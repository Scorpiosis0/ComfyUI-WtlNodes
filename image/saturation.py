import torch
import cv2
import numpy as np
import time
import threading
import base64
import io
from PIL import Image
import server
import math

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, saturation: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (saturation)

def _get_params(
    node_id: str,
    saturation: float,
) -> tuple[float]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (saturation))

def _set_flag(node_id: str, flag: str) -> None:
    """Mark a button press – ``flag`` must be ``'apply'`` or ``'skip'``."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        flags = entry.setdefault("flags", {})
        flags[flag] = True

def _check_and_clear_flag(node_id: str, flag: str) -> bool:
    """Return True once if the flag was set; afterwards it is cleared."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id)
        if not entry:
            return False
        flags = entry.get("flags", {})
        if flags.get(flag):
            flags[flag] = False          # clear it for the next poll
            return True
        return False

def _clear_all(node_id: str) -> None:
    """Remove *everything* stored for a node – used at the start of a run."""
    with _CONTROL_LOCK:
        _CONTROL_STORE.pop(node_id, None)

def _saturation_hsv(image, saturation):

    img_np = image.cpu().numpy()[0]  # shape: H, W, C
    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    v = max_c
    s = np.where(max_c != 0, delta / max_c, 0)

    h = np.zeros_like(max_c)
    mask = (delta != 0)
    # red max
    mask_r = mask & (r == max_c)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    # green max
    mask_g = mask & (g == max_c)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    # blue max
    mask_b = mask & (b == max_c)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    h = h % 360

    # Apply saturation
    s = np.clip(s * (1+saturation/100), 0, 1)

    # HSV -> RGB
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c
    h_i = (h / 60).astype(int)

    r_out = np.zeros_like(h)
    g_out = np.zeros_like(h)
    b_out = np.zeros_like(h)

    # 0-5
    mask0 = h_i == 0
    r_out[mask0], g_out[mask0], b_out[mask0] = c[mask0], x[mask0], 0
    mask1 = h_i == 1
    r_out[mask1], g_out[mask1], b_out[mask1] = x[mask1], c[mask1], 0
    mask2 = h_i == 2
    r_out[mask2], g_out[mask2], b_out[mask2] = 0, c[mask2], x[mask2]
    mask3 = h_i == 3
    r_out[mask3], g_out[mask3], b_out[mask3] = 0, x[mask3], c[mask3]
    mask4 = h_i == 4
    r_out[mask4], g_out[mask4], b_out[mask4] = x[mask4], 0, c[mask4]
    mask5 = h_i == 5
    r_out[mask5], g_out[mask5], b_out[mask5] = c[mask5], 0, x[mask5]

    cur_image_np = np.stack([r_out + m, g_out + m, b_out + m], axis=-1)
    cur_image = torch.from_numpy(cur_image_np).unsqueeze(0).to(image.device)
    cur_image = torch.clamp(cur_image, 0.0, 1.0)
    return(cur_image)

class saturationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
                "saturation":("FLOAT",{
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "auto_apply": ("BOOLEAN", {
                    "default": False,
                    "label_on": "On",
                    "label_off": "Off"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="saturation"
    OUTPUT_NODE = True
    CATEGORY = "WtlNodes/image"

    def saturation(self, image, saturation, auto_apply, unique_id=None, prompt=None, extra_pnginfo=None):
        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        # Early‑out if the user pressed **Skip**
        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        result = image

        if unique_id and not auto_apply:
            uid = str(unique_id)
            while True:
                # Grab the *latest* slider values sent by the UI
                cur_saturation = _get_params(uid, saturation)
                # Build a temporary image with those live values
                cur_image = _saturation_hsv(image, cur_saturation)
                
                # Convert to PIL Image
                i = 255. * cur_image[0].cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                # Resize to 1MP (1 megapixel = 1,000,000 pixels)
                width, height = img.size
                current_pixels = width * height
                
                if current_pixels > 1000000:
                    # Calculate scale factor to get to 1MP
                    scale = math.sqrt(1000000 / current_pixels)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save to buffer with NO compression for speed
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", compress_level=0)
                buffer.seek(0)
                
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Send preview via websocket WITHOUT returning
                if hasattr(server.PromptServer, "instance"):
                    server.PromptServer.instance.send_sync(
                        "executed",
                        {
                            "node": uid,
                            "output": {
                                "ram_preview": [img_base64]
                            },
                            "prompt_id": None
                        }
                    )

                #  Check for button presses
                if _check_and_clear_flag(uid, "apply"):
                    saturation = cur_saturation
                    break   # exit preview loop

                if _check_and_clear_flag(uid, "skip"):
                    return {"result": (image,)}
                
                time.sleep(0.5)
                
            # Apply final effect after exiting loop
            result = _saturation_hsv(image, saturation)
        else:
            # Auto-apply mode
            result = _saturation_hsv(image, saturation)
                
        return {"result": (result,)}
    
NODE_CLASS_MAPPINGS = {"saturationNode": saturationNode}
NODE_DISPLAY_NAME_MAPPINGS = {"saturationNode": "Saturation (HSV)"}