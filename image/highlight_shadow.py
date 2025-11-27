import torch
import cv2
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, shadow_adjustment: float, highlight_adjustment: float, midpoint: float, feather_radius: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (shadow_adjustment, highlight_adjustment, midpoint, feather_radius,)

def _get_params(node_id: str, *defaults: float) -> tuple[float, ...]:
    """Return stored params if available, otherwise return the provided defaults."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", defaults)

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

def _process(image, shadow_adjustment, highlight_adjustment, midpoint, feather_radius):
    # ComfyUI images are in format [B, H, W, C]
    device = image.device
    
    # Clamp input to valid range
    img = torch.clamp(image, 0, 1)
    
    # Convert RGB to HSV
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    
    max_c = torch.maximum(torch.maximum(r, g), b)
    min_c = torch.minimum(torch.minimum(r, g), b)
    delta = max_c - min_c
    
    v = max_c
    s = torch.where(max_c != 0, delta / max_c, torch.zeros_like(max_c))
    
    # Compute hue
    h = torch.zeros_like(max_c)
    mask = delta != 0
    
    mask_r = mask & (r == max_c)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    
    mask_g = mask & (g == max_c)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    
    mask_b = mask & (b == max_c)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    h = h % 360
    
    # Create luminance-based masks from V channel
    shadow_mask = torch.clamp((midpoint - v) / (midpoint + 1e-10), 0, 1)
    highlight_mask = torch.clamp((v - midpoint) / (1.0 - midpoint + 1e-10), 0, 1)
    
    # Apply Gaussian blur to feather the masks using cv2
    if feather_radius > 0:

        ksize = int(feather_radius * 2) | 1  # Ensure odd kernel size
        if ksize < 3:
            ksize = 3

        # Remove batch dimension for OpenCV (expects HxW)
        shadow_np = shadow_mask[0].cpu().numpy()
        highlight_np = highlight_mask[0].cpu().numpy()

        # Gaussian blur
        shadow_blur = cv2.GaussianBlur(shadow_np, (ksize, ksize), feather_radius / 3.0)
        highlight_blur = cv2.GaussianBlur(highlight_np, (ksize, ksize), feather_radius / 3.0)

        # Re-wrap into torch with batch dimension restored
        shadow_mask = torch.from_numpy(shadow_blur).unsqueeze(0).to(device)
        highlight_mask = torch.from_numpy(highlight_blur).unsqueeze(0).to(device)
    
    # Apply adjustments to V channel
    v_adjusted = v.clone()
    
    if shadow_adjustment != 0:
        v_adjusted = v_adjusted + (shadow_adjustment/100) * shadow_mask
    
    if highlight_adjustment != 0:
        v_adjusted = v_adjusted + (highlight_adjustment/100) * highlight_mask
    
    v_adjusted = torch.clamp(v_adjusted, 0, 1)
    
    # HSV -> RGB conversion
    c = v_adjusted * s
    x = c * (1 - torch.abs((h / 60) % 2 - 1))
    m = v_adjusted - c
    
    h_i = (h / 60).long()
    
    r_out = torch.zeros_like(h)
    g_out = torch.zeros_like(h)
    b_out = torch.zeros_like(h)
    
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
    
    rgb_adjusted = torch.stack([r_out + m, g_out + m, b_out + m], dim=-1)
    rgb_adjusted = torch.clamp(rgb_adjusted, 0, 1)
    
    return rgb_adjusted

class HighlightShadowC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_adjustment": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "highlight_adjustment": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "midpoint": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "feather_radius": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 1.0,
                }),
                "apply_type": (["none","auto_apply","apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_highlight_shadow"
    CATEGORY = "WtlNodes/image"

    def adjust_highlight_shadow(self, image, shadow_adjustment, highlight_adjustment, midpoint, feather_radius, apply_type, unique_id=None):

        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                # Process all images at once (original behavior)
                while True:
                    cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                    cur_image = _process(image, cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius)
                    _send_ram_preview(cur_image, uid)

                    #  Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        params = (cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius)
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (image,)}
                    
                    time.sleep(0.25)

                # Apply final effect after exiting loop
                result = _process(image, cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius)

            else:
                # Process images one by one
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]  # Keep batch dimension
                    
                    while True:
                        cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                        cur_image = _process(single_image, cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius)
                        _send_ram_preview(cur_image, uid)

                        # Check for button presses
                        if _check_and_clear_flag(uid, "apply"):
                            final_params = (cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius)
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            # Skip this image, use original
                            result_list.append(single_image)
                            final_params = None
                            break
                        
                        time.sleep(0.25)

                    # Apply final effect for this image if not skipped
                    if final_params is not None:
                        processed = _process(image, cur_shadow_adjustment, cur_highlight_adjustment, cur_midpoint, cur_feather_radius)
                        result_list.append(processed)

                # Concatenate all processed images back into a batch
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode (always processes all images the same way)
            result = _process(image, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"HighlightShadow": HighlightShadowC}
NODE_DISPLAY_NAME_MAPPINGS = {"HighlightShadow": "Highlight & Shadow"}