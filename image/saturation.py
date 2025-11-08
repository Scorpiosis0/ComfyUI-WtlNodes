import torch
import cv2
import numpy as np
import time
import threading
from nodes import PreviewImage

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

class saturationNode(PreviewImage):
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
                })
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="saturation"
    CATEGORY = "WtlNodes/image"

    def saturation(self, image, saturation, unique_id=None, prompt=None, extra_pnginfo=None,):

        # -------------------------------------------------------------
        #  1️⃣  Clean any stale data for this node (mirrors old file‑cleanup)
        # -------------------------------------------------------------
        if unique_id:
            uid = str(unique_id)          # ensure consistent key type
            _clear_all(uid)

        # -------------------------------------------------------------
        #  2️⃣  Early‑out if the user pressed **Skip**
        # -------------------------------------------------------------
        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            batch_size = image.shape[0]
            return (image)

        # -------------------------------------------------------------
        #  3️⃣  Convert tensors to numpy for the heavy lifting
        # -------------------------------------------------------------

        batch_size = image.shape[0]
        result = image.clone()

        for b in range(batch_size):
            print(f"Hello")
            cur_image = image[b:b+1]
            # ---- preview loop (only when we have a UI node id) ---- #?
            if unique_id is not None:
                print(f"No unique ID exist!")

            if unique_id:
                print(f"Anyone here?")
                uid = str(unique_id)
                print(
                    f"[SAT] Starting preview loop for node {uid}. "
                    "Adjust sliders, then press **Apply Effect**."
                )
                while True:
                    # Grab the *latest* slider values sent by the UI
                    cur_saturation = _get_params(uid, saturation)
                    print(f"And here?")
                    # Build a temporary mask with those live values
                    grayscale = (image[..., 0:1] * 0.299 + image[..., 1:2] * 0.587 + image[..., 2:3] * 0.114)
                    grayscale_rgb = grayscale.repeat(1, 1, 1, 3)

                    cur_image = grayscale_rgb + (image - grayscale_rgb) * (1+cur_saturation/100)
                    cur_image = torch.clamp(cur_image, 0.0, 1.0)
                    # Show a quick preview in the UI
                    self._preview_image(cur_image, uid, prompt, extra_pnginfo)

                    # -------------------------------------------------
                    #  Check for button presses
                    # -------------------------------------------------
                    if _check_and_clear_flag(uid, "apply"):
                        print(f"[SAT] **Apply** pressed for node {uid}.")
                        saturation = (
                            cur_saturation
                        )
                        image = cur_image
                        break   # exit preview loop

                    if _check_and_clear_flag(uid, "skip"):
                        print(f"[SAT] **Skip** pressed for node {uid}.")
                        return (image)

                    # Throttle the loop a little so we don’t hammer the CPU
                    time.sleep(0.5)
            print(f"U there?")
            # ---- real effect (unchanged) -----------------------------
            grayscale = (image[..., 0:1] * 0.299 + image[..., 1:2] * 0.587 + image[..., 2:3] * 0.114)
            grayscale_rgb = grayscale.repeat(1, 1, 1, 3)

            result[b:b+1] = grayscale_rgb + (image - grayscale_rgb) * (1+saturation/100)
            result[b:b+1] = torch.clamp(result[b:b+1], 0.0, 1.0)
        return (result,)
    
    # -----------------------------------------------------------------
    #  Helper that pushes a mask preview to the UI (unchanged)
    # -----------------------------------------------------------------
    def _preview_image(self, image, unique_id, prompt, extra_pnginfo):
        """Preview the image in real‑time."""
        try:
            result = self.save_images(
                image,
                filename_prefix="sat_preview_",
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )

            # Send preview to UI via websocket (ComfyUI internal API)
            import server

            if hasattr(server.PromptServer, "instance"):
                server.PromptServer.instance.send_sync(
                    "executed",
                    {
                        "node": unique_id,
                        "output": {"images": result["ui"]["images"]},
                        "prompt_id": None,
                    },
                )
        except Exception as e:
            print(f"[SAT] Preview error: {e}")
    
NODE_CLASS_MAPPINGS = {"saturationNode": saturationNode}
NODE_DISPLAY_NAME_MAPPINGS = {"saturationNode": "Saturation"}