import torch
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, contrast: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (contrast,)

def _get_params(node_id: str, contrast: float) -> tuple[float]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (contrast,))

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

class contrast:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
                "contrast":("FLOAT",{
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
                "apply_all": ("BOOLEAN", {
                    "default": False,
                    "label_on": "On",
                    "label_off": "Off"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                },
            }
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="contrast"
    CATEGORY = "WtlNodes/image"
    
    def contrast(self, image, contrast, auto_apply, apply_all, unique_id=None):

        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}
        
        if unique_id and not auto_apply:
            uid = str(unique_id)
            
            if apply_all:
                # Process all images at once (original behavior)
                while True:
                    cur_contrast = _get_params(uid, contrast)[0]
                    pivot = 0.5
                    cur_image = pivot + (image - pivot) * (1 + cur_contrast / 100)
                    cur_image = torch.clamp(cur_image, 0.0, 1.0)
                    _send_ram_preview(cur_image, uid)

                    # Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        contrast = cur_contrast
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (image,)}
                    
                    time.sleep(0.25)
                
                # Apply final effect after exiting loop
                pivot = 0.5
                result = pivot + (image - pivot) * (1 + contrast / 100)
                result = torch.clamp(result, 0.0, 1.0)
            
            else:
                # Process images one by one
                batch_size = image.shape[0]
                result_list = []
                
                for i in range(batch_size):
                    single_image = image[i:i+1]  # Keep batch dimension
                    
                    while True:
                        cur_contrast = _get_params(uid, contrast)[0]
                        pivot = 0.5
                        cur_image = pivot + (single_image - pivot) * (1 + cur_contrast / 100)
                        cur_image = torch.clamp(cur_image, 0.0, 1.0)
                        _send_ram_preview(cur_image, uid)

                        # Check for button presses
                        if _check_and_clear_flag(uid, "apply"):
                            final_contrast = cur_contrast
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            # Skip this image, use original
                            result_list.append(single_image)
                            final_contrast = None
                            break
                        
                        time.sleep(0.25)
                    
                    # Apply final effect for this image if not skipped
                    if final_contrast is not None:
                        pivot = 0.5
                        processed = pivot + (single_image - pivot) * (1 + final_contrast / 100)
                        processed = torch.clamp(processed, 0.0, 1.0)
                        result_list.append(processed)
                
                # Concatenate all processed images back into a batch
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode (always processes all images the same way)
            pivot = 0.5
            result = pivot + (image - pivot) * (1 + contrast / 100)
            result = torch.clamp(result, 0.0, 1.0)
                
        return {"result": (result,)}
            
NODE_CLASS_MAPPINGS = {"Contrast": contrast}
NODE_DISPLAY_NAME_MAPPINGS = {"Contrast": "Contrast"}