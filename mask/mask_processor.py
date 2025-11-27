import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, dilate_erode: int, feather: int) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (dilate_erode, feather)

def _get_params(node_id: str, *defaults: int) -> tuple[int, ...]:
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
            flags[flag] = False
            return True
        return False

def _clear_all(node_id: str) -> None:
    """Remove *everything* stored for a node – used at the start of a run."""
    with _CONTROL_LOCK:
        _CONTROL_STORE.pop(node_id, None)

def apply_mask_processing(mask, dilate_erode, feather):
    """Apply mask processing with dilate/erode and feather effects."""
    # Convert tensor to numpy
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    results = []
    
    for b in range(batch_size):
        # Get single mask and convert to uint8 (0-255)
        m = mask_np[b]
        m_uint8 = (m * 255).astype(np.uint8)
        
        # Step 1: Expand or Shrink (Morphological operations)
        if dilate_erode != 0:
            m_uint8 = _expand_or_shrink(m_uint8, dilate_erode)
        
        # Step 2: Feather (Gaussian blur)
        if feather > 0:
            m_uint8 = _feather_mask(m_uint8, feather)
        
        # Convert back to float (0-1)
        m_float = m_uint8.astype(np.float32) / 255.0
        results.append(m_float)
    
    # Convert back to torch tensor
    output = torch.from_numpy(np.stack(results)).float()
    return output

def _expand_or_shrink(mask, amount):      
    # Create kernel for morphological operation
    kernel_size = abs(amount) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if amount > 0:
        # Expand (dilation)
        result = cv2.dilate(mask, kernel, iterations=1)
    else:
        # Shrink (erosion)
        result = cv2.erode(mask, kernel, iterations=1)
    
    return result

def _feather_mask(mask, radius):
    # Gaussian blur for soft edges
    # Kernel size must be odd
    kernel_size = radius * 2 + 1
    
    # Apply gaussian blur
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    return blurred

class MaskProcessorC:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "dilate_erode": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "apply_type": (["none", "auto_apply","apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "WtlNodes/mask"

    def process_mask(self, mask, dilate_erode, feather, apply_type, unique_id=None):
        
        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                # Process all images at once (original behavior)
                while True:
                    cur_dilate_erode, cur_feather = _get_params(uid, dilate_erode, feather)
                    cur_filtered = apply_mask_processing(mask, cur_dilate_erode, cur_feather)
                    _send_ram_preview(cur_filtered, uid)

                    #  Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        final_dilate_erode, final_feather = cur_dilate_erode, cur_feather
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (mask,)}
                    
                    time.sleep(0.25)

                # Apply final effect after exiting loop
                result = apply_mask_processing(mask, dilate_erode, feather)
            
            else:
                # Process masks one by one
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]  # Keep batch dimension
                    
                    while True:
                        cur_dilate_erode, cur_feather = _get_params(uid, dilate_erode, feather)
                        cur_processed = apply_mask_processing(single_mask, cur_dilate_erode, cur_feather)
                        
                        # Convert mask to image for preview
                        preview_image = cur_processed.unsqueeze(-1).repeat(1, 1, 1, 3)
                        _send_ram_preview(preview_image, uid)

                        # Check for button presses
                        if _check_and_clear_flag(uid, "apply"):
                            final_dilate_erode, final_feather = cur_dilate_erode, cur_feather
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            # Skip this mask, use original
                            result_list.append(single_mask)
                            final_dilate_erode = None
                            break
                        
                        time.sleep(0.25)

                    # Apply final effect for this mask if not skipped
                    if final_dilate_erode is not None:
                        processed = apply_mask_processing(single_mask, final_dilate_erode, final_feather)
                        result_list.append(processed)

                # Concatenate all processed masks back into a batch
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode (always processes all masks the same way)
            result = apply_mask_processing(mask, dilate_erode, feather)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"MaskProcessor": MaskProcessorC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskProcessor": "Mask Processor"}