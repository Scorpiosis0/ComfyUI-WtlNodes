import torch
import numpy as np
from scipy import ndimage
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, area_x: int, area_y: int, keep: str) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (area_x, area_y, keep)

def _get_params(node_id: str, area_x: int, area_y: int, keep: str) -> tuple[int, int, str]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (area_x, area_y, keep))

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

def apply_mask_filter(masks, area_x, area_y, keep):
    """Apply mask filtering based on area criteria."""
    # Handle different mask shapes
    if len(masks.shape) == 4:
        # Shape is [B, 1, H, W], squeeze to [B, H, W]
        masks = masks.squeeze(1)
    
    filtered_masks = []
    
    for i in range(masks.shape[0]):
        mask = masks[i]
        
        # Convert to numpy for connected component analysis
        mask_np = mask.cpu().numpy()
        
        # Threshold the mask (in case it has values between 0 and 1)
        binary_mask = mask_np > 0.5
        
        # Find connected components
        labeled_mask, num_features = ndimage.label(binary_mask)
        
        # Create output mask
        filtered_mask = np.zeros_like(mask_np)
        
        # Check each connected component
        for component_id in range(1, num_features + 1):
            # Get the component
            component = (labeled_mask == component_id)
            # Calculate area
            area = np.sum(component)
            
            # Determine if we should keep this component based on keep parameter
            should_keep = False
            if keep == "above_x":
                should_keep = area >= area_x
            elif keep == "bellow_x":
                should_keep = area <= area_x
            elif keep == "between_x_y":
                should_keep = area_x <= area <= area_y
            
            if should_keep:
                # Keep this component
                filtered_mask[component] = mask_np[component]
        
        # Convert back to torch tensor
        filtered_mask_tensor = torch.from_numpy(filtered_mask).to(mask.device).to(mask.dtype)
        filtered_masks.append(filtered_mask_tensor)
    
    # Stack filtered masks back into a batch
    result = torch.stack(filtered_masks, dim=0)
    
    return result

class MaskFilterC:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "area_x": ("INT", {
                    "default": 3000,
                    "min": 0,
                    "max": 10000000,
                    "step": 100,
                    "display": "number"
                }),
                "area_y": ("INT", {
                    "default": 5000,
                    "min": 0,
                    "max": 10000000,
                    "step": 100,
                    "display": "number"
                }),
                "keep": (["above_x", "bellow_x", "between_x_y"],),
                "apply_type": (["none", "auto_apply","apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "filter_masks"
    CATEGORY = "WtlNodes/mask"
    
    def filter_masks(self, masks, area_x, area_y, keep, apply_type, unique_id=None):
        
        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (masks,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                # Process all images at once (original behavior)
                while True:
                    cur_area_x, cur_area_y, cur_keep = _get_params(uid, area_x, area_y, keep)
                    cur_filtered = apply_mask_filter(masks, cur_area_x, cur_area_y, cur_keep)
                    _send_ram_preview(cur_filtered, uid)

                    #  Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        final_area_x, final_area_y, final_keep = cur_area_x, cur_area_y, cur_keep
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (masks,)}
                    
                    time.sleep(0.25)

                # Apply final effect after exiting loop
                result = apply_mask_filter(masks, area_x, area_y, keep)

            else:
                # Process masks one by one
                batch_size = masks.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = masks[i:i+1]  # Keep batch dimension
                    
                    while True:
                        cur_area_x, cur_area_y, cur_keep = _get_params(uid, area_x, area_y, keep)
                        cur_filtered = apply_mask_filter(single_mask, cur_area_x, cur_area_y, cur_keep)
                        
                        # Convert mask to image for preview
                        preview_image = cur_filtered.unsqueeze(-1).repeat(1, 1, 1, 3)
                        _send_ram_preview(preview_image, uid)

                        # Check for button presses
                        if _check_and_clear_flag(uid, "apply"):
                            final_area_x, final_area_y, final_keep = cur_area_x, cur_area_y, cur_keep
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            # Skip this mask, use original
                            result_list.append(single_mask)
                            final_area_x = None
                            break
                        
                        time.sleep(0.25)

                    # Apply final effect for this mask if not skipped
                    if final_area_x is not None:
                        processed = apply_mask_filter(single_mask, final_area_x, final_area_y, final_keep)
                        result_list.append(processed)

                # Concatenate all processed masks back into a batch
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode (always processes all masks the same way)
            result = apply_mask_filter(masks, area_x, area_y, keep)
                
        return {"result": (result,)}

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {"MaskFilter": MaskFilterC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskFilter": "Mask Filter"}