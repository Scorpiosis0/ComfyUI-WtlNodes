import torch
import cv2
import numpy as np
import time
import threading
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, focus: float, rng: float, edge: int, hard_focus: float, blur: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (focus, rng, edge, hard_focus, blur)

def _get_params(node_id: str, default_focus: float, default_range: float, default_edge: int, default_hard_focus: float, default_blur: float) -> tuple[float, float, int, float, float]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (default_focus, default_range, default_edge, default_hard_focus, default_blur))

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

def _apply_dof_to_image(img, depth, focus_depth, focus_range, hard_focus_range, edge_fix, blur_strength):
    """Apply DOF effect to a single image and return result + mask"""
    
    if depth.shape[-1] > 1:
        depth = np.mean(depth, axis=-1, keepdims=True)

    # Normalise depth to 0‑1
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Build mask with hard focus range
    hard_zone_min = focus_depth - hard_focus_range
    hard_zone_max = focus_depth + hard_focus_range
    
    blur_mask = np.zeros_like(depth)
    
    # For depths below hard zone
    below_mask = depth < hard_zone_min
    blur_mask[below_mask] = (hard_zone_min - depth[below_mask]) / focus_range
    
    # For depths above hard zone
    above_mask = depth > hard_zone_max
    blur_mask[above_mask] = (depth[above_mask] - hard_zone_max) / focus_range
    
    # Clip to 0-1 range
    blur_mask = np.clip(blur_mask, 0, 1).squeeze()
    
    # Edge Fix
    if edge_fix > 0:
        kernel_size = abs(edge_fix) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blur_mask = cv2.dilate(blur_mask, kernel, iterations=1)
        blur_mask = cv2.erode(blur_mask, kernel, iterations=1)

    # Apply blur
    img_uint8 = (img * 255).astype(np.uint8)
    kernel_size = int(blur_strength * 2) * 2 + 1
    kernel_size = max(1, kernel_size)

    if kernel_size > 1:
        blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        blurred = blurred.astype(np.float32) / 255.0
    else:
        blurred = img

    # Blend original and blurred based on mask
    blur_mask_3ch = np.expand_dims(blur_mask, axis=-1)
    result = img * (1 - blur_mask_3ch) + blurred * blur_mask_3ch

    return result, blur_mask

class DepthDOFNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "focus_depth": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                }),
                "blur_strength": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "hard_focus_range": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "round": 0.01,
                }),
                "focus_range": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                }),
                "edge_fix": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                }),

            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "blur_mask")
    FUNCTION = "apply_dof"
    CATEGORY = "WtlNodes/image"

    def __init__(self):
        super().__init__()

    def apply_dof(self, image, depth_map, focus_depth, blur_strength, focus_range, hard_focus_range, edge_fix, unique_id=None, prompt=None, extra_pnginfo=None):

        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        # Early‑out if the user pressed **Skip**
        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            batch_size = image.shape[0]
            empty_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]))
            return (image, empty_mask)

        # Convert tensors to numpy
        img_np = image.cpu().numpy()
        depth_np = depth_map.cpu().numpy()
        batch_size = img_np.shape[0]

        if unique_id:
            uid = str(unique_id)

            # Process images one by one with different parameters
            results = []
            masks = []

            for b in range(batch_size):
                img = img_np[b]
                depth = depth_np[b]

                while True:
                    cur_focus, cur_range, cur_edge, cur_hard_range, cur_blur = _get_params(
                        uid, focus_depth, focus_range, edge_fix, hard_focus_range, blur_strength
                    )

                    preview_result, preview_mask = _apply_dof_to_image(
                        img, depth, cur_focus, cur_range, cur_hard_range, cur_edge, cur_blur
                    )
                    
                    # Convert mask to RGB for preview
                    mask_rgb = np.stack([preview_mask, preview_mask, preview_mask], axis=-1)
                    mask_tensor = torch.from_numpy(mask_rgb).unsqueeze(0).float()
                    _send_ram_preview(mask_tensor, uid)

                    # Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        final_params = (cur_focus, cur_range, cur_edge, cur_hard_range, cur_blur)
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        # Skip this image, use original
                        results.append(img)
                        masks.append(np.zeros((img.shape[0], img.shape[1])))
                        final_params = None
                        break
                    
                    time.sleep(0.25)

                # Apply final effect for this image if not skipped
                if final_params is not None:
                    result, mask = _apply_dof_to_image(
                        img, depth, final_params[0], final_params[1], 
                        final_params[3], final_params[2], final_params[4]
                    )
                    results.append(result)
                    masks.append(mask)

            output_img = torch.from_numpy(np.stack(results)).float()
            output_mask = torch.from_numpy(np.stack(masks)).float()

        else:
            # Auto-apply mode
            results = []
            masks = []

            for b in range(batch_size):
                result, mask = _apply_dof_to_image(
                    img_np[b], depth_np[b], focus_depth, focus_range, 
                    hard_focus_range, edge_fix, blur_strength
                )
                results.append(result)
                masks.append(mask)

            output_img = torch.from_numpy(np.stack(results)).float()
            output_mask = torch.from_numpy(np.stack(masks)).float()

        print(f"[DOF] Effect applied – returning final image.")
        return (output_img, output_mask)

NODE_CLASS_MAPPINGS = {"DepthDOFNode": DepthDOFNode}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthDOFNode": "Depth of Field (DOF)"}