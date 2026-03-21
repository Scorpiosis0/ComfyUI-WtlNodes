import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, focus: float, rng: float, edge: int, hard_focus: float, blur: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (focus, rng, edge, hard_focus, blur)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, focus: float, rng: float, edge: int, hard_focus: float, blur: float) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (focus, rng, edge, hard_focus, blur))

def _check_and_clear_params_changed(node_id: str) -> bool:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id)
        if not entry:
            return False
        if entry.get("params_changed"):
            entry["params_changed"] = False
            return True
        return False

def _set_processing_time(node_id: str, ms: int) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["processing_time_ms"] = ms
        entry["processing_complete"] = True

def _get_processing_time(node_id: str) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return (entry.get("processing_time_ms", 0), entry.get("processing_complete", False))

def _set_flag(node_id: str, flag: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry.setdefault("flags", {})[flag] = True

def _check_and_clear_flag(node_id: str, flag: str) -> bool:
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
    with _CONTROL_LOCK:
        _CONTROL_STORE.pop(node_id, None)

def _apply_single(img, depth, focus_depth, focus_range, hard_focus_range, edge_fix, blur_strength):
    if depth.shape[-1] > 1:
        depth = np.mean(depth, axis=-1, keepdims=True)

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    hard_zone_min = focus_depth - hard_focus_range
    hard_zone_max = focus_depth + hard_focus_range
    blur_mask = np.zeros_like(depth)
    below = depth < hard_zone_min
    blur_mask[below] = (hard_zone_min - depth[below]) / focus_range
    above = depth > hard_zone_max
    blur_mask[above] = (depth[above] - hard_zone_max) / focus_range
    blur_mask = np.clip(blur_mask, 0, 1).squeeze()

    if edge_fix > 0:
        ksize = abs(edge_fix) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        blur_mask = cv2.dilate(blur_mask, kernel, iterations=1)
        blur_mask = cv2.erode(blur_mask, kernel, iterations=1)

    img_uint8 = (img * 255).astype(np.uint8)
    kernel_size = max(1, int(blur_strength * 2) * 2 + 1)
    if kernel_size > 1:
        blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0).astype(np.float32) / 255.0
    else:
        blurred = img

    blur_mask_3ch = np.expand_dims(blur_mask, axis=-1)
    result = img * (1 - blur_mask_3ch) + blurred * blur_mask_3ch
    return result, blur_mask

class DepthOfFieldC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "focus_depth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "blur_strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "round": 0.1}),
                "hard_focus_range": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "round": 0.001}),
                "focus_range": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "edge_fix": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
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

    def apply_dof(self, image, depth_map, focus_depth, blur_strength, focus_range,
                  hard_focus_range, edge_fix, unique_id=None, prompt=None, extra_pnginfo=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            empty_mask = torch.zeros((image.shape[0], image.shape[1], image.shape[2]))
            return (image, empty_mask)

        img_np = image.cpu().numpy()
        depth_np = depth_map.cpu().numpy()
        batch_size = img_np.shape[0]

        if unique_id:
            uid = str(unique_id)
            results = []
            masks = []

            for b in range(batch_size):
                img = img_np[b]
                depth = depth_np[b]

                cur = _get_params(uid, focus_depth, focus_range, edge_fix, hard_focus_range, blur_strength)
                t0 = time.time()
                preview_result, preview_mask = _apply_single(img, depth, cur[0], cur[1], cur[3], cur[2], cur[4])
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                mask_rgb = np.stack([preview_mask] * 3, axis=-1)
                _send_ram_preview(torch.from_numpy(mask_rgb).unsqueeze(0).float(), uid)

                final = None
                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, focus_depth, focus_range, edge_fix, hard_focus_range, blur_strength)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            results.append(img)
                            masks.append(np.zeros((img.shape[0], img.shape[1])))
                            final = None
                            break
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, focus_depth, focus_range, edge_fix, hard_focus_range, blur_strength)
                    t0 = time.time()
                    preview_result, preview_mask = _apply_single(img, depth, cur[0], cur[1], cur[3], cur[2], cur[4])
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    mask_rgb = np.stack([preview_mask] * 3, axis=-1)
                    _send_ram_preview(torch.from_numpy(mask_rgb).unsqueeze(0).float(), uid)

                if final is not None:
                    result, mask = _apply_single(img, depth, final[0], final[1], final[3], final[2], final[4])
                    results.append(result)
                    masks.append(mask)

            output_img = torch.from_numpy(np.stack(results)).float()
            output_mask = torch.from_numpy(np.stack(masks)).float()
        else:
            results, masks = [], []
            for b in range(batch_size):
                result, mask = _apply_single(img_np[b], depth_np[b], focus_depth, focus_range,
                                             hard_focus_range, edge_fix, blur_strength)
                results.append(result)
                masks.append(mask)
            output_img = torch.from_numpy(np.stack(results)).float()
            output_mask = torch.from_numpy(np.stack(masks)).float()

        return (output_img, output_mask)


NODE_CLASS_MAPPINGS = {"DepthDOF": DepthOfFieldC}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthDOF": "Depth of Field (DOF)"}
