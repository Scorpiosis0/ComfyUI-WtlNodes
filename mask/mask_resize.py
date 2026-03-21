import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, resize_by: bool, width: int, height: int, multiplier: float,
                interpolation: str, fit_mode: str, enhanced_visibility: bool) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, resize_by: bool, width: int, height: int, multiplier: float,
                interpolation: str, fit_mode: str, enhanced_visibility: bool) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility))

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
        flags = entry.setdefault("flags", {})
        flags[flag] = True

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

def apply_resize(mask, resize_by, width, height, multiplier, interp_method, fit_mode):
    mask_np = mask.cpu().numpy()
    results = []

    for b in range(mask_np.shape[0]):
        m = mask_np[b]
        orig_h, orig_w = m.shape[:2]
        mask_uint8 = (m * 255).astype(np.uint8)

        target_w = max(1, int(orig_w * multiplier)) if resize_by else max(1, width)
        target_h = max(1, int(orig_h * multiplier)) if resize_by else max(1, height)
        h, w = mask_uint8.shape[:2]

        if fit_mode == "adjust":
            result_mask = cv2.resize(mask_uint8, (target_w, target_h), interpolation=interp_method)
        elif fit_mode == "crop":
            ar = w / h
            tar = target_w / target_h
            if ar > tar:
                new_h, new_w = target_h, int(target_h * ar)
            else:
                new_w, new_h = target_w, int(target_w / ar)
            resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)
            sx, sy = (new_w - target_w) // 2, (new_h - target_h) // 2
            result_mask = resized[sy:sy + target_h, sx:sx + target_w]
        elif fit_mode == "fit":
            ar = w / h
            tar = target_w / target_h
            if ar > tar:
                new_w, new_h = target_w, int(target_w / ar)
            else:
                new_h, new_w = target_h, int(target_h * ar)
            resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)
            result_mask = np.zeros((target_h, target_w), dtype=mask_uint8.dtype)
            sx, sy = (target_w - new_w) // 2, (target_h - new_h) // 2
            result_mask[sy:sy + new_h, sx:sx + new_w] = resized

        results.append(result_mask.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

def apply_resize_preview(mask, resize_by, width, height, multiplier, interp_method, fit_mode, enhanced_visibility):
    mask_np = mask.cpu().numpy()
    results = []

    for b in range(mask_np.shape[0]):
        m = mask_np[b]
        orig_h, orig_w = m.shape[:2]
        mask_uint8 = (m * 255).astype(np.uint8)

        target_w = max(1, int(orig_w * multiplier)) if resize_by else max(1, width)
        target_h = max(1, int(orig_h * multiplier)) if resize_by else max(1, height)
        h, w = mask_uint8.shape[:2]

        if fit_mode == "adjust":
            result_mask = cv2.resize(mask_uint8, (target_w, target_h), interpolation=interp_method)
            rgb = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2RGB)
        elif fit_mode == "crop":
            ar = w / h
            tar = target_w / target_h
            if ar > tar:
                new_h, new_w = target_h, int(target_h * ar)
            else:
                new_w, new_h = target_w, int(target_w / ar)
            resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)
            sx, sy = (new_w - target_w) // 2, (new_h - target_h) // 2
            result_mask = resized[sy:sy + target_h, sx:sx + target_w]
            rgb = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2RGB)
        elif fit_mode == "fit":
            ar = w / h
            tar = target_w / target_h
            if ar > tar:
                new_w, new_h = target_w, int(target_w / ar)
            else:
                new_h, new_w = target_h, int(target_h * ar)
            resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)
            result_mask = np.zeros((target_h, target_w), dtype=mask_uint8.dtype)
            sx, sy = (target_w - new_w) // 2, (target_h - new_h) // 2
            result_mask[sy:sy + new_h, sx:sx + new_w] = resized
            rgb = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2RGB)
            if enhanced_visibility:
                padding_mask = np.ones((target_h, target_w), dtype=bool)
                padding_mask[sy:sy + new_h, sx:sx + new_w] = False
                rgb[padding_mask] = np.array([255, 0, 0], dtype=np.uint8)

        results.append(rgb.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

class MaskResizeC:
    INTERPOLATION_METHODS = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "resize_by": ("BOOLEAN", {"default": False, "label_on": "Multiplier", "label_off": "Absolute"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()), {"default": "nearest"}),
                "fit_mode": (["crop", "adjust", "fit"],),
                "enhanced_visibility": ("BOOLEAN", {"default": False}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "resize"
    CATEGORY = "WtlNodes/mask"

    def resize(self, mask, resize_by, width, height, multiplier, interpolation,
               fit_mode, enhanced_visibility, apply_type, unique_id=None):
        interp_method = self.INTERPOLATION_METHODS[interpolation]

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur = _get_params(uid, resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility)
                cur_method = self.INTERPOLATION_METHODS[cur[4]]
                start_time = time.time()
                preview = apply_resize_preview(mask, cur[0], cur[1], cur[2], cur[3], cur_method, cur[5], cur[6])
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (mask,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility)
                    cur_method = self.INTERPOLATION_METHODS[cur[4]]
                    start_time = time.time()
                    preview = apply_resize_preview(mask, cur[0], cur[1], cur[2], cur[3], cur_method, cur[5], cur[6])
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(preview, uid)

                final_method = self.INTERPOLATION_METHODS[final[4]]
                result = apply_resize(mask, final[0], final[1], final[2], final[3], final_method, final[5])

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]

                    cur = _get_params(uid, resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility)
                    cur_method = self.INTERPOLATION_METHODS[cur[4]]
                    start_time = time.time()
                    preview = apply_resize_preview(single_mask, cur[0], cur[1], cur[2], cur[3], cur_method, cur[5], cur[6])
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(preview, uid)

                    final = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final = _get_params(uid, resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_mask)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, resize_by, width, height, multiplier, interpolation, fit_mode, enhanced_visibility)
                        cur_method = self.INTERPOLATION_METHODS[cur[4]]
                        start_time = time.time()
                        preview = apply_resize_preview(single_mask, cur[0], cur[1], cur[2], cur[3], cur_method, cur[5], cur[6])
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        final_method = self.INTERPOLATION_METHODS[final[4]]
                        result_list.append(apply_resize(single_mask, final[0], final[1], final[2], final[3], final_method, final[5]))

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_resize(mask, resize_by, width, height, multiplier, interp_method, fit_mode)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"MaskResize": MaskResizeC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskResize": "Mask Resize"}
