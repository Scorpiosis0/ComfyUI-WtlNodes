import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, zoom: float, interpolation: str, translate_x: int,
                translate_y: int, enhanced_visibility: bool) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (zoom, interpolation, translate_x, translate_y, enhanced_visibility)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, zoom: float, interpolation: str, translate_x: int,
                translate_y: int, enhanced_visibility: bool) -> tuple[float, str, int, int, bool]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (zoom, interpolation, translate_x, translate_y, enhanced_visibility))

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

def apply_zoom(mask, zoom, interp_method, tx, ty):
    mask_np = mask.cpu().numpy()
    results = []

    for b in range(mask_np.shape[0]):
        m = mask_np[b]
        h, w = m.shape[:2]
        mask_uint8 = (m * 255).astype(np.uint8)

        new_w, new_h = int(w * zoom), int(h * zoom)
        zoomed = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)

        result = np.zeros((h, w), dtype=mask_uint8.dtype)

        if zoom > 1.0:
            # Crop center of zoomed mask, shifted by tx/ty
            src_x = (new_w - w) // 2 - tx
            src_y = (new_h - h) // 2 - ty
            # Safe blit: handle translation that goes outside zoomed bounds
            result = np.zeros((h, w), dtype=mask_uint8.dtype)
            sx1 = max(0, src_x)
            sy1 = max(0, src_y)
            dx1 = max(0, -src_x)
            dy1 = max(0, -src_y)
            cw = min(new_w - sx1, w - dx1)
            ch = min(new_h - sy1, h - dy1)
            if cw > 0 and ch > 0:
                result[dy1:dy1 + ch, dx1:dx1 + cw] = zoomed[sy1:sy1 + ch, sx1:sx1 + cw]
        else:
            # Place zoomed mask centered on canvas, shifted by tx/ty
            dst_x = (w - new_w) // 2 + tx
            dst_y = (h - new_h) // 2 + ty
            src_x1 = max(0, -dst_x)
            src_y1 = max(0, -dst_y)
            dst_x1 = max(0, dst_x)
            dst_y1 = max(0, dst_y)
            copy_w = min(new_w - src_x1, w - dst_x1)
            copy_h = min(new_h - src_y1, h - dst_y1)
            if copy_w > 0 and copy_h > 0:
                result[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] =                     zoomed[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

        results.append(result.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

def apply_zoom_preview(mask, zoom, interp_method, tx, ty, enhanced_visibility):
    mask_np = mask.cpu().numpy()
    results = []

    for b in range(mask_np.shape[0]):
        m = mask_np[b]
        h, w = m.shape[:2]
        mask_uint8 = (m * 255).astype(np.uint8)

        new_w, new_h = int(w * zoom), int(h * zoom)
        zoomed = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)

        result = np.zeros((h, w), dtype=mask_uint8.dtype)
        # Track which pixels are padding (no content)
        valid = np.zeros((h, w), dtype=np.uint8)

        if zoom > 1.0:
            src_x = (new_w - w) // 2 - tx
            src_y = (new_h - h) // 2 - ty
            # Safe blit: handle translation that goes outside zoomed bounds
            result = np.zeros((h, w), dtype=mask_uint8.dtype)
            sx1 = max(0, src_x)
            sy1 = max(0, src_y)
            dx1 = max(0, -src_x)
            dy1 = max(0, -src_y)
            cw = min(new_w - sx1, w - dx1)
            ch = min(new_h - sy1, h - dy1)
            if cw > 0 and ch > 0:
                result[dy1:dy1 + ch, dx1:dx1 + cw] = zoomed[sy1:sy1 + ch, sx1:sx1 + cw]
            valid[dy1:dy1 + ch, dx1:dx1 + cw] = 255
        else:
            dst_x = (w - new_w) // 2 + tx
            dst_y = (h - new_h) // 2 + ty
            src_x1 = max(0, -dst_x)
            src_y1 = max(0, -dst_y)
            dst_x1 = max(0, dst_x)
            dst_y1 = max(0, dst_y)
            copy_w = min(new_w - src_x1, w - dst_x1)
            copy_h = min(new_h - src_y1, h - dst_y1)
            if copy_w > 0 and copy_h > 0:
                result[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] =                     zoomed[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]
                valid[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = 255

        rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        if enhanced_visibility:
            rgb[valid == 0] = np.array([255, 0, 0], dtype=np.uint8)

        results.append(rgb.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

class MaskZoomC:
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
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "translate_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "translate_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()), {"default": "nearest"}),
                "enhanced_visibility": ("BOOLEAN", {"default": False}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "zoom"
    CATEGORY = "WtlNodes/mask"

    def zoom(self, mask, zoom, interpolation, translate_x, translate_y, enhanced_visibility, apply_type, unique_id=None):
        interp_method = self.INTERPOLATION_METHODS[interpolation]

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                cur_method = self.INTERPOLATION_METHODS[cur[1]]
                start_time = time.time()
                preview = apply_zoom_preview(mask, cur[0], cur_method, cur[2], cur[3], cur[4])
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (mask,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                    cur_method = self.INTERPOLATION_METHODS[cur[1]]
                    start_time = time.time()
                    preview = apply_zoom_preview(mask, cur[0], cur_method, cur[2], cur[3], cur[4])
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(preview, uid)

                final_method = self.INTERPOLATION_METHODS[final[1]]
                result = apply_zoom(mask, final[0], final_method, final[2], final[3])

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]

                    cur = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                    cur_method = self.INTERPOLATION_METHODS[cur[1]]
                    start_time = time.time()
                    preview = apply_zoom_preview(single_mask, cur[0], cur_method, cur[2], cur[3], cur[4])
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
                                final = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_mask)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                        cur_method = self.INTERPOLATION_METHODS[cur[1]]
                        start_time = time.time()
                        preview = apply_zoom_preview(single_mask, cur[0], cur_method, cur[2], cur[3], cur[4])
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        final_method = self.INTERPOLATION_METHODS[final[1]]
                        result_list.append(apply_zoom(single_mask, final[0], final_method, final[2], final[3]))

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_zoom(mask, zoom, interp_method, translate_x, translate_y)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"MaskZoom": MaskZoomC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskZoom": "Mask Zoom"}