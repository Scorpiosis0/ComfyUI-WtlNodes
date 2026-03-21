import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, zoom: float, interpolation: str, translate_x: int, translate_y: int, bg_color: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (zoom, interpolation, translate_x, translate_y, bg_color)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, zoom: float, interpolation: str, translate_x: int, translate_y: int, bg_color: str) -> tuple[float, str, int, int, str]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (zoom, interpolation, translate_x, translate_y, bg_color))

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


def apply_zoom_translate(image, zoom, interp_method, tx, ty, bg_color):
    img_np = image.cpu().numpy()
    results = []

    bg_fill = 255 if bg_color == "white" else 0

    for b in range(img_np.shape[0]):
        img = img_np[b]
        h, w = img.shape[:2]
        img_uint8 = (img * 255).astype(np.uint8)

        new_w = int(w * zoom)
        new_h = int(h * zoom)
        zoomed = cv2.resize(img_uint8, (new_w, new_h), interpolation=interp_method)

        # Allocate output canvas with background fill
        result = np.full((h, w, img.shape[2]), bg_fill, dtype=img_uint8.dtype)

        if zoom > 1.0:
            # Crop center of zoomed image, shifted by tx/ty
            src_x = (new_w - w) // 2 - tx
            src_y = (new_h - h) // 2 - ty
            # Safe blit: handle translation that goes outside zoomed bounds
            bg_fill_val = 255 if bg_color == "white" else 0
            result = np.full((h, w, img.shape[2]), bg_fill_val, dtype=img_uint8.dtype)
            sx1 = max(0, src_x)
            sy1 = max(0, src_y)
            dx1 = max(0, -src_x)
            dy1 = max(0, -src_y)
            cw = min(new_w - sx1, w - dx1)
            ch = min(new_h - sy1, h - dy1)
            if cw > 0 and ch > 0:
                result[dy1:dy1 + ch, dx1:dx1 + cw] = zoomed[sy1:sy1 + ch, sx1:sx1 + cw]
        else:
            # Place zoomed image centered on canvas, shifted by tx/ty
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

class ImageZoomC:
    
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
                "image": ("IMAGE",),
                "zoom": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01
                }),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()),{
                    "default": "bilinear",
                }),
                "translate_x": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1
                }),
                "translate_y": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1
                }),
                "bg_color": (["black", "white"],),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "zoom_translate"
    CATEGORY = "WtlNodes/image"

    def zoom_translate(self, image, zoom, interpolation, translate_x, translate_y, bg_color, apply_type, unique_id=None):
        interp_method = self.INTERPOLATION_METHODS[interpolation]

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(uid, "skip"):
            return {"result": (image,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur_zoom, cur_interp, cur_tx, cur_ty, cur_bg = _get_params(uid, zoom, interpolation, translate_x, translate_y, bg_color)
                cur_method = self.INTERPOLATION_METHODS[cur_interp]
                start_time = time.time()
                initial_image = apply_zoom_translate(image, cur_zoom, cur_method, cur_tx, cur_ty, cur_bg)
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(initial_image, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_zoom, final_interp, final_tx, final_ty, final_bg = _get_params(uid, zoom, interpolation, translate_x, translate_y, bg_color)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_zoom, cur_interp, cur_tx, cur_ty, cur_bg = _get_params(uid, zoom, interpolation, translate_x, translate_y, bg_color)
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]
                    start_time = time.time()
                    cur_image = apply_zoom_translate(image, cur_zoom, cur_method, cur_tx, cur_ty, cur_bg)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_image, uid)

                final_method = self.INTERPOLATION_METHODS[final_interp]
                result = apply_zoom_translate(image, final_zoom, final_method, final_tx, final_ty, final_bg)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]

                    cur_zoom, cur_interp, cur_tx, cur_ty, cur_bg = _get_params(uid, zoom, interpolation, translate_x, translate_y, bg_color)
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]
                    start_time = time.time()
                    initial_image = apply_zoom_translate(single_image, cur_zoom, cur_method, cur_tx, cur_ty, cur_bg)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(initial_image, uid)

                    final_zoom = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_zoom, final_interp, final_tx, final_ty, final_bg = _get_params(uid, zoom, interpolation, translate_x, translate_y, bg_color)
                                final_method = self.INTERPOLATION_METHODS[final_interp]
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_image)
                                final_zoom = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_zoom, cur_interp, cur_tx, cur_ty, cur_bg = _get_params(uid, zoom, interpolation, translate_x, translate_y, bg_color)
                        cur_method = self.INTERPOLATION_METHODS[cur_interp]
                        start_time = time.time()
                        cur_image = apply_zoom_translate(single_image, cur_zoom, cur_method, cur_tx, cur_ty, cur_bg)
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(cur_image, uid)

                    if final_zoom is not None:
                        result_list.append(apply_zoom_translate(single_image, final_zoom, final_method, final_tx, final_ty, final_bg))

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_zoom_translate(image, zoom, interp_method, translate_x, translate_y, bg_color)

        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"ImageZoom": ImageZoomC}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageZoom": "Image Zoom"}