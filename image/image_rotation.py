import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, rotate: float, interpolation: str, fit_mode: str, bg_color: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (rotate, interpolation, fit_mode, bg_color)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, rotate: float, interpolation: str, fit_mode: str, bg_color: str) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (rotate, interpolation, fit_mode, bg_color))

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

def _apply(image, angle, interp_method, fit_mode, bg_color):
    img_np = image.cpu().numpy()
    bg_value = (255, 255, 255) if bg_color == "white" else (0, 0, 0)
    results = []

    for b in range(img_np.shape[0]):
        img = img_np[b]
        h, w = img.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        img_uint8 = (img * 255).astype(np.uint8)

        if fit_mode == "crop":
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated = cv2.warpAffine(img_uint8, M, (w, h), flags=interp_method,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "fit":
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            scale = min(w / (w * np.cos(angle_rad) + h * np.sin(angle_rad)),
                        h / (w * np.sin(angle_rad) + h * np.cos(angle_rad)))
            M = cv2.getRotationMatrix2D(center, -angle, scale)
            rotated = cv2.warpAffine(img_uint8, M, (w, h), flags=interp_method,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "adjust":
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            if angle_rad < 0.001:
                scale = 1.0
            else:
                scale = min(w / (w * np.cos(angle_rad) + h * np.sin(angle_rad)),
                            h / (w * np.sin(angle_rad) + h * np.cos(angle_rad)))
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(img_uint8, M, (w, h),
                                     flags=interp_method | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "none":
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += new_w / 2 - center[0]
            M[1, 2] += new_h / 2 - center[1]
            rotated = cv2.warpAffine(img_uint8, M, (new_w, new_h), flags=interp_method,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        results.append(rotated.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

class ImageRotationC:
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
                "rotate": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0, "round": 0.1}),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()), {"default": "bilinear"}),
                "fit_mode": (["crop", "fit", "adjust", "none"],),
                "bg_color": (["black", "white"],),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate"
    CATEGORY = "WtlNodes/image"

    def rotate(self, image, rotate, interpolation, fit_mode, bg_color, apply_type, unique_id=None):
        interp_method = self.INTERPOLATION_METHODS[interpolation]

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                t0 = time.time()
                preview = _apply(image, cur[0], self.INTERPOLATION_METHODS[cur[1]], cur[2], cur[3])
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                    t0 = time.time()
                    preview = _apply(image, cur[0], self.INTERPOLATION_METHODS[cur[1]], cur[2], cur[3])
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                result = _apply(image, final[0], self.INTERPOLATION_METHODS[final[1]], final[2], final[3])

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single = image[i:i+1]

                    cur = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                    t0 = time.time()
                    preview = _apply(single, cur[0], self.INTERPOLATION_METHODS[cur[1]], cur[2], cur[3])
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                    final = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                        t0 = time.time()
                        preview = _apply(single, cur[0], self.INTERPOLATION_METHODS[cur[1]], cur[2], cur[3])
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        result_list.append(_apply(single, final[0], self.INTERPOLATION_METHODS[final[1]], final[2], final[3]))

                result = torch.cat(result_list, dim=0)
        else:
            result = _apply(image, rotate, interp_method, fit_mode, bg_color)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"ImageRotation": ImageRotationC}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageRotation": "Image Rotation"}
