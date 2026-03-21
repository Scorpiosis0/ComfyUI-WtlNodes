import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, translate_x: int, translate_y: int, bg_color: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (translate_x, translate_y, bg_color)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, translate_x: int, translate_y: int, bg_color: str) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (translate_x, translate_y, bg_color))

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

def _apply(image, tx, ty, bg_color):
    img_np = image.cpu().numpy()
    bg_value = (255, 255, 255) if bg_color == "white" else (0, 0, 0)
    results = []

    for b in range(img_np.shape[0]):
        img = img_np[b]
        h, w = img.shape[:2]
        img_uint8 = (img * 255).astype(np.uint8)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img_uint8, M, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)
        results.append(translated.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

class ImageTranslationC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "translate_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "translate_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "bg_color": (["black", "white"],),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "translate"
    CATEGORY = "WtlNodes/image"

    def translate(self, image, translate_x, translate_y, bg_color, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur = _get_params(uid, translate_x, translate_y, bg_color)
                t0 = time.time()
                preview = _apply(image, *cur)
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, translate_x, translate_y, bg_color)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, translate_x, translate_y, bg_color)
                    t0 = time.time()
                    preview = _apply(image, *cur)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                result = _apply(image, *final)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single = image[i:i+1]

                    cur = _get_params(uid, translate_x, translate_y, bg_color)
                    t0 = time.time()
                    preview = _apply(single, *cur)
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
                                final = _get_params(uid, translate_x, translate_y, bg_color)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, translate_x, translate_y, bg_color)
                        t0 = time.time()
                        preview = _apply(single, *cur)
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        result_list.append(_apply(single, *final))

                result = torch.cat(result_list, dim=0)
        else:
            result = _apply(image, translate_x, translate_y, bg_color)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"ImageTranslation": ImageTranslationC}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageTranslation": "Image Translation"}
