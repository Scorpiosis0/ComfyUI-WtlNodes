import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, offset_x: int, offset_y: int, red_scale: float,
                blue_scale: float, center_x: float, center_y: float, falloff: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, *defaults) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", defaults)

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

def _apply(image, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff):
    img_np = image.cpu().numpy()
    results = []

    for b in range(img_np.shape[0]):
        img = img_np[b]
        h, w, c = img.shape
        img_uint8 = (img * 255).astype(np.uint8)

        if c == 3:
            b_ch, g_ch, r_ch = cv2.split(img_uint8)
        elif c == 4:
            b_ch, g_ch, r_ch, a_ch = cv2.split(img_uint8)
        else:
            results.append(img)
            continue

        center_px_x = center_x * w
        center_px_y = center_y * h
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dx = x_coords - center_px_x
        dy = y_coords - center_px_y
        distance = np.sqrt(dx**2 + dy**2)
        max_distance = np.sqrt((w / 2)**2 + (h / 2)**2)
        distance_norm = distance / max_distance
        strength = np.power(distance_norm, falloff)
        angle = np.arctan2(dy, dx)

        r_offset_x = strength * offset_x * red_scale * np.cos(angle)
        r_offset_y = strength * offset_y * red_scale * np.sin(angle)
        b_offset_x = -strength * offset_x * blue_scale * np.cos(angle)
        b_offset_y = -strength * offset_y * blue_scale * np.sin(angle)

        map_r_x = (x_coords + r_offset_x).astype(np.float32)
        map_r_y = (y_coords + r_offset_y).astype(np.float32)
        map_b_x = (x_coords + b_offset_x).astype(np.float32)
        map_b_y = (y_coords + b_offset_y).astype(np.float32)

        r_displaced = cv2.remap(r_ch, map_r_x, map_r_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        b_displaced = cv2.remap(b_ch, map_b_x, map_b_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        if c == 3:
            result_uint8 = cv2.merge([b_displaced, g_ch, r_displaced])
        else:
            result_uint8 = cv2.merge([b_displaced, g_ch, r_displaced, a_ch])

        results.append(result_uint8.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

class ChromaticAberrationC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset_x": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "red_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01}),
                "blue_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "falloff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "WtlNodes/image"

    def apply_effect(self, image, offset_x, offset_y, red_scale, blue_scale,
                     center_x, center_y, falloff, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur = _get_params(uid, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)
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
                            final = _get_params(uid, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)
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

                    cur = _get_params(uid, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)
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
                                final = _get_params(uid, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)
                        t0 = time.time()
                        preview = _apply(single, *cur)
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        result_list.append(_apply(single, *final))

                result = torch.cat(result_list, dim=0)
        else:
            result = _apply(image, offset_x, offset_y, red_scale, blue_scale, center_x, center_y, falloff)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"ChromaticAberration": ChromaticAberrationC}
NODE_DISPLAY_NAME_MAPPINGS = {"ChromaticAberration": "Chromatic Aberration"}
