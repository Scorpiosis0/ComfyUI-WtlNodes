import torch
import math
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, temperature: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (temperature,)
        old_params = entry.get("params")
        if old_params != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, temperature: float) -> tuple[float]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (temperature,))

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

def kelvin_to_rgb(temp_k):
    temp_k = max(1000, min(40000, temp_k))
    temp = temp_k / 100.0

    if temp <= 66:
        red = 255.0
    else:
        red = max(0, min(255, 329.698727446 * ((temp - 60) ** -0.1332047592)))

    if temp <= 66:
        green = max(0, min(255, 99.4708025861 * math.log(temp) - 161.1195681661))
    else:
        green = max(0, min(255, 288.1221695283 * ((temp - 60) ** -0.0755148492)))

    if temp >= 66:
        blue = 255.0
    elif temp <= 19:
        blue = 0.0
    else:
        blue = max(0, min(255, 138.5177312231 * math.log(temp - 10) - 305.0447927307))

    return (red / 255.0, green / 255.0, blue / 255.0)

def _apply_temperature(image, temperature):
    r_mult, g_mult, b_mult = kelvin_to_rgb(temperature)
    r_ref, g_ref, b_ref = kelvin_to_rgb(6500.0)

    r_mult = r_mult / r_ref
    g_mult = g_mult / g_ref
    b_mult = b_mult / b_ref

    multipliers = torch.tensor([r_mult, g_mult, b_mult], dtype=image.dtype, device=image.device)
    adjusted = image * multipliers

    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=image.dtype, device=image.device)
    original_lum = torch.sum(image * luma_weights, dim=-1, keepdim=True)
    adjusted_lum = torch.sum(adjusted * luma_weights, dim=-1, keepdim=True)
    scale = torch.where(adjusted_lum > 0.001, original_lum / adjusted_lum, torch.ones_like(adjusted_lum))

    return torch.clamp(adjusted * scale, 0.0, 1.0)

class ColorTemperatureC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 6500.0,
                    "min": 1000.0,
                    "max": 40000.0,
                    "step": 100.0,
                    "round": 0.1,
                    "display": "temperature (in Kelvin)",
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "WtlNodes/image"

    def apply_effect(self, image, temperature, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur_temperature = _get_params(uid, temperature)[0]
                start_time = time.time()
                initial_image = _apply_temperature(image, cur_temperature)
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(initial_image, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_temperature = _get_params(uid, temperature)[0]
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_temperature = _get_params(uid, temperature)[0]
                    start_time = time.time()
                    cur_image = _apply_temperature(image, cur_temperature)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_image, uid)

                result = _apply_temperature(image, final_temperature)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]

                    cur_temperature = _get_params(uid, temperature)[0]
                    start_time = time.time()
                    initial_image = _apply_temperature(single_image, cur_temperature)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(initial_image, uid)

                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_temperature = _get_params(uid, temperature)[0]
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_image)
                                final_temperature = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_temperature = _get_params(uid, temperature)[0]
                        start_time = time.time()
                        cur_image = _apply_temperature(single_image, cur_temperature)
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(cur_image, uid)

                    if final_temperature is not None:
                        result_list.append(_apply_temperature(single_image, final_temperature))

                result = torch.cat(result_list, dim=0)
        else:
            result = _apply_temperature(image, temperature)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"Temperature": ColorTemperatureC}
NODE_DISPLAY_NAME_MAPPINGS = {"Temperature": "Temperature (Tanner Helland's algorithm)"}
