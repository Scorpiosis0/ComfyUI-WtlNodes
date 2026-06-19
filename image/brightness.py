import torch
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, brightness: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (brightness,)
        old_params = entry.get("params")
        if old_params != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, brightness: float) -> tuple[float]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (brightness,))

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

def _apply_brightness(image, brightness):
    result = image * (1 + brightness / 100)
    return torch.clamp(result, 0.0, 1.0)

class BrightnessC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "brightness"
    CATEGORY = "WtlNodes/image"

    def brightness(self, image, brightness, apply_type, unique_id=None):

        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                # Send initial preview with current parameters
                cur_brightness = _get_params(uid, brightness)[0]
                start_time = time.time()
                initial_image = _apply_brightness(image, cur_brightness)
                processing_ms = int((time.time() - start_time) * 1000)
                _set_processing_time(uid, processing_ms)
                _send_ram_preview(initial_image, uid)

                while True:
                    # Wait for parameter change or button press
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_brightness = _get_params(uid, brightness)[0]
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    # If apply button was pressed, exit loop
                    if not triggered:
                        break

                    # Get current params and process
                    cur_brightness = _get_params(uid, brightness)[0]
                    start_time = time.time()
                    cur_image = _apply_brightness(image, cur_brightness)
                    processing_ms = int((time.time() - start_time) * 1000)
                    _set_processing_time(uid, processing_ms)
                    _send_ram_preview(cur_image, uid)

                # Apply final effect after exiting loop
                result = _apply_brightness(image, final_brightness)
                result = torch.clamp(result, 0.0, 1.0)

            else:
                # Process images one by one
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]

                    # Send initial preview with current parameters
                    cur_brightness = _get_params(uid, brightness)[0]
                    start_time = time.time()
                    initial_image = _apply_brightness(single_image, cur_brightness)
                    processing_ms = int((time.time() - start_time) * 1000)
                    _set_processing_time(uid, processing_ms)
                    _send_ram_preview(initial_image, uid)

                    while True:
                        # Wait for parameter change or button press
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_brightness = _get_params(uid, brightness)[0]
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_image)
                                final_brightness = None
                                break
                            time.sleep(0.05)

                        # If apply or skip button was pressed, exit loop
                        if not triggered:
                            break

                        # Get current params and process
                        cur_brightness = _get_params(uid, brightness)[0]
                        start_time = time.time()
                        cur_image = _apply_brightness(single_image, cur_brightness)
                        processing_ms = int((time.time() - start_time) * 1000)
                        _set_processing_time(uid, processing_ms)
                        _send_ram_preview(cur_image, uid)

                    # Apply final effect for this image if not skipped
                    if final_brightness is not None:
                        processed = _apply_brightness(single_image, final_brightness)
                        result_list.append(processed)

                # Concatenate all processed images back into a batch
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode
            result = _apply_brightness(image, brightness)
            result = torch.clamp(result, 0.0, 1.0)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"Brightness": BrightnessC}
NODE_DISPLAY_NAME_MAPPINGS = {"Brightness": "Brightness"}