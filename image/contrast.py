import torch
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, contrast: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (contrast,)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, contrast: float) -> tuple[float]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (contrast,))

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

def _apply(image, contrast):
    pivot = 0.5
    result = pivot + (image - pivot) * (1 + contrast / 100)
    return torch.clamp(result, 0.0, 1.0)

class ContrastC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "round": 0.1,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "contrast"
    CATEGORY = "WtlNodes/image"

    def contrast(self, image, contrast, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur_contrast = _get_params(uid, contrast)[0]
                t0 = time.time()
                preview = _apply(image, cur_contrast)
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_contrast = _get_params(uid, contrast)[0]
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_contrast = _get_params(uid, contrast)[0]
                    t0 = time.time()
                    preview = _apply(image, cur_contrast)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                result = _apply(image, final_contrast)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single = image[i:i+1]

                    cur_contrast = _get_params(uid, contrast)[0]
                    t0 = time.time()
                    preview = _apply(single, cur_contrast)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                    final_contrast = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_contrast = _get_params(uid, contrast)[0]
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single)
                                final_contrast = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_contrast = _get_params(uid, contrast)[0]
                        t0 = time.time()
                        preview = _apply(single, cur_contrast)
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final_contrast is not None:
                        result_list.append(_apply(single, final_contrast))

                result = torch.cat(result_list, dim=0)
        else:
            result = _apply(image, contrast)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"Contrast": ContrastC}
NODE_DISPLAY_NAME_MAPPINGS = {"Contrast": "Contrast"}
