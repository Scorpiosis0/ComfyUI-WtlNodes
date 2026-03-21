import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, dilate_erode: int, feather: int) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (dilate_erode, feather)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, *defaults: int) -> tuple[int, ...]:
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

def _expand_or_shrink(mask, amount):
    kernel_size = abs(amount) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if amount > 0:
        return cv2.dilate(mask, kernel, iterations=1)
    else:
        return cv2.erode(mask, kernel, iterations=1)

def _feather_mask(mask, radius):
    kernel_size = radius * 2 + 1
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

def apply_mask_processing(mask, dilate_erode, feather):
    mask_np = mask.cpu().numpy()
    results = []

    for b in range(mask_np.shape[0]):
        m = mask_np[b]
        m_uint8 = (m * 255).astype(np.uint8)

        if dilate_erode != 0:
            m_uint8 = _expand_or_shrink(m_uint8, dilate_erode)
        if feather > 0:
            m_uint8 = _feather_mask(m_uint8, feather)

        results.append(m_uint8.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

class MaskProcessorC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "dilate_erode": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "WtlNodes/mask"

    def process_mask(self, mask, dilate_erode, feather, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur_dilate_erode, cur_feather = _get_params(uid, dilate_erode, feather)
                start_time = time.time()
                initial = apply_mask_processing(mask, cur_dilate_erode, cur_feather)
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(initial, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_dilate_erode, final_feather = _get_params(uid, dilate_erode, feather)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (mask,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_dilate_erode, cur_feather = _get_params(uid, dilate_erode, feather)
                    start_time = time.time()
                    cur_processed = apply_mask_processing(mask, cur_dilate_erode, cur_feather)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_processed, uid)

                result = apply_mask_processing(mask, final_dilate_erode, final_feather)

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]

                    cur_dilate_erode, cur_feather = _get_params(uid, dilate_erode, feather)
                    start_time = time.time()
                    initial = apply_mask_processing(single_mask, cur_dilate_erode, cur_feather)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    preview_image = initial.unsqueeze(-1).repeat(1, 1, 1, 3)
                    _send_ram_preview(preview_image, uid)

                    final_dilate_erode = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_dilate_erode, final_feather = _get_params(uid, dilate_erode, feather)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_mask)
                                final_dilate_erode = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_dilate_erode, cur_feather = _get_params(uid, dilate_erode, feather)
                        start_time = time.time()
                        cur_processed = apply_mask_processing(single_mask, cur_dilate_erode, cur_feather)
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        preview_image = cur_processed.unsqueeze(-1).repeat(1, 1, 1, 3)
                        _send_ram_preview(preview_image, uid)

                    if final_dilate_erode is not None:
                        result_list.append(apply_mask_processing(single_mask, final_dilate_erode, final_feather))

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_mask_processing(mask, dilate_erode, feather)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"MaskProcessor": MaskProcessorC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskProcessor": "Mask Processor"}
