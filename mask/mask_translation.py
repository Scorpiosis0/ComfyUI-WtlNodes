import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, translate_x: int, translate_y: int, enhanced_visibility: bool) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (translate_x, translate_y, enhanced_visibility)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, translate_x: int, translate_y: int, enhanced_visibility: bool) -> tuple[int, int, bool]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (translate_x, translate_y, enhanced_visibility))

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

def apply_translation(mask, tx, ty):
    mask_np = mask.cpu().numpy()
    results = []

    for b in range(mask_np.shape[0]):
        m = mask_np[b]
        h, w = m.shape[:2]
        mask_uint8 = (m * 255).astype(np.uint8)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(mask_uint8, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        results.append(translated.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

def apply_translation_preview(mask, tx, ty, enhanced_visibility):
    mask_np = mask.cpu().numpy()
    results = []

    for b in range(mask_np.shape[0]):
        m = mask_np[b]
        h, w = m.shape[:2]
        mask_uint8 = (m * 255).astype(np.uint8)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(mask_uint8, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        ones = np.ones_like(mask_uint8, dtype=np.uint8) * 255
        valid = cv2.warpAffine(ones, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        rgb = cv2.cvtColor(translated, cv2.COLOR_GRAY2RGB)
        if enhanced_visibility:
            rgb[valid == 0] = np.array([255, 0, 0], dtype=np.uint8)

        results.append(rgb.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

class MaskTranslationC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "translate_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "translate_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "enhanced_visibility": ("BOOLEAN", {"default": False}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "translate"
    CATEGORY = "WtlNodes/mask"

    def translate(self, mask, translate_x, translate_y, enhanced_visibility, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur_tx, cur_ty, cur_enh = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                start_time = time.time()
                preview = apply_translation_preview(mask, cur_tx, cur_ty, cur_enh)
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_tx, final_ty, _ = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (mask,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_tx, cur_ty, cur_enh = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                    start_time = time.time()
                    preview = apply_translation_preview(mask, cur_tx, cur_ty, cur_enh)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(preview, uid)

                result = apply_translation(mask, final_tx, final_ty)

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]

                    cur_tx, cur_ty, cur_enh = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                    start_time = time.time()
                    preview = apply_translation_preview(single_mask, cur_tx, cur_ty, cur_enh)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(preview, uid)

                    final_tx = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_tx, final_ty, _ = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_mask)
                                final_tx = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_tx, cur_ty, cur_enh = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                        start_time = time.time()
                        preview = apply_translation_preview(single_mask, cur_tx, cur_ty, cur_enh)
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(preview, uid)

                    if final_tx is not None:
                        result_list.append(apply_translation(single_mask, final_tx, final_ty))

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_translation(mask, translate_x, translate_y)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"MaskTranslation": MaskTranslationC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskTranslation": "Mask Translation"}
