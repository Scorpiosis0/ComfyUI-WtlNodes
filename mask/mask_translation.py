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
        entry["params"] = (translate_x, translate_y, enhanced_visibility)

def _get_params(node_id: str, translate_x: int, translate_y: int, enhanced_visibility: bool) -> tuple[int, int, bool]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (translate_x, translate_y, enhanced_visibility))

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
    batch_size = mask_np.shape[0]
    results = []
    
    for b in range(batch_size):
        m = mask_np[b]
        h, w = m.shape[:2]
        
        mask_uint8 = (m * 255).astype(np.uint8)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(mask_uint8, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        mask_float = translated.astype(np.float32) / 255.0
        results.append(mask_float)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_translation_preview(mask, tx, ty, enhanced_visibility):
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    results = []
    
    for b in range(batch_size):
        m = mask_np[b]
        h, w = m.shape[:2]
        
        mask_uint8 = (m * 255).astype(np.uint8)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(mask_uint8, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Validity mask for padding detection
        ones = np.ones_like(mask_uint8, dtype=np.uint8) * 255
        valid = cv2.warpAffine(ones, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Convert to RGB
        rgb = cv2.cvtColor(translated, cv2.COLOR_GRAY2RGB)
        
        # Highlight padding in red if enhanced visibility is on
        if enhanced_visibility:
            red_bg = np.array([255, 0, 0], dtype=np.uint8)
            padding_pixels = (valid == 0)
            rgb[padding_pixels] = red_bg
        
        results.append(rgb.astype(np.float32) / 255.0)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

class MaskTranslationC:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
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
                "enhanced_visibility": ("BOOLEAN", {"default": False}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
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
                while True:
                    cur_tx, cur_ty, cur_enh = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                    
                    # PREVIEW (RED/BLACK)
                    preview_image = apply_translation_preview(mask, cur_tx, cur_ty, cur_enh)
                    _send_ram_preview(preview_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        translate_x, translate_y, enhanced_visibility = cur_tx, cur_ty, cur_enh
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (mask,)}
                    
                    time.sleep(0.25)

                result = apply_translation(mask, translate_x, translate_y)

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]
                    
                    while True:
                        cur_tx, cur_ty, cur_enh = _get_params(uid, translate_x, translate_y, enhanced_visibility)
                        
                        preview_image = apply_translation_preview(single_mask, cur_tx, cur_ty, cur_enh)
                        _send_ram_preview(preview_image, uid)

                        if _check_and_clear_flag(uid, "apply"):
                            final_tx, final_ty = cur_tx, cur_ty
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_mask)
                            final_tx = None
                            break
                        
                        time.sleep(0.25)

                    if final_tx is not None:
                        processed = apply_translation(single_mask, final_tx, final_ty)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_translation(mask, translate_x, translate_y)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"MaskTranslation": MaskTranslationC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskTranslation": "Mask Translation"}