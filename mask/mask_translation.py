import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, translate_x: int, translate_y: int) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (translate_x, translate_y)

def _get_params(node_id: str, translate_x: int, translate_y: int) -> tuple[int, int]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (translate_x, translate_y))

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
        translated = cv2.warpAffine(mask_uint8, translation_matrix, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        mask_float = translated.astype(np.float32) / 255.0
        results.append(mask_float)
    
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
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "translate"
    CATEGORY = "WtlNodes/mask"

    def translate(self, mask, translate_x, translate_y, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                while True:
                    cur_tx, cur_ty = _get_params(uid, translate_x, translate_y)
                    cur_mask = apply_translation(mask, cur_tx, cur_ty)
                    
                    # Convert mask to image for preview
                    preview_image = cur_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    _send_ram_preview(preview_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        translate_x, translate_y = cur_tx, cur_ty
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
                        cur_tx, cur_ty = _get_params(uid, translate_x, translate_y)
                        cur_mask = apply_translation(single_mask, cur_tx, cur_ty)
                        
                        # Convert mask to image for preview
                        preview_image = cur_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
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