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

def apply_translation(image, tx, ty):
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w = img.shape[:2]
        
        img_uint8 = (img * 255).astype(np.uint8)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img_uint8, translation_matrix, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        img_float = translated.astype(np.float32) / 255.0
        results.append(img_float)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

class ImageTranslationC:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "translate"
    CATEGORY = "WtlNodes/image"

    def translate(self, image, translate_x, translate_y, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                while True:
                    cur_tx, cur_ty = _get_params(uid, translate_x, translate_y)
                    cur_image = apply_translation(image, cur_tx, cur_ty)
                    _send_ram_preview(cur_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        translate_x, translate_y = cur_tx, cur_ty
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (image,)}
                    
                    time.sleep(0.25)

                result = apply_translation(image, translate_x, translate_y)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]
                    
                    while True:
                        cur_tx, cur_ty = _get_params(uid, translate_x, translate_y)
                        cur_image = apply_translation(single_image, cur_tx, cur_ty)
                        _send_ram_preview(cur_image, uid)

                        if _check_and_clear_flag(uid, "apply"):
                            final_tx, final_ty = cur_tx, cur_ty
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_image)
                            final_tx = None
                            break
                        
                        time.sleep(0.25)

                    if final_tx is not None:
                        processed = apply_translation(single_image, final_tx, final_ty)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_translation(image, translate_x, translate_y)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"ImageTranslation": ImageTranslationC}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageTranslation": "Image Translation"}