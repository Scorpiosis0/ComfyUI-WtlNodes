import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, resize_by: bool, width: int, height: int, multiplier: float, interpolation: str, fit_mode: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (resize_by, width, height, multiplier, interpolation, fit_mode)

def _get_params(node_id: str, resize_by: bool, width: int, height: int, multiplier: float, interpolation: str, fit_mode: str) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (resize_by, width, height, multiplier, interpolation, fit_mode))

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

def apply_resize(image, resize_by, width, height, multiplier, interp_method, fit_mode):
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        orig_h, orig_w = img.shape[:2]
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if resize_by:
            target_w = max(1, int(orig_w * multiplier))
            target_h = max(1, int(orig_h * multiplier))
        else:
            target_w = max(1, width)
            target_h = max(1, height)
        
        h, w = img_uint8.shape[:2]
        
        if fit_mode == "adjust":
            result_img = cv2.resize(img_uint8, (target_w, target_h), interpolation=interp_method)
        
        elif fit_mode == "crop":
            aspect_ratio = w / h
            target_aspect = target_w / target_h
            
            if aspect_ratio > target_aspect:
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            else:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            
            resized = cv2.resize(img_uint8, (new_w, new_h), interpolation=interp_method)
            
            start_x = (new_w - target_w) // 2
            start_y = (new_h - target_h) // 2
            
            result_img = resized[start_y:start_y + target_h, start_x:start_x + target_w]
        
        elif fit_mode == "fit":
            aspect_ratio = w / h
            target_aspect = target_w / target_h
            
            if aspect_ratio > target_aspect:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            
            resized = cv2.resize(img_uint8, (new_w, new_h), interpolation=interp_method)
            
            result_img = np.zeros((target_h, target_w, img_uint8.shape[2]), dtype=img_uint8.dtype)
            
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            
            result_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        img_float = result_img.astype(np.float32) / 255.0
        results.append(img_float)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

class ImageResizeC:
    
    INTERPOLATION_METHODS = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_by": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Multiplier",
                    "label_off": "Absolute"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1
                }),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()),{
                    "default": "bilinear",
                }),
                "fit_mode": (["crop", "adjust", "fit"],),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "WtlNodes/image"

    def resize(self, image, resize_by, width, height, multiplier, interpolation, fit_mode, 
               apply_type, unique_id=None):
        interp_method = self.INTERPOLATION_METHODS[interpolation]
        
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                while True:
                    cur_params = _get_params(uid, resize_by, width, height, multiplier, 
                                            interpolation, fit_mode)
                    cur_resize_by, cur_w, cur_h, cur_mult, cur_interp, cur_fit = cur_params
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]
                    cur_image = apply_resize(image, cur_resize_by, cur_w, cur_h, cur_mult, 
                                            cur_method, cur_fit)
                    _send_ram_preview(cur_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        resize_by, width, height = cur_resize_by, cur_w, cur_h
                        multiplier, interpolation, fit_mode = cur_mult, cur_interp, cur_fit
                        interp_method = cur_method
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (image,)}
                    
                    time.sleep(0.25)

                result = apply_resize(image, resize_by, width, height, multiplier, 
                                     interp_method, fit_mode)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]
                    
                    while True:
                        cur_params = _get_params(uid, resize_by, width, height, multiplier, 
                                                interpolation, fit_mode)
                        cur_resize_by, cur_w, cur_h, cur_mult, cur_interp, cur_fit = cur_params
                        cur_method = self.INTERPOLATION_METHODS[cur_interp]
                        cur_image = apply_resize(single_image, cur_resize_by, cur_w, cur_h, 
                                                cur_mult, cur_method, cur_fit)
                        _send_ram_preview(cur_image, uid)

                        if _check_and_clear_flag(uid, "apply"):
                            final_params = (cur_resize_by, cur_w, cur_h, cur_mult, 
                                          cur_interp, cur_fit, cur_method)
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_image)
                            final_params = None
                            break
                        
                        time.sleep(0.25)

                    if final_params is not None:
                        f_resize_by, f_w, f_h, f_mult, f_interp, f_fit, f_method = final_params
                        processed = apply_resize(single_image, f_resize_by, f_w, f_h, 
                                               f_mult, f_method, f_fit)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_resize(image, resize_by, width, height, multiplier, 
                                 interp_method, fit_mode)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"ImageResize": ImageResizeC}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageResize": "Image Resize"}