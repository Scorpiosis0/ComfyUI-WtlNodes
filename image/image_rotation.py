import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, rotate: float, interpolation: str, fit_mode: str, bg_color: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (rotate, interpolation, fit_mode, bg_color)

def _get_params(node_id: str, rotate: float, interpolation: str, fit_mode: str, bg_color: str) -> tuple[float, str, str, str]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (rotate, interpolation, fit_mode, bg_color))

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

def apply_rotation(image, angle, interp_method, fit_mode, bg_color):
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    # Determine background value
    if bg_color == "white":
        bg_value = (255, 255, 255)
    elif bg_color == "transparent":
        bg_value = (0, 0, 0)  # Will handle alpha separately if needed
    else:  # black
        bg_value = (0, 0, 0)
    
    for b in range(batch_size):
        img = img_np[b]
        h, w = img.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if fit_mode == "crop":
            # Normal rotation - crop what goes outside, can have black corners
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated = cv2.warpAffine(img_uint8, rotation_matrix, (w, h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)
            img_float = rotated.astype(np.float32) / 255.0
            results.append(img_float)
            
        elif fit_mode == "fit":
            # Scale down so entire rotated image fits within original dimensions
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            
            # Calculate scale to fit entire rotated image in original canvas
            scale = min(w / (w * np.cos(angle_rad) + h * np.sin(angle_rad)), h / (w * np.sin(angle_rad) + h * np.cos(angle_rad)))
            
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            rotated = cv2.warpAffine(img_uint8, rotation_matrix, (w, h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)
            
            img_float = rotated.astype(np.float32) / 255.0
            results.append(img_float)
            
        elif fit_mode == "adjust":
            # Fills canvas
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            
            # Calculate scale to fit entire rotated image in original canvas
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            if angle_rad < 0.001:
                scale = 1.0
            else:
                rotated_w = w * cos_a + h * sin_a
                rotated_h = w * sin_a + h * cos_a
                scale = min(w / rotated_w, h / rotated_h)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(img_uint8, rotation_matrix, (w, h), flags=interp_method | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)
            
            img_float = rotated.astype(np.float32) / 255.0
            results.append(img_float)
            
        elif fit_mode == "none":
            # Change canvas size to fit entire rotated image
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            rotated = cv2.warpAffine(img_uint8, rotation_matrix, (new_w, new_h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)
            
            img_float = rotated.astype(np.float32) / 255.0
            results.append(img_float)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

class ImageRotationC:
    
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
                "rotate": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()),{
                    "default": "bilinear",
                }),
                "fit_mode": (["crop", "fit", "adjust", "none"],),
                "bg_color": (["black", "white"],),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate"
    CATEGORY = "WtlNodes/image"

    def rotate(self, image, rotate, interpolation, fit_mode, bg_color, apply_type, unique_id=None):
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
                    cur_rotate, cur_interp, cur_fit, cur_bg = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]
                    cur_image = apply_rotation(image, cur_rotate, cur_method, cur_fit, cur_bg)
                    _send_ram_preview(cur_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        rotate, interpolation, fit_mode, bg_color = cur_rotate, cur_interp, cur_fit, cur_bg
                        interp_method = cur_method
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (image,)}
                    
                    time.sleep(0.25)

                result = apply_rotation(image, rotate, interp_method, fit_mode, bg_color)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]
                    
                    while True:
                        cur_rotate, cur_interp, cur_fit, cur_bg = _get_params(uid, rotate, interpolation, fit_mode, bg_color)
                        cur_method = self.INTERPOLATION_METHODS[cur_interp]
                        cur_image = apply_rotation(single_image, cur_rotate, cur_method, cur_fit, cur_bg)
                        _send_ram_preview(cur_image, uid)

                        if _check_and_clear_flag(uid, "apply"):
                            final_rotate, final_interp, final_fit, final_bg = cur_rotate, cur_interp, cur_fit, cur_bg
                            final_method = cur_method
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_image)
                            final_rotate = None
                            break
                        
                        time.sleep(0.25)

                    if final_rotate is not None:
                        processed = apply_rotation(single_image, final_rotate, final_method, final_fit, final_bg)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_rotation(image, rotate, interp_method, fit_mode, bg_color)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"ImageRotation": ImageRotationC}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageRotate": "Image Rotation"}