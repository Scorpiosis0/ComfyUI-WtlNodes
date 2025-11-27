import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, rotate: float, interpolation: str, fit_mode: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (rotate, interpolation, fit_mode)

def _get_params(node_id: str, rotate: float, interpolation: str, fit_mode: str) -> tuple[float, str, str]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (rotate, interpolation, fit_mode))

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

def apply_rotation(mask, angle, interp_method, fit_mode):
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    results = []
    
    for b in range(batch_size):
        m = mask_np[b]
        h, w = m.shape[:2]
        center = (w // 2, h // 2)
        
        mask_uint8 = (m * 255).astype(np.uint8)
        
        if fit_mode == "crop":
            # Normal rotation - crop what goes outside
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), 
                                    flags=interp_method, borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))
            mask_float = rotated.astype(np.float32) / 255.0
            results.append(mask_float)
            
        elif fit_mode == "fit":
            # Scale down so entire rotated mask fits within original dimensions
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            
            # Calculate scale to fit entire rotated mask in original canvas
            scale = min(w / (w * np.cos(angle_rad) + h * np.sin(angle_rad)),
                       h / (w * np.sin(angle_rad) + h * np.cos(angle_rad)))
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), 
                                    flags=interp_method, borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))
            
            mask_float = rotated.astype(np.float32) / 255.0
            results.append(mask_float)
            
        elif fit_mode == "adjust":
            # Scale up to fill canvas with no black background
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            
            # Calculate scale factor to eliminate black corners
            if w >= h:
                scale = 1.0 / (np.cos(angle_rad) + np.sin(angle_rad) * h / w)
            else:
                scale = 1.0 / (np.cos(angle_rad) + np.sin(angle_rad) * w / h)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), 
                                    flags=interp_method, borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))
            
            mask_float = rotated.astype(np.float32) / 255.0
            results.append(mask_float)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

class MaskRotationC:
    
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
                "mask": ("MASK",),
                "rotate": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 0.1
                }),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()),{
                    "default": "nearest",
                }),
                "fit_mode": (["crop", "adjust", "fit"],),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "rotate"
    CATEGORY = "WtlNodes/mask"

    def rotate(self, mask, rotate, interpolation, fit_mode, apply_type, unique_id=None):
        interp_method = self.INTERPOLATION_METHODS[interpolation]
        
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                while True:
                    cur_rotate, cur_interp, cur_fit = _get_params(uid, rotate, interpolation, fit_mode)
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]
                    cur_mask = apply_rotation(mask, cur_rotate, cur_method, cur_fit)
                    
                    # Convert mask to image for preview
                    preview_image = cur_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    _send_ram_preview(preview_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        rotate, interpolation, fit_mode = cur_rotate, cur_interp, cur_fit
                        interp_method = cur_method
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (mask,)}
                    
                    time.sleep(0.25)

                result = apply_rotation(mask, rotate, interp_method, fit_mode)

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]
                    
                    while True:
                        cur_rotate, cur_interp, cur_fit = _get_params(uid, rotate, interpolation, fit_mode)
                        cur_method = self.INTERPOLATION_METHODS[cur_interp]
                        cur_mask = apply_rotation(single_mask, cur_rotate, cur_method, cur_fit)
                        
                        # Convert mask to image for preview
                        preview_image = cur_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                        _send_ram_preview(preview_image, uid)

                        if _check_and_clear_flag(uid, "apply"):
                            final_rotate, final_interp, final_fit = cur_rotate, cur_interp, cur_fit
                            final_method = cur_method
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_mask)
                            final_rotate = None
                            break
                        
                        time.sleep(0.25)

                    if final_rotate is not None:
                        processed = apply_rotation(single_mask, final_rotate, final_method, final_fit)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_rotation(mask, rotate, interp_method, fit_mode)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"MaskRotation": MaskRotationC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskRotation": "Mask Rotation"}