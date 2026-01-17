import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, zoom: float, interpolation: str, translate_x: int, translate_y: int, enhanced_visibility: bool) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (zoom, interpolation, translate_x, translate_y, enhanced_visibility)

def _get_params(node_id: str, zoom: float, interpolation: str, translate_x: int, translate_y: int, enhanced_visibility: bool) -> tuple[float, str, int, int, bool]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (zoom, interpolation, translate_x, translate_y, enhanced_visibility))

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

def apply_zoom(mask, zoom, interp_method, tx, ty):
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    results = []
    
    for b in range(batch_size):
        m = mask_np[b]
        h, w = m.shape[:2]
        
        mask_uint8 = (m * 255).astype(np.uint8)
        
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        
        zoomed = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)

        # --- Translation ---
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(zoomed, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        if zoom > 1.0:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result_mask = zoomed[start_y:start_y + h, start_x:start_x + w]
        else:
            result_mask = np.zeros((h, w), dtype=mask_uint8.dtype)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result_mask[start_y:start_y + new_h, start_x:start_x + new_w] = zoomed
        
        mask_float = translated.astype(np.float32) / 255.0
        results.append(mask_float)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_zoom_preview(mask, zoom, interp_method, tx, ty, enhanced_visibility):
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    results = []
    
    for b in range(batch_size):
        m = mask_np[b]
        h, w = m.shape[:2]
        
        mask_uint8 = (m * 255).astype(np.uint8)
        
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        
        zoomed = cv2.resize(mask_uint8, (new_w, new_h), interpolation=interp_method)

        # --- Translation ---
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(zoomed, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Validity mask for padding detection
        ones = np.ones_like(zoomed, dtype=np.uint8) * 255
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

class MaskZoomC:
    
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
                "zoom": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01
                }),
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
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()),{
                    "default": "nearest",
                }),
                "enhanced_visibility": ("BOOLEAN", {"default": False}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "zoom"
    CATEGORY = "WtlNodes/mask"

    def zoom(self, mask, zoom, interpolation, translate_x, translate_y, enhanced_visibility, apply_type, unique_id=None):
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
                    cur_zoom, cur_interp, cur_tx, cur_ty, cur_enh = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]
                    
                    # PREVIEW (RED/BLACK)
                    preview_image = apply_zoom_preview(mask, cur_zoom, cur_method, cur_tx, cur_ty, cur_enh)
                    _send_ram_preview(preview_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        zoom, interpolation, translate_x, translate_y, enhanced_visibility = cur_zoom, cur_interp, cur_tx, cur_ty, cur_enh
                        interp_method = cur_method
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (mask,)}
                    
                    time.sleep(0.25)

                result = apply_zoom(mask, zoom, interp_method, translate_x, translate_y)

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]
                    
                    while True:
                        cur_zoom, cur_interp, cur_tx, cur_ty, cur_enh = _get_params(uid, zoom, interpolation, translate_x, translate_y, enhanced_visibility)
                        cur_method = self.INTERPOLATION_METHODS[cur_interp]
                        
                        preview_image = apply_zoom_preview(single_mask, cur_zoom, cur_method, cur_tx, cur_ty, cur_enh)
                        _send_ram_preview(preview_image, uid)

                        if _check_and_clear_flag(uid, "apply"):
                            final_zoom, final_interp, final_tx, final_ty = cur_zoom, cur_interp, cur_tx, cur_ty
                            final_method = cur_method
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_mask)
                            final_zoom = None
                            break
                        
                        time.sleep(0.25)

                    if final_zoom is not None:
                        processed = apply_zoom(single_mask, final_zoom, final_method, final_tx, final_ty)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_zoom(mask, zoom, interp_method, translate_x, translate_y)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"MaskZoom": MaskZoomC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskZoom": "Mask Zoom"}