import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, zoom: float, interpolation: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (zoom, interpolation)

def _get_params(node_id: str, zoom: float, interpolation: str) -> tuple[float, str]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (zoom, interpolation))

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

def apply_zoom(mask, zoom, interp_method):
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
        
        if zoom > 1.0:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result_mask = zoomed[start_y:start_y + h, start_x:start_x + w]
        else:
            result_mask = np.zeros((h, w), dtype=mask_uint8.dtype)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result_mask[start_y:start_y + new_h, start_x:start_x + new_w] = zoomed
        
        mask_float = result_mask.astype(np.float32) / 255.0
        results.append(mask_float)
    
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
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()),{
                    "default": "nearest",
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "zoom"
    CATEGORY = "WtlNodes/mask"

    def zoom(self, mask, zoom, interpolation, apply_type, unique_id=None):
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
                    cur_zoom, cur_interp = _get_params(uid, zoom, interpolation)
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]
                    cur_mask = apply_zoom(mask, cur_zoom, cur_method)
                    
                    # Convert mask to image for preview
                    preview_image = cur_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    _send_ram_preview(preview_image, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        zoom, interpolation = cur_zoom, cur_interp
                        interp_method = cur_method
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (mask,)}
                    
                    time.sleep(0.25)

                result = apply_zoom(mask, zoom, interp_method)

            else:
                batch_size = mask.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = mask[i:i+1]
                    
                    while True:
                        cur_zoom, cur_interp = _get_params(uid, zoom, interpolation)
                        cur_method = self.INTERPOLATION_METHODS[cur_interp]
                        cur_mask = apply_zoom(single_mask, cur_zoom, cur_method)
                        
                        # Convert mask to image for preview
                        preview_image = cur_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                        _send_ram_preview(preview_image, uid)

                        if _check_and_clear_flag(uid, "apply"):
                            final_zoom, final_interp = cur_zoom, cur_interp
                            final_method = cur_method
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_mask)
                            final_zoom = None
                            break
                        
                        time.sleep(0.25)

                    if final_zoom is not None:
                        processed = apply_zoom(single_mask, final_zoom, final_method)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_zoom(mask, zoom, interp_method)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"MaskZoom": MaskZoomC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskZoom": "Mask Zoom"}