import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, rotate: float, interpolation: str, fit_mode: str, enhanced_visibility: bool) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (rotate, interpolation, fit_mode, enhanced_visibility)

def _get_params(node_id: str, rotate: float, interpolation: str, fit_mode: str, enhanced_visibility: bool):
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (rotate, interpolation, fit_mode, enhanced_visibility))

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


# Core rotation (always black bg for the final output)
def apply_rotation(mask, angle, interp_method, fit_mode):
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    results = []

    bg_value = 0  # ALWAYS black for final output

    for b in range(batch_size):
        m = mask_np[b]
        h, w = m.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)

        mask_uint8 = (m * 255).astype(np.uint8)

        if fit_mode == "crop":
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "fit":
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            scale = min(w / (w * np.cos(angle_rad) + h * np.sin(angle_rad)),
                        h / (w * np.sin(angle_rad) + h * np.cos(angle_rad)))

            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "adjust":
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad

            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            if angle_rad < 0.001:
                scale = 1.0
            else:
                rotated_w = w * cos_a + h * sin_a
                rotated_h = w * sin_a + h * cos_a
                scale = min(w / rotated_w, h / rotated_h)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), flags=interp_method | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "none":
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)

            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]

            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (new_w, new_h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        results.append(rotated.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()


# Preview rotation: background may be red
def apply_rotation_preview(mask, angle, interp_method, fit_mode, enhanced_visibility):
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    results = []

    for b in range(batch_size):
        m = mask_np[b]
        h, w = m.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)

        mask_uint8 = (m * 255).astype(np.uint8)
        bg_value = 0  # Black background for rotated mask

        # Apply fit_mode logic
        if fit_mode == "crop":
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "fit":
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            scale = min(w / (w * np.cos(angle_rad) + h * np.sin(angle_rad)),
                        h / (w * np.sin(angle_rad) + h * np.cos(angle_rad)))
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "adjust":
            # Preview: just do standard rotation without scaling
            angle_rad = np.radians(abs(angle) % 180)
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad

            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            if angle_rad < 0.001:
                scale = 1.0
            else:
                rotated_w = w * cos_a + h * sin_a
                rotated_h = w * sin_a + h * cos_a
                scale = min(w / rotated_w, h / rotated_h)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (w, h), flags=interp_method | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        elif fit_mode == "none":
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            rotated = cv2.warpAffine(mask_uint8, rotation_matrix, (new_w, new_h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)

        # Validity mask for padding detection
        ones = np.ones_like(mask_uint8, dtype=np.uint8) * 255
        valid = cv2.warpAffine(ones, rotation_matrix, rotated.shape[::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Convert rotated mask to RGB for preview
        rgb = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)

        # Highlight padding in red if enhanced visibility is on
        if enhanced_visibility and fit_mode in ["crop", "fit", "none"]:
            red_bg = np.array([255, 0, 0], dtype=np.uint8)
            padding_pixels = (valid == 0)
            rgb[padding_pixels] = red_bg

        results.append(rgb.astype(np.float32) / 255.0)

    return torch.from_numpy(np.stack(results)).float()

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
                    "step": 1.0,
                    "round": 0.1,
                }),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()), {"default": "nearest"}),
                "fit_mode": (["crop", "fit", "adjust", "none"],),
                "enhanced_visibility": ("BOOLEAN", {"default": False}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "rotate"
    CATEGORY = "WtlNodes/mask"

    def rotate(self, mask, rotate, interpolation, fit_mode, enhanced_visibility, apply_type, unique_id=None):
        interp_method = self.INTERPOLATION_METHODS[interpolation]

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (mask,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                while True:
                    cur_rotate, cur_interp, cur_fit, cur_enh = _get_params(uid, rotate, interpolation, fit_mode, enhanced_visibility)
                    cur_method = self.INTERPOLATION_METHODS[cur_interp]

                    # PREVIEW (RED/BLACK)
                    preview_mask = apply_rotation_preview(mask, cur_rotate, cur_method, cur_fit, cur_enh)
                    _send_ram_preview(preview_mask, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        rotate, interpolation, fit_mode, enhanced_visibility = cur_rotate, cur_interp, cur_fit, cur_enh
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
                        cur_rotate, cur_interp, cur_fit, cur_enh = _get_params(uid, rotate, interpolation, fit_mode, enhanced_visibility)
                        cur_method = self.INTERPOLATION_METHODS[cur_interp]

                        preview_mask = apply_rotation_preview(single_mask, cur_rotate, cur_method, cur_fit, cur_enh)
                        _send_ram_preview(preview_mask, uid)

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