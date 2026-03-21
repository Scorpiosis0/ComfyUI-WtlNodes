import torch
import threading
import time
import numpy as np
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, dither_method: str, r_levels: int, g_levels: int,
                b_levels: int, dither_scale: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (dither_method, r_levels, g_levels, b_levels, dither_scale)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, dither_method: str, r_levels: int, g_levels: int,
                b_levels: int, dither_scale: float) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (dither_method, r_levels, g_levels, b_levels, dither_scale))

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
        entry.setdefault("flags", {})[flag] = True

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

class DitherC:
    _bayer_matrix_cache = {}
    _blue_noise_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dither_method": (["none", "bayer", "arithmetic_add", "blue_noise"],),
                "r_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "g_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "b_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "dither_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 5.0, "step": 0.25}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dither"
    CATEGORY = "WtlNodes/image"

    @staticmethod
    def generate_bayer_matrix(n):
        if n == 1:
            return np.array([[0]], dtype=np.float32)
        if n in DitherC._bayer_matrix_cache:
            return DitherC._bayer_matrix_cache[n]
        smaller = DitherC.generate_bayer_matrix(n // 2)
        result = np.vstack([
            np.hstack([4 * smaller, 4 * smaller + 2]),
            np.hstack([4 * smaller + 3, 4 * smaller + 1])
        ]).astype(np.float32)
        DitherC._bayer_matrix_cache[n] = result
        return result

    @staticmethod
    def generate_blue_noise(seed=0):
        if seed in DitherC._blue_noise_cache:
            return DitherC._blue_noise_cache[seed]
        np.random.seed(seed)
        size = 256
        blue_noise = np.zeros((4, size, size), dtype=np.uint8)
        for ch in range(4):
            white = np.random.rand(size, size)
            bn = white.copy()
            for _ in range(5):
                from scipy.ndimage import gaussian_filter
                smooth = gaussian_filter(bn, sigma=2.0)
                bn = white - 0.7 * smooth
                bn = (bn - bn.min()) / (bn.max() - bn.min() + 1e-8)
            blue_noise[ch] = (bn * 255).astype(np.uint8)
        DitherC._blue_noise_cache[seed] = blue_noise
        return blue_noise

    @staticmethod
    def posterize_no_dither(image, levels_per_channel):
        result = image.clone()
        for c in range(3):
            levels = max(2, levels_per_channel[c])
            ch = result[:, :, :, c]
            result[:, :, :, c] = torch.clamp(torch.floor(ch * (levels - 1) + 0.5) / (levels - 1), 0, 1)
        return result

    @staticmethod
    def bayer_dither(image, levels_per_channel, dither_scale=1.0):
        batch_size, height, width, _ = image.shape
        device = image.device
        bayer = DitherC.generate_bayer_matrix(8)
        bayer_t = torch.from_numpy(bayer).to(device).float()
        result = image.clone()
        for c in range(3):
            levels = max(2, levels_per_channel[c])
            y_coords = torch.arange(height, device=device, dtype=torch.int32).view(height, 1).expand(height, width)
            x_coords = torch.arange(width, device=device, dtype=torch.int32).view(1, width).expand(height, width)
            bayer_y = (y_coords / dither_scale).int() % 8
            bayer_x = (x_coords / dither_scale).int() % 8
            bayer_vals = bayer_t[bayer_y, bayer_x]
            threshold = (bayer_vals - 32.5) / 64.0 / levels
            dithered = result[:, :, :, c] + threshold.unsqueeze(0)
            result[:, :, :, c] = torch.clamp(torch.floor(dithered * (levels - 1) + 0.5) / (levels - 1), 0, 1)
        return result

    @staticmethod
    def arithmetic_add_dither(image, levels_per_channel, dither_scale=1.0):
        _, height, width, _ = image.shape
        device = image.device
        result = image.clone()
        for c in range(3):
            levels = max(2, levels_per_channel[c])
            y_coords = (torch.arange(height, device=device, dtype=torch.int32).view(height, 1) / dither_scale).int().expand(height, width)
            x_coords = (torch.arange(width, device=device, dtype=torch.int32).view(1, width) / dither_scale).int().expand(height, width)
            mask = (((x_coords + c * 67) + y_coords * 236) * 119) & 255
            threshold = (mask.float() - 128.0) / 256.0 / levels
            dithered = result[:, :, :, c] + threshold.unsqueeze(0)
            result[:, :, :, c] = torch.clamp(torch.floor(dithered * (levels - 1) + 0.5) / (levels - 1), 0, 1)
        return result

    @staticmethod
    def blue_noise_dither(image, levels_per_channel, dither_scale=1.0):
        _, height, width, _ = image.shape
        device = image.device
        blue_noise = DitherC.generate_blue_noise(seed=0)
        blue_noise_t = torch.from_numpy(blue_noise).to(device).float()
        result = image.clone()
        for c in range(3):
            levels = max(2, levels_per_channel[c])
            y_coords = (torch.arange(height, device=device, dtype=torch.int32).view(height, 1) / dither_scale).int() % 256
            x_coords = (torch.arange(width, device=device, dtype=torch.int32).view(1, width) / dither_scale).int() % 256
            noise_vals = blue_noise_t[c, y_coords, x_coords]
            threshold = (noise_vals - 128.0) / 257.0 / levels
            dithered = result[:, :, :, c] + threshold.unsqueeze(0)
            result[:, :, :, c] = torch.clamp(torch.floor(dithered * (levels - 1) + 0.5) / (levels - 1), 0, 1)
        return result

    @staticmethod
    def apply_dither(image, dither_method, r_levels, g_levels, b_levels, dither_scale=1.0):
        levels = [r_levels, g_levels, b_levels]
        if dither_method == "none":
            return DitherC.posterize_no_dither(image, levels)
        elif dither_method == "bayer":
            return DitherC.bayer_dither(image, levels, dither_scale)
        elif dither_method == "arithmetic_add":
            return DitherC.arithmetic_add_dither(image, levels, dither_scale)
        elif dither_method == "blue_noise":
            return DitherC.blue_noise_dither(image, levels, dither_scale)
        return image

    def dither(self, image, dither_method, r_levels, g_levels, b_levels, dither_scale, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                t0 = time.time()
                preview = self.apply_dither(image, *cur)
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                    t0 = time.time()
                    preview = self.apply_dither(image, *cur)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                result = self.apply_dither(image, *final)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single = image[i:i+1]

                    cur = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                    t0 = time.time()
                    preview = self.apply_dither(single, *cur)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                    final = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                        t0 = time.time()
                        preview = self.apply_dither(single, *cur)
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        result_list.append(self.apply_dither(single, *final))

                result = torch.cat(result_list, dim=0)
        else:
            result = self.apply_dither(image, dither_method, r_levels, g_levels, b_levels, dither_scale)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"Dither": DitherC}
NODE_DISPLAY_NAME_MAPPINGS = {"Dither": "Dither"}
