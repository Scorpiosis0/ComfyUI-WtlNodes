import torch
import threading
import time
import math
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, intensity: float, grain_size: float, monochrome: bool) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (intensity, grain_size, monochrome)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, intensity: float, grain_size: float, monochrome: bool) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (intensity, grain_size, monochrome))

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

def _get_or_create_seed(node_id: str) -> int:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        if "seed" not in entry:
            entry["seed"] = torch.randint(0, 2**31 - 1, (1,)).item()
        return entry["seed"]

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

def _generate_perlin_noise(shape, scale=1.0, device='cpu', seed=None):
    batch, height, width, channels = shape
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    scaled_h = max(1, int(height / scale))
    scaled_w = max(1, int(width / scale))
    base_noise = torch.randn(batch, scaled_h, scaled_w, channels, device=device, generator=generator)

    noise = torch.nn.functional.interpolate(
        base_noise.permute(0, 3, 1, 2),
        size=(height, width),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1)

    if scale > 1.5:
        detail_scale = scale / 2.0
        detail_h = max(1, int(height / detail_scale))
        detail_w = max(1, int(width / detail_scale))
        detail_gen = torch.Generator(device=device).manual_seed(seed + 1) if seed is not None else None
        detail_noise = torch.randn(batch, detail_h, detail_w, channels, device=device, generator=detail_gen)
        detail_noise = torch.nn.functional.interpolate(
            detail_noise.permute(0, 3, 1, 2),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)
        noise = noise + detail_noise * 0.5

    return noise / (noise.std() + 1e-8) * 0.5

def _apply(image, intensity, grain_size, monochrome, seed=None):
    if intensity <= 0:
        return image

    batch, height, width, channels = image.shape
    device = image.device

    if monochrome:
        grain = _generate_perlin_noise((batch, height, width, 1), scale=grain_size, device=device, seed=seed)
        grain = grain.repeat(1, 1, 1, channels)
    else:
        grain = _generate_perlin_noise((batch, height, width, channels), scale=grain_size, device=device, seed=seed)

    grain = grain * (intensity / 100.0 * 0.15)
    return torch.clamp(image + grain, 0.0, 1.0)

class FilmGrainC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0, "round": 0.1,
                }),
                "grain_size": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "round": 0.1,
                }),
                "monochrome": ("BOOLEAN", {"default": True}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "film_grain"
    CATEGORY = "WtlNodes/image"

    def film_grain(self, image, intensity, grain_size, monochrome, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)
            seed = _get_or_create_seed(uid)

            if apply_type == "apply_all":
                cur = _get_params(uid, intensity, grain_size, monochrome)
                t0 = time.time()
                preview = _apply(image, *cur, seed=seed)
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, intensity, grain_size, monochrome)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, intensity, grain_size, monochrome)
                    t0 = time.time()
                    preview = _apply(image, *cur, seed=seed)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                result = _apply(image, *final, seed=seed)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single = image[i:i+1]
                    image_seed = seed + i

                    cur = _get_params(uid, intensity, grain_size, monochrome)
                    t0 = time.time()
                    preview = _apply(single, *cur, seed=image_seed)
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
                                final = _get_params(uid, intensity, grain_size, monochrome)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, intensity, grain_size, monochrome)
                        t0 = time.time()
                        preview = _apply(single, *cur, seed=image_seed)
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        result_list.append(_apply(single, *final, seed=image_seed))

                result = torch.cat(result_list, dim=0)
        else:
            result = _apply(image, intensity, grain_size, monochrome, seed=None)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"FilmGrain": FilmGrainC}
NODE_DISPLAY_NAME_MAPPINGS = {"FilmGrain": "Film Grain"}