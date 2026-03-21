import torch
import torch.nn.functional as F
import threading
import time
import math
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, scanline_intensity: float, scanline_width: float, 
                curvature: float, chromatic_aberration: float, halation: float,
                phosphor_dots: float, noise: float, vignette: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (scanline_intensity, scanline_width, curvature, 
                          chromatic_aberration, halation, phosphor_dots, noise, vignette)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, scanline_intensity: float, scanline_width: float,
                curvature: float, chromatic_aberration: float, halation: float,
                phosphor_dots: float, noise: float, vignette: float) -> tuple:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (scanline_intensity, scanline_width, curvature,
                                   chromatic_aberration, halation, phosphor_dots, noise, vignette))

def _set_flag(node_id: str, flag: str) -> None:
    """Mark a button press – ``flag`` must be ``'apply'`` or ``'skip'``."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        flags = entry.setdefault("flags", {})
        flags[flag] = True

def _check_and_clear_flag(node_id: str, flag: str) -> bool:
    """Return True once if the flag was set; afterwards it is cleared."""
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
    """Remove *everything* stored for a node – used at the start of a run."""
    with _CONTROL_LOCK:
        _CONTROL_STORE.pop(node_id, None)

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
    """Get or create a static seed for this node's session."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        if "seed" not in entry:
            entry["seed"] = torch.randint(0, 2**31 - 1, (1,)).item()
        return entry["seed"]

def apply_scanlines(image, intensity, width):
    """Apply horizontal scanlines to the image with gaussian falloff."""
    batch, height, width_px, channels = image.shape
    
    # Create smooth scanline pattern with gaussian-like falloff
    y = torch.arange(height, device=image.device, dtype=torch.float32)
    
    # Create a sine wave pattern for smooth scanlines
    frequency = math.pi / width
    scanline_pattern = torch.sin(y * frequency)
    
    # Convert to 0-1 range and apply intensity
    scanline_pattern = scanline_pattern * 0.5 + 0.5
    scanline_pattern = torch.pow(scanline_pattern, 0.5)
    
    # Apply intensity
    scanline_pattern = 1.0 - (intensity * (1.0 - scanline_pattern))
    
    # Reshape to apply across width
    scanline_pattern = scanline_pattern.view(1, height, 1, 1)
    
    return image * scanline_pattern

def apply_curvature(image, amount):
    """Apply barrel distortion to simulate curved CRT screen with regular black borders."""
    batch, height, width, channels = image.shape
    device = image.device
    
    # Use aspect ratio to make coordinates square-ish for regular borders
    aspect = width / height
    
    # Create normalized coordinates with aspect correction
    if aspect > 1:
        # Wider than tall
        y = torch.linspace(-1, 1, height, device=device)
        x = torch.linspace(-aspect, aspect, width, device=device)
    else:
        # Taller than wide
        y = torch.linspace(-1/aspect, 1/aspect, height, device=device)
        x = torch.linspace(-1, 1, width, device=device)
    
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate distance from center
    r = torch.sqrt(grid_x**2 + grid_y**2)
    
    # Apply barrel distortion
    distortion = 1.0 + amount * r**2
    
    # Warp coordinates
    warped_x = grid_x * distortion
    warped_y = grid_y * distortion
    
    # Normalize back to [-1, 1] range for grid_sample
    if aspect > 1:
        warped_x = warped_x / aspect
    else:
        warped_y = warped_y * aspect
    
    # Create edge mask with subtle falloff
    edge_dist_x = torch.abs(warped_x)
    edge_dist_y = torch.abs(warped_y)
    edge_dist = torch.maximum(edge_dist_x, edge_dist_y)
    
    # Soft falloff near edges
    falloff_start = 0.98
    edge_mask = torch.clamp((1.0 - edge_dist) / (1.0 - falloff_start), 0.0, 1.0)
    edge_mask = edge_mask.unsqueeze(0).unsqueeze(-1)
    
    # Stack for grid_sample
    grid = torch.stack([warped_x, warped_y], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    
    # Permute image for grid_sample
    image_permuted = image.permute(0, 3, 1, 2)
    
    # Apply warping with zeros padding
    warped = F.grid_sample(image_permuted, grid, mode='bilinear', 
                           padding_mode='zeros', align_corners=True)
    
    # Permute back
    warped = warped.permute(0, 2, 3, 1)
    
    # Apply edge mask
    warped = warped * edge_mask
    
    return warped

def apply_chromatic_aberration(image, amount):
    """Apply RGB channel separation for chromatic aberration effect."""
    batch, height, width, channels = image.shape
    
    # Calculate offset in pixels
    offset_pixels = max(1, int(amount * 2))
    
    # Split into RGB channels
    r = image[..., 0:1]
    g = image[..., 1:2] 
    b = image[..., 2:3]
    
    # Red channel - shift right
    r_shifted = torch.roll(r, shifts=offset_pixels, dims=2)
    
    # Blue channel - shift left  
    b_shifted = torch.roll(b, shifts=-offset_pixels, dims=2)
    
    # Combine channels
    result = torch.cat([r_shifted, g, b_shifted], dim=-1)
    
    return result

def apply_halation(image, amount):
    """Apply screen bleeding/halation glow effect."""
    batch, height, width, channels = image.shape
    
    # Create a blur kernel size based on amount
    kernel_size = max(3, int(amount * 20) | 1)
    
    # Permute for conv2d
    image_permuted = image.permute(0, 3, 1, 2)
    
    # Create Gaussian kernel
    sigma = kernel_size / 4.0
    kernel_range = torch.arange(kernel_size, device=image.device, dtype=torch.float32)
    kernel_range = kernel_range - kernel_size // 2
    
    gauss_1d = torch.exp(-(kernel_range**2) / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    gauss_2d = gauss_2d / gauss_2d.sum()
    
    # Apply blur to each channel
    kernel = gauss_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    
    padding = kernel_size // 2
    blurred = F.conv2d(image_permuted, kernel, padding=padding, groups=channels)
    
    # Add glow on top of original
    result = image_permuted + (blurred * amount * 0.3)
    result = torch.clamp(result, 0.0, 1.0)
    
    # Permute back
    return result.permute(0, 2, 3, 1)

def apply_phosphor_dots(image, intensity):
    """Apply RGB phosphor dot structure for CRT authenticity."""
    batch, height, width, channels = image.shape
    device = image.device
    
    # Create RGB subpixel pattern
    x = torch.arange(width, device=device, dtype=torch.float32)
    y = torch.arange(height, device=device, dtype=torch.float32)
    
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Create RGB mask
    r_mask = torch.cos(grid_x * math.pi * 2 / 3) * 0.5 + 0.5
    g_mask = torch.cos((grid_x - 1) * math.pi * 2 / 3) * 0.5 + 0.5
    b_mask = torch.cos((grid_x - 2) * math.pi * 2 / 3) * 0.5 + 0.5
    
    # Add vertical variation
    vertical_mod = torch.sin(grid_y * math.pi / 2) * 0.3 + 0.7
    
    r_mask = r_mask * vertical_mod
    g_mask = g_mask * vertical_mod
    b_mask = b_mask * vertical_mod
    
    # Stack masks
    rgb_mask = torch.stack([r_mask, g_mask, b_mask], dim=-1).unsqueeze(0)
    
    # Apply intensity
    rgb_mask = 1.0 - (intensity * (1.0 - rgb_mask))
    
    return image * rgb_mask

def apply_noise(image, amount, seed=None):
    """Apply signal noise/static with deterministic seed."""
    # Use seed for deterministic noise
    if seed is not None:
        generator = torch.Generator(device=image.device).manual_seed(seed)
        noise = torch.randn_like(image, generator=generator) * amount * 0.1
    else:
        noise = torch.randn_like(image) * amount * 0.1
    
    return torch.clamp(image + noise, 0.0, 1.0)

def apply_vignette(image, amount):
    """Apply vignetting (darkening at edges)."""
    batch, height, width, channels = image.shape
    device = image.device
    
    # Create radial gradient from center
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate distance from center
    r = torch.sqrt(grid_x**2 + grid_y**2)
    
    # Create vignette mask
    vignette_mask = 1.0 - (r * amount * 0.7)
    vignette_mask = torch.clamp(vignette_mask, 0.0, 1.0)
    
    # Apply to all channels
    vignette_mask = vignette_mask.view(1, height, width, 1)
    
    return image * vignette_mask

class CRTEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scanline_intensity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "scanline_width": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                }),
                "curvature": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                }),
                "chromatic_aberration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                }),
                "halation": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "phosphor_dots": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "noise": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "vignette": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_crt"
    CATEGORY = "WtlNodes/image"
    
    def apply_crt(self, image, scanline_intensity, scanline_width, curvature,
                  chromatic_aberration, halation, phosphor_dots, noise, vignette,
                  apply_type, unique_id=None):

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)
            seed = _get_or_create_seed(uid)

            if apply_type == "apply_all":
                cur_si, cur_sw, cur_curv, cur_ca, cur_hal, cur_ph, cur_noise, cur_vig = _get_params(
                    uid, scanline_intensity, scanline_width, curvature,
                    chromatic_aberration, halation, phosphor_dots, noise, vignette
                )
                start_time = time.time()
                cur_result = image
                cur_result = apply_curvature(cur_result, cur_curv)
                cur_result = apply_chromatic_aberration(cur_result, cur_ca)
                cur_result = apply_halation(cur_result, cur_hal)
                cur_result = apply_phosphor_dots(cur_result, cur_ph)
                cur_result = apply_scanlines(cur_result, cur_si, cur_sw)
                cur_result = apply_vignette(cur_result, cur_vig)
                cur_result = apply_noise(cur_result, cur_noise, seed)
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(cur_result, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_si, final_sw, final_curv, final_ca, final_hal, final_ph, final_noise, final_vig = _get_params(
                                uid, scanline_intensity, scanline_width, curvature,
                                chromatic_aberration, halation, phosphor_dots, noise, vignette
                            )
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_si, cur_sw, cur_curv, cur_ca, cur_hal, cur_ph, cur_noise, cur_vig = _get_params(
                        uid, scanline_intensity, scanline_width, curvature,
                        chromatic_aberration, halation, phosphor_dots, noise, vignette
                    )
                    start_time = time.time()
                    cur_result = image
                    cur_result = apply_curvature(cur_result, cur_curv)
                    cur_result = apply_chromatic_aberration(cur_result, cur_ca)
                    cur_result = apply_halation(cur_result, cur_hal)
                    cur_result = apply_phosphor_dots(cur_result, cur_ph)
                    cur_result = apply_scanlines(cur_result, cur_si, cur_sw)
                    cur_result = apply_vignette(cur_result, cur_vig)
                    cur_result = apply_noise(cur_result, cur_noise, seed)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_result, uid)

                result = image
                result = apply_curvature(result, final_curv)
                result = apply_chromatic_aberration(result, final_ca)
                result = apply_halation(result, final_hal)
                result = apply_phosphor_dots(result, final_ph)
                result = apply_scanlines(result, final_si, final_sw)
                result = apply_vignette(result, final_vig)
                result = apply_noise(result, final_noise, seed)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]
                    image_seed = seed + i

                    cur_si, cur_sw, cur_curv, cur_ca, cur_hal, cur_ph, cur_noise, cur_vig = _get_params(
                        uid, scanline_intensity, scanline_width, curvature,
                        chromatic_aberration, halation, phosphor_dots, noise, vignette
                    )
                    start_time = time.time()
                    cur_result = single_image
                    cur_result = apply_curvature(cur_result, cur_curv)
                    cur_result = apply_chromatic_aberration(cur_result, cur_ca)
                    cur_result = apply_halation(cur_result, cur_hal)
                    cur_result = apply_phosphor_dots(cur_result, cur_ph)
                    cur_result = apply_scanlines(cur_result, cur_si, cur_sw)
                    cur_result = apply_vignette(cur_result, cur_vig)
                    cur_result = apply_noise(cur_result, cur_noise, image_seed)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_result, uid)

                    final_si = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_si, final_sw, final_curv, final_ca, final_hal, final_ph, final_noise, final_vig = _get_params(
                                    uid, scanline_intensity, scanline_width, curvature,
                                    chromatic_aberration, halation, phosphor_dots, noise, vignette
                                )
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_image)
                                final_si = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_si, cur_sw, cur_curv, cur_ca, cur_hal, cur_ph, cur_noise, cur_vig = _get_params(
                            uid, scanline_intensity, scanline_width, curvature,
                            chromatic_aberration, halation, phosphor_dots, noise, vignette
                        )
                        start_time = time.time()
                        cur_result = single_image
                        cur_result = apply_curvature(cur_result, cur_curv)
                        cur_result = apply_chromatic_aberration(cur_result, cur_ca)
                        cur_result = apply_halation(cur_result, cur_hal)
                        cur_result = apply_phosphor_dots(cur_result, cur_ph)
                        cur_result = apply_scanlines(cur_result, cur_si, cur_sw)
                        cur_result = apply_vignette(cur_result, cur_vig)
                        cur_result = apply_noise(cur_result, cur_noise, image_seed)
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(cur_result, uid)

                    if final_si is not None:
                        processed = single_image
                        processed = apply_curvature(processed, final_curv)
                        processed = apply_chromatic_aberration(processed, final_ca)
                        processed = apply_halation(processed, final_hal)
                        processed = apply_phosphor_dots(processed, final_ph)
                        processed = apply_scanlines(processed, final_si, final_sw)
                        processed = apply_vignette(processed, final_vig)
                        processed = apply_noise(processed, final_noise, image_seed)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)

        else:
            result = image
            result = apply_curvature(result, curvature)
            result = apply_chromatic_aberration(result, chromatic_aberration)
            result = apply_halation(result, halation)
            result = apply_phosphor_dots(result, phosphor_dots)
            result = apply_scanlines(result, scanline_intensity, scanline_width)
            result = apply_vignette(result, vignette)
            result = apply_noise(result, noise, seed=None)

        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"CRTEffect": CRTEffect}
NODE_DISPLAY_NAME_MAPPINGS = {"CRTEffect": "CRT TV Effect"}