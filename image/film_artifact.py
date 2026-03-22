import torch
import threading
import time
import math
import pickle
from pathlib import Path
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()
_ARTIFACT_CACHE = None
_CACHE_LOCK = threading.Lock()

def _load_artifact_cache():
    """Load the pre-generated artifact cache (only once)."""
    global _ARTIFACT_CACHE
    
    with _CACHE_LOCK:
        if _ARTIFACT_CACHE is not None:
            return _ARTIFACT_CACHE
        
        cache_path = Path(__file__).parent / 'film_artifacts_cache.pkl'
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Artifact cache not found at {cache_path}\n"
                f"Please run 'python generate_artifact_cache.py' first to create the cache."
            )
        
        print(f"Loading film artifact cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            _ARTIFACT_CACHE = pickle.load(f)
        
        print(f"✓ Cache loaded: {len(_ARTIFACT_CACHE['scratches']['patterns'])} scratches, "
              f"{len(_ARTIFACT_CACHE['hairs']['shapes'])} hairs")
        
        return _ARTIFACT_CACHE

def _set_params(node_id: str, intensity: float, scratch_density: float, scratch_max_length: float,
                scratch_max_width: int, dust_density: float, dust_max_size: float, hair_density: float, 
                hair_max_length: float, light_leak_intensity: float, vignette_strength: float, seed: int) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = {
            "intensity": intensity,
            "seed": seed,
            "scratch_density": scratch_density,
            "scratch_max_length": scratch_max_length,
            "scratch_max_width": scratch_max_width,
            "dust_density": dust_density,
            "dust_max_size": dust_max_size,
            "hair_density": hair_density,
            "hair_max_length": hair_max_length,
            "light_leak_intensity": light_leak_intensity,
            "vignette_strength": vignette_strength,
        }
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, intensity: float, scratch_density: float, scratch_max_length: float,
                scratch_max_width: int, dust_density: float, dust_max_size: float, hair_density: float, 
                hair_max_length: float, light_leak_intensity: float, vignette_strength: float, seed: int) -> tuple:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        params = entry.get("params", {
            "intensity": intensity,
            "seed": seed,
            "scratch_density": scratch_density,
            "scratch_max_length": scratch_max_length,
            "scratch_max_width": scratch_max_width,
            "dust_density": dust_density,
            "dust_max_size": dust_max_size,
            "hair_density": hair_density,
            "hair_max_length": hair_max_length,
            "light_leak_intensity": light_leak_intensity,
            "vignette_strength": vignette_strength,
        })
        return (params["intensity"], params["scratch_density"], params["scratch_max_length"],
                params["scratch_max_width"], params["dust_density"], params["dust_max_size"], 
                params["hair_density"], params["hair_max_length"], params["light_leak_intensity"], 
                params["vignette_strength"], params["seed"])

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

def _clear_all(node_id: str) -> None:
    with _CONTROL_LOCK:
        _CONTROL_STORE.pop(node_id, None)

def generate_perlin_noise(shape, scale=1.0, device='cpu', seed=None):
    """Generate Perlin-like noise using vectorized operations with optional seed."""
    batch, height, width, channels = shape
    
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    scaled_h = max(1, int(height / scale))
    scaled_w = max(1, int(width / scale))
    
    base_noise = torch.randn(batch, scaled_h, scaled_w, channels, device=device, generator=generator)
    
    noise = torch.nn.functional.interpolate(
        base_noise.permute(0, 3, 1, 2),
        size=(height, width),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1)
    
    noise = noise / (noise.std() + 1e-8) * 0.5
    
    return noise

def apply_scratches_from_cache(image, density, max_length, max_width, seed, device, cache):
    """
    Apply vertical scratches using pre-generated patterns from cache.
    FAST: Just scaling and placement, no generation.
    """
    batch, height, width, channels = image.shape
    
    if density <= 0:
        return torch.zeros(batch, height, width, 1, device=device)
    
    num_scratches = int((density / 100.0) * 40)
    if num_scratches == 0:
        return torch.zeros(batch, height, width, 1, device=device)
    
    mask = torch.zeros(batch, height, width, 1, device=device)
    
    # Get cache data
    patterns = cache['scratches']['patterns']
    lengths = cache['scratches']['lengths']
    widths_pref = cache['scratches']['widths']
    total_cached = patterns.shape[0]
    
    # Use seed to select scratches
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, total_cached, (num_scratches,), generator=generator)
    
    for idx in indices:
        idx = idx.item()
        
        # Get cached pattern
        pattern_length = lengths[idx].item()
        pattern = patterns[idx, :pattern_length]
        width_pref = widths_pref[idx].item()
        
        # Scale to desired length
        desired_length = int(height * (max_length / 100.0))
        if desired_length < 10:
            continue
        
        # Interpolate pattern to desired length
        scaled_pattern = torch.nn.functional.interpolate(
            pattern.unsqueeze(0).unsqueeze(0),
            size=desired_length,
            mode='linear',
            align_corners=False
        ).squeeze()
        
        # Random placement
        x_pos = torch.randint(0, width, (1,), generator=generator).item()
        start_y = torch.randint(0, max(1, height - desired_length + 1), (1,), generator=generator).item()
        
        # Width (scaled by max_width preference)
        scratch_width = max(1, int(width_pref * (max_width / 10.0)))
        
        # Apply to mask
        x_start = max(0, x_pos - scratch_width // 2)
        x_end = min(width, x_pos + scratch_width // 2 + 1)
        end_y = min(start_y + desired_length, height)
        actual_len = end_y - start_y
        
        if x_start < x_end and actual_len > 0:
            opacity_2d = scaled_pattern[:actual_len].view(-1, 1).expand(-1, x_end - x_start).to(device)
            current = mask[:, start_y:end_y, x_start:x_end, 0]
            mask[:, start_y:end_y, x_start:x_end, 0] = torch.maximum(current, opacity_2d)
    
    return mask

def apply_hairs_from_cache(image, density, max_length, seed, device, cache):
    """
    Apply hair fibers using PIL for fast line rendering.
    PIL is MUCH faster than pixel-by-pixel tensor operations.
    """
    from PIL import Image, ImageDraw
    import numpy as np
    
    batch, height, width, channels = image.shape
    
    if density <= 0:
        return torch.zeros(batch, height, width, 1, device=device)
    
    # Density directly controls number (0-100 -> 0-50 hairs)
    num_hairs = max(1, int(density / 2))
    
    # Render at 2x for antialiasing
    super_h = height * 2
    super_w = width * 2
    
    # Create PIL image (CPU rendering - FAST)
    pil_img = Image.new('L', (super_w, super_h), 0)  # Grayscale
    draw = ImageDraw.Draw(pil_img)
    
    # Get cache data
    shapes = cache['hairs']['shapes']
    thicknesses = cache['hairs']['thicknesses']
    intensities = cache['hairs']['intensities']
    total_cached = shapes.shape[0]
    
    # Use seed to select hairs
    generator = torch.Generator().manual_seed(seed + 2000)
    indices = torch.randint(0, total_cached, (num_hairs,), generator=generator)
    
    for idx in indices:
        idx = idx.item()
        
        # Get cached shape (normalized [0, 1])
        shape = shapes[idx].cpu().numpy()  # [100, 2]
        thickness = thicknesses[idx].item()
        intensity = intensities[idx].item()
        
        # Scale to image with desired length
        length_scale = max_length / 100.0
        
        # Scale coordinates
        scaled_x = shape[:, 0] * super_w
        scaled_y = shape[:, 1] * super_h * length_scale
        
        # Random placement
        offset_x = torch.rand(1, generator=generator).item() * super_w * 0.5
        offset_y = torch.rand(1, generator=generator).item() * max(1, super_h - scaled_y.max())
        
        final_x = scaled_x + offset_x
        final_y = scaled_y + offset_y
        
        # Create list of points for PIL
        points = [(final_x[i], final_y[i]) for i in range(len(final_x))]
        
        # Draw polyline (all segments at once!)
        # PIL draws lines FAST - this is what Paint does internally
        line_width = max(1, int(thickness))
        color = int(intensity * 255)  # 0-255 grayscale value
        
        draw.line(points, fill=color, width=line_width)
    
    # Convert PIL image to tensor
    img_array = np.array(pil_img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    super_mask = torch.from_numpy(img_array).to(device).unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
    super_mask = super_mask.repeat(batch, 1, 1, 1)
    
    # Downsample to original resolution
    mask = torch.nn.functional.interpolate(
        super_mask.permute(0, 3, 1, 2),
        size=(height, width),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1)
    
    return mask

def apply_dust_spots(image, density, max_size, seed, device):
    """Apply simple gaussian dust spots - fast local-only computation."""
    if density <= 0:
        return torch.zeros_like(image), torch.zeros_like(image)
    
    batch, height, width, channels = image.shape
    black_dust = torch.zeros(batch, height, width, 1, device=device)
    white_dust = torch.zeros(batch, height, width, 1, device=device)
    
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    
    # Density directly controls number of spots (0-100 -> 0-200 spots)
    num_spots = int(density * 2)
    if num_spots == 0:
        return black_dust, white_dust
    
    # Generate spot parameters
    spot_centers = []
    
    for i in range(num_spots):
        # Random position
        cx = torch.rand(1, generator=generator).item() * width
        cy = torch.rand(1, generator=generator).item() * height
        
        # Check for overlap with existing spots (simple collision avoidance)
        too_close = False
        for existing_cx, existing_cy, existing_r in spot_centers:
            dist = math.sqrt((cx - existing_cx)**2 + (cy - existing_cy)**2)
            if dist < existing_r * 1.5:  # Avoid if too close
                too_close = True
                break
        
        if too_close:
            continue
        
        # Random size
        radius = 3.0 + torch.rand(1, generator=generator).item() * (max_size / 100.0) * 297.0
        
        # Random intensity and type
        is_black = torch.rand(1, generator=generator).item() < 0.85
        if is_black:
            intensity = 0.15 + torch.rand(1, generator=generator).item() * 0.50
        else:
            intensity = 0.10 + torch.rand(1, generator=generator).item() * 0.35
        
        spot_centers.append((cx, cy, radius))
        
        # Only compute in local region (feathered mask approach)
        # Bounding box with padding
        padding = int(radius * 3)  # 3 sigma for gaussian
        x_min = max(0, int(cx - padding))
        x_max = min(width, int(cx + padding) + 1)
        y_min = max(0, int(cy - padding))
        y_max = min(height, int(cy + padding) + 1)
        
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # Create local coordinate grid
        local_x = torch.arange(x_min, x_max, device=device, dtype=torch.float32).view(1, -1)
        local_y = torch.arange(y_min, y_max, device=device, dtype=torch.float32).view(-1, 1)
        
        # Compute distance in local region only
        dist = torch.sqrt((local_x - cx) ** 2 + (local_y - cy) ** 2)
        
        # Gaussian falloff (feathered edge)
        spot_mask = torch.exp(-(dist ** 2) / (2 * (radius / 2.5) ** 2))
        spot_mask = torch.clamp(spot_mask, 0.0, 1.0) * intensity
        
        # Apply to appropriate mask (local region only)
        if is_black:
            black_dust[:, y_min:y_max, x_min:x_max, 0] = torch.maximum(
                black_dust[:, y_min:y_max, x_min:x_max, 0], 
                spot_mask
            )
        else:
            white_dust[:, y_min:y_max, x_min:x_max, 0] = torch.maximum(
                white_dust[:, y_min:y_max, x_min:x_max, 0], 
                spot_mask
            )
    
    return black_dust, white_dust

def apply_light_leaks(image, intensity, seed, device):
    """Apply light leak effects using Perlin noise."""
    if intensity <= 0:
        return torch.zeros_like(image)
    
    batch, height, width, channels = image.shape
    
    leak_noise = generate_perlin_noise(
        (batch, height, width, 1),
        scale=min(height, width) / 3.0,
        device=device,
        seed=seed + 3000
    )
    
    leak_noise = torch.clamp(leak_noise, 0.0, 1.0)
    threshold = 0.6
    leak_noise = torch.clamp((leak_noise - threshold) / (1.0 - threshold), 0.0, 1.0)
    
    leak_strength = (intensity / 100.0) * 0.4
    leak = leak_noise * leak_strength
    leak = leak.repeat(1, 1, 1, channels)
    
    return leak

def apply_vignette(image, strength, device):
    """Apply vignette darkening to edges."""
    if strength <= 0:
        return torch.ones_like(image)
    
    batch, height, width, channels = image.shape
    
    y_coords = torch.linspace(-1, 1, height, device=device).view(-1, 1)
    x_coords = torch.linspace(-1, 1, width, device=device).view(1, -1)
    
    dist = torch.sqrt(x_coords ** 2 + y_coords ** 2)
    
    vignette_amount = strength / 100.0
    vignette = 1.0 - (dist / math.sqrt(2)) * vignette_amount
    vignette = torch.clamp(vignette, 0.0, 1.0)
    
    vignette = vignette.unsqueeze(0).unsqueeze(-1).repeat(batch, 1, 1, channels)
    
    return vignette

def apply_film_artifacts(image, intensity, scratch_density, scratch_max_length, scratch_max_width,
                         dust_density, dust_max_size, hair_density, hair_max_length,
                         light_leak_intensity, vignette_strength, seed, cache):
    """Apply all film artifacts using cache for scratches and hairs."""
    import time as timing_module
    device = image.device
    
    opacity_multiplier = intensity / 100.0
    
    # Generate artifacts with timing
    start = timing_module.time()
    scratches = apply_scratches_from_cache(
        image, scratch_density, scratch_max_length, scratch_max_width, seed, device, cache
    )
    print(f"[FilmArtifacts] Scratches: {(timing_module.time() - start)*1000:.2f}ms")
    
    start = timing_module.time()
    black_dust, white_dust = apply_dust_spots(
        image, dust_density, dust_max_size, seed, device
    )
    print(f"[FilmArtifacts] Dust: {(timing_module.time() - start)*1000:.2f}ms")
    
    start = timing_module.time()
    hairs = apply_hairs_from_cache(
        image, hair_density, hair_max_length, seed, device, cache
    )
    print(f"[FilmArtifacts] Hairs: {(timing_module.time() - start)*1000:.2f}ms")
    
    start = timing_module.time()
    light_leaks = apply_light_leaks(
        image, light_leak_intensity, seed, device
    )
    print(f"[FilmArtifacts] Light leaks: {(timing_module.time() - start)*1000:.2f}ms")
    
    start = timing_module.time()
    vignette = apply_vignette(
        image, vignette_strength, device
    )
    print(f"[FilmArtifacts] Vignette: {(timing_module.time() - start)*1000:.2f}ms")
    
    # Apply opacity multiplier and expand masks
    start = timing_module.time()
    
    # Fuse opacity multiplication with mask expansion
    scratches = (scratches * opacity_multiplier).expand(-1, -1, -1, image.shape[3])
    black_dust = (black_dust * opacity_multiplier).expand(-1, -1, -1, image.shape[3])
    white_dust = (white_dust * opacity_multiplier).expand(-1, -1, -1, image.shape[3])
    hairs = (hairs * opacity_multiplier).expand(-1, -1, -1, image.shape[3])
    light_leaks = light_leaks * opacity_multiplier
    vignette = 1.0 - (1.0 - vignette) * opacity_multiplier
    
    # Pre-combine all screen-blended masks (scratches + hairs + white_dust)
    # Screen blend: 1 - (1-a)(1-b)(1-c) = 1 - (1-a) * (1-b) * (1-c)
    inv_result = (1.0 - scratches) * (1.0 - hairs) * (1.0 - white_dust)
    combined_bright = 1.0 - inv_result
    
    # Apply blending (fewer operations, more in-place)
    result = image * vignette  # Multiply blend
    result *= (1.0 - black_dust)  # Multiply blend (in-place)
    result += light_leaks  # Additive (in-place)
    
    # Apply combined screen blend
    result = 1.0 - (1.0 - result) * (1.0 - combined_bright)
    result.clamp_(0.0, 1.0)  # In-place clamp
    
    print(f"[FilmArtifacts] Blending: {(timing_module.time() - start)*1000:.2f}ms")
    
    return result

class FilmArtifactsC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                }),
                "scratch_density": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "scratch_max_length": ("FLOAT", {
                    "default": 80.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "scratch_max_width": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "dust_density": ("FLOAT", {
                    "default": 40.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "dust_max_size": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "hair_density": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "hair_max_length": ("FLOAT", {
                    "default": 60.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "light_leak_intensity": ("FLOAT", {
                    "default": 20.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "vignette_strength": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "film_artifacts"
    CATEGORY = "WtlNodes/image"
    
    def film_artifacts(self, image, intensity, seed, scratch_density, scratch_max_length,
                      scratch_max_width, dust_density, dust_max_size, hair_density, hair_max_length,
                      light_leak_intensity, vignette_strength, apply_type, unique_id=None):
        
        # Load cache (only once)
        cache = _load_artifact_cache()
        
        # Clean stale data
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                (cur_intensity, cur_scratch_density, cur_scratch_max_length, cur_scratch_max_width,
                 cur_dust_density, cur_dust_max_size, cur_hair_density,
                 cur_hair_max_length, cur_light_leak_intensity,
                 cur_vignette_strength, cur_seed) = _get_params(
                    uid, intensity, scratch_density, scratch_max_length, scratch_max_width,
                    dust_density, dust_max_size, hair_density, hair_max_length,
                    light_leak_intensity, vignette_strength, seed
                )
                start_time = time.time()
                initial_image = apply_film_artifacts(
                    image, cur_intensity, cur_scratch_density, cur_scratch_max_length,
                    cur_scratch_max_width, cur_dust_density, cur_dust_max_size,
                    cur_hair_density, cur_hair_max_length, cur_light_leak_intensity,
                    cur_vignette_strength, cur_seed, cache
                )
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(initial_image, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_p = _get_params(
                                uid, intensity, scratch_density, scratch_max_length, scratch_max_width,
                                dust_density, dust_max_size, hair_density, hair_max_length,
                                light_leak_intensity, vignette_strength, seed
                            )
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    (cur_intensity, cur_scratch_density, cur_scratch_max_length, cur_scratch_max_width,
                     cur_dust_density, cur_dust_max_size, cur_hair_density,
                     cur_hair_max_length, cur_light_leak_intensity,
                     cur_vignette_strength, cur_seed) = _get_params(
                        uid, intensity, scratch_density, scratch_max_length, scratch_max_width,
                        dust_density, dust_max_size, hair_density, hair_max_length,
                        light_leak_intensity, vignette_strength, seed
                    )
                    start_time = time.time()
                    cur_image = apply_film_artifacts(
                        image, cur_intensity, cur_scratch_density, cur_scratch_max_length,
                        cur_scratch_max_width, cur_dust_density, cur_dust_max_size,
                        cur_hair_density, cur_hair_max_length, cur_light_leak_intensity,
                        cur_vignette_strength, cur_seed, cache
                    )
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_image, uid)

                result = apply_film_artifacts(
                    image, final_p[0], final_p[1], final_p[2], final_p[3], final_p[4],
                    final_p[5], final_p[6], final_p[7], final_p[8], final_p[9], final_p[10], cache
                )

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]

                    (cur_intensity, cur_scratch_density, cur_scratch_max_length, cur_scratch_max_width,
                     cur_dust_density, cur_dust_max_size, cur_hair_density,
                     cur_hair_max_length, cur_light_leak_intensity,
                     cur_vignette_strength, cur_seed) = _get_params(
                        uid, intensity, scratch_density, scratch_max_length, scratch_max_width,
                        dust_density, dust_max_size, hair_density, hair_max_length,
                        light_leak_intensity, vignette_strength, seed
                    )
                    image_seed = cur_seed + i
                    start_time = time.time()
                    initial_image = apply_film_artifacts(
                        single_image, cur_intensity, cur_scratch_density, cur_scratch_max_length,
                        cur_scratch_max_width, cur_dust_density, cur_dust_max_size,
                        cur_hair_density, cur_hair_max_length, cur_light_leak_intensity,
                        cur_vignette_strength, image_seed, cache
                    )
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(initial_image, uid)

                    final_params = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_params = _get_params(
                                    uid, intensity, scratch_density, scratch_max_length, scratch_max_width,
                                    dust_density, dust_max_size, hair_density, hair_max_length,
                                    light_leak_intensity, vignette_strength, seed
                                )
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_image)
                                final_params = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        (cur_intensity, cur_scratch_density, cur_scratch_max_length, cur_scratch_max_width,
                         cur_dust_density, cur_dust_max_size, cur_hair_density,
                         cur_hair_max_length, cur_light_leak_intensity,
                         cur_vignette_strength, cur_seed) = _get_params(
                            uid, intensity, scratch_density, scratch_max_length, scratch_max_width,
                            dust_density, dust_max_size, hair_density, hair_max_length,
                            light_leak_intensity, vignette_strength, seed
                        )
                        image_seed = cur_seed + i
                        start_time = time.time()
                        cur_image = apply_film_artifacts(
                            single_image, cur_intensity, cur_scratch_density, cur_scratch_max_length,
                            cur_scratch_max_width, cur_dust_density, cur_dust_max_size,
                            cur_hair_density, cur_hair_max_length, cur_light_leak_intensity,
                            cur_vignette_strength, image_seed, cache
                        )
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(cur_image, uid)

                    if final_params is not None:
                        image_seed = final_params[10] + i
                        processed = apply_film_artifacts(
                            single_image, final_params[0], final_params[1], final_params[2],
                            final_params[3], final_params[4], final_params[5],
                            final_params[6], final_params[7], final_params[8],
                            final_params[9], image_seed, cache
                        )
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_film_artifacts(
                image, intensity, scratch_density, scratch_max_length,
                scratch_max_width, dust_density, dust_max_size,
                hair_density, hair_max_length, light_leak_intensity,
                vignette_strength, seed, cache
            )
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"FilmArtifacts": FilmArtifactsC}
NODE_DISPLAY_NAME_MAPPINGS = {"FilmArtifacts": "Film Artifacts"}