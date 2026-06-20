import torch
import torch.nn.functional as F
import threading
import time
import math
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# param store helpers
# ---------------------------------------------------------------------------

def _set_params(node_id: str, *args) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        if entry.get("params") != args:
            entry["params"] = args
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, *defaults) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", defaults)

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


# ---------------------------------------------------------------------------
# effect functions
# ---------------------------------------------------------------------------

def _gauss_1d(size, sigma, device):
    kr = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(kr ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def _gauss_sep_2d(img_p, k, sigma, channels):
    """
    Separable Gaussian: two 1D conv passes instead of one 2D pass.
    O(2k·H·W) vs O(k²·H·W) — roughly k/2× faster for large kernels such as
    the bloom outer radius at high resolutions.
    """
    k = max(3, k | 1)
    g1d = _gauss_1d(k, sigma, img_p.device)
    pad = k // 2
    kh = g1d.view(1, 1, 1, k).repeat(channels, 1, 1, 1)
    x = F.conv2d(img_p, kh, padding=(0, pad), groups=channels)
    kv = g1d.view(1, 1, k, 1).repeat(channels, 1, 1, 1)
    return F.conv2d(x, kv, padding=(pad, 0), groups=channels)


def apply_phosphor_tint(image, strength):
    """
    Converts toward a high-contrast P31 CRT green-phosphor image.
    At strength=1: fully desaturated into green/black with crushed darks.
    """
    if strength < 0.001:
        return image
    luma = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    luma_c = torch.pow(luma, 1.6)
    r = torch.pow(luma_c, 2.5) * 0.38
    g = luma_c
    b = torch.pow(luma_c, 4.0) * 0.05
    green = torch.clamp(torch.stack([r, g, b], dim=-1), 0.0, 1.0)
    return image * (1.0 - strength) + green * strength


def apply_defocus(image, amount):
    """
    Small isotropic Gaussian blur simulating CRT electron beam spot size.
    Real CRT phosphors emit light slightly beyond the beam footprint, creating
    a subtle per-pixel halo. Models beam defocus and aperture-grille bleed.
    """
    if amount < 0.001:
        return image
    channels = image.shape[3]
    sigma = max(0.3, amount * 2.5)
    k = max(3, int(sigma * 4) | 1)
    img_p = image.permute(0, 3, 1, 2)
    return _gauss_sep_2d(img_p, k, sigma, channels).permute(0, 2, 3, 1)


def apply_phosphor_dots(image, intensity, dot_size):
    """RGB sub-pixel phosphor triad overlay."""
    if intensity < 0.001:
        return image
    _, height, width, _ = image.shape
    device = image.device
    period = max(0.5, float(dot_size))
    x = torch.arange(width, device=device, dtype=torch.float32)
    y = torch.arange(height, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    r_mask = torch.cos(grid_x * math.pi * 2 / period) * 0.5 + 0.5
    g_mask = torch.cos((grid_x - period / 3) * math.pi * 2 / period) * 0.5 + 0.5
    b_mask = torch.cos((grid_x - 2 * period / 3) * math.pi * 2 / period) * 0.5 + 0.5
    v_mod = torch.sin(grid_y * math.pi / (period * 0.667)) * 0.3 + 0.7
    rgb_mask = torch.stack([r_mask * v_mod, g_mask * v_mod, b_mask * v_mod], dim=-1)
    rgb_mask = 1.0 - intensity * (1.0 - rgb_mask.unsqueeze(0))
    return image * rgb_mask


def apply_halation(image, amount):
    """Full-frame soft glow — phosphor persistence / screen-glass diffusion."""
    if amount < 0.001:
        return image
    channels = image.shape[3]
    k = max(3, int(amount * 20) | 1)
    img_p = image.permute(0, 3, 1, 2)
    blurred = _gauss_sep_2d(img_p, k, k / 4.0, channels)
    result = img_p + blurred * amount * 0.6
    return torch.clamp(result, 0.0, 1.0).permute(0, 2, 3, 1)


def apply_bloom(image, strength):
    """
    Multi-scale threshold bloom applied to image content.
    Two Gaussian passes (2% and 9% of short dimension) blended 65/35.
    Both use separable convolution for speed on large images.
    """
    if strength < 0.001:
        return image
    _, height, width, channels = image.shape
    threshold = 0.42
    luma = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    bright_mask = torch.clamp((luma - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    bright_p = (image * bright_mask.unsqueeze(-1)).permute(0, 3, 1, 2)
    short = min(height, width)
    k1 = max(5, int(short * 0.02) | 1)
    k2 = max(11, int(short * 0.09) | 1)
    b1 = _gauss_sep_2d(bright_p, k1, k1 / 3.0, channels)
    b2 = _gauss_sep_2d(bright_p, k2, k2 / 3.0, channels)
    combined = (b1 * 0.65 + b2 * 0.35).permute(0, 2, 3, 1)
    return torch.clamp(image + combined * strength * 6.0, 0.0, 1.0)


def apply_beam(image, intensity, position, width, glow):
    """
    Stylistic raster beam trace — a single bright horizontal line visible
    on low-persistence phosphors (radar monitors, P31/P39 long-persistence CRTs).

    Beam brightness is modulated per-column by the underlying image luma:
    brighter content causes stronger phosphor excitation, producing a subtle
    horizontal ripple along the beam that gives it an organic feel.

    Core: Gaussian with sigma=width pixels, amplitude 2.5× (clips to white).
    Glow: Laplacian decay inside a Hann (cos²) window. The Hann envelope reaches
    zero with zero slope at its boundary, making the outer glow edge invisible
    on dark backgrounds.
    """
    if intensity < 0.001:
        return image
    batch, height, width_px = image.shape[:3]
    device = image.device

    y_pos = (position / 100.0) * (height - 1)
    y = torch.arange(height, device=device, dtype=torch.float32) - y_pos

    sigma_core = max(0.3, float(width))
    sharp = torch.exp(-(y ** 2) / (2.0 * sigma_core ** 2))

    if glow > 0.001:
        radius = max(2.0, glow * height * 0.25)
        dist_norm = (torch.abs(y) / radius).clamp(0.0, 1.0)
        hann = torch.cos(dist_norm * math.pi * 0.5) ** 2
        tau = max(0.5, glow * height * 0.07)
        wide = torch.exp(-torch.abs(y) / tau) * hann
    else:
        wide = torch.zeros_like(sharp)

    profile = (sharp * 2.5 + wide * glow * 0.8) * intensity  # [H]

    # Per-column luma modulation: brighter image → stronger beam excitation
    row_idx = max(0, min(height - 1, int(round(y_pos))))
    luma_row = (image[:, row_idx, :, 0] * 0.299 +
                image[:, row_idx, :, 1] * 0.587 +
                image[:, row_idx, :, 2] * 0.114)  # [B, W]
    luma_mod = 1.0 + luma_row * 0.35              # [B, W] — up to +35% on bright areas

    beam_map = profile.view(1, height, 1) * luma_mod.view(batch, 1, width_px)  # [B, H, W]
    return torch.clamp(image + beam_map.unsqueeze(-1), 0.0, 1.0)


def apply_scanlines(image, intensity, spacing):
    """
    Periodic dark horizontal bands matching the inter-row structure of a real CRT.
    Models the shadow mask between phosphor rows: bright emission centres separated
    by dark gaps, with period = spacing pixels per scanline.

    line_mod = cos(π·y / spacing)²
      → reaches 1.0 at phosphor-row centres (y = 0, spacing, 2·spacing, …)
      → reaches 0.0 at midpoints between rows (y = spacing/2, 3·spacing/2, …)
    """
    if intensity < 0.001:
        return image
    height = image.shape[1]
    device = image.device
    y = torch.arange(height, device=device, dtype=torch.float32)
    period = max(1.0, float(spacing))
    line_mod = torch.cos(y * math.pi / period) ** 2  # [H], period = spacing pixels
    brightness = 1.0 - intensity * (1.0 - line_mod)  # [H]
    return image * brightness.view(1, height, 1, 1)


def apply_chromatic_aberration(image, amount):
    if amount < 0.001:
        return image
    offset = max(1, int(amount * 2))
    r = torch.roll(image[..., 0:1], shifts=offset, dims=2)
    b = torch.roll(image[..., 2:3], shifts=-offset, dims=2)
    return torch.cat([r, image[..., 1:2], b], dim=-1)


def apply_curvature(image, amount):
    """Barrel distortion — pure NDC coordinates give even black borders."""
    if amount < 0.001:
        return image
    batch, height, width, _ = image.shape
    device = image.device
    x = torch.linspace(-1, 1, width, device=device)
    y = torch.linspace(-1, 1, height, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    r2 = grid_x ** 2 + grid_y ** 2
    distortion = 1.0 + amount * r2
    grid = torch.stack([grid_x * distortion, grid_y * distortion], dim=-1)
    grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
    warped = F.grid_sample(image.permute(0, 3, 1, 2), grid,
                           mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped.permute(0, 2, 3, 1)


def apply_vignette(image, amount):
    if amount < 0.001:
        return image
    height, width = image.shape[1], image.shape[2]
    device = image.device
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    mask = torch.clamp(1.0 - r * amount * 0.7, 0.0, 1.0).view(1, height, width, 1)
    return image * mask


def apply_noise(image, amount, seed=None):
    if amount < 0.001:
        return image
    if seed is not None:
        generator = torch.Generator(device=image.device).manual_seed(seed)
        noise = torch.randn_like(image, generator=generator) * amount * 0.1
    else:
        noise = torch.randn_like(image) * amount * 0.1
    return torch.clamp(image + noise, 0.0, 1.0)


# ---------------------------------------------------------------------------
# effect chain
# ---------------------------------------------------------------------------

def _run_effects(image,
                 beam_i, beam_pos, beam_w, beam_glow,
                 sl_i, sl_spacing,
                 curv, ca, hal, bl, pt, pd, pd_size, defocus, n, vig, grayscale, noise_seed):
    if grayscale:
        luma = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        image = luma.unsqueeze(-1).expand_as(image)
    result = apply_phosphor_tint(image, pt)
    result = apply_defocus(result, defocus)
    result = apply_phosphor_dots(result, pd, pd_size)
    result = apply_halation(result, hal)
    result = apply_bloom(result, bl)
    result = apply_scanlines(result, sl_i, sl_spacing)
    result = apply_beam(result, beam_i, beam_pos, beam_w, beam_glow)
    result = apply_noise(result, n, noise_seed)     # before curvature — no noise on black border
    result = apply_curvature(result, curv)
    result = apply_chromatic_aberration(result, ca)
    result = apply_vignette(result, vig)
    return result


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class CRTEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":                ("IMAGE",),
                # --- Raster beam trace (stylistic bright sweep line) ---
                "beam_intensity":       ("FLOAT", {"default": 0.75,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "beam_position":        ("FLOAT", {"default": 60.0, "min": 0.0,  "max": 100.0, "step": 0.1}),
                "beam_width":           ("FLOAT", {"default": 2.5,  "min": 0.3,  "max": 20.0,  "step": 0.5}),
                "beam_glow":            ("FLOAT", {"default": 0.6,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                # --- CRT scanlines (periodic phosphor-row structure) ---
                "scanline_intensity":   ("FLOAT", {"default": 0.6,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "scanline_spacing":     ("INT",   {"default": 8,    "min": 2,    "max": 16,      "step": 1}),
                # --- Screen optics ---
                "curvature":            ("FLOAT", {"default": 0.1, "min": 0.0,  "max": 0.5,   "step": 0.01}),
                "chromatic_aberration": ("FLOAT", {"default": 1.5,  "min": 0.0,  "max": 5.0,   "step": 0.1}),
                "halation":             ("FLOAT", {"default": 0.5,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "bloom":                ("FLOAT", {"default": 0.35, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                # --- Phosphor ---
                "phosphor_tint":        ("FLOAT", {"default": 0.7,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "phosphor_dots":        ("FLOAT", {"default": 0.2, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "phosphor_dot_size":    ("FLOAT", {"default": 5.0,  "min": 1.0,  "max": 12.0,  "step": 0.5}),
                "defocus":              ("FLOAT", {"default": 0.4,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                # --- Overall ---
                "noise":                ("FLOAT",   {"default": 0.3,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "vignette":             ("FLOAT",   {"default": 0.4,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "grayscale":            ("BOOLEAN", {"default": False}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_crt"
    CATEGORY = "WtlNodes/image"

    def apply_crt(self, image,
                  beam_intensity, beam_position, beam_width, beam_glow,
                  scanline_intensity, scanline_spacing,
                  curvature, chromatic_aberration,
                  halation, bloom,
                  phosphor_tint, phosphor_dots, phosphor_dot_size,
                  defocus, noise, vignette, grayscale,
                  apply_type, unique_id=None):

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        def pack_defaults():
            return _get_params(uid,
                               beam_intensity, beam_position, beam_width, beam_glow,
                               scanline_intensity, scanline_spacing,
                               curvature, chromatic_aberration,
                               halation, bloom,
                               phosphor_tint, phosphor_dots, phosphor_dot_size,
                               defocus, noise, vignette, grayscale)

        def run(img, p, nseed):
            (bi, bp, bw, bg,
             sl_i, sl_sp,
             curv, ca, hal, bl, pt, pd, pd_size, df, n, vig, gs) = p
            return _run_effects(img, bi, bp, bw, bg,
                                sl_i, sl_sp,
                                curv, ca, hal, bl, pt, pd, pd_size, df, n, vig, gs, nseed)

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)
            noise_seed = _get_or_create_seed(uid)

            if apply_type == "apply_all":
                p = pack_defaults()
                t0 = time.time()
                preview = run(image, p, noise_seed)
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _send_ram_preview(preview, uid)

                final_p = None
                while True:
                    changed = False
                    while not changed:
                        if _check_and_clear_params_changed(uid):
                            changed = True; break
                        if _check_and_clear_flag(uid, "apply"):
                            final_p = pack_defaults(); break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)
                    if not changed:
                        break
                    p = pack_defaults()
                    t0 = time.time()
                    preview = run(image, p, noise_seed)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                result = run(image, final_p, noise_seed)

            else:
                batch_size = image.shape[0]
                result_list = []
                for i in range(batch_size):
                    single = image[i:i + 1]
                    img_noise_seed = noise_seed + i
                    p = pack_defaults()
                    t0 = time.time()
                    preview = run(single, p, img_noise_seed)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                    final_p = None
                    while True:
                        changed = False
                        while not changed:
                            if _check_and_clear_params_changed(uid):
                                changed = True; break
                            if _check_and_clear_flag(uid, "apply"):
                                final_p = pack_defaults(); break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single); final_p = None; break
                            time.sleep(0.05)
                        if not changed:
                            break
                        p = pack_defaults()
                        t0 = time.time()
                        preview = run(single, p, img_noise_seed)
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final_p is not None:
                        result_list.append(run(single, final_p, img_noise_seed))

                result = torch.cat(result_list, dim=0)

        else:
            p = (beam_intensity, beam_position, beam_width, beam_glow,
                 scanline_intensity, scanline_spacing,
                 curvature, chromatic_aberration,
                 halation, bloom,
                 phosphor_tint, phosphor_dots, phosphor_dot_size,
                 defocus, noise, vignette, grayscale)
            result = run(image, p, None)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"CRTEffect": CRTEffect}
NODE_DISPLAY_NAME_MAPPINGS = {"CRTEffect": "CRT TV Effect"}
