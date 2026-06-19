import torch
import math
import comfy.sample
import comfy.model_management
import comfy.utils


# ---------------------------------------------------------------------------
# Tile coordinates
# ---------------------------------------------------------------------------

def _build_1d(total, n):
    if n == 1:
        return [(0, total)], total
    tile = math.ceil(total / n)
    coords = []
    for i in range(n):
        s = i * tile
        e = min(s + tile, total)
        coords.append((s, e))
    if len(coords) >= 2:
        ls, le = coords[-1]
        if le - ls < tile:
            coords[-1] = (max(0, total - tile), total)
    return coords, tile


def _build_tiles(H, W, n_rows, n_cols):
    rows, tile_h = _build_1d(H, n_rows)
    cols, tile_w = _build_1d(W, n_cols)
    tiles = []
    for (y0, y1) in rows:
        for (x0, x1) in cols:
            tiles.append((y0, y1, x0, x1))
    return tiles, tile_h, tile_w


def _get_seam_positions(rows_coords, cols_coords):
    """
    Return internal seam positions (not image edges).
    seam_ys: y latent positions between row tiles
    seam_xs: x latent positions between col tiles
    """
    seam_ys = []
    seam_xs = []
    # Internal boundaries only — where one tile ends and the next begins
    for i in range(len(rows_coords) - 1):
        # midpoint between end of tile i and start of tile i+1
        seam_ys.append((rows_coords[i][1] + rows_coords[i + 1][0]) // 2)
    for i in range(len(cols_coords) - 1):
        seam_xs.append((cols_coords[i][1] + cols_coords[i + 1][0]) // 2)
    return seam_ys, seam_xs


# ---------------------------------------------------------------------------
# Cosine mask
# ---------------------------------------------------------------------------

def _cosine_mask(th, tw, device, dtype,
                 clamp_top, clamp_left, clamp_bottom, clamp_right):
    def cosine_1d(n):
        t = torch.linspace(0, torch.pi, n, device=device, dtype=dtype)
        return (1 - torch.cos(t)) / 2

    wy = cosine_1d(th)
    wx = cosine_1d(tw)

    if clamp_top:    wy[:th // 2] = 1.0
    if clamp_bottom: wy[th // 2:] = 1.0
    if clamp_left:   wx[:tw // 2] = 1.0
    if clamp_right:  wx[tw // 2:] = 1.0

    return (wy.view(th, 1) * wx.view(1, tw)).unsqueeze(0).unsqueeze(0)


def _make_strip_mask(flat_size, feather_lat, device, dtype):
    """
    1-D mask for a seam strip:
    - flat 1.0 zone of flat_size tokens in center
    - gaussian falloff of feather_lat tokens on each side
    Total strip size = flat_size + 2 * feather_lat
    Returns tensor of shape (flat_size + 2*feather_lat,).
    """
    total = flat_size + 2 * feather_lat
    mask = torch.ones(total, device=device, dtype=dtype)

    if feather_lat > 0:
        # Gaussian falloff: exp(-(x/sigma)^2)
        # x goes from 0 at flat edge to feather_lat at strip edge
        # sigma chosen so mask reaches ~0.01 at strip edge
        sigma = feather_lat / 2.15  # ~2.15 sigma reaches ~0.01
        x = torch.arange(feather_lat, device=device, dtype=dtype)
        gauss = torch.exp(-(x / sigma) ** 2)

        # Left side: gaussian from flat edge outward (reversed — 1 at flat, 0 at edge)
        mask[:feather_lat] = gauss.flip(0)
        # Right side: gaussian from flat edge outward
        mask[feather_lat + flat_size:] = gauss

    return mask


# ---------------------------------------------------------------------------
# Tile grid visualization
# ---------------------------------------------------------------------------

def _draw_tile_grid(H_lat, W_lat, tiles):
    H_px = H_lat * 8
    W_px = W_lat * 8
    img  = torch.zeros(H_px, W_px, 3, dtype=torch.float32)
    red  = torch.tensor([1.0, 0.0, 0.0])
    lw   = 2

    for (y0, y1, x0, x1) in tiles:
        py0, py1 = y0 * 8, y1 * 8
        px0, px1 = x0 * 8, x1 * 8
        img[max(0, py0):min(H_px, py0 + lw), px0:px1] = red
        img[max(0, py1 - lw):min(H_px, py1), px0:px1] = red
        img[py0:py1, max(0, px0):min(W_px, px0 + lw)] = red
        img[py0:py1, max(0, px1 - lw):min(W_px, px1)] = red

    return img.unsqueeze(0)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TiledSamplerCustomAdvanced:

    TILE_FACTORS = ["/2", "/4", "/8"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise":              ("NOISE",),
                "guider":             ("GUIDER",),
                "sampler":            ("SAMPLER",),
                "sigmas":             ("SIGMAS",),
                "latent_image":       ("LATENT",),
                "tile_factor":        (cls.TILE_FACTORS, {
                    "default": "/2",
                    "tooltip": "Number of tiles along the short side.",
                }),
                "context_size":       ("INT", {
                    "default": 256, "min": 0, "max": 512, "step": 32,
                    "tooltip": "Extra pixels of surrounding image the model sees as context.",
                }),
                "seam_flat_width":    ("INT", {
                    "default": 128, "min": 8, "max": 256, "step": 8,
                    "tooltip": "Width of the fully regenerated zone in pixels, centered on each seam. "
                               "This zone gets mask=1.0 — full inpainting strength.",
                }),
                "seam_feather":       ("INT", {
                    "default": 128, "min": 0, "max": 256, "step": 8,
                    "tooltip": "Width of gaussian falloff on each side of the flat zone. "
                               "Total sampled strip = seam_flat_width + 2 * seam_feather. "
                               "Larger = smoother blend into surrounding image.",
                }),
                "seam_fix_max_sigma": ("FLOAT", {
                    "default": 
                    0.7, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Max sigma for the seam fix pass. Input sigmas are rescaled "
                               "so the peak equals this value, steps stay proportional.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("output", "tiles")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image,
               tile_factor, context_size,
               seam_flat_width, seam_feather, seam_fix_max_sigma):

        latent = latent_image.copy()
        latent_samples = latent["samples"]

        latent_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_samples, latent.get("downscale_ratio_spacial", None)
        )

        B, C, H_lat, W_lat = latent_samples.shape

        short_lat   = min(H_lat, W_lat)
        long_lat    = max(H_lat, W_lat)
        factor      = int(tile_factor[1:])
        ctx_lat = context_size // 8

        n_short = factor
        _, tile_lat = _build_1d(short_lat, n_short)
        n_long = math.ceil(long_lat / tile_lat)

        if H_lat <= W_lat:
            n_rows, n_cols = n_short, n_long
        else:
            n_rows, n_cols = n_long, n_short

        tiles, tile_h, tile_w = _build_tiles(H_lat, W_lat, n_rows, n_cols)

        # Store row/col coords for seam position calculation
        rows_coords, _ = _build_1d(H_lat, n_rows)
        cols_coords, _ = _build_1d(W_lat, n_cols)
        seam_ys, seam_xs = _get_seam_positions(rows_coords, cols_coords)

        noise_mask  = latent.get("noise_mask", None)
        total_tiles = len(tiles)

        print(f"[TiledSampler] {n_rows}×{n_cols} = {total_tiles} tile(s) | "
              f"tile={tile_lat*8}px context={context_size}px")

        tile_grid_img = _draw_tile_grid(H_lat, W_lat, tiles)

        # ---------------------------------------------------------------
        # Pass 1: tiled sampling
        # ---------------------------------------------------------------
        canvas = latent_samples.clone()

        accum  = torch.zeros_like(latent_samples)
        weight = torch.zeros(B, 1, H_lat, W_lat,
                             device=latent_samples.device, dtype=latent_samples.dtype)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        pbar = comfy.utils.ProgressBar(total_tiles) if comfy.utils.PROGRESS_BAR_ENABLED else None

        for tile_idx, (y0, y1, x0, x1) in enumerate(tiles):
            th, tw = y1 - y0, x1 - x0

            cy0 = max(0,     y0 - ctx_lat)
            cy1 = min(H_lat, y1 + ctx_lat)
            cx0 = max(0,     x0 - ctx_lat)
            cx1 = min(W_lat, x1 + ctx_lat)
            ry0 = y0 - cy0;  ry1 = ry0 + th
            rx0 = x0 - cx0;  rx1 = rx0 + tw

            print(f"[TiledSampler] Tile {tile_idx + 1}/{total_tiles} "
                  f"x={x0*8}:{x1*8} y={y0*8}:{y1*8} ({tw*8}×{th*8}px)")

            padded_in = canvas[:, :, cy0:cy1, cx0:cx1]

            tile_latent = latent.copy()
            tile_latent["samples"] = padded_in

            tile_mask = None
            if noise_mask is not None:
                tile_mask = noise_mask[:, :, cy0:cy1, cx0:cx1]
                tile_latent["noise_mask"] = tile_mask

            result = guider.sample(
                noise.generate_noise(tile_latent),
                padded_in, sampler, sigmas,
                denoise_mask=tile_mask, disable_pbar=disable_pbar, seed=noise.seed,
            ).to(comfy.model_management.intermediate_device())

            tile_out = result[:, :, ry0:ry1, rx0:rx1]
            canvas[:, :, y0:y1, x0:x1] = tile_out.detach()

            # Flat weight — no overlap so no blending needed, seam fix handles boundaries
            accum [:, :, y0:y1, x0:x1] = tile_out
            weight[:, :, y0:y1, x0:x1] = 1.0

            if pbar is not None:
                pbar.update(1)

        tiled_result = accum / weight.clamp(min=1e-6)

        # ---------------------------------------------------------------
        # Pass 2: seam fix — strip-based, one crop per seam line
        # ---------------------------------------------------------------
        final = tiled_result.clone()

        if (len(seam_ys) > 0 or len(seam_xs) > 0) and seam_fix_max_sigma > 0:
            flat_half    = max(1, (seam_flat_width // 8) // 2)
            feather_lat  = max(1, seam_feather // 8)
            seam_half    = flat_half + feather_lat  # total half-width of sampled strip
            # Rescale sigmas so peak = seam_fix_max_sigma, steps stay proportional
            peak = sigmas.max()
            seam_sigmas = sigmas * (seam_fix_max_sigma / peak.clamp(min=1e-6))
            total_seams  = len(seam_ys) + len(seam_xs)
            seam_idx     = 0

            print(f"[TiledSampler] Seam fix | {total_seams} seam(s) | "
                  f"flat={seam_flat_width}px feather={seam_feather}px "
                  f"total_strip={seam_flat_width + 2*seam_feather}px "
                  f"max_sigma={seam_fix_max_sigma} (input peak={sigmas.max():.3f})")

            seam_pbar = comfy.utils.ProgressBar(total_seams) if comfy.utils.PROGRESS_BAR_ENABLED else None

            # Horizontal seams (y boundaries)
            for sy in seam_ys:
                y0 = max(0, sy - seam_half)
                y1 = min(H_lat, sy + seam_half + 1)
                strip_h = y1 - y0

                # Context padding on top and bottom
                cy0 = max(0, y0 - ctx_lat)
                cy1 = min(H_lat, y1 + ctx_lat)
                ry0 = y0 - cy0
                ry1 = ry0 + strip_h

                # Full width strip crop from final latent
                strip_in = final[:, :, cy0:cy1, :]  # (B, C, padded_h, W)

                # Build mask: flat 1.0 on seam, feathered edges
                # mask shape matches strip (not padded) — (strip_h, W_lat)
                flat_h = strip_h - 2 * feather_lat
                strip_mask_1d = _make_strip_mask(max(1, flat_h), feather_lat,
                                                  final.device, final.dtype)
                mask_h = strip_mask_1d.shape[0]
                strip_mask = strip_mask_1d.view(1, 1, mask_h, 1).expand(B, 1, mask_h, W_lat)

                # Pad mask to padded height (0 in context zones)
                full_mask = torch.zeros(B, 1, cy1 - cy0, W_lat,
                                        device=final.device, dtype=final.dtype)
                full_mask[:, :, ry0:ry1, :] = strip_mask

                strip_latent = latent.copy()
                strip_latent["samples"] = strip_in
                strip_latent["noise_mask"] = full_mask

                seam_idx += 1
                print(f"[TiledSampler] Seam {seam_idx}/{total_seams} "
                      f"horizontal y={y0*8}:{y1*8}px ({strip_h*8}px tall)")

                strip_result = guider.sample(
                    noise.generate_noise(strip_latent),
                    strip_in, sampler, seam_sigmas,
                    denoise_mask=full_mask, disable_pbar=disable_pbar, seed=noise.seed,
                ).to(comfy.model_management.intermediate_device())

                # Blend back using mask
                strip_out = strip_result[:, :, ry0:ry1, :]
                final[:, :, y0:y1, :] = (
                    final[:, :, y0:y1, :] * (1.0 - strip_mask) +
                    strip_out * strip_mask
                )

                if seam_pbar:
                    seam_pbar.update(1)

            # Vertical seams (x boundaries)
            for sx in seam_xs:
                x0 = max(0, sx - seam_half)
                x1 = min(W_lat, sx + seam_half + 1)
                strip_w = x1 - x0

                # Context padding left and right
                cx0 = max(0, x0 - ctx_lat)
                cx1 = min(W_lat, x1 + ctx_lat)
                rx0 = x0 - cx0
                rx1 = rx0 + strip_w

                strip_in = final[:, :, :, cx0:cx1]  # (B, C, H, padded_w)

                flat_w = strip_w - 2 * feather_lat
                strip_mask_1d = _make_strip_mask(max(1, flat_w), feather_lat,
                                                  final.device, final.dtype)
                mask_w = strip_mask_1d.shape[0]
                strip_mask = strip_mask_1d.view(1, 1, 1, mask_w).expand(B, 1, H_lat, mask_w)

                full_mask = torch.zeros(B, 1, H_lat, cx1 - cx0,
                                        device=final.device, dtype=final.dtype)
                full_mask[:, :, :, rx0:rx1] = strip_mask

                strip_latent = latent.copy()
                strip_latent["samples"] = strip_in
                strip_latent["noise_mask"] = full_mask

                seam_idx += 1
                print(f"[TiledSampler] Seam {seam_idx}/{total_seams} "
                      f"vertical x={x0*8}:{x1*8}px ({strip_w*8}px wide)")

                strip_result = guider.sample(
                    noise.generate_noise(strip_latent),
                    strip_in, sampler, seam_sigmas,
                    denoise_mask=full_mask, disable_pbar=disable_pbar, seed=noise.seed,
                ).to(comfy.model_management.intermediate_device())

                strip_out = strip_result[:, :, :, rx0:rx1]
                final[:, :, :, x0:x1] = (
                    final[:, :, :, x0:x1] * (1.0 - strip_mask) +
                    strip_out * strip_mask
                )

                if seam_pbar:
                    seam_pbar.update(1)

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = final
        return (out, tile_grid_img)


NODE_CLASS_MAPPINGS = {
    "TiledSamplerCustomAdvanced": TiledSamplerCustomAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledSamplerCustomAdvanced": "Tiled Sampler (Custom Advanced)",
}