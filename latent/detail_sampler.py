import logging
import torch
import torch.nn.functional as F
import comfy.sample
import comfy.model_management
import comfy.utils


# ---------------------------------------------------------------------------
# Detection maps
#
# Detection reads x0 straight out of the sampler's callback partway through the
# run. The generation itself is never interrupted, so the base image is exactly
# what a plain sampler would have produced — the detail pass is purely additive.
# ---------------------------------------------------------------------------

def _chan_norm(t):
    """Scale each channel by its own std so high-variance latent channels
    don't dominate the reduction across channels."""
    return t / (t.std(dim=(2, 3), keepdim=True) + 1e-6)


def _hf_energy(x0, pool):
    """Local high-frequency energy of x0, pooled to a coarse grid.

    A region whose spectrum is pressed against the latent Nyquist limit is the
    one that is resolution-starved — that is what this measures.
    """
    blur = F.avg_pool2d(x0, 3, stride=1, padding=1)
    hf = _chan_norm(x0) - _chan_norm(blur)
    hf = hf.abs().mean(dim=1, keepdim=True)
    return F.avg_pool2d(hf, pool, stride=pool)


def _residual(x0_cur, x0_prev, pool):
    """How much x0 is still moving, pooled to the same coarse grid."""
    d = (_chan_norm(x0_cur) - _chan_norm(x0_prev)).abs().mean(dim=1, keepdim=True)
    return F.avg_pool2d(d, pool, stride=pool)


def _unit(t):
    lo, hi = t.min(), t.max()
    return (t - lo) / (hi - lo + 1e-8)


def _detail_score(x0_cur, x0_prev, pool):
    """Structure that exists AND has stopped moving.

    High HF alone also selects detail the model is still actively changing.
    Low residual alone selects flat sky. The product is what we want.
    """
    hf = _unit(_hf_energy(x0_cur, pool))
    if x0_prev is None:
        return hf
    res = _unit(_residual(x0_cur, x0_prev, pool))
    return hf * (1.0 - res)


# ---------------------------------------------------------------------------
# Blob selection
# ---------------------------------------------------------------------------

def _connected_components(mask, max_iter=None):
    """Label 8-connected blobs by iterative max-propagation.

    Each foreground cell starts with a unique id; a 3x3 max-pool masked back to
    the foreground floods the largest id through each blob until stable. Pure
    torch — no scipy dependency for one function. Equality is checked every 8
    iterations rather than every one, since each check forces a device sync.
    """
    h, w = mask.shape[-2:]
    if max_iter is None:
        max_iter = h + w
    labels = torch.arange(1, h * w + 1, device=mask.device,
                          dtype=torch.float32).view(1, 1, h, w) * mask
    for _ in range(max_iter // 8 + 1):
        prev = labels
        for _ in range(8):
            labels = F.max_pool2d(labels, 3, stride=1, padding=1) * mask
        if torch.equal(labels, prev):
            break
    return labels


def _select_blobs(score, threshold, min_area, max_regions):
    """Threshold the score map, label islands, keep the strongest few.

    Ranking is by MEAN score, not total: total area would let one large mass
    win every slot, which is the whole problem islands are meant to solve.
    """
    mask = (score >= threshold).float()
    if float(mask.sum()) == 0.0:
        return [], None

    labels = _connected_components(mask)
    flat = labels.flatten().long()
    counts = torch.bincount(flat)
    sums = torch.bincount(flat, weights=score.flatten().float(),
                          minlength=counts.numel())
    means = sums / counts.clamp(min=1)

    valid = torch.nonzero(counts >= max(1, int(min_area))).flatten()
    valid = valid[valid > 0]
    if valid.numel() == 0:
        return [], None

    order = torch.argsort(means[valid], descending=True)
    chosen = valid[order][:max_regions]
    return [{"label": int(l), "score": float(means[int(l)]), "area": int(counts[int(l)])}
            for l in chosen.tolist()], labels


def _fit_axis(lo, hi, limit, pad, mult):
    """Grow [lo,hi) by pad, round out to a multiple of `mult`, clamp to limit.

    Latent dims must stay divisible by 8 — SDXL's UNet downsamples three times,
    and DiT patching needs alignment too. No size ceiling: the island's own
    extent decides the window.
    """
    lo = max(0, lo - pad)
    hi = min(limit, hi + pad)
    want = ((hi - lo + mult - 1) // mult) * mult
    want = min(max(want, mult), max(mult, (limit // mult) * mult))
    c = (lo + hi) / 2
    a = min(max(0, int(round(c - want / 2))), max(0, limit - want))
    return a, a + want


def _blob_box(blob_mask, H, W, pad, mult=8):
    """Crop window for an island: its bounding box plus context, nothing else.

    Follows the island's real extent and aspect — no fixed size, no forced
    square, no ceiling.
    """
    ys, xs = torch.nonzero(blob_mask[0, 0], as_tuple=True)
    y0, y1 = _fit_axis(int(ys.min()), int(ys.max()) + 1, H, pad, mult)
    x0, x1 = _fit_axis(int(xs.min()), int(xs.max()) + 1, W, pad, mult)
    return y0, y1, x0, x1


def _subschedule(sigmas, start_idx, n_steps):
    """Resample the tail of the schedule to n_steps, preserving its curve."""
    tail = sigmas[start_idx:]
    if tail.numel() < 2:
        tail = sigmas[-2:]
    if tail.numel() - 1 == n_steps:
        return tail
    pos = torch.linspace(0, tail.numel() - 1, n_steps + 1, device=tail.device)
    lo = pos.floor().long().clamp(0, tail.numel() - 1)
    hi = pos.ceil().long().clamp(0, tail.numel() - 1)
    frac = (pos - lo.to(pos.dtype)).to(tail.dtype)
    return tail[lo] * (1 - frac) + tail[hi] * frac


def _soften(m, radius):
    """Box-blur a binary island mask twice for a smooth composite falloff.

    Done at latent resolution — a 97-tap box blur at pixel resolution would be
    O(k^2) per pixel and far too slow. The upsample to pixel space smooths it
    the rest of the way.
    """
    r = int(radius)
    if r <= 0:
        return m.clamp(0, 1)
    k = 2 * r + 1
    m = F.avg_pool2d(m, k, stride=1, padding=r)
    m = F.avg_pool2d(m, k, stride=1, padding=r)
    return (m / (m.max() + 1e-8)).clamp(0, 1)


def _snap(v, mult):
    return max(mult, int(round(v / mult)) * mult)


def _resize_img(img, w, h, method="lanczos"):
    """(B,H,W,C) -> resized (B,H,W,C). comfy's helper wants BCHW."""
    out = comfy.utils.common_upscale(img.movedim(-1, 1).float(), w, h, method, "disabled")
    return out.movedim(1, -1)


# comfy's VAE.encode/decode already retry tiled on OOM, but only on OOM — and
# only if the fallback itself fits. Forcing tiled up front keeps the peak low
# instead of spiking first and recovering second.

def _vae_encode(vae, pixels, tiled):
    if tiled:
        return vae.encode_tiled(pixels)
    return vae.encode(pixels)


def _vae_decode(vae, samples, tiled):
    if tiled:
        return vae.decode_tiled(samples)
    return vae.decode(samples)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class DetailSamplerCustomAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise":            ("NOISE",),
                "guider":           ("GUIDER",),
                "sampler":          ("SAMPLER",),
                "sigmas":           ("SIGMAS",),
                "latent_image":     ("LATENT",),
                "vae":              ("VAE",),
                "detect_at":        ("FLOAT", {
                    "default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Point in the schedule the detail map is read from. "
                               "The generation itself always runs to completion — "
                               "this only decides when to look.",
                }),
                "mask_threshold":   ("FLOAT", {
                    "default": 0.35, "min": 0.02, "max": 0.95, "step": 0.01,
                    "tooltip": "Cutoff on the normalised detail score. Everything "
                               "above it becomes mask; connected areas become one "
                               "island each. Lower = larger islands that can merge.",
                }),
                "detail_scale":     ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25,
                    "tooltip": "Upscale applied to every island equally, in pixel "
                               "space, before re-rendering.",
                }),
                "detail_denoise":   ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of each island is re-rendered, as a fraction "
                               "of the schedule. The strength dial: 0.3 refines, "
                               "0.7+ reinvents.",
                }),
                "detail_steps":     ("INT", {
                    "default": 12, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Steps per island.",
                }),
                "detail_strength":  ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Composite opacity of the detailed result. 0 = detector "
                               "only: emits the mask, skips every detail pass, returns "
                               "the base image untouched.",
                }),
                "min_blob_size":    ("INT", {
                    "default": 48, "min": 8, "max": 512, "step": 8,
                    "tooltip": "Islands smaller than this many pixels square are "
                               "discarded as specks.",
                }),
                "context_pad":      ("INT", {
                    "default": 64, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Surrounding pixels included in each crop. Too little "
                               "and the crop loses the context it needs to stay "
                               "consistent with the scene.",
                }),
                "max_regions":      ("INT", {
                    "default": 3, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Islands are ranked by mean score; this many are kept.",
                }),
                "preserve_context":  ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Hold everything outside the island fixed during the "
                               "detail pass (noise_mask inpainting) instead of "
                               "regenerating the whole crop and blending after. The "
                               "island is then rendered against the real surroundings "
                               "at every step, so it fits what it is pasted into. "
                               "Off = the previous behaviour, for A/B.",
                }),
                "tiled_vae":        ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force tiled VAE encode/decode for the island crops. "
                               "comfy already retries tiled after an OOM, but only "
                               "after spiking first; this keeps the peak low from "
                               "the start. Turn on if large islands crash.",
                }),
                "feather":          ("INT", {
                    "default": 48, "min": 0, "max": 256, "step": 8,
                    "tooltip": "Composite falloff in pixels, applied to the island "
                               "shape.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT")
    RETURN_NAMES = ("image", "mask", "latent")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image, vae,
               detect_at, mask_threshold, detail_scale, detail_denoise,
               detail_steps, detail_strength, min_blob_size, context_pad,
               max_regions, preserve_context, tiled_vae, feather):

        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent["samples"],
            latent.get("downscale_ratio_spacial", None)
        )
        latent["samples"] = samples
        latent.pop("downscale_ratio_spacial", None)

        noise_mask = latent.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        intermediate = comfy.model_management.intermediate_device()
        total_steps = len(sigmas) - 1

        # -------------------------------------------------------------------
        # Pass 1 — the full generation, uninterrupted. The callback only reads
        # x0; it never writes, so this is byte-identical to a plain sampler.
        # -------------------------------------------------------------------
        detect_step = min(max(1, int(round(total_steps * detect_at))), total_steps - 1) \
            if total_steps > 1 else 0
        cap = {"cur": None, "prev": None}

        def capture(step, denoised, x, total):
            if step <= detect_step:
                cap["prev"] = cap["cur"]
                cap["cur"] = denoised.detach()

        base_lat = guider.sample(
            noise.generate_noise(latent), samples, sampler, sigmas,
            denoise_mask=noise_mask, callback=capture,
            disable_pbar=disable_pbar, seed=noise.seed,
        ).to(intermediate)

        out_latent = latent.copy()
        out_latent["samples"] = base_lat

        image = _vae_decode(vae, base_lat, tiled_vae)
        if image.ndim == 5:  # video VAEs fold frames in; take the first
            image = image[0]
        img_h, img_w = image.shape[1], image.shape[2]

        if samples.ndim != 4 or cap["cur"] is None:
            logging.warning("[DetailSampler] no detail map available - base image only")
            return (image, torch.zeros(1, img_h, img_w), out_latent)

        H, W = samples.shape[-2:]
        ratio = max(1, img_h // H)
        align = ratio * 8  # latent must stay /8, so pixels must stay /(ratio*8)

        # -------------------------------------------------------------------
        # Detection
        # -------------------------------------------------------------------
        pool = max(1, min(H, W) // 32)
        x0 = cap["cur"].float()
        prev = cap["prev"].float() if cap["prev"] is not None else None
        coarse = _detail_score(x0[:1], prev[:1] if prev is not None else None, pool)
        score = _unit(F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False))

        min_area = max(1, (max(1, min_blob_size // ratio)) ** 2)
        blobs, labels = _select_blobs(score, mask_threshold, min_area, max_regions)

        if not blobs:
            logging.info(f"[DetailSampler] no island above threshold "
                         f"{mask_threshold:.2f} - base image only")
            return (image, torch.zeros(1, img_h, img_w), out_latent)

        for blob in blobs:
            blob["mask"] = (labels == blob["label"]).float()
            blob["box"] = _blob_box(blob["mask"], H, W, context_pad // ratio)

        # Union of softened island masks, at pixel resolution
        acc = torch.zeros(1, 1, H, W, device=score.device, dtype=torch.float32)
        for blob in blobs:
            y0, y1, x0b, x1b = blob["box"]
            soft = _soften(blob["mask"][:, :, y0:y1, x0b:x1b], feather // ratio)
            blob["soft"] = soft
            acc[:, :, y0:y1, x0b:x1b] = torch.maximum(acc[:, :, y0:y1, x0b:x1b], soft)
        mask_px = F.interpolate(acc, size=(img_h, img_w),
                                mode="bilinear", align_corners=False)[0].cpu().clamp(0, 1)

        if detail_strength <= 0.0:
            logging.info(f"[DetailSampler] detector-only (strength 0) | "
                         f"{len(blobs)} island(s) | base image untouched")
            return (image, mask_px, out_latent)

        start_idx = min(max(0, int(round((1.0 - detail_denoise) * total_steps))),
                        max(0, total_steps - 1))
        sigmas_d = _subschedule(sigmas, start_idx, max(1, detail_steps))

        logging.info(
            f"[DetailSampler] detected at step {detect_step}/{total_steps} | "
            f"{len(blobs)} island(s) | x{detail_scale} | denoise {detail_denoise:.2f} "
            f"(sigma {float(sigmas_d[0]):.3f}) | {len(sigmas_d) - 1} step(s) | "
            f"strength {detail_strength:.2f} | "
            f"context {'held (noise_mask)' if preserve_context else 'regenerated'}")

        # -------------------------------------------------------------------
        # Pass 2 — a regular detailer, one island at a time.
        #
        # Crop the finished image, upscale in PIXEL space (latent interpolation
        # is what made this mushy before), re-render, scale back, composite.
        # Nothing is ever resampled down into the latent grid, so the added
        # detail survives into the output.
        # -------------------------------------------------------------------
        gen = torch.Generator(device="cpu").manual_seed((noise.seed or 0) + 0x5EED)
        result = image.clone().float()

        for i, blob in enumerate(blobs):
            y0, y1, x0b, x1b = blob["box"]
            py0, py1 = y0 * ratio, min(img_h, y1 * ratio)
            px0, px1 = x0b * ratio, min(img_w, x1b * ratio)
            ch, cw = py1 - py0, px1 - px0
            uh, uw = _snap(ch * detail_scale, align), _snap(cw * detail_scale, align)

            logging.info(f"[DetailSampler]   island {i + 1}/{len(blobs)} "
                         f"{cw}x{ch}px -> {uw}x{uh}px "
                         f"(score {blob['score']:.2f}, area {blob['area']})")

            crop = result[:, py0:py1, px0:px1, :]
            up = _resize_img(crop, uw, uh).clamp(0, 1)

            crop_lat = _vae_encode(vae, up, tiled_vae)
            del up
            fresh = torch.randn(crop_lat.shape, generator=gen,
                                dtype=torch.float32, device="cpu").to(crop_lat.device)

            # denoise_mask pins everything outside the island to the original
            # content on every model call, while still presenting it to the
            # model at the right noise level. Without it the surroundings get
            # regenerated too, so the island is rendered against a context that
            # drifts — and then thrown away at composite time.
            dmask = None
            if preserve_context:
                dmask = F.interpolate(blob["soft"].float(),
                                      size=tuple(crop_lat.shape[-2:]),
                                      mode="bilinear", align_corners=False)
                dmask = dmask.clamp(0, 1).to(crop_lat.device)

            done = guider.sample(
                fresh, crop_lat, sampler, sigmas_d, denoise_mask=dmask,
                disable_pbar=disable_pbar, seed=noise.seed,
            ).to(intermediate)
            del crop_lat, fresh, dmask

            # The sampler's working set is still resident here; releasing it
            # before the VAE decode is what keeps the peak from stacking.
            comfy.model_management.soft_empty_cache()

            patch = _vae_decode(vae, done, tiled_vae)
            del done
            if patch.ndim == 5:
                patch = patch[0]
            patch = _resize_img(patch, cw, ch).clamp(0, 1).to(result.device, result.dtype)

            m = F.interpolate(blob["soft"], size=(ch, cw), mode="bilinear",
                              align_corners=False)[0, 0]
            m = (m * detail_strength).unsqueeze(-1).to(result.device, result.dtype)
            base = result[:, py0:py1, px0:px1, :]
            result[:, py0:py1, px0:px1, :] = base + m * (patch - base)

            del crop, patch, base, m

        return (result.clamp(0, 1), mask_px, out_latent)


NODE_CLASS_MAPPINGS = {
    "DetailSamplerCustomAdvanced": DetailSamplerCustomAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailSamplerCustomAdvanced": "Detail Sampler (Custom Advanced) [Not Working]",
}
