import logging
import torch
import comfy.patcher_extension
import comfy.model_patcher


class CacheHolder:
    def __init__(self, cache_interval, start_percent, end_percent, verbose, output_channels):
        self.name = "WtlCache"
        self.cache_interval = max(1, int(cache_interval))
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.verbose = verbose
        self.output_channels = output_channels
        self.start_t = 0.0
        self.end_t = 0.0
        self.model_sampling = None
        self.sigma_schedule = None
        self.first_cond_uuid = None
        self.initial_step = True
        self.skip_current_step = False
        self.window_step = 0
        self.uuid_cache_diffs = {}
        self.uuid_x0 = {}
        self.uuid_prev_x0 = {}
        self.total_steps_skipped = 0
        self.state_metadata = None
        self.cached_sigma = 1.0
        self.prev_sigma = 1.0
        # None = untested, True/False latched after first full compute (per-model, not per-run)
        self.x0_valid = None

    def prepare_timesteps(self, model_sampling):
        self.model_sampling = model_sampling
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def set_sigma_schedule(self, sigmas):
        try:
            self.sigma_schedule = sigmas.detach().to("cpu").flatten()
        except Exception:
            self.sigma_schedule = None

    def is_past_end_timestep(self, sigmas) -> bool:
        return not (sigmas[0] > self.end_t).item()

    def in_window(self, sigmas) -> bool:
        return (sigmas[0] <= self.start_t).item()

    def has_first_cond_uuid(self, uuids) -> bool:
        return self.first_cond_uuid in uuids

    def can_apply_cache_diff(self, uuids) -> bool:
        return all(uuid in self.uuid_cache_diffs for uuid in uuids)

    def check_metadata(self, x) -> bool:
        metadata = (x.device, x.dtype, x.shape[1:])
        if self.state_metadata is None:
            self.state_metadata = metadata
            return True
        if metadata == self.state_metadata:
            return True
        logging.warning(f"{self.name} - tensor shape/dtype/device changed, resetting state")
        self.reset()
        return False

    def _c_in(self, sigma, ref):
        sigma_t = torch.tensor([sigma], device=ref.device, dtype=ref.dtype)
        ones = torch.ones((1,) + tuple(ref.shape[1:]), device=ref.device, dtype=ref.dtype)
        return self.model_sampling.calculate_input(sigma_t, ones).flatten()[0]

    def _validate_x0(self, output, xc, x_orig, x0, sigma):
        # Confirm the denoised formula is invertible as model_output = (x_orig - x0) / sigma.
        # True for eps and flow parameterizations; v-pred would fail form_ok and disable x0 mode.
        mo_rec = (x_orig - x0) / sigma
        form_ok = torch.allclose(mo_rec, output, rtol=1e-2, atol=1e-2)
        sigma_t = torch.tensor([sigma], device=xc.device, dtype=xc.dtype)
        ones = torch.ones((1,) + tuple(xc.shape[1:]), device=xc.device, dtype=xc.dtype)
        lin_ok = torch.allclose(self.model_sampling.calculate_input(sigma_t, 2 * ones),
                                2 * self.model_sampling.calculate_input(sigma_t, ones),
                                rtol=1e-3, atol=1e-4)
        ok = bool(form_ok and lin_ok)
        logging.info(f"{self.name} - x0 mode "
                     f"{'VALID' if ok else 'INVALID -> disabling'} "
                     f"(invertible={bool(form_ok)}, linear_input={bool(lin_ok)})")
        return ok

    def _maybe_compute_x0(self, output, xc, sigma):
        if self.model_sampling is None or sigma <= 1e-6:
            return None
        if self.x0_valid is False:
            return None
        try:
            c_in = self._c_in(sigma, xc)
            if not torch.isfinite(c_in) or c_in == 0:
                self.x0_valid = False
                return None
            x_orig = xc / c_in
            sigma_t = torch.tensor([sigma], device=xc.device, dtype=xc.dtype)
            x0 = self.model_sampling.calculate_denoised(sigma_t, output, x_orig)
            if self.x0_valid is None:
                self.x0_valid = self._validate_x0(output, xc, x_orig, x0, sigma)
            return x0 if self.x0_valid else None
        except Exception as e:
            logging.warning(f"{self.name} - x0 reconstruction failed ({e}); disabling")
            self.x0_valid = False
            return None

    def update_cache_diff(self, output, x, uuids, sigma):
        self.prev_sigma = self.cached_sigma
        self.cached_sigma = sigma
        for uuid in uuids:
            if uuid in self.uuid_x0:
                self.uuid_prev_x0[uuid] = self.uuid_x0[uuid]
        if output.shape[1:] != x.shape[1:]:
            slicing = []
            skip_dim = True
            for dim_o, dim_x in zip(output.shape, x.shape):
                if not skip_dim and dim_o != dim_x:
                    slicing.append(slice(dim_x - dim_o, None))
                else:
                    slicing.append(slice(None))
                skip_dim = False
            x = x[tuple(slicing)]
        diff = output - x
        x0_full = self._maybe_compute_x0(output, x, sigma)
        batch_offset = diff.shape[0] // len(uuids)
        for i, uuid in enumerate(uuids):
            self.uuid_cache_diffs[uuid] = diff[i * batch_offset:(i + 1) * batch_offset, ...]
            if x0_full is not None:
                self.uuid_x0[uuid] = x0_full[i * batch_offset:(i + 1) * batch_offset, ...]

    def _step_index(self, sigma):
        if self.sigma_schedule is None:
            return None
        return int((self.sigma_schedule - float(sigma)).abs().argmin().item())

    def _extrapolation_t(self, sigma):
        # Parameterize by step position in the actual polled schedule rather than raw sigma
        # value, so geometry respects whatever curve shape the scheduler produced.
        i_now = self._step_index(sigma)
        i_c = self._step_index(self.cached_sigma)
        i_p = self._step_index(self.prev_sigma)
        if i_now is None or i_c is None or i_p is None:
            return None
        denom = i_c - i_p
        return None if denom == 0 else (i_now - i_c) / denom

    def _adaptive_cap(self, x0_cur, x0_prev):
        # Allow further extrapolation when x0 is stable (barely moving), clamp tight when
        # it is still converging fast. Self-calibrates per region of the sigma schedule.
        stability = (x0_cur - x0_prev).norm().item() / (x0_cur.norm().item() + 1e-8)
        return max(0.5, min(2.0, 1.0 / (stability * 4.0 + 0.5)))

    def predict_diff(self, uuid, sigma, xc):
        if self.x0_valid and uuid in self.uuid_x0 and sigma > 1e-6 and self.model_sampling is not None:
            x0_cur = self.uuid_x0[uuid]
            if uuid in self.uuid_prev_x0:
                t_raw = self._extrapolation_t(sigma)
                if t_raw is not None:
                    cap = self._adaptive_cap(x0_cur, self.uuid_prev_x0[uuid])
                    t = max(0.0, min(t_raw, cap))
                    x0_now = x0_cur + t * (x0_cur - self.uuid_prev_x0[uuid])
                    if self.verbose:
                        logging.info(f"{self.name} - x0 t={t_raw:.3f} -> {t:.3f} (cap {cap:.2f})")
                else:
                    x0_now = x0_cur
            else:
                # One anchor: hold x0 flat. x0 tends toward the final image (not zero),
                # so an origin-ratio would be wrong here.
                x0_now = x0_cur
            c_in = self._c_in(sigma, xc)
            x_orig = xc / c_in
            return (x_orig - x0_now) / sigma - xc

        # Fallback for models where x0 reconstruction is not supported.
        s2 = self.cached_sigma
        return self.uuid_cache_diffs[uuid] * (sigma / s2 if s2 > 0 else 1.0)

    def apply_cache_diff(self, x, uuids, sigma):
        if self.first_cond_uuid in uuids:
            self.total_steps_skipped += 1
        batch_offset = x.shape[0] // len(uuids)
        for i, uuid in enumerate(uuids):
            xc_i = x[i * batch_offset:(i + 1) * batch_offset, ...]
            predicted = self.predict_diff(uuid, sigma, xc_i)
            batch_slice = [slice(i * batch_offset, (i + 1) * batch_offset)]
            if x.shape[1:] != predicted.shape[1:]:
                slicing = []
                skip_this_dim = True
                for dim_u, dim_x in zip(predicted.shape, x.shape):
                    if skip_this_dim:
                        skip_this_dim = False
                        continue
                    if dim_u != dim_x:
                        slicing.append(slice(dim_x - dim_u, None))
                    else:
                        slicing.append(slice(None))
                batch_slice = batch_slice + slicing
            x[tuple(batch_slice)] += predicted.to(x.device)
        return x

    def reset(self):
        self.first_cond_uuid = None
        self.initial_step = True
        self.skip_current_step = False
        self.window_step = 0
        del self.uuid_cache_diffs
        self.uuid_cache_diffs = {}
        self.uuid_x0 = {}
        self.uuid_prev_x0 = {}
        self.total_steps_skipped = 0
        self.state_metadata = None
        self.cached_sigma = 1.0
        self.prev_sigma = 1.0
        return self

    def clone(self):
        return CacheHolder(self.cache_interval, self.start_percent, self.end_percent,
                           self.verbose, self.output_channels)


def cache_forward_wrapper(executor, *args, **kwargs):
    transformer_options = args[-1]
    if not isinstance(transformer_options, dict):
        transformer_options = kwargs.get("transformer_options")
    cache: CacheHolder = transformer_options["wtlcache"]

    x = args[0][:, :cache.output_channels]
    sigmas = transformer_options["sigmas"]
    uuids = transformer_options["uuids"]

    if sigmas is not None and cache.is_past_end_timestep(sigmas):
        return executor(*args, **kwargs)

    if cache.in_window(sigmas):
        cache.check_metadata(x)
        can_apply = cache.can_apply_cache_diff(uuids)

        if cache.skip_current_step and can_apply:
            return cache.apply_cache_diff(x, uuids, sigmas[0].item())

        if cache.initial_step:
            cache.first_cond_uuid = uuids[0]
            cache.initial_step = False

        if cache.has_first_cond_uuid(uuids):
            compute_full = (cache.window_step % cache.cache_interval == 0)
            cache.window_step += 1
            if not compute_full and can_apply:
                cache.skip_current_step = True
                return cache.apply_cache_diff(x, uuids, sigmas[0].item())

    full_output = executor(*args, **kwargs)
    output = full_output[:, :cache.output_channels]
    cache.update_cache_diff(output, x, uuids, sigmas[0].item())
    return full_output


def cache_calc_cond_batch_wrapper(executor, *args, **kwargs):
    model_options = args[-1]
    cache: CacheHolder = model_options["transformer_options"]["wtlcache"]
    cache.skip_current_step = False
    return executor(*args, **kwargs)


def cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    orig_model_options = guider.model_options
    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        cache = guider.model_options["transformer_options"]["wtlcache"].clone().prepare_timesteps(
            guider.model_patcher.model.model_sampling
        )
        cache.set_sigma_schedule(args[3])
        guider.model_options["transformer_options"]["wtlcache"] = cache
        logging.info(f"{cache.name} enabled - interval: {cache.cache_interval}, "
                     f"window: [{cache.start_percent}, {cache.end_percent}]")
        return executor(*args, **kwargs)
    finally:
        cache = guider.model_options["transformer_options"]["wtlcache"]
        total_steps = len(args[3]) - 1
        try:
            speedup = total_steps / (total_steps - cache.total_steps_skipped)
        except ZeroDivisionError:
            speedup = 1.0
        logging.info(f"{cache.name} - skipped {cache.total_steps_skipped}/{total_steps} steps "
                     f"({speedup:.2f}x speedup).")
        cache.reset()
        guider.model_options = orig_model_options


class CacheAcceleratorC:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cache_interval": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "WtlNodes/sampling"

    def patch(self, model, cache_interval, start_percent, end_percent, verbose):
        model = model.clone()
        holder = CacheHolder(
            cache_interval, start_percent, end_percent, verbose,
            output_channels=model.model.latent_format.latent_channels,
        )
        model.model_options["transformer_options"]["wtlcache"] = holder
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "wtlcache", cache_sample_wrapper)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, "wtlcache", cache_calc_cond_batch_wrapper)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "wtlcache", cache_forward_wrapper)
        return (model,)


NODE_CLASS_MAPPINGS = {"CacheAccelerator": CacheAcceleratorC}
NODE_DISPLAY_NAME_MAPPINGS = {"CacheAccelerator": "Cache Accelerator"}
