import bisect
import logging
import math
import torch
import comfy.patcher_extension
import comfy.model_patcher


class CacheStream:
    """x0 history of one calc_cond_batch output: one pass of a sampler eval x one cond slot."""

    def __init__(self, signature, reason):
        self.signature = signature
        self.reason = reason
        self.anchors = []  # (tau, x0), oldest first
        self.last_seen = None

    def add(self, tau, x0, max_anchors):
        x0 = x0.detach().clone()
        if self.anchors and abs(tau - self.anchors[-1][0]) < 1e-3:
            # Same schedule position computed twice (e.g. Heun corrector then next step): keep the newer state.
            self.anchors[-1] = (tau, x0)
        else:
            self.anchors.append((tau, x0))
            del self.anchors[:-max_anchors]


def _patch_fingerprint(transformer_options):
    # Extra guidance passes (PAG, SLG, SAG...) are told apart from the main pass by their patches.
    replace = transformer_options.get("patches_replace", {})
    patches = transformer_options.get("patches", {})
    return (tuple(sorted((str(k), tuple(sorted(str(b) for b in v))) for k, v in replace.items())),
            tuple(sorted((str(k), len(v)) for k, v in patches.items())))


def _active_uuids(cond, sigma):
    # Same timestep-range test as samplers.get_area_and_mult: a prompt switch invalidates history.
    active = []
    for c in cond:
        start, end = c.get("timestep_start"), c.get("timestep_end")
        if start is not None and sigma > start:
            continue
        if end is not None and sigma < end:
            continue
        active.append(c.get("uuid"))
    return tuple(active)


def _follow_through(anchors):
    # Fraction of the previous x0 move that the latest move continued (per unit of schedule time), in [0, 1]:
    # 1 = steady trend, extrapolate fully; 0 = the move reversed (oscillation), hold x0.
    (t0, a0), (t1, a1), (t2, a2) = anchors[-3:]
    m1 = (a1 - a0) / (t1 - t0)
    m2 = (a2 - a1) / (t2 - t1)
    r = torch.dot(m2.flatten(), m1.flatten()).item() / max(torch.dot(m1.flatten(), m1.flatten()).item(), 1e-20)
    return min(max(r, 0.0), 1.0)


def _extrapolate(anchors, tau_now):
    (tau_prev, x0_prev), (tau_cur, x0_cur) = anchors[-2:]
    step = (tau_now - tau_cur) / (tau_cur - tau_prev) * (x0_cur - x0_prev)
    return x0_cur + _follow_through(anchors) * step


def _lowpass(x, sigma=2.0):
    # Separable Gaussian blur over the two spatial (last) dims of a latent; sigma in latent pixels.
    r = max(1, int(3 * sigma))
    c = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-c ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    shape = x.shape
    y = x.reshape(-1, 1, shape[-2], shape[-1])
    mode = "reflect" if min(shape[-2], shape[-1]) > r else "replicate"
    y = torch.nn.functional.conv2d(torch.nn.functional.pad(y, (r, r, 0, 0), mode=mode), g.view(1, 1, 1, -1))
    y = torch.nn.functional.conv2d(torch.nn.functional.pad(y, (0, 0, r, r), mode=mode), g.view(1, 1, -1, 1))
    return y.reshape(shape)


class CacheHolder:
    name = "WtlCache"
    min_anchors = 3
    max_anchors = 3

    def __init__(self, cache_interval, start_percent, end_percent, verbose):
        self.cache_interval = max(1, int(cache_interval))
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.verbose = verbose
        self.schedule = None
        self.steps = 0
        self.start_pos = 0.0
        self.end_pos = 0.0
        self.reset()

    def prepare(self, sigmas):
        neg_log = [-math.log(s) for s in sigmas.detach().flatten().float().cpu().tolist() if s > 0]
        if any(b < a for a, b in zip(neg_log, neg_log[1:])):
            raise RuntimeError(f"{self.name}: the sigma schedule is not a decreasing curve (e.g. flipped or restart "
                               f"sigmas), so steps cannot be cached. Remove Cache Accelerator from this workflow.")
        self.schedule = neg_log if len(neg_log) >= 2 else None
        # The window is a fraction of the sampling steps; positions are continuous step indices (see tau).
        self.steps = sigmas.numel() - 1
        self.start_pos = self.start_percent * self.steps
        self.end_pos = self.end_percent * self.steps
        return self

    def window_steps(self):
        first = math.ceil(self.start_pos - 1e-6)
        last = min(math.ceil(self.end_pos - 1e-6), self.steps)
        if last <= first:
            return f"no steps of {self.steps} (nothing will be skipped)"
        return f"steps {first + 1}-{last} of {self.steps}"

    def reset(self):
        self.streams = {}
        self.depth = 0
        self.eval_index = -1
        self.window_evals = 0
        self.pass_index = 0
        self.tau_now = 0.0
        self.last_tau = None
        self.in_window = False
        self.recording = False
        self.skip_planned = False
        self.model_called = False
        self.forced_reason = None
        self.total_evals = 0
        self.predicted_evals = 0
        self.forced = {}
        return self

    def clone(self):
        return type(self)(self.cache_interval, self.start_percent, self.end_percent, self.verbose)

    def tau(self, sigma):
        # Position on the polled schedule as a continuous step index (interpolated in log-sigma), so the
        # geometry follows the scheduler and off-grid evals (2S/SDE midpoints, churn) are not snapped.
        a = self.schedule
        if a is None:
            return 0.0  # single-step schedule: nothing can be skipped
        x = -math.log(max(sigma, 1e-12))
        j = min(max(bisect.bisect_right(a, x) - 1, 0), len(a) - 2)
        d = a[j + 1] - a[j]
        return j + ((x - a[j]) / d if d > 1e-12 else 0.0)

    def begin_eval(self, sigma):
        self.depth += 1
        if self.depth > 1:
            return
        self.eval_index += 1
        self.total_evals += 1
        self.pass_index = 0
        self.model_called = False
        self.forced_reason = None
        self.tau_now = self.tau(sigma)
        if self.last_tau is not None and self.tau_now < self.last_tau - 1e-3:
            # Schedule moved backwards (restart-style sampling): the history belongs to another trajectory.
            self.streams = {}
        self.last_tau = self.tau_now
        self.recording = self.tau_now < self.end_pos - 1e-6
        self.in_window = self.start_pos - 1e-6 <= self.tau_now < self.end_pos - 1e-6
        self.skip_planned = False
        if self.in_window:
            self.skip_planned = self.window_evals % self.cache_interval != 0
            self.window_evals += 1

    def end_eval(self):
        self.depth -= 1
        if self.depth > 0:
            return
        if self.in_window and self.pass_index == 0:
            raise RuntimeError(f"{self.name}: this sampler/guider does not go through calc_cond_batch, so its "
                               f"steps cannot be cached. Remove Cache Accelerator from this workflow.")
        if self.skip_planned:
            if self.model_called:
                reason = self.forced_reason or "unknown"
                self.forced[reason] = self.forced.get(reason, 0) + 1
            else:
                self.predicted_evals += 1
        if self.verbose:
            state = "predicted" if self.skip_planned and not self.model_called else "computed"
            logging.info(f"{self.name} - eval {self.eval_index} (step pos {self.tau_now:.2f}): {state}")

    def abort_eval(self):
        self.depth -= 1

    def streams_for_pass(self, conds, x_in, timestep, model_options):
        p = self.pass_index
        self.pass_index += 1
        to = model_options.get("transformer_options", {})
        window = getattr(to.get("context_window"), "index_list", None)
        base = (tuple(x_in.shape), _patch_fingerprint(to), tuple(window) if window is not None else None)
        sigma = float(timestep.flatten()[0])
        streams = []
        for i, cond in enumerate(conds):
            if cond is None:
                streams.append(None)
                continue
            signature = base + (_active_uuids(cond, sigma),)
            s = self.streams.get((p, i))
            if s is None or s.signature != signature or s.last_seen != self.eval_index - 1:
                if s is None:
                    reason = "no history yet"
                elif s.signature != signature:
                    reason = "conds or pass changed"
                else:
                    reason = "pass was absent last step"
                s = CacheStream(signature, reason)
                self.streams[(p, i)] = s
            s.last_seen = self.eval_index
            streams.append(s)
        return streams

    def predict_pass(self, streams, x_in):
        for s in streams:
            if s is not None and len(s.anchors) < self.min_anchors:
                self.forced_reason = self.forced_reason or s.reason
                return None
        return self.predict(streams, x_in)

    def predict(self, streams, x_in):
        # Coarse structure (composition, shapes, colour areas) from the direction-aware prediction, fine detail
        # from plain linear extrapolation, whose slight overshoot reads as crispness rather than drift.
        faithful = self._direction(streams, x_in)
        if x_in.ndim < 4:
            return faithful
        out = []
        for s, f in zip(streams, faithful):
            if s is None:
                out.append(f)
            else:
                linear = self._linear(s)
                out.append(linear + _lowpass(f - linear))
        return out

    def _direction(self, streams, x_in):
        if len(streams) == 2 and None not in streams:
            # cond/uncond pass: the CFG guidance difference can oscillate step to step while the uncond base
            # trends smoothly, so each gets its own follow-through.
            c, u = streams
            guidance = [(tau, xc - xu) for (tau, xc), (_, xu) in zip(c.anchors, u.anchors)]
            base = _extrapolate(u.anchors, self.tau_now)
            return [base + _extrapolate(guidance, self.tau_now), base]
        return [torch.zeros_like(x_in) if s is None else _extrapolate(s.anchors, self.tau_now) for s in streams]

    def _adaptive_cap(self, x0_cur, x0_prev):
        # Allow further extrapolation when x0 is stable (barely moving), clamp tight when
        # it is still converging fast. Self-calibrates per region of the sigma schedule.
        stability = (x0_cur - x0_prev).norm().item() / (x0_cur.norm().item() + 1e-8)
        return max(0.5, min(2.0, 1.0 / (stability * 4.0 + 0.5)))

    def _linear(self, stream):
        (tau_prev, x0_prev), (tau_cur, x0_cur) = stream.anchors[-2:]
        t_raw = (self.tau_now - tau_cur) / (tau_cur - tau_prev)
        cap = self._adaptive_cap(x0_cur, x0_prev)
        t = max(0.0, min(t_raw, cap))
        if self.verbose:
            logging.info(f"{self.name} - x0 t={t_raw:.3f} -> {t:.3f} (cap {cap:.2f})")
        return x0_cur + t * (x0_cur - x0_prev)

    def record_pass(self, streams, out):
        for s, o in zip(streams, out):
            if s is not None:
                s.add(self.tau_now, o, self.max_anchors)

    def log_summary(self):
        if self.total_evals == 0:
            return
        computed = self.total_evals - self.predicted_evals
        msg = (f"{self.name} - {self.total_evals} model evals: {self.predicted_evals} predicted, {computed} computed "
               f"({self.total_evals / max(computed, 1):.2f}x fewer model calls)")
        if self.forced:
            msg += "; planned skips that ran for real: " + ", ".join(f"{n}x {r}" for r, n in self.forced.items())
        logging.info(msg)


def cache_calc_cond_batch_wrapper(executor, *args, **kwargs):
    _, conds, x_in, timestep, model_options = args[:5]
    cache: CacheHolder = model_options["transformer_options"]["wtlcache"]
    if cache.depth == 0:
        raise RuntimeError(f"{cache.name}: calc_cond_batch was called outside a sampler evaluation, so this "
                           f"sampling path cannot be cached. Remove Cache Accelerator from this workflow.")
    streams = cache.streams_for_pass(conds, x_in, timestep, model_options)
    if cache.skip_planned:
        predicted = cache.predict_pass(streams, x_in)
        if predicted is not None:
            return predicted
    out = executor(*args, **kwargs)
    cache.model_called = True
    if cache.recording:
        cache.record_pass(streams, out)
    return out


def cache_predict_noise_wrapper(executor, *args, **kwargs):
    cache: CacheHolder = executor.class_obj.model_options["transformer_options"]["wtlcache"]
    cache.begin_eval(float(args[1].flatten()[0]))
    try:
        out = executor(*args, **kwargs)
    except BaseException:
        cache.abort_eval()
        raise
    cache.end_eval()
    return out


def cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    orig_model_options = guider.model_options
    sigmas = args[3] if len(args) > 3 else kwargs["sigmas"]
    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        cache = guider.model_options["transformer_options"]["wtlcache"].clone().prepare(sigmas)
        guider.model_options["transformer_options"]["wtlcache"] = cache
        logging.info(f"{cache.name} enabled - interval: {cache.cache_interval}, "
                     f"window: [{cache.start_percent}, {cache.end_percent}] = {cache.window_steps()}")
        return executor(*args, **kwargs)
    finally:
        cache = guider.model_options["transformer_options"]["wtlcache"]
        cache.log_summary()
        cache.reset()
        guider.model_options = orig_model_options


class CacheAcceleratorC:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cache_interval": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Inside the window, run the model every N evaluations and predict the ones in "
                               "between. 1 = never skip."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Where skipping starts, as a fraction of the sampling steps (0.2 of 40 steps = "
                               "from step 9). The console shows which steps the window covers."
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Where skipping stops, as a fraction of the sampling steps (0.9 of 40 steps = "
                               "up to step 36). The steps after it always run fully."
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log every evaluation (computed / predicted) and the extrapolation values."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "WtlNodes/sampling"

    def patch(self, model, cache_interval, start_percent, end_percent, verbose):
        if "wtlcache" in model.model_options.get("transformer_options", {}):
            raise RuntimeError("Cache Accelerator is already applied to this model; use only one per model chain.")
        model = model.clone()
        model.model_options["transformer_options"]["wtlcache"] = CacheHolder(
            cache_interval, start_percent, end_percent, verbose)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "wtlcache", cache_sample_wrapper)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE, "wtlcache", cache_predict_noise_wrapper)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, "wtlcache", cache_calc_cond_batch_wrapper)
        return (model,)


NODE_CLASS_MAPPINGS = {"CacheAccelerator": CacheAcceleratorC}
NODE_DISPLAY_NAME_MAPPINGS = {"CacheAccelerator": "Cache Accelerator"}
