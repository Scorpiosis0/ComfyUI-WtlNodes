import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import threading
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()
_LAST_COMPUTE: dict[str, tuple] = {}  # uid -> (compute_params, result, blur_mask, in_focus, out_focus, border)

def _set_params(node_id: str, focal_point: float, focus_falloff: float,
                focal_plane: float, blur_strength: float, in_focus_mask_fix: int,
                bokeh_shape: str, highlight_factor: float, highlight_threshold_low: float,
                highlight_threshold_high: float, preview_mode: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (focal_point, focus_falloff, focal_plane, blur_strength,
                      in_focus_mask_fix, bokeh_shape, highlight_factor,
                      highlight_threshold_low, highlight_threshold_high, preview_mode)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, focal_point: float, focus_falloff: float,
                focal_plane: float, blur_strength: float, in_focus_mask_fix: int,
                bokeh_shape: str, highlight_factor: float, highlight_threshold_low: float,
                highlight_threshold_high: float, preview_mode: str) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (focal_point, focus_falloff, focal_plane, blur_strength,
                                    in_focus_mask_fix, bokeh_shape, highlight_factor,
                                    highlight_threshold_low, highlight_threshold_high,
                                    preview_mode))

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
    _LAST_COMPUTE.pop(node_id, None)


def _make_preview_tensor(result_np, blur_mask_np, in_focus_mask_np, preview_mode):
    if preview_mode == "image":
        preview = result_np
    elif preview_mode == "in_focus_mask":
        preview = np.stack([in_focus_mask_np] * 3, axis=-1)
    else:  # "blur_mask"
        preview = np.stack([blur_mask_np] * 3, axis=-1)
    return torch.from_numpy(preview).unsqueeze(0).float()


def create_bokeh_kernel(size, shape='circle'):
    if size <= 1:
        return None
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    if shape == 'circle':
        y, x = np.ogrid[-center:size-center, -center:size-center]
        mask = x**2 + y**2 <= center**2
        kernel[mask] = 1.0
    elif shape == 'hexagon':
        y, x = np.ogrid[-center:size-center, -center:size-center]
        angle = np.arctan2(y, x)
        distance = np.sqrt(x**2 + y**2)
        hex_radius = center / np.cos(angle % (np.pi/3) - np.pi/6)
        mask = distance <= hex_radius * 0.95
        kernel[mask] = 1.0
    elif shape == 'octagon':
        y, x = np.ogrid[-center:size-center, -center:size-center]
        angle = np.arctan2(y, x)
        distance = np.sqrt(x**2 + y**2)
        oct_radius = center / np.cos(angle % (np.pi/4) - np.pi/8)
        mask = distance <= oct_radius * 0.95
        kernel[mask] = 1.0
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    return kernel


def apply_depth_aware_blur(img_tensor, blur_mask_tensor, max_blur_strength, bokeh_shape,
                           highlight_threshold_low, highlight_threshold_high, highlight_factor,
                           in_focus_mask_fix):
    """
    8-level depth-graduated bokeh blur.

    Each pixel is assigned a kernel size proportional to its blur_mask value, interpolated
    between the two nearest precomputed levels. In-focus pixels (blur_mask < 0.01) have their
    original values restored after blurring. The masked convolution zeroes out in-focus pixels
    before each level's convolution to prevent sharp edges from leaking into background bokeh.
    """
    device = img_tensor.device
    in_focus_threshold = 0.01
    in_focus_mask = blur_mask_tensor < in_focus_threshold
    out_of_focus_mask = ~in_focus_mask
    border_mask = torch.zeros_like(in_focus_mask)

    if in_focus_mask_fix > 0:
        in_focus_mask_np = in_focus_mask.cpu().numpy().astype(np.uint8)
        kernel_size_fix = in_focus_mask_fix * 2 + 1
        kernel_fix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_fix, kernel_size_fix))
        in_focus_mask_dilated_np = cv2.dilate(in_focus_mask_np, kernel_fix, iterations=1)
        in_focus_mask_dilated = torch.from_numpy(in_focus_mask_dilated_np).to(device).bool()
        border_mask = in_focus_mask_dilated & ~in_focus_mask
        out_of_focus_mask = out_of_focus_mask & ~in_focus_mask_dilated
        in_focus_mask = in_focus_mask_dilated

    luminance = 0.299 * img_tensor[:, :, 0] + 0.587 * img_tensor[:, :, 1] + 0.114 * img_tensor[:, :, 2]
    if highlight_factor > 0:
        v = (luminance - highlight_threshold_low) / (highlight_threshold_high - highlight_threshold_low + 1e-8)
        v = torch.clamp(v, 0, 1)
        weights = torch.exp(v * 10.0 * highlight_factor * np.log(2))
    else:
        weights = torch.ones_like(luminance)

    weighted_img = img_tensor * weights.unsqueeze(-1)

    num_levels = 8
    max_kernel = int(max_blur_strength * 2) * 2 + 1
    max_kernel = max(1, max_kernel)
    kernel_sizes = np.linspace(1, max_kernel, num_levels).astype(int)
    kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]

    masked_img = weighted_img.clone()
    masked_img[in_focus_mask] = 0.0
    masked_weights = weights.clone()
    masked_weights[in_focus_mask] = 0.0

    blur_levels = []
    blur_weights_levels = []
    for kernel_size in kernel_sizes:
        if kernel_size <= 1:
            blur_levels.append(weighted_img.clone())
            blur_weights_levels.append(weights.clone())
            continue
        kernel_np = create_bokeh_kernel(kernel_size, bokeh_shape)
        if kernel_np is None:
            blur_levels.append(weighted_img.clone())
            blur_weights_levels.append(weights.clone())
            continue
        kernel_torch = torch.from_numpy(kernel_np).to(device).float()
        masked_img_4d = masked_img.permute(2, 0, 1).unsqueeze(0)
        masked_weights_4d = masked_weights.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel_torch.unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2
        blurred_weighted_4d = F.conv2d(masked_img_4d, kernel_4d.repeat(3, 1, 1, 1), padding=pad, groups=3)
        blurred_weights_4d = F.conv2d(masked_weights_4d, kernel_4d, padding=pad)
        blur_levels.append(blurred_weighted_4d.squeeze(0).permute(1, 2, 0))
        blur_weights_levels.append(blurred_weights_4d.squeeze(0).squeeze(0))

    blur_mask_scaled = blur_mask_tensor * (num_levels - 1)
    level_indices = torch.clamp(torch.floor(blur_mask_scaled).long(), 0, num_levels - 2)
    blend_factor = blur_mask_scaled - level_indices.float()
    blend_factor_3ch = blend_factor.unsqueeze(-1)

    result_weighted = torch.zeros_like(weighted_img)
    result_weights = torch.zeros_like(weights)
    for i in range(num_levels - 1):
        at_level = (level_indices == i).float()
        blended_w = blur_levels[i] * (1 - blend_factor_3ch) + blur_levels[i + 1] * blend_factor_3ch
        blended_wt = blur_weights_levels[i] * (1 - blend_factor) + blur_weights_levels[i + 1] * blend_factor
        result_weighted += blended_w * at_level.unsqueeze(-1)
        result_weights += blended_wt * at_level

    blurred = torch.clamp(result_weighted / (result_weights.unsqueeze(-1) + 1e-8), 0, 1)

    result = blurred.clone()
    result[in_focus_mask & ~border_mask] = img_tensor[in_focus_mask & ~border_mask]
    result[border_mask] = img_tensor[border_mask]

    return result, in_focus_mask.float(), out_of_focus_mask.float(), border_mask.float()


def _apply_dof_to_image(img, depth, focal_point, focus_falloff, focal_plane,
                        blur_strength, bokeh_shape, highlight_threshold_low, highlight_threshold_high,
                        highlight_factor, in_focus_mask_fix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.from_numpy(img).float().to(device)
    depth_tensor = torch.from_numpy(depth).float().to(device)

    if depth_tensor.shape[-1] > 1:
        depth_tensor = depth_tensor.mean(dim=-1, keepdim=True)

    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)

    hard_zone_min = focal_point - focal_plane
    hard_zone_max = focal_point + focal_plane

    blur_mask = torch.zeros_like(depth_tensor)
    below = depth_tensor < hard_zone_min
    blur_mask[below] = (hard_zone_min - depth_tensor[below]) / (focus_falloff + 1e-8)
    above = depth_tensor > hard_zone_max
    blur_mask[above] = (depth_tensor[above] - hard_zone_max) / (focus_falloff + 1e-8)
    blur_mask = torch.clamp(blur_mask, 0, 1).squeeze()

    result, in_focus_mask, out_of_focus_mask, border_mask = apply_depth_aware_blur(
        img_tensor, blur_mask, blur_strength, bokeh_shape,
        highlight_threshold_low, highlight_threshold_high, highlight_factor,
        in_focus_mask_fix
    )

    return (result.cpu().numpy(), blur_mask.cpu().numpy(),
            in_focus_mask.cpu().numpy(), out_of_focus_mask.cpu().numpy(), border_mask.cpu().numpy())


class CameraDepthOfFieldC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "focal_point": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "blur_strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "round": 0.1}),
                "focal_plane": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "round": 0.001}),
                "focus_falloff": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "in_focus_mask_fix": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "bokeh_shape": (["circle", "hexagon", "octagon"], {"default": "circle"}),
                "highlight_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "highlight_threshold_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "highlight_threshold_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "preview_mode": (["blur_mask", "in_focus_mask", "image"], {"default": "blur_mask"}),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "blur_mask", "in_focus_mask", "out_of_focus_mask", "border_mask")
    FUNCTION = "apply_dof"
    CATEGORY = "WtlNodes/image"

    def apply_dof(self, image, depth_map, focal_point, blur_strength, focus_falloff, focal_plane,
                  in_focus_mask_fix, bokeh_shape, highlight_factor, highlight_threshold_low,
                  highlight_threshold_high, preview_mode, apply_type,
                  unique_id=None, prompt=None, extra_pnginfo=None):

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            batch_size = image.shape[0]
            empty = torch.zeros((batch_size, image.shape[1], image.shape[2]))
            full = torch.ones((batch_size, image.shape[1], image.shape[2]))
            return (image, empty, full, empty, empty)

        img_np = image.cpu().numpy()
        depth_np = depth_map.cpu().numpy()
        batch_size = img_np.shape[0]

        # compute_params excludes preview_mode — 9 elements
        def _run(img, depth, cp):
            fp, ff, fpl, bs, ifmf, bsh, hf, htl, hth = cp
            return _apply_dof_to_image(img, depth, fp, ff, fpl, bs, bsh, htl, hth, hf, ifmf)

        def _defaults(uid):
            return _get_params(uid, focal_point, focus_falloff, focal_plane, blur_strength,
                               in_focus_mask_fix, bokeh_shape, highlight_factor,
                               highlight_threshold_low, highlight_threshold_high,
                               preview_mode)

        if not unique_id or apply_type == "auto_apply":
            results, blur_masks, in_focus_masks, out_of_focus_masks, border_masks = [], [], [], [], []
            cp = (focal_point, focus_falloff, focal_plane, blur_strength,
                  in_focus_mask_fix, bokeh_shape, highlight_factor,
                  highlight_threshold_low, highlight_threshold_high)
            for b in range(batch_size):
                result, bm, ifm, ofm, borm = _run(img_np[b], depth_np[b], cp)
                results.append(result); blur_masks.append(bm)
                in_focus_masks.append(ifm); out_of_focus_masks.append(ofm); border_masks.append(borm)
            return (
                torch.from_numpy(np.stack(results)).float(),
                torch.from_numpy(np.stack(blur_masks)).float(),
                torch.from_numpy(np.stack(in_focus_masks)).float(),
                torch.from_numpy(np.stack(out_of_focus_masks)).float(),
                torch.from_numpy(np.stack(border_masks)).float(),
            )

        uid = str(unique_id)
        results, blur_masks, in_focus_masks, out_of_focus_masks, border_masks = [], [], [], [], []

        for b in range(batch_size):
            img = img_np[b]
            depth = depth_np[b]

            cur = _defaults(uid)
            cp = cur[:9]
            pm = cur[9]

            # Skip recompute if only preview_mode changed
            cached = _LAST_COMPUTE.get(uid)
            if cached and cached[0] == cp:
                result_np, blur_mask_np, ifm_np, ofm_np, bm_np = cached[1:]
            else:
                t0 = time.time()
                result_np, blur_mask_np, ifm_np, ofm_np, bm_np = _run(img, depth, cp)
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _LAST_COMPUTE[uid] = (cp, result_np, blur_mask_np, ifm_np, ofm_np, bm_np)

            _send_ram_preview(_make_preview_tensor(result_np, blur_mask_np, ifm_np, pm), uid)

            final_params = None
            while True:
                triggered = False
                while not triggered:
                    if _check_and_clear_params_changed(uid):
                        triggered = True; break
                    if _check_and_clear_flag(uid, "apply"):
                        final_params = _defaults(uid); break
                    if _check_and_clear_flag(uid, "skip"):
                        results.append(img)
                        blur_masks.append(np.zeros((img.shape[0], img.shape[1])))
                        in_focus_masks.append(np.ones((img.shape[0], img.shape[1])))
                        out_of_focus_masks.append(np.zeros((img.shape[0], img.shape[1])))
                        border_masks.append(np.zeros((img.shape[0], img.shape[1])))
                        final_params = None; break
                    time.sleep(0.05)
                if not triggered:
                    break

                cur = _defaults(uid)
                cp = cur[:9]
                pm = cur[9]

                cached = _LAST_COMPUTE.get(uid)
                if cached and cached[0] == cp:
                    result_np, blur_mask_np, ifm_np, ofm_np, bm_np = cached[1:]
                else:
                    t0 = time.time()
                    result_np, blur_mask_np, ifm_np, ofm_np, bm_np = _run(img, depth, cp)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _LAST_COMPUTE[uid] = (cp, result_np, blur_mask_np, ifm_np, ofm_np, bm_np)

                _send_ram_preview(_make_preview_tensor(result_np, blur_mask_np, ifm_np, pm), uid)

            if final_params is not None:
                cp_final = final_params[:9]
                cached = _LAST_COMPUTE.get(uid)
                if cached and cached[0] == cp_final:
                    result, bm, ifm, ofm, borm = cached[1:]
                else:
                    result, bm, ifm, ofm, borm = _run(img, depth, cp_final)
                results.append(result); blur_masks.append(bm)
                in_focus_masks.append(ifm); out_of_focus_masks.append(ofm); border_masks.append(borm)

        return (
            torch.from_numpy(np.stack(results)).float(),
            torch.from_numpy(np.stack(blur_masks)).float(),
            torch.from_numpy(np.stack(in_focus_masks)).float(),
            torch.from_numpy(np.stack(out_of_focus_masks)).float(),
            torch.from_numpy(np.stack(border_masks)).float(),
        )


NODE_CLASS_MAPPINGS = {"CameraDepthDOF": CameraDepthOfFieldC}
NODE_DISPLAY_NAME_MAPPINGS = {"CameraDepthDOF": "Camera Depth of Field"}
