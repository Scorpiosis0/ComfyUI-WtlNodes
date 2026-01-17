import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import threading
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, focus: float, rng: float, edge: int, hard_focus: float, blur: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (focus, rng, edge, hard_focus, blur)

def _get_params(node_id: str, default_focus: float, default_range: float, default_edge: int, default_hard_focus: float, default_blur: float) -> tuple[float, float, int, float, float]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (default_focus, default_range, default_edge, default_hard_focus, default_blur))

def _set_flag(node_id: str, flag: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        flags = entry.setdefault("flags", {})
        flags[flag] = True

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


def create_bokeh_kernel(size, shape='circle'):
    """Create a shaped bokeh kernel for lens blur effect"""
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


def apply_masked_blur_optimized(img_tensor, blur_mask_tensor, kernel_size, bokeh_shape, 
                               highlight_threshold_low, highlight_threshold_high, highlight_factor,
                               in_focus_mask_fix, blur_fixed_edge):
    """Optimized masked blur using PyTorch for GPU acceleration."""
    if kernel_size <= 1:
        return img_tensor, torch.zeros_like(blur_mask_tensor), torch.ones_like(blur_mask_tensor), torch.zeros_like(blur_mask_tensor)
    
    device = img_tensor.device
    
    # Step 1: Create in-focus mask
    in_focus_threshold = 0.01
    in_focus_mask = blur_mask_tensor < in_focus_threshold
    out_of_focus_mask = ~in_focus_mask
    
    # Apply in-focus mask fix (expansion) and track the border zone
    border_mask = torch.zeros_like(in_focus_mask)
    if in_focus_mask_fix > 0:
        in_focus_mask_np = in_focus_mask.cpu().numpy().astype(np.uint8)
        kernel_size_fix = in_focus_mask_fix * 2 + 1
        kernel_fix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_fix, kernel_size_fix))
        in_focus_mask_dilated_np = cv2.dilate(in_focus_mask_np, kernel_fix, iterations=1)
        in_focus_mask_dilated = torch.from_numpy(in_focus_mask_dilated_np).to(device).bool()
        
        # Border mask = dilated area minus original in-focus area
        border_mask = in_focus_mask_dilated & ~in_focus_mask
        
        # Update masks: expanded in-focus area should not be in out-of-focus
        out_of_focus_mask = out_of_focus_mask & ~in_focus_mask_dilated
        in_focus_mask = in_focus_mask_dilated
    
    # Calculate highlight weights on GPU
    luminance = 0.299 * img_tensor[:, :, 0] + 0.587 * img_tensor[:, :, 1] + 0.114 * img_tensor[:, :, 2]
    
    if highlight_factor > 0:
        v = (luminance - highlight_threshold_low) / (highlight_threshold_high - highlight_threshold_low + 1e-8)
        v = torch.clamp(v, 0, 1)
        scaled_factor = 10.0 * highlight_factor * np.log(2)
        weights = torch.exp(v * scaled_factor)
    else:
        weights = torch.ones_like(luminance)
    
    weights_3ch = weights.unsqueeze(-1)
    weighted_img = img_tensor * weights_3ch
    
    # Create bokeh kernel on GPU
    kernel_np = create_bokeh_kernel(kernel_size, bokeh_shape)
    if kernel_np is None:
        return img_tensor, in_focus_mask.float(), out_of_focus_mask.float(), border_mask.float()
    
    kernel_torch = torch.from_numpy(kernel_np).to(device).float()
    
    # Step 2: Create masked tensors
    masked_img = weighted_img.clone()
    masked_img[in_focus_mask] = 0.0
    
    masked_weights = weights.clone()
    masked_weights[in_focus_mask] = 0.0
    
    validity_mask = out_of_focus_mask.float()
    
    # Step 3: Apply convolution
    masked_img_4d = masked_img.permute(2, 0, 1).unsqueeze(0)
    masked_weights_4d = masked_weights.unsqueeze(0).unsqueeze(0)
    
    kernel_4d = kernel_torch.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    
    blurred_weighted_4d = F.conv2d(masked_img_4d, kernel_4d.repeat(3, 1, 1, 1), padding=pad, groups=3)
    blurred_weights_4d = F.conv2d(masked_weights_4d, kernel_4d, padding=pad)
    
    blurred_weighted = blurred_weighted_4d.squeeze(0).permute(1, 2, 0)
    blurred_weights = blurred_weights_4d.squeeze(0).squeeze(0)
    
    blurred_weights_3ch = blurred_weights.unsqueeze(-1)
    blurred = blurred_weighted / (blurred_weights_3ch + 1e-8)
    blurred = torch.clamp(blurred, 0, 1)
    
    # Step 4: Composite sharp areas back
    result = blurred.clone()
    result[in_focus_mask & ~border_mask] = img_tensor[in_focus_mask & ~border_mask]
    
    # Step 5: Apply subtle Gaussian blur to border zone if enabled
    if blur_fixed_edge and border_mask.any():
        border_img = img_tensor.clone()
        border_img[~border_mask] = 0.0
        
        gaussian_kernel_size = 5
        sigma = 1.0
        gaussian_kernel = torch.zeros((gaussian_kernel_size, gaussian_kernel_size), device=device)
        center = gaussian_kernel_size // 2
        
        for i in range(gaussian_kernel_size):
            for j in range(gaussian_kernel_size):
                x, y = i - center, j - center
                gaussian_kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel_4d = gaussian_kernel.unsqueeze(0).unsqueeze(0)
        
        border_img_4d = border_img.permute(2, 0, 1).unsqueeze(0)
        border_mask_4d = border_mask.float().unsqueeze(0).unsqueeze(0)
        
        pad_g = gaussian_kernel_size // 2
        blurred_border_4d = F.conv2d(border_img_4d, gaussian_kernel_4d.repeat(3, 1, 1, 1), padding=pad_g, groups=3)
        border_validity_4d = F.conv2d(border_mask_4d, gaussian_kernel_4d, padding=pad_g)
        
        blurred_border = blurred_border_4d.squeeze(0).permute(1, 2, 0)
        border_validity = border_validity_4d.squeeze(0).squeeze(0).unsqueeze(-1)
        
        blurred_border = blurred_border / (border_validity + 1e-8)
        blurred_border = torch.clamp(blurred_border, 0, 1)
        
        result[border_mask] = blurred_border[border_mask]
    else:
        result[border_mask] = img_tensor[border_mask]
    
    print(f"[Camera DOF] Masked blur: in-focus={in_focus_mask.sum().item()}, out-of-focus={out_of_focus_mask.sum().item()}, border={border_mask.sum().item()}")
    
    return result, in_focus_mask.float(), out_of_focus_mask.float(), border_mask.float()


def apply_depth_aware_masked_blur_optimized(img_tensor, blur_mask_tensor, max_blur_strength, bokeh_shape, 
                                           highlight_threshold_low, highlight_threshold_high, highlight_factor,
                                           in_focus_mask_fix, blur_fixed_edge):
    """Optimized variable blur with masking using PyTorch"""
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
        scaled_factor = 10.0 * highlight_factor * np.log(2)
        weights = torch.exp(v * scaled_factor)
    else:
        weights = torch.ones_like(luminance)
    
    weights_3ch = weights.unsqueeze(-1)
    weighted_img = img_tensor * weights_3ch
    
    num_levels = 8
    max_kernel = int(max_blur_strength * 2) * 2 + 1
    max_kernel = max(1, max_kernel)
    
    kernel_sizes = np.linspace(1, max_kernel, num_levels).astype(int)
    kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
    
    print(f"[Camera DOF] Depth-aware masked blur (GPU): {num_levels} levels, kernels: {kernel_sizes}")
    
    masked_img = weighted_img.clone()
    masked_img[in_focus_mask] = 0.0
    
    masked_weights = weights.clone()
    masked_weights[in_focus_mask] = 0.0
    
    validity_mask = out_of_focus_mask.float()
    
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
        
        blurred_weighted = blurred_weighted_4d.squeeze(0).permute(1, 2, 0)
        blurred_weights = blurred_weights_4d.squeeze(0).squeeze(0)
        
        blur_levels.append(blurred_weighted)
        blur_weights_levels.append(blurred_weights)
    
    blur_mask_scaled = blur_mask_tensor * (num_levels - 1)
    level_indices = torch.floor(blur_mask_scaled).long()
    level_indices = torch.clamp(level_indices, 0, num_levels - 2)
    blend_factor = blur_mask_scaled - level_indices.float()
    blend_factor_3ch = blend_factor.unsqueeze(-1)
    
    result_weighted = torch.zeros_like(weighted_img)
    result_weights = torch.zeros_like(weights)
    
    for i in range(num_levels - 1):
        at_level = (level_indices == i).float()
        at_level_3ch = at_level.unsqueeze(-1)
        
        lower_weighted = blur_levels[i]
        upper_weighted = blur_levels[i + 1]
        lower_weights = blur_weights_levels[i]
        upper_weights = blur_weights_levels[i + 1]
        
        blended_weighted = lower_weighted * (1 - blend_factor_3ch) + upper_weighted * blend_factor_3ch
        blended_weights = lower_weights * (1 - blend_factor) + upper_weights * blend_factor
        
        result_weighted += blended_weighted * at_level_3ch
        result_weights += blended_weights * at_level
    
    at_max = (level_indices == num_levels - 2).float()
    at_max_3ch = at_max.unsqueeze(-1)
    result_weighted += blur_levels[-1] * at_max_3ch
    result_weights += blur_weights_levels[-1] * at_max
    
    result_weights_3ch = result_weights.unsqueeze(-1)
    blurred = result_weighted / (result_weights_3ch + 1e-8)
    blurred = torch.clamp(blurred, 0, 1)
    
    result = blurred.clone()
    result[in_focus_mask & ~border_mask] = img_tensor[in_focus_mask & ~border_mask]
    
    if blur_fixed_edge and border_mask.any():
        border_img = img_tensor.clone()
        border_img[~border_mask] = 0.0
        
        gaussian_kernel_size = 5
        sigma = 1.0
        gaussian_kernel = torch.zeros((gaussian_kernel_size, gaussian_kernel_size), device=device)
        center = gaussian_kernel_size // 2
        
        for i in range(gaussian_kernel_size):
            for j in range(gaussian_kernel_size):
                x, y = i - center, j - center
                gaussian_kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel_4d = gaussian_kernel.unsqueeze(0).unsqueeze(0)
        
        border_img_4d = border_img.permute(2, 0, 1).unsqueeze(0)
        border_mask_4d = border_mask.float().unsqueeze(0).unsqueeze(0)
        
        pad_g = gaussian_kernel_size // 2
        blurred_border_4d = F.conv2d(border_img_4d, gaussian_kernel_4d.repeat(3, 1, 1, 1), padding=pad_g, groups=3)
        border_validity_4d = F.conv2d(border_mask_4d, gaussian_kernel_4d, padding=pad_g)
        
        blurred_border = blurred_border_4d.squeeze(0).permute(1, 2, 0)
        border_validity = border_validity_4d.squeeze(0).squeeze(0).unsqueeze(-1)
        
        blurred_border = blurred_border / (border_validity + 1e-8)
        blurred_border = torch.clamp(blurred_border, 0, 1)
        
        result[border_mask] = blurred_border[border_mask]
    else:
        result[border_mask] = img_tensor[border_mask]
    
    return result, in_focus_mask.float(), out_of_focus_mask.float(), border_mask.float()


def _apply_dof_to_image(img, depth, focal_point, focus_falloff, focal_plane, edge_fix, 
                       blur_strength, bokeh_shape, highlight_threshold_low, highlight_threshold_high, 
                       highlight_factor, depth_aware_blur, in_focus_mask_fix, blur_fixed_edge):
    """Apply DOF effect with simple masking approach - returns result and all masks"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.from_numpy(img).float().to(device)
    depth_tensor = torch.from_numpy(depth).float().to(device)
    
    if depth_tensor.shape[-1] > 1:
        depth_tensor = depth_tensor.mean(dim=-1, keepdim=True)

    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)

    hard_zone_min = focal_point - focal_plane
    hard_zone_max = focal_point + focal_plane
    
    blur_mask_tensor = torch.zeros_like(depth_tensor)
    
    below_mask = depth_tensor < hard_zone_min
    blur_mask_tensor[below_mask] = (hard_zone_min - depth_tensor[below_mask]) / (focus_falloff + 1e-8)
    
    above_mask = depth_tensor > hard_zone_max
    blur_mask_tensor[above_mask] = (depth_tensor[above_mask] - hard_zone_max) / (focus_falloff + 1e-8)
    
    blur_mask_tensor = torch.clamp(blur_mask_tensor, 0, 1).squeeze()
    
    if edge_fix > 0:
        blur_mask_np = blur_mask_tensor.cpu().numpy()
        kernel_size = abs(edge_fix) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blur_mask_np = cv2.dilate(blur_mask_np, kernel, iterations=1)
        blur_mask_np = cv2.erode(blur_mask_np, kernel, iterations=1)
        blur_mask_tensor = torch.from_numpy(blur_mask_np).to(device)
    
    if depth_aware_blur:
        blurred, in_focus_mask, out_of_focus_mask, border_mask = apply_depth_aware_masked_blur_optimized(
            img_tensor, blur_mask_tensor, blur_strength, bokeh_shape,
            highlight_threshold_low, highlight_threshold_high, highlight_factor,
            in_focus_mask_fix, blur_fixed_edge
        )
    else:
        kernel_size = int(blur_strength * 2) * 2 + 1
        kernel_size = max(1, kernel_size)
        blurred, in_focus_mask, out_of_focus_mask, border_mask = apply_masked_blur_optimized(
            img_tensor, blur_mask_tensor, kernel_size, bokeh_shape,
            highlight_threshold_low, highlight_threshold_high, highlight_factor,
            in_focus_mask_fix, blur_fixed_edge
        )
    
    blur_mask_3ch = blur_mask_tensor.unsqueeze(-1)
    result = img_tensor * (1 - blur_mask_3ch) + blurred * blur_mask_3ch
    
    result_np = result.cpu().numpy()
    blur_mask_np = blur_mask_tensor.cpu().numpy()
    in_focus_mask_np = in_focus_mask.cpu().numpy()
    out_of_focus_mask_np = out_of_focus_mask.cpu().numpy()
    border_mask_np = border_mask.cpu().numpy()
    
    return result_np, blur_mask_np, in_focus_mask_np, out_of_focus_mask_np, border_mask_np


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
                "edge_fix": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "in_focus_mask_fix": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "bokeh_shape": (["circle", "hexagon", "octagon"], {"default": "circle"}),
                "highlight_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "highlight_threshold_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "highlight_threshold_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "depth_aware_blur": ("BOOLEAN", {"default": False}),
                "blur_fixed_edge": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "blur_mask", "in_focus_mask", "out_of_focus_mask", "border_mask")
    FUNCTION = "apply_dof"
    CATEGORY = "WtlNodes/image"

    def apply_dof(self, image, depth_map, focal_point, blur_strength, focus_falloff, focal_plane, 
                  edge_fix, in_focus_mask_fix, bokeh_shape, highlight_factor, highlight_threshold_low, highlight_threshold_high, 
                  depth_aware_blur, blur_fixed_edge, unique_id=None, prompt=None, extra_pnginfo=None):

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            batch_size = image.shape[0]
            empty_blur_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]))
            full_in_focus = torch.ones((batch_size, image.shape[1], image.shape[2]))
            empty_out_focus = torch.zeros((batch_size, image.shape[1], image.shape[2]))
            empty_border = torch.zeros((batch_size, image.shape[1], image.shape[2]))
            return (image, empty_blur_mask, full_in_focus, empty_out_focus, empty_border)

        img_np = image.cpu().numpy()
        depth_np = depth_map.cpu().numpy()
        batch_size = img_np.shape[0]

        if unique_id:
            uid = str(unique_id)
            results = []
            blur_masks = []
            in_focus_masks = []
            out_of_focus_masks = []
            border_masks = []

            for b in range(batch_size):
                img = img_np[b]
                depth = depth_np[b]

                while True:
                    cur_focus, cur_range, cur_edge, cur_hard_range, cur_blur = _get_params(
                        uid, focal_point, focus_falloff, edge_fix, focal_plane, blur_strength
                    )

                    preview_result, preview_blur_mask, preview_in_focus, preview_out_focus, preview_border = _apply_dof_to_image(
                        img, depth, cur_focus, cur_range, cur_hard_range, 
                        cur_edge, cur_blur, bokeh_shape, highlight_threshold_low, 
                        highlight_threshold_high, highlight_factor, depth_aware_blur, in_focus_mask_fix, blur_fixed_edge
                    )
                    
                    mask_rgb = np.stack([preview_blur_mask, preview_blur_mask, preview_blur_mask], axis=-1)
                    mask_tensor = torch.from_numpy(mask_rgb).unsqueeze(0).float()
                    _send_ram_preview(mask_tensor, uid)

                    if _check_and_clear_flag(uid, "apply"):
                        final_params = (cur_focus, cur_range, cur_edge, cur_hard_range, cur_blur)
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        results.append(img)
                        blur_masks.append(np.zeros((img.shape[0], img.shape[1])))
                        in_focus_masks.append(np.ones((img.shape[0], img.shape[1])))
                        out_of_focus_masks.append(np.zeros((img.shape[0], img.shape[1])))
                        border_masks.append(np.zeros((img.shape[0], img.shape[1])))
                        final_params = None
                        break
                    
                    time.sleep(0.25)

                if final_params is not None:
                    result, blur_mask, in_focus, out_focus, border = _apply_dof_to_image(
                        img, depth, final_params[0], final_params[1], 
                        final_params[3], final_params[2], final_params[4],
                        bokeh_shape, highlight_threshold_low, highlight_threshold_high, 
                        highlight_factor, depth_aware_blur, in_focus_mask_fix, blur_fixed_edge
                    )
                    results.append(result)
                    blur_masks.append(blur_mask)
                    in_focus_masks.append(in_focus)
                    out_of_focus_masks.append(out_focus)
                    border_masks.append(border)

            output_img = torch.from_numpy(np.stack(results)).float()
            output_blur_mask = torch.from_numpy(np.stack(blur_masks)).float()
            output_in_focus = torch.from_numpy(np.stack(in_focus_masks)).float()
            output_out_focus = torch.from_numpy(np.stack(out_of_focus_masks)).float()
            output_border = torch.from_numpy(np.stack(border_masks)).float()

        else:
            results = []
            blur_masks = []
            in_focus_masks = []
            out_of_focus_masks = []
            border_masks = []

            for b in range(batch_size):
                result, blur_mask, in_focus, out_focus, border = _apply_dof_to_image(
                    img_np[b], depth_np[b], focal_point, focus_falloff, 
                    focal_plane, edge_fix, blur_strength,
                    bokeh_shape, highlight_threshold_low, highlight_threshold_high, 
                    highlight_factor, depth_aware_blur, in_focus_mask_fix, blur_fixed_edge
                )
                results.append(result)
                blur_masks.append(blur_mask)
                in_focus_masks.append(in_focus)
                out_of_focus_masks.append(out_focus)
                border_masks.append(border)

            output_img = torch.from_numpy(np.stack(results)).float()
            output_blur_mask = torch.from_numpy(np.stack(blur_masks)).float()
            output_in_focus = torch.from_numpy(np.stack(in_focus_masks)).float()
            output_out_focus = torch.from_numpy(np.stack(out_of_focus_masks)).float()
            output_border = torch.from_numpy(np.stack(border_masks)).float()

        blur_mode = "depth-aware" if depth_aware_blur else "flat"
        print(f"[Camera DOF] Simple masked blur applied ({blur_mode} mode), mask fix: {in_focus_mask_fix}px, border blur: {blur_fixed_edge}")
        return (output_img, output_blur_mask, output_in_focus, output_out_focus, output_border)

NODE_CLASS_MAPPINGS = {"CameraDepthDOF": CameraDepthOfFieldC}
NODE_DISPLAY_NAME_MAPPINGS = {"CameraDepthDOF": "Camera Depth of Field (WIP)"}