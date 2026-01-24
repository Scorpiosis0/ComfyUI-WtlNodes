import torch
import threading
import time
import numpy as np
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, dither_method: str, r_levels: int, g_levels: int, b_levels: int, dither_scale: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (dither_method, r_levels, g_levels, b_levels, dither_scale)

def _get_params(node_id: str, dither_method: str, r_levels: int, g_levels: int, b_levels: int, dither_scale: float) -> tuple[str, int, int, int, float]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (dither_method, r_levels, g_levels, b_levels, dither_scale))

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

class DitherC:
    # Cache for Bayer matrix
    _bayer_matrix_cache = {}
    # Cache for blue noise
    _blue_noise_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dither_method": (["none", "bayer", "arithmetic_add", "blue_noise"],),
                "r_levels": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                }),
                "g_levels": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                }),
                "b_levels": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                }),
                "dither_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 5.0,
                    "step": 0.25,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dither"
    CATEGORY = "WtlNodes/image"
    
    @staticmethod
    def generate_bayer_matrix(n):
        """Generate Bayer matrix of size n×n (n must be power of 2)."""
        if n == 1:
            return np.array([[0]], dtype=np.float32)
        
        # Check cache first
        if n in DitherC._bayer_matrix_cache:
            return DitherC._bayer_matrix_cache[n]
        
        smaller = DitherC.generate_bayer_matrix(n // 2)
        top_left = 4 * smaller
        top_right = 4 * smaller + 2
        bottom_left = 4 * smaller + 3
        bottom_right = 4 * smaller + 1
        
        result = np.vstack([
            np.hstack([top_left, top_right]),
            np.hstack([bottom_left, bottom_right])
        ])
        
        # Do NOT normalize - keep values as 1 to (n*n)
        result = result.astype(np.float32)
        
        DitherC._bayer_matrix_cache[n] = result
        return result
    
    @staticmethod
    def generate_blue_noise(seed=0):
        cache_key = seed
        if cache_key in DitherC._blue_noise_cache:
            return DitherC._blue_noise_cache[cache_key]
        
        np.random.seed(seed)
        # Generate blue noise using Poisson disc sampling approximation
        size = 256
        blue_noise = np.zeros((4, size, size), dtype=np.uint8)
        
        # Generate per-channel blue noise
        for ch in range(4):
            white = np.random.rand(size, size)
            # Simple blue noise approximation: iteratively filter and re-randomize
            bn = white.copy()
            for iteration in range(5):
                from scipy.ndimage import gaussian_filter
                smooth = gaussian_filter(bn, sigma=2.0)
                bn = white - 0.7 * smooth
                bn = (bn - bn.min()) / (bn.max() - bn.min() + 1e-8)
            blue_noise[ch] = (bn * 255).astype(np.uint8)
        
        DitherC._blue_noise_cache[cache_key] = blue_noise
        return blue_noise
    
    @staticmethod
    def posterize_no_dither(image, levels_per_channel):
        """Simple posterization without dithering (quantization only)."""
        # image: [B, H, W, C]
        # levels_per_channel: [r_levels, g_levels, b_levels]
        
        result = image.clone()
        
        for c in range(3):
            levels = levels_per_channel[c]
            if levels < 2:
                levels = 2
            
            # Quantize to n levels
            channel_data = result[:, :, :, c]
            quantized = torch.floor(channel_data * (levels - 1) + 0.5) / (levels - 1)
            quantized = torch.clamp(quantized, 0.0, 1.0)
            result[:, :, :, c] = quantized
        
        return result
    
    @staticmethod
    def bayer_dither(image, levels_per_channel, dither_scale=1.0):
        """Ordered dithering using Bayer matrix (vectorized)."""
        batch_size, height, width, channels = image.shape
        device = image.device
        
        # Use fixed 8x8 Bayer matrix
        bayer = DitherC.generate_bayer_matrix(8)
        bayer_t = torch.from_numpy(bayer).to(device).float()  # Shape: [8, 8]
        
        result = image.clone()
        
        for c in range(3):
            levels = max(2, levels_per_channel[c])
            
            # Create coordinate grids [height, width]
            y_coords = torch.arange(height, device=device, dtype=torch.int32).view(height, 1).expand(height, width)
            x_coords = torch.arange(width, device=device, dtype=torch.int32).view(1, width).expand(height, width)
            
            # Scale coordinates and modulo to get index into 8x8 Bayer matrix
            # Dividing by dither_scale makes the pattern bigger (less frequent tiling)
            bayer_y = (y_coords / dither_scale).int() % 8
            bayer_x = (x_coords / dither_scale).int() % 8
            
            # Index into Bayer matrix
            bayer_vals = bayer_t[bayer_y, bayer_x]
            
            # Scale threshold based on 8x8 matrix (values 1-64)
            max_val = 64
            threshold = (bayer_vals - (max_val + 1) / 2.0) / max_val / levels
            
            # Apply dithering to all batches at once
            channel_data = result[:, :, :, c]
            dithered = channel_data + threshold.unsqueeze(0)  # Add batch dimension
            quantized = torch.floor(dithered * (levels - 1) + 0.5) / (levels - 1)
            result[:, :, :, c] = torch.clamp(quantized, 0.0, 1.0)
        
        return result
    
    @staticmethod
    def arithmetic_add_dither(image, levels_per_channel, dither_scale=1.0):
        """Arithmetic addition dithering using vectorized position-based pattern."""
        batch_size, height, width, channels = image.shape
        device = image.device
        
        result = image.clone()
        
        for c in range(3):
            levels = max(2, levels_per_channel[c])
            
            # Create coordinate grids [height, width], scaled by dither_scale
            y_coords = (torch.arange(height, device=device, dtype=torch.int32).view(height, 1) / dither_scale).int().expand(height, width)
            x_coords = (torch.arange(width, device=device, dtype=torch.int32).view(1, width) / dither_scale).int().expand(height, width)
            
            # Vectorized GIMP formula: (((x + c*67) + y*236) * 119) & 255
            mask = (((x_coords + c * 67) + y_coords * 236) * 119) & 255
            mask = mask.float()
            
            # Convert to threshold: (mask - 128) / 256.0 / levels
            threshold = (mask - 128.0) / 256.0 / levels
            
            # Apply dithering to all batches at once
            channel_data = result[:, :, :, c]
            dithered = channel_data + threshold.unsqueeze(0)  # Add batch dimension
            quantized = torch.floor(dithered * (levels - 1) + 0.5) / (levels - 1)
            result[:, :, :, c] = torch.clamp(quantized, 0.0, 1.0)
        
        return result
    
    @staticmethod
    def blue_noise_dither(image, levels_per_channel, dither_scale=1.0):
        """Dithering using blue noise texture (vectorized)."""
        batch_size, height, width, channels = image.shape
        device = image.device
        
        result = image.clone()
        
        # Get blue noise texture (4 channels, 256x256 each, u8 values 0-255)
        blue_noise = DitherC.generate_blue_noise(seed=0)
        blue_noise_t = torch.from_numpy(blue_noise).to(device).float()  # Convert to tensor first
        
        for c in range(3):
            levels = max(2, levels_per_channel[c])
            
            # Create coordinate grids for indexing into blue noise (256x256 tiling)
            # Scale coordinates to control pattern size
            y_coords = (torch.arange(height, device=device, dtype=torch.int32).view(height, 1) / dither_scale).int() % 256
            x_coords = (torch.arange(width, device=device, dtype=torch.int32).view(1, width) / dither_scale).int() % 256
            
            # Index into blue noise texture (now a torch tensor)
            noise_vals = blue_noise_t[c, y_coords, x_coords]
            
            # GIMP formula: (noise - 128) / 257.0 / levels
            threshold = (noise_vals - 128.0) / 257.0 / levels
            
            # Apply dithering to all batches at once
            channel_data = result[:, :, :, c]
            dithered = channel_data + threshold.unsqueeze(0)  # Add batch dimension
            quantized = torch.floor(dithered * (levels - 1) + 0.5) / (levels - 1)
            result[:, :, :, c] = torch.clamp(quantized, 0.0, 1.0)
        
        return result
    
    @staticmethod
    def apply_dither(image, dither_method, r_levels, g_levels, b_levels, dither_scale=1.0):
        """Apply the selected dithering method."""
        levels = [r_levels, g_levels, b_levels]
        
        if dither_method == "none":
            return DitherC.posterize_no_dither(image, levels)
        elif dither_method == "bayer":
            return DitherC.bayer_dither(image, levels, dither_scale)
        elif dither_method == "arithmetic_add":
            return DitherC.arithmetic_add_dither(image, levels, dither_scale)
        elif dither_method == "blue_noise":
            return DitherC.blue_noise_dither(image, levels, dither_scale)
        else:
            return image
    
    def dither(self, image, dither_method, r_levels, g_levels, b_levels, dither_scale, apply_type, unique_id=None):
        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)
        
        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}
        
        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)
            
            if apply_type == "apply_all":
                # Process all images at once
                while True:
                    cur_method, cur_r, cur_g, cur_b, cur_scale = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                    cur_image = self.apply_dither(image, cur_method, cur_r, cur_g, cur_b, cur_scale)
                    _send_ram_preview(cur_image, uid)
                    
                    # Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        dither_method, r_levels, g_levels, b_levels, dither_scale = cur_method, cur_r, cur_g, cur_b, cur_scale
                        break
                    
                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (image,)}
                    
                    time.sleep(0.25)
                
                # Apply final effect
                result = self.apply_dither(image, dither_method, r_levels, g_levels, b_levels, dither_scale)
            
            else:
                # Process images one by one
                batch_size = image.shape[0]
                result_list = []
                
                for i in range(batch_size):
                    single_image = image[i:i+1]
                    
                    while True:
                        cur_method, cur_r, cur_g, cur_b, cur_scale = _get_params(uid, dither_method, r_levels, g_levels, b_levels, dither_scale)
                        cur_image = self.apply_dither(single_image, cur_method, cur_r, cur_g, cur_b, cur_scale)
                        _send_ram_preview(cur_image, uid)
                        
                        # Check for button presses
                        if _check_and_clear_flag(uid, "apply"):
                            final_params = (cur_method, cur_r, cur_g, cur_b, cur_scale)
                            break
                        
                        if _check_and_clear_flag(uid, "skip"):
                            result_list.append(single_image)
                            final_params = None
                            break
                        
                        time.sleep(0.25)
                    
                    # Apply final effect for this image if not skipped
                    if final_params is not None:
                        processed = self.apply_dither(single_image, *final_params)
                        result_list.append(processed)
                
                # Concatenate all processed images
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode
            result = self.apply_dither(image, dither_method, r_levels, g_levels, b_levels, dither_scale)
        
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"Dither": DitherC}
NODE_DISPLAY_NAME_MAPPINGS = {"Dither": "Dither"}

# Note: Make sure your __init__.py has "image.dithering" in SUBMODULES
# and the NODE_HANDLERS dict includes the "dit" entry as shown in your example