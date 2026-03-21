import torch
import cv2
import numpy as np
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, filter_type: str, strength: float, edge_threshold: float, 
                neon_hue: float, neon_blur: float) -> None:
    """Write the newest slider values for *node_id* and set trigger flag."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (filter_type, strength, edge_threshold, neon_hue, neon_blur)
        old_params = entry.get("params")
        
        # Only set trigger if params actually changed
        if old_params != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False  # Mark as not complete

def _get_params(node_id: str, *defaults) -> tuple:
    """Return stored params if available, otherwise return the provided defaults."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", defaults)

def _check_and_clear_params_changed(node_id: str) -> bool:
    """Return True if params changed since last check, then clear the flag."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id)
        if not entry:
            return False
        if entry.get("params_changed"):
            entry["params_changed"] = False
            return True
        return False

def _set_processing_time(node_id: str, ms: int) -> None:
    """Store the processing time in milliseconds and mark as complete."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["processing_time_ms"] = ms
        entry["processing_complete"] = True

def _get_processing_time(node_id: str) -> tuple:
    """Get the last processing time and completion status."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return (entry.get("processing_time_ms", 0), entry.get("processing_complete", False))

def _set_current_image(node_id: str, image) -> None:
    """Store the current processed image for stacking effects."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["current_image"] = image

def _get_current_image(node_id: str):
    """Get the stored current image, or None if not available."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("current_image", None)

def _set_flag(node_id: str, flag: str) -> None:
    """Mark a button press – ``flag`` must be ``'apply'``, ``'skip'``, or ``'apply_again'``."""
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

def apply_bw_filter(image, strength):
    """Apply black and white filter."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Convert to grayscale
        if c == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            # Convert back to 3 channels
            bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif c == 4:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2GRAY)
            # Convert back to 4 channels (keep alpha)
            bw_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            bw = np.dstack([bw_rgb, img_uint8[:, :, 3]])
        else:
            results.append(img)
            continue
        
        # Blend with original based on strength
        blended = cv2.addWeighted(img_uint8, 1 - strength, bw, strength, 0)
        
        # Convert back to float
        result_float = blended.astype(np.float32) / 255.0
        results.append(result_float)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_sepia_filter(image, strength):
    """Apply sepia tone filter."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    # Sepia transformation matrix
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        if c == 3:
            # Apply sepia transformation
            sepia = np.dot(img, sepia_matrix.T)
            sepia = np.clip(sepia, 0, 1)
            
            # Blend with original based on strength
            blended = img * (1 - strength) + sepia * strength
            results.append(blended)
            
        elif c == 4:
            # Apply to RGB channels only
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4]
            
            sepia = np.dot(rgb, sepia_matrix.T)
            sepia = np.clip(sepia, 0, 1)
            
            # Blend with original based on strength
            blended_rgb = rgb * (1 - strength) + sepia * strength
            blended = np.concatenate([blended_rgb, alpha], axis=2)
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_duotone_filter(image, strength):
    """Apply duotone filter (shadow color: dark blue, highlight color: orange)."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    # Define shadow and highlight colors (can be customized)
    shadow_color = np.array([0.1, 0.15, 0.3])  # Dark blue
    highlight_color = np.array([1.0, 0.7, 0.3])  # Orange
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        if c == 3:
            # Convert to grayscale for luminance
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            gray = gray[:, :, np.newaxis]  # Add channel dimension
            
            # Map grayscale to duotone colors
            duotone = shadow_color * (1 - gray) + highlight_color * gray
            duotone = np.clip(duotone, 0, 1)
            
            # Blend with original
            blended = img * (1 - strength) + duotone * strength
            results.append(blended)
            
        elif c == 4:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4]
            
            gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            gray = gray[:, :, np.newaxis]
            
            duotone = shadow_color * (1 - gray) + highlight_color * gray
            duotone = np.clip(duotone, 0, 1)
            
            blended_rgb = rgb * (1 - strength) + duotone * strength
            blended = np.concatenate([blended_rgb, alpha], axis=2)
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_invert_filter(image, strength):
    """Apply color inversion filter."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        if c == 3:
            # Invert RGB channels
            inverted = 1.0 - img
            blended = img * (1 - strength) + inverted * strength
            results.append(blended)
            
        elif c == 4:
            # Invert RGB, keep alpha
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4]
            
            inverted_rgb = 1.0 - rgb
            blended_rgb = rgb * (1 - strength) + inverted_rgb * strength
            blended = np.concatenate([blended_rgb, alpha], axis=2)
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_cartoon_filter(image, strength, edge_threshold):
    """Apply cartoon/posterize effect with adjustable edge threshold."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if c >= 3:
            # Bilateral filter for edge-preserving smoothing
            smooth = cv2.bilateralFilter(img_uint8, 9, 75, 75)
            
            # Edge detection with adjustable threshold
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            # Use edge_threshold to control sensitivity (0-255 range)
            threshold_value = int(edge_threshold * 255)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 9, threshold_value // 25)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Combine smooth colors with edges
            cartoon = cv2.bitwise_and(smooth, edges)
            
            if c == 4:
                # Preserve alpha channel
                cartoon = np.dstack([cartoon, img_uint8[:, :, 3]])
            
            cartoon_float = cartoon.astype(np.float32) / 255.0
            blended = img * (1 - strength) + cartoon_float * strength
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_sketch_filter(image, strength):
    """Apply pencil sketch effect."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if c >= 3:
            # Convert to grayscale
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Invert grayscale
            inverted = 255 - gray
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
            
            # Invert blurred image
            inverted_blurred = 255 - blurred
            
            # Create sketch by dividing
            sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
            
            if c == 4:
                sketch = np.dstack([sketch, img_uint8[:, :, 3]])
            
            sketch_float = sketch.astype(np.float32) / 255.0
            blended = img * (1 - strength) + sketch_float * strength
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_neon_filter(image, strength, edge_threshold, hue_shift, blur_amount):
    """Apply neon/glow effect with smooth antialiased edges and bright bloom."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if c >= 3:
            # Convert to grayscale
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Edge detection with adjustable threshold
            threshold1 = int(edge_threshold * 100)
            threshold2 = int(edge_threshold * 200)
            edges = cv2.Canny(gray, threshold1, threshold2)
            
            # Multi-stage antialiasing for ultra-smooth lines
            # Stage 1: Light blur to smooth jagged edges
            sharp_edges = cv2.GaussianBlur(edges, (5, 5), 1.0)
            # Stage 2: Additional subtle blur for silky smoothness
            sharp_edges = cv2.GaussianBlur(sharp_edges, (3, 3), 0.7)
            
            # Create glow/bloom by blurring for the glow layer
            kernel_size = max(3, int(blur_amount * 40) | 1)
            glow = cv2.GaussianBlur(edges, (kernel_size, kernel_size), blur_amount * 20)
            
            # Colorize based on hue shift (0-360 degrees)
            hue_angle = hue_shift * 2 * np.pi
            
            # Create sharp neon lines (bright, saturated)
            neon_sharp = np.zeros_like(img_uint8, dtype=np.float32)
            neon_sharp[:, :, 0] = sharp_edges * (0.5 + 0.5 * np.cos(hue_angle))  # Red
            neon_sharp[:, :, 1] = sharp_edges * (0.5 + 0.5 * np.cos(hue_angle + 2.094))  # Green
            neon_sharp[:, :, 2] = sharp_edges * (0.5 + 0.5 * np.cos(hue_angle + 4.189))  # Blue
            
            # Create glow layer (brighter, softer bloom)
            neon_glow = np.zeros_like(img_uint8, dtype=np.float32)
            neon_glow[:, :, 0] = glow * (0.5 + 0.5 * np.cos(hue_angle))
            neon_glow[:, :, 1] = glow * (0.5 + 0.5 * np.cos(hue_angle + 2.094))
            neon_glow[:, :, 2] = glow * (0.5 + 0.5 * np.cos(hue_angle + 4.189))
            
            # Combine: very bright sharp core + bright glow
            # Sharp edges are 4x brighter, glow is 1.5x brighter
            neon = np.clip(neon_sharp * 4.0 + neon_glow * 1.5, 0, 255).astype(np.uint8)
            
            if c == 4:
                neon = np.dstack([neon, img_uint8[:, :, 3]])
            
            neon_float = neon.astype(np.float32) / 255.0
            
            # Additive blending for glow effect
            blended = np.clip(img * (1 - strength) + neon_float * strength, 0, 1)
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_high_contrast_filter(image, strength):
    """Apply high contrast filter - simple contrast boost around midpoint."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        if c >= 3:
            # Simple contrast formula: contrast = (value - 0.5) * factor + 0.5
            # This keeps 0.5 (middle gray) the same, makes darks darker and brights brighter
            contrast_factor = 2.5  # Adjust strength of contrast
            
            high_contrast = (img - 0.5) * contrast_factor + 0.5
            high_contrast = np.clip(high_contrast, 0, 1)
            
            if c == 4:
                # Keep alpha channel unchanged
                high_contrast = np.concatenate([high_contrast[:, :, :3], img[:, :, 3:4]], axis=2)
            
            blended = img * (1 - strength) + high_contrast * strength
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_watercolor_filter(image, strength):
    """Apply watercolor painting effect - smooth color regions with simplified palette."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if c >= 3:
            # Apply bilateral filter for smooth watercolor regions
            watercolor = cv2.bilateralFilter(img_uint8, 9, 100, 100)
            
            # Apply median blur to reduce artifacts
            watercolor = cv2.medianBlur(watercolor, 5)
            
            # Simplified color quantization using integer division (much faster than K-means)
            # Reduce to limited color palette
            color_levels = 32  # Number of levels per channel (32^3 = 32768 colors)
            watercolor = (watercolor // (256 // color_levels)) * (256 // color_levels)
            watercolor = watercolor.astype(np.uint8)
            
            # Final light smoothing to blend color regions
            watercolor = cv2.bilateralFilter(watercolor, 5, 50, 50)
            
            if c == 4:
                watercolor = np.dstack([watercolor, img_uint8[:, :, 3]])
            
            watercolor_float = watercolor.astype(np.float32) / 255.0
            blended = img * (1 - strength) + watercolor_float * strength
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_emboss_filter(image, strength):
    """Apply emboss effect - creates 3D relief appearance."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    # Emboss kernel
    kernel = np.array([[-2, -1, 0],
                      [-1,  1, 1],
                      [ 0,  1, 2]])
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if c >= 3:
            # Convert to grayscale first for consistent embossing
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Apply emboss filter
            embossed_gray = cv2.filter2D(gray, -1, kernel)
            
            # Add neutral gray (128) to center the values - this prevents brightening
            embossed_gray = cv2.add(embossed_gray, 128)
            embossed_gray = np.clip(embossed_gray, 0, 255)
            
            # Convert back to RGB
            embossed = cv2.cvtColor(embossed_gray, cv2.COLOR_GRAY2RGB)
            
            if c == 4:
                embossed = np.dstack([embossed[:, :, :3], img_uint8[:, :, 3]])
            
            embossed_float = embossed.astype(np.float32) / 255.0
            blended = img * (1 - strength) + embossed_float * strength
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_infrared_filter(image, strength):
    """Apply thermal/infrared camera effect with heat map colors (cold=blue, warm=red)."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        if c >= 3:
            # Convert to grayscale to get "heat" intensity
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_float = gray.astype(np.float32) / 255.0
            
            # Create thermal colormap
            # Cold (dark) = blue/purple, Medium = red/yellow, Hot (bright) = white/yellow
            thermal = np.zeros_like(img)
            
            # Blue channel: high in cold areas (dark), low in hot areas
            thermal[:, :, 2] = np.clip(1.0 - gray_float * 1.5, 0, 1)
            
            # Green channel: peaks in medium temperatures
            thermal[:, :, 1] = np.clip(np.sin(gray_float * np.pi) * 1.2, 0, 1)
            
            # Red channel: high in hot areas (bright)
            thermal[:, :, 0] = np.clip(gray_float * 1.3, 0, 1)
            
            # Enhance contrast for more dramatic thermal look
            thermal = np.clip((thermal - 0.3) * 1.6 + 0.3, 0, 1)
            
            if c == 4:
                thermal = np.concatenate([thermal[:, :, :3], img[:, :, 3:4]], axis=2)
            
            blended = img * (1 - strength) + thermal * strength
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_pastel_filter(image, strength):
    """Apply soft pastel effect - desaturated colors with soft blur."""
    img_np = image.cpu().numpy()
    batch_size = img_np.shape[0]
    results = []
    
    for b in range(batch_size):
        img = img_np[b]
        h, w, c = img.shape
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if c >= 3:
            # Bilateral filter for soft smoothing (the "smudging" you like)
            smooth = cv2.bilateralFilter(img_uint8, 9, 60, 60)
            
            # Convert to HSV and simply cut saturation in half
            hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.5  # Saturation / 2
            
            pastel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            if c == 4:
                pastel = np.dstack([pastel, img_uint8[:, :, 3]])
            
            pastel_float = pastel.astype(np.float32) / 255.0
            blended = img * (1 - strength) + pastel_float * strength
            results.append(blended)
        else:
            results.append(img)
    
    output = torch.from_numpy(np.stack(results)).float()
    return output

def apply_image_filter(image, filter_type, strength, edge_threshold, neon_hue, neon_blur):
    """Apply the selected filter with given parameters."""
    # Even for "none", return immediately but still counts as processing
    if filter_type == "none":
        return image
    elif filter_type == "b&w":
        return apply_bw_filter(image, strength)
    elif filter_type == "sepia":
        return apply_sepia_filter(image, strength)
    elif filter_type == "duotone":
        return apply_duotone_filter(image, strength)
    elif filter_type == "invert":
        return apply_invert_filter(image, strength)
    elif filter_type == "cartoon":
        return apply_cartoon_filter(image, strength, edge_threshold)
    elif filter_type == "sketch":
        return apply_sketch_filter(image, strength)
    elif filter_type == "neon":
        return apply_neon_filter(image, strength, edge_threshold, neon_hue, neon_blur)
    elif filter_type == "high_contrast":
        return apply_high_contrast_filter(image, strength)
    elif filter_type == "emboss":
        return apply_emboss_filter(image, strength)
    elif filter_type == "infrared":
        return apply_infrared_filter(image, strength)
    else:
        return image


class ImageFiltersC:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filter_type": (["none", "b&w", "sepia", "duotone", "invert", "cartoon", "sketch", 
                                "neon", "high_contrast", "emboss", "infrared"],),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "neon_hue": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "neon_blur": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "WtlNodes/image"

    def apply_effect(self, image, filter_type, strength, edge_threshold, neon_hue, 
                     neon_blur, apply_type, unique_id=None):
        
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
                # Start with the original image
                stacked_image = image
                
                # Send initial preview with current parameters
                cur_filter_type, cur_strength, cur_edge_threshold, cur_neon_hue, cur_neon_blur = _get_params(
                    uid, filter_type, strength, edge_threshold, neon_hue, neon_blur
                )
                
                start_time = time.time()
                initial_filtered = apply_image_filter(
                    stacked_image, cur_filter_type, cur_strength, cur_edge_threshold, 
                    cur_neon_hue, cur_neon_blur
                )
                processing_ms = int((time.time() - start_time) * 1000)
                _set_processing_time(uid, processing_ms)
                _send_ram_preview(initial_filtered, uid)
                
                while True:
                    # Wait for parameter change or button press
                    triggered = False
                    apply_pressed = False
                    apply_again_pressed = False
                    
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            apply_pressed = True
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply_again"):
                            apply_again_pressed = True
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)  # Short sleep to avoid busy-waiting
                    
                    # If apply button was pressed, exit loop
                    if apply_pressed:
                        final_params = _get_params(
                            uid, filter_type, strength, edge_threshold, neon_hue, neon_blur
                        )
                        break
                    
                    # Get current params and process
                    cur_filter_type, cur_strength, cur_edge_threshold, cur_neon_hue, cur_neon_blur = _get_params(
                        uid, filter_type, strength, edge_threshold, neon_hue, neon_blur
                    )
                    
                    # Time the processing
                    start_time = time.time()
                    cur_filtered = apply_image_filter(
                        stacked_image, cur_filter_type, cur_strength, cur_edge_threshold, 
                        cur_neon_hue, cur_neon_blur
                    )
                    processing_ms = int((time.time() - start_time) * 1000)
                    _set_processing_time(uid, processing_ms)
                    
                    _send_ram_preview(cur_filtered, uid)
                    
                    # Check if apply_again was triggered - stack the effect!
                    if apply_again_pressed:
                        stacked_image = cur_filtered.clone()

                # Apply final effect after exiting loop
                result = apply_image_filter(
                    stacked_image, final_params[0], final_params[1], final_params[2],
                    final_params[3], final_params[4]
                )
            
            else:
                # Process images one by one
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]  # Keep batch dimension
                    stacked_single = single_image
                    
                    # Send initial preview with current parameters
                    cur_filter_type, cur_strength, cur_edge_threshold, cur_neon_hue, cur_neon_blur = _get_params(
                        uid, filter_type, strength, edge_threshold, neon_hue, neon_blur
                    )
                    
                    start_time = time.time()
                    initial_processed = apply_image_filter(
                        stacked_single, cur_filter_type, cur_strength, cur_edge_threshold,
                        cur_neon_hue, cur_neon_blur
                    )
                    processing_ms = int((time.time() - start_time) * 1000)
                    _set_processing_time(uid, processing_ms)
                    _send_ram_preview(initial_processed, uid)
                    
                    while True:
                        # Wait for parameter change or button press
                        triggered = False
                        apply_pressed = False
                        apply_again_pressed = False
                        skip_pressed = False
                        
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                apply_pressed = True
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply_again"):
                                apply_again_pressed = True
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                skip_pressed = True
                                triggered = True
                                break
                            time.sleep(0.05)  # Short sleep to avoid busy-waiting
                        
                        # Handle skip
                        if skip_pressed:
                            result_list.append(single_image)
                            final_params = None
                            break
                        
                        # Handle apply
                        if apply_pressed:
                            final_params = _get_params(
                                uid, filter_type, strength, edge_threshold, neon_hue, neon_blur
                            )
                            break
                        
                        # Get current params and process
                        cur_filter_type, cur_strength, cur_edge_threshold, cur_neon_hue, cur_neon_blur = _get_params(
                            uid, filter_type, strength, edge_threshold, neon_hue, neon_blur
                        )
                        
                        # Time the processing
                        start_time = time.time()
                        cur_processed = apply_image_filter(
                            stacked_single, cur_filter_type, cur_strength, cur_edge_threshold,
                            cur_neon_hue, cur_neon_blur
                        )
                        processing_ms = int((time.time() - start_time) * 1000)
                        _set_processing_time(uid, processing_ms)
                        
                        _send_ram_preview(cur_processed, uid)
                        
                        # Check if apply_again was triggered - stack the effect!
                        if apply_again_pressed:
                            stacked_single = cur_processed.clone()

                    # Apply final effect for this image if not skipped
                    if final_params is not None:
                        processed = apply_image_filter(
                            stacked_single, final_params[0], final_params[1], final_params[2],
                            final_params[3], final_params[4]
                        )
                        result_list.append(processed)

                # Concatenate all processed images back into a batch
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode (always processes all images the same way)
            result = apply_image_filter(
                image, filter_type, strength, edge_threshold, neon_hue, neon_blur
            )
                
        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"ImageFilters": ImageFiltersC}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageFilters": "Image Filters"}