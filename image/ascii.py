import torch
import threading
import time
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()
_FONT_CACHE: dict[tuple, any] = {}  # Cache loaded fonts by (name, size, bold, italic)
_DENSITY_CACHE: dict[tuple, str] = {}  # Cache character density ramps by (font_name, bold, italic)
_LAST_FONT_NAME: str = None  # Track last font to clear old cache

def calculate_char_density_ramp(font, font_name, bold, italic):
    """
    Calculate character density ramp for the given font configuration.
    Returns a string of characters sorted from light to dark.
    """
    # Check cache first
    cache_key = (font_name, bold, italic)
    
    # Clear old font's cache if font changed
    global _LAST_FONT_NAME
    if _LAST_FONT_NAME is not None and _LAST_FONT_NAME != font_name:
        # Remove all cache entries for the old font
        keys_to_remove = [k for k in _DENSITY_CACHE.keys() if k[0] == _LAST_FONT_NAME]
        for key in keys_to_remove:
            _DENSITY_CACHE.pop(key, None)
        print(f"[ASCII Effect] Cleared density cache for old font '{_LAST_FONT_NAME}'")
    _LAST_FONT_NAME = font_name
    
    if cache_key in _DENSITY_CACHE:
        return _DENSITY_CACHE[cache_key]
    
    print(f"[ASCII Effect] Calculating character density ramp for font '{font_name}' (bold={bold}, italic={italic})...")
    
    # ASCII printable characters (32-126), excluding space which we'll add manually at the start
    char_range = list(range(33, 127))
    
    densities = []
    
    # Create a test image to measure character densities
    test_size = 100
    
    for char_code in char_range:
        char = chr(char_code)
        
        # Skip control characters and other invisibles
        if char in '\t\n\r\v\f':
            continue
        
        # Create white background image
        img = Image.new('L', (test_size, test_size), 255)
        draw = ImageDraw.Draw(img)
        
        # Draw character in black
        try:
            draw.text((10, 10), char, fill=0, font=font)
        except:
            # Skip characters that can't be rendered
            continue
        
        # Convert to numpy and calculate density
        img_array = np.array(img)
        
        # Count black/dark pixels (pixel value < 200)
        dark_pixels = np.sum(img_array < 200)
        total_pixels = test_size * test_size
        density = dark_pixels / total_pixels
        
        densities.append((char, density))
    
    # Sort by density (ascending - light to dark)
    densities.sort(key=lambda x: x[1])
    
    # Build the character ramp string, always starting with space
    char_ramp = ' ' + ''.join([char for char, _ in densities])
    
    # Cache the result
    _DENSITY_CACHE[cache_key] = char_ramp
    
    print(f"[ASCII Effect] Density ramp calculated: {len(char_ramp)} characters")
    
    return char_ramp

def get_available_fonts():
    """Get list of available TrueType fonts on the system."""
    fonts = ["Default"]  # Always include default option
    
    # Add custom fonts directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    custom_fonts_dir = os.path.join(os.path.dirname(current_dir), "font")
    
    # Common font directories
    font_dirs = []
    
    # Add custom fonts first (priority)
    if os.path.exists(custom_fonts_dir):
        font_dirs.append(custom_fonts_dir)
    
    if sys.platform == "win32":
        windows_fonts = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        font_dirs.append(windows_fonts)
    
    # Linux font directories
    font_dirs.extend([
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts")
    ])
    
    # macOS font directories
    font_dirs.extend([
        "/Library/Fonts",
        "/System/Library/Fonts",
        os.path.expanduser("~/Library/Fonts")
    ])
    
    found_fonts = set()
    
    for font_dir in font_dirs:
        if not os.path.exists(font_dir):
            continue
        
        for root, dirs, files in os.walk(font_dir):
            for file in files:
                if file.lower().endswith(('.ttf', '.otf')):
                    # Get font name without extension
                    font_name = os.path.splitext(file)[0]
                    # Remove common suffixes
                    for suffix in ['-Regular', '-Bold', '-Italic', '-BoldItalic', 
                                   'Regular', 'Bold', 'Italic', 'BoldItalic', ' Regular']:
                        if font_name.endswith(suffix):
                            font_name = font_name[:-len(suffix)]
                    found_fonts.add(font_name.strip())
    
    fonts.extend(sorted(found_fonts))
    print(f"[ASCII Effect] Found {len(fonts)-1} fonts on system")
    
    return fonts

# Cache available fonts at module load
AVAILABLE_FONTS = get_available_fonts()

def _set_params(node_id: str, red_weight: float, green_weight: float, blue_weight: float, 
                char_set: str, char_size: float, background: str, font_name: str,
                bold: bool, italic: bool, spacing: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (red_weight, green_weight, blue_weight, char_set, char_size, 
                          background, font_name, bold, italic, spacing)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, red_weight: float, green_weight: float, blue_weight: float,
                char_set: str, char_size: float, background: str, font_name: str,
                bold: bool, italic: bool, spacing: float) -> tuple:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        stored_params = entry.get("params", None)
        
        # If we have stored params, return them
        if stored_params is not None:
            return stored_params
        
        # Otherwise return the defaults
        return (red_weight, green_weight, blue_weight, char_set, char_size, 
                background, font_name, bold, italic, spacing)

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
            flags[flag] = False          # clear it for the next poll
            return True
        return False

def _clear_all(node_id: str) -> None:
    """Remove *everything* stored for a node – used at the start of a run."""
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


# Static character density strings (from light to dark) - for special modes
STATIC_CHAR_SETS = {
    "Numbers": " 1234567890",
    "Letters": " .,'ilI|/\\cCoO08B@",
    "Special Characters": " .'`-_=+*#@"
}

def apply_ascii_effect(image_tensor, red_weight, green_weight, blue_weight, char_set,
                      char_size, background, font_name, bold, italic, spacing):
    """
    Apply ASCII effect to a single image tensor.
    
    Args:
        image_tensor: torch.Tensor of shape (1, H, W, C) in range [0, 1]
        red_weight: float 0-255
        green_weight: float 0-255
        blue_weight: float 0-255
        char_set: str - key from CHAR_SETS
        char_size: int - font size in pixels
        background: str - "black" or "white"
        font_name: str - name of font to use
        bold: bool - make font bold
        italic: bool - make font italic
        spacing: float - character spacing multiplier (affects both horizontal and vertical)
    
    Returns:
        torch.Tensor of shape (1, H, W, C) in range [0, 1]
    """
    # Convert to numpy and scale to 0-255
    img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    height, width, channels = img_np.shape
    
    # Check font cache first
    font_cache_key = (font_name, int(char_size), bold, italic)
    font = _FONT_CACHE.get(font_cache_key)
    
    if font is None:
        # Try to load the requested font
        font_loaded = False
        
        if font_name != "Default":
            # Common font directories
            font_dirs = []
            
            # Add custom fonts directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            custom_fonts_dir = os.path.join(os.path.dirname(current_dir), "font")
            if os.path.exists(custom_fonts_dir):
                font_dirs.append(custom_fonts_dir)
            
            if sys.platform == "win32":
                windows_fonts = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
                font_dirs.append(windows_fonts)
            
            font_dirs.extend([
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                "/Library/Fonts",
                "/System/Library/Fonts",
                os.path.expanduser("~/Library/Fonts")
            ])
            
            # Build possible font file names
            suffixes = []
            if bold and italic:
                suffixes = ['-BoldItalic', 'BoldItalic', '-Bold-Italic', ' Bold Italic', 'bd_it', 'z']
            elif bold:
                suffixes = ['-Bold', 'Bold', 'bd', 'b']
            elif italic:
                suffixes = ['-Italic', 'Italic', 'i', 'it']
            else:
                suffixes = ['-Regular', 'Regular', '', '-Normal', 'Normal']
            
            # Try to find the font file
            for font_dir in font_dirs:
                if not os.path.exists(font_dir):
                    continue
                
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if not file.lower().endswith(('.ttf', '.otf')):
                            continue
                        
                        file_base = os.path.splitext(file)[0]
                        
                        # Check if this file matches our font
                        for suffix in suffixes:
                            if file_base == font_name + suffix or file_base.lower() == (font_name + suffix).lower():
                                font_path = os.path.join(root, file)
                                try:
                                    font = ImageFont.truetype(font_path, int(char_size))
                                    font_loaded = True
                                    print(f"[ASCII Effect] Loaded font: {font_path}")
                                    break
                                except Exception as e:
                                    print(f"[ASCII Effect] Failed to load {font_path}: {e}")
                        
                        if font_loaded:
                            break
                    
                    if font_loaded:
                        break
                
                if font_loaded:
                    break
        
        # Fallback to default fonts if custom font not loaded
        if not font_loaded:
            if font_name != "Default":
                print(f"[ASCII Effect] Could not find font '{font_name}', falling back to default")
            
            try:
                font = ImageFont.truetype("cour.ttf", int(char_size))
                print("[ASCII Effect] Loaded default font: cour.ttf")
            except:
                try:
                    font = ImageFont.truetype("Courier New.ttf", int(char_size))
                    print("[ASCII Effect] Loaded default font: Courier New.ttf")
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", int(char_size))
                        print("[ASCII Effect] Loaded default font: DejaVuSansMono.ttf")
                    except:
                        font = ImageFont.load_default()
                        print("[ASCII Effect] Loaded PIL default font")
        
        # Cache the loaded font
        _FONT_CACHE[font_cache_key] = font
    
    # Get character set based on selection
    if char_set == "ASCII Table":
        # Use dynamic density calculation for ASCII Table
        chars = calculate_char_density_ramp(font, font_name, bold, italic)
    else:
        # Use static character sets for other modes
        chars = STATIC_CHAR_SETS[char_set]
    
    # Calculate character dimensions
    dummy_img = Image.new('RGB', (100, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), "M", font=font)
    char_width = (bbox[2] - bbox[0]) * spacing
    char_height = (bbox[3] - bbox[1]) * spacing
    
    # Calculate grid dimensions
    cols = max(1, int(width / char_width))
    rows = max(1, int(height / char_height))
    
    # Calculate block size
    block_width = width / cols
    block_height = height / rows
    
    # Create output image
    bg_color = (0, 0, 0) if background == "black" else (255, 255, 255)
    output_img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(output_img)
    
    # Process each block
    for row in range(rows):
        for col in range(cols):
            # Calculate block boundaries
            x1 = int(col * block_width)
            y1 = int(row * block_height)
            x2 = int((col + 1) * block_width)
            y2 = int((row + 1) * block_height)
            
            # Extract block
            block = img_np[y1:y2, x1:x2]
            
            if block.size == 0:
                continue
            
            # Calculate average color values
            r_avg = np.mean(block[:, :, 0])
            g_avg = np.mean(block[:, :, 1])
            b_avg = np.mean(block[:, :, 2])
            
            # Calculate brightness using standard grayscale conversion for character selection
            # This ensures character selection is independent of color weights
            brightness = (r_avg * 0.299 + g_avg * 0.587 + b_avg * 0.114)
            brightness = brightness / 255.0  # Normalize to 0-1
            
            # Map brightness to character index
            char_index = int(brightness * (len(chars) - 1))
            char_index = min(char_index, len(chars) - 1)
            char = chars[char_index]
            
            # Calculate weighted color (weights act as multipliers, normalized to 0-1 range)
            # RGB weights only affect the color, not which character is selected
            weight_r = red_weight / 255.0
            weight_g = green_weight / 255.0
            weight_b = blue_weight / 255.0
            
            color_r = int(r_avg * weight_r)
            color_g = int(g_avg * weight_g)
            color_b = int(b_avg * weight_b)
            
            # Handle case where all weights are 0
            if red_weight == 0 and green_weight == 0 and blue_weight == 0:
                if background == "black":
                    color_r = color_g = color_b = 255  # White text on black
                else:
                    color_r = color_g = color_b = 0    # Black text on white
            
            color_r = min(255, max(0, color_r))
            color_g = min(255, max(0, color_g))
            color_b = min(255, max(0, color_b))
            
            char_color = (color_r, color_g, color_b)
            
            # Draw character
            draw.text((x1, y1), char, fill=char_color, font=font)
    
    # Convert back to tensor
    output_np = np.array(output_img).astype(np.float32) / 255.0
    output_tensor = torch.from_numpy(output_np).unsqueeze(0)
    
    return output_tensor


class ASCIIC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_weight": ("FLOAT", {
                    "default": 255.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "green_weight": ("FLOAT", {
                    "default": 255.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "blue_weight": ("FLOAT", {
                    "default": 255.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "char_set": (["ASCII Table", "Numbers", "Letters", "Special Characters"],),
                "char_size": ("FLOAT", {
                    "default": 10.0,
                    "min": 4.0,
                    "max": 50.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "background": (["black", "white"],),
                "font_name": (AVAILABLE_FONTS,),
                "bold": ("BOOLEAN", {
                    "default": False,
                }),
                "italic": ("BOOLEAN", {
                    "default": False,
                }),
                "spacing": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "round": 0.01,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ascii_effect"
    CATEGORY = "WtlNodes/image"
    
    def ascii_effect(self, image, red_weight, green_weight, blue_weight, char_set,
                     char_size, background, font_name, bold, italic, spacing, apply_type, unique_id=None):

        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur_params = _get_params(uid, red_weight, green_weight, blue_weight,
                                         char_set, char_size, background, font_name, bold, italic, spacing)
                cur_red, cur_green, cur_blue, cur_char_set, cur_char_size, cur_bg, cur_font, cur_bold, cur_italic, cur_spacing = cur_params
                start_time = time.time()
                initial_image = apply_ascii_effect(image, cur_red, cur_green, cur_blue,
                                                   cur_char_set, cur_char_size, cur_bg,
                                                   cur_font, cur_bold, cur_italic, cur_spacing)
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(initial_image, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_params = _get_params(uid, red_weight, green_weight, blue_weight,
                                                       char_set, char_size, background, font_name, bold, italic, spacing)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_params = _get_params(uid, red_weight, green_weight, blue_weight,
                                             char_set, char_size, background, font_name, bold, italic, spacing)
                    cur_red, cur_green, cur_blue, cur_char_set, cur_char_size, cur_bg, cur_font, cur_bold, cur_italic, cur_spacing = cur_params
                    start_time = time.time()
                    cur_image = apply_ascii_effect(image, cur_red, cur_green, cur_blue,
                                                   cur_char_set, cur_char_size, cur_bg,
                                                   cur_font, cur_bold, cur_italic, cur_spacing)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_image, uid)

                f_red, f_green, f_blue, f_char_set, f_char_size, f_bg, f_font, f_bold, f_italic, f_spacing = final_params
                result = apply_ascii_effect(image, f_red, f_green, f_blue,
                                            f_char_set, f_char_size, f_bg,
                                            f_font, f_bold, f_italic, f_spacing)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]

                    cur_params = _get_params(uid, red_weight, green_weight, blue_weight,
                                             char_set, char_size, background, font_name, bold, italic, spacing)
                    cur_red, cur_green, cur_blue, cur_char_set, cur_char_size, cur_bg, cur_font, cur_bold, cur_italic, cur_spacing = cur_params
                    start_time = time.time()
                    initial_image = apply_ascii_effect(single_image, cur_red, cur_green, cur_blue,
                                                       cur_char_set, cur_char_size, cur_bg,
                                                       cur_font, cur_bold, cur_italic, cur_spacing)
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
                                final_params = _get_params(uid, red_weight, green_weight, blue_weight,
                                                           char_set, char_size, background, font_name, bold, italic, spacing)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_image)
                                final_params = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_params = _get_params(uid, red_weight, green_weight, blue_weight,
                                                 char_set, char_size, background, font_name, bold, italic, spacing)
                        cur_red, cur_green, cur_blue, cur_char_set, cur_char_size, cur_bg, cur_font, cur_bold, cur_italic, cur_spacing = cur_params
                        start_time = time.time()
                        cur_image = apply_ascii_effect(single_image, cur_red, cur_green, cur_blue,
                                                       cur_char_set, cur_char_size, cur_bg,
                                                       cur_font, cur_bold, cur_italic, cur_spacing)
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        _send_ram_preview(cur_image, uid)

                    if final_params is not None:
                        f_red, f_green, f_blue, f_char_set, f_char_size, f_bg, f_font, f_bold, f_italic, f_spacing = final_params
                        processed = apply_ascii_effect(single_image, f_red, f_green, f_blue,
                                                       f_char_set, f_char_size, f_bg,
                                                       f_font, f_bold, f_italic, f_spacing)
                        result_list.append(processed)

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_ascii_effect(image, red_weight, green_weight, blue_weight,
                                        char_set, char_size, background, font_name, bold, italic, spacing)

        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"ASCII": ASCIIC}
NODE_DISPLAY_NAME_MAPPINGS = {"ASCII": "ASCII Effect"}