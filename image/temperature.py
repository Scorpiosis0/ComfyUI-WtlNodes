import torch
import math
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, temperature: float) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (temperature,)

def _get_params(node_id: str, temperature: float) -> tuple[float]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (temperature,))

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

def kelvin_to_rgb(temp_k):

        # Convert color temperature in Kelvin to RGB multipliers. Based on Tanner Helland's approximation.
        # Clamp temperature to valid range
        temp_k = max(1000, min(40000, temp_k))
        
        # Work with temp/100 for the algorithm
        temp = temp_k / 100.0
        
        # Calculate Red
        if temp <= 66:
            red = 255.0
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))
        
        # Calculate Green
        if temp <= 66:
            green = temp
            green = 99.4708025861 * math.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))
        
        # Calculate Blue
        if temp >= 66:
            blue = 255.0
        elif temp <= 19:
            blue = 0.0
        else:
            blue = temp - 10
            blue = 138.5177312231 * math.log(blue) - 305.0447927307
        blue = max(0, min(255, blue))
        
        # Normalize to 0-1 range
        return (red / 255.0, green / 255.0, blue / 255.0)

def _apply_temperature(image, temperature):
        # Apply color temperature adjustment to image tensor.
        # Get RGB multipliers for the target temperature
        r_mult, g_mult, b_mult = kelvin_to_rgb(temperature)
        
        # Get reference multipliers (6500K = neutral daylight)
        r_ref, g_ref, b_ref = kelvin_to_rgb(6500.0)
        
        # Normalize multipliers so 6500K = no change
        r_mult = r_mult / r_ref
        g_mult = g_mult / g_ref
        b_mult = b_mult / b_ref
        
        # Create multiplier tensor on the same device as input
        multipliers = torch.tensor([r_mult, g_mult, b_mult], dtype=image.dtype, device=image.device)
        
        # Apply temperature adjustment
        # image shape is (B, H, W, C), we want to multiply along C dimension
        adjusted = image * multipliers
        
        # Preserve luminosity using Rec. 709 coefficients
        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=image.dtype, device=image.device)
        
        # Calculate original and adjusted luminosity
        original_lum = torch.sum(image * luma_weights, dim=-1, keepdim=True)
        adjusted_lum = torch.sum(adjusted * luma_weights, dim=-1, keepdim=True)
        
        # Avoid division by zero
        scale = torch.where(adjusted_lum > 0.001, original_lum / adjusted_lum, torch.ones_like(adjusted_lum))
        
        out = adjusted * scale
        
        # Clamp to valid range [0, 1]
        out = torch.clamp(out, 0.0, 1.0)

        return out

class ColorTemperatureC:
    """
    ComfyUI Node for adjusting color temperature using Tanner Helland's approximation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 6500.0,
                    "min": 1000.0,
                    "max": 40000.0,
                    "step": 100.0,
                    "round": 0.1,
                    "display": "temperature (in Kelvin)",
                }),
                "apply_type": (["none","auto_apply","apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "WtlNodes/image"

    def apply_effect(self, image, temperature, apply_type, unique_id=None):

        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                # Process all images at once (original behavior)
                while True:
                    cur_temperature = _get_params(uid, temperature)[0]
                    cur_image = _apply_temperature(image, cur_temperature)
                    _send_ram_preview(cur_image, uid)

                    # Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        temperature = cur_temperature
                        break

                    if _check_and_clear_flag(uid, "skip"):
                        return {"result": (image,)}
                    
                    time.sleep(0.25)

                result = _apply_temperature(image, temperature)

            else:
                # Process images one by one
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]  # Keep batch dimension
                    
                    while True:
                        cur_temperature = _get_params(uid, temperature)[0]
                        cur_image = _apply_temperature(single_image, cur_temperature)
                        _send_ram_preview(cur_image, uid)

                        # Check for button presses
                        if _check_and_clear_flag(uid, "apply"):
                            final_temperature = cur_temperature
                            break

                        if _check_and_clear_flag(uid, "skip"):
                            # Skip this image, use original
                            result_list.append(single_image)
                            final_temperature = None
                            break
                        
                        time.sleep(0.25)
                    
                    # Apply final effect for this image if not skipped
                    if final_temperature is not None:
                        processed = _apply_temperature(single_image, final_temperature)
                        result_list.append(processed)
                
                # Concatenate all processed images back into a batch
                result = torch.cat(result_list, dim=0)

        else:
            result =_apply_temperature(image, temperature)
        
        return {"result": (result,)}

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"Temperature": ColorTemperatureC}
NODE_DISPLAY_NAME_MAPPINGS = {"Temperature": "Temperature (Tanner Helland's algorithm)"}