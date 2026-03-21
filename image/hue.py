import torch
import time
import threading
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, hue: float) -> None:
    """Write the newest slider values for *node_id* and set trigger flag."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (hue,)
        old_params = entry.get("params")
        
        # Only set trigger if params actually changed
        if old_params != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False  # Mark as not complete

def _get_params(node_id: str, hue: float) -> tuple[float]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (hue,))

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

def _hue_hsv(image, hue):
    # image: (1, H, W, C) in range [0,1]
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    # max/min per pixel
    max_c = torch.maximum(torch.maximum(r, g), b)
    min_c = torch.minimum(torch.minimum(r, g), b)
    delta = max_c - min_c

    v = max_c
    s = torch.where(max_c != 0, delta / max_c, torch.zeros_like(max_c))

    # compute hue (same logic as NumPy)
    h = torch.zeros_like(max_c)
    mask = delta != 0

    mask_r = mask & (r == max_c)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)

    mask_g = mask & (g == max_c)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)

    mask_b = mask & (b == max_c)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    h = h % 360

    # Apply hue
    h = (h + hue) % 360

    # HSV -> RGB
    c = v * s
    x = c * (1 - torch.abs((h / 60) % 2 - 1))
    m = v - c

    h_i = (h / 60).long()

    r_out = torch.zeros_like(h)
    g_out = torch.zeros_like(h)
    b_out = torch.zeros_like(h)

    mask0 = h_i == 0
    r_out[mask0], g_out[mask0], b_out[mask0] = c[mask0], x[mask0], 0

    mask1 = h_i == 1
    r_out[mask1], g_out[mask1], b_out[mask1] = x[mask1], c[mask1], 0

    mask2 = h_i == 2
    r_out[mask2], g_out[mask2], b_out[mask2] = 0, c[mask2], x[mask2]

    mask3 = h_i == 3
    r_out[mask3], g_out[mask3], b_out[mask3] = 0, x[mask3], c[mask3]

    mask4 = h_i == 4
    r_out[mask4], g_out[mask4], b_out[mask4] = x[mask4], 0, c[mask4]

    mask5 = h_i == 5
    r_out[mask5], g_out[mask5], b_out[mask5] = c[mask5], 0, x[mask5]

    out = torch.stack([r_out + m, g_out + m, b_out + m], dim=-1)
    out = torch.clamp(out, 0.0, 1.0)

    return (out)

class HueC:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
                "hue":("FLOAT",{
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "apply_type": (["none","auto_apply","apply_all"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="hue"
    CATEGORY = "WtlNodes/image"

    def hue (self, image, hue, apply_type, unique_id=None):

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
                
                # Send initial preview with current parameters
                cur_hue = _get_params(uid, hue)[0]
                start_time = time.time()
                initial_image = _hue_hsv(image, cur_hue)
                processing_ms = int((time.time() - start_time) * 1000)
                _set_processing_time(uid, processing_ms)
                _send_ram_preview(initial_image, uid)
                
                while True:
                    # Wait for parameter change or button press
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_hue = _get_params(uid, hue)[0]
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)  # Short sleep to avoid busy-waiting
                    
                    # If apply button was pressed, exit loop
                    if not triggered:
                        break
                    
                    # Get current params and process
                    cur_hue = _get_params(uid, hue)[0]
                    start_time = time.time()
                    cur_image = _hue_hsv(image, cur_hue)
                    processing_ms = int((time.time() - start_time) * 1000)
                    _set_processing_time(uid, processing_ms)
                    _send_ram_preview(cur_image, uid)

                # Apply final effect after exiting loop
                result = _hue_hsv(image, final_hue)
                result = torch.clamp(result, 0.0, 1.0)

            else:
                # Process images one by one
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_image = image[i:i+1]  # Keep batch dimension
                    
                    # Send initial preview with current parameters
                    cur_hue = _get_params(uid, hue)[0]
                    start_time = time.time()
                    initial_image = _hue_hsv(single_image, cur_hue)
                    processing_ms = int((time.time() - start_time) * 1000)
                    _set_processing_time(uid, processing_ms)
                    _send_ram_preview(initial_image, uid)
                    
                    while True:
                        # Wait for parameter change or button press
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_hue = _get_params(uid, hue)[0]
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_image)
                                final_hue = None
                                break
                            time.sleep(0.05)  # Short sleep to avoid busy-waiting
                        
                        # If apply or skip button was pressed, exit loop
                        if not triggered:
                            break
                        
                        # Get current params and process
                        cur_hue = _get_params(uid, hue)[0]
                        start_time = time.time()
                        cur_image = _hue_hsv(single_image, cur_hue)
                        processing_ms = int((time.time() - start_time) * 1000)
                        _set_processing_time(uid, processing_ms)
                        _send_ram_preview(cur_image, uid)

                    # Apply final effect for this image if not skipped
                    if final_hue is not None:
                        processed = _hue_hsv(single_image, final_hue)
                        result_list.append(processed)

                # Concatenate all processed images back into a batch
                result = torch.cat(result_list, dim=0)
        else:
            # Auto-apply mode (always processes all images the same way)
            result = _hue_hsv(image, hue)
            result = torch.clamp(result, 0.0, 1.0)
                
        return {"result": (result,)}

NODE_CLASS_MAPPINGS = {"Hue": HueC}
NODE_DISPLAY_NAME_MAPPINGS = {"Hue": "Hue"}