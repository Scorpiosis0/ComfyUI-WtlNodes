import torch
import cv2
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, shadow_adjustment: float, highlight_adjustment: float,
                midpoint: float, feather_radius: float) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, *defaults: float) -> tuple:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", defaults)

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

def _apply(image, shadow_adjustment, highlight_adjustment, midpoint, feather_radius):
    device = image.device
    img = torch.clamp(image, 0, 1)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    max_c = torch.maximum(torch.maximum(r, g), b)
    min_c = torch.minimum(torch.minimum(r, g), b)
    delta = max_c - min_c

    v = max_c
    s = torch.where(max_c != 0, delta / max_c, torch.zeros_like(max_c))

    h = torch.zeros_like(max_c)
    mask = delta != 0
    mask_r = mask & (r == max_c)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    mask_g = mask & (g == max_c)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    mask_b = mask & (b == max_c)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    h = h % 360

    shadow_mask = torch.clamp((midpoint - v) / (midpoint + 1e-10), 0, 1)
    highlight_mask = torch.clamp((v - midpoint) / (1.0 - midpoint + 1e-10), 0, 1)

    if feather_radius > 0:
        ksize = max(3, int(feather_radius * 2) | 1)
        shadow_np = shadow_mask[0].cpu().numpy()
        highlight_np = highlight_mask[0].cpu().numpy()
        shadow_blur = cv2.GaussianBlur(shadow_np, (ksize, ksize), feather_radius / 3.0)
        highlight_blur = cv2.GaussianBlur(highlight_np, (ksize, ksize), feather_radius / 3.0)
        shadow_mask = torch.from_numpy(shadow_blur).unsqueeze(0).to(device)
        highlight_mask = torch.from_numpy(highlight_blur).unsqueeze(0).to(device)

    v_adjusted = v.clone()
    if shadow_adjustment != 0:
        v_adjusted = v_adjusted + (shadow_adjustment / 100) * shadow_mask
    if highlight_adjustment != 0:
        v_adjusted = v_adjusted + (highlight_adjustment / 100) * highlight_mask
    v_adjusted = torch.clamp(v_adjusted, 0, 1)

    c = v_adjusted * s
    x = c * (1 - torch.abs((h / 60) % 2 - 1))
    m = v_adjusted - c
    h_i = (h / 60).long()

    r_out = torch.zeros_like(h)
    g_out = torch.zeros_like(h)
    b_out = torch.zeros_like(h)

    for idx, (rc, gc, bc) in enumerate([(c, x, 0), (x, c, 0), (0, c, x),
                                         (0, x, c), (x, 0, c), (c, 0, x)]):
        m_ = h_i == idx
        r_out[m_] = (rc[m_] if isinstance(rc, torch.Tensor) else rc)
        g_out[m_] = (gc[m_] if isinstance(gc, torch.Tensor) else gc)
        b_out[m_] = (bc[m_] if isinstance(bc, torch.Tensor) else bc)

    rgb = torch.stack([r_out + m, g_out + m, b_out + m], dim=-1)
    return torch.clamp(rgb, 0, 1)

class HighlightShadowC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_adjustment": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "round": 0.1,
                }),
                "highlight_adjustment": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "round": 0.1,
                }),
                "midpoint": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "feather_radius": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0,
                }),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_highlight_shadow"
    CATEGORY = "WtlNodes/image"

    def adjust_highlight_shadow(self, image, shadow_adjustment, highlight_adjustment,
                                midpoint, feather_radius, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (image,)}

        if unique_id and apply_type != "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                t0 = time.time()
                preview = _apply(image, *cur)
                _set_processing_time(uid, int((time.time() - t0) * 1000))
                _send_ram_preview(preview, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (image,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                    t0 = time.time()
                    preview = _apply(image, *cur)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                result = _apply(image, *final)

            else:
                batch_size = image.shape[0]
                result_list = []

                for i in range(batch_size):
                    single = image[i:i+1]

                    cur = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                    t0 = time.time()
                    preview = _apply(single, *cur)
                    _set_processing_time(uid, int((time.time() - t0) * 1000))
                    _send_ram_preview(preview, uid)

                    final = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single)
                                final = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur = _get_params(uid, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)
                        t0 = time.time()
                        preview = _apply(single, *cur)
                        _set_processing_time(uid, int((time.time() - t0) * 1000))
                        _send_ram_preview(preview, uid)

                    if final is not None:
                        result_list.append(_apply(single, *final))  # fixed: was using full `image`

                result = torch.cat(result_list, dim=0)
        else:
            result = _apply(image, shadow_adjustment, highlight_adjustment, midpoint, feather_radius)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"HighlightShadow": HighlightShadowC}
NODE_DISPLAY_NAME_MAPPINGS = {"HighlightShadow": "Highlight & Shadow"}
