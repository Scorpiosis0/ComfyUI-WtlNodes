import torch
import numpy as np
from scipy import ndimage
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, area_x: int, area_y: int, keep: str) -> None:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        new_params = (area_x, area_y, keep)
        if entry.get("params") != new_params:
            entry["params"] = new_params
            entry["params_changed"] = True
            entry["processing_complete"] = False

def _get_params(node_id: str, area_x: int, area_y: int, keep: str) -> tuple[int, int, str]:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (area_x, area_y, keep))

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

def apply_mask_filter(masks, area_x, area_y, keep):
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)

    filtered_masks = []

    for i in range(masks.shape[0]):
        mask = masks[i]
        mask_np = mask.cpu().numpy()
        binary_mask = mask_np > 0.5
        labeled_mask, num_features = ndimage.label(binary_mask)
        filtered_mask = np.zeros_like(mask_np)

        for component_id in range(1, num_features + 1):
            component = (labeled_mask == component_id)
            area = np.sum(component)

            should_keep = False
            if keep == "above_x":
                should_keep = area >= area_x
            elif keep == "bellow_x":
                should_keep = area <= area_x
            elif keep == "between_x_y":
                should_keep = area_x <= area <= area_y

            if should_keep:
                filtered_mask[component] = mask_np[component]

        filtered_masks.append(torch.from_numpy(filtered_mask).to(mask.device).to(mask.dtype))

    return torch.stack(filtered_masks, dim=0)

class MaskFilterC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "area_x": ("INT", {"default": 3000, "min": 0, "max": 10000000, "step": 100, "display": "number"}),
                "area_y": ("INT", {"default": 5000, "min": 0, "max": 10000000, "step": 100, "display": "number"}),
                "keep": (["above_x", "bellow_x", "between_x_y"],),
                "apply_type": (["none", "auto_apply", "apply_all"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "filter_masks"
    CATEGORY = "WtlNodes/mask"

    def filter_masks(self, masks, area_x, area_y, keep, apply_type, unique_id=None):
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)

        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            return {"result": (masks,)}

        if unique_id and not apply_type == "auto_apply":
            uid = str(unique_id)

            if apply_type == "apply_all":
                cur_area_x, cur_area_y, cur_keep = _get_params(uid, area_x, area_y, keep)
                start_time = time.time()
                initial = apply_mask_filter(masks, cur_area_x, cur_area_y, cur_keep)
                _set_processing_time(uid, int((time.time() - start_time) * 1000))
                _send_ram_preview(initial, uid)

                while True:
                    triggered = False
                    while not triggered:
                        if _check_and_clear_params_changed(uid):
                            triggered = True
                            break
                        if _check_and_clear_flag(uid, "apply"):
                            final_area_x, final_area_y, final_keep = _get_params(uid, area_x, area_y, keep)
                            break
                        if _check_and_clear_flag(uid, "skip"):
                            return {"result": (masks,)}
                        time.sleep(0.05)

                    if not triggered:
                        break

                    cur_area_x, cur_area_y, cur_keep = _get_params(uid, area_x, area_y, keep)
                    start_time = time.time()
                    cur_filtered = apply_mask_filter(masks, cur_area_x, cur_area_y, cur_keep)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    _send_ram_preview(cur_filtered, uid)

                result = apply_mask_filter(masks, final_area_x, final_area_y, final_keep)

            else:
                batch_size = masks.shape[0]
                result_list = []

                for i in range(batch_size):
                    single_mask = masks[i:i+1]

                    cur_area_x, cur_area_y, cur_keep = _get_params(uid, area_x, area_y, keep)
                    start_time = time.time()
                    initial = apply_mask_filter(single_mask, cur_area_x, cur_area_y, cur_keep)
                    _set_processing_time(uid, int((time.time() - start_time) * 1000))
                    preview_image = initial.unsqueeze(-1).repeat(1, 1, 1, 3)
                    _send_ram_preview(preview_image, uid)

                    final_area_x = None
                    while True:
                        triggered = False
                        while not triggered:
                            if _check_and_clear_params_changed(uid):
                                triggered = True
                                break
                            if _check_and_clear_flag(uid, "apply"):
                                final_area_x, final_area_y, final_keep = _get_params(uid, area_x, area_y, keep)
                                break
                            if _check_and_clear_flag(uid, "skip"):
                                result_list.append(single_mask)
                                final_area_x = None
                                break
                            time.sleep(0.05)

                        if not triggered:
                            break

                        cur_area_x, cur_area_y, cur_keep = _get_params(uid, area_x, area_y, keep)
                        start_time = time.time()
                        cur_filtered = apply_mask_filter(single_mask, cur_area_x, cur_area_y, cur_keep)
                        _set_processing_time(uid, int((time.time() - start_time) * 1000))
                        preview_image = cur_filtered.unsqueeze(-1).repeat(1, 1, 1, 3)
                        _send_ram_preview(preview_image, uid)

                    if final_area_x is not None:
                        result_list.append(apply_mask_filter(single_mask, final_area_x, final_area_y, final_keep))

                result = torch.cat(result_list, dim=0)
        else:
            result = apply_mask_filter(masks, area_x, area_y, keep)

        return {"result": (result,)}


NODE_CLASS_MAPPINGS = {"MaskFilter": MaskFilterC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskFilter": "Mask Filter"}
