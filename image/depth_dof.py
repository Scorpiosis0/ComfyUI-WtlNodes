import torch
import cv2
import numpy as np
import time
import threading
from nodes import PreviewImage

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

def _set_params(node_id: str, focus: float, rng: float, edge: int) -> None:
    """Write the newest slider values for *node_id*."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.setdefault(node_id, {})
        entry["params"] = (focus, rng, edge)

def _get_params(
    node_id: str,
    default_focus: float,
    default_range: float,
    default_edge: int,
) -> tuple[float, float, int]:
    """Return the latest parameters (or the defaults if nothing was set)."""
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id, {})
        return entry.get("params", (default_focus, default_range, default_edge))

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

class DepthDOFNode(PreviewImage):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "focus_depth": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                }),
                "blur_strength": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                }),
                "focus_range": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                }),
                "edge_fix": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                }),
                "auto_apply": ("BOOLEAN", {
                    "default": False,
                    "label_on": "On",
                    "label_off": "Off"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "blur_mask")
    FUNCTION = "apply_dof"
    CATEGORY = "image/effects"

    def __init__(self):
        super().__init__()
        # No on‑disk directory needed any more.

    # Core processing – only the preview‑loop part talks to the RAM store.
    def apply_dof(self, image, depth_map, focus_depth, blur_strength, focus_range, edge_fix, auto_apply, unique_id=None, prompt=None, extra_pnginfo=None,):

        # Clean any stale data for this node (mirrors old file‑cleanup)
        if unique_id:
            uid = str(unique_id)          # ensure consistent key type
            _clear_all(uid)

        # Early‑out if the user pressed **Skip**
        if unique_id and _check_and_clear_flag(str(unique_id), "skip"):
            batch_size = image.shape[0]
            empty_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]))
            return (image, empty_mask)

        # Convert tensors to numpy for the heavy lifting
        img_np   = image.cpu().numpy()
        depth_np = depth_map.cpu().numpy()

        batch_size = img_np.shape[0]
        results = []
        masks = []

        for b in range(batch_size):
            # image / depth preparation (identical to original)
            img = img_np[b]
            depth = depth_np[b]

            if depth.shape[-1] > 1:               # make depth single‑channel
                depth = np.mean(depth, axis=-1, keepdims=True)

            # Normalise depth to 0‑1
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            # preview loop (only when we have a UI node id)
            if unique_id:
                uid = str(unique_id)
                while True & auto_apply == False:
                    # Grab the *latest* slider values sent by the UI
                    cur_focus, cur_range, cur_edge = _get_params(uid, focus_depth, focus_range, edge_fix)

                    # Build a temporary mask with those live values
                    cur_mask = np.abs(depth - cur_focus)
                    cur_mask = np.clip(cur_mask / cur_range, 0, 1).squeeze()
                    kernel_size = abs(cur_edge) * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    cur_mask = cv2.dilate(cur_mask, kernel, iterations=1)
                    cur_mask = cv2.erode(cur_mask, kernel, iterations=1)
                    # Show a quick preview in the UI
                    self._preview_mask(cur_mask, uid, prompt, extra_pnginfo)
                    print(f"Sent Preview to def.")

                    # Check for button presses
                    if _check_and_clear_flag(uid, "apply"):
                        print(f"[DOF] **Apply** pressed for node {uid}.")
                        focus_depth, focus_range, edge_fix = (cur_focus, cur_range, cur_edge)
                        blur_mask = cur_mask          # final mask for this batch
                        break   # exit preview loop

                    if _check_and_clear_flag(uid, "skip"):
                        print(f"[DOF] **Skip** pressed for node {uid}.")
                        empty_mask = torch.zeros(
                            (batch_size, image.shape[1], image.shape[2])
                        )
                        return (image, empty_mask)

                    # Throttle the loop a little so we don’t hammer the CPU
                    time.sleep(1)

            # ---- real effect (unchanged) -----------------------------
            img_uint8 = (img * 255).astype(np.uint8)

            # Compute a Gaussian kernel size from the blur strength
            kernel_size = int(blur_strength * 2) * 2 + 1
            kernel_size = max(1, kernel_size)

            if kernel_size > 1:
                blurred = cv2.GaussianBlur(
                    img_uint8, (kernel_size, kernel_size), 0)
                blurred = blurred.astype(np.float32) / 255.0
            else:
                blurred = img

            # Blend original and blurred based on mask
            blur_mask_3ch = np.expand_dims(blur_mask, axis=-1)
            result = img * (1 - blur_mask_3ch) + blurred * blur_mask_3ch

            results.append(result)
            masks.append(blur_mask)

        # -------------------------------------------------------------
        #  4️⃣  Convert back to torch tensors and return
        # -------------------------------------------------------------
        output_img  = torch.from_numpy(np.stack(results)).float()
        output_mask = torch.from_numpy(np.stack(masks)).float()
        print(f"[DOF] Effect applied – returning final image.")
        return (output_img, output_mask)

    # -----------------------------------------------------------------
    #  Helper that pushes a mask preview to the UI (unchanged)
    # -----------------------------------------------------------------
    def _preview_mask(self, mask, unique_id, prompt, extra_pnginfo):
        """Preview the depth mask in real‑time."""
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        mask_tensor = torch.from_numpy(mask_rgb[np.newaxis, ...]).float()

        try:
            result = self.save_images(mask_tensor, filename_prefix="dof_preview_", prompt=prompt, extra_pnginfo=extra_pnginfo)

            # Send preview to UI via websocket (ComfyUI internal API)
            import server
            print(f"Sent Preview to WebSocket.")

            if hasattr(server.PromptServer, "instance"):
                server.PromptServer.instance.send_sync("executed", {"node": unique_id, "output": {"images": result["ui"]["images"]}, "prompt_id": None})

        except Exception as e:
            print(f"[DOF] Preview error: {e}")

# -----------------------------------------------------------------
#  Node registration for ComfyUI (unchanged)
# -----------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"DepthDOFNode": DepthDOFNode}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthDOFNode": "Depth of Field (DOF)"}