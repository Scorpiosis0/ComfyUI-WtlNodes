import torch
import threading
import time
from ..helper.ram_preview import _send_ram_preview

_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()

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

class RAMPreviewImageC:
    def __init__(self):
        self.compress_level = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to preview."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                },
        }
    

    RETURN_TYPES = ()
    FUNCTION = "preview_images"
    OUTPUT_NODE = True
    CATEGORY = "WtlNodes/image"
    DESCRIPTION = "Previews images using RAM storage only (no disk I/O)."

    def preview_images(self, images, unique_id=None):

        # Clean any stale data for this node
        if unique_id:
            uid = str(unique_id)
            _clear_all(uid)
        
        _send_ram_preview(images, uid, resize=False)
        return (images,)
    
NODE_CLASS_MAPPINGS = {"RAMPreviewImage": RAMPreviewImageC}
NODE_DISPLAY_NAME_MAPPINGS = {"RAMPreviewImage": "RAM Preview Image"}