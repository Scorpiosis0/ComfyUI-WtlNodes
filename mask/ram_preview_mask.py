import torch
from ..helper.ram_preview import _send_ram_preview


class RAMPreviewMaskC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_mask"
    OUTPUT_NODE = True
    CATEGORY = "WtlNodes/mask"
    DESCRIPTION = "Previews a mask using RAM storage only (no disk I/O)."

    def preview_mask(self, mask, unique_id=None):
        uid = str(unique_id) if unique_id else "preview"
        mask_rgb = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        _send_ram_preview(mask_rgb, uid, resize=False)
        return ()


NODE_CLASS_MAPPINGS = {"RAMPreviewMask": RAMPreviewMaskC}
NODE_DISPLAY_NAME_MAPPINGS = {"RAMPreviewMask": "RAM Preview Mask"}
