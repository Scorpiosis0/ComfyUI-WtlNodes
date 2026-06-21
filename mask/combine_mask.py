from ..helper.ram_preview import _send_ram_preview


class WtlMaskCombinerC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "WtlNodes/mask"
    DESCRIPTION = "Previews multiple masks of different sizes using RAM storage only (no disk I/O) and outputs them as a MASK list. Each mask keeps its original dimensions — no cropping. Connect mask_1, mask_2… — new slots appear as you connect them."

    def execute(self, unique_id=None, **kwargs):
        # With INPUT_IS_LIST every value (including hidden) arrives wrapped in a list.
        if isinstance(unique_id, list):
            unique_id = unique_id[0] if unique_id else None
        uid = str(unique_id) if unique_id else "combine_mask"

        frames = []
        i = 1
        while f"mask_{i}" in kwargs:
            v = kwargs[f"mask_{i}"]
            if v is not None:
                for item in v:          # each item is a MASK tensor (B, H, W)
                    if item is None:
                        continue
                    for frame in item:  # iterate batch dim -> (H, W)
                        frames.append(frame)
            i += 1

        if not frames:
            import torch
            return ([torch.zeros(1, 64, 64)],)

        rgb_frames = [f.unsqueeze(-1).repeat(1, 1, 3) for f in frames]
        _send_ram_preview(rgb_frames, uid, resize=False)
        return ([f.unsqueeze(0) for f in frames],)


NODE_CLASS_MAPPINGS = {"WtlMaskCombiner": WtlMaskCombinerC}
NODE_DISPLAY_NAME_MAPPINGS = {"WtlMaskCombiner": "Mask Combiner"}
