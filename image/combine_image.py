from ..helper.ram_preview import _send_ram_preview


class WtlImageCombinerC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "WtlNodes/image"
    DESCRIPTION = "Previews multiple images of different sizes using RAM storage only (no disk I/O) and outputs them as an IMAGE list. Each image keeps its original dimensions — no cropping. Connect image_1, image_2… — new slots appear as you connect them."

    def execute(self, unique_id=None, **kwargs):
        # With INPUT_IS_LIST every value (including hidden) arrives wrapped in a list.
        if isinstance(unique_id, list):
            unique_id = unique_id[0] if unique_id else None
        uid = str(unique_id) if unique_id else "combine_image"

        frames = []
        i = 1
        while f"image_{i}" in kwargs:
            v = kwargs[f"image_{i}"]
            if v is not None:
                for item in v:          # each item is an IMAGE tensor (B, H, W, C)
                    if item is None:
                        continue
                    for frame in item:  # iterate batch dim -> (H, W, C)
                        frames.append(frame)
            i += 1

        if not frames:
            import torch
            return ([torch.zeros(1, 64, 64, 3)],)

        _send_ram_preview(frames, uid, resize=False)
        return ([f.unsqueeze(0) for f in frames],)


NODE_CLASS_MAPPINGS = {"WtlImageCombiner": WtlImageCombinerC}
NODE_DISPLAY_NAME_MAPPINGS = {"WtlImageCombiner": "Image Combiner"}
