import torch
import comfy.utils


class WtlMaskBatchC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("batch (crops to mask 1 size)",)
    FUNCTION = "execute"
    CATEGORY = "WtlNodes/mask"
    DESCRIPTION = "Outputs multiple masks as a single stacked MASK tensor. All frames are cropped to mask_1's dimensions. Connect mask_1, mask_2… — new slots appear as you connect them."

    def execute(self, **kwargs):
        frames = []
        i = 1
        while f"mask_{i}" in kwargs:
            v = kwargs[f"mask_{i}"]
            if v is not None:
                for frame in v:
                    frames.append(frame)
            i += 1
        if not frames:
            return (torch.zeros(1, 64, 64),)
        h, w = frames[0].shape[0], frames[0].shape[1]
        adjusted = []
        for f in frames:
            if f.shape[0] == h and f.shape[1] == w:
                adjusted.append(f)
            else:
                # Scale-then-center-crop to match mask 1 (ComfyUI's default batch behavior).
                bchw = f.unsqueeze(0).unsqueeze(0)
                bchw = comfy.utils.common_upscale(bchw, w, h, "bilinear", "center")
                adjusted.append(bchw.squeeze(0).squeeze(0))
        return (torch.stack(adjusted),)


NODE_CLASS_MAPPINGS = {"WtlMaskBatch": WtlMaskBatchC}
NODE_DISPLAY_NAME_MAPPINGS = {"WtlMaskBatch": "Mask Batch"}
