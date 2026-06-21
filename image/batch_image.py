import torch
import comfy.utils


class WtlImageBatchC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("batch (crops to image 1 size)",)
    FUNCTION = "execute"
    CATEGORY = "WtlNodes/image"
    DESCRIPTION = "Outputs multiple images as a single stacked IMAGE tensor. All frames are cropped to image_1's dimensions. Connect image_1, image_2… — new slots appear as you connect them."

    def execute(self, **kwargs):
        frames = []
        i = 1
        while f"image_{i}" in kwargs:
            v = kwargs[f"image_{i}"]
            if v is not None:
                for frame in v:
                    frames.append(frame)
            i += 1
        if not frames:
            return (torch.zeros(1, 64, 64, 3),)
        h, w = frames[0].shape[0], frames[0].shape[1]
        adjusted = []
        for f in frames:
            if f.shape[0] == h and f.shape[1] == w:
                adjusted.append(f)
            else:
                # Scale-then-center-crop to match frame 1 (ComfyUI's default batch behavior).
                bchw = f.movedim(-1, 0).unsqueeze(0)
                bchw = comfy.utils.common_upscale(bchw, w, h, "bilinear", "center")
                adjusted.append(bchw.squeeze(0).movedim(0, -1))
        return (torch.stack(adjusted),)


NODE_CLASS_MAPPINGS = {"WtlImageBatch": WtlImageBatchC}
NODE_DISPLAY_NAME_MAPPINGS = {"WtlImageBatch": "Image Batch"}
