class DepthPro:
    @classmethod

    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Positive values expand (grow) the mask, negative values shrink (erode) it"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_mask"
    CATEGORY = "depth"