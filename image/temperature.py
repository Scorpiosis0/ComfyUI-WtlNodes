import torch
import math

class temperature:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "kelvin": ("INT", {
                    "default": 6500,    # Standard daylight
                    "min": 1000,        # Candle flame
                    "max": 12000,       # Clear blue sky
                    "step": 100,
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "temperature"
    CATEGORY = "WtlNodes/image"
    
    def temperature(self, image, kelvin):

        result = torch.clamp(result, 0.0, 1.0)
        
        return (result,)


# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "Temperature": temperature
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Temperature": "Color Temperature"
}