import torch

class saturation:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
                "saturation":("FLOAT",{
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                })
            }}
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="saturation"
    CATEGORY = "WtlNodes/image"

    def saturation(self,image,saturation):

        grayscale = (image[..., 0:1] * 0.299 + image[..., 1:2] * 0.587 + image[..., 2:3] * 0.114)
        grayscale_rgb = grayscale.repeat(1, 1, 1, 3)

        result = grayscale_rgb + (image - grayscale_rgb) * (1+saturation/100)
        result = torch.clamp(result, 0.0, 1.0)

        return(result,)
    
NODE_CLASS_MAPPINGS = {"Saturation": saturation}
NODE_DISPLAY_NAME_MAPPINGS = {"Saturation": "Saturation"}