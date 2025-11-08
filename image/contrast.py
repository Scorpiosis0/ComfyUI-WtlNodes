import torch

class contrast:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
                "contrast":("FLOAT",{
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                })
            }}
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="contrast"
    CATEGORY = "WtlNodes/image"
    
    def contrast(self, image, contrast):

        pivot = 0.5
        result = pivot + (image - pivot) * (1+contrast/100)
        result = torch.clamp(result, 0.0, 1.0)
        
        return (result,)
    
NODE_CLASS_MAPPINGS = {
    "Contrast": contrast
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Contrast": "Contrast"
}