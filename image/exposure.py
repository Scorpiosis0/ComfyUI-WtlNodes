import torch

class exposure:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
                "exposure":("FLOAT",{
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                })
            }}
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="exposure"
    CATEGORY = "WtlNodes/image"
    
    def exposure(self,image,exposure):

        result = image * (2**(exposure/100))
        result = torch.clamp(result, 0.0, 1.0)

        return(result,)
    
NODE_CLASS_MAPPINGS = {"Exposure": exposure}
NODE_DISPLAY_NAME_MAPPINGS = {"Exposure": "Exposure"}