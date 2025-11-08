import torch

class brightness:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
                "brightness":("FLOAT",{
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.1,
                })
            }}
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION="brightness"
    CATEGORY = "WtlNodes/image"
    
    def brightness(self,image,brightness):

        result = image * (1+brightness/100)
        result = torch.clamp(result, 0.0, 1.0)

        return(result,)
    
NODE_CLASS_MAPPINGS = {
    "Brightness": brightness
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Brightness": "Brightness"
}