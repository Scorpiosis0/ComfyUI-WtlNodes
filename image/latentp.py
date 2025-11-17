import torch

class AdvancedEmptyLatent:
    # Define aspect ratios and their corresponding resolutions
    RATIOS = {
        "1:1": "1:1",
        "3:2": "3:2",
        "4:3": "4:3",
        "5:3": "5:3",
        "16:9": "16:9",
        "16:10": "16:10",
        "21:9": "21:9",
        "32:9": "32:9",
    }
    
    # Resolutions for each ratio (in pixels, will be divided by 8 for latent)
    # Format: {ratio_key: [resolutions_list]}
    RESOLUTIONS = {
        "1:1": [
            "512x512",
            "768x768",
            "1024x1024",
            "1536x1536",
            "2048x2048",
        ],
        "3:2": [
            "960x640",
            "1152x768",
            "1344x896",
            "1536x1024",
            "1728x1152",
            "1920x1280",
        ],
        "4:3": [
            "768x576",
            "1024x768",
            "1280x960",
            "1536x1152",
            "1792x1344",
        ],
        "5:3": [
            "960x576",
            "1280x768",
            "1600x960",
            "1920x1152",
        ],
        "16:9": [
            "1024x576",
            "2048x1152",
        ],
        "16:10": [
            "1024x640",
            "2048x1280",
        ],
        "21:9": [
            "1344x576",
            "2688x1152",
        ],
        "32:9": [
            "2048x576",
            "4096x1152",
        ],
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Flatten all resolutions into a single list for validation
        all_resolutions = list(set([res for res_list in cls.RESOLUTIONS.values() for res in res_list]))
        
        return {
            "required": {
                "use_ratio": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Ratio Mode",
                    "label_off": "Manual Mode"
                }),
                "portrait_landscape": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Potrait",
                    "label_off": "Landscape"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "ratio": (list(cls.RATIOS.values()),),
                "resolution": (all_resolutions,),  # All possible resolutions
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 256
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("Latent", "Latent Width", "Latent Height", "Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "WtlNodes/latent"

    def generate(self, use_ratio, portrait_landscape, width, height, ratio, resolution, batch_size):
        if use_ratio:
            if portrait_landscape:
                # Parse resolution string (e.g., "1024x1024")
                h, w = map(int, resolution.split('x'))
            else:
                w, h = map(int, resolution.split('x'))
        else:
            # Use manual width/height
            w, h = width, height
        
        # Convert pixel dimensions to latent dimensions (SDXL uses 8x compression)
        latent_width = w // 8
        latent_height = h // 8
        
        # Create empty latent (SDXL uses 4 channels)
        latent = torch.zeros([batch_size, 4, latent_height, latent_width])
        
        return ({"samples": latent}, latent_width, latent_height, w, h)

NODE_CLASS_MAPPINGS = {"AdvancedEmptyLatent": AdvancedEmptyLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"AdvancedEmptyLatent": "Empty Latent (Advanced)"}