import torch
import numpy as np
import base64
import io
from PIL import Image
import server
from ..helper.ram_preview import _send_ram_preview

class RAMImageCompareC:
    def __init__(self):
        self.compress_level = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "compare_mode": (["slide", "click"], {"default": "slide"}),
                "image_a": ("IMAGE", {"tooltip": "First image to compare"}),
                "image_b": ("IMAGE", {"tooltip": "Second image to compare"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = "WtlNodes/image"
    DESCRIPTION = "Compare two images with a slide or click reveal (RAM only, no disk I/O)."

    def compare_images(self, compare_mode="slide", image_a=None, image_b=None, unique_id=None):
        if not unique_id:
            return {}
            
        uid = str(unique_id)
        
        # Convert both images to base64
        images_base64 = []
        
        for img_tensor in [image_a, image_b]:
            if img_tensor is None:
                images_base64.append(None)
                continue
                
            try:
                i = 255. * img_tensor[0].cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", compress_level=0)
                buffer.seek(0)
                
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images_base64.append(img_base64)
            except Exception as e:
                print(f"[RAM Compare] Error processing image: {e}")
                images_base64.append(None)
        
        # Send both images in one message
        if hasattr(server.PromptServer, "instance"):
            server.PromptServer.instance.send_sync(
                "executed",
                {
                    "node": uid,
                    "output": {
                        "ram_preview": images_base64  # Send as array of 2 images
                    },
                    "prompt_id": None
                }
            )
        
        return {}
    
NODE_CLASS_MAPPINGS = {"RAMImageCompare": RAMImageCompareC}
NODE_DISPLAY_NAME_MAPPINGS = {"RAMImageCompare": "RAM Image Compare"}