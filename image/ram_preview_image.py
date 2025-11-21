import io
import base64
import numpy as np
from PIL import Image

class RAMPreviewImageC:
    def __init__(self):
        self.compress_level = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to preview."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_images"
    OUTPUT_NODE = True
    CATEGORY = "WtlNodes/experimental"
    DESCRIPTION = "Previews images using RAM storage only (no disk I/O)."

    def preview_images(self, images):
        previews = []
        
        for image in images:
            # Convert tensor to PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Save to RAM buffer
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", compress_level=self.compress_level)
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            previews.append(img_base64)
        
        return {
            "ui": {
                "ram_preview": previews  # Custom key for our extension
            }
        }
    
NODE_CLASS_MAPPINGS = {"RAMPreviewImage": RAMPreviewImageC}
NODE_DISPLAY_NAME_MAPPINGS = {"RAMPreviewImage": "RAM Preview Image"}