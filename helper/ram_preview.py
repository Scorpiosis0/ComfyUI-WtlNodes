import numpy as np
import base64
import io
from PIL import Image
import server
import math

def _send_ram_preview(image_tensor, unique_id):
    """Send RAM preview via websocket (no disk I/O)."""
    try:
        # Convert to PIL Image
        i = 255. * image_tensor[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Resize to 1MP (1 megapixel = 1,000,000 pixels)
        width, height = img.size
        current_pixels = width * height
        
        if current_pixels > 1000000:
            # Calculate scale factor to get to 1MP
            scale = math.sqrt(1000000 / current_pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to buffer with NO compression for speed
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=0)
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Send preview via websocket
        if hasattr(server.PromptServer, "instance"):
            server.PromptServer.instance.send_sync(
                "executed",
                {
                    "node": unique_id,
                    "output": {
                        "ram_preview": [img_base64]
                    },
                    "prompt_id": None
                }
            )
    except Exception as e:
        print(f"[RAM Preview] Error: {e}")