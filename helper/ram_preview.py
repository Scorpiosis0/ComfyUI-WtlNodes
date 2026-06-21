import numpy as np
import base64
import io
from PIL import Image
import server
import math

def _send_ram_preview(image_tensor, unique_id, resize=True):
    """Send RAM preview via websocket (no disk I/O)."""
    try:
        images_base64 = []

        # Accepts either a (B, H, W, C) tensor or a plain list of (H, W, C) tensors.
        for frame in image_tensor:
            i = 255. * frame.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if resize:
                width, height = img.size
                current_pixels = width * height
                if current_pixels > 1_000_000:
                    scale = math.sqrt(1_000_000 / current_pixels)
                    img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG", compress_level=0)
            buffer.seek(0)
            images_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

        if hasattr(server.PromptServer, "instance"):
            server.PromptServer.instance.send_sync(
                "executed",
                {
                    "node": unique_id,
                    "output": {
                        "ram_preview": images_base64
                    },
                    "prompt_id": None
                }
            )
    except Exception as e:
        print(f"[RAM Preview] Error: {e}")