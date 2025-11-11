import torch
import numpy as np
import cv2

def live_preview_resizer(image):
    # Resize to 1MP
    image = image[0].cpu().numpy()  # shape: H,W,C
    H, W, C = image.shape
    target_pixels = 1024*1024
    scale = (target_pixels / (H * W)) ** 0.5
    new_H = max(1, int(H * scale))
    new_W = max(1, int(W * scale))
    image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
    # Convert back to tensor
    image = torch.from_numpy(image).unsqueeze(0).to(image.device)
    image = torch.clamp(image, 0.0, 1.0)
    return(image)