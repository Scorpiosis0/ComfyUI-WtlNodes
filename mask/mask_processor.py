import torch
import cv2
import numpy as np

class MaskProcessor:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Positive values expand (grow) the mask, negative values shrink (erode) it"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Softens the edges of the mask (blur radius)"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "mask"

    def process_mask(self, mask, expand, feather):
        # Convert tensor to numpy
        # ComfyUI masks are [B, H, W] format
        mask_np = mask.cpu().numpy()
        batch_size = mask_np.shape[0]
        results = []
        
        for b in range(batch_size):
            # Get single mask and convert to uint8 (0-255)
            m = mask_np[b]
            m_uint8 = (m * 255).astype(np.uint8)
            
            # Step 1: Expand or Shrink (Morphological operations)
            if expand != 0:
                m_uint8 = self._expand_or_shrink(m_uint8, expand)
            
            # Step 2: Feather (Gaussian blur)
            if feather > 0:
                m_uint8 = self._feather_mask(m_uint8, feather)
            
            # Convert back to float (0-1)
            m_float = m_uint8.astype(np.float32) / 255.0
            results.append(m_float)
        
        # Convert back to torch tensor
        output = torch.from_numpy(np.stack(results)).float()
        return (output,)
    
    def _expand_or_shrink(self, mask, amount):
        """
        Expand (dilate) or shrink (erode) the mask
        Positive amount = expand/grow
        Negative amount = shrink/erode
        """        
        # Create kernel for morphological operation
        # Larger kernel = more aggressive expansion/shrinking
        kernel_size = abs(amount) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if amount > 0:
            # Expand (dilation) - grows white areas
            result = cv2.dilate(mask, kernel, iterations=1)
        else:
            # Shrink (erosion) - shrinks white areas
            result = cv2.erode(mask, kernel, iterations=1)
        
        return result
    
    def _feather_mask(self, mask, radius):
        """
        Feather (blur) the mask edges for smooth transitions
        """
        # Gaussian blur for soft edges
        # Kernel size must be odd
        kernel_size = radius * 2 + 1
        
        # Apply gaussian blur
        blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        return blurred

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"MaskProcessor": MaskProcessor}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskProcessor": "Mask Processor"}

"""
USAGE NOTES:

1. EXPAND/SHRINK:
   - Positive values (1-100): Expands/grows the mask outward
   - Negative values (-1 to -100): Shrinks/erodes the mask inward
   - Uses morphological operations with ellipse kernel
   - Works on any mask shape, not just rectangles!

2. FEATHER:
   - Softens the edges of the mask
   - Higher values = softer/more blurred edges
   - Creates smooth transitions between masked and unmasked areas
   - Useful for blending effects

3. ORDER OF OPERATIONS:
   - First: Expand/Shrink is applied
   - Second: Feather is applied
   - This order gives the best results for most use cases

4. COMMON USE CASES:
   - Expand mask before feathering for soft, large selections
   - Shrink mask to remove edge artifacts
   - Feather only for smooth blending
   - Combine all three for precise control

5. TIPS:
   - Start with small values and increase gradually
   - Expand + Feather combo is great for soft selections
   - Shrink can help clean up noisy masks
   - Feather radius of 5-10 usually looks natural
"""