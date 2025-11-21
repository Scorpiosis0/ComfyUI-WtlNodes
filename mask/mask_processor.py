import torch
import cv2
import numpy as np

class MaskProcessorC:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "dilate_erode": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "WtlNodes/mask"

    def process_mask(self, mask, dilate_erode, feather):
        # Convert tensor to numpy
        mask_np = mask.cpu().numpy()
        batch_size = mask_np.shape[0]
        results = []
        
        for b in range(batch_size):
            # Get single mask and convert to uint8 (0-255)
            m = mask_np[b]
            m_uint8 = (m * 255).astype(np.uint8)
            
            # Step 1: Expand or Shrink (Morphological operations)
            if dilate_erode != 0:
                m_uint8 = self._expand_or_shrink(m_uint8, dilate_erode)
            
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
        # Create kernel for morphological operation
        kernel_size = abs(amount) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if amount > 0:
            # Expand (dilation)
            result = cv2.dilate(mask, kernel, iterations=1)
        else:
            # Shrink (erosion)
            result = cv2.erode(mask, kernel, iterations=1)
        
        return result
    
    def _feather_mask(self, mask, radius):
        # Gaussian blur for soft edges
        # Kernel size must be odd
        kernel_size = radius * 2 + 1
        
        # Apply gaussian blur
        blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        return blurred

NODE_CLASS_MAPPINGS = {"MaskProcessor": MaskProcessorC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskProcessor": "Mask Processor"}