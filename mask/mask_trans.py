import torch
import cv2
import numpy as np

class MaskTransformNode:
    
    INTERPOLATION_METHODS = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "resize_by": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Multiplier",
                    "label_off": "Absolute"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1
                }),
                "interpolation": (list(cls.INTERPOLATION_METHODS.keys()),{
                    "default": "nearest",
                }),
                "fit_mode": (["crop", "adjust", "fit"],),
                "rotate": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 0.1
                }),
                "translate_x": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1
                }),
                "translate_y": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1
                }),
                "zoom": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "transform"
    CATEGORY = "mask/transform"

    def transform(self, mask, resize_by, width, height, multiplier, interpolation, 
                  fit_mode, rotate, translate_x, translate_y, zoom):
        # Convert tensor to numpy
        mask_np = mask.cpu().numpy()
        batch_size = mask_np.shape[0]
        results = []
        
        interp_method = self.INTERPOLATION_METHODS[interpolation]
        
        for b in range(batch_size):
            mask = mask_np[b]
            orig_h, orig_w = mask.shape[:2]
            
            # Convert to uint8 for OpenCV processing
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Step 1: Apply zoom (before other transforms)
            if zoom != 1.0:
                mask_uint8 = self._apply_zoom(mask_uint8, zoom, interp_method)
            
            # Step 2: Apply rotation
            if rotate != 0.0:
                mask_uint8 = self._apply_rotation(mask_uint8, rotate, interp_method)
            
            # Step 3: Apply translation
            if translate_x != 0 or translate_y != 0:
                mask_uint8 = self._apply_translation(mask_uint8, translate_x, translate_y)
            
            # Step 4: Calculate target dimensions
            if resize_by:
                target_w = int(orig_w * multiplier)
                target_h = int(orig_h * multiplier)
            else:
                target_w = width
                target_h = height
            
            # Ensure dimensions are at least 1
            target_w = max(1, target_w)
            target_h = max(1, target_h)
            
            # Step 5: Apply resize with fit mode
            mask_uint8 = self._apply_resize(mask_uint8, target_w, target_h, 
                                           fit_mode, interp_method)
            
            # Convert back to float32
            mask_float = mask_uint8.astype(np.float32) / 255.0
            results.append(mask_float)
        
        # Convert back to torch tensor
        output = torch.from_numpy(np.stack(results)).float()
        return (output,)
    
    def _apply_zoom(self, mask, zoom, interp_method):
        """Apply zoom by scaling around center"""
        h, w = mask.shape[:2]
        
        # Calculate new dimensions
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        
        # Resize mask
        zoomed = cv2.resize(mask, (new_w, new_h), interpolation=interp_method)
        
        if zoom > 1.0:
            # Crop from center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            return zoomed[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad to original size
            result = np.zeros((h, w), dtype=mask.dtype)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result[start_y:start_y + new_h, start_x:start_x + new_w] = zoomed
            return result
    
    def _apply_rotation(self, mask, angle, interp_method):
        """Rotate mask around center"""
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(mask, rotation_matrix, (new_w, new_h),
                                 flags=interp_method, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        
        return rotated
    
    def _apply_translation(self, mask, tx, ty):
        """Translate mask"""
        h, w = mask.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        translated = cv2.warpAffine(mask, translation_matrix, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
        return translated
    
    def _apply_resize(self, mask, target_w, target_h, fit_mode, interp_method):
        """Apply resize with different fit modes"""
        h, w = mask.shape[:2]
        
        if fit_mode == "adjust":
            # Simply resize to target dimensions (may change aspect ratio)
            return cv2.resize(mask, (target_w, target_h), interpolation=interp_method)
        
        elif fit_mode == "crop":
            # Resize and crop to maintain aspect ratio
            aspect_ratio = w / h
            target_aspect = target_w / target_h
            
            if aspect_ratio > target_aspect:
                # mask is wider - fit height and crop width
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            else:
                # mask is taller - fit width and crop height
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            
            # Resize
            resized = cv2.resize(mask, (new_w, new_h), interpolation=interp_method)
            
            # Crop from center
            start_x = (new_w - target_w) // 2
            start_y = (new_h - target_h) // 2
            
            return resized[start_y:start_y + target_h, start_x:start_x + target_w]
        
        elif fit_mode == "fit":
            # Resize and add black bars to maintain aspect ratio
            aspect_ratio = w / h
            target_aspect = target_w / target_h
            
            if aspect_ratio > target_aspect:
                # mask is wider - fit width and add bars top/bottom
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                # mask is taller - fit height and add bars left/right
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            
            # Resize
            resized = cv2.resize(mask, (new_w, new_h), interpolation=interp_method)
            
            # Create canvas with black bars
            result = np.zeros((target_h, target_w, mask.shape[2]), dtype=mask.dtype)
            
            # Center the resized mask
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            
            result[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            
            return result
        
        return mask


NODE_CLASS_MAPPINGS = {
    "MaskTransformNode": MaskTransformNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskTransformNode": "Mask Transform (Advanced)"
}


"""
USAGE NOTES:

1. RESIZE MODES:
   - resize_by = False: Use absolute width/height values
   - resize_by = True: Use multiplier (width/height hidden, needs dynamic UI)

2. INTERPOLATION METHODS:
   - nearest: Fast, pixelated look
   - bilinear: Smooth, faster than bicubic
   - bicubic: High quality, good balance
   - lanczos: Highest quality, slower

3. FIT MODES:
   - crop: Maintains aspect ratio, crops excess (no black bars, may lose content)
   - adjust: Stretches/squashes to fit (changes aspect ratio)
   - fit: Maintains aspect ratio, adds black bars (letterboxing/pillarboxing)

4. TRANSFORM ORDER:
   Zoom → Rotation → Translation → Resize
   This order provides the most intuitive results

5. COORDINATES:
   - translate_x: Positive moves right, negative moves left
   - translate_y: Positive moves down, negative moves up
   - rotate: Positive is clockwise, negative is counter-clockwise

6. DYNAMIC UI:
   When implementing dynamic behavior:
   - Hide width/height when resize_by = True, show multiplier
   - Hide multiplier when resize_by = False, show width/height
"""