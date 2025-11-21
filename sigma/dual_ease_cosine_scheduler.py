import torch
import numpy as np

# Dual ease version with top and bottom control
class DualEaseCosineSchedulerC:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT",{
                    "default": 20,
                    "min": 1,
                    "max": 10000
                }),
                "sigma_max": ("FLOAT",{
                    "default": 14.6146,
                    "min": 0.0, "max": 1000.0,
                    "step": 0.01
                }),
                "sigma_min": ("FLOAT",{
                    "default": 0.0292,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.01
                }),
                "rho_start": ("FLOAT",{
                    "default": 5.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "rho_end": ("FLOAT",{
                    "default": 1.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "WtlNodes/sigma"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho_start, rho_end):
        # x from 0 to 1 - use steps not steps+1, then append zero
        x = np.linspace(0, 1, steps, dtype=np.float64)
        
        sigmas = []
        for xi in x:
            # Variable exponent that transitions from rho_start to rho_end
            rho = rho_start + (rho_end - rho_start) * xi
            
            # Apply the formula with variable rho
            cos_component = (1 + np.cos(np.pi * xi)) / 2
            eased = cos_component ** rho
            sigma = sigma_max - (sigma_max - sigma_min) * (1 - eased)
            sigmas.append(float(sigma))
        
        # Convert to numpy for safety checks
        sigmas = np.array(sigmas, dtype=np.float64)
        
        # Ensure strictly decreasing with minimum spacing (SDE fix)
        min_spacing = 1e-4
        for i in range(1, len(sigmas)):
            if sigmas[i] >= sigmas[i-1] - min_spacing:
                sigmas[i] = sigmas[i-1] - min_spacing
        
        # Append zero like Karras
        sigmas = np.append(sigmas, 0.0)
        
        sigmas = torch.FloatTensor(sigmas)
        return (sigmas,)

NODE_CLASS_MAPPINGS = {"DualEaseCosineScheduler": DualEaseCosineSchedulerC}
NODE_DISPLAY_NAME_MAPPINGS = {"DualEaseCosineScheduler": "Dual Ease Cosine Scheduler"}