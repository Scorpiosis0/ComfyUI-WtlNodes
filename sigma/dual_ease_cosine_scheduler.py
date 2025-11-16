import torch
import numpy as np
from io import BytesIO
from PIL import Image

# Dual ease version with top and bottom control
class DualEaseCosineScheduler:
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
    CATEGORY = "sampling/custom"

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

# Visualizer node
class VisualizeSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "visualize"
    CATEGORY = "sampling/custom"

    def visualize(self, sigmas):
        sigmas_np = sigmas.cpu().numpy() if hasattr(sigmas, 'cpu') else np.array(sigmas)
        img = self.generate_plot(sigmas_np)
        
        import folder_paths
        import os
        
        output_dir = folder_paths.get_temp_directory()
        filename = "sigma_viz.png"
        file_path = os.path.join(output_dir, filename)
        img.save(file_path)
        
        return {"ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}}
    
    def generate_plot(self, sigmas):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            steps = list(range(len(sigmas)))
            ax.plot(steps, sigmas, 'b-', linewidth=3)
            ax.scatter(steps, sigmas, c='blue', s=40, alpha=0.6, zorder=5)
            
            ax.set_xlabel('Step', fontsize=14, fontweight='bold')
            ax.set_ylabel('Sigma', fontsize=14, fontweight='bold')
            ax.set_title('Sigma Schedule', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            sigma_max = sigmas[0]
            sigma_min = sigmas[-1]
            ax.axhline(y=sigma_max, color='red', linestyle='--', alpha=0.7, linewidth=2)
            if sigma_min > 0:
                ax.axhline(y=sigma_min, color='green', linestyle='--', alpha=0.7, linewidth=2)
            
            ax.text(0.02, 0.98, f'Max: {sigma_max:.4f}', transform=ax.transAxes, 
                   verticalalignment='top', fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax.text(0.02, 0.05, f'Min: {sigma_min:.4f}', transform=ax.transAxes, 
                   verticalalignment='bottom', fontsize=11, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax.text(0.98, 0.98, f'Steps: {len(sigmas)}', transform=ax.transAxes, 
                   ha='right', va='top', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_facecolor('#f0f0f0')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
            plt.close(fig)
            
            return img
        except Exception as e:
            print(f"Plot error: {e}")
            return Image.new('RGB', (800, 600), color='white')

NODE_CLASS_MAPPINGS = {"DualEaseCosineScheduler": DualEaseCosineScheduler}
NODE_DISPLAY_NAME_MAPPINGS = {"DualEaseCosineScheduler": "Dual Ease Cosine Scheduler"}