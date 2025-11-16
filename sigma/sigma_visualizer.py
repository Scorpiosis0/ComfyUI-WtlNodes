import torch
import numpy as np
from io import BytesIO
from PIL import Image

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

NODENODE_CLASS_MAPPINGS = {"VisualizeSigmas": VisualizeSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"VisualizeSigmas": "Visualize Sigma Schedule"}