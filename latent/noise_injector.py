import torch
import math

class LatentNoiseC:
    NOISE_TYPES = [
        "Gaussian",
        "White",
        "Perlin",
        "Simplex",
        "Worley",
        "Voronoi",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "noise_type": (cls.NOISE_TYPES,),
                "noise_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                }),
                "scale": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Scale/frequency of procedural noise patterns"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("Latent",)
    FUNCTION = "add_noise"
    CATEGORY = "WtlNodes/latent"

    def add_noise(self, latent, seed, noise_type, noise_strength, scale):
        samples = latent["samples"]
        batch, channels, height, width = samples.shape
        
        # Create generator for seed
        generator = torch.Generator(device=samples.device).manual_seed(seed)
        
        # Generate noise based on type
        if noise_type == "Gaussian":
            noise = self._gaussian_noise(samples.shape, samples.dtype, samples.device, generator)
        elif noise_type == "White":
            noise = self._white_noise(samples.shape, samples.dtype, samples.device, generator)
        elif noise_type == "Perlin":
            noise = self._perlin_noise(batch, channels, height, width, samples.dtype, samples.device, generator, scale)
        elif noise_type == "Simplex":
            noise = self._simplex_noise(batch, channels, height, width, samples.dtype, samples.device, generator, scale)
        elif noise_type == "Worley":
            noise = self._worley_noise(batch, channels, height, width, samples.dtype, samples.device, generator, scale)
        elif noise_type == "Voronoi":
            noise = self._voronoi_noise(batch, channels, height, width, samples.dtype, samples.device, generator, scale)
        else:
            noise = self._gaussian_noise(samples.shape, samples.dtype, samples.device, generator)
        
        # Add noise to latent
        noisy_samples = samples + (noise * noise_strength)
        
        return ({"samples": noisy_samples},)
    
    def _gaussian_noise(self, shape, dtype, device, generator):
        """Standard gaussian/normal distribution noise"""
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)
    
    def _white_noise(self, shape, dtype, device, generator):
        """Uniform random noise"""
        return (torch.rand(shape, dtype=dtype, device=device, generator=generator) - 0.5) * 2.0
    
    def _perlin_noise(self, batch, channels, height, width, dtype, device, generator, scale):
        """Perlin noise - smooth gradient noise"""
        noise = torch.zeros((batch, channels, height, width), dtype=dtype, device=device)
        
        for b in range(batch):
            for c in range(channels):
                # Generate random gradients
                grid_h = int(height / scale) + 2
                grid_w = int(width / scale) + 2
                
                gradients = torch.randn((grid_h, grid_w, 2), dtype=dtype, device=device, generator=generator)
                gradients = gradients / (torch.norm(gradients, dim=2, keepdim=True) + 1e-8)
                
                # Create coordinate grids
                y = torch.linspace(0, grid_h - 1, height, dtype=dtype, device=device)
                x = torch.linspace(0, grid_w - 1, width, dtype=dtype, device=device)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                # Get grid cell corners
                y0 = yy.long()
                x0 = xx.long()
                y1 = torch.clamp(y0 + 1, max=grid_h - 1)
                x1 = torch.clamp(x0 + 1, max=grid_w - 1)
                
                # Interpolation weights
                fy = yy - y0.float()
                fx = xx - x0.float()
                
                # Fade function for smooth interpolation
                fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
                fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
                
                # Get gradients at corners
                g00 = gradients[y0, x0]
                g01 = gradients[y0, x1]
                g10 = gradients[y1, x0]
                g11 = gradients[y1, x1]
                
                # Distance vectors
                d00 = torch.stack([xx - x0.float(), yy - y0.float()], dim=-1)
                d01 = torch.stack([xx - x1.float(), yy - y0.float()], dim=-1)
                d10 = torch.stack([xx - x0.float(), yy - y1.float()], dim=-1)
                d11 = torch.stack([xx - x1.float(), yy - y1.float()], dim=-1)
                
                # Dot products
                n00 = (g00 * d00).sum(dim=-1)
                n01 = (g01 * d01).sum(dim=-1)
                n10 = (g10 * d10).sum(dim=-1)
                n11 = (g11 * d11).sum(dim=-1)
                
                # Bilinear interpolation
                n0 = n00 * (1 - fx) + n01 * fx
                n1 = n10 * (1 - fx) + n11 * fx
                result = n0 * (1 - fy) + n1 * fy
                
                noise[b, c] = result
        
        return noise
    
    def _simplex_noise(self, batch, channels, height, width, dtype, device, generator, scale):
        """Simplex noise - improved Perlin with fewer artifacts (vectorized)"""
        noise = torch.zeros((batch, channels, height, width), dtype=dtype, device=device)
        
        F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        G2 = (3.0 - math.sqrt(3.0)) / 6.0
        
        for b in range(batch):
            for c in range(channels):
                # Generate permutation table
                perm_size = 512
                perm = torch.randperm(perm_size, generator=generator, device=device)
                perm = torch.cat([perm, perm])  # Repeat for wraparound
                
                # Gradient vectors
                grad3 = torch.tensor([
                    [1, 1], [-1, 1], [1, -1], [-1, -1],
                    [1, 0], [-1, 0], [0, 1], [0, -1],
                    [1, 1], [-1, 1], [1, -1], [-1, -1]
                ], dtype=dtype, device=device)
                
                # Create coordinate grids (vectorized!)
                y_coords = torch.arange(height, dtype=dtype, device=device).view(-1, 1) / scale
                x_coords = torch.arange(width, dtype=dtype, device=device).view(1, -1) / scale
                
                # Broadcast to full grid
                y_grid = y_coords.expand(height, width)
                x_grid = x_coords.expand(height, width)
                
                # Skew input space
                s = (x_grid + y_grid) * F2
                i_s = torch.floor(x_grid + s).long()
                j_s = torch.floor(y_grid + s).long()
                
                t = (i_s + j_s).float() * G2
                X0 = i_s.float() - t
                Y0 = j_s.float() - t
                x0 = x_grid - X0
                y0 = y_grid - Y0
                
                # Determine simplex (vectorized)
                i1 = (x0 > y0).long()
                j1 = (x0 <= y0).long()
                
                # Offsets for corners
                x1 = x0 - i1.float() + G2
                y1 = y0 - j1.float() + G2
                x2 = x0 - 1.0 + 2.0 * G2
                y2 = y0 - 1.0 + 2.0 * G2
                
                # Hash coordinates (wrap to 0-255)
                ii = i_s & 255
                jj = j_s & 255
                
                # Get gradient indices
                gi0 = perm[ii + perm[jj]] % 12
                gi1 = perm[ii + i1 + perm[jj + j1]] % 12
                gi2 = perm[ii + 1 + perm[jj + 1]] % 12
                
                # Calculate contribution from corners (vectorized)
                t0 = 0.5 - x0*x0 - y0*y0
                t0 = torch.clamp(t0, min=0.0)
                t0_sq = t0 * t0
                n0 = t0_sq * t0_sq * (grad3[gi0, 0] * x0 + grad3[gi0, 1] * y0)
                
                t1 = 0.5 - x1*x1 - y1*y1
                t1 = torch.clamp(t1, min=0.0)
                t1_sq = t1 * t1
                n1 = t1_sq * t1_sq * (grad3[gi1, 0] * x1 + grad3[gi1, 1] * y1)
                
                t2 = 0.5 - x2*x2 - y2*y2
                t2 = torch.clamp(t2, min=0.0)
                t2_sq = t2 * t2
                n2 = t2_sq * t2_sq * (grad3[gi2, 0] * x2 + grad3[gi2, 1] * y2)
                
                noise[b, c] = 70.0 * (n0 + n1 + n2)
        
        return noise
    
    def _worley_noise(self, batch, channels, height, width, dtype, device, generator, scale):
        """Worley/Cellular noise - based on distance to random points"""
        noise = torch.zeros((batch, channels, height, width), dtype=dtype, device=device)
        
        for b in range(batch):
            for c in range(channels):
                # Generate random feature points
                num_points = int(scale * 2)
                points_y = torch.rand(num_points, dtype=dtype, device=device, generator=generator) * height
                points_x = torch.rand(num_points, dtype=dtype, device=device, generator=generator) * width
                
                # Create coordinate grid
                y = torch.arange(height, dtype=dtype, device=device).view(-1, 1)
                x = torch.arange(width, dtype=dtype, device=device).view(1, -1)
                
                # Calculate distances to all points
                min_dist = torch.full((height, width), float('inf'), dtype=dtype, device=device)
                
                for p in range(num_points):
                    dist = torch.sqrt((y - points_y[p])**2 + (x - points_x[p])**2)
                    min_dist = torch.min(min_dist, dist)
                
                # Normalize to [-1, 1] based on expected cell size
                # Average cell size is roughly sqrt(total_area / num_points)
                expected_max_dist = math.sqrt((height * width) / num_points) if num_points > 0 else 1.0
                noise[b, c] = torch.clamp((min_dist / expected_max_dist) * 2.0 - 1.0, -1.0, 1.0)
        
        return noise
    
    def _voronoi_noise(self, batch, channels, height, width, dtype, device, generator, scale):
        """Voronoi noise - regions based on nearest point"""
        noise = torch.zeros((batch, channels, height, width), dtype=dtype, device=device)
        
        for b in range(batch):
            for c in range(channels):
                # Generate random feature points with associated values
                num_points = int(scale * 2)
                points_y = torch.rand(num_points, dtype=dtype, device=device, generator=generator) * height
                points_x = torch.rand(num_points, dtype=dtype, device=device, generator=generator) * width
                values = torch.randn(num_points, dtype=dtype, device=device, generator=generator)
                
                # Create coordinate grid
                y = torch.arange(height, dtype=dtype, device=device).view(-1, 1)
                x = torch.arange(width, dtype=dtype, device=device).view(1, -1)
                
                # Find nearest point for each pixel
                min_dist = torch.full((height, width), float('inf'), dtype=dtype, device=device)
                nearest_val = torch.zeros((height, width), dtype=dtype, device=device)
                
                for p in range(num_points):
                    dist = torch.sqrt((y - points_y[p])**2 + (x - points_x[p])**2)
                    mask = dist < min_dist
                    min_dist = torch.where(mask, dist, min_dist)
                    nearest_val = torch.where(mask, values[p], nearest_val)
                
                noise[b, c] = nearest_val
        
        return noise


NODE_CLASS_MAPPINGS = {"LatentNoise": LatentNoiseC}
NODE_DISPLAY_NAME_MAPPINGS = {"LatentNoise": "Add Noise to Latent (WIP)"}