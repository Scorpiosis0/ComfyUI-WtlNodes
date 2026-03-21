"""
Film Artifact Cache Generator

Generates pre-computed film artifacts (scratches and hairs) and saves them to a .pkl file.
Run this once to create the cache, then the Film Artifacts node will load it.

Usage: python generate_artifact_cache.py
"""

import torch
import pickle
import math
from pathlib import Path

def generate_scratch_patterns(num_scratches=2000, max_points=2000, seed=42):
    """
    Generate scratch opacity patterns.
    
    Returns:
        dict with:
            - patterns: [num_scratches, max_points] normalized opacity values
            - lengths: [num_scratches] actual number of valid points per scratch
            - widths: [num_scratches] width preference (1-10 scale)
    """
    print(f"Generating {num_scratches} scratch patterns...")
    
    generator = torch.Generator().manual_seed(seed)
    
    patterns = torch.zeros(num_scratches, max_points)
    lengths = torch.zeros(num_scratches, dtype=torch.long)
    widths = torch.zeros(num_scratches)
    
    for i in range(num_scratches):
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_scratches} scratches...")
        
        # Random length (10% to 100% of max)
        length = int(torch.rand(1, generator=generator).item() * 0.9 * max_points) + int(0.1 * max_points)
        lengths[i] = length
        
        # Random width preference (exponential distribution for thin scratches)
        width_rand = torch.rand(1, generator=generator).item()
        widths[i] = 1 + (width_rand ** 3) * 9  # 1-10, heavily favoring 1-2
        
        # Generate opacity profile
        t = torch.arange(length, dtype=torch.float32)
        
        # Base intensity
        base_intensity = 0.05 + torch.rand(1, generator=generator).item() * 0.30
        
        # Fade in/out
        fade_in_len = min(20, length // 4)
        fade_out_len = min(20, length // 4)
        fade = torch.ones(length)
        if fade_in_len > 0:
            fade[:fade_in_len] = torch.linspace(0, 1, fade_in_len)
        if fade_out_len > 0:
            fade[-fade_out_len:] = torch.linspace(1, 0, fade_out_len)
        
        # Organic variation (sine waves)
        variation = torch.zeros(length)
        num_waves = 3
        for w in range(num_waves):
            freq = 0.05 + torch.rand(1, generator=generator).item() * 0.15
            phase = torch.rand(1, generator=generator).item() * 2 * math.pi
            variation += torch.sin(t * freq + phase)
        variation = (variation / num_waves) * 0.5 + 0.5
        
        # Combine
        opacity = base_intensity * variation * fade
        
        # Add random gaps
        if torch.rand(1, generator=generator).item() < 0.4:
            num_gaps = torch.randint(1, 4, (1,), generator=generator).item()
            for _ in range(num_gaps):
                gap_pos = torch.randint(0, length, (1,), generator=generator).item()
                gap_size = torch.randint(2, 8, (1,), generator=generator).item()
                gap_start = max(0, gap_pos - gap_size // 2)
                gap_end = min(length, gap_pos + gap_size // 2)
                opacity[gap_start:gap_end] *= torch.rand(1, generator=generator).item() * 0.3
        
        # Store
        patterns[i, :length] = opacity
    
    return {
        'patterns': patterns,
        'lengths': lengths,
        'widths': widths
    }

def generate_hair_shapes(num_hairs=5000, points_per_hair=100, seed=42):
    """
    Generate hair curve shapes in normalized [0, 1] coordinate space.
    
    Returns:
        dict with:
            - shapes: [num_hairs, points_per_hair, 2] normalized (x, y) coordinates
            - thicknesses: [num_hairs] thickness values (2.0-4.5 range)
            - intensities: [num_hairs] opacity values (0.2-0.6 range)
    """
    print(f"Generating {num_hairs} hair shapes...")
    
    generator = torch.Generator().manual_seed(seed + 1000)
    
    shapes = torch.zeros(num_hairs, points_per_hair, 2)
    thicknesses = torch.zeros(num_hairs)
    intensities = torch.zeros(num_hairs)
    
    for i in range(num_hairs):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_hairs} hairs...")
        
        # Random parameters
        angle = torch.rand(1, generator=generator).item() * 2 * math.pi
        curve_amplitude = (torch.rand(1, generator=generator).item() - 0.5) * 0.3  # Normalized to 0-1 space
        curve_frequency = 1.5 + torch.rand(1, generator=generator).item() * 1.5
        
        thicknesses[i] = 2.0 + torch.rand(1, generator=generator).item() * 2.5
        intensities[i] = 0.2 + torch.rand(1, generator=generator).item() * 0.4
        
        # Generate curve in normalized space [0, 1]
        t = torch.linspace(0, 1, points_per_hair)
        
        # Start at random position in normalized space
        start_x = torch.rand(1, generator=generator).item()
        start_y = torch.rand(1, generator=generator).item()
        
        # Base direction (normalized length is always 0 to 1)
        base_x = start_x + torch.cos(torch.tensor(angle)) * t
        base_y = start_y + torch.sin(torch.tensor(angle)) * t
        
        # Add curve
        perpendicular_angle = angle + math.pi / 2
        curve_offset = torch.sin(t * math.pi * curve_frequency) * curve_amplitude * t
        
        final_x = base_x + math.cos(perpendicular_angle) * curve_offset
        final_y = base_y + math.sin(perpendicular_angle) * curve_offset
        
        # Store (keeping values that may go outside [0,1] - we'll wrap/clamp at runtime)
        shapes[i, :, 0] = final_x
        shapes[i, :, 1] = final_y
    
    return {
        'shapes': shapes,
        'thicknesses': thicknesses,
        'intensities': intensities
    }

def main():
    print("=" * 60)
    print("Film Artifact Cache Generator")
    print("=" * 60)
    
    # Generate artifacts
    scratches = generate_scratch_patterns(num_scratches=2000, max_points=2000, seed=42)
    hairs = generate_hair_shapes(num_hairs=5000, points_per_hair=100, seed=42)
    
    # Package into cache
    cache = {
        'version': '1.0',
        'scratches': scratches,
        'hairs': hairs,
    }
    
    # Save to file
    output_path = Path(__file__).parent / 'film_artifacts_cache.pkl'
    print(f"\nSaving cache to: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Print stats
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Cache generated successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Scratches: {scratches['patterns'].shape[0]}")
    print(f"  Hairs: {hairs['shapes'].shape[0]}")
    print(f"\nYou can now use the Film Artifacts node - it will load this cache automatically.")
    print("=" * 60)

if __name__ == '__main__':
    main()