import numpy as np
from pathlib import Path
import nrrd

def append_csv(path, data):
    with open(path, "a") as f:
        f.write(data + "\n")

def create_sphere(size, center, radius):
    """Create a 3D sphere."""
    x, y, z = np.ogrid[:size, :size, :size]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    return (dist <= radius).astype(np.float32)

def add_gaussian_noise(image, mean=0, std=2e-11):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 2e-10)

def create_multiple_spheres(size, sphere_params):
    """Create an image with multiple spheres.
    
    Args:
        size: Size of the 3D volume
        sphere_params: List of tuples (center, radius) for each sphere
    """
    image = np.zeros((size, size, size), dtype=np.float32)
    for center, radius in sphere_params:
        sphere = create_sphere(size, center, radius)
        image = np.maximum(image, sphere)
    image = image * 1e-10
    return image

def main():
    # Create output directory
    output_dir = Path("sample")
    output_dir.mkdir(exist_ok=True)
    sample_csv = output_dir / "sample.csv"
    append_csv(sample_csv, "x,y,partition")
    
    # Image size
    size = 100
    
    # Define different sphere configurations
    sphere_configs = [
        # Single large sphere
        [((50, 50, 50), 30)],
        
        # Two overlapping spheres
        [((40, 40, 40), 25), ((60, 60, 60), 25)],
        
        # Three spheres of different sizes
        [((30, 30, 30), 15), ((50, 50, 50), 25), ((70, 70, 70), 20)],
        
        # Four small spheres
        [((30, 30, 50), 10), ((30, 70, 50), 10),
         ((70, 30, 50), 10), ((70, 70, 50), 10)],
        
        # Five spheres in a cross pattern
        [((50, 50, 50), 15),  # Center
         ((50, 30, 50), 10),  # Top
         ((50, 70, 50), 10),  # Bottom
         ((30, 50, 50), 10),  # Left
         ((70, 50, 50), 10)]  # Right
    ]
    
    # Generate and save images
    for i, sphere_params in enumerate(sphere_configs):
        # Create reference image
        ref_image = create_multiple_spheres(size, sphere_params)
        
        # Create noisy version
        noisy_image = add_gaussian_noise(ref_image)
        
        # Save reference image
        nrrd.write(str(output_dir / f"ref_{i+1}.nrrd"), ref_image)
        
        # Save noisy image
        nrrd.write(str(output_dir / f"noisy_{i+1}.nrrd"), noisy_image)
        
        # append pair to csv
        append_csv(sample_csv, f"noisy_{i+1}.nrrd,ref_{i+1}.nrrd,{i}")

        print(f"Created image pair {i+1} with {len(sphere_params)} spheres")

if __name__ == "__main__":
    main() 