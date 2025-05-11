import nrrd
from pathlib import Path
import matplotlib.pyplot as plt
import re

def extract_number(filename):
    """Extract the number from a filename like ref_123.nrrd or noisy_123.nrrd."""
    match = re.search(r'_(\d+)\.', filename.name)
    return int(match.group(1)) if match else 0

def load_image_pair(ref_path, noisy_path):
    """Load a pair of reference and noisy images."""
    ref_data, _ = nrrd.read(str(ref_path))
    noisy_data, _ = nrrd.read(str(noisy_path))
    return ref_data, noisy_data

def save_middle_slice(ref_data, noisy_data, output_path):
    """Save the middle slice of a pair of images as a PNG."""
    # Get the middle slice
    slice_idx = ref_data.shape[0] // 2
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Sample {output_path.stem.split('_')[1]} - Middle Slice", fontsize=16)
    
    # Plot the slices
    im1 = ax1.imshow(ref_data[slice_idx], cmap='gray')
    im2 = ax2.imshow(noisy_data[slice_idx], cmap='gray')
    
    ax1.set_title('Reference')
    ax2.set_title('Noisy')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def save_denoised_preview(ref_data, noisy_data, denoised_data, output_path):
    """Save the middle slice of reference, noisy, and denoised images as a PNG."""
    # Get the middle slice
    slice_idx = ref_data.shape[0] // 2
    
    # Create the figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Sample {output_path.stem.split('_')[2]} - Middle Slice with Denoising", fontsize=16)
    
    # Plot the slices
    im1 = ax1.imshow(ref_data[slice_idx], cmap='gray')
    im2 = ax2.imshow(noisy_data[slice_idx], cmap='gray')
    im3 = ax3.imshow(denoised_data[slice_idx], cmap='gray')
    
    ax1.set_title('Reference')
    ax2.set_title('Noisy')
    ax3.set_title('Denoised')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.colorbar(im3, ax=ax3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Get all image pairs
    sample_dir = Path("sample")
    
    # Get all files and organize them by their numbers
    ref_files = {extract_number(f): f for f in sample_dir.glob("ref_*.nrrd")}
    noisy_files = {extract_number(f): f for f in sample_dir.glob("noisy_*.nrrd")}
    denoised_files = {extract_number(f): f for f in sample_dir.glob("denoised_*.nrrd")}
    
    # Get all unique sample numbers
    all_numbers = sorted(set(ref_files.keys()) | set(noisy_files.keys()) | set(denoised_files.keys()))
    
    # Create preview for each pair that has both ref and noisy
    for num in all_numbers:
        if num in ref_files and num in noisy_files:
            ref_path = ref_files[num]
            noisy_path = noisy_files[num]
            print(f"Processing {ref_path.name} and {noisy_path.name}")
            ref_data, noisy_data = load_image_pair(ref_path, noisy_path)
            
            # Create output path for the preview
            output_path = sample_dir / f"preview_{num}.png"
            
            # Save the middle slice
            save_middle_slice(ref_data, noisy_data, output_path)
            print(f"Saved preview to {output_path}")
            
            # If we also have a denoised version, create that preview
            if num in denoised_files:
                denoised_path = denoised_files[num]
                print(f"Processing denoised preview for {denoised_path.name}")
                denoised_data, _ = nrrd.read(str(denoised_path))
                
                # Create output path for the denoised preview
                output_path = sample_dir / f"preview_denoised_{num}.png"
                
                # Save the denoised preview
                save_denoised_preview(ref_data, noisy_data, denoised_data, output_path)
                print(f"Saved denoised preview to {output_path}")

if __name__ == "__main__":
    main() 