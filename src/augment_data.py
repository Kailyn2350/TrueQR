import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import glob
from tqdm import tqdm
import sys

# --- Configuration ---
# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directories (relative to the script's location)
TRUE_DATA_PATHS = [
    os.path.join(BASE_DIR, "True"),
    os.path.join(BASE_DIR, "grid_output")
]
FALSE_DATA_PATH = os.path.join(BASE_DIR, "grid_output_false")

# Output directory (relative to the script's location)
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "augmented_data")
AUGMENTED_TRUE_DIR = os.path.join(OUTPUT_BASE_DIR, "true")
AUGMENTED_FALSE_DIR = os.path.join(OUTPUT_BASE_DIR, "false")

# Augmentation settings
NUM_AUGMENTATIONS_PER_IMAGE = 10
BRIGHTNESS_RANGE = (0.7, 1.3)  # Min/max brightness factor
NOISE_INTENSITY = 25           # Max noise intensity (0-255)

# --- Helper Functions ---
def add_noise(image, intensity):
    """Adds random Gaussian noise to a PIL image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.randn(*img_array.shape) * intensity
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img_array)

def adjust_brightness(image, factor):
    """Adjusts the brightness of a PIL image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def find_image_files(paths):
    """Finds all image files (png, jpg, jpeg) in a list of directories."""
    image_files = []
    for path in paths:
        if not os.path.isdir(path):
            print(f"Warning: Directory not found - {path}")
            continue
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            # Using recursive glob to find files in subdirectories
            image_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
    return list(set(image_files)) # Return unique file paths

def process_and_augment_images(source_paths, target_dir, label):
    """
    Processes a list of source directories, augments the images,
    and saves them to the target directory.
    """
    print(f"\n--- Processing {label} data ---")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    image_files = find_image_files(source_paths)
    if not image_files:
        print(f"No images found for '{label}' data. Please check the paths.")
        return
        
    print(f"Found {len(image_files)} images for augmentation.")

    for image_path in tqdm(image_files, desc=f"Augmenting {label} images"):
        try:
            original_image = Image.open(image_path).convert("RGB")
            # Sanitize filename to handle various characters
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            sanitized_filename = "".join(c for c in base_filename if c.isalnum() or c in ('-', '_')).rstrip()

            for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                augmented_image = original_image.copy() # Use copy() to avoid modifying the original
                
                # Apply a random combination of augmentations
                # 1. Adjust Brightness
                brightness_factor = random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
                augmented_image = adjust_brightness(augmented_image, brightness_factor)

                # 2. Add Noise
                if random.random() > 0.3: # Apply noise 70% of the time
                    noise_level = random.uniform(5, NOISE_INTENSITY)
                    augmented_image = add_noise(augmented_image, noise_level)

                # Save the augmented image
                output_filename = f"{sanitized_filename}_aug_{i+1}.jpg"
                output_path = os.path.join(target_dir, output_filename)
                augmented_image.save(output_path, "JPEG", quality=95)

        except Exception as e:
            print(f"Could not process {image_path}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting data augmentation process...")
    
    # Create the base output directory
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)

    # Process TRUE data
    process_and_augment_images(TRUE_DATA_PATHS, AUGMENTED_TRUE_DIR, "True")

    # Process FALSE data
    process_and_augment_images([FALSE_DATA_PATH], AUGMENTED_FALSE_DIR, "False")

    print("\nData augmentation complete!")
    print(f"Augmented 'true' data saved in: {AUGMENTED_TRUE_DIR}")
    print(f"Augmented 'false' data saved in: {AUGMENTED_FALSE_DIR}")
