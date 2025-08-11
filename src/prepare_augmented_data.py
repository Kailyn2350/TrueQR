'''
This script prepares the AUGMENTED data for training the multi-input model.
It generates signatures for both the true and false image datasets from the augmented_data folder
and saves them to a JSON file.
'''

import os
import json
import cv2
import numpy as np
from test_verify import compute_signature

# Get the root directory of the project (one level up from src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

TRUE_DATA_DIR = os.path.join(ROOT_DIR, "augmented_data", "true")
FALSE_DATA_DIR = os.path.join(ROOT_DIR, "augmented_data", "false")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
OUTPUT_JSON = os.path.join(CONFIG_DIR, "augmented_training_data.json")

def process_directory(directory, label):
    data = []
    print(f"Processing directory: {directory}")
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found: {directory}")
        return data

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not read image {file_path}")
                    continue
                
                signature = compute_signature(image)
                
                for key, value in signature.items():
                    if isinstance(value, (np.integer, np.floating)):
                        signature[key] = value.item()

                data.append({
                    "file_path": file_path,
                    "signature": signature,
                    "label": label
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return data

def main():
    print("Starting augmented data preparation...")
    
    true_data = process_directory(TRUE_DATA_DIR, 1)
    false_data = process_directory(FALSE_DATA_DIR, 0)
    
    all_data = true_data + false_data
    
    output = {"data": all_data}
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Successfully created {OUTPUT_JSON} with {len(all_data)} entries.")

if __name__ == "__main__":
    main()
