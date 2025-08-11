'''
This script prepares the data for training the multi-input model.
It generates signatures for both the true and false image datasets and saves them to a JSON file.
'''

import os
import json
import cv2
import numpy as np
from test_verify import compute_signature

# Get the root directory of the project (one level up from src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

TRUE_DATA_DIR = os.path.join(ROOT_DIR, "True_data", "secured")
FALSE_DATA_DIR = os.path.join(ROOT_DIR, "False_data", "simulated_copies")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
OUTPUT_JSON = os.path.join(CONFIG_DIR, "training_data.json")

def process_directory(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(directory, filename)
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not read image {file_path}")
                    continue
                
                # All images are based on the same parameters for generation
                # so we can use the default parameters for compute_signature
                signature = compute_signature(image)
                
                # Convert numpy types to native python types for json serialization
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
    print("Starting data preparation...")
    
    print(f"Processing TRUE images from: {TRUE_DATA_DIR}")
    true_data = process_directory(TRUE_DATA_DIR, 1)
    
    print(f"Processing FALSE images from: {FALSE_DATA_DIR}")
    false_data = process_directory(FALSE_DATA_DIR, 0)
    
    all_data = true_data + false_data
    
    output = {"data": all_data}
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Successfully created {OUTPUT_JSON} with {len(all_data)} entries.")

if __name__ == "__main__":
    main()
