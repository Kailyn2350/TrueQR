import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
import json
from test_verify import compute_signature

# --- Configuration ---
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

MODEL_PATH = os.path.join(ROOT_DIR, "results", "true_qr_classifier_augmented.keras")
NORM_PARAMS_PATH = os.path.join(ROOT_DIR, "results", "signature_normalization.json")
SECURED_DIR = os.path.join(ROOT_DIR, "True_data", "secured")
SIMULATED_COPIES_DIR = os.path.join(ROOT_DIR, "False_data", "simulated_copies")
AUGMENTED_TRUE_DIR = os.path.join(ROOT_DIR, "augmented_data", "true")
AUGMENTED_FALSE_DIR = os.path.join(ROOT_DIR, "augmented_data", "false")

IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['false', 'true']

# --- 1. Load Model and Normalization Parameters ---
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

print(f"Loading normalization parameters from: {NORM_PARAMS_PATH}")
with open(NORM_PARAMS_PATH, 'r') as f:
    norm_params = json.load(f)
mean = np.array(norm_params['mean'])
std = np.array(norm_params['std'])
print("Normalization parameters loaded.")

# --- 2. Prediction Function ---
def predict_image(image_path):
    """Loads an image, computes signature, preprocesses, and returns the model's prediction."""
    # Load image for model input
    img_for_model = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img_for_model)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Load image for signature calculation
    img_for_sig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_for_sig is None:
        raise ValueError(f"Could not read image for signature: {image_path}")

    # Compute and normalize signature
    signature = compute_signature(img_for_sig)
    sig_vector = np.array([[signature['phash'], signature['hf_strength'], signature['fft_peak_ratio']]])
    normalized_sig = (sig_vector - mean) / std
    normalized_sig = normalized_sig.astype(np.float32)

    # Predict
    predictions = model.predict({'image_input': img_array, 'signature_input': normalized_sig})
    score = predictions[0][0]

    class_name = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
    return class_name, score, img_for_model

# --- 3. Get Image Samples ---
def get_random_images(directory, num_images=4):
    """Gets a list of random image paths from a directory."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_files:
        print(f"No images found in: {directory}")
        return []
    return random.sample(all_files, min(len(all_files), num_images))

secured_images = get_random_images(SECURED_DIR)
simulated_images = get_random_images(SIMULATED_COPIES_DIR)
aug_true_images = get_random_images(AUGMENTED_TRUE_DIR)
aug_false_images = get_random_images(AUGMENTED_FALSE_DIR)

all_samples = {
    "Augmented (True)": aug_true_images,
    "Augmented (False)": aug_false_images,
    "Secured (Expected: True)": secured_images,
    "Simulated Copies (Expected: False)": simulated_images,
}

# --- 4. Visualize Predictions ---
total_categories = len(all_samples)
valid_categories = {k: v for k, v in all_samples.items() if v}
num_valid_categories = len(valid_categories)

if num_valid_categories == 0:
    print("No images found in any of the specified directories. Exiting.")
else:
    fig, axes = plt.subplots(num_valid_categories, 4, figsize=(15, 4 * num_valid_categories), squeeze=False)
    fig.tight_layout(pad=5.0)

    category_idx = 0
    for category, images in all_samples.items():
        if not images:
            continue

        for j, image_path in enumerate(images):
            if j >= 4: break
            try:
                class_name, score, img = predict_image(image_path)
                ax = axes[category_idx, j]
                ax.imshow(img)
                ax.set_title(f"{category}\nPrediction: {class_name} ({score:.2f})", fontsize=10)
                ax.axis('off')
            except Exception as e:
                print(f"Could not process {image_path}: {e}")
                axes[category_idx, j].axis('off')
        
        for j in range(len(images), 4):
            axes[category_idx, j].axis('off')

        category_idx += 1

    results_path = os.path.join(ROOT_DIR, "results", "visual_test_results_augmented.png")
    plt.savefig(results_path)
    print(f"Saved visual test results to {results_path}")
