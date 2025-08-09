import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys

# --- Configuration ---
# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

MODEL_PATH = os.path.join(ROOT_DIR, "results", "true_qr_classifier.keras")
SECURED_DIR = os.path.join(ROOT_DIR, "True_data", "secured")
SIMULATED_COPIES_DIR = os.path.join(ROOT_DIR, "False_data", "simulated_copies")
AUGMENTED_TRUE_DIR = os.path.join(ROOT_DIR, "augmented_data", "true")
AUGMENTED_FALSE_DIR = os.path.join(ROOT_DIR, "augmented_data", "false")


IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['false', 'true'] # From train_model.py

# --- 1. Load Model ---
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# --- 2. Prediction Function ---
def predict_image(image_path):
    """Loads an image, preprocesses it, and returns the model's prediction."""
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = predictions[0][0]

    class_name = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
    return class_name, score, img

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
# Adjust grid size based on how many categories have images
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
            class_name, score, img = predict_image(image_path)

            ax = axes[category_idx, j]
            ax.imshow(img)
            ax.set_title(f"{category}\nPrediction: {class_name} ({score:.2f})", fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots for this category
        for j in range(len(images), 4):
            axes[category_idx, j].axis('off')

        category_idx += 1

    plt.show()