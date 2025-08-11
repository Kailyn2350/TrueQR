import os
import cv2
import numpy as np
import tensorflow as tf
import json
from test_verify import compute_signature

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

MODEL_PATH = os.path.join(ROOT_DIR, "results", "true_qr_classifier_augmented.keras")
NORM_PARAMS_PATH = os.path.join(ROOT_DIR, "results", "signature_normalization.json")
# Let's pick a file we know exists from the augmented data
IMAGE_PATH = os.path.join(ROOT_DIR, "augmented_data", "true", "image_0.jpg") # Assuming jpg, will correct if needed

IMG_HEIGHT = 224
IMG_WIDTH = 224

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

# --- 2. Prediction Function (simplified) ---
def predict_single_image(image_path):
    print(f"Predicting for image: {image_path}")
    # Load image for model input
    img_for_model = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img_for_model)
    img_array = tf.expand_dims(img_array, 0)

    # Load image for signature calculation
    img_for_sig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_for_sig is None:
        print(f"Error: Could not read image for signature: {image_path}")
        return

    # Compute and normalize signature
    signature = compute_signature(img_for_sig)
    sig_vector = np.array([[signature['phash'], signature['hf_strength'], signature['fft_peak_ratio']]])
    normalized_sig = (sig_vector - mean) / std
    normalized_sig = normalized_sig.astype(np.float32)

    # Predict
    print("Calling model.predict...")
    predictions = model.predict({'image_input': img_array, 'signature_input': normalized_sig})
    score = predictions[0][0]
    print(f"Prediction score: {score}")

# --- 3. Run Prediction ---
if os.path.exists(IMAGE_PATH):
    predict_single_image(IMAGE_PATH)
else:
    print(f"Error: Test image not found at {IMAGE_PATH}")
    # As a fallback, let's read the json and get a valid path
    with open(os.path.join(ROOT_DIR, "config", "augmented_training_data.json"), 'r') as f:
        data = json.load(f)['data']
    if data:
        first_true_image = next((item['file_path'] for item in data if item['label'] == 1), None)
        if first_true_image:
            print(f"Found a valid image path: {first_true_image}")
            predict_single_image(first_true_image)
        else:
            print("Could not find any true images in the json data.")
