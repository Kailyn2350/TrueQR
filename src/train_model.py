import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

# --- Configuration ---
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

DATA_JSON_PATH = os.path.join(ROOT_DIR, "config", "training_data.json")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 25 # Increased epochs for potentially more complex task
VALIDATION_SPLIT = 0.2

# --- 1. Load Data from JSON ---
print("Loading data from JSON...")
with open(DATA_JSON_PATH, 'r') as f:
    all_data = json.load(f)['data']

# Separate data and labels
file_paths = [item['file_path'] for item in all_data]
labels = [item['label'] for item in all_data]
signatures = np.array([[item['signature']['phash'], item['signature']['hf_strength'], item['signature']['fft_peak_ratio']] for item in all_data])

# --- 2. Normalize Signatures ---
print("Normalizing signature data...")
mean = np.mean(signatures, axis=0)
std = np.std(signatures, axis=0)

# Avoid division by zero if a feature has no variance
std[std == 0] = 1.0 

normalized_signatures = (signatures - mean) / std

# Save normalization params for inference
norm_params = {'mean': mean.tolist(), 'std': std.tolist()}

norm_params_path = os.path.join(RESULTS_DIR, 'signature_normalization.json')
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(norm_params_path, 'w') as f:
    json.dump(norm_params, f)
print(f"Saved signature normalization parameters to {norm_params_path}")

# --- 3. Create Datasets ---
print("Creating train and validation datasets...")

# Split data into training and validation sets
train_paths, val_paths, train_sigs, val_sigs, train_labels, val_labels = train_test_split(
    file_paths, normalized_signatures, labels, 
    test_size=VALIDATION_SPLIT, 
    random_state=42, 
    stratify=labels
)

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img

def create_dataset(paths, signatures, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    signature_ds = tf.data.Dataset.from_tensor_slices(signatures.astype(np.float32))
    
    # Combine image and signature datasets
    input_ds = tf.data.Dataset.zip((image_ds, signature_ds))
    
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    return tf.data.Dataset.zip((input_ds, label_ds))

train_ds = create_dataset(train_paths, train_sigs, train_labels)
val_ds = create_dataset(val_paths, val_sigs, val_labels)

# Batch and prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(buffer_size=len(train_paths)).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# --- 4. Build Multi-input Model ---
print("Building multi-input model...")

# Image Input Branch
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

image_input = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='image_input')
x = preprocess_input(image_input)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.Model(inputs=image_input, outputs=x)

# Signature Input Branch
signature_input = tf.keras.Input(shape=(3,), name='signature_input')
y = tf.keras.layers.Dense(16, activation='relu')(signature_input)
y = tf.keras.layers.Dense(8, activation='relu')(y)
y = tf.keras.Model(inputs=signature_input, outputs=y)

# Concatenate branches
combined = tf.keras.layers.concatenate([x.output, y.output])

# Classifier Head
z = tf.keras.layers.Dropout(0.3)(combined)
z = tf.keras.layers.Dense(64, activation='relu')(z)
z = tf.keras.layers.Dense(1, activation='sigmoid')(z)

model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)

# --- 5. Compile Model ---
print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# --- 6. Train Model ---
print(f"\nStarting training for {EPOCHS} epochs...")

# Need to adapt the dataset to return a dictionary of inputs
def adapt_dataset(inputs, label):
    image_tensor, signature_tensor = inputs
    return {'image_input': image_tensor, 'signature_input': signature_tensor}, label

train_ds_adapted = train_ds.map(adapt_dataset)
val_ds_adapted = val_ds.map(adapt_dataset)

history = model.fit(
    train_ds_adapted,
    validation_data=val_ds_adapted,
    epochs=EPOCHS
)
print("\nTraining complete.")

# --- 7. Evaluate and Save ---
print("Evaluating model and saving results...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

history_plot_path = os.path.join(RESULTS_DIR, 'training_history_multi_input.png')
plt.savefig(history_plot_path)
print(f"Saved training history plot to: {history_plot_path}")

model_save_path = os.path.join(RESULTS_DIR, 'true_qr_classifier_augmented.keras')
model.save(model_save_path)
print(f"Saved trained model to: {model_save_path}")