import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import TensorBoard
import datetime
from sklearn.model_selection import train_test_split
from unet_models import build_Unet


# Loss functions for segmentation
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1 - dice

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# Metrics for segmentation
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3]) - intersection
    return tf.keras.backend.mean((intersection + smooth) / (union + smooth))



IMAGE_PATH = '../DATA/ISIC/ISIC2018_Task1-2_Training_Input'
MASK_PATH = '../DATA/ISIC/ISIC2018_Task1_Training_GroundTruth'

image_names = sorted(glob(os.path.join(IMAGE_PATH, '*.jpg')))
mask_names = sorted(glob(os.path.join(MASK_PATH, '*.png')))



# Function to load and preprocess images and masks
def preprocess_data(img_path, mask_path):
    # Load image
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    
    # Load mask
    mask = load_img(mask_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    mask = img_to_array(mask) / 255.0
    mask = (mask > 0.5).astype(np.float32)  # Binarize
    
    return img, mask

# Create TensorFlow dataset
def create_dataset(image_paths, mask_paths):
    def load_data(img_path, mask_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE])
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.cast(mask > 0.5, tf.float32)
        
        return img, mask
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda x, y: load_data(x, y), 
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset



# Split into train, validation, test
BATCH_SIZE = 8
IMG_SIZE = 256
train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = train_test_split(
    image_names, mask_names, test_size=0.2, random_state=42)

train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
    train_img_paths, train_mask_paths, test_size=0.15, random_state=42)

print(f"Train: {len(train_img_paths)}, Validation: {len(val_img_paths)}, Test: {len(test_img_paths)}")

# Create datasets
train_dataset = create_dataset(train_img_paths, train_mask_paths)
val_dataset = create_dataset(val_img_paths, val_mask_paths)
test_dataset = create_dataset(test_img_paths, test_mask_paths)

# Configure for training
train_dataset = train_dataset.shuffle(buffer_size=min(len(train_dataset), 100)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# Build and compile U-Net model
unet_model = build_Unet((256, 256, 3), 4, 0.1)
unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, iou_coef, 'binary_accuracy']
)

# Callback for Early Stopping and model saving
log_dir = os.path.join("logs", "unet_isic_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('stored_models/weights/unet_isic_segmentation.h5', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
]

# Train model
unet_history = unet_model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=callbacks
)

# store the best model
tf.saved_model.save(unet_model, 'stored_models/unet_skin_lesion_saved_model')

# Evaluate and compare models
unet_evaluation = unet_model.evaluate(test_dataset)

print("U-Net Evaluation:")
for name, value in zip(unet_model.metrics_names, unet_evaluation):
    print(f"{name}: {value:.4f}")