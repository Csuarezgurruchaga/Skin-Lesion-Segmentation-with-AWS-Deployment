import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image

def preprocess_for_prediction(image_bytes, target_size=(256, 256)):
    """Preprocess an image from bytes for model prediction."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def postprocess_prediction(prediction):
    """Convert model output to a binary mask and prepare for visualization."""
    # Apply threshold to get binary mask
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    
    # Remove batch dimension and channel dimension
    if binary_mask.shape[0] == 1:  # Remove batch dimension if present
        binary_mask = binary_mask[0]
    
    if binary_mask.shape[-1] == 1:  # Remove channel dimension for grayscale
        binary_mask = binary_mask[..., 0]
    
    return binary_mask

def overlay_segmentation(original_image, segmentation_mask, alpha=0.5):
    """Create an overlay of the segmentation on the original image."""
    # Ensure segmentation mask is the right shape
    if original_image.shape[:2] != segmentation_mask.shape[:2]:
        segmentation_mask = cv2.resize(
            segmentation_mask, 
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create a red overlay for the segmentation
    overlay = np.zeros_like(original_image)
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        # For RGB images
        overlay[..., 0] = segmentation_mask  # Red channel
    
    # Combine original and overlay
    result = cv2.addWeighted(original_image, 1, overlay, alpha, 0)
    
    return result

def image_to_bytes(image, format='JPEG'):
    """Convert OpenCV image to bytes."""
    # Convert BGR to RGB if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(image)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()
