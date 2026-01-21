import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def load_and_preprocess_image(image_file, target_size=(224, 224)):
    """
    Loads an image from a file object (Streamlit UploadedFile) or path,
    resizes it, and applies EfficientNet preprocessing.

    Args:
        image_file: File path or file-like object (bytes).
        target_size: Tuple (height, width).

    Returns:
        preprocessed_image: A numpy array of shape (1, height, width, 3) ready for inference.
        original_image: The original image as a numpy array (RGB) for display.
    """
    # Read the file bytes
    if hasattr(image_file, 'read'):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Reset pointer for other uses if needed, though usually consumed
        image_file.seek(0)
    else:
        # Assume it's a path
        image = cv2.imread(image_file)
    
    if image is None:
        raise ValueError("Could not decode image.")

    # Convert BGR to RGB (OpenCV uses BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image_resized = cv2.resize(image_rgb, target_size)

    # Convert to float32 and expand dims
    image_batch = np.expand_dims(image_resized, axis=0)

    # Preprocess input (EfficientNet expects 0-255 or -1 to 1 depending on version, 
    # the built-in preprocess_input handles it correctly)
    preprocessed_image = efficientnet_preprocess(image_batch)

    return preprocessed_image, image_rgb
