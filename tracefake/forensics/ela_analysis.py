import cv2
import numpy as np
import os
from PIL import Image, ImageChops

def perform_ela(image_file, quality=90):
    """
    Performs Error Level Analysis (ELA) on an image.
    
    Args:
        image_file: File path or file-like object (bytes).
        quality: Quality level for the re-saved JPEG (default 90).
        
    Returns:
        ela_image: Numpy array representing the ELA result (RGB).
    """
    try:
        # Load image with PIL
        if hasattr(image_file, 'read'):
            image_file.seek(0)
            original = Image.open(image_file).convert('RGB')
            image_file.seek(0)
        else:
            original = Image.open(image_file).convert('RGB')

        # Save to buffer at specified quality
        # We need to save it to a temporary buffer to simulate compression
        import io
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        
        # Open compressed image
        compressed = Image.open(buffer).convert('RGB')
        
        # Calculate difference
        # ELA = |Original - Compressed|
        ela_image = ImageChops.difference(original, compressed)
        
        # Extrema ensures we pick up the min/max differences
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            max_diff = 1 # Avoid division by zero
            
        # Scale to make it visible
        scale = 255.0 / max_diff
        ela_image = ImageEnhance_brightness(ela_image, scale)
        
        return np.array(ela_image)

    except Exception as e:
        print(f"Error performing ELA: {e}")
        # Return a blank image or original in case of failure
        return np.zeros((224, 224, 3), dtype=np.uint8)

def ImageEnhance_brightness(image, scale):
    """
    Helper to manually scale brightness since we calculated precise scale factor.
    """
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(scale)
