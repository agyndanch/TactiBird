"""
TactiBird Overlay - Image Processing Utilities
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray, enhance_contrast: bool = True, 
                    denoise: bool = True, sharpen: bool = False) -> np.ndarray:
    """
    Preprocess image for better analysis
    
    Args:
        image: Input image
        enhance_contrast: Whether to enhance contrast
        denoise: Whether to apply denoising
        sharpen: Whether to apply sharpening
        
    Returns:
        Preprocessed image
    """
    try:
        processed = image.copy()
        
        # Convert to BGR if needed
        if len(processed.shape) == 4:  # BGRA
            processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)
        
        # Enhance contrast
        if enhance_contrast:
            processed = enhance_image_contrast(processed)
        
        # Denoise
        if denoise:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # Sharpen
        if sharpen:
            processed = sharpen_image(processed)
        
        return processed
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return image

def enhance_image_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    """
    Enhance image contrast
    
    Args:
        image: Input image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        Contrast enhanced image
    """
    try:
        # Apply contrast and brightness adjustment
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    except Exception as e:
        logger.error(f"Contrast enhancement failed: {e}")
        return image

def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Sharpen image using unsharp masking
    
    Args:
        image: Input image
        strength: Sharpening strength (0.5-2.0)
        
    Returns:
        Sharpened image
    """
    try:
        # Create sharpening kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * strength
        
        # Apply kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    except Exception as e:
        logger.error(f"Image sharpening failed: {e}")
        return image

def extract_region(image: np.ndarray, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
    """
    Extract region from image with bounds checking
    
    Args:
        image: Source image
        x, y: Top-left coordinates
        width, height: Region dimensions
        
    Returns:
        Extracted region or None if invalid
    """
    try:
        img_height, img_width = image.shape[:2]
        
        # Clamp coordinates to image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        # Clamp dimensions
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))
        
        # Extract region
        region = image[y:y+height, x:x+width]
        
        if region.size == 0:
            logger.warning(f"Empty region extracted: ({x}, {y}, {width}, {height})")
            return None
        
        return region
        
    except Exception as e:
        logger.error(f"Region extraction failed: {e}")
        return None

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        target_width, target_height = target_size
        
        if maintain_aspect:
            # Calculate scaling factor
            height, width = image.shape[:2]
            scale_x = target_width / width
            scale_y = target_height / height
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Pad if necessary
            if new_width != target_width or new_height != target_height:
                resized = pad_image(resized, target_size)
        else:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        return resized
        
    except Exception as e:
        logger.error(f"Image resizing failed: {e}")
        return image

def pad_image(image: np.ndarray, target_size: Tuple[int, int], 
             color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Pad image to target size
    
    Args:
        image: Input image
        target_size: (width, height)
        color: Padding color (BGR)
        
    Returns:
        Padded image
    """
    try:
        target_width, target_height = target_size
        height, width = image.shape[:2]
        
        # Calculate padding
        pad_x = max(0, target_width - width)
        pad_y = max(0, target_height - height)
        
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        
        # Apply padding
        if len(image.shape) == 3:
            padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=color)
        else:
            padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=color[0])
        
        return padded
        
    except Exception as e:
        logger.error(f"Image padding failed: {e}")
        return image

def crop_to_content(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop image to remove empty borders
    
    Args:
        image: Input image
        threshold: Pixel intensity threshold for content detection
        
    Returns:
        Cropped image
    """
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find content bounds
        coords = np.argwhere(gray > threshold)
        
        if len(coords) == 0:
            return