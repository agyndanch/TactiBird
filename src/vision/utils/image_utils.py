"""
TactiBird Overlay - Image Processing Utilities
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
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
            return image
        
        # Get bounding box
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        # Crop image
        cropped = image[y0:y1, x0:x1]
        return cropped
        
    except Exception as e:
        logger.error(f"Image cropping failed: {e}")
        return image

def detect_edges(image: np.ndarray, low_threshold: int = 50, 
                high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges in image using Canny edge detection
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
        
    Returns:
        Edge-detected image
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
        
    except Exception as e:
        logger.error(f"Edge detection failed: {e}")
        return image

def find_contours(image: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """
    Find contours in image
    
    Args:
        image: Input image (preferably binary)
        min_area: Minimum contour area to keep
        
    Returns:
        List of contours
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        return filtered_contours
        
    except Exception as e:
        logger.error(f"Contour detection failed: {e}")
        return []

def get_image_histogram(image: np.ndarray, channels: List[int] = None) -> List[np.ndarray]:
    """
    Calculate image histogram
    
    Args:
        image: Input image
        channels: Channels to analyze (default: all)
        
    Returns:
        List of histograms for each channel
    """
    try:
        if channels is None:
            if len(image.shape) == 3:
                channels = [0, 1, 2]
            else:
                channels = [0]
        
        histograms = []
        for channel in channels:
            if len(image.shape) == 3:
                hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
            else:
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            histograms.append(hist)
        
        return histograms
        
    except Exception as e:
        logger.error(f"Histogram calculation failed: {e}")
        return []

def apply_morphological_operations(image: np.ndarray, operation: str = "close", 
                                 kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological operations to image
    
    Args:
        image: Input image (preferably binary)
        operation: Operation type ("open", "close", "erode", "dilate")
        kernel_size: Size of morphological kernel
        
    Returns:
        Processed image
    """
    try:
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply operation
        if operation == "open":
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == "erode":
            result = cv2.erode(image, kernel, iterations=1)
        elif operation == "dilate":
            result = cv2.dilate(image, kernel, iterations=1)
        else:
            logger.warning(f"Unknown morphological operation: {operation}")
            return image
        
        return result
        
    except Exception as e:
        logger.error(f"Morphological operation failed: {e}")
        return image

def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
    """
    Convert image color space
    
    Args:
        image: Input image
        conversion: Conversion type (e.g., "BGR2HSV", "BGR2GRAY")
        
    Returns:
        Converted image
    """
    try:
        conversion_map = {
            "BGR2HSV": cv2.COLOR_BGR2HSV,
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "HSV2BGR": cv2.COLOR_HSV2BGR,
            "GRAY2BGR": cv2.COLOR_GRAY2BGR,
            "RGB2BGR": cv2.COLOR_RGB2BGR
        }
        
        if conversion in conversion_map:
            converted = cv2.cvtColor(image, conversion_map[conversion])
            return converted
        else:
            logger.warning(f"Unknown color conversion: {conversion}")
            return image
            
    except Exception as e:
        logger.error(f"Color conversion failed: {e}")
        return image

def threshold_image(image: np.ndarray, threshold_type: str = "binary", 
                   threshold_value: int = 127) -> Tuple[np.ndarray, int]:
    """
    Apply thresholding to image
    
    Args:
        image: Input image (grayscale)
        threshold_type: Type of thresholding
        threshold_value: Threshold value (ignored for adaptive methods)
        
    Returns:
        Tuple of (thresholded image, threshold value used)
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if threshold_type == "binary":
            ret_val, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_type == "binary_inv":
            ret_val, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        elif threshold_type == "otsu":
            ret_val, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == "adaptive_mean":
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            ret_val = threshold_value
        elif threshold_type == "adaptive_gaussian":
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            ret_val = threshold_value
        else:
            logger.warning(f"Unknown threshold type: {threshold_type}")
            return gray, threshold_value
        
        return thresh, ret_val
        
    except Exception as e:
        logger.error(f"Image thresholding failed: {e}")
        return image, threshold_value

def calculate_image_quality(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate image quality metrics
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of quality metrics
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate metrics
        metrics = {}
        
        # Variance of Laplacian (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = laplacian_var
        
        # Brightness (mean intensity)
        metrics['brightness'] = np.mean(gray)
        
        # Contrast (standard deviation)
        metrics['contrast'] = np.std(gray)
        
        # Signal-to-noise ratio estimate
        noise_estimate = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        signal_estimate = np.std(gray)
        metrics['snr'] = signal_estimate / max(noise_estimate, 1e-10)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Image quality calculation failed: {e}")
        return {}

def save_debug_image(image: np.ndarray, filename: str, create_dirs: bool = True):
    """
    Save image for debugging purposes
    
    Args:
        image: Image to save
        filename: Output filename
        create_dirs: Whether to create directories if they don't exist
    """
    try:
        if create_dirs:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(filename, image)
        logger.debug(f"Debug image saved: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save debug image: {e}")

def compare_images(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compare two images using structural similarity
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    try:
        # Ensure images are the same size
        if image1.shape != image2.shape:
            height, width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
            image1 = cv2.resize(image1, (width, height))
            image2 = cv2.resize(image2, (width, height))
        
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
        
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2
        
        # Calculate mean squared error
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        
        # Convert to similarity score
        max_pixel_value = 255.0
        similarity = 1.0 - (mse / (max_pixel_value ** 2))
        
        return max(0.0, similarity)
        
    except Exception as e:
        logger.error(f"Image comparison failed: {e}")
        return 0.0

def create_mask_from_color(image: np.ndarray, color: Tuple[int, int, int], 
                          tolerance: int = 10) -> np.ndarray:
    """
    Create a mask for pixels matching a specific color
    
    Args:
        image: Input image
        color: Target color (BGR)
        tolerance: Color tolerance
        
    Returns:
        Binary mask
    """
    try:
        # Convert color to numpy array
        target_color = np.array(color)
        
        # Create color ranges
        lower_bound = np.maximum(target_color - tolerance, 0)
        upper_bound = np.minimum(target_color + tolerance, 255)
        
        # Create mask
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        return mask
        
    except Exception as e:
        logger.error(f"Color mask creation failed: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8)