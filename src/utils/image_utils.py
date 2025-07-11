"""
TactiBird - Image Utilities Module
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any

class ImageUtils:
    """Utility functions for image processing"""
    
    @staticmethod
    def crop_region(image: np.ndarray, region: Dict[str, int]) -> np.ndarray:
        """Crop image to specified region"""
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to specified dimensions"""
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 10) -> np.ndarray:
        """Enhance image contrast"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)