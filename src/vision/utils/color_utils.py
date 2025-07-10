"""
TactiBird Overlay - Color Detection Utilities
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class ColorSpace(Enum):
    """Supported color spaces"""
    BGR = "bgr"
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"
    GRAY = "gray"

class TFTColors:
    """TFT-specific color definitions"""
    
    # Champion tier colors (HSV ranges)
    TIER_COLORS = {
        1: [(0, 0, 100), (180, 30, 200)],      # Gray - Tier 1
        2: [(35, 50, 50), (85, 255, 255)],     # Green - Tier 2
        3: [(100, 50, 50), (130, 255, 255)],   # Blue - Tier 3
        4: [(130, 50, 100), (160, 255, 255)],  # Purple - Tier 4
        5: [(15, 100, 150), (35, 255, 255)]    # Gold - Tier 5
    }
    
    # Trait activation colors
    TRAIT_COLORS = {
        'bronze': [(15, 82, 138), (25, 255, 255)],
        'silver': [(0, 0, 180), (180, 30, 255)],
        'gold': [(15, 100, 200), (35, 255, 255)],
        'prismatic': [(140, 100, 100), (170, 255, 255)]
    }
    
    # UI element colors
    UI_COLORS = {
        'gold': [(15, 100, 100), (35, 255, 255)],
        'health_high': [(35, 50, 50), (85, 255, 255)],
        'health_medium': [(15, 100, 100), (35, 255, 255)],
        'health_low': [(0, 100, 100), (10, 255, 255)],
        'mana_blue': [(100, 50, 50), (130, 255, 255)],
        'experience': [(130, 50, 100), (160, 255, 255)]
    }
    
    # Item rarity colors
    ITEM_COLORS = {
        'component': [(0, 0, 100), (180, 30, 200)],    # Gray
        'completed': [(100, 100, 100), (130, 255, 255)], # Blue
        'special': [(130, 100, 100), (160, 255, 255)]     # Purple
    }

class ColorDetector:
    """Utility class for color-based detection"""
    
    def __init__(self):
        self.color_cache = {}
        self.calibration_data = {}
        
    def detect_color_in_region(self, image: np.ndarray, color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]], 
                              color_space: ColorSpace = ColorSpace.HSV) -> Dict[str, Any]:
        """
        Detect specific color in image region
        
        Args:
            image: Input image
            color_range: (lower_bound, upper_bound) in specified color space
            color_space: Color space to use for detection
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Convert image to specified color space
            converted = self._convert_color_space(image, color_space)
            
            # Create color mask
            lower_bound, upper_bound = color_range
            mask = cv2.inRange(converted, np.array(lower_bound), np.array(upper_bound))
            
            # Calculate statistics
            total_pixels = mask.shape[0] * mask.shape[1]
            color_pixels = np.sum(mask > 0)
            color_ratio = color_pixels / total_pixels if total_pixels > 0 else 0
            
            # Find contours of colored regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get largest colored region
            largest_contour = None
            largest_area = 0
            
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > largest_area:
                        largest_area = area
                        largest_contour = contour
            
            # Calculate center of largest region
            center = None
            if largest_contour is not None:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            return {
                'detected': color_ratio > 0.01,  # At least 1% of pixels
                'color_ratio': color_ratio,
                'color_pixels': color_pixels,
                'total_pixels': total_pixels,
                'largest_area': largest_area,
                'center': center,
                'mask': mask,
                'contours': contours
            }
            
        except Exception as e:
            logger.error(f"Color detection failed: {e}")
            return {
                'detected': False,
                'color_ratio': 0,
                'color_pixels': 0,
                'total_pixels': 0,
                'largest_area': 0,
                'center': None,
                'mask': None,
                'contours': []
            }
    
    def detect_champion_tier(self, champion_region: np.ndarray) -> Optional[int]:
        """Detect champion tier based on color"""
        try:
            best_tier = None
            best_ratio = 0
            
            for tier, color_range in TFTColors.TIER_COLORS.items():
                result = self.detect_color_in_region(champion_region, color_range, ColorSpace.HSV)
                
                if result['color_ratio'] > best_ratio:
                    best_ratio = result['color_ratio']
                    best_tier = tier
            
            # Only return tier if confidence is high enough
            if best_ratio > 0.05:  # At least 5% of pixels
                return best_tier
            
            return None
            
        except Exception as e:
            logger.error(f"Champion tier detection failed: {e}")
            return None
    
    def detect_trait_activation(self, trait_region: np.ndarray) -> str:
        """Detect trait activation level based on color"""
        try:
            best_level = 'inactive'
            best_ratio = 0
            
            for level, color_range in TFTColors.TRAIT_COLORS.items():
                result = self.detect_color_in_region(trait_region, color_range, ColorSpace.HSV)
                
                if result['color_ratio'] > best_ratio:
                    best_ratio = result['color_ratio']
                    best_level = level
            
            # Only return activation level if confidence is high enough
            if best_ratio > 0.1:  # At least 10% of pixels
                return best_level
            
            return 'inactive'
            
        except Exception as e:
            logger.error(f"Trait activation detection failed: {e}")
            return 'inactive'
    
    def detect_health_level(self, health_region: np.ndarray) -> str:
        """Detect health level based on color"""
        try:
            # Check for different health levels
            for level in ['health_high', 'health_medium', 'health_low']:
                color_range = TFTColors.UI_COLORS[level]
                result = self.detect_color_in_region(health_region, color_range, ColorSpace.HSV)
                
                if result['color_ratio'] > 0.3:  # At least 30% of pixels
                    return level.replace('health_', '')
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Health level detection failed: {e}")
            return 'unknown'
    
    def detect_gold_elements(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect gold-colored UI elements"""
        try:
            gold_range = TFTColors.UI_COLORS['gold']
            result = self.detect_color_in_region(image, gold_range, ColorSpace.HSV)
            
            if not result['detected'] or not result['contours']:
                return []
            
            # Convert contours to bounding boxes
            gold_regions = []
            for contour in result['contours']:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    gold_regions.append((x, y, w, h))
            
            return gold_regions
            
        except Exception as e:
            logger.error(f"Gold element detection failed: {e}")
            return []
    
    def analyze_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[Tuple[int, int, int], float]]:
        """Analyze dominant colors in image using K-means clustering"""
        try:
            # Reshape image to list of pixels
            pixels = image.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Calculate color percentages
            labels = labels.flatten()
            total_pixels = len(labels)
            
            dominant_colors = []
            for i in range(k):
                count = np.sum(labels == i)
                percentage = count / total_pixels
                color = tuple(map(int, centers[i]))
                dominant_colors.append((color, percentage))
            
            # Sort by percentage
            dominant_colors.sort(key=lambda x: x[1], reverse=True)
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Dominant color analysis failed: {e}")
            return []
    
    def _convert_color_space(self, image: np.ndarray, target_space: ColorSpace) -> np.ndarray:
        """Convert image to target color space"""
        try:
            if target_space == ColorSpace.HSV:
                return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif target_space == ColorSpace.RGB:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif target_space == ColorSpace.LAB:
                return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif target_space == ColorSpace.GRAY:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:  # BGR or unknown
                return image
                
        except Exception as e:
            logger.error(f"Color space conversion failed: {e}")
            return image
    
    def create_color_mask(self, image: np.ndarray, color_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]], 
                         color_space: ColorSpace = ColorSpace.HSV) -> np.ndarray:
        """Create combined mask for multiple color ranges"""
        try:
            converted = self._convert_color_space(image, color_space)
            combined_mask = np.zeros(converted.shape[:2], dtype=np.uint8)
            
            for lower, upper in color_ranges:
                mask = cv2.inRange(converted, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            return combined_mask
            
        except Exception as e:
            logger.error(f"Color mask creation failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def filter_by_color(self, image: np.ndarray, color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]], 
                       color_space: ColorSpace = ColorSpace.HSV) -> np.ndarray:
        """Filter image to show only specified color range"""
        try:
            mask = self.create_color_mask(image, [color_range], color_space)
            result = cv2.bitwise_and(image, image, mask=mask)
            return result
            
        except Exception as e:
            logger.error(f"Color filtering failed: {e}")
            return image
    
    def calculate_color_similarity(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], 
                                  color_space: ColorSpace = ColorSpace.BGR) -> float:
        """Calculate similarity between two colors (0-1, where 1 is identical)"""
        try:
            # Convert single colors to small images for color space conversion
            img1 = np.uint8([[color1]])
            img2 = np.uint8([[color2]])
            
            # Convert to specified color space
            conv1 = self._convert_color_space(img1, color_space)
            conv2 = self._convert_color_space(img2, color_space)
            
            # Calculate Euclidean distance
            diff = np.linalg.norm(conv1[0, 0] - conv2[0, 0])
            
            # Normalize to 0-1 range (max distance depends on color space)
            max_distance = 441.67 if color_space == ColorSpace.BGR else 441.67  # sqrt(255^2 + 255^2 + 255^2)
            similarity = 1 - (diff / max_distance)
            
            return max(0, min(1, similarity))
            
        except Exception as e:
            logger.error(f"Color similarity calculation failed: {e}")
            return 0.0
    
    def calibrate_colors(self, calibration_images: List[Tuple[np.ndarray, str]], 
                        known_colors: Dict[str, Tuple[int, int, int]]) -> Dict[str, Any]:
        """Calibrate color detection using known samples"""
        try:
            calibration_results = {}
            
            for image, label in calibration_images:
                if label in known_colors:
                    expected_color = known_colors[label]
                    
                    # Analyze dominant colors in image
                    dominant_colors = self.analyze_dominant_colors(image, k=3)
                    
                    if dominant_colors:
                        detected_color = dominant_colors[0][0]  # Most dominant color
                        similarity = self.calculate_color_similarity(expected_color, detected_color)
                        
                        calibration_results[label] = {
                            'expected': expected_color,
                            'detected': detected_color,
                            'similarity': similarity,
                            'dominant_colors': dominant_colors
                        }
            
            # Store calibration data
            self.calibration_data.update(calibration_results)
            
            return calibration_results
            
        except Exception as e:
            logger.error(f"Color calibration failed: {e}")
            return {}
    
    def get_color_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive color statistics for an image"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Calculate statistics
            stats = {
                'bgr': {
                    'mean': np.mean(image, axis=(0, 1)).tolist(),
                    'std': np.std(image, axis=(0, 1)).tolist(),
                    'min': np.min(image, axis=(0, 1)).tolist(),
                    'max': np.max(image, axis=(0, 1)).tolist()
                },
                'hsv': {
                    'mean': np.mean(hsv, axis=(0, 1)).tolist(),
                    'std': np.std(hsv, axis=(0, 1)).tolist()
                },
                'brightness': float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))),
                'contrast': float(np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))),
                'dominant_colors': self.analyze_dominant_colors(image, k=5)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Color statistics calculation failed: {e}")
            return {}
    
    def detect_ui_elements_by_color(self, image: np.ndarray) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Detect various UI elements based on their colors"""
        try:
            ui_elements = {}
            
            # Detect different UI elements
            for element_type, color_range in TFTColors.UI_COLORS.items():
                result = self.detect_color_in_region(image, color_range, ColorSpace.HSV)
                
                if result['detected'] and result['contours']:
                    regions = []
                    for contour in result['contours']:
                        area = cv2.contourArea(contour)
                        if area > 50:  # Minimum area for UI elements
                            x, y, w, h = cv2.boundingRect(contour)
                            regions.append((x, y, w, h))
                    
                    ui_elements[element_type] = regions
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"UI element detection by color failed: {e}")
            return {}