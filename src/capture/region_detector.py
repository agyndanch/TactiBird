"""
TactiBird Overlay - Game Region Detection
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DetectedRegion:
    """Represents a detected game region"""
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    template_match: bool = False

class RegionDetector:
    """Detects TFT game regions automatically"""
    
    def __init__(self):
        self.ui_templates = {}
        self.cached_regions = {}
        self.confidence_threshold = 0.7
        
        # Standard aspect ratios for different regions
        self.region_ratios = {
            'board': (4/3, 3/2),  # Board is roughly rectangular
            'shop': (5/1, 6/1),   # Shop is wide and short
            'gold': (2/1, 3/1),   # Gold display is small rectangle
            'health': (2/1, 3/1), # Health similar to gold
            'level': (1/1, 2/1)   # Level is more square
        }
        
        logger.info("Region detector initialized")
    
    def detect_game_window(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the TFT game window bounds"""
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for large rectangular contours
            height, width = screenshot.shape[:2]
            min_area = width * height * 0.5  # At least half the screen
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular (4 corners)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = w / h
                        
                        # TFT window should have reasonable aspect ratio
                        if 1.2 <= aspect_ratio <= 2.0:
                            return (x, y, w, h)
            
            # Fallback: assume full screen is game window
            return (0, 0, width, height)
            
        except Exception as e:
            logger.error(f"Game window detection failed: {e}")
            return None
    
    def detect_ui_regions(self, screenshot: np.ndarray) -> Dict[str, DetectedRegion]:
        """Detect UI regions using template matching and heuristics"""
        regions = {}
        
        try:
            # First detect game window
            game_window = self.detect_game_window(screenshot)
            if not game_window:
                logger.warning("Could not detect game window")
                return regions
            
            gx, gy, gw, gh = game_window
            game_region = screenshot[gy:gy+gh, gx:gx+gw]
            
            # Detect individual regions within game window
            regions['board'] = self._detect_board_region(game_region, gx, gy)
            regions['shop'] = self._detect_shop_region(game_region, gx, gy)
            regions['gold'] = self._detect_gold_region(game_region, gx, gy)
            regions['health'] = self._detect_health_region(game_region, gx, gy)
            regions['level'] = self._detect_level_region(game_region, gx, gy)
            regions['traits'] = self._detect_traits_region(game_region, gx, gy)
            
            # Filter out None results
            regions = {k: v for k, v in regions.items() if v is not None}
            
            logger.info(f"Detected {len(regions)} UI regions")
            return regions
            
        except Exception as e:
            logger.error(f"UI region detection failed: {e}")
            return {}
    
    def _detect_board_region(self, game_image: np.ndarray, offset_x: int, offset_y: int) -> Optional[DetectedRegion]:
        """Detect the game board region"""
        try:
            height, width = game_image.shape[:2]
            
            # Board is typically in the center-left area
            # Estimate based on typical TFT layout
            board_x = int(width * 0.1)
            board_y = int(height * 0.15)
            board_w = int(width * 0.6)
            board_h = int(height * 0.6)
            
            # Validate using hexagonal pattern detection
            board_region = game_image[board_y:board_y+board_h, board_x:board_x+board_w]
            confidence = self._validate_board_pattern(board_region)
            
            if confidence > 0.5:
                return DetectedRegion(
                    name='board',
                    bbox=(offset_x + board_x, offset_y + board_y, board_w, board_h),
                    confidence=confidence
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Board detection failed: {e}")
            return None
    
    def _detect_shop_region(self, game_image: np.ndarray, offset_x: int, offset_y: int) -> Optional[DetectedRegion]:
        """Detect the shop region"""
        try:
            height, width = game_image.shape[:2]
            
            # Shop is typically at the bottom center
            shop_x = int(width * 0.2)
            shop_y = int(height * 0.75)
            shop_w = int(width * 0.6)
            shop_h = int(height * 0.2)
            
            # Look for shop-like UI elements
            shop_region = game_image[shop_y:shop_y+shop_h, shop_x:shop_x+shop_w]
            confidence = self._validate_shop_pattern(shop_region)
            
            if confidence > 0.4:
                return DetectedRegion(
                    name='shop',
                    bbox=(offset_x + shop_x, offset_y + shop_y, shop_w, shop_h),
                    confidence=confidence
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Shop detection failed: {e}")
            return None
    
    def _detect_gold_region(self, game_image: np.ndarray, offset_x: int, offset_y: int) -> Optional[DetectedRegion]:
        """Detect the gold display region"""
        try:
            # Gold is typically in bottom-left corner
            height, width = game_image.shape[:2]
            
            # Search in bottom-left quadrant
            search_region = game_image[int(height*0.7):height, 0:int(width*0.3)]
            
            # Look for gold-colored elements
            gold_areas = self._find_gold_colored_regions(search_region)
            
            if gold_areas:
                # Take the largest gold-colored area
                best_area = max(gold_areas, key=lambda x: x[2] * x[3])
                gx, gy, gw, gh = best_area
                
                return DetectedRegion(
                    name='gold',
                    bbox=(offset_x + gx, offset_y + int(height*0.7) + gy, gw, gh),
                    confidence=0.8
                )
            
            # Fallback to estimated position
            gold_x = int(width * 0.05)
            gold_y = int(height * 0.85)
            gold_w = int(width * 0.08)
            gold_h = int(height * 0.06)
            
            return DetectedRegion(
                name='gold',
                bbox=(offset_x + gold_x, offset_y + gold_y, gold_w, gold_h),
                confidence=0.5
            )
            
        except Exception as e:
            logger.error(f"Gold detection failed: {e}")
            return None
    
    def _detect_health_region(self, game_image: np.ndarray, offset_x: int, offset_y: int) -> Optional[DetectedRegion]:
        """Detect the health display region"""
        try:
            height, width = game_image.shape[:2]
            
            # Health is typically near gold, slightly above
            health_x = int(width * 0.05)
            health_y = int(height * 0.78)
            health_w = int(width * 0.08)
            health_h = int(height * 0.06)
            
            # Look for red/green health bar colors
            health_region = game_image[health_y:health_y+health_h, health_x:health_x+health_w]
            confidence = self._validate_health_colors(health_region)
            
            return DetectedRegion(
                name='health',
                bbox=(offset_x + health_x, offset_y + health_y, health_w, health_h),
                confidence=max(0.5, confidence)
            )
            
        except Exception as e:
            logger.error(f"Health detection failed: {e}")
            return None
    
    def _detect_level_region(self, game_image: np.ndarray, offset_x: int, offset_y: int) -> Optional[DetectedRegion]:
        """Detect the level display region"""
        try:
            height, width = game_image.shape[:2]
            
            # Level is typically in bottom-left area
            level_x = int(width * 0.05)
            level_y = int(height * 0.71)
            level_w = int(width * 0.08)
            level_h = int(height * 0.06)
            
            return DetectedRegion(
                name='level',
                bbox=(offset_x + level_x, offset_y + level_y, level_w, level_h),
                confidence=0.6
            )
            
        except Exception as e:
            logger.error(f"Level detection failed: {e}")
            return None
    
    def _detect_traits_region(self, game_image: np.ndarray, offset_x: int, offset_y: int) -> Optional[DetectedRegion]:
        """Detect the traits panel region"""
        try:
            height, width = game_image.shape[:2]
            
            # Traits panel is typically on the left side
            traits_x = int(width * 0.02)
            traits_y = int(height * 0.15)
            traits_w = int(width * 0.15)
            traits_h = int(height * 0.5)
            
            # Look for vertical list pattern
            traits_region = game_image[traits_y:traits_y+traits_h, traits_x:traits_x+traits_w]
            confidence = self._validate_traits_pattern(traits_region)
            
            return DetectedRegion(
                name='traits',
                bbox=(offset_x + traits_x, offset_y + traits_y, traits_w, traits_h),
                confidence=max(0.5, confidence)
            )
            
        except Exception as e:
            logger.error(f"Traits detection failed: {e}")
            return None
    
    def _validate_board_pattern(self, board_region: np.ndarray) -> float:
        """Validate if region contains hexagonal board pattern"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY)
            
            # Look for hexagonal/circular patterns
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 50,
                param1=50, param2=30, minRadius=20, maxRadius=100
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Board should have multiple circular/hexagonal spaces
                if len(circles) >= 10:  # Minimum expected hex spaces
                    return 0.8
                elif len(circles) >= 5:
                    return 0.6
            
            return 0.3
            
        except Exception:
            return 0.3
    
    def _validate_shop_pattern(self, shop_region: np.ndarray) -> float:
        """Validate if region contains shop pattern"""
        try:
            # Look for rectangular shop slots
            gray = cv2.cvtColor(shop_region, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for rectangular shapes
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangular_contours = 0
            for contour in contours:
                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if roughly rectangular and reasonable size
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if 1000 < area < 50000:  # Reasonable size for shop slot
                        rectangular_contours += 1
            
            # Shop should have multiple rectangular slots
            if rectangular_contours >= 3:
                return 0.7
            elif rectangular_contours >= 1:
                return 0.5
            
            return 0.3
            
        except Exception:
            return 0.3
    
    def _find_gold_colored_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find regions with gold-like colors"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define gold color range in HSV
            lower_gold = np.array([15, 100, 100])
            upper_gold = np.array([35, 255, 255])
            
            # Create mask for gold colors
            mask = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            gold_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    gold_regions.append((x, y, w, h))
            
            return gold_regions
            
        except Exception:
            return []
    
    def _validate_health_colors(self, health_region: np.ndarray) -> float:
        """Validate if region contains health bar colors"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(health_region, cv2.COLOR_BGR2HSV)
            
            # Check for red (low health) or green (high health) colors
            # Red range
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Green range
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            
            # Create masks
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Combine masks
            health_mask = red_mask1 + red_mask2 + green_mask
            
            # Calculate percentage of health-colored pixels
            total_pixels = health_region.shape[0] * health_region.shape[1]
            health_pixels = np.sum(health_mask > 0)
            
            return min(1.0, health_pixels / total_pixels * 5)  # Scale up confidence
            
        except Exception:
            return 0.4
    
    def _validate_traits_pattern(self, traits_region: np.ndarray) -> float:
        """Validate if region contains traits list pattern"""
        try:
            # Look for vertical list pattern
            gray = cv2.cvtColor(traits_region, cv2.COLOR_BGR2GRAY)
            
            # Apply horizontal edge detection to find list items
            kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
            edges = cv2.filter2D(gray, -1, kernel)
            
            # Look for horizontal lines (trait separators)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                horizontal_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is roughly horizontal
                    if abs(y2 - y1) < 5 and abs(x2 - x1) > 20:
                        horizontal_lines += 1
                
                # Traits panel should have multiple horizontal separators
                if horizontal_lines >= 3:
                    return 0.7
                elif horizontal_lines >= 1:
                    return 0.5
            
            return 0.4
            
        except Exception:
            return 0.4
    
    def calibrate_regions(self, screenshot: np.ndarray, known_regions: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, DetectedRegion]:
        """Calibrate region detection using known good regions"""
        try:
            calibrated = {}
            
            for region_name, (x, y, w, h) in known_regions.items():
                calibrated[region_name] = DetectedRegion(
                    name=region_name,
                    bbox=(x, y, w, h),
                    confidence=1.0
                )
            
            # Cache for future use
            self.cached_regions = calibrated.copy()
            
            logger.info(f"Calibrated {len(calibrated)} regions")
            return calibrated
            
        except Exception as e:
            logger.error(f"Region calibration failed: {e}")
            return {}
    
    def get_region_info(self) -> Dict[str, any]:
        """Get information about detected regions"""
        return {
            "cached_regions": len(self.cached_regions),
            "confidence_threshold": self.confidence_threshold,
            "supported_regions": list(self.region_ratios.keys())
        }