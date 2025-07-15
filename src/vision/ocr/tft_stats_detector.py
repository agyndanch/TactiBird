"""
TactiBird - Stats Detection Module
"""

import logging
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from src.vision.ocr.text_recognizer import TextRecognizer
from src.utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)

@dataclass
class GameStats:
    """Game statistics data class"""
    gold: int = 0
    health: int = 100
    level: int = 1
    stage: int = 1
    round_num: int = 1
    confidence: float = 0.0
    in_game: bool = False
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'gold': self.gold,
            'health': self.health,
            'level': self.level,
            'stage': self.stage,
            'round': self.round_num,
            'confidence': self.confidence,
            'in_game': self.in_game,
            'timestamp': self.timestamp
        }

class TFTStatsDetector:
    """Detects player statistics from game screen"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regions = config['capture']['regions']
        self.ocr_confidence = config['vision']['ocr_confidence']
        
        # Initialize OCR
        self.text_recognizer = TextRecognizer(config)
        
        # Stats tracking
        self.last_stats = GameStats()
        self.detection_history = []
        
        logger.info("Stats detector initialized")
    
    async def detect_stats(self, screenshot: np.ndarray) -> Optional[GameStats]:
        """Detect all game statistics from screenshot"""
        try:
            # Check if we're in game first
            if not self._is_in_game(screenshot):
                return GameStats(in_game=False)
            
            stats = GameStats(in_game=True)
            confidences = []
            
            # Detect gold
            gold_result = await self._detect_gold(screenshot)
            if gold_result:
                stats.gold = gold_result[0]
                confidences.append(gold_result[1])
            
            # Detect health
            health_result = await self._detect_health(screenshot)
            if health_result:
                stats.health = health_result[0]
                confidences.append(health_result[1])
            
            # Detect level
            level_result = await self._detect_level(screenshot)
            if level_result:
                stats.level = level_result[0]
                confidences.append(level_result[1])
            
            # Detect stage/round
            stage_result = await self._detect_stage_round(screenshot)
            if stage_result:
                stats.stage = stage_result[0]
                stats.round_num = stage_result[1]
                confidences.append(stage_result[2])
            
            # Calculate overall confidence
            stats.confidence = np.mean(confidences) if confidences else 0.0
            stats.timestamp = self._get_timestamp()
            
            # Validate and smooth stats
            stats = self._validate_stats(stats)
            
            # Update tracking
            self.last_stats = stats
            self._update_history(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error detecting stats: {e}")
            return None
    
    async def _detect_gold(self, screenshot: np.ndarray) -> Optional[Tuple[int, float]]:
        """Detect gold amount"""
        try:
            region = self.regions['gold']
            gold_img = ImageUtils.crop_region(screenshot, region)
            
            # Preprocess for better OCR
            processed = self._preprocess_number_region(gold_img)
            
            # OCR detection
            result = await self.text_recognizer.recognize_numbers(processed)
            
            if result and result.confidence > self.ocr_confidence:
                gold = self._parse_gold(result.text)
                if gold is not None:
                    return gold, result.confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting gold: {e}")
            return None
    
    async def _detect_health(self, screenshot: np.ndarray) -> Optional[Tuple[int, float]]:
        """Detect health amount"""
        try:
            region = self.regions['health']
            health_img = ImageUtils.crop_region(screenshot, region)
            
            # Preprocess for better OCR
            processed = self._preprocess_number_region(health_img)
            
            # OCR detection
            result = await self.text_recognizer.recognize_numbers(processed)
            
            if result and result.confidence > self.ocr_confidence:
                health = self._parse_health(result.text)
                if health is not None:
                    return health, result.confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting health: {e}")
            return None
    
    async def _detect_level(self, screenshot: np.ndarray) -> Optional[Tuple[int, float]]:
        """Detect player level"""
        try:
            region = self.regions['level']
            level_img = ImageUtils.crop_region(screenshot, region)
            
            # Preprocess for better OCR
            processed = self._preprocess_number_region(level_img)
            
            # OCR detection
            result = await self.text_recognizer.recognize_numbers(processed)
            
            if result and result.confidence > self.ocr_confidence:
                level = self._parse_level(result.text)
                if level is not None:
                    return level, result.confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting level: {e}")
            return None
    
    async def _detect_stage_round(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Detect stage and round"""
        try:
            region = self.regions['stage']
            stage_img = ImageUtils.crop_region(screenshot, region)
            
            # Preprocess for better OCR
            processed = self._preprocess_text_region(stage_img)
            
            # OCR detection
            result = await self.text_recognizer.recognize_text(processed)
            
            if result and result.confidence > self.ocr_confidence:
                stage_round = self._parse_stage_round(result.text)
                if stage_round:
                    return stage_round[0], stage_round[1], result.confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting stage/round: {e}")
            return None
    
    def _is_in_game(self, screenshot: np.ndarray) -> bool:
        """Check if player is currently in a game"""
        try:
            # Look for game UI elements
            # Check for gold region
            gold_region = self.regions['gold']
            gold_img = ImageUtils.crop_region(screenshot, gold_region)
            
            # Check if the region contains typical game UI colors
            # Gold UI typically has golden/yellow colors
            hsv = cv2.cvtColor(gold_img, cv2.COLOR_BGR2HSV)
            
            # Define golden color range
            lower_gold = np.array([15, 100, 100])
            upper_gold = np.array([35, 255, 255])
            
            mask = cv2.inRange(hsv, lower_gold, upper_gold)
            gold_pixels = np.sum(mask > 0)
            
            # If we have enough golden pixels, likely in game
            total_pixels = gold_img.shape[0] * gold_img.shape[1]
            gold_ratio = gold_pixels / total_pixels
            
            return gold_ratio > 0.1  # At least 10% golden pixels
            
        except Exception as e:
            logger.debug(f"Error checking game state: {e}")
            return False
    
    def _preprocess_number_region(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image region for number OCR"""
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Increase contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Resize for better OCR
            height, width = cleaned.shape
            if height < 30:
                scale = 30 / height
                new_width = int(width * scale)
                cleaned = cv2.resize(cleaned, (new_width, 30), interpolation=cv2.INTER_CUBIC)
            
            return cleaned
            
        except Exception as e:
            logger.debug(f"Error preprocessing number region: {e}")
            return img
    
    def _preprocess_text_region(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image region for text OCR"""
        try:
            # Similar to number preprocessing but more gentle
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Mild contrast enhancement
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=5)
            
            # Apply adaptive threshold for varying lighting
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Resize for better OCR
            height, width = thresh.shape
            if height < 25:
                scale = 25 / height
                new_width = int(width * scale)
                thresh = cv2.resize(thresh, (new_width, 25), interpolation=cv2.INTER_CUBIC)
            
            return thresh
            
        except Exception as e:
            logger.debug(f"Error preprocessing text region: {e}")
            return img
    
    def _parse_gold(self, text: str) -> Optional[int]:
        """Parse gold amount from OCR text"""
        try:
            # Remove non-numeric characters except commas
            cleaned = ''.join(c for c in text if c.isdigit() or c == ',')
            if not cleaned:
                return None
            
            # Remove commas and convert to int
            gold = int(cleaned.replace(',', ''))
            
            # Validate reasonable range (0-999)
            if 0 <= gold <= 999:
                return gold
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    def _parse_health(self, text: str) -> Optional[int]:
        """Parse health amount from OCR text"""
        try:
            # Remove non-numeric characters
            cleaned = ''.join(c for c in text if c.isdigit())
            if not cleaned:
                return None
            
            health = int(cleaned)
            
            # Validate reasonable range (0-100)
            if 0 <= health <= 100:
                return health
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    def _parse_level(self, text: str) -> Optional[int]:
        """Parse level from OCR text"""
        try:
            # Remove non-numeric characters
            cleaned = ''.join(c for c in text if c.isdigit())
            if not cleaned:
                return None
            
            level = int(cleaned)
            
            # Validate reasonable range (1-9)
            if 1 <= level <= 9:
                return level
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    def _parse_stage_round(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse stage and round from OCR text (e.g., '2-3' or 'Stage 2-3')"""
        try:
            import re
            
            # Look for pattern like "2-3" or "Stage 2-3"
            pattern = r'(\d+)-(\d+)'
            match = re.search(pattern, text)
            
            if match:
                stage = int(match.group(1))
                round_num = int(match.group(2))
                
                # Validate reasonable ranges
                if 1 <= stage <= 7 and 1 <= round_num <= 7:
                    return stage, round_num
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    def _validate_stats(self, stats: GameStats) -> GameStats:
        """Validate and smooth statistics using history"""
        try:
            # If we have previous stats, use them to validate current ones
            if self.last_stats and self.last_stats.in_game:
                # Gold shouldn't change drastically in one frame - INCREASE THRESHOLD
                if abs(stats.gold - self.last_stats.gold) > 100:  # Changed from 50 to 100
                    # If change is too large, use smoothed value
                    stats.gold = self._smooth_value(stats.gold, self.last_stats.gold, 0.5)  # Changed from 0.7 to 0.5

                # Health should only decrease or stay same (except healing items)
                if stats.health > self.last_stats.health + 10:  # Changed from 5 to 10
                    stats.health = self.last_stats.health

                # Level should only increase or stay same
                if stats.level < self.last_stats.level:
                    stats.level = self.last_stats.level

                # REMOVE OVERLY STRICT STAGE/ROUND VALIDATION
                # Comment out or remove these lines:
                # if (stats.stage < self.last_stats.stage or 
                #     (stats.stage == self.last_stats.stage and stats.round_num < self.last_stats.round_num)):
                #     stats.stage = self.last_stats.stage
                #     stats.round_num = self.last_stats.round_num

            return stats
        
        except Exception as e:
            logger.debug(f"Error validating stats: {e}")
            return stats
    
    def _smooth_value(self, new_value: int, old_value: int, alpha: float) -> int:
        """Exponential smoothing for numeric values"""
        return int(alpha * new_value + (1 - alpha) * old_value)
    
    def _update_history(self, stats: GameStats):
        """Update detection history for analysis"""
        self.detection_history.append(stats)
        
        # Keep only last 100 entries
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def get_default_regions(self, screen_width: int, screen_height: int) -> Dict[str, Dict[str, int]]:
        """Get default regions based on screen resolution"""
        try:
            # Calculate regions based on common TFT UI layouts
            # These are approximate positions that work for most resolutions
            
            if screen_width >= 1920:  # 1920x1080 or higher
                return {
                    "board": {"x": 560, "y": 150, "w": 800, "h": 600},
                    "shop": {"x": 560, "y": 850, "w": 800, "h": 120},
                    "gold": {"x": 1520, "y": 850, "w": 80, "h": 30},
                    "health": {"x": 100, "y": 100, "w": 60, "h": 25},
                    "level": {"x": 500, "y": 850, "w": 50, "h": 25},
                    "stage": {"x": 960, "y": 20, "w": 120, "h": 30}
                }
            elif screen_width >= 1366:  # 1366x768
                return {
                    "board": {"x": 400, "y": 100, "w": 566, "h": 425},
                    "shop": {"x": 400, "y": 600, "w": 566, "h": 85},
                    "gold": {"x": 1070, "y": 600, "w": 60, "h": 25},
                    "health": {"x": 70, "y": 70, "w": 50, "h": 20},
                    "level": {"x": 350, "y": 600, "w": 40, "h": 20},
                    "stage": {"x": 683, "y": 15, "w": 85, "h": 25}
                }
            else:  # Lower resolutions
                scale = screen_width / 1920
                return {
                    "board": {"x": int(560 * scale), "y": int(150 * scale), 
                             "w": int(800 * scale), "h": int(600 * scale)},
                    "shop": {"x": int(560 * scale), "y": int(850 * scale), 
                            "w": int(800 * scale), "h": int(120 * scale)},
                    "gold": {"x": int(1520 * scale), "y": int(850 * scale), 
                            "w": int(80 * scale), "h": int(30 * scale)},
                    "health": {"x": int(100 * scale), "y": int(100 * scale), 
                              "w": int(60 * scale), "h": int(25 * scale)},
                    "level": {"x": int(500 * scale), "y": int(850 * scale), 
                             "w": int(50 * scale), "h": int(25 * scale)},
                    "stage": {"x": int(960 * scale), "y": int(20 * scale), 
                             "w": int(120 * scale), "h": int(30 * scale)}
                }
                
        except Exception as e:
            logger.error(f"Error calculating default regions: {e}")
            return self.regions
    
    def get_detection_accuracy(self) -> Dict[str, float]:
        """Get accuracy metrics for recent detections"""
        try:
            if len(self.detection_history) < 10:
                return {"insufficient_data": True}
            
            recent_stats = self.detection_history[-50:]  # Last 50 detections
            
            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in recent_stats if s.confidence > 0])
            
            # Calculate consistency (how often values stay the same or change logically)
            consistency_scores = []
            for i in range(1, len(recent_stats)):
                prev = recent_stats[i-1]
                curr = recent_stats[i]
                
                # Score based on logical changes
                score = 0
                if curr.gold >= prev.gold - 10:  # Gold can decrease slightly due to interest
                    score += 1
                if curr.health <= prev.health:  # Health should not increase unexpectedly
                    score += 1
                if curr.level >= prev.level:  # Level should not decrease
                    score += 1
                
                consistency_scores.append(score / 3)
            
            avg_consistency = np.mean(consistency_scores)
            
            return {
                "average_confidence": float(avg_confidence),
                "consistency_score": float(avg_consistency),
                "total_detections": len(recent_stats),
                "in_game_rate": sum(1 for s in recent_stats if s.in_game) / len(recent_stats)
            }
            
        except Exception as e:
            logger.error(f"Error calculating detection accuracy: {e}")
            return {"error": str(e)}