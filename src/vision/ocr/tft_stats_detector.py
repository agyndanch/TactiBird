"""
TFT Economy Overlay - Enhanced OCR for Gold/Health Detection
"""

import cv2
import numpy as np
import pytesseract
import logging
import re
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GameStats:
    """Container for detected game statistics"""
    gold: Optional[int] = None
    health: Optional[int] = None
    level: Optional[int] = None
    stage: Optional[int] = None
    round_num: Optional[int] = None
    confidence: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence is None:
            self.confidence = {}

class TFTStatsDetector:
    """Enhanced OCR detector for TFT game statistics"""
    
    def __init__(self):
        # OCR patterns for different stats
        self.patterns = {
            'gold': [
                r'(\d+)\s*[gG]',  # "50g", "50 g"
                r'[gG][oO][lL][dD]?\s*[:\-]?\s*(\d+)',  # "Gold: 50", "Gold-50"
                r'(\d+)\s*[gG][oO][lL][dD]?',  # "50gold", "50 Gold"
            ],
            'health': [
                r'(\d+)\s*[hH][pP]?',  # "40hp", "40 HP", "40h"
                r'[hH][eE][aA][lL][tT][hH]?\s*[:\-]?\s*(\d+)',  # "Health: 40"
                r'(\d+)\s*/\s*100',  # "40/100"
            ],
            'level': [
                r'[lL][vV][lL]?\s*[:\-]?\s*(\d+)',  # "Lvl: 5", "Level 5"
                r'[lL][eE][vV][eE][lL]\s*(\d+)',  # "Level 5"
            ],
            'stage': [
                r'(\d+)\-(\d+)',  # "3-2" format
                r'[sS][tT][aA][gG][eE]\s*(\d+)',  # "Stage 3"
            ]
        }
        
        # Valid ranges for validation
        self.valid_ranges = {
            'gold': (0, 200),  # TFT gold range
            'health': (0, 100),  # TFT health range  
            'level': (1, 11),   # TFT level range
            'stage': (1, 7),    # TFT stage range
            'round': (1, 7)     # TFT round range
        }
        
        # Common OCR misreads
        self.ocr_corrections = {
            'O': '0', 'o': '0', 'I': '1', 'l': '1', 'S': '5', 
            'B': '8', 'G': '6', 'Z': '2', 'A': '4'
        }
    
    def detect_stats(self, screenshot: np.ndarray, regions: Dict[str, Tuple[int, int, int, int]]) -> GameStats:
        """
        Detect game statistics from screenshot
        
        Args:
            screenshot: Full screenshot image
            regions: Dictionary of regions {stat_name: (x, y, w, h)}
        
        Returns:
            GameStats object with detected values
        """
        stats = GameStats()
        
        for stat_name, region in regions.items():
            try:
                # Extract region from screenshot
                x, y, w, h = region
                roi = screenshot[y:y+h, x:x+w]
                
                if stat_name == 'gold':
                    value, confidence = self._detect_gold(roi)
                    stats.gold = value
                    stats.confidence['gold'] = confidence
                    
                elif stat_name == 'health':
                    value, confidence = self._detect_health(roi)
                    stats.health = value
                    stats.confidence['health'] = confidence
                    
                elif stat_name == 'level':
                    value, confidence = self._detect_level(roi)
                    stats.level = value
                    stats.confidence['level'] = confidence
                    
                elif stat_name == 'stage_round':
                    stage, round_num, confidence = self._detect_stage_round(roi)
                    stats.stage = stage
                    stats.round_num = round_num
                    stats.confidence['stage_round'] = confidence
                    
            except Exception as e:
                logger.error(f"Error detecting {stat_name}: {e}")
                stats.confidence[stat_name] = 0.0
        
        return stats
    
    def _detect_gold(self, roi: np.ndarray) -> Tuple[Optional[int], float]:
        """Detect gold amount from region of interest"""
        try:
            # Preprocess image for better OCR
            processed = self._preprocess_gold_image(roi)
            
            # Try multiple OCR configurations
            configs = [
                '--psm 7 -c tessedit_char_whitelist=0123456789gGold ',
                '--psm 8 -c tessedit_char_whitelist=0123456789gGold ',
                '--psm 13 -c tessedit_char_whitelist=0123456789gGold '
            ]
            
            best_value = None
            best_confidence = 0.0
            
            for config in configs:
                try:
                    # Extract text with confidence
                    data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                    text = ' '.join([data['text'][i] for i in range(len(data['text'])) if data['conf'][i] > 30])
                    
                    # Parse gold value
                    value = self._parse_gold_value(text)
                    if value is not None:
                        confidence = np.mean([conf for conf in data['conf'] if conf > 0])
                        if confidence > best_confidence:
                            best_value = value
                            best_confidence = confidence
                            
                except Exception as e:
                    logger.debug(f"OCR config failed: {e}")
                    continue
            
            return best_value, best_confidence / 100.0
            
        except Exception as e:
            logger.error(f"Gold detection failed: {e}")
            return None, 0.0
    
    def _detect_health(self, roi: np.ndarray) -> Tuple[Optional[int], float]:
        """Detect health amount from region of interest"""
        try:
            # Preprocess for health (often red text)
            processed = self._preprocess_health_image(roi)
            
            configs = [
                '--psm 7 -c tessedit_char_whitelist=0123456789hHPealte/ ',
                '--psm 8 -c tessedit_char_whitelist=0123456789hHPealte/ '
            ]
            
            best_value = None
            best_confidence = 0.0
            
            for config in configs:
                try:
                    data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                    text = ' '.join([data['text'][i] for i in range(len(data['text'])) if data['conf'][i] > 30])
                    
                    value = self._parse_health_value(text)
                    if value is not None:
                        confidence = np.mean([conf for conf in data['conf'] if conf > 0])
                        if confidence > best_confidence:
                            best_value = value
                            best_confidence = confidence
                            
                except Exception as e:
                    logger.debug(f"Health OCR config failed: {e}")
                    continue
            
            return best_value, best_confidence / 100.0
            
        except Exception as e:
            logger.error(f"Health detection failed: {e}")
            return None, 0.0
    
    def _detect_level(self, roi: np.ndarray) -> Tuple[Optional[int], float]:
        """Detect player level"""
        try:
            processed = self._preprocess_text_image(roi)
            
            text = pytesseract.image_to_string(processed, config='--psm 7 -c tessedit_char_whitelist=0123456789LvlLevel ')
            value = self._parse_level_value(text)
            
            # Rough confidence based on successful parsing
            confidence = 0.8 if value is not None else 0.0
            
            return value, confidence
            
        except Exception as e:
            logger.error(f"Level detection failed: {e}")
            return None, 0.0
    
    def _detect_stage_round(self, roi: np.ndarray) -> Tuple[Optional[int], Optional[int], float]:
        """Detect stage and round (e.g., '3-2')"""
        try:
            processed = self._preprocess_text_image(roi)
            
            text = pytesseract.image_to_string(processed, config='--psm 7 -c tessedit_char_whitelist=0123456789-Stage ')
            stage, round_num = self._parse_stage_round(text)
            
            confidence = 0.8 if stage is not None and round_num is not None else 0.0
            
            return stage, round_num, confidence
            
        except Exception as e:
            logger.error(f"Stage/round detection failed: {e}")
            return None, None, 0.0
    
    def _preprocess_gold_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for gold detection"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Upscale for better OCR
            scale_factor = 3
            height, width = gray.shape
            gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
            
            # Enhance yellow/gold colored text
            # Gold text is often yellow on dark background
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Gold image preprocessing failed: {e}")
            return image
    
    def _preprocess_health_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for health detection"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Upscale
            scale_factor = 3
            height, width = gray.shape
            gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
            
            # Health is often red text - enhance contrast
            gray = cv2.equalizeHist(gray)
            
            # Binary threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Health image preprocessing failed: {e}")
            return image
    
    def _preprocess_text_image(self, image: np.ndarray) -> np.ndarray:
        """General text preprocessing"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Upscale
            scale_factor = 2
            height, width = gray.shape
            gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
            
            # Denoise
            gray = cv2.medianBlur(gray, 3)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return image
    
    def _parse_gold_value(self, text: str) -> Optional[int]:
        """Parse gold value from OCR text"""
        try:
            # Clean up text
            cleaned_text = self._correct_ocr_text(text)
            
            # Try each gold pattern
            for pattern in self.patterns['gold']:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
                    if self.valid_ranges['gold'][0] <= value <= self.valid_ranges['gold'][1]:
                        return value
            
            # Fallback: look for any reasonable number
            numbers = re.findall(r'\d+', cleaned_text)
            for num_str in numbers:
                value = int(num_str)
                if self.valid_ranges['gold'][0] <= value <= self.valid_ranges['gold'][1]:
                    return value
            
            return None
            
        except Exception as e:
            logger.debug(f"Gold parsing failed for '{text}': {e}")
            return None
    
    def _parse_health_value(self, text: str) -> Optional[int]:
        """Parse health value from OCR text"""
        try:
            cleaned_text = self._correct_ocr_text(text)
            
            for pattern in self.patterns['health']:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
                    if self.valid_ranges['health'][0] <= value <= self.valid_ranges['health'][1]:
                        return value
            
            # Fallback
            numbers = re.findall(r'\d+', cleaned_text)
            for num_str in numbers:
                value = int(num_str)
                if self.valid_ranges['health'][0] <= value <= self.valid_ranges['health'][1]:
                    return value
            
            return None
            
        except Exception as e:
            logger.debug(f"Health parsing failed for '{text}': {e}")
            return None
    
    def _parse_level_value(self, text: str) -> Optional[int]:
        """Parse level value from OCR text"""
        try:
            cleaned_text = self._correct_ocr_text(text)
            
            for pattern in self.patterns['level']:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
                    if self.valid_ranges['level'][0] <= value <= self.valid_ranges['level'][1]:
                        return value
            
            return None
            
        except Exception as e:
            logger.debug(f"Level parsing failed for '{text}': {e}")
            return None
    
    def _parse_stage_round(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse stage and round from text like '3-2'"""
        try:
            cleaned_text = self._correct_ocr_text(text)
            
            # Look for X-Y format
            match = re.search(r'(\d+)\-(\d+)', cleaned_text)
            if match:
                stage = int(match.group(1))
                round_num = int(match.group(2))
                
                if (self.valid_ranges['stage'][0] <= stage <= self.valid_ranges['stage'][1] and
                    self.valid_ranges['round'][0] <= round_num <= self.valid_ranges['round'][1]):
                    return stage, round_num
            
            return None, None
            
        except Exception as e:
            logger.debug(f"Stage/round parsing failed for '{text}': {e}")
            return None, None
    
    def _correct_ocr_text(self, text: str) -> str:
        """Apply common OCR corrections"""
        result = text
        for wrong, correct in self.ocr_corrections.items():
            result = result.replace(wrong, correct)
        return result
    
    def get_default_regions(self, screenshot_width: int, screenshot_height: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Get default detection regions based on common TFT UI positions"""
        return {
            'gold': (
                int(screenshot_width * 0.02),   # x: left side
                int(screenshot_height * 0.88),  # y: bottom area
                int(screenshot_width * 0.12),   # w: small width
                int(screenshot_height * 0.06)   # h: small height
            ),
            'health': (
                int(screenshot_width * 0.02),   # x: left side, near gold
                int(screenshot_height * 0.82),  # y: slightly above gold
                int(screenshot_width * 0.12),   # w: small width
                int(screenshot_height * 0.06)   # h: small height
            ),
            'level': (
                int(screenshot_width * 0.02),   # x: left side
                int(screenshot_height * 0.76),  # y: above health
                int(screenshot_width * 0.12),   # w: small width
                int(screenshot_height * 0.06)   # h: small height
            ),
            'stage_round': (
                int(screenshot_width * 0.45),   # x: center-ish
                int(screenshot_height * 0.02),  # y: top of screen
                int(screenshot_width * 0.10),   # w: medium width
                int(screenshot_height * 0.06)   # h: small height
            )
        }