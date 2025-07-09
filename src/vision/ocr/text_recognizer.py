"""
TactiBird Overlay - Text Recognizer (OCR)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available, OCR functionality will be limited")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextRecognizer:
    """Handles text recognition (OCR) for game UI elements"""
    
    def __init__(self, engine: str = "tesseract"):
        self.engine = engine
        self.easyocr_reader = None
        
        # Initialize OCR engines
        self._init_engines()
        
        # Text processing patterns
        self.number_pattern = re.compile(r'\d+')
        self.gold_pattern = re.compile(r'(\d+)\s*g', re.IGNORECASE)
        self.health_pattern = re.compile(r'(\d+)\s*hp', re.IGNORECASE)
        self.level_pattern = re.compile(r'level\s*(\d+)', re.IGNORECASE)
        
        logger.info(f"Text recognizer initialized with engine: {engine}")
    
    def _init_engines(self):
        """Initialize OCR engines"""
        if self.engine == "easyocr" and EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.engine = "tesseract"
        
        if self.engine == "tesseract" and not TESSERACT_AVAILABLE:
            logger.error("Tesseract not available, text recognition will not work")
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> str:
        """Extract text from image"""
        try:
            if preprocess:
                image = self._preprocess_for_ocr(image)
            
            if self.engine == "easyocr" and self.easyocr_reader:
                return self._extract_with_easyocr(image)
            elif self.engine == "tesseract" and TESSERACT_AVAILABLE:
                return self._extract_with_tesseract(image)
            else:
                logger.warning("No OCR engine available")
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def _extract_with_tesseract(self, image: np.ndarray) -> str:
        """Extract text using Tesseract"""
        try:
            # Configure Tesseract for better number recognition
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /'
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""
    
    def _extract_with_easyocr(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image)
            text_parts = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    text_parts.append(text)
            
            return ' '.join(text_parts)
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize if too small
            height, width = gray.shape
            if height < 30 or width < 30:
                scale_factor = max(30 / height, 30 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply threshold to get black text on white background
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def extract_number(self, image: np.ndarray) -> Optional[int]:
        """Extract a single number from image"""
        text = self.extract_text(image)
        match = self.number_pattern.search(text)
        
        if match:
            try:
                return int(match.group())
            except ValueError:
                pass
        
        return None
    
    def extract_gold_value(self, image: np.ndarray) -> Optional[int]:
        """Extract gold value from image"""
        text = self.extract_text(image)
        
        # Try gold pattern first
        match = self.gold_pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        # Fallback to any number
        return self.extract_number(image)
    
    def extract_health_value(self, image: np.ndarray) -> Optional[int]:
        """Extract health value from image"""
        text = self.extract_text(image)
        
        # Try health pattern first
        match = self.health_pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        # Fallback to any number
        return self.extract_number(image)
    
    def extract_level_value(self, image: np.ndarray) -> Optional[int]:
        """Extract level value from image"""
        text = self.extract_text(image)
        
        # Try level pattern first
        match = self.level_pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        # Fallback to any number
        return self.extract_number(image)
    
    def extract_stage_round(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Extract stage and round from image (e.g., "2-1")"""
        text = self.extract_text(image)
        
        # Look for pattern like "2-1" or "2 1"
        stage_round_pattern = re.compile(r'(\d+)[-\s](\d+)')
        match = stage_round_pattern.search(text)
        
        if match:
            try:
                stage = int(match.group(1))
                round_num = int(match.group(2))
                return (stage, round_num)
            except ValueError:
                pass
        
        return None
    
    def extract_champion_names(self, image: np.ndarray) -> List[str]:
        """Extract champion names from image"""
        text = self.extract_text(image)
        
        # This would need a comprehensive list of champion names
        # For now, return words that look like champion names
        words = text.split()
        champion_names = []
        
        for word in words:
            # Filter out obvious non-champion words
            if len(word) >= 3 and word.isalpha() and word[0].isupper():
                champion_names.append(word)
        
        return champion_names
    
    def extract_trait_info(self, image: np.ndarray) -> List[Dict[str, any]]:
        """Extract trait information from image"""
        text = self.extract_text(image)
        traits = []
        
        # Look for trait patterns like "Assassin (3/3)" or "Academy 2/4"
        trait_pattern = re.compile(r'([A-Za-z]+)\s*[(\[]?(\d+)[/](\d+)[)\]]?')
        
        for match in trait_pattern.finditer(text):
            trait_name = match.group(1)
            current_count = int(match.group(2))
            required_count = int(match.group(3))
            
            traits.append({
                'name': trait_name,
                'current': current_count,
                'required': required_count,
                'active': current_count >= required_count
            })
        
        return traits
    
    def is_text_present(self, image: np.ndarray, target_text: str, 
                       confidence_threshold: float = 0.8) -> bool:
        """Check if specific text is present in image"""
        extracted_text = self.extract_text(image).lower()
        target_text = target_text.lower()
        
        # Simple substring check
        if target_text in extracted_text:
            return True
        
        # Fuzzy matching for OCR errors
        return self._fuzzy_match(extracted_text, target_text, confidence_threshold)
    
    def _fuzzy_match(self, text1: str, text2: str, threshold: float) -> bool:
        """Simple fuzzy matching for OCR errors"""
        # Calculate simple similarity based on character overlap
        if not text1 or not text2:
            return False
        
        # Remove spaces and normalize
        text1 = re.sub(r'\s+', '', text1)
        text2 = re.sub(r'\s+', '', text2)
        
        if len(text2) == 0:
            return False
        
        # Count matching characters
        matches = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        similarity = matches / max(len(text1), len(text2))
        
        return similarity >= threshold
    
    def get_text_confidence(self, image: np.ndarray) -> float:
        """Get confidence score for OCR results"""
        try:
            if self.engine == "tesseract" and TESSERACT_AVAILABLE:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                return sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            elif self.engine == "easyocr" and self.easyocr_reader:
                results = self.easyocr_reader.readtext(image)
                if results:
                    confidences = [conf for (_, _, conf) in results]
                    return sum(confidences) / len(confidences)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def debug_save_processed_image(self, image: np.ndarray, filename: str):
        """Save preprocessed image for debugging"""
        try:
            processed = self._preprocess_for_ocr(image)
            cv2.imwrite(filename, processed)
            logger.info(f"Saved processed image: {filename}")
        except Exception as e:
            logger.error(f"Failed to save processed image: {e}")