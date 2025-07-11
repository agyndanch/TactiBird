"""
TactiBird - Text Recognition Module
"""

import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR recognition result"""
    text: str = ""
    confidence: float = 0.0
    bbox: tuple = (0, 0, 0, 0)  # (x, y, w, h)

class TextRecognizer:
    """Text recognition using OCR"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config['vision']['ocr_confidence']
        
        # Try to import pytesseract
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.ocr_available = True
        except ImportError:
            logger.warning("Pytesseract not available, OCR disabled")
            self.ocr_available = False
    
    async def recognize_text(self, image: np.ndarray) -> Optional[OCRResult]:
        """Recognize text in image"""
        if not self.ocr_available:
            return None
        
        try:
            # Use pytesseract for text recognition
            text = self.pytesseract.image_to_string(image, config='--psm 7')
            
            # Get confidence data
            data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence
            )
            
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return None
    
    async def recognize_numbers(self, image: np.ndarray) -> Optional[OCRResult]:
        """Recognize numbers in image"""
        if not self.ocr_available:
            return None
        
        try:
            # Configure for numbers only
            config = '--psm 7 -c tessedit_char_whitelist=0123456789,'
            text = self.pytesseract.image_to_string(image, config=config)
            
            # Get confidence
            data = self.pytesseract.image_to_data(image, config=config, output_type=self.pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence
            )
            
        except Exception as e:
            logger.debug(f"Number OCR error: {e}")
            return None
    
    async def test_detection(self):
        """Test OCR detection capabilities"""
        print("Testing OCR detection...")
        
        if not self.ocr_available:
            print("❌ OCR not available - install pytesseract")
            return
        
        # Create test image with text
        test_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "Test 123", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        result = await self.recognize_text(test_img)
        if result:
            print(f"✅ OCR working - detected: '{result.text}' (confidence: {result.confidence:.2f})")
        else:
            print("❌ OCR test failed")