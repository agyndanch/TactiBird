"""
TactiBird Overlay - Trait Detection
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from src.data.models.game_state import Trait
from src.vision.templates.template_matcher import TemplateMatcher
from src.vision.ocr.text_recognizer import TextRecognizer
from src.vision.utils.image_utils import preprocess_image

logger = logging.getLogger(__name__)

@dataclass
class TraitDetection:
    """Represents a detected trait"""
    trait: Trait
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    text_region: Optional[np.ndarray] = None

class TraitDetector:
    """Detects traits/synergies in TFT screenshots"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.template_matcher = TemplateMatcher()
        self.text_recognizer = TextRecognizer()
        
        # Detection settings
        self.confidence_threshold = 0.7
        self.text_confidence_threshold = 0.6
        
        # Trait visual indicators
        self.trait_colors = {
            'bronze': [(15, 82, 138), (25, 255, 255)],  # HSV range for bronze
            'silver': [(0, 0, 180), (180, 30, 255)],    # HSV range for silver
            'gold': [(15, 100, 200), (35, 255, 255)],   # HSV range for gold
            'prismatic': [(140, 100, 100), (170, 255, 255)]  # HSV range for prismatic
        }
        
        logger.info("Trait detector initialized")
    
    def detect_traits_in_panel(self, traits_panel: np.ndarray) -> List[TraitDetection]:
        """Detect active traits in the traits panel"""
        try:
            detections = []
            
            # Preprocess panel image
            processed = preprocess_image(traits_panel, enhance_contrast=True, denoise=True)
            
            # Segment panel into individual trait regions
            trait_regions = self._segment_traits_panel(processed)
            
            for i, region in enumerate(trait_regions):
                if region.size == 0:
                    continue
                
                # Try template matching first
                trait_detection = self._detect_trait_by_template(region)
                
                if not trait_detection:
                    # Try OCR-based detection
                    trait_detection = self._detect_trait_by_ocr(region)
                
                if trait_detection:
                    # Adjust bounding box to panel coordinates
                    x, y, w, h = trait_detection.bounding_box
                    region_y_offset = i * (traits_panel.shape[0] // len(trait_regions))
                    adjusted_bbox = (x, y + region_y_offset, w, h)
                    trait_detection.bounding_box = adjusted_bbox
                    
                    detections.append(trait_detection)
            
            logger.debug(f"Detected {len(detections)} traits in panel")
            return detections
            
        except Exception as e:
            logger.error(f"Trait panel detection failed: {e}")
            return []
    
    def detect_trait_from_champions(self, champions: List) -> List[Trait]:
        """Detect traits from champion composition"""
        try:
            trait_counts = {}
            
            # Count traits from all champions
            for champion in champions:
                champion_traits = self.data_manager.get_champion_traits(champion.name)
                
                for trait_name in champion_traits:
                    trait_counts[trait_name] = trait_counts.get(trait_name, 0) + 1
            
            # Create trait objects
            detected_traits = []
            for trait_name, count in trait_counts.items():
                trait_data = self.data_manager.get_trait_data(trait_name)
                breakpoints = self.data_manager.get_trait_breakpoints(trait_name)
                
                # Determine active level
                active_level = 0
                is_active = False
                for i, breakpoint in enumerate(sorted(breakpoints)):
                    if count >= breakpoint:
                        active_level = i + 1
                        is_active = True
                    else:
                        break
                
                trait = Trait(
                    name=trait_name,
                    current_count=count,
                    required_counts=breakpoints,
                    is_active=is_active,
                    active_level=active_level
                )
                
                detected_traits.append(trait)
            
            return detected_traits
            
        except Exception as e:
            logger.error(f"Trait detection from champions failed: {e}")
            return []
    
    def _segment_traits_panel(self, panel_image: np.ndarray) -> List[np.ndarray]:
        """Segment traits panel into individual trait regions"""
        try:
            height, width = panel_image.shape[:2]
            regions = []
            
            # Traits are typically displayed vertically
            # Estimate number of visible traits (usually 6-10)
            max_traits = 10
            region_height = height // max_traits
            
            for i in range(max_traits):
                y_start = i * region_height
                y_end = min((i + 1) * region_height, height)
                
                if y_end > y_start:
                    region = panel_image[y_start:y_end, 0:width]
                    
                    # Check if region contains content
                    if self._has_trait_content(region):
                        regions.append(region)
                    else:
                        regions.append(np.array([]))  # Empty region
                else:
                    break
            
            return regions
            
        except Exception as e:
            logger.error(f"Traits panel segmentation failed: {e}")
            return []
    
    def _has_trait_content(self, region: np.ndarray) -> bool:
        """Check if region contains trait content"""
        try:
            if region.size == 0:
                return False
            
            # Convert to grayscale for analysis
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Check for text-like patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Regions with traits should have some edge content
            return edge_density > 0.02
            
        except Exception:
            return False
    
    def _detect_trait_by_template(self, region: np.ndarray) -> Optional[TraitDetection]:
        """Detect trait using template matching"""
        try:
            trait_templates = self.template_matcher.get_trait_templates()
            
            best_match = None
            best_confidence = 0.0
            
            for trait_name, template in trait_templates.items():
                matches = self.template_matcher.match_template(
                    region, template, threshold=self.confidence_threshold
                )
                
                if matches and matches[0]['confidence'] > best_confidence:
                    best_confidence = matches[0]['confidence']
                    best_match = (trait_name, matches[0])
            
            if best_match:
                trait_name, match = best_match
                
                # Try to extract count information from nearby text
                trait_counts = self._extract_trait_counts_from_region(region)
                
                trait_data = self.data_manager.get_trait_data(trait_name)
                breakpoints = self.data_manager.get_trait_breakpoints(trait_name)
                
                current_count = trait_counts.get('current', 0)
                required_count = trait_counts.get('required', breakpoints[0] if breakpoints else 3)
                
                # Determine if trait is active
                is_active = current_count >= required_count
                active_level = 0
                if is_active:
                    for i, bp in enumerate(sorted(breakpoints)):
                        if current_count >= bp:
                            active_level = i + 1
                
                trait = Trait(
                    name=trait_name,
                    current_count=current_count,
                    required_counts=breakpoints,
                    is_active=is_active,
                    active_level=active_level
                )
                
                return TraitDetection(
                    trait=trait,
                    confidence=match['confidence'],
                    bounding_box=match['bbox'],
                    text_region=region
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Template-based trait detection failed: {e}")
            return None
    
    def _detect_trait_by_ocr(self, region: np.ndarray) -> Optional[TraitDetection]:
        """Detect trait using OCR"""
        try:
            # Extract text from region
            text = self.text_recognizer.extract_text(region)
            
            if not text.strip():
                return None
            
            # Parse trait information from text
            trait_info = self._parse_trait_text(text)
            
            if trait_info:
                trait_name = trait_info['name']
                current_count = trait_info.get('current', 0)
                required_count = trait_info.get('required', 0)
                
                # Get trait data
                trait_data = self.data_manager.get_trait_data(trait_name)
                breakpoints = self.data_manager.get_trait_breakpoints(trait_name)
                
                # Determine if trait is active
                is_active = current_count >= required_count
                active_level = 0
                if is_active:
                    for i, bp in enumerate(sorted(breakpoints)):
                        if current_count >= bp:
                            active_level = i + 1
                
                trait = Trait(
                    name=trait_name,
                    current_count=current_count,
                    required_counts=breakpoints,
                    is_active=is_active,
                    active_level=active_level
                )
                
                # Calculate confidence based on text recognition quality
                confidence = self.text_recognizer.get_text_confidence(region)
                
                return TraitDetection(
                    trait=trait,
                    confidence=confidence,
                    bounding_box=(0, 0, region.shape[1], region.shape[0]),
                    text_region=region
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"OCR-based trait detection failed: {e}")
            return None
    
    def _extract_trait_counts_from_region(self, region: np.ndarray) -> Dict[str, int]:
        """Extract trait count information from region"""
        try:
            text = self.text_recognizer.extract_text(region)
            return self._parse_trait_counts(text)
            
        except Exception as e:
            logger.debug(f"Trait count extraction failed: {e}")
            return {}
    
    def _parse_trait_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse trait information from OCR text"""
        try:
            import re
            
            # Look for patterns like "Assassin (3/3)" or "Academy 2/4"
            pattern = r'([A-Za-z]+)\s*[(\[]?(\d+)[/](\d+)[)\]]?'
            match = re.search(pattern, text)
            
            if match:
                trait_name = match.group(1).lower()
                current_count = int(match.group(2))
                required_count = int(match.group(3))
                
                return {
                    'name': trait_name,
                    'current': current_count,
                    'required': required_count
                }
            
            # Alternative pattern: just trait name
            trait_pattern = r'([A-Za-z]+)'
            trait_match = re.search(trait_pattern, text)
            
            if trait_match:
                trait_name = trait_match.group(1).lower()
                
                # Try to extract numbers separately
                numbers = re.findall(r'\d+', text)
                if len(numbers) >= 2:
                    return {
                        'name': trait_name,
                        'current': int(numbers[0]),
                        'required': int(numbers[1])
                    }
                elif len(numbers) == 1:
                    return {
                        'name': trait_name,
                        'current': int(numbers[0]),
                        'required': 0
                    }
                else:
                    return {
                        'name': trait_name,
                        'current': 0,
                        'required': 0
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Trait text parsing failed: {e}")
            return None
    
    def _parse_trait_counts(self, text: str) -> Dict[str, int]:
        """Parse trait count numbers from text"""
        try:
            import re
            
            # Look for "current/required" pattern
            count_match = re.search(r'(\d+)[/](\d+)', text)
            
            if count_match:
                return {
                    'current': int(count_match.group(1)),
                    'required': int(count_match.group(2))
                }
            
            # Look for standalone numbers
            numbers = re.findall(r'\d+', text)
            if len(numbers) >= 2:
                return {
                    'current': int(numbers[0]),
                    'required': int(numbers[1])
                }
            elif len(numbers) == 1:
                return {
                    'current': int(numbers[0]),
                    'required': 0
                }
            
            return {}
            
        except Exception:
            return {}
    
    def detect_trait_activation_level(self, region: np.ndarray) -> str:
        """Detect trait activation level by color"""
        try:
            if len(region.shape) == 3:
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            else:
                return 'inactive'
            
            # Check for different trait activation colors
            for level, (lower, upper) in self.trait_colors.items():
                lower_bound = np.array(lower)
                upper_bound = np.array(upper)
                
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                color_ratio = np.sum(mask > 0) / mask.size
                
                if color_ratio > 0.1:  # At least 10% of region has this color
                    return level
            
            return 'inactive'
            
        except Exception as e:
            logger.debug(f"Trait activation level detection failed: {e}")
            return 'inactive'
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get trait detection statistics"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'text_confidence_threshold': self.text_confidence_threshold,
            'available_templates': len(self.template_matcher.get_trait_templates()),
            'supported_activation_levels': list(self.trait_colors.keys())
        }
    
    def update_settings(self, **kwargs):
        """Update detection settings"""
        if 'confidence_threshold' in kwargs:
            self.confidence_threshold = kwargs['confidence_threshold']
        if 'text_confidence_threshold' in kwargs:
            self.text_confidence_threshold = kwargs['text_confidence_threshold']
        
        logger.info("Trait detection settings updated")