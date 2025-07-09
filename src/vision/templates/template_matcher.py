"""
TactiBird Overlay - Template Matcher
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

logger = logging.getLogger(__name__)

class TemplateMatcher:
    """Handles template matching for champion, item, and UI element detection"""
    
    def __init__(self):
        self.champion_templates = {}
        self.item_templates = {}
        self.ui_templates = {}
        self.trait_templates = {}
        
        # Template matching parameters
        self.match_threshold = 0.8
        self.scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # Multi-scale matching
        
        # Load templates
        self._load_templates()
        
        logger.info("Template matcher initialized")
    
    def _load_templates(self):
        """Load all template images"""
        try:
            self._load_champion_templates()
            self._load_item_templates()
            self._load_ui_templates()
            self._load_trait_templates()
            logger.info("Templates loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
    
    def _load_champion_templates(self):
        """Load champion template images"""
        champions_path = Path("data/templates/champions")
        if not champions_path.exists():
            logger.warning("Champion templates directory not found")
            return
        
        for template_file in champions_path.glob("*.png"):
            champion_name = template_file.stem
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
                if template is not None:
                    self.champion_templates[champion_name] = template
                    logger.debug(f"Loaded champion template: {champion_name}")
            except Exception as e:
                logger.error(f"Failed to load champion template {champion_name}: {e}")
    
    def _load_item_templates(self):
        """Load item template images"""
        items_path = Path("data/templates/items")
        if not items_path.exists():
            logger.warning("Item templates directory not found")
            return
        
        for template_file in items_path.glob("*.png"):
            item_name = template_file.stem
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
                if template is not None:
                    self.item_templates[item_name] = template
                    logger.debug(f"Loaded item template: {item_name}")
            except Exception as e:
                logger.error(f"Failed to load item template {item_name}: {e}")
    
    def _load_ui_templates(self):
        """Load UI element templates"""
        ui_path = Path("data/templates/ui")
        if not ui_path.exists():
            logger.warning("UI templates directory not found")
            return
        
        for template_file in ui_path.glob("*.png"):
            ui_element = template_file.stem
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
                if template is not None:
                    self.ui_templates[ui_element] = template
                    logger.debug(f"Loaded UI template: {ui_element}")
            except Exception as e:
                logger.error(f"Failed to load UI template {ui_element}: {e}")
    
    def _load_trait_templates(self):
        """Load trait template images"""
        traits_path = Path("data/templates/traits")
        if not traits_path.exists():
            logger.warning("Trait templates directory not found")
            return
        
        for template_file in traits_path.glob("*.png"):
            trait_name = template_file.stem
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
                if template is not None:
                    self.trait_templates[trait_name] = template
                    logger.debug(f"Loaded trait template: {trait_name}")
            except Exception as e:
                logger.error(f"Failed to load trait template {trait_name}: {e}")
    
    def match_template(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = None, method: int = cv2.TM_CCOEFF_NORMED,
                      multi_scale: bool = True) -> List[Dict[str, Any]]:
        """
        Match template in image with optional multi-scale matching
        
        Args:
            image: Source image to search in
            template: Template image to find
            threshold: Confidence threshold (default: self.match_threshold)
            method: OpenCV template matching method
            multi_scale: Whether to try multiple scales
            
        Returns:
            List of matches with confidence scores and locations
        """
        if threshold is None:
            threshold = self.match_threshold
        
        matches = []
        
        try:
            if multi_scale:
                # Try multiple scales
                for scale in self.scale_factors:
                    scaled_template = self._scale_template(template, scale)
                    scale_matches = self._match_single_scale(image, scaled_template, threshold, method)
                    
                    # Adjust coordinates for scale
                    for match in scale_matches:
                        match['scale'] = scale
                        match['bbox'] = self._adjust_bbox_for_scale(match['bbox'], scale)
                    
                    matches.extend(scale_matches)
            else:
                matches = self._match_single_scale(image, template, threshold, method)
            
            # Remove overlapping matches
            matches = self._remove_overlapping_matches(matches)
            
            # Sort by confidence
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
        
        return matches
    
    def _scale_template(self, template: np.ndarray, scale: float) -> np.ndarray:
        """Scale template image"""
        height, width = template.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def _match_single_scale(self, image: np.ndarray, template: np.ndarray, 
                           threshold: float, method: int) -> List[Dict[str, Any]]:
        """Match template at single scale"""
        matches = []
        
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            return matches
        
        # Perform template matching
        result = cv2.matchTemplate(image, template, method)
        
        # Find all matches above threshold
        locations = np.where(result >= threshold)
        
        template_h, template_w = template.shape[:2]
        
        for y, x in zip(locations[0], locations[1]):
            confidence = result[y, x]
            
            match = {
                'confidence': float(confidence),
                'bbox': (x, y, template_w, template_h),
                'center': (x + template_w // 2, y + template_h // 2),
                'scale': 1.0
            }
            matches.append(match)
        
        return matches
    
    def _adjust_bbox_for_scale(self, bbox: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
        """Adjust bounding box coordinates for scale"""
        x, y, w, h = bbox
        return (x, y, int(w / scale), int(h / scale))
    
    def _remove_overlapping_matches(self, matches: List[Dict[str, Any]], 
                                   overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Remove overlapping matches using Non-Maximum Suppression"""
        if len(matches) <= 1:
            return matches
        
        # Sort by confidence
        matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
        
        filtered_matches = []
        
        for current_match in matches:
            is_overlapping = False
            
            for existing_match in filtered_matches:
                if self._calculate_overlap(current_match['bbox'], existing_match['bbox']) > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_matches.append(current_match)
        
        return filtered_matches
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left >= right or top >= bottom:
            return 0.0
        
        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_champion(self, image: np.ndarray, champion_name: str, 
                     threshold: float = None) -> List[Dict[str, Any]]:
        """Find specific champion in image"""
        if champion_name not in self.champion_templates:
            logger.warning(f"Template not found for champion: {champion_name}")
            return []
        
        template = self.champion_templates[champion_name]
        return self.match_template(image, template, threshold)
    
    def find_item(self, image: np.ndarray, item_name: str, 
                 threshold: float = None) -> List[Dict[str, Any]]:
        """Find specific item in image"""
        if item_name not in self.item_templates:
            logger.warning(f"Template not found for item: {item_name}")
            return []
        
        template = self.item_templates[item_name]
        return self.match_template(image, template, threshold)
    
    def find_ui_element(self, image: np.ndarray, element_name: str, 
                       threshold: float = None) -> List[Dict[str, Any]]:
        """Find specific UI element in image"""
        if element_name not in self.ui_templates:
            logger.warning(f"Template not found for UI element: {element_name}")
            return []
        
        template = self.ui_templates[element_name]
        return self.match_template(image, template, threshold)
    
    def find_trait(self, image: np.ndarray, trait_name: str, 
                  threshold: float = None) -> List[Dict[str, Any]]:
        """Find specific trait in image"""
        if trait_name not in self.trait_templates:
            logger.warning(f"Template not found for trait: {trait_name}")
            return []
        
        template = self.trait_templates[trait_name]
        return self.match_template(image, template, threshold)
    
    def find_all_champions(self, image: np.ndarray, 
                          threshold: float = None) -> Dict[str, List[Dict[str, Any]]]:
        """Find all champions in image"""
        results = {}
        
        for champion_name, template in self.champion_templates.items():
            matches = self.match_template(image, template, threshold)
            if matches:
                results[champion_name] = matches
        
        return results
    
    def find_all_items(self, image: np.ndarray, 
                      threshold: float = None) -> Dict[str, List[Dict[str, Any]]]:
        """Find all items in image"""
        results = {}
        
        for item_name, template in self.item_templates.items():
            matches = self.match_template(image, template, threshold)
            if matches:
                results[item_name] = matches
        
        return results
    
    def get_champion_templates(self) -> Dict[str, np.ndarray]:
        """Get all champion templates"""
        return self.champion_templates.copy()
    
    def get_item_templates(self) -> Dict[str, np.ndarray]:
        """Get all item templates"""
        return self.item_templates.copy()
    
    def get_ui_templates(self) -> Dict[str, np.ndarray]:
        """Get all UI templates"""
        return self.ui_templates.copy()
    
    def get_trait_templates(self) -> Dict[str, np.ndarray]:
        """Get all trait templates"""
        return self.trait_templates.copy()
    
    def add_template(self, category: str, name: str, template: np.ndarray):
        """Add a new template"""
        if category == "champion":
            self.champion_templates[name] = template
        elif category == "item":
            self.item_templates[name] = template
        elif category == "ui":
            self.ui_templates[name] = template
        elif category == "trait":
            self.trait_templates[name] = template
        else:
            logger.warning(f"Unknown template category: {category}")
        
        logger.info(f"Added {category} template: {name}")
    
    def save_template(self, category: str, name: str, template: np.ndarray):
        """Save template to file"""
        category_paths = {
            "champion": "data/templates/champions",
            "item": "data/templates/items",
            "ui": "data/templates/ui",
            "trait": "data/templates/traits"
        }
        
        if category not in category_paths:
            logger.error(f"Unknown template category: {category}")
            return
        
        template_path = Path(category_paths[category])
        template_path.mkdir(parents=True, exist_ok=True)
        
        file_path = template_path / f"{name}.png"
        
        try:
            cv2.imwrite(str(file_path), template)
            self.add_template(category, name, template)
            logger.info(f"Saved template: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save template {name}: {e}")
    
    def update_threshold(self, threshold: float):
        """Update default matching threshold"""
        self.match_threshold = threshold
        logger.info(f"Updated matching threshold to {threshold}")
    
    def get_template_info(self) -> Dict[str, int]:
        """Get information about loaded templates"""
        return {
            "champions": len(self.champion_templates),
            "items": len(self.item_templates),
            "ui_elements": len(self.ui_templates),
            "traits": len(self.trait_templates),
            "total": len(self.champion_templates) + len(self.item_templates) + 
                    len(self.ui_templates) + len(self.trait_templates)
        }