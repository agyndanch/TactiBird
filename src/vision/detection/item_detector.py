"""
TactiBird Overlay - Item Detection
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from src.data.models.game_state import Item
from src.vision.templates.template_matcher import TemplateMatcher
from src.vision.utils.image_utils import preprocess_image

logger = logging.getLogger(__name__)

@dataclass
class ItemDetection:
    """Represents a detected item"""
    item: Item
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    center_point: Tuple[int, int]
    slot_index: Optional[int] = None  # For items on champions (0-2)

class ItemDetector:
    """Detects items in TFT screenshots"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.template_matcher = TemplateMatcher()
        
        # Detection settings
        self.confidence_threshold = 0.75
        self.component_threshold = 0.7
        self.completed_item_threshold = 0.8
        
        # Item categories for different detection strategies
        self.item_categories = {
            'components': [
                'bf_sword', 'recurve_bow', 'needlessly_large_rod', 'tear_of_the_goddess',
                'chain_vest', 'negatron_cloak', 'giants_belt', 'spatula', 'glove'
            ],
            'completed_items': [],  # Will be populated from data
            'special_items': ['thiefs_gloves', 'force_of_nature', 'spatula_items']
        }
        
        # Load item data
        self._load_item_categories()
        
        logger.info("Item detector initialized")
    
    def _load_item_categories(self):
        """Load item categories from data manager"""
        try:
            # This would load from actual game data
            # For now, use some example completed items
            self.item_categories['completed_items'] = [
                'infinity_edge', 'bloodthirster', 'last_whisper', 'guardian_angel',
                'rabadons_deathcap', 'archangels_staff', 'morellonomicon', 'blue_buff',
                'bramble_vest', 'dragons_claw', 'warmogs_armor', 'gargoyle_stoneplate'
            ]
            
        except Exception as e:
            logger.warning(f"Failed to load item categories: {e}")
    
    def detect_items_on_champion(self, champion_region: np.ndarray) -> List[ItemDetection]:
        """Detect items on a specific champion"""
        try:
            detections = []
            
            # Preprocess image
            processed = preprocess_image(champion_region, enhance_contrast=True)
            
            # Items are typically in small slots around the champion
            item_slots = self._identify_item_slots(processed)
            
            for slot_idx, slot_region in enumerate(item_slots):
                if slot_region.size == 0:
                    continue
                
                # Try to detect item in this slot
                item_detection = self._detect_item_in_slot(slot_region, slot_idx)
                
                if item_detection:
                    detections.append(item_detection)
            
            logger.debug(f"Detected {len(detections)} items on champion")
            return detections
            
        except Exception as e:
            logger.error(f"Champion item detection failed: {e}")
            return []
    
    def detect_bench_items(self, bench_region: np.ndarray) -> List[ItemDetection]:
        """Detect items on the bench"""
        try:
            detections = []
            
            # Preprocess image
            processed = preprocess_image(bench_region, enhance_contrast=True)
            
            # Bench items are typically in a horizontal row
            item_regions = self._segment_bench_items(processed)
            
            for region in item_regions:
                item_detection = self._detect_item_in_region(region)
                if item_detection:
                    detections.append(item_detection)
            
            logger.debug(f"Detected {len(detections)} items on bench")
            return detections
            
        except Exception as e:
            logger.error(f"Bench item detection failed: {e}")
            return []
    
    def detect_carousel_items(self, carousel_region: np.ndarray) -> List[ItemDetection]:
        """Detect items in carousel (draft phase)"""
        try:
            detections = []
            
            # Preprocess image
            processed = preprocess_image(carousel_region, enhance_contrast=True)
            
            # Carousel items are on champions arranged in a circle
            champion_regions = self._identify_carousel_champions(processed)
            
            for region in champion_regions:
                item_detections = self.detect_items_on_champion(region)
                detections.extend(item_detections)
            
            logger.debug(f"Detected {len(detections)} items in carousel")
            return detections
            
        except Exception as e:
            logger.error(f"Carousel item detection failed: {e}")
            return []
    
    def _identify_item_slots(self, champion_region: np.ndarray) -> List[np.ndarray]:
        """Identify item slot regions on a champion"""
        try:
            height, width = champion_region.shape[:2]
            slots = []
            
            # Items are typically in bottom-right corner of champion
            # Each champion can have up to 3 items
            slot_size = min(width // 4, height // 4)
            
            # Define 3 item slot positions
            slot_positions = [
                (width - slot_size, height - slot_size),           # Bottom-right
                (width - slot_size, height - 2 * slot_size),       # Middle-right
                (width - 2 * slot_size, height - slot_size)        # Bottom-middle
            ]
            
            for x, y in slot_positions:
                if x >= 0 and y >= 0:
                    slot = champion_region[y:y + slot_size, x:x + slot_size]
                    slots.append(slot)
                else:
                    slots.append(np.array([]))  # Empty slot
            
            return slots
            
        except Exception as e:
            logger.error(f"Item slot identification failed: {e}")
            return []
    
    def _detect_item_in_slot(self, slot_region: np.ndarray, slot_index: int) -> Optional[ItemDetection]:
        """Detect item in a specific slot"""
        try:
            if slot_region.size == 0:
                return None
            
            # First try to detect completed items
            item_detection = self._match_completed_items(slot_region)
            
            if not item_detection:
                # Try to detect components
                item_detection = self._match_item_components(slot_region)
            
            if item_detection:
                item_detection.slot_index = slot_index
                
            return item_detection
            
        except Exception as e:
            logger.debug(f"Item detection in slot {slot_index} failed: {e}")
            return None
    
    def _detect_item_in_region(self, region: np.ndarray) -> Optional[ItemDetection]:
        """Detect item in a general region"""
        try:
            # Try completed items first (higher priority)
            item_detection = self._match_completed_items(region)
            
            if not item_detection:
                # Try components
                item_detection = self._match_item_components(region)
            
            return item_detection
            
        except Exception as e:
            logger.debug(f"Item detection in region failed: {e}")
            return None
    
    def _match_completed_items(self, region: np.ndarray) -> Optional[ItemDetection]:
        """Match completed items using template matching"""
        try:
            item_templates = self.template_matcher.get_item_templates()
            
            best_match = None
            best_confidence = 0.0
            
            for item_name in self.item_categories['completed_items']:
                if item_name in item_templates:
                    template = item_templates[item_name]
                    
                    matches = self.template_matcher.match_template(
                        region, template, threshold=self.completed_item_threshold
                    )
                    
                    if matches and matches[0]['confidence'] > best_confidence:
                        best_confidence = matches[0]['confidence']
                        best_match = (item_name, matches[0])
            
            if best_match:
                item_name, match = best_match
                item_data = self.data_manager.get_item_data(item_name)
                
                item = Item(
                    name=item_name,
                    components=item_data.get('components', []) if item_data else [],
                    stats=item_data.get('stats', {}) if item_data else {},
                    is_completed=True
                )
                
                return ItemDetection(
                    item=item,
                    confidence=match['confidence'],
                    bounding_box=match['bbox'],
                    center_point=match['center']
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Completed item matching failed: {e}")
            return None
    
    def _match_item_components(self, region: np.ndarray) -> Optional[ItemDetection]:
        """Match item components using template matching"""
        try:
            item_templates = self.template_matcher.get_item_templates()
            
            best_match = None
            best_confidence = 0.0
            
            for component_name in self.item_categories['components']:
                if component_name in item_templates:
                    template = item_templates[component_name]
                    
                    matches = self.template_matcher.match_template(
                        region, template, threshold=self.component_threshold
                    )
                    
                    if matches and matches[0]['confidence'] > best_confidence:
                        best_confidence = matches[0]['confidence']
                        best_match = (component_name, matches[0])
            
            if best_match:
                component_name, match = best_match
                component_data = self.data_manager.get_item_data(component_name)
                
                item = Item(
                    name=component_name,
                    components=[],  # Components don't have sub-components
                    stats=component_data.get('stats', {}) if component_data else {},
                    is_completed=False
                )
                
                return ItemDetection(
                    item=item,
                    confidence=match['confidence'],
                    bounding_box=match['bbox'],
                    center_point=match['center']
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Component matching failed: {e}")
            return None
    
    def _segment_bench_items(self, bench_region: np.ndarray) -> List[np.ndarray]:
        """Segment bench region into individual item slots"""
        try:
            height, width = bench_region.shape[:2]
            item_regions = []
            
            # Bench typically has 9-10 slots horizontally
            num_slots = 10
            slot_width = width // num_slots
            
            for i in range(num_slots):
                x_start = i * slot_width
                x_end = min((i + 1) * slot_width, width)
                
                # Items are typically in the top portion of bench
                y_start = 0
                y_end = height // 2
                
                slot_region = bench_region[y_start:y_end, x_start:x_end]
                item_regions.append(slot_region)
            
            return item_regions
            
        except Exception as e:
            logger.error(f"Bench segmentation failed: {e}")
            return []
    
    def _identify_carousel_champions(self, carousel_region: np.ndarray) -> List[np.ndarray]:
        """Identify champion regions in carousel"""
        try:
            # This is a simplified implementation
            # Real carousel detection would need to identify circular arrangement
            
            height, width = carousel_region.shape[:2]
            champion_regions = []
            
            # Assume champions are arranged in a grid for simplicity
            rows, cols = 2, 5  # Typical carousel layout
            
            for row in range(rows):
                for col in range(cols):
                    x_start = col * (width // cols)
                    x_end = (col + 1) * (width // cols)
                    y_start = row * (height // rows)
                    y_end = (row + 1) * (height // rows)
                    
                    champion_region = carousel_region[y_start:y_end, x_start:x_end]
                    champion_regions.append(champion_region)
            
            return champion_regions
            
        except Exception as e:
            logger.error(f"Carousel champion identification failed: {e}")
            return []
    
    def analyze_item_combinations(self, detected_items: List[ItemDetection]) -> Dict[str, Any]:
        """Analyze possible item combinations"""
        try:
            analysis = {
                'components': [],
                'completed_items': [],
                'possible_combinations': [],
                'optimization_suggestions': []
            }
            
            # Separate components and completed items
            for detection in detected_items:
                if detection.item.is_completed:
                    analysis['completed_items'].append(detection.item.name)
                else:
                    analysis['components'].append(detection.item.name)
            
            # Find possible combinations
            component_counts = {}
            for component in analysis['components']:
                component_counts[component] = component_counts.get(component, 0) + 1
            
            # Check for possible item combinations
            item_recipes = self._get_item_recipes()
            
            for item_name, recipe in item_recipes.items():
                if self._can_craft_item(component_counts, recipe):
                    analysis['possible_combinations'].append({
                        'item': item_name,
                        'recipe': recipe,
                        'priority': self._get_item_priority(item_name)
                    })
            
            # Generate optimization suggestions
            analysis['optimization_suggestions'] = self._generate_item_suggestions(
                analysis['components'], analysis['completed_items']
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Item combination analysis failed: {e}")
            return {}
    
    def _get_item_recipes(self) -> Dict[str, List[str]]:
        """Get item crafting recipes"""
        # This would come from the data manager in a real implementation
        return {
            'infinity_edge': ['bf_sword', 'glove'],
            'bloodthirster': ['bf_sword', 'negatron_cloak'],
            'last_whisper': ['recurve_bow', 'glove'],
            'rabadons_deathcap': ['needlessly_large_rod', 'needlessly_large_rod'],
            'archangels_staff': ['tear_of_the_goddess', 'needlessly_large_rod'],
            'blue_buff': ['tear_of_the_goddess', 'tear_of_the_goddess'],
            'bramble_vest': ['chain_vest', 'chain_vest'],
            'dragons_claw': ['negatron_cloak', 'negatron_cloak'],
            'warmogs_armor': ['giants_belt', 'giants_belt']
        }
    
    def _can_craft_item(self, component_counts: Dict[str, int], recipe: List[str]) -> bool:
        """Check if an item can be crafted with available components"""
        try:
            recipe_counts = {}
            for component in recipe:
                recipe_counts[component] = recipe_counts.get(component, 0) + 1
            
            for component, needed in recipe_counts.items():
                if component_counts.get(component, 0) < needed:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _get_item_priority(self, item_name: str) -> int:
        """Get item crafting priority (1-10)"""
        # This would come from meta analysis
        priority_map = {
            'infinity_edge': 9,
            'rabadons_deathcap': 9,
            'archangels_staff': 8,
            'bloodthirster': 8,
            'last_whisper': 8,
            'bramble_vest': 7,
            'dragons_claw': 7,
            'warmogs_armor': 6,
            'blue_buff': 7
        }
        return priority_map.get(item_name, 5)
    
    def _generate_item_suggestions(self, components: List[str], completed_items: List[str]) -> List[str]:
        """Generate item optimization suggestions"""
        suggestions = []
        
        try:
            # Count components
            component_counts = {}
            for component in components:
                component_counts[component] = component_counts.get(component, 0) + 1
            
            # Suggest using excess components
            for component, count in component_counts.items():
                if count >= 3:
                    suggestions.append(f"Use excess {component} components ({count} available)")
            
            # Suggest high-priority crafts
            recipes = self._get_item_recipes()
            for item_name, recipe in recipes.items():
                if self._can_craft_item(component_counts, recipe):
                    priority = self._get_item_priority(item_name)
                    if priority >= 8:
                        suggestions.append(f"Craft {item_name} (high priority)")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return []
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get item detection statistics"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'component_threshold': self.component_threshold,
            'completed_item_threshold': self.completed_item_threshold,
            'available_templates': len(self.template_matcher.get_item_templates()),
            'tracked_components': len(self.item_categories['components']),
            'tracked_completed_items': len(self.item_categories['completed_items'])
        }
    
    def update_thresholds(self, **kwargs):
        """Update detection thresholds"""
        if 'confidence_threshold' in kwargs:
            self.confidence_threshold = kwargs['confidence_threshold']
        if 'component_threshold' in kwargs:
            self.component_threshold = kwargs['component_threshold']
        if 'completed_item_threshold' in kwargs:
            self.completed_item_threshold = kwargs['completed_item_threshold']
        
        logger.info("Item detection thresholds updated")