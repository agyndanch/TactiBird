"""
TactiBird Overlay - Champion Detection
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from src.data.models.game_state import Champion, ChampionTier, Position
from src.vision.templates.template_matcher import TemplateMatcher
from src.vision.utils.image_utils import preprocess_image

logger = logging.getLogger(__name__)

@dataclass
class ChampionDetection:
    """Represents a detected champion"""
    champion: Champion
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    center_point: Tuple[int, int]
    level: int = 1
    items: List[str] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []

class ChampionDetector:
    """Detects champions in TFT screenshots"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.template_matcher = TemplateMatcher()
        
        # Detection settings
        self.confidence_threshold = 0.7
        self.max_champions_per_region = 28  # Max board size
        self.level_detection_enabled = True
        self.item_detection_enabled = True
        
        # Champion appearance variations
        self.star_level_templates = {}
        self.chosen_champion_templates = {}
        
        # Load additional templates
        self._load_star_templates()
        self._load_chosen_templates()
        
        logger.info("Champion detector initialized")
    
    def detect_champions_in_region(self, image: np.ndarray, region_type: str = "board") -> List[ChampionDetection]:
        """
        Detect champions in a specific region
        
        Args:
            image: Image to analyze
            region_type: Type of region (board, bench, shop)
            
        Returns:
            List of detected champions
        """
        try:
            detections = []
            
            # Preprocess image for better detection
            processed_image = preprocess_image(image, enhance_contrast=True, denoise=True)
            
            # Get champion templates
            champion_templates = self.template_matcher.get_champion_templates()
            
            for champion_name, template in champion_templates.items():
                # Perform template matching
                matches = self.template_matcher.match_template(
                    processed_image,
                    template,
                    threshold=self.confidence_threshold,
                    multi_scale=True
                )
                
                for match in matches:
                    # Create champion object
                    champion = self._create_champion_from_match(champion_name, match, region_type)
                    
                    # Detect additional details
                    champion_detection = ChampionDetection(
                        champion=champion,
                        confidence=match['confidence'],
                        bounding_box=match['bbox'],
                        center_point=match['center']
                    )
                    
                    # Detect champion level (stars)
                    if self.level_detection_enabled:
                        champion_detection.level = self._detect_champion_level(
                            processed_image, match['bbox']
                        )
                        champion_detection.champion.level = champion_detection.level
                    
                    # Detect items
                    if self.item_detection_enabled and region_type in ["board", "bench"]:
                        champion_detection.items = self._detect_champion_items(
                            processed_image, match['bbox']
                        )
                        champion_detection.champion.items = champion_detection.items
                    
                    detections.append(champion_detection)
            
            # Remove overlapping detections
            filtered_detections = self._remove_duplicate_detections(detections)
            
            # Sort by confidence
            filtered_detections.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit number of detections
            max_detections = self._get_max_detections_for_region(region_type)
            filtered_detections = filtered_detections[:max_detections]
            
            logger.debug(f"Detected {len(filtered_detections)} champions in {region_type}")
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Champion detection failed: {e}")
            return []
    
    def detect_board_champions(self, board_image: np.ndarray) -> List[ChampionDetection]:
        """Detect champions on the game board"""
        try:
            detections = self.detect_champions_in_region(board_image, "board")
            
            # Add board position information
            for detection in detections:
                board_position = self._calculate_board_position(
                    detection.center_point, board_image.shape
                )
                detection.champion.position = board_position
            
            return detections
            
        except Exception as e:
            logger.error(f"Board champion detection failed: {e}")
            return []
    
    def detect_bench_champions(self, bench_image: np.ndarray) -> List[ChampionDetection]:
        """Detect champions on the bench"""
        return self.detect_champions_in_region(bench_image, "bench")
    
    def detect_shop_champions(self, shop_image: np.ndarray) -> List[ChampionDetection]:
        """Detect champions in the shop"""
        return self.detect_champions_in_region(shop_image, "shop")
    
    def _create_champion_from_match(self, champion_name: str, match: Dict, region_type: str) -> Champion:
        """Create champion object from template match"""
        try:
            # Get champion data from data manager
            champion_data = self.data_manager.get_champion_data(champion_name)
            
            if champion_data:
                tier = ChampionTier(champion_data.get('tier', 1))
                traits = champion_data.get('traits', [])
            else:
                # Fallback to default values
                tier = ChampionTier.ONE
                traits = []
            
            # Determine position based on region
            position = None
            if region_type == "board":
                # Will be calculated later in detect_board_champions
                position = None
            elif region_type == "bench":
                # Bench positions are typically numbered 0-8
                position = Position(0, -1)  # Placeholder
            
            champion = Champion(
                name=champion_name,
                tier=tier,
                traits=traits,
                position=position,
                level=1,  # Will be detected later
                items=[],  # Will be detected later
                is_chosen=self._is_chosen_champion(match)
            )
            
            return champion
            
        except Exception as e:
            logger.error(f"Champion creation failed for {champion_name}: {e}")
            # Return basic champion as fallback
            return Champion(
                name=champion_name,
                tier=ChampionTier.ONE,
                traits=[],
                level=1
            )
    
    def _detect_champion_level(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """Detect champion level (number of stars)"""
        try:
            x, y, w, h = bbox
            
            # Extract region around champion for star detection
            star_region_y = max(0, y + h - int(h * 0.3))  # Bottom 30% of champion
            star_region = image[star_region_y:y + h, x:x + w]
            
            if star_region.size == 0:
                return 1
            
            # Look for star patterns
            level = self._count_stars_in_region(star_region)
            
            # Validate level (1-3 stars)
            return max(1, min(3, level))
            
        except Exception as e:
            logger.debug(f"Champion level detection failed: {e}")
            return 1
    
    def _count_stars_in_region(self, region: np.ndarray) -> int:
        """Count star patterns in image region"""
        try:
            # Convert to HSV for better star color detection
            if len(region.shape) == 3:
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            else:
                hsv = cv2.cvtColor(region, cv2.COLOR_GRAY2HSV)
            
            # Define star color range (yellow/gold)
            lower_star = np.array([15, 100, 100])
            upper_star = np.array([35, 255, 255])
            
            # Create mask for star colors
            star_mask = cv2.inRange(hsv, lower_star, upper_star)
            
            # Find contours that could be stars
            contours, _ = cv2.findContours(star_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            star_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                # Stars should be small but visible
                if 10 < area < 500:
                    # Check if contour is roughly star-shaped (complex shape)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # Stars are less circular than circles
                        if 0.3 < circularity < 0.8:
                            star_count += 1
            
            return min(3, star_count)  # Max 3 stars
            
        except Exception as e:
            logger.debug(f"Star counting failed: {e}")
            return 1
    
    def _detect_champion_items(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[str]:
        """Detect items on a champion"""
        try:
            x, y, w, h = bbox
            
            # Items are typically in the bottom-right corner of champion
            item_region_x = x + int(w * 0.6)
            item_region_y = y + int(h * 0.6)
            item_region_w = int(w * 0.4)
            item_region_h = int(h * 0.4)
            
            item_region = image[item_region_y:item_region_y + item_region_h,
                              item_region_x:item_region_x + item_region_w]
            
            if item_region.size == 0:
                return []
            
            # Use template matching to detect items
            detected_items = []
            item_templates = self.template_matcher.get_item_templates()
            
            for item_name, template in item_templates.items():
                matches = self.template_matcher.match_template(
                    item_region, template, threshold=0.8
                )
                
                if matches:
                    detected_items.append(item_name)
                    
                    # Champions can have max 3 items
                    if len(detected_items) >= 3:
                        break
            
            return detected_items
            
        except Exception as e:
            logger.debug(f"Item detection failed: {e}")
            return []
    
    def _calculate_board_position(self, center_point: Tuple[int, int], image_shape: Tuple[int, int]) -> Position:
        """Calculate board position from pixel coordinates"""
        try:
            cx, cy = center_point
            height, width = image_shape[:2]
            
            # TFT board is hexagonal, approximate with grid
            # This is simplified - real implementation would need calibration
            cols = 7
            rows = 4
            
            # Calculate grid position
            col = int((cx / width) * cols)
            row = int((cy / height) * rows)
            
            # Clamp to valid range
            col = max(0, min(cols - 1, col))
            row = max(0, min(rows - 1, row))
            
            return Position(col, row)
            
        except Exception as e:
            logger.debug(f"Board position calculation failed: {e}")
            return Position(0, 0)
    
    def _is_chosen_champion(self, match: Dict) -> bool:
        """Detect if champion is a chosen champion (TFT Set 4/4.5 feature)"""
        try:
            # Chosen champions have special visual effects
            # This would require specific template matching or color detection
            # For now, return False as this feature may not be in current TFT
            return False
            
        except Exception:
            return False
    
    def _remove_duplicate_detections(self, detections: List[ChampionDetection]) -> List[ChampionDetection]:
        """Remove overlapping detections of the same champion"""
        try:
            if len(detections) <= 1:
                return detections
            
            # Sort by confidence (highest first)
            sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
            
            filtered = []
            overlap_threshold = 0.5
            
            for current in sorted_detections:
                is_duplicate = False
                
                for existing in filtered:
                    # Check if same champion type
                    if current.champion.name == existing.champion.name:
                        # Check spatial overlap
                        overlap = self._calculate_bbox_overlap(
                            current.bounding_box, existing.bounding_box
                        )
                        
                        if overlap > overlap_threshold:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    filtered.append(current)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Duplicate removal failed: {e}")
            return detections
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        try:
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
            
        except Exception:
            return 0.0
    
    def _get_max_detections_for_region(self, region_type: str) -> int:
        """Get maximum expected detections for region type"""
        max_detections = {
            "board": 28,    # Max board size
            "bench": 9,     # Bench size
            "shop": 5       # Shop size
        }
        return max_detections.get(region_type, 10)
    
    def _load_star_templates(self):
        """Load star templates for level detection"""
        try:
            star_templates_path = Path("data/templates/ui/stars/")
            if star_templates_path.exists():
                for star_file in star_templates_path.glob("*.png"):
                    level = int(star_file.stem.split('_')[-1])  # e.g., "star_1.png"
                    template = cv2.imread(str(star_file))
                    if template is not None:
                        self.star_level_templates[level] = template
            
            logger.debug(f"Loaded {len(self.star_level_templates)} star templates")
            
        except Exception as e:
            logger.warning(f"Failed to load star templates: {e}")
    
    def _load_chosen_templates(self):
        """Load chosen champion templates"""
        try:
            chosen_templates_path = Path("data/templates/ui/chosen/")
            if chosen_templates_path.exists():
                for chosen_file in chosen_templates_path.glob("*.png"):
                    champion_name = chosen_file.stem
                    template = cv2.imread(str(chosen_file))
                    if template is not None:
                        self.chosen_champion_templates[champion_name] = template
            
            logger.debug(f"Loaded {len(self.chosen_champion_templates)} chosen templates")
            
        except Exception as e:
            logger.warning(f"Failed to load chosen templates: {e}")
    
    def calibrate_detection(self, test_images: List[np.ndarray], 
                          known_champions: List[List[str]]) -> Dict[str, float]:
        """Calibrate detection parameters using test data"""
        try:
            if len(test_images) != len(known_champions):
                logger.error("Mismatch between test images and known champions")
                return {}
            
            results = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'optimal_threshold': self.confidence_threshold
            }
            
            # Test different confidence thresholds
            thresholds = np.arange(0.5, 1.0, 0.05)
            best_f1 = 0.0
            best_threshold = self.confidence_threshold
            
            for threshold in thresholds:
                self.confidence_threshold = threshold
                
                total_tp = 0  # True positives
                total_fp = 0  # False positives
                total_fn = 0  # False negatives
                
                for img, known_champs in zip(test_images, known_champions):
                    detections = self.detect_champions_in_region(img, "board")
                    detected_names = [d.champion.name for d in detections]
                    
                    # Calculate metrics
                    tp = len(set(detected_names) & set(known_champs))
                    fp = len(set(detected_names) - set(known_champs))
                    fn = len(set(known_champs) - set(detected_names))
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                
                # Calculate metrics for this threshold
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    results['precision'] = precision
                    results['recall'] = recall
                    results['f1_score'] = f1
            
            # Set optimal threshold
            self.confidence_threshold = best_threshold
            results['optimal_threshold'] = best_threshold
            
            logger.info(f"Calibration complete - F1: {best_f1:.3f}, Threshold: {best_threshold:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Detection calibration failed: {e}")
            return {}
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics and settings"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'max_champions_per_region': self.max_champions_per_region,
            'level_detection_enabled': self.level_detection_enabled,
            'item_detection_enabled': self.item_detection_enabled,
            'available_templates': len(self.template_matcher.get_champion_templates()),
            'star_templates': len(self.star_level_templates),
            'chosen_templates': len(self.chosen_champion_templates)
        }
    
    def update_settings(self, **kwargs):
        """Update detection settings"""
        if 'confidence_threshold' in kwargs:
            self.confidence_threshold = kwargs['confidence_threshold']
        if 'level_detection_enabled' in kwargs:
            self.level_detection_enabled = kwargs['level_detection_enabled']
        if 'item_detection_enabled' in kwargs:
            self.item_detection_enabled = kwargs['item_detection_enabled']
        
        logger.info("Detection settings updated")
    
    def export_detections(self, detections: List[ChampionDetection], 
                         format: str = "json") -> Dict[str, Any]:
        """Export detections in specified format"""
        try:
            if format == "json":
                return {
                    'detections': [
                        {
                            'champion': {
                                'name': d.champion.name,
                                'tier': d.champion.tier.value,
                                'level': d.champion.level,
                                'items': d.champion.items,
                                'traits': d.champion.traits,
                                'position': {
                                    'x': d.champion.position.x,
                                    'y': d.champion.position.y
                                } if d.champion.position else None
                            },
                            'confidence': d.confidence,
                            'bounding_box': d.bounding_box,
                            'center_point': d.center_point
                        }
                        for d in detections
                    ],
                    'detection_count': len(detections),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {}

class ChampionTracker:
    """Track champions across multiple frames"""
    
    def __init__(self, max_tracking_frames: int = 10):
        self.max_tracking_frames = max_tracking_frames
        self.tracked_champions = {}
        self.frame_history = []
        
    def update_tracking(self, detections: List[ChampionDetection], frame_id: int):
        """Update champion tracking with new detections"""
        try:
            # Add frame to history
            self.frame_history.append({
                'frame_id': frame_id,
                'detections': detections,
                'timestamp': datetime.now()
            })
            
            # Limit history size
            if len(self.frame_history) > self.max_tracking_frames:
                self.frame_history.pop(0)
            
            # Update tracked champions
            for detection in detections:
                champion_key = f"{detection.champion.name}_{detection.center_point}"
                
                if champion_key not in self.tracked_champions:
                    self.tracked_champions[champion_key] = {
                        'champion': detection.champion,
                        'first_seen': frame_id,
                        'last_seen': frame_id,
                        'confidence_history': [detection.confidence],
                        'position_history': [detection.center_point],
                        'stable': False
                    }
                else:
                    tracked = self.tracked_champions[champion_key]
                    tracked['last_seen'] = frame_id
                    tracked['confidence_history'].append(detection.confidence)
                    tracked['position_history'].append(detection.center_point)
                    
                    # Mark as stable if seen for multiple frames
                    if len(tracked['confidence_history']) >= 3:
                        tracked['stable'] = True
            
            # Remove old tracked champions
            self._cleanup_old_tracks(frame_id)
            
        except Exception as e:
            logger.error(f"Champion tracking update failed: {e}")
    
    def get_stable_champions(self) -> List[Champion]:
        """Get champions that have been stable across multiple frames"""
        try:
            stable_champions = []
            
            for tracked_data in self.tracked_champions.values():
                if tracked_data['stable']:
                    champion = tracked_data['champion']
                    
                    # Use average confidence
                    avg_confidence = np.mean(tracked_data['confidence_history'])
                    if avg_confidence >= 0.7:  # High confidence threshold
                        stable_champions.append(champion)
            
            return stable_champions
            
        except Exception as e:
            logger.error(f"Failed to get stable champions: {e}")
            return []
    
    def _cleanup_old_tracks(self, current_frame: int):
        """Remove old champion tracks"""
        try:
            max_age = 5  # Frames
            
            to_remove = []
            for key, tracked_data in self.tracked_champions.items():
                if current_frame - tracked_data['last_seen'] > max_age:
                    to_remove.append(key)
            
            for key in to_remove:
                del self.tracked_champions[key]
                
        except Exception as e:
            logger.error(f"Track cleanup failed: {e}")
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        try:
            stable_count = sum(1 for t in self.tracked_champions.values() if t['stable'])
            
            return {
                'total_tracked': len(self.tracked_champions),
                'stable_champions': stable_count,
                'frame_history_size': len(self.frame_history),
                'tracking_accuracy': stable_count / len(self.tracked_champions) if self.tracked_champions else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get tracking stats: {e}")
            return {}