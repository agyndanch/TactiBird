"""
TactiBird Overlay - Board State Analyzer
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.data.models.game_state import BoardState, Champion, Position, ChampionTier, Trait
from src.vision.templates.template_matcher import TemplateMatcher
from src.vision.utils.image_utils import preprocess_image, extract_region
from src.vision.ocr.text_recognizer import TextRecognizer

logger = logging.getLogger(__name__)

@dataclass
class ChampionDetection:
    """Detected champion on board"""
    champion: Champion
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h

class BoardAnalyzer:
    """Analyzes the TFT board state from screenshots"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.template_matcher = TemplateMatcher()
        self.text_recognizer = TextRecognizer()
        
        # Board configuration
        self.board_size = (7, 4)  # TFT board is 7x4
        self.hex_positions = self._generate_hex_positions()
        
        # Detection thresholds
        self.champion_confidence_threshold = 0.7
        self.trait_confidence_threshold = 0.8
        
        logger.info("Board analyzer initialized")
    
    def _generate_hex_positions(self) -> List[Position]:
        """Generate all valid hexagonal board positions"""
        positions = []
        
        # TFT board layout (approximate)
        for row in range(4):
            for col in range(7):
                # Skip invalid positions based on TFT hex layout
                if (row == 0 and col in [0, 6]) or (row == 3 and col in [0, 6]):
                    continue
                positions.append(Position(col, row))
        
        return positions
    
    def analyze(self, screenshot: np.ndarray) -> Optional[BoardState]:
        """
        Analyze screenshot to extract board state
        
        Args:
            screenshot: Screenshot image
            
        Returns:
            BoardState object or None if analysis fails
        """
        try:
            # Extract board region
            board_region = self._extract_board_region(screenshot)
            if board_region is None:
                logger.warning("Could not extract board region")
                return None
            
            # Preprocess image
            processed_board = preprocess_image(board_region)
            
            # Detect champions
            champion_detections = self._detect_champions(processed_board)
            
            # Detect traits
            active_traits = self._detect_traits(screenshot)
            
            # Create board state
            board_state = BoardState()
            
            # Add detected champions
            for detection in champion_detections:
                if detection.confidence >= self.champion_confidence_threshold:
                    board_state.add_champion(detection.champion)
            
            # Add active traits
            board_state.active_traits = active_traits
            
            logger.debug(f"Board analysis complete - {len(board_state.champions)} champions, {len(active_traits)} active traits")
            return board_state
            
        except Exception as e:
            logger.error(f"Board analysis failed: {e}")
            return None
    
    def _extract_board_region(self, screenshot: np.ndarray) -> Optional[np.ndarray]:
        """Extract the board region from screenshot"""
        # TODO: Implement dynamic board region detection
        # For now, use configured region
        height, width = screenshot.shape[:2]
        
        # Approximate board location (adjust based on resolution)
        board_x = int(width * 0.1)
        board_y = int(height * 0.2)
        board_w = int(width * 0.6)
        board_h = int(height * 0.5)
        
        return extract_region(screenshot, board_x, board_y, board_w, board_h)
    
    def _detect_champions(self, board_image: np.ndarray) -> List[ChampionDetection]:
        """Detect champions on the board"""
        detections = []
        
        try:
            # Get champion templates
            champion_templates = self.template_matcher.get_champion_templates()
            
            for champion_name, template in champion_templates.items():
                matches = self.template_matcher.match_template(
                    board_image, 
                    template, 
                    threshold=self.champion_confidence_threshold
                )
                
                for match in matches:
                    # Determine position on board
                    position = self._pixel_to_board_position(match['center'])
                    
                    if position:
                        # Create champion object
                        champion = Champion(
                            name=champion_name,
                            tier=self._get_champion_tier(champion_name),
                            position=position
                        )
                        
                        # Detect champion level and items
                        champion.level = self._detect_champion_level(board_image, match['bbox'])
                        champion.items = self._detect_champion_items(board_image, match['bbox'])
                        
                        detection = ChampionDetection(
                            champion=champion,
                            confidence=match['confidence'],
                            bounding_box=match['bbox']
                        )
                        
                        detections.append(detection)
            
        except Exception as e:
            logger.error(f"Champion detection failed: {e}")
        
        return detections
    
    def _detect_traits(self, screenshot: np.ndarray) -> List[Trait]:
        """Detect active traits from the traits panel"""
        traits = []
        
        try:
            # Extract traits region
            traits_region = self._extract_traits_region(screenshot)
            if traits_region is None:
                return traits
            
            # Use OCR to read trait information
            trait_text = self.text_recognizer.extract_text(traits_region)
            
            # Parse trait information
            traits = self._parse_trait_text(trait_text)
            
        except Exception as e:
            logger.error(f"Trait detection failed: {e}")
        
        return traits
    
    def _extract_traits_region(self, screenshot: np.ndarray) -> Optional[np.ndarray]:
        """Extract the traits panel region"""
        height, width = screenshot.shape[:2]
        
        # Approximate traits panel location
        traits_x = int(width * 0.02)
        traits_y = int(height * 0.2)
        traits_w = int(width * 0.15)
        traits_h = int(height * 0.6)
        
        return extract_region(screenshot, traits_x, traits_y, traits_w, traits_h)
    
    def _pixel_to_board_position(self, pixel_pos: Tuple[int, int]) -> Optional[Position]:
        """Convert pixel coordinates to board position"""
        # TODO: Implement accurate pixel-to-hex conversion
        # This is a simplified version
        x, y = pixel_pos
        
        # Rough conversion (needs calibration)
        board_x = min(6, max(0, int(x / 100)))
        board_y = min(3, max(0, int(y / 100)))
        
        return Position(board_x, board_y)
    
    def _get_champion_tier(self, champion_name: str) -> ChampionTier:
        """Get champion tier from data"""
        # TODO: Load from data manager
        # Placeholder implementation
        tier_map = {
            # This would be loaded from game data
            "graves": ChampionTier.ONE,
            "nidalee": ChampionTier.ONE,
            "tristana": ChampionTier.TWO,
            "lux": ChampionTier.THREE,
            "jinx": ChampionTier.FOUR,
            "kayn": ChampionTier.FIVE
        }
        
        return tier_map.get(champion_name.lower(), ChampionTier.ONE)
    
    def _detect_champion_level(self, board_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """Detect champion level (stars)"""
        # TODO: Implement star detection
        # For now, assume level 1
        return 1
    
    def _detect_champion_items(self, board_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[str]:
        """Detect items on champion"""
        # TODO: Implement item detection
        # For now, return empty list
        return []
    
    def _parse_trait_text(self, text: str) -> List[Trait]:
        """Parse trait information from OCR text"""
        traits = []
        
        try:
            lines = text.strip().split('\n')
            
            for line in lines:
                # Parse trait format: "Assassin (3/3)"
                if '(' in line and ')' in line:
                    trait_name = line.split('(')[0].strip()
                    trait_count = line.split('(')[1].split(')')[0]
                    
                    if '/' in trait_count:
                        current, required = trait_count.split('/')
                        current_count = int(current.strip())
                        required_count = int(required.strip())
                        
                        trait = Trait(
                            name=trait_name,
                            current_count=current_count,
                            required_counts=[required_count],
                            is_active=current_count >= required_count,
                            active_level=1 if current_count >= required_count else 0
                        )
                        
                        traits.append(trait)
            
        except Exception as e:
            logger.error(f"Trait parsing failed: {e}")
        
        return traits
    
    def calibrate_board_positions(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Calibrate board position detection"""
        # TODO: Implement calibration routine
        return {"status": "not_implemented"}