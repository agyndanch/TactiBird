"""
TactiBird Overlay - Shop Analyzer
"""

import logging
import cv2
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.data.models.game_state import ShopState, Champion, ChampionTier
from src.vision.templates.template_matcher import TemplateMatcher
from src.vision.utils.image_utils import preprocess_image, extract_region
from src.vision.ocr.text_recognizer import TextRecognizer

logger = logging.getLogger(__name__)

@dataclass
class ShopSlot:
    """Represents a shop slot"""
    champion: Optional[Champion]
    cost: int
    position: int  # 0-4 for the 5 shop slots
    confidence: float

class ShopAnalyzer:
    """Analyzes the TFT shop from screenshots"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.template_matcher = TemplateMatcher()
        self.text_recognizer = TextRecognizer()
        
        # Shop configuration
        self.shop_slots = 5
        self.slot_width = 120  # Approximate width of each shop slot
        self.slot_height = 150  # Approximate height of each shop slot
        
        # Detection thresholds
        self.champion_confidence_threshold = 0.7
        self.cost_confidence_threshold = 0.8
        
        logger.info("Shop analyzer initialized")
    
    def analyze(self, screenshot: np.ndarray) -> Optional[ShopState]:
        """
        Analyze screenshot to extract shop state
        
        Args:
            screenshot: Screenshot image
            
        Returns:
            ShopState object or None if analysis fails
        """
        try:
            # Extract shop region
            shop_region = self._extract_shop_region(screenshot)
            if shop_region is None:
                logger.warning("Could not extract shop region")
                return None
            
            # Preprocess image
            processed_shop = preprocess_image(shop_region)
            
            # Analyze each shop slot
            shop_slots = self._analyze_shop_slots(processed_shop)
            
            # Extract additional shop information
            reroll_cost = self._detect_reroll_cost(shop_region)
            can_reroll = self._can_reroll(shop_region)
            
            # Create shop state
            shop_state = ShopState(
                available_champions=[slot.champion for slot in shop_slots if slot.champion],
                reroll_cost=reroll_cost,
                can_reroll=can_reroll
            )
            
            logger.debug(f"Shop analysis complete - {len(shop_state.available_champions)} champions available")
            return shop_state
            
        except Exception as e:
            logger.error(f"Shop analysis failed: {e}")
            return None
    
    def _extract_shop_region(self, screenshot: np.ndarray) -> Optional[np.ndarray]:
        """Extract the shop region from screenshot"""
        height, width = screenshot.shape[:2]
        
        # Approximate shop location (bottom center of screen)
        shop_x = int(width * 0.2)
        shop_y = int(height * 0.75)
        shop_w = int(width * 0.6)
        shop_h = int(height * 0.2)
        
        return extract_region(screenshot, shop_x, shop_y, shop_w, shop_h)
    
    def _analyze_shop_slots(self, shop_image: np.ndarray) -> List[ShopSlot]:
        """Analyze individual shop slots"""
        slots = []
        shop_width = shop_image.shape[1]
        
        for i in range(self.shop_slots):
            # Calculate slot region
            slot_x = int((shop_width / self.shop_slots) * i)
            slot_w = int(shop_width / self.shop_slots)
            slot_region = extract_region(shop_image, slot_x, 0, slot_w, shop_image.shape[0])
            
            # Analyze this slot
            slot = self._analyze_single_slot(slot_region, i)
            slots.append(slot)
        
        return slots
    
    def _analyze_single_slot(self, slot_image: np.ndarray, position: int) -> ShopSlot:
        """Analyze a single shop slot"""
        champion = None
        cost = 0
        confidence = 0.0
        
        try:
            # Detect champion in slot
            champion_detection = self._detect_champion_in_slot(slot_image)
            if champion_detection:
                champion, confidence = champion_detection
                
                # Get cost from champion data
                if champion:
                    cost = self.data_manager.get_champion_tier(champion.name).value
            
            # Alternative: detect cost directly from image
            if cost == 0:
                cost = self._detect_cost_in_slot(slot_image)
            
        except Exception as e:
            logger.error(f"Failed to analyze slot {position}: {e}")
        
        return ShopSlot(
            champion=champion,
            cost=cost,
            position=position,
            confidence=confidence
        )
    
    def _detect_champion_in_slot(self, slot_image: np.ndarray) -> Optional[Tuple[Champion, float]]:
        """Detect champion in a shop slot"""
        try:
            # Get champion templates
            champion_templates = self.template_matcher.get_champion_templates()
            
            best_match = None
            best_confidence = 0.0
            
            for champion_name, template in champion_templates.items():
                matches = self.template_matcher.match_template(
                    slot_image,
                    template,
                    threshold=self.champion_confidence_threshold
                )
                
                if matches:
                    match = matches[0]  # Take best match
                    if match['confidence'] > best_confidence:
                        best_confidence = match['confidence']
                        best_match = champion_name
            
            if best_match and best_confidence >= self.champion_confidence_threshold:
                # Create champion object
                champion = Champion(
                    name=best_match,
                    tier=self.data_manager.get_champion_tier(best_match),
                    traits=self.data_manager.get_champion_traits(best_match)
                )
                
                return champion, best_confidence
            
        except Exception as e:
            logger.error(f"Champion detection failed: {e}")
        
        return None
    
    def _detect_cost_in_slot(self, slot_image: np.ndarray) -> int:
        """Detect champion cost in shop slot"""
        try:
            # Extract bottom portion where cost is typically displayed
            cost_region = extract_region(
                slot_image,
                0,
                int(slot_image.shape[0] * 0.8),
                slot_image.shape[1],
                int(slot_image.shape[0] * 0.2)
            )
            
            # Use OCR to read cost
            cost_text = self.text_recognizer.extract_text(cost_region)
            
            # Parse cost from text
            import re
            cost_match = re.search(r'\d+', cost_text)
            if cost_match:
                cost = int(cost_match.group())
                if 1 <= cost <= 5:  # Valid TFT costs
                    return cost
            
        except Exception as e:
            logger.debug(f"Cost detection failed: {e}")
        
        return 0
    
    def _detect_reroll_cost(self, shop_region: np.ndarray) -> int:
        """Detect reroll cost"""
        try:
            # Extract reroll button area (typically bottom right)
            reroll_region = extract_region(
                shop_region,
                int(shop_region.shape[1] * 0.8),
                int(shop_region.shape[0] * 0.7),
                int(shop_region.shape[1] * 0.2),
                int(shop_region.shape[0] * 0.3)
            )
            
            # Use OCR to read reroll cost
            reroll_text = self.text_recognizer.extract_text(reroll_region)
            
            # Parse cost from text
            import re
            cost_match = re.search(r'\d+', reroll_text)
            if cost_match:
                return int(cost_match.group())
            
        except Exception as e:
            logger.debug(f"Reroll cost detection failed: {e}")
        
        return 2  # Default reroll cost
    
    def _can_reroll(self, shop_region: np.ndarray) -> bool:
        """Check if reroll is available"""
        # TODO: Implement reroll availability detection
        # This could check if the reroll button is enabled/clickable
        return True
    
    def get_shop_odds(self, player_level: int) -> Dict[int, float]:
        """Get shop odds for each champion tier based on player level"""
        # TFT shop odds (approximate)
        odds_table = {
            1: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
            2: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
            3: {1: 0.75, 2: 0.25, 3: 0.0, 4: 0.0, 5: 0.0},
            4: {1: 0.55, 2: 0.30, 3: 0.15, 4: 0.0, 5: 0.0},
            5: {1: 0.45, 2: 0.33, 3: 0.20, 4: 0.02, 5: 0.0},
            6: {1: 0.30, 2: 0.40, 3: 0.25, 4: 0.05, 5: 0.0},
            7: {1: 0.19, 2: 0.30, 3: 0.35, 4: 0.15, 5: 0.01},
            8: {1: 0.16, 2: 0.20, 3: 0.25, 4: 0.32, 5: 0.07},
            9: {1: 0.09, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.31},
        }
        
        return odds_table.get(player_level, odds_table[9])
    
    def analyze_shop_value(self, shop_state: ShopState, player_level: int, current_comp: List[str] = None) -> Dict[str, Any]:
        """Analyze the value of current shop"""
        if not shop_state.available_champions:
            return {"value": "empty", "recommendations": []}
        
        analysis = {
            "value": "average",
            "recommendations": [],
            "tier_distribution": {},
            "comp_synergy": 0.0
        }
        
        # Analyze tier distribution
        for champion in shop_state.available_champions:
            tier = champion.tier.value
            analysis["tier_distribution"][tier] = analysis["tier_distribution"].get(tier, 0) + 1
        
        # Calculate shop value based on player level and odds
        shop_odds = self.get_shop_odds(player_level)
        expected_value = 0.0
        
        for tier, count in analysis["tier_distribution"].items():
            expected_count = shop_odds.get(tier, 0.0) * 5  # 5 shop slots
            if count > expected_count * 1.2:  # 20% above expected
                expected_value += 1.0
            elif count < expected_count * 0.8:  # 20% below expected
                expected_value -= 0.5
        
        # Determine overall value
        if expected_value > 1.5:
            analysis["value"] = "excellent"
        elif expected_value > 0.5:
            analysis["value"] = "good"
        elif expected_value < -1.0:
            analysis["value"] = "poor"
        
        # Check composition synergy
        if current_comp:
            synergy_score = 0
            for champion in shop_state.available_champions:
                if champion.name.lower() in [c.lower() for c in current_comp]:
                    synergy_score += 2  # Upgrade potential
                
                # Check trait synergy
                champion_traits = self.data_manager.get_champion_traits(champion.name)
                for trait in champion_traits:
                    if any(trait in self.data_manager.get_champion_traits(comp_champ) for comp_champ in current_comp):
                        synergy_score += 1
            
            analysis["comp_synergy"] = synergy_score / len(shop_state.available_champions)
        
        return analysis