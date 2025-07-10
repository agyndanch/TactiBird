"""
TactiBird Overlay - Positioning Coach
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from src.ai.coaches.base_coach import BaseCoach, CoachingSuggestion
from src.data.models.game_state import GameState

logger = logging.getLogger(__name__)

class PositioningCoach(BaseCoach):
    """Coach that provides positioning advice for optimal team setup"""
    
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.name = "Positioning Coach"
        self.enabled = True
        self.weight = 1.0
        
        logger.info("Positioning coach initialized")
    
    def get_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Generate positioning suggestions based on current board state"""
        suggestions = []
        
        try:
            # Basic positioning suggestions
            if game_state.board:
                suggestions.extend(self._get_basic_positioning_advice(game_state))
                suggestions.extend(self._get_protection_advice(game_state))
                suggestions.extend(self._get_synergy_positioning_advice(game_state))
            
        except Exception as e:
            logger.error(f"Error generating positioning suggestions: {e}")
        
        return suggestions
    
    def _get_basic_positioning_advice(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Get basic positioning advice"""
        suggestions = []
        
        # Example basic positioning advice
        suggestions.append(CoachingSuggestion(
            type="positioning",
            message="Place tanks in the front row to protect your carries",
            priority=5,
            timestamp=datetime.now(),
            context={"advice_type": "basic", "category": "tank_positioning"}
        ))
        
        if game_state.is_late_game():
            suggestions.append(CoachingSuggestion(
                type="positioning",
                message="Late game: Spread units to avoid AoE damage",
                priority=7,
                timestamp=datetime.now(),
                context={"advice_type": "late_game", "category": "aoe_protection"}
            ))
        
        return suggestions
    
    def _get_protection_advice(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Get advice about protecting important units"""
        suggestions = []
        
        # Example protection advice
        if game_state.player and game_state.player.health <= 30:
            suggestions.append(CoachingSuggestion(
                type="positioning",
                message="Low health: Consider more defensive positioning",
                priority=8,
                timestamp=datetime.now(),
                context={"advice_type": "defensive", "health": game_state.player.health}
            ))
        
        return suggestions
    
    def _get_synergy_positioning_advice(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Get advice about positioning for synergies"""
        suggestions = []
        
        # Example synergy-based positioning
        suggestions.append(CoachingSuggestion(
            type="positioning",
            message="Keep similar synergy units close for optimal effect",
            priority=4,
            timestamp=datetime.now(),
            context={"advice_type": "synergy", "category": "grouping"}
        ))
        
        return suggestions
    
    def is_applicable(self, game_state: GameState) -> bool:
        """Check if positioning advice is relevant"""
        if not self.enabled:
            return False
        
        # Only provide positioning advice if we have board state
        return game_state.board is not None