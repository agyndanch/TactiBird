"""
TactiBird Overlay - Base Coach Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.data.models.game_state import GameState

@dataclass
class CoachingSuggestion:
    """Represents a coaching suggestion"""
    type: str  # economy, composition, positioning, items, etc.
    message: str
    priority: int  # 1-10, higher is more important
    timestamp: datetime
    context: Dict[str, Any]  # Additional context data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.type,
            'message': self.message,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }

class BaseCoach(ABC):
    """Base class for all coaching modules"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.name = "Base Coach"
        self.enabled = True
        self.weight = 1.0
        
    @abstractmethod
    def get_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """
        Generate coaching suggestions based on current game state
        
        Args:
            game_state: Current game state
            
        Returns:
            List of coaching suggestions
        """
        pass
    
    def is_applicable(self, game_state: GameState) -> bool:
        """
        Check if this coach is applicable for the current game state
        
        Args:
            game_state: Current game state
            
        Returns:
            True if coach should provide suggestions
        """
        return self.enabled
    
    def get_priority_adjustment(self, suggestion: CoachingSuggestion, game_state: GameState) -> int:
        """
        Adjust suggestion priority based on current context
        
        Args:
            suggestion: The suggestion to adjust
            game_state: Current game state
            
        Returns:
            Adjusted priority value
        """
        base_priority = suggestion.priority
        
        # Adjust based on coach weight
        adjusted_priority = int(base_priority * self.weight)
        
        # Context-based adjustments
        if game_state.player.health <= 20:
            # Increase priority for survival suggestions
            if suggestion.type in ['positioning', 'items']:
                adjusted_priority += 2
        
        if game_state.is_late_game():
            # Late game prioritizes positioning and items over economy
            if suggestion.type == 'economy':
                adjusted_priority -= 1
            elif suggestion.type in ['positioning', 'items']:
                adjusted_priority += 1
        
        return max(1, min(10, adjusted_priority))
    
    def validate_suggestion(self, suggestion: CoachingSuggestion, game_state: GameState) -> bool:
        """
        Validate if a suggestion is still relevant
        
        Args:
            suggestion: The suggestion to validate
            game_state: Current game state
            
        Returns:
            True if suggestion is still valid
        """
        # Basic validation - suggestions shouldn't be too old
        time_diff = datetime.now() - suggestion.timestamp
        if time_diff.total_seconds() > 30:  # 30 seconds old
            return False
        
        return True
    
    def get_coach_info(self) -> Dict[str, Any]:
        """Get information about this coach"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'weight': self.weight,
            'type': self.__class__.__name__
        }