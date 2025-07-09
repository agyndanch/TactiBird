"""
TactiBird Overlay - Economy Coach
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from src.ai.coaches.base_coach import BaseCoach, CoachingSuggestion
from src.data.models.game_state import GameState, GamePhase

logger = logging.getLogger(__name__)

class EconomyCoach(BaseCoach):
    """Provides economy management coaching"""
    
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.name = "Economy Coach"
        
        # Economy thresholds and rules
        self.interest_breakpoints = [10, 20, 30, 40, 50]
        self.safe_health_threshold = 40
        self.critical_health_threshold = 20
        
    def get_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Generate economy-focused suggestions"""
        suggestions = []
        
        try:
            # Skip if not in planning phase
            if game_state.phase != GamePhase.PLANNING:
                return suggestions
            
            # Analyze current economy situation
            suggestions.extend(self._check_interest_optimization(game_state))
            suggestions.extend(self._check_leveling_timing(game_state))
            suggestions.extend(self._check_reroll_advice(game_state))
            suggestions.extend(self._check_emergency_economy(game_state))
            suggestions.extend(self._check_winstreak_economy(game_state))
            
        except Exception as e:
            logger.error(f"Error generating economy suggestions: {e}")
        
        return suggestions
    
    def _check_interest_optimization(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Check for interest optimization opportunities"""
        suggestions = []
        current_gold = game_state.player.gold
        
        # Find next interest breakpoint
        next_breakpoint = None
        for breakpoint in self.interest_breakpoints:
            if current_gold < breakpoint:
                next_breakpoint = breakpoint
                break
        
        if next_breakpoint:
            gold_needed = next_breakpoint - current_gold
            
            if 1 <= gold_needed <= 3:
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message=f"Save {gold_needed} more gold to reach {next_breakpoint}g interest breakpoint",
                    priority=7,
                    timestamp=datetime.now(),
                    context={
                        "current_gold": current_gold,
                        "target_gold": next_breakpoint,
                        "gold_needed": gold_needed
                    }
                ))
        
        # Warn about losing interest
        if current_gold >= 10:
            current_interest = min(current_gold // 10, 5)
            if current_gold % 10 >= 8:  # Close to losing interest
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message=f"Careful! You'll lose {current_interest} interest if you spend",
                    priority=6,
                    timestamp=datetime.now(),
                    context={
                        "current_interest": current_interest,
                        "gold_over_threshold": current_gold % 10
                    }
                ))
        
        return suggestions
    
    def _check_leveling_timing(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Analyze leveling timing decisions"""
        suggestions = []
        player = game_state.player
        
        # Early game leveling advice
        if game_state.is_early_game():
            if game_state.stage == 2 and game_state.round == 1:
                if player.gold >= 4:
                    suggestions.append(CoachingSuggestion(
                        type="economy",
                        message="Consider leveling to 4 at 2-1 for better shop odds",
                        priority=8,
                        timestamp=datetime.now(),
                        context={
                            "stage": game_state.stage,
                            "round": game_state.round,
                            "current_level": player.level
                        }
                    ))
            
            elif game_state.stage == 2 and game_state.round == 5:
                if player.gold >= 8 and player.level < 5:
                    suggestions.append(CoachingSuggestion(
                        type="economy",
                        message="Consider leveling to 5 at 2-5 for economy",
                        priority=7,
                        timestamp=datetime.now(),
                        context={
                            "stage": game_state.stage,
                            "round": game_state.round,
                            "current_level": player.level
                        }
                    ))
        
        # Mid game leveling
        elif game_state.is_mid_game():
            if player.level < 6 and player.gold >= 20:
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message="Good economy - consider leveling to 6 for 4-cost access",
                    priority=6,
                    timestamp=datetime.now(),
                    context={
                        "current_level": player.level,
                        "gold": player.gold
                    }
                ))
        
        # Late game leveling
        elif game_state.is_late_game():
            if player.level < 8 and player.gold >= 30:
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message="Late game - prioritize leveling to 8 for 5-cost champions",
                    priority=9,
                    timestamp=datetime.now(),
                    context={
                        "current_level": player.level,
                        "gold": player.gold
                    }
                ))
        
        return suggestions
    
    def _check_reroll_advice(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Provide reroll timing advice"""
        suggestions = []
        player = game_state.player
        
        # Don't reroll if low on gold and healthy
        if player.gold < 20 and player.health > self.safe_health_threshold:
            if game_state.is_early_game():
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message="Avoid rerolling - focus on economy in early game",
                    priority=6,
                    timestamp=datetime.now(),
                    context={
                        "gold": player.gold,
                        "health": player.health,
                        "phase": "early_game"
                    }
                ))
        
        # Reroll aggressively if low health
        elif player.health <= self.critical_health_threshold:
            if player.gold >= 10:
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message="Low health! Reroll aggressively to stabilize",
                    priority=10,
                    timestamp=datetime.now(),
                    context={
                        "gold": player.gold,
                        "health": player.health,
                        "urgency": "critical"
                    }
                ))
        
        # Power spike timings
        elif game_state.stage == 3 and game_state.round >= 2:
            if player.level >= 6 and player.gold >= 30:
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message="Good time to reroll for 3-cost upgrades at level 6",
                    priority=7,
                    timestamp=datetime.now(),
                    context={
                        "stage": game_state.stage,
                        "level": player.level,
                        "gold": player.gold
                    }
                ))
        
        return suggestions
    
    def _check_emergency_economy(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Check for emergency economy situations"""
        suggestions = []
        player = game_state.player
        
        # Very low health - prioritize immediate power
        if player.health <= 10:
            suggestions.append(CoachingSuggestion(
                type="economy",
                message="EMERGENCY: Spend all gold to survive!",
                priority=10,
                timestamp=datetime.now(),
                context={
                    "health": player.health,
                    "urgency": "emergency"
                }
            ))
        
        # Low health but manageable
        elif player.health <= self.critical_health_threshold:
            suggestions.append(CoachingSuggestion(
                type="economy",
                message="Low health - prioritize board strength over economy",
                priority=9,
                timestamp=datetime.now(),
                context={
                    "health": player.health,
                    "urgency": "high"
                }
            ))
        
        return suggestions
    
    def _check_winstreak_economy(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Check for win streak economy opportunities"""
        suggestions = []
        player = game_state.player
        
        # Win streak bonus economy
        if player.win_streak >= 3:
            if player.gold >= 30:
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message=f"Win streak active! Extra gold from streak - consider investing",
                    priority=6,
                    timestamp=datetime.now(),
                    context={
                        "win_streak": player.win_streak,
                        "gold": player.gold
                    }
                ))
        
        # Loss streak economy
        elif player.loss_streak >= 3:
            if player.gold >= 20:
                suggestions.append(CoachingSuggestion(
                    type="economy",
                    message="Loss streak - good time to invest in board strength",
                    priority=7,
                    timestamp=datetime.now(),
                    context={
                        "loss_streak": player.loss_streak,
                        "gold": player.gold
                    }
                ))
        
        return suggestions
    
    def analyze_economy_trend(self, recent_states: List[GameState]) -> Dict[str, Any]:
        """Analyze economy trends over recent game states"""
        if len(recent_states) < 2:
            return {}
        
        current = recent_states[-1]
        previous = recent_states[-2]
        
        gold_change = current.player.gold - previous.player.gold
        level_change = current.player.level - previous.player.level
        
        return {
            "gold_trend": "increasing" if gold_change > 0 else "decreasing" if gold_change < 0 else "stable",
            "gold_change": gold_change,
            "level_change": level_change,
            "economy_strength": current.get_economy_strength(),
            "interest_efficiency": min(current.player.gold // 10, 5)
        }