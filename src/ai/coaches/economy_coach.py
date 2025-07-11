"""
TactiBird - Economy Coach Module
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class EconomySuggestion:
    """Economy coaching suggestion"""
    type: str  # 'level', 'reroll', 'save', 'interest'
    priority: int  # 1-5, 5 being highest
    message: str
    reasoning: str
    gold_cost: int = 0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'priority': self.priority,
            'message': self.message,
            'reasoning': self.reasoning,
            'gold_cost': self.gold_cost,
            'confidence': self.confidence
        }

class EconomyCoach:
    """AI coach for TFT economy management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.max_suggestions = config.get('max_suggestions', 5)
        
        # Economy thresholds
        self.interest_breakpoints = [10, 20, 30, 40, 50]
        self.level_costs = {2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 80, 9: 110}
        
        # Stage-specific economy guidelines
        self.economy_guidelines = self._init_economy_guidelines()
        
        logger.info("Economy coach initialized")
    
    def _init_economy_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Initialize stage-specific economy guidelines"""
        return {
            "early": {  # Stage 1-2
                "target_gold": 20,
                "level_priority": "low",
                "reroll_threshold": 30,
                "interest_priority": "high"
            },
            "mid": {  # Stage 3-4
                "target_gold": 30,
                "level_priority": "medium", 
                "reroll_threshold": 20,
                "interest_priority": "medium"
            },
            "late": {  # Stage 5+
                "target_gold": 40,
                "level_priority": "high",
                "reroll_threshold": 10,
                "interest_priority": "low"
            }
        }
    
    async def get_suggestions(self, game_state) -> List[Dict[str, Any]]:
        """Get economy coaching suggestions based on game state"""
        try:
            suggestions = []
            
            if not game_state.stats or not game_state.stats.in_game:
                return suggestions
            
            stats = game_state.stats
            
            # Determine game phase
            phase = self._get_game_phase(stats.stage, stats.round_num)
            guidelines = self.economy_guidelines.get(phase, self.economy_guidelines["mid"])
            
            # Generate suggestions based on current state
            await self._check_interest_optimization(stats, guidelines, suggestions)
            await self._check_leveling_opportunity(stats, guidelines, suggestions)
            await self._check_reroll_timing(stats, guidelines, suggestions, game_state)
            await self._check_economy_health(stats, guidelines, suggestions)
            await self._check_all_in_timing(stats, guidelines, suggestions, game_state)
            
            # Sort by priority and return top suggestions
            suggestions.sort(key=lambda x: x['priority'], reverse=True)
            return suggestions[:self.max_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating economy suggestions: {e}")
            return []
    
    async def _check_interest_optimization(self, stats, guidelines: Dict[str, Any], 
                                         suggestions: List[Dict[str, Any]]):
        """Check for interest optimization opportunities"""
        try:
            current_gold = stats.gold
            
            # Find next interest breakpoint
            next_breakpoint = None
            for breakpoint in self.interest_breakpoints:
                if current_gold < breakpoint:
                    next_breakpoint = breakpoint
                    break
            
            if next_breakpoint:
                gold_needed = next_breakpoint - current_gold
                
                # If close to breakpoint, suggest saving
                if 1 <= gold_needed <= 5 and guidelines["interest_priority"] != "low":
                    suggestion = EconomySuggestion(
                        type="interest",
                        priority=4,
                        message=f"Save {gold_needed} gold to reach {next_breakpoint}g interest breakpoint",
                        reasoning=f"Interest income increases from {current_gold // 10} to {next_breakpoint // 10} gold per round",
                        gold_cost=-gold_needed,  # Negative because it's saving
                        confidence=0.9
                    )
                    suggestions.append(suggestion.to_dict())
            
            # Warn about gold overflow
            if current_gold > 50:
                suggestion = EconomySuggestion(
                    type="save",
                    priority=5,
                    message="Consider spending gold - maximum interest reached",
                    reasoning="You're at max interest (5g/round). Extra gold provides no benefit.",
                    confidence=1.0
                )
                suggestions.append(suggestion.to_dict())
                
        except Exception as e:
            logger.debug(f"Error checking interest optimization: {e}")
    
    async def _check_leveling_opportunity(self, stats, guidelines: Dict[str, Any], 
                                        suggestions: List[Dict[str, Any]]):
        """Check if leveling up is advisable"""
        try:
            current_level = stats.level
            current_gold = stats.gold
            
            if current_level >= 9:  # Max level
                return
            
            level_cost = self.level_costs.get(current_level + 1, 999)
            
            # Check if we can afford to level
            if current_gold >= level_cost:
                priority = self._calculate_level_priority(stats, guidelines)
                
                if priority > 2:  # Only suggest if reasonably important
                    suggestion = EconomySuggestion(
                        type="level",
                        priority=priority,
                        message=f"Level to {current_level + 1} ({level_cost}g)",
                        reasoning=self._get_level_reasoning(stats, guidelines),
                        gold_cost=level_cost,
                        confidence=0.8
                    )
                    suggestions.append(suggestion.to_dict())
                    
        except Exception as e:
            logger.debug(f"Error checking leveling opportunity: {e}")
    
    async def _check_reroll_timing(self, stats, guidelines: Dict[str, Any], 
                                 suggestions: List[Dict[str, Any]], game_state):
        """Check if rerolling is advisable"""
        try:
            current_gold = stats.gold
            
            # Don't suggest rerolls if we can't afford them
            if current_gold < 2:
                return
            
            # Check if we should be rerolling based on comp strength
            if hasattr(game_state, 'board_state') and game_state.board_state:
                comp_strength = self._evaluate_comp_strength(game_state.board_state)
                
                if comp_strength < 0.6:  # Weak composition
                    if current_gold >= guidelines["reroll_threshold"]:
                        suggestion = EconomySuggestion(
                            type="reroll",
                            priority=3,
                            message="Consider rerolling for stronger units",
                            reasoning="Current board composition is weak. Rerolling may improve your chances.",
                            gold_cost=2,
                            confidence=0.7
                        )
                        suggestions.append(suggestion.to_dict())
                        
        except Exception as e:
            logger.debug(f"Error checking reroll timing: {e}")
    
    async def _check_economy_health(self, stats, guidelines: Dict[str, Any], 
                                  suggestions: List[Dict[str, Any]]):
        """Check overall economy health and provide warnings"""
        try:
            current_gold = stats.gold
            current_health = stats.health
            
            # Low gold warning
            if current_gold < 10 and stats.stage >= 3:
                suggestion = EconomySuggestion(
                    type="save",
                    priority=4,
                    message="Economy is low - prioritize saving",
                    reasoning="Low gold reduces your options. Focus on economy unless health is critical.",
                    confidence=0.8
                )
                suggestions.append(suggestion.to_dict())
            
            # Health vs economy balance
            if current_health <= 20:
                suggestion = EconomySuggestion(
                    type="reroll",
                    priority=5,
                    message="Low health - prioritize board strength over economy",
                    reasoning="Health is critical. Consider spending gold to stabilize your board.",
                    confidence=0.9
                )
                suggestions.append(suggestion.to_dict())
                
        except Exception as e:
            logger.debug(f"Error checking economy health: {e}")
    
    async def _check_all_in_timing(self, stats, guidelines: Dict[str, Any], 
                                 suggestions: List[Dict[str, Any]], game_state):
        """Check if it's time to go all-in"""
        try:
            current_health = stats.health
            current_stage = stats.stage
            current_round = stats.round_num
            
            # Stage 6+ with low health = all-in time
            if current_stage >= 6 and current_health <= 30:
                suggestion = EconomySuggestion(
                    type="reroll",
                    priority=5,
                    message="All-in time - spend gold to maximize board strength",
                    reasoning="Late game with low health. Economy is less important than survival.",
                    confidence=0.95
                )
                suggestions.append(suggestion.to_dict())
            
            # Final stages
            elif current_stage >= 7:
                suggestion = EconomySuggestion(
                    type="reroll",
                    priority=4,
                    message="Late game - prioritize board strength",
                    reasoning="Game is ending soon. Focus on maximizing your current board.",
                    confidence=0.9
                )
                suggestions.append(suggestion.to_dict())
                
        except Exception as e:
            logger.debug(f"Error checking all-in timing: {e}")
    
    def _get_game_phase(self, stage: int, round_num: int) -> str:
        """Determine current game phase"""
        if stage <= 2:
            return "early"
        elif stage <= 4:
            return "mid"
        else:
            return "late"
    
    def _calculate_level_priority(self, stats, guidelines: Dict[str, Any]) -> int:
        """Calculate priority for leveling up"""
        priority = 1
        
        # Base priority from guidelines
        level_priority = guidelines.get("level_priority", "medium")
        if level_priority == "high":
            priority += 2
        elif level_priority == "medium":
            priority += 1
        
        # Adjust based on current level and stage
        if stats.level < stats.stage + 1:  # Behind on levels
            priority += 1
        
        # Adjust based on health
        if stats.health <= 30:
            priority += 1
        
        return min(priority, 5)
    
    def _get_level_reasoning(self, stats, guidelines: Dict[str, Any]) -> str:
        """Get reasoning for level suggestion"""
        reasons = []
        
        if stats.level < stats.stage + 1:
            reasons.append("behind on levels for current stage")
        
        if guidelines.get("level_priority") == "high":
            reasons.append("high priority phase for leveling")
        
        if stats.health <= 30:
            reasons.append("low health requires stronger board")
        
        if not reasons:
            reasons.append("good opportunity to gain board strength")
        
        return "Level up: " + ", ".join(reasons)
    
    def _evaluate_comp_strength(self, board_state) -> float:
        """Evaluate current composition strength (0.0 - 1.0)"""
        try:
            # This is a simplified evaluation
            # In a real implementation, this would analyze synergies, unit tiers, etc.
            
            if not hasattr(board_state, 'champions') or not board_state.champions:
                return 0.0
            
            # Count units and estimate strength
            unit_count = len(board_state.champions)
            
            # Base strength on unit count
            strength = min(unit_count / 8, 1.0)  # Full board = 8 units
            
            # TODO: Add synergy analysis, unit tier analysis, etc.
            
            return strength
            
        except Exception as e:
            logger.debug(f"Error evaluating comp strength: {e}")
            return 0.5  # Default to medium strength