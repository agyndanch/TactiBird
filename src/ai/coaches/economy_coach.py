"""
TactiBird - Economy Coach Module with Playstyle Support
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class EconomySuggestion:
    """Economy coaching suggestion"""
    type: str  # 'level', 'reroll', 'save', 'interest', 'Economy', 'Rolling'
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
    """AI coach for TFT economy management with playstyle support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.max_suggestions = config.get('max_suggestions', 5)
        self.current_playstyle = None
        
        # Economy thresholds
        self.interest_breakpoints = [10, 20, 30, 40, 50]
        self.level_costs = {2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 80, 9: 110}
        
        # Stage-specific economy guidelines
        self.economy_guidelines = self._init_economy_guidelines()
        
        # Playstyle-specific strategies
        self.playstyle_strategies = self._init_playstyle_strategies()
        
        logger.info("Economy coach initialized with playstyle support")
    
    def set_playstyle(self, playstyle: str):
        """Set the current playstyle for contextualized suggestions"""
        self.current_playstyle = playstyle
        logger.info(f"Playstyle set to: {playstyle}")
    
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
    
    def _init_playstyle_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize playstyle-specific strategies"""
        return {
            "1-cost-reroll": {
                "name": "1-Cost Reroll",
                "description": "Focus on 3-starring 1-cost units",
                "economy_priority": "high",
                "level_strategy": "stay_low",
                "reroll_timing": "early",
                "target_levels": [4, 5],
                "gold_thresholds": {
                    "start_rolling": 50,
                    "minimum_econ": 30,
                    "all_in": 30
                }
            },
            "2-cost-reroll": {
                "name": "2-Cost Reroll",
                "description": "Focus on 3-starring 2-cost units",
                "economy_priority": "high",
                "level_strategy": "slow_level",
                "reroll_timing": "mid",
                "target_levels": [5, 6],
                "gold_thresholds": {
                    "start_rolling": 50,
                    "minimum_econ": 20,
                    "all_in": 20
                }
            },
            "3-cost-reroll": {
                "name": "3-Cost Reroll",
                "description": "Focus on 3-starring 3-cost units",
                "economy_priority": "medium",
                "level_strategy": "normal",
                "reroll_timing": "mid_late",
                "target_levels": [6, 7],
                "gold_thresholds": {
                    "start_rolling": 50,
                    "minimum_econ": 30,
                    "all_in": 40
                }
            },
            "fast-8": {
                "name": "Fast 8",
                "description": "Rush to level 8 for 5-cost units",
                "economy_priority": "critical",
                "level_strategy": "aggressive",
                "reroll_timing": "late",
                "target_levels": [8, 9],
                "gold_thresholds": {
                    "start_rolling": 50,
                    "minimum_econ": 50,
                    "all_in": 0
                }
            }
        }
    
    async def get_suggestions(self, game_state) -> List[Dict[str, Any]]:
        """Get economy coaching suggestions based on game state and playstyle"""
        try:
            suggestions = []
            
            if not game_state.stats or not game_state.stats.in_game:
                return suggestions
            
            stats = game_state.stats
            
            # Get playstyle-specific suggestions if playstyle is set
            if self.current_playstyle:
                playstyle_suggestions = await self._get_playstyle_suggestions(stats)
                suggestions.extend(playstyle_suggestions)
            
            # Determine game phase
            phase = self._get_game_phase(stats.stage, stats.round_num)
            guidelines = self.economy_guidelines.get(phase, self.economy_guidelines["mid"])
            
            # Generate general economy suggestions
            await self._check_interest_optimization(stats, guidelines, suggestions)
            await self._check_leveling_opportunity(stats, guidelines, suggestions)
            await self._check_reroll_timing(stats, guidelines, suggestions, game_state)
            await self._check_economy_health(stats, guidelines, suggestions)
            await self._check_all_in_timing(stats, guidelines, suggestions, game_state)
            
            # Remove duplicates and sort by priority
            unique_suggestions = self._remove_duplicate_suggestions(suggestions)
            unique_suggestions.sort(key=lambda x: x['priority'], reverse=True)
            return unique_suggestions[:self.max_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating economy suggestions: {e}")
            return []
    
    async def _get_playstyle_suggestions(self, stats) -> List[Dict[str, Any]]:
        """Generate playstyle-specific suggestions"""
        suggestions = []
        
        if not self.current_playstyle or self.current_playstyle not in self.playstyle_strategies:
            return suggestions
        
        strategy = self.playstyle_strategies[self.current_playstyle]
        gold = stats.gold or 0
        level = stats.level or 1
        stage = stats.stage or 1
        
        try:
            if self.current_playstyle == "1-cost-reroll":
                suggestions.extend(self._get_1_cost_reroll_suggestions(gold, level, stage, strategy))
            elif self.current_playstyle == "2-cost-reroll":
                suggestions.extend(self._get_2_cost_reroll_suggestions(gold, level, stage, strategy))
            elif self.current_playstyle == "3-cost-reroll":
                suggestions.extend(self._get_3_cost_reroll_suggestions(gold, level, stage, strategy))
            elif self.current_playstyle == "fast-8":
                suggestions.extend(self._get_fast_8_suggestions(gold, level, stage, strategy))
            
        except Exception as e:
            logger.error(f"Error generating {self.current_playstyle} suggestions: {e}")
        
        return suggestions
    
    def _get_1_cost_reroll_suggestions(self, gold: int, level: int, stage: int, strategy: Dict) -> List[Dict[str, Any]]:
        """Generate 1-cost reroll specific suggestions"""
        suggestions = []
        
        # Level guidance
        if level < 6:
            suggestions.append({
                'type': 'Economy',
                'priority': 4,
                'message': 'Stay at level 4-5. Don\'t level until you have your 1-cost carry 3-starred.',
                'reasoning': '1-cost reroll strategy requires staying low level for better 1-cost odds',
                'confidence': 0.9
            })
        
        # Rolling guidance
        if gold > strategy["gold_thresholds"]["start_rolling"]:
            suggestions.append({
                'type': 'Rolling',
                'priority': 4,
                'message': 'Start rolling down to find your 1-cost carry. Maintain 50+ gold for interest.',
                'reasoning': 'Sufficient gold to start rolling while maintaining economy',
                'confidence': 0.85
            })
        elif gold < strategy["gold_thresholds"]["minimum_econ"] and stage >= 3:
            suggestions.append({
                'type': 'Economy',
                'priority': 5,
                'message': 'Focus on building economy before rolling. You need more gold.',
                'reasoning': 'Insufficient gold for 1-cost reroll strategy',
                'confidence': 0.9
            })
        
        # Stage-specific advice
        if stage >= 4 and level > 6:
            suggestions.append({
                'type': 'Economy',
                'priority': 3,
                'message': 'Consider transitioning - you may be too high level for 1-cost reroll.',
                'reasoning': 'Level too high for optimal 1-cost odds',
                'confidence': 0.8
            })
        
        return suggestions
    
    def _get_2_cost_reroll_suggestions(self, gold: int, level: int, stage: int, strategy: Dict) -> List[Dict[str, Any]]:
        """Generate 2-cost reroll specific suggestions"""
        suggestions = []
        
        # Level guidance
        if level < 5 and stage >= 3:
            suggestions.append({
                'type': 'Economy',
                'priority': 3,
                'message': 'Level to 5 for better 2-cost odds, then stabilize.',
                'reasoning': 'Level 5 provides optimal 2-cost unit odds',
                'confidence': 0.85
            })
        elif level > 6 and stage < 5:
            suggestions.append({
                'type': 'Economy',
                'priority': 4,
                'message': 'You\'re too high level for 2-cost reroll. Consider fast-8 transition.',
                'reasoning': 'Level too high for optimal 2-cost strategy',
                'confidence': 0.8
            })
        
        # Rolling guidance
        if gold > strategy["gold_thresholds"]["start_rolling"] and level >= 5:
            suggestions.append({
                'type': 'Rolling',
                'priority': 4,
                'message': 'Start rolling for your 2-cost carries. Maintain some economy.',
                'reasoning': 'Good position to start rolling for 2-cost units',
                'confidence': 0.85
            })
        
        return suggestions
    
    def _get_3_cost_reroll_suggestions(self, gold: int, level: int, stage: int, strategy: Dict) -> List[Dict[str, Any]]:
        """Generate 3-cost reroll specific suggestions"""
        suggestions = []
        
        # Level guidance
        if level < 6 and stage >= 4:
            suggestions.append({
                'type': 'Economy',
                'priority': 4,
                'message': 'Level to 6 for optimal 3-cost odds.',
                'reasoning': 'Level 6 provides best 3-cost unit odds',
                'confidence': 0.85
            })
        
        # Rolling guidance
        if gold > strategy["gold_thresholds"]["start_rolling"] and level >= 6:
            suggestions.append({
                'type': 'Rolling',
                'priority': 4,
                'message': 'Good position to roll for 3-cost upgrades.',
                'reasoning': 'Sufficient gold and level for 3-cost rolling',
                'confidence': 0.8
            })
        elif level < 6 and gold > 30:
            suggestions.append({
                'type': 'Economy',
                'priority': 3,
                'message': 'Consider leveling to 6 before rolling for 3-costs.',
                'reasoning': 'Better 3-cost odds at level 6',
                'confidence': 0.75
            })
        
        return suggestions
    
    def _get_fast_8_suggestions(self, gold: int, level: int, stage: int, strategy: Dict) -> List[Dict[str, Any]]:
        """Generate fast-8 specific suggestions"""
        suggestions = []
        
        # Economy requirements
        if stage >= 4 and gold < strategy["gold_thresholds"]["minimum_econ"]:
            suggestions.append({
                'type': 'Economy',
                'priority': 5,
                'message': 'Fast 8 needs strong economy. Focus on econ before aggressive leveling.',
                'reasoning': 'Insufficient gold for fast-8 strategy',
                'confidence': 0.9
            })
        
        # Leveling guidance
        if level < 8 and gold >= strategy["gold_thresholds"]["minimum_econ"] and stage >= 4:
            suggestions.append({
                'type': 'Economy',
                'priority': 4,
                'message': 'Continue leveling to 8. Don\'t get distracted by rolling.',
                'reasoning': 'Fast-8 requires reaching level 8 quickly',
                'confidence': 0.85
            })
        
        # Rolling guidance
        if level >= 8 and gold > 30:
            suggestions.append({
                'type': 'Rolling',
                'priority': 5,
                'message': 'You\'re at 8! Start rolling for 4-cost and 5-cost units.',
                'reasoning': 'Reached target level for fast-8 strategy',
                'confidence': 0.9
            })
        elif level >= 8 and gold < 20:
            suggestions.append({
                'type': 'Economy',
                'priority': 4,
                'message': 'At level 8 but need more gold to roll effectively.',
                'reasoning': 'Insufficient gold for effective rolling at level 8',
                'confidence': 0.8
            })
        
        return suggestions
    
    def _remove_duplicate_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate suggestions based on message similarity"""
        unique_suggestions = []
        seen_messages = set()
        
        for suggestion in suggestions:
            message_key = suggestion['message'].lower().strip()
            if message_key not in seen_messages:
                seen_messages.add(message_key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    async def _check_interest_optimization(self, stats, guidelines: Dict[str, Any], suggestions: List[Dict[str, Any]]):
        """Check for interest optimization opportunities"""
        try:
            gold = stats.gold or 0
            
            # Check if player is close to next interest breakpoint
            for breakpoint in self.interest_breakpoints:
                if gold < breakpoint and gold >= breakpoint - 5:
                    gold_needed = breakpoint - gold
                    suggestion = EconomySuggestion(
                        type="interest",
                        priority=3,
                        message=f"Consider saving {gold_needed} more gold to reach {breakpoint} (next interest tier)",
                        reasoning=f"Currently at {gold} gold, {gold_needed} away from {breakpoint} interest tier",
                        confidence=0.8
                    )
                    suggestions.append(suggestion.to_dict())
                    break
                    
        except Exception as e:
            logger.debug(f"Error checking interest optimization: {e}")
    
    async def _check_leveling_opportunity(self, stats, guidelines: Dict[str, Any], suggestions: List[Dict[str, Any]]):
        """Check for leveling opportunities"""
        try:
            gold = stats.gold or 0
            level = stats.level or 1
            
            if level < 9:  # Can still level up
                level_cost = self.level_costs.get(level + 1, 0)
                
                # Check if player can afford to level and maintain some economy
                if gold >= level_cost + 20:  # Level cost + some buffer
                    priority = self._calculate_level_priority(stats, guidelines)
                    reasoning = self._get_level_reasoning(stats, guidelines)
                    
                    suggestion = EconomySuggestion(
                        type="level",
                        priority=priority,
                        message=f"Consider leveling to {level + 1} (costs {level_cost} gold)",
                        reasoning=reasoning,
                        gold_cost=level_cost,
                        confidence=0.75
                    )
                    suggestions.append(suggestion.to_dict())
                    
        except Exception as e:
            logger.debug(f"Error checking leveling opportunity: {e}")
    
    async def _check_reroll_timing(self, stats, guidelines: Dict[str, Any], suggestions: List[Dict[str, Any]], game_state):
        """Check for optimal reroll timing"""
        try:
            gold = stats.gold or 0
            health = stats.health or 100
            
            reroll_threshold = guidelines.get("reroll_threshold", 20)
            
            # High priority reroll if low health and sufficient gold
            if health <= 30 and gold >= reroll_threshold + 10:
                suggestion = EconomySuggestion(
                    type="reroll",
                    priority=4,
                    message="Low health - consider rolling to strengthen your board",
                    reasoning=f"Health at {health}, rolling may be necessary for survival",
                    confidence=0.8
                )
                suggestions.append(suggestion.to_dict())
            
            # Suggest holding if gold is below threshold
            elif gold < reroll_threshold:
                suggestion = EconomySuggestion(
                    type="save",
                    priority=2,
                    message=f"Consider saving gold before rolling (current: {gold}, suggested minimum: {reroll_threshold})",
                    reasoning="Building economy before rolling improves long-term position",
                    confidence=0.7
                )
                suggestions.append(suggestion.to_dict())
                
        except Exception as e:
            logger.debug(f"Error checking reroll timing: {e}")
    
    async def _check_economy_health(self, stats, guidelines: Dict[str, Any], suggestions: List[Dict[str, Any]]):
        """Check overall economy health"""
        try:
            gold = stats.gold or 0
            stage = stats.stage or 1
            
            target_gold = guidelines.get("target_gold", 30)
            
            # Warn if economy is behind target for current stage
            if gold < target_gold * 0.7:  # 30% below target
                suggestion = EconomySuggestion(
                    type="save",
                    priority=3,
                    message=f"Economy behind target for stage {stage}. Focus on building gold.",
                    reasoning=f"Current gold: {gold}, target for stage: {target_gold}",
                    confidence=0.8
                )
                suggestions.append(suggestion.to_dict())
                
        except Exception as e:
            logger.debug(f"Error checking economy health: {e}")
    
    async def _check_all_in_timing(self, stats, guidelines: Dict[str, Any], suggestions: List[Dict[str, Any]], game_state):
        """Check if it's time to go all-in"""
        try:
            health = stats.health or 100
            gold = stats.gold or 0
            current_stage = stats.stage or 1
            
            # Critical health - suggest all-in
            if health <= 20:
                suggestion = EconomySuggestion(
                    type="reroll",
                    priority=5,
                    message="Critical health! Consider going all-in to survive.",
                    reasoning="Economy is less important than survival.",
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
            return 0.5  # Placeholder
        except Exception as e:
            logger.debug(f"Error evaluating composition strength: {e}")
            return 0.5