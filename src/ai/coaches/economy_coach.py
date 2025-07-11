"""
TFT Economy Overlay - Economy Coach (Simplified)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EconomySuggestion:
    """Simple economy suggestion"""
    message: str
    priority: int  # 1-10, higher is more urgent
    timestamp: datetime
    category: str  # interest, leveling, spending, emergency

class EconomyCoach:
    """Simplified economy management coach focused on gold/health decisions"""
    
    def __init__(self):
        self.name = "Economy Coach"
        
        # Economy rules and thresholds
        self.interest_breakpoints = [10, 20, 30, 40, 50]
        self.safe_health_threshold = 40
        self.critical_health_threshold = 20
        self.emergency_health_threshold = 10
        
    def get_suggestions(self, gold: int, health: int, level: int, stage: int, round_num: int) -> List[EconomySuggestion]:
        """Generate economy suggestions based on current game state"""
        suggestions = []
        
        try:
            # Emergency spending (very low health)
            if health <= self.emergency_health_threshold:
                suggestions.append(EconomySuggestion(
                    message=f"EMERGENCY: {health} HP! Spend all {gold} gold to survive!",
                    priority=10,
                    timestamp=datetime.now(),
                    category="emergency"
                ))
                return suggestions  # Return only emergency advice
            
            # Critical health - prioritize strength over economy
            elif health <= self.critical_health_threshold:
                suggestions.append(EconomySuggestion(
                    message=f"Low health ({health} HP) - prioritize board strength over economy",
                    priority=9,
                    timestamp=datetime.now(),
                    category="emergency"
                ))
            
            # Interest optimization
            suggestions.extend(self._check_interest_optimization(gold, health))
            
            # Leveling advice
            suggestions.extend(self._check_leveling_advice(gold, health, level, stage, round_num))
            
            # Stage-specific economy advice
            suggestions.extend(self._check_stage_economy(gold, health, level, stage, round_num))
            
        except Exception as e:
            logger.error(f"Error generating economy suggestions: {e}")
        
        return suggestions
    
    def _check_interest_optimization(self, gold: int, health: int) -> List[EconomySuggestion]:
        """Check for interest optimization opportunities"""
        suggestions = []
        
        # Don't optimize interest if health is critical
        if health <= self.critical_health_threshold:
            return suggestions
        
        # Find next interest breakpoint
        next_breakpoint = None
        for breakpoint in self.interest_breakpoints:
            if gold < breakpoint:
                next_breakpoint = breakpoint
                break
        
        if next_breakpoint:
            difference = next_breakpoint - gold
            
            if difference <= 3 and gold >= 7:  # Close to breakpoint
                suggestions.append(EconomySuggestion(
                    message=f"Save {difference} more gold to reach {next_breakpoint} (next interest breakpoint)",
                    priority=6,
                    timestamp=datetime.now(),
                    category="interest"
                ))
            elif gold >= 50:  # Max interest
                suggestions.append(EconomySuggestion(
                    message="At max interest (50g) - consider spending excess gold",
                    priority=5,
                    timestamp=datetime.now(),
                    category="interest"
                ))
        
        return suggestions
    
    def _check_leveling_advice(self, gold: int, health: int, level: int, stage: int, round_num: int) -> List[EconomySuggestion]:
        """Provide leveling timing advice"""
        suggestions = []
        
        # Early game leveling (Stage 2)
        if stage == 2:
            if round_num == 5 and level < 5 and gold >= 8:  # 2-5 level to 5
                suggestions.append(EconomySuggestion(
                    message="Standard timing: Level to 5 at 2-5 for better economy",
                    priority=7,
                    timestamp=datetime.now(),
                    category="leveling"
                ))
        
        # Mid game leveling (Stage 3-4)
        elif stage in [3, 4]:
            if level < 6 and gold >= 20 and health > self.safe_health_threshold:
                suggestions.append(EconomySuggestion(
                    message="Good economy - consider leveling to 6 for 4-cost champion access",
                    priority=6,
                    timestamp=datetime.now(),
                    category="leveling"
                ))
            elif level < 7 and gold >= 28 and stage == 4:
                suggestions.append(EconomySuggestion(
                    message="Stage 4 - consider pushing to level 7 if stabilized",
                    priority=7,
                    timestamp=datetime.now(),
                    category="leveling"
                ))
        
        # Late game leveling (Stage 5+)
        elif stage >= 5:
            if level < 8 and gold >= 30:
                suggestions.append(EconomySuggestion(
                    message="Late game - prioritize leveling to 8 for 5-cost champions",
                    priority=9,
                    timestamp=datetime.now(),
                    category="leveling"
                ))
        
        return suggestions
    
    def _check_stage_economy(self, gold: int, health: int, level: int, stage: int, round_num: int) -> List[EconomySuggestion]:
        """Stage-specific economy advice"""
        suggestions = []
        
        # Early game (Stage 1-2)
        if stage <= 2:
            if gold < 10 and health > 80:
                suggestions.append(EconomySuggestion(
                    message="Early game - focus on building economy, avoid excessive spending",
                    priority=5,
                    timestamp=datetime.now(),
                    category="spending"
                ))
        
        # Krugs (Stage 3-1)
        elif stage == 3 and round_num == 1:
            if gold >= 30:
                suggestions.append(EconomySuggestion(
                    message="Pre-Krugs: Good economy. Consider small upgrades for PvE round",
                    priority=6,
                    timestamp=datetime.now(),
                    category="spending"
                ))
        
        # Power spike timing (Stage 3-2, first PvP after Krugs)
        elif stage == 3 and round_num == 2:
            if level >= 6 and gold >= 20:
                suggestions.append(EconomySuggestion(
                    message="Post-Krugs power spike: Good time to roll for 3-cost upgrades at level 6",
                    priority=7,
                    timestamp=datetime.now(),
                    category="spending"
                ))
        
        return suggestions
    
    def get_economy_status(self, gold: int, health: int, level: int, stage: int) -> Dict[str, Any]:
        """Get overall economy status summary"""
        # Calculate interest
        interest = min(gold // 10, 5)
        
        # Determine economy strength
        if stage <= 2:
            # Early game benchmarks
            target_gold = stage * 15  # Rough benchmark
            economy_strength = "strong" if gold >= target_gold else "average" if gold >= target_gold * 0.7 else "weak"
        else:
            # Mid/late game - more complex
            if gold >= 50:
                economy_strength = "very_strong"
            elif gold >= 30:
                economy_strength = "strong"
            elif gold >= 20:
                economy_strength = "average"
            else:
                economy_strength = "weak"
        
        # Determine spending priority
        if health <= self.emergency_health_threshold:
            spending_priority = "emergency_all"
        elif health <= self.critical_health_threshold:
            spending_priority = "strength_focus"
        elif gold >= 50:
            spending_priority = "can_spend"
        else:
            spending_priority = "save_for_interest"
        
        return {
            "gold": gold,
            "health": health,
            "interest": interest,
            "economy_strength": economy_strength,
            "spending_priority": spending_priority,
            "next_interest_breakpoint": self._get_next_interest_breakpoint(gold)
        }
    
    def _get_next_interest_breakpoint(self, gold: int) -> Optional[int]:
        """Get the next interest breakpoint"""
        for breakpoint in self.interest_breakpoints:
            if gold < breakpoint:
                return breakpoint
        return None