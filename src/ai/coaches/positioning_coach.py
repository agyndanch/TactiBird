"""
TactiBird Overlay - Positioning Coach
"""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
import math

from src.ai.coaches.base_coach import BaseCoach, CoachingSuggestion
from src.data.models.game_state import GameState, Champion, Position

logger = logging.getLogger(__name__)

class PositioningCoach(BaseCoach):
    """Provides positioning and board arrangement coaching"""
    
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.name = "Positioning Coach"
        
        # Board layout (TFT hex grid)
        self.board_width = 7
        self.board_height = 4
        
        # Positioning concepts
        self.frontline_rows = [0, 1]  # Bottom two rows
        self.backline_rows = [2, 3]   # Top two rows
        self.corner_positions = [(0, 1), (0, 2), (6, 1), (6, 2)]
        self.center_positions = [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2)]
        
    def get_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Generate positioning-focused suggestions"""
        suggestions = []
        
        try:
            # Analyze current positioning
            pos_analysis = self._analyze_current_positioning(game_state)
            
            # Generate suggestions
            suggestions.extend(self._suggest_frontline_positioning(game_state, pos_analysis))
            suggestions.extend(self._suggest_backline_positioning(game_state, pos_analysis))
            suggestions.extend(self._suggest_carry_protection(game_state, pos_analysis))
            suggestions.extend(self._suggest_spread_vs_clump(game_state))
            suggestions.extend(self._suggest_corner_positioning(game_state))
            suggestions.extend(self._suggest_counter_positioning(game_state))
            
        except Exception as e:
            logger.error(f"Error generating positioning suggestions: {e}")
        
        return suggestions
    
    def _analyze_current_positioning(self, game_state: GameState) -> Dict[str, Any]:
        """Analyze current board positioning"""
        analysis = {
            "frontline_count": 0,
            "backline_count": 0,
            "carries_exposed": [],
            "tanks_mispositioned": [],
            "formation": "unknown",
            "spread_factor": 0.0,
            "vulnerability_score": 0.0
        }
        
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        
        for champion in game_state.board.champions:
            if champion.position:
                row = champion.position.y
                
                # Count frontline vs backline
                if row in self.frontline_rows:
                    analysis["frontline_count"] += 1
                elif self._is_traditional_tank(champion):
                tanks.append(champion)
        
        return tanks
    
    def _identify_formation(self, game_state: GameState) -> str:
        """Identify current formation type"""
        positions = [champ.position for champ in game_state.board.champions if champ.position]
        
        if not positions:
            return "unknown"
        
        # Calculate formation characteristics
        front_heavy = sum(1 for pos in positions if pos.y <= 1) / len(positions)
        back_heavy = sum(1 for pos in positions if pos.y >= 2) / len(positions)
        spread_x = max(pos.x for pos in positions) - min(pos.x for pos in positions) if positions else 0
        
        if front_heavy > 0.6:
            return "aggressive"
        elif back_heavy > 0.6:
            return "defensive"
        elif spread_x >= 5:
            return "spread"
        else:
            return "compact"
    
    def _calculate_spread_factor(self, game_state: GameState) -> float:
        """Calculate how spread out the units are (0=clumped, 1=spread)"""
        positions = [champ.position for champ in game_state.board.champions if champ.position]
        
        if len(positions) < 2:
            return 0.0
        
        # Calculate average distance between all units
        total_distance = 0
        count = 0
        
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                distance = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
                total_distance += distance
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        max_distance = math.sqrt(self.board_width**2 + self.board_height**2)
        
        return min(1.0, avg_distance / max_distance * 3)  # Scaled factor
    
    def _calculate_vulnerability(self, game_state: GameState, carries: List[Champion]) -> float:
        """Calculate overall vulnerability score"""
        if not carries:
            return 0.0
        
        total_vulnerability = 0
        
        for carry in carries:
            if not carry.position:
                continue
            
            # Distance to nearest tank
            min_tank_distance = self._get_min_tank_distance(carry, game_state)
            
            # Position vulnerability (frontline = more vulnerable)
            position_vuln = 1.0 if carry.position.y <= 1 else 0.5
            
            # Edge vulnerability (corners are safer)
            edge_vuln = 0.3 if self._is_in_corner(carry.position) else 0.7
            
            # Tank protection factor
            tank_protection = max(0, 1.0 - min_tank_distance / 3.0)
            
            carry_vulnerability = (position_vuln + edge_vuln) * (1.0 - tank_protection)
            total_vulnerability += carry_vulnerability
        
        return total_vulnerability / len(carries)
    
    def _calculate_protection_level(self, carry: Champion, game_state: GameState) -> float:
        """Calculate protection level for a specific carry"""
        if not carry.position:
            return 0.0
        
        protection = 0.0
        
        # Backline positioning
        if carry.position.y >= 2:
            protection += 0.4
        
        # Corner positioning
        if self._is_in_corner(carry.position):
            protection += 0.3
        
        # Tank proximity
        min_tank_distance = self._get_min_tank_distance(carry, game_state)
        if min_tank_distance <= 2:
            protection += 0.3
        
        return min(1.0, protection)
    
    def _get_protection_strategies(self, carry: Champion, game_state: GameState) -> List[str]:
        """Get protection strategies for a carry"""
        strategies = []
        
        if not carry.position:
            return strategies
        
        # Move to backline
        if carry.position.y <= 1:
            strategies.append("move to backline")
        
        # Corner positioning
        if not self._is_in_corner(carry.position):
            strategies.append("position in corner")
        
        # Add frontline
        tanks = self._identify_tanks(game_state)
        if len(tanks) < 2:
            strategies.append("add more frontline units")
        
        # Spread positioning
        if self._calculate_spread_factor(game_state) < 0.3:
            strategies.append("spread units to avoid AOE")
        
        return strategies
    
    def _get_min_tank_distance(self, carry: Champion, game_state: GameState) -> float:
        """Get minimum distance to nearest tank"""
        if not carry.position:
            return float('inf')
        
        tanks = self._identify_tanks(game_state)
        
        if not tanks:
            return float('inf')
        
        min_distance = float('inf')
        
        for tank in tanks:
            if tank.position:
                distance = math.sqrt(
                    (carry.position.x - tank.position.x)**2 + 
                    (carry.position.y - tank.position.y)**2
                )
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _is_in_corner(self, position: Position) -> bool:
        """Check if position is in a corner"""
        return (position.x, position.y) in self.corner_positions
    
    def _is_ranged_carry(self, champion: Champion) -> bool:
        """Check if champion is a ranged carry"""
        # This would check champion data for range
        # Placeholder implementation
        ranged_champions = ["Jinx", "Tristana", "Lux", "Brand", "Karma"]
        return champion.name in ranged_champions
    
    def _is_carry_item(self, item: str) -> bool:
        """Check if item is a carry item"""
        carry_items = [
            "Infinity Edge", "Last Whisper", "Bloodthirster", 
            "Archangel's Staff", "Rabadon's Deathcap", "Spear of Shojin"
        ]
        return item in carry_items
    
    def _is_tank_item(self, item: str) -> bool:
        """Check if item is a tank item"""
        tank_items = [
            "Bramble Vest", "Dragon's Claw", "Gargoyle Stoneplate", 
            "Warmog's Armor", "Frozen Heart", "Sunfire Cape"
        ]
        return item in tank_items
    
    def _is_traditional_tank(self, champion: Champion) -> bool:
        """Check if champion is traditionally a tank"""
        tank_champions = ["Leona", "Braum", "Nautilus", "Thresh", "Sejuani", "Mundo"]
        return champion.name in tank_champions
    
    def _analyze_enemy_threats(self, game_state: GameState) -> Dict[str, int]:
        """Analyze enemy threat types"""
        # This would analyze opponent compositions if available
        # For now, return placeholder threats
        return {
            "aoe_damage": 2,
            "single_target": 3,
            "assassins": 1,
            "backline_access": 2
        }
    
    def _analyze_enemy_positioning_patterns(self, game_state: GameState) -> Dict[str, int]:
        """Analyze common enemy positioning patterns"""
        # This would analyze opponent positioning if available
        # For now, return placeholder patterns
        return {
            "corner_carry": 3,
            "spread_formation": 2,
            "assassin_flank": 1,
            "frontline_heavy": 2
        }
    
    def _get_counter_positioning(self, enemy_pattern: str) -> str:
        """Get counter-positioning strategy for enemy pattern"""
        counters = {
            "corner_carry": "use assassins or position opposite corner",
            "spread_formation": "clump units and focus fire",
            "assassin_flank": "protect backline with tanks",
            "frontline_heavy": "spread out and use range advantage"
        }
        return counters.get(enemy_pattern, "")
    
    def get_optimal_positioning(self, game_state: GameState) -> Dict[str, Position]:
        """Get optimal positioning suggestions for all champions"""
        suggestions = {}
        
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        supports = [c for c in game_state.board.champions if c not in carries and c not in tanks]
        
        # Position tanks in frontline
        frontline_positions = self._get_available_frontline_positions()
        for i, tank in enumerate(tanks[:len(frontline_positions)]):
            suggestions[tank.name] = frontline_positions[i]
        
        # Position carries in backline corners
        backline_positions = self._get_available_backline_positions()
        corner_positions = [pos for pos in backline_positions if self._is_in_corner(pos)]
        
        for i, carry in enumerate(carries[:len(corner_positions)]):
            suggestions[carry.name] = corner_positions[i]
        
        # Position remaining carries in backline
        remaining_backline = [pos for pos in backline_positions if pos not in corner_positions]
        remaining_carries = carries[len(corner_positions):]
        
        for i, carry in enumerate(remaining_carries[:len(remaining_backline)]):
            suggestions[carry.name] = remaining_backline[i]
        
        # Position supports
        remaining_positions = self._get_remaining_positions(suggestions.values())
        for i, support in enumerate(supports[:len(remaining_positions)]):
            suggestions[support.name] = remaining_positions[i]
        
        return suggestions
    
    def _get_available_frontline_positions(self) -> List[Position]:
        """Get available frontline positions"""
        positions = []
        for row in self.frontline_rows:
            for col in range(1, self.board_width - 1):  # Avoid corners for tanks
                positions.append(Position(col, row))
        return positions
    
    def _get_available_backline_positions(self) -> List[Position]:
        """Get available backline positions"""
        positions = []
        for row in self.backline_rows:
            for col in range(self.board_width):
                positions.append(Position(col, row))
        return sorted(positions, key=lambda p: self._is_in_corner(p), reverse=True)
    
    def _get_remaining_positions(self, used_positions: List[Position]) -> List[Position]:
        """Get remaining available positions"""
        all_positions = []
        for row in range(self.board_height):
            for col in range(self.board_width):
                pos = Position(col, row)
                if pos not in used_positions:
                    all_positions.append(pos)
        return all_positions
    
    def analyze_positioning_matchup(self, game_state: GameState, enemy_positions: List[Position]) -> Dict[str, Any]:
        """Analyze positioning matchup against specific enemy formation"""
        analysis = {
            "threat_level": 0,
            "vulnerabilities": [],
            "counter_moves": [],
            "positioning_score": 0.0
        }
        
        if not enemy_positions:
            return analysis
        
        # Analyze enemy formation pattern
        enemy_pattern = self._identify_enemy_formation(enemy_positions)
        
        # Check for specific threats
        if enemy_pattern == "corner_carry":
            analysis["threat_level"] = 7
            analysis["vulnerabilities"].append("Enemy carry protected in corner")
            analysis["counter_moves"].append("Use assassins or long-range units")
        
        elif enemy_pattern == "spread_formation":
            analysis["threat_level"] = 5
            analysis["vulnerabilities"].append("Enemy spread against AOE")
            analysis["counter_moves"].append("Focus fire with clumped positioning")
        
        elif enemy_pattern == "frontline_heavy":
            analysis["threat_level"] = 6
            analysis["vulnerabilities"].append("Strong frontline but weak backline")
            analysis["counter_moves"].append("Spread out and focus backline")
        
        # Calculate positioning score based on current setup
        analysis["positioning_score"] = self._calculate_positioning_matchup_score(
            game_state, enemy_positions, enemy_pattern
        )
        
        return analysis
    
    def _identify_enemy_formation(self, positions: List[Position]) -> str:
        """Identify enemy formation pattern"""
        if not positions:
            return "unknown"
        
        # Analyze formation characteristics
        front_units = sum(1 for pos in positions if pos.y <= 1)
        back_units = sum(1 for pos in positions if pos.y >= 2)
        corner_units = sum(1 for pos in positions if self._is_in_corner(pos))
        spread_factor = self._calculate_spread_factor_from_positions(positions)
        
        if corner_units >= 2:
            return "corner_carry"
        elif spread_factor > 0.7:
            return "spread_formation"
        elif front_units > back_units * 1.5:
            return "frontline_heavy"
        elif back_units > front_units * 1.5:
            return "backline_heavy"
        else:
            return "balanced"
    
    def _calculate_spread_factor_from_positions(self, positions: List[Position]) -> float:
        """Calculate spread factor from position list"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0
        count = 0
        
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                distance = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
                total_distance += distance
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        max_distance = math.sqrt(self.board_width**2 + self.board_height**2)
        
        return min(1.0, avg_distance / max_distance * 3)
    
    def _calculate_positioning_matchup_score(self, game_state: GameState, 
                                           enemy_positions: List[Position], 
                                           enemy_pattern: str) -> float:
        """Calculate how well positioned we are against enemy"""
        score = 0.5  # Base score
        
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        
        # Score carry protection
        for carry in carries:
            if carry.position:
                protection = self._calculate_protection_level(carry, game_state)
                score += protection * 0.2
        
        # Score based on enemy pattern
        if enemy_pattern == "corner_carry":
            # Reward having assassins or backline access
            assassin_count = sum(1 for c in game_state.board.champions 
                               if "Assassin" in self.data_manager.get_champion_traits(c.name))
            score += min(0.3, assassin_count * 0.1)
        
        elif enemy_pattern == "spread_formation":
            # Reward clumped positioning for focus fire
            our_spread = self._calculate_spread_factor(game_state)
            score += (1.0 - our_spread) * 0.2
        
        elif enemy_pattern == "frontline_heavy":
            # Reward spread positioning to avoid frontline
            our_spread = self._calculate_spread_factor(game_state)
            score += our_spread * 0.2
        
        return min(1.0, max(0.0, score))
    
    def suggest_positioning_adjustments(self, game_state: GameState, 
                                      enemy_analysis: Dict[str, Any]) -> List[CoachingSuggestion]:
        """Suggest specific positioning adjustments based on enemy analysis"""
        suggestions = []
        
        if not enemy_analysis or enemy_analysis["threat_level"] < 5:
            return suggestions
        
        # High threat level - suggest immediate adjustments
        for counter_move in enemy_analysis.get("counter_moves", []):
            suggestions.append(CoachingSuggestion(
                type="positioning",
                message=f"Counter enemy formation: {counter_move}",
                priority=8,
                timestamp=datetime.now(),
                context={
                    "enemy_threat": enemy_analysis["threat_level"],
                    "counter_strategy": counter_move
                }
            ))
        
        # Specific vulnerability fixes
        for vulnerability in enemy_analysis.get("vulnerabilities", []):
            if "corner" in vulnerability.lower():
                suggestions.append(CoachingSuggestion(
                    type="positioning",
                    message="Enemy carry in corner - position assassins for flanking",
                    priority=7,
                    timestamp=datetime.now(),
                    context={"vulnerability": vulnerability}
                ))
            
            elif "spread" in vulnerability.lower():
                suggestions.append(CoachingSuggestion(
                    type="positioning",
                    message="Enemy spread formation - group units for focus fire",
                    priority=6,
                    timestamp=datetime.now(),
                    context={"vulnerability": vulnerability}
                ))
        
        return suggestions
    
    def get_positioning_heatmap(self, game_state: GameState) -> Dict[str, float]:
        """Generate positioning heatmap showing optimal positions"""
        heatmap = {}
        
        # Initialize all positions with base score
        for row in range(self.board_height):
            for col in range(self.board_width):
                pos_key = f"{col},{row}"
                heatmap[pos_key] = 0.5  # Base score
        
        # Adjust scores based on positioning principles
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        
        # Higher scores for backline positions (for carries)
        for row in self.backline_rows:
            for col in range(self.board_width):
                pos_key = f"{col},{row}"
                heatmap[pos_key] += 0.3
        
        # Higher scores for corner positions (for carries)
        for pos in self.corner_positions:
            pos_key = f"{pos[0]},{pos[1]}"
            heatmap[pos_key] += 0.2
        
        # Higher scores for frontline positions (for tanks)
        for row in self.frontline_rows:
            for col in range(1, self.board_width - 1):  # Avoid corners for tanks
                pos_key = f"{col},{row}"
                heatmap[pos_key] += 0.2 if len(tanks) > 0 else 0.0
        
        # Adjust based on existing units
        for champion in game_state.board.champions:
            if champion.position:
                pos_key = f"{champion.position.x},{champion.position.y}"
                # Reduce score for occupied positions
                heatmap[pos_key] = 0.0
                
                # Increase scores for adjacent positions (synergy bonus)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        adj_x = champion.position.x + dx
                        adj_y = champion.position.y + dy
                        
                        if (0 <= adj_x < self.board_width and 
                            0 <= adj_y < self.board_height):
                            adj_key = f"{adj_x},{adj_y}"
                            if heatmap.get(adj_key, 0) > 0:  # Not occupied
                                heatmap[adj_key] += 0.1
        
        # Normalize scores to 0-1 range
        max_score = max(heatmap.values()) if heatmap.values() else 1.0
        for pos_key in heatmap:
            heatmap[pos_key] = min(1.0, heatmap[pos_key] / max_score)
        
        return heatmap
    
    def validate_positioning(self, game_state: GameState) -> Dict[str, Any]:
        """Validate current positioning and return detailed analysis"""
        validation = {
            "overall_score": 0.0,
            "issues": [],
            "strengths": [],
            "recommendations": []
        }
        
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        
        # Check carry positioning
        carry_scores = []
        for carry in carries:
            if carry.position:
                protection = self._calculate_protection_level(carry, game_state)
                carry_scores.append(protection)
                
                if protection < 0.3:
                    validation["issues"].append(f"{carry.name} poorly protected")
                elif protection > 0.7:
                    validation["strengths"].append(f"{carry.name} well protected")
        
        # Check tank positioning
        tank_scores = []
        for tank in tanks:
            if tank.position and tank.position.y in self.frontline_rows:
                tank_scores.append(1.0)
                validation["strengths"].append(f"{tank.name} properly positioned in frontline")
            elif tank.position:
                tank_scores.append(0.3)
                validation["issues"].append(f"{tank.name} should be in frontline")
        
        # Check formation balance
        frontline_count = sum(1 for c in game_state.board.champions 
                            if c.position and c.position.y in self.frontline_rows)
        backline_count = sum(1 for c in game_state.board.champions 
                           if c.position and c.position.y in self.backline_rows)
        
        if frontline_count < 2 and len(game_state.board.champions) >= 5:
            validation["issues"].append("Insufficient frontline presence")
            validation["recommendations"].append("Add more frontline units")
        
        if backline_count == 0 and len(carries) > 0:
            validation["issues"].append("No backline protection")
            validation["recommendations"].append("Move carries to backline")
        
        # Calculate overall score
        all_scores = carry_scores + tank_scores
        if all_scores:
            validation["overall_score"] = sum(all_scores) / len(all_scores)
        
        # Add formation score
        formation_score = self._calculate_formation_score(game_state)
        validation["overall_score"] = (validation["overall_score"] + formation_score) / 2
        
        return validation
    
    def _calculate_formation_score(self, game_state: GameState) -> float:
        """Calculate overall formation quality score"""
        if not game_state.board.champions:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        # Score based on positioning principles
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        
        # Carry protection score (40% weight)
        if carries:
            carry_protection = sum(self._calculate_protection_level(c, game_state) 
                                 for c in carries if c.position) / len(carries)
            score += carry_protection * 0.4
            total_weight += 0.4
        
        # Tank positioning score (30% weight)
        if tanks:
            tank_positioning = sum(1.0 if t.position and t.position.y in self.frontline_rows else 0.3 
                                 for t in tanks if t.position) / len(tanks)
            score += tank_positioning * 0.3
            total_weight += 0.3
        
        # Formation balance score (20% weight)
        formation_balance = self._calculate_formation_balance(game_state)
        score += formation_balance * 0.2
        total_weight += 0.2
        
        # Spread/clump appropriateness (10% weight)
        spread_score = self._calculate_spread_appropriateness(game_state)
        score += spread_score * 0.1
        total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_formation_balance(self, game_state: GameState) -> float:
        """Calculate how balanced the formation is"""
        champions = [c for c in game_state.board.champions if c.position]
        if not champions:
            return 0.0
        
        frontline_count = sum(1 for c in champions if c.position.y in self.frontline_rows)
        backline_count = sum(1 for c in champions if c.position.y in self.backline_rows)
        
        total_units = len(champions)
        
        # Ideal ratio is roughly 40% frontline, 60% backline
        ideal_frontline_ratio = 0.4
        actual_frontline_ratio = frontline_count / total_units if total_units > 0 else 0
        
        # Score based on how close to ideal ratio
        ratio_diff = abs(ideal_frontline_ratio - actual_frontline_ratio)
        balance_score = max(0.0, 1.0 - ratio_diff * 2)  # Penalty for deviation
        
        return balance_score
    
    def _calculate_spread_appropriateness(self, game_state: GameState) -> float:
        """Calculate if current spread level is appropriate"""
        current_spread = self._calculate_spread_factor(game_state)
        
        # TODO: This would ideally consider enemy composition
        # For now, assume moderate spread (0.4-0.6) is generally good
        ideal_spread = 0.5
        spread_diff = abs(current_spread - ideal_spread)
        
        return max(0.0, 1.0 - spread_diff)
    
    def get_advanced_positioning_tips(self, game_state: GameState) -> List[str]:
        """Get advanced positioning tips based on current state"""
        tips = []
        
        # Check for common positioning mistakes
        carries = self._identify_carries(game_state)
        
        # Tip: Corner positioning for ranged carries
        for carry in carries:
            if carry.position and self._is_ranged_carry(carry):
                if not self._is_in_corner(carry.position):
                    tips.append(f"Consider corner positioning for {carry.name} (ranged carry)")
        
        # Tip: Tank positioning
        tanks = self._identify_tanks(game_state)
        if len(tanks) == 1:
            tips.append("Single tank setup - consider positioning in center front")
        elif len(tanks) >= 2:
            tips.append("Multiple tanks - spread them across frontline for better coverage")
        
        # Tip: Formation spacing
        spread_factor = self._calculate_spread_factor(game_state)
        if spread_factor < 0.3:
            tips.append("Very clumped formation - vulnerable to AOE damage")
        elif spread_factor > 0.8:
            tips.append("Very spread formation - may lack focus fire potential")
        
        # Tip: Late game positioning
        if game_state.is_late_game():
            tips.append("Late game: Prioritize carry protection over economy positioning")
        
        return tips row in self.backline_rows:
                    analysis["backline_count"] += 1
                
                # Check for mispositioned units
                if champion in carries and row in self.frontline_rows:
                    analysis["carries_exposed"].append(champion.name)
                elif champion in tanks and row in self.backline_rows:
                    analysis["tanks_mispositioned"].append(champion.name)
        
        # Analyze formation
        analysis["formation"] = self._identify_formation(game_state)
        
        # Calculate spread factor
        analysis["spread_factor"] = self._calculate_spread_factor(game_state)
        
        # Calculate vulnerability
        analysis["vulnerability_score"] = self._calculate_vulnerability(game_state, carries)
        
        return analysis
    
    def _suggest_frontline_positioning(self, game_state: GameState, analysis: Dict[str, Any]) -> List[CoachingSuggestion]:
        """Suggest frontline positioning improvements"""
        suggestions = []
        
        # Check if frontline is too weak
        if analysis["frontline_count"] < 2 and len(game_state.board.champions) >= 5:
            suggestions.append(CoachingSuggestion(
                type="positioning",
                message="Add more units to frontline for protection",
                priority=7,
                timestamp=datetime.now(),
                context={
                    "current_frontline": analysis["frontline_count"],
                    "recommended_min": 2
                }
            ))
        
        # Check for tanks in wrong position
        if analysis["tanks_mispositioned"]:
            for tank in analysis["tanks_mispositioned"]:
                suggestions.append(CoachingSuggestion(
                    type="positioning",
                    message=f"Move {tank} to frontline (tank should be in front)",
                    priority=8,
                    timestamp=datetime.now(),
                    context={
                        "champion": tank,
                        "role": "tank",
                        "issue": "wrong_position"
                    }
                ))
        
        return suggestions
    
    def _suggest_backline_positioning(self, game_state: GameState, analysis: Dict[str, Any]) -> List[CoachingSuggestion]:
        """Suggest backline positioning improvements"""
        suggestions = []
        
        # Check for exposed carries
        if analysis["carries_exposed"]:
            for carry in analysis["carries_exposed"]:
                suggestions.append(CoachingSuggestion(
                    type="positioning",
                    message=f"Move {carry} to backline for safety",
                    priority=9,
                    timestamp=datetime.now(),
                    context={
                        "champion": carry,
                        "role": "carry",
                        "issue": "exposed"
                    }
                ))
        
        # Suggest backline density
        if analysis["backline_count"] > analysis["frontline_count"] + 2:
            suggestions.append(CoachingSuggestion(
                type="positioning",
                message="Too many units in backline - move some forward",
                priority=6,
                timestamp=datetime.now(),
                context={
                    "backline_count": analysis["backline_count"],
                    "frontline_count": analysis["frontline_count"]
                }
            ))
        
        return suggestions
    
    def _suggest_carry_protection(self, game_state: GameState, analysis: Dict[str, Any]) -> List[CoachingSuggestion]:
        """Suggest carry protection strategies"""
        suggestions = []
        
        carries = self._identify_carries(game_state)
        
        for carry in carries:
            if carry.position:
                protection_level = self._calculate_protection_level(carry, game_state)
                
                if protection_level < 0.5:  # Poorly protected
                    protection_strategies = self._get_protection_strategies(carry, game_state)
                    
                    if protection_strategies:
                        suggestions.append(CoachingSuggestion(
                            type="positioning",
                            message=f"Better protect {carry.name}: {protection_strategies[0]}",
                            priority=8,
                            timestamp=datetime.now(),
                            context={
                                "champion": carry.name,
                                "protection_level": protection_level,
                                "strategies": protection_strategies
                            }
                        ))
        
        return suggestions
    
    def _suggest_spread_vs_clump(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest whether to spread or clump units"""
        suggestions = []
        
        # Analyze enemy threats
        enemy_threats = self._analyze_enemy_threats(game_state)
        spread_factor = self._calculate_spread_factor(game_state)
        
        # Against AOE threats, suggest spreading
        if enemy_threats.get("aoe_damage", 0) >= 3 and spread_factor < 0.4:
            suggestions.append(CoachingSuggestion(
                type="positioning",
                message="Spread units to avoid AOE damage",
                priority=7,
                timestamp=datetime.now(),
                context={
                    "threat": "aoe_damage",
                    "current_spread": spread_factor,
                    "recommended": "spread"
                }
            ))
        
        # Against single-target threats, suggest clumping for protection
        elif enemy_threats.get("single_target", 0) >= 3 and spread_factor > 0.7:
            suggestions.append(CoachingSuggestion(
                type="positioning",
                message="Group units together for mutual protection",
                priority=6,
                timestamp=datetime.now(),
                context={
                    "threat": "single_target",
                    "current_spread": spread_factor,
                    "recommended": "clump"
                }
            ))
        
        return suggestions
    
    def _suggest_corner_positioning(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest corner positioning strategies"""
        suggestions = []
        
        carries = self._identify_carries(game_state)
        
        # Check if any carries could benefit from corner positioning
        for carry in carries:
            if carry.position and not self._is_in_corner(carry.position):
                # Suggest corner positioning for range carries
                if self._is_ranged_carry(carry):
                    suggestions.append(CoachingSuggestion(
                        type="positioning",
                        message=f"Consider corner positioning for {carry.name} (ranged carry)",
                        priority=6,
                        timestamp=datetime.now(),
                        context={
                            "champion": carry.name,
                            "strategy": "corner_positioning",
                            "reason": "ranged_carry_safety"
                        }
                    ))
        
        return suggestions
    
    def _suggest_counter_positioning(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest positioning to counter enemy compositions"""
        suggestions = []
        
        # Analyze common enemy patterns
        enemy_patterns = self._analyze_enemy_positioning_patterns(game_state)
        
        for pattern, frequency in enemy_patterns.items():
            if frequency >= 2:  # Common pattern
                counter_strategy = self._get_counter_positioning(pattern)
                
                if counter_strategy:
                    suggestions.append(CoachingSuggestion(
                        type="positioning",
                        message=f"Counter enemy {pattern}: {counter_strategy}",
                        priority=7,
                        timestamp=datetime.now(),
                        context={
                            "enemy_pattern": pattern,
                            "frequency": frequency,
                            "counter_strategy": counter_strategy
                        }
                    ))
        
        return suggestions
    
    # Helper methods
    
    def _identify_carries(self, game_state: GameState) -> List[Champion]:
        """Identify carry champions"""
        carries = []
        
        for champion in game_state.board.champions:
            # High cost champions are typically carries
            if champion.tier.value >= 4:
                carries.append(champion)
            # Or champions with carry items
            elif any(self._is_carry_item(item) for item in champion.items):
                carries.append(champion)
        
        return carries
    
    def _identify_tanks(self, game_state: GameState) -> List[Champion]:
        """Identify tank champions"""
        tanks = []
        
        for champion in game_state.board.champions:
            # Champions with tank items or traditionally tanky
            if any(self._is_tank_item(item) for item in champion.items):
                tanks.append(champion)
            elif self._is_traditional_tank(champion):
                tanks.append(champion)
            # Low-cost champions in frontline positions often serve as tanks
            elif (champion.tier.value <= 2 and 
                  champion.position and 
                  champion.position.y in self.frontline_rows):
                tanks.append(champion)
        
        return tanks