"""
TactiBird Overlay - Composition Coach
"""

import logging
from typing import List, Dict, Any, Set
from datetime import datetime
from collections import Counter

from src.ai.coaches.base_coach import BaseCoach, CoachingSuggestion
from src.data.models.game_state import GameState, Champion, ChampionTier

logger = logging.getLogger(__name__)

class CompositionCoach(BaseCoach):
    """Provides team composition coaching"""
    
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.name = "Composition Coach"
        
        # Composition analysis thresholds
        self.min_trait_activation = 2
        self.optimal_board_size = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9
        }
        
    def get_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Generate composition-focused suggestions"""
        suggestions = []
        
        try:
            # Analyze current composition
            comp_analysis = self._analyze_current_composition(game_state)
            
            # Generate suggestions based on analysis
            suggestions.extend(self._suggest_trait_completions(game_state, comp_analysis))
            suggestions.extend(self._suggest_champion_upgrades(game_state))
            suggestions.extend(self._suggest_meta_compositions(game_state))
            suggestions.extend(self._suggest_pivot_opportunities(game_state))
            suggestions.extend(self._suggest_board_optimization(game_state))
            
        except Exception as e:
            logger.error(f"Error generating composition suggestions: {e}")
        
        return suggestions
    
    def _analyze_current_composition(self, game_state: GameState) -> Dict[str, Any]:
        """Analyze the current team composition"""
        board_champions = game_state.board.champions
        active_traits = game_state.board.active_traits
        
        analysis = {
            "champion_count": len(board_champions),
            "trait_counts": {},
            "tier_distribution": {},
            "carry_potential": [],
            "synergy_strength": 0.0,
            "comp_identity": None
        }
        
        # Count traits from champions
        for champion in board_champions:
            champion_traits = self.data_manager.get_champion_traits(champion.name)
            for trait in champion_traits:
                analysis["trait_counts"][trait] = analysis["trait_counts"].get(trait, 0) + 1
            
            # Count tier distribution
            tier = champion.tier.value
            analysis["tier_distribution"][tier] = analysis["tier_distribution"].get(tier, 0) + 1
            
            # Identify potential carries (4+ cost champions)
            if tier >= 4:
                analysis["carry_potential"].append(champion.name)
        
        # Calculate synergy strength
        analysis["synergy_strength"] = self._calculate_synergy_strength(analysis["trait_counts"])
        
        # Try to identify composition
        analysis["comp_identity"] = self._identify_composition(board_champions)
        
        return analysis
    
    def _calculate_synergy_strength(self, trait_counts: Dict[str, int]) -> float:
        """Calculate overall synergy strength of current traits"""
        total_strength = 0.0
        
        for trait_name, count in trait_counts.items():
            breakpoints = self.data_manager.get_trait_breakpoints(trait_name)
            
            # Find the highest activated breakpoint
            activated_level = 0
            for breakpoint in sorted(breakpoints):
                if count >= breakpoint:
                    activated_level += 1
                else:
                    break
            
            # Weight by activation level and trait value
            trait_strength = activated_level * (count / max(breakpoints)) if breakpoints else 0
            total_strength += trait_strength
        
        return total_strength
    
    def _identify_composition(self, champions: List[Champion]) -> Dict[str, Any]:
        """Try to identify what composition the player is going for"""
        champion_names = [champ.name for champ in champions]
        
        # Check against known meta compositions
        meta_comps = self.data_manager.get_meta_compositions()
        
        best_match = None
        best_score = 0.0
        
        for comp in meta_comps:
            core_champions = comp.get("core_champions", [])
            overlap = len(set(champion_names).intersection(set(core_champions)))
            score = overlap / len(core_champions) if core_champions else 0
            
            if score > best_score and score >= 0.3:  # At least 30% overlap
                best_score = score
                best_match = {
                    "name": comp["name"],
                    "score": score,
                    "missing_champions": list(set(core_champions) - set(champion_names)),
                    "tier": comp.get("tier", "Unknown"),
                    "carry": comp.get("carry", ""),
                    "key_traits": comp.get("key_traits", [])
                }
        
        return best_match
    
    def _suggest_trait_completions(self, game_state: GameState, analysis: Dict[str, Any]) -> List[CoachingSuggestion]:
        """Suggest trait completions that are close to activation"""
        suggestions = []
        trait_counts = analysis["trait_counts"]
        
        for trait_name, current_count in trait_counts.items():
            breakpoints = self.data_manager.get_trait_breakpoints(trait_name)
            
            for breakpoint in breakpoints:
                needed = breakpoint - current_count
                
                # Suggest if 1-2 champions away from activation
                if 1 <= needed <= 2:
                    priority = 8 if needed == 1 else 6
                    
                    suggestions.append(CoachingSuggestion(
                        type="composition",
                        message=f"Add {needed} more {trait_name} champion(s) to activate {trait_name} ({breakpoint})",
                        priority=priority,
                        timestamp=datetime.now(),
                        context={
                            "trait": trait_name,
                            "current_count": current_count,
                            "target_count": breakpoint,
                            "needed": needed
                        }
                    ))
                    break  # Only suggest the closest breakpoint
        
        return suggestions
    
    def _suggest_champion_upgrades(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest champion upgrade opportunities"""
        suggestions = []
        
        # Count champions on board and bench
        all_champions = game_state.board.champions + game_state.player.bench_champions
        champion_counts = Counter(champ.name for champ in all_champions)
        
        for champion_name, count in champion_counts.items():
            # Find champions that can be upgraded
            board_champion = next((c for c in game_state.board.champions if c.name == champion_name), None)
            
            if board_champion:
                current_level = board_champion.level
                
                # Suggest 2-star upgrade (need 3 copies)
                if current_level == 1 and count >= 3:
                    suggestions.append(CoachingSuggestion(
                        type="composition",
                        message=f"Upgrade {champion_name} to 2-star (have {count}/3)",
                        priority=7,
                        timestamp=datetime.now(),
                        context={
                            "champion": champion_name,
                            "current_level": current_level,
                            "target_level": 2,
                            "copies_available": count
                        }
                    ))
                
                # Suggest 3-star upgrade (need 9 copies total)
                elif current_level == 2 and count >= 9:
                    suggestions.append(CoachingSuggestion(
                        type="composition",
                        message=f"Consider 3-starring {champion_name} (have {count}/9)",
                        priority=5,
                        timestamp=datetime.now(),
                        context={
                            "champion": champion_name,
                            "current_level": current_level,
                            "target_level": 3,
                            "copies_available": count
                        }
                    ))
        
        return suggestions
    
    def _suggest_meta_compositions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest transitioning to meta compositions"""
        suggestions = []
        
        if game_state.is_early_game():
            return suggestions  # Don't suggest meta comps too early
        
        current_champions = [champ.name for champ in game_state.board.champions]
        meta_comps = self.data_manager.get_meta_compositions()
        
        for comp in meta_comps:
            core_champions = comp.get("core_champions", [])
            overlap = len(set(current_champions).intersection(set(core_champions)))
            
            # Suggest if player has some champions from a strong comp
            if overlap >= 2 and overlap < len(core_champions):
                missing = list(set(core_champions) - set(current_champions))
                
                suggestions.append(CoachingSuggestion(
                    type="composition",
                    message=f"Consider transitioning to {comp['name']} (Tier {comp.get('tier', '?')}) - need: {', '.join(missing[:3])}",
                    priority=6,
                    timestamp=datetime.now(),
                    context={
                        "composition": comp["name"],
                        "tier": comp.get("tier", "Unknown"),
                        "missing_champions": missing,
                        "current_overlap": overlap,
                        "total_needed": len(core_champions)
                    }
                ))
        
        return suggestions
    
    def _suggest_pivot_opportunities(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest when to pivot compositions"""
        suggestions = []
        
        # Only suggest pivots in mid-late game
        if not (game_state.is_mid_game() or game_state.is_late_game()):
            return suggestions
        
        current_comp = self._analyze_current_composition(game_state)
        
        # Suggest pivot if synergy strength is low
        if current_comp["synergy_strength"] < 2.0 and game_state.player.gold >= 30:
            suggestions.append(CoachingSuggestion(
                type="composition",
                message="Weak synergies detected - consider pivoting to a stronger composition",
                priority=7,
                timestamp=datetime.now(),
                context={
                    "synergy_strength": current_comp["synergy_strength"],
                    "reason": "weak_synergies"
                }
            ))
        
        # Suggest pivot if health is low and comp isn't working
        if game_state.player.health <= 30 and current_comp["synergy_strength"] < 3.0:
            suggestions.append(CoachingSuggestion(
                type="composition",
                message="Low health - force a strong composition to stabilize",
                priority=9,
                timestamp=datetime.now(),
                context={
                    "health": game_state.player.health,
                    "synergy_strength": current_comp["synergy_strength"],
                    "reason": "low_health_stabilize"
                }
            ))
        
        return suggestions
    
    def _suggest_board_optimization(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest board size and positioning optimizations"""
        suggestions = []
        
        player_level = game_state.player.level
        current_board_size = len(game_state.board.champions)
        optimal_size = self.optimal_board_size.get(player_level, player_level)
        
        # Suggest playing fewer units if over-fielding
        if current_board_size > optimal_size:
            suggestions.append(CoachingSuggestion(
                type="composition",
                message=f"Consider benching weaker units (playing {current_board_size}/{optimal_size})",
                priority=5,
                timestamp=datetime.now(),
                context={
                    "current_size": current_board_size,
                    "optimal_size": optimal_size,
                    "player_level": player_level
                }
            ))
        
        # Suggest filling board if under-utilizing
        elif current_board_size < optimal_size and game_state.player.bench_champions:
            suggestions.append(CoachingSuggestion(
                type="composition",
                message=f"Add more units to the board ({current_board_size}/{optimal_size})",
                priority=6,
                timestamp=datetime.now(),
                context={
                    "current_size": current_board_size,
                    "optimal_size": optimal_size,
                    "player_level": player_level,
                    "bench_available": len(game_state.player.bench_champions)
                }
            ))
        
        return suggestions
    
    def analyze_shop_for_composition(self, game_state: GameState) -> Dict[str, Any]:
        """Analyze shop champions for composition building"""
        if not game_state.shop or not game_state.shop.available_champions:
            return {"recommendations": [], "priority_buys": []}
        
        current_comp = self._analyze_current_composition(game_state)
        shop_champions = game_state.shop.available_champions
        
        analysis = {
            "recommendations": [],
            "priority_buys": [],
            "trait_completions": [],
            "upgrades_available": []
        }
        
        # Check for trait completions in shop
        for champion in shop_champions:
            champion_traits = self.data_manager.get_champion_traits(champion.name)
            
            for trait in champion_traits:
                current_count = current_comp["trait_counts"].get(trait, 0)
                breakpoints = self.data_manager.get_trait_breakpoints(trait)
                
                for breakpoint in breakpoints:
                    if current_count == breakpoint - 1:  # One away from activation
                        analysis["trait_completions"].append({
                            "champion": champion.name,
                            "trait": trait,
                            "current_count": current_count,
                            "target_count": breakpoint
                        })
        
        # Check for upgrade opportunities
        board_champions = {champ.name: champ for champ in game_state.board.champions}
        
        for champion in shop_champions:
            if champion.name in board_champions:
                board_champ = board_champions[champion.name]
                if board_champ.level < 3:  # Can still be upgraded
                    analysis["upgrades_available"].append({
                        "champion": champion.name,
                        "current_level": board_champ.level,
                        "tier": champion.tier.value
                    })
        
        return analysis