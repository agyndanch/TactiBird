"""
TactiBird Overlay - Item Coach
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from collections import Counter

from src.ai.coaches.base_coach import BaseCoach, CoachingSuggestion
from src.data.models.game_state import GameState, Champion, Item

logger = logging.getLogger(__name__)

class ItemCoach(BaseCoach):
    """Provides item building and optimization coaching"""
    
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.name = "Item Coach"
        
        # Item priority categories
        self.carry_items = ["Infinity Edge", "Last Whisper", "Bloodthirster", "Archangel's Staff", "Rabadon's Deathcap"]
        self.tank_items = ["Bramble Vest", "Dragon's Claw", "Gargoyle Stoneplate", "Warmog's Armor"]
        self.utility_items = ["Zephyr", "Shroud of Stillness", "Chalice of Power", "Locket of the Iron Solari"]
        
    def get_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Generate item-focused suggestions"""
        suggestions = []
        
        try:
            # Analyze current items
            item_analysis = self._analyze_current_items(game_state)
            
            # Generate suggestions
            suggestions.extend(self._suggest_item_combinations(game_state, item_analysis))
            suggestions.extend(self._suggest_item_positioning(game_state))
            suggestions.extend(self._suggest_item_priorities(game_state))
            suggestions.extend(self._suggest_component_usage(game_state))
            suggestions.extend(self._suggest_item_pivots(game_state))
            
        except Exception as e:
            logger.error(f"Error generating item suggestions: {e}")
        
        return suggestions
    
    def _analyze_current_items(self, game_state: GameState) -> Dict[str, Any]:
        """Analyze current item distribution and efficiency"""
        analysis = {
            "completed_items": [],
            "component_items": [],
            "item_distribution": {},
            "carries_itemized": [],
            "tanks_itemized": [],
            "utility_coverage": 0.0
        }
        
        # Analyze items on board champions
        for champion in game_state.board.champions:
            champion_items = champion.items
            analysis["item_distribution"][champion.name] = len(champion_items)
            
            for item in champion_items:
                if self._is_completed_item(item):
                    analysis["completed_items"].append(item)
                    
                    # Categorize by function
                    if item in self.carry_items:
                        analysis["carries_itemized"].append(champion.name)
                    elif item in self.tank_items:
                        analysis["tanks_itemized"].append(champion.name)
                    elif item in self.utility_items:
                        analysis["utility_coverage"] += 1
                else:
                    analysis["component_items"].append(item)
        
        # Analyze bench items
        bench_items = game_state.player.items_on_bench
        for item in bench_items:
            if self._is_completed_item(item.name):
                analysis["completed_items"].append(item.name)
            else:
                analysis["component_items"].append(item.name)
        
        return analysis
    
    def _suggest_item_combinations(self, game_state: GameState, analysis: Dict[str, Any]) -> List[CoachingSuggestion]:
        """Suggest item combinations from available components"""
        suggestions = []
        
        # Count available components
        component_counts = Counter(analysis["component_items"])
        
        # Find possible item combinations
        possible_items = self._find_craftable_items(component_counts)
        
        for item_name, components, priority in possible_items:
            # Check if this item would be good for current composition
            relevance = self._calculate_item_relevance(item_name, game_state)
            
            if relevance > 0.5:  # Only suggest relevant items
                adjusted_priority = min(10, priority + int(relevance * 3))
                
                suggestions.append(CoachingSuggestion(
                    type="items",
                    message=f"Craft {item_name} from {' + '.join(components)}",
                    priority=adjusted_priority,
                    timestamp=datetime.now(),
                    context={
                        "item": item_name,
                        "components": components,
                        "relevance": relevance,
                        "category": self._get_item_category(item_name)
                    }
                ))
        
        return suggestions
    
    def _suggest_item_positioning(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest optimal item positioning on champions"""
        suggestions = []
        
        # Find carries and tanks
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        
        for champion in game_state.board.champions:
            champion_role = self._determine_champion_role(champion, carries, tanks)
            optimal_items = self._get_optimal_items_for_role(champion_role, champion)
            current_items = set(champion.items)
            
            # Check for suboptimal itemization
            if champion_role == "carry" and not any(item in self.carry_items for item in current_items):
                if len(champion.items) < 3:
                    suggestions.append(CoachingSuggestion(
                        type="items",
                        message=f"Itemize {champion.name} (carry) with damage items",
                        priority=8,
                        timestamp=datetime.now(),
                        context={
                            "champion": champion.name,
                            "role": champion_role,
                            "current_items": list(current_items),
                            "recommended": optimal_items[:3]
                        }
                    ))
            
            elif champion_role == "tank" and not any(item in self.tank_items for item in current_items):
                if len(champion.items) < 2:
                    suggestions.append(CoachingSuggestion(
                        type="items",
                        message=f"Give {champion.name} (tank) defensive items",
                        priority=7,
                        timestamp=datetime.now(),
                        context={
                            "champion": champion.name,
                            "role": champion_role,
                            "current_items": list(current_items),
                            "recommended": optimal_items[:2]
                        }
                    ))
        
        return suggestions
    
    def _suggest_item_priorities(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest item priority based on game stage and composition"""
        suggestions = []
        
        # Early game priorities
        if game_state.is_early_game():
            suggestions.append(CoachingSuggestion(
                type="items",
                message="Prioritize basic damage items (BF Sword, Rod) in early game",
                priority=6,
                timestamp=datetime.now(),
                context={
                    "stage": "early_game",
                    "priority_components": ["B.F. Sword", "Needlessly Large Rod"]
                }
            ))
        
        # Mid game priorities
        elif game_state.is_mid_game():
            carry_count = len(self._identify_carries(game_state))
            
            if carry_count == 0:
                suggestions.append(CoachingSuggestion(
                    type="items",
                    message="Find a carry champion and itemize them heavily",
                    priority=9,
                    timestamp=datetime.now(),
                    context={
                        "stage": "mid_game",
                        "issue": "no_carry"
                    }
                ))
            
            # Check for frontline
            tank_count = len(self._identify_tanks(game_state))
            if tank_count == 0:
                suggestions.append(CoachingSuggestion(
                    type="items",
                    message="Build tank items for frontline protection",
                    priority=7,
                    timestamp=datetime.now(),
                    context={
                        "stage": "mid_game",
                        "issue": "no_tanks"
                    }
                ))
        
        # Late game priorities
        elif game_state.is_late_game():
            # Focus on optimizing existing carries
            suggestions.append(CoachingSuggestion(
                type="items",
                message="Optimize carry items and add utility/counter items",
                priority=8,
                timestamp=datetime.now(),
                context={
                    "stage": "late_game",
                    "focus": "optimization"
                }
            ))
        
        return suggestions
    
    def _suggest_component_usage(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest how to use individual components"""
        suggestions = []
        
        # Count components on bench
        bench_components = [item.name for item in game_state.player.items_on_bench 
                          if not self._is_completed_item(item.name)]
        
        component_counts = Counter(bench_components)
        
        # Suggest using excess components
        for component, count in component_counts.items():
            if count >= 3:  # Too many of same component
                suggestions.append(CoachingSuggestion(
                    type="items",
                    message=f"Use excess {component} components ({count} available)",
                    priority=5,
                    timestamp=datetime.now(),
                    context={
                        "component": component,
                        "count": count,
                        "possible_items": self._get_items_using_component(component)
                    }
                ))
        
        # Suggest completing items from pairs
        for component, count in component_counts.items():
            if count == 2:
                same_component_items = self._get_items_from_duplicate_components(component)
                if same_component_items:
                    suggestions.append(CoachingSuggestion(
                        type="items",
                        message=f"Consider making {same_component_items[0]} from two {component}",
                        priority=6,
                        timestamp=datetime.now(),
                        context={
                            "component": component,
                            "count": count,
                            "item": same_component_items[0]
                        }
                    ))
        
        return suggestions
    
    def _suggest_item_pivots(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Suggest item build pivots based on game state"""
        suggestions = []
        
        # Suggest counter items based on enemy compositions
        enemy_threats = self._analyze_enemy_threats(game_state)
        
        for threat_type, threat_level in enemy_threats.items():
            if threat_level >= 3:  # Significant threat
                counter_items = self._get_counter_items(threat_type)
                
                if counter_items:
                    suggestions.append(CoachingSuggestion(
                        type="items",
                        message=f"Build {counter_items[0]} to counter {threat_type} comps",
                        priority=7,
                        timestamp=datetime.now(),
                        context={
                            "threat_type": threat_type,
                            "threat_level": threat_level,
                            "counter_items": counter_items
                        }
                    ))
        
        return suggestions
    
    # Helper methods
    
    def _is_completed_item(self, item_name: str) -> bool:
        """Check if item is a completed item (not a component)"""
        # Basic components
        components = [
            "B.F. Sword", "Recurve Bow", "Needlessly Large Rod", "Tear of the Goddess",
            "Chain Vest", "Negatron Cloak", "Giant's Belt", "Spatula", "Glove"
        ]
        return item_name not in components
    
    def _find_craftable_items(self, component_counts: Counter) -> List[Tuple[str, List[str], int]]:
        """Find items that can be crafted from available components"""
        craftable = []
        
        # Item recipes (simplified)
        recipes = {
            "Infinity Edge": (["B.F. Sword", "Glove"], 8),
            "Bloodthirster": (["B.F. Sword", "Negatron Cloak"], 7),
            "Last Whisper": (["Recurve Bow", "Glove"], 8),
            "Rabadon's Deathcap": (["Needlessly Large Rod", "Needlessly Large Rod"], 8),
            "Archangel's Staff": (["Tear of the Goddess", "Needlessly Large Rod"], 7),
            "Bramble Vest": (["Chain Vest", "Chain Vest"], 6),
            "Dragon's Claw": (["Negatron Cloak", "Negatron Cloak"], 6),
            "Warmog's Armor": (["Giant's Belt", "Giant's Belt"], 5)
        }
        
        for item_name, (components, priority) in recipes.items():
            if all(component_counts[comp] >= components.count(comp) for comp in set(components)):
                craftable.append((item_name, components, priority))
        
        return sorted(craftable, key=lambda x: x[2], reverse=True)
    
    def _calculate_item_relevance(self, item_name: str, game_state: GameState) -> float:
        """Calculate how relevant an item is for current composition"""
        relevance = 0.5  # Base relevance
        
        # Check if item fits composition
        comp_analysis = self._analyze_composition_needs(game_state)
        
        if item_name in self.carry_items and comp_analysis.get("needs_carry_items", False):
            relevance += 0.3
        elif item_name in self.tank_items and comp_analysis.get("needs_tank_items", False):
            relevance += 0.3
        elif item_name in self.utility_items and comp_analysis.get("needs_utility", False):
            relevance += 0.2
        
        return min(1.0, relevance)
    
    def _get_item_category(self, item_name: str) -> str:
        """Get item category"""
        if item_name in self.carry_items:
            return "carry"
        elif item_name in self.tank_items:
            return "tank"
        elif item_name in self.utility_items:
            return "utility"
        return "other"
    
    def _identify_carries(self, game_state: GameState) -> List[Champion]:
        """Identify carry champions on board"""
        carries = []
        
        for champion in game_state.board.champions:
            # High cost champions are typically carries
            if champion.tier.value >= 4:
                carries.append(champion)
            # Or champions with carry items
            elif any(item in self.carry_items for item in champion.items):
                carries.append(champion)
        
        return carries
    
    def _identify_tanks(self, game_state: GameState) -> List[Champion]:
        """Identify tank champions on board"""
        tanks = []
        
        for champion in game_state.board.champions:
            # Champions with tank items
            if any(item in self.tank_items for item in champion.items):
                tanks.append(champion)
            # Or traditionally tanky champions (would need trait/role data)
            elif self._is_traditional_tank(champion):
                tanks.append(champion)
        
        return tanks
    
    def _determine_champion_role(self, champion: Champion, carries: List[Champion], tanks: List[Champion]) -> str:
        """Determine champion's role in composition"""
        if champion in carries:
            return "carry"
        elif champion in tanks:
            return "tank"
        elif champion.tier.value >= 4:
            return "carry"
        elif champion.tier.value <= 2:
            return "support"
        else:
            return "flex"
    
    def _get_optimal_items_for_role(self, role: str, champion: Champion) -> List[str]:
        """Get optimal items for champion role"""
        if role == "carry":
            # Get specific recommendations for this champion
            recommended = self.data_manager.get_item_recommendations(champion.name)
            return recommended if recommended else self.carry_items[:3]
        elif role == "tank":
            return self.tank_items[:2]
        elif role == "support":
            return self.utility_items[:2]
        else:
            return []
    
    def _analyze_composition_needs(self, game_state: GameState) -> Dict[str, bool]:
        """Analyze what types of items the composition needs"""
        carries = self._identify_carries(game_state)
        tanks = self._identify_tanks(game_state)
        
        return {
            "needs_carry_items": len(carries) > 0 and any(len(c.items) < 3 for c in carries),
            "needs_tank_items": len(tanks) == 0 or any(len(t.items) < 2 for t in tanks),
            "needs_utility": len(game_state.board.champions) >= 6
        }
    
    def _get_items_using_component(self, component: str) -> List[str]:
        """Get items that use a specific component"""
        # Simplified mapping
        component_items = {
            "B.F. Sword": ["Infinity Edge", "Bloodthirster", "Spear of Shojin"],
            "Needlessly Large Rod": ["Rabadon's Deathcap", "Archangel's Staff", "Morellonomicon"],
            "Chain Vest": ["Bramble Vest", "Gargoyle Stoneplate", "Sunfire Cape"],
            "Glove": ["Infinity Edge", "Last Whisper", "Hand of Justice"]
        }
        return component_items.get(component, [])
    
    def _get_items_from_duplicate_components(self, component: str) -> List[str]:
        """Get items made from two of the same component"""
        duplicate_items = {
            "Needlessly Large Rod": ["Rabadon's Deathcap"],
            "Chain Vest": ["Bramble Vest"],
            "Negatron Cloak": ["Dragon's Claw"],
            "Giant's Belt": ["Warmog's Armor"]
        }
        return duplicate_items.get(component, [])
    
    def _analyze_enemy_threats(self, game_state: GameState) -> Dict[str, int]:
        """Analyze enemy composition threats"""
        # This would analyze opponent data if available
        # For now, return placeholder
        return {
            "ap_burst": 2,
            "ad_carry": 3,
            "assassins": 1
        }
    
    def _get_counter_items(self, threat_type: str) -> List[str]:
        """Get items that counter specific threat types"""
        counters = {
            "ap_burst": ["Dragon's Claw", "Chalice of Power"],
            "ad_carry": ["Bramble Vest", "Frozen Heart"],
            "assassins": ["Zephyr", "Shroud of Stillness"]
        }
        return counters.get(threat_type, [])
    
    def _is_traditional_tank(self, champion: Champion) -> bool:
        """Check if champion is traditionally a tank"""
        # This would check champion traits/roles
        # Placeholder implementation
        tank_champions = ["Leona", "Braum", "Nautilus", "Thresh"]
        return champion.name in tank_champions