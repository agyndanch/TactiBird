"""
TactiBird Overlay - Game State Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

class GamePhase(Enum):
    """TFT game phases"""
    CAROUSEL = "carousel"
    PLANNING = "planning"
    COMBAT = "combat"
    UNKNOWN = "unknown"

class ChampionTier(Enum):
    """Champion tier/cost"""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

@dataclass
class Position:
    """Board position"""
    x: int
    y: int
    
    def to_hex(self) -> str:
        """Convert to hexagonal coordinate string"""
        return f"{self.x},{self.y}"

@dataclass
class Champion:
    """Champion data model"""
    name: str
    tier: ChampionTier
    level: int = 1
    items: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    position: Optional[Position] = None
    is_chosen: bool = False
    
    @property
    def cost(self) -> int:
        """Get champion cost based on tier"""
        return self.tier.value

@dataclass
class Item:
    """Item data model"""
    name: str
    components: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    is_completed: bool = True

@dataclass
class Trait:
    """Trait/synergy data model"""
    name: str
    current_count: int
    required_counts: List[int] = field(default_factory=list)
    is_active: bool = False
    active_level: int = 0

@dataclass
class BoardState:
    """Current board state"""
    champions: List[Champion] = field(default_factory=list)
    positioned_champions: Dict[str, Champion] = field(default_factory=dict)  # position_hex -> champion
    active_traits: List[Trait] = field(default_factory=list)
    
    def get_champion_at(self, position: Position) -> Optional[Champion]:
        """Get champion at specific position"""
        return self.positioned_champions.get(position.to_hex())
    
    def add_champion(self, champion: Champion):
        """Add champion to board"""
        self.champions.append(champion)
        if champion.position:
            self.positioned_champions[champion.position.to_hex()] = champion
    
    def remove_champion(self, champion: Champion):
        """Remove champion from board"""
        if champion in self.champions:
            self.champions.remove(champion)
        if champion.position:
            pos_key = champion.position.to_hex()
            if pos_key in self.positioned_champions:
                del self.positioned_champions[pos_key]

@dataclass
class ShopState:
    """Current shop state"""
    available_champions: List[Champion] = field(default_factory=list)
    reroll_cost: int = 2
    can_reroll: bool = True
    
    def get_champions_by_tier(self, tier: ChampionTier) -> List[Champion]:
        """Get champions of specific tier in shop"""
        return [champ for champ in self.available_champions if champ.tier == tier]

@dataclass
class PlayerState:
    """Player's current state"""
    level: int = 1
    experience: int = 0
    gold: int = 0
    health: int = 100
    win_streak: int = 0
    loss_streak: int = 0
    bench_champions: List[Champion] = field(default_factory=list)
    items_on_bench: List[Item] = field(default_factory=list)
    
    @property
    def experience_needed(self) -> int:
        """Experience needed for next level"""
        exp_requirements = {1: 0, 2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 80, 9: 100}
        return exp_requirements.get(self.level + 1, 100) - self.experience

@dataclass
class OpponentState:
    """Opponent information"""
    name: str = ""
    level: int = 1
    health: int = 100
    board_power: float = 0.0
    visible_champions: List[Champion] = field(default_factory=list)
    last_comp_seen: List[str] = field(default_factory=list)

@dataclass
class GameState:
    """Complete game state"""
    timestamp: datetime
    phase: GamePhase = GamePhase.UNKNOWN
    stage: int = 1
    round: int = 1
    time_remaining: float = 0.0
    
    # Player state
    player: PlayerState = field(default_factory=PlayerState)
    board: BoardState = field(default_factory=BoardState)
    shop: ShopState = field(default_factory=ShopState)
    
    # Opponents
    opponents: List[OpponentState] = field(default_factory=list)
    
    # Meta information
    confidence: float = 0.0  # How confident we are in this state reading
    errors: List[str] = field(default_factory=list)  # Any errors during state extraction
    
    @property
    def stage_round_str(self) -> str:
        """Get stage-round as string (e.g., "2-1")"""
        return f"{self.stage}-{self.round}"
    
    def is_early_game(self) -> bool:
        """Check if in early game (stages 1-2)"""
        return self.stage <= 2
    
    def is_mid_game(self) -> bool:
        """Check if in mid game (stages 3-4)"""
        return 3 <= self.stage <= 4
    
    def is_late_game(self) -> bool:
        """Check if in late game (stages 5+)"""
        return self.stage >= 5
    
    def is_carousel_round(self) -> bool:
        """Check if current round is a carousel"""
        return self.round in [1, 4] and self.stage > 1  # Simplified carousel detection
    
    def get_board_value(self) -> int:
        """Calculate total gold value of board"""
        total_value = 0
        for champion in self.board.champions:
            # Base cost + upgrade cost
            base_cost = champion.cost
            upgrade_cost = 0
            if champion.level == 2:
                upgrade_cost = base_cost * 2  # 3 copies needed
            elif champion.level == 3:
                upgrade_cost = base_cost * 8  # 9 copies needed
            total_value += base_cost + upgrade_cost
        return total_value
    
    def get_economy_strength(self) -> str:
        """Evaluate economy strength"""
        if self.player.gold >= 50:
            return "excellent"
        elif self.player.gold >= 30:
            return "strong"
        elif self.player.gold >= 20:
            return "decent"
        elif self.player.gold >= 10:
            return "weak"
        else:
            return "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'phase': self.phase.value,
            'stage': self.stage,
            'round': self.round,
            'time_remaining': self.time_remaining,
            'player': {
                'level': self.player.level,
                'gold': self.player.gold,
                'health': self.player.health,
                'experience': self.player.experience
            },
            'board': {
                'champion_count': len(self.board.champions),
                'active_traits': [trait.name for trait in self.board.active_traits if trait.is_active]
            },
            'shop': {
                'available_count': len(self.shop.available_champions)
            },
            'confidence': self.confidence
        }