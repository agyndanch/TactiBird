"""
TactiBird Overlay - Data Manager
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.data.models.game_state import Champion, ChampionTier, Item, Trait

logger = logging.getLogger(__name__)

class DataManager:
    """Manages game data, caching, and API interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_size = config.get('cache_size', 100)
        self.api_timeout = config.get('api_timeout', 30)
        self.use_local_cache = config.get('use_local_cache', True)
        self.auto_update = config.get('auto_update', True)
        
        # Data storage
        self.champions_data = {}
        self.items_data = {}
        self.traits_data = {}
        self.compositions_data = {}
        
        # Cache
        self.cache = {}
        self.last_update = None
        
        # Initialize data
        self._load_data()
        
        logger.info("Data manager initialized")
    
    def _load_data(self):
        """Load game data from files"""
        try:
            self._load_champions_data()
            self._load_items_data()
            self._load_traits_data()
            self._load_compositions_data()
            logger.info("Game data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load game data: {e}")
    
    def _load_champions_data(self):
        """Load champion data"""
        champions_file = Path("data/champions/champion_data.json")
        if champions_file.exists():
            with open(champions_file, 'r') as f:
                self.champions_data = json.load(f)
        else:
            self.champions_data = self._get_default_champions_data()
            self._save_champions_data()
    
    def _load_items_data(self):
        """Load item data"""
        items_file = Path("data/items/item_data.json")
        if items_file.exists():
            with open(items_file, 'r') as f:
                self.items_data = json.load(f)
        else:
            self.items_data = self._get_default_items_data()
            self._save_items_data()
    
    def _load_traits_data(self):
        """Load trait data"""
        traits_file = Path("data/traits/trait_data.json")
        if traits_file.exists():
            with open(traits_file, 'r') as f:
                self.traits_data = json.load(f)
        else:
            self.traits_data = self._get_default_traits_data()
            self._save_traits_data()
    
    def _load_compositions_data(self):
        """Load composition data"""
        comps_file = Path("data/compositions/meta_comps.json")
        if comps_file.exists():
            with open(comps_file, 'r') as f:
                self.compositions_data = json.load(f)
        else:
            self.compositions_data = self._get_default_compositions_data()
            self._save_compositions_data()
    
    def get_champion_data(self, champion_name: str) -> Optional[Dict[str, Any]]:
        """Get champion data by name"""
        return self.champions_data.get(champion_name.lower())
    
    def get_champion_tier(self, champion_name: str) -> ChampionTier:
        """Get champion tier"""
        data = self.get_champion_data(champion_name)
        if data:
            tier = data.get('tier', 1)
            return ChampionTier(tier)
        return ChampionTier.ONE
    
    def get_champion_traits(self, champion_name: str) -> List[str]:
        """Get champion traits"""
        data = self.get_champion_data(champion_name)
        return data.get('traits', []) if data else []
    
    def get_item_data(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Get item data by name"""
        return self.items_data.get(item_name.lower())
    
    def get_trait_data(self, trait_name: str) -> Optional[Dict[str, Any]]:
        """Get trait data by name"""
        return self.traits_data.get(trait_name.lower())
    
    def get_trait_breakpoints(self, trait_name: str) -> List[int]:
        """Get trait activation breakpoints"""
        data = self.get_trait_data(trait_name)
        return data.get('breakpoints', [3, 6, 9]) if data else [3, 6, 9]
    
    def get_meta_compositions(self) -> List[Dict[str, Any]]:
        """Get current meta compositions"""
        return self.compositions_data.get('meta_comps', [])
    
    def find_composition_by_champions(self, champions: List[str]) -> Optional[Dict[str, Any]]:
        """Find composition that matches given champions"""
        for comp in self.get_meta_compositions():
            comp_champions = set(champ.lower() for champ in comp.get('core_champions', []))
            player_champions = set(champ.lower() for champ in champions)
            
            # Check if player has at least 60% of core champions
            overlap = len(comp_champions.intersection(player_champions))
            if overlap >= len(comp_champions) * 0.6:
                return comp
        
        return None
    
    def get_item_recommendations(self, champion_name: str) -> List[str]:
        """Get recommended items for champion"""
        data = self.get_champion_data(champion_name)
        return data.get('recommended_items', []) if data else []
    
    def cache_get(self, key: str) -> Any:
        """Get value from cache"""
        return self.cache.get(key)
    
    def cache_set(self, key: str, value: Any):
        """Set value in cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def _get_default_champions_data(self) -> Dict[str, Any]:
        """Get default champion data"""
        return {
            "graves": {
                "tier": 1,
                "traits": ["Gunslinger", "Outlaw"],
                "recommended_items": ["Infinity Edge", "Last Whisper", "Bloodthirster"]
            },
            "nidalee": {
                "tier": 1,
                "traits": ["Yordle", "Shapeshifter"],
                "recommended_items": ["Rabadon's Deathcap", "Ionic Spark", "Morellonomicon"]
            },
            "tristana": {
                "tier": 2,
                "traits": ["Yordle", "Gunslinger"],
                "recommended_items": ["Rapid Firecannon", "Infinity Edge", "Phantom Dancer"]
            },
            "lux": {
                "tier": 3,
                "traits": ["Academy", "Arcanist"],
                "recommended_items": ["Archangel's Staff", "Rabadon's Deathcap", "Blue Buff"]
            },
            "jinx": {
                "tier": 4,
                "traits": ["Scrap", "Twinshot"],
                "recommended_items": ["Last Whisper", "Infinity Edge", "Hurricane"]
            },
            "kayn": {
                "tier": 5,
                "traits": ["Challenger", "Assassin"],
                "recommended_items": ["Infinity Edge", "Bloodthirster", "Guardian Angel"]
            }
        }
    
    def _get_default_items_data(self) -> Dict[str, Any]:
        """Get default item data"""
        return {
            "infinity_edge": {
                "components": ["B.F. Sword", "Glove"],
                "stats": {"critical_strike_chance": 75, "critical_strike_damage": 225}
            },
            "rabadon's_deathcap": {
                "components": ["Needlessly Large Rod", "Needlessly Large Rod"],
                "stats": {"ability_power": 75}
            },
            "bloodthirster": {
                "components": ["B.F. Sword", "Negatron Cloak"],
                "stats": {"attack_damage": 15, "spell_vamp": 25}
            }
        }
    
    def _get_default_traits_data(self) -> Dict[str, Any]:
        """Get default trait data"""
        return {
            "academy": {
                "breakpoints": [3, 5, 7],
                "description": "Academy units gain Attack Damage and Ability Power"
            },
            "arcanist": {
                "breakpoints": [2, 4, 6, 8],
                "description": "Arcanists gain Ability Power"
            },
            "assassin": {
                "breakpoints": [2, 4, 6],
                "description": "Assassins gain Critical Strike Chance and Critical Strike Damage"
            }
        }
    
    def _get_default_compositions_data(self) -> Dict[str, Any]:
        """Get default composition data"""
        return {
            "meta_comps": [
                {
                    "name": "Academy Lux",
                    "tier": "S",
                    "core_champions": ["Lux", "Graves", "Leona", "Katarina"],
                    "key_traits": ["Academy", "Arcanist"],
                    "carry": "Lux",
                    "recommended_items": {
                        "Lux": ["Archangel's Staff", "Rabadon's Deathcap", "Blue Buff"]
                    }
                },
                {
                    "name": "Jinx Carry",
                    "tier": "A",
                    "core_champions": ["Jinx", "Vi", "Ezreal", "Zilean"],
                    "key_traits": ["Scrap", "Twinshot"],
                    "carry": "Jinx",
                    "recommended_items": {
                        "Jinx": ["Last Whisper", "Infinity Edge", "Hurricane"]
                    }
                }
            ]
        }
    
    def _save_champions_data(self):
        """Save champion data to file"""
        Path("data/champions").mkdir(parents=True, exist_ok=True)
        with open("data/champions/champion_data.json", 'w') as f:
            json.dump(self.champions_data, f, indent=2)
    
    def _save_items_data(self):
        """Save item data to file"""
        Path("data/items").mkdir(parents=True, exist_ok=True)
        with open("data/items/item_data.json", 'w') as f:
            json.dump(self.items_data, f, indent=2)
    
    def _save_traits_data(self):
        """Save trait data to file"""
        Path("data/traits").mkdir(parents=True, exist_ok=True)
        with open("data/traits/trait_data.json", 'w') as f:
            json.dump(self.traits_data, f, indent=2)
    
    def _save_compositions_data(self):
        """Save composition data to file"""
        Path("data/compositions").mkdir(parents=True, exist_ok=True)
        with open("data/compositions/meta_comps.json", 'w') as f:
            json.dump(self.compositions_data, f, indent=2)
    
    async def update_data_from_api(self):
        """Update data from external APIs"""
        if not self.auto_update:
            return
        
        # Check if update is needed
        if self.last_update and datetime.now() - self.last_update < timedelta(hours=6):
            return
        
        try:
            # TODO: Implement actual API calls to Riot API or community sites
            logger.info("Data update from API not implemented yet")
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Failed to update data from API: {e}")