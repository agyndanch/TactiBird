"""
TactiBird Overlay - Main Application Class
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.capture.screen_capture import ScreenCapture
from src.vision.detection.board_analyzer import BoardAnalyzer
from src.vision.detection.shop_analyzer import ShopAnalyzer
from src.ai.coaches.economy_coach import EconomyCoach
from src.ai.coaches.composition_coach import CompositionCoach
from src.ai.coaches.item_coach import ItemCoach
from src.overlay.server.websocket_server import WebSocketServer
from src.data.models.game_state import GameState
from src.data.managers.data_manager import DataManager

@dataclass
class CoachingSuggestion:
    """Represents a coaching suggestion"""
    type: str
    message: str
    priority: int
    timestamp: datetime
    context: Dict[str, Any]

class TFTCoachingApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize all application components"""
        try:
            # Data management
            self.data_manager = DataManager(self.config.get('data', {}))
            
            # Screen capture
            self.screen_capture = ScreenCapture(self.config.get('capture', {}))
            
            # Vision components
            self.board_analyzer = BoardAnalyzer(self.data_manager)
            self.shop_analyzer = ShopAnalyzer(self.data_manager)
            
            # AI coaches
            self.economy_coach = EconomyCoach(self.data_manager)
            self.composition_coach = CompositionCoach(self.data_manager)
            self.item_coach = ItemCoach(self.data_manager)
            
            # Overlay server
            self.websocket_server = WebSocketServer(
                port=self.config.get('overlay', {}).get('port', 8765)
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run(self):
        """Main application loop"""
        self.running = True
        self.logger.info("Starting TactiBird Overlay...")
        
        try:
            # Start the overlay server
            await self.websocket_server.start()
            
            # Start the main coaching loop
            await self._coaching_loop()
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            raise
        finally:
            await self.stop()
    
    async def _coaching_loop(self):
        """Main coaching loop that captures, analyzes, and provides suggestions"""
        update_interval = self.config.get('ai', {}).get('update_interval', 0.5)
        
        while self.running:
            try:
                # Capture current game state
                screenshot = await self.screen_capture.capture_async()
                if screenshot is None:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Analyze the game state
                game_state = await self._analyze_game_state(screenshot)
                if game_state is None:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Generate coaching suggestions
                suggestions = await self._generate_suggestions(game_state)
                
                # Send suggestions to overlay
                if suggestions:
                    await self.websocket_server.broadcast_suggestions(suggestions)
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in coaching loop: {e}")
                await asyncio.sleep(update_interval)
    
    async def _analyze_game_state(self, screenshot) -> Optional[GameState]:
        """Analyze screenshot to extract game state"""
        try:
            # Analyze board state
            board_state = await asyncio.get_event_loop().run_in_executor(
                None, self.board_analyzer.analyze, screenshot
            )
            
            # Analyze shop state
            shop_state = await asyncio.get_event_loop().run_in_executor(
                None, self.shop_analyzer.analyze, screenshot
            )
            
            # Create game state object
            game_state = GameState(
                board=board_state,
                shop=shop_state,
                timestamp=datetime.now()
            )
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"Error analyzing game state: {e}")
            return None
    
    async def _generate_suggestions(self, game_state: GameState) -> list[CoachingSuggestion]:
        """Generate coaching suggestions based on game state"""
        suggestions = []
        
        try:
            # Get suggestions from each coach
            coaches = [
                self.economy_coach,
                self.composition_coach,
                self.positioning_coach,
                self.item_coach
            ]
            
            for coach in coaches:
                coach_suggestions = await asyncio.get_event_loop().run_in_executor(
                    None, coach.get_suggestions, game_state
                )
                suggestions.extend(coach_suggestions)
            
            # Sort by priority
            suggestions.sort(key=lambda x: x.priority, reverse=True)
            
            # Limit number of suggestions
            max_suggestions = self.config.get('ai', {}).get('max_suggestions', 5)
            return suggestions[:max_suggestions]
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []
    
    async def stop(self):
        """Stop the application"""
        self.logger.info("Stopping TactiBird Overlay...")
        self.running = False
        
        # Stop overlay server
        if hasattr(self, 'websocket_server'):
            await self.websocket_server.stop()
        
        # Cleanup screen capture
        if hasattr(self, 'screen_capture'):
            self.screen_capture.cleanup()
        
        self.logger.info("Application stopped")