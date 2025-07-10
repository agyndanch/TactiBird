"""
Fixed TFTCoachingApp - adds missing positioning_coach and fixes tesseract
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import pytesseract
import os

from src.capture.screen_capture import ScreenCapture
from src.vision.detection.board_analyzer import BoardAnalyzer
from src.vision.detection.shop_analyzer import ShopAnalyzer
from src.ai.coaches.economy_coach import EconomyCoach
from src.ai.coaches.composition_coach import CompositionCoach
from src.ai.coaches.item_coach import ItemCoach
from src.overlay.server.websocket_server import WebSocketServer
from src.data.models.game_state import GameState
from src.data.managers.data_manager import DataManager
from src.ai.coaches.base_coach import CoachingSuggestion

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    # Try common installation paths
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.environ.get('USERNAME', '')),
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

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
            
            # Create a placeholder positioning coach if it doesn't exist
            try:
                from src.ai.coaches.positioning_coach import PositioningCoach
                self.positioning_coach = PositioningCoach(self.data_manager)
            except ImportError:
                self.logger.warning("PositioningCoach not found, creating placeholder")
                self.positioning_coach = self._create_placeholder_coach("positioning")
            
            # Overlay server
            self.websocket_server = WebSocketServer(
                port=self.config.get('overlay', {}).get('port', 8765)
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _create_placeholder_coach(self, coach_type: str):
        """Create a placeholder coach that returns no suggestions"""
        class PlaceholderCoach:
            def __init__(self, data_manager):
                self.data_manager = data_manager
                self.name = f"{coach_type.title()} Coach (Placeholder)"
                self.enabled = False
            
            def get_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
                return []
            
            def is_applicable(self, game_state: GameState) -> bool:
                return False
        
        return PlaceholderCoach(self.data_manager)
    
    async def run(self):
        """Main application loop"""
        self.running = True
        self.logger.info("Starting TactiBird Overlay...")

        try:
            # Start the overlay server first and wait for it to be ready
            self.logger.info("Starting WebSocket server...")
            await self.websocket_server.start()

            # Verify server is running
            if not self.websocket_server.is_running():
                raise RuntimeError("WebSocket server failed to start properly")

            self.logger.info(f"WebSocket server is running on port {self.websocket_server.port}")

            # Add a small delay to ensure server is fully ready
            await asyncio.sleep(0.5)

            # Now start the main coaching loop
            self.logger.info("Starting coaching loop...")
            await self._coaching_loop()

        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
            raise
        finally:
            await self.stop()
    
    async def _coaching_loop(self):
        """Main coaching loop that captures, analyzes, and provides suggestions"""
        update_interval = self.config.get('ai', {}).get('update_interval', 0.5)
        
        while self.running:
            try:
                # Capture screen
                screenshot = await self.screen_capture.capture_async()
                if screenshot is None:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Analyze game state
                game_state = await self._analyze_game_state(screenshot)
                if game_state is None:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Generate suggestions
                suggestions = await self._generate_suggestions(game_state)
                
                # Broadcast to overlay
                if suggestions:
                    await self.websocket_server.broadcast_suggestions(suggestions)
                
                # Broadcast game state
                await self.websocket_server.broadcast_game_state(game_state)
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in coaching loop: {e}", exc_info=True)
                await asyncio.sleep(update_interval)
    
    async def _analyze_game_state(self, screenshot) -> Optional[GameState]:
        """Analyze current game state from screenshot"""
        try:
            # Run analysis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Analyze board
            board_state = await loop.run_in_executor(
                None, self.board_analyzer.analyze, screenshot
            )
            
            # Analyze shop
            shop_state = await loop.run_in_executor(
                None, self.shop_analyzer.analyze, screenshot
            )
            
            # Create game state with required timestamp
            current_time = datetime.now()
            game_state = GameState(timestamp=current_time)
            
            # Set analyzed states
            game_state.board = board_state if board_state is not None else GameState(timestamp=current_time).board
            game_state.shop = shop_state if shop_state is not None else GameState(timestamp=current_time).shop
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"Game state analysis failed: {e}")
            return None
    
    async def _generate_suggestions(self, game_state: GameState) -> List[CoachingSuggestion]:
        """Generate coaching suggestions from all coaches"""
        suggestions = []
        
        try:
            coaches = [
                self.economy_coach,
                self.composition_coach,
                self.item_coach,
                self.positioning_coach
            ]
            
            for coach in coaches:
                if coach.is_applicable(game_state):
                    coach_suggestions = coach.get_suggestions(game_state)
                    suggestions.extend(coach_suggestions)
            
            # Sort by priority (highest first)
            suggestions.sort(key=lambda s: s.priority, reverse=True)
            
            # Limit to max suggestions
            max_suggestions = self.config.get('ai', {}).get('max_suggestions', 5)
            suggestions = suggestions[:max_suggestions]
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []
    
    async def stop(self):
        """Stop the application gracefully"""
        self.logger.info("Stopping TactiBird Overlay...")
        self.running = False
        
        try:
            # Stop WebSocket server
            if hasattr(self, 'websocket_server'):
                await self.websocket_server.stop()
            
            # Cleanup screen capture
            if hasattr(self, 'screen_capture'):
                self.screen_capture.cleanup()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        self.logger.info("TactiBird Overlay stopped")