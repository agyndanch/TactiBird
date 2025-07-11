"""
TFT Economy Overlay - Simplified Main Application
Focuses only on economy management with OCR-based stat detection
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

import mss
import cv2
import numpy as np

from src.vision.ocr.tft_stats_detector import TFTStatsDetector, GameStats
from src.ai.coaches.economy_coach import EconomyCoach
from src.overlay.server.websocket_server import WebSocketServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TactiBirdOverlay:
    """Simplified TFT overlay focused on economy management"""
    
    def __init__(self, config_path: str = "config.json"):
        # Use the centralized config loader instead of custom one
        from src.config import load_config
        self.config = load_config(config_path)
        self.running = False
        
        # Initialize components with config
        self.stats_detector = TFTStatsDetector(self.config)  # Pass config here
        self.economy_coach = EconomyCoach(self.config)       # Pass config here too
        self.websocket_server = None
        
        # Screen capture
        self.sct = mss.mss()
        
        # Detection regions (will be auto-detected or configured)
        self.detection_regions = {}
        
        # Game state tracking
        self.last_stats = GameStats()
        self.stats_history = []
        
        logger.info("TFT Economy Overlay initialized")

    async def start(self):
        """Start the overlay application"""
        logger.info("Starting TactiBird Overlay...")
        self.running = True
        
        # Start websocket server
        if self.websocket_server is None:
            port = self.config.get('overlay', {}).get('port', 8765)
            self.websocket_server = WebSocketServer(port)
            await self.websocket_server.start()
        
        # Start main loop
        await self._main_loop()
    
    async def stop(self):
        """Stop the overlay application"""
        logger.info("Stopping TactiBird Overlay...")
        self.running = False
        
        if self.websocket_server:
            await self.websocket_server.stop()
    
    async def _main_loop(self):
        """Main application loop"""
        fps = self.config.get('capture', {}).get('fps', 2)
        frame_time = 1.0 / fps
        
        try:
            while self.running:
                try:
                    loop_start = time.time()
                    
                    # Capture screen
                    screenshot = await self._capture_screen()
                    if screenshot is None:
                        await asyncio.sleep(frame_time)
                        continue
                    
                    # Detect stats
                    stats = await self.stats_detector.detect_stats(screenshot)
                    if stats and stats.in_game:
                        # Update game state
                        self.last_stats = stats
                        self.stats_history.append(stats)
                        
                        # Generate coaching suggestions
                        suggestions = await self.economy_coach.analyze(stats)
                        
                        # Send update to UI
                        await self._send_update(stats, suggestions)
                    
                    # Maintain frame rate
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, frame_time - elapsed)
                    await asyncio.sleep(sleep_time)
                    
                except asyncio.CancelledError:
                    # Handle graceful shutdown
                    logger.info("Main loop cancelled, shutting down gracefully")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(frame_time)
        
        except asyncio.CancelledError:
            logger.info("Main loop task cancelled")
        finally:
            logger.info("Main loop finished")
    
    async def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture game screen"""
        try:
            # Use MSS for screen capture
            monitor = self.sct.monitors[1]  # Primary monitor
            screenshot = self.sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None
    
    async def _send_update(self, stats: GameStats, suggestions: list):
        """Send update to overlay UI"""
        if self.websocket_server:
            update_data = {
                'type': 'stats_update',
                'stats': stats.to_dict(),
                'suggestions': [s.to_dict() for s in suggestions],
                'timestamp': time.time()
            }
            await self.websocket_server.broadcast(update_data)