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
        
        # Start websocket server - FIXED: Pass the full config instead of just the port
        if self.websocket_server is None:
            # Pass the entire config to WebSocketServer constructor
            self.websocket_server = WebSocketServer(self.config)
            # Store the server instance for proper cleanup
            self.server_task = await self.websocket_server.start_server()
        
        # Start main loop
        await self._main_loop()
    
    async def stop(self):
        """Stop the overlay application"""
        logger.info("Stopping TactiBird Overlay...")
        self.running = False
        
        if self.websocket_server:
            await self.websocket_server.stop_server()  # Use stop_server method
            if hasattr(self, 'server_task') and self.server_task:
                self.server_task.close()
                try:
                    await self.server_task.wait_closed()
                except Exception as e:
                    logger.error(f"Error closing server: {e}")
    
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
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(frame_time)
                    
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
            raise
    
    async def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture screenshot"""
        try:
            # Use configured monitor or default to primary
            monitor_config = self.config.get('capture', {}).get('monitor', {})
            if monitor_config:
                monitor = monitor_config
            else:
                monitor = self.sct.monitors[1]  # Primary monitor
            
            # Capture screenshot
            screenshot = self.sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert BGRA to BGR (remove alpha channel)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None
    
    async def _send_update(self, stats: GameStats, suggestions: list):
        """Send update to websocket clients"""
        try:
            if self.websocket_server:
                # Create game state object
                game_state = {
                    'stats': stats,
                    'suggestions': suggestions,
                    'timestamp': time.time()
                }
                
                # Update game state and broadcast to clients
                await self.websocket_server.update_game_state(game_state)
                
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
    
    def get_stats_history(self, limit: int = 10) -> list:
        """Get recent stats history"""
        return self.stats_history[-limit:] if self.stats_history else []
    
    def clear_stats_history(self):
        """Clear stats history"""
        self.stats_history.clear()
        logger.info("Stats history cleared")
    
    async def calibrate_regions(self):
        """Auto-calibrate detection regions"""
        try:
            logger.info("Starting region calibration...")
            
            # Take screenshot for calibration
            screenshot = await self._capture_screen()
            if screenshot is None:
                logger.error("Failed to capture screenshot for calibration")
                return False
            
            # Use stats detector to find regions
            regions = await self.stats_detector.auto_detect_regions(screenshot)
            if regions:
                self.detection_regions = regions
                logger.info(f"Calibrated regions: {list(regions.keys())}")
                return True
            else:
                logger.warning("Failed to auto-detect regions")
                return False
                
        except Exception as e:
            logger.error(f"Error during region calibration: {e}")
            return False