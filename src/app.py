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
        self._shutdown_started = False
        
        # Initialize components with config
        self.stats_detector = TFTStatsDetector(self.config)
        self.economy_coach = EconomyCoach(self.config)
        self.websocket_server = None
        self.server_task = None
        self.main_loop_task = None
        
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
        
        try:
            # Start websocket server
            if self.websocket_server is None:
                self.websocket_server = WebSocketServer(self.config)
                self.server_task = await self.websocket_server.start_server()
            
            # Start main loop
            self.main_loop_task = asyncio.create_task(self._main_loop())
            
            # Wait for main loop to complete
            await self.main_loop_task
            
        except asyncio.CancelledError:
            logger.info("Application start was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            raise
        finally:
            # Ensure we mark as not running
            self.running = False
    
    async def stop(self):
        """Stop the overlay application"""
        if self._shutdown_started:
            logger.debug("Shutdown already in progress")
            return
        
        self._shutdown_started = True
        logger.info("Stopping TactiBird Overlay...")
        self.running = False
        
        try:
            # Cancel main loop first
            if self.main_loop_task and not self.main_loop_task.done():
                self.main_loop_task.cancel()
                try:
                    await asyncio.wait_for(self.main_loop_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.debug("Main loop cancelled")
            
            # Stop websocket server
            if self.websocket_server:
                await self.websocket_server.stop_server()
            
            # Close server task
            if self.server_task:
                self.server_task.close()
                try:
                    await self.server_task.wait_closed()
                except Exception as e:
                    logger.debug(f"Server task close error: {e}")
                    
        except Exception as e:
            logger.error(f"Error during stop: {e}")
    
    async def _main_loop(self):
        """Main application loop"""
        fps = self.config.get('capture', {}).get('fps', 2)
        frame_time = 1.0 / fps
        
        logger.info(f"Starting main loop with {fps} FPS")
        
        try:
            while self.running:
                try:
                    loop_start = time.time()
                    
                    # Check if we should continue running
                    if not self.running:
                        break
                    
                    # Capture screen
                    screenshot = await self._capture_screen()
                    if screenshot is None:
                        # Always await sleep to allow for cancellation
                        await asyncio.sleep(frame_time)
                        continue
                    
                    # Detect stats
                    stats = await self.stats_detector.detect_stats(screenshot)
                    if stats and stats.in_game:
                        # Update game state
                        self.last_stats = stats
                        self.stats_history.append(stats)
                        
                        # Create game state object for coach (with .stats attribute)
                        game_state_for_coach = type('GameState', (), {
                            'stats': stats,
                            'timestamp': time.time()
                        })()
                        
                        # Generate coaching suggestions using get_suggestions method
                        suggestions = await self.economy_coach.get_suggestions(game_state_for_coach)
                        
                        # Send update to UI (pass stats and suggestions separately)
                        await self._send_update(stats, suggestions)
                    
                    # Maintain frame rate - always sleep to allow cancellation
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.01, frame_time - elapsed)  # Minimum 10ms sleep
                    await asyncio.sleep(sleep_time)
                    
                except asyncio.CancelledError:
                    logger.info("Main loop cancelled")
                    break
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt in main loop")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    # Always sleep to allow cancellation even on errors
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("Main loop task cancelled")
            raise  # Re-raise to propagate cancellation
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt caught in main loop")
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
        finally:
            logger.info("Main loop stopped")
            self.running = False
    
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
                    # Create game state dictionary that matches WebSocketServer expectations
                    game_state = {
                        'stats': stats,
                        'suggestions': suggestions,
                        'timestamp': time.time()
                    }
                    await self.websocket_server.update_game_state(game_state)

            except Exception as e:
                logger.error(f"Failed to send update: {e}")