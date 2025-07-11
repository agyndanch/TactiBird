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
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize components
        self.stats_detector = TFTStatsDetector()
        self.economy_coach = EconomyCoach()
        self.websocket_server = None
        
        # Screen capture
        self.sct = mss.mss()
        
        # Detection regions (will be auto-detected or configured)
        self.detection_regions = {}
        
        # Game state tracking
        self.last_stats = GameStats()
        self.stats_history = []
        
        logger.info("TFT Economy Overlay initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "capture": {
                "fps": 2,  # Lower FPS for economy overlay
                "monitor": 0
            },
            "detection": {
                "confidence_threshold": 0.7,
                "auto_detect_regions": True
            },
            "overlay": {
                "port": 8765,
                "update_interval": 1.0
            },
            "economy": {
                "suggestion_cooldown": 5.0  # Seconds between similar suggestions
            }
        }
    
    async def start(self):
        """Start the overlay application"""
        try:
            self.running = True
            
            # Start WebSocket server for overlay UI
            self.websocket_server = WebSocketServer(
                port=self.config["overlay"]["port"]
            )
            await self.websocket_server.start()
            
            # Initialize detection regions
            await self._initialize_detection_regions()
            
            # Start main detection loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Failed to start overlay: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the overlay application"""
        self.running = False
        
        if self.websocket_server:
            await self.websocket_server.stop()
        
        logger.info("TFT Economy Overlay stopped")
    
    async def _initialize_detection_regions(self):
        """Initialize or auto-detect screen regions for OCR"""
        monitor = self.sct.monitors[self.config["capture"]["monitor"]]
        screen_width = monitor["width"]
        screen_height = monitor["height"]
        
        if self.config["detection"]["auto_detect_regions"]:
            # Use default regions based on screen size
            self.detection_regions = self.stats_detector.get_default_regions(
                screen_width, screen_height
            )
            logger.info("Using auto-detected regions")
        else:
            # Load custom regions from config
            self.detection_regions = self.config.get("regions", {})
            logger.info("Using configured regions")
        
        logger.info(f"Detection regions: {self.detection_regions}")
    
    async def _main_loop(self):
        """Main detection and coaching loop"""
        update_interval = self.config["overlay"]["update_interval"]
        
        while self.running:
            try:
                start_time = time.time()
                
                # Capture screenshot
                screenshot = self._capture_screenshot()
                if screenshot is None:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Detect game statistics
                stats = self.stats_detector.detect_stats(screenshot, self.detection_regions)
                
                # Only process if we have valid stats
                if self._is_valid_stats(stats):
                    # Generate economy suggestions
                    suggestions = self._get_economy_suggestions(stats)
                    
                    # Send to overlay UI
                    await self._send_to_overlay(stats, suggestions)
                    
                    # Update tracking
                    self.last_stats = stats
                    self.stats_history.append(stats)
                    
                    # Keep only recent history (last 10 readings)
                    if len(self.stats_history) > 10:
                        self.stats_history.pop(0)
                
                # Maintain update rate
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(update_interval)
    
    def _capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot of the specified monitor"""
        try:
            monitor = self.sct.monitors[self.config["capture"]["monitor"]]
            screenshot = self.sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def _is_valid_stats(self, stats: GameStats) -> bool:
        """Check if detected stats are valid"""
        # Require at least gold and health to be detected
        return (stats.gold is not None and 
                stats.health is not None and
                stats.confidence.get('gold', 0) > self.config["detection"]["confidence_threshold"] and
                stats.confidence.get('health', 0) > self.config["detection"]["confidence_threshold"])
    
    def _get_economy_suggestions(self, stats: GameStats) -> list:
        """Generate economy suggestions based on current stats"""
        if not self._is_valid_stats(stats):
            return []
        
        # Use default values if not detected
        level = stats.level or 5
        stage = stats.stage or 3
        round_num = stats.round_num or 1
        
        suggestions = self.economy_coach.get_suggestions(
            gold=stats.gold,
            health=stats.health,
            level=level,
            stage=stage,
            round_num=round_num
        )
        
        # Filter suggestions based on cooldown
        filtered_suggestions = self._filter_suggestions_by_cooldown(suggestions)
        
        return [
            {
                "message": s.message,
                "priority": s.priority,
                "category": s.category,
                "timestamp": s.timestamp.isoformat()
            }
            for s in filtered_suggestions
        ]
    
    def _filter_suggestions_by_cooldown(self, suggestions) -> list:
        """Filter suggestions to avoid spam"""
        # TODO: Implement cooldown logic to prevent repeated suggestions
        # For now, just return all suggestions
        return suggestions
    
    async def _send_to_overlay(self, stats: GameStats, suggestions: list):
        """Send data to overlay UI via WebSocket"""
        if not self.websocket_server:
            return
        
        # Get economy status
        economy_status = self.economy_coach.get_economy_status(
            gold=stats.gold,
            health=stats.health,
            level=stats.level or 5,
            stage=stats.stage or 3
        )
        
        # Prepare data for overlay
        overlay_data = {
            "type": "update",
            "stats": {
                "gold": stats.gold,
                "health": stats.health,
                "level": stats.level,
                "stage": stats.stage,
                "round": stats.round_num,
                "confidence": stats.confidence
            },
            "economy": economy_status,
            "suggestions": suggestions,
            "timestamp": time.time()
        }
        
        await self.websocket_server.broadcast(json.dumps(overlay_data))
    
    def calibrate_regions(self, screenshot: np.ndarray) -> Dict[str, tuple]:
        """Manual calibration of detection regions (for setup)"""
        # TODO: Implement interactive region selection
        # For now, return default regions
        height, width = screenshot.shape[:2]
        return self.stats_detector.get_default_regions(width, height)

class WebSocketServer:
    """Simple WebSocket server for overlay communication"""
    
    def __init__(self, port: int):
        self.port = port
        self.clients = set()
        self.server = None
    
    async def start(self):
        """Start the WebSocket server"""
        import websockets
        
        async def handle_client(websocket, path):
            self.clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.clients.remove(websocket)
        
        self.server = await websockets.serve(handle_client, "localhost", self.port)
        logger.info(f"WebSocket server started on port {self.port}")
    
    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )

# Entry point
async def main():
    """Main entry point"""
    overlay = TactiBirdOverlay()
    
    try:
        await overlay.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await overlay.stop()

if __name__ == "__main__":
    asyncio.run(main())