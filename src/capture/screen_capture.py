"""
TactiBird Overlay - Screen Capture Module
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import mss
import cv2

logger = logging.getLogger(__name__)

class ScreenCapture:
    """Handles screen capture for TFT game analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = config.get('monitor', 0)
        self.fps = config.get('fps', 30)
        self.regions = config.get('regions', {})
        self.enable_debug = config.get('enable_debug', False)
        
        # Initialize screen capture
        self.sct = mss.mss()
        self.monitors = self.sct.monitors
        self.target_monitor = self.monitors[self.monitor + 1] if self.monitor < len(self.monitors) - 1 else self.monitors[1]
        
        # Performance tracking
        self.last_capture_time = 0
        self.capture_count = 0
        self.avg_capture_time = 0
        
        logger.info(f"Screen capture initialized - Monitor: {self.monitor}, FPS: {self.fps}")
        logger.info(f"Target monitor: {self.target_monitor}")
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture full screen"""
        start_time = time.time()
        
        try:
            # Capture screenshot
            screenshot = self.sct.grab(self.target_monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            if self.enable_debug:
                logger.debug(f"Screen capture successful - Shape: {img.shape}")
            
            return img
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def capture_region(self, region_name: str) -> Optional[np.ndarray]:
        """Capture specific region of the screen"""
        if region_name not in self.regions:
            logger.warning(f"Unknown region: {region_name}")
            return None
        
        region = self.regions[region_name]
        
        # Calculate absolute coordinates
        monitor_region = {
            'top': self.target_monitor['top'] + region['y'],
            'left': self.target_monitor['left'] + region['x'],
            'width': region['w'],
            'height': region['h']
        }
        
        start_time = time.time()
        
        try:
            # Capture region
            screenshot = self.sct.grab(monitor_region)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            if self.enable_debug:
                logger.debug(f"Region capture successful - {region_name}: {img.shape}")
            
            return img
            
        except Exception as e:
            logger.error(f"Region capture failed for {region_name}: {e}")
            return None
    
    def capture_multiple_regions(self, region_names: list) -> Dict[str, np.ndarray]:
        """Capture multiple regions at once"""
        results = {}
        
        for region_name in region_names:
            img = self.capture_region(region_name)
            if img is not None:
                results[region_name] = img
        
        return results
    
    async def capture_async(self) -> Optional[np.ndarray]:
        """Asynchronous screen capture"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.capture_screen)
    
    async def capture_region_async(self, region_name: str) -> Optional[np.ndarray]:
        """Asynchronous region capture"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.capture_region, region_name)
    
    def get_game_window_bounds(self) -> Optional[Dict[str, int]]:
        """Get TFT game window bounds (placeholder for window detection)"""
        # TODO: Implement actual game window detection
        # This could use Windows API, X11, or other platform-specific methods
        logger.warning("Game window detection not implemented, using full monitor")
        return {
            'x': self.target_monitor['left'],
            'y': self.target_monitor['top'],
            'width': self.target_monitor['width'],
            'height': self.target_monitor['height']
        }
    
    def update_regions(self, new_regions: Dict[str, Dict[str, int]]):
        """Update capture regions"""
        self.regions.update(new_regions)
        logger.info(f"Updated capture regions: {list(new_regions.keys())}")
    
    def save_screenshot(self, img: np.ndarray, filename: str):
        """Save screenshot for debugging"""
        try:
            cv2.imwrite(filename, img)
            logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
    
    def _update_performance_metrics(self, start_time: float):
        """Update performance tracking metrics"""
        capture_time = time.time() - start_time
        self.capture_count += 1
        self.avg_capture_time = (self.avg_capture_time * (self.capture_count - 1) + capture_time) / self.capture_count
        self.last_capture_time = capture_time
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get capture performance statistics"""
        return {
            'last_capture_time': self.last_capture_time,
            'avg_capture_time': self.avg_capture_time,
            'capture_count': self.capture_count,
            'estimated_fps': 1.0 / max(self.avg_capture_time, 0.001)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'sct'):
            self.sct.close()
        logger.info("Screen capture cleanup completed")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()