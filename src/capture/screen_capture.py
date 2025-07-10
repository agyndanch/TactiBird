"""
Fixed screen capture module to handle threading issues
"""

import asyncio
import logging
import time
import threading
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import mss
import cv2

logger = logging.getLogger(__name__)

class ScreenCapture:
    """Handles screen capture for TFT game analysis - Fixed for threading issues"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = config.get('monitor', 0)
        self.fps = config.get('fps', 30)
        self.regions = config.get('regions', {})
        self.enable_debug = config.get('enable_debug', False)
        
        # Thread-local storage for MSS instances
        self._local = threading.local()
        
        # Get initial monitor info
        with mss.mss() as sct:
            self.monitors = sct.monitors
            self.target_monitor = self.monitors[self.monitor + 1] if self.monitor < len(self.monitors) - 1 else self.monitors[1]
        
        # Performance tracking
        self.last_capture_time = 0
        self.capture_count = 0
        self.avg_capture_time = 0
        
        logger.info(f"Screen capture initialized - Monitor: {self.monitor}, FPS: {self.fps}")
        logger.info(f"Target monitor: {self.target_monitor}")
    
    def _get_sct(self):
        """Get thread-local MSS instance"""
        if not hasattr(self._local, 'sct'):
            self._local.sct = mss.mss()
        return self._local.sct
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture full screen - Fixed for threading issues"""
        start_time = time.time()
        
        try:
            # Get thread-local MSS instance
            sct = self._get_sct()
            
            # Capture screenshot
            screenshot = sct.grab(self.target_monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Update performance tracking
            capture_time = time.time() - start_time
            self.capture_count += 1
            self.avg_capture_time = ((self.avg_capture_time * (self.capture_count - 1)) + capture_time) / self.capture_count
            self.last_capture_time = capture_time
            
            if self.enable_debug and self.capture_count % 30 == 0:  # Log every 30 captures
                logger.debug(f"Screen capture stats - Count: {self.capture_count}, "
                           f"Avg time: {self.avg_capture_time:.4f}s, "
                           f"Last time: {self.last_capture_time:.4f}s")
            
            return img
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def capture_region(self, region_name: str) -> Optional[np.ndarray]:
        """Capture specific region - Fixed for threading issues"""
        if region_name not in self.regions:
            logger.warning(f"Region '{region_name}' not configured")
            return None
        
        region_config = self.regions[region_name]
        
        try:
            # Get thread-local MSS instance
            sct = self._get_sct()
            
            # Calculate absolute coordinates
            monitor_left = self.target_monitor['left']
            monitor_top = self.target_monitor['top']
            
            region = {
                'left': monitor_left + region_config['x'],
                'top': monitor_top + region_config['y'],
                'width': region_config['w'],
                'height': region_config['h']
            }
            
            # Capture region
            screenshot = sct.grab(region)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
        except Exception as e:
            logger.error(f"Region capture failed for '{region_name}': {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self._local, 'sct'):
                self._local.sct.close()
                delattr(self._local, 'sct')
            logger.info("Screen capture cleanup completed")
        except Exception as e:
            logger.error(f"Error during screen capture cleanup: {e}")
    
    def get_fps_limit(self) -> float:
        """Get the delay needed to maintain target FPS"""
        return 1.0 / self.fps
    
    def should_capture(self) -> bool:
        """Check if enough time has passed for next capture based on FPS limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_capture_time
        return time_since_last >= self.get_fps_limit()
    
    async def capture_async(self) -> Optional[np.ndarray]:
        """Async wrapper for screen capture"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.capture_screen)
    
    async def capture_region_async(self, region_name: str) -> Optional[np.ndarray]:
        """Async wrapper for region capture"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.capture_region, region_name)