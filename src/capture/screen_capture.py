"""
TactiBird - Screen Capture Module
"""

import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import mss
import cv2

logger = logging.getLogger(__name__)

class ScreenCapture:
    """Screen capture functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fps = config.get('fps', 30)
        self.method = config.get('method', 'mss')
        self.window_title = config.get('window_title', 'Teamfight Tactics')
        
        # Initialize capture method
        if self.method == 'mss':
            self.sct = mss.mss()
        
        self.running = False
        self.last_screenshot = None
        
    async def start(self):
        """Start screen capture"""
        self.running = True
        logger.info(f"Screen capture started with {self.method} method")
    
    async def stop(self):
        """Stop screen capture"""
        self.running = False
        logger.info("Screen capture stopped")
    
    async def capture(self) -> Optional[np.ndarray]:
        """Capture screenshot"""
        try:
            if self.method == 'mss':
                return self._capture_mss()
            else:
                logger.error(f"Unsupported capture method: {self.method}")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None
    
    def _capture_mss(self) -> Optional[np.ndarray]:
        """Capture using MSS library"""
        try:
            # Capture primary monitor
            monitor = self.sct.monitors[1]  # Index 0 is all monitors combined
            screenshot = self.sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            self.last_screenshot = img
            return img
            
        except Exception as e:
            logger.error(f"MSS capture error: {e}")
            return None