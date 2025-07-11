"""
TactiBird - Region Calibration Module
"""

import asyncio
import logging
import cv2
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RegionCalibrator:
    """Interactive region calibration tool"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regions = config['capture']['regions'].copy()
        
    async def run(self):
        """Run interactive calibration"""
        print("Region Calibration Tool")
        print("=" * 30)
        print("This tool helps you set up screen regions for your resolution.")
        print("Follow the on-screen instructions to calibrate each region.")
        print()
        
        try:
            from src.capture.screen_capture import ScreenCapture
            
            # Initialize screen capture
            capture = ScreenCapture(self.config['capture'])
            await capture.start()
            
            # Take a screenshot for calibration
            screenshot = await capture.capture()
            if screenshot is None:
                print("❌ Failed to capture screenshot")
                return
            
            print(f"Screenshot captured: {screenshot.shape[1]}x{screenshot.shape[0]}")
            print()
            
            # Calibrate each region
            regions_to_calibrate = ['gold', 'health', 'level', 'stage', 'shop', 'board']
            
            for region_name in regions_to_calibrate:
                print(f"Calibrating {region_name} region...")
                await self._calibrate_region(screenshot, region_name)
            
            # Save updated configuration
            self.config['capture']['regions'] = self.regions
            
            from src.config import save_config
            save_config(self.config, 'config.json')
            
            print("✅ Calibration complete! Updated config.json")
            
            await capture.stop()
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            print(f"❌ Calibration failed: {e}")
    
    async def _calibrate_region(self, screenshot: np.ndarray, region_name: str):
        """Calibrate a specific region"""
        # For now, use default regions based on screen size
        # In a full implementation, this would open an interactive GUI
        
        height, width = screenshot.shape[:2]
        
        if region_name == 'gold':
            # Bottom right area
            self.regions[region_name] = {
                'x': int(width * 0.85), 'y': int(height * 0.85),
                'w': int(width * 0.1), 'h': int(height * 0.05)
            }
        elif region_name == 'health':
            # Top left area
            self.regions[region_name] = {
                'x': int(width * 0.05), 'y': int(height * 0.05),
                'w': int(width * 0.08), 'h': int(height * 0.04)
            }
        elif region_name == 'level':
            # Bottom left area
            self.regions[region_name] = {
                'x': int(width * 0.25), 'y': int(height * 0.85),
                'w': int(width * 0.06), 'h': int(height * 0.04)
            }
        elif region_name == 'stage':
            # Top center area
            self.regions[region_name] = {
                'x': int(width * 0.45), 'y': int(height * 0.02),
                'w': int(width * 0.1), 'h': int(height * 0.04)
            }
        elif region_name == 'shop':
            # Bottom center area
            self.regions[region_name] = {
                'x': int(width * 0.25), 'y': int(height * 0.75),
                'w': int(width * 0.5), 'h': int(height * 0.1)
            }
        elif region_name == 'board':
            # Center area
            self.regions[region_name] = {
                'x': int(width * 0.25), 'y': int(height * 0.15),
                'w': int(width * 0.5), 'h': int(height * 0.55)
            }
        
        region = self.regions[region_name]
        print(f"   {region_name}: x={region['x']}, y={region['y']}, w={region['w']}, h={region['h']}")