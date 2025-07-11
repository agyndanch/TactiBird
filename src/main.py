"""
TFT Economy Overlay - Main Entry Point
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.app import TFTEconomyOverlay

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/economy_overlay.log')
        ]
    )

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='TFT Economy Overlay')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--calibrate', action='store_true', help='Run region calibration')
    parser.add_argument('--test-ocr', action='store_true', help='Test OCR detection')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.calibrate:
        logger.info("Starting region calibration...")
        # TODO: Implement calibration mode
        print("Calibration mode not yet implemented")
        return
    
    if args.test_ocr:
        logger.info("Testing OCR detection...")
        # TODO: Implement OCR test mode
        print("OCR test mode not yet implemented")
        return
    
    # Start the main overlay application
    logger.info("Starting TFT Economy Overlay...")
    overlay = TFTEconomyOverlay(args.config)
    
    try:
        await overlay.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await overlay.stop()

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the application
    asyncio.run(main())