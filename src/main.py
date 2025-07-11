"""
TactiBird Overlay - Main Entry Point
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app import TactiBirdOverlay
from src.config import load_config, ensure_directories

def setup_logging():
    """Setup logging configuration"""
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'tactibird.log')
        ]
    )

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='TactiBird Overlay - AI-powered TFT coaching')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--calibrate', action='store_true', help='Run region calibration')
    parser.add_argument('--test-ocr', action='store_true', help='Test OCR detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Ensure required directories exist
    ensure_directories()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    if args.calibrate:
        logger.info("Starting region calibration...")
        from src.utils.calibration import RegionCalibrator
        calibrator = RegionCalibrator(config)
        await calibrator.run()
        return 0
    
    if args.test_ocr:
        logger.info("Testing OCR detection...")
        from src.vision.ocr.text_recognizer import TextRecognizer
        recognizer = TextRecognizer(config)
        await recognizer.test_detection()
        return 0
    
    # Start the main overlay application
    logger.info("Starting TactiBird Overlay...")
    overlay = TactiBirdOverlay(config)
    
    try:
        await overlay.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    finally:
        await overlay.stop()
    
    return 0

if __name__ == "__main__":
    # Run the application
    exit_code = asyncio.run(main())
    sys.exit(exit_code)