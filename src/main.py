#!/usr/bin/env python3
"""
TactiBird Overlay - Main Application Entry Point
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add both src and parent directory to path to work from anywhere
current_dir = Path(__file__).parent
root_dir = current_dir.parent

# Add src directory to path
sys.path.insert(0, str(current_dir))
# Add root directory to path (in case config.json is there)
sys.path.insert(0, str(root_dir))

from app import TFTCoachingApp
from logger import setup_logger
from config import load_config

def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting TactiBird Overlay")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create and run the application
        app = TFTCoachingApp(config)
        asyncio.run(app.run())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("TactiBird Overlay stopped")

if __name__ == "__main__":
    main()