"""
TactiBird Overlay - Logging Configuration (Fixed)
"""

import logging
import logging.handlers
import sys
import re
from pathlib import Path
from typing import Optional

def parse_file_size(size_str: str) -> int:
    """
    Parse file size string (e.g., '10M', '5MB', '1GB') to bytes
    
    Args:
        size_str: Size string like '10M', '5MB', '100KB', etc.
        
    Returns:
        Size in bytes
    """
    if isinstance(size_str, int):
        return size_str
    
    # Remove whitespace and convert to uppercase
    size_str = str(size_str).strip().upper()
    
    # Define size multipliers
    multipliers = {
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    # Use regex to extract number and unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGB]*)$', size_str)
    if not match:
        # If no valid format, default to 10MB
        return 10 * 1024 * 1024
    
    number, unit = match.groups()
    number = float(number)
    
    # Default to bytes if no unit specified
    if not unit:
        return int(number)
    
    # Find matching multiplier
    multiplier = multipliers.get(unit, 1)
    return int(number * multiplier)

def setup_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: str = "logs/app.log",
    max_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration for the application
    
    Args:
        name: Logger name (default: root logger)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_size: Maximum log file size before rotation (e.g., '10MB', '5M', '1GB')
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    try:
        # Parse max_size to bytes
        max_bytes = parse_file_size(max_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        # Log the error but continue with console logging
        console_handler.setFormatter(detailed_formatter)  # Use detailed format for console if file fails
        logger.warning(f"Failed to setup file logging: {e}")
        logger.info("Continuing with console logging only")
    
    # Set third-party library log levels to reduce noise
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('mss').setLevel(logging.WARNING)
    logging.getLogger('cv2').setLevel(logging.WARNING)
    
    logger.info(f"Logger initialized - Level: {level}, File: {log_file}")
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)

class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.debug(f"Completed {self.operation} in {elapsed:.3f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {elapsed:.3f}s: {exc_val}")

# Helper function for easy performance logging
def log_performance(logger: logging.Logger, operation: str):
    """Decorator or context manager for performance logging"""
    return PerformanceLogger(logger, operation)