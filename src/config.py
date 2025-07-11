"""
TactiBird Overlay - Configuration Management
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "capture": {
        "fps": 30,
        "method": "mss",  # mss, win32, or dxcam
        "regions": {
            "board": {"x": 100, "y": 200, "w": 800, "h": 600},
            "shop": {"x": 200, "y": 850, "w": 600, "h": 150},
            "gold": {"x": 50, "y": 950, "w": 100, "h": 50},
            "health": {"x": 900, "y": 950, "w": 100, "h": 50},
            "level": {"x": 150, "y": 950, "w": 80, "h": 40},
            "stage": {"x": 960, "y": 10, "w": 120, "h": 40}
        },
        "window_title": "Teamfight Tactics"
    },
    "ai": {
        "confidence_threshold": 0.8,
        "max_suggestions": 5,
        "update_interval": 0.5,
        "economy_weight": 1.0,
        "composition_weight": 1.0,
        "positioning_weight": 0.8
    },
    "overlay": {
        "port": 8765,
        "host": "localhost",
        "transparent": True,
        "always_on_top": True,
        "width": 400,
        "height": 600,
        "position": {"x": 10, "y": 10}
    },
    "vision": {
        "template_threshold": 0.8,
        "ocr_confidence": 0.7,
        "preprocessing": {
            "denoise": True,
            "sharpen": True,
            "contrast_enhance": True
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "max_files": 5,
        "max_size_mb": 10
    },
    "performance": {
        "max_fps": 60,
        "adaptive_quality": True,
        "low_power_mode": False
    }
}

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file with fallback to defaults"""
    config_file = Path(config_path)
    
    try:
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Merge with defaults
            config = merge_config(DEFAULT_CONFIG, user_config)
            logger.info(f"Configuration loaded from {config_path}")
            
            # Validate configuration
            validate_config(config)
            
            return config
        else:
            logger.warning(f"Config file {config_path} not found, creating default")
            save_config(DEFAULT_CONFIG, config_path)
            return DEFAULT_CONFIG.copy()
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: str = "config.json"):
    """Save configuration to file"""
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")

def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    
    return result

def validate_config(config: Dict[str, Any]):
    """Validate configuration values"""
    errors = []
    
    # Validate capture settings
    capture = config.get('capture', {})
    if capture.get('fps', 0) <= 0:
        errors.append("Capture FPS must be positive")
    
    # Validate regions
    regions = capture.get('regions', {})
    required_regions = ['board', 'shop', 'gold', 'health', 'level', 'stage']
    
    for region_name in required_regions:
        if region_name not in regions:
            errors.append(f"Missing required region: {region_name}")
            continue
            
        region = regions[region_name]
        if not all(k in region for k in ['x', 'y', 'w', 'h']):
            errors.append(f"Region '{region_name}' missing required coordinates")
        if any(region.get(k, 0) < 0 for k in ['x', 'y', 'w', 'h']):
            errors.append(f"Region '{region_name}' has negative coordinates")
    
    # Validate AI settings
    ai = config.get('ai', {})
    threshold = ai.get('confidence_threshold', 0)
    if not 0 <= threshold <= 1:
        errors.append("AI confidence threshold must be between 0 and 1")
    
    if ai.get('max_suggestions', 0) <= 0:
        errors.append("Max suggestions must be positive")
    
    if ai.get('update_interval', 0) <= 0:
        errors.append("Update interval must be positive")
    
    # Validate overlay settings
    overlay = config.get('overlay', {})
    port = overlay.get('port', 0)
    if not 1024 <= port <= 65535:
        errors.append("Overlay port must be between 1024 and 65535")
    
    # Validate vision settings
    vision = config.get('vision', {})
    template_threshold = vision.get('template_threshold', 0)
    if not 0 <= template_threshold <= 1:
        errors.append("Template threshold must be between 0 and 1")
    
    ocr_confidence = vision.get('ocr_confidence', 0)
    if not 0 <= ocr_confidence <= 1:
        errors.append("OCR confidence must be between 0 and 1")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")

def get_data_path() -> Path:
    """Get the data directory path"""
    return Path(__file__).parent.parent / "data"

def get_template_path() -> Path:
    """Get the template directory path"""
    return get_data_path() / "templates"

def get_model_path() -> Path:
    """Get the model directory path"""
    return get_data_path() / "models"

def get_log_path() -> Path:
    """Get the log directory path"""
    return Path(__file__).parent.parent / "logs"

def get_cache_path() -> Path:
    """Get the cache directory path"""
    return Path(__file__).parent.parent / "cache"

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        get_data_path(),
        get_template_path(),
        get_model_path(),
        get_log_path(),
        get_cache_path(),
        get_cache_path() / "templates",
        get_data_path() / "champions",
        get_data_path() / "items",
        get_data_path() / "traits",
        get_data_path() / "compositions"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """Get nested configuration value using dot notation"""
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def set_config_value(config: Dict[str, Any], key_path: str, value: Any):
    """Set nested configuration value using dot notation"""
    keys = key_path.split('.')
    current = config
    
    # Navigate to parent of target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value