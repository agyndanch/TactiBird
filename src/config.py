"""
TactiBird Overlay - Configuration Management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "capture": {
        "fps": 30,
        "regions": {
            "board": {"x": 100, "y": 200, "w": 800, "h": 600},
            "shop": {"x": 200, "y": 850, "w": 600, "h": 150},
            "gold": {"x": 50, "y": 950, "w": 100, "h": 50},
            "health": {"x": 50, "y": 900, "w": 100, "h": 50},
            "level": {"x": 50, "y": 850, "w": 100, "h": 50},
            "traits": {"x": 50, "y": 200, "w": 150, "h": 600}
        },
        "monitor": 0,
        "enable_debug": False
    },
    "ai": {
        "confidence_threshold": 0.8,
        "max_suggestions": 5,
        "update_interval": 0.5,
        "enable_ml_models": True,
        "model_path": "data/models/",
        "coaches": {
            "economy": {"enabled": True, "weight": 1.0},
            "composition": {"enabled": True, "weight": 1.0},
            "positioning": {"enabled": True, "weight": 0.8},
            "items": {"enabled": True, "weight": 0.9}
        }
    },
    "overlay": {
        "port": 8765,
        "transparent": True,
        "always_on_top": True,
        "position": {"x": 0, "y": 0},
        "size": {"width": 1920, "height": 1080},
        "theme": "dark",
        "opacity": 0.9
    },
    "vision": {
        "ocr_engine": "tesseract",
        "template_threshold": 0.8,
        "preprocessing": {
            "enable_denoising": True,
            "enable_sharpening": True,
            "contrast_adjustment": 1.2
        }
    },
    "data": {
        "auto_update": True,
        "cache_size": 100,
        "api_timeout": 30,
        "use_local_cache": True
    },
    "logging": {
        "level": "INFO",
        "file": "logs/app.log",
        "max_size": "10MB",
        "backup_count": 5
    }
}

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config = merge_config(config, file_config)
                logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Config file {config_path} not found, using defaults")
        # Create default config file
        save_config(config, config_path)
    
    # Validate configuration
    validate_config(config)
    
    return config

def save_config(config: Dict[str, Any], config_path: str = "config.json"):
    """Save configuration to file"""
    try:
        # Create directories if they don't exist
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
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
    for region_name, region in regions.items():
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

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        get_data_path(),
        get_template_path(),
        get_model_path(),
        get_log_path(),
        Path("cache"),
        Path("cache/templates")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")