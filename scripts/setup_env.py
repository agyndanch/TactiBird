#!/usr/bin/env python3
"""
TFT AI Coaching Overlay - Environment Setup Script
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/champions",
        "data/items", 
        "data/traits",
        "data/compositions",
        "data/templates/champions",
        "data/templates/items",
        "data/templates/traits",
        "data/templates/ui",
        "data/models",
        "logs",
        "cache",
        "cache/templates",
        "config",
        "assets/icons",
        "assets/sounds",
        "assets/images",
        "tests/test_data/screenshots",
        "tests/test_data/game_states"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_python_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Python requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install Python requirements: {e}")
        return False

def check_node_js():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Node.js version: {version}")
            return True
        else:
            print("ERROR: Node.js not found")
            return False
    except FileNotFoundError:
        print("ERROR: Node.js not installed")
        print("Please install Node.js from https://nodejs.org/")
        return False

def install_node_requirements():
    """Install Node.js requirements"""
    if not check_node_js():
        return False
    
    print("Installing Node.js requirements...")
    
    try:
        subprocess.check_call(["npm", "install"])
        print("✓ Node.js requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install Node.js requirements: {e}")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ Tesseract OCR: {version_line}")
            return True
        else:
            print("WARNING: Tesseract OCR not found")
            return False
    except FileNotFoundError:
        print("WARNING: Tesseract OCR not installed")
        print_tesseract_install_instructions()
        return False

def print_tesseract_install_instructions():
    """Print Tesseract installation instructions"""
    system = platform.system().lower()
    
    print("\nTesseract OCR Installation Instructions:")
    if system == "windows":
        print("Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki")
        print("Or use: winget install UB-Mannheim.TesseractOCR")
    elif system == "darwin":  # macOS
        print("macOS: brew install tesseract")
    elif system == "linux":
        print("Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("Fedora: sudo dnf install tesseract")
        print("Arch: sudo pacman -S tesseract")
    
    print("Add Tesseract to your system PATH after installation.")

def create_default_config():
    """Create default configuration file"""
    config_path = Path("config.json")
    
    if config_path.exists():
        print("✓ Configuration file already exists")
        return
    
    default_config = {
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
    
    import json
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print("✓ Created default configuration file")

def create_placeholder_files():
    """Create placeholder files for development"""
    placeholders = [
        ("src/__init__.py", "# TFT AI Coaching Overlay"),
        ("src/capture/__init__.py", "# Screen capture module"),
        ("src/vision/__init__.py", "# Computer vision module"),
        ("src/ai/__init__.py", "# AI coaching module"),
        ("src/data/__init__.py", "# Data management module"),
        ("src/overlay/__init__.py", "# Overlay module"),
        ("src/utils/__init__.py", "# Utilities module"),
        ("tests/__init__.py", "# Tests"),
        ("README.md", "# TFT AI Coaching Overlay\n\nAI-powered coaching for Teamfight Tactics"),
        ("LICENSE", "MIT License\n\n[Add full license text here]"),
        (".env.example", "# Environment variables\nDEBUG=false\nLOG_LEVEL=INFO")
    ]
    
    for file_path, content in placeholders:
        path = Path(file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            print(f"✓ Created placeholder: {file_path}")

def main():
    """Main setup function"""
    print("TFT AI Coaching Overlay - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Create placeholder files
    print("\nCreating placeholder files...")
    create_placeholder_files()
    
    # Create default configuration
    print("\nSetting up configuration...")
    create_default_config()
    
    # Install Python requirements
    print("\nInstalling Python dependencies...")
    if not install_python_requirements():
        print("WARNING: Some Python packages may have failed to install")
    
    # Install Node.js requirements
    print("\nInstalling Node.js dependencies...")
    if not install_node_requirements():
        print("WARNING: Node.js dependencies not installed")
    
    # Check Tesseract
    print("\nChecking Tesseract OCR...")
    check_tesseract()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Install Tesseract OCR if not already installed")
    print("2. Configure screen regions in config.json for your resolution")
    print("3. Download TFT champion/item templates to data/templates/")
    print("4. Run: python main.py")
    print("\nFor Electron overlay:")
    print("5. Run: npm start")

if __name__ == "__main__":
    main()