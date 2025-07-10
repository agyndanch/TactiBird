#!/usr/bin/env python3
"""
TactiBird Overlay - Build Script
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import argparse
import platform

class TactiBirdBuilder:
    """Build system for TactiBird overlay"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.src_dir = self.root_dir / "src"
        self.build_dir = self.root_dir / "build"
        self.dist_dir = self.root_dir / "dist"
        self.platform = platform.system().lower()
        
    def clean(self):
        """Clean build artifacts"""
        print("🧹 Cleaning build artifacts...")
        
        # Remove build directories
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            print(f"   Removed {self.build_dir}")
        
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
            print(f"   Removed {self.dist_dir}")
        
        # Remove Python cache
        for cache_dir in self.root_dir.rglob("__pycache__"):
            shutil.rmtree(cache_dir)
            print(f"   Removed {cache_dir}")
        
        # Remove .pyc files
        for pyc_file in self.root_dir.rglob("*.pyc"):
            pyc_file.unlink()
        
        print("✅ Clean complete")
    
    def setup_environment(self):
        """Setup build environment"""
        print("🔧 Setting up build environment...")
        
        # Create build directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        (self.dist_dir / "installers").mkdir(exist_ok=True)
        
        print("✅ Environment setup complete")
    
    def install_dependencies(self):
        """Install Python and Node.js dependencies"""
        print("📦 Installing dependencies...")
        
        # Install Python dependencies
        print("   Installing Python dependencies...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Python dependencies failed: {result.stderr}")
            return False
        
        # Install Node.js dependencies
        if (self.root_dir / "package.json").exists():
            print("   Installing Node.js dependencies...")
            result = subprocess.run(["npm", "install"], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Node.js dependencies failed: {result.stderr}")
                return False
        
        print("✅ Dependencies installed")
        return True
    
    def build_python(self):
        """Build Python components"""
        print("🐍 Building Python components...")
        
        # Copy source files
        build_src = self.build_dir / "src"
        if build_src.exists():
            shutil.rmtree(build_src)
        
        shutil.copytree(self.src_dir, build_src)
        
        # Copy data files
        data_dir = self.root_dir / "data"
        if data_dir.exists():
            shutil.copytree(data_dir, self.build_dir / "data")
        
        # Copy config files
        config_files = ["config.json", "requirements.txt"]
        for config_file in config_files:
            config_path = self.root_dir / config_file
            if config_path.exists():
                shutil.copy2(config_path, self.build_dir)
        
        print("✅ Python build complete")
        return True
    
    def build_electron(self):
        """Build Electron application"""
        print("⚡ Building Electron application...")
        
        if not (self.root_dir / "package.json").exists():
            print("   No package.json found, skipping Electron build")
            return True
        
        # Build for current platform
        platform_flag = {
            "windows": "--win",
            "darwin": "--mac",
            "linux": "--linux"
        }.get(self.platform, "--linux")
        
        result = subprocess.run([
            "npm", "run", "build", platform_flag
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Electron build failed: {result.stderr}")
            return False
        
        print("✅ Electron build complete")
        return True
    
    def build_installer(self):
        """Build platform-specific installer"""
        print("📦 Building installer...")
        
        if self.platform == "windows":
            return self._build_windows_installer()
        elif self.platform == "darwin":
            return self._build_macos_installer()
        elif self.platform == "linux":
            return self._build_linux_installer()
        else:
            print(f"   Installer not supported for {self.platform}")
            return True
    
    def _build_windows_installer(self):
        """Build Windows installer"""
        print("   Building Windows installer...")
        
        # Use Electron Builder (already configured in package.json)
        result = subprocess.run([
            "npm", "run", "build-win"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Windows installer build failed: {result.stderr}")
            return False
        
        print("✅ Windows installer built")
        return True
    
    def _build_macos_installer(self):
        """Build macOS installer"""
        print("   Building macOS installer...")
        
        result = subprocess.run([
            "npm", "run", "build-mac"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ macOS installer build failed: {result.stderr}")
            return False
        
        print("✅ macOS installer built")
        return True
    
    def _build_linux_installer(self):
        """Build Linux installer"""
        print("   Building Linux installer...")
        
        result = subprocess.run([
            "npm", "run", "build-linux"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Linux installer build failed: {result.stderr}")
            return False
        
        print("✅ Linux installer built")
        return True
    
    def run_tests(self):
        """Run test suite"""
        print("🧪 Running tests...")
        
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Tests failed: {result.stderr}")
            return False
        
        print("✅ Tests passed")
        return True
    
    def lint_code(self):
        """Run code linting"""
        print("🔍 Linting code...")
        
        # Python linting with flake8
        print("   Linting Python code...")
        result = subprocess.run([
            sys.executable, "-m", "flake8", "src/", "tests/"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️  Python linting issues: {result.stdout}")
        
        # JavaScript linting with ESLint
        if (self.root_dir / "package.json").exists():
            print("   Linting JavaScript code...")
            result = subprocess.run([
                "npm", "run", "lint"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠️  JavaScript linting issues: {result.stdout}")
        
        print("✅ Linting complete")
        return True
    
    def create_portable_version(self):
        """Create portable version"""
        print("📦 Creating portable version...")
        
        portable_dir = self.dist_dir / "portable"
        portable_dir.mkdir(exist_ok=True)
        
        # Copy built application
        if (self.build_dir / "src").exists():
            shutil.copytree(self.build_dir / "src", portable_dir / "src")
        
        if (self.build_dir / "data").exists():
            shutil.copytree(self.build_dir / "data", portable_dir / "data")
        
        # Copy config and requirements
        for file_name in ["config.json", "requirements.txt"]:
            src_file = self.build_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, portable_dir)
        
        # Create startup script
        if self.platform == "windows":
            startup_script = portable_dir / "start.bat"
            startup_script.write_text(
                "@echo off\n"
                "python src/main.py\n"
                "pause\n"
            )
        else:
            startup_script = portable_dir / "start.sh"
            startup_script.write_text(
                "#!/bin/bash\n"
                "python3 src/main.py\n"
            )
            startup_script.chmod(0o755)
        
        # Create README
        readme = portable_dir / "README.txt"
        readme.write_text(
            "TactiBird Portable\n"
            "==================\n\n"
            "1. Install Python 3.8+ and required dependencies:\n"
            "   pip install -r requirements.txt\n\n"
            "2. Run the application:\n"
            f"   {'start.bat' if self.platform == 'windows' else './start.sh'}\n\n"
            "3. Configure the overlay by editing config.json\n\n"
            "For more information, visit: https://github.com/agyndanch/TactiBird\n"
        )
        
        print("✅ Portable version created")
        return True
    
    def generate_documentation(self):
        """Generate documentation"""
        print("📚 Generating documentation...")
        
        docs_dir = self.build_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Copy existing docs
        source_docs = self.root_dir / "docs"
        if source_docs.exists():
            for doc_file in source_docs.glob("*.md"):
                shutil.copy2(doc_file, docs_dir)
        
        # Generate API docs (if sphinx is available)
        try:
            result = subprocess.run([
                sys.executable, "-m", "sphinx.cmd.build",
                "-b", "html",
                "docs/",
                str(docs_dir / "html")
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   Sphinx documentation generated")
        except FileNotFoundError:
            print("   Sphinx not available, skipping API docs")
        
        print("✅ Documentation generated")
        return True
    
    def package_release(self, version):
        """Package release files"""
        print(f"📦 Packaging release v{version}...")
        
        release_dir = self.dist_dir / f"tactibird-v{version}"
        release_dir.mkdir(exist_ok=True)
        
        # Copy installers
        installers_dir = self.dist_dir / "installers"
        if installers_dir.exists():
            for installer in installers_dir.glob("*"):
                shutil.copy2(installer, release_dir)
        
        # Copy portable version
        portable_dir = self.dist_dir / "portable"
        if portable_dir.exists():
            portable_archive = release_dir / f"tactibird-portable-v{version}"
            shutil.make_archive(str(portable_archive), 'zip', portable_dir)
        
        # Copy documentation
        docs_dir = self.build_dir / "docs"
        if docs_dir.exists():
            docs_archive = release_dir / f"tactibird-docs-v{version}"
            shutil.make_archive(str(docs_archive), 'zip', docs_dir)
        
        # Create release notes
        release_notes = release_dir / "RELEASE_NOTES.txt"
        release_notes.write_text(
            f"TactiBird v{version}\n"
            "================\n\n"
            "Release files:\n"
            "- Installer for your platform\n"
            "- Portable version (ZIP)\n"
            "- Documentation (ZIP)\n\n"
            "Installation instructions and changelog available at:\n"
            "https://github.com/agyndanch/TactiBird\n"
        )
        
        print(f"✅ Release v{version} packaged")
        return True
    
    def full_build(self, version=None):
        """Run full build process"""
        print("🚀 Starting full build process...")
        
        steps = [
            ("Clean", self.clean),
            ("Setup Environment", self.setup_environment),
            ("Install Dependencies", self.install_dependencies),
            ("Run Tests", self.run_tests),
            ("Lint Code", self.lint_code),
            ("Build Python", self.build_python),
            ("Build Electron", self.build_electron),
            ("Build Installer", self.build_installer),
            ("Create Portable", self.create_portable_version),
            ("Generate Docs", self.generate_documentation),
        ]
        
        if version:
            steps.append(("Package Release", lambda: self.package_release(version)))
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ Build failed at step: {step_name}")
                return False
        
        print("\n🎉 Build completed successfully!")
        print(f"   Build artifacts: {self.build_dir}")
        print(f"   Distribution files: {self.dist_dir}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="TactiBird Build Script")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--build", action="store_true", help="Build application")
    parser.add_argument("--full", action="store_true", help="Full build process")
    parser.add_argument("--version", help="Version for release packaging")
    parser.add_argument("--portable", action="store_true", help="Create portable version")
    
    args = parser.parse_args()
    
    builder = TactiBirdBuilder()
    
    if args.clean:
        return builder.clean()
    elif args.test:
        return builder.run_tests()
    elif args.lint:
        return builder.lint_code()
    elif args.portable:
        builder.setup_environment()
        builder.build_python()
        return builder.create_portable_version()
    elif args.build:
        builder.setup_environment()
        builder.build_python()
        return builder.build_electron()
    elif args.full:
        return builder.full_build(args.version)
    else:
        # Default: build application
        builder.setup_environment()
        builder.build_python()
        return builder.build_electron()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)