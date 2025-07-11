"""
TactiBird - File Utilities Module
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import json
import pickle
import time

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> bool:
        """Ensure directory exists, create if it doesn't"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """Check if file exists"""
        return Path(file_path).exists()
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
        """Get file size in bytes"""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"Failed to get size for {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_modification_time(file_path: Union[str, Path]) -> Optional[float]:
        """Get file modification time as timestamp"""
        try:
            return Path(file_path).stat().st_mtime
        except Exception as e:
            logger.error(f"Failed to get modification time for {file_path}: {e}")
            return None
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path], 
                 preserve_metadata: bool = True) -> bool:
        """Copy file with optional metadata preservation"""
        try:
            dst_path = Path(dst)
            FileUtils.ensure_directory(dst_path.parent)
            
            if preserve_metadata:
                shutil.copy2(src, dst)
            else:
                shutil.copy(src, dst)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy {src} to {dst}: {e}")
            return False
    
    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Move file"""
        try:
            dst_path = Path(dst)
            FileUtils.ensure_directory(dst_path.parent)
            
            shutil.move(src, dst)
            return True
            
        except Exception as e:
            logger.error(f"Failed to move {src} to {dst}: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> bool:
        """Delete file safely"""
        try:
            Path(file_path).unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False
    
    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", 
                  recursive: bool = True) -> List[Path]:
        """Find files matching pattern"""
        try:
            directory = Path(directory)
            
            if recursive:
                return list(directory.rglob(pattern))
            else:
                return list(directory.glob(pattern))
                
        except Exception as e:
            logger.error(f"Failed to find files in {directory}: {e}")
            return []
    
    @staticmethod
    def read_text_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
        """Read text file content"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_text_file(file_path: Union[str, Path], content: str, 
                       encoding: str = 'utf-8') -> bool:
        """Write text content to file"""
        try:
            file_path = Path(file_path)
            FileUtils.ensure_directory(file_path.parent)
            
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write text file {file_path}: {e}")
            return False
    
    @staticmethod
    def read_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_json_file(file_path: Union[str, Path], data: Dict[str, Any], 
                       indent: int = 2) -> bool:
        """Write data to JSON file"""
        try:
            file_path = Path(file_path)
            FileUtils.ensure_directory(file_path.parent)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON file {file_path}: {e}")
            return False
    
    @staticmethod
    def read_pickle_file(file_path: Union[str, Path]) -> Optional[Any]:
        """Read pickle file"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to read pickle file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_pickle_file(file_path: Union[str, Path], data: Any) -> bool:
        """Write data to pickle file"""
        try:
            file_path = Path(file_path)
            FileUtils.ensure_directory(file_path.parent)
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to write pickle file {file_path}: {e}")
            return False
    
    @staticmethod
    def get_directory_size(directory: Union[str, Path]) -> int:
        """Get total size of directory in bytes"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        continue
            return total_size
        except Exception as e:
            logger.error(f"Failed to get directory size for {directory}: {e}")
            return 0
    
    @staticmethod
    def clean_directory(directory: Union[str, Path], max_age_days: int = 7) -> int:
        """Clean old files from directory"""
        try:
            directory = Path(directory)
            if not directory.exists():
                return 0
            
            current_time = time.time()
            cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
            deleted_count = 0
            
            for file_path in directory.iterdir():
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            deleted_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to delete old file {file_path}: {e}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clean directory {directory}: {e}")
            return 0
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Union[str, Path] = None) -> Optional[Path]:
        """Create backup of file with timestamp"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return None
            
            if backup_dir is None:
                backup_dir = file_path.parent / "backups"
            
            backup_dir = Path(backup_dir)
            FileUtils.ensure_directory(backup_dir)
            
            # Create timestamped backup filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name
            
            if FileUtils.copy_file(file_path, backup_path):
                return backup_path
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            return None
    
    @staticmethod
    def compress_directory(directory: Union[str, Path], output_path: Union[str, Path] = None) -> Optional[Path]:
        """Compress directory to zip file"""
        try:
            import zipfile
            
            directory = Path(directory)
            if not directory.exists():
                return None
            
            if output_path is None:
                output_path = directory.parent / f"{directory.name}.zip"
            
            output_path = Path(output_path)
            FileUtils.ensure_directory(output_path.parent)
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in directory.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(directory)
                        zipf.write(file_path, arcname)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to compress directory {directory}: {e}")
            return None
    
    @staticmethod
    def extract_zip(zip_path: Union[str, Path], extract_to: Union[str, Path] = None) -> bool:
        """Extract zip file"""
        try:
            import zipfile
            
            zip_path = Path(zip_path)
            if not zip_path.exists():
                return False
            
            if extract_to is None:
                extract_to = zip_path.parent / zip_path.stem
            
            extract_to = Path(extract_to)
            FileUtils.ensure_directory(extract_to)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_to)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract zip {zip_path}: {e}")
            return False

class TempFileManager:
    """Manager for temporary files and directories"""
    
    def __init__(self, cleanup_on_exit: bool = True):
        self.cleanup_on_exit = cleanup_on_exit
        self.temp_files = []
        self.temp_dirs = []
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'tactibird_', 
                        content: str = None) -> Path:
        """Create temporary file"""
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            temp_path = Path(temp_path)
            
            if content is not None:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                os.close(fd)  # Close file descriptor if no content
            
            self.temp_files.append(temp_path)
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            return Path()
    
    def create_temp_dir(self, suffix: str = '', prefix: str = 'tactibird_') -> Path:
        """Create temporary directory"""
        try:
            temp_dir = Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix))
            self.temp_dirs.append(temp_dir)
            return temp_dir
            
        except Exception as e:
            logger.error(f"Failed to create temp directory: {e}")
            return Path()
    
    def cleanup(self):
        """Clean up temporary files and directories"""
        try:
            # Clean up files
            for temp_file in self.temp_files:
                FileUtils.delete_file(temp_file)
            
            # Clean up directories
            for temp_dir in self.temp_dirs:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            self.temp_files.clear()
            self.temp_dirs.clear()
            
        except Exception as e:
            logger.error(f"Temp cleanup failed: {e}")
    
    def __del__(self):
        """Destructor - cleanup if enabled"""
        if self.cleanup_on_exit:
            self.cleanup()

class ConfigManager:
    """Manager for configuration files with automatic backup and validation"""
    
    def __init__(self, config_path: Union[str, Path], backup_count: int = 5):
        self.config_path = Path(config_path)
        self.backup_count = backup_count
        self.backup_dir = self.config_path.parent / "config_backups"
    
    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration with automatic backup"""
        try:
            if not self.config_path.exists():
                return None
            
            # Create backup before loading
            self._create_backup()
            
            return FileUtils.read_json_file(self.config_path)
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._restore_from_backup()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration with validation"""
        try:
            # Validate config before saving
            if not self._validate_config(config):
                return False
            
            # Create backup of current config
            if self.config_path.exists():
                self._create_backup()
            
            # Save new config
            success = FileUtils.write_json_file(self.config_path, config)
            
            if success:
                # Clean old backups
                self._cleanup_old_backups()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """Create timestamped backup of current config"""
        try:
            if not self.config_path.exists():
                return True
            
            FileUtils.ensure_directory(self.backup_dir)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_{timestamp}.json"
            backup_path = self.backup_dir / backup_name
            
            return FileUtils.copy_file(self.config_path, backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create config backup: {e}")
            return False
    
    def _restore_from_backup(self) -> Optional[Dict[str, Any]]:
        """Restore config from most recent backup"""
        try:
            if not self.backup_dir.exists():
                return None
            
            # Find most recent backup
            backups = list(self.backup_dir.glob("config_*.json"))
            if not backups:
                return None
            
            latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
            config = FileUtils.read_json_file(latest_backup)
            
            if config and self._validate_config(config):
                logger.info(f"Restored config from backup: {latest_backup}")
                return config
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return None
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        try:
            # Basic validation - ensure required sections exist
            required_sections = ['capture', 'ai', 'overlay', 'vision']
            
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required config section: {section}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only the most recent ones"""
        try:
            if not self.backup_dir.exists():
                return
            
            backups = list(self.backup_dir.glob("config_*.json"))
            if len(backups) <= self.backup_count:
                return
            
            # Sort by modification time and remove oldest
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for old_backup in backups[self.backup_count:]:
                FileUtils.delete_file(old_backup)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")

# Global temp file manager instance
temp_manager = TempFileManager()