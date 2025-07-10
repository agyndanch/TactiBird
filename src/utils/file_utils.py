"""
TactiBird Overlay - File Utilities
"""

import os
import json
import csv
import pickle
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import zipfile
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> bool:
        """Ensure directory exists, create if necessary"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    @staticmethod
    def safe_read_json(file_path: Union[str, Path], default: Any = None) -> Any:
        """Safely read JSON file with fallback"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {file_path}")
            return default
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return default
        except Exception as e:
            logger.error(f"Failed to read JSON {file_path}: {e}")
            return default
    
    @staticmethod
    def safe_write_json(file_path: Union[str, Path], data: Any, indent: int = 2, 
                       backup: bool = True) -> bool:
        """Safely write JSON file with backup"""
        try:
            file_path = Path(file_path)
            
            # Create backup if file exists and backup is requested
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                shutil.copy2(file_path, backup_path)
            
            # Ensure parent directory exists
            FileUtils.ensure_directory(file_path.parent)
            
            # Write to temporary file first
            temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            # Atomic move
            temp_path.replace(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write JSON {file_path}: {e}")
            return False
    
    @staticmethod
    def safe_read_csv(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[Dict[str, str]]:
        """Safely read CSV file"""
        try:
            data = []
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            return data
        except Exception as e:
            logger.error(f"Failed to read CSV {file_path}: {e}")
            return []
    
    @staticmethod
    def safe_write_csv(file_path: Union[str, Path], data: List[Dict[str, Any]], 
                      fieldnames: Optional[List[str]] = None) -> bool:
        """Safely write CSV file"""
        try:
            if not data:
                return True
            
            file_path = Path(file_path)
            FileUtils.ensure_directory(file_path.parent)
            
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write CSV {file_path}: {e}")
            return False
    
    @staticmethod
    def safe_read_pickle(file_path: Union[str, Path], default: Any = None) -> Any:
        """Safely read pickle file"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to read pickle {file_path}: {e}")
            return default
    
    @staticmethod
    def safe_write_pickle(file_path: Union[str, Path], data: Any) -> bool:
        """Safely write pickle file"""
        try:
            file_path = Path(file_path)
            FileUtils.ensure_directory(file_path.parent)
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write pickle {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
        """Calculate file hash"""
        try:
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """Get file size in bytes"""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"Failed to get size for {file_path}: {e}")
            return 0
    
    @staticmethod
    def get_file_modified_time(file_path: Union[str, Path]) -> Optional[datetime]:
        """Get file modification time"""
        try:
            timestamp = Path(file_path).stat().st_mtime
            return datetime.fromtimestamp(timestamp)
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
    def clean_directory(directory: Union[str, Path], pattern: str = "*", 
                       max_age_days: Optional[int] = None) -> int:
        """Clean directory by removing old files"""
        try:
            directory = Path(directory)
            
            if not directory.exists():
                return 0
            
            files = list(directory.glob(pattern))
            deleted_count = 0
            
            for file_path in files:
                if file_path.is_file():
                    should_delete = True
                    
                    if max_age_days is not None:
                        mod_time = FileUtils.get_file_modified_time(file_path)
                        if mod_time:
                            age = (datetime.now() - mod_time).days
                            should_delete = age > max_age_days
                    
                    if should_delete:
                        if FileUtils.delete_file(file_path):
                            deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clean directory {directory}: {e}")
            return 0
    
    @staticmethod
    def archive_directory(directory: Union[str, Path], archive_path: Union[str, Path], 
                         compression: str = 'zip') -> bool:
        """Create archive from directory"""
        try:
            directory = Path(directory)
            archive_path = Path(archive_path)
            
            FileUtils.ensure_directory(archive_path.parent)
            
            if compression == 'zip':
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in directory.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(directory)
                            zipf.write(file_path, arcname)
            else:
                # Use shutil for other formats
                shutil.make_archive(str(archive_path.with_suffix('')), compression, directory)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive {directory}: {e}")
            return False
    
    @staticmethod
    def extract_archive(archive_path: Union[str, Path], extract_to: Union[str, Path]) -> bool:
        """Extract archive to directory"""
        try:
            archive_path = Path(archive_path)
            extract_to = Path(extract_to)
            
            FileUtils.ensure_directory(extract_to)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zipf:
                    zipf.extractall(extract_to)
            else:
                shutil.unpack_archive(archive_path, extract_to)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False

class ConfigManager:
    """Configuration file manager"""
    
    def __init__(self, config_file: Union[str, Path]):
        self.config_file = Path(config_file)
        self.config_data = {}
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            self.config_data = FileUtils.safe_read_json(self.config_file, {})
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            return FileUtils.safe_write_json(self.config_file, self.config_data)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            keys = key.split('.')
            data = self.config_data
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in data:
                    data[k] = {}
                data = data[k]
            
            # Set value
            data[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """Update multiple configuration values"""
        try:
            for key, value in updates.items():
                self.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self.get(key) is not None
    
    def remove(self, key: str) -> bool:
        """Remove configuration key"""
        try:
            keys = key.split('.')
            data = self.config_data
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in data:
                    return False
                data = data[k]
            
            # Remove key
            if keys[-1] in data:
                del data[keys[-1]]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove config {key}: {e}")
            return False
    
    def reset_to_defaults(self, defaults: Dict[str, Any]) -> bool:
        """Reset configuration to defaults"""
        try:
            self.config_data = defaults.copy()
            return self.save_config()
        except Exception as e:
            logger.error(f"Failed to reset config: {e}")
            return False
    
    def backup_config(self, backup_path: Optional[Union[str, Path]] = None) -> bool:
        """Create backup of current configuration"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.config_file.with_suffix(f".backup_{timestamp}.json")
            
            return FileUtils.copy_file(self.config_file, backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup config: {e}")
            return False
    
    def restore_config(self, backup_path: Union[str, Path]) -> bool:
        """Restore configuration from backup"""
        try:
            if FileUtils.copy_file(backup_path, self.config_file):
                return self.load_config()
            return False
            
        except Exception as e:
            logger.error(f"Failed to restore config: {e}")
            return False

class DataExporter:
    """Export data to various formats"""
    
    @staticmethod
    def export_to_json(data: Any, output_path: Union[str, Path], 
                      pretty: bool = True) -> bool:
        """Export data to JSON format"""
        try:
            indent = 2 if pretty else None
            return FileUtils.safe_write_json(output_path, data, indent=indent)
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], output_path: Union[str, Path],
                     fieldnames: Optional[List[str]] = None) -> bool:
        """Export data to CSV format"""
        try:
            return FileUtils.safe_write_csv(output_path, data, fieldnames)
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    @staticmethod
    def export_game_session(session_data: Dict[str, Any], 
                           output_dir: Union[str, Path]) -> bool:
        """Export complete game session data"""
        try:
            output_dir = Path(output_dir)
            FileUtils.ensure_directory(output_dir)
            
            # Export main session data
            session_file = output_dir / "session.json"
            if not DataExporter.export_to_json(session_data, session_file):
                return False
            
            # Export game states if available
            if 'game_states' in session_data:
                states_file = output_dir / "game_states.json"
                DataExporter.export_to_json(session_data['game_states'], states_file)
            
            # Export suggestions if available
            if 'suggestions' in session_data:
                suggestions_file = output_dir / "suggestions.csv"
                suggestions_data = []
                
                for suggestion in session_data['suggestions']:
                    suggestions_data.append({
                        'timestamp': suggestion.get('timestamp', ''),
                        'type': suggestion.get('type', ''),
                        'message': suggestion.get('message', ''),
                        'priority': suggestion.get('priority', 0),
                        'context': str(suggestion.get('context', {}))
                    })
                
                DataExporter.export_to_csv(suggestions_data, suggestions_file)
            
            # Create summary
            summary = {
                'export_timestamp': datetime.now().isoformat(),
                'session_duration': session_data.get('duration', 0),
                'total_suggestions': len(session_data.get('suggestions', [])),
                'game_states_count': len(session_data.get('game_states', [])),
                'files_exported': ['session.json']
            }
            
            if 'game_states' in session_data:
                summary['files_exported'].append('game_states.json')
            if 'suggestions' in session_data:
                summary['files_exported'].append('suggestions.csv')
            
            summary_file = output_dir / "export_summary.json"
            DataExporter.export_to_json(summary, summary_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Game session export failed: {e}")
            return False

class LogManager:
    """Log file management utilities"""
    
    @staticmethod
    def rotate_logs(log_file: Union[str, Path], max_size_mb: int = 10, 
                   backup_count: int = 5) -> bool:
        """Rotate log files when they get too large"""
        try:
            log_file = Path(log_file)
            
            if not log_file.exists():
                return True
            
            # Check file size
            size_mb = FileUtils.get_file_size(log_file) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                # Rotate existing backups
                for i in range(backup_count - 1, 0, -1):
                    old_backup = log_file.with_suffix(f"{log_file.suffix}.{i}")
                    new_backup = log_file.with_suffix(f"{log_file.suffix}.{i + 1}")
                    
                    if old_backup.exists():
                        if new_backup.exists():
                            FileUtils.delete_file(new_backup)
                        FileUtils.move_file(old_backup, new_backup)
                
                # Move current log to .1
                backup_file = log_file.with_suffix(f"{log_file.suffix}.1")
                if backup_file.exists():
                    FileUtils.delete_file(backup_file)
                
                FileUtils.move_file(log_file, backup_file)
                
                # Create new empty log file
                log_file.touch()
            
            return True
            
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")
            return False
    
    @staticmethod
    def clean_old_logs(log_directory: Union[str, Path], 
                      max_age_days: int = 30) -> int:
        """Clean old log files"""
        try:
            return FileUtils.clean_directory(log_directory, "*.log*", max_age_days)
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
            return 0
    
    @staticmethod
    def archive_logs(log_directory: Union[str, Path], 
                    archive_path: Union[str, Path]) -> bool:
        """Archive log files"""
        try:
            return FileUtils.archive_directory(log_directory, archive_path)
        except Exception as e:
            logger.error(f"Log archiving failed: {e}")
            return False

class TempFileManager:
    """Temporary file management"""
    
    def __init__(self, cleanup_on_exit: bool = True):
        self.temp_files = []
        self.temp_dirs = []
        self.cleanup_on_exit = cleanup_on_exit
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'tactibird_', 
                        delete: bool = False) -> Path:
        """Create temporary file"""
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=suffix, prefix=prefix, delete=delete
            )
            
            temp_path = Path(temp_file.name)
            
            if not delete:
                temp_file.close()
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

# Global temp file manager instance
temp_manager = TempFileManager()