"""
TactiBird Overlay - Template Manager
"""

import cv2
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import pickle

logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages template loading, caching, and updates"""
    
    def __init__(self):
        self.template_cache = {}
        self.template_metadata = {}
        self.cache_file = Path("cache/template_cache.pkl")
        self.metadata_file = Path("cache/template_metadata.json")
        
        # Template directories
        self.template_dirs = {
            'champions': Path("data/templates/champions"),
            'items': Path("data/templates/items"),
            'traits': Path("data/templates/traits"),
            'ui': Path("data/templates/ui")
        }
        
        # Supported image formats
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        
        # Initialize
        self._ensure_directories()
        self._load_cache()
        
        logger.info("Template manager initialized")
    
    def _ensure_directories(self):
        """Ensure all template directories exist"""
        for category, directory in self.template_dirs.items():
            directory.mkdir(parents=True, exist_ok=True)
        
        # Ensure cache directory exists
        Path("cache").mkdir(parents=True, exist_ok=True)
    
    def load_all_templates(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load all templates from directories"""
        try:
            all_templates = {}
            
            for category, directory in self.template_dirs.items():
                templates = self.load_templates_from_directory(directory, category)
                all_templates[category] = templates
                
                logger.info(f"Loaded {len(templates)} {category} templates")
            
            return all_templates
            
        except Exception as e:
            logger.error(f"Failed to remove template {name}: {e}")
            return False
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about loaded templates"""
        try:
            info = {
                'total_templates': len(self.template_cache),
                'categories': {},
                'cache_size_mb': 0,
                'last_updated': None
            }
            
            # Count templates by category
            for cache_key in self.template_cache.keys():
                category = cache_key.split('_')[0]
                info['categories'][category] = info['categories'].get(category, 0) + 1
            
            # Calculate cache size (approximate)
            total_pixels = 0
            for template in self.template_cache.values():
                total_pixels += template.size
            info['cache_size_mb'] = (total_pixels * 3) / (1024 * 1024)  # Assuming 3 bytes per pixel
            
            # Get last update time
            if self.template_metadata:
                timestamps = [meta.get('cached_at', '') for meta in self.template_metadata.values()]
                info['last_updated'] = max(timestamps) if timestamps else None
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get template info: {e}")
            return {}
    
    def update_templates(self) -> Dict[str, int]:
        """Update all templates from directories"""
        try:
            update_stats = {
                'updated': 0,
                'added': 0,
                'removed': 0,
                'errors': 0
            }
            
            # Track existing templates
            existing_templates = set(self.template_cache.keys())
            current_templates = set()
            
            # Load all templates
            for category, directory in self.template_dirs.items():
                if not directory.exists():
                    continue
                
                for file_path in directory.iterdir():
                    if file_path.suffix.lower() in self.supported_formats:
                        template_name = file_path.stem
                        cache_key = f"{category}_{template_name}"
                        current_templates.add(cache_key)
                        
                        try:
                            # Check if template needs updating
                            if self._is_template_cached(file_path, category):
                                continue  # Template is up-to-date
                            
                            # Load and process template
                            template = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                            
                            if template is not None:
                                processed_template = self._preprocess_template(template, category)
                                
                                # Update cache
                                if cache_key in self.template_cache:
                                    update_stats['updated'] += 1
                                else:
                                    update_stats['added'] += 1
                                
                                self._cache_template(file_path, category, processed_template)
                            else:
                                update_stats['errors'] += 1
                                
                        except Exception as e:
                            logger.error(f"Failed to update template {template_name}: {e}")
                            update_stats['errors'] += 1
            
            # Remove templates that no longer exist
            removed_templates = existing_templates - current_templates
            for cache_key in removed_templates:
                if cache_key in self.template_cache:
                    del self.template_cache[cache_key]
                if cache_key in self.template_metadata:
                    del self.template_metadata[cache_key]
                update_stats['removed'] += 1
            
            # Save updated cache
            self.save_cache()
            
            logger.info(f"Template update complete: {update_stats}")
            return update_stats
            
        except Exception as e:
            logger.error(f"Template update failed: {e}")
            return {'updated': 0, 'added': 0, 'removed': 0, 'errors': 1}
    
    def validate_templates(self) -> Dict[str, List[str]]:
        """Validate all templates and return issues"""
        try:
            validation_results = {
                'valid': [],
                'invalid': [],
                'missing_files': [],
                'corrupted': []
            }
            
            for cache_key, metadata in self.template_metadata.items():
                template_path = Path(metadata['path'])
                
                # Check if file exists
                if not template_path.exists():
                    validation_results['missing_files'].append(cache_key)
                    continue
                
                # Check if template is valid
                if cache_key not in self.template_cache:
                    validation_results['invalid'].append(cache_key)
                    continue
                
                template = self.template_cache[cache_key]
                
                # Basic validation
                if template is None or template.size == 0:
                    validation_results['corrupted'].append(cache_key)
                    continue
                
                # Check dimensions
                if len(template.shape) != 3 or template.shape[2] != 3:
                    validation_results['invalid'].append(cache_key)
                    continue
                
                validation_results['valid'].append(cache_key)
            
            logger.info(f"Template validation: {len(validation_results['valid'])} valid, "
                       f"{len(validation_results['invalid'])} invalid")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return {'valid': [], 'invalid': [], 'missing_files': [], 'corrupted': []}
    
    def export_templates(self, output_dir: Path, categories: List[str] = None) -> bool:
        """Export templates to directory"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            
            for cache_key, template in self.template_cache.items():
                category = cache_key.split('_')[0]
                
                # Filter by categories if specified
                if categories and category not in categories:
                    continue
                
                template_name = '_'.join(cache_key.split('_')[1:])
                
                # Create category subdirectory
                category_dir = output_dir / category
                category_dir.mkdir(exist_ok=True)
                
                # Export template
                output_path = category_dir / f"{template_name}.png"
                success = cv2.imwrite(str(output_path), template)
                
                if success:
                    exported_count += 1
                else:
                    logger.warning(f"Failed to export template: {cache_key}")
            
            logger.info(f"Exported {exported_count} templates to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Template export failed: {e}")
            return False
    
    def import_templates(self, import_dir: Path, overwrite: bool = False) -> Dict[str, int]:
        """Import templates from directory"""
        try:
            import_stats = {
                'imported': 0,
                'skipped': 0,
                'errors': 0
            }
            
            if not import_dir.exists():
                logger.error(f"Import directory does not exist: {import_dir}")
                return import_stats
            
            # Scan for template files
            for category_dir in import_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                
                category = category_dir.name
                
                # Check if category is valid
                if category not in self.template_dirs:
                    logger.warning(f"Unknown template category: {category}")
                    continue
                
                # Import templates from this category
                for template_file in category_dir.iterdir():
                    if template_file.suffix.lower() not in self.supported_formats:
                        continue
                    
                    template_name = template_file.stem
                    cache_key = f"{category}_{template_name}"
                    
                    try:
                        # Check if template already exists
                        if cache_key in self.template_cache and not overwrite:
                            import_stats['skipped'] += 1
                            continue
                        
                        # Load template
                        template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
                        
                        if template is not None:
                            # Add template
                            success = self.add_template(template_name, template, category)
                            
                            if success:
                                import_stats['imported'] += 1
                            else:
                                import_stats['errors'] += 1
                        else:
                            import_stats['errors'] += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to import template {template_name}: {e}")
                        import_stats['errors'] += 1
            
            logger.info(f"Template import complete: {import_stats}")
            return import_stats
            
        except Exception as e:
            logger.error(f"Template import failed: {e}")
            return {'imported': 0, 'skipped': 0, 'errors': 1}
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize template cache"""
        try:
            optimization_stats = {
                'original_size': len(self.template_cache),
                'optimized_size': 0,
                'removed_duplicates': 0,
                'removed_corrupted': 0
            }
            
            # Remove corrupted templates
            corrupted_keys = []
            for cache_key, template in self.template_cache.items():
                if template is None or template.size == 0:
                    corrupted_keys.append(cache_key)
            
            for key in corrupted_keys:
                del self.template_cache[key]
                if key in self.template_metadata:
                    del self.template_metadata[key]
                optimization_stats['removed_corrupted'] += 1
            
            # Remove duplicate templates (based on hash)
            template_hashes = {}
            duplicate_keys = []
            
            for cache_key, template in self.template_cache.items():
                # Calculate template hash
                template_hash = hashlib.md5(template.tobytes()).hexdigest()
                
                if template_hash in template_hashes:
                    # Found duplicate
                    duplicate_keys.append(cache_key)
                else:
                    template_hashes[template_hash] = cache_key
            
            for key in duplicate_keys:
                del self.template_cache[key]
                if key in self.template_metadata:
                    del self.template_metadata[key]
                optimization_stats['removed_duplicates'] += 1
            
            optimization_stats['optimized_size'] = len(self.template_cache)
            
            # Save optimized cache
            self.save_cache()
            
            logger.info(f"Cache optimization complete: {optimization_stats}")
            return optimization_stats
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {}
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        try:
            stats = {
                'total_templates': len(self.template_cache),
                'total_metadata': len(self.template_metadata),
                'categories': {},
                'average_template_size': 0,
                'cache_files_exist': {
                    'cache': self.cache_file.exists(),
                    'metadata': self.metadata_file.exists()
                },
                'memory_usage_mb': 0
            }
            
            if not self.template_cache:
                return stats
            
            # Calculate statistics
            total_pixels = 0
            category_counts = {}
            template_sizes = []
            
            for cache_key, template in self.template_cache.items():
                category = cache_key.split('_')[0]
                category_counts[category] = category_counts.get(category, 0) + 1
                
                if template is not None:
                    total_pixels += template.size
                    template_sizes.append(template.size)
            
            stats['categories'] = category_counts
            stats['average_template_size'] = sum(template_sizes) / len(template_sizes) if template_sizes else 0
            stats['memory_usage_mb'] = (total_pixels * 3) / (1024 * 1024)  # Bytes to MB
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {}
    
    def __del__(self):
        """Destructor - save cache on exit"""
        try:
            self.save_cache()
        except Exception:
            passFailed to load all templates: {e}")
            return {}
    
    def load_templates_from_directory(self, directory: Path, category: str) -> Dict[str, np.ndarray]:
        """Load templates from a specific directory"""
        try:
            templates = {}
            
            if not directory.exists():
                logger.warning(f"Template directory does not exist: {directory}")
                return templates
            
            for file_path in directory.iterdir():
                if file_path.suffix.lower() in self.supported_formats:
                    template_name = file_path.stem
                    
                    # Check if template is in cache and up-to-date
                    if self._is_template_cached(file_path, category):
                        template = self._get_cached_template(file_path, category)
                        if template is not None:
                            templates[template_name] = template
                            continue
                    
                    # Load template from file
                    template = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                    
                    if template is not None:
                        # Preprocess template
                        processed_template = self._preprocess_template(template, category)
                        templates[template_name] = processed_template
                        
                        # Cache the template
                        self._cache_template(file_path, category, processed_template)
                        
                        logger.debug(f"Loaded template: {template_name}")
                    else:
                        logger.warning(f"Failed to load template: {file_path}")
            
            return templates
            
        except Exception as e:
            logger.error(f"Failed to load templates from {directory}: {e}")
            return {}
    
    def _preprocess_template(self, template: np.ndarray, category: str) -> np.ndarray:
        """Preprocess template for better matching"""
        try:
            processed = template.copy()
            
            # Category-specific preprocessing
            if category == 'champions':
                # Champions might need specific preprocessing
                processed = self._preprocess_champion_template(processed)
            elif category == 'items':
                # Items might need different preprocessing
                processed = self._preprocess_item_template(processed)
            elif category == 'ui':
                # UI elements might need edge enhancement
                processed = self._preprocess_ui_template(processed)
            
            # General preprocessing
            processed = self._apply_general_preprocessing(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Template preprocessing failed: {e}")
            return template
    
    def _preprocess_champion_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess champion templates"""
        try:
            # Remove background (assuming champions have consistent background)
            processed = self._remove_background(template)
            
            # Enhance champion features
            processed = self._enhance_features(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Champion template preprocessing failed: {e}")
            return template
    
    def _preprocess_item_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess item templates"""
        try:
            # Items are usually small and detailed
            # Enhance contrast and sharpness
            processed = cv2.convertScaleAbs(template, alpha=1.2, beta=10)
            
            # Apply sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Item template preprocessing failed: {e}")
            return template
    
    def _preprocess_ui_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess UI element templates"""
        try:
            # UI elements benefit from edge enhancement
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Combine original with edges
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            processed = cv2.addWeighted(template, 0.7, edges_colored, 0.3, 0)
            
            return processed
            
        except Exception as e:
            logger.error(f"UI template preprocessing failed: {e}")
            return template
    
    def _apply_general_preprocessing(self, template: np.ndarray) -> np.ndarray:
        """Apply general preprocessing to all templates"""
        try:
            # Normalize size if too large
            height, width = template.shape[:2]
            max_size = 200
            
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Apply slight Gaussian blur to reduce noise
            template = cv2.GaussianBlur(template, (3, 3), 0)
            
            return template
            
        except Exception as e:
            logger.error(f"General preprocessing failed: {e}")
            return template
    
    def _remove_background(self, template: np.ndarray) -> np.ndarray:
        """Remove background from template"""
        try:
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            
            # Create mask for background (assuming background is dark/uniform)
            lower_bg = np.array([0, 0, 0])
            upper_bg = np.array([180, 255, 50])
            bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
            
            # Invert mask to get foreground
            fg_mask = cv2.bitwise_not(bg_mask)
            
            # Apply mask to template
            result = cv2.bitwise_and(template, template, mask=fg_mask)
            
            return result
            
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return template
    
    def _enhance_features(self, template: np.ndarray) -> np.ndarray:
        """Enhance template features"""
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Feature enhancement failed: {e}")
            return template
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for cache validation"""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return hashlib.md5(file_content).hexdigest()
        except Exception:
            return ""
    
    def _is_template_cached(self, file_path: Path, category: str) -> bool:
        """Check if template is cached and up-to-date"""
        try:
            cache_key = f"{category}_{file_path.stem}"
            
            if cache_key not in self.template_metadata:
                return False
            
            # Check if file has been modified
            current_hash = self._get_file_hash(file_path)
            cached_hash = self.template_metadata[cache_key].get('hash', '')
            
            return current_hash == cached_hash
            
        except Exception:
            return False
    
    def _get_cached_template(self, file_path: Path, category: str) -> Optional[np.ndarray]:
        """Get template from cache"""
        try:
            cache_key = f"{category}_{file_path.stem}"
            return self.template_cache.get(cache_key)
        except Exception:
            return None
    
    def _cache_template(self, file_path: Path, category: str, template: np.ndarray):
        """Cache template with metadata"""
        try:
            cache_key = f"{category}_{file_path.stem}"
            
            # Store template in cache
            self.template_cache[cache_key] = template
            
            # Store metadata
            self.template_metadata[cache_key] = {
                'path': str(file_path),
                'category': category,
                'hash': self._get_file_hash(file_path),
                'size': template.shape,
                'cached_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Template caching failed: {e}")
    
    def _load_cache(self):
        """Load template cache from disk"""
        try:
            # Load template cache
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.template_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.template_cache)} templates from cache")
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.template_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.template_metadata)} templates")
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.template_cache = {}
            self.template_metadata = {}
    
    def save_cache(self):
        """Save template cache to disk"""
        try:
            # Save template cache
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.template_cache, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.template_metadata, f, indent=2)
            
            logger.info("Template cache saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def clear_cache(self):
        """Clear template cache"""
        try:
            self.template_cache.clear()
            self.template_metadata.clear()
            
            # Remove cache files
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            logger.info("Template cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def add_template(self, name: str, template: np.ndarray, category: str) -> bool:
        """Add a new template"""
        try:
            # Preprocess template
            processed = self._preprocess_template(template, category)
            
            # Save to appropriate directory
            template_dir = self.template_dirs[category]
            template_path = template_dir / f"{name}.png"
            
            success = cv2.imwrite(str(template_path), processed)
            
            if success:
                # Add to cache
                cache_key = f"{category}_{name}"
                self.template_cache[cache_key] = processed
                self.template_metadata[cache_key] = {
                    'path': str(template_path),
                    'category': category,
                    'hash': self._get_file_hash(template_path),
                    'size': processed.shape,
                    'cached_at': datetime.now().isoformat()
                }
                
                logger.info(f"Added template: {name} ({category})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add template {name}: {e}")
            return False
    
    def remove_template(self, name: str, category: str) -> bool:
        """Remove a template"""
        try:
            # Remove from cache
            cache_key = f"{category}_{name}"
            if cache_key in self.template_cache:
                del self.template_cache[cache_key]
            if cache_key in self.template_metadata:
                del self.template_metadata[cache_key]
            
            # Remove file
            template_path = self.template_dirs[category] / f"{name}.png"
            if template_path.exists():
                template_path.unlink()
            
            logger.info(f"Removed template: {name} ({category})")
            return True
            
        except Exception as e:
            logger.error(f"