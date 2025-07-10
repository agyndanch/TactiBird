"""
TactiBird Overlay - Capture Utilities
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class CaptureUtils:
    """Utility functions for screen capture operations"""
    
    @staticmethod
    def validate_region(region: Dict[str, int], screen_bounds: Tuple[int, int]) -> bool:
        """Validate if a region is within screen bounds"""
        try:
            x, y = region.get('x', 0), region.get('y', 0)
            w, h = region.get('w', 0), region.get('h', 0)
            screen_w, screen_h = screen_bounds
            
            # Check if region is completely within screen
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return False
            
            if x + w > screen_w or y + h > screen_h:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Region validation failed: {e}")
            return False
    
    @staticmethod
    def normalize_region(region: Dict[str, int], screen_bounds: Tuple[int, int]) -> Dict[str, int]:
        """Normalize region coordinates to screen bounds"""
        try:
            x, y = region.get('x', 0), region.get('y', 0)
            w, h = region.get('w', 0), region.get('h', 0)
            screen_w, screen_h = screen_bounds
            
            # Clamp to screen bounds
            x = max(0, min(x, screen_w - 1))
            y = max(0, min(y, screen_h - 1))
            w = max(1, min(w, screen_w - x))
            h = max(1, min(h, screen_h - y))
            
            return {'x': x, 'y': y, 'w': w, 'h': h}
            
        except Exception as e:
            logger.error(f"Region normalization failed: {e}")
            return region
    
    @staticmethod
    def scale_region(region: Dict[str, int], scale_factor: float) -> Dict[str, int]:
        """Scale region by a factor"""
        try:
            return {
                'x': int(region['x'] * scale_factor),
                'y': int(region['y'] * scale_factor),
                'w': int(region['w'] * scale_factor),
                'h': int(region['h'] * scale_factor)
            }
        except Exception as e:
            logger.error(f"Region scaling failed: {e}")
            return region
    
    @staticmethod
    def region_to_bbox(region: Dict[str, int]) -> Tuple[int, int, int, int]:
        """Convert region dict to bounding box tuple"""
        return (region['x'], region['y'], region['w'], region['h'])
    
    @staticmethod
    def bbox_to_region(bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
        """Convert bounding box tuple to region dict"""
        x, y, w, h = bbox
        return {'x': x, 'y': y, 'w': w, 'h': h}
    
    @staticmethod
    def calculate_region_overlap(region1: Dict[str, int], region2: Dict[str, int]) -> float:
        """Calculate overlap ratio between two regions"""
        try:
            x1, y1, w1, h1 = region1['x'], region1['y'], region1['w'], region1['h']
            x2, y2, w2, h2 = region2['x'], region2['y'], region2['w'], region2['h']
            
            # Calculate intersection
            left = max(x1, x2)
            top = max(y1, y2)
            right = min(x1 + w1, x2 + w2)
            bottom = min(y1 + h1, y2 + h2)
            
            if left >= right or top >= bottom:
                return 0.0
            
            intersection = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Overlap calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def expand_region(region: Dict[str, int], expansion: int, screen_bounds: Tuple[int, int]) -> Dict[str, int]:
        """Expand region by specified pixels"""
        try:
            expanded = {
                'x': region['x'] - expansion,
                'y': region['y'] - expansion,
                'w': region['w'] + 2 * expansion,
                'h': region['h'] + 2 * expansion
            }
            
            return CaptureUtils.normalize_region(expanded, screen_bounds)
            
        except Exception as e:
            logger.error(f"Region expansion failed: {e}")
            return region
    
    @staticmethod
    def merge_regions(regions: List[Dict[str, int]]) -> Optional[Dict[str, int]]:
        """Merge multiple regions into bounding box"""
        try:
            if not regions:
                return None
            
            min_x = min(r['x'] for r in regions)
            min_y = min(r['y'] for r in regions)
            max_x = max(r['x'] + r['w'] for r in regions)
            max_y = max(r['y'] + r['h'] for r in regions)
            
            return {
                'x': min_x,
                'y': min_y,
                'w': max_x - min_x,
                'h': max_y - min_y
            }
            
        except Exception as e:
            logger.error(f"Region merge failed: {e}")
            return None

class ImageProcessor:
    """Image processing utilities for captured frames"""
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int], 
                                maintain_aspect: bool = True) -> np.ndarray:
        """Resize image while optionally maintaining aspect ratio"""
        try:
            target_w, target_h = target_size
            current_h, current_w = image.shape[:2]
            
            if maintain_aspect:
                # Calculate scale to fit within target size
                scale_w = target_w / current_w
                scale_h = target_h / current_h
                scale = min(scale_w, scale_h)
                
                new_w = int(current_w * scale)
                new_h = int(current_h * scale)
                
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Pad to target size if needed
                if new_w != target_w or new_h != target_h:
                    # Create black background
                    result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
                    
                    # Center the resized image
                    start_x = (target_w - new_w) // 2
                    start_y = (target_h - new_h) // 2
                    
                    result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
                    return result
                else:
                    return resized
            else:
                return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                
        except Exception as e:
            logger.error(f"Image resize failed: {e}")
            return image
    
    @staticmethod
    def apply_roi_mask(image: np.ndarray, regions: List[Dict[str, int]]) -> np.ndarray:
        """Apply mask to keep only specified regions of interest"""
        try:
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create mask for all regions
            for region in regions:
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                mask[y:y+h, x:x+w] = 255
            
            # Apply mask
            if len(image.shape) == 3:
                mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                return cv2.bitwise_and(image, mask_3d)
            else:
                return cv2.bitwise_and(image, mask)
                
        except Exception as e:
            logger.error(f"ROI mask application failed: {e}")
            return image
    
    @staticmethod
    def detect_motion(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 25) -> Tuple[bool, float]:
        """Detect motion between two frames"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Threshold the difference
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate motion percentage
            motion_pixels = np.sum(thresh > 0)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            motion_ratio = motion_pixels / total_pixels
            
            # Consider motion detected if more than 1% of pixels changed
            motion_detected = motion_ratio > 0.01
            
            return motion_detected, motion_ratio
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return False, 0.0
    
    @staticmethod
    def calculate_image_quality(image: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Normalize metrics
            normalized_sharpness = min(100, sharpness / 100)  # Normalize to 0-100
            normalized_brightness = brightness / 255 * 100    # Normalize to 0-100
            normalized_contrast = min(100, contrast / 50)     # Normalize to 0-100
            
            return {
                'sharpness': normalized_sharpness,
                'brightness': normalized_brightness,
                'contrast': normalized_contrast,
                'overall_quality': (normalized_sharpness + normalized_contrast) / 2
            }
            
        except Exception as e:
            logger.error(f"Image quality calculation failed: {e}")
            return {'sharpness': 0, 'brightness': 0, 'contrast': 0, 'overall_quality': 0}

class PerformanceMonitor:
    """Monitor capture performance and optimize settings"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.processing_times = []
        self.memory_usage = []
        
    def record_frame_time(self, frame_time: float):
        """Record time taken to capture a frame"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def record_processing_time(self, processing_time: float):
        """Record time taken to process a frame"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        try:
            stats = {}
            
            if self.frame_times:
                stats['avg_frame_time'] = sum(self.frame_times) / len(self.frame_times)
                stats['max_frame_time'] = max(self.frame_times)
                stats['min_frame_time'] = min(self.frame_times)
                stats['fps'] = 1.0 / stats['avg_frame_time'] if stats['avg_frame_time'] > 0 else 0
            
            if self.processing_times:
                stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
                stats['max_processing_time'] = max(self.processing_times)
                
            return stats
            
        except Exception as e:
            logger.error(f"Performance stats calculation failed: {e}")
            return {}
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest performance optimizations"""
        suggestions = []
        stats = self.get_performance_stats()
        
        if 'fps' in stats:
            if stats['fps'] < 15:
                suggestions.append("Consider reducing capture FPS or region sizes")
            elif stats['fps'] > 60:
                suggestions.append("FPS is high, could increase capture quality")
        
        if 'avg_processing_time' in stats and 'avg_frame_time' in stats:
            processing_ratio = stats['avg_processing_time'] / stats['avg_frame_time']
            if processing_ratio > 0.8:
                suggestions.append("Processing is bottleneck, optimize image processing")
        
        return suggestions

class RegionCalibrator:
    """Calibrate and fine-tune capture regions"""
    
    def __init__(self):
        self.calibration_data = {}
        
    def start_interactive_calibration(self, screenshot: np.ndarray) -> Dict[str, Dict[str, int]]:
        """Start interactive region calibration (would need GUI implementation)"""
        # This would typically open a GUI for manual region selection
        # For now, return default regions
        height, width = screenshot.shape[:2]
        
        return {
            'board': {'x': int(width*0.1), 'y': int(height*0.2), 'w': int(width*0.6), 'h': int(height*0.5)},
            'shop': {'x': int(width*0.2), 'y': int(height*0.75), 'w': int(width*0.6), 'h': int(height*0.2)},
            'gold': {'x': int(width*0.05), 'y': int(height*0.85), 'w': int(width*0.08), 'h': int(height*0.06)},
            'health': {'x': int(width*0.05), 'y': int(height*0.78), 'w': int(width*0.08), 'h': int(height*0.06)},
            'level': {'x': int(width*0.05), 'y': int(height*0.71), 'w': int(width*0.08), 'h': int(height*0.06)},
            'traits': {'x': int(width*0.02), 'y': int(height*0.15), 'w': int(width*0.15), 'h': int(height*0.5)}
        }
    
    def auto_calibrate_regions(self, screenshots: List[np.ndarray]) -> Dict[str, Dict[str, int]]:
        """Auto-calibrate regions using multiple screenshots"""
        try:
            if not screenshots:
                return {}
            
            # Analyze multiple screenshots to find consistent regions
            region_candidates = {}
            
            for screenshot in screenshots:
                from src.capture.region_detector import RegionDetector
                detector = RegionDetector()
                detected = detector.detect_ui_regions(screenshot)
                
                for region_name, region_data in detected.items():
                    if region_name not in region_candidates:
                        region_candidates[region_name] = []
                    
                    region_candidates[region_name].append({
                        'x': region_data.bbox[0],
                        'y': region_data.bbox[1], 
                        'w': region_data.bbox[2],
                        'h': region_data.bbox[3],
                        'confidence': region_data.confidence
                    })
            
            # Average the regions with high confidence
            calibrated_regions = {}
            for region_name, candidates in region_candidates.items():
                high_conf_candidates = [c for c in candidates if c['confidence'] > 0.6]
                
                if high_conf_candidates:
                    avg_region = {
                        'x': int(sum(c['x'] for c in high_conf_candidates) / len(high_conf_candidates)),
                        'y': int(sum(c['y'] for c in high_conf_candidates) / len(high_conf_candidates)),
                        'w': int(sum(c['w'] for c in high_conf_candidates) / len(high_conf_candidates)),
                        'h': int(sum(c['h'] for c in high_conf_candidates) / len(high_conf_candidates))
                    }
                    calibrated_regions[region_name] = avg_region
            
            logger.info(f"Auto-calibrated {len(calibrated_regions)} regions")
            return calibrated_regions
            
        except Exception as e:
            logger.error(f"Auto-calibration failed: {e}")
            return {}
    
    def save_calibration(self, regions: Dict[str, Dict[str, int]], filename: str = "calibration.json"):
        """Save calibration data to file"""
        try:
            calibration_path = Path("config") / filename
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(calibration_path, 'w') as f:
                json.dump(regions, f, indent=2)
            
            logger.info(f"Calibration saved to {calibration_path}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
    
    def load_calibration(self, filename: str = "calibration.json") -> Dict[str, Dict[str, int]]:
        """Load calibration data from file"""
        try:
            calibration_path = Path("config") / filename
            
            if calibration_path.exists():
                with open(calibration_path, 'r') as f:
                    regions = json.load(f)
                
                logger.info(f"Calibration loaded from {calibration_path}")
                return regions
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return {}

class CaptureOptimizer:
    """Optimize capture settings based on system performance"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.optimization_history = []
    
    def optimize_for_performance(self, current_config: Dict[str, Any], 
                                target_fps: float = 30) -> Dict[str, Any]:
        """Optimize capture configuration for target performance"""
        try:
            optimized_config = current_config.copy()
            stats = self.performance_monitor.get_performance_stats()
            
            if 'fps' in stats:
                current_fps = stats['fps']
                
                if current_fps < target_fps * 0.8:  # Below 80% of target
                    # Reduce quality/regions to improve performance
                    if 'regions' in optimized_config:
                        # Reduce region sizes by 10%
                        for region_name, region in optimized_config['regions'].items():
                            region['w'] = int(region['w'] * 0.9)
                            region['h'] = int(region['h'] * 0.9)
                    
                    # Reduce FPS target
                    if 'fps' in optimized_config:
                        optimized_config['fps'] = max(15, optimized_config['fps'] - 5)
                        
                elif current_fps > target_fps * 1.2:  # Above 120% of target
                    # Can afford to increase quality
                    if 'regions' in optimized_config:
                        # Increase region sizes by 5%
                        for region_name, region in optimized_config['regions'].items():
                            region['w'] = int(region['w'] * 1.05)
                            region['h'] = int(region['h'] * 1.05)
            
            self.optimization_history.append({
                'timestamp': time.time(),
                'old_config': current_config,
                'new_config': optimized_config,
                'stats': stats
            })
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return current_config
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization history and recommendations"""
        return {
            'optimization_count': len(self.optimization_history),
            'performance_stats': self.performance_monitor.get_performance_stats(),
            'suggestions': self.performance_monitor.suggest_optimizations(),
            'recent_optimizations': self.optimization_history[-5:] if self.optimization_history else []
        }