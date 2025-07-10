"""
TactiBird Overlay - OCR Image Preprocessor
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

class PreprocessingMode(Enum):
    """Different preprocessing modes for different text types"""
    NUMBERS = "numbers"      # Optimized for reading numbers
    TEXT = "text"           # Optimized for reading text
    MIXED = "mixed"         # General purpose
    HIGH_CONTRAST = "high_contrast"  # For low contrast text

class OCRPreprocessor:
    """Preprocess images for better OCR results"""
    
    def __init__(self):
        self.preprocessing_configs = {
            PreprocessingMode.NUMBERS: {
                'resize_factor': 3.0,
                'denoise_strength': 5,
                'contrast_alpha': 1.5,
                'brightness_beta': 20,
                'blur_kernel': (1, 1),
                'morph_kernel': (2, 2),
                'threshold_method': cv2.THRESH_BINARY + cv2.THRESH_OTSU
            },
            PreprocessingMode.TEXT: {
                'resize_factor': 2.5,
                'denoise_strength': 3,
                'contrast_alpha': 1.3,
                'brightness_beta': 15,
                'blur_kernel': (3, 3),
                'morph_kernel': (1, 1),
                'threshold_method': cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            },
            PreprocessingMode.MIXED: {
                'resize_factor': 2.0,
                'denoise_strength': 4,
                'contrast_alpha': 1.4,
                'brightness_beta': 10,
                'blur_kernel': (2, 2),
                'morph_kernel': (1, 1),
                'threshold_method': cv2.THRESH_BINARY + cv2.THRESH_OTSU
            },
            PreprocessingMode.HIGH_CONTRAST: {
                'resize_factor': 4.0,
                'denoise_strength': 7,
                'contrast_alpha': 2.0,
                'brightness_beta': 30,
                'blur_kernel': (1, 1),
                'morph_kernel': (3, 3),
                'threshold_method': cv2.THRESH_BINARY + cv2.THRESH_OTSU
            }
        }
        
        logger.info("OCR preprocessor initialized")
    
    def preprocess(self, image: np.ndarray, mode: PreprocessingMode = PreprocessingMode.MIXED,
                  custom_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Preprocess image for OCR
        
        Args:
            image: Input image
            mode: Preprocessing mode
            custom_config: Custom preprocessing configuration
            
        Returns:
            Preprocessed image
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image provided for preprocessing")
                return image
            
            # Use custom config or default for mode
            config = custom_config if custom_config else self.preprocessing_configs[mode]
            
            # Start preprocessing pipeline
            processed = image.copy()
            
            # Step 1: Convert to grayscale if needed
            processed = self._convert_to_grayscale(processed)
            
            # Step 2: Resize for better OCR
            processed = self._resize_image(processed, config['resize_factor'])
            
            # Step 3: Denoise
            processed = self._denoise_image(processed, config['denoise_strength'])
            
            # Step 4: Adjust contrast and brightness
            processed = self._adjust_contrast_brightness(
                processed, config['contrast_alpha'], config['brightness_beta']
            )
            
            # Step 5: Apply Gaussian blur to smooth text
            processed = self._apply_blur(processed, config['blur_kernel'])
            
            # Step 6: Threshold to binary
            processed = self._apply_threshold(processed, config['threshold_method'])
            
            # Step 7: Morphological operations to clean up
            processed = self._apply_morphology(processed, config['morph_kernel'])
            
            # Step 8: Final cleanup
            processed = self._final_cleanup(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed"""
        try:
            if len(image.shape) == 3:
                # Convert BGR to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            return gray
            
        except Exception as e:
            logger.error(f"Grayscale conversion failed: {e}")
            return image
    
    def _resize_image(self, image: np.ndarray, resize_factor: float) -> np.ndarray:
        """Resize image for better OCR accuracy"""
        try:
            if resize_factor == 1.0:
                return image
            
            height, width = image.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
            interpolation = cv2.INTER_CUBIC if resize_factor > 1.0 else cv2.INTER_AREA
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            return resized
            
        except Exception as e:
            logger.error(f"Image resize failed: {e}")
            return image
    
    def _denoise_image(self, image: np.ndarray, strength: int) -> np.ndarray:
        """Remove noise from image"""
        try:
            if strength <= 0:
                return image
            
            # Apply Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
            return denoised
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return image
    
    def _adjust_contrast_brightness(self, image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
        """Adjust contrast and brightness"""
        try:
            # Apply: new_image = alpha * image + beta
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return adjusted
            
        except Exception as e:
            logger.error(f"Contrast/brightness adjustment failed: {e}")
            return image
    
    def _apply_blur(self, image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """Apply Gaussian blur to smooth text"""
        try:
            if kernel_size[0] <= 1 and kernel_size[1] <= 1:
                return image
            
            # Ensure kernel size is odd
            kx = kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1
            ky = kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1
            
            blurred = cv2.GaussianBlur(image, (kx, ky), 0)
            return blurred
            
        except Exception as e:
            logger.error(f"Blur application failed: {e}")
            return image
    
    def _apply_threshold(self, image: np.ndarray, threshold_method: int) -> np.ndarray:
        """Apply thresholding to create binary image"""
        try:
            _, thresholded = cv2.threshold(image, 0, 255, threshold_method)
            return thresholded
            
        except Exception as e:
            logger.error(f"Thresholding failed: {e}")
            return image
    
    def _apply_morphology(self, image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """Apply morphological operations"""
        try:
            if kernel_size[0] <= 0 or kernel_size[1] <= 0:
                return image
            
            # Create morphological kernel
            kernel = np.ones(kernel_size, np.uint8)
            
            # Apply closing to connect broken characters
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
            # Apply opening to remove noise
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            return opened
            
        except Exception as e:
            logger.error(f"Morphological operations failed: {e}")
            return image
    
    def _final_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Final cleanup operations"""
        try:
            # Remove small noise components
            cleaned = self._remove_small_components(image)
            
            # Ensure proper padding around text
            padded = self._add_padding(cleaned)
            
            return padded
            
        except Exception as e:
            logger.error(f"Final cleanup failed: {e}")
            return image
    
    def _remove_small_components(self, image: np.ndarray, min_area: int = 10) -> np.ndarray:
        """Remove small connected components (noise)"""
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
            
            # Create output image
            cleaned = np.zeros_like(image)
            
            # Keep components larger than min_area
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    cleaned[labels == i] = 255
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Small component removal failed: {e}")
            return image
    
    def _add_padding(self, image: np.ndarray, padding: int = 10) -> np.ndarray:
        """Add padding around the image"""
        try:
            padded = cv2.copyMakeBorder(
                image, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=0
            )
            return padded
            
        except Exception as e:
            logger.error(f"Padding failed: {e}")
            return image
    
    def preprocess_for_numbers(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for number recognition"""
        return self.preprocess(image, PreprocessingMode.NUMBERS)
    
    def preprocess_for_text(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for text recognition"""
        return self.preprocess(image, PreprocessingMode.TEXT)
    
    def preprocess_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing based on image characteristics"""
        try:
            # Analyze image characteristics
            analysis = self._analyze_image(image)
            
            # Choose best preprocessing mode
            if analysis['low_contrast']:
                mode = PreprocessingMode.HIGH_CONTRAST
            elif analysis['likely_numbers']:
                mode = PreprocessingMode.NUMBERS
            elif analysis['complex_text']:
                mode = PreprocessingMode.TEXT
            else:
                mode = PreprocessingMode.MIXED
            
            return self.preprocess(image, mode)
            
        except Exception as e:
            logger.error(f"Adaptive preprocessing failed: {e}")
            return self.preprocess(image, PreprocessingMode.MIXED)
    
    def _analyze_image(self, image: np.ndarray) -> Dict[str, bool]:
        """Analyze image characteristics for adaptive preprocessing"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            analysis = {}
            
            # Check contrast
            contrast = np.std(gray)
            analysis['low_contrast'] = contrast < 30
            
            # Check if image is likely to contain numbers (small, regular patterns)
            height, width = gray.shape
            analysis['likely_numbers'] = height < 50 and width < 200
            
            # Check for complex text patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            analysis['complex_text'] = edge_density > 0.1
            
            # Check brightness
            brightness = np.mean(gray)
            analysis['dark_image'] = brightness < 80
            analysis['bright_image'] = brightness > 180
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                'low_contrast': False,
                'likely_numbers': False,
                'complex_text': False,
                'dark_image': False,
                'bright_image': False
            }
    
    def batch_preprocess(self, images: List[np.ndarray], 
                        mode: PreprocessingMode = PreprocessingMode.MIXED) -> List[np.ndarray]:
        """Preprocess multiple images"""
        try:
            processed_images = []
            
            for image in images:
                processed = self.preprocess(image, mode)
                processed_images.append(processed)
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Batch preprocessing failed: {e}")
            return images
    
    def compare_preprocessing_methods(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compare different preprocessing methods on the same image"""
        try:
            results = {}
            
            for mode in PreprocessingMode:
                try:
                    processed = self.preprocess(image, mode)
                    results[mode.value] = processed
                except Exception as e:
                    logger.error(f"Preprocessing with {mode.value} failed: {e}")
                    results[mode.value] = image
            
            return results
            
        except Exception as e:
            logger.error(f"Preprocessing comparison failed: {e}")
            return {'original': image}
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about preprocessing capabilities"""
        return {
            'available_modes': [mode.value for mode in PreprocessingMode],
            'default_configs': {mode.value: config for mode, config in self.preprocessing_configs.items()},
            'supported_operations': [
                'grayscale_conversion',
                'resize',
                'denoise',
                'contrast_brightness_adjustment',
                'gaussian_blur',
                'thresholding',
                'morphological_operations',
                'noise_removal',
                'padding'
            ]
        }
    
    def save_preprocessing_result(self, image: np.ndarray, filename: str):
        """Save preprocessed image for debugging"""
        try:
            cv2.imwrite(filename, image)
            logger.info(f"Preprocessed image saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save preprocessed image: {e}")
    
    def create_custom_config(self, resize_factor: float = 2.0, denoise_strength: int = 4,
                           contrast_alpha: float = 1.4, brightness_beta: int = 10,
                           blur_kernel: Tuple[int, int] = (2, 2),
                           morph_kernel: Tuple[int, int] = (1, 1),
                           threshold_method: int = cv2.THRESH_BINARY + cv2.THRESH_OTSU) -> Dict[str, Any]:
        """Create custom preprocessing configuration"""
        return {
            'resize_factor': resize_factor,
            'denoise_strength': denoise_strength,
            'contrast_alpha': contrast_alpha,
            'brightness_beta': brightness_beta,
            'blur_kernel': blur_kernel,
            'morph_kernel': morph_kernel,
            'threshold_method': threshold_method
        }
        