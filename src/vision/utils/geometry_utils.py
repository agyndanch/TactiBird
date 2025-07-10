"""
TactiBird Overlay - Geometry Utilities
"""

import numpy as np
import cv2
import logging
from typing import Tuple, List, Dict, Optional, Any
import math

logger = logging.getLogger(__name__)

class Point:
    """2D Point class"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Rectangle:
    """Rectangle class"""
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    @property
    def center(self) -> Point:
        """Get center point of rectangle"""
        return Point(self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        """Get area of rectangle"""
        return self.width * self.height
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside rectangle"""
        return (self.x <= point.x <= self.x + self.width and 
                self.y <= point.y <= self.y + self.height)
    
    def intersects(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another"""
        return not (self.x + self.width < other.x or 
                   other.x + other.width < self.x or
                   self.y + self.height < other.y or 
                   other.y + other.height < self.y)
    
    def intersection_area(self, other: 'Rectangle') -> float:
        """Calculate intersection area with another rectangle"""
        if not self.intersects(other):
            return 0.0
        
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        
        return (right - left) * (bottom - top)
    
    def __repr__(self):
        return f"Rectangle({self.x}, {self.y}, {self.width}, {self.height})"

class GeometryUtils:
    """Utility functions for geometric calculations"""
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        try:
            x1, y1 = point1
            x2, y2 = point2
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate angle between two points in degrees"""
        try:
            x1, y1 = point1
            x2, y2 = point2
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            return angle_deg
        except Exception as e:
            logger.error(f"Angle calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def rotate_point(point: Tuple[float, float], center: Tuple[float, float], angle_deg: float) -> Tuple[float, float]:
        """Rotate point around center by given angle"""
        try:
            x, y = point
            cx, cy = center
            angle_rad = math.radians(angle_deg)
            
            # Translate to origin
            x_translated = x - cx
            y_translated = y - cy
            
            # Rotate
            x_rotated = x_translated * math.cos(angle_rad) - y_translated * math.sin(angle_rad)
            y_rotated = x_translated * math.sin(angle_rad) + y_translated * math.cos(angle_rad)
            
            # Translate back
            x_final = x_rotated + cx
            y_final = y_rotated + cy
            
            return (x_final, y_final)
        except Exception as e:
            logger.error(f"Point rotation failed: {e}")
            return point
    
    @staticmethod
    def calculate_bbox_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        try:
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            rect1 = Rectangle(x1, y1, w1, h1)
            rect2 = Rectangle(x2, y2, w2, h2)
            
            intersection = rect1.intersection_area(rect2)
            union = rect1.area + rect2.area - intersection
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.error(f"Bbox overlap calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def find_contour_center(contour: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find center of contour using moments"""
        try:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return (cx, cy)
            return None
        except Exception as e:
            logger.error(f"Contour center calculation failed: {e}")
            return None
    
    @staticmethod
    def calculate_contour_area(contour: np.ndarray) -> float:
        """Calculate area of contour"""
        try:
            return cv2.contourArea(contour)
        except Exception as e:
            logger.error(f"Contour area calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_contour_perimeter(contour: np.ndarray, closed: bool = True) -> float:
        """Calculate perimeter of contour"""
        try:
            return cv2.arcLength(contour, closed)
        except Exception as e:
            logger.error(f"Contour perimeter calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def fit_ellipse_to_contour(contour: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """Fit ellipse to contour"""
        try:
            if len(contour) >= 5:  # Need at least 5 points for ellipse fitting
                ellipse = cv2.fitEllipse(contour)
                return ellipse
            return None
        except Exception as e:
            logger.error(f"Ellipse fitting failed: {e}")
            return None
    
    @staticmethod
    def approximate_contour(contour: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
        """Approximate contour with fewer points"""
        try:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return approx
        except Exception as e:
            logger.error(f"Contour approximation failed: {e}")
            return contour
    
    @staticmethod
    def is_contour_convex(contour: np.ndarray) -> bool:
        """Check if contour is convex"""
        try:
            return cv2.isContourConvex(contour)
        except Exception as e:
            logger.error(f"Convexity check failed: {e}")
            return False
    
    @staticmethod
    def calculate_shape_similarity(contour1: np.ndarray, contour2: np.ndarray) -> float:
        """Calculate shape similarity between two contours using Hu moments"""
        try:
            moments1 = cv2.moments(contour1)
            moments2 = cv2.moments(contour2)
            
            hu1 = cv2.HuMoments(moments1).flatten()
            hu2 = cv2.HuMoments(moments2).flatten()
            
            # Calculate similarity using normalized correlation
            correlation = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
            
            # Convert to similarity (lower correlation = higher similarity)
            similarity = 1.0 / (1.0 + correlation)
            
            return similarity
        except Exception as e:
            logger.error(f"Shape similarity calculation failed: {e}")
            return 0.0

class HexagonalGrid:
    """Utility class for hexagonal grid calculations (TFT board)"""
    
    def __init__(self, center: Tuple[float, float], hex_size: float):
        self.center = center
        self.hex_size = hex_size
        self.width = hex_size * 2
        self.height = hex_size * math.sqrt(3)
    
    def hex_to_pixel(self, q: int, r: int) -> Tuple[float, float]:
        """Convert hexagonal coordinates to pixel coordinates"""
        try:
            x = self.hex_size * (3.0/2.0 * q)
            y = self.hex_size * (math.sqrt(3.0)/2.0 * q + math.sqrt(3.0) * r)
            
            return (self.center[0] + x, self.center[1] + y)
        except Exception as e:
            logger.error(f"Hex to pixel conversion failed: {e}")
            return self.center
    
    def pixel_to_hex(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pixel coordinates to hexagonal coordinates"""
        try:
            # Translate to hex center
            x_rel = x - self.center[0]
            y_rel = y - self.center[1]
            
            # Convert to fractional hex coordinates
            q_frac = (2.0/3.0 * x_rel) / self.hex_size
            r_frac = (-1.0/3.0 * x_rel + math.sqrt(3.0)/3.0 * y_rel) / self.hex_size
            s_frac = -q_frac - r_frac
            
            # Round to nearest hex
            q_round = round(q_frac)
            r_round = round(r_frac)
            s_round = round(s_frac)
            
            # Handle rounding errors
            q_diff = abs(q_round - q_frac)
            r_diff = abs(r_round - r_frac)
            s_diff = abs(s_round - s_frac)
            
            if q_diff > r_diff and q_diff > s_diff:
                q_round = -r_round - s_round
            elif r_diff > s_diff:
                r_round = -q_round - s_round
            
            return (q_round, r_round)
        except Exception as e:
            logger.error(f"Pixel to hex conversion failed: {e}")
            return (0, 0)
    
    def get_hex_neighbors(self, q: int, r: int) -> List[Tuple[int, int]]:
        """Get neighboring hexagon coordinates"""
        try:
            directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
            neighbors = []
            
            for dq, dr in directions:
                neighbors.append((q + dq, r + dr))
            
            return neighbors
        except Exception as e:
            logger.error(f"Hex neighbors calculation failed: {e}")
            return []
    
    def hex_distance(self, q1: int, r1: int, q2: int, r2: int) -> int:
        """Calculate distance between two hex coordinates"""
        try:
            return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2
        except Exception as e:
            logger.error(f"Hex distance calculation failed: {e}")
            return 0
    
    def get_hex_corners(self, q: int, r: int) -> List[Tuple[float, float]]:
        """Get corner points of hexagon"""
        try:
            center_x, center_y = self.hex_to_pixel(q, r)
            corners = []
            
            for i in range(6):
                angle = 60 * i
                angle_rad = math.radians(angle)
                x = center_x + self.hex_size * math.cos(angle_rad)
                y = center_y + self.hex_size * math.sin(angle_rad)
                corners.append((x, y))
            
            return corners
        except Exception as e:
            logger.error(f"Hex corners calculation failed: {e}")
            return []

class TFTBoardGeometry:
    """TFT-specific board geometry utilities"""
    
    def __init__(self, board_bounds: Tuple[int, int, int, int]):
        self.board_bounds = board_bounds  # x, y, width, height
        self.board_size = (7, 4)  # TFT board is 7x4
        
        # Calculate grid properties
        x, y, w, h = board_bounds
        self.cell_width = w / self.board_size[0]
        self.cell_height = h / self.board_size[1]
        
    def board_to_pixel(self, board_x: int, board_y: int) -> Tuple[float, float]:
        """Convert board coordinates to pixel coordinates"""
        try:
            x, y, w, h = self.board_bounds
            
            # Calculate pixel position (center of cell)
            pixel_x = x + (board_x + 0.5) * self.cell_width
            pixel_y = y + (board_y + 0.5) * self.cell_height
            
            return (pixel_x, pixel_y)
        except Exception as e:
            logger.error(f"Board to pixel conversion failed: {e}")
            return (0, 0)
    
    def pixel_to_board(self, pixel_x: float, pixel_y: float) -> Tuple[int, int]:
        """Convert pixel coordinates to board coordinates"""
        try:
            x, y, w, h = self.board_bounds
            
            # Calculate relative position within board
            rel_x = pixel_x - x
            rel_y = pixel_y - y
            
            # Convert to board coordinates
            board_x = int(rel_x / self.cell_width)
            board_y = int(rel_y / self.cell_height)
            
            # Clamp to valid range
            board_x = max(0, min(self.board_size[0] - 1, board_x))
            board_y = max(0, min(self.board_size[1] - 1, board_y))
            
            return (board_x, board_y)
        except Exception as e:
            logger.error(f"Pixel to board conversion failed: {e}")
            return (0, 0)
    
    def is_valid_position(self, board_x: int, board_y: int) -> bool:
        """Check if board position is valid"""
        try:
            # TFT board has some invalid positions due to hexagonal layout
            if board_y == 0:  # Front row
                return 0 <= board_x <= 6
            elif board_y == 1:  # Second row
                return 0 <= board_x <= 6
            elif board_y == 2:  # Third row
                return 0 <= board_x <= 6
            elif board_y == 3:  # Back row
                return 1 <= board_x <= 5  # Corners not available
            else:
                return False
        except Exception:
            return False
    
    def get_adjacent_positions(self, board_x: int, board_y: int) -> List[Tuple[int, int]]:
        """Get adjacent board positions"""
        try:
            adjacent = []
            
            # Check all 8 directions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = board_x + dx
                    new_y = board_y + dy
                    
                    if self.is_valid_position(new_x, new_y):
                        adjacent.append((new_x, new_y))
            
            return adjacent
        except Exception as e:
            logger.error(f"Adjacent positions calculation failed: {e}")
            return []
    
    def calculate_positioning_score(self, positions: List[Tuple[int, int]], 
                                  formation_type: str = "standard") -> float:
        """Calculate positioning score based on formation type"""
        try:
            if not positions:
                return 0.0
            
            score = 0.0
            
            if formation_type == "standard":
                # Prefer front line for tanks, back line for carries
                for x, y in positions:
                    if y <= 1:  # Front positions
                        score += 1.0
                    elif y >= 2:  # Back positions
                        score += 0.8
            
            elif formation_type == "spread":
                # Prefer spread out positions
                for i, pos1 in enumerate(positions):
                    for pos2 in positions[i+1:]:
                        distance = GeometryUtils.calculate_distance(pos1, pos2)
                        score += min(1.0, distance / 3.0)  # Normalize distance
            
            elif formation_type == "clumped":
                # Prefer close positions
                for i, pos1 in enumerate(positions):
                    for pos2 in positions[i+1:]:
                        distance = GeometryUtils.calculate_distance(pos1, pos2)
                        score += max(0.0, 1.0 - distance / 3.0)  # Inverse distance
            
            return score / len(positions) if positions else 0.0
            
        except Exception as e:
            logger.error(f"Positioning score calculation failed: {e}")
            return 0.0