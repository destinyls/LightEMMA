"""
Fallback implementations for Argoverse utilities
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Fallback implementation for argoverse.utils.json_utils.read_json_file
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def centerline_to_polygon(centerline: np.ndarray, lane_width: float = 3.5, visualize: bool = False) -> Optional[np.ndarray]:
    """
    Fallback implementation for argoverse.utils.centerline_utils.centerline_to_polygon
    Convert lane centerline to polygon
    """
    if len(centerline) < 2:
        return None
    
    # Create polygon by offsetting centerline
    polygon_points = []
    half_width = lane_width / 2
    
    for i in range(len(centerline)):
        if i == 0:
            # First point: use vector to next point
            direction = centerline[i+1] - centerline[i]
        elif i == len(centerline) - 1:
            # Last point: use vector from previous point
            direction = centerline[i] - centerline[i-1]
        else:
            # Middle points: use average of vectors
            direction = (centerline[i+1] - centerline[i-1]) / 2
        
        # Normalize and get perpendicular
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
            perpendicular = np.array([-direction[1], direction[0]])
            
            # Add points on both sides
            left_point = centerline[i] + perpendicular * half_width
            right_point = centerline[i] - perpendicular * half_width
            
            polygon_points.append([left_point, right_point])
    
    if polygon_points:
        # Arrange points to form a closed polygon
        left_side = [p[0] for p in polygon_points]
        right_side = [p[1] for p in reversed(polygon_points)]
        
        return np.array(left_side + right_side)
    
    return None


def interp_arc(t: int, points: np.ndarray) -> np.ndarray:
    """
    Fallback implementation for argoverse.utils.interpolate.interp_arc
    Interpolate points along an arc
    """
    if len(points) < 2:
        return points
    
    try:
        from scipy.interpolate import interp1d
        
        # Create parameter for original points based on cumulative distance
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
        
        # Normalize distances to [0, 1]
        if distances[-1] > 0:
            distances = distances / distances[-1]
        
        # Create new parameter array
        t_new = np.linspace(0, 1, t)
        
        # Interpolate x and y coordinates
        f_x = interp1d(distances, points[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
        f_y = interp1d(distances, points[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        
        return np.column_stack([f_x(t_new), f_y(t_new)])
        
    except ImportError:
        # Fallback to simple linear interpolation without scipy
        t_original = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, t)
        
        # Simple linear interpolation
        x_interp = np.interp(t_new, t_original, points[:, 0])
        y_interp = np.interp(t_new, t_original, points[:, 1])
        
        return np.column_stack([x_interp, y_interp])


def compute_polygon_bboxes(polygons: List[np.ndarray]) -> List[tuple]:
    """
    Fallback implementation for argoverse.utils.manhattan_search.compute_polygon_bboxes
    Compute bounding boxes for polygons
    """
    bboxes = []
    for polygon in polygons:
        if len(polygon) > 0:
            min_x, min_y = np.min(polygon, axis=0)
            max_x, max_y = np.max(polygon, axis=0)
            bboxes.append((min_x, min_y, max_x, max_y))
        else:
            bboxes.append((0, 0, 0, 0))
    
    return bboxes


class ArgoverseMap:
    """
    Fallback implementation for argoverse.map_representation.map_api.ArgoverseMap
    Basic map representation for compatibility
    """
    
    def __init__(self, map_data: Dict[str, Any] = None):
        self.map_data = map_data or {}
        self.city_name = "fallback_city"
    
    def get_lane_ids(self) -> List[str]:
        """Get all lane IDs"""
        return [lane['token'] for lane in self.map_data.get('lane', [])]
    
    def get_lane_segment_polygon(self, lane_id: str) -> Optional[np.ndarray]:
        """Get polygon for a lane segment"""
        for lane in self.map_data.get('lane', []):
            if lane['token'] == lane_id:
                # Find associated polygon
                polygon_token = lane['polygon_token']
                for polygon in self.map_data.get('polygon', []):
                    if polygon['token'] == polygon_token:
                        # Convert to coordinates (simplified)
                        coords = []
                        for node_token in polygon['exterior_node_tokens']:
                            for node in self.map_data.get('node', []):
                                if node['token'] == node_token:
                                    coords.append([node['x'], node['y']])
                                    break
                        return np.array(coords) if coords else None
        return None
    
    def get_lane_segment_centerline(self, lane_id: str) -> Optional[np.ndarray]:
        """Get centerline for a lane segment"""
        polygon = self.get_lane_segment_polygon(lane_id)
        if polygon is not None and len(polygon) >= 4:
            # Simple centerline computation
            n_points = len(polygon) // 2
            centerline = []
            for i in range(n_points):
                p1 = polygon[i]
                p2 = polygon[-(i+1)]
                center = (p1 + p2) / 2
                centerline.append(center)
            return np.array(centerline)
        return None 