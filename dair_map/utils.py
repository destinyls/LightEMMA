"""
Utility functions for DAIR map visualization and processing
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Try to import Argoverse utilities, fall back to local implementations
try:
    from argoverse.utils.json_utils import read_json_file
    from argoverse.utils.centerline_utils import centerline_to_polygon
    ARGOVERSE_AVAILABLE = True
except ImportError:
    from .argoverse_fallback import read_json_file, centerline_to_polygon
    ARGOVERSE_AVAILABLE = False


def load_map_data(map_path: str) -> Dict[str, Any]:
    """
    Load map data from JSON file using Argoverse JSON utilities
    
    Args:
        map_path: Path to the map JSON file
        
    Returns:
        Loaded map data dictionary
    """
    if not Path(map_path).exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")
    
    return read_json_file(map_path)


def get_node_coordinates_dict(map_data: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """
    Create a dictionary mapping node tokens to coordinates for fast lookup
    
    Args:
        map_data: Map data dictionary
        
    Returns:
        Dictionary mapping node tokens to (x, y) coordinates
    """
    node_coords = {}
    for node in map_data.get('node', []):
        node_coords[node['token']] = (node['x'], node['y'])
    
    return node_coords


def polygon_to_coordinates(polygon_data: Dict, node_coords: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Convert polygon data to coordinate array using precomputed node coordinates
    
    Args:
        polygon_data: Polygon data dictionary
        node_coords: Dictionary mapping node tokens to coordinates
        
    Returns:
        Array of polygon coordinates
    """
    coordinates = []
    for node_token in polygon_data['exterior_node_tokens']:
        if node_token in node_coords:
            coordinates.append(list(node_coords[node_token]))
        else:
            # Handle missing nodes by skipping or using default coordinates
            print(f"Warning: Node token {node_token} not found")
            continue
    
    return np.array(coordinates)


def extract_lane_centerline(lane_data: Dict, map_data: Dict[str, Any], 
                          node_coords: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
    """
    Extract lane centerline from lane data
    
    Args:
        lane_data: Lane data dictionary
        map_data: Full map data dictionary
        node_coords: Precomputed node coordinates dictionary
        
    Returns:
        Array of centerline coordinates
    """
    if node_coords is None:
        node_coords = get_node_coordinates_dict(map_data)
    
    polygon_token = lane_data['polygon_token']
    
    # Find the polygon associated with this lane
    for polygon in map_data.get('polygon', []):
        if polygon['token'] == polygon_token:
            coords = polygon_to_coordinates(polygon, node_coords)
            
            if len(coords) >= 4:
                # Compute centerline by averaging opposite sides of the polygon
                # This is a simplified approach - more sophisticated methods could be used
                n_points = len(coords) // 2
                centerline = []
                
                for i in range(n_points):
                    p1 = coords[i]
                    p2 = coords[-(i+1)]
                    center = (p1 + p2) / 2
                    centerline.append(center)
                
                return np.array(centerline)
            else:
                return coords
    
    return np.array([])


def compute_lane_polygon_from_centerline(centerline: np.ndarray, lane_width: float = 3.5) -> np.ndarray:
    """
    Convert lane centerline to polygon using Argoverse utilities
    
    Args:
        centerline: Array of centerline points
        lane_width: Width of the lane in meters
        
    Returns:
        Array of polygon coordinates
    """
    try:
        # Use Argoverse utility to convert centerline to polygon
        polygon_coords = centerline_to_polygon(centerline, visualize=False)
        return polygon_coords
    except Exception as e:
        print(f"Warning: Could not convert centerline to polygon using Argoverse: {e}")
        
        # Fallback: create simple polygon by offsetting centerline
        if len(centerline) < 2:
            return centerline
        
        # Compute perpendicular vectors
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
    
    return centerline


def filter_map_elements_by_bbox(map_data: Dict[str, Any], 
                               bbox: Tuple[float, float, float, float],
                               element_types: List[str] = None) -> Dict[str, Any]:
    """
    Filter map elements by bounding box
    
    Args:
        map_data: Full map data dictionary
        bbox: Bounding box (min_x, min_y, max_x, max_y)
        element_types: List of element types to filter ['polygon', 'lane', 'ped_crossing', etc.]
        
    Returns:
        Filtered map data dictionary
    """
    if element_types is None:
        element_types = ['polygon', 'lane', 'ped_crossing', 'road_segment']
    
    min_x, min_y, max_x, max_y = bbox
    node_coords = get_node_coordinates_dict(map_data)
    filtered_data = {'version': map_data.get('version', '1.3')}
    
    # Filter nodes first
    filtered_nodes = []
    filtered_node_tokens = set()
    
    for node in map_data.get('node', []):
        x, y = node['x'], node['y']
        if min_x <= x <= max_x and min_y <= y <= max_y:
            filtered_nodes.append(node)
            filtered_node_tokens.add(node['token'])
    
    filtered_data['node'] = filtered_nodes
    
    # Filter other elements based on whether their nodes are in the bbox
    for element_type in element_types:
        if element_type not in map_data:
            continue
        
        filtered_elements = []
        
        for element in map_data[element_type]:
            # Check if element has nodes in the bounding box
            element_in_bbox = False
            
            if 'exterior_node_tokens' in element:
                # For polygons, ped_crossings, etc.
                for node_token in element['exterior_node_tokens']:
                    if node_token in filtered_node_tokens:
                        element_in_bbox = True
                        break
            elif 'polygon_token' in element:
                # For lanes - check the associated polygon
                polygon_token = element['polygon_token']
                for polygon in map_data.get('polygon', []):
                    if polygon['token'] == polygon_token:
                        for node_token in polygon['exterior_node_tokens']:
                            if node_token in filtered_node_tokens:
                                element_in_bbox = True
                                break
                        break
            
            if element_in_bbox:
                filtered_elements.append(element)
        
        filtered_data[element_type] = filtered_elements
    
    return filtered_data


def convert_to_argoverse_format(map_data: Dict[str, Any], map_name: str = "dair_map") -> Dict[str, Any]:
    """
    Convert DAIR map data to Argoverse-compatible format
    
    Args:
        map_data: DAIR map data dictionary
        map_name: Name for the converted map
        
    Returns:
        Argoverse-compatible map data dictionary
    """
    node_coords = get_node_coordinates_dict(map_data)
    
    argoverse_data = {
        'map_name': map_name,
        'lanes': [],
        'polygons': [],
        'pedestrian_crossings': [],
        'metadata': {
            'version': map_data.get('version', '1.3'),
            'coordinate_system': 'local',
            'units': 'meters'
        }
    }
    
    # Convert lanes
    for lane_data in map_data.get('lane', []):
        try:
            centerline = extract_lane_centerline(lane_data, map_data, node_coords)
            if len(centerline) > 1:
                # Create polygon from centerline
                polygon_coords = compute_lane_polygon_from_centerline(centerline)
                
                argoverse_lane = {
                    'id': lane_data['token'],
                    'centerline': centerline.tolist(),
                    'left_boundary': [],  # Could be computed from lane dividers
                    'right_boundary': [],  # Could be computed from lane dividers
                    'polygon': polygon_coords.tolist() if len(polygon_coords) > 0 else [],
                    'lane_type': lane_data.get('lane_type', 'VEHICLE'),
                    'turn_direction': 'NONE',  # Could be inferred from geometry
                    'predecessors': [],  # Would need connectivity information
                    'successors': []  # Would need connectivity information
                }
                argoverse_data['lanes'].append(argoverse_lane)
        except Exception as e:
            print(f"Warning: Could not convert lane {lane_data.get('token', 'unknown')}: {e}")
            continue
    
    # Convert road polygons
    for polygon_data in map_data.get('polygon', []):
        try:
            coords = polygon_to_coordinates(polygon_data, node_coords)
            if len(coords) > 2:
                argoverse_polygon = {
                    'id': polygon_data['token'],
                    'coordinates': coords.tolist(),
                    'type': 'ROAD_SURFACE'
                }
                argoverse_data['polygons'].append(argoverse_polygon)
        except Exception as e:
            print(f"Warning: Could not convert polygon {polygon_data.get('token', 'unknown')}: {e}")
            continue
    
    # Convert pedestrian crossings
    for crossing_data in map_data.get('ped_crossing', []):
        try:
            coords = polygon_to_coordinates(crossing_data, node_coords)
            if len(coords) > 2:
                argoverse_crossing = {
                    'id': crossing_data['token'],
                    'coordinates': coords.tolist(),
                    'type': 'PEDESTRIAN_CROSSING'
                }
                argoverse_data['pedestrian_crossings'].append(argoverse_crossing)
        except Exception as e:
            print(f"Warning: Could not convert crossing {crossing_data.get('token', 'unknown')}: {e}")
            continue
    
    # Compute map bounds
    if map_data.get('node'):
        all_x = [node['x'] for node in map_data['node']]
        all_y = [node['y'] for node in map_data['node']]
        argoverse_data['bounds'] = {
            'min_x': min(all_x),
            'min_y': min(all_y),
            'max_x': max(all_x),
            'max_y': max(all_y)
        }
    
    return argoverse_data


def save_argoverse_map(argoverse_data: Dict[str, Any], output_path: str) -> None:
    """
    Save Argoverse-format map data to JSON file
    
    Args:
        argoverse_data: Argoverse-compatible map data
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(argoverse_data, f, indent=2)
    
    print(f"Argoverse map saved to: {output_path}")


def get_map_statistics(map_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about the map data
    
    Args:
        map_data: Map data dictionary
        
    Returns:
        Dictionary with map statistics
    """
    stats = {
        'version': map_data.get('version', 'unknown'),
        'total_nodes': len(map_data.get('node', [])),
        'total_polygons': len(map_data.get('polygon', [])),
        'total_lines': len(map_data.get('line', [])),
        'total_lanes': len(map_data.get('lane', [])),
        'total_road_segments': len(map_data.get('road_segment', [])),
        'total_ped_crossings': len(map_data.get('ped_crossing', [])),
        'lane_types': {},
        'bounds': None
    }
    
    # Count lane types
    for lane in map_data.get('lane', []):
        lane_type = lane.get('lane_type', 'UNKNOWN')
        stats['lane_types'][lane_type] = stats['lane_types'].get(lane_type, 0) + 1
    
    # Compute bounds
    if map_data.get('node'):
        all_x = [node['x'] for node in map_data['node']]
        all_y = [node['y'] for node in map_data['node']]
        stats['bounds'] = {
            'min_x': min(all_x),
            'min_y': min(all_y),
            'max_x': max(all_x),
            'max_y': max(all_y),
            'width': max(all_x) - min(all_x),
            'height': max(all_y) - min(all_y)
        }
    
    return stats


def validate_map_data(map_data: Dict[str, Any]) -> List[str]:
    """
    Validate map data and return list of issues found
    
    Args:
        map_data: Map data dictionary
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Check required fields
    if 'version' not in map_data:
        issues.append("Missing 'version' field")
    
    if 'node' not in map_data or not map_data['node']:
        issues.append("No nodes found in map data")
        return issues
    
    # Check node structure
    node_tokens = set()
    for i, node in enumerate(map_data.get('node', [])):
        if 'token' not in node:
            issues.append(f"Node {i} missing 'token' field")
        else:
            if node['token'] in node_tokens:
                issues.append(f"Duplicate node token: {node['token']}")
            node_tokens.add(node['token'])
        
        if 'x' not in node or 'y' not in node:
            issues.append(f"Node {i} missing coordinate fields")
    
    # Check polygon references
    for i, polygon in enumerate(map_data.get('polygon', [])):
        if 'exterior_node_tokens' not in polygon:
            issues.append(f"Polygon {i} missing 'exterior_node_tokens'")
            continue
        
        for node_token in polygon['exterior_node_tokens']:
            if node_token not in node_tokens:
                issues.append(f"Polygon {i} references unknown node: {node_token}")
    
    # Check lane references
    polygon_tokens = {p['token'] for p in map_data.get('polygon', [])}
    for i, lane in enumerate(map_data.get('lane', [])):
        if 'polygon_token' not in lane:
            issues.append(f"Lane {i} missing 'polygon_token'")
        elif lane['polygon_token'] not in polygon_tokens:
            issues.append(f"Lane {i} references unknown polygon: {lane['polygon_token']}")
    
    return issues 