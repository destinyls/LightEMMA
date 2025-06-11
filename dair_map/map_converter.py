"""
Map format conversion utilities for DAIR V2X-Seq dataset
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

from .config import MapConfig, get_config
from .utils import (
    get_node_coordinates_dict,
    polygon_to_coordinates,
    extract_lane_centerline,
    compute_lane_polygon_from_centerline,
    convert_to_argoverse_format,
    save_argoverse_map
)


@dataclass
class ConversionResult:
    """Result of a map conversion operation"""
    success: bool
    output_path: Optional[str] = None
    source_format: Optional[str] = None
    target_format: Optional[str] = None
    elements_converted: Optional[Dict[str, int]] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    
    def __str__(self):
        if self.success:
            return f"Conversion successful: {self.source_format} -> {self.target_format}"
        else:
            return f"Conversion failed: {self.errors}"


class MapConverter:
    """
    Class for converting between different map formats
    """
    
    def __init__(self, config: Optional[MapConfig] = None):
        """
        Initialize the map converter
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or get_config()
        self.supported_conversions = {
            ('v2x_seq', 'argoverse'): self._convert_v2x_seq_to_argoverse,
            ('v2x_seq', 'nuscenes'): self._convert_v2x_seq_to_nuscenes,
            ('argoverse', 'v2x_seq'): self._convert_argoverse_to_v2x_seq,
        }
    
    def get_supported_conversions(self) -> List[Tuple[str, str]]:
        """Get list of supported conversion pairs"""
        return list(self.supported_conversions.keys())
    
    def is_conversion_supported(self, source_format: str, target_format: str) -> bool:
        """Check if conversion between formats is supported"""
        return (source_format, target_format) in self.supported_conversions
    
    def convert_map(self, 
                   map_data: Dict[str, Any],
                   source_format: str,
                   target_format: str,
                   output_path: Optional[str] = None,
                   map_name: str = "converted_map") -> ConversionResult:
        """
        Convert map data between formats
        
        Args:
            map_data: Source map data
            source_format: Source format identifier
            target_format: Target format identifier  
            output_path: Optional output file path
            map_name: Name for the converted map
            
        Returns:
            ConversionResult object
        """
        if not self.is_conversion_supported(source_format, target_format):
            return ConversionResult(
                success=False,
                source_format=source_format,
                target_format=target_format,
                errors=[f"Conversion from {source_format} to {target_format} is not supported"]
            )
        
        try:
            # Get conversion function
            converter_func = self.supported_conversions[(source_format, target_format)]
            
            # Perform conversion
            result_data, warnings = converter_func(map_data, map_name)
            
            # Count converted elements
            elements_converted = self._count_elements(result_data, target_format)
            
            # Save to file if output path specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                output_path_str = str(output_path)
            else:
                output_path_str = None
            
            return ConversionResult(
                success=True,
                output_path=output_path_str,
                source_format=source_format,
                target_format=target_format,
                elements_converted=elements_converted,
                warnings=warnings
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                source_format=source_format,
                target_format=target_format,
                errors=[str(e)]
            )
    
    def _count_elements(self, map_data: Dict[str, Any], format_type: str) -> Dict[str, int]:
        """Count elements in converted map data"""
        counts = {}
        
        if format_type == 'argoverse':
            counts['lanes'] = len(map_data.get('lanes', []))
            counts['polygons'] = len(map_data.get('polygons', []))
            counts['pedestrian_crossings'] = len(map_data.get('pedestrian_crossings', []))
            
        elif format_type == 'v2x_seq':
            counts['nodes'] = len(map_data.get('node', []))
            counts['polygons'] = len(map_data.get('polygon', []))
            counts['lanes'] = len(map_data.get('lane', []))
            counts['ped_crossings'] = len(map_data.get('ped_crossing', []))
            
        elif format_type == 'nuscenes':
            counts['lanes'] = len(map_data.get('lane', []))
            counts['road_segments'] = len(map_data.get('road_segment', []))
            counts['stop_lines'] = len(map_data.get('stop_line', []))
            
        return counts
    
    def _convert_v2x_seq_to_argoverse(self, 
                                     map_data: Dict[str, Any], 
                                     map_name: str) -> Tuple[Dict[str, Any], List[str]]:
        """Convert V2X-Seq format to Argoverse format"""
        warnings = []
        
        # Use existing utility function
        argoverse_data = convert_to_argoverse_format(map_data, map_name)
        
        # Enhanced conversion with additional processing
        node_coords = get_node_coordinates_dict(map_data)
        
        # Process lanes with better connectivity information
        for lane_data in map_data.get('lane', []):
            try:
                lane_token = lane_data['token']
                
                # Find corresponding lane in converted data
                for argoverse_lane in argoverse_data['lanes']:
                    if argoverse_lane['id'] == lane_token:
                        # Add additional lane metadata
                        argoverse_lane['lane_type'] = lane_data.get('lane_type', 'VEHICLE')
                        
                        # Try to infer turn direction from geometry
                        centerline = np.array(argoverse_lane['centerline'])
                        if len(centerline) > 2:
                            turn_direction = self._infer_turn_direction(centerline)
                            argoverse_lane['turn_direction'] = turn_direction
                        
                        break
                        
            except Exception as e:
                warnings.append(f"Could not process lane {lane_data.get('token', 'unknown')}: {e}")
        
        return argoverse_data, warnings
    
    def _convert_v2x_seq_to_nuscenes(self, 
                                    map_data: Dict[str, Any], 
                                    map_name: str) -> Tuple[Dict[str, Any], List[str]]:
        """Convert V2X-Seq format to NuScenes format"""
        warnings = []
        node_coords = get_node_coordinates_dict(map_data)
        
        nuscenes_data = {
            'version': '1.0',
            'map_name': map_name,
            'lane': [],
            'road_segment': [],
            'stop_line': [],
            'ped_crossing': [],
            'road_divider': [],
            'lane_divider': []
        }
        
        # Convert lanes
        for lane_data in map_data.get('lane', []):
            try:
                centerline = extract_lane_centerline(lane_data, map_data, node_coords)
                if len(centerline) > 1:
                    nuscenes_lane = {
                        'token': lane_data['token'],
                        'road_segment_token': '',  # Would need road segment mapping
                        'lane_type': lane_data.get('lane_type', 'car'),
                        'left_lane_divider_tokens': [],
                        'right_lane_divider_tokens': [],
                        'centerline': centerline.tolist()
                    }
                    nuscenes_data['lane'].append(nuscenes_lane)
                    
            except Exception as e:
                warnings.append(f"Could not convert lane {lane_data.get('token', 'unknown')}: {e}")
        
        # Convert pedestrian crossings
        for crossing_data in map_data.get('ped_crossing', []):
            try:
                coords = polygon_to_coordinates(crossing_data, node_coords)
                if len(coords) > 2:
                    nuscenes_crossing = {
                        'token': crossing_data['token'],
                        'polygon': coords.tolist()
                    }
                    nuscenes_data['ped_crossing'].append(nuscenes_crossing)
                    
            except Exception as e:
                warnings.append(f"Could not convert crossing {crossing_data.get('token', 'unknown')}: {e}")
        
        return nuscenes_data, warnings
    
    def _convert_argoverse_to_v2x_seq(self, 
                                     map_data: Dict[str, Any], 
                                     map_name: str) -> Tuple[Dict[str, Any], List[str]]:
        """Convert Argoverse format to V2X-Seq format"""
        warnings = []
        
        v2x_data = {
            'version': '1.3',
            'map_name': map_name,
            'node': [],
            'polygon': [],
            'lane': [],
            'ped_crossing': []
        }
        
        # Generate nodes from lane centerlines and polygons
        node_id_counter = 0
        node_lookup = {}  # (x, y) -> node_token
        
        def get_or_create_node(x: float, y: float) -> str:
            nonlocal node_id_counter
            
            # Round coordinates to avoid duplicate nodes
            key = (round(x, 3), round(y, 3))
            
            if key in node_lookup:
                return node_lookup[key]
            
            node_token = f"node_{node_id_counter:06d}"
            node_id_counter += 1
            
            v2x_data['node'].append({
                'token': node_token,
                'x': float(x),
                'y': float(y)
            })
            
            node_lookup[key] = node_token
            return node_token
        
        # Convert lanes
        for lane_data in map_data.get('lanes', []):
            try:
                centerline = np.array(lane_data['centerline'])
                
                if len(centerline) > 1:
                    # Create polygon from centerline
                    polygon_coords = compute_lane_polygon_from_centerline(centerline)
                    
                    # Create nodes for polygon
                    node_tokens = []
                    for point in polygon_coords:
                        node_token = get_or_create_node(point[0], point[1])
                        node_tokens.append(node_token)
                    
                    # Create polygon
                    polygon_token = f"polygon_{lane_data['id']}"
                    v2x_data['polygon'].append({
                        'token': polygon_token,
                        'exterior_node_tokens': node_tokens
                    })
                    
                    # Create lane
                    v2x_data['lane'].append({
                        'token': lane_data['id'],
                        'polygon_token': polygon_token,
                        'lane_type': lane_data.get('lane_type', 'VEHICLE')
                    })
                    
            except Exception as e:
                warnings.append(f"Could not convert lane {lane_data.get('id', 'unknown')}: {e}")
        
        # Convert pedestrian crossings
        for crossing_data in map_data.get('pedestrian_crossings', []):
            try:
                coords = np.array(crossing_data['coordinates'])
                
                # Create nodes for crossing
                node_tokens = []
                for point in coords:
                    node_token = get_or_create_node(point[0], point[1])
                    node_tokens.append(node_token)
                
                # Create crossing
                v2x_data['ped_crossing'].append({
                    'token': crossing_data['id'],
                    'exterior_node_tokens': node_tokens
                })
                
            except Exception as e:
                warnings.append(f"Could not convert crossing {crossing_data.get('id', 'unknown')}: {e}")
        
        return v2x_data, warnings
    
    def _infer_turn_direction(self, centerline: np.ndarray) -> str:
        """Infer turn direction from lane centerline geometry"""
        if len(centerline) < 3:
            return 'NONE'
        
        # Calculate total angular change
        vectors = np.diff(centerline, axis=0)
        angles = []
        
        for i in range(len(vectors) - 1):
            v1 = vectors[i]
            v2 = vectors[i + 1]
            
            # Skip if vectors are too short
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            
            # Calculate cross product to determine turn direction
            cross = np.cross(v1, v2)
            angle = np.arccos(cos_angle)
            
            if cross < 0:
                angle = -angle
            
            angles.append(angle)
        
        if not angles:
            return 'NONE'
        
        total_angle = sum(angles)
        
        # Threshold for determining turn direction (in radians)
        threshold = np.pi / 6  # 30 degrees
        
        if total_angle > threshold:
            return 'LEFT'
        elif total_angle < -threshold:
            return 'RIGHT'
        else:
            return 'NONE'
    
    def batch_convert(self, 
                     input_dir: str,
                     output_dir: str,
                     source_format: str,
                     target_format: str,
                     pattern: str = "*.json") -> List[ConversionResult]:
        """
        Batch convert multiple map files
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            source_format: Source format identifier
            target_format: Target format identifier
            pattern: File pattern to match
            
        Returns:
            List of ConversionResult objects
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for input_file in input_path.glob(pattern):
            try:
                # Load source map
                with open(input_file, 'r') as f:
                    map_data = json.load(f)
                
                # Determine output filename
                output_file = output_path / f"{input_file.stem}_{target_format}.json"
                
                # Convert map
                result = self.convert_map(
                    map_data=map_data,
                    source_format=source_format,
                    target_format=target_format,
                    output_path=str(output_file),
                    map_name=input_file.stem
                )
                
                results.append(result)
                
            except Exception as e:
                results.append(ConversionResult(
                    success=False,
                    errors=[f"Failed to process {input_file}: {e}"]
                ))
        
        return results
    
    def validate_converted_map(self, 
                              converted_data: Dict[str, Any], 
                              target_format: str) -> List[str]:
        """
        Validate converted map data
        
        Args:
            converted_data: Converted map data
            target_format: Target format identifier
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if target_format == 'argoverse':
            # Validate Argoverse format
            if 'lanes' not in converted_data:
                issues.append("Missing 'lanes' field")
            
            for i, lane in enumerate(converted_data.get('lanes', [])):
                if 'id' not in lane:
                    issues.append(f"Lane {i} missing 'id' field")
                if 'centerline' not in lane or len(lane['centerline']) < 2:
                    issues.append(f"Lane {i} has invalid centerline")
        
        elif target_format == 'v2x_seq':
            # Validate V2X-Seq format
            if 'node' not in converted_data:
                issues.append("Missing 'node' field")
            if 'polygon' not in converted_data:
                issues.append("Missing 'polygon' field")
            
            # Check node references
            node_tokens = {node['token'] for node in converted_data.get('node', [])}
            for i, polygon in enumerate(converted_data.get('polygon', [])):
                if 'exterior_node_tokens' not in polygon:
                    issues.append(f"Polygon {i} missing 'exterior_node_tokens'")
                else:
                    for node_token in polygon['exterior_node_tokens']:
                        if node_token not in node_tokens:
                            issues.append(f"Polygon {i} references unknown node: {node_token}")
        
        return issues
    
    def get_conversion_report(self, results: List[ConversionResult]) -> Dict[str, Any]:
        """
        Generate a summary report for batch conversion results
        
        Args:
            results: List of conversion results
            
        Returns:
            Summary report dictionary
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        all_warnings = []
        all_errors = []
        
        for result in results:
            if result.warnings:
                all_warnings.extend(result.warnings)
            if result.errors:
                all_errors.extend(result.errors)
        
        return {
            'total_conversions': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'total_warnings': len(all_warnings),
            'total_errors': len(all_errors),
            'unique_warnings': list(set(all_warnings)),
            'unique_errors': list(set(all_errors))
        } 