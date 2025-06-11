"""
DAIR Map Processing and Visualization Module

This module provides tools for loading, processing, and visualizing maps from the DAIR V2X-Seq dataset.
It supports both original V2X-Seq format and Argoverse-compatible format.

Main Classes:
    - DAIRMapManager: Main interface for map operations
    - MapVisualizer: Specialized class for map visualization
    - MapDataLoader: Class for loading and processing map data
    - MapConverter: Class for converting between different map formats

Example Usage:
    ```python
    from dair_map import DAIRMapManager
    
    # Initialize the map manager
    manager = DAIRMapManager()
    
    # Load a map
    manager.load_map("yizhuang02")
    
    # Visualize the map
    manager.visualize(elements=['lanes', 'centerlines'], save_path="output.png")
    
    # Get map statistics
    stats = manager.get_statistics()
    ```
"""

from .map_manager import DAIRMapManager
from .map_visualizer import MapVisualizer  
from .map_data_loader import MapDataLoader
from .map_converter import MapConverter
from .config import MapConfig, VisualizationConfig

# Import utility functions for backward compatibility
from .utils import (
    load_map_data,
    get_node_coordinates_dict, 
    polygon_to_coordinates,
    extract_lane_centerline,
    compute_lane_polygon_from_centerline,
    filter_map_elements_by_bbox,
    convert_to_argoverse_format,
    save_argoverse_map,
    get_map_statistics,
    validate_map_data
)

__version__ = "1.0.0"
__author__ = "DAIR Team"

__all__ = [
    # Main classes
    "DAIRMapManager",
    "MapVisualizer", 
    "MapDataLoader",
    "MapConverter",
    "MapConfig",
    "VisualizationConfig",
    
    # Utility functions
    "load_map_data",
    "get_node_coordinates_dict",
    "polygon_to_coordinates", 
    "extract_lane_centerline",
    "compute_lane_polygon_from_centerline",
    "filter_map_elements_by_bbox",
    "convert_to_argoverse_format",
    "save_argoverse_map",
    "get_map_statistics",
    "validate_map_data"
] 