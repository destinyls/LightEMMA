"""
Main DAIR Map Manager - Unified interface for map operations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from .config import MapConfig, VisualizationConfig, get_config
from .map_data_loader import MapDataLoader, MapInfo
from .map_converter import MapConverter, ConversionResult
from .map_visualizer import MapVisualizer
from .utils import (
    get_map_statistics,
    validate_map_data,
    filter_map_elements_by_bbox
)


class DAIRMapManager:
    """
    Main interface for DAIR map operations
    
    This class provides a unified interface for loading, processing, converting,
    and visualizing maps from the DAIR V2X-Seq dataset.
    """
    
    def __init__(self, config: Optional[MapConfig] = None):
        """
        Initialize the DAIR Map Manager
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or get_config()
        
        # Initialize components
        self.data_loader = MapDataLoader(self.config)
        self.converter = MapConverter(self.config)
        self.visualizer = MapVisualizer(self.config)
        
        # Current state
        self.current_map_name: Optional[str] = None
        self.current_map_data: Optional[Dict[str, Any]] = None
        
        print(f"DAIR Map Manager initialized")
        print(f"Available maps: {len(self.get_available_maps())}")
    
    # ========== Map Discovery and Loading ==========
    
    def get_available_maps(self) -> List[str]:
        """Get list of available map names"""
        return self.data_loader.get_available_maps()
    
    def get_map_info(self, map_name: str) -> Optional[MapInfo]:
        """Get information about a specific map"""
        return self.data_loader.get_map_info(map_name)
    
    def refresh_map_list(self):
        """Refresh the list of available maps"""
        self.data_loader.refresh_map_list()
        print(f"Map list refreshed. Available maps: {len(self.get_available_maps())}")
    
    def load_map(self, map_name: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a map
        
        Args:
            map_name: Name of the map to load
            force_reload: Force reload even if already loaded
            
        Returns:
            Map data dictionary
        """
        try:
            map_data = self.data_loader.load_map(map_name, force_reload)
            self.current_map_name = map_name
            self.current_map_data = map_data
            
            # Update visualizer with current map
            self.visualizer.set_current_map(map_name, map_data)
            
            return map_data
            
        except Exception as e:
            print(f"Failed to load map {map_name}: {e}")
            raise
    
    def is_map_loaded(self, map_name: str) -> bool:
        """Check if a map is currently loaded"""
        return self.data_loader.is_map_loaded(map_name)
    
    def unload_map(self, map_name: Optional[str] = None):
        """Unload a map from memory"""
        if map_name:
            self.data_loader.unload_map(map_name)
            if map_name == self.current_map_name:
                self.current_map_name = None
                self.current_map_data = None
                self.visualizer.clear_current_map()
        else:
            # Unload current map
            if self.current_map_name:
                self.unload_map(self.current_map_name)
    
    def unload_all_maps(self):
        """Unload all maps from memory"""
        self.data_loader.unload_all_maps()
        self.current_map_name = None
        self.current_map_data = None
        self.visualizer.clear_current_map()
    
    # ========== Map Information and Statistics ==========
    
    def get_statistics(self, map_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a map
        
        Args:
            map_name: Map name, uses current map if None
            
        Returns:
            Statistics dictionary
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        return self.data_loader.get_map_statistics(target_map)
    
    def get_bounds(self, map_name: Optional[str] = None) -> Tuple[float, float, float, float]:
        """
        Get bounds for a map
        
        Args:
            map_name: Map name, uses current map if None
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        return self.data_loader.get_map_bounds(target_map)
    
    def validate_map(self, map_name: Optional[str] = None) -> List[str]:
        """
        Validate a map and return list of issues
        
        Args:
            map_name: Map name, uses current map if None
            
        Returns:
            List of validation issues
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        map_data = self.data_loader.load_map(target_map)
        return validate_map_data(map_data)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get information about memory usage"""
        return self.data_loader.get_memory_usage()
    
    # ========== Visualization ==========
    
    def visualize(self,
                 map_name: Optional[str] = None,
                 elements: Optional[List[str]] = None,
                 bbox: Optional[Tuple[float, float, float, float]] = None,
                 figsize: Optional[Tuple[int, int]] = None,
                 show_direction: bool = True,
                 arrow_spacing: int = 50,
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 **style_kwargs) -> None:
        """
        Visualize a map
        
        Args:
            map_name: Map name, uses current map if None
            elements: List of elements to visualize
            bbox: Bounding box (min_x, min_y, max_x, max_y)
            figsize: Figure size
            show_direction: Whether to show direction arrows
            arrow_spacing: Spacing between arrows in meters
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **style_kwargs: Additional style parameters
        """
        # Determine target map
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        # Load map if not current
        if target_map != self.current_map_name:
            self.load_map(target_map)
        
        # Create visualization config
        viz_config = VisualizationConfig(
            elements=elements or self.config.default_elements,
            bbox=bbox,
            figsize=figsize or self.config.default_figsize,
            show_direction=show_direction,
            arrow_spacing=arrow_spacing,
            save_path=save_path,
            show_plot=show_plot
        )
        
        # Merge with base config and apply style overrides
        viz_config.merge_with_base_config(self.config)
        for element, styles in style_kwargs.items():
            if element in viz_config.style_overrides:
                viz_config.style_overrides[element].update(styles)
        
        # Perform visualization
        self.visualizer.visualize_with_config(viz_config)
    
    def create_distance_visualization(self,
                                    center: Tuple[float, float],
                                    distance: Union[int, float],
                                    map_name: Optional[str] = None,
                                    show_direction: bool = True,
                                    save_path: Optional[str] = None) -> str:
        """
        Create distance-limited visualization
        
        Args:
            center: Center point (x, y)
            distance: Distance range in meters
            map_name: Map name, uses current map if None
            show_direction: Whether to show direction arrows
            save_path: Path to save the figure
            
        Returns:
            Path to the saved visualization
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        # Load map if not current
        if target_map != self.current_map_name:
            self.load_map(target_map)
        
        # Calculate bounding box
        center_x, center_y = center
        half_dist = distance / 2
        bbox = (center_x - half_dist, center_y - half_dist,
                center_x + half_dist, center_y + half_dist)
        
        # Generate save path if not provided
        if not save_path:
            output_dir = Path(self.config.output_dir)
            suffix = "_with_arrows" if show_direction else "_no_arrows"
            save_path = str(output_dir / f"distance_{target_map}_{distance}m{suffix}.png")
        
        # Create visualization
        self.visualize(
            map_name=target_map,
            elements=['lanes', 'centerlines', 'crosswalks', 'stoplines'],
            bbox=bbox,
            show_direction=show_direction,
            save_path=save_path,
            show_plot=False
        )
        
        return save_path
    
    def create_region_comparison(self,
                               bbox: Tuple[float, float, float, float],
                               map_name: Optional[str] = None,
                               save_path: Optional[str] = None) -> str:
        """
        Create a comparison visualization showing different element types
        
        Args:
            bbox: Bounding box (min_x, min_y, max_x, max_y)
            map_name: Map name, uses current map if None
            save_path: Path to save the figure
            
        Returns:
            Path to the saved visualization
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        # Load map if not current
        if target_map != self.current_map_name:
            self.load_map(target_map)
        
        # Generate save path if not provided
        if not save_path:
            output_dir = Path(self.config.output_dir)
            save_path = str(output_dir / f"comparison_{target_map}.png")
        
        # Create comparison visualization
        self.visualizer.create_comparison_plot(bbox, save_path)
        
        return save_path
    
    # ========== Map Conversion ==========
    
    def convert_map(self,
                   target_format: str,
                   map_name: Optional[str] = None,
                   output_path: Optional[str] = None) -> ConversionResult:
        """
        Convert current or specified map to target format
        
        Args:
            target_format: Target format ('argoverse', 'nuscenes', etc.)
            map_name: Map name, uses current map if None
            output_path: Output file path
            
        Returns:
            ConversionResult object
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        # Load map data
        map_data = self.data_loader.load_map(target_map)
        map_info = self.data_loader.get_map_info(target_map)
        
        if not map_info:
            raise ValueError(f"Map info not found for {target_map}")
        
        # Generate output path if not provided
        if not output_path:
            output_dir = Path(self.config.output_dir)
            output_path = str(output_dir / f"{target_map}_{target_format}.json")
        
        # Perform conversion
        return self.converter.convert_map(
            map_data=map_data,
            source_format=map_info.format_type,
            target_format=target_format,
            output_path=output_path,
            map_name=target_map
        )
    
    def batch_convert_maps(self,
                          target_format: str,
                          map_names: Optional[List[str]] = None,
                          output_dir: Optional[str] = None) -> List[ConversionResult]:
        """
        Convert multiple maps to target format
        
        Args:
            target_format: Target format
            map_names: List of map names, uses all available if None
            output_dir: Output directory
            
        Returns:
            List of ConversionResult objects
        """
        target_maps = map_names or self.get_available_maps()
        results = []
        
        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for map_name in target_maps:
            try:
                output_file = output_path / f"{map_name}_{target_format}.json"
                result = self.convert_map(
                    target_format=target_format,
                    map_name=map_name,
                    output_path=str(output_file)
                )
                results.append(result)
                
            except Exception as e:
                results.append(ConversionResult(
                    success=False,
                    errors=[f"Failed to convert {map_name}: {e}"]
                ))
        
        return results
    
    # ========== Map Processing ==========
    
    def filter_map_by_region(self,
                            bbox: Tuple[float, float, float, float],
                            map_name: Optional[str] = None,
                            element_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Filter map elements by bounding box
        
        Args:
            bbox: Bounding box (min_x, min_y, max_x, max_y)
            map_name: Map name, uses current map if None
            element_types: Element types to filter
            
        Returns:
            Filtered map data
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        map_data = self.data_loader.load_map(target_map)
        
        return filter_map_elements_by_bbox(
            map_data=map_data,
            bbox=bbox,
            element_types=element_types
        )
    
    def get_lanes_in_region(self,
                           bbox: Tuple[float, float, float, float],
                           map_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get lanes within a bounding box
        
        Args:
            bbox: Bounding box (min_x, min_y, max_x, max_y)
            map_name: Map name, uses current map if None
            
        Returns:
            List of lane data dictionaries
        """
        target_map = map_name or self.current_map_name
        if not target_map:
            raise ValueError("No map specified and no current map loaded")
        
        # Load map if not current
        if target_map != self.current_map_name:
            self.load_map(target_map)
        
        return self.visualizer.get_lanes_in_region(bbox)
    
    # ========== Configuration Management ==========
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        self.config.update(**kwargs)
        
        # Update components with new config
        self.data_loader.config = self.config
        self.converter.config = self.config
        self.visualizer.config = self.config
    
    def get_config(self) -> MapConfig:
        """Get current configuration"""
        return self.config
    
    def reset_config(self):
        """Reset configuration to defaults"""
        from .config import MapConfig
        self.config = MapConfig()
        self.update_config()
    
    # ========== Cache Management ==========
    
    def clear_cache(self, map_name: Optional[str] = None):
        """Clear cache files"""
        self.data_loader.clear_cache(map_name)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        cache_dir = Path(self.config.cache_dir)
        
        if not cache_dir.exists():
            return {'cache_exists': False}
        
        cache_files = list(cache_dir.glob("*_cache.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_exists': True,
            'cache_dir': str(cache_dir),
            'cache_files_count': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        }
    
    # ========== Utility Methods ==========
    
    def print_summary(self):
        """Print a summary of available maps and current state"""
        print("\n" + "="*60)
        print("DAIR Map Manager Summary")
        print("="*60)
        
        # Available maps
        available_maps = self.get_available_maps()
        print(f"Available maps: {len(available_maps)}")
        for i, map_name in enumerate(available_maps[:10], 1):  # Show first 10
            map_info = self.get_map_info(map_name)
            format_type = map_info.format_type if map_info else "unknown"
            status = "loaded" if self.is_map_loaded(map_name) else "not loaded"
            print(f"  {i:2d}. {map_name} ({format_type}) - {status}")
        
        if len(available_maps) > 10:
            print(f"  ... and {len(available_maps) - 10} more")
        
        # Current map
        print(f"\nCurrent map: {self.current_map_name or 'None'}")
        
        if self.current_map_name:
            stats = self.get_statistics()
            bounds = self.get_bounds()
            print(f"Map bounds: ({bounds[0]:.1f}, {bounds[1]:.1f}) -> ({bounds[2]:.1f}, {bounds[3]:.1f})")
            print(f"Elements: {self._format_stats(stats)}")
        
        # Memory usage
        memory_info = self.get_memory_usage()
        print(f"\nMemory usage: {memory_info['total_size_mb']:.1f} MB")
        print(f"Loaded maps: {memory_info['loaded_maps_count']}")
        
        # Cache info
        cache_info = self.get_cache_info()
        if cache_info['cache_exists']:
            print(f"Cache: {cache_info['cache_files_count']} files, {cache_info['total_size_mb']:.1f} MB")
        else:
            print("Cache: empty")
        
        print("="*60)
    
    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """Format statistics for display"""
        parts = []
        for key, value in stats.items():
            if key.startswith('total_') and isinstance(value, int) and value > 0:
                element_name = key.replace('total_', '')
                parts.append(f"{element_name}={value}")
        return ", ".join(parts)
    
    def __str__(self):
        return f"DAIRMapManager(maps={len(self.get_available_maps())}, current={self.current_map_name})"
    
    def __repr__(self):
        return self.__str__() 