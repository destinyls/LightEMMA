"""
Map data loading and management for DAIR V2X-Seq dataset
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import hashlib
import pickle
from dataclasses import dataclass

from .config import MapConfig, get_config
from .utils import (
    load_map_data, 
    get_node_coordinates_dict,
    polygon_to_coordinates,
    extract_lane_centerline,
    get_map_statistics,
    validate_map_data,
    read_json_file
)


@dataclass
class MapInfo:
    """Container for map metadata and information"""
    name: str
    file_path: str
    format_type: str  # 'v2x_seq', 'argoverse', etc.
    bounds: Optional[Tuple[float, float, float, float]] = None
    statistics: Optional[Dict[str, Any]] = None
    last_modified: Optional[float] = None
    
    def __str__(self):
        return f"MapInfo(name={self.name}, format={self.format_type})"


class MapDataLoader:
    """
    Class for loading and managing map data from various sources and formats
    """
    
    def __init__(self, config: Optional[MapConfig] = None):
        """
        Initialize the map data loader
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or get_config()
        self._available_maps: Dict[str, MapInfo] = {}
        self._loaded_maps: Dict[str, Dict[str, Any]] = {}
        self._node_coords_cache: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        # Discover available maps
        self._discover_maps()
    
    def _discover_maps(self):
        """Discover available maps in the configured directories"""
        self._available_maps.clear()
        
        # Check V2X-Seq format maps (expansion directory)
        if Path(self.config.expansion_dir).exists():
            self._discover_v2x_seq_maps()
        
        # Check Argoverse format maps (maps directory)
        if Path(self.config.maps_dir).exists():
            self._discover_argoverse_maps()
    
    def _discover_v2x_seq_maps(self):
        """Discover V2X-Seq format maps"""
        expansion_dir = Path(self.config.expansion_dir)
        
        for json_file in expansion_dir.glob("*.json"):
            map_name = json_file.stem
            
            map_info = MapInfo(
                name=map_name,
                file_path=str(json_file),
                format_type='v2x_seq',
                last_modified=json_file.stat().st_mtime
            )
            
            self._available_maps[map_name] = map_info
    
    def _discover_argoverse_maps(self):
        """Discover Argoverse format maps"""
        maps_dir = Path(self.config.maps_dir)
        
        # Look for argoverse-style JSON files
        for json_file in maps_dir.glob("*.json"):
            # Skip files in expansion subdirectory
            if "expansion" in str(json_file):
                continue
            
            map_name = json_file.stem
            
            # Check if it's not already discovered as V2X-Seq format
            if map_name not in self._available_maps:
                map_info = MapInfo(
                    name=map_name,
                    file_path=str(json_file),
                    format_type='argoverse',
                    last_modified=json_file.stat().st_mtime
                )
                
                self._available_maps[map_name] = map_info
    
    def get_available_maps(self) -> List[str]:
        """Get list of available map names"""
        return list(self._available_maps.keys())
    
    def get_map_info(self, map_name: str) -> Optional[MapInfo]:
        """Get information about a specific map"""
        return self._available_maps.get(map_name)
    
    def refresh_map_list(self):
        """Refresh the list of available maps"""
        self._discover_maps()
    
    def _get_cache_path(self, map_name: str) -> Path:
        """Get cache file path for a map"""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{map_name}_cache.pkl"
    
    def _get_cache_key(self, map_info: MapInfo) -> str:
        """Generate cache key based on map file and modification time"""
        key_data = f"{map_info.file_path}_{map_info.last_modified}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_from_cache(self, map_name: str) -> Optional[Dict[str, Any]]:
        """Load map data from cache if available and valid"""
        if not self.config.enable_cache:
            return None
        
        cache_path = self._get_cache_path(map_name)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is valid
            map_info = self._available_maps.get(map_name)
            if map_info and cached_data.get('cache_key') == self._get_cache_key(map_info):
                return cached_data['map_data']
        
        except Exception as e:
            print(f"Warning: Could not load cache for {map_name}: {e}")
        
        return None
    
    def _save_to_cache(self, map_name: str, map_data: Dict[str, Any]):
        """Save map data to cache"""
        if not self.config.enable_cache:
            return
        
        try:
            map_info = self._available_maps.get(map_name)
            if not map_info:
                return
            
            cache_data = {
                'cache_key': self._get_cache_key(map_info),
                'map_data': map_data
            }
            
            cache_path = self._get_cache_path(map_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        except Exception as e:
            print(f"Warning: Could not save cache for {map_name}: {e}")
    
    def load_map(self, map_name: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load map data
        
        Args:
            map_name: Name of the map to load
            force_reload: Force reload even if map is cached in memory
            
        Returns:
            Map data dictionary
            
        Raises:
            ValueError: If map is not found
            FileNotFoundError: If map file is missing
        """
        if map_name not in self._available_maps:
            raise ValueError(f"Map '{map_name}' not found. Available maps: {self.get_available_maps()}")
        
        # Check if already loaded in memory
        if not force_reload and map_name in self._loaded_maps:
            return self._loaded_maps[map_name]
        
        map_info = self._available_maps[map_name]
        
        # Try loading from cache first
        if not force_reload:
            cached_data = self._load_from_cache(map_name)
            if cached_data is not None:
                self._loaded_maps[map_name] = cached_data
                return cached_data
        
        # Load from file
        print(f"Loading map: {map_name} ({map_info.format_type} format)")
        
        if not Path(map_info.file_path).exists():
            raise FileNotFoundError(f"Map file not found: {map_info.file_path}")
        
        try:
            if map_info.format_type == 'v2x_seq':
                map_data = self._load_v2x_seq_map(map_info.file_path)
            elif map_info.format_type == 'argoverse':
                map_data = self._load_argoverse_map(map_info.file_path)
            else:
                # Try generic JSON loading
                map_data = load_map_data(map_info.file_path)
            
            # Validate if enabled
            if self.config.validate_on_load:
                issues = validate_map_data(map_data)
                if issues:
                    if self.config.strict_validation:
                        raise ValueError(f"Map validation failed: {issues}")
                    else:
                        print(f"Warning: Map validation issues found: {issues}")
            
            # Compute and cache statistics
            statistics = get_map_statistics(map_data)
            map_info.statistics = statistics
            
            # Compute bounds
            if statistics.get('bounds'):
                bounds = statistics['bounds']
                map_info.bounds = (bounds['min_x'], bounds['min_y'], 
                                 bounds['max_x'], bounds['max_y'])
            
            # Cache the loaded data
            self._loaded_maps[map_name] = map_data
            self._save_to_cache(map_name, map_data)
            
            # Cache node coordinates for V2X-Seq format
            if map_info.format_type == 'v2x_seq':
                self._node_coords_cache[map_name] = get_node_coordinates_dict(map_data)
            
            print(f"Map loaded successfully!")
            print(f"  - Format: {map_info.format_type}")
            print(f"  - Bounds: {map_info.bounds}")
            if statistics:
                print(f"  - Elements: {self._format_statistics(statistics)}")
            
            return map_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load map {map_name}: {e}")
    
    def _load_v2x_seq_map(self, file_path: str) -> Dict[str, Any]:
        """Load V2X-Seq format map"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _load_argoverse_map(self, file_path: str) -> Dict[str, Any]:
        """Load Argoverse format map"""
        return read_json_file(file_path)
    
    def _format_statistics(self, stats: Dict[str, Any]) -> str:
        """Format statistics for display"""
        parts = []
        for key, value in stats.items():
            if key.startswith('total_') and isinstance(value, int):
                element_name = key.replace('total_', '')
                if value > 0:
                    parts.append(f"{element_name}={value}")
        return ", ".join(parts)
    
    def get_map_statistics(self, map_name: str) -> Dict[str, Any]:
        """
        Get statistics for a map
        
        Args:
            map_name: Name of the map
            
        Returns:
            Dictionary with map statistics
        """
        map_info = self.get_map_info(map_name)
        if not map_info:
            raise ValueError(f"Map '{map_name}' not found")
        
        # Return cached statistics if available
        if map_info.statistics:
            return map_info.statistics
        
        # Load map to compute statistics
        map_data = self.load_map(map_name)
        return map_info.statistics or get_map_statistics(map_data)
    
    def get_map_bounds(self, map_name: str) -> Tuple[float, float, float, float]:
        """
        Get bounds for a map
        
        Args:
            map_name: Name of the map
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        map_info = self.get_map_info(map_name)
        if not map_info:
            raise ValueError(f"Map '{map_name}' not found")
        
        # Return cached bounds if available
        if map_info.bounds:
            return map_info.bounds
        
        # Load map to compute bounds
        map_data = self.load_map(map_name)
        return map_info.bounds or (0, 0, 100, 100)
    
    def get_node_coordinates(self, map_name: str) -> Dict[str, Tuple[float, float]]:
        """
        Get node coordinates dictionary for V2X-Seq format maps
        
        Args:
            map_name: Name of the map
            
        Returns:
            Dictionary mapping node tokens to coordinates
        """
        if map_name in self._node_coords_cache:
            return self._node_coords_cache[map_name]
        
        map_data = self.load_map(map_name)
        node_coords = get_node_coordinates_dict(map_data)
        self._node_coords_cache[map_name] = node_coords
        
        return node_coords
    
    def is_map_loaded(self, map_name: str) -> bool:
        """Check if a map is currently loaded in memory"""
        return map_name in self._loaded_maps
    
    def unload_map(self, map_name: str):
        """Unload a map from memory"""
        if map_name in self._loaded_maps:
            del self._loaded_maps[map_name]
        if map_name in self._node_coords_cache:
            del self._node_coords_cache[map_name]
    
    def unload_all_maps(self):
        """Unload all maps from memory"""
        self._loaded_maps.clear()
        self._node_coords_cache.clear()
    
    def get_loaded_maps(self) -> List[str]:
        """Get list of currently loaded map names"""
        return list(self._loaded_maps.keys())
    
    def clear_cache(self, map_name: Optional[str] = None):
        """
        Clear cache files
        
        Args:
            map_name: Specific map name to clear, or None to clear all
        """
        cache_dir = Path(self.config.cache_dir)
        
        if map_name:
            cache_file = self._get_cache_path(map_name)
            if cache_file.exists():
                cache_file.unlink()
                print(f"Cleared cache for {map_name}")
        else:
            # Clear all cache files
            for cache_file in cache_dir.glob("*_cache.pkl"):
                cache_file.unlink()
            print("Cleared all map cache files")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get information about memory usage"""
        import sys
        
        total_size = 0
        map_sizes = {}
        
        for map_name, map_data in self._loaded_maps.items():
            size = sys.getsizeof(map_data)
            map_sizes[map_name] = size
            total_size += size
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'loaded_maps_count': len(self._loaded_maps),
            'map_sizes': map_sizes
        } 