"""
DAIR Map Visualizer for V2X-Seq dataset using Argoverse format
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
from typing import Dict, List, Tuple, Optional, Any, Union
import os
from pathlib import Path
import re

# Handle both direct execution and module import
try:
    from .config import MapConfig, VisualizationConfig, get_config
except ImportError:
    # Direct execution fallback
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import MapConfig, VisualizationConfig, get_config


def interp_arc(t: int, points: np.ndarray) -> np.ndarray:
    """
    Interpolate points along an arc - local implementation
    
    Args:
        t: Number of interpolated points
        points: Original points array
        
    Returns:
        Interpolated points array
    """
    if len(points) < 2:
        return points
    
    # Simple linear interpolation
    from scipy.interpolate import interp1d
    
    # Calculate cumulative distances
    distances = [0]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i-1])
        distances.append(distances[-1] + dist)
    
    # Normalize distances to [0, 1]
    total_distance = distances[-1]
    if total_distance == 0:
        return points
    
    normalized_distances = np.array(distances) / total_distance
    
    # Create interpolation functions
    f_x = interp1d(normalized_distances, points[:, 0], kind='linear')
    f_y = interp1d(normalized_distances, points[:, 1], kind='linear')
    
    # Generate new parameter values
    new_params = np.linspace(0, 1, t)
    
    # Interpolate
    new_x = f_x(new_params)
    new_y = f_y(new_params)
    
    return np.column_stack([new_x, new_y])


class MapVisualizer:
    """
    Map visualizer for DAIR V2X-seq dataset using Argoverse format
    """
    
    def __init__(self, config: Optional[MapConfig] = None):
        """
        Initialize the map visualizer
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or get_config()
        
        # Current state
        self.current_map_data: Optional[Dict[str, Any]] = None
        self.current_map_name: Optional[str] = None
        
        # Performance optimization caches
        self._junction_paths_cache: Optional[List] = None
        self._parsed_coords_cache: Dict[str, np.ndarray] = {}
        
    def set_current_map(self, map_name: str, map_data: Dict[str, Any]):
        """Set the current map data and clear caches"""
        self.current_map_name = map_name
        self.current_map_data = map_data
        # Clear caches when map changes
        self._junction_paths_cache = None
        self._parsed_coords_cache.clear()
    
    def clear_current_map(self):
        """Clear the current map data and caches"""
        self.current_map_name = None
        self.current_map_data = None
        self._junction_paths_cache = None
        self._parsed_coords_cache.clear()
    
    def _parse_coordinate(self, coord_str: str) -> Tuple[float, float]:
        """
        Parse coordinate string in format "(x, y)" to float tuple
        
        Args:
            coord_str: Coordinate string like "(2818.276806, -118.348421)"
            
        Returns:
            Tuple of (x, y) coordinates
        """
        # Remove parentheses and split by comma
        coord_str = coord_str.strip('()')
        x_str, y_str = coord_str.split(',')
        return float(x_str.strip()), float(y_str.strip())
    
    def _parse_coordinate_list(self, coord_list: List[str]) -> np.ndarray:
        """
        Parse list of coordinate strings to numpy array with caching
        
        Args:
            coord_list: List of coordinate strings
            
        Returns:
            Numpy array of shape (N, 2) with coordinates
        """
        # Create cache key from coordinate list
        cache_key = str(hash(tuple(coord_list)))
        
        if cache_key in self._parsed_coords_cache:
            return self._parsed_coords_cache[cache_key]
        
        coordinates = []
        for coord_str in coord_list:
            x, y = self._parse_coordinate(coord_str)
            coordinates.append([x, y])
        
        result = np.array(coordinates)
        self._parsed_coords_cache[cache_key] = result
        return result
    
    def _get_junction_paths(self):
        """
        Get cached junction paths for efficient point-in-polygon tests
        
        Returns:
            List of matplotlib.path.Path objects for all junctions
        """
        if self._junction_paths_cache is not None:
            return self._junction_paths_cache
        
        if not self.current_map_data or 'JUNCTION' not in self.current_map_data:
            self._junction_paths_cache = []
            return self._junction_paths_cache
        
        from matplotlib.path import Path as MplPath
        junction_paths = []
        
        for junction_id, junction_data in self.current_map_data.get('JUNCTION', {}).items():
            try:
                # Get junction polygon coordinates
                if 'polygon' in junction_data:
                    coords = self._parse_coordinate_list(junction_data['polygon'])
                elif 'boundary' in junction_data:
                    coords = self._parse_coordinate_list(junction_data['boundary'])
                else:
                    continue
                
                if len(coords) > 2:
                    path = MplPath(coords)
                    junction_paths.append(path)
                        
            except (ValueError, KeyError):
                continue
        
        self._junction_paths_cache = junction_paths
        return self._junction_paths_cache
    
    def visualize_with_config(self, viz_config: VisualizationConfig):
        """
        Visualize map using configuration object
        
        Args:
            viz_config: Visualization configuration
        """
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        fig, ax = plt.subplots(1, 1, figsize=viz_config.figsize)
        
        # Get styles from config
        styles = viz_config.style_overrides
        
        # Visualize elements based on configuration
        if 'lanes' in viz_config.elements:
            lane_style = styles.get('lanes', self.config.get_visualization_style('lanes'))
            if viz_config.show_direction:
                self._plot_lanes_with_direction(ax, **lane_style)
            else:
                self._plot_lanes(ax, **lane_style)
        
        if 'centerlines' in viz_config.elements:
            centerline_style = styles.get('centerlines', self.config.get_visualization_style('centerlines'))
            
            # Get junction filtering settings from config or use base config defaults
            filter_arrows = viz_config.filter_arrows_in_junctions
            if filter_arrows is None:
                filter_arrows = self.config.filter_arrows_in_junctions
            
            junction_buffer = viz_config.arrow_junction_buffer  
            if junction_buffer is None:
                junction_buffer = self.config.arrow_junction_buffer
                
            self._plot_centerlines(ax, show_direction=viz_config.show_direction, 
                                 arrow_spacing=viz_config.arrow_spacing,
                                 arrow_size=viz_config.arrow_size,
                                 max_arrows_per_lane=viz_config.max_arrows_per_lane,
                                 skip_small_lanes=viz_config.skip_small_lanes,
                                 filter_arrows_in_junctions=filter_arrows,
                                 arrow_junction_buffer=junction_buffer,
                                 **centerline_style)
        
        if 'crosswalks' in viz_config.elements:
            crosswalk_style = styles.get('crosswalks', self.config.get_visualization_style('crosswalks'))
            self._plot_crosswalks(ax, **crosswalk_style)
        
        if 'stoplines' in viz_config.elements:
            stopline_style = styles.get('stoplines', self.config.get_visualization_style('stoplines'))
            self._plot_stoplines(ax, **stopline_style)
        
        if 'junctions' in viz_config.elements:
            junction_style = styles.get('junctions', self.config.get_visualization_style('junctions'))
            self._plot_junctions(ax, **junction_style)
        
        # Set view
        if viz_config.bbox:
            ax.set_xlim(viz_config.bbox[0], viz_config.bbox[2])
            ax.set_ylim(viz_config.bbox[1], viz_config.bbox[3])
        else:
            self._auto_fit_view(ax)
        
        ax.set_aspect('equal')
        title = f'DAIR Map Visualization: {self.current_map_name}'
        if viz_config.show_direction:
            title += ' (with Direction Arrows)'
        ax.set_title(title)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        self._add_legend(ax, viz_config.elements, viz_config.show_direction)
        
        plt.tight_layout()
        
        if viz_config.save_path:
            plt.savefig(viz_config.save_path, dpi=viz_config.dpi, bbox_inches='tight')
            print(f"Visualization saved to: {viz_config.save_path}")
        
        if viz_config.show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_comparison_plot(self, bbox: Tuple[float, float, float, float], save_path: str):
        """
        Create a comparison plot showing different element types
        
        Args:
            bbox: Bounding box (min_x, min_y, max_x, max_y)
            save_path: Path to save the plot
        """
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f'{self.current_map_name} - Element Comparison', fontsize=16)
        
        # Plot 1: Lanes only
        lane_style = self.config.get_visualization_style('lanes')
        self._plot_lanes(axes[0, 0], **lane_style)
        axes[0, 0].set_xlim(bbox[0], bbox[2])
        axes[0, 0].set_ylim(bbox[1], bbox[3])
        axes[0, 0].set_title('Lanes')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Centerlines only
        centerline_style = self.config.get_visualization_style('centerlines')
        self._plot_centerlines(axes[0, 1], show_direction=True, **centerline_style)
        axes[0, 1].set_xlim(bbox[0], bbox[2])
        axes[0, 1].set_ylim(bbox[1], bbox[3])
        axes[0, 1].set_title('Centerlines with Direction')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Crosswalks only
        crosswalk_style = self.config.get_visualization_style('crosswalks')
        self._plot_crosswalks(axes[1, 0], **crosswalk_style)
        axes[1, 0].set_xlim(bbox[0], bbox[2])
        axes[1, 0].set_ylim(bbox[1], bbox[3])
        axes[1, 0].set_title('Crosswalks')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: All elements combined
        self._plot_lanes(axes[1, 1], alpha=0.3, **lane_style)
        self._plot_centerlines(axes[1, 1], show_direction=True, **centerline_style)
        self._plot_crosswalks(axes[1, 1], **crosswalk_style)
        stopline_style = self.config.get_visualization_style('stoplines')
        self._plot_stoplines(axes[1, 1], **stopline_style)
        axes[1, 1].set_xlim(bbox[0], bbox[2])
        axes[1, 1].set_ylim(bbox[1], bbox[3])
        axes[1, 1].set_title('All Elements Combined')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.default_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {save_path}")
    
    def visualize_map(self, 
                     map_name: Optional[str] = None,
                     elements: List[str] = None,
                     bbox: Optional[Tuple[float, float, float, float]] = None,
                     figsize: Tuple[int, int] = (15, 15),
                     save_path: Optional[str] = None,
                     show_direction: bool = True,
                     arrow_spacing: int = 50) -> None:
        """
        Legacy method for backward compatibility
        """
        # Create visualization config from parameters
        viz_config = VisualizationConfig(
            elements=elements or self.config.default_elements,
            bbox=bbox,
            figsize=figsize,
            show_direction=show_direction,
            arrow_spacing=arrow_spacing,
            save_path=save_path,
            show_plot=True
        )
        
        viz_config.merge_with_base_config(self.config)
        self.visualize_with_config(viz_config)
        
    def _add_legend(self, ax, elements: List[str], show_direction: bool):
        """Add legend to the plot"""
        legend_elements = []
        
        if 'lanes' in elements:
            legend_elements.append(patches.Patch(color=self.config.lane_color, label='Lanes'))
        
        if 'centerlines' in elements:
            if show_direction:
                legend_elements.append(plt.Line2D([0], [0], color=self.config.centerline_color, 
                                                linewidth=2, label='Centerlines with Direction'))
            else:
                legend_elements.append(plt.Line2D([0], [0], color=self.config.centerline_color, 
                                                linewidth=2, label='Centerlines'))
        
        if 'crosswalks' in elements:
            legend_elements.append(patches.Patch(color=self.config.crosswalk_color, label='Crosswalks'))
        
        if 'stoplines' in elements:
            legend_elements.append(plt.Line2D([0], [0], color=self.config.stopline_color, 
                                            linewidth=3, label='Stop Lines'))
        
        if 'junctions' in elements:
            legend_elements.append(patches.Patch(color=self.config.junction_color, label='Junctions'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_lanes(self, ax, **kwargs):
        """Plot lane polygons using boundary information"""
        if not self.current_map_data:
            return
            
        lane_polygons = []
        
        for lane_id, lane_data in self.current_map_data.get('LANE', {}).items():
            try:
                # Get centerline, left and right boundaries
                centerline = self._parse_coordinate_list(lane_data.get('centerline', []))
                left_boundary = self._parse_coordinate_list(lane_data.get('left_boundary', []))
                right_boundary = self._parse_coordinate_list(lane_data.get('right_boundary', []))
                
                # Create polygon from boundaries if available
                if len(left_boundary) > 0 and len(right_boundary) > 0:
                    # Combine left boundary + reversed right boundary to form polygon
                    polygon_coords = np.vstack([left_boundary, right_boundary[::-1]])
                    if len(polygon_coords) > 2:
                        polygon = patches.Polygon(polygon_coords, closed=True, **kwargs)
                        lane_polygons.append(polygon)
                elif len(centerline) > 1:
                    # Fallback: create thin polygon around centerline
                    # Calculate perpendicular vectors for lane width
                    lane_width = self.config.default_lane_width
                    
                    left_coords = []
                    right_coords = []
                    
                    for i in range(len(centerline)):
                        if i == 0:
                            # First point: use direction to next point
                            direction = centerline[i+1] - centerline[i]
                        elif i == len(centerline) - 1:
                            # Last point: use direction from previous point
                            direction = centerline[i] - centerline[i-1]
                        else:
                            # Middle point: use average direction
                            direction = centerline[i+1] - centerline[i-1]
                        
                        # Normalize direction and get perpendicular
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            perpendicular = np.array([-direction[1], direction[0]])
                            
                            left_coords.append(centerline[i] + perpendicular * lane_width / 2)
                            right_coords.append(centerline[i] - perpendicular * lane_width / 2)
                    
                    if len(left_coords) > 0:
                        polygon_coords = np.vstack([left_coords, right_coords[::-1]])
                        polygon = patches.Polygon(polygon_coords, closed=True, **kwargs)
                        lane_polygons.append(polygon)
                        
            except (ValueError, KeyError) as e:
                continue  # Skip problematic lanes
        
        if lane_polygons:
            collection = PatchCollection(lane_polygons, match_original=True)
            ax.add_collection(collection)
    
    def _plot_centerlines(self, ax, show_direction=True, arrow_spacing=50, arrow_size=10, 
                         max_arrows_per_lane=None, skip_small_lanes=False, 
                         filter_arrows_in_junctions=None, arrow_junction_buffer=None, **kwargs):
        """Plot lane centerlines with optional direction arrows (performance optimized)"""
        if not self.current_map_data:
            return
        
        # Use config defaults if parameters not provided
        if filter_arrows_in_junctions is None:
            filter_arrows_in_junctions = self.config.filter_arrows_in_junctions
        if arrow_junction_buffer is None:
            arrow_junction_buffer = self.config.arrow_junction_buffer
            
        centerlines = []
        total_lanes = len(self.current_map_data.get('LANE', {}))
        processed_lanes = 0
        
        print(f"📊 Processing {total_lanes} lanes for centerline visualization...")
        if not filter_arrows_in_junctions:
            print("   🔄 Arrow junction filtering is DISABLED")
        else:
            print(f"   🛡️  Arrow junction filtering is ENABLED (buffer: {arrow_junction_buffer}m)")
        
        for lane_id, lane_data in self.current_map_data.get('LANE', {}).items():
            try:
                centerline = self._parse_coordinate_list(lane_data.get('centerline', []))
                if len(centerline) > 1:
                    # Performance optimization: skip very short lanes if requested
                    if skip_small_lanes:
                        lane_length = self._calculate_lane_length(centerline)
                        if lane_length < arrow_spacing:
                            continue
                    
                    centerlines.append(centerline)
                    
                    # Add direction arrows if requested
                    if show_direction:
                        arrow_style = self.config.get_visualization_style('arrows')
                        self._add_direction_arrows(ax, centerline, arrow_spacing, arrow_size, 
                                                 max_arrows_per_lane=max_arrows_per_lane,
                                                 filter_arrows_in_junctions=filter_arrows_in_junctions,
                                                 arrow_junction_buffer=arrow_junction_buffer,
                                                 **arrow_style)
                        
                processed_lanes += 1
                     
            except (ValueError, KeyError):
                continue
        
        print(f"✅ Completed processing {processed_lanes} lanes")
        
        if centerlines:
            line_collection = LineCollection(centerlines, **kwargs)
            ax.add_collection(line_collection)
    
    def _calculate_lane_length(self, centerline: np.ndarray) -> float:
        """Calculate total length of a lane centerline"""
        if len(centerline) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(centerline)):
            total_length += np.linalg.norm(centerline[i] - centerline[i-1])
        return total_length
    
    def _is_point_in_junction(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside any junction (optimized with caching)
        
        Args:
            point: 2D point as numpy array [x, y]
            
        Returns:
            True if the point is inside any junction, False otherwise
        """
        junction_paths = self._get_junction_paths()
        
        for path in junction_paths:
            if path.contains_point(point):
                return True
        
        return False
    
    def _are_points_in_junctions(self, points: np.ndarray) -> np.ndarray:
        """
        Check which points are inside junctions (batch operation for efficiency)
        
        Args:
            points: Array of points with shape (N, 2)
            
        Returns:
            Boolean array indicating which points are in junctions
        """
        if len(points) == 0:
            return np.array([], dtype=bool)
        
        junction_paths = self._get_junction_paths()
        if not junction_paths:
            return np.zeros(len(points), dtype=bool)
        
        # Use batch contains_points for efficiency
        in_junction = np.zeros(len(points), dtype=bool)
        
        for path in junction_paths:
            # contains_points is faster for multiple points
            in_this_junction = path.contains_points(points)
            in_junction |= in_this_junction
        
        return in_junction

    def _add_direction_arrows(self, ax, centerline, spacing=50, arrow_size=10, 
                             max_arrows_per_lane=None, filter_arrows_in_junctions=True,
                             arrow_junction_buffer=0.0, **kwargs):
        """
        Add direction arrows along a centerline, with configurable junction filtering (optimized)
        
        Args:
            ax: Matplotlib axis
            centerline: Array of centerline points
            spacing: Distance between arrows in meters
            arrow_size: Size of arrow heads
            max_arrows_per_lane: Maximum number of arrows per lane (for performance)
            filter_arrows_in_junctions: Whether to filter arrows inside junctions
            arrow_junction_buffer: Buffer distance around junctions (meters)
        """
        if len(centerline) < 2:
            return
        
        # Calculate cumulative distances along centerline
        distances = [0]
        for i in range(1, len(centerline)):
            dist = np.linalg.norm(centerline[i] - centerline[i-1])
            distances.append(distances[-1] + dist)
        
        total_length = distances[-1]
        if total_length < spacing:
            return
        
        # Place arrows at regular intervals
        arrow_positions = np.arange(spacing/2, total_length - spacing/2, spacing)
        
        # Apply max arrows per lane limit
        if max_arrows_per_lane and len(arrow_positions) > max_arrows_per_lane:
            # Sample evenly distributed arrows
            indices = np.linspace(0, len(arrow_positions)-1, max_arrows_per_lane, dtype=int)
            arrow_positions = arrow_positions[indices]
        
        if len(arrow_positions) == 0:
            return
        
        # Pre-calculate all arrow points and directions for batch processing
        arrow_points = []
        arrow_directions = []
        
        for pos in arrow_positions:
            # Find the segment containing this position
            segment_idx = np.searchsorted(distances, pos) - 1
            if segment_idx < 0 or segment_idx >= len(centerline) - 1:
                continue
            
            # Interpolate position within the segment
            seg_start_dist = distances[segment_idx]
            seg_end_dist = distances[segment_idx + 1]
            
            if seg_end_dist - seg_start_dist == 0:
                continue
                
            ratio = (pos - seg_start_dist) / (seg_end_dist - seg_start_dist)
            
            # Interpolate position
            point = centerline[segment_idx] + ratio * (centerline[segment_idx + 1] - centerline[segment_idx])
            
            # Calculate direction vector
            direction = centerline[segment_idx + 1] - centerline[segment_idx]
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                direction = direction / direction_norm
                arrow_points.append(point)
                arrow_directions.append(direction)
        
        if not arrow_points:
            return
        
        # Apply junction filtering if enabled
        if filter_arrows_in_junctions:
            # Batch check which points are in junctions (with buffer if specified)
            arrow_points_array = np.array(arrow_points)
            
            if arrow_junction_buffer > 0:
                # Apply buffer around junctions - check points displaced by buffer distance
                in_junctions = np.zeros(len(arrow_points), dtype=bool)
                for i, (point, direction) in enumerate(zip(arrow_points, arrow_directions)):
                    # Check original point and points displaced by buffer in multiple directions
                    test_points = [
                        point,
                        point + np.array([arrow_junction_buffer, 0]),
                        point - np.array([arrow_junction_buffer, 0]),
                        point + np.array([0, arrow_junction_buffer]),
                        point - np.array([0, arrow_junction_buffer])
                    ]
                    in_junctions[i] = any(self._is_point_in_junction(tp) for tp in test_points)
            else:
                # Standard junction filtering without buffer
                in_junctions = self._are_points_in_junctions(arrow_points_array)
        else:
            # No junction filtering - allow all arrows
            in_junctions = np.zeros(len(arrow_points), dtype=bool)
        
        # Draw arrows (skip those in junctions if filtering is enabled)
        arrow_color = kwargs.get('color', self.config.arrow_color)
        
        drawn_arrows = 0
        for i, (point, direction) in enumerate(zip(arrow_points, arrow_directions)):
            if not in_junctions[i]:  # Draw if not in junction or filtering disabled
                ax.annotate('', 
                           xy=point + direction * arrow_size/2, 
                           xytext=point - direction * arrow_size/2,
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2, alpha=0.8))
                drawn_arrows += 1
        
        # Optional: log filtering results for debugging
        if filter_arrows_in_junctions and len(arrow_points) > 0:
            filtered_count = len(arrow_points) - drawn_arrows
            if filtered_count > 0:
                pass  # Uncomment for debugging: print(f"      Filtered {filtered_count}/{len(arrow_points)} arrows in junctions")
    
    def _plot_lanes_with_direction(self, ax, show_direction=True, **kwargs):
        """Plot lane polygons with optional direction indicators"""
        self._plot_lanes(ax, **kwargs)
        
        if show_direction:
            # Also add direction arrows for lane boundaries
            for lane_id, lane_data in self.current_map_data.get('LANE', {}).items():
                try:
                    centerline = self._parse_coordinate_list(lane_data.get('centerline', []))
                    if len(centerline) > 1:
                        # Add fewer, larger arrows for lane polygons
                        arrow_style = self.config.get_visualization_style('arrows')
                        self._add_direction_arrows(ax, centerline, spacing=100, arrow_size=15, 
                                                 color='darkblue', alpha=0.6)
                except (ValueError, KeyError):
                    continue
    
    def _plot_crosswalks(self, ax, **kwargs):
        """Plot pedestrian crosswalks"""
        if not self.current_map_data:
            return
            
        crosswalk_polygons = []
        
        for crosswalk_id, crosswalk_data in self.current_map_data.get('CROSSWALK', {}).items():
            try:
                # Crosswalks might have polygon information or boundary points
                if 'polygon' in crosswalk_data:
                    coords = self._parse_coordinate_list(crosswalk_data['polygon'])
                elif 'boundary' in crosswalk_data:
                    coords = self._parse_coordinate_list(crosswalk_data['boundary'])
                else:
                    continue
                
                if len(coords) > 2:
                    polygon = patches.Polygon(coords, closed=True, **kwargs)
                    crosswalk_polygons.append(polygon)
                    
            except (ValueError, KeyError):
                continue
        
        if crosswalk_polygons:
            collection = PatchCollection(crosswalk_polygons, match_original=True)
            ax.add_collection(collection)
    
    def _plot_stoplines(self, ax, **kwargs):
        """Plot stop lines"""
        if not self.current_map_data:
            return
            
        stoplines = []
        
        for stopline_id, stopline_data in self.current_map_data.get('STOPLINE', {}).items():
            try:
                # Stop lines are typically line segments
                if 'polygon' in stopline_data:
                    coords = self._parse_coordinate_list(stopline_data['polygon'])
                elif 'line' in stopline_data:
                    coords = self._parse_coordinate_list(stopline_data['line'])
                else:
                    continue
                
                if len(coords) > 1:
                    stoplines.append(coords)
                    
            except (ValueError, KeyError):
                continue
        
        if stoplines:
            line_collection = LineCollection(stoplines, **kwargs)
            ax.add_collection(line_collection)
    
    def _plot_junctions(self, ax, **kwargs):
        """Plot junctions"""
        if not self.current_map_data:
            return
            
        junction_polygons = []
        
        for junction_id, junction_data in self.current_map_data.get('JUNCTION', {}).items():
            try:
                # Junctions might have polygon boundary
                if 'polygon' in junction_data:
                    coords = self._parse_coordinate_list(junction_data['polygon'])
                elif 'boundary' in junction_data:
                    coords = self._parse_coordinate_list(junction_data['boundary'])
                else:
                    continue
                
                if len(coords) > 2:
                    polygon = patches.Polygon(coords, closed=True, **kwargs)
                    junction_polygons.append(polygon)
                    
            except (ValueError, KeyError):
                continue
        
        if junction_polygons:
            collection = PatchCollection(junction_polygons, match_original=True)
            ax.add_collection(collection)
    
    def _auto_fit_view(self, ax):
        """Auto-fit the view to show all data"""
        if not self.current_map_data:
            return
            
        all_x, all_y = [], []
        
        # Collect coordinates from all lanes
        for lane_id, lane_data in self.current_map_data.get('LANE', {}).items():
            try:
                centerline = self._parse_coordinate_list(lane_data.get('centerline', []))
                if len(centerline) > 0:
                    all_x.extend(centerline[:, 0])
                    all_y.extend(centerline[:, 1])
            except (ValueError, KeyError):
                continue
        
        if all_x and all_y:
            margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * self.config.bbox_margin_ratio
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    def get_map_bounds(self) -> Tuple[float, float, float, float]:
        """Get map bounds (min_x, min_y, max_x, max_y)"""
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        all_x, all_y = [], []
        
        # Collect coordinates from all lanes
        for lane_id, lane_data in self.current_map_data.get('LANE', {}).items():
            try:
                centerline = self._parse_coordinate_list(lane_data.get('centerline', []))
                if len(centerline) > 0:
                    all_x.extend(centerline[:, 0])
                    all_y.extend(centerline[:, 1])
            except (ValueError, KeyError):
                continue
        
        if not all_x or not all_y:
            return (0, 0, 100, 100)  # Default bounds
        
        return (min(all_x), min(all_y), max(all_x), max(all_y))
    
    def get_lanes_in_region(self, bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """
        Get lanes within a bounding box
        
        Args:
            bbox: (min_x, min_y, max_x, max_y)
            
        Returns:
            List of lane data dictionaries
        """
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        min_x, min_y, max_x, max_y = bbox
        lanes_in_region = []
        
        for lane_id, lane_data in self.current_map_data.get('LANE', {}).items():
            try:
                centerline = self._parse_coordinate_list(lane_data.get('centerline', []))
                if len(centerline) > 0:
                    # Check if any point of the centerline is in the bbox
                    x_coords = centerline[:, 0]
                    y_coords = centerline[:, 1]
                    
                    if (np.any((x_coords >= min_x) & (x_coords <= max_x) & 
                              (y_coords >= min_y) & (y_coords <= max_y))):
                        lane_data_copy = lane_data.copy()
                        lane_data_copy['lane_id'] = lane_id
                        lanes_in_region.append(lane_data_copy)
            except (ValueError, KeyError):
                continue
        
        return lanes_in_region
    
    def get_lane_centerline(self, lane_id: str) -> np.ndarray:
        """
        Get centerline for a specific lane
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Numpy array of centerline coordinates
        """
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        lane_data = self.current_map_data.get('LANE', {}).get(lane_id)
        if not lane_data:
            raise ValueError(f"Lane {lane_id} not found")
        
        return self._parse_coordinate_list(lane_data.get('centerline', []))
    
    def interpolate_lane_centerline(self, lane_id: str, num_points: int = 100) -> np.ndarray:
        """
        Interpolate lane centerline
        
        Args:
            lane_id: ID of the lane
            num_points: Number of points for interpolation
            
        Returns:
            Interpolated centerline points
        """
        centerline = self.get_lane_centerline(lane_id)
        
        if len(centerline) < 2:
            return centerline
        
        # Use local interpolation implementation
        return interp_arc(t=num_points, points=centerline)
    
    def create_distance_limited_visualization(self, 
                                            center_x: float, 
                                            center_y: float, 
                                            distance_range: Union[float, Tuple[float, float]], 
                                            show_direction: bool = True,
                                            save_path: Optional[str] = None,
                                            dpi: int = 150,
                                            figsize: Tuple[int, int] = (15, 15)) -> str:
        """
        Create distance-limited map visualization centered at specified point
        
        Args:
            center_x: Center point X coordinate
            center_y: Center point Y coordinate
            distance_range: Distance range in meters. Can be single value or tuple (min_dist, max_dist)
            show_direction: Whether to show direction arrows
            save_path: Output file path. If None, auto-generated based on parameters
            dpi: Output resolution
            figsize: Figure size
            
        Returns:
            Path to saved visualization file
        """
        if not self.current_map_data or not self.current_map_name:
            raise ValueError("No map data loaded. Please load a map first.")
        
        print(f"\n🎯 Creating distance-limited visualization: center({center_x:.1f}, {center_y:.1f})")
        
        if isinstance(distance_range, (int, float)):
            # Single distance value, create square region
            half_dist = distance_range / 2
            bbox = (center_x - half_dist, center_y - half_dist, 
                    center_x + half_dist, center_y + half_dist)
            print(f"   Distance range: ±{distance_range/2:.0f}m (square region)")
        else:
            # Distance range tuple
            min_dist, max_dist = distance_range
            bbox = (center_x - max_dist/2, center_y - max_dist/2,
                    center_x + max_dist/2, center_y + max_dist/2)
            print(f"   Distance range: {min_dist:.0f}m - {max_dist:.0f}m")
        
        # Get lanes in region
        lanes_in_region = self.get_lanes_in_region(bbox)
        print(f"   Lanes in region: {len(lanes_in_region)}")
        
        direction_text = " (with Direction Arrows)" if show_direction else ""
        print(f"   Direction arrows: {'enabled' if show_direction else 'disabled'}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{self.current_map_name} - Distance Limited Visualization ({distance_range}m){direction_text}', fontsize=16)
        
        # Subplot 1: Lanes only
        if show_direction:
            self._plot_lanes_with_direction(axes[0, 0], alpha=0.7, color='lightblue', edgecolor='blue')
        else:
            self._plot_lanes(axes[0, 0], alpha=0.7, color='lightblue', edgecolor='blue')
        axes[0, 0].set_xlim(bbox[0], bbox[2])
        axes[0, 0].set_ylim(bbox[1], bbox[3])
        axes[0, 0].set_title('Lanes' + (' with Direction' if show_direction else ''))
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Subplot 2: Centerlines only
        self._plot_centerlines(axes[0, 1], show_direction=show_direction, arrow_spacing=30,
                              color='red', linewidth=2)
        axes[0, 1].set_xlim(bbox[0], bbox[2])
        axes[0, 1].set_ylim(bbox[1], bbox[3])
        axes[0, 1].set_title('Centerlines' + (' with Direction' if show_direction else ''))
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Subplot 3: Crosswalks only
        self._plot_crosswalks(axes[1, 0], alpha=0.8, color='orange', edgecolor='darkorange')
        axes[1, 0].set_xlim(bbox[0], bbox[2])
        axes[1, 0].set_ylim(bbox[1], bbox[3])
        axes[1, 0].set_title('Crosswalks')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: All elements combined
        if show_direction:
            self._plot_lanes_with_direction(axes[1, 1], alpha=0.5, color='lightblue', edgecolor='blue')
        else:
            self._plot_lanes(axes[1, 1], alpha=0.5, color='lightblue', edgecolor='blue')
        
        self._plot_centerlines(axes[1, 1], show_direction=show_direction, arrow_spacing=40,
                              color='red', linewidth=1)
        self._plot_crosswalks(axes[1, 1], alpha=0.8, color='orange', edgecolor='darkorange')
        self._plot_stoplines(axes[1, 1], color='red', linewidth=3)
        axes[1, 1].set_xlim(bbox[0], bbox[2])
        axes[1, 1].set_ylim(bbox[1], bbox[3])
        axes[1, 1].set_title('All Elements' + (' with Direction' if show_direction else ''))
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark center point on all subplots
        for ax in axes.flat:
            ax.plot(center_x, center_y, 'ro', markersize=8, label='Center Point')
            ax.legend()
        
        # Generate output filename if not provided
        if save_path is None:
            if isinstance(distance_range, (int, float)):
                suffix = "_with_arrows" if show_direction else "_no_arrows"
                save_path = f'distance_limited_{self.current_map_name}_{distance_range}m{suffix}.png'
            else:
                suffix = "_with_arrows" if show_direction else "_no_arrows"
                save_path = f'distance_limited_{self.current_map_name}_{distance_range[0]}-{distance_range[1]}m{suffix}.png'
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved visualization file: {save_path}")
        return save_path
    
    def create_circular_region_visualization(self,
                                           center_x: float,
                                           center_y: float,
                                           radius: float,
                                           show_direction: bool = True,
                                           save_path: Optional[str] = None,
                                           dpi: int = 150,
                                           figsize: Tuple[int, int] = (12, 12)) -> str:
        """
        Create circular region map visualization
        
        Args:
            center_x: Center point X coordinate
            center_y: Center point Y coordinate
            radius: Radius in meters
            show_direction: Whether to show direction arrows
            save_path: Output file path. If None, auto-generated
            dpi: Output resolution
            figsize: Figure size
            
        Returns:
            Path to saved visualization file
        """
        if not self.current_map_data or not self.current_map_name:
            raise ValueError("No map data loaded. Please load a map first.")
        
        print(f"\n🔵 Creating circular region visualization: center({center_x:.1f}, {center_y:.1f}), radius {radius}m")
        print(f"   Direction arrows: {'enabled' if show_direction else 'disabled'}")
        
        # Create square bounding box surrounding circular region
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        
        # Get lanes in region
        lanes_in_region = self.get_lanes_in_region(bbox)
        print(f"   Lanes in bounding box: {len(lanes_in_region)}")
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Draw map elements
        if show_direction:
            self._plot_lanes_with_direction(ax, alpha=0.6, color='lightblue', edgecolor='blue')
        else:
            self._plot_lanes(ax, alpha=0.6, color='lightblue', edgecolor='blue')
        
        self._plot_centerlines(ax, show_direction=show_direction, arrow_spacing=35,
                              color='red', linewidth=1)
        self._plot_crosswalks(ax, alpha=0.8, color='orange', edgecolor='darkorange')
        
        # Set display range
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect('equal')
        
        # Draw circular boundary
        circle = plt.Circle((center_x, center_y), radius, fill=False, color='red', linewidth=3, linestyle='--')
        ax.add_patch(circle)
        
        # Mark center point
        ax.plot(center_x, center_y, 'ro', markersize=10, label='Center Point')
        
        direction_text = " (with Direction Arrows)" if show_direction else ""
        ax.set_title(f'{self.current_map_name} - Circular Region Visualization (Radius {radius}m){direction_text}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Generate output filename if not provided
        if save_path is None:
            suffix = "_with_arrows" if show_direction else "_no_arrows"
            save_path = f'circular_region_{self.current_map_name}_{radius}m{suffix}.png'
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved circular region visualization: {save_path}")
        return save_path
    
    def create_custom_bbox_visualization(self,
                                       min_x: float,
                                       min_y: float,
                                       max_x: float,
                                       max_y: float,
                                       show_direction: bool = True,
                                       save_path: Optional[str] = None,
                                       dpi: int = 150,
                                       figsize: Tuple[int, int] = (12, 10)) -> str:
        """
        Create custom bounding box map visualization
        
        Args:
            min_x, min_y, max_x, max_y: Bounding box coordinates
            show_direction: Whether to show direction arrows
            save_path: Output file path. If None, auto-generated
            dpi: Output resolution
            figsize: Figure size
            
        Returns:
            Path to saved visualization file
        """
        if not self.current_map_data or not self.current_map_name:
            raise ValueError("No map data loaded. Please load a map first.")
        
        print(f"\n📍 Creating custom bounding box visualization: ({min_x:.1f}, {min_y:.1f}) -> ({max_x:.1f}, {max_y:.1f})")
        
        bbox = (min_x, min_y, max_x, max_y)
        width = max_x - min_x
        height = max_y - min_y
        
        # Get lanes in region
        lanes_in_region = self.get_lanes_in_region(bbox)
        print(f"   Region size: {width:.0f} x {height:.0f} meters")
        print(f"   Lanes in region: {len(lanes_in_region)}")
        print(f"   Direction arrows: {'enabled' if show_direction else 'disabled'}")
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Draw map elements
        if show_direction:
            self._plot_lanes_with_direction(ax, alpha=0.6, color='lightblue', edgecolor='blue')
        else:
            self._plot_lanes(ax, alpha=0.6, color='lightblue', edgecolor='blue')
        
        self._plot_centerlines(ax, show_direction=show_direction, arrow_spacing=40,
                              color='red', linewidth=1.5)
        self._plot_crosswalks(ax, alpha=0.8, color='orange', edgecolor='darkorange')
        self._plot_stoplines(ax, color='red', linewidth=3)
        
        # Set display range
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        
        direction_text = " (with Direction Arrows)" if show_direction else ""
        ax.set_title(f'{self.current_map_name} - Custom BBox ({width:.0f}x{height:.0f}m){direction_text}')
        ax.grid(True, alpha=0.3)
        
        # Generate output filename if not provided
        if save_path is None:
            suffix = "_with_arrows" if show_direction else "_no_arrows"
            save_path = f'custom_bbox_{self.current_map_name}_{width:.0f}x{height:.0f}m{suffix}.png'
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved custom bounding box visualization: {save_path}")
        return save_path
    
    def create_region_comparison_visualization(self,
                                             center_x: float,
                                             center_y: float,
                                             distances: List[float],
                                             comparison_type: str = 'square',
                                             show_direction: bool = True,
                                             save_path: Optional[str] = None,
                                             dpi: int = 150) -> str:
        """
        Create comparison visualization showing multiple distance ranges
        
        Args:
            center_x: Center point X coordinate
            center_y: Center point Y coordinate
            distances: List of distances to compare
            comparison_type: Type of comparison ('square' or 'circular')
            show_direction: Whether to show direction arrows
            save_path: Output file path. If None, auto-generated
            dpi: Output resolution
            
        Returns:
            Path to saved visualization file
        """
        if not self.current_map_data or not self.current_map_name:
            raise ValueError("No map data loaded. Please load a map first.")
        
        print(f"\n📊 Creating region comparison visualization: {len(distances)} regions")
        print(f"   Center: ({center_x:.1f}, {center_y:.1f})")
        print(f"   Distances: {distances}")
        print(f"   Type: {comparison_type}")
        
        # Determine grid layout
        n_plots = len(distances)
        if n_plots <= 2:
            rows, cols = 1, n_plots
            figsize = (6 * n_plots, 6)
        elif n_plots <= 4:
            rows, cols = 2, 2
            figsize = (12, 12)
        else:
            rows = int(np.ceil(np.sqrt(n_plots)))
            cols = int(np.ceil(n_plots / rows))
            figsize = (6 * cols, 6 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        direction_text = " (with Direction Arrows)" if show_direction else ""
        fig.suptitle(f'{self.current_map_name} - {comparison_type.title()} Region Comparison{direction_text}', fontsize=16)
        
        for i, distance in enumerate(distances):
            ax = axes[i]
            
            if comparison_type == 'circular':
                # Circular region
                bbox = (center_x - distance, center_y - distance, center_x + distance, center_y + distance)
                
                # Draw map elements
                if show_direction:
                    self._plot_lanes_with_direction(ax, alpha=0.6, color='lightblue', edgecolor='blue')
                else:
                    self._plot_lanes(ax, alpha=0.6, color='lightblue', edgecolor='blue')
                
                self._plot_centerlines(ax, show_direction=show_direction, arrow_spacing=50,
                                      color='red', linewidth=1)
                
                # Draw circular boundary
                circle = plt.Circle((center_x, center_y), distance, fill=False, color='red', linewidth=2, linestyle='--')
                ax.add_patch(circle)
                
                title = f'Radius {distance}m'
            else:
                # Square region
                half_dist = distance / 2
                bbox = (center_x - half_dist, center_y - half_dist, 
                        center_x + half_dist, center_y + half_dist)
                
                # Draw map elements
                if show_direction:
                    self._plot_lanes_with_direction(ax, alpha=0.6, color='lightblue', edgecolor='blue')
                else:
                    self._plot_lanes(ax, alpha=0.6, color='lightblue', edgecolor='blue')
                
                self._plot_centerlines(ax, show_direction=show_direction, arrow_spacing=50,
                                      color='red', linewidth=1)
                
                title = f'±{distance/2:.0f}m'
            
            # Set display range and formatting
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Mark center point
            ax.plot(center_x, center_y, 'ro', markersize=6)
            
            # Count lanes in region
            lanes_count = len(self.get_lanes_in_region(bbox))
            ax.text(0.02, 0.98, f'Lanes: {lanes_count}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Generate output filename if not provided
        if save_path is None:
            distances_str = '_'.join(map(str, distances))
            suffix = "_with_arrows" if show_direction else "_no_arrows"
            save_path = f'{comparison_type}_comparison_{self.current_map_name}_{distances_str}m{suffix}.png'
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved region comparison visualization: {save_path}")
        return save_path


def main():
    """Example usage of the DAIR Map Visualizer with advanced features"""
    print("🎨 DAIR Map Visualizer - Demo")
    print("=" * 50)
    
    try:
        # Try to use DAIRMapManager for full functionality
        try:
            from .map_manager import DAIRMapManager
        except ImportError:
            try:
                from map_manager import DAIRMapManager
            except ImportError:
                print("⚠️ DAIRMapManager not available. Using basic demo.")
                basic_demo()
                return
        
        # Initialize manager and visualizer
        config = MapConfig()
        manager = DAIRMapManager(config)
        visualizer = MapVisualizer(config)
        
        # Get available maps
        available_maps = manager.get_available_maps()
        print(f"📍 Available maps: {len(available_maps)}")
        
        if not available_maps:
            print("❌ No maps found. Please check your data directory.")
            print("📁 Expected location: data/v2x-seq-nuscenes/cooperative/maps/")
            return
        
        # Use first available map
        map_name = available_maps[0]
        print(f"🗺️  Loading map: {map_name}")
        
        # Load map
        map_data = manager.load_map(map_name)
        visualizer.set_current_map(map_name, map_data)
        
        # Get map bounds and center
        bounds = visualizer.get_map_bounds()
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        
        print(f"📊 Map bounds: ({bounds[0]:.1f}, {bounds[1]:.1f}) -> ({bounds[2]:.1f}, {bounds[3]:.1f})")
        print(f"🎯 Map center: ({center_x:.1f}, {center_y:.1f})")
        
        print("\n🚀 Creating visualizations...")
        
        # Demo 1: Basic visualization
        print("1. Creating basic map visualization...")
        from .config import VisualizationConfig
        viz_config = VisualizationConfig(
            elements=['lanes', 'centerlines'],
            show_direction=True,
            save_path=f"demo_basic_{map_name}.png",
            show_plot=False
        )
        visualizer.visualize_with_config(viz_config)
        
        # Demo 2: Distance-limited visualization
        print("2. Creating distance-limited visualization...")
        visualizer.create_distance_limited_visualization(
            center_x=center_x,
            center_y=center_y,
            distance_range=500,
            show_direction=True,
            save_path=f"demo_distance_{map_name}.png"
        )
        
        # Demo 3: Circular region visualization
        print("3. Creating circular region visualization...")
        visualizer.create_circular_region_visualization(
            center_x=center_x,
            center_y=center_y,
            radius=300,
            show_direction=True,
            save_path=f"demo_circular_{map_name}.png"
        )
        
        print("\n✅ Demo completed successfully!")
        print("📁 Generated visualization files:")
        print(f"   - demo_basic_{map_name}.png")
        print(f"   - demo_distance_{map_name}.png")
        print(f"   - demo_circular_{map_name}.png")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("ℹ️  Running basic demo instead...")
        basic_demo()


def basic_demo():
    """Basic demo without map loading"""
    print("\n📝 Basic Demo - MapVisualizer Features")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = MapVisualizer()
    
    print("🔧 Available visualization methods:")
    methods = [m for m in dir(visualizer) if m.startswith('create_') or m.startswith('visualize')]
    for i, method in enumerate(methods, 1):
        print(f"   {i}. {method}")
    
    print("\n💡 Usage example:")
    print("""
    from dair_map import MapVisualizer, DAIRMapManager
    
    # Load map data
    manager = DAIRMapManager()
    manager.load_map("yizhuang02")
    map_data = manager.get_map_data("yizhuang02")
    
    # Create visualizer
    visualizer = MapVisualizer()
    visualizer.set_current_map("yizhuang02", map_data)
    
    # Create advanced visualizations
    visualizer.create_distance_limited_visualization(1500, 1000, 600)
    visualizer.create_circular_region_visualization(1500, 1000, 400)
    """)
    
    print("\n📖 For more examples, run:")
    print("   python dair_map/example_advanced_visualization.py")


if __name__ == "__main__":
    main()
