"""
DAIR Map Visualizer using Argoverse API format
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path

# Try to import Argoverse utilities, fall back to local implementations
try:
    from argoverse.map_representation.map_api import ArgoverseMap
    from argoverse.utils.centerline_utils import centerline_to_polygon
    from argoverse.utils.json_utils import read_json_file
    from argoverse.utils.manhattan_search import compute_polygon_bboxes
    from argoverse.utils.interpolate import interp_arc
    ARGOVERSE_AVAILABLE = True
    print("Using Argoverse API")
except ImportError:
    from .argoverse_fallback import (
        ArgoverseMap, centerline_to_polygon, read_json_file, 
        compute_polygon_bboxes, interp_arc
    )
    ARGOVERSE_AVAILABLE = False
    print("Argoverse API not available, using fallback implementations")


class MapVisualizer:
    """
    Map visualizer for DAIR V2X-seq dataset using Argoverse format
    """
    
    def __init__(self, maps_dir: str = "data/v2x-seq-nuscenes/cooperative/maps"):
        """
        Initialize the map visualizer
        
        Args:
            maps_dir: Path to the maps directory
        """
        self.maps_dir = Path(maps_dir)
        self.expansion_dir = self.maps_dir / "expansion"
        self.available_maps = self._get_available_maps()
        self.current_map_data = None
        self.current_map_name = None
        
    def _get_available_maps(self) -> List[str]:
        """Get list of available map names"""
        if not self.expansion_dir.exists():
            return []
        
        maps = []
        for json_file in self.expansion_dir.glob("*.json"):
            maps.append(json_file.stem)
        return sorted(maps)
    
    def load_map(self, map_name: str) -> Dict[str, Any]:
        """
        Load map data from JSON file
        
        Args:
            map_name: Name of the map (e.g., 'yizhuang02')
            
        Returns:
            Loaded map data dictionary
        """
        map_path = self.expansion_dir / f"{map_name}.json"
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")
        
        print(f"Loading map: {map_name}")
        self.current_map_data = read_json_file(str(map_path))
        self.current_map_name = map_name
        
        print(f"Map loaded successfully!")
        print(f"- Polygons: {len(self.current_map_data.get('polygon', []))}")
        print(f"- Lines: {len(self.current_map_data.get('line', []))}")
        print(f"- Nodes: {len(self.current_map_data.get('node', []))}")
        print(f"- Lanes: {len(self.current_map_data.get('lane', []))}")
        print(f"- Road segments: {len(self.current_map_data.get('road_segment', []))}")
        print(f"- Pedestrian crossings: {len(self.current_map_data.get('ped_crossing', []))}")
        
        return self.current_map_data
    
    def _get_node_coordinates(self, node_token: str) -> Tuple[float, float]:
        """Get node coordinates by token"""
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        for node in self.current_map_data.get('node', []):
            if node['token'] == node_token:
                return (node['x'], node['y'])
        
        raise ValueError(f"Node token not found: {node_token}")
    
    def _polygon_to_coordinates(self, polygon_data: Dict) -> np.ndarray:
        """Convert polygon data to coordinate array"""
        coordinates = []
        for node_token in polygon_data['exterior_node_tokens']:
            x, y = self._get_node_coordinates(node_token)
            coordinates.append([x, y])
        
        return np.array(coordinates)
    
    def _get_lane_centerline(self, lane_data: Dict) -> np.ndarray:
        """
        Extract lane centerline from lane data
        Uses the polygon associated with the lane to compute centerline
        """
        polygon_token = lane_data['polygon_token']
        
        # Find the polygon
        for polygon in self.current_map_data.get('polygon', []):
            if polygon['token'] == polygon_token:
                coords = self._polygon_to_coordinates(polygon)
                
                # Simple centerline computation - can be improved
                if len(coords) >= 4:
                    # Get approximate centerline by averaging opposite sides
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
    
    def visualize_map(self, 
                     map_name: Optional[str] = None,
                     elements: List[str] = None,
                     bbox: Optional[Tuple[float, float, float, float]] = None,
                     figsize: Tuple[int, int] = (15, 15),
                     save_path: Optional[str] = None) -> None:
        """
        Visualize map elements
        
        Args:
            map_name: Map to visualize (if different from current)
            elements: List of elements to visualize ['polygons', 'lanes', 'centerlines', 'crossings']
            bbox: Bounding box (min_x, min_y, max_x, max_y) for visualization
            figsize: Figure size
            save_path: Path to save the figure
        """
        if map_name and map_name != self.current_map_name:
            self.load_map(map_name)
        
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        if elements is None:
            elements = ['polygons', 'lanes', 'centerlines', 'crossings']
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Visualize polygons (road areas)
        if 'polygons' in elements:
            self._plot_polygons(ax, alpha=0.3, color='lightgray', edgecolor='gray')
        
        # Visualize lanes
        if 'lanes' in elements:
            self._plot_lanes(ax, color='blue', alpha=0.5)
        
        # Visualize lane centerlines
        if 'centerlines' in elements:
            self._plot_centerlines(ax, color='red', linewidth=1)
        
        # Visualize pedestrian crossings
        if 'crossings' in elements:
            self._plot_crossings(ax, color='orange', alpha=0.7)
        
        # Set bounding box if provided
        if bbox:
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
        else:
            # Auto-fit to data
            self._auto_fit_view(ax)
        
        ax.set_aspect('equal')
        ax.set_title(f'Map Visualization: {self.current_map_name}')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = []
        if 'polygons' in elements:
            legend_elements.append(patches.Patch(color='lightgray', label='Road Areas'))
        if 'lanes' in elements:
            legend_elements.append(patches.Patch(color='blue', label='Lanes'))
        if 'centerlines' in elements:
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, label='Centerlines'))
        if 'crossings' in elements:
            legend_elements.append(patches.Patch(color='orange', label='Ped Crossings'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map visualization saved to: {save_path}")
        
        plt.show()
    
    def _plot_polygons(self, ax, **kwargs):
        """Plot road polygons"""
        polygons = []
        
        for polygon_data in self.current_map_data.get('polygon', []):
            try:
                coords = self._polygon_to_coordinates(polygon_data)
                if len(coords) > 2:
                    polygon = patches.Polygon(coords, closed=True, **kwargs)
                    polygons.append(polygon)
            except ValueError:
                continue  # Skip if node not found
        
        if polygons:
            collection = PatchCollection(polygons, match_original=True)
            ax.add_collection(collection)
    
    def _plot_lanes(self, ax, **kwargs):
        """Plot lane polygons"""
        lane_polygons = []
        
        for lane_data in self.current_map_data.get('lane', []):
            try:
                polygon_token = lane_data['polygon_token']
                
                # Find the polygon
                for polygon in self.current_map_data.get('polygon', []):
                    if polygon['token'] == polygon_token:
                        coords = self._polygon_to_coordinates(polygon)
                        if len(coords) > 2:
                            polygon_patch = patches.Polygon(coords, closed=True, **kwargs)
                            lane_polygons.append(polygon_patch)
                        break
            except ValueError:
                continue
        
        if lane_polygons:
            collection = PatchCollection(lane_polygons, match_original=True)
            ax.add_collection(collection)
    
    def _plot_centerlines(self, ax, **kwargs):
        """Plot lane centerlines"""
        centerlines = []
        
        for lane_data in self.current_map_data.get('lane', []):
            try:
                centerline = self._get_lane_centerline(lane_data)
                if len(centerline) > 1:
                    centerlines.append(centerline)
            except ValueError:
                continue
        
        if centerlines:
            # Use centerline_to_polygon from argoverse if needed for visualization
            line_collection = LineCollection(centerlines, **kwargs)
            ax.add_collection(line_collection)
    
    def _plot_crossings(self, ax, **kwargs):
        """Plot pedestrian crossings"""
        crossing_polygons = []
        
        for crossing in self.current_map_data.get('ped_crossing', []):
            try:
                coords = self._polygon_to_coordinates(crossing)
                if len(coords) > 2:
                    polygon = patches.Polygon(coords, closed=True, **kwargs)
                    crossing_polygons.append(polygon)
            except ValueError:
                continue
        
        if crossing_polygons:
            collection = PatchCollection(crossing_polygons, match_original=True)
            ax.add_collection(collection)
    
    def _auto_fit_view(self, ax):
        """Auto-fit the view to show all data"""
        all_x, all_y = [], []
        
        # Collect all coordinates
        for node in self.current_map_data.get('node', []):
            all_x.append(node['x'])
            all_y.append(node['y'])
        
        if all_x and all_y:
            margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.05
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    def get_map_bounds(self) -> Tuple[float, float, float, float]:
        """Get map bounds (min_x, min_y, max_x, max_y)"""
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        all_x = [node['x'] for node in self.current_map_data.get('node', [])]
        all_y = [node['y'] for node in self.current_map_data.get('node', [])]
        
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
        
        for lane_data in self.current_map_data.get('lane', []):
            try:
                centerline = self._get_lane_centerline(lane_data)
                if len(centerline) > 0:
                    # Check if any point of the centerline is in the bbox
                    x_coords = centerline[:, 0]
                    y_coords = centerline[:, 1]
                    
                    if (np.any((x_coords >= min_x) & (x_coords <= max_x) & 
                              (y_coords >= min_y) & (y_coords <= max_y))):
                        lanes_in_region.append(lane_data)
            except ValueError:
                continue
        
        return lanes_in_region
    
    def interpolate_lane_centerline(self, lane_data: Dict, num_points: int = 100) -> np.ndarray:
        """
        Interpolate lane centerline using argoverse interpolation utilities
        
        Args:
            lane_data: Lane data dictionary
            num_points: Number of points for interpolation
            
        Returns:
            Interpolated centerline points
        """
        centerline = self._get_lane_centerline(lane_data)
        
        if len(centerline) < 2:
            return centerline
        
        # Use argoverse interpolation
        try:
            interpolated = interp_arc(t=num_points, points=centerline)
            return interpolated
        except:
            # Fallback to simple linear interpolation if argoverse function fails
            from scipy.interpolate import interp1d
            t_original = np.linspace(0, 1, len(centerline))
            t_new = np.linspace(0, 1, num_points)
            
            f_x = interp1d(t_original, centerline[:, 0], kind='linear')
            f_y = interp1d(t_original, centerline[:, 1], kind='linear')
            
            return np.column_stack([f_x(t_new), f_y(t_new)])
    
    def export_to_argoverse_format(self, output_dir: str) -> None:
        """
        Export current map data to Argoverse-compatible format
        
        Args:
            output_dir: Directory to save the exported data
        """
        if not self.current_map_data:
            raise ValueError("No map data loaded")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert and save in Argoverse format
        argoverse_data = {
            'map_name': self.current_map_name,
            'lanes': [],
            'polygons': [],
            'bounds': self.get_map_bounds()
        }
        
        # Convert lanes
        for lane_data in self.current_map_data.get('lane', []):
            try:
                centerline = self._get_lane_centerline(lane_data)
                if len(centerline) > 1:
                    # Convert to polygon using argoverse utility
                    polygon_coords = centerline_to_polygon(centerline)
                    
                    argoverse_lane = {
                        'id': lane_data['token'],
                        'centerline': centerline.tolist(),
                        'polygon': polygon_coords.tolist() if polygon_coords is not None else [],
                        'lane_type': lane_data.get('lane_type', 'VEHICLE')
                    }
                    argoverse_data['lanes'].append(argoverse_lane)
            except:
                continue
        
        # Save the converted data
        output_file = output_path / f"{self.current_map_name}_argoverse.json"
        with open(output_file, 'w') as f:
            json.dump(argoverse_data, f, indent=2)
        
        print(f"Map exported to Argoverse format: {output_file}")


def main():
    """Example usage of the map visualizer"""
    # Initialize visualizer
    visualizer = MapVisualizer()
    
    print("Available maps:", visualizer.available_maps)
    
    if visualizer.available_maps:
        # Load and visualize the first available map
        map_name = visualizer.available_maps[0]
        visualizer.load_map(map_name)
        
        # Get map bounds
        bounds = visualizer.get_map_bounds()
        print(f"Map bounds: {bounds}")
        
        # Visualize with all elements
        visualizer.visualize_map(
            elements=['polygons', 'lanes', 'centerlines', 'crossings'],
            save_path=f"map_visualization_{map_name}.png"
        )
        
        # Visualize a specific region
        min_x, min_y, max_x, max_y = bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        region_size = min(max_x - min_x, max_y - min_y) * 0.3
        
        region_bbox = (
            center_x - region_size/2,
            center_y - region_size/2,
            center_x + region_size/2,
            center_y + region_size/2
        )
        
        visualizer.visualize_map(
            elements=['lanes', 'centerlines'],
            bbox=region_bbox,
            save_path=f"map_region_{map_name}.png"
        )


if __name__ == "__main__":
    main()
