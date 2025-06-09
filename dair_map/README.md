# DAIR Map Visualizer

A comprehensive map visualization tool for DAIR V2X-seq-NuScenes dataset using Argoverse API format.

## Features

- **Load and visualize** V2X-seq-NuScenes cooperative map data
- **Argoverse API integration** using centerline utilities, JSON utilities, and interpolation
- **Multiple visualization modes**: polygons, lanes, centerlines, pedestrian crossings
- **Region-based visualization** with bounding box filtering
- **Lane analysis tools** with centerline extraction and interpolation
- **Format conversion** to Argoverse-compatible format
- **Export functionality** for processed map data

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy matplotlib pathlib typing
pip install argoverse-api  # For Argoverse utilities
```

## Quick Start

```python
from dair_map import MapVisualizer

# Initialize visualizer
visualizer = MapVisualizer()

# Check available maps
print("Available maps:", visualizer.available_maps)

# Load a map
visualizer.load_map('yizhuang02')

# Visualize the full map
visualizer.visualize_map(
    elements=['polygons', 'lanes', 'centerlines', 'crossings'],
    save_path='map_visualization.png'
)
```

## Usage Examples

### Basic Map Loading and Visualization

```python
from dair_map import MapVisualizer

# Initialize with default maps directory
visualizer = MapVisualizer()

# Or specify custom directory
visualizer = MapVisualizer(maps_dir="custom/path/to/maps")

# Load map data
map_data = visualizer.load_map('yizhuang02')

# Full map visualization
visualizer.visualize_map(
    elements=['polygons', 'lanes', 'centerlines', 'crossings'],
    figsize=(15, 15),
    save_path='full_map.png'
)
```

### Region-Based Visualization

```python
# Get map bounds
bounds = visualizer.get_map_bounds()
min_x, min_y, max_x, max_y = bounds

# Define region of interest
region_bbox = (min_x + 100, min_y + 100, min_x + 500, min_y + 500)

# Visualize specific region
visualizer.visualize_map(
    elements=['lanes', 'centerlines'],
    bbox=region_bbox,
    figsize=(10, 10),
    save_path='region_view.png'
)

# Get lanes in region
lanes_in_region = visualizer.get_lanes_in_region(region_bbox)
print(f"Found {len(lanes_in_region)} lanes in region")
```

### Lane Analysis

```python
# Get all lanes
map_data = visualizer.current_map_data
lanes = map_data.get('lane', [])

for lane in lanes[:5]:  # Analyze first 5 lanes
    # Extract centerline
    centerline = visualizer._get_lane_centerline(lane)
    
    # Interpolate centerline with more points
    interpolated = visualizer.interpolate_lane_centerline(lane, num_points=100)
    
    print(f"Lane {lane['token']}: {len(centerline)} -> {len(interpolated)} points")
```

### Convert to Argoverse Format

```python
from dair_map.utils import convert_to_argoverse_format, save_argoverse_map

# Load map
visualizer.load_map('yizhuang02')

# Convert to Argoverse format
argoverse_data = convert_to_argoverse_format(
    visualizer.current_map_data, 
    map_name='yizhuang02'
)

# Save converted data
save_argoverse_map(argoverse_data, 'argoverse_maps/yizhuang02_argoverse.json')

# Or use built-in export
visualizer.export_to_argoverse_format('exported_maps')
```

### Map Statistics and Validation

```python
from dair_map.utils import get_map_statistics, validate_map_data

# Get map statistics
stats = get_map_statistics(map_data)
print("Map Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Validate map data
issues = validate_map_data(map_data)
if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Map data is valid!")
```

## API Reference

### MapVisualizer Class

#### Constructor
```python
MapVisualizer(maps_dir: str = "data/v2x-seq-nuscenes/cooperative/maps")
```

#### Main Methods

- `load_map(map_name: str)` - Load map data from JSON file
- `visualize_map(map_name, elements, bbox, figsize, save_path)` - Visualize map elements
- `get_map_bounds()` - Get map coordinate bounds
- `get_lanes_in_region(bbox)` - Filter lanes by bounding box
- `interpolate_lane_centerline(lane_data, num_points)` - Interpolate lane centerline
- `export_to_argoverse_format(output_dir)` - Export to Argoverse format

#### Visualization Elements

- `polygons` - Road surface polygons
- `lanes` - Lane polygons
- `centerlines` - Lane centerlines
- `crossings` - Pedestrian crossings

### Utility Functions

```python
from dair_map.utils import (
    load_map_data,
    convert_to_argoverse_format, 
    get_map_statistics,
    validate_map_data,
    filter_map_elements_by_bbox
)
```

## Data Format

The visualizer expects V2X-seq-NuScenes map data in JSON format with the following structure:

```json
{
  "version": "1.3",
  "node": [
    {"token": "node_id", "x": 123.45, "y": 678.90}
  ],
  "polygon": [
    {
      "token": "polygon_id",
      "exterior_node_tokens": ["node_id1", "node_id2", ...],
      "holes": []
    }
  ],
  "lane": [
    {
      "token": "lane_id",
      "polygon_token": "polygon_id",
      "lane_type": "CAR",
      "from_edge_line_token": "line_id1",
      "to_edge_line_token": "line_id2"
    }
  ],
  "ped_crossing": [...],
  "road_segment": [...],
  ...
}
```

## Argoverse Integration

The implementation uses the following Argoverse API components:

```python
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import centerline_to_polygon
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.manhattan_search import compute_polygon_bboxes
from argoverse.utils.interpolate import interp_arc
```

## Examples

Run the example script to see all features in action:

```bash
cd dair_map
python example.py
```

This will generate several visualization files demonstrating different capabilities:
- `full_map_*.png` - Complete map visualization
- `region_lanes_*.png` - Regional lane visualization  
- `lane_analysis_*.png` - Individual lane analysis
- `custom_visualization_*.png` - Custom styling examples

## Directory Structure

```
dair_map/
├── __init__.py          # Package initialization
├── map_visualizer.py    # Main MapVisualizer class
├── utils.py            # Utility functions
├── example.py          # Usage examples
└── README.md           # This documentation
```

## Notes

- **Map Data Path**: Default path is `data/v2x-seq-nuscenes/cooperative/maps/expansion/`
- **Coordinate System**: Maps use local coordinate system in meters
- **Performance**: Large maps (100MB+) may take several seconds to load
- **Memory Usage**: Keep in mind memory requirements for large maps

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that map files exist in the specified directory
2. **Import errors**: Ensure Argoverse API is properly installed
3. **Memory issues**: For large maps, consider filtering by region
4. **Visualization issues**: Adjust figsize parameter for better display

### Performance Tips

- Use `bbox` parameter to visualize only regions of interest
- Filter elements using the `elements` parameter
- Use `get_lanes_in_region()` for efficient spatial queries
- Cache loaded map data to avoid repeated loading 