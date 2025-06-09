"""
Example script demonstrating DAIR Map Visualizer usage
"""

from dair_map import MapVisualizer, load_map_data, convert_to_argoverse_format, get_map_statistics
import matplotlib.pyplot as plt


def basic_visualization_example():
    """Basic example of loading and visualizing a map"""
    print("=== Basic Visualization Example ===")
    
    # Initialize the visualizer
    visualizer = MapVisualizer()
    
    # Check available maps
    print(f"Available maps: {visualizer.available_maps}")
    
    if not visualizer.available_maps:
        print("No maps found. Please check the data directory.")
        return
    
    # Load the first available map
    map_name = visualizer.available_maps[0]
    print(f"Loading map: {map_name}")
    
    map_data = visualizer.load_map(map_name)
    
    # Get map statistics
    stats = get_map_statistics(map_data)
    print(f"\nMap Statistics:")
    for key, value in stats.items():
        if key != 'bounds':
            print(f"  {key}: {value}")
    
    if stats['bounds']:
        bounds = stats['bounds']
        print(f"  Map bounds: ({bounds['min_x']:.1f}, {bounds['min_y']:.1f}) to ({bounds['max_x']:.1f}, {bounds['max_y']:.1f})")
        print(f"  Map size: {bounds['width']:.1f} x {bounds['height']:.1f} meters")
    
    # Visualize the full map
    print("\nGenerating full map visualization...")
    visualizer.visualize_map(
        elements=['polygons', 'lanes', 'centerlines', 'crossings'],
        figsize=(12, 12),
        save_path=f"full_map_{map_name}.png"
    )


def region_visualization_example():
    """Example of visualizing specific regions of the map"""
    print("\n=== Region Visualization Example ===")
    
    visualizer = MapVisualizer()
    
    if not visualizer.available_maps:
        print("No maps found.")
        return
    
    # Load map
    map_name = visualizer.available_maps[0]
    visualizer.load_map(map_name)
    
    # Get map bounds
    bounds = visualizer.get_map_bounds()
    min_x, min_y, max_x, max_y = bounds
    
    # Define a smaller region (center 30% of the map)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    region_size = min(max_x - min_x, max_y - min_y) * 0.3
    
    region_bbox = (
        center_x - region_size/2,
        center_y - region_size/2,
        center_x + region_size/2,
        center_y + region_size/2
    )
    
    print(f"Visualizing region: {region_bbox}")
    
    # Visualize the region with different elements
    visualizer.visualize_map(
        elements=['lanes', 'centerlines'],
        bbox=region_bbox,
        figsize=(10, 10),
        save_path=f"region_lanes_{map_name}.png"
    )
    
    # Get lanes in this region
    lanes_in_region = visualizer.get_lanes_in_region(region_bbox)
    print(f"Found {len(lanes_in_region)} lanes in the specified region")


def lane_analysis_example():
    """Example of analyzing individual lanes"""
    print("\n=== Lane Analysis Example ===")
    
    visualizer = MapVisualizer()
    
    if not visualizer.available_maps:
        print("No maps found.")
        return
    
    # Load map
    map_name = visualizer.available_maps[0]
    map_data = visualizer.load_map(map_name)
    
    # Analyze first few lanes
    lanes = map_data.get('lane', [])[:5]  # First 5 lanes
    
    fig, axes = plt.subplots(1, len(lanes), figsize=(15, 3))
    if len(lanes) == 1:
        axes = [axes]
    
    for i, lane_data in enumerate(lanes):
        try:
            # Get original centerline
            centerline = visualizer._get_lane_centerline(lane_data)
            
            # Get interpolated centerline
            interpolated = visualizer.interpolate_lane_centerline(lane_data, num_points=50)
            
            # Plot both
            ax = axes[i]
            if len(centerline) > 0:
                ax.plot(centerline[:, 0], centerline[:, 1], 'b-', label='Original', linewidth=2)
            if len(interpolated) > 0:
                ax.plot(interpolated[:, 0], interpolated[:, 1], 'r--', label='Interpolated', linewidth=1)
            
            ax.set_title(f"Lane {i+1}\n{lane_data.get('lane_type', 'UNKNOWN')}")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            print(f"Lane {i+1}: {len(centerline)} original points, {len(interpolated)} interpolated points")
            
        except Exception as e:
            print(f"Error processing lane {i+1}: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig(f"lane_analysis_{map_name}.png", dpi=150, bbox_inches='tight')
    plt.show()


def argoverse_conversion_example():
    """Example of converting to Argoverse format"""
    print("\n=== Argoverse Conversion Example ===")
    
    visualizer = MapVisualizer()
    
    if not visualizer.available_maps:
        print("No maps found.")
        return
    
    # Load map
    map_name = visualizer.available_maps[0]
    map_data = visualizer.load_map(map_name)
    
    # Convert to Argoverse format
    print("Converting to Argoverse format...")
    argoverse_data = convert_to_argoverse_format(map_data, map_name)
    
    print(f"Conversion completed:")
    print(f"  - Lanes: {len(argoverse_data['lanes'])}")
    print(f"  - Polygons: {len(argoverse_data['polygons'])}")
    print(f"  - Pedestrian crossings: {len(argoverse_data['pedestrian_crossings'])}")
    
    # Save the converted data
    from dair_map.utils import save_argoverse_map
    output_path = f"argoverse_maps/{map_name}_argoverse.json"
    save_argoverse_map(argoverse_data, output_path)
    
    # Also use the built-in export function
    visualizer.export_to_argoverse_format("argoverse_export")


def custom_visualization_example():
    """Example of custom visualization with specific styling"""
    print("\n=== Custom Visualization Example ===")
    
    visualizer = MapVisualizer()
    
    if not visualizer.available_maps:
        print("No maps found.")
        return
    
    # Load map
    map_name = visualizer.available_maps[0]
    visualizer.load_map(map_name)
    
    # Create custom visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Plot 1: Only road polygons
    visualizer._plot_polygons(axes[0, 0], alpha=0.6, color='lightblue', edgecolor='darkblue')
    axes[0, 0].set_title('Road Polygons Only')
    axes[0, 0].set_aspect('equal')
    visualizer._auto_fit_view(axes[0, 0])
    
    # Plot 2: Only lanes
    visualizer._plot_lanes(axes[0, 1], alpha=0.7, color='green', edgecolor='darkgreen')
    axes[0, 1].set_title('Lanes Only')
    axes[0, 1].set_aspect('equal')
    visualizer._auto_fit_view(axes[0, 1])
    
    # Plot 3: Only centerlines
    visualizer._plot_centerlines(axes[1, 0], color='red', linewidth=0.5)
    axes[1, 0].set_title('Centerlines Only')
    axes[1, 0].set_aspect('equal')
    visualizer._auto_fit_view(axes[1, 0])
    
    # Plot 4: Everything combined
    visualizer._plot_polygons(axes[1, 1], alpha=0.3, color='lightgray', edgecolor='gray')
    visualizer._plot_lanes(axes[1, 1], alpha=0.5, color='blue', edgecolor='darkblue')
    visualizer._plot_centerlines(axes[1, 1], color='red', linewidth=0.8)
    visualizer._plot_crossings(axes[1, 1], alpha=0.8, color='orange', edgecolor='darkorange')
    axes[1, 1].set_title('All Elements Combined')
    axes[1, 1].set_aspect('equal')
    visualizer._auto_fit_view(axes[1, 1])
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"custom_visualization_{map_name}.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run all examples"""
    print("DAIR Map Visualizer Examples")
    print("============================")
    
    try:
        # Run examples
        basic_visualization_example()
        region_visualization_example()
        lane_analysis_example()
        argoverse_conversion_example()
        custom_visualization_example()
        
        print("\n=== All examples completed successfully! ===")
        print("Check the generated PNG files for visualizations.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 