#!/usr/bin/env python3
"""
Demo script to create a basic map visualization
"""

import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add dair_map to path
sys.path.append('dair_map')

try:
    from dair_map.map_visualizer import MapVisualizer
    from dair_map.utils import get_map_statistics
    
    print('DAIR Map Visualizer Demo')
    print('=' * 40)
    
    # Initialize visualizer
    visualizer = MapVisualizer()
    print(f'Available maps: {len(visualizer.available_maps)}')
    
    if visualizer.available_maps:
        # Load the first map
        map_name = visualizer.available_maps[0]
        print(f'Loading map: {map_name}')
        
        map_data = visualizer.load_map(map_name)
        
        # Get statistics
        stats = get_map_statistics(map_data)
        print(f'Map size: {stats["bounds"]["width"]:.0f} x {stats["bounds"]["height"]:.0f} meters')
        
        # Create a regional visualization (smaller area for demo)
        bounds = visualizer.get_map_bounds()
        min_x, min_y, max_x, max_y = bounds
        
        # Define a 1000x1000 meter region in the center
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        region_size = 1000
        
        demo_bbox = (
            center_x - region_size/2,
            center_y - region_size/2,
            center_x + region_size/2,
            center_y + region_size/2
        )
        
        print(f'Creating demo visualization for region: {demo_bbox}')
        
        # Create the visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'DAIR Map Demo: {map_name}', fontsize=16)
        print("stage 000")
        # Plot 1: Road polygons only
        visualizer._plot_polygons(axes[0, 0], alpha=0.7, color='lightblue', edgecolor='blue')
        axes[0, 0].set_xlim(demo_bbox[0], demo_bbox[2])
        axes[0, 0].set_ylim(demo_bbox[1], demo_bbox[3])
        axes[0, 0].set_title('Road Polygons')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.3)
        print("stage 001")
        # Plot 2: Lanes only
        visualizer._plot_lanes(axes[0, 1], alpha=0.7, color='green', edgecolor='darkgreen')
        axes[0, 1].set_xlim(demo_bbox[0], demo_bbox[2])
        axes[0, 1].set_ylim(demo_bbox[1], demo_bbox[3])
        axes[0, 1].set_title('Lanes')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        print("stage 002")
        # Plot 3: Centerlines only
        visualizer._plot_centerlines(axes[1, 0], color='red', linewidth=1)
        axes[1, 0].set_xlim(demo_bbox[0], demo_bbox[2])
        axes[1, 0].set_ylim(demo_bbox[1], demo_bbox[3])
        axes[1, 0].set_title('Lane Centerlines')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
        print("stage 003")
        # Plot 4: All elements combined
        visualizer._plot_polygons(axes[1, 1], alpha=0.3, color='lightgray', edgecolor='gray')
        visualizer._plot_lanes(axes[1, 1], alpha=0.5, color='blue', edgecolor='darkblue')
        visualizer._plot_centerlines(axes[1, 1], color='red', linewidth=0.8)
        visualizer._plot_crossings(axes[1, 1], alpha=0.8, color='orange', edgecolor='darkorange')
        axes[1, 1].set_xlim(demo_bbox[0], demo_bbox[2])
        axes[1, 1].set_ylim(demo_bbox[1], demo_bbox[3])
        axes[1, 1].set_title('All Elements')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.3)
        print("stage 004")
        # Save the visualization
        output_file = f'dair_map_demo_{map_name}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f'Demo visualization saved to: {output_file}')
        
        # Get lane count in demo region
        lanes_in_region = visualizer.get_lanes_in_region(demo_bbox)
        print(f'Demo region contains {len(lanes_in_region)} lanes')
        
        print('\n✅ Demo completed successfully!')
        print('🗺️  Map visualization features are working correctly.')
        print(f'📁 Check the generated file: {output_file}')
        
    else:
        print('❌ No maps found in the data directory.')
        
except Exception as e:
    print(f'❌ Error during demo: {e}')
    import traceback
    traceback.print_exc() 