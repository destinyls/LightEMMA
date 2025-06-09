import os
import re
import ast
import argparse
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import json
from matplotlib.patches import Polygon

from utils import *
from vlm import ModelHandler


class EgoTrajectoryVisualizer:
    """
    Visualizer for ego vehicle trajectory in world coordinates with optional point cloud background and 3D bounding boxes
    """
    def __init__(self, figsize=(12, 12), nusc=None):
        self.figsize = figsize
        self.nusc = nusc  # NuScenes instance for point cloud loading
        self.trajectory_colors = {
            'ego_trajectory': '#FF0000',  # Red for ego vehicle trajectory
            'ego_current': '#FF4500',     # Orange red for current position
            'ego_start': '#00FF00',       # Green for start position
            'ego_end': '#0000FF',         # Blue for end position
            'trajectory_line': '#FF6B6B', # Light red for trajectory line
        }
        
        # Define colors for different object categories (same as visualize_bev_coop_veh.py)
        self.category_colors = {
            'car': 'blue',
            'pedestrian': 'orange',
            'bicycle': 'magenta',
            'bus': 'cyan',
            'truck': 'red',
            'motorcycle': 'yellow',
            'default': 'gray'
        }
        
        # Define colors for different directions (based on compass/orientation)
        self.direction_colors = {
            'front': '#00FF00',           # Green - straight ahead
            'front left': '#00FFFF',      # Cyan - front-left
            'front right': '#FFFF00',     # Yellow - front-right  
            'left': '#0000FF',            # Blue - left side
            'right': '#FF8000',           # Orange - right side
            'rear': '#FF0000',            # Red - behind
            'rear left': '#8000FF',       # Purple - rear-left
            'rear right': '#FF0080',      # Pink - rear-right
            'very close front': '#80FF80',     # Light green variants for very close
            'very close front left': '#80FFFF',
            'very close front right': '#FFFF80',
            'very close left': '#8080FF',
            'very close right': '#FFB080',
            'very close rear': '#FF8080',
            'very close rear left': '#B080FF',
            'very close rear right': '#FF80B0',
            'close front': '#40FF40',          # Medium variants for close
            'close front left': '#40FFFF',
            'close front right': '#FFFF40',
            'close left': '#4040FF',
            'close right': '#FF9040',
            'close rear': '#FF4040',
            'close rear left': '#9040FF',
            'close rear right': '#FF4090',
            'far front': '#00CC00',            # Darker variants for far
            'far front left': '#00CCCC',
            'far front right': '#CCCC00',
            'far left': '#0000CC',
            'far right': '#CC6600',
            'far rear': '#CC0000',
            'far rear left': '#6600CC',
            'far rear right': '#CC0066',
            'default': '#808080'               # Gray for unknown directions
        }
    
    def get_point_cloud(self, sample_token):
        """
        Get LiDAR point cloud for a sample and transform to world coordinates
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            points: Nx4 array of point cloud points in world coordinates (x, y, z, intensity)
        """
        if self.nusc is None:
            print("Warning: NuScenes instance not provided, cannot load point cloud")
            return None
            
        try:
            sample = self.nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
            
            # Load point cloud in sensor coordinates
            pc = LidarPointCloud.from_file(lidar_path)
            
            # Get calibration and pose information
            calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            # Transform from sensor to ego vehicle coordinates
            pc.rotate(Quaternion(calibrated_sensor['rotation']).rotation_matrix)
            pc.translate(np.array(calibrated_sensor['translation']))
            
            # Transform from ego vehicle to world coordinates
            pc.rotate(Quaternion(ego_pose['rotation']).rotation_matrix)
            pc.translate(np.array(ego_pose['translation']))
            
            return pc.points.T  # Nx4 array (x, y, z, intensity) in world coordinates
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return None
    
    def filter_points_in_range(self, points, point_cloud_range):
        """
        Filter points within the specified range
        
        Args:
            points: Nx4 array of point cloud points (x, y, z, intensity)
            point_cloud_range: [xmin, ymin, xmax, ymax] range for filtering
            
        Returns:
            filtered_points: Points within the specified range
        """
        if points is None or len(points) == 0:
            return None
            
        x_min, y_min, x_max, y_max = point_cloud_range
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )
        return points[mask]
    
    def get_world_boxes(self, sample_token):
        """
        Get 3D bounding boxes for a sample in world coordinates
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            boxes: List of box dictionaries with position, size, rotation, and category in world frame
        """
        if self.nusc is None:
            print("Warning: NuScenes instance not provided, cannot load boxes")
            return []
            
        try:
            sample = self.nusc.get('sample', sample_token)
            boxes = []
            
            # Get all annotations for this sample (already in world coordinates)
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                
                # Get category - extract the general category type
                category = ann['category_name']
                if '.' in category:
                    category = category.split('.')[1]  # e.g., 'vehicle.car' -> 'car'
                
                # Create box dictionary (annotations are already in world coordinates)
                box = {
                    'position': ann['translation'],  # [x, y, z] in world coordinates
                    'size': ann['size'],             # [width, length, height]
                    'rotation': ann['rotation'],     # quaternion [w, x, y, z]
                    'category': category
                }
                boxes.append(box)
            
            return boxes
        except Exception as e:
            print(f"Error loading 3D boxes: {e}")
            return []
    
    def filter_boxes_in_range(self, boxes, point_cloud_range):
        """
        Filter boxes that are within or partially overlapping with the specified range
        
        Args:
            boxes: List of box dictionaries with position, size, rotation, and category
            point_cloud_range: [xmin, ymin, xmax, ymax] range for filtering
            
        Returns:
            filtered_boxes: Boxes within or partially overlapping with the specified range
        """
        if not boxes:
            return []
            
        x_min, y_min, x_max, y_max = point_cloud_range
        filtered_boxes = []
        
        for box in boxes:
            x, y, _ = box['position']
            width, length, _ = box['size']
            
            # Get a conservative estimate of box bounds by considering max dimensions
            # This simple check will include boxes that might partially overlap with range
            max_dimension = max(length, width) / 2
            
            # Check if the box is completely outside the range
            if (x + max_dimension < x_min or 
                x - max_dimension > x_max or 
                y + max_dimension < y_min or 
                y - max_dimension > y_max):
                continue
            
            filtered_boxes.append(box)
            
        return filtered_boxes
    
    def _plot_box(self, ax, box, box_alpha=0.7, arrow_alpha=0.8):
        """
        Plot a 3D box in BEV (Bird's Eye View)
        
        Args:
            ax: Matplotlib axis
            box: Box dictionary with position, size, rotation, and category
            box_alpha: Transparency for 3D bounding boxes
            arrow_alpha: Transparency for direction arrows
        """
        # Extract box parameters
        x, y, z = box['position']
        width, length, height = box['size']
        quaternion = box['rotation']
        category = box['category']
        
        # Convert quaternion to yaw (rotation around z-axis)
        yaw = quaternion_to_yaw(quaternion)
        
        # Get color for category
        color = self.category_colors.get(category, self.category_colors['default'])
        
        # Calculate corners of the rectangle
        corners = self._get_corners(x, y, length, width, yaw)
        
        # Plot rectangle
        rect = Polygon(corners, fill=True, alpha=box_alpha, color=color, 
                      edgecolor='black', linewidth=1, zorder=7)
        ax.add_patch(rect)
        
        # Plot direction arrow
        arrow_length = length * 0.5
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        arrow = Arrow(x, y, dx, dy, width=width*0.5, color='red', 
                     alpha=arrow_alpha, zorder=8)
        ax.add_patch(arrow)
    
    def _get_corners(self, x, y, length, width, yaw):
        """
        Get the four corners of a rotated rectangle
        
        Args:
            x, y: Center coordinates
            length, width: Rectangle dimensions
            yaw: Rotation angle in radians
            
        Returns:
            corners: List of (x, y) coordinates for the corners
        """
        # Calculate half dimensions
        half_length = length / 2
        half_width = width / 2
        
        # Calculate corners (centered at origin, unrotated)
        corners = [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ]
        
        # Rotate corners
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotated_corners = []
        for cx, cy in corners:
            # Rotate
            rx = cx * cos_yaw - cy * sin_yaw
            ry = cx * sin_yaw + cy * cos_yaw
            # Translate
            rx += x
            ry += y
            rotated_corners.append((rx, ry))
        
        return rotated_corners
    
    def visualize_scene_trajectory(self, ego_positions, ego_headings, scene_name, 
                                 save_path=None, show_plot=True, current_index=None,
                                 show_orientation_arrows=True, trajectory_line_width=3, 
                                 point_size=50, show_ego_axes=True, ego_axes_interval=5,
                                 show_point_cloud=False, first_sample_token=None,
                                 point_cloud_alpha=0.6, point_cloud_size=1.0,
                                 show_3d_boxes=False, box_alpha=0.7, arrow_alpha=0.8):
        """
        Visualize ego vehicle trajectory for a complete scene in world coordinates with optional point cloud background and 3D bounding boxes
        
        Args:
            ego_positions: List of ego positions [(x, y), ...]
            ego_headings: List of ego headings [yaw1, yaw2, ...]
            scene_name: Name of the scene
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
            current_index: Index of current position to highlight (optional)
            show_orientation_arrows: Whether to show orientation arrows
            trajectory_line_width: Width of the trajectory line
            point_size: Size of trajectory points
            show_ego_axes: Whether to show ego coordinate axes
            ego_axes_interval: Interval for showing ego axes (every N points)
            show_point_cloud: Whether to show point cloud background from first frame
            first_sample_token: Sample token for the first frame (required if show_point_cloud=True or show_3d_boxes=True)
            point_cloud_alpha: Transparency of point cloud points
            point_cloud_size: Size of point cloud points
            show_3d_boxes: Whether to show 3D bounding boxes from first frame
            box_alpha: Transparency for 3D bounding boxes
            arrow_alpha: Transparency for direction arrows in boxes
            
        Returns:
            fig, ax: Figure and axis objects
        """
        if not ego_positions or len(ego_positions) < 2:
            print("Insufficient trajectory points for visualization!")
            return None, None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract trajectory coordinates
        traj_x = [pos[0] for pos in ego_positions]
        traj_y = [pos[1] for pos in ego_positions]
        
        # Calculate trajectory bounds and adjust visualization range
        traj_x_min, traj_x_max = min(traj_x), max(traj_x)
        traj_y_min, traj_y_max = min(traj_y), max(traj_y)
        
        # Calculate trajectory span
        traj_x_span = traj_x_max - traj_x_min
        traj_y_span = traj_y_max - traj_y_min
        
        # Add padding based on trajectory span (minimum 50m, or 20% of span)
        padding_x = max(50, traj_x_span * 0.2)
        padding_y = max(50, traj_y_span * 0.2)
        
        # Ensure minimum visualization area (at least 200m x 200m)
        min_range = 200
        if traj_x_span + 2 * padding_x < min_range:
            padding_x = (min_range - traj_x_span) / 2
        if traj_y_span + 2 * padding_y < min_range:
            padding_y = (min_range - traj_y_span) / 2
        
        # Calculate adjusted range to fully cover trajectory
        adjusted_range = [
            traj_x_min - padding_x,
            traj_y_min - padding_y,
            traj_x_max + padding_x,
            traj_y_max + padding_y
        ]
        
        print(f"Trajectory bounds: X[{traj_x_min:.1f}, {traj_x_max:.1f}], Y[{traj_y_min:.1f}, {traj_y_max:.1f}]")
        print(f"Trajectory span: X={traj_x_span:.1f}m, Y={traj_y_span:.1f}m")
        print(f"Applied padding: X={padding_x:.1f}m, Y={padding_y:.1f}m")
        print(f"Final visualization range: X[{adjusted_range[0]:.1f}, {adjusted_range[2]:.1f}], Y[{adjusted_range[1]:.1f}, {adjusted_range[3]:.1f}]")
        
        # Load and render point cloud background if requested
        first_frame_points = None
        if show_point_cloud and first_sample_token and self.nusc:
            try:
                print("Loading point cloud from first frame...")
                first_frame_points = self.get_point_cloud(first_sample_token)
                if first_frame_points is not None:
                    print(f"Loaded {len(first_frame_points)} total points")
                    first_frame_points = self.filter_points_in_range(first_frame_points, adjusted_range)
                    if first_frame_points is not None:
                        print(f"Filtered to {len(first_frame_points)} points within visualization range")
                    else:
                        print("No points found within visualization range")
                else:
                    print("Failed to load point cloud")
            except Exception as e:
                print(f"Error loading point cloud: {e}")
                first_frame_points = None
        
        # Load and render 3D bounding boxes if requested
        first_frame_boxes = []
        if show_3d_boxes and first_sample_token and self.nusc:
            try:
                print("Loading 3D bounding boxes from first frame...")
                first_frame_boxes = self.get_world_boxes_from_infrastructure(first_sample_token)
                if first_frame_boxes:
                    print(f"Loaded {len(first_frame_boxes)} total boxes")
                    first_frame_boxes = self.filter_boxes_in_range(first_frame_boxes, adjusted_range)
                    print(f"Filtered to {len(first_frame_boxes)} boxes within visualization range")
                else:
                    print("No 3D boxes found in first frame")
            except Exception as e:
                print(f"Error loading 3D boxes: {e}")
                first_frame_boxes = []
        
        # Set clean background color (same as visualize_bev_seq.py)
        ax.set_facecolor('#FAFAFA')  # Very light gray background
        
        # Plot point cloud background if available (use fixed parameters like visualize_bev_seq.py)
        scatter = None
        if first_frame_points is not None and len(first_frame_points) > 0:
            print("Rendering point cloud background...")
            # Use height (z) for coloring, with a suitable colormap (same parameters as visualize_bev_seq.py)
            scatter = ax.scatter(first_frame_points[:, 0], first_frame_points[:, 1], 
                               s=1.0, c=first_frame_points[:, 2], cmap='viridis', 
                               alpha=0.6, vmin=first_frame_points[:, 2].min(), 
                               vmax=first_frame_points[:, 2].max(), zorder=3)
        
        # Plot 3D bounding boxes if available
        if first_frame_boxes:
            print(f"Rendering {len(first_frame_boxes)} 3D bounding boxes...")
            for box in first_frame_boxes:
                self._plot_box(ax, box, box_alpha=box_alpha, arrow_alpha=arrow_alpha)
        
        # Plot trajectory line
        ax.plot(traj_x, traj_y, color=self.trajectory_colors['trajectory_line'], 
               linewidth=trajectory_line_width, alpha=0.8, zorder=5, 
               label=f'Ego Trajectory ({len(ego_positions)} points)')
        
        # Plot trajectory points
        ax.scatter(traj_x, traj_y, c=self.trajectory_colors['ego_trajectory'], 
                  s=point_size, alpha=0.7, zorder=6, edgecolors='white', linewidth=1)
        
        # Mark start and end positions
        if len(ego_positions) > 0:
            # Start position
            start_x, start_y = ego_positions[0][0], ego_positions[0][1]
            ax.plot(start_x, start_y, 'o', color=self.trajectory_colors['ego_start'], 
                   markersize=12, markeredgecolor='white', markeredgewidth=2, 
                   label='Start Position', zorder=8)
            
            # End position
            if len(ego_positions) > 1:
                end_x, end_y = ego_positions[-1][0], ego_positions[-1][1]
                ax.plot(end_x, end_y, 's', color=self.trajectory_colors['ego_end'], 
                       markersize=12, markeredgecolor='white', markeredgewidth=2, 
                       label='End Position', zorder=8)
        
        # Highlight current position if specified
        if current_index is not None and 0 <= current_index < len(ego_positions):
            curr_x, curr_y = ego_positions[current_index][0], ego_positions[current_index][1]
            ax.plot(curr_x, curr_y, 'o', color=self.trajectory_colors['ego_current'], 
                   markersize=15, markeredgecolor='white', markeredgewidth=3, 
                   label='Current Position', zorder=9)
        
        # Show orientation arrows if requested
        if show_orientation_arrows and len(ego_headings) > 1:
            # Show arrows at regular intervals
            arrow_step = max(1, len(ego_positions) // 10)  # Show about 10 arrows
            for i in range(0, len(ego_positions), arrow_step):
                if i < len(ego_headings):
                    x, y = ego_positions[i][0], ego_positions[i][1]
                    yaw = ego_headings[i]
                    
                    arrow_length = 8
                    dx = arrow_length * np.cos(yaw)
                    dy = arrow_length * np.sin(yaw)
                    
                    ax.arrow(x, y, dx, dy, head_width=3, head_length=2, 
                            fc=self.trajectory_colors['ego_current'], 
                            ec='white', linewidth=1, alpha=0.8, zorder=7)
        
        # Show ego coordinate axes if requested
        if show_ego_axes and len(ego_positions) > 0:
            # Show ego axes at regular intervals
            axes_step = max(1, ego_axes_interval)
            for i in range(0, len(ego_positions), axes_step):
                if i < len(ego_headings):
                    self.draw_ego_coordinate_axes(
                        ax, ego_positions[i], ego_headings[i],
                        axis_length=12, line_width=2, alpha=0.8, zorder=9
                    )
        
        # Set axis limits
        ax.set_xlim(adjusted_range[0], adjusted_range[2])
        ax.set_ylim(adjusted_range[1], adjusted_range[3])
        
        # Set axis labels and title
        ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
        
        title = f'Ego Vehicle Trajectory\nScene: {scene_name}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add trajectory info text box (same format as visualize_bev_seq.py)
        distance = self._calculate_trajectory_distance(ego_positions)
        
        # Calculate visualization area
        vis_width = adjusted_range[2] - adjusted_range[0]
        vis_height = adjusted_range[3] - adjusted_range[1]
        
        # Prepare info text (same format as visualize_bev_seq.py)
        info_text = (f"Trajectory Info:\n"
                    f"Points: {len(ego_positions)}\n"
                    f"Distance: {distance:.1f}m\n"
                    f"View Area: {vis_width:.0f}×{vis_height:.0f}m")
        
        # Add point cloud info if available
        if first_frame_points is not None:
            info_text += f"\nLiDAR Points: {len(first_frame_points):,}"
        
        # Add 3D box category info if available
        if first_frame_boxes:
            info_text += f"\n3D Boxes: {len(first_frame_boxes)}"
            # Count boxes by category
            category_counts = {}
            for box in first_frame_boxes:
                category = box['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Add category breakdown
            for category, count in sorted(category_counts.items()):
                info_text += f"\n  {category}: {count}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightblue', alpha=0.8), zorder=10)
        
        # Add legend for 3D box categories if boxes are displayed
        if first_frame_boxes:
            # Create legend handles for box categories
            handles = []
            labels = []
            
            # Get unique categories from displayed boxes
            displayed_categories = set(box['category'] for box in first_frame_boxes)
            
            for category in sorted(displayed_categories):
                color = self.category_colors.get(category, self.category_colors['default'])
                handle = plt.Rectangle((0, 0), 1, 1, color=color, alpha=box_alpha)
                handles.append(handle)
                labels.append(f'{category}')
            
            # Add legend if there are categories to show
            if handles:
                ax.legend(handles, labels, loc='upper right', 
                         title='Object Categories', framealpha=0.9, 
                         bbox_to_anchor=(0.98, 0.98))
        else:
            # Add simple legend for trajectory elements
            trajectory_handles = []
            trajectory_labels = []
            
            # Trajectory line
            trajectory_handles.append(plt.Line2D([0], [0], color=self.trajectory_colors['trajectory_line'], 
                                               linewidth=trajectory_line_width, alpha=0.8))
            trajectory_labels.append('Ego Trajectory')
            
            # Start position
            if len(ego_positions) > 0:
                trajectory_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor=self.trajectory_colors['ego_start'], 
                                                   markersize=8, markeredgecolor='white', 
                                                   markeredgewidth=2, linestyle='None'))
                trajectory_labels.append('Start Position')
            
            # End position
            if len(ego_positions) > 1:
                trajectory_handles.append(plt.Line2D([0], [0], marker='s', color='w', 
                                                   markerfacecolor=self.trajectory_colors['ego_end'], 
                                                   markersize=8, markeredgecolor='white', 
                                                   markeredgewidth=2, linestyle='None'))
                trajectory_labels.append('End Position')
            
            # Add legend
            if trajectory_handles:
                ax.legend(trajectory_handles, trajectory_labels, loc='upper right', 
                         framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        
        # Save if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Trajectory visualization saved to: {save_path}")
        
        # Show plot if required
        if show_plot:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def draw_ego_coordinate_axes(self, ax, position, heading, axis_length=15, 
                                line_width=3, alpha=0.9, zorder=10):
        """
        Draw ego vehicle coordinate axes (x: forward, y: left) in world coordinates
        
        Args:
            ax: Matplotlib axis object
            position: Ego vehicle position (x, y)
            heading: Ego vehicle heading (yaw angle in radians)
            axis_length: Length of coordinate axes in meters
            line_width: Width of axis lines
            alpha: Transparency of axes
            zorder: Drawing order
        """
        ego_x, ego_y = position[0], position[1]
        yaw = heading
        
        # Calculate axis directions
        # X-axis (forward direction) - Red
        x_axis_dx = axis_length * np.cos(yaw)
        x_axis_dy = axis_length * np.sin(yaw)
        
        # Y-axis (left direction) - Green  
        y_axis_dx = axis_length * np.cos(yaw + np.pi/2)
        y_axis_dy = axis_length * np.sin(yaw + np.pi/2)
        
        # Draw X-axis (forward, red)
        ax.arrow(ego_x, ego_y, x_axis_dx, x_axis_dy, 
                head_width=axis_length*0.15, head_length=axis_length*0.1,
                fc='red', ec='darkred', linewidth=line_width, 
                alpha=alpha, zorder=zorder)
        
        # Draw Y-axis (left, green)
        ax.arrow(ego_x, ego_y, y_axis_dx, y_axis_dy,
                head_width=axis_length*0.15, head_length=axis_length*0.1,
                fc='green', ec='darkgreen', linewidth=line_width,
                alpha=alpha, zorder=zorder)
        
        # Add text labels
        # X-axis label
        x_label_x = ego_x + x_axis_dx * 1.2
        x_label_y = ego_y + x_axis_dy * 1.2
        ax.text(x_label_x, x_label_y, 'X', fontsize=12, fontweight='bold',
               color='red', ha='center', va='center', zorder=zorder+1)
        
        # Y-axis label  
        y_label_x = ego_x + y_axis_dx * 1.2
        y_label_y = ego_y + y_axis_dy * 1.2
        ax.text(y_label_x, y_label_y, 'Y', fontsize=12, fontweight='bold',
               color='green', ha='center', va='center', zorder=zorder+1)
        
        # Add origin point
        ax.plot(ego_x, ego_y, 'ko', markersize=8, markeredgecolor='white', 
               markeredgewidth=2, zorder=zorder+1)
    
    def _calculate_trajectory_distance(self, positions):
        """
        Calculate total distance traveled along the trajectory
        
        Args:
            positions: List of positions [(x, y), ...]
            
        Returns:
            total_distance: Total distance in meters
        """
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            prev_pos = np.array(positions[i-1])
            curr_pos = np.array(positions[i])
            distance = np.linalg.norm(curr_pos - prev_pos)
            total_distance += distance
        
        return total_distance

    def overlay_trajectory_on_first_frame(self, first_frame_image_path, ego_positions, 
                                        ego_headings, camera_params, scene_name,
                                        save_path=None, trajectory_color=(255, 0, 0),
                                        line_width=3, point_size=8, show_arrows=True,
                                        arrow_interval=5, alpha=0.8):
        """
        Overlay the complete sequence trajectory on the first frame image
        
        Args:
            first_frame_image_path: Path to the first frame image
            ego_positions: List of ego positions [(x, y), ...] in world coordinates
            ego_headings: List of ego headings [yaw1, yaw2, ...] in radians
            camera_params: Camera parameters including rotation, translation, and intrinsic matrix
            scene_name: Name of the scene
            save_path: Path to save the visualization
            trajectory_color: RGB color for trajectory (default: red)
            line_width: Width of trajectory line
            point_size: Size of trajectory points
            show_arrows: Whether to show orientation arrows
            arrow_interval: Interval for showing arrows (every N points)
            alpha: Transparency of trajectory overlay
            
        Returns:
            success: Boolean indicating if visualization was successful
        """
        try:
            # Load the first frame image
            original_img = cv2.imread(first_frame_image_path)
            if original_img is None:
                print(f"Failed to load image from path: {first_frame_image_path}")
                return False

            img = original_img.copy()
            
            # Get first frame ego position and heading for transformation
            first_ego_pos = ego_positions[0]
            first_ego_heading = ego_headings[0]
            
            # --- Form Transformation Matrices ---
            # Ego to global transformation matrix
            T_ego_global = np.eye(4)
            T_ego_global[:3, :3] = np.array([
                [np.cos(first_ego_heading), -np.sin(first_ego_heading), 0],
                [np.sin(first_ego_heading), np.cos(first_ego_heading), 0],
                [0, 0, 1],
            ])
            T_ego_global[:3, 3] = np.array([first_ego_pos[0], first_ego_pos[1], 0])

            # Camera to ego transformation matrix
            T_cam_ego = np.eye(4)
            # T_cam_ego[:3, :3] = Quaternion(camera_params["rotation"]).rotation_matrix
            T_cam_ego[:3, :3] = np.array(camera_params["rotation"])
            T_cam_ego[:3, 3] = np.array(camera_params["translation"])

            # Camera to global transformation matrix
            T_cam_global = T_ego_global @ T_cam_ego
            T_global_cam = np.linalg.inv(T_cam_global)

            # Transform world trajectory points to camera coordinates
            points3d_world = [np.array([pos[0], pos[1], 0.0]) for pos in ego_positions]
            points3d_cam = np.array(
                [(T_global_cam @ np.append(p, 1))[:3] for p in points3d_world]
            )
            
            # Filter points that are in front of the camera
            valid = points3d_cam[:, 2] > 0
            if not valid.any():
                print("No trajectory points are visible in the camera view")
                return False
                
            # Project valid points onto the image plane
            points_valid = points3d_cam[valid]
            valid_indices = np.where(valid)[0]
            
            proj = (np.array(camera_params["camera_intrinsic"]) @ points_valid.T).T
            points2d_img = proj[:, :2] / proj[:, 2][:, np.newaxis]
            
            # Filter points that are within image bounds
            img_height, img_width = img.shape[:2]
            in_bounds = (
                (points2d_img[:, 0] >= 0) & (points2d_img[:, 0] < img_width) &
                (points2d_img[:, 1] >= 0) & (points2d_img[:, 1] < img_height)
            )
            
            if not in_bounds.any():
                print("No trajectory points are within image bounds")
                return False
            
            # Get final valid points and their indices
            final_points = points2d_img[in_bounds]
            final_indices = valid_indices[in_bounds]
            
            print(f"Projecting {len(final_points)} trajectory points onto image (with 1.5m downward offset)")
            
            # Create figure for overlay
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.set_position([0, 0, 1, 1])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)

            # Convert BGR color to RGB and normalize
            color_rgb = (trajectory_color[2]/255, trajectory_color[1]/255, trajectory_color[0]/255, alpha)
            
            if len(final_points) > 1:
                # Draw trajectory line
                x_coords = final_points[:, 0]
                y_coords = final_points[:, 1]
                ax.plot(x_coords, y_coords, color=color_rgb, linewidth=line_width, 
                       linestyle='solid', alpha=alpha, label='Ego Trajectory')
                
                # Draw trajectory points
                ax.scatter(x_coords, y_coords, c=[color_rgb], s=point_size**2, 
                          alpha=alpha, edgecolors='white', linewidth=1, zorder=6)
                
                # Draw orientation arrows if requested
                if show_arrows and len(final_indices) > 1:
                    arrow_step = max(1, arrow_interval)
                    for i in range(0, len(final_indices), arrow_step):
                        if i < len(final_indices) - 1:
                            # Get current and next point for arrow direction
                            curr_idx = final_indices[i]
                            next_idx = final_indices[min(i + 1, len(final_indices) - 1)]
                            
                            if curr_idx < len(ego_headings):
                                # Use heading for arrow direction
                                yaw = ego_headings[curr_idx]
                                
                                # Project arrow direction to image coordinates
                                # Apply same 1.5m downward offset for arrow start point
                                arrow_start_world = np.array([ego_positions[curr_idx][0], ego_positions[curr_idx][1], -1.5])
                                arrow_length_world = 5.0  # 5 meters in world coordinates
                                arrow_end_world = arrow_start_world + np.array([
                                    arrow_length_world * np.cos(yaw),
                                    arrow_length_world * np.sin(yaw),
                                    0
                                ])
                                
                                # Transform arrow end to camera coordinates
                                arrow_end_cam = (T_global_cam @ np.append(arrow_end_world, 1))[:3]
                                
                                if arrow_end_cam[2] > 0:  # Check if arrow end is in front of camera
                                    # Project to image coordinates
                                    arrow_end_proj = np.array(camera_params["camera_intrinsic"]) @ arrow_end_cam
                                    arrow_end_img = arrow_end_proj[:2] / arrow_end_proj[2]
                                    
                                    # Draw arrow
                                    start_point = final_points[i]
                                    arrow_vector = arrow_end_img - start_point
                                    arrow_norm = np.linalg.norm(arrow_vector)
                                    
                                    if arrow_norm > 10:  # Only draw if arrow is long enough in pixels
                                        # Normalize and scale arrow
                                        arrow_vector = arrow_vector / arrow_norm * 20  # 20 pixel length
                                        
                                        ax.annotate('', xy=start_point + arrow_vector, xytext=start_point,
                                                  arrowprops=dict(arrowstyle='->', color=color_rgb, 
                                                                lw=line_width, mutation_scale=15))
            
            # Add start and end markers
            if len(final_points) > 0:
                # Start point (green)
                ax.plot(final_points[0, 0], final_points[0, 1], 'o', 
                       color='green', markersize=12, markeredgecolor='white', 
                       markeredgewidth=2, label='Start', zorder=8)
                
                # End point (blue) 
                if len(final_points) > 1:
                    ax.plot(final_points[-1, 0], final_points[-1, 1], 's', 
                           color='blue', markersize=12, markeredgecolor='white', 
                           markeredgewidth=2, label='End', zorder=8)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
            
            # Add title
            ax.text(0.5, 0.02, f'Ego Trajectory Overlay - Scene: {scene_name}', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   ha='center', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            # Convert matplotlib figure to OpenCV image
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            # Remove any extra padding
            if buf.shape[0] > img.shape[0]:
                buf = buf[:img.shape[0], :, :]
            if buf.shape[1] > img.shape[1]:
                buf = buf[:, :img.shape[1], :]
                
            result_img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
            # Close matplotlib figure to prevent memory leaks
            plt.close(fig)

            # Save the result
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result_img)
                print(f"Trajectory overlay saved to: {save_path}")
            else:
                # Use default path
                default_path = f"trajectory_overlay_{scene_name}.png"
                cv2.imwrite(default_path, result_img)
                print(f"Trajectory overlay saved to: {default_path}")
            
            return True
            
        except Exception as e:
            print(f"Error creating trajectory overlay: {e}")
            return False

    def visualize_current_frame_in_trajectory(self, ego_positions, ego_headings, scene_name,
                                            current_index, current_sample_token, 
                                            save_path=None, show_plot=False,
                                            show_point_cloud=True, show_3d_boxes=True,
                                            point_cloud_alpha=0.6, point_cloud_size=1.0,
                                            box_alpha=0.7, arrow_alpha=0.8,
                                            trajectory_line_width=2, point_size=30,
                                            show_descriptive_labels=True, label_font_size=9,
                                            leader_line_alpha=0.8, label_background_alpha=0.9):
        """
        Visualize current frame position in complete trajectory with point cloud and 3D boxes
        
        Args:
            ego_positions: List of ego positions [(x, y), ...] for complete trajectory
            ego_headings: List of ego headings [yaw1, yaw2, ...] for complete trajectory
            scene_name: Name of the scene
            current_index: Index of current frame in the trajectory
            current_sample_token: Sample token for the current frame
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
            show_point_cloud: Whether to show point cloud background from current frame
            show_3d_boxes: Whether to show 3D bounding boxes from current frame
            point_cloud_alpha: Transparency of point cloud points
            point_cloud_size: Size of point cloud points
            box_alpha: Transparency for 3D bounding boxes
            arrow_alpha: Transparency for direction arrows in boxes
            trajectory_line_width: Width of the trajectory line
            point_size: Size of trajectory points
            show_descriptive_labels: Whether to show descriptive text labels for objects
            label_font_size: Font size for descriptive labels
            leader_line_alpha: Transparency for leader lines
            label_background_alpha: Transparency for label backgrounds
            
        Returns:
            fig, ax: Figure and axis objects
        """
        if not ego_positions or len(ego_positions) < 2:
            print("Insufficient trajectory points for visualization!")
            return None, None
        
        if current_index < 0 or current_index >= len(ego_positions):
            print(f"Invalid current_index: {current_index}")
            return None, None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract trajectory coordinates
        traj_x = [pos[0] for pos in ego_positions]
        traj_y = [pos[1] for pos in ego_positions]
        
        # Get current position for centering the view
        curr_pos = ego_positions[current_index]
        curr_x, curr_y = curr_pos[0], curr_pos[1]
        
        # Calculate visualization range centered on current position
        view_range = 100  # 100m x 100m view
        adjusted_range = [
            curr_x - view_range,
            curr_y - view_range,
            curr_x + view_range,
            curr_y + view_range
        ]
        
        print(f"Current position: X={curr_x:.1f}, Y={curr_y:.1f}")
        print(f"Visualization range: X[{adjusted_range[0]:.1f}, {adjusted_range[2]:.1f}], Y[{adjusted_range[1]:.1f}, {adjusted_range[3]:.1f}]")
        
        # Load and render point cloud background if requested (reuse existing logic)
        current_frame_points = None
        if show_point_cloud and current_sample_token and self.nusc:
            try:
                print("Loading point cloud from current frame...")
                current_frame_points = self.get_point_cloud(current_sample_token)
                if current_frame_points is not None:
                    print(f"Loaded {len(current_frame_points)} total points")
                    current_frame_points = self.filter_points_in_range(current_frame_points, adjusted_range)
                    if current_frame_points is not None:
                        print(f"Filtered to {len(current_frame_points)} points within visualization range")
                    else:
                        print("No points found within visualization range")
                else:
                    print("Failed to load point cloud")
            except Exception as e:
                print(f"Error loading point cloud: {e}")
                current_frame_points = None
        
        # Load and render 3D bounding boxes if requested (reuse existing logic)
        current_frame_boxes = []
        if show_3d_boxes and current_sample_token and self.nusc:
            try:
                current_frame_boxes = self.get_world_boxes_from_infrastructure(current_sample_token)
                if current_frame_boxes:
                    current_frame_boxes = self.filter_boxes_in_range(current_frame_boxes, adjusted_range)
                else:
                    print("No 3D boxes found in current frame")
            except Exception as e:
                print(f"Error loading 3D boxes: {e}")
                current_frame_boxes = []
        
        # Get ego-coordinate boxes and descriptive information for text labels
        ego_coordinate_boxes = []
        descriptive_boxes = []
        if show_descriptive_labels and current_sample_token and self.nusc:
            try:
                ego_coordinate_boxes = self.get_ego_boxes(current_sample_token)
                if ego_coordinate_boxes:
                    # Filter boxes within reasonable range for labeling (closer to ego)
                    label_range = min(50, view_range)  # Use 50m or current view range, whichever is smaller
                    ego_filtered_boxes = []
                    for box in ego_coordinate_boxes:
                        pos = np.array(box['position'])
                        distance = np.sqrt(pos[0]**2 + pos[1]**2)
                        if distance <= label_range:
                            ego_filtered_boxes.append(box)
                    
                    descriptive_boxes = self.to_descriptive_box(ego_filtered_boxes)
                else:
                    print("No ego coordinate boxes found for descriptive labels")
            except Exception as e:
                print(f"Error loading ego coordinate boxes for labels: {e}")
        
        # Set clean background color (reuse existing style)
        ax.set_facecolor('#FAFAFA')
        
        # Plot point cloud background if available (reuse existing logic)
        if current_frame_points is not None and len(current_frame_points) > 0:
            ax.scatter(current_frame_points[:, 0], current_frame_points[:, 1], 
                      s=point_cloud_size, c=current_frame_points[:, 2], cmap='viridis', 
                      alpha=point_cloud_alpha, vmin=current_frame_points[:, 2].min(), 
                      vmax=current_frame_points[:, 2].max(), zorder=3)
        
        # Plot 3D bounding boxes if available (reuse existing logic)
        if current_frame_boxes:
            for box in current_frame_boxes:
                self._plot_box(ax, box, box_alpha=box_alpha, arrow_alpha=arrow_alpha)
        
        # Plot complete trajectory line (reuse existing style)
        ax.plot(traj_x, traj_y, color=self.trajectory_colors['trajectory_line'], 
               linewidth=trajectory_line_width, alpha=0.6, zorder=5, 
               label=f'Complete Trajectory ({len(ego_positions)} points)')
        
        # Plot trajectory points with different colors for past/future
        past_indices = list(range(0, current_index))
        future_indices = list(range(current_index + 1, len(ego_positions)))
        
        # Plot past trajectory points
        if past_indices:
            past_x = [traj_x[i] for i in past_indices]
            past_y = [traj_y[i] for i in past_indices]
            ax.scatter(past_x, past_y, c='gray', s=point_size, alpha=0.5, zorder=6, 
                      edgecolors='white', linewidth=1, label='Past Trajectory')
        
        # Plot future trajectory points
        if future_indices:
            future_x = [traj_x[i] for i in future_indices]
            future_y = [traj_y[i] for i in future_indices]
            ax.scatter(future_x, future_y, c='lightblue', s=point_size, alpha=0.7, zorder=6, 
                      edgecolors='white', linewidth=1, label='Future Trajectory')
        
        # Highlight current position (reuse existing style)
        ax.plot(curr_x, curr_y, 'o', color=self.trajectory_colors['ego_current'], 
               markersize=15, markeredgecolor='white', markeredgewidth=3, 
               label='Current Position', zorder=9)
        
        # Show current orientation arrow
        if current_index < len(ego_headings):
            yaw = ego_headings[current_index]
            arrow_length = 12
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            
            ax.arrow(curr_x, curr_y, dx, dy, head_width=4, head_length=3, 
                    fc=self.trajectory_colors['ego_current'], 
                    ec='white', linewidth=2, alpha=0.9, zorder=8)
        
        # Draw current ego coordinate axes (reuse existing method)
        if current_index < len(ego_headings):
            self.draw_ego_coordinate_axes(
                ax, curr_pos, ego_headings[current_index],
                axis_length=15, line_width=3, alpha=0.9, zorder=10
            )
        
        # Add descriptive text labels with leader lines
        if show_descriptive_labels and descriptive_boxes:
            self._add_descriptive_labels_with_leaders(
                ax, descriptive_boxes, curr_pos, ego_headings[current_index],
                adjusted_range, label_font_size, leader_line_alpha, label_background_alpha
            )
        
        # Set axis limits
        ax.set_xlim(adjusted_range[0], adjusted_range[2])
        ax.set_ylim(adjusted_range[1], adjusted_range[3])
        
        # Set axis labels and title
        ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
        
        title = f'Current Frame in Trajectory\nScene: {scene_name}, Frame: {current_index}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add frame info text box (reuse existing style)
        total_distance = self._calculate_trajectory_distance(ego_positions)
        current_distance = self._calculate_trajectory_distance(ego_positions[:current_index+1])
        
        info_text = (f"Frame Info:\n"
                    f"Current: {current_index}/{len(ego_positions)-1}\n"
                    f"Progress: {current_distance:.1f}m/{total_distance:.1f}m\n"
                    f"View: {view_range*2}×{view_range*2}m")
        
        # Add point cloud info if available
        if current_frame_points is not None:
            info_text += f"\nLiDAR Points: {len(current_frame_points)}"
        
        # Add 3D box category info if available
        if current_frame_boxes:
            info_text += f"\n3D Boxes: {len(current_frame_boxes)}"
            # Count boxes by category
            category_counts = {}
            for box in current_frame_boxes:
                category = box['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Add category breakdown
            for category, count in sorted(category_counts.items()):
                info_text += f"\n  {category}: {count}"
        
        # Add descriptive labels info
        if descriptive_boxes:
            info_text += f"\nLabeled Objects: {len(descriptive_boxes)}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightblue', alpha=0.8), zorder=10)
        
        # Add legend (reuse existing logic)
        if current_frame_boxes:
            # Create legend handles for box categories and directions
            handles = []
            labels = []
            
            # Add trajectory elements first
            handles.append(plt.Line2D([0], [0], color=self.trajectory_colors['trajectory_line'], 
                                   linewidth=trajectory_line_width, alpha=0.6))
            labels.append('Complete Trajectory')
            
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=self.trajectory_colors['ego_current'], 
                                   markersize=8, markeredgecolor='white', 
                                   markeredgewidth=2, linestyle='None'))
            labels.append('Current Position')
            
            if show_descriptive_labels and descriptive_boxes:
                # Show direction-based colors when descriptive labels are enabled
                displayed_directions = set()
                for desc_box in descriptive_boxes:
                    direction = desc_box['relative_direction'].replace('-', ' ')
                    displayed_directions.add(direction)
                
                for direction in sorted(displayed_directions):
                    color = self.direction_colors.get(direction, self.direction_colors['default'])
                    handle = plt.Rectangle((0, 0), 1, 1, color=color, alpha=label_background_alpha)
                    handles.append(handle)
                    labels.append(f'{direction}')
                
                title = 'Elements & Directions'
            else:
                # Show category-based colors when descriptive labels are disabled
                displayed_categories = set(box['category'] for box in current_frame_boxes)
                
                for category in sorted(displayed_categories):
                    color = self.category_colors.get(category, self.category_colors['default'])
                    handle = plt.Rectangle((0, 0), 1, 1, color=color, alpha=box_alpha)
                    handles.append(handle)
                    labels.append(f'{category}')
                
                title = 'Elements & Categories'
            
            ax.legend(handles, labels, loc='upper right', 
                     title=title, framealpha=0.9, 
                     bbox_to_anchor=(0.98, 0.98))
        else:
            # Simple legend for trajectory elements
            trajectory_handles = []
            trajectory_labels = []
            
            trajectory_handles.append(plt.Line2D([0], [0], color=self.trajectory_colors['trajectory_line'], 
                                               linewidth=trajectory_line_width, alpha=0.6))
            trajectory_labels.append('Complete Trajectory')
            
            trajectory_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=self.trajectory_colors['ego_current'], 
                                               markersize=8, markeredgecolor='white', 
                                               markeredgewidth=2, linestyle='None'))
            trajectory_labels.append('Current Position')
            
            if past_indices:
                trajectory_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor='gray', 
                                                   markersize=6, markeredgecolor='white', 
                                                   markeredgewidth=1, linestyle='None'))
                trajectory_labels.append('Past Trajectory')
            
            if future_indices:
                trajectory_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor='lightblue', 
                                                   markersize=6, markeredgecolor='white', 
                                                   markeredgewidth=1, linestyle='None'))
                trajectory_labels.append('Future Trajectory')
            
            ax.legend(trajectory_handles, trajectory_labels, loc='upper right', 
                     framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        
        # Save if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Current frame visualization saved to: {save_path}")
        
        # Show plot if required
        if show_plot:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax

    def get_ego_boxes_from_infrastructure(self, sample_token):
        """
        Get 3D bounding boxes for a sample in infrastructure coordinates and convert to ego coordinates
        Args:
            sample_token: Token of the sample
            
        Returns:
            boxes: List of box dictionaries with position, size, rotation, and category in ego frame
        """
        if self.nusc is None:
            print("Warning: NuScenes instance not provided, cannot load infrastructure boxes")
            return []
        
        try:
            # Load v2i_pair.json for infrastructure frame mapping
            v2i_path = "data/v2x-seq-nuscenes/cooperative/v2i_pair.json"
            try:
                with open(v2i_path, 'r') as f:
                    v2i_pairs = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load v2i_pair.json: {e}")
                return []
            
            # Get vehicle_sequence from scene
            sample = self.nusc.get('sample', sample_token)
            scene = self.nusc.get('scene', sample['scene_token'])
            vehicle_sequence = scene['token']
            vehicle_frame = sample_token
            
            # Check if this vehicle frame has a paired infrastructure frame
            if not (vehicle_sequence in v2i_pairs and 
                    vehicle_frame in v2i_pairs[vehicle_sequence]):
                print(f"No paired infrastructure frame found for vehicle frame {vehicle_frame}")
                return []
            
            infrastructure_info = v2i_pairs[vehicle_sequence][vehicle_frame]
            infrastructure_frame = infrastructure_info['infrastructure_frame']
            
            # Load infrastructure annotation file
            infra_annotation_path = f"data/v2x-seq-nuscenes/cooperative/infrastructure-side/label/virtuallidar/{infrastructure_frame}.json"
            try:
                with open(infra_annotation_path, 'r') as f:
                    infra_annotations = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load infrastructure annotations from {infra_annotation_path}: {e}")
                return []
            
            # Load infrastructure calibration for coordinate transformation
            calib_path = f"data/v2x-seq-nuscenes/cooperative/infrastructure-side/calib/virtuallidar_to_world/{infrastructure_frame}.json"
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load infrastructure calibration from {calib_path}: {e}")
                return []
            
            infra_to_world = {
                'translation': calib_data['translation'],
                'rotation': calib_data['rotation']
            }
            
            # Get ego pose information for world to ego transformation
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            ego_to_world = {
                'translation': ego_pose['translation'],
                'rotation': ego_pose['rotation']
            }
            
            # Object type mapping from infrastructure to standard category names
            category_map = {
                'Car': 'car',
                'Truck': 'truck',
                'Bus': 'bus',
                'Cyclist': 'bicycle',
                'Motorcyclist': 'motorcycle',
                'Pedestrian': 'pedestrian',
                'Van': 'car',
                'Trailer': 'truck',
                'Emergency_vehicle': 'car',
                'Construction_vehicle': 'truck'
            }
            
            # Parse infrastructure annotations and transform to ego coordinates
            boxes = []
            for i, ann in enumerate(infra_annotations):
                try:
                    # Validate required fields exist
                    required_fields = ['3d_location', '3d_dimensions', 'rotation', 'type']
                    if not all(field in ann for field in required_fields):
                        print(f"Warning: Missing required fields in annotation {i}: {list(ann.keys())}")
                        continue
                    
                    # Validate 3d_location structure
                    if not all(coord in ann['3d_location'] for coord in ['x', 'y', 'z']):
                        print(f"Warning: Invalid 3d_location structure in annotation {i}")
                        continue
                    
                    # Validate 3d_dimensions structure
                    if not all(dim in ann['3d_dimensions'] for dim in ['w', 'l', 'h']):
                        print(f"Warning: Invalid 3d_dimensions structure in annotation {i}")
                        continue
                    
                    # Extract annotation information with error handling
                    obj_type = str(ann['type'])
                    # Map object type to standard category names
                    category = category_map.get(obj_type, obj_type.lower())
                    position_infra = [
                        float(ann['3d_location']['x']), 
                        float(ann['3d_location']['y']), 
                        float(ann['3d_location']['z'])
                    ]
                    size = [
                        float(ann['3d_dimensions']['w']), 
                        float(ann['3d_dimensions']['l']), 
                        float(ann['3d_dimensions']['h'])
                    ]  # [width, length, height]
                    rotation_yaw = float(ann['rotation'])  # rotation around z-axis in radians
                    
                    # Step 1: Transform from infrastructure to world coordinates
                    box_in_infra = {
                        'translation': position_infra,
                        'rotation': [rotation_yaw, 0, 0, 0]
                    }
                    box_in_world = self.infra_to_world_box(box_in_infra, infra_to_world)
                    
                    # Step 2: Transform from world to ego coordinates  
                    world_box = {
                        'translation': box_in_world['translation'],
                        'rotation': box_in_world['rotation']
                    }
                    ego_box = self.world_to_ego_box(world_box, ego_to_world)
                    
                    # Create box dictionary in ego coordinates
                    box = {
                        'position': ego_box['translation'],                                             # [x, y, z] in ego coordinates
                        'size': size,                                                                   # [width, length, height]
                        'rotation': ego_box['rotation'],                                                # quaternion [w, x, y, z] in ego frame
                        'category': category,
                        'token': ann.get('token', f'infra_{infrastructure_frame}_{i}'),                # Infrastructure annotation token
                        'track_id': ann.get('track_id', None),                                         # Track ID for temporal consistency
                        'source': 'infrastructure',                                                    # Mark as infrastructure source
                        'infrastructure_frame': infrastructure_frame,                                  # Reference to infrastructure frame
                        'original_type': obj_type                                                      # Keep original type for reference
                    }
                    boxes.append(box)
                    
                except (ValueError, TypeError, KeyError) as e:
                    print(f"Error processing infrastructure annotation {i}: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error processing infrastructure annotation {i}: {e}")
                    continue
            
            return boxes
            
        except Exception as e:
            print(f"Error loading infrastructure 3D boxes in ego coordinates: {e}")
            return []
    
    def get_world_boxes_from_infrastructure(self, sample_token):
        """
        Get 3D bounding boxes for a sample in infrastructure coordinates and convert to world coordinates
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            boxes: List of box dictionaries with position, size, rotation, and category in world frame
        """
        if self.nusc is None:
            print("Warning: NuScenes instance not provided, cannot load infrastructure boxes")
            return []
        
        try:
            # Load v2i_pair.json for infrastructure frame mapping
            v2i_path = "data/v2x-seq-nuscenes/cooperative/v2i_pair.json"
            try:
                with open(v2i_path, 'r') as f:
                    v2i_pairs = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load v2i_pair.json: {e}")
                return []
            
            # Get vehicle_sequence from scene
            sample = self.nusc.get('sample', sample_token)
            scene = self.nusc.get('scene', sample['scene_token'])
            vehicle_sequence = scene['token']
            vehicle_frame = sample_token
            
            # Check if this vehicle frame has a paired infrastructure frame
            if not (vehicle_sequence in v2i_pairs and 
                    vehicle_frame in v2i_pairs[vehicle_sequence]):
                print(f"No paired infrastructure frame found for vehicle frame {vehicle_frame}")
                return []
            
            infrastructure_info = v2i_pairs[vehicle_sequence][vehicle_frame]
            infrastructure_frame = infrastructure_info['infrastructure_frame']
            
            print(f"Found paired infrastructure frame: {infrastructure_frame}")
            
            # Load infrastructure annotation file
            infra_annotation_path = f"data/v2x-seq-nuscenes/cooperative/infrastructure-side/label/virtuallidar/{infrastructure_frame}.json"
            try:
                with open(infra_annotation_path, 'r') as f:
                    infra_annotations = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load infrastructure annotations from {infra_annotation_path}: {e}")
                return []
            
            # Load infrastructure calibration for coordinate transformation
            calib_path = f"data/v2x-seq-nuscenes/cooperative/infrastructure-side/calib/virtuallidar_to_world/{infrastructure_frame}.json"
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load infrastructure calibration from {calib_path}: {e}")
                return []
            infra_to_world = {
                'translation': calib_data['translation'],
                'rotation': calib_data['rotation']
            }
            # Object type mapping from infrastructure to standard category names
            category_map = {
                'Car': 'car',
                'Truck': 'truck',
                'Bus': 'bus',
                'Cyclist': 'bicycle',
                'Motorcyclist': 'motorcycle',
                'Pedestrian': 'pedestrian',
                'Van': 'car',
                'Trailer': 'truck',
                'Emergency_vehicle': 'car',
                'Construction_vehicle': 'truck'
            }
            
            # Parse infrastructure annotations and transform to world coordinates
            boxes = []
            for i, ann in enumerate(infra_annotations):
                try:
                    # Validate required fields exist
                    required_fields = ['3d_location', '3d_dimensions', 'rotation', 'type']
                    if not all(field in ann for field in required_fields):
                        print(f"Warning: Missing required fields in annotation {i}: {list(ann.keys())}")
                        continue
                    
                    # Validate 3d_location structure
                    if not all(coord in ann['3d_location'] for coord in ['x', 'y', 'z']):
                        print(f"Warning: Invalid 3d_location structure in annotation {i}")
                        continue
                    
                    # Validate 3d_dimensions structure
                    if not all(dim in ann['3d_dimensions'] for dim in ['w', 'l', 'h']):
                        print(f"Warning: Invalid 3d_dimensions structure in annotation {i}")
                        continue
                    
                    # Extract annotation information with error handling
                    obj_type = str(ann['type'])
                    # Map object type to standard category names
                    category = category_map.get(obj_type, obj_type.lower())
                    position_infra = [
                        float(ann['3d_location']['x']), 
                        float(ann['3d_location']['y']), 
                        float(ann['3d_location']['z'])
                    ]
                    size = [
                        float(ann['3d_dimensions']['w']), 
                        float(ann['3d_dimensions']['l']), 
                        float(ann['3d_dimensions']['h'])
                    ]  # [width, length, height]
                    rotation_yaw = float(ann['rotation'])  # rotation around z-axis in radians
                    box_in_infra = {
                        'translation': position_infra,
                        'rotation': [rotation_yaw, 0, 0, 0]
                    }
                    box_in_world = self.infra_to_world_box(box_in_infra, infra_to_world)                    
                    # Create box dictionary in world coordinates
                    box = {
                        'position': box_in_world['translation'],                                        # [x, y, z] in world coordinates
                        'size': size,                                                                   # [width, length, height]
                        'rotation': box_in_world['rotation'],                                           # quaternion [w, x, y, z] in world frame
                        'category': category,
                        'token': ann.get('token', f'infra_{infrastructure_frame}_{i}'),                # Infrastructure annotation token
                        'track_id': ann.get('track_id', None),                                         # Track ID for temporal consistency
                        'source': 'infrastructure',                                                    # Mark as infrastructure source
                        'infrastructure_frame': infrastructure_frame,                                  # Reference to infrastructure frame
                        'original_type': obj_type                                                      # Keep original type for reference
                    }
                    boxes.append(box)
                    
                except (ValueError, TypeError, KeyError) as e:
                    print(f"Error processing infrastructure annotation {i}: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error processing infrastructure annotation {i}: {e}")
                    continue
            
            return boxes
            
        except Exception as e:
            print(f"Error loading infrastructure 3D boxes: {e}")
            return []
    

    def get_ego_boxes(self, sample_token):
        """
        Get 3D bounding boxes for a sample and convert from world coordinates to ego coordinates
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            boxes: List of box dictionaries with position, size, rotation, and category in ego frame
        """
        if self.nusc is None:
            print("Warning: NuScenes instance not provided, cannot load boxes")
            return []
            
        try:
            sample = self.nusc.get('sample', sample_token)
            boxes = []
            
            # Get ego pose information
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            # Get transformation from world to ego
            ego_to_world = {
                'translation': ego_pose['translation'],
                'rotation': ego_pose['rotation']
            }
            
            # Get all annotations for this sample (already in world coordinates)
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                
                # Convert world to ego coordinates
                ego_box = self.world_to_ego_box(ann, ego_to_world)
                
                # Get category - extract the general category type
                category = ann['category_name']
                if '.' in category:
                    category = category.split('.')[1]  # e.g., 'vehicle.car' -> 'car'
                
                # Create box dictionary
                box = {
                    'position': ego_box['translation'],      # [x, y, z] in ego coordinates
                    'size': ann['size'],                     # [width, length, height]
                    'rotation': ego_box['rotation'],         # quaternion [w, x, y, z] in ego frame
                    'category': category,
                    'token': ann_token,                      # Add annotation token for reference
                    'instance_token': ann.get('instance_token', None),  # Instance tracking token
                    'visibility_token': ann.get('visibility_token', None)  # Visibility information
                }
                boxes.append(box)
            
            return boxes
        except Exception as e:
            print(f"Error loading 3D boxes in ego coordinates: {e}")
            return []
    
    def infra_to_world_box(self, box_in_infra, infra_to_world):
        """
        Transform a box from infrastructure coordinate to world coordinate
        
        Args:
            box_in_infra: Box in infrastructure coordinate (annotation dictionary)
            infra_to_world: Infrastructure to world transformation

        Returns:
            Box in world coordinate
        """
        # Create transformation matrix from infrastructure to world
        infra_to_world_mat = self.transform_matrix(infra_to_world['translation'], infra_to_world['rotation'])
        # Transform box position from infrastructure to world
        infra_position = np.array(box_in_infra['translation']).reshape(3, 1)
        infra_position_hom = np.vstack([infra_position, np.ones((1, 1))])
        world_position_hom = infra_to_world_mat @ infra_position_hom
        world_position = world_position_hom[:3].flatten().tolist()
        # Transform box rotation from infrastructure to world
        infra_rotation = Quaternion(box_in_infra['rotation'])
        infra_rotation_mat = infra_rotation.rotation_matrix
        infra_rotation_mat_hom = np.eye(4)
        infra_rotation_mat_hom[:3, :3] = infra_rotation_mat
        # Only rotate, don't translate for rotation transformation
        rot_infra_to_world_mat = np.copy(infra_to_world_mat)
        rot_infra_to_world_mat[:3, 3] = 0
        world_rotation_mat_hom = rot_infra_to_world_mat @ infra_rotation_mat_hom
        world_rotation_mat = world_rotation_mat_hom[:3, :3]
        
        # Ensure the rotation matrix is orthogonal using SVD
        U, _, Vt = np.linalg.svd(world_rotation_mat)
        world_rotation_mat = U @ Vt
        
        world_rotation = Quaternion(matrix=world_rotation_mat)
        
        return {
            'translation': world_position,
            'rotation': [world_rotation.w, world_rotation.x, world_rotation.y, world_rotation.z]
        }
    
    def world_to_ego_box(self, box_in_world, ego_to_world):
        """
        Transform a box from world coordinate to ego coordinate
        
        Args:
            box_in_world: Box in world coordinate (annotation dictionary)
            ego_to_world: Ego vehicle to world transformation
            
        Returns:
            Box in ego coordinate
        """
        # Create transformation matrix from ego to world
        ego_to_world_mat = self.transform_matrix(ego_to_world['translation'], ego_to_world['rotation'])
        
        # Compute world to ego matrix
        world_to_ego_mat = np.linalg.inv(ego_to_world_mat)
        
        # Transform box position from world to ego
        world_position = np.array(box_in_world['translation']).reshape(3, 1)
        world_position_hom = np.vstack([world_position, np.ones((1, 1))])
        ego_position_hom = world_to_ego_mat @ world_position_hom
        ego_position = ego_position_hom[:3].flatten().tolist()
        
        # Transform box rotation from world to ego
        world_rotation = Quaternion(box_in_world['rotation'])
        world_rotation_mat = world_rotation.rotation_matrix
        world_rotation_mat_hom = np.eye(4)
        world_rotation_mat_hom[:3, :3] = world_rotation_mat
        
        # Only rotate, don't translate for rotation transformation
        rot_world_to_ego_mat = np.copy(world_to_ego_mat)
        rot_world_to_ego_mat[:3, 3] = 0
        
        ego_rotation_mat_hom = rot_world_to_ego_mat @ world_rotation_mat_hom
        ego_rotation_mat = ego_rotation_mat_hom[:3, :3]
        ego_rotation = Quaternion(matrix=ego_rotation_mat)
        
        return {
            'translation': ego_position,
            'rotation': [ego_rotation.w, ego_rotation.x, ego_rotation.y, ego_rotation.z]
        }
    
    def transform_matrix(self, translation, rotation):
        """
        Create a 4x4 transformation matrix from translation and rotation
        
        Args:
            translation: Translation vector in one of the following formats:
                       - [x, y, z] (list or 1D array)
                       - [[x], [y], [z]] (list of lists or 2D array)
            rotation: Rotation in one of the following formats:
                     - 3x3 rotation matrix (numpy array)
                     - Standard quaternion [w, x, y, z] (pyquaternion format)
            
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        
        # Handle translation - support both [x, y, z] and [[x], [y], [z]] formats
        if isinstance(translation, (list, tuple)):
            translation = np.array(translation, dtype=float)
        elif isinstance(translation, np.ndarray):
            translation = translation.astype(float)
        else:
            raise ValueError(f"Invalid translation type: {type(translation)}")
        
        # Handle different translation shapes
        if translation.ndim == 1:
            # Format: [x, y, z]
            if len(translation) != 3:
                raise ValueError(f"1D translation must have 3 elements, got {len(translation)}")
            translation_vector = translation
        elif translation.ndim == 2:
            # Format: [[x], [y], [z]] or [[x, y, z]]
            if translation.shape == (3, 1):
                # [[x], [y], [z]] - reshape to 1D
                translation_vector = translation.flatten()
            elif translation.shape == (1, 3):
                # [[x, y, z]] - reshape to 1D
                translation_vector = translation.flatten()
            else:
                raise ValueError(f"2D translation must be shape (3,1) or (1,3), got {translation.shape}")
        else:
            raise ValueError(f"Translation must be 1D or 2D array, got {translation.ndim}D")
        
        transform[:3, 3] = translation_vector
        
        # Handle rotation based on input type and format
        rotation = np.array(rotation)
        # Case 1: 3x3 rotation matrix
        if rotation.shape == (3, 3):
            transform[:3, :3] = rotation
        # Case 2: Standard quaternion [w, x, y, z]
        elif len(rotation) == 4:
            q = Quaternion(rotation)
            transform[:3, :3] = q.rotation_matrix
        else:
            raise ValueError(f"Unsupported rotation format. Expected 3x3 matrix or 4-element quaternion [w,x,y,z], got shape {rotation.shape}")
        
        return transform

    def to_descriptive_box(self, geometry_boxes):
        """
        Convert geometry boxes in ego coordinate system to descriptive format
        
        Args:
            geometry_boxes: List of box dictionaries with position, size, rotation, and category in ego frame
                           Each box contains: {'position': [x, y, z], 'size': [w, l, h], 'rotation': quaternion, 'category': str}
            
        Returns:
            descriptive_boxes: List of descriptive box dictionaries with:
                             - relative_direction: Direction relative to ego (e.g., "front-left", "rear-right")
                             - distance: Distance from ego vehicle (meters)
                             - target_orientation: Direction the target is facing relative to ego
                             - category: Object category
                             - detailed_info: Additional detailed information
        """
        descriptive_boxes = []
        
        for box in geometry_boxes:
            try:
                # Extract box information
                position = np.array(box['position'])  # [x, y, z] in ego coordinates
                size = box['size']  # [width, length, height]
                rotation = box['rotation']  # quaternion [w, x, y, z]
                category = box['category']
                
                # Calculate distance from ego (at origin)
                distance = np.sqrt(position[0]**2 + position[1]**2)
                
                # Calculate relative direction from ego to target
                relative_direction = self._get_relative_direction(position[0], position[1])
                
                # Calculate target orientation relative to ego
                target_orientation = self._get_target_orientation(rotation)
                
                # Calculate relative heading (angle from ego to target)
                relative_angle = np.arctan2(position[1], position[0])  # y, x for correct angle
                relative_angle_deg = np.degrees(relative_angle)
                
                # Normalize angle to [-180, 180]
                if relative_angle_deg > 180:
                    relative_angle_deg -= 360
                elif relative_angle_deg < -180:
                    relative_angle_deg += 360
                
                # Create descriptive box
                descriptive_box = {
                    'category': category,
                    'relative_direction': relative_direction,
                    'distance': round(distance, 2),
                    'target_orientation': target_orientation,
                    'detailed_info': {
                        'relative_angle_deg': round(relative_angle_deg, 1),
                        'position_ego': [round(p, 2) for p in position.tolist()],
                        'size': [round(s, 2) for s in size],
                        'distance_breakdown': {
                            'longitudinal': round(abs(position[0]), 2),  # front/rear distance
                            'lateral': round(abs(position[1]), 2),       # left/right distance
                            'vertical': round(abs(position[2]), 2)       # up/down distance
                        }
                    }
                }
                
                descriptive_boxes.append(descriptive_box)
                
            except Exception as e:
                print(f"Error processing box: {e}")
                continue
        
        return descriptive_boxes
    
    def _get_relative_direction(self, x, y):
        """
        Get relative direction description based on position in ego coordinates
        
        Args:
            x: Forward/backward position (positive = forward)
            y: Left/right position (positive = left)
            
        Returns:
            direction: String description of relative direction
        """
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(y, x))
        
        # Normalize to [0, 360)
        if angle < 0:
            angle += 360
        
        # Define direction sectors (8 directions)
        if angle >= 337.5 or angle < 22.5:
            direction = "front"
        elif 22.5 <= angle < 67.5:
            direction = "front-left"
        elif 67.5 <= angle < 112.5:
            direction = "left"
        elif 112.5 <= angle < 157.5:
            direction = "rear-left"
        elif 157.5 <= angle < 202.5:
            direction = "rear"
        elif 202.5 <= angle < 247.5:
            direction = "rear-right"
        elif 247.5 <= angle < 292.5:
            direction = "right"
        elif 292.5 <= angle < 337.5:
            direction = "front-right"
        else:
            direction = "unknown"
        
        # Add distance-based refinement
        distance = np.sqrt(x**2 + y**2)
        if distance < 5:
            direction = "very-close-" + direction
        elif distance < 15:
            direction = "close-" + direction
        elif distance > 50:
            direction = "far-" + direction
        
        return direction
    
    def _get_target_orientation(self, quaternion):
        """
        Get target orientation description relative to ego vehicle
        
        Args:
            quaternion: Rotation quaternion [w, x, y, z]
            
        Returns:
            orientation: String description of target's facing direction relative to ego
        """
        try:
            # Convert quaternion to rotation matrix
            q = Quaternion(quaternion)
            rotation_matrix = q.rotation_matrix
            
            # Extract forward vector (x-axis) of the target in ego coordinates
            target_forward = rotation_matrix[:, 0]  # First column is x-axis (forward direction)
            
            # Calculate the yaw angle of target's forward direction
            target_yaw = np.arctan2(target_forward[1], target_forward[0])
            target_yaw_deg = np.degrees(target_yaw)
            
            # Normalize to [-180, 180]
            if target_yaw_deg > 180:
                target_yaw_deg -= 360
            elif target_yaw_deg < -180:
                target_yaw_deg += 360
            
            # Classify orientation
            if -22.5 <= target_yaw_deg < 22.5:
                orientation = "facing-same-direction"  # Same direction as ego
            elif 22.5 <= target_yaw_deg < 67.5:
                orientation = "facing-left"
            elif 67.5 <= target_yaw_deg < 112.5:
                orientation = "facing-strong-left"
            elif 112.5 <= target_yaw_deg < 157.5:
                orientation = "facing-rear-left"
            elif 157.5 <= target_yaw_deg or target_yaw_deg < -157.5:
                orientation = "facing-opposite"  # Opposite to ego
            elif -157.5 <= target_yaw_deg < -112.5:
                orientation = "facing-rear-right"
            elif -112.5 <= target_yaw_deg < -67.5:
                orientation = "facing-strong-right"
            elif -67.5 <= target_yaw_deg < -22.5:
                orientation = "facing-right"
            else:
                orientation = "unknown-orientation"
            
            return f"{orientation} ({target_yaw_deg:.1f}°)"
            
        except Exception as e:
            print(f"Error calculating target orientation: {e}")
            return "unknown-orientation"

    def _add_descriptive_labels_with_leaders(self, ax, descriptive_boxes, ego_pos, ego_heading,
                                           view_range, font_size=9, leader_alpha=0.8, 
                                           background_alpha=0.9):
        """
        Add descriptive text labels with leader lines to avoid overlap, including target orientation visualization
        
        Args:
            ax: Matplotlib axis object
            descriptive_boxes: List of descriptive box dictionaries
            ego_pos: Current ego position (x, y)
            ego_heading: Current ego heading in radians
            view_range: [x_min, y_min, x_max, y_max] view boundaries
            font_size: Font size for labels
            leader_alpha: Transparency for leader lines
            background_alpha: Transparency for label backgrounds
        """
        if not descriptive_boxes:
            return
        
        # Transform ego coordinates to world coordinates for each object
        ego_x, ego_y = ego_pos[0], ego_pos[1]
        cos_heading, sin_heading = np.cos(ego_heading), np.sin(ego_heading)
        
        # Prepare object data for labeling
        label_data = []
        for i, desc_box in enumerate(descriptive_boxes):
            # Get object position in ego coordinates
            ego_position = desc_box['detailed_info']['position_ego']
            obj_x_ego, obj_y_ego = ego_position[0], ego_position[1]
            
            # Transform from ego to world coordinates
            obj_x_world = ego_x + obj_x_ego * cos_heading - obj_y_ego * sin_heading
            obj_y_world = ego_y + obj_x_ego * sin_heading + obj_y_ego * cos_heading
            
            # Extract information from descriptive box
            category = desc_box['category']
            distance = desc_box['distance']
            relative_dir = desc_box['relative_direction']
            target_orientation = desc_box['target_orientation']
            
            # Extract angle from target_orientation string (e.g., "facing-same-direction (15.3°)")
            orientation_angle = None
            orientation_desc = target_orientation.replace('-', ' ')
            angle_match = re.search(r'\(([-\d.]+)°\)', target_orientation)
            if angle_match:
                try:
                    orientation_angle = float(angle_match.group(1))
                    orientation_desc = target_orientation.split(' (')[0].replace('-', ' ')
                except ValueError:
                    orientation_angle = None
            
            # Create enhanced label text with target orientation
            label_text = (f"{category}\n{distance:.1f}m\n"
                         f"{relative_dir.replace('-', ' ')}\n"
                         f"{orientation_desc}")
            
            label_data.append({
                'box_index': i,
                'world_pos': (obj_x_world, obj_y_world),
                'ego_pos': (obj_x_ego, obj_y_ego),
                'label_text': label_text,
                'category': category,
                'distance': distance,
                'relative_direction': relative_dir,
                'target_orientation': target_orientation,
                'orientation_angle': orientation_angle,
                'orientation_desc': orientation_desc
            })
        
        # Sort by distance (label closer objects first)
        label_data.sort(key=lambda x: x['distance'])
        
        # Calculate label positions to avoid overlap
        label_positions = self._calculate_non_overlapping_label_positions(
            label_data, view_range, font_size
        )
        
        # Draw labels, leader lines, and orientation arrows
        for data, label_pos in zip(label_data, label_positions):
            box_pos = data['world_pos']
            label_text = data['label_text']
            category = data['category']
            orientation_angle = data['orientation_angle']
            relative_direction = data['relative_direction']
            
            # Get color based on direction instead of category
            direction_key = relative_direction.replace('-', ' ')
            color = self.direction_colors.get(direction_key, self.direction_colors['default'])
            
            # Draw leader line from object to label
            if label_pos != box_pos:  # Only draw line if label is moved
                ax.plot([box_pos[0], label_pos[0]], [box_pos[1], label_pos[1]], 
                       color=color, linewidth=1.5, alpha=leader_alpha, 
                       linestyle='--', zorder=11)
                
                # Add a small circle at the object position
                ax.plot(box_pos[0], box_pos[1], 'o', color=color, 
                       markersize=4, alpha=0.8, zorder=12)
            
            # Draw target orientation arrow if angle is available
            if orientation_angle is not None:
                self._draw_target_orientation_arrow(
                    ax, box_pos, ego_heading, orientation_angle, color, leader_alpha
                )
            
            # Draw label text with direction-based background color
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor=color, 
                            alpha=background_alpha, edgecolor='white', linewidth=1)
            
            ax.text(label_pos[0], label_pos[1], label_text, 
                   fontsize=font_size, ha='center', va='center',
                   bbox=bbox_props, zorder=13, color='white', weight='bold')
    
    def _draw_target_orientation_arrow(self, ax, object_pos, ego_heading, target_angle_deg, 
                                     color, alpha=0.8):
        """
        Draw an arrow showing the target object's orientation
        
        Args:
            ax: Matplotlib axis object
            object_pos: Object position (x, y) in world coordinates
            ego_heading: Current ego heading in radians
            target_angle_deg: Target orientation angle in degrees relative to ego
            color: Color for the arrow
            alpha: Transparency for the arrow
        """
        obj_x, obj_y = object_pos
        
        # Convert target angle from degrees to radians
        target_angle_rad = np.radians(target_angle_deg)
        
        # Calculate target's absolute heading (ego_heading + relative_target_angle)
        target_heading = ego_heading + target_angle_rad
        
        # Arrow parameters
        arrow_length = 8  # meters
        arrow_width = 2   # meters
        
        # Calculate arrow end position
        dx = arrow_length * np.cos(target_heading)
        dy = arrow_length * np.sin(target_heading)
        
        # Draw orientation arrow (without numerical angle display)
        ax.arrow(obj_x, obj_y, dx, dy, 
                head_width=arrow_width, head_length=arrow_length*0.2, 
                fc=color, ec='white', linewidth=1.5, 
                alpha=alpha, zorder=12, 
                length_includes_head=True)
    
    def _calculate_non_overlapping_label_positions(self, label_data, view_range, font_size):
        """
        Calculate label positions that avoid overlap using a simple placement strategy
        
        Args:
            label_data: List of label data dictionaries
            view_range: [x_min, y_min, x_max, y_max] view boundaries
            font_size: Font size for estimating text dimensions
            
        Returns:
            label_positions: List of (x, y) positions for labels
        """
        if not label_data:
            return []
        
        x_min, y_min, x_max, y_max = view_range
        
        # Estimate label dimensions based on font size
        # Approximate character width and height in data coordinates
        char_width = (x_max - x_min) * 0.008  # Rough estimate
        char_height = (y_max - y_min) * 0.015  # Rough estimate
        
        # Calculate text bounds for each label (now with 4 lines: category, distance, direction, orientation)
        text_bounds = []
        for data in label_data:
            lines = data['label_text'].split('\n')
            max_line_length = max(len(line) for line in lines)
            text_width = max_line_length * char_width
            text_height = len(lines) * char_height  # Now typically 4 lines
            text_bounds.append((text_width, text_height))
        
        # Initialize positions at object locations
        positions = [data['world_pos'] for data in label_data]
        
        # Define placement offsets around objects (8 directions)
        offset_directions = [
            (1, 1),    # NE
            (0, 1),    # N
            (-1, 1),   # NW
            (-1, 0),   # W
            (-1, -1),  # SW
            (0, -1),   # S
            (1, -1),   # SE
            (1, 0),    # E
        ]
        
        # Adjust positions to avoid overlaps
        for i, (data, bounds) in enumerate(zip(label_data, text_bounds)):
            obj_pos = data['world_pos']
            text_width, text_height = bounds
            
            # Try different offset positions
            best_pos = obj_pos
            min_overlap_score = float('inf')
            
            # Base offset distance (minimum distance from object) - increased for larger labels
            base_offset = max(text_width, text_height) * 0.8  # Increased from 0.7 to 0.8
            
            for offset_mult in [1.0, 1.5, 2.0, 2.5]:  # Added 2.5 for more distance options
                for dx, dy in offset_directions:
                    # Calculate candidate position
                    offset_distance = base_offset * offset_mult
                    candidate_x = obj_pos[0] + dx * offset_distance
                    candidate_y = obj_pos[1] + dy * offset_distance
                    
                    # Check if position is within view bounds (with some margin)
                    margin = max(text_width, text_height) * 0.5
                    if (candidate_x - text_width/2 < x_min + margin or 
                        candidate_x + text_width/2 > x_max - margin or
                        candidate_y - text_height/2 < y_min + margin or 
                        candidate_y + text_height/2 > y_max - margin):
                        continue
                    
                    # Calculate overlap score with existing labels
                    overlap_score = 0
                    for j in range(i):  # Only check against already placed labels
                        other_pos = positions[j]
                        other_width, other_height = text_bounds[j]
                        
                        # Check if rectangles overlap
                        x_overlap = max(0, min(candidate_x + text_width/2, 
                                              other_pos[0] + other_width/2) - 
                                          max(candidate_x - text_width/2, 
                                              other_pos[0] - other_width/2))
                        y_overlap = max(0, min(candidate_y + text_height/2, 
                                              other_pos[1] + other_height/2) - 
                                          max(candidate_y - text_height/2, 
                                              other_pos[1] - other_height/2))
                        
                        overlap_area = x_overlap * y_overlap
                        overlap_score += overlap_area
                    
                    # Add penalty for distance from object (prefer closer positions)
                    distance_penalty = np.sqrt((candidate_x - obj_pos[0])**2 + 
                                             (candidate_y - obj_pos[1])**2) * 0.1
                    total_score = overlap_score + distance_penalty
                    
                    if total_score < min_overlap_score:
                        min_overlap_score = total_score
                        best_pos = (candidate_x, candidate_y)
            
            positions[i] = best_pos
        
        return positions

def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: End-to-End Autonomous Driving")
    parser.add_argument("--model", type=str, default="qwen2.5-3b", 
                        help="Model to use for reasoning (default: gpt-4o, "
                        "options: gpt-4o, gpt-4.1, claude-3.7, claude-3.5, "
                        "gemini-2.5, gemini-2.0, qwen2.5-7b, qwen2.5-72b, "
                        "deepseek-vl2-16b, deepseek-vl2-28b, llama-3.2-11b, "
                        "llama-3.2-90b)")
    parser.add_argument("--model_weights", type=str, default="/home/yanglei/QWen/Qwen2.5-VL-3B-Instruct/",
                        help="Path to local model weights (for local models only)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file (default: config.yaml)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Optional: Specific scene name to process.")
    parser.add_argument("--all_scenes", action="store_true",
                        help="Process all scenes instead of random sampling")
    parser.add_argument("--continue_dir", type=str, default=None,
                        help="Path to the directory with previously processed scene JSON files to resume processing")
    
    # Trajectory visualization arguments
    parser.add_argument("--visualize_trajectory", action="store_true",
                        help="Enable trajectory visualization for each scene")
    parser.add_argument("--viz_output_dir", type=str, default="trajectory_visualizations",
                        help="Directory to save trajectory visualizations")
    parser.add_argument("--show_plot", action="store_true",
                        help="Display trajectory plots (default: False, only save)")
    parser.add_argument("--show_orientation_arrows", action="store_true", default=False,
                        help="Show orientation arrows on trajectory")
    parser.add_argument("--show_ego_axes", action="store_true", default=True,
                        help="Show ego coordinate axes on trajectory")
    parser.add_argument("--ego_axes_interval", type=int, default=10,
                        help="Interval for showing ego axes (every N trajectory points)")
    
    # Point cloud visualization arguments
    parser.add_argument("--show_point_cloud", action="store_true",
                        help="Show LiDAR point cloud background in trajectory visualization")
    parser.add_argument("--point_cloud_alpha", type=float, default=0.6,
                        help="Transparency of point cloud points (0.0-1.0)")
    parser.add_argument("--point_cloud_size", type=float, default=1.0,
                        help="Size of point cloud points")
    
    # 3D bounding box visualization arguments
    parser.add_argument("--show_3d_boxes", action="store_true",
                        help="Show 3D bounding boxes from first frame in trajectory visualization")
    parser.add_argument("--box_alpha", type=float, default=0.7,
                        help="Transparency of 3D bounding boxes (0.0-1.0)")
    parser.add_argument("--arrow_alpha", type=float, default=0.8,
                        help="Transparency of direction arrows in 3D boxes (0.0-1.0)")
    
    # First frame trajectory overlay arguments
    parser.add_argument("--overlay_trajectory_on_first_frame", action="store_true",
                        help="Overlay complete sequence trajectory on first frame image")
    parser.add_argument("--overlay_output_dir", type=str, default="trajectory_overlays",
                        help="Directory to save trajectory overlay images")
    parser.add_argument("--overlay_color", type=str, default="255,0,0",
                        help="RGB color for trajectory overlay (format: R,G,B)")
    parser.add_argument("--overlay_line_width", type=int, default=3,
                        help="Width of trajectory line in overlay")
    parser.add_argument("--overlay_point_size", type=int, default=8,
                        help="Size of trajectory points in overlay")
    parser.add_argument("--overlay_show_arrows", action="store_true", default=True,
                        help="Show orientation arrows in trajectory overlay")
    parser.add_argument("--overlay_arrow_interval", type=int, default=5,
                        help="Interval for showing arrows in overlay (every N points)")
    parser.add_argument("--overlay_alpha", type=float, default=0.8,
                        help="Transparency of trajectory overlay (0.0-1.0)")
    
    # Frame-by-frame visualization arguments
    parser.add_argument("--visualize_current_frame", action="store_true",
                        help="Enable frame-by-frame visualization showing current position in complete trajectory")
    parser.add_argument("--frame_viz_output_dir", type=str, default="frame_visualizations",
                        help="Directory to save frame-by-frame visualizations")
    parser.add_argument("--frame_viz_show_plot", action="store_true", default=False,
                        help="Display frame visualization plots (default: False, only save)")
    parser.add_argument("--frame_viz_show_point_cloud", action="store_true", default=True,
                        help="Show point cloud background in frame visualization")
    parser.add_argument("--frame_viz_show_3d_boxes", action="store_true", default=True,
                        help="Show 3D bounding boxes in frame visualization")
    parser.add_argument("--frame_viz_interval", type=int, default=1,
                        help="Interval for saving frame visualizations (every N frames)")
    
    # Descriptive label arguments for frame visualization
    parser.add_argument("--show_descriptive_labels", action="store_true", default=True,
                        help="Show descriptive text labels for objects in frame visualization")
    parser.add_argument("--label_font_size", type=int, default=7,
                        help="Font size for descriptive labels")
    parser.add_argument("--leader_line_alpha", type=float, default=0.8,
                        help="Transparency for leader lines connecting labels to objects (0.0-1.0)")
    parser.add_argument("--label_background_alpha", type=float, default=0.9,
                        help="Transparency for label background boxes (0.0-1.0)")
    
    return parser.parse_args()


def run_prediction():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Configure output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Use the provided directory for continuation, or create a new one
    if args.continue_dir:
        results_dir = args.continue_dir
        print(f"Continuing from existing directory: {results_dir}")
    else:
        results_dir = f"{config['data']['results']}/{args.model}_{timestamp}/output"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created new results directory: {results_dir}")
    
    # Initialize random seed for reproducibility
    random.seed(42)
    
    # Load NuScenes parameters from config
    OBS_LEN = config["prediction"]["obs_len"]
    FUT_LEN = config["prediction"]["fut_len"]
    EXT_LEN = config["prediction"]["ext_len"]
    TTL_LEN = OBS_LEN + FUT_LEN + EXT_LEN 
    
    # Initialize model
    model_handler = ModelHandler(args.model, args.config, model_weights=args.model_weights)
    model_handler.model_instance, model_handler.processor = model_handler.initialize_model()
    
    # Initialize NuScenes dataset
    nusc = NuScenes(version=config["data"]["version"], dataroot=config["data"]["root"], verbose=True)
    # Select scenes to process
    if args.scene:
        # Find the specific scene by name
        selected_scenes = [scene for scene in nusc.scene if scene["name"] == args.scene]
        if not selected_scenes:
            print(f"Scene '{args.scene}' not found in dataset")
            return
    else:
        # Process all scenes if no specific scene is specified
        selected_scenes = nusc.scene
        print(f"Processing all {len(selected_scenes)} scenes")
    # Get list of already processed scenes if continuing
    processed_scene_names = []
    if args.continue_dir:
        for filename in os.listdir(args.continue_dir):
            if filename.endswith('.json'):
                processed_scene_names.append(filename.replace('.json', ''))
        print(f"Found {len(processed_scene_names)} previously processed scenes")
    
    # Process each selected scene
    for scene in selected_scenes:
        scene_name = scene["name"]

        # Skip already processed scenes when in continuation mode
        if args.continue_dir and scene_name in processed_scene_names:
            print(f"Skipping already processed scene: {scene_name}")
            continue

        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        description = scene["description"]
        
        print(f"\nProcessing scene '{scene_name}': {description}")
        
        # Create scene data structure
        scene_data = {
            "scene_info": {
                "name": scene_name,
                "description": description,
                "first_sample_token": first_sample_token,
                "last_sample_token": last_sample_token
            },
            "frames": [],
            "metadata": {
                "model": args.model,
                "timestamp": timestamp,
                "total_frames": 0
            }
        }
        
        # Collect scene data
        camera_params = []
        front_camera_images = []
        ego_positions = []
        ego_headings = []
        timestamps = []
        sample_tokens = []
        
        curr_sample_token = first_sample_token
        
        # Retrieve all frames in the scene
        while curr_sample_token:
            sample = nusc.get("sample", curr_sample_token)
            sample_tokens.append(curr_sample_token)
            if "v2x-seq" not in nusc.dataroot:
                cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                front_camera_images.append(
                    os.path.join(nusc.dataroot, cam_front_data["filename"])
                )
            else:
                cam_front_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                front_camera_image = os.path.join(nusc.dataroot, cam_front_data["filename"].replace("velodyne", "image").replace("bin", "jpg"))
                front_camera_images.append(front_camera_image)
            
            # Get the camera parameters
            camera_param_path = os.path.join(nusc.dataroot, "v1.0-trainval/calibrated_sensor.json")
            with open(camera_param_path, 'r') as f:
                camera_param = json.load(f)[0]
            camera_params.append(camera_param)
            
            # Get ego vehicle state
            ego_state = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_positions.append(tuple(ego_state["translation"][0:2]))
            ego_headings.append(quaternion_to_yaw(ego_state["rotation"]))
            timestamps.append(ego_state["timestamp"])
            
            # Move to next sample or exit loop if at the end
            curr_sample_token = (
                sample["next"] if curr_sample_token != last_sample_token else None
            )
        
        num_frames = len(front_camera_images)
        
        # Check if we have enough frames
        if num_frames < TTL_LEN:
            print(f"Skipping '{scene_name}', insufficient frames ({num_frames} < {TTL_LEN}).")
            continue

        # Visualize trajectory if requested
        if args.visualize_trajectory and len(ego_positions) > 1:
            print(f"Creating trajectory visualization for scene '{scene_name}'...")
            
            # Initialize trajectory visualizer
            trajectory_visualizer = EgoTrajectoryVisualizer(nusc=nusc)
            
            # Create visualization output directory
            viz_dir = args.viz_output_dir
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create visualization save path
            viz_save_path = os.path.join(viz_dir, f"{scene_name}_trajectory.png")
            
            try:
                # Create trajectory visualization
                fig, ax = trajectory_visualizer.visualize_scene_trajectory(
                    ego_positions=ego_positions,
                    ego_headings=ego_headings,
                    scene_name=scene_name,
                    save_path=viz_save_path,
                    show_plot=args.show_plot,
                    show_orientation_arrows=args.show_orientation_arrows,
                    show_ego_axes=args.show_ego_axes,
                    ego_axes_interval=args.ego_axes_interval,
                    show_point_cloud=args.show_point_cloud,
                    first_sample_token=first_sample_token,
                    point_cloud_alpha=args.point_cloud_alpha,
                    point_cloud_size=args.point_cloud_size,
                    show_3d_boxes=args.show_3d_boxes,
                    box_alpha=args.box_alpha,
                    arrow_alpha=args.arrow_alpha
                )
                
                if fig is not None:
                    print(f"Trajectory visualization saved to: {viz_save_path}")
                else:
                    print(f"Failed to create trajectory visualization for scene '{scene_name}'")        
            except Exception as e:
                print(f"Error creating trajectory visualization for scene '{scene_name}': {e}")
                continue
        
        # Overlay trajectory on first frame if requested
        if args.overlay_trajectory_on_first_frame and len(ego_positions) > 1:
            print(f"Creating trajectory overlay on first frame for scene '{scene_name}'...")
            
            # Initialize trajectory visualizer if not already done
            if not args.visualize_trajectory:
                trajectory_visualizer = EgoTrajectoryVisualizer(nusc=nusc)
            
            # Create overlay output directory
            overlay_dir = args.overlay_output_dir
            os.makedirs(overlay_dir, exist_ok=True)
            
            # Parse overlay color
            try:
                color_parts = args.overlay_color.split(',')
                if len(color_parts) == 3:
                    overlay_color = tuple(int(c.strip()) for c in color_parts)
                else:
                    print(f"Invalid color format: {args.overlay_color}, using default red")
                    overlay_color = (255, 0, 0)
            except:
                print(f"Error parsing color: {args.overlay_color}, using default red")
                overlay_color = (255, 0, 0)
            
            # Create overlay save path
            overlay_save_path = os.path.join(overlay_dir, f"{scene_name}_trajectory_overlay.png")
            
            try:
                # Get first frame image path and camera parameters
                first_frame_image = front_camera_images[0]
                first_frame_camera_params = camera_params[0]
                
                # Create trajectory overlay
                success = trajectory_visualizer.overlay_trajectory_on_first_frame(
                    first_frame_image_path=first_frame_image,
                    ego_positions=ego_positions,
                    ego_headings=ego_headings,
                    camera_params=first_frame_camera_params,
                    scene_name=scene_name,
                    save_path=overlay_save_path,
                    trajectory_color=overlay_color,
                    line_width=args.overlay_line_width,
                    point_size=args.overlay_point_size,
                    show_arrows=args.overlay_show_arrows,
                    arrow_interval=args.overlay_arrow_interval,
                    alpha=args.overlay_alpha
                )
                
                if success:
                    print(f"Trajectory overlay saved to: {overlay_save_path}")
                else:
                    print(f"Failed to create trajectory overlay for scene '{scene_name}'")
            except Exception as e:
                print(f"Error creating trajectory overlay for scene '{scene_name}': {e}")
                continue
        
        # Initialize trajectory visualizer for frame-by-frame visualization if requested
        if args.visualize_current_frame:
            if not (args.visualize_trajectory or args.overlay_trajectory_on_first_frame):
                trajectory_visualizer = EgoTrajectoryVisualizer(nusc=nusc)
            
            # Create frame visualization output directory
            frame_viz_dir = args.frame_viz_output_dir
            os.makedirs(frame_viz_dir, exist_ok=True)
            
            # Create scene-specific subdirectory
            scene_viz_dir = os.path.join(frame_viz_dir, scene_name)
            os.makedirs(scene_viz_dir, exist_ok=True)
            
            print(f"Frame-by-frame visualization enabled for scene '{scene_name}', saving to: {scene_viz_dir}")
        
        # Initialize trajectory visualizer for processing boxes (always needed)
        if 'trajectory_visualizer' not in locals():
            trajectory_visualizer = EgoTrajectoryVisualizer(nusc=nusc)
        
        # Process each frame in the scene
        for i in range(0, num_frames - TTL_LEN, 1):
            try:
                cur_index = i + OBS_LEN + 1
                frame_index = i  # The relative index in the processed subset
                
                image_path = front_camera_images[cur_index]
                print(f"Processing frame {i} from {scene_name}, image: {image_path}")
                
                # Extract image ID from filename
                match = re.search(r"(\d+)(?=\.jpg$)", image_path)
                image_id = match.group(1) if match else None
                
                sample_token = sample_tokens[cur_index]
                camera_param = camera_params[cur_index]
                
                # Get current position and heading
                cur_pos = ego_positions[cur_index]
                cur_heading = ego_headings[cur_index]

                # Get observation data (past positions and timestamps)
                obs_pos = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]
                obs_pos = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
                obs_time = timestamps[cur_index - OBS_LEN - 1 : cur_index + 1]
                
                # Calculate past speeds and curvatures
                prev_speed = compute_speed(obs_pos, obs_time)
                prev_curvatures = compute_curvature(obs_pos)
                prev_actions = list(zip(prev_speed, prev_curvatures))
                
                # Get future positions and timestamps (ground truth)
                fut_pos = ego_positions[cur_index - 1 : cur_index + FUT_LEN + 1]
                fut_pos = global_to_ego_frame(cur_pos, cur_heading, fut_pos)
                fut_time = timestamps[cur_index - 1 : cur_index + FUT_LEN + 1]
                
                # Calculate ground truth speeds and curvatures
                gt_speed = compute_speed(fut_pos, fut_time)
                gt_curvatures = compute_curvature(fut_pos)
                gt_actions = list(zip(gt_speed, gt_curvatures))
                
                # Remove extra indices used for speed and curvature calculation
                fut_pos = fut_pos[2:]
                
                # Create frame-by-frame visualization if requested
                if args.visualize_current_frame and (frame_index % args.frame_viz_interval == 0):
                    try:
                        print(f"Creating frame visualization for frame {frame_index} (cur_index {cur_index}) in scene '{scene_name}'...")
                        
                        # Create frame visualization save path
                        frame_viz_save_path = os.path.join(scene_viz_dir, f"frame_{frame_index:04d}_cur_{cur_index:04d}.png")
                        
                        # Create frame visualization
                        fig, ax = trajectory_visualizer.visualize_current_frame_in_trajectory(
                            ego_positions=ego_positions,
                            ego_headings=ego_headings,
                            scene_name=scene_name,
                            current_index=cur_index,
                            current_sample_token=sample_token,
                            save_path=frame_viz_save_path,
                            show_plot=args.frame_viz_show_plot,
                            show_point_cloud=args.frame_viz_show_point_cloud,
                            show_3d_boxes=args.frame_viz_show_3d_boxes,
                            point_cloud_alpha=args.point_cloud_alpha,
                            point_cloud_size=args.point_cloud_size,
                            box_alpha=args.box_alpha,
                            arrow_alpha=args.arrow_alpha,
                            show_descriptive_labels=args.show_descriptive_labels,
                            label_font_size=args.label_font_size,
                            leader_line_alpha=args.leader_line_alpha,
                            label_background_alpha=args.label_background_alpha
                        )
                        
                        if fig is not None:
                            print(f"Frame visualization saved to: {frame_viz_save_path}")
                        else:
                            print(f"Failed to create frame visualization for frame {frame_index}")
                    except Exception as e:
                        print(f"Error creating frame visualization for frame {frame_index} in scene '{scene_name}': {e}")
                # current_frame_boxes = trajectory_visualizer.get_ego_boxes(sample_token)
                # current_frame_boxes = trajectory_visualizer.get_ego_boxes_from_infrastructure(sample_token)
                current_frame_boxes = trajectory_visualizer.get_ego_boxes(sample_token)
                descriptive_boxes = trajectory_visualizer.to_descriptive_box(current_frame_boxes)
                # Define prompts for LLM inference
                scene_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "You must observe and analyze the movements of vehicles and pedestrians, "
                    "lane markings, traffic lights, and any relevant objects in the scene. "
                    "describe what you observe, but do not infer the ego's action. "
                    "generate your response in plain text in one paragraph without any formating. "
                )
                
                # Run scene description inference
                scene_description, scene_tokens, scene_time = model_handler.get_response(
                    prompt=scene_prompt,
                    image_path=image_path
                )
                print("Scene description:", scene_description)
                
                # Generate intent prompt based on scene description
                intent_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The scene is described as follows: "
                    f"{scene_description} "
                    "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left."
                    "A negative curvature indicates the ego is turning right. "
                    "What was the ego's previous intent? "
                    "Was it accelerating (by how much), decelerating (by how much), or maintaining speed? "
                    "Was it turning left (by how much), turning right (by how much), or following the lane? "
                    "Taking into account the ego's previous intent, how should it drive in the next 3 seconds? "
                    "Should the ego accelerate (by how much), decelerate (by how much), or maintain speed? "
                    "Should the ego turn left (by how much), turn right (by how much), or follow the lane?  "
                    "Generate your response in plain text in one paragraph without any formating. "
                )
                
                # Run driving intent inference
                driving_intent, intent_tokens, intent_time = model_handler.get_response(
                    prompt=intent_prompt,
                    image_path=image_path
                )
                print("Driving intent:", driving_intent)
                
                # Generate perception text from descriptive boxes
                perception_text = descriptive_boxes_to_text(descriptive_boxes, max_objects=8, prioritize_close=True)
                print("Perception text:", perception_text)
                
                # Generate waypoint prompt based on scene, intent, and perception
                waypoint_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The scene is described as follows: "
                    f"{scene_description} "
                    "The detected objects and their spatial relationships are: "
                    f"{perception_text} "
                    "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is "
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is "
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left. "
                    "A negative curvature indicates the ego is turning right. "
                    "The high-level driving instructions are as follows: "
                    f"{driving_intent} "
                    "Based on the scene description, detected objects, ego vehicle's motion history, and driving instructions, "
                    "predict the speed and curvature for the next 6 waypoints, with 0.5-second resolution. "
                    "Consider the positions and movements of surrounding objects when planning the trajectory. "
                    "The predicted speed and curvature changes must obey the physical constraints of the vehicle. "
                    "Predict Exactly 6 pairs of speed and curvature, in the format: "
                    "[(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)]. "
                    "ONLY return the answers in the required format, do not include punctuation or text."
                )
                
                # Run waypoint prediction inference
                pred_actions_str, waypoint_tokens, waypoint_time = model_handler.get_response(
                    prompt=waypoint_prompt,
                    image_path=image_path
                )
                print("Predicted actions:", pred_actions_str)
                
                # Prepare frame data structure
                frame_data = {
                    "frame_index": frame_index,
                    "sample_token": sample_token,
                    "image_path": image_path,
                    "timestamp": timestamps[cur_index],
                    "camera_params": {
                        "rotation": camera_param["rotation"],
                        "translation": camera_param["translation"],
                        "camera_intrinsic": camera_param["camera_intrinsic"]
                    },
                    "ego_info": {
                        "position": cur_pos,
                        "heading": cur_heading,
                        "obs_positions": obs_pos,
                        "obs_actions": prev_actions,
                        "gt_positions": fut_pos,
                        "gt_actions": gt_actions
                    },
                    "scene_objects": {
                        "geometric_boxes": current_frame_boxes,
                        "descriptive_boxes": descriptive_boxes
                    },
                    "inference": {
                        "scene_prompt": format_long_text(scene_prompt),
                        "scene_description": format_long_text(scene_description),
                        "intent_prompt": format_long_text(intent_prompt),
                        "driving_intent": format_long_text(driving_intent),
                        "perception_text": format_long_text(perception_text),
                        "waypoint_prompt": format_long_text(waypoint_prompt),
                        "pred_actions_str": pred_actions_str
                    },
                    "token_usage": {
                        "scene_prompt": scene_tokens,
                        "intent_prompt": intent_tokens,
                        "waypoint_prompt": waypoint_tokens
                    },
                    "time_usage": {
                        "scene_prompt": scene_time,
                        "intent_prompt": intent_time,
                        "waypoint_prompt": waypoint_time
                    }
                }
                
                # Try to parse predicted actions and generate trajectory
                try:
                    pred_actions = ast.literal_eval(pred_actions_str)
                    if isinstance(pred_actions, list) and len(pred_actions) > 0:
                        prediction = integrate_driving_commands(pred_actions, dt=0.5)
                        frame_data["predictions"] = {
                            "pred_actions": pred_actions,
                            "trajectory": prediction
                        }
                    else:
                        frame_data["predictions"] = {
                            "pred_actions_str": pred_actions_str
                        }
                except Exception as e:
                    frame_data["predictions"] = {
                        "pred_actions_str": pred_actions_str
                    }
                
                # Add frame data to scene
                scene_data["frames"].append(frame_data)
                
            except Exception as e:
                print(f"Error processing frame {i} in {scene_name}: {e}")
                continue
        
        # Update total frames count
        scene_data["metadata"]["total_frames"] = len(scene_data["frames"])
        
        # Save scene data
        scene_file_path = f"{results_dir}/{scene_name}.json"
        save_dict_to_json(scene_data, scene_file_path)
        print(f"Scene data saved to {scene_file_path} with {len(scene_data['frames'])} frames")


def descriptive_boxes_to_text(descriptive_boxes, max_objects=10, prioritize_close=True):
    """
    Convert descriptive boxes to natural language text suitable for LLM prompts
    
    Args:
        descriptive_boxes: List of descriptive box dictionaries from to_descriptive_box()
        max_objects: Maximum number of objects to include in the description
        prioritize_close: Whether to prioritize closer objects
        
    Returns:
        perception_text: Natural language description of the surrounding objects
    """
    if not descriptive_boxes:
        return "No objects detected in the surrounding area."
    
    # Sort objects by distance (closest first) if prioritizing close objects
    if prioritize_close:
        sorted_boxes = sorted(descriptive_boxes, key=lambda x: x['distance'])
    else:
        sorted_boxes = descriptive_boxes
    
    # Limit to max_objects
    sorted_boxes = sorted_boxes[:max_objects]
    
    # Group objects by category for better organization
    category_groups = {}
    for box in sorted_boxes:
        category = box['category']
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(box)
    
    # Generate text description
    perception_parts = []
    
    # Add summary
    total_objects = len(sorted_boxes)
    categories = list(category_groups.keys())
    if total_objects > 0:
        category_summary = ", ".join(categories)
        perception_parts.append(f"Detected {total_objects} objects: {category_summary}.")
    
    # Add detailed descriptions for each category
    for category, boxes in category_groups.items():
        category_descriptions = []
        
        for box in boxes:
            # Extract information
            relative_dir = box['relative_direction'].replace('-', ' ')
            distance = box['distance']
            target_orientation = box['target_orientation']
            
            # Get detailed position info
            detailed_info = box.get('detailed_info', {})
            position_ego = detailed_info.get('position_ego', [0, 0, 0])
            relative_angle = detailed_info.get('relative_angle_deg', 0)
            distance_breakdown = detailed_info.get('distance_breakdown', {})
            
            # Format distance description
            if distance < 5:
                distance_desc = f"very close ({distance:.1f}m)"
            elif distance < 15:
                distance_desc = f"close ({distance:.1f}m)"
            elif distance < 30:
                distance_desc = f"nearby ({distance:.1f}m)"
            else:
                distance_desc = f"distant ({distance:.1f}m)"
            
            # Format direction with more context
            direction_desc = relative_dir
            if 'front' in relative_dir:
                if abs(relative_angle) < 15:
                    direction_desc = f"directly ahead"
                elif 'left' in relative_dir:
                    direction_desc = f"ahead and to the left"
                elif 'right' in relative_dir:
                    direction_desc = f"ahead and to the right"
            elif 'rear' in relative_dir:
                if 'left' in relative_dir:
                    direction_desc = f"behind and to the left"
                elif 'right' in relative_dir:
                    direction_desc = f"behind and to the right"
                else:
                    direction_desc = f"directly behind"
            elif relative_dir == 'left':
                direction_desc = f"to the left side"
            elif relative_dir == 'right':
                direction_desc = f"to the right side"
            
            # Format orientation description
            orientation_desc = ""
            if 'facing same direction' in target_orientation:
                orientation_desc = "moving in the same direction"
            elif 'facing opposite' in target_orientation:
                orientation_desc = "moving in the opposite direction"
            elif 'facing left' in target_orientation:
                orientation_desc = "turning left"
            elif 'facing right' in target_orientation:
                orientation_desc = "turning right"
            elif 'facing' in target_orientation:
                orientation_desc = target_orientation.split(' (')[0].replace('facing ', '').replace('-', ' ')
            
            # Combine into description
            obj_desc = f"{distance_desc} {direction_desc}"
            if orientation_desc and orientation_desc != "unknown orientation":
                obj_desc += f", {orientation_desc}"
            
            category_descriptions.append(obj_desc)
        
        # Format category description
        if len(category_descriptions) == 1:
            perception_parts.append(f"A {category} is {category_descriptions[0]}.")
        else:
            # Multiple objects of same category
            if len(category_descriptions) == 2:
                perception_parts.append(f"Two {category}s: one {category_descriptions[0]}, another {category_descriptions[1]}.")
            else:
                main_desc = ", ".join(category_descriptions[:-1])
                last_desc = category_descriptions[-1]
                perception_parts.append(f"Multiple {category}s: {main_desc}, and one {last_desc}.")
    
    # Add spatial context for important objects
    critical_objects = [box for box in sorted_boxes if box['distance'] < 15 and 'front' in box['relative_direction']]
    if critical_objects:
        critical_desc = []
        for box in critical_objects[:3]:  # Limit to 3 most critical
            category = box['category']
            distance = box['distance']
            critical_desc.append(f"{category} at {distance:.1f}m ahead")
        
        if len(critical_desc) > 0:
            perception_parts.append(f"Critical objects requiring attention: {', '.join(critical_desc)}.")
    
    # Join all parts
    perception_text = " ".join(perception_parts)
    
    return perception_text


if __name__ == "__main__":
    run_prediction()