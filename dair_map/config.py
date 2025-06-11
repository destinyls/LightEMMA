"""
Configuration management for DAIR Map module
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class MapConfig:
    """Configuration class for DAIR Map operations"""
    
    # Data paths
    maps_dir: str = "data/v2x-seq-nuscenes/cooperative/maps"
    expansion_dir: str = None  # Will be set to maps_dir/expansion if None
    output_dir: str = "output"
    
    # Visualization settings
    default_figsize: Tuple[int, int] = (15, 15)
    default_dpi: int = 300
    default_lane_width: float = 3.5
    default_arrow_spacing: int = 50
    default_arrow_size: int = 10
    
    # Element colors and styles
    lane_color: str = "lightblue"
    lane_edge_color: str = "blue"
    lane_alpha: float = 0.5
    
    centerline_color: str = "red"
    centerline_width: float = 1.0
    
    crosswalk_color: str = "orange"
    crosswalk_edge_color: str = "darkorange"
    crosswalk_alpha: float = 0.7
    
    stopline_color: str = "red"
    stopline_width: float = 3.0
    
    junction_color: str = "lightgreen"
    junction_alpha: float = 0.3
    
    # Arrow settings
    arrow_color: str = "red"
    arrow_alpha: float = 0.8
    arrow_width: float = 2.0
    # Arrow filtering settings (控制箭头过滤行为)
    filter_arrows_in_junctions: bool = False  # 是否过滤交叉路口内的箭头
    arrow_junction_buffer: float = 0.0  # 交叉路口周围的缓冲区距离(米)
    
    # Default elements to visualize
    default_elements: List[str] = field(default_factory=lambda: [
        'lanes', 'centerlines', 'crosswalks', 'stoplines'
    ])
    
    # Supported map formats
    supported_formats: List[str] = field(default_factory=lambda: [
        'v2x_seq', 'argoverse', 'nuscenes'
    ])
    
    # Coordinate system settings
    coordinate_system: str = "local"  # "local" or "global"
    units: str = "meters"
    
    # Processing settings
    interpolation_points: int = 100
    bbox_margin_ratio: float = 0.05  # Margin as percentage of map size
    
    # Validation settings
    validate_on_load: bool = True
    strict_validation: bool = False
    
    # Caching settings
    enable_cache: bool = True
    cache_dir: str = ".cache/dair_map"
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set expansion directory if not provided
        if self.expansion_dir is None:
            self.expansion_dir = str(Path(self.maps_dir) / "expansion")
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for dir_path in [self.output_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MapConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def get_visualization_style(self, element_type: str) -> Dict[str, Any]:
        """Get visualization style for specific element type"""
        styles = {
            'lanes': {
                'color': self.lane_color,
                'edgecolor': self.lane_edge_color,
                'alpha': self.lane_alpha
            },
            'centerlines': {
                'color': self.centerline_color,
                'linewidth': self.centerline_width
            },
            'crosswalks': {
                'color': self.crosswalk_color,
                'edgecolor': self.crosswalk_edge_color,
                'alpha': self.crosswalk_alpha
            },
            'stoplines': {
                'color': self.stopline_color,
                'linewidth': self.stopline_width
            },
            'junctions': {
                'color': self.junction_color,
                'alpha': self.junction_alpha
            },
            'arrows': {
                'color': self.arrow_color,
                'alpha': self.arrow_alpha,
                'lw': self.arrow_width
            }
        }
        
        return styles.get(element_type, {})
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check directory paths
        if not Path(self.maps_dir).exists():
            issues.append(f"Maps directory not found: {self.maps_dir}")
        
        if not Path(self.expansion_dir).exists():
            issues.append(f"Expansion directory not found: {self.expansion_dir}")
        
        # Check numeric values
        if self.default_figsize[0] <= 0 or self.default_figsize[1] <= 0:
            issues.append("Figure size must be positive")
        
        if self.default_dpi <= 0:
            issues.append("DPI must be positive")
        
        if self.default_lane_width <= 0:
            issues.append("Lane width must be positive")
        
        # Check alpha values
        for attr in ['lane_alpha', 'crosswalk_alpha', 'junction_alpha', 'arrow_alpha']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                issues.append(f"{attr} must be between 0 and 1")
        
        # Check supported elements
        all_elements = ['lanes', 'centerlines', 'crosswalks', 'stoplines', 'junctions']
        for element in self.default_elements:
            if element not in all_elements:
                issues.append(f"Unknown element type: {element}")
        
        return issues


@dataclass
class VisualizationConfig:
    """Configuration for map visualization"""
    elements: List[str] = field(default_factory=lambda: ['lanes', 'centerlines'])
    bbox: Optional[Tuple[float, float, float, float]] = None
    figsize: Tuple[int, int] = (12, 12)
    dpi: int = 150
    save_path: Optional[str] = None
    show_plot: bool = True
    show_direction: bool = True
    arrow_spacing: int = 50
    arrow_size: int = 10
    style_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance optimization options
    fast_mode: bool = True  # Enables various performance optimizations
    max_arrows_per_lane: Optional[int] = None  # Limit arrows per lane
    skip_small_lanes: bool = True  # Skip lanes shorter than arrow_spacing
    
    # Arrow filtering options (箭头过滤选项)
    filter_arrows_in_junctions: Optional[bool] = False  # 是否过滤交叉路口内的箭头 (None表示使用基础配置)
    arrow_junction_buffer: Optional[float] = None  # 交叉路口周围的缓冲区距离(米) (None表示使用基础配置)
    
    def __post_init__(self):
        """Apply fast mode optimizations if enabled"""
        if self.fast_mode:
            self.show_direction = False  # Disable arrows for speed
            self.dpi = max(100, self.dpi // 2)  # Reduce DPI
            self.arrow_spacing = max(100, self.arrow_spacing * 2)  # Increase spacing
            self.skip_small_lanes = True
            if self.max_arrows_per_lane is None:
                self.max_arrows_per_lane = 10
        
        # Ensure valid arrow spacing
        if self.arrow_spacing <= 0:
            self.arrow_spacing = 50
    
    def merge_with_base_config(self, base_config: 'MapConfig'):
        """Merge with base configuration"""
        # Apply base config defaults if not overridden
        if not hasattr(self, '_merged'):
            for element in self.elements:
                if element not in self.style_overrides:
                    self.style_overrides[element] = base_config.get_visualization_style(element)
            self._merged = True


# Global default configuration
default_config = MapConfig()


def get_config() -> MapConfig:
    """Get the global default configuration"""
    return default_config


def set_config(config: MapConfig):
    """Set the global default configuration"""
    global default_config
    default_config = config


def update_config(**kwargs):
    """Update the global default configuration"""
    global default_config
    default_config.update(**kwargs) 