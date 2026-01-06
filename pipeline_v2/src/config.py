"""
Centralized configuration for TreeNet AI Pipeline v2.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import os


@dataclass
class DataPaths:
    """Data paths configuration."""
    
    # Main data directory
    data_root: Path = Path('/storage/lukovic/Data/FORWARDS/treenet/server_data')
    meteo_root: Path = Path('/storage/lukovic/Data/FORWARDS/treenet/meteo_data')
    
    # Output directory
    output_root: Path = Path('/storage/lukovic/Data/FORWARDS/treenet/processed')
    
    # Development paths (for Mac)
    dev_data_root: Path = Path('/Users/lukovic/data/FORWARDS/TreeNet/server_data')
    dev_meteo_root: Path = Path('/Users/lukovic/data/FORWARDS/TreeNet/meteo_data')
    
    def get_paths(self, dev_mode: bool = False) -> Tuple[Path, Path]:
        """Get data and meteo paths based on environment."""
        if dev_mode:
            return self.dev_data_root, self.dev_meteo_root
        return self.data_root, self.meteo_root
    
    @classmethod
    def from_env(cls) -> DataPaths:
        """Create paths from environment variables."""
        return cls(
            data_root=Path(os.getenv('TREENET_DATA_PATH', 
                                     '/storage/lukovic/Data/FORWARDS/treenet/server_data')),
            meteo_root=Path(os.getenv('TREENET_METEO_PATH', 
                                      '/storage/lukovic/Data/FORWARDS/treenet/meteo_data')),
            output_root=Path(os.getenv('TREENET_OUTPUT_PATH',
                                       '/storage/lukovic/Data/FORWARDS/treenet/processed'))
        )


@dataclass
class SegmentConfig:
    """Segmentation configuration."""
    
    # Segment parameters
    segment_days: int = 30  # Length of each segment in days
    stride_days: int = 10   # Overlap between consecutive segments
    
    # Data resolution
    resolution_minutes: int = 10  # 10-minute resolution for local sensors
    hourly_target_resolution: bool = True  # Convert targets to hourly
    
    # Temporal parameters
    steps_per_hour: int = 6  # 60 / 10 = 6
    hours_per_day: int = 24
    
    @property
    def input_steps(self) -> int:
        """Total 10-minute steps in a segment: 30 days * 24 h * 6 steps/h = 4320."""
        return self.segment_days * self.hours_per_day * self.steps_per_hour
    
    @property
    def output_steps(self) -> int:
        """Total hourly steps in a segment: 30 days * 24 h = 720."""
        return self.segment_days * self.hours_per_day
    
    @property
    def stride_steps(self) -> int:
        """Stride in 10-minute steps."""
        return self.stride_days * self.hours_per_day * self.steps_per_hour


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Timezone
    local_tz: str = 'Europe/Zurich'
    
    # Sensor variable names in metadata
    temperature_var: str = 'air temperature'
    humidity_var: str = 'relative humidity'
    dendrometer_var: str = 'tree stem radius change'
    
    # Channel configuration
    input_channels: List[str] = field(default_factory=lambda: [
        'temp_treenet',   # Local temperature (10-min)
        'rh_treenet',     # Local relative humidity (10-min)
        'stem',           # Tree stem radius change (10-min)
        'tas',            # Global average daily temperature
        'tasmax',         # Global max daily temperature
        'tasmin',         # Global min daily temperature
        'rh',             # Global daily relative humidity
        'vpd',            # Global vapor pressure deficit
        'gh',             # Global radiation
        'pr',             # Global precipitation
        'doy'             # Day of year
    ])
    
    target_channels: List[str] = field(default_factory=lambda: [
        'local_T',   # Cleaned local temperature (hourly)
        'local_RH',  # Cleaned local relative humidity (hourly)
        'stem'       # Cleaned stem radius change (hourly)
    ])
    
    # Meteo columns
    meteo_columns: List[str] = field(default_factory=lambda: [
        'ts', 'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr'
    ])
    
    @property
    def n_input_channels(self) -> int:
        return len(self.input_channels)
    
    @property
    def n_target_channels(self) -> int:
        return len(self.target_channels)


@dataclass
class SplitConfig:
    """Train/test split configuration."""
    
    test_ratio: float = 0.2  # 20% for test
    random_seed: int = 42
    
    # Site filtering
    min_sensors_per_site: int = 3  # Must have at least 1 of each sensor type
    
    @property
    def train_ratio(self) -> float:
        return 1.0 - self.test_ratio


@dataclass
class NormalizationConfig:
    """Normalization configuration."""
    
    method: str = 'minmax'  # 'minmax' or 'zscore'
    scope: str = 'year'     # 'year' or 'segment'
    clip_percentile: float = 0.0  # Clip outliers at percentiles (0 = no clipping)


@dataclass
class GapConfig:
    """Gap injection configuration for training."""
    
    enabled: bool = True
    
    # Gap parameters
    min_gap_days: int = 1
    max_gap_days: int = 12
    
    # How many gaps to inject per segment
    min_gaps_per_segment: int = 1
    max_gaps_per_segment: int = 3
    
    # Channel selection probability
    gap_channel_prob: float = 0.5
    
    # Random seed
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    
    # Architecture
    model_type: str = 'tcn'
    n_filters: int = 64
    kernel_size: int = 3
    n_blocks: int = 4
    dropout_rate: float = 0.2
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 3e-4
    
    # Loss weights
    recon_masked_weight: float = 1.0
    recon_unmasked_weight: float = 0.05
    hourly_weight: float = 1.0
    
    # Callbacks
    early_stop_patience: int = 10
    reduce_lr_patience: int = 4
    min_lr: float = 1e-6
    
    # Optimization
    use_mixed_precision: bool = False


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration."""
    
    # Hampel filter for outlier detection
    use_hampel_filter: bool = True
    hampel_window: int = 13
    hampel_sigmas: float = 3.0
    
    # Misalignment correction
    use_misalignment_correction: bool = True
    max_lag_steps: int = 144  # 24 hours * 6 steps/hour
    
    # Detrending
    use_detrending: bool = False
    lowess_frac: float = 0.02
    lowess_iter: int = 3


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all sub-configs."""
    
    paths: DataPaths = field(default_factory=DataPaths)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # Global settings
    verbose: bool = True
    save_intermediate: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # TODO: Implement YAML parsing with nested dataclasses
        raise NotImplementedError("YAML loading not yet implemented")
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        config_dict = asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.split.test_ratio > 0 and self.split.test_ratio < 1, \
            "test_ratio must be between 0 and 1"
        assert self.segment.segment_days > 0, "segment_days must be positive"
        assert self.segment.stride_days > 0, "stride_days must be positive"
        assert self.gap.min_gap_days <= self.gap.max_gap_days, \
            "min_gap_days must be <= max_gap_days"
        assert self.model.batch_size > 0, "batch_size must be positive"
        assert self.model.epochs > 0, "epochs must be positive"


# Default configuration instance
default_config = PipelineConfig()
