"""Tests for configuration module."""

import pytest
from src.config import (
    PipelineConfig,
    DataPaths,
    SegmentConfig,
    DataConfig,
    SplitConfig,
    NormalizationConfig,
    GapConfig,
    ModelConfig,
    PreprocessingConfig
)


class TestDataPaths:
    """Test DataPaths dataclass."""
    
    def test_default_paths(self):
        """Test default path configuration."""
        paths = DataPaths()
        
        assert paths.data_root is not None
        assert paths.meteo_root is not None
        assert paths.output_root is not None
        assert paths.model_data_dir is not None
    
    def test_custom_paths(self):
        """Test custom path configuration."""
        paths = DataPaths(
            data_root='/custom/data',
            output_root='/custom/output'
        )
        
        assert paths.data_root == '/custom/data'
        assert paths.output_root == '/custom/output'


class TestSegmentConfig:
    """Test SegmentConfig dataclass."""
    
    def test_default_segment_config(self):
        """Test default segment configuration."""
        config = SegmentConfig()
        
        assert config.segment_days == 30
        assert config.stride_days == 10
    
    def test_input_steps_calculation(self):
        """Test input steps calculation."""
        config = SegmentConfig(segment_days=30)
        
        # 30 days * 24 hours * 6 (10-min intervals per hour)
        expected_steps = 30 * 24 * 6
        assert config.input_steps == expected_steps
        assert config.input_steps == 4320
    
    def test_output_steps_calculation(self):
        """Test output steps calculation."""
        config = SegmentConfig(segment_days=30)
        
        # 30 days * 24 hours
        expected_steps = 30 * 24
        assert config.output_steps == expected_steps
        assert config.output_steps == 720
    
    def test_custom_segment_length(self):
        """Test custom segment length."""
        config = SegmentConfig(segment_days=45)
        
        assert config.input_steps == 45 * 24 * 6
        assert config.output_steps == 45 * 24


class TestDataConfig:
    """Test DataConfig dataclass."""
    
    def test_default_channels(self):
        """Test default channel configuration."""
        config = DataConfig()
        
        # Check input channels
        assert len(config.input_channels) == 11
        assert 'temp_treenet' in config.input_channels
        assert 'rh_treenet' in config.input_channels
        assert 'stem' in config.input_channels
        assert 'doy' in config.input_channels
        
        # Check target channels
        assert len(config.target_channels) == 3
        assert 'local_T' in config.target_channels
        assert 'local_RH' in config.target_channels
        assert 'stem' in config.target_channels
    
    def test_variable_names(self):
        """Test sensor variable names."""
        config = DataConfig()
        
        assert config.temperature_var == 'air temperature'
        assert config.humidity_var == 'relative humidity'
        assert config.dendrometer_var == 'tree stem radius change'


class TestGapConfig:
    """Test GapConfig dataclass."""
    
    def test_default_gap_config(self):
        """Test default gap configuration."""
        config = GapConfig()
        
        assert config.enabled is True
        assert config.min_gap_days == 1
        assert config.max_gap_days == 12
        assert config.min_gaps_per_segment == 1
        assert config.max_gaps_per_segment == 3
    
    def test_disable_gaps(self):
        """Test disabling gap injection."""
        config = GapConfig(enabled=False)
        
        assert config.enabled is False
    
    def test_custom_gap_range(self):
        """Test custom gap size range."""
        config = GapConfig(
            min_gap_days=2,
            max_gap_days=7
        )
        
        assert config.min_gap_days == 2
        assert config.max_gap_days == 7


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.n_blocks == 4
        assert config.n_filters == 64
        assert config.kernel_size == 3
        assert config.dropout_rate == 0.1
    
    def test_custom_model_config(self):
        """Test custom model configuration."""
        config = ModelConfig(
            n_blocks=6,
            n_filters=128,
            dropout_rate=0.2
        )
        
        assert config.n_blocks == 6
        assert config.n_filters == 128
        assert config.dropout_rate == 0.2


class TestPipelineConfig:
    """Test PipelineConfig integration."""
    
    def test_default_pipeline_config(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()
        
        # Check all nested configs exist
        assert isinstance(config.paths, DataPaths)
        assert isinstance(config.segment, SegmentConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.split, SplitConfig)
        assert isinstance(config.normalization, NormalizationConfig)
        assert isinstance(config.gap, GapConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
    
    def test_modify_nested_config(self):
        """Test modifying nested configuration."""
        config = PipelineConfig()
        
        # Modify segment config
        config.segment.segment_days = 45
        assert config.segment.segment_days == 45
        assert config.segment.input_steps == 45 * 24 * 6
        
        # Modify gap config
        config.gap.enabled = False
        assert config.gap.enabled is False
        
        # Modify model config
        config.model.n_blocks = 6
        assert config.model.n_blocks == 6
    
    def test_config_immutability(self):
        """Test that config values are properly set."""
        config = PipelineConfig()
        
        original_days = config.segment.segment_days
        config.segment.segment_days = 60
        
        assert config.segment.segment_days == 60
        assert config.segment.segment_days != original_days
    
    def test_preprocessing_config(self):
        """Test preprocessing configuration."""
        config = PipelineConfig()
        
        assert config.preprocessing.local_timezone == 'Europe/Zurich'
        assert config.preprocessing.utc_timezone == 'UTC'


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_segment_days_positive(self):
        """Test that segment days must be positive."""
        config = SegmentConfig(segment_days=30)
        assert config.segment_days > 0
    
    def test_stride_days_positive(self):
        """Test that stride days must be positive."""
        config = SegmentConfig(stride_days=10)
        assert config.stride_days > 0
    
    def test_gap_days_range(self):
        """Test gap days range is valid."""
        config = GapConfig(min_gap_days=1, max_gap_days=12)
        assert config.min_gap_days <= config.max_gap_days
        assert config.min_gap_days > 0
    
    def test_model_parameters_positive(self):
        """Test model parameters are positive."""
        config = ModelConfig()
        assert config.n_blocks > 0
        assert config.n_filters > 0
        assert config.kernel_size > 0
        assert 0 <= config.dropout_rate < 1
