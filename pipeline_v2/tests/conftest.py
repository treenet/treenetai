"""pytest configuration and shared fixtures."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from src.config import (
    DataPaths,
    SegmentConfig,
    DataConfig,
    GapConfig,
    ModelConfig,
    PreprocessingConfig,
    PipelineConfig
)


@pytest.fixture
def sample_config():
    """Create sample pipeline configuration for testing."""
    return PipelineConfig()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_10min_data():
    """Create sample 10-minute resolution data."""
    # 1 day of 10-min data (144 timesteps)
    index = pd.date_range(
        '2021-01-01 00:00:00',
        periods=144,
        freq='10min',
        tz='UTC'
    )
    
    data = {
        'value': np.sin(np.linspace(0, 4*np.pi, 144)) + np.random.normal(0, 0.1, 144)
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_hourly_data():
    """Create sample hourly resolution data."""
    # 1 day of hourly data (24 timesteps)
    index = pd.date_range(
        '2021-01-01 00:00:00',
        periods=24,
        freq='h',
        tz='UTC'
    )
    
    data = {
        'value': np.sin(np.linspace(0, 4*np.pi, 24)) + np.random.normal(0, 0.1, 24)
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_daily_data():
    """Create sample daily resolution data."""
    # 30 days of daily data
    index = pd.date_range(
        '2021-01-01',
        periods=30,
        freq='D',
        tz='Europe/Zurich'
    )
    
    data = {
        'tas': 10 + 5 * np.sin(np.linspace(0, 2*np.pi, 30)),
        'tasmax': 15 + 5 * np.sin(np.linspace(0, 2*np.pi, 30)),
        'tasmin': 5 + 5 * np.sin(np.linspace(0, 2*np.pi, 30)),
        'rh': 60 + 20 * np.cos(np.linspace(0, 2*np.pi, 30)),
        'vpd': 0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 30)),
        'gh': 100 + 50 * np.sin(np.linspace(0, 2*np.pi, 30)),
        'pr': np.random.exponential(2, 30)
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_segment_input():
    """Create sample segment input array (30 days, 10-min, 11 channels)."""
    n_steps = 30 * 24 * 6  # 4320 timesteps
    
    index = pd.date_range(
        '2021-01-01 00:00:00',
        periods=n_steps,
        freq='10min',
        tz='UTC'
    )
    
    # Create 11 channels with different patterns
    data = {
        'temp_treenet': 15 + 5 * np.sin(np.linspace(0, 30*2*np.pi, n_steps)),
        'rh_treenet': 60 + 20 * np.cos(np.linspace(0, 30*2*np.pi, n_steps)),
        'stem': 10 + 2 * np.sin(np.linspace(0, 30*np.pi, n_steps)),
        'tas': np.repeat(10 + 5 * np.sin(np.linspace(0, 2*np.pi, 30)), 144),
        'tasmax': np.repeat(15 + 5 * np.sin(np.linspace(0, 2*np.pi, 30)), 144),
        'tasmin': np.repeat(5 + 5 * np.sin(np.linspace(0, 2*np.pi, 30)), 144),
        'rh': np.repeat(60 + 20 * np.cos(np.linspace(0, 2*np.pi, 30)), 144),
        'vpd': np.repeat(0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 30)), 144),
        'gh': np.repeat(100 + 50 * np.sin(np.linspace(0, 2*np.pi, 30)), 144),
        'pr': np.repeat(np.random.exponential(2, 30), 144),
        'doy': np.linspace(1, 31, n_steps)
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_segment_output():
    """Create sample segment output array (30 days, hourly, 3 channels)."""
    n_steps = 30 * 24  # 720 timesteps
    
    index = pd.date_range(
        '2021-01-01 00:00:00',
        periods=n_steps,
        freq='h',
        tz='UTC'
    )
    
    data = {
        'local_T': 15 + 5 * np.sin(np.linspace(0, 30*2*np.pi, n_steps)),
        'local_RH': 60 + 20 * np.cos(np.linspace(0, 30*2*np.pi, n_steps)),
        'stem': 10 + 2 * np.sin(np.linspace(0, 30*np.pi, n_steps))
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_normalized_segment():
    """Create sample normalized segment (values in [0, 1])."""
    n_steps = 30 * 24 * 6  # 4320 timesteps
    
    index = pd.date_range(
        '2021-01-01 00:00:00',
        periods=n_steps,
        freq='10min',
        tz='UTC'
    )
    
    # Normalized values [0, 1]
    data = {
        'temp_treenet': np.random.uniform(0, 1, n_steps),
        'rh_treenet': np.random.uniform(0, 1, n_steps),
        'stem': np.random.uniform(0, 1, n_steps)
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_normalization_params():
    """Create sample normalization parameters."""
    return {
        'temp_treenet': {'min': 10.0, 'max': 20.0, 'diff': 10.0},
        'rh_treenet': {'min': 40.0, 'max': 80.0, 'diff': 40.0},
        'stem': {'min': 8.0, 'max': 12.0, 'diff': 4.0}
    }


@pytest.fixture
def sample_gap_spec():
    """Create sample gap specification."""
    return [
        {'channel': 0, 'start_idx': 100, 'end_idx': 300},
        {'channel': 1, 'start_idx': 500, 'end_idx': 800},
        {'channel': 2, 'start_idx': 1000, 'end_idx': 1200}
    ]


@pytest.fixture
def sample_numpy_segment():
    """Create sample numpy segment array."""
    return np.random.rand(4320, 11).astype(np.float32)


@pytest.fixture
def sample_numpy_target():
    """Create sample numpy target array."""
    return np.random.rand(720, 3).astype(np.float32)
