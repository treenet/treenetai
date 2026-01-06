# API Documentation

Comprehensive reference for using pipeline_v2 modules programmatically.

## Table of Contents

- [Configuration](#configuration)
- [Data Loading](#data-loading)
- [Data Processing](#data-processing)
- [Segmentation](#segmentation)
- [Gap Injection](#gap-injection)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)

---

## Configuration

### `PipelineConfig`

Central configuration class with nested dataclasses for all pipeline parameters.

**Location:** `src/config.py`

**Usage:**
```python
from config import PipelineConfig

# Load default configuration
config = PipelineConfig()

# Access nested configs
print(config.paths.data_root)
print(config.segment.segment_days)  # Default: 30
print(config.data.input_channels)   # 11 channels
print(config.gap.max_gap_days)      # Default: 12

# Modify configuration
config.segment.segment_days = 45  # Change to 45-day segments
config.gap.enabled = False        # Disable gap injection
```

**Key Properties:**
- `paths: DataPaths` - All file paths
- `segment: SegmentConfig` - Segment extraction settings
- `data: DataConfig` - Channel definitions
- `split: SplitConfig` - Train/test split settings
- `normalization: NormalizationConfig` - Normalization strategy
- `gap: GapConfig` - Gap injection parameters
- `model: ModelConfig` - TCN architecture
- `preprocessing: PreprocessingConfig` - Timezone and timestamps

### `SegmentConfig`

Configure segment extraction parameters.

```python
from config import SegmentConfig

seg_config = SegmentConfig(
    segment_days=30,      # Segment length in days
    stride_days=10,       # Stride between segments
)

# Computed properties
input_steps = seg_config.input_steps    # 4320 for 30 days @ 10-min
output_steps = seg_config.output_steps  # 720 for 30 days @ hourly
```

---

## Data Loading

### `DataLoaders`

Load raw sensor and meteo data from feather/CSV files.

**Location:** `src/data/loaders.py`

**Usage:**
```python
from pathlib import Path
from data.loaders import DataLoaders
from config import PipelineConfig

config = PipelineConfig()
loader = DataLoaders(config)

# Load metadata
metadata = loader.load_metadata()
print(f"Total sensors: {len(metadata)}")

# Load specific sensor
series_id = 123
temp_df = loader.load_thermometer_l1(series_id)
print(temp_df.head())

# Load meteo data
site_id = 1
year = 2021
meteo_df = loader.load_meteotest_data(site_id, year)
print(meteo_df.columns)

# Get sites with complete data
complete_sites = loader.get_sites_with_complete_data(
    metadata=metadata,
    year=year
)
print(f"Sites with all sensors: {len(complete_sites)}")
```

**Methods:**
- `load_metadata() -> pd.DataFrame` - Load sensor metadata
- `load_thermometer_l1(series_id: int) -> pd.DataFrame` - Temperature (10-min)
- `load_hygrometer_l1(series_id: int) -> pd.DataFrame` - Humidity (10-min)
- `load_dendrometer_l2(series_id: int) -> pd.DataFrame` - Stem input (10-min)
- `load_dendrometer_lm(series_id: int) -> pd.DataFrame` - Stem target (10-min + hourly)
- `load_meteotest_data(site_id: int, year: int) -> pd.DataFrame` - Global meteo (daily)
- `get_sites_with_complete_data(metadata, year) -> List[Dict]` - Filter complete sites

---

## Data Processing

### `TimestampProcessor`

Handle UTC conversion and timezone safety.

**Location:** `src/data/processors.py`

**Usage:**
```python
from data.processors import TimestampProcessor

processor = TimestampProcessor(local_tz='Europe/Zurich')

# Convert local timestamps to UTC
df_utc = processor.to_utc_index(df_local, source_tz='Europe/Zurich')

# Convert UTC back to local
df_local = processor.to_local_index(df_utc, target_tz='Europe/Zurich')
```

**Methods:**
- `to_utc_index(df, source_tz) -> pd.DataFrame` - Convert to UTC (DST-safe)
- `to_local_index(df, target_tz) -> pd.DataFrame` - Convert to local time

### `DataResampler`

Resample time series to different resolutions.

**Usage:**
```python
from data.processors import DataResampler

resampler = DataResampler(local_tz='Europe/Zurich')

# Resample to hourly (subsampling at exact hours)
df_hourly = resampler.resample_to_hourly(df_10min)

# Resample to daily mean (in local timezone)
df_daily = resampler.resample_to_daily(df_10min)
```

**Methods:**
- `resample_to_hourly(df) -> pd.DataFrame` - Subsample to hourly
- `resample_to_daily(df) -> pd.DataFrame` - Aggregate to daily mean

### `DataMerger`

Combine multiple sensor streams.

**Usage:**
```python
from data.processors import DataMerger

merger = DataMerger()

# Merge local sensors
local_df = merger.merge_local_sensors(
    temp_df=temp_df,
    rh_df=rh_df,
    stem_df=stem_df
)

# Create full input array (11 channels)
input_df = merger.create_input_array(
    local_df=local_df,
    meteo_df=meteo_df
)
print(input_df.columns)  # ['temp_treenet', 'rh_treenet', 'stem', 'tas', ..., 'doy']

# Create target array (3 channels)
target_df = merger.create_target_array(
    temp_df=temp_hourly,
    rh_df=rh_hourly,
    stem_df=stem_hourly
)
print(target_df.columns)  # ['local_T', 'local_RH', 'stem']
```

**Methods:**
- `merge_local_sensors(temp_df, rh_df, stem_df) -> pd.DataFrame`
- `create_input_array(local_df, meteo_df) -> pd.DataFrame` - 11 channels
- `create_target_array(temp_df, rh_df, stem_df) -> pd.DataFrame` - 3 channels

---

## Segmentation

### `Normalizer`

Year-level min-max normalization.

**Location:** `src/data/segmentation.py`

**Usage:**
```python
from data.segmentation import Normalizer

normalizer = Normalizer()

# Compute year-level normalization parameters
norm_params = normalizer.compute_normalization_params(
    df_year=df,
    channels=['temp_treenet', 'rh_treenet', 'stem']
)

# Apply normalization
df_normalized = normalizer.normalize(
    df=df,
    params=norm_params,
    channels=['temp_treenet', 'rh_treenet', 'stem']
)

# Reverse normalization
df_original = normalizer.denormalize(
    df_normalized=df_normalized,
    params=norm_params,
    channels=['temp_treenet', 'rh_treenet', 'stem']
)
```

**Methods:**
- `compute_normalization_params(df_year, channels) -> Dict[str, Dict]`
- `normalize(df, params, channels) -> pd.DataFrame`
- `denormalize(df_normalized, params, channels) -> pd.DataFrame`

### `SegmentExtractor`

Extract complete 30-day segments with stride.

**Usage:**
```python
from data.segmentation import SegmentExtractor, SegmentMetadata
from config import SegmentConfig

config = SegmentConfig(segment_days=30, stride_days=10)
extractor = SegmentExtractor(config)

# Find complete segments
segments = extractor.find_complete_segments(
    input_df=input_df,    # 10-min resolution
    target_df=target_df   # Hourly resolution
)

for seg in segments:
    print(f"Segment: {seg.window_start_utc} to {seg.window_end_utc}")
    print(f"  Input: {seg.input_start_idx} to {seg.input_end_idx}")
    print(f"  Target: {seg.target_start_idx} to {seg.target_end_idx}")
```

**Methods:**
- `find_complete_segments(input_df, target_df) -> List[SegmentMetadata]`

### `SegmentBuilder`

Orchestrate full segmentation pipeline.

**Usage:**
```python
from data.segmentation import SegmentBuilder
from config import PipelineConfig

config = PipelineConfig()
builder = SegmentBuilder(config)

# Build segments for a sensor combination
result = builder.build_segments_for_combination(
    site_id=1,
    combination_id=0,
    sensor_ids={
        'thermometer': 123,
        'hygrometer': 456,
        'dendrometer': 789
    },
    year=2021
)

# Access results
train_input = result['train_input_segments']
train_output = result['train_output_segments']
metadata = result['metadata']
```

**Methods:**
- `build_segments_for_combination(site_id, combination_id, sensor_ids, year) -> Dict`

---

## Gap Injection

### `GapGenerator`

Generate random gaps for training augmentation.

**Location:** `src/gaps/gap_injection.py`

**Usage:**
```python
from gaps.gap_injection import GapGenerator
from config import GapConfig

gap_config = GapConfig(
    enabled=True,
    min_gap_days=1,
    max_gap_days=12,
    min_gaps_per_segment=1,
    max_gaps_per_segment=3
)

generator = GapGenerator(gap_config)

# Generate gap specification
gap_spec = generator.generate_gaps(
    segment_length=4320,  # 30 days @ 10-min
    n_channels=11
)

print(f"Number of gaps: {len(gap_spec)}")
for gap in gap_spec:
    print(f"  Channel {gap['channel']}: steps {gap['start_idx']} to {gap['end_idx']}")
```

**Methods:**
- `generate_gaps(segment_length, n_channels) -> List[Dict]`

### `GapInjector`

Apply gaps to data segments.

**Usage:**
```python
from gaps.gap_injection import GapInjector

injector = GapInjector()

# Inject gaps into a segment
X_gapped = injector.inject_gaps(
    X=X_segment,        # Shape: (4320, 11)
    gap_spec=gap_spec
)

# Check gap coverage
coverage = injector.compute_gap_coverage(X_gapped)
print(f"Gap coverage: {coverage:.1%}")
```

**Methods:**
- `inject_gaps(X, gap_spec) -> np.ndarray`
- `compute_gap_coverage(X) -> float`

---

## Model Architecture

### `TCNBlock`

Single TCN block with dilated causal convolutions.

**Location:** `src/models/tcn.py`

**Usage:**
```python
import tensorflow as tf
from models.tcn import TCNBlock

# Create TCN block
tcn_block = TCNBlock(
    n_filters=64,
    kernel_size=3,
    dilation_rate=2,
    dropout_rate=0.1
)

# Apply to input
x = tf.random.normal((32, 4320, 11))  # (batch, time, channels)
y = tcn_block(x, training=True)
print(y.shape)  # (32, 4320, 64)
```

**Parameters:**
- `n_filters: int` - Number of convolutional filters
- `kernel_size: int` - Kernel size (default: 3)
- `dilation_rate: int` - Dilation factor (default: 1)
- `dropout_rate: float` - Dropout rate (default: 0.1)

### `TCNModel`

Full TCN architecture with multi-task learning.

**Usage:**
```python
from models.tcn import TCNModel
from config import ModelConfig

model_config = ModelConfig(
    n_blocks=4,
    n_filters=64,
    kernel_size=3,
    dropout_rate=0.1
)

# Create model
model = TCNModel(
    config=model_config,
    n_input_channels=11,
    n_target_channels=3
)

# Build model
model.build(input_shape=(None, 4320, 11))
print(model.summary())

# Forward pass
X = tf.random.normal((32, 4320, 11))
recon, hourly = model(X, training=False)

print(recon.shape)   # (32, 4320, 3) - 10-min reconstruction
print(hourly.shape)  # (32, 720, 3) - hourly prediction
```

**Methods:**
- `call(inputs, training) -> Tuple[tf.Tensor, tf.Tensor]`
- `get_receptive_field() -> int` - Compute receptive field size

---

## Training

### `DataGenerator`

Keras Sequence for batch generation with gap injection.

**Location:** `src/models/training.py`

**Usage:**
```python
from models.training import DataGenerator
from config import PipelineConfig

config = PipelineConfig()

# Create generator
train_gen = DataGenerator(
    input_segments=train_input,     # List of input DataFrames
    output_segments=train_output,   # List of output DataFrames
    batch_size=32,
    gap_config=config.gap,
    shuffle=True
)

# Use in training
model.fit(train_gen, epochs=100)

# Get a single batch
X_batch, (Y_recon_batch, Y_hourly_batch) = train_gen[0]
print(X_batch.shape)         # (32, 4320, 11)
print(Y_recon_batch.shape)   # (32, 4320, 3)
print(Y_hourly_batch.shape)  # (32, 720, 3)
```

**Methods:**
- `__len__() -> int` - Number of batches per epoch
- `__getitem__(idx) -> Tuple` - Get batch (X, (Y_recon, Y_hourly))
- `on_epoch_end()` - Shuffle data

### `ModelTrainer`

Orchestrate training with callbacks and experiment tracking.

**Usage:**
```python
from models.training import ModelTrainer
from config import PipelineConfig

config = PipelineConfig()

trainer = ModelTrainer(
    config=config,
    experiment_name='baseline_tcn'
)

# Train model
history = trainer.train(
    train_input=train_input,
    train_output=train_output,
    val_input=val_input,
    val_output=val_output,
    epochs=100,
    batch_size=32
)

# Access training history
print(history.history.keys())
# ['loss', 'recon_loss', 'hourly_loss', 'val_loss', ...]

# Load best model
best_model = tf.keras.models.load_model(
    trainer.experiment_dir / 'best_model.keras'
)
```

**Methods:**
- `train(train_input, train_output, val_input, val_output, epochs, batch_size) -> History`
- `evaluate(model, test_input, test_output) -> Dict`

---

## Evaluation

### `GapFillingMetrics`

Compute gap filling performance metrics.

**Location:** `src/gaps/metrics.py`

**Usage:**
```python
from gaps.metrics import GapFillingMetrics

metrics = GapFillingMetrics()

# Compute metrics
results = metrics.compute_metrics(
    y_true=y_true,      # Ground truth
    y_pred=y_pred,      # Predictions
    gap_mask=gap_mask   # Boolean mask (True = gap)
)

print(results)
# {
#     'mae_overall': 0.15,
#     'mae_gaps': 0.22,
#     'mae_non_gaps': 0.12,
#     'rmse_overall': 0.20,
#     'rmse_gaps': 0.30,
#     'rmse_non_gaps': 0.15,
#     'r2_overall': 0.85,
#     'r2_gaps': 0.75,
#     'r2_non_gaps': 0.90
# }
```

**Methods:**
- `compute_metrics(y_true, y_pred, gap_mask) -> Dict[str, float]`
- `compute_per_channel_metrics(y_true, y_pred, gap_mask, channel_names) -> pd.DataFrame`

---

## Visualization

### `SegmentPlotter`

Plot processed segments for validation.

**Location:** `src/visualization/plot_segments.py`

**Usage:**
```python
from pathlib import Path
from visualization.plot_segments import SegmentPlotter

plotter = SegmentPlotter(local_tz='Europe/Zurich')

# Load segments
data_dir = Path('processed/model_data')
combo_ids, input_segs, output_segs, metadata = plotter.load_segments(
    data_dir=data_dir,
    split='train'
)

# Plot specific segments for a site
output_dir = Path('plots')
n_plots = plotter.plot_segments_for_site(
    data_dir=data_dir,
    site_id=1,
    year=2021,
    output_dir=output_dir,
    split='train',
    max_segments=10
)

# Generate summary statistics
plotter.plot_summary_stats(
    data_dir=data_dir,
    output_path=output_dir / 'summary.png',
    split='train'
)
```

**Methods:**
- `load_segments(data_dir, split) -> Tuple` - Load pickled segments
- `plot_segment(input_df, output_df, ...) -> None` - Plot single segment
- `plot_segments_for_site(data_dir, site_id, year, output_dir, split, max_segments) -> int`
- `plot_summary_stats(data_dir, output_path, split) -> None`

### `RawDataComparator`

Compare denormalized segments with raw data.

**Location:** `src/visualization/compare_raw.py`

**Usage:**
```python
from pathlib import Path
from visualization.compare_raw import RawDataComparator

comparator = RawDataComparator(local_tz='Europe/Zurich')

# Compare a segment
comparator.compare_segment(
    data_dir=Path('processed/model_data'),
    raw_root=Path('/storage/lukovic/Data/FORWARDS/treenet/server_data'),
    meteo_root=Path('/storage/lukovic/Data/FORWARDS/treenet/meteo_data'),
    split='train',
    combo_id=0,
    seg_idx=0,
    year=2021,
    site_id=1,
    output_dir=Path('comparisons')
)
```

**Methods:**
- `load_segment_metadata(data_dir, split, combo_id, seg_idx) -> Dict`
- `denormalize_segment(df_norm, mins, diffs, channels) -> pd.DataFrame`
- `load_raw_sensor(raw_root, sensor_type, sensor_id, column) -> pd.DataFrame`
- `load_raw_meteo(meteo_root, site_id, year) -> pd.DataFrame`
- `compare_segment(data_dir, raw_root, meteo_root, split, combo_id, seg_idx, year, site_id, output_dir) -> None`

---

## Complete Example

### Build Custom Pipeline

```python
from pathlib import Path
from config import PipelineConfig, SegmentConfig
from data.loaders import DataLoaders
from data.segmentation import SegmentBuilder
from models.tcn import TCNModel
from models.training import ModelTrainer

# 1. Configure pipeline
config = PipelineConfig()
config.segment.segment_days = 45  # Use 45-day segments
config.gap.max_gap_days = 7       # Smaller gaps

# 2. Load data
loader = DataLoaders(config)
metadata = loader.load_metadata()
complete_sites = loader.get_sites_with_complete_data(metadata, year=2021)

# 3. Build segments
builder = SegmentBuilder(config)
segments = builder.build_segments_for_combination(
    site_id=complete_sites[0]['site_id'],
    combination_id=0,
    sensor_ids=complete_sites[0]['sensors'],
    year=2021
)

# 4. Train model
trainer = ModelTrainer(config, experiment_name='custom_45day')
history = trainer.train(
    train_input=segments['train_input_segments'],
    train_output=segments['train_output_segments'],
    val_input=segments['test_input_segments'],
    val_output=segments['test_output_segments'],
    epochs=50,
    batch_size=16
)

# 5. Visualize results
from visualization.plot_segments import SegmentPlotter

plotter = SegmentPlotter()
plotter.plot_segments_for_site(
    data_dir=Path(config.paths.model_data_dir),
    site_id=complete_sites[0]['site_id'],
    year=2021,
    output_dir=Path('custom_plots'),
    split='train',
    max_segments=5
)
```

---

## Error Handling

### Common Exceptions

**`FileNotFoundError`**
```python
try:
    df = loader.load_thermometer_l1(series_id=999)
except FileNotFoundError as e:
    print(f"Sensor file not found: {e}")
```

**`ValueError`**
```python
try:
    segments = extractor.find_complete_segments(input_df, target_df)
except ValueError as e:
    print(f"Invalid segment configuration: {e}")
```

**`RuntimeError`**
```python
try:
    meta = comparator.load_segment_metadata(data_dir, 'train', combo_id=99, seg_idx=0)
except RuntimeError as e:
    print(f"Segment metadata not found: {e}")
```

---

## Type Hints

All modules use comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

def process_data(
    df: pd.DataFrame,
    columns: List[str],
    normalize: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Process DataFrame with specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to process
        normalize: Whether to normalize data
        
    Returns:
        Tuple of (processed DataFrame, statistics dict)
    """
    pass
```

---

## Performance Tips

1. **Batch loading**: Load multiple sensors at once to reduce I/O
2. **Caching**: Cache normalization parameters for repeated use
3. **GPU memory**: Use smaller batch sizes if running out of memory
4. **Parallel processing**: Use multiprocessing for segment extraction
5. **Data types**: Use `float32` instead of `float64` to save memory

```python
# Example: Efficient batch loading
sensor_ids = [123, 456, 789]
dfs = [loader.load_thermometer_l1(sid) for sid in sensor_ids]
combined = pd.concat(dfs, keys=sensor_ids)
```

---

## Logging

All modules use Python's logging framework:

```python
import logging
from utils import setup_logging

# Setup logging
logger = setup_logging('my_script', level=logging.DEBUG)

# Use logger
logger.info("Starting data processing")
logger.debug(f"Processing {len(segments)} segments")
logger.warning("Missing data detected")
logger.error("Failed to load sensor data")
```

---

## Testing

### Unit Test Example

```python
import pytest
from data.processors import DataResampler

def test_resample_to_hourly():
    """Test hourly resampling preserves timestamps."""
    resampler = DataResampler(local_tz='Europe/Zurich')
    
    # Create 10-min data
    index = pd.date_range('2021-01-01', periods=144, freq='10min', tz='UTC')
    df = pd.DataFrame({'value': range(144)}, index=index)
    
    # Resample to hourly
    df_hourly = resampler.resample_to_hourly(df)
    
    # Check
    assert len(df_hourly) == 24
    assert df_hourly.index.freq == 'h'
```

Run tests:
```bash
pytest tests/ -v
```

---

## Contributing

When adding new modules:

1. **Type hints**: Add comprehensive type annotations
2. **Docstrings**: Use Google-style docstrings
3. **Logging**: Use logging instead of print statements
4. **Error handling**: Raise appropriate exceptions
5. **Tests**: Add unit tests for new functionality
6. **Documentation**: Update this API doc

---

## Support

For questions or issues:
- Check [README.md](README.md) for overview
- Check [VISUALIZATION.md](VISUALIZATION.md) for plotting
- Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
- Review code comments and docstrings in source files
