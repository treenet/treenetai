# TreeNet AI Pipeline v2

Modular pipeline for dendrometer and climate data processing with gap filling using Temporal Convolutional Networks (TCN).

## Overview

This pipeline processes raw sensor data from TreeNet monitoring sites to:
1. **Clean and align** multi-sensor time series data (temperature, humidity, dendrometer)
2. **Extract 30-day segments** with strict completeness requirements
3. **Train TCN models** for gap filling and hourly prediction
4. **Evaluate performance** on test sites
5. **Visualize segments** to validate data quality

### Key Features

- **Year-level normalization** for consistent scaling
- **UTC-safe timestamp handling** for DST transitions
- **Many-to-one mapping** support (multiple input combos → same output)
- **Gap injection** for training data augmentation (1-12 day gaps)
- **Multi-task learning**: reconstruction + hourly prediction
- **Modular design** with clear separation of concerns
- **Comprehensive visualization** for validation

### Documentation

- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation guide
- **[QUICKSTART.md](QUICKSTART.md)** - Fast setup and first run
- **[VISUALIZATION.md](VISUALIZATION.md)** - Guide for plotting segments
- **[API.md](API.md)** - Complete API reference for programmatic use
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and features
- **[tests/README.md](tests/README.md)** - Testing and development guide

## Project Structure

```
pipeline_v2/
├── src/
│   ├── config.py              # Centralized configuration
│   ├── utils.py               # Utility functions
│   ├── data/
│   │   ├── loaders.py         # Data loading from raw files
│   │   ├── processors.py      # UTC conversion, resampling, merging
│   │   ├── segmentation.py    # 30-day segment extraction
│   │   └── validation.py      # Data quality checks
│   ├── gaps/
│   │   ├── gap_injection.py   # Random gap generation
│   │   └── metrics.py         # Gap filling evaluation metrics
│   ├── models/
│   │   ├── tcn.py             # TCN architecture
│   │   └── training.py        # Training pipeline
│   └── visualization/
│       ├── plot_segments.py   # Segment visualization
│       └── compare_raw.py     # Raw data comparison
├── 1_build_segments.py        # CLI: Build segments from raw data
├── 2_train_model.py           # CLI: Train TCN model
├── 3_evaluate.py              # CLI: Evaluate trained model
├── 4_visualize_segments.py    # CLI: Visualize processed segments
├── 5_compare_with_raw.py      # CLI: Compare with raw data
├── configs/
│   └── default.yaml           # Default configuration
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── QUICKSTART.md              # Quick setup guide
├── VISUALIZATION.md           # Visualization guide
└── IMPLEMENTATION_SUMMARY.md  # Technical overview
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.2+ (for GPU training, optional)

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install requirements
pip install -r requirements.txt
```

## Usage

### 1. Build Segments from Raw Data

Extract 30-day segments with complete coverage:

```bash
python 1_build_segments.py \\
    --data-root /storage/lukovic/Data/FORWARDS/treenet/server_data \\
    --meteo-root /storage/lukovic/Data/FORWARDS/treenet/meteo_data \\
    --output-root /storage/lukovic/Data/FORWARDS/treenet/processed \\
    --year 2020 \\
    --test-ratio 0.2 \\
    --segment-days 30 \\
    --stride-days 10 \\
    --verbose
```

**Parameters:**
- `--data-root`: Directory with raw sensor data (thermometer_l1/, hygrometer_l1/, etc.)
- `--meteo-root`: Directory with meteotest CSV files
- `--output-root`: Output directory for processed segments
- `--year`: Year to process
- `--test-ratio`: Fraction of sites to use for testing (default: 0.2)
- `--segment-days`: Length of each segment in days (default: 30)
- `--stride-days`: Overlap between consecutive segments (default: 10)

**Outputs:**
```
processed/model_data/
├── train_input_segments_numpy.pkl    # (N, 4320, 11) - training inputs
├── train_output_segments_numpy.pkl   # (N, 720, 3) - training targets
├── test_input_segments_numpy.pkl     # (M, 4320, 11) - test inputs
├── test_output_segments_numpy.pkl    # (M, 720, 3) - test targets
├── train_segment_ids.pkl             # Metadata with normalization params
├── test_segment_ids.pkl              # Test metadata
└── model_*_data_combination_ids.pkl  # Sensor ID mappings
```

### 2. Train TCN Model

Train the model with gap injection:

```bash
python 2_train_model.py \\
    --data-dir /storage/lukovic/Data/FORWARDS/treenet/processed/model_data \\
    --output-dir ./experiments \\
    --experiment-name baseline_tcn \\
    --epochs 100 \\
    --batch-size 32 \\
    --n-blocks 4 \\
    --n-filters 64 \\
    --max-gap-days 12 \\
    --verbose
```

**Parameters:**
- `--data-dir`: Directory with processed segments from step 1
- `--output-dir`: Base directory for experiment outputs
- `--experiment-name`: Optional name for this experiment
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--n-blocks`: Number of TCN blocks (default: 4)
- `--n-filters`: Filters per block (default: 64)
- `--max-gap-days`: Maximum gap length for augmentation (default: 12)
- `--no-gaps`: Disable gap injection (not recommended)

**Outputs:**
```
experiments/20260106_120000_baseline_tcn/
├── best_model.keras              # Best model checkpoint
├── final_model.keras             # Final model after training
├── training_history.csv          # Loss curves
├── evaluation_metrics.json       # Test set metrics
├── config.json                   # Full configuration used
├── training.log                  # Training logs
└── tensorboard/                  # TensorBoard logs
```

### 3. Evaluate Model

```bash
python 3_evaluate.py \\
    --model-path experiments/20260106_120000_baseline_tcn/best_model.keras \\
    --data-dir /storage/lukovic/Data/FORWARDS/treenet/processed/model_data \\
    --output-dir evaluation_results
```

### 4. Visualize Segments

Inspect processed segments to validate data quality:

```bash
# Plot segments for specific site and year
python 4_visualize_segments.py --site 1 --year 2021 --split train

# Generate summary statistics
python 4_visualize_segments.py --summary --split train

# Plot all segments (limited per site)
python 4_visualize_segments.py --all --split train --max-per-site 5
```

**Parameters:**
- `--site`: Site ID to visualize (requires --year)
- `--year`: Year to filter
- `--split`: 'train' or 'test' (default: train)
- `--summary`: Generate summary statistics only
- `--all`: Plot segments for all sites
- `--max-per-site`: Maximum segments per site (default: 10)
- `--no-globals`: Hide global channels from plots

**Outputs:**
```
processed/model_data/visualizations/train/
├── summary_statistics.png            # Overview of all segments
└── site_1/
    └── year_2021/
        ├── segment_site1_combo0_seg0.png
        ├── segment_site1_combo0_seg1.png
        └── ...
```

Each segment plot shows:
- **Top panel**: 11 input channels at 10-min resolution (local sensors + global meteo)
- **Bottom panel**: 3 target channels at hourly resolution
- **Metadata**: Site, combination, segment index, date range, coverage stats

### 5. Compare with Raw Data

Validate data integrity by comparing denormalized segments with original raw files:

```bash
# Compare specific segment
python 5_compare_with_raw.py \\
    --data-dir processed/model_data \\
    --raw-root /storage/lukovic/Data/FORWARDS/treenet/server_data \\
    --meteo-root /storage/lukovic/Data/FORWARDS/treenet/meteo_data \\
    --split train \\
    --combo-id 0 \\
    --seg-idx 0 \\
    --year 2021 \\
    --site 1 \\
    --output-dir comparisons

# Compare all segments for a combination
python 5_compare_with_raw.py \\
    --data-dir processed/model_data \\
    --raw-root /storage/lukovic/Data/FORWARDS/treenet/server_data \\
    --meteo-root /storage/lukovic/Data/FORWARDS/treenet/meteo_data \\
    --split train \\
    --combo-id 0 \\
    --all-segments \\
    --year 2021 \\
    --site 1 \\
    --output-dir comparisons
```

**What it does:**
1. Loads normalized segment data
2. Denormalizes to original scale using stored min/max values
3. Loads corresponding raw sensor/meteo files
4. Plots both side-by-side for visual comparison

This ensures:
- ✅ Normalization/denormalization is reversible
- ✅ No data corruption during processing
- ✅ Time windows match exactly
- ✅ Resolution conversions are accurate

## Configuration

The pipeline uses a centralized configuration system in [src/config.py](src/config.py). Key parameters:

### Data Configuration

```python
@dataclass
class DataConfig:
    # Sensor variable names in metadata
    temperature_var: str = 'air temperature'
    humidity_var: str = 'relative humidity'
    dendrometer_var: str = 'tree stem radius change'
    
    # 11 input channels
    input_channels: List[str] = [
        'temp_treenet',   # Local temperature (10-min)
        'rh_treenet',     # Local RH (10-min)
        'stem',           # Stem radius change (10-min)
        'tas',            # Global avg temp (daily)
        'tasmax',         # Global max temp (daily)
        'tasmin',         # Global min temp (daily)
        'rh',             # Global RH (daily)
        'vpd',            # Vapor pressure deficit (daily)
        'gh',             # Global radiation (daily)
        'pr',             # Precipitation (daily)
        'doy'             # Day of year
    ]
    
    # 3 target channels (hourly)
    target_channels: List[str] = [
        'local_T',   # Cleaned temperature
        'local_RH',  # Cleaned relative humidity
        'stem'       # Cleaned stem radius change
    ]
```

### Segment Configuration

```python
@dataclass
class SegmentConfig:
    segment_days: int = 30    # ← Adjustable
    stride_days: int = 10     # ← Adjustable
    
    @property
    def input_steps(self) -> int:
        return self.segment_days * 24 * 6  # 4320 for 30 days
    
    @property
    def output_steps(self) -> int:
        return self.segment_days * 24  # 720 for 30 days
```

### Gap Configuration

```python
@dataclass
class GapConfig:
    enabled: bool = True
    min_gap_days: int = 1
    max_gap_days: int = 12
    min_gaps_per_segment: int = 1
    max_gaps_per_segment: int = 3
    gap_channel_prob: float = 0.5
```

## Data Format

### Raw Data Structure

Expected directory structure under `data_root`:

```
server_data/
├── metadata_all.pkl                   # Metadata for all sensors
├── thermometer_l1/
│   └── thermometer_l1_series_id_*.ftr  # Raw temperature (10-min)
├── hygrometer_l1/
│   └── hygrometer_l1_series_id_*.ftr   # Raw humidity (10-min)
├── dendrometer_l2/
│   └── dendrometer_l2_series_id_*.ftr  # Raw stem (10-min)
├── dendrometer_lm/
│   └── dendrometer_lm_series_id_*.ftr  # Ground truth (10-min + hourly)
└── meteo_data/
    └── site_*.csv                      # Global meteo (daily)
```

### Metadata Format

`metadata_all.pkl` must contain:
- `series_id`: Unique sensor ID
- `site_id`: Site ID
- `variable_name`: Sensor type ('air temperature', 'relative humidity', 'tree stem radius change')
- `series_start`: Start date
- `series_stop`: End date

## Model Architecture

### TCN (Temporal Convolutional Network)

The model uses a **multi-task architecture**:

1. **Encoder**: Stack of dilated causal convolution blocks
   - Exponentially increasing dilation rates (1, 2, 4, 8, ...)
   - Residual connections
   - Batch normalization + dropout

2. **Decoder Branch 1**: Reconstruction (10-min resolution)
   - Predicts masked input channels
   - Learns to fill gaps in local sensor data

3. **Decoder Branch 2**: Hourly prediction
   - Downsamples via average pooling (6:1 ratio)
   - Predicts cleaned hourly targets

### Loss Function

Weighted multi-task loss:
```
L = w_recon * MSE(recon, input) + w_hourly * MSE(hourly, target)
```

Where:
- `w_recon`: Weight for reconstruction (higher for masked regions)
- `w_hourly`: Weight for hourly prediction

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black src/
pylint src/
```

### Adding New Features

1. Data loaders: Edit `src/data/loaders.py`
2. Processing: Edit `src/data/processors.py`
3. Model architecture: Edit `src/models/tcn.py`
4. Training logic: Edit `src/models/training.py`

## Performance

### Expected Metrics

On test set (30-day segments, 12-day gaps):
- **Reconstruction MAE**: 0.05-0.15 (normalized units)
- **Hourly prediction MAE**: 0.08-0.20 (normalized units)
- **R² score**: 0.80-0.95

### Training Time

- **CPU**: ~4-6 hours per 100 epochs
- **GPU (V100)**: ~30-45 minutes per 100 epochs

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python 2_train_model.py --batch-size 16
```

### Low Performance

- Increase model capacity: `--n-blocks 6 --n-filters 128`
- Train longer: `--epochs 200`
- Check data quality: Visualize segments

### DST Issues

The pipeline handles DST transitions automatically using UTC-safe conversion. If you encounter timezone errors, check that raw data timestamps are properly formatted.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{treenetai_pipeline_v2,
  title={TreeNet AI Pipeline v2},
  author={Lukovic, Mirko},
  year={2026},
  url={https://github.com/username/treenetai}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues:
- Email: mirko.lukovic@example.com
- Issues: https://github.com/username/treenetai/issues

## Acknowledgments

- Based on improvements documented in `PROJECT_IMPROVEMENTS.md`
- Inspired by original pipeline in `../pipeline/monthly/`
- Uses TensorFlow/Keras for deep learning
