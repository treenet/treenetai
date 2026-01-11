# TreeNet AI Pipeline v2 - Project Context

**Purpose**: This file documents project-specific context details to help maintain continuity across sessions. Reference this document at the start of any new session.

**Last Updated**: 2026-01-11

---

## 0. Project Goal & Overview

### High-Level Objective
Convert raw tree physiological time series data (starting from year 2023) into **clean 1-hour 3-channel time series data with no gaps**.

### The Gap-Filling Problem
TreeNet sensors collect high-frequency data (10-minute intervals) measuring:
- Air temperature, humidity
- Tree stem radius changes (dendrometer data)

This raw data contains **gaps** (missing measurements) due to:
- Sensor malfunctions
- Data transmission failures
- Maintenance periods
- Environmental damage

The pipeline trains a **deep learning model** to reconstruct missing data and produce clean, continuous time series.

### Long-Term Goal: Generate Ground Truth for 2024-2025

**CRITICAL**: There is **NO ground truth data** (target/cleaned data) for the following channels for years 2024 and 2025:
- **Temperature** (thermometer)
- **Relative humidity** (hygrometer)

The ultimate goal is to:
1. Train the model on historical data (2019-2023) where ground truth exists
2. Use the trained model to **generate clean temperature and humidity data** for 2024-2025
3. This generated data will serve as the new "ground truth" for those years

This means the model must learn to:
- Clean noisy sensor data
- Fill gaps in the raw input
- Produce reliable hourly outputs comparable to manually-cleaned historical data

### End-to-End Workflow
1. **Raw data** → 10-minute sensor measurements with gaps
2. **Segment building** → Extract 30-day overlapping windows
3. **Model training** → TCN encoder-decoder learns to reconstruct from context
4. **Gap filling** → Model fills missing data using surrounding valid data
5. **Output** → Clean 1-hour resolution time series (TWD, GRO, MDS)

---

## 1. Data Directory Structure

### Root Directories
```
/storage/lukovic/Data/FORWARDS/treenet/
├── server_data/           # Raw sensor and metadata files
│   ├── dendrometer_l2/    # Level 2 dendrometer data (clean)
│   ├── dendrometer_lm/    # Dendrometer LM model outputs (ground truth)
│   ├── hygrometer_l1/     # Level 1 hygrometer data
│   ├── thermometer_l1/    # Level 1 thermometer data  
│   ├── swp_l1/            # Soil water potential data
│   ├── meteo_data/        # Daily meteorological data (Swiss sites only)
│   └── metadata_*.pkl     # Various metadata pickle files
└── processed/             # All processed model data, experiments, logs, reports
```

### Storage Directory Convention

**CRITICAL: All outputs, logs, reports, and results should be stored in:**
```
/storage/lukovic/Data/FORWARDS/treenet/processed/
```

**Subdirectory Naming Convention:** `{country}_{scope}_{normalization}`

| Component | Options | Description |
|-----------|---------|-------------|
| `{country}` | `swiss`, `netherlands`, `all` | Country of data origin |
| `{scope}` | `full`, `subset`, `test` | Data scope (all sites vs limited) |
| `{normalization}` | `yearly_norm`, `segment_norm` | Normalization method used |

**Directory Internal Structure:**
```
{run_name}/
├── logs/              # Console logs, segment building logs
├── temp/              # Temporary/intermediate files during processing
├── model_data/        # Processed segments and intermediate time series
│   ├── intermediate_timeseries/
│   ├── segments/
│   └── reports/
└── experiments/       # Training experiments with timestamps
    └── YYYYMMDD_HHMMSS_{name}/
        ├── best_model.keras
        ├── final_model.keras
        ├── config.json
        ├── evaluation_metrics.json
        ├── training_history.csv
        └── tensorboard/
```

### Current Active Datasets (Cleaned 2026-01-11)
- **Yearly-norm**: `/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_yearly_norm_with_filter/`
- **Segment-norm**: `/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/`
- **Both datasets**: 25,517 segments (with data quality filter applied)

**Note**: All old test directories and partial combination datasets have been deleted. Only the two above remain.

### Ground Truth Data Availability

**IMPORTANT**: Ground truth (target/cleaned) data is NOT available for all years:

| Channel | 2019-2023 | 2024-2025 |
|---------|-----------|-----------|
| Temperature (thermometer) | ✅ LM data available | ❌ NO ground truth |
| Relative Humidity (hygrometer) | ✅ LM data available | ❌ NO ground truth |
| Stem Radius (dendrometer) | ✅ LM data available | ✅ LM data available |

**Implications for training:**
- Training data should primarily come from **2019-2023** where all ground truth exists
- Years 2024-2025 can be used for stem channel but NOT for temperature/humidity
- The trained model will be used to **generate** temperature/humidity ground truth for 2024-2025

### Default Paths
| Path Type | Location |
|-----------|----------|
| Raw data | `/storage/lukovic/Data/FORWARDS/treenet/server_data` |
| Meteo data | `/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data` |
| **Processed (all)** | `/storage/lukovic/Data/FORWARDS/treenet/processed/` |
| Pipeline code | `/home/lukovic/codes/treenetai/pipeline_v2` |
| **Visualizations** | `/home/lukovic/data/treenet/visualizations/` |

### Temporary & Data Storage Rules ⚠️ CRITICAL

**Temporary files directory:**
```
/home/lukovic/data/treenet/temp
```

**ABSOLUTE RULE - NEVER store images or data files in:**
```
/home/lukovic/codes/       ← NO images, NO data files, NO generated output
```
This directory tree is for **CODE ONLY** (Python scripts, configuration files, documentation).

**Default storage for temporary data and images:**
```
/home/lukovic/data/treenet/
├── temp/              # All temporary files
├── visualizations/    # Generated plots and images
└── <custom>/          # New subdirectories as needed
```

**Rationale:**
1. Keep code repositories clean and small for version control
2. Prevent accidental commits of large binary files
3. Separate concerns: code vs. generated artifacts
4. Centralized data location for backup and cleanup

---

## 2. File Naming Conventions

### CRITICAL: series_id vs site_id

| Identifier | What It Represents | Used In |
|------------|-------------------|---------|
| **series_id** | Individual sensor/instrument | Sensor data filenames |
| **site_id** | Measurement location (has multiple sensors) | Meteo filenames, metadata |

### Sensor Data Files (use `series_id`)
```
dendrometer_l2_series_id_{SERIES_ID}.ftr
dendrometer_lm_series_id_{SERIES_ID}.ftr
hygrometer_l1_series_id_{SERIES_ID}.ftr
thermometer_l1_series_id_{SERIES_ID}.ftr
swp_l1_series_id_{SERIES_ID}.ftr
```

### Meteo Data Files (use `site_id`)
```
meteo_data_site_id_{SITE_ID}.csv
```

---

## 3. Tree Growth Physics and Stem Radius Signal

### Important Physical Properties

**Stem Radius Growth Pattern:**
- The stem radius signal has a **non-decreasing long-term trend** across the year
- Trees accumulate new cells over time (cambial growth)
- Daily fluctuations exist due to water uptake and transpiration
- In the long term, the stem radius **increases** over the course of a year

**Yearly Pattern:**
- **Minimum stem radius**: Typically at the **beginning of the year** (after dormant winter period)
- **Maximum stem radius**: Typically towards the **end of the year** (after growing season)
- Short-term fluctuations can cause temporary decreases, but the overall trend is increasing

**Implication for Normalization:**
- The starting point (offset) of stem radius is physically irrelevant
- Any offset can be added without changing the physical meaning
- Year-level normalization should account for this growth pattern
- The min/max within a year reliably capture the annual range

---

## 4. Normalization Strategy

### The Problem (Discovered 2025-01-10)

The INPUT stem (from `dendrometer_l2`, raw 10-min data) and OUTPUT stem (from `dendrometer_lm`, cleaned hourly data) were being normalized with **DIFFERENT parameters** because they come from different data sources:

| Data | Source | Example Values |
|------|--------|----------------|
| INPUT stem | `dendrometer_l2` (raw 10-min) | min=4248, diff=14141 |
| OUTPUT stem | `dendrometer_lm` (LM-cleaned hourly) | min=-12, diff=29564 |

This made the learning task unnecessarily complex since the model had to implicitly learn scale transformations.

### Solution: Aligned Yearly Normalization (NORM_SCOPE="year")

When normalizing at year level, apply this procedure:

1. **Select each year** in the data
2. **Find first common valid timestamp**: Identify the earliest timestamp where BOTH input stem and output stem have valid (non-NaN) values
3. **Shift to zero**: Subtract the values at this common timestamp from both signals, so they both "start from zero"
4. **Find common time range**: Identify ALL timestamps where BOTH signals have valid data
5. **Compute normalization INDEPENDENTLY from common range**: Each signal uses its own min/max from the common time range

**Critical Insight (Updated 2025-01-10):**
- Normalization must be computed from the **COMMON TIME RANGE** only
- Each signal must be normalized **INDEPENDENTLY** (input uses input params, output uses output params)
- **CANNOT use output's params for input** because during inference, only input is available
- After shifting to zero at first common timestamp:
  - Min values will be similar (~0) for both signals
  - Max values (and diff) may differ slightly due to noise differences between L2 and LM
  - The difference should be small since they measure the same physical quantity (same tree)

**Example - Site 51 T640_H617_D667 (2023):**
- RAW DATA:
  - Input: 14,595 to 17,996 µm (Δ=3,400)
  - Output: 35,366 to 39,962 µm (Δ=4,596)
- Common time range: June 29 to Dec 31, 2023 (3,974 hourly timestamps)
- AFTER ALIGNMENT (independent normalization from common range):
  - Input normalization: min=-3.6, diff≈3,399
  - Output normalization: min=-4.0, diff≈3,399
  - **Min values match!** (both ~0 due to alignment)
  - **Diff values similar** (may vary slightly due to L2/LM noise differences)
  - Segment-level variation ratio: ~1.0 (target)

**Why this works:**
- Both signals start from the same reference point (zero at first common timestamp)
- Since min is the same (~0), any difference in diff represents actual signal variation differences
- Normalization params from common range ensure we don't include data outside valid segment regions
- Independent normalization allows inference with input-only data

**Implementation**: `Normalizer.align_stem_signals_yearly()` in `src/data/segmentation.py`
**Visualization**: `10_visualize_alignment.py` shows raw, aligned, and normalized data

### Per-Year Independent Normalization (Updated 2025-01-10)

**Key Change**: With `NORM_SCOPE="year"`, normalization is now computed **per-year independently** rather than globally across all years.

**Why this matters:**
- Some years may have data quality issues (e.g., sensor problems) that produce outliers
- Global normalization would allow one bad year to affect all other years
- Per-year normalization isolates each year's data

**Trade-off:**
- Segments that span December-January boundary cannot be extracted
- Loss of ~5-10% segments per site-year boundary
- Acceptable trade-off for data quality

**Implementation in `build_segments_for_combination()`:**
1. For each valid year in the data:
2. Compute year-specific normalization parameters
3. Normalize only that year's data
4. Extract segments within that year only
5. Store year-specific normalization params in metadata

### Normalization Scope Comparison (Updated 2026-01-11)

Two normalization strategies are available via the `--norm-scope` argument:

| Aspect | `year` (default) | `segment` |
|--------|------------------|-----------|
| **Range computed from** | Entire year's data | Each 30-day segment |
| **Cross-year segments** | ❌ Not allowed | ✅ Allowed (Dec→Jan) |
| **Normalized value range** | Portion of [0,1] for segments | Full [0,1] for each segment |
| **When to use** | Comparable scales across segments | Maximum local contrast |

**Example - Winter Temperature Segment:**
- Yearly-norm: Uses ~57% of normalized range (only winter temps)
- Segment-norm: Uses full 0-1 range (local min/max)

**Dataset Comparison (with data quality filter applied):**
| Metric | Yearly-Norm | Segment-Norm |
|--------|-------------|--------------|
| Total Segments | 25,517 | 25,517 |
| Cross-year Segments | 0 | 369 (1.47%) |
| Shared Segments | 22,727 | 22,727 |
| Unique Segments | 2,387 | 2,387 |

### Data Quality Filtering (Added 2025-01-10)

**Purpose**: Automatically detect and exclude years where input (L2) and output (LM) stem signals are incompatible due to sensor issues.

**Detection Method**: Compare the ratio of `output_range / input_range` for each year's stem signal.

| Ratio Range | Interpretation | Action |
|-------------|----------------|--------|
| < 0.5 | Input has outliers/spikes (L2 corrupted) | **FILTER** |
| 0.5 - 2.0 | Normal variance between L2 and LM | **KEEP** |
| > 2.0 | Output has issues or input too smooth | **FILTER** |

**Example - Site 51 T640_H619_D667:**
| Year | Input Range | Output Range | Ratio | Status |
|------|-------------|--------------|-------|--------|
| 2019 | 21,638 µm | 1,386 µm | 0.064 | ❌ FILTERED |
| 2020 | 1,948 µm | 1,948 µm | 1.00 | ✅ Valid |
| 2021 | 3,638 µm | 3,034 µm | 0.83 | ✅ Valid |
| 2022 | 3,067 µm | 3,067 µm | 1.00 | ✅ Valid |
| 2023 | 3,399 µm | 4,596 µm | 1.35 | ✅ Valid |
| 2024 | 2,232 µm | 2,222 µm | 1.00 | ✅ Valid |
| 2025 | 2,177 µm | 11 µm | 0.005 | ❌ FILTERED |

**Implementation**: `Normalizer.filter_valid_years()` and `Normalizer.check_stem_quality()` in `src/data/segmentation.py`

**Result after filtering:**
- Mean stem ratio: ~1.0 (ideal)
- Std: ~0.1 (very consistent)
- All segment ratios in range [0.85, 1.22]

### ⚠️ IMPORTANT: Filtered Years Are NOT Used

**Filtered years are COMPLETELY EXCLUDED from segment extraction.**

When a year fails the quality check (ratio outside [0.5, 2.0]):
1. That year is removed from `valid_years` list
2. NO segments are extracted from that year
3. The data exists but is not used for training

**Code reference** (`src/data/segmentation.py`, line ~744):
```python
# Apply data quality filtering - only keep years that passed quality check
if valid_years_quality is not None:
    valid_years = [y for y in valid_years if y in valid_years_quality]
```

**Rationale:**
- Bad quality years would corrupt model training
- Input spikes (ratio < 0.5) teach wrong patterns
- Output issues (ratio > 2.0) provide unreliable targets

### Filtering Logging & Visualization (Added 2026-01-11)

When segment building encounters years that fail data quality checks:

1. **Log File Entry**: Each filtered year is logged with full details:
   ```
   FILTERED: site3_T18_H20_D22 year 2021 - ratio=0.042, input_range=5234.2, output_range=219.8 - Ratio 0.042 < 0.5 (input has outliers/spikes)
   ```

2. **Plot Generation**: A 2-panel plot is saved showing:
   - Top panel: Input stem (L2) signal for the filtered year
   - Bottom panel: Output stem (LM) signal for the filtered year
   - Title shows combination ID, year, and reason for filtering

3. **Output Location**: `{run_dir}/logs/filtered_plots/filtered_{combo_str}_year{year}.png`

4. **Summary**: Total filtered years reported at end of segment building

### Segment-Level Normalization (NORM_SCOPE="segment") - CRITICAL

When using segment-level normalization:

**Key Principle:** Normalization must be computed **AFTER gap injection** using only values **OUTSIDE the gap region**.

**Rationale:**
- In real gap-filling scenarios, there will be a gap in the segment to fill
- We cannot know if the min/max values were in the gap region (now missing)
- Training must reflect this uncertainty
- Some segments may have reconstructed values **outside [0, 1]** - this is expected and correct

**Implementation:**
1. Segments are stored **raw (unnormalized)** when using `NORM_SCOPE="segment"`
2. During training, for each batch:
   a. Extract raw segment
   b. Inject gap (set gap region to NaN)
   c. Compute normalization params from NON-GAP region only
   d. Normalize the entire segment using these params
   e. Gap region remains NaN (for model to fill)
3. Predictions may exceed [0, 1] - this is physically valid
4. Implementation is in `SegmentNormalizer` class in `src/models/training.py`

---

## 5. Model Architecture

### Overview

The TreeNet AI gap-filling model is a **multi-task Temporal Convolutional Network (TCN)** that takes raw 10-minute sensor data with gaps and produces clean hourly predictions.

### Architecture Diagram

```
                    ┌─────────────────────────────────┐
                    │         TCN ENCODER             │
     INPUT          │  4 blocks, 64 filters/block     │
 [input_x, mask]    │  Dilations: 1, 2, 4, 8          │
 (4320, 22)         │  Shared feature extraction      │
        │           └─────────────────────────────────┘
        │                         │
        ├─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         │                         ▼
┌───────────────┐                 │            ┌───────────────────┐
│   BRANCH 1    │                 │            │     BRANCH 2      │
│  Recon Head   │                 │            │    Hourly Head    │
│ Conv1D → 11   │                 │            │ AvgPool(6) → 3    │
└───────────────┘                 │            └───────────────────┘
        │                         │                         │
        ▼                         │                         ▼
┌───────────────┐                 │            ┌───────────────────┐
│ recon_output  │                 │            │   hourly_output   │
│ (4320, 11)    │                 │            │    (720, 3)       │
│ 10-min recon  │                 │            │  Hourly cleaned   │
└───────────────┘                 │            └───────────────────┘
```

### Input Specification
- **Resolution**: 10-minute intervals
- **Window size**: 4320 timesteps = 30 days
- **Channels**: 11 (data) + 11 (mask) = 22 total

#### Channel Categories

##### Local Sensor Channels (0-2) - TARGET FOR GAP FILLING
| Channel | Variable | Source | Resolution | Notes |
|---------|----------|--------|------------|-------|
| 0 | temp_treenet | Local thermometer | 10-min | Below canopy temperature |
| 1 | rh_treenet | Local hygrometer | 10-min | Below canopy humidity |
| 2 | stem | Dendrometer | 10-min | Stem radius change |

**These are the ONLY channels where gaps are injected during training.**

##### Global Meteo Channels (3-10) - NEVER GAPPED
| Channel | Variable | Source | Resolution | Notes |
|---------|----------|--------|------------|-------|
| 3 | tas | MeteoSwiss | Daily | Mean air temperature |
| 4 | tasmax | MeteoSwiss | Daily | Max air temperature |
| 5 | tasmin | MeteoSwiss | Daily | Min air temperature |
| 6 | rh | MeteoSwiss | Daily | Relative humidity |
| 7 | vpd | MeteoSwiss | Daily | Vapor pressure deficit |
| 8 | gh | MeteoSwiss | Daily | Global horizontal irradiance |
| 9 | pr | MeteoSwiss | Daily | Precipitation |
| 10 | doy | Computed | N/A | Day of year (1-365) |

**These channels are NEVER gapped** because global meteo data is professionally maintained and gap-free. They serve as auxiliary information to help the model fill gaps.

### Output Specification
- **Resolution**: 1-hour intervals
- **Window size**: 720 timesteps = 30 days
- **Channels**: 3

| Channel | Variable | Description |
|---------|----------|-------------|
| 0 | TWD | Tree Water Deficit |
| 1 | GRO | Stem Growth |
| 2 | MDS | Maximum Daily Shrinkage |

### Model Details
- **Architecture**: TCN-based encoder-decoder with optional attention
- **Multi-task outputs**:
  - `recon_output`: 10-min reconstruction (4320×11)
  - `hourly_output`: 1-hour predictions (720×3)
- **Parameters**: ~98k trainable (base), ~130k with attention
- **Inputs**: `[input_x, input_mask]` where mask=1 for valid data

### Attention Mechanism (Optional)

The model can include MultiHeadAttention after the TCN encoder:
```python
# Config for attention
USE_ATTENTION = True
ATTENTION_HEADS = 8
ATTENTION_KEY_DIM = 64
```

---

## 6. Hyperparameters

All hyperparameters are centralized in `src/config.py`.

### Key Dataclasses

| Dataclass | Key Parameters | Description |
|-----------|----------------|-------------|
| `ModelConfig` | `n_filters=64`, `kernel_size=3`, `n_blocks=4`, `dropout_rate=0.2`, `batch_size=32`, `epochs=100`, `learning_rate=3e-4` | Model architecture & training |
| `GapConfig` | `min_gap_days=1`, `max_gap_days=12`, `min_gaps_per_segment=1`, `max_gaps_per_segment=3` | Gap injection for augmentation |
| `NormalizationConfig` | `method='minmax'`, `scope='year'` | Data normalization settings |
| `SegmentConfig` | `segment_days=30`, `stride_days=10` | Segment extraction settings |
| `SplitConfig` | `test_ratio=0.2`, `random_seed=42` | Train/test split |

### Hyperparameter Tuning Notes
- **batch_size=16**: Required on single GPU due to memory constraints (24GB RTX 3090)
- **batch_size=32**: Works with 2+ GPUs or mixed precision
- **n_filters**: Higher values (128) may improve accuracy but increase memory
- **n_blocks**: 4 blocks provides sufficient receptive field for 30-day segments

---

## 7. Experimental Results

### Current Best Results (January 2025)

**Target**: Stem MSE < 0.025

| Experiment | Config | Stem MSE | Stem MAE | Stem R² |
|------------|--------|----------|----------|---------|
| Baseline | 4 blocks, 64 filters | 0.0379 | 0.195 | 0.32 |
| Attention v1 | +8 heads, 64 key_dim | 0.0336 | 0.140 | 0.42 |
| **Attention v2** | 128 filters, 5 blocks | **0.0305** | 0.137 | 0.47 |
| Target | - | **<0.025** | - | - |

### Experiment Paths
```
/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/experiments/
├── 20260110_145342_with_attention/     # Attention v1
├── 20260110_153125_attention_v2/       # Attention v2 (current best)
```

---

## 8. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean Absolute Error | Average deviation from true value |
| **MSE** | Mean Squared Error | Penalizes large errors more |
| **R²** | Coefficient of Determination | Proportion of variance explained (1.0 = perfect) |

---

## 9. Site Coverage Summary

### Total Sites
- **162 total sites** in metadata
- **100 Swiss sites** (have valid meteo data from MeteoSwiss)
- **62 non-Swiss sites** (Netherlands: 60, Estonia: 1, Austria: 1)

### Sites with Complete Sensor Coverage
For training, a site needs ALL THREE sensor types:
1. `air temperature` (thermometer_l1)
2. `relative humidity` (hygrometer_l1)  
3. `tree stem radius change` (dendrometer_l2 + dendrometer_lm)

| Category | Count |
|----------|-------|
| All sites with 3 sensors | 63 |
| **Swiss sites with 3 sensors** | **52** (usable) |
| Non-Swiss with 3 sensors | 11 (unusable - no meteo) |

### Train/Test Split
- **Site-level split**: Entire sites are assigned to train OR test (no mixing)
- Default 80/20 ratio: 42 train sites, 10 test sites
- Random seed 42 for reproducibility

---

## 10. Project Structure

```
pipeline_v2/
├── src/                              # Source code (~2,500 lines)
│   ├── config.py                     # Configuration system
│   ├── utils.py                      # Utilities & logging
│   ├── data/                         # Data loading and processing
│   │   ├── loaders.py                # Data file loaders
│   │   ├── processors.py             # Preprocessing
│   │   ├── segmentation.py           # Segment extraction
│   │   └── validation.py             # Data validation
│   ├── gaps/                         # Gap injection
│   │   ├── gap_injection.py          # Gap augmentation
│   │   └── metrics.py                # Evaluation metrics
│   ├── models/                       # Model architecture
│   │   ├── tcn.py                    # TCN network
│   │   └── training.py               # Training pipeline
│   └── visualization/                # Plotting tools
│       ├── plot_segments.py
│       └── compare_raw.py
├── tests/                            # Test suite (110 tests)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_processors.py
│   ├── test_segmentation.py
│   └── test_gap_injection.py
├── 1_build_segments.py               # CLI: Build segments
├── 2_train_model.py                  # CLI: Train model
├── 3_evaluate.py                     # CLI: Evaluate
├── 4_visualize_segments.py           # CLI: Visualize segments
├── 5_compare_with_raw.py             # CLI: Compare with raw
├── 6_reconstruct_timeseries.py       # CLI: Gap-fill timeseries
├── 7_visualize_reconstruction.py     # CLI: Before/after plots
├── 8_visualize_predictions.py        # CLI: Predictions vs truth
├── requirements.txt                  # Dependencies
├── PROJECT_CONTEXT.md                # This file
└── README.md                         # Main documentation
```

---

## 11. Logging System

### Overview

The pipeline uses Python's `logging` module for structured, consistent logging across all scripts. Each script writes to both console and a dedicated log file.

### Log File Locations

| Script | Log File | Location |
|--------|----------|----------|
| `1_build_segments.py` | `build_segments.log` | `{output_dir}/logs/` |
| `2_train_model.py` | `training.log` | `{experiment_dir}/` |
| `3_evaluate.py` | `evaluation.log` | `{output_dir}/` |

### Logging Setup

The `setup_logging()` function in `src/utils.py` configures logging:

```python
from src.utils import setup_logging

# Returns a logger instance
log = setup_logging("my_process.log", name="my_module", verbose=True)

# Use the logger
log.info("Processing started")
log.warning("Missing data for site 5")
log.error("Failed to load file: %s", filename)
```

**Parameters:**
- `log_file`: Path to log file (created if doesn't exist)
- `name`: Logger name (default: 'treenet')
- `verbose`: If True, logs DEBUG level; if False, logs INFO level

### Log Format

```
2026-01-11 10:23:45 - treenet - INFO - Processing site 36...
2026-01-11 10:23:46 - treenet - WARNING - Missing hygrometer data for series 234
```

### Module-Level Logging

For modules imported by main scripts (e.g., `training.py`, `loaders.py`), use `get_logger()`:

```python
from src.utils import get_logger

logger = get_logger('treenet.loaders')
logger.warning("Skipping corrupted segment")
```

---

## 12. File Nomenclature & Data Storage

### Segment Data Formats

Segments are stored in two formats for different purposes:

| Format | File Pattern | Purpose | Contains |
|--------|--------------|---------|----------|
| **NumPy (Training)** | `{split}_numpy.pkl` | Model training | Arrays only: `input_segments`, `output_segments`, `masks` |
| **DataFrame (Inspection)** | `{split}_segments.pkl` | Human inspection | Full metadata + timestamps |

### Training Data Format (NumPy)

Location: `{output_dir}/model_data/segments/`

```python
# Load training data
with open("train_numpy.pkl", "rb") as f:
    data = pickle.load(f)

# Contents:
data['input_segments']   # Shape: (N, 4320, 11) - 10-min inputs
data['output_segments']  # Shape: (N, 720, 3)  - hourly outputs
data['masks']           # Shape: (N, 4320, 11) - valid data mask
```

### Inspection Data Format (DataFrame)

Location: `{output_dir}/model_data/segments/`

```python
# Load for inspection (includes timestamps)
df = pd.read_pickle("train_segments.pkl")

# Key columns:
# - segment_id: Unique identifier
# - site_id, series_id: Location info
# - year, start_date, end_date: Temporal bounds
# - input_segment: Array (4320, 11)
# - output_segment: Array (720, 3)
# - timestamps_input: DatetimeIndex (4320,)
# - timestamps_output: DatetimeIndex (720,)
# - norm_params: Dict with normalization parameters
```

### Intermediate Timeseries Files

Location: `{output_dir}/model_data/intermediate_timeseries/`

Pattern: `{split}_input_site{id}_T{t}_H{h}_D{d}.ftr`

Example: `train_input_site36_T334_H328_D276.ftr`

- **Format**: Feather (fast I/O)
- **Contents**: Merged timeseries with all channels before segmentation
- **Purpose**: Debugging, inspection, intermediate caching

### Reports & Metadata

Location: `{output_dir}/model_data/reports/`

| File | Format | Contents |
|------|--------|----------|
| `segment_building_report.json` | JSON | Summary statistics |
| `segment_metadata.pkl` | Pickle | Full segment metadata |
| `combo_metadata.csv` | CSV | Per-combination summary |
| `yearly_stats.csv` | CSV | Per-year statistics |

### Experiment Outputs

Location: `{output_dir}/experiments/{timestamp}_{name}/`

| File | Contents |
|------|----------|
| `best_model.keras` | Best checkpoint (by val_loss) |
| `final_model.keras` | Final epoch model |
| `config.json` | Training configuration |
| `training_history.csv` | Epoch-by-epoch metrics |
| `evaluation_metrics.json` | Test set evaluation |
| `training.log` | Training log file |
| `tensorboard/` | TensorBoard event files |

---

## 13. Pipeline Scripts

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `1_build_segments.py` | Extract 30-day segments | `--country`, `--run-name`, `--norm-scope` |
| `2_train_model.py` | Train TCN model | `--epochs`, `--batch-size`, `--data-dir` |
| `3_evaluate.py` | Evaluate performance | `--experiment-dir` |
| `4_visualize_segments.py` | Visualize segments | `--split`, `--n-samples` |
| `5_compare_with_raw.py` | Compare with raw data | |
| `6_reconstruct_timeseries.py` | **Gap-filling main script** | `--site-id`, `--model-path` |
| `7_visualize_reconstruction.py` | Before/after plots | `--site-id` |
| `8_visualize_predictions.py` | Model predictions vs truth | `--experiment-dir`, `--n-samples` |

### Quick Commands

```bash
# Activate environment
source /home/lukovic/pyenv/lamella/bin/activate

# Navigate to pipeline
cd /home/lukovic/codes/treenetai/pipeline_v2

# Build segments (all Swiss sites, yearly normalization)
python 1_build_segments.py --run-name swiss_full_yearly_norm --country Switzerland --max-combinations -1

# Train model (on GPU 1, batch size 16)
CUDA_VISIBLE_DEVICES=1 python 2_train_model.py \
    --data-dir /storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/model_data \
    --output-dir /storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/experiments \
    --epochs 100 --batch-size 16 --verbose

# Visualize predictions
python 8_visualize_predictions.py \
    --experiment-dir /storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/experiments/LATEST \
    --n-samples 10 --split test
```

---

## 14. Known Technical Issues & Design Decisions

### Issue 1: Hygrometer Sensor Drift

**Problem**: Hygrometer hardware causes signal drift over time:
1. **Baseline Drift**: Signal gradually shifts from true value
2. **Saturation Cap**: After some time, signal never reaches 100% RH

**Current Status**: Not yet addressed - requires model enhancement

### Issue 2: Segment Length Trade-offs

**Current Setting**: 30-day segments (4320 timesteps)

**Ideal Goal**: Year-long segments to capture seasonal patterns and long-term sensor drift

**Problem**: Too many gaps in raw data to compile year-long uninterrupted segments

**Strategy**: Start with 30-day, gradually extend once model demonstrates good gap-filling

### Issue 3: Timezone Handling

**Raw Data**: Timestamps in local timezone (`Europe/Zurich`)

**Solution**: All timestamps converted to **UTC** during processing to avoid DST issues

### Issue 4: Daily Meteo + UTC Alignment

**Problem**: Daily meteo data needs alignment with 10-minute UTC timestamps

**Solution**: Use "civil day" concept - convert UTC to local time, extract calendar date, match to meteo

### Issue 5: NFS Silent Write Failures

**Problem**: Feather file writes to NFS can silently fail

**Solution**: File verification after each write (check existence and size > 0)

---

## 15. Python Environment & Technology Stack

- **Python version**: 3.10.12
- **Virtual environment**: `/home/lukovic/pyenv/lamella/bin/python`
- **Activation**: `source /home/lukovic/pyenv/lamella/bin/activate`

### Core Dependencies
| Package | Purpose |
|---------|---------|
| TensorFlow/Keras | Deep learning framework |
| pandas | Time series manipulation |
| numpy | Numerical computing |
| feather-format | Fast file I/O |
| matplotlib | Visualization |

### Development Tools
| Tool | Purpose |
|------|---------|
| pytest | Unit testing |
| pytest-cov | Coverage reports |
| Black | Code formatting |
| Pylint | Linting |

---

## 16. GPU & Hardware

- **Available GPUs**: 6x NVIDIA RTX 3090 (24GB each)
- **Check availability**: `nvidia-smi`
- **Select GPU**: `CUDA_VISIBLE_DEVICES=1 python ...`

| Batch Size | Memory Used | Notes |
|------------|-------------|-------|
| 64 | ~22+ GB | OOM on single GPU |
| 32 | ~14 GB | Works on single GPU |
| 16 | ~8 GB | Safe for single GPU |

---

## 17. Gap Filling & Reconstruction Workflow

### Gap Analysis
Gaps in time series are identified as periods where timestamps differ by more than the expected interval (10-min for input, 1-hour for output):

| Gap Size | Action |
|----------|--------|
| ≤12 days | Fillable by model |
| >12 days | Marked for alternative handling |

### Reconstruction Process

1. **Gap Analysis** - Identify all gaps ≤12 days in the time series
2. **Segment Creation** - Create 30-day segments centered on each gap
3. **Model Inference** - Run model to predict filled values
4. **Gap Merging** - Merge predicted values back into original time series
5. **Output** - Save reconstructed time series

### Multi-Channel Gap Handling

| Scenario | Handling |
|----------|----------|
| All channels gapped | Model reconstructs all from context |
| Temperature only gapped | Model uses RH/stem context to fill temp |
| Stem only gapped | Model uses temp/RH/meteo context to fill stem |
| Meteo never gapped | Global meteo always available as anchor |

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **MAE** | < 0.15 | Mean absolute error on filled gaps |
| **Coverage** | > 95% | Percentage of gaps successfully filled |
| **Stem MSE** | < 0.025 | Primary target for stem channel |
| **Speed** | < 1 min/site | Reconstruction throughput |

---

## 18. Metadata Files

| File | Description | Key Columns |
|------|-------------|-------------|
| `metadata_all.pkl` | Master metadata | site_id, series_id, country |
| `metadata_data_dendro_l2.pkl` | Dendrometer L2 | series_id, site_id |
| `metadata_data_dendro_lm.pkl` | Dendrometer LM (ground truth) | series_id, site_id |
| `metadata_data_all_l1_humidity.pkl` | Hygrometer | series_id, site_id |
| `metadata_data_all_l1_temperature.pkl` | Thermometer | series_id, site_id |

---

## 19. Project Statistics & Implementation Status

### Code Base
- **Total Lines:** ~7,000+
  - Python code: ~2,500 lines
  - Documentation: ~4,500 lines
  - Tests: ~1,000 lines (110 tests)

### Core Modules
| Module | Location | Purpose |
|--------|----------|---------|
| config | `src/config.py` | Configuration dataclasses |
| loaders | `src/data/loaders.py` | Data loading utilities |
| processors | `src/data/processors.py` | Data preprocessing |
| segmentation | `src/data/segmentation.py` | Segment extraction |
| gap_injection | `src/gaps/gap_injection.py` | Gap augmentation |
| tcn | `src/models/tcn.py` | TCN architecture |
| training | `src/models/training.py` | Training pipeline |
| metrics | `src/gaps/metrics.py` | Evaluation metrics |
| visualization | `src/visualization/` | Plotting tools |

### CLI Tools (5 main + 3 additional)
| Script | Status |
|--------|--------|
| `1_build_segments.py` | ✅ Complete |
| `2_train_model.py` | ✅ Complete |
| `3_evaluate.py` | ✅ Complete |
| `4_visualize_segments.py` | ✅ Complete |
| `5_compare_with_raw.py` | ✅ Complete |
| `6_reconstruct_timeseries.py` | ✅ Complete |
| `7_visualize_reconstruction.py` | ✅ Complete |
| `8_visualize_predictions.py` | ✅ Complete |

### Testing
- ✅ Configuration tests (40 tests)
- ✅ Processing tests (25 tests)
- ✅ Segmentation tests (20 tests)
- ✅ Gap injection tests (25 tests)
- ❌ Model tests (TODO)
- ❌ Training tests (TODO)
- **Current coverage:** ~50% of codebase (110 tests)

### Performance Benchmarks
| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Segment extraction | ~5-10 min | 1 year, 10 sites |
| Training (100 epochs) | ~2-3 hours | Single GPU RTX 3090 |
| Inference | Real-time | Hourly predictions |
| Visualization | ~1 sec | Per segment plot |

### Memory Usage
| Component | Memory | Notes |
|-----------|--------|-------|
| Segment storage | ~500 MB | Per year (pickle) |
| Training | ~4-6 GB | batch_size=32 |
| Training | ~8 GB | batch_size=16 (safe) |

---

## 20. Session Continuity Checklist

When starting a new session, verify:
1. ☐ Python environment: `/home/lukovic/pyenv/lamella/bin/python`
2. ☐ Data paths exist and are accessible
3. ☐ GPU available if training
4. ☐ Review this PROJECT_CONTEXT.md for context
5. ☐ Check current dataset paths in Section 1

---

## 21. Development Priorities

### Current Focus: Stem MSE < 0.025

**Approaches being tested:**
1. ✅ Attention mechanism (8 heads, 64 key_dim) - improved to 0.0305
2. ✅ Larger model (128 filters, 5 blocks) - improved to 0.0305
3. 🔄 **Aligned stem normalization** - IN PROGRESS
4. ⏳ Segment-level gap-aware normalization

**Next Steps:**
1. Rebuild segments with aligned stem normalization
2. Train model with corrected normalization
3. Evaluate improvement in stem MSE

### Visualization Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `4_visualize_segments.py` | View raw segments | `/home/lukovic/data/treenet/visualizations/` |
| `7_visualize_reconstruction.py` | Gap-filling results | `/home/lukovic/data/treenet/visualizations/` |
| `8_visualize_predictions.py` | Model predictions | `/home/lukovic/data/treenet/visualizations/` |

---

## 22. TODO / Future Tasks

### High Priority

1. **☐ Data Quality Filter: Recover Filtered Years**
   - **Issue**: Years failing quality check (ratio outside [0.5, 2.0]) are completely excluded
   - **Impact**: ~124 year-segments worth of data not used for training
   - **Proposed approach**:
     - Develop a cleaning/preprocessing step specifically for filtered years
     - Apply spike detection and removal to L2 input data
     - Re-align L2 with LM after cleaning
     - Re-run quality check to validate cleaned years
   - **Location of filtered year info**: `{run_dir}/logs/filtered_plots/`
   - **Status**: Logging and visualization implemented (2026-01-11)

2. **☐ Achieve stem MSE < 0.025**
   - Current best: ~0.0305
   - Approaches to try: aligned normalization, larger models, attention tuning

### Medium Priority

3. **☐ Generate ground truth for 2024-2025**
   - Use trained model to produce clean temperature/humidity data
   - Validate against available LM stem data for those years

4. **☐ Implement segment-level gap-aware normalization**
   - Normalize AFTER gap injection using only values outside gap region

### Low Priority / Research

5. **☐ Investigate outlier detection methods**
   - Automated spike detection in L2 data
   - Statistical vs. ML-based approaches

6. **☐ Cross-validation study**
   - Evaluate model generalization across different sites/years

---

**Maintained by**: TreeNet AI Pipeline v2 Development
