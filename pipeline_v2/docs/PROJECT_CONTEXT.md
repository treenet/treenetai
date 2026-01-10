# TreeNet AI Pipeline v2 - Project Context

**Purpose**: This file documents project-specific context details to help maintain continuity across sessions. Reference this document at the start of any new session.

**Last Updated**: 2025-01-11

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

### Current Active Dataset
- **Path**: `/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/`
- **Description**: Full Swiss dataset (52 sites), yearly min-max normalization
- **Segments**: 27,435 total (26,996 train + 439 test)

### Default Paths
| Path Type | Location |
|-----------|----------|
| Raw data | `/storage/lukovic/Data/FORWARDS/treenet/server_data` |
| Meteo data | `/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data` |
| **Processed (all)** | `/storage/lukovic/Data/FORWARDS/treenet/processed/` |
| **Current dataset** | `/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/` |
| Pipeline code | `/home/lukovic/codes/treenetai/pipeline_v2` |
| **Visualizations** | `/home/lukovic/data/treenet/visualizations/` |

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
4. **Find last common valid timestamp**: Find the timestamp closest to year-end where both signals have valid values
5. **Extract aligned interval**: Use only the data between first and last common timestamps
6. **Normalize independently**: Even with independent normalization, parameters will be similar because both signals start from zero
7. **Repeat for each year**: Apply this procedure to each available year

**Why this works:**
- Both signals start from the same reference point (zero at first common timestamp)
- The normalization ranges will be similar since they measure the same physical quantity
- Preserves year-level consistency for temporal patterns

**Implementation**: `Normalizer.align_stem_signals_yearly()` in `src/data/segmentation.py`

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

## 10. Pipeline Scripts

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

## 11. Known Technical Issues & Design Decisions

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

## 12. Python Environment

- **Python version**: 3.10.12
- **Virtual environment**: `/home/lukovic/pyenv/lamella/bin/python`
- **Activation**: `source /home/lukovic/pyenv/lamella/bin/activate`

### Key Packages
- TensorFlow/Keras (deep learning)
- pandas, numpy (data manipulation)
- feather-format (fast file I/O)
- matplotlib (visualization)

---

## 13. GPU & Hardware

- **Available GPUs**: 6x NVIDIA RTX 3090 (24GB each)
- **Check availability**: `nvidia-smi`
- **Select GPU**: `CUDA_VISIBLE_DEVICES=1 python ...`

| Batch Size | Memory Used | Notes |
|------------|-------------|-------|
| 64 | ~22+ GB | OOM on single GPU |
| 32 | ~14 GB | Works on single GPU |
| 16 | ~8 GB | Safe for single GPU |

---

## 14. Metadata Files

| File | Description | Key Columns |
|------|-------------|-------------|
| `metadata_all.pkl` | Master metadata | site_id, series_id, country |
| `metadata_data_dendro_l2.pkl` | Dendrometer L2 | series_id, site_id |
| `metadata_data_dendro_lm.pkl` | Dendrometer LM (ground truth) | series_id, site_id |
| `metadata_data_all_l1_humidity.pkl` | Hygrometer | series_id, site_id |
| `metadata_data_all_l1_temperature.pkl` | Thermometer | series_id, site_id |

---

## 15. Session Continuity Checklist

When starting a new session, verify:
1. ☐ Python environment: `/home/lukovic/pyenv/lamella/bin/python`
2. ☐ Data paths exist and are accessible
3. ☐ GPU available if training
4. ☐ Review this PROJECT_CONTEXT.md for context
5. ☐ Check current dataset: `/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/`

---

## 16. Development Priorities

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

**Maintained by**: TreeNet AI Pipeline v2 Development
