# TreeNet AI Pipeline v2 - Project Context

**Purpose**: This file documents project-specific context details to help maintain continuity across sessions. Reference this document at the start of any new session.

**Last Updated**: 2026-01-12 (Gap types documentation, constrained RH visualization)

---

## ⚠️ CRITICAL INSTRUCTION FOR AI ASSISTANTS

**ALWAYS UPDATE THIS FILE** with any new information discovered during conversations:
- New findings, limitations, or insights about the data or model
- New scripts created or significant modifications to existing scripts
- New output directories or file naming conventions
- Solutions to problems encountered
- Important technical details that would help future sessions

This ensures continuity across sessions and prevents rediscovery of the same issues.

---

## Change Log

### 2026-01-12 - Comprehensive Test Set Evaluation (2020-2021 with Stem Alignment)
- **Re-evaluated with stem alignment** for proper scale calibration
- **Years 2020-2021** selected as optimal (260,106 hours vs 175,387 for 2021-2022)
- **Summary Metrics (WITH Stem Alignment)**:
  | Channel | Mean Correlation | Mean MAE | Mean R² |
  |---------|------------------|----------|---------|
  | Temperature | 0.914 | 2.25°C | 0.83 |
  | Relative Humidity | 0.791 | 8.42% | 0.57 |
  | Stem | 0.800 | 25.8 μm | 0.75 |
- **CRITICAL**: Stem alignment is REQUIRED for multi-year reconstruction evaluation
  - Without alignment: R² = -2857 (severely negative due to scale mismatch)
  - With alignment: R² = 0.75 (proper evaluation)
- **New/Modified scripts**:
  - `batch_evaluate_test_set.py` - Now accepts `--years` argument and auto-applies stem alignment
  - `visualize_boxplots.py` - Box-and-whisker plots for metric comparison
- **Output directories**:
  - `/home/lukovic/data/treenet/test_set_evaluation_unconstrained_2020_2021_aligned/` - Current results
  - `/home/lukovic/data/treenet/test_set_evaluation_unconstrained_2021_2022/` - Prior results (no alignment)
- **Boxplots generated**: `boxplot_correlation_*.png`, `boxplot_mae_*.png`, `boxplot_r2_*.png`, `evaluation_comparison_*.png`

### 2026-01-12 - Batch Test Set Evaluation (Unconstrained Model, NO alignment)
- **Evaluated unconstrained model on ALL 20 test combinations** for years 2021-2022
- **4 combinations failed** (site86_*_D925): insufficient data for 30-day windows
- **16 combinations successfully evaluated** with full metrics and visualizations
- **Summary Metrics (WITHOUT Stem Alignment - for comparison)**:
  | Channel | Mean Correlation | Mean MAE | Mean R² |
  |---------|------------------|----------|---------|
  | Temperature | 0.855 | 2.94°C | 0.70 |
  | Relative Humidity | 0.757 | 8.90% | 0.50 |
  | Stem | 0.743 | 1782 μm | -2857 (scale issue) |
- **Key observation**: Stem R² is severely negative due to scale mismatch (operational denorm)
  - Stem needs scale alignment via `--align-stem` flag or LM-based denormalization
- **New script**: `batch_evaluate_test_set.py` - Comprehensive batch evaluation
- **Output directory**: `/home/lukovic/data/treenet/test_set_evaluation_unconstrained_2021_2022/`
  - 16 × stacked visualizations (`stacked_with_gaps_*.png`)
  - 16 × reconstruction feather files (`reconstructed_*.ftr`)
  - `evaluation_metrics.json` - Full per-combination metrics
  - `evaluation_summary.txt` - Aggregated summary

### 2026-01-12 - Constrained RH Model Evaluation & Gap Documentation
- **Constrained RH model trained**: penalty_weight=0.1, 13 epochs fine-tuning from unconstrained model
- **Model path**: `/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260112_190347_constrained_rh_v2_finetune_constrained_rh/best_model.keras`
- **RH constraint comparison** (403 test segments, 290,160 predictions):
  - Below-0 violations: 61 → 37 (**39% reduction**)
  - Above-1 violations: 0 → 2 (minor increase)
  - MAE: 0.0625 → 0.0639 (2.3% trade-off)
- **New scripts created**:
  - `compare_rh_constraint.py` - Quick RH comparison
  - `compare_rh_detailed.py` - Comprehensive RH analysis
  - `compare_reconstruction_stacked.py` - Side-by-side reconstruction comparison
  - `visualize_stacked_with_gaps_constrained.py` - 9-row stacked visualization with gap shading
- **Modified scripts**:
  - `6_reconstruct_timeseries.py`: Added `.ftr` extension support, `constrained_hourly_loss`, string combo-ids
  - `src/models/tcn.py`: Added `constrained_hourly_loss` to `TCNModel.load()`
- **Documented gap types**: Missing timestamps vs NaN values (critical for gap detection)
- **Output files**:
  - `/home/lukovic/data/treenet/rh_constraint_comparison/` - Comparison visualizations
  - `/home/lukovic/data/treenet/reconstructions_constrained_site22/` - Constrained reconstructions

### 2026-01-12 - Code Cleanup and Script Unification
- **Merged training scripts**: `2_train_model.py` now supports `--fine-tune`, `--constrain-rh`, `--use-attention` flags
- **Merged reconstruction scripts**: `6_reconstruct_timeseries.py` now supports `--align-stem` flag
- **Merged visualization scripts**: `12_visualize_gap_filling.py` now supports `--denorm {normalized,ideal,operational}` flag
- **Archived superseded files**: Moved 6 scripts to `archive/superseded/`
- **Generated visualizations**:
  - Denormalized (ideal/LM params): `/home/lukovic/data/treenet/gap_filling_visualization_denorm/`
  - Operational (input params): `/home/lukovic/data/treenet/gap_filling_visualization_operational/`
- **Documented critical limitation**: LM parameters not available at inference time (see Section 22)

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

### Train/Test (Holdout) Set Design

**Site-based split**: The raw data is collected from various TreeNet monitoring sites across Switzerland. These sites are split into two disjoint sets:

| Set | Purpose | Description |
|-----|---------|-------------|
| **Training Set** | Model learning | Sites used to train the model. The model learns patterns from these sites' data. |
| **Test (Holdout) Set** | Generalization evaluation | Sites **never seen during training**. Used to evaluate how well the model generalizes to completely new sites. |

**Why this matters:**
- The model must reconstruct sensor data for **any** TreeNet site, not just the ones it trained on
- A site-based holdout ensures we test **true generalization** - can the model handle a new tree/sensor/location?
- If we only split by time (e.g., train on 2019-2022, test on 2023), we might overfit to site-specific patterns
- Holdout sites test whether the model learned **general** tree physiological patterns vs. **site-specific** quirks

**Current test sites** (as of 2026-01-12):
- Site 22 (Dendrometers: 120, 121)
- Site 72 (Dendrometers: 849, 850)
- Site 86 (Dendrometers: 911, 922, 925, 937)

None of these 8 site/dendrometer combinations appear in training data.

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

### Gap-Filling Visualization Outputs (Updated 2026-01-12)

| Directory | Description | Generated by |
|-----------|-------------|--------------|
| `/home/lukovic/data/treenet/gap_filling_visualization/` | Normalized [0,1] figures | `12_visualize_gap_filling.py --denorm normalized` |
| `/home/lukovic/data/treenet/gap_filling_visualization_denorm/` | Denormalized with LM params (ideal/reference) | `12_visualize_gap_filling.py --denorm ideal` |
| `/home/lukovic/data/treenet/gap_filling_visualization_operational/` | Denormalized with input params (realistic) | `12_visualize_gap_filling.py --denorm operational` |

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

### Gap Injection Details (IMPORTANT)

**How gaps are distributed across channels:**

When `n_gaps=2`, **2 separate gap regions** are injected per segment. For **each gap**, a channel is **randomly selected** from the gappable channels [T, RH, Stem].

**Key behavior (line 154 in `gap_injection.py`):**
```python
for _ in range(n_gaps):  # For each gap
    channel = self.rng.choice(channels_to_gap)  # Randomly pick a channel
```

**This means:**
- **Most common case:** Each gap goes to a different channel (2 channels with 1 gap each)
- **Less common case:** Both gaps go to the same channel (1 channel with 2 disjoint gaps)

| Outcome | Probability | Example |
|---------|-------------|---------|
| 2 different channels | ~67% | T=7d gap, RH=7d gap, Stem=0d |
| Same channel (2 gaps) | ~33% | T=0d, RH=7d gap + 7d gap, Stem=0d |

**This is intentional** - the model needs to learn to handle various gap scenarios including:
- Single gap in one channel
- Multiple disjoint gaps in the same channel  
- Overlapping gaps (rare, due to random placement)

**Typical gap coverage per gapped channel (with `n_gaps=2`, `gap_days=12`):**
- Single gap: 40% of 30-day segment missing (12/30 days)
- Two gaps in same channel: Up to 80% missing (24/30 days) - rare

**Gappable channels:** Only local sensor channels (0, 1, 2) can have gaps. Global meteo channels (3-10) are NEVER gapped.

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
├── 20260110_153125_attention_v2/       # Attention v2 (current best - yearly norm)
```

### Current Best Model (Segment-Norm with Attention)
```
/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/
└── 20260111_152352_segment_norm_attention/
    └── best_model.keras                # Currently used for gap-filling visualizations
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
├── archive/                          # Old/superseded code
│   ├── superseded/                   # Scripts merged into unified versions (2026-01-12)
│   │   ├── 12_visualize_gap_filling.py    # → merged into new 12_visualize_gap_filling.py
│   │   ├── 14_reconstruct_from_intermediate.py  # → merged into 6_reconstruct_timeseries.py
│   │   ├── 15_reconstruct_with_alignment.py     # → merged into 6_reconstruct_timeseries.py
│   │   ├── 16_train_constrained.py             # → merged into 2_train_model.py
│   │   ├── 17_visualize_gap_filling_denorm.py  # → merged into 12_visualize_gap_filling.py
│   │   ├── 18_visualize_gap_filling_operational.py  # → merged into 12_visualize_gap_filling.py
│   │   ├── 2_train_model.py                    # Original training script
│   │   └── 6_reconstruct_timeseries.py         # Original reconstruction script
│   └── ...                           # Other archived code
├── 1_build_segments.py               # CLI: Build segments
├── 2_train_model.py                  # CLI: Train model (UNIFIED: --fine-tune, --constrain-rh)
├── 3_evaluate.py                     # CLI: Evaluate (3 modes)
├── 4_visualize_segments.py           # CLI: Visualize segments
├── 5_compare_with_raw.py             # CLI: Compare with raw
├── 6_reconstruct_timeseries.py       # CLI: Gap-fill (UNIFIED: --align-stem)
├── 7_visualize_reconstruction.py     # CLI: Before/after plots
├── 8_visualize_predictions.py        # CLI: Predictions vs truth
├── 10_plot_gap_evaluation.py         # CLI: Gap evaluation box plots
├── 11_visualize_gap_injection.py     # CLI: Gap injection visualization
├── 12_visualize_gap_filling.py       # CLI: Gap filling viz (UNIFIED: --denorm)
├── 13_batch_reconstruct.py           # CLI: Batch reconstruction
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

### Core Pipeline Scripts (Unified - 2026-01-12)

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `1_build_segments.py` | Extract 30-day segments | `--country`, `--run-name`, `--norm-scope` |
| `2_train_model.py` | **Unified training script** | `--fine-tune`, `--constrain-rh`, `--use-attention` |
| `3_evaluate.py` | **Unified evaluation script** | `--mode`, `--model-path`, `--recon-path` |
| `4_visualize_segments.py` | Visualize segments | `--split`, `--n-samples` |
| `5_compare_with_raw.py` | Compare with raw data | |
| `6_reconstruct_timeseries.py` | **Unified reconstruction** | `--align-stem`, `--site-id`, `--model-path` |
| `7_visualize_reconstruction.py` | Before/after plots | `--site-id` |
| `8_visualize_predictions.py` | Model predictions vs truth | `--experiment-dir`, `--n-samples` |
| `10_plot_gap_evaluation.py` | Gap-filling box plots | `--gap-days`, `--model-path` |
| `11_visualize_gap_injection.py` | Visualize gap injection | `--gap-days`, `--n-gaps` |
| `12_visualize_gap_filling.py` | **Unified visualization** | `--denorm {normalized,ideal,operational}` |
| `13_batch_reconstruct.py` | Batch reconstruction | `--sites`, `--model-path` |

### 2_train_model.py - Unified Training Script (Updated 2026-01-12)

This script supports multiple training modes via flags:

| Flag | Description | Use Case |
|------|-------------|----------|
| `--use-attention` | Enable attention mechanism | Better temporal modeling |
| `--fine-tune` | Load existing model and continue training | Resume from checkpoint |
| `--constrain-rh` | Constrain RH output to [0%, 100%] | Physical plausibility |

**Example Usage:**
```bash
# Standard training with attention
python 2_train_model.py --use-attention

# Fine-tune existing model with RH constraints
python 2_train_model.py --fine-tune --constrain-rh --model-path experiments/best_model.keras
```

### 6_reconstruct_timeseries.py - Unified Reconstruction Script (Updated 2026-01-12)

Supports optional stem alignment for correct absolute scale:

| Flag | Description | Use Case |
|------|-------------|----------|
| `--align-stem` | Compute scale/offset from Nov-Dec overlap | Correct absolute stem values |

**Why alignment is needed:**
- Model outputs are in normalized space
- Stem normalization uses different scales for L2 (input) vs LM (output)
- Alignment uses known overlap period to compute optimal transformation

### 12_visualize_gap_filling.py - Unified Visualization Script (Updated 2026-01-12)

Supports three denormalization modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `normalized` | Keep [0,1] normalized values | Relative comparisons |
| `ideal` | Use LM params (output_min/diff) | **Reference only** - not available in production |
| `operational` | Use input params (input_min/diff) | **Realistic deployment** - what we'll have in practice |

**Example Usage:**
```bash
# Generate normalized figures (default)
python 12_visualize_gap_filling.py --denorm normalized

# Generate operational figures (realistic scenario)
python 12_visualize_gap_filling.py --denorm operational
```

**⚠️ IMPORTANT:** The `ideal` mode uses LM parameters which are NOT available during real gap-filling (that's what we're trying to generate!). Use `operational` mode to see realistic performance.

### Archived/Superseded Scripts (in `archive/superseded/`)

The following scripts were merged into unified versions:
- `16_train_constrained.py` → merged into `2_train_model.py` (`--constrain-rh`)
- `14_reconstruct_from_intermediate.py` → merged into `6_reconstruct_timeseries.py`
- `15_reconstruct_with_alignment.py` → merged into `6_reconstruct_timeseries.py` (`--align-stem`)
- `17_visualize_gap_filling_denorm.py` → merged into `12_visualize_gap_filling.py` (`--denorm ideal`)
- `18_visualize_gap_filling_operational.py` → merged into `12_visualize_gap_filling.py` (`--denorm operational`)

### 3_evaluate.py - Unified Evaluation Script (Details)

This script provides **three evaluation modes** via the `--mode` flag:

| Mode | Description | Required Arguments |
|------|-------------|-------------------|
| `segments` | Evaluate on 30-day test segments | `--model-path`, `--data-dir` |
| `reconstruction` | Compare full reconstruction vs LM data | `--recon-path`, `--lm-path` or `--lm-dir` |
| `synthetic-gaps` | Inject synthetic gaps and evaluate filling | `--model-path`, `--gap-days` |

**Example Usage:**

```bash
# Mode 1: Evaluate on test segments (30-day windows)
python 3_evaluate.py --mode segments \
    --model-path experiments/best_model.keras \
    --data-dir /path/to/model_data

# Mode 2: Evaluate full reconstructed time series vs LM data
python 3_evaluate.py --mode reconstruction \
    --recon-path reconstructions/site22.ftr \
    --lm-path site22_lm.ftr

# Mode 3: Synthetic gap injection evaluation
python 3_evaluate.py --mode synthetic-gaps \
    --model-path experiments/best_model.keras \
    --gap-days 1 7 12
```

**Outputs:**
- `segments`: `general_evaluation_results.json`
- `reconstruction`: `reconstruction_evaluation_results.json`
- `synthetic-gaps`: `synthetic_gap_evaluation_results.json`

### 10_plot_gap_evaluation.py - Gap Evaluation Plots

Generates **box-and-whisker plots** showing gap-filling performance across channels.

**Key Arguments:**
- `--gap-days`: List of gap lengths to evaluate (e.g., `1 7 12`)
- `--n-gaps`: Number of channels with gaps per segment (default: 2)
- `--model-path`: Path to trained model

**Output Files:** (saved to `--output-dir`)
- `gap_evaluation_error_30d_segments_{gaps}d.png` - Absolute error distribution
- `gap_evaluation_mse_30d_segments_{gaps}d.png` - MSE distribution
- `gap_evaluation_correlation_30d_segments_{gaps}d.png` - Correlation distribution

**Title Format:** "(N × 30-day test segments, 2 channels with gaps)"

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

### Bash Wrapper Scripts (Updated 2026-01-12)

Located in `scripts/` directory. These provide a user-friendly interface with configurable settings.

| Script | Python Target | Purpose |
|--------|---------------|---------|
| `1_build_segments.sh` | `1_build_segments.py` | Build training segments |
| `2_train_model.sh` | `2_train_model.py` | Train model (NEW: fine-tune, constrain-rh) |
| `3_evaluate_model.sh` | `3_evaluate.py` | Evaluate model |
| `4_visualize_segments.sh` | `4_visualize_segments.py` | Visualize segments |
| `6_reconstruct_timeseries.sh` | `6_reconstruct_timeseries.py` | Gap-fill (NEW: align-stem) |
| `7_visualize_reconstruction.sh` | `7_visualize_reconstruction.py` | Before/after plots |
| `8_visualize_predictions.sh` | `8_visualize_predictions.py` | Predictions vs truth |
| `12_visualize_gap_filling.sh` | `12_visualize_gap_filling.py` | Gap-filling viz (NEW: denorm modes) |
| `run_pipeline.sh` | Multiple | Full pipeline execution |

**How to use bash scripts:**
1. Edit the USER CONFIGURATION section at the top of the script
2. Run with `./scripts/<script>.sh`
3. Preview command with `./scripts/<script>.sh --dry-run`

**Example - Training with new flags:**
```bash
# Edit scripts/2_train_model.sh:
FINE_TUNE="true"
MODEL_PATH="/path/to/best_model.keras"
CONSTRAIN_RH="true"

# Then run:
./scripts/2_train_model.sh
```

**Example - Visualization with denormalization:**
```bash
# Edit scripts/12_visualize_gap_filling.sh:
DENORM_MODE="operational"  # Options: normalized, ideal, operational

# Then run:
./scripts/12_visualize_gap_filling.sh
```

---

## 14. Complete Pipeline Workflow

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TreeNet AI Gap-Filling Pipeline                       │
└─────────────────────────────────────────────────────────────────────────────┘

   RAW DATA                  PROCESSED DATA                 MODEL OUTPUTS
┌──────────────┐          ┌─────────────────┐          ┌─────────────────────┐
│ server_data/ │          │   model_data/   │          │    experiments/     │
│  ├─dendro_l2/│   ──►    │  ├─segments/    │   ──►    │  └─YYYYMMDD_HHMMSS/ │
│  ├─thermo_l1/│          │  ├─intermediate/│          │     ├─best_model    │
│  ├─hygro_l1/ │          │  └─reports/     │          │     ├─config.json   │
│  └─meteo/    │          └─────────────────┘          │     └─metrics.json  │
└──────────────┘                                       └─────────────────────┘
       │                          │                              │
       │ 1_build_segments.py      │ 2_train_model.py            │
       └──────────────────────────┴──────────────────────────────┘
                                                                 │
                                                    3_evaluate.py │
                                                                 │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GAP-FILLING                                     │
└─────────────────────────────────────────────────────────────────────────────┘
       │                                                         │
       │ 6_reconstruct_timeseries.py (--align-stem)             │
       │                                                         │
       ▼                                                         │
┌─────────────────┐          ┌─────────────────────────────────────────────┐
│  reconstructed/ │   ──►    │ 12_visualize_gap_filling.py (--denorm MODE) │
│  └─site*.ftr    │          └─────────────────────────────────────────────┘
└─────────────────┘                           │
                                              ▼
                              ┌───────────────────────────────────┐
                              │ gap_filling_visualization[_MODE]/ │
                              │   └─*.png figures                 │
                              └───────────────────────────────────┘
```

### Stage-by-Stage Workflow

**Stage 1: Build Segments**
```bash
./scripts/1_build_segments.sh
# or: python 1_build_segments.py --run-name swiss_segment_norm_all_combos --norm-scope segment
```
- Input: Raw sensor data from `server_data/`
- Output: Training segments in `model_data/segments/`
- Time: ~30-60 minutes for all Swiss sites

**Stage 2: Train Model**
```bash
./scripts/2_train_model.sh
# or: python 2_train_model.py --use-attention --data-dir .../model_data
```
- Input: Segments from Stage 1
- Output: Trained model in `experiments/YYYYMMDD_HHMMSS/`
- Time: ~2-4 hours (100 epochs, single GPU)

**Stage 3: Evaluate Model**
```bash
./scripts/3_evaluate_model.sh
# or: python 3_evaluate.py --mode segments --model-path .../best_model.keras
```
- Input: Trained model + test segments
- Output: Metrics JSON in experiment directory

**Stage 4: Reconstruct Time Series (Gap-Filling)**
```bash
./scripts/6_reconstruct_timeseries.sh
# or: python 6_reconstruct_timeseries.py --align-stem --site-id 22
```
- Input: Trained model + intermediate time series
- Output: Gap-filled `.ftr` files in `reconstructed/`
- NEW: `--align-stem` flag for correct absolute scale

**Stage 5: Visualize Results**
```bash
./scripts/12_visualize_gap_filling.sh
# or: python 12_visualize_gap_filling.py --denorm operational --n-samples 10
```
- Input: Trained model + test segments
- Output: PNG figures in visualization directories
- NEW: `--denorm` modes for different output scales

### Denormalization Modes Explained

When visualizing or reconstructing, the model outputs normalized values in [0,1]. Three modes are available:

| Mode | Output Directory | Y-axis Labels | Use Case |
|------|------------------|---------------|----------|
| `normalized` | `gap_filling_visualization/` | [0,1] range | Comparing relative patterns |
| `ideal` | `gap_filling_visualization_denorm/` | Physical units (°C, %, μm) | **Reference only** - requires LM data |
| `operational` | `gap_filling_visualization_operational/` | Physical units (approx.) | **Production use** - uses input params |

**⚠️ Critical**: In production deployment (real gap-filling):
- LM (ground truth) data is NOT available - that's what we're generating!
- Only `operational` mode is realistic for deployed models
- T and RH: Small error (~1-2°C, ~2-5% RH) - acceptable
- Stem: Larger error (~2-3× scale) - use `--align-stem` for correction

### ⚠️ CRITICAL: Stem Alignment for Multi-Year Reconstruction (Added 2026-01-12)

**THE PROBLEM**: When reconstructing multi-year time series using operational denormalization, the stem channel suffers from severe scale mismatch because each 30-day segment uses its own min/max for normalization.

**THE EVIDENCE**:
- Without alignment: Stem R² = -2857 (severely negative, predictions worse than mean)
- With alignment: Stem R² = 0.75 (excellent - proper evaluation)

**THE SOLUTION**: **ALWAYS use stem alignment** when:
1. Evaluating reconstruction quality vs LM ground truth
2. Generating multi-year continuous time series
3. Comparing reconstruction across different sensor combinations

**How Alignment Works**:
```python
# Linear regression: LM_stem = slope * recon_stem + intercept
from scipy import stats
slope, intercept, r_value, _, _ = stats.linregress(recon_values, lm_values)
aligned_stem = recon_stem * slope + intercept
```

**Scripts with Stem Alignment**:
- `batch_evaluate_test_set.py`: Automatic alignment when LM data available
- `6_reconstruct_timeseries.py`: Use `--align-stem` flag

**Note**: T and RH don't require alignment because their physical scales are consistent.

---

## 15. Known Technical Issues & Design Decisions

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

## 16. Python Environment & Technology Stack

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

## 17. GPU & Hardware

- **Available GPUs**: 6x NVIDIA RTX 3090 (24GB each)
- **Check availability**: `nvidia-smi`
- **Select GPU**: `CUDA_VISIBLE_DEVICES=1 python ...`

| Batch Size | Memory Used | Notes |
|------------|-------------|-------|
| 64 | ~22+ GB | OOM on single GPU |
| 32 | ~14 GB | Works on single GPU |
| 16 | ~8 GB | Safe for single GPU |

---

## 18. Gap Filling & Reconstruction Workflow

### Gap Analysis
Gaps in time series are identified as periods where timestamps differ by more than the expected interval (10-min for input, 1-hour for output):

| Gap Size | Action |
|----------|--------|
| ≤12 days | Fillable by model |
| >12 days | Marked for alternative handling |

### High-Level Reconstruction Process

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

### Detailed Multi-Year Reconstruction: Step-by-Step (Updated 2026-01-12)

The reconstruction process uses a **sliding window approach** to process long time series (2+ years) through a model trained on 30-day segments.

#### Phase 1: Data Loading

**Input**: Intermediate timeseries file (`.feather`)
- Contains merged data: `temp_treenet`, `rh_treenet`, `stem`, `tas`, `tasmax`, `tasmin`, `rh`, `vpd`, `gh`, `pr`, `doy`
- 10-minute resolution (6 samples per hour)
- Continuous time index

```python
df = pd.read_feather(intermediate_file)
df_range = df[(df['ts'] >= year_start) & (df['ts'] < year_end)]
```

#### Phase 2: Sliding Window Setup

**Parameters**:
- **Window size**: 30 days = 4,320 input samples (10-min resolution)
- **Stride**: 24 hours = 144 samples (default)
- **Result**: Significant overlap between consecutive windows

**Example for 2-year reconstruction**:
```
Year range: 2021-01-01 to 2022-12-31 (730 days)
Input samples: 730 × 24 × 6 = 105,120 samples
Windows: (105,120 - 4,320) / 144 + 1 ≈ 701 windows
```

#### Phase 3: Per-Window Processing (Repeated for Each Window)

**Step 3.1: Extract Window**
```
Window i starts at: i × stride_samples (= i × 144)
Window i ends at:   i × stride_samples + 4,320
```

**Step 3.2: Segment-Level Normalization**

⚠️ **CRITICAL**: Each window is normalized **independently** using its own min/max:

```python
for each channel in [temp, rh, stem, meteo...]:
    min_val = np.nanmin(window[:, channel])
    max_val = np.nanmax(window[:, channel])
    window_norm[:, channel] = (window[:, channel] - min_val) / (max_val - min_val)
    
    # Store for later denormalization
    norm_params[channel] = {'min': min_val, 'max': max_val}
```

**Step 3.3: Create Input Mask**
```python
mask = (~np.isnan(window)).astype(float)  # 1 where valid, 0 where NaN
input_array = np.nan_to_num(window_norm, nan=0.0)  # Replace NaN with 0
```

**Step 3.4: Model Prediction**
```python
predictions = model.predict([input_array, mask])
# predictions[1] = hourly_output: shape (720, 3) = 30 days × 24 hours × 3 channels
```

**Step 3.5: Denormalization (Using Input Parameters)**

Since LM parameters not available at inference:
```python
for channel in [T, RH, stem]:
    input_min = norm_params[input_channel]['min']
    input_diff = norm_params[input_channel]['max'] - input_min
    output_denorm[:, channel] = pred[:, channel] * input_diff + input_min
```

**Step 3.6: Assign Timestamps**
```python
segment_start = window.timestamp[0]
hourly_times = pd.date_range(start=segment_start, periods=720, freq='1H')
```

#### Phase 4: Combine Overlapping Predictions

After processing all windows, most timestamps have **multiple predictions**:

```
Timestamp           | Window 1 | Window 2 | Window 3 | ...
2021-01-01 00:00    | pred_1   |   -      |    -     |
2021-01-01 01:00    | pred_1   |   -      |    -     |
...
2021-01-02 00:00    | pred_1   | pred_2   |    -     |  ← Overlap begins
2021-01-02 01:00    | pred_1   | pred_2   |    -     |
...
2021-01-03 00:00    | pred_1   | pred_2   | pred_3   |  ← More overlap
```

**Aggregation**: Simple **averaging** of overlapping predictions:
```python
reconstructed = all_predictions.groupby('ts').agg({
    'recon_T': 'mean',
    'recon_RH': 'mean',
    'recon_stem': 'mean'
}).reset_index()
```

#### Phase 5: Stem Scale Alignment (Optional, --align-stem)

**Problem**: Denormalization using input params gives wrong absolute stem scale.

**Solution**: Use Nov-Dec overlap period to compute scale correction.

**Step 5.1: Extended Reconstruction Start**
```
Requested: 2021-2022
Actual reconstruction: Nov 2020 - Dec 2022 (starts 2 months early)
```

**Step 5.2: Extract Alignment Period**
```python
recon_align = reconstructed[Nov 2020 : Jan 2021]  # Reconstructed
lm_align = df_lm[Nov 2020 : Jan 2021]  # Ground truth LM
```

**Step 5.3: Compute Linear Transformation**
```python
# Linear regression: LM = scale × recon + offset
slope, intercept, r, _, _ = linregress(recon_align['stem'], lm_align['stem'])
```

**Step 5.4: Apply Correction to All Data**
```python
reconstructed['stem'] = reconstructed['stem'] * slope + intercept
```

#### Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECONSTRUCTION FLOW                           │
└─────────────────────────────────────────────────────────────────┘

RAW INPUT (2 years, 10-min resolution):
|████████████████████████████████████████████████████████████████|
0                                                              105,120

SLIDING WINDOWS (30-day each, 24-hour stride):
|▓▓▓▓▓▓▓▓▓▓▓|                                  Window 1
    |▓▓▓▓▓▓▓▓▓▓▓|                              Window 2
        |▓▓▓▓▓▓▓▓▓▓▓|                          Window 3
            |▓▓▓▓▓▓▓▓▓▓▓|                      Window 4
                ...
                                          |▓▓▓▓▓▓▓▓▓▓▓| Window 701

PER-WINDOW PROCESSING:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Segment Normalize│ → │ Model Predict    │ → │ Denormalize      │
│ (local min/max)  │    │ (4320→720)       │    │ (input params)   │
└──────────────────┘    └──────────────────┘    └──────────────────┘

OVERLAP AVERAGING:
│pred_1│pred_2│pred_3│pred_4│...
    │pred_2│pred_3│pred_4│pred_5│...
        │pred_3│pred_4│pred_5│pred_6│...
═══════════════════════════════════════════
        FINAL = mean(overlapping predictions)

OUTPUT (2 years, 1-hour resolution):
|████████████████████████████████████████████████████████████████|
0                                                              17,520
```

#### Reconstruction Computational Cost

| Time Range | Windows | Predictions | Approximate Time |
|------------|---------|-------------|------------------|
| 1 year | ~350 | ~350 | ~30 seconds |
| 2 years | ~700 | ~700 | ~60 seconds |
| 5 years | ~1,750 | ~1,750 | ~2-3 minutes |

Times assume GPU inference with batch size 1 per window.

---

## 19. Metadata Files

| File | Description | Key Columns |
|------|-------------|-------------|
| `metadata_all.pkl` | Master metadata | site_id, series_id, country |
| `metadata_data_dendro_l2.pkl` | Dendrometer L2 | series_id, site_id |
| `metadata_data_dendro_lm.pkl` | Dendrometer LM (ground truth) | series_id, site_id |
| `metadata_data_all_l1_humidity.pkl` | Hygrometer | series_id, site_id |
| `metadata_data_all_l1_temperature.pkl` | Thermometer | series_id, site_id |

---

## 20. Project Statistics & Implementation Status

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

## 21. Session Continuity Checklist

When starting a new session, verify:
1. ☐ Python environment: `/home/lukovic/pyenv/lamella/bin/python`
2. ☐ Data paths exist and are accessible
3. ☐ GPU available if training
4. ☐ Review this PROJECT_CONTEXT.md for context
5. ☐ Check current dataset paths in Section 1

---

## 22. Development Priorities

### ✅ COMPLETED: Stem MSE < 0.025 (2026-01-11)

**Final Result:** MSE = 0.00078, R² = 0.9886 (far exceeds target!)

**Successful Configuration:**
- Segment-level normalization
- Attention mechanism (8 heads, 64 key_dim)
- 128 filters, 5 TCN blocks
- Dataset: 25,517 segments (swiss_segment_norm_all_combos)

**Full Results:**
| Channel | MAE | R² | MSE |
|---------|-----|----|----|
| local_T | 0.043 | 0.9342 | - |
| local_RH | 0.063 | 0.8985 | - |
| stem | 0.022 | 0.9886 | 0.00078 |

### Current Focus: PATH 1 Denormalization

**Issue:** Model outputs in relative [0,1] scale. Stem values cannot be directly converted to absolute units without LM reference.

**Next Steps:**
1. Test reconstruction on additional sites
2. Evaluate whether relative output is acceptable for downstream analysis
3. Consider PATH 2 approach for absolute stem values

### Visualization Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `4_visualize_segments.py` | View raw segments | `/home/lukovic/data/treenet/visualizations/` |
| `7_visualize_reconstruction.py` | Gap-filling results | `/home/lukovic/data/treenet/visualizations/` |
| `8_visualize_predictions.py` | Model predictions | `/home/lukovic/data/treenet/visualizations/` |

---

## 23. Signal Reconstruction Strategy (Added 2026-01-11)

### Problem Statement

The ultimate goal is to reconstruct raw sensor signals to produce:
1. **Clean, gap-free L1/L2 data** (10-min temperature, humidity, dendrometer)
2. **Clean LM-quality hourly data** for all channels

### Path Analysis

Two main approaches have been identified:

#### PATH 1: Direct L2→LM Reconstruction (Current Implementation)

**Description:**
Use the existing trained model to directly convert raw L2/L1 input to LM-quality output.

```
Input:  L1/L2 10-min multi-channel (11 channels) with gaps ≤12 days
Model:  TCN + Attention (current)
Output: LM 1-hour 3-channel (temp, humidity, stem) cleaned
```

**Pros:**
- ✅ Model already trained and performing well (val_loss ~0.006)
- ✅ Single inference step
- ✅ Direct path to clean hourly data

**Cons:**
- ❌ Cannot handle gaps >12 days without iteration
- ❌ Input resolution lost (10-min → 1-hour)
- ❌ If input quality is poor, output may also be poor

**Improvement Ideas:**
1. Train with larger gaps (up to 15-20 days)
2. Train with clean data (no gaps) to also learn signal cleaning
3. Iterative gap-filling for very large gaps

---

#### PATH 2: Staged Gap-Filling + Cleaning (Proposed)

**Description:**
Split the problem into separate stages with specialized models.

**Stage A: L1/L2 Gap Filling (10-min → 10-min)**
```
Input:  L1/L2 10-min multi-channel (11 channels) with gaps
Model:  TCN_GapFill_L2
Output: L1/L2 10-min multi-channel (3 channels: temp, humidity, stem) without gaps
```
- Train on L2 segments where gaps are artificially injected
- Target = original L2 values before gap injection
- Does NOT clean signal, just fills gaps

**Stage B: LM Gap Filling (1-hour → 1-hour)**
```
Input:  LM 1-hour 3-channel with gaps
Model:  TCN_GapFill_LM
Output: LM 1-hour 3-channel without gaps
```
- Similar to Stage A but for LM data
- Useful for recovering LM ground truth where gaps exist

**Stage C: L2→LM Cleaning (Yearly)**
```
Input:  Complete L2 10-min data (from Stage A) for full year
Model:  TCN_Clean_Yearly
Output: Clean LM 1-hour data for full year
```
- Works on longer segments (yearly)
- Focuses on signal cleaning, not gap-filling
- Input has no gaps (filled in Stage A)

**Pros:**
- ✅ Each model specialized for one task
- ✅ Can handle unlimited gap sizes (by filling first)
- ✅ Preserves 10-min resolution where needed
- ✅ Modular: can improve individual stages

**Cons:**
- ❌ More complex pipeline (3 models)
- ❌ More training data needed for each model
- ❌ Error propagation between stages
- ❌ 2024-2025: No LM ground truth for temp/humidity (Stage C difficult)

---

### Recommended Strategy: Hybrid Approach

**Phase 1: Start with PATH 1 (Current)**
1. Use trained model for L2→LM reconstruction on gaps ≤12 days
2. Evaluate reconstruction quality
3. If quality is good, iterate for larger gaps

**Phase 2: Develop PATH 2 Stage A (L2 Gap-Filling)**
1. Create L2→L2 gap-filling model (same resolution)
2. Train on artificially gapped L2 data
3. Use to fill gaps before L2→LM conversion

**Phase 3: Combine**
1. Stage A: Fill L2 gaps (any size)
2. PATH 1: Convert gap-free L2 to LM-quality output
3. Result: Complete, clean LM-quality time series

### Implementation Priority

| Priority | Task | Model | Status |
|----------|------|-------|--------|
| 1 | PATH 1 reconstruction | Current | ✅ Tested (2026-01-11) |
| 2 | Stage A: L2→L2 gap-fill | New | ⏳ Pending |
| 3 | Stage B: LM→LM gap-fill | New | ⏳ Future |
| 4 | Stage C: Yearly cleaning | New | ⏳ Research |

### PATH 1 Denormalization Challenge (2026-01-11)

**Problem Identified:**
During training with segment-level normalization, BOTH input and output are normalized to [0,1] using each segment's OWN min/max values. This means:

1. **Input normalization**: Uses L2 min/max (e.g., stem: -85,519 to 18,390)
2. **Output normalization**: Uses LM min/max (e.g., stem: -13 to 29,552)

These are **independent** normalizations. The model learns to map from normalized input space to normalized output space, but we lose the absolute scale information.

**Consequences:**
- When reconstructing without LM data (the whole point of PATH 1), we cannot perfectly denormalize the output
- For Temperature and RH, the L2 and LM scales are similar, so using input params works OK
- For Stem, L2 and LM have very different scales (ratio ~0.38), causing large absolute errors

**Validation Results (Site 3, Segment-Norm+Attention Model):**

| Channel | MAE (absolute) | Correlation | Normalized MAE |
|---------|---------------|-------------|----------------|
| local_T | 0.67°C | **0.993** | 0.018 |
| local_RH | 3.2% | **0.966** | 0.042 |
| stem | 11,649 µm | 0.475 | 1.56 |

**Key Observations:**
1. Temperature and RH work excellently (>96% correlation)
2. Stem has moderate temporal correlation but wrong absolute scale
3. The model captures patterns correctly, but denormalization is the issue

**Output Modes in `6_reconstruct_timeseries_v2.py`:**
- `--output-mode normalized`: Keep output in [0,1] range (relative values)
- `--output-mode input_scale`: Use input normalization params (approximation)

### CRITICAL: Operational Denormalization Limitation (2026-01-12)

**⚠️ FUNDAMENTAL PROBLEM: LM Constants Not Available at Inference Time**

When the model is deployed for real gap-filling (its intended purpose), we will **NOT have access to LM data** - because that's exactly what we're trying to generate! This creates a fundamental mismatch:

| Training | Inference (Real Use) |
|----------|---------------------|
| Model learns: normalized_input → normalized_LM_output | We have: normalized_input → normalized_output |
| Output denorm: uses LM min/max | **We cannot use LM min/max** (not available) |
| Result: Perfect denormalization | **CANNOT properly denormalize** |

**What This Means:**
1. The model outputs are in **LM-normalized space** [0,1]
2. To convert to physical units, we'd need `output_min` and `output_diff` from LM data
3. **But we don't have LM data** - if we did, we wouldn't need gap-filling!

**Segment Metadata Structure:**
```python
# For each segment, we have:
segment_meta.input_min    # e.g., {'temp_treenet': -14.56, 'stem': 6863}
segment_meta.input_diff   # e.g., {'temp_treenet': 23.41, 'stem': 380}
segment_meta.output_min   # e.g., {'local_T': -14.33, 'stem': 2638}  # ← FROM LM!
segment_meta.output_diff  # e.g., {'local_T': 23.33, 'stem': 375}    # ← FROM LM!
```

**Impact by Channel:**
| Channel | Input (L1/L2) Range | Output (LM) Range | Difference | Impact |
|---------|---------------------|-------------------|------------|--------|
| Temperature | ~[-20°C, +35°C] | ~[-20°C, +35°C] | Small | ✅ Using input params is OK |
| RH | ~[20%, 100%] | ~[15%, 100%] | Small | ✅ Using input params is OK |
| Stem | ~[6000, 7500 μm] | ~[2500, 3500 μm] | **LARGE** | ❌ Input params give WRONG scale |

**Why Stem Is Different:**
- L2 dendrometer data has a different baseline than LM processed data
- LM processing applies significant corrections/calibrations
- The **ratio** between L2 and LM stem ranges is typically ~0.3-0.5
- Using input params for denormalization gives values ~2-3× too large

**Practical Solutions:**

1. **For T and RH (RECOMMENDED):**
   - Use input normalization parameters for denormalization
   - Error is small (~1-2°C or ~2-5% RH)
   - Acceptable for most applications

2. **For Stem (OPTIONS):**
   - **Option A**: Keep output normalized [0,1] = relative changes only
   - **Option B**: Use scale alignment from known overlap period (current `15_reconstruct_with_alignment.py` approach)
   - **Option C**: Train a different model architecture that learns absolute values
   - **Option D**: Accept ~2-3× error in absolute scale, patterns are correct

**This is why `15_reconstruct_with_alignment.py` exists:**
- It uses Nov-Dec overlap period (where LM data exists) to compute optimal scale+offset
- Then applies that transformation to full reconstruction
- Result: Correct absolute scale after alignment

**CONCLUSION:** The segment-norm approach is fundamentally limited for producing absolute-scale outputs without reference data. This is acceptable for T/RH but problematic for stem. **Consider global normalization** if absolute outputs are critical.

### Stem Evaluation: Alignment Procedure (2026-01-11)

**IMPORTANT**: For stem (dendrometer) data, the absolute scale is **arbitrary**. The baseline shifts over time due to sensor resets, calibration, and environmental factors. Therefore:

**Correct Evaluation Procedure for Stem:**
1. Reconstruct the time series normally
2. **Align/shift** the reconstructed stem to minimize MAE with ground truth
3. The optimal shift is: `shift = median(LM_stem - recon_stem)`
4. Compute correlation and MAE **after alignment**

**Why This Matters:**
- A correlation of 0.999 with MAE of 4000 µm doesn't mean poor reconstruction
- After optimal shift, MAE of 30 µm (0.5% of range) is achievable
- The **pattern** is what matters, not the absolute baseline

**Test Site Results (Held-out Sites, NOT in Training):**

| Test Site | T Corr | RH Corr | Stem Corr | Stem MAE (aligned) | Stem Normalized MAE |
|-----------|--------|---------|-----------|-------------------|---------------------|
| Site 22 (D=120) | 0.983 | 0.927 | **0.999** | 30 µm | **0.54%** ✅ |
| Site 86 (D=911) | 0.966 | 0.849 | 0.250 | 2014 µm | 156% ⚠️ |

**Key Findings:**
- Site 22: Excellent reconstruction across all channels
- Site 86: T and RH good, but stem has issues (likely data quality problem in source data)

**Visualization Files (Test Sites):**
- `/home/lukovic/data/treenet/visualizations/reconstructions/test_sites/`

---

## 24. Gap Tracking and Per-Channel Visualization

### Per-Channel Gap Tracking (Added 2026-01-11)

The reconstruction script (`6_reconstruct_timeseries_v2.py`) now tracks gaps **per-channel** independently:

**Output Columns:**
| Column | Description |
|--------|-------------|
| `is_gap` | Overall gap (from merged input analysis) |
| `is_gap_T` | Temperature sensor (L1 thermometer) gaps |
| `is_gap_RH` | Humidity sensor (L1 hygrometer) gaps |
| `is_gap_stem` | Stem sensor (L2 dendrometer) gaps |

**How gaps are identified:**
- For each hourly output timestamp, check if the underlying raw sensor has data within that hour
- If no sensor samples exist within that hour → marked as gap for that channel
- Each channel is checked independently against its own sensor source

**Important: These are INPUT gaps (L1/L2), NOT output (LM) gaps!**
- Grey shading in visualizations indicates where the model is **inferring/gap-filling** values
- Rather than translating actual measurements

### Sensor Co-Location and Gap Patterns

**Why gaps may appear identical across channels:**

At TreeNet sites, sensors are typically on the **same data logger/station**:
- Thermometer (temperature)
- Hygrometer (humidity)
- Dendrometer (stem)

**When the data logger fails** (power loss, connectivity issues, etc.), **all sensors go down together**.

**Example - Site 86, Year 2019:**
| Timestamp Range | T Gap | RH Gap | Stem Gap | Cause |
|-----------------|-------|--------|----------|-------|
| 2019-07-16 13:00-14:00 | ✓ | ✓ | ✓ | System outage |
| 2019-12-30 23:00 to 2019-12-31 22:00 | ✓ | ✓ | ✓ | System outage |

These identical patterns are **NOT a bug** - they reflect real system-wide outages.

**However**, sensors CAN have different gap patterns when:
- Individual sensor failures (sensor-specific problems)
- Sensor maintenance/replacement
- Different operational start/end dates

**Example - Site 86, Year 2020 (different patterns):**
| Metric | Temperature | Humidity | Stem |
|--------|-------------|----------|------|
| Gap hours | 69 | 67 | **25** |
| T+RH but not Stem | 42 hours | - | - |

In 2020, there are **42 hours** where T and RH have gaps but Stem does NOT.

### Visualization Script (v3)

**Script:** `7_visualize_reconstruction_v3.py`

**Features:**
- **450 DPI** (triple resolution for publication quality)
- **Per-channel gap shading**: Each subplot shows gaps specific to that channel
- **Colors**: Blue (LM ground truth), Red (reconstruction), Grey (gap regions)
- **Stem alignment**: Applies optimal vertical shift for visual comparison
- **Statistics**: Correlation and MAE displayed on each subplot

**Usage:**
```bash
python 7_visualize_reconstruction_v3.py \
    --recon-path <reconstructed_file.ftr> \
    --site-id <site> \
    --years 2019 2020 \
    --output-dir <output_directory>
```

---

## 24.5. Test Site Data Quality Analysis (Added 2026-01-12)

### Train/Test Split Design

**IMPORTANT**: All test site/dendrometer combinations are **TEST-ONLY** (holdout):
- None of the 8 test dendrometers appear in the training set
- This is by design for proper generalization evaluation
- The model has never seen these specific sensors during training

**Test Combinations:**
| Site | Dendrometer | Training Status |
|------|-------------|-----------------|
| 22 | 120 | TEST ONLY |
| 22 | 121 | TEST ONLY |
| 72 | 849 | TEST ONLY |
| 72 | 850 | TEST ONLY |
| 86 | 911 | TEST ONLY |
| 86 | 922 | TEST ONLY |
| 86 | 925 | TEST ONLY |
| 86 | 937 | TEST ONLY |

### L2 Data Discontinuities in Test Dendrometers

**Analysis performed**: Checked for month-to-month mean value jumps > 1000 µm

| Site | Dendro | Data Range | Value Range (µm) | Discontinuities |
|------|--------|------------|------------------|-----------------|
| 22 | 120 | 2014-03 to 2025-06 | 0 - 13,140 | **2** (Jun-Jul 2017) |
| 22 | 121 | 2014-03 to 2025-06 | -3,211 - 9,752 | **2** (Jul 2017, Jan 2020) |
| 72 | 849 | 2014-04 to 2025-06 | 1,760 - 11,110 | **1** (Jan 2022: 9,279 µm jump!) |
| 72 | 850 | 2014-04 to 2025-06 | 4,109 - 17,358 | **4** (Jun 2015, May 2019, Jan 2020, Jan 2022) |
| **86** | **911** | 2016-11 to 2022-08 | 1,465 - 9,983 | **3** (Jan 2018, Jul 2018, **Jan 2019: 8,373 µm jump!**) |
| 86 | 922 | 2017-02 to 2025-06 | 3,503 - 5,966 | **0** ✓ |
| 86 | 925 | 2010-10 to 2016-09 | 3,448 - 5,003 | **0** ✓ |
| 86 | 937 | 2016-11 to 2025-06 | 8,374 - 15,119 | **3** (Jan 2022, Jan 2025, May 2025) |

### Root Cause: Site 86 D=911 January 2019 Reconstruction Issue

**Problem observed**: Poor stem reconstruction at beginning of January 2019

**Investigation findings**:
1. L2 data has a **massive discontinuity in December 2018**:
   - Before: ~1,500 µm
   - After: ~9,900 µm
   - Jump magnitude: **8,373 µm**
2. LM (ground truth) is stable at ~5,800 µm throughout
3. This is a **sensor reset/calibration change**, not real tree growth
4. Segments starting January 1, 2019 see L2 values at ~9,900 µm but LM targets at ~5,800 µm
5. The model hasn't been trained on this dendrometer (test-only) so cannot learn this offset

**Implication**: The model performs well on sites with continuous L2 data (e.g., Site 22 D=120 for recent years, Site 86 D=922), but struggles with sensor resets that create L2↔LM mismatches.

### Recommendations

1. **Focus evaluation on stable dendrometers**: D=922 and D=925 (Site 86) have no discontinuities
2. **Document known discontinuities**: Use this analysis to interpret reconstruction quality
3. **Consider preprocessing**: For production use, implement discontinuity detection and handling
4. **Acknowledge limitation**: Model trained on L2→LM transformation assumes consistent sensor calibration

---

## 24.6. Gap-Filling Performance Evaluation (Added 2026-01-12)

### Evaluation Methodology

**Synthetic Gap Injection**: Since we have paired input/output data (30-day segments with no gaps), we evaluate gap-filling by:
1. Taking clean test segments (input and target)
2. Injecting synthetic gaps into input data (first 3 channels only)
3. Passing gapped input through the model
4. Comparing model output to original target
5. Computing metrics for gap regions only vs. non-gap regions

**Script**: `9_evaluate_synthetic_gaps.py`

**Parameters for main evaluation**:
- Test segments: 403 (all)
- Gap length: 7 days
- Gaps per segment: 2
- Gap percentage: ~15%

### Results: 7-Day Gaps (All 403 Test Segments)

#### Entire 30-Day Segments
| Channel | MAE | RMSE | Correlation | R² | N samples |
|---------|-----|------|-------------|-----|-----------|
| local_T | 0.047 | 0.062 | **0.959** | 0.919 | 290,160 |
| local_RH | 0.068 | 0.091 | **0.938** | 0.880 | 290,160 |
| stem | 0.032 | 0.052 | **0.981** | 0.960 | 290,160 |

#### Gap Regions ONLY (True Gap-Filling Performance)
| Channel | MAE | RMSE | Correlation | R² | N samples |
|---------|-----|------|-------------|-----|-----------|
| local_T | 0.064 | 0.082 | **0.926** | 0.855 | 43,285 |
| local_RH | 0.091 | 0.118 | **0.895** | 0.800 | 43,793 |
| stem | 0.091 | 0.117 | **0.898** | 0.806 | 43,198 |

#### Non-Gap Regions (Comparison Baseline)
| Channel | MAE | RMSE | Correlation | R² | N samples |
|---------|-----|------|-------------|-----|-----------|
| local_T | 0.044 | 0.058 | **0.965** | 0.930 | 246,875 |
| local_RH | 0.064 | 0.085 | **0.946** | 0.894 | 246,367 |
| stem | 0.022 | 0.028 | **0.995** | 0.988 | 246,962 |

### Key Findings

1. **Gap-filling works well**: Correlation remains >0.89 for all channels even in gap regions
2. **Temperature is best**: Gap correlation 0.926 (only 4% drop from non-gap)
3. **Stem has largest gap penalty**: Gap correlation drops from 0.995 to 0.898 (~10% drop)
4. **All channels meet quality threshold**: Correlation >0.8 for gap regions indicates reliable gap-filling

**Note**: Values are normalized [0,1], so MAE=0.064 means ~6.4% of normalized range error.

### Performance Degradation by Region

| Channel | Non-Gap Corr | Gap Corr | Degradation |
|---------|--------------|----------|-------------|
| local_T | 0.965 | 0.926 | -4.0% |
| local_RH | 0.946 | 0.895 | -5.4% |
| stem | 0.995 | 0.898 | -9.7% |

Stem channel shows largest degradation in gaps, likely because:
1. Dendrometer data has more complex temporal dynamics
2. Model relies more heavily on local context for stem reconstruction
3. Temperature and humidity correlate better with COSMO reanalysis (auxiliary input)

### Interpretation

The model can fill **7-day gaps** (14% of segment) with approximately:
- **93% correlation** for temperature
- **90% correlation** for humidity
- **90% correlation** for stem radius

This demonstrates the model has learned meaningful patterns from surrounding context and auxiliary meteorological data to reconstruct missing local sensor measurements.

---

## 24.7. Production Reconstruction with Scale Alignment (Added 2026-01-12)

### The Scale Alignment Problem

When reconstructing time series from raw L1/L2 data, the **stem channel** has a scale mismatch:
- Model outputs are in the scale of the INPUT (L1/L2 raw dendrometer)
- Ground truth (LM) has a different scale/baseline
- Even with perfect correlation, R² can be negative due to scale mismatch

**Root cause**: Segment-wise normalization uses INPUT data's min/max, but different sensor calibrations and processing between L1/L2 and LM create different baselines.

### Solution: Nov-Dec Prior Year Alignment

**Strategy**: Use overlapping data from November-December of the year BEFORE the requested period to calibrate scale.

**Algorithm**:
1. Start reconstruction from **Nov 1 of (year_start - 1)** instead of Jan 1 of year_start
2. Run full reconstruction pipeline for extended period
3. Compute **scale and offset** alignment using Nov-Dec overlap period where both reconstruction and LM ground truth exist:
   ```
   gt = scale * recon + offset
   ```
4. Apply linear transformation to entire stem channel
5. Return only the requested year range

**Script**: `15_reconstruct_with_alignment.py`

### Alignment Methods Comparison

| Method | Formula | When Used |
|--------|---------|-----------|
| Scale + Offset | `aligned = scale * recon + offset` | Default - uses linear regression |
| Offset Only | `aligned = recon + offset` | When reconstruction std < 0.01 |

### Results: Test Set 2021-2022 (17 combinations)

#### Temperature & Humidity (Excellent across all methods)
| Channel | Correlation | R² | Norm MAE |
|---------|-------------|-----|----------|
| Temperature | 0.9880 ± 0.0053 | 0.9751 ± 0.0108 | 0.0212 ± 0.0034 |
| Humidity | 0.9561 ± 0.0185 | 0.8913 ± 0.0499 | 0.0561 ± 0.0131 |

#### Stem (By Dendrometer - Scale+Offset Alignment)

| Dendrometer | Site | Correlation | R² | Status |
|-------------|------|-------------|-----|--------|
| D120 | 22 | 0.9900 | 0.8418 | ✓ Good |
| D121 | 22 | 0.9973 | 0.9809 | ✓ Excellent |
| D922 | 86 | 0.9969 | 0.7683 | ✓ Good |
| D937 | 86 | 0.9278 | 0.8465 | ✓ Good |
| D911 | 86 | 0.9887 | 0.2951 | ⚠ Moderate (improved from R²=-21 with offset-only) |
| D577 | 49 | 0.5324 | -2.67 | ✗ Poor (data quality issue) |
| D849 | 72 | -0.6557 | -37.8 | ✗ Poor (fundamental prediction failure) |
| D850 | 72 | -0.6234 | -28.8 | ✗ Poor (fundamental prediction failure) |

### Key Findings

1. **14 of 17 combinations** have positive stem R² after scale+offset alignment
2. **D911** dramatically improved: R² went from **-21** to **+0.30** with scale alignment
3. **Problematic sensors** (D849, D850 on site 72) show negative correlation - this is a data quality or model limitation issue, not alignment

### Scale Factors Observed

| Category | Scale Factor | Example |
|----------|--------------|---------|
| Near unity | 0.93 - 1.0 | D121 (scale=0.9344), D937 (scale=0.96) |
| Moderate | 0.65 - 0.90 | D120 (scale=0.8720), D922 (scale=0.66) |
| Extreme | 0.17 - 0.26 | D911 (scale=0.17), D577 (scale=0.26) |

**Interpretation**: Large scale factors (far from 1.0) indicate significant calibration differences between L1/L2 and LM processing pipelines for those specific dendrometers.

### Output Files

**Aligned reconstructions**: `/home/lukovic/data/treenet/reconstructions_aligned_v2_2021_2022/`
- `aligned_{combo_id}.ftr` - Reconstructed and aligned time series
- `aligned_results.json` - Metrics summary

**Visualizations**: 
- `stacked_with_gaps_{combo_id}.png` - 9-row stacked plots (Input/Recon/GT for each channel) with gap shading
- `aligned_comparison_{combo_id}.png` - 3-panel plots showing T/RH/Stem comparison with ground truth

### Visualization Gap Shading Interpretation (Added 2026-01-12)

**IMPORTANT**: The red shading in `stacked_with_gaps_*.png` visualizations marks periods where **ALL THREE input channels** (temperature, humidity, stem) are **simultaneously missing** in the input data.

- This is NOT per-channel gap detection
- Gaps are detected at the segment level (same time window applies to all channels)
- The shading helps identify where the model had **no local sensor context** to work with
- Reconstruction quality is typically lower in these shaded regions

**Gap Detection Criteria**:
- Minimum gap duration: 12 hours
- All three L1/L2 input channels must have NaN values
- For site86_T920_H917_D911 (2021-2022): Found 9 gap regions totaling ~3900 hours

### Understanding Data Gaps in Raw Input (Added 2026-01-12)

**IMPORTANT**: Data gaps in TreeNet sensor data can manifest in TWO different ways:

#### Gap Type 1: Missing Timestamps (No Row Exists)
The time series has missing rows - there is no record at all for certain timestamps. This typically occurs when:
- Sensor communication failed completely
- Data transmission was interrupted
- Sensor was offline for maintenance

**Detection method**: 
- Reindex the dataframe to a complete 10-minute time grid
- Missing timestamps become NaN values after reindexing

**Example** (site22, 2021-2022):
```
Expected samples (2 years, 10-min): 105,120
Actual samples: 102,821
Missing timestamps: 2,299 (gap regions at end of Dec 2021 and Dec 2022)
```

#### Gap Type 2: Present Timestamps with NaN Values
The timestamp exists in the data, but the value is NaN (missing). This typically occurs when:
- Sensor recorded a timestamp but measurement failed
- Quality control flagged and removed bad values
- Partial data transmission

**Detection method**: Check for NaN values in existing rows

#### Combined Gap Detection

When visualizing or analyzing gaps, **ALWAYS**:
1. First reindex to a complete time grid (e.g., 10-min intervals for raw data, 1-hour for processed)
2. Then check for NaN values
3. Both types of gaps will now appear as NaN

**Code pattern for proper gap detection:**
```python
# Create complete time index
complete_idx = pd.date_range(start, end, freq='10min', tz='UTC')
# Reindex - missing timestamps become NaN
df_complete = df.reindex(complete_idx)
# Now check for gaps (both types are NaN)
is_gap = df_complete['column'].isna()
```

### Known Limitation: Stem Amplitude Compression (2026-01-12)

For some dendrometers (especially D911), the scale alignment based on Nov-Dec warmup period can **compress the amplitude** of seasonal variations:

| Metric | Input (L2) | Reconstruction | Ground Truth (LM) |
|--------|------------|----------------|-------------------|
| Range (D911) | 184 μm | **29 μm** | 190 μm |

**Root cause**: The scale factor (0.17) was estimated during the dormant Nov-Dec period when stem variation was minimal. This biased the estimate low.

**Result**: High correlation (0.99) but wrong amplitude - the model captures the **shape** but not the **magnitude** of seasonal growth.

**Potential fixes**:
1. Use a longer calibration period (full prior year)
2. Variance-matching instead of linear regression for scale factor
3. Season-specific alignment

### Known Limitation: RH Exceeding Physical Bounds (2026-01-12)

**Issue**: Relative humidity reconstruction can exceed 100% (saturation point), which is physically impossible.

**Analysis for site22_T119_H118_D120 (2021-2022)**:

| Source | Min | Max | Samples >100% |
|--------|-----|-----|---------------|
| Input (L1) | 16.3% | 100.0% | 0 |
| Reconstruction | 4.2% | **109.5%** | 125 (0.71%) |
| Ground Truth (LM) | 14.0% | 100.0% | 0 |

**Observations**:
1. The model occasionally predicts RH values above 100%, especially during gap regions
2. During gaps, the model extrapolates without local context, leading to unphysical values
3. The reconstruction follows GT better than Input (expected since GT is the training target)

**Correlation Analysis**:
- Reconstruction vs Ground Truth: r = 0.937
- Reconstruction vs Input: r = 0.944

**Potential Solutions**:

| Solution | Pros | Cons |
|----------|------|------|
| **Post-processing clipping** | Simple, ensures physical bounds | May distort near-saturation dynamics |
| **Constrained loss function** | Model learns physical limits | Requires retraining |
| **Sigmoid/tanh output activation** | Built-in bounds | May compress valid variation |
| **Physics-informed regularization** | Preserves physical relationships | Complex implementation |

**Recommended approach**: Apply **post-processing clipping** as immediate fix:
```python
reconstructed_rh = np.clip(reconstructed_rh, 0, 100)
```

This is justified because:
- Only 0.71% of samples exceed 100%
- Maximum exceedance is modest (109.5%)
- Physical bounds are well-defined for RH

---

## 26. Current Pipeline Limitations (2026-01-12)

### Limitation 1: Denormalization at Inference (CRITICAL)

**Problem**: Segment-level normalization uses DIFFERENT parameters for input (L2) and output (LM). At inference time, we don't have LM data (that's what we're generating!).

| Aspect | Training | Inference |
|--------|----------|-----------|
| Input normalization | Use L2 min/max | Use L2 min/max ✅ |
| Output denormalization | Use LM min/max | **NO LM available** ❌ |

**Impact by channel:**
- **Temperature**: ~1-2°C error - acceptable
- **Relative Humidity**: ~2-5% error - acceptable  
- **Stem**: ~2-3× scale error - problematic

**Current workarounds:**
- `--denorm operational`: Use input params (approximate for T/RH)
- `--align-stem`: Use overlap period to compute scale correction

### Limitation 2: Segment Boundary Effects

**Problem**: 30-day segments are normalized independently. Predictions at segment boundaries may have discontinuities when reconstructing longer time series.

**Symptom**: Small jumps in reconstructed signal at segment join points.

**Current mitigation**: Overlapping segments with averaging in overlap regions.

### Limitation 3: Single-Stage Learning

**Problem**: Model learns TWO tasks simultaneously:
1. Gap-filling (reconstruct missing 10-min data)
2. Quality enhancement (10-min → 1-hour, L2 → LM quality)

**Impact**: Conflating tasks may limit performance on both.

### Limitation 4: Stem Channel Complexity

**Problem**: Stem dendrometer signals have:
- Non-periodic long-term trends (growth)
- Arbitrary baseline (sensor-specific)
- Different scale in L2 vs LM (post-processing effects)

**Impact**: Stem predictions have correct patterns but wrong absolute scale.

### Limitation 5: No Physical Constraints in Model

**Problem**: Current model doesn't enforce physical bounds:
- RH should be in [0%, 100%]
- Temperature should be physically plausible
- Stem radius should be non-decreasing over long periods

**Status**: RH constraint being added via `--constrain-rh` flag.

### Limitation 6: Site Generalization for Outliers

**Problem**: Some holdout test sites (e.g., Site 86, D911) have unusual data patterns that the model hasn't learned.

**Impact**: Poor correlation (e.g., stem R²=0.25) on some test site/sensor combinations.

---

## 27. Proposed Alternative Approach (For Future Implementation)

This section documents a potentially improved approach, to be considered for future development.

### Core Idea: Two-Stage Pipeline

Instead of one model learning both gap-filling AND quality enhancement:

```
CURRENT (Single-Stage):
L2 (with gaps) → [Single Model] → LM-quality output

PROPOSED (Two-Stage):
L2 (with gaps) → [Stage 1: Gap-Fill] → L2 (no gaps) → [Stage 2: Enhance] → LM-quality
```

**Stage 1: Gap-Filling Model**
- Input: L2 data (10-min) with gaps
- Output: L2 data (10-min) WITHOUT gaps
- **Key advantage**: Input and output are SAME scale → no denormalization problem!

**Stage 2: Quality Enhancement Model**
- Input: Gap-filled L2 data (from Stage 1)
- Output: LM-quality hourly data
- Only trained on gap-free data

### Proposed: Global Normalization

Instead of segment-level normalization (current approach):

```python
# Current: segment-level (different for each 30-day window)
segment_norm = (segment - segment.min()) / (segment.max() - segment.min())

# Proposed: global normalization (same constants for ALL data)
GLOBAL_STATS = {
    'temp': {'min': -30, 'max': 50},      # °C, physical bounds
    'rh':   {'min': 0, 'max': 100},       # %, physical bounds  
    'stem': {'min': 0, 'max': 50000},     # μm, typical range
}
global_norm = (data - GLOBAL_STATS[channel]['min']) / (GLOBAL_STATS[channel]['max'] - GLOBAL_STATS[channel]['min'])
```

**Benefits:**
- Same normalization everywhere → denormalization is trivial
- Model learns absolute values, not just relative patterns
- Cross-segment predictions are consistent

### Proposed: Masked Autoencoder Architecture

Replace TCN with Temporal Masked Autoencoder (inspired by computer vision MAE):

```
Architecture:
1. Encoder: Process ONLY non-gap timesteps (masked attention)
2. Positional encoding: Preserve temporal position information
3. Decoder: Reconstruct ALL timesteps including gaps
4. Loss: Compute only on gap regions
```

**Benefits:**
- Natural fit for gap-filling task
- No information leakage from gap regions
- Attention can capture long-range dependencies

### Proposed: Curriculum Learning

Instead of random gap injection from start:

```
Phase 1 (Epochs 1-30):   Small gaps (1-3 days)
Phase 2 (Epochs 31-60):  Medium gaps (3-7 days)  
Phase 3 (Epochs 61-100): Large gaps (7-14 days)
```

**Benefits:**
- Learn easy patterns first
- Gradually develop long-range modeling capability
- More stable training

### Proposed: Channel-Specific Decoders

Different channels have different characteristics:

| Channel | Characteristic | Strategy |
|---------|---------------|----------|
| Temperature | Strong daily/seasonal cycle | Leverage meteo heavily |
| RH | Correlated with T | Condition on T prediction |
| Stem | Long-term trend, no cycle | Need longer context |

**Proposed architecture:**
```
Shared Encoder → Channel-specific Decoders
         │
         ├── T Decoder (meteo-conditioned)
         ├── RH Decoder (T-conditioned + meteo)
         └── Stem Decoder (long-context attention)
```

### Proposed: Calibration-Based Deployment

For production deployment where LM data is unavailable:

```
Calibration Workflow:
1. Identify small period where LM exists (e.g., Nov-Dec prior year)
2. Run model on that period (gap-free input)
3. Compare model output to LM → compute calibration offsets
4. Apply calibration to all future predictions
```

This formalizes the current `--align-stem` approach as standard practice.

### Implementation Priority

If implementing the alternative approach:

1. **High priority**: Global normalization (simplest, biggest impact)
2. **Medium priority**: Two-stage pipeline (architectural change)
3. **Lower priority**: MAE architecture, curriculum learning (research experiments)

---

## 28. TODO / Future Tasks

### High Priority

1. **✅ Achieve stem MSE < 0.025** (COMPLETED 2026-01-11)
   - **Final result**: MSE = 0.00078, R² = 0.9886
   - Model: segment-norm + attention (128 filters, 5 blocks, 8 heads)
   - Path: `/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260111_152352_segment_norm_attention/best_model.keras`

2. **✅ Implement PATH 1 reconstruction** (COMPLETED 2026-01-11)
   - Script: `6_reconstruct_timeseries_v2.py`
   - Tested on Site 3
   - Results: T correlation 0.993, RH correlation 0.966, stem correlation 0.475
   - **Limitation**: Stem denormalization requires LM reference (see Section 22)

3. **✅ Batch reconstruction 2023-2024** (COMPLETED 2026-01-12)
   - Script: `13_batch_reconstruct.py`
   - Processed: 84 combinations from 10 Swiss sites
   - Output: `/storage/lukovic/Data/FORWARDS/treenet/reconstructed_2023_2024/`
   - Sites: 3, 9, 11, 14, 21, 26, 33, 36, 65, 66

4. **✅ Scale alignment for stem channel** (COMPLETED 2026-01-12)
   - Script: `15_reconstruct_with_alignment.py`
   - Strategy: Nov-Dec prior year scale+offset alignment
   - Result: 14/17 test combinations have positive R²
   - Key improvement: D911 went from R²=-21 to R²=+0.30

5. **☐ Data Quality Filter: Recover Filtered Years**
   - **Issue**: Years failing quality check (ratio outside [0.5, 2.0]) are completely excluded
   - **Impact**: ~124 year-segments worth of data not used for training
   - **Proposed approach**:
     - Develop a cleaning/preprocessing step specifically for filtered years
     - Apply spike detection and removal to L2 input data
     - Re-align L2 with LM after cleaning
     - Re-run quality check to validate cleaned years
   - **Location of filtered year info**: `{run_dir}/logs/filtered_plots/`
   - **Status**: Logging and visualization implemented (2026-01-11)

### Medium Priority

4. **☐ Resolve stem denormalization for PATH 1**
   - Options:
     a. Accept relative output (normalized [0,1])
     b. Develop scale calibration using partial LM data
     c. Train PATH 2 model (LM→LM) for absolute values
   
5. **☐ Develop Stage A: L2→L2 gap-filling model**
   - Create new segment builder for L2→L2 training
   - Train and evaluate model
   - Integrate into reconstruction pipeline

6. **☐ Generate ground truth for 2024-2025**
   - Use trained model to produce clean temperature/humidity data
   - Validate against available LM stem data for those years

### Low Priority / Research

7. **☐ Investigate outlier detection methods**
   - Automated spike detection in L2 data
   - Statistical vs. ML-based approaches

8. **☐ Cross-validation study**
   - Evaluate model generalization across different sites/years

9. **☐ Yearly segment processing (Stage C)**
   - Research feasibility given ground truth limitations

---

**Maintained by**: TreeNet AI Pipeline v2 Development
