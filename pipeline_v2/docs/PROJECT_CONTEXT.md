# TreeNet AI Pipeline v2 - Project Context

**Purpose**: This file documents project-specific context details to help maintain continuity across sessions. Reference this document at the start of any new session.

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
├── processed/             # Processed model data and segments (old, 25 sites)
└── processed_full_swiss/  # Full rebuild with all 52 Swiss sites
```

> **Note**: The `climate/` directory has been removed as it is no longer relevant for this project.

### Default Paths
| Path Type | Location |
|-----------|----------|
| Raw data | `/storage/lukovic/Data/FORWARDS/treenet/server_data` |
| Meteo data | `/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data` |
| Processed (old) | `/storage/lukovic/Data/FORWARDS/treenet/processed` |
| Processed (full) | `/storage/lukovic/Data/FORWARDS/treenet/processed_full_swiss` |
| Pipeline code | `/home/lukovic/codes/treenetai/pipeline_v2` |

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
**Example**: `dendrometer_l2_series_id_1007.ftr` → data for sensor series_id 1007

### Meteo Data Files (use `site_id`)
```
meteo_data_site_id_{SITE_ID}.csv
```
**Example**: `meteo_data_site_id_3.csv` → daily meteo for site 3 (Bachtel-Forest)

---

## 3. Metadata Files

| File | Description | Key Columns |
|------|-------------|-------------|
| `metadata_all.pkl` | Master metadata | site_id, series_id, country, variable_name |
| `metadata_data_dendro_l2.pkl` | Dendrometer L2 | series_id, site_id |
| `metadata_data_dendro_lm.pkl` | Dendrometer LM (ground truth) | series_id, site_id |
| `metadata_data_all_l1_humidity.pkl` | Hygrometer | series_id, site_id |
| `metadata_data_all_l1_temperature.pkl` | Thermometer | series_id, site_id |
| `metadata_data_all_l1_swp.pkl` | Soil water potential | series_id, site_id |

---

## 4. Site Coverage Summary

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

### Data Availability Observations

When building segments from the 52 Swiss sites, many sensor combinations return **0 segments**. There are three main reasons:

#### 1. Missing LM (Ground Truth) Data
Not all dendrometer L2 sensors have corresponding LM (manually processed) data. The LM data represents human-validated ground truth and is required for supervised learning. Some sensors have L2 data but no LM data yet.

#### 2. Insufficient Temporal Overlap
Sensor combinations may have overlapping operation periods too short to form complete 30-day segments. This happens when:
- Sensors were installed at different times
- One sensor was replaced/removed while others continued
- Gaps in raw data exceed the segment requirements

#### 3. New Sites Without Ground Truth
Sites with higher IDs (e.g., 157+) are newer installations. These have raw sensor data but no LM (ground truth) data yet. **This is precisely why we are building this model** - to provide an automated process for creating LM data for new sites where manual processing hasn't been done yet.

### Latest Build Results (January 2025)
| Split | Combinations | Segments |
|-------|--------------|----------|
| Train | 84 | 3,393 |
| Test | 4 | 71 |

Output: `/storage/lukovic/Data/FORWARDS/treenet/processed_full_swiss/processed/model_data`

---

## 5. Meteo Data Details

### Swiss Sites (Valid)
- **Source**: MeteoSwiss daily gridded data
- **Coverage**: 1981-01-01 to 2024-12-31 (16,071 days)
- **Variables**: 
  - `ff`: wind speed (m/s)
  - `gh`: global radiation (W/m²)
  - `pr`: precipitation (mm)
  - `tas`: mean temperature (°C)
  - `tasmax`: max temperature (°C)
  - `tasmin`: min temperature (°C)
  - `rh`: relative humidity (%)
  - `vpd`: vapor pressure deficit (kPa)
- **Data quality**: ~100% valid values

### Non-Swiss Sites (Invalid)
- Meteo files exist but contain **100% NaN values**
- These sites are automatically filtered out during segment building
- The `--country Switzerland` flag (default) excludes them

---

## 6. Model Architecture

### Input Specification
- **Resolution**: 10-minute intervals
- **Window size**: 4320 timesteps = 30 days
- **Channels**: 11

#### Channel Categories

The 11 input channels are divided into two distinct categories with different purposes:

##### Local Sensor Channels (0-2) - TARGET FOR GAP FILLING
| Channel | Variable | Source | Resolution | Notes |
|---------|----------|--------|------------|-------|
| 0 | temp_treenet | Local thermometer | 10-min | Below canopy temperature |
| 1 | rh_treenet | Local hygrometer | 10-min | Below canopy humidity |
| 2 | stem | Dendrometer | 10-min | Stem radius change |

**These are the ONLY channels where gaps are injected during training.** These channels represent local measurements at the site level, below the tree canopy. They are subject to sensor malfunctions, data transmission failures, and other issues that cause missing data.

##### Global Meteo Channels (3-10) - NEVER GAPPED
| Channel | Variable | Source | Resolution | Notes |
|---------|----------|--------|------------|-------|
| 3 | tas | MeteoSwiss | Daily | Mean air temperature (above canopy) |
| 4 | tasmax | MeteoSwiss | Daily | Max air temperature (above canopy) |
| 5 | tasmin | MeteoSwiss | Daily | Min air temperature (above canopy) |
| 6 | rh | MeteoSwiss | Daily | Relative humidity (above canopy) |
| 7 | vpd | MeteoSwiss | Daily | Vapor pressure deficit |
| 8 | gh | MeteoSwiss | Daily | Global horizontal irradiance |
| 9 | pr | MeteoSwiss | Daily | Precipitation |
| 10 | doy | Computed | N/A | Day of year (1-365) |

**These channels are NEVER gapped** because:
1. **Data quality**: Global meteo data is professionally maintained and gap-free
2. **Reference role**: They serve as auxiliary information to help the model fill gaps
3. **Temporal context**: Channels 3-9 provide atmospheric context; channel 10 (day of year) indicates seasonal patterns
4. **Spatial scale**: Global meteo represents regional conditions (km scale) vs local sensors (m scale)

##### Why This Matters

The model learns to use **correlations** between local sensor data and global meteo data:
- If local temperature (ch 0) has a gap, the model uses global temperature (ch 3-5) plus humidity and radiation patterns to estimate the missing values
- If stem radius (ch 2) has a gap, the model uses temperature, humidity, and precipitation patterns to predict the physiological response
- The day-of-year channel (ch 10) is especially important - it tells the model whether it's winter/summer, which dramatically affects expected values

**Gap Injection Configuration** (in `src/gaps/gap_injection.py`):
```python
GAPPABLE_CHANNELS = [0, 1, 2]  # Only local sensor channels
# Global meteo channels (3-10) are never modified
```

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
- **Architecture**: TCN-based encoder-decoder
- **Multi-task outputs**:
  - `recon_output`: 10-min reconstruction (4320×11)
  - `hourly_output`: 1-hour predictions (720×3)
- **Parameters**: ~98k trainable
- **Inputs**: `[input_x, input_mask]` where mask=1 for valid data

### Model Usage
```python
# Model expects 2 inputs
predictions = model.predict([input_data, mask])
# Returns: [recon_output, hourly_output]
```

---

## 7. Intermediate Data Files

During segment building, full multi-year time series are saved for each sensor combination before segmentation.

### Input Time Series (11-channel, 10-min resolution)
```
intermediate_timeseries/{split}_input_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}.ftr
```
**Example**: `train_input_site3_T9_H7_D18.ftr`

### Output/Target Time Series (3-channel, hourly resolution)
```
intermediate_timeseries/{split}_output_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}.ftr
```
**Example**: `train_output_site3_T9_H7_D18.ftr`

### File Format Details
| Property | Description |
|----------|-------------|
| Format | Feather (`.ftr`) - fast binary, pandas-compatible |
| Structure | DataFrame with `ts` column (timestamp) + data columns |
| Coverage | ENTIRE available time series (not segmented) |
| Location | `{output_root}/processed/model_data/intermediate_timeseries/` |

### Use Cases
1. **Reconstruction**: Apply trained model to fill gaps in full multi-year time series
2. **Visualization**: Plot entire time series before/after gap-filling
3. **Debugging**: Verify data quality before segmentation
4. **Analysis**: Study long-term patterns and sensor drift

---

## 8. Python Environment

- **Python version**: 3.10.12
- **Virtual environment**: `/home/lukovic/pyenv/lamella/bin/python`
- **Activation**: `source /home/lukovic/pyenv/lamella/bin/activate`

### Key Packages
- TensorFlow/Keras (deep learning)
- pandas, numpy (data manipulation)
- feather-format (fast file I/O)
- matplotlib (visualization)

---

## 9. Pipeline Scripts

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `1_build_segments.py` | Extract 30-day segments | `--country`, `--max-sites`, `--max-combinations` |
| `2_preprocess.py` | Prepare for training | |
| `3_train_model.py` | Train TCN model | `--epochs`, `--batch-size` |
| `4_evaluate_model.py` | Evaluate performance | |
| `5_export_predictions.py` | Export predictions | |
| `6_reconstruct_timeseries.py` | **Gap-filling main script** | `--site-id`, `--model-path` |
| `7_visualize_reconstruction.py` | Before/after plots | `--site-id` |

### Key Script: 1_build_segments.py
```bash
# Build with all Swiss sites (default behavior)
python 1_build_segments.py --max-sites -1

# Build with specific sites
python 1_build_segments.py --force-sites 3,4,10

# Build with all countries (not recommended)
python 1_build_segments.py --country all
```

### Key Script: 6_reconstruct_timeseries.py
```bash
# Reconstruct a single site
python 6_reconstruct_timeseries.py --site-id 3 --model-path /path/to/model.keras
```

---

## 10. Known Technical Issues & Design Decisions

This section documents important technical challenges and the rationale behind design decisions.

### Issue 1: Hygrometer Sensor Drift

**Problem**: The hygrometer hardware inherently causes signal drift over time. Two specific manifestations:

1. **Baseline Drift**: The signal gradually shifts up or down from its true value
2. **Saturation Cap**: After some time, the signal never reaches 100% relative humidity, but saturates at 95% or lower

**Challenge**: When the signal saturates at <100%, we cannot determine whether:
- The signal should be shifted to 100% (sensor degradation)
- The relative humidity is genuinely at that lower value (real measurement)

**Future Goal**: The model should eventually be able to detect and correct these drift patterns. This is one motivation for using longer input segments (see Issue 3).

**Current Status**: Not yet addressed - requires model enhancement

---

### Issue 2: Segment Length Trade-offs

**Current Setting**: 30-day segments (4320 timesteps at 10-min resolution)

**Ideal Goal**: Year-long segments to capture:
- Seasonal patterns
- Long-term sensor drift (especially hygrometer - see Issue 1)
- Full annual growth cycles

**Problem**: Due to low quality of raw data (too many gaps), it is difficult to compile a single uninterrupted 11-channel segment spanning an entire year. Many sensor combinations have overlapping coverage of only a few months.

**Strategy**:
1. Start with 30-day segments (proven to work)
2. Evaluate model performance
3. Gradually extend segment length once model demonstrates good gap-filling capability
4. Find optimal trade-off between segment length and model performance

**Note**: Longer segments improve the chances of identifying hygrometer drift patterns, as drift accumulates over time.

---

### Issue 3: Timezone Handling

**Raw Data Storage**: Timestamps stored in local timezone:
- `Europe/Zurich` 
- Sometimes `Etc/GMT-1` (equivalent for Swiss winter time)

**Problem**: Local timezone causes issues:
1. DST transitions create gaps (spring forward) or duplicates (fall back)
2. Complicates merging data from different sources
3. Makes segment extraction unreliable across DST boundaries

**Historical Note on DST in TreeNet Data**:
- **Older data**: Some historical data may switch between summer time (CEST, UTC+2) and winter time (CET, UTC+1) following actual DST transitions
- **Recent and current data**: In later years (especially present and future), all timestamps are recorded in **local winter time (CET, UTC+1)** regardless of the actual season. This simplifies processing but must be handled correctly.
- The pipeline assumes `Europe/Zurich` timezone which handles both cases appropriately

**Solution**: All timestamps converted to **UTC** during processing:
```python
class TimestampProcessor:
    """Converts between local timezone and UTC."""
    def __init__(self, local_tz: str = 'Europe/Zurich'):
        self.local_tz = local_tz
    
    def to_utc_index(self, ts_series: pd.Series) -> pd.DatetimeIndex:
        # Handle both tz-naive and tz-aware inputs
        # Localize to Europe/Zurich then convert to UTC
        ...
```

**DST Handling**:
- `nonexistent='shift_forward'`: Handle DST gaps (spring forward)
- `ambiguous='NaT'`: Handle DST overlaps (fall back) as missing data

---

### Issue 4: Daily Meteo Data + UTC Timestamp Alignment

**Problem**: Global meteo data is daily resolution (only has dates, no times), while sensor data has 10-minute UTC timestamps. How to properly align them?

**Complication**: A UTC timestamp might fall on different calendar dates depending on timezone:
- `2023-06-15 23:30:00 UTC` = `2023-06-16 01:30:00 Europe/Zurich` (summer)
- Same physical moment, different calendar dates!

**Solution**: Use "civil day" concept:

```python
def process_meteo_daily(self, meteo_df: pd.DataFrame) -> pd.DataFrame:
    """Process daily meteo data: convert to local civil days."""
    
    # Convert to local timezone first
    ts_local = self.ts_processor.to_local_series(meteo_df['ts'])
    
    # Get the local civil day (midnight in local time)
    civil_day = ts_local.dt.normalize()
    
    # Index meteo data by civil day
    result.index = civil_day
    ...
```

**How It Works**:
1. Sensor timestamp (UTC) → Convert to local time (Europe/Zurich)
2. Local time → Extract the local calendar date
3. Match to meteo data for that calendar date

**Example**:
- Sensor reading: `2023-06-15 23:30:00 UTC`
- Local time: `2023-06-16 01:30:00 Europe/Zurich`
- Civil day: `2023-06-16` → Use meteo data for June 16th

This ensures the meteo data reflects what the tree actually experienced on that local calendar day, not the UTC calendar day.

---

## 11. Common Code Issues & Solutions

### Issue 1: Timestamp Alignment
**Problem**: Segment start not aligned to 10-minute grid
**Solution**: Use `.floor('10min')` on segment start timestamp
```python
seg_start = seg_start.floor('10min')
```

### Issue 2: Datetime Dtype Mismatch
**Problem**: `datetime64[us]` vs `datetime64[ns]` comparison fails
**Solution**: Explicit conversion
```python
input_seg_cols.index = pd.to_datetime(input_seg_cols.index).tz_convert('UTC')
```

### Issue 3: Meteo File Not Found
**Problem**: Old code looking for `site_{ID}.csv`
**Solution**: Updated to `meteo_data_site_id_{ID}.csv`

### Issue 4: Model Input Shape
**Problem**: Model expects `[input, mask]` not just `input`
**Solution**: Always pass mask array
```python
predictions = model.predict([input_data, mask])
```

---

## 12. Performance Metrics

### Current Best Results (Site 3 test)
- **Fill rate**: 87% (76/87 gaps filled)
- **MAE**: 49.14 μm
- **Skipped gaps**: 11 (near data boundaries)

### Training Dataset (old, 25 sites)
- Train segments: 25,896
- Test segments: ~5,000
- Early stopping: Epoch 16

---

## 13. GPU & Hardware

- **Available GPUs**: 6x NVIDIA RTX 3090
- **Default GPU**: GPU 0
- **Check availability**: `nvidia-smi`

---

## 14. Quick Reference Commands

```bash
# Activate environment
source /home/lukovic/pyenv/lamella/bin/activate

# Navigate to pipeline
cd /home/lukovic/codes/treenetai/pipeline_v2

# Build segments (all Swiss sites)
python 1_build_segments.py --max-sites -1 --country Switzerland

# Train model
python 3_train_model.py --data-dir /path/to/processed

# Reconstruct time series
python 6_reconstruct_timeseries.py --site-id 3

# Check GPU
nvidia-smi
```

---

## 15. Session Continuity Checklist

When starting a new session, verify:
1. ☐ Python environment: `/home/lukovic/pyenv/lamella/bin/python`
2. ☐ Data paths exist and are accessible
3. ☐ GPU available if training
4. ☐ Review recent changes in git

---

**Last Updated**: January 2025
**Maintained by**: TreeNet AI Pipeline v2 Development
