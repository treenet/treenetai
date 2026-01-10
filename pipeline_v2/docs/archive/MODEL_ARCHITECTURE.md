# TreeNet AI - Model Architecture & Data Flow

**Last Updated**: 2026-01-09

---

## Overview

The TreeNet AI gap-filling model is a **multi-task Temporal Convolutional Network (TCN)** that takes raw 10-minute sensor data with gaps and produces clean hourly predictions. The architecture is designed around the constraints of the available ground truth data.

---

## Data Context

### Input Data (Raw L1/L2 Sensors)

The raw TreeNet sensor data has the following characteristics:

| Sensor | Variable | Resolution | Issues |
|--------|----------|------------|--------|
| Thermometer (L1) | Air Temperature | 10 min | Gaps, noise, outliers |
| Hygrometer (L1) | Relative Humidity | 10 min | Gaps, noise, outliers, **sensor drift** |
| Dendrometer (L2) | Stem Radius Change | 10 min | Gaps, unrealistic jumps, artifacts |

**Key Issues:**
- **Gaps**: Missing data periods (minutes to days)
- **Artifacts**: Sensor malfunctions, unrealistic jumps (dendrometer)
- **Noise**: Random measurement noise
- **Sensor Drift**: Gradual drift over time (specifically in hygrometers - a known issue with RH sensors)
- Found ~43,000 unrealistic jumps across 10 dendrometer sensors

### Ground Truth Data (Curated LM)

The "LM" (Laboratorium-cleaned) data serves as ground truth:

| Variable | Resolution | Source |
|----------|------------|--------|
| Temperature (local_T) | **1 hour** | Averaged/patched from multiple sensors per site |
| Relative Humidity (local_RH) | **1 hour** | Averaged/patched from multiple sensors per site |
| Stem Radius Change (stem) | 10 min or 1 hour | Manually cleaned individual sensor |

**Critical Constraints:**

1. **Resolution Mismatch**: Ground truth T and RH are only available at 1-hour resolution, not 10-minute
2. **Many-to-One Relationship**: Multiple input sensor combinations map to a single curated output
   - Site may have 3 thermometers × 2 hygrometers × 4 dendrometers = 24 combinations
   - All 24 combinations must predict the same cleaned output
3. **Processing Method**: Curated data may be averages, best-sensor selections, or sensor patches
4. **No Direct 10-min Ground Truth**: For T and RH, we cannot evaluate 10-min reconstruction directly

**Future Improvements:**
When 10-minute curated data with sensor-level correspondence becomes available, the model can be retrained to produce 10-minute outputs with direct supervision.

---

## Model Architecture

### TCN Multi-Task Design

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
        │                         │                         │
        │    TRAINING LOSS        │     TRAINING LOSS       │
        ▼                         │                         ▼
┌───────────────┐                 │            ┌───────────────────┐
│ Compare with  │                 │            │  Compare with     │
│ Original X    │                 │            │  LM Ground Truth  │
│ (pre-gap)     │                 │            │  (hourly)         │
└───────────────┘                 │            └───────────────────┘
```

### Input Details

**Two inputs concatenated (22 channels total):**

1. **`input_x`** - Data tensor `(batch, 4320, 11)`
   - Shape: 30 days × 24 hours × 6 steps/hour = 4320 timesteps
   - 11 channels (see below)
   - Gaps are filled with 0.0 (or interpolated values at inference)

2. **`input_mask`** - Binary mask `(batch, 4320, 11)`
   - 1.0 = valid data point
   - 0.0 = gap (missing/masked data)

**11 Input Channels:**

| Index | Channel | Source | Resolution | Has Gaps? |
|-------|---------|--------|------------|-----------|
| 0 | temp_treenet | Thermometer L1 | 10 min | ✅ Yes |
| 1 | rh_treenet | Hygrometer L1 | 10 min | ✅ Yes |
| 2 | stem | Dendrometer L2 | 10 min | ✅ Yes |
| 3 | tas | MeteoTest grid | Daily → 10 min | ❌ Complete |
| 4 | tasmax | MeteoTest grid | Daily → 10 min | ❌ Complete |
| 5 | tasmin | MeteoTest grid | Daily → 10 min | ❌ Complete |
| 6 | rh | MeteoTest grid | Daily → 10 min | ❌ Complete |
| 7 | vpd | MeteoTest grid | Daily → 10 min | ❌ Complete |
| 8 | gh | MeteoTest grid | Daily → 10 min | ❌ Complete |
| 9 | pr | MeteoTest grid | Daily → 10 min | ❌ Complete |
| 10 | doy | Day of Year | Computed | ❌ Complete |

**Note**: Only channels 0-2 (local sensors) have gaps. Channels 3-10 (gridded meteo) are always complete and provide context for gap filling.

### Output Details

**Two outputs (multi-task):**

1. **`recon_output`** - Reconstructed 10-min data `(batch, 4320, 11)`
   - Same resolution and channels as input
   - Trained to reconstruct the **original** input (before gap injection)
   - **Use case**: Gap-filling at 10-min resolution
   - **Limitation**: No direct ground truth for T/RH at 10-min → quality uncertain

2. **`hourly_output`** - Cleaned hourly predictions `(batch, 720, 3)`
   - 30 days × 24 hours = 720 timesteps
   - 3 channels: local_T, local_RH, stem
   - **Use case**: Primary output for supervised learning
   - **Advantage**: Direct comparison with LM ground truth

---

## How Gap Handling Works

### Training Phase

1. **Complete segments** are loaded (no original gaps)
2. **Synthetic gaps** are injected randomly:
   - Gap duration: 1-12 days (configurable)
   - Gap count: 1-3 per segment
   - Gap channels: Only 0, 1, 2 (local sensors)
   - Gap representation: Values set to 0.0, mask set to 0.0

3. **Model learns to**:
   - Identify gaps via the mask (mask=0 means gap)
   - Use context from non-gapped channels (meteo data)
   - Use temporal context from before/after the gap
   - Predict both 10-min reconstruction AND hourly output

### Inference Phase

1. **Real gaps** in input data are:
   - Filled with interpolated values (linear interpolation)
   - Marked in mask as 0.0

2. **Model predicts**:
   - 10-min reconstruction (all 11 channels)
   - Hourly output (3 channels) → **This is used for gap-filling**

3. **Hourly predictions** are used to fill gaps in the original time series

---

## Training Configuration

From `config.json`:

```json
{
  "model": {
    "n_filters": 64,
    "kernel_size": 3,
    "n_blocks": 4,
    "dropout_rate": 0.2,
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.0003
  },
  "gap": {
    "enabled": true,
    "min_gap_days": 1,
    "max_gap_days": 12,
    "min_gaps_per_segment": 1,
    "max_gaps_per_segment": 3
  }
}
```

### Loss Weights

```python
loss_weights = {
    'recon_output': 1.0,    # Reconstruction loss
    'hourly_output': 1.0    # Hourly prediction loss
}
```

---

## Current Limitations

1. **10-min Reconstruction Quality**: Cannot directly evaluate T/RH at 10-min (no ground truth)
2. **Artifact Removal**: Model is primarily designed for gap-filling, not artifact removal
   - Unrealistic jumps (dendrometer) may persist in 10-min output
   - Sensor drift (hygrometer) - a known issue with RH sensors - may not be fully corrected
   - Hourly output benefits from averaging/smoothing
3. **Many-to-One Mapping**: Same ground truth used for all sensor combinations at a site
4. **Model Capacity**: Current model trained on only ~29 segments (needs full dataset)

---

## Pipeline Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `1_build_segments.py` | Extract 30-day segments | Raw sensor data | Normalized segments |
| `2_train_model.py` | Train TCN model | Segments + gap injection | Trained model |
| `3_evaluate.py` | Evaluate model | Test segments | Metrics |
| `6_reconstruct_timeseries.py` | Fill gaps in full time series | Model + raw data | Reconstructed series |

---

## Future Improvements

1. **10-min Ground Truth**: When available, retrain with direct 10-min supervision
2. **Artifact Handling**: Add preprocessing steps or auxiliary loss for artifact detection
3. **Sensor-Specific Models**: Train separate models per sensor type if curated data becomes sensor-specific
4. **Extended Segments**: Test 60/90-day segments for drift correction

---

## References

- TCN Architecture: [tcn.py](../src/models/tcn.py)
- Gap Injection: [gap_injection.py](../src/gaps/gap_injection.py)
- Training Pipeline: [training.py](../src/models/training.py)
- Reconstruction: [6_reconstruct_timeseries.py](../6_reconstruct_timeseries.py)
