# Segment-Level Normalization Implementation Results

**Date**: 2026-01-08  
**Status**: ✅ Successfully Implemented and Tested

---

## Summary

Segment-level normalization has been implemented as an alternative to year-level normalization. This allows the model to better handle data quality issues (e.g., unrealistic jumps in dendrometer data) by normalizing each 30-day segment independently rather than using year-wide statistics.

---

## Implementation Details

### 1. Modified Files

#### `src/data/segmentation.py`
- **Normalizer class** (line 51):
  - Added `__init__(norm_scope='year')` parameter
  - Updated docstring to document both 'year' and 'segment' scopes
  - No changes to `compute_normalization_params()`, `normalize()`, or `denormalize()` methods (work for both scopes)

- **SegmentBuilder class** (line 330):
  - Added `norm_scope` parameter to `__init__`
  - Updated `build_segments_for_combination()` to handle two workflows:
    - **Year-level**: Normalize full data → find segments → extract
    - **Segment-level**: Find segments → extract raw → normalize each individually
  - Each segment's metadata stores its own normalization parameters

#### `1_build_segments.py`
- Added `--norm-scope` CLI argument (choices: 'year' or 'segment', default: 'year')
- Pass `norm_scope` to SegmentBuilder initialization
- Added configuration printout showing active normalization scope

### 2. How It Works

**Year-Level Normalization (Original)**:
```
1. Load full year(s) of data
2. Compute min/max across all data
3. Normalize using year-wide parameters
4. Extract 30-day segments
5. All segments share same normalization parameters
```

**Segment-Level Normalization (New)**:
```
1. Load full year(s) of data (raw, not normalized)
2. Find valid 30-day segments
3. For each segment:
   a. Extract raw 30-day window
   b. Compute min/max for this segment only
   c. Normalize using segment-specific parameters
   d. Store normalization params in metadata
4. Each segment has unique normalization parameters
```

### 3. Normalization Parameters Storage

With segment-level normalization, the `SegmentMetadata` object for each segment contains:
- `input_min`: Dict of minimum values for each input channel (11 channels)
- `input_diff`: Dict of ranges (max - min) for each input channel
- `output_min`: Dict of minimum values for each output channel (3 channels)
- `output_diff`: Dict of ranges for each output channel

These parameters are needed for:
- **Training**: Already normalized, no action needed
- **Reconstruction**: Denormalize model predictions using segment-specific parameters

---

## Test Results

### Test Configuration
- Site: 3
- Combinations: 5 randomly selected
- Segment length: 30 days
- Stride: 10 days
- Normalization: segment-level

### Output
```
Total combinations: 3 (with valid segments)
Total segments: 29
  - Combo 2: 3 segments
  - Combo 4: 21 segments
  - Combo 5: 5 segments
```

### Normalization Parameter Analysis

**Input (10-min) Dendrometer Channel**:
- Min values: Mean = 7,746.80, Std = 5,985.50
- Range values: Mean = 275.20, Std = 300.61

**Output (hourly) Dendrometer Channel**:
- Min values: Mean = 16,547.48, Std = 4,098.27
- Range values: Mean = 273.55, Std = 301.17

**By Sensor Combination**:
| Combo | Segments | Min (mean ± std) | Range (mean ± std) |
|-------|----------|------------------|--------------------|
| 2     | 3        | 8,456 ± 247      | 723 ± 89          |
| 4     | 21       | 16,366 ± 2,506   | 259 ± 290         |
| 5     | 5        | 22,166 ± 47      | 64 ± 54           |

### Key Findings

1. **High Variability**: Standard deviations are large (e.g., Combo 4: ±2,506 for min values)
   - This is EXPECTED and DESIRABLE
   - Indicates that normalization adapts to local data characteristics
   - Different segments from same sensor have different normalization

2. **Sensor-Specific Patterns**:
   - Combo 5 has much larger min values (22,166) and smaller ranges (64)
   - Suggests different dendrometer sensor or different growth stage
   - Segment-level normalization handles this naturally

3. **Example: Combo 2, Segment 0 vs Segment 2**:
   - **Segment 0** (May 7 - Jun 6, 2022):
     - Stem min: 7,236, range: 822
   - **Segment 2** (May 27 - Jun 26, 2022):
     - Stem min: 7,837, range: 607
   - 20-day overlap but different normalization parameters
   - Adapts to seasonal growth patterns

---

## Comparison: Year-Level vs Segment-Level

### Year-Level Normalization
**Advantages**:
- Consistent normalization across all segments from same year
- Model sees comparable value ranges across different time periods
- Easier to interpret (one set of parameters per year)

**Disadvantages**:
- If data has large jumps/spikes, normalization parameters are distorted
- All segments affected by outliers in other parts of the year
- Cannot adapt to seasonal patterns or drift

### Segment-Level Normalization
**Advantages**:
- Robust to data quality issues (jumps, spikes, drift)
- Adapts to seasonal patterns and local conditions
- Each segment normalized based on its own data distribution
- Jumps within a segment are normalized appropriately

**Disadvantages**:
- More complex (different parameters per segment)
- Requires storing more metadata
- Model must be robust to varying input distributions

---

## Usage

### Build Segments with Segment-Level Normalization

```bash
cd /home/lukovic/codes/treenetai/pipeline_v2

# Single site test
python 1_build_segments.py \
  --force-sites 3 \
  --max-combinations 5 \
  --norm-scope segment \
  --meteo-root /storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data \
  --output-root /home/lukovic/data/treenet/pipeline_v2_test

# Full dataset with segment-level normalization
python 1_build_segments.py \
  --max-combinations -1 \
  --norm-scope segment \
  --meteo-root /storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data \
  --output-root /home/lukovic/data/treenet/pipeline_v2_segment_norm
```

### Training Model with Segment-Level Data

No changes needed to `2_train_model.py` - it automatically uses the normalization parameters from metadata for denormalization during evaluation.

```bash
python 2_train_model.py \
  --data-dir /home/lukovic/data/treenet/pipeline_v2_segment_norm/processed/model_data \
  --output-dir /home/lukovic/data/treenet/pipeline_v2/experiments \
  --epochs 100 \
  --batch-size 32
```

---

## Next Steps

### 1. Full Dataset Extraction
Run unlimited segment extraction with segment-level normalization:
```bash
python 1_build_segments.py \
  --max-combinations -1 \
  --norm-scope segment \
  --meteo-root /storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data
```

Expected: Similar total segment count (~27,000) but with segment-specific normalization

### 2. Model Training Comparison
Train two models and compare:
- Model A: Year-level normalization (current 27,435 segments)
- Model B: Segment-level normalization (new extraction)

Compare metrics:
- Gap filling MAE
- Reconstruction quality
- Performance on segments with data quality issues

### 3. Time Series Reconstruction Module
Implement `6_reconstruct_timeseries.py` as per IMPLEMENTATION_PLAN.md:
- Gap detection in full time series
- Segment creation around gaps
- Model inference with proper normalization
- Denormalization using segment-specific parameters
- Patching reconstructed values back into original series

### 4. Extended Segment Lengths
Test with 60-day and 90-day segments:
- Better drift detection (especially for RH)
- Requires model architecture adjustment (longer sequences)
- Evaluate trade-off: drift detection vs training data quantity

---

## Files Modified

```
/home/lukovic/codes/treenetai/pipeline_v2/
├── src/data/segmentation.py       # Modified: Normalizer and SegmentBuilder classes
├── 1_build_segments.py            # Modified: Added --norm-scope CLI argument
└── IMPLEMENTATION_PLAN.md         # Created: Full implementation plan document
```

---

## Verification

To verify segment-level normalization is working:

```python
import pickle
from pathlib import Path

# Load metadata
metadata_path = Path('/home/lukovic/data/treenet/pipeline_v2_test/processed/model_data/test_segment_ids.pkl')
with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

# Check first 3 segments have different normalization parameters
for i in range(3):
    print(f"Segment {i}:")
    print(f"  Stem min: {metadata[i].output_min['stem']:.2f}")
    print(f"  Stem range: {metadata[i].output_diff['stem']:.2f}")
```

Expected: Different min/range values for each segment (not identical).

---

**Implementation Complete** ✅
