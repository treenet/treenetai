# TreeNet AI Pipeline v2 - Implementation Plan
## Next Steps for Gap Filling and Time Series Reconstruction

**Date**: 2026-01-08  
**Context**: After discovering unrealistic jumps in L2 dendrometer data and testing initial model training

---

## Current Status ✅

### Completed
1. **Segment Extraction**: 27,435 segments created (1,131 train combos, 59 test combos)
2. **Model Architecture**: TCN with 4 blocks, 64 filters (~99k parameters)
3. **Initial Training**: 2 epochs tested successfully on 3,313 segments
4. **Gap Injection**: Working correctly (1-12 days, 1-3 gaps per segment)
5. **Visualization**: Gap filling analysis created for 3 samples
6. **Data Quality Analysis**: Identified unrealistic jumps in L2 data
   - Sensor 20: 57 jumps (max: 9,927 units) in Apr 2024 - Mar 2025
   - 10 sensors checked: ALL have jumps (43,207 total)
   - Only 21% of jumps occur near gaps
7. **Intermediate Files**: 1,190 .ftr files saved with full time series

### Key Findings
- **L2 Raw Data Issues**: Unrealistic jumps common across sensors
- **LM Target Data**: Clean and stable (no jumps)
- **Current Normalization**: Year-level (may be problematic with jumps)
- **Segment Selection**: Strict "no gaps" policy helps but doesn't eliminate all jumps

---

## Priority 1: Segment-Level Normalization 🎯

### Problem
- Year-level normalization uses statistics from entire year
- Years with large jumps produce incorrect normalization constants
- Can distort valid data within clean segments

### Solution
Add configurable normalization scope to pipeline:
- **Year-level** (current): Normalize using yearly statistics
- **Segment-level** (new): Normalize each 30-day segment independently
- **Hybrid**: Year-level for stable channels, segment-level for problematic ones

### Implementation

#### 1. Modify `src/data/normalization.py`

Add parameter to `Normalizer` class:
```python
class Normalizer:
    def __init__(self, norm_scope='year'):  # 'year' or 'segment'
        self.norm_scope = norm_scope
        self.method = 'minmax'  # Can be extended to 'zscore'
```

Add segment-level normalization method:
```python
def normalize_segment(self, segment_df, channels):
    """
    Normalize a single segment using its own statistics.
    
    Args:
        segment_df: DataFrame with one segment
        channels: List of channel names to normalize
    
    Returns:
        normalized_df, norm_params
    """
    norm_params = {}
    result = segment_df.copy()
    
    for ch in channels:
        values = segment_df[ch].values
        min_val = values.min()
        max_val = values.max()
        
        if max_val > min_val:
            result[ch] = (values - min_val) / (max_val - min_val)
            norm_params[ch] = {'min': min_val, 'max': max_val}
        else:
            result[ch] = 0.5  # Constant value
            norm_params[ch] = {'min': min_val, 'max': min_val}
    
    return result, norm_params
```

#### 2. Update `src/data/segmentation.py`

Modify `SegmentBuilder.__init__`:
```python
def __init__(
    self,
    segment_days: int = 30,
    stride_days: int = 10,
    norm_method: str = 'minmax',
    norm_scope: str = 'year'  # NEW
):
    self.norm_scope = norm_scope
    self.normalizer = Normalizer(norm_scope=norm_scope)
```

Update `build_segments_for_combination`:
```python
if self.norm_scope == 'segment':
    # Extract segments first, then normalize each
    for start, end in valid_segments:
        input_seg = self.extractor.extract_segment(input_df, start, end)
        output_seg = self.extractor.extract_segment(output_df, start, end)
        
        # Normalize segment
        input_norm, input_params = self.normalizer.normalize_segment(
            input_seg, input_channels)
        output_norm, output_params = self.normalizer.normalize_segment(
            output_seg, target_channels)
        
        # Store with normalization parameters
        all_input_segments.append((input_norm, input_params))
        all_output_segments.append((output_norm, output_params))

elif self.norm_scope == 'year':
    # Current implementation (year-level normalization)
    ...
```

#### 3. Update CLI Arguments

Modify `1_build_segments.py`:
```python
parser.add_argument(
    '--norm-scope',
    type=str,
    default='year',
    choices=['year', 'segment'],
    help='Normalization scope: year-level or segment-level'
)
```

### Testing Strategy
1. Run with `--norm-scope segment` on small dataset (1 site, 5 combos)
2. Compare segment statistics before/after normalization
3. Verify model training works with segment-level normalized data
4. Compare model performance: year-level vs segment-level

---

## Priority 2: Time Series Reconstruction Module 🔧

### Goal
Reconstruct complete multi-year time series by filling gaps using trained model on test sites.

### Module: `6_reconstruct_timeseries.py`

#### Workflow

**Step 1: Gap Analysis**
```python
def analyze_gaps(df, max_gap_days=12):
    """
    Identify all gaps in time series that are ≤12 days.
    
    Returns:
        gap_info: List of (start_idx, end_idx, gap_days) tuples
    """
    df['time_diff'] = df['ts'].diff()
    expected = pd.Timedelta('10 minutes')  # or '1 hour' for output
    tolerance = pd.Timedelta('15 minutes')
    
    gaps = []
    for idx in range(1, len(df)):
        if df.iloc[idx]['time_diff'] > tolerance:
            gap_days = df.iloc[idx]['time_diff'].days
            if 0 < gap_days <= max_gap_days:
                gaps.append({
                    'start_idx': idx - 1,
                    'end_idx': idx,
                    'start_time': df.iloc[idx-1]['ts'],
                    'end_time': df.iloc[idx]['ts'],
                    'gap_days': gap_days
                })
    
    return gaps
```

**Step 2: Create Segments Around Gaps**
```python
def create_gap_filling_segments(
    input_df, 
    output_df, 
    gaps, 
    segment_days=30
):
    """
    Create 30-day segments centered on each gap.
    
    Ensures:
    - Segment contains the gap
    - Segment has no other gaps
    - Segment boundaries don't overlap with gaps
    
    Returns:
        segments: List of (input_seg, output_seg, gap_info) tuples
    """
    segments = []
    segment_hours = segment_days * 24
    
    for gap in gaps:
        gap_start_time = gap['start_time']
        
        # Try centering segment on gap
        seg_start = gap_start_time - pd.Timedelta(days=segment_days//2)
        seg_end = seg_start + pd.Timedelta(days=segment_days)
        
        # Extract candidate segment
        input_seg = input_df[
            (input_df['ts'] >= seg_start) & 
            (input_df['ts'] < seg_end)
        ]
        output_seg = output_df[
            (output_df['ts'] >= seg_start) & 
            (output_df['ts'] < seg_end)
        ]
        
        # Verify segment validity (no other gaps)
        if is_valid_segment(input_seg, output_seg, gap):
            segments.append({
                'input': input_seg,
                'output': output_seg,
                'gap': gap,
                'segment_start': seg_start,
                'segment_end': seg_end
            })
    
    return segments
```

**Step 3: Prepare Model Input**
```python
def prepare_segment_for_model(
    segment_data,
    meteo_data,
    normalizer,
    norm_scope='segment'
):
    """
    Combine sensor data with meteo data and normalize.
    
    Creates 11-channel input:
    - temp_treenet, rh_treenet, stem (local)
    - tas, tasmax, tasmin, rh, vpd, gh, pr (global)
    - doy (day of year)
    """
    # Merge with meteo
    input_df = merge_with_meteo(segment_data['input'], meteo_data)
    
    # Add day of year
    input_df['doy'] = input_df['ts'].dt.dayofyear / 365.0
    
    # Normalize
    if norm_scope == 'segment':
        input_norm, norm_params = normalizer.normalize_segment(
            input_df, 
            channels=['temp_treenet', 'rh_treenet', 'stem', 
                     'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy']
        )
    else:
        # Use year-level normalization
        input_norm, norm_params = normalizer.normalize_by_year(input_df)
    
    # Convert to model input format (4320, 11)
    X = input_norm.values.reshape(1, -1, 11)
    mask = np.ones_like(X)  # No gaps for reconstruction
    
    return X, mask, norm_params
```

**Step 4: Model Prediction & Gap Filling**
```python
def fill_gaps_in_segment(
    model,
    segment_data,
    gap_info,
    norm_params
):
    """
    Use model to predict values and fill gaps.
    
    Returns:
        filled_segment: DataFrame with gaps filled
        filled_indices: Indices where values were filled
    """
    # Get model predictions
    predictions = model.predict({
        'input_x': segment_data['X'],
        'input_mask': segment_data['mask']
    })
    
    recon_pred = predictions[0]  # Reconstruction (4320, 11)
    hourly_pred = predictions[1]  # Hourly (720, 3)
    
    # Denormalize predictions
    recon_denorm = denormalize(recon_pred, norm_params)
    hourly_denorm = denormalize(hourly_pred, norm_params)
    
    # Extract gap region
    gap_start_idx = find_index(segment_data, gap_info['start_time'])
    gap_end_idx = find_index(segment_data, gap_info['end_time'])
    
    # Fill gap with predicted values
    filled_values = recon_denorm[0, gap_start_idx:gap_end_idx, :]
    
    return filled_values, (gap_start_idx, gap_end_idx)
```

**Step 5: Reconstruct Full Time Series**
```python
def reconstruct_timeseries(
    model,
    test_site_id,
    sensor_ids,
    data_dir,
    output_dir,
    norm_scope='segment'
):
    """
    Main reconstruction pipeline for one test site.
    
    Args:
        model: Trained TCN model
        test_site_id: Site ID to reconstruct
        sensor_ids: (thermo_id, hygro_id, dendro_id)
        data_dir: Path to raw data
        output_dir: Where to save reconstructed series
        norm_scope: 'segment' or 'year'
    
    Returns:
        reconstruction_metrics: MAE, coverage, etc.
    """
    # Load raw data for test site
    thermo_df = load_sensor_data(data_dir, 'thermometer', sensor_ids[0])
    hygro_df = load_sensor_data(data_dir, 'hygrometer', sensor_ids[1])
    dendro_l2_df = load_sensor_data(data_dir, 'dendrometer_l2', sensor_ids[2])
    dendro_lm_df = load_sensor_data(data_dir, 'dendrometer_lm', sensor_ids[2])
    
    # Process into input/output format
    input_df = create_input_array(thermo_df, hygro_df, dendro_l2_df, meteo)
    output_df = create_target_array(dendro_lm_df)
    
    # Analyze gaps
    input_gaps = analyze_gaps(input_df, max_gap_days=12)
    output_gaps = analyze_gaps(output_df, max_gap_days=12)
    
    print(f"Site {test_site_id}:")
    print(f"  Input gaps: {len(input_gaps)}")
    print(f"  Output gaps: {len(output_gaps)}")
    
    # Create segments around gaps
    segments = create_gap_filling_segments(input_df, output_df, input_gaps)
    print(f"  Created {len(segments)} gap-filling segments")
    
    # Process each segment
    filled_data = []
    for seg in segments:
        X, mask, norm_params = prepare_segment_for_model(seg, meteo, normalizer, norm_scope)
        filled = fill_gaps_in_segment(model, {'X': X, 'mask': mask}, seg['gap'], norm_params)
        filled_data.append(filled)
    
    # Merge filled segments back into original time series
    reconstructed = merge_predictions(input_df, output_df, filled_data)
    
    # Save reconstructed time series
    save_path = output_dir / f"reconstructed_site{test_site_id}_T{sensor_ids[0]}_H{sensor_ids[1]}_D{sensor_ids[2]}.ftr"
    reconstructed.to_feather(save_path)
    
    # Calculate metrics (comparing filled values to LM ground truth)
    metrics = calculate_reconstruction_metrics(reconstructed, dendro_lm_df)
    
    return metrics, reconstructed
```

#### CLI Interface
```bash
python 6_reconstruct_timeseries.py \
    --model-path experiments/20260108_134031/best_model.keras \
    --test-sites 3,32,43 \
    --data-dir /storage/lukovic/Data/FORWARDS/treenet/server_data \
    --output-dir /home/lukovic/data/treenet/pipeline_v2/reconstructions \
    --norm-scope segment \
    --max-gap-days 12
```

---

## Priority 3: Extended Segment Lengths 📏

### Motivation
- **30-day segments**: Good for gap filling, but may miss slow drifts (e.g., RH sensor degradation)
- **60-day segments**: Better drift detection, but reduces training data
- **90-day segments**: Maximum drift detection, minimal training data

### Strategy
1. Train separate models for each segment length:
   - Model_30: Current 30-day model
   - Model_60: 60-day segments
   - Model_90: 90-day segments

2. Compare performance:
   - Gap filling accuracy (MAE)
   - Drift detection capability
   - Training data requirements
   - Computational cost

3. Hybrid approach:
   - Use Model_30 for pure gap filling
   - Use Model_60/90 for drift correction
   - Cascade: Model_30 fills gaps → Model_90 corrects drifts

### Implementation
```python
# In 1_build_segments.py, add:
parser.add_argument(
    '--segment-days',
    type=int,
    default=30,
    choices=[30, 60, 90],
    help='Segment length in days'
)

# Will automatically adjust:
# - input_steps = segment_days * 24 * 6  (10-min resolution)
# - output_steps = segment_days * 24  (hourly resolution)
# - Model architecture may need adjustment for longer sequences
```

### Expected Results

| Segment Length | Training Segments | Gap Fill MAE | Drift Detection | Training Time |
|----------------|-------------------|--------------|-----------------|---------------|
| 30 days        | ~27,000          | ✓✓✓          | ✗               | 2-4 hours     |
| 60 days        | ~10,000 (est)    | ✓✓           | ✓✓              | 4-8 hours     |
| 90 days        | ~4,000 (est)     | ✓            | ✓✓✓             | 6-12 hours    |

---

## Testing Strategy 🧪

### Phase 1: Segment-Level Normalization
1. Run `1_build_segments.py` with `--norm-scope segment` on 1 test site
2. Compare normalization statistics: segment vs year level
3. Train model with segment-normalized data (10 epochs)
4. Compare performance metrics with year-level model

### Phase 2: Time Series Reconstruction
1. Implement `6_reconstruct_timeseries.py`
2. Test on single test site (Site 3)
3. Visualize reconstructed vs ground truth
4. Calculate metrics: MAE, gap coverage, reconstruction quality

### Phase 3: Extended Segments
1. Create 60-day and 90-day segments
2. Train models for each length
3. Compare gap filling and drift detection performance
4. Decide on optimal segment length or hybrid approach

---

## Success Metrics 📊

### Gap Filling
- **MAE < 0.15**: Excellent reconstruction
- **Coverage > 95%**: Most gaps successfully filled
- **Speed**: < 1 min per site reconstruction

### Drift Detection
- **RH drift detection**: Model identifies when max(RH) < 100% consistently
- **Temperature drift**: Detects sensor bias over time
- **Stem drift**: Identifies unrealistic growth patterns

### Production Readiness
- **Test site reconstruction**: All 12 test sites successfully reconstructed
- **Multi-year coverage**: Handles data from 2012-2025
- **Robustness**: Works with various gap patterns and lengths

---

## Next Session Checklist ☑️

1. [ ] Implement segment-level normalization in `src/data/normalization.py`
2. [ ] Update `src/data/segmentation.py` to use new normalization
3. [ ] Add `--norm-scope` CLI argument to `1_build_segments.py`
4. [ ] Test segment-level normalization on 1 site
5. [ ] Create `6_reconstruct_timeseries.py` scaffold
6. [ ] Implement gap analysis function
7. [ ] Implement segment creation around gaps
8. [ ] Implement model prediction and gap filling
9. [ ] Test reconstruction on Site 3
10. [ ] Compare with ground truth (LM data)

---

## File Locations 📁

```
/home/lukovic/data/treenet/pipeline_v2/
├── experiments/              # Model checkpoints
├── gap_analysis_results/     # Gap filling visualizations
├── dendrometer_jumps_analysis.png  # Jump analysis plot
├── IMPLEMENTATION_PLAN.md    # This file
└── reconstructions/          # To be created: reconstructed time series

/home/lukovic/codes/treenetai/pipeline_v2/
├── 1_build_segments.py       # Segment extraction (needs --norm-scope)
├── 2_train_model.py          # Model training
├── 6_reconstruct_timeseries.py  # To be created
└── src/
    ├── data/
    │   ├── normalization.py  # Needs segment-level normalization
    │   └── segmentation.py   # Needs norm_scope parameter
    └── models/
        └── training.py       # Model training (already working)

/storage/lukovic/Data/FORWARDS/treenet/
├── processed/processed/model_data/
│   ├── train_input_segments.pkl      # 25,896 segments
│   ├── test_input_segments.pkl       # 1,539 segments
│   └── intermediate_timeseries/      # 1,190 .ftr files
└── server_data/
    ├── dendrometer_l2/       # Raw input data (with jumps)
    ├── dendrometer_lm/       # Clean target data
    ├── thermometer/
    └── hygrometer/
```

---

## Questions to Address 🤔

1. **Normalization Parameters Storage**: 
   - With segment-level norm, each segment has its own min/max
   - Need to store these for reconstruction (for denormalization)
   - How: Save norm_params alongside each segment

2. **Gap Overlapping**:
   - If two gaps are 25 days apart, segments may overlap
   - Strategy: Allow overlapping segments, merge predictions intelligently
   - Or: Prioritize larger gaps first

3. **Multi-Channel Gaps**:
   - What if Temperature has gap but RH doesn't?
   - Current: Model handles with input mask
   - Reconstruction: Fill only the gapped channel, keep others unchanged

4. **Drift Correction Without Ground Truth**:
   - Training uses LM (clean) as target
   - At inference, no LM available
   - Strategy: Model learns to predict "clean" version from "raw" input
   - RH drift: Model learns max(RH) should be ~100%

5. **Computational Efficiency**:
   - Test site reconstruction processes many segments
   - Consider: Batch processing, parallel processing
   - GPU utilization for faster inference

---

## Long-Term Vision 🚀

### Production Pipeline
```
Raw Data (L1, L2) → Gap Analysis → Segment Creation → 
→ Model Inference → Gap Filling → Drift Correction → 
→ Reconstructed Clean Time Series → Archive/Database
```

### Continuous Improvement
1. **Active Learning**: Use reconstruction errors to identify difficult cases
2. **Model Ensemble**: Combine 30/60/90-day models for best results
3. **Sensor-Specific Models**: Train separate models for problematic sensors
4. **Real-Time Processing**: Process new data as it arrives

### Research Directions
1. **Attention Mechanisms**: Focus on gap regions
2. **Multi-Task Learning**: Joint gap filling + drift detection
3. **Transfer Learning**: Use pre-trained models on new sites
4. **Uncertainty Quantification**: Confidence intervals for predictions

---

**End of Implementation Plan**
