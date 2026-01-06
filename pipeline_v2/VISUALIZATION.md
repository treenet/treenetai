# Visualization Guide

Quick reference for visualizing and validating processed segments.

## Overview

The visualization module provides two complementary tools:

1. **Segment Visualization** (`4_visualize_segments.py`):
   - Plot normalized segments showing all input/target channels
   - Generate summary statistics across datasets
   - Inspect coverage and data distributions

2. **Raw Data Comparison** (`5_compare_with_raw.py`):
   - Compare denormalized segments with original raw files
   - Validate data integrity through processing pipeline
   - Ensure normalization is reversible

## Quick Start

### 1. Plot Segments for Specific Site

```bash
python 4_visualize_segments.py --site 1 --year 2021 --split train
```

This creates plots in `processed/model_data/visualizations/train/site_1/year_2021/`.

### 2. Generate Summary Statistics

```bash
python 4_visualize_segments.py --summary --split train
```

Creates `summary_statistics.png` showing:
- Histogram of segments per combination
- Distribution of input/output lengths
- Overall coverage statistics

### 3. Compare with Raw Data

```bash
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
```

This validates that denormalized segments match raw data exactly.

## Understanding the Plots

### Segment Plots (4_visualize_segments.py)

Each segment plot contains:

#### Top Panel: Input Channels (10-minute resolution)
- **Solid lines**: Local sensors (temp, RH, stem)
- **Dashed lines**: Global meteo (tas, tasmax, tasmin, rh, vpd, gh, pr) sampled at noon
- **Twin y-axis**: Left for local (solid), right for global (dashed)

#### Bottom Panel: Target Channels (hourly resolution)
- **local_T**: Cleaned temperature target
- **local_RH**: Cleaned humidity target
- **stem**: Cleaned stem radius target

### Metadata in Title
```
Year 2021 • Site 1 • Combo 23 • Segment 5
2021-06-15 → 2021-07-14 • Input: 4320 steps • Output: 720 steps
```

## Common Use Cases

### Check Data Coverage for Site
```bash
# See how many segments were extracted
python 4_visualize_segments.py --site 1 --year 2021 --split train --max-per-site 50
```

### Validate Normalization
```bash
# Check if values are properly scaled [0, 1]
python 4_visualize_segments.py --site 1 --year 2021 --split train
```
Look for outliers or unnormalized values.

### Compare Train vs Test
```bash
# Train set
python 4_visualize_segments.py --site 1 --year 2021 --split train

# Test set (different sites)
python 4_visualize_segments.py --site 8 --year 2021 --split test
```

### Quick Overview of All Sites
```bash
# Limited to 5 segments per site
python 4_visualize_segments.py --all --split train --max-per-site 5
```

## Programmatic Usage

### In Python Scripts

```python
from pathlib import Path
from visualization.plot_segments import SegmentPlotter

# Initialize plotter
plotter = SegmentPlotter(local_tz='Europe/Zurich')

# Load segments
data_dir = Path('/storage/lukovic/Data/FORWARDS/treenet/processed/model_data')
combo_ids, input_segs, output_segs, metadata = plotter.load_segments(
    data_dir=data_dir,
    split='train'
)

# Plot specific segments
output_dir = Path('./my_plots')
plotter.plot_segments_for_site(
    data_dir=data_dir,
    site_id=1,
    year=2021,
    output_dir=output_dir,
    split='train',
    max_segments=10
)

# Generate summary
plotter.plot_summary_stats(
    data_dir=data_dir,
    output_path=output_dir / 'summary.png',
    split='train'
)
```

## Troubleshooting

### No plots generated
- **Check if segments exist**: Ensure `1_build_segments.py` completed successfully
- **Verify site/year**: Check that the site had data for the specified year
- **Check split**: Use correct split ('train' or 'test')

### Plots look wrong
- **Unnormalized data**: Values should be in [0, 1] range
- **Missing channels**: Some combinations may not have all sensors
- **Coverage gaps**: Check for NaN values (shouldn't appear in normalized data)

### Memory issues
- **Use --max-per-site**: Limit number of plots created
- **Process one site at a time**: Don't use --all for large datasets

## Advanced Options

### Custom Output Directory
```bash
python 4_visualize_segments.py \\
    --site 1 --year 2021 --split train \\
    --output-dir /path/to/custom/output
```

### Hide Global Channels
```bash
# Show only local sensors (cleaner plots)
python 4_visualize_segments.py \\
    --site 1 --year 2021 --split train \\
    --no-globals
```

### Override Data Directory
```bash
python 4_visualize_segments.py \\
    --site 1 --year 2021 --split train \\
    --data-dir /custom/path/to/processed/data
```

## Raw Data Comparison

### Purpose

The raw data comparison tool (`5_compare_with_raw.py`) validates data integrity by:
1. Loading normalized segments
2. Denormalizing to original scale
3. Loading corresponding raw sensor/meteo files
4. Plotting both side-by-side for visual verification

This ensures:
- ✅ Normalization/denormalization is **reversible**
- ✅ No **data corruption** during processing
- ✅ Time windows **match exactly**
- ✅ Resolution conversions are **accurate**

### Usage

#### Compare Single Segment
```bash
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
```

#### Compare All Segments for a Combination
```bash
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

### Understanding Raw Comparison Plots

Each comparison plot shows:
- **Blue solid line**: Raw data from original .ftr/.csv files (10-min or daily)
- **Red dashed line**: Denormalized segment data (10-min input or hourly target)

If processing is correct, the lines should **overlap perfectly** (within floating-point precision).

### What Gets Compared

**Input channels:**
- `temp_treenet`: Raw thermometer_l1 vs denormalized segment input
- `rh_treenet`: Raw hygrometer_l1 vs denormalized segment input
- `stem`: Raw dendrometer_l2 vs denormalized segment input
- Global channels: Raw site CSV daily values vs segment daily means

**Target channels:**
- `local_T`: Raw thermometer 10-min vs denormalized hourly targets
- `local_RH`: Raw hygrometer 10-min vs denormalized hourly targets
- `stem`: Raw dendrometer_lm 10-min vs denormalized hourly targets

### When to Use

Use raw data comparison when:
- **First time running pipeline**: Validate implementation correctness
- **After configuration changes**: Ensure new settings don't break data processing
- **Debugging issues**: Identify where data corruption might occur
- **Quality assurance**: Spot-check random segments periodically

### Troubleshooting

**Lines don't overlap perfectly:**
- Check normalization parameters (min/max) are stored correctly
- Verify time zones are handled consistently (UTC vs local)
- Ensure segment window matches raw data slice exactly
- Check for floating-point precision issues (small differences <1e-6 are OK)

**Missing raw data files:**
- Verify `--raw-root` points to server_data directory
- Ensure sensor ID files exist: `{sensor_type}_series_id_{ID}.ftr`
- Check meteo files exist: `site_{ID}.csv`

**Wrong time window:**
- Segment metadata may be corrupted
- Rebuild segments with `1_build_segments.py`

## Expected Output Structure

```
visualizations/
├── train/
│   ├── summary_statistics.png
│   ├── site_1/
│   │   └── year_2021/
│   │       ├── segment_site1_combo0_seg0.png
│   │       ├── segment_site1_combo0_seg1.png
│   │       └── ...
│   └── site_2/
│       └── year_2021/
│           └── ...
└── test/
    ├── summary_statistics.png
    └── site_8/
        └── year_2021/
            └── ...

comparisons/
└── combo_0/
    ├── seg_0/
    │   ├── y2021_site1_combo0_seg0_temp_treenet.png
    │   ├── y2021_site1_combo0_seg0_rh_treenet.png
    │   ├── y2021_site1_combo0_seg0_stem_input.png
    │   ├── y2021_site1_combo0_seg0_local_T_target.png
    │   ├── y2021_site1_combo0_seg0_local_RH_target.png
    │   ├── y2021_site1_combo0_seg0_stem_target.png
    │   └── y2021_site1_combo0_seg0_*.png (global channels)
    └── seg_1/
        └── ...
```

## Tips

1. **Start with summary**: Always check `--summary` first to understand overall data distribution
2. **Sample randomly**: Use `--max-per-site 5` to get representative samples
3. **Check edge cases**: Visualize first and last segments of each year
4. **Validate train/test split**: Ensure test sites are truly different from train sites
5. **Look for artifacts**: Check for normalization errors, timezone issues, or resampling problems
6. **Use raw comparison sparingly**: Only needed for QA, not routine inspection
7. **Spot-check randomly**: Compare 1-2 segments per combination, not all
