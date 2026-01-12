#!/usr/bin/env python3
"""
Visualize constrained RH reconstruction in 9-row stacked format.

For each channel (T, RH, Stem), shows 3 rows:
1. Raw Input (L1/L2 data)
2. Reconstruction (from constrained RH model)
3. Ground Truth (LM data)

Gap regions are shown with light red shading where ALL THREE input channels
are simultaneously missing.

Usage:
    python visualize_stacked_with_gaps_constrained.py \
        --combo-id site22_T119_H118_D120 \
        --years 2021 2022

Author: TreeNet AI Pipeline
Date: 2026-01-12
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.feather as feather
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor
from src.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize stacked with gaps (constrained RH)')
    parser.add_argument('--combo-id', type=str, required=True,
                        help='Combination ID, e.g. site22_T119_H118_D120')
    parser.add_argument('--recon-path', type=str,
                        default='/home/lukovic/data/treenet/reconstructions_constrained_site22/test_input_site22_T119_H118_D120_reconstructed.feather',
                        help='Path to constrained reconstruction file')
    parser.add_argument('--years', type=int, nargs='+', default=[2021, 2022],
                        help='Years to visualize')
    parser.add_argument('--data-dir', type=str,
                        default='/storage/lukovic/Data/FORWARDS/treenet/server_data',
                        help='Root directory with raw sensor data')
    parser.add_argument('--meteo-root', type=str,
                        default='/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data',
                        help='Directory with meteo CSV files')
    parser.add_argument('--output-dir', type=str,
                        default='/home/lukovic/data/treenet/rh_constraint_comparison',
                        help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output figure')
    return parser.parse_args()


def parse_combo_id(combo_id: str):
    """Parse combination ID like 'site22_T119_H118_D120' into components."""
    parts = combo_id.split('_')
    site_id = int(parts[0].replace('site', ''))
    thermo_id = int(parts[1].replace('T', ''))
    hygro_id = int(parts[2].replace('H', ''))
    dendro_id = int(parts[3].replace('D', ''))
    return site_id, thermo_id, hygro_id, dendro_id


def load_raw_input_data(loaders: DataLoaders, processor: DataProcessor,
                        thermo_id: int, hygro_id: int, dendro_id: int,
                        years: list) -> pd.DataFrame:
    """Load and merge L1/L2 raw input data."""
    dfs = []
    
    print(f"  Loading thermometer L1 (ID={thermo_id})...")
    thermo_raw = loaders.load_thermometer_l1(thermo_id)
    if thermo_raw is not None:
        thermo_proc = processor.process_sensor_dataframe(thermo_raw, keep_all_columns=True)
        # Column is 'value' after processing - rename to input_T
        if 'value' in thermo_proc.columns:
            thermo_proc = thermo_proc.rename(columns={'value': 'input_T'})
            thermo_proc = thermo_proc.reset_index()  # ts is the index
            dfs.append(thermo_proc[['ts', 'input_T']])
            print(f"    Loaded: {len(thermo_proc):,} samples")
    
    print(f"  Loading hygrometer L1 (ID={hygro_id})...")
    hygro_raw = loaders.load_hygrometer_l1(hygro_id)
    if hygro_raw is not None:
        hygro_proc = processor.process_sensor_dataframe(hygro_raw, keep_all_columns=True)
        if 'value' in hygro_proc.columns:
            hygro_proc = hygro_proc.rename(columns={'value': 'input_RH'})
            hygro_proc = hygro_proc.reset_index()
            dfs.append(hygro_proc[['ts', 'input_RH']])
            print(f"    Loaded: {len(hygro_proc):,} samples")
    
    print(f"  Loading dendrometer L2 (ID={dendro_id})...")
    dendro_raw = loaders.load_dendrometer_l2(dendro_id)
    if dendro_raw is not None:
        dendro_proc = processor.process_sensor_dataframe(dendro_raw, keep_all_columns=True)
        if 'value' in dendro_proc.columns:
            dendro_proc = dendro_proc.rename(columns={'value': 'input_stem'})
            dendro_proc = dendro_proc.reset_index()
            dfs.append(dendro_proc[['ts', 'input_stem']])
            print(f"    Loaded: {len(dendro_proc):,} samples")
    
    if not dfs:
        return None
    
    # Merge all
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='ts', how='outer')
    
    merged = merged.set_index('ts').sort_index()
    
    # Filter to years
    merged = merged[(merged.index.year >= min(years)) & (merged.index.year <= max(years))]
    
    # Reindex to complete time series to detect missing timestamps as gaps
    complete_idx = pd.date_range(
        f'{min(years)}-01-01', 
        f'{max(years)}-12-31 23:50:00', 
        freq='10min', 
        tz='UTC'
    )
    merged = merged.reindex(complete_idx)
    print(f"  After reindex to complete 10-min grid: {merged.isna().sum().sum():,} missing values")
    
    return merged


def load_lm_data(loaders: DataLoaders, processor: DataProcessor, dendro_id: int, years: list) -> pd.DataFrame:
    """Load LM ground truth data."""
    print(f"  Loading LM data (dendro_id={dendro_id})...")
    lm_raw = loaders.load_dendrometer_lm(dendro_id)
    if lm_raw is None:
        return None
    
    lm_processed = processor.process_sensor_dataframe(lm_raw, keep_all_columns=True)
    lm_df = processor.merger.create_target_array(lm_processed)
    
    # Rename columns for clarity
    lm_df = lm_df.rename(columns={
        'local_T': 'lm_T',
        'local_RH': 'lm_RH',
        'stem': 'lm_stem'
    })
    
    lm_df = lm_df.sort_index()
    lm_df = lm_df[(lm_df.index.year >= min(years)) & (lm_df.index.year <= max(years))]
    
    return lm_df


def identify_input_gap_regions(input_df: pd.DataFrame, min_gap_hours: int = 12):
    """
    Identify gap regions where ALL THREE input channels are simultaneously missing.
    
    Returns list of (start, end) tuples for contiguous gap regions.
    """
    gaps = []
    
    if input_df is None or len(input_df) == 0:
        return gaps
    
    # Create combined mask: True where all three are NaN
    has_T = ~input_df['input_T'].isna() if 'input_T' in input_df.columns else pd.Series(False, index=input_df.index)
    has_RH = ~input_df['input_RH'].isna() if 'input_RH' in input_df.columns else pd.Series(False, index=input_df.index)
    has_stem = ~input_df['input_stem'].isna() if 'input_stem' in input_df.columns else pd.Series(False, index=input_df.index)
    
    all_missing = (~has_T) & (~has_RH) & (~has_stem)
    
    # Find contiguous gap regions
    in_gap = False
    gap_start = None
    
    for i, (idx, is_missing) in enumerate(zip(input_df.index, all_missing)):
        if is_missing and not in_gap:
            gap_start = idx
            in_gap = True
        elif not is_missing and in_gap:
            # End of gap
            gap_duration_hours = (input_df.index[i-1] - gap_start).total_seconds() / 3600
            if gap_duration_hours >= min_gap_hours:
                gaps.append((gap_start, input_df.index[i-1]))
            in_gap = False
    
    # Handle gap that extends to end
    if in_gap and gap_start is not None:
        gap_duration_hours = (input_df.index[-1] - gap_start).total_seconds() / 3600
        if gap_duration_hours >= min_gap_hours:
            gaps.append((gap_start, input_df.index[-1]))
    
    return gaps


def identify_per_channel_gaps(input_df: pd.DataFrame, channel: str, min_gap_hours: int = 4):
    """
    Identify gap regions for a specific input channel.
    
    Returns list of (start, end) tuples.
    """
    gaps = []
    
    if input_df is None or len(input_df) == 0 or channel not in input_df.columns:
        return gaps
    
    is_missing = input_df[channel].isna()
    
    in_gap = False
    gap_start = None
    
    for i, (idx, missing) in enumerate(zip(input_df.index, is_missing)):
        if missing and not in_gap:
            gap_start = idx
            in_gap = True
        elif not missing and in_gap:
            gap_duration_hours = (input_df.index[i-1] - gap_start).total_seconds() / 3600
            if gap_duration_hours >= min_gap_hours:
                gaps.append((gap_start, input_df.index[i-1]))
            in_gap = False
    
    if in_gap and gap_start is not None:
        gap_duration_hours = (input_df.index[-1] - gap_start).total_seconds() / 3600
        if gap_duration_hours >= min_gap_hours:
            gaps.append((gap_start, input_df.index[-1]))
    
    return gaps


def create_stacked_visualization(
    input_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    lm_df: pd.DataFrame,
    gap_regions: list,
    per_channel_gaps: dict,
    combo_id: str,
    years: list,
    output_dir: Path,
    dpi: int = 300
):
    """
    Create 9-row stacked visualization with gap shading.
    
    Layout:
    Row 1-3: Temperature (Input, Recon, GT)
    Row 4-6: Relative Humidity (Input, Recon, GT)
    Row 7-9: Stem Radius (Input, Recon, GT)
    """
    # Colors - updated per user request
    color_input = '#2ca02c'    # Green for raw input
    color_recon = '#d62728'    # Red for reconstruction
    color_lm = '#1f77b4'       # Blue for ground truth
    color_gap = '#ffcccc'      # Light red for gap shading
    
    # Create figure
    fig, axes = plt.subplots(9, 1, figsize=(20, 24), sharex=True)
    
    # Channel configuration
    channels = [
        {
            'name': 'Temperature',
            'input_col': 'input_T',
            'recon_col': 'recon_T',
            'lm_col': 'lm_T',
            'ylabel': '°C',
            'rows': [0, 1, 2]
        },
        {
            'name': 'Relative Humidity',
            'input_col': 'input_RH',
            'recon_col': 'recon_RH',
            'lm_col': 'lm_RH',
            'ylabel': '%',
            'rows': [3, 4, 5]
        },
        {
            'name': 'Stem Radius',
            'input_col': 'input_stem',
            'recon_col': 'recon_stem',
            'lm_col': 'lm_stem',
            'ylabel': 'μm',
            'rows': [6, 7, 8]
        }
    ]
    
    for ch in channels:
        row_input, row_recon, row_lm = ch['rows']
        
        # Get per-channel gaps
        ch_gaps = per_channel_gaps.get(ch['input_col'], [])
        
        # Row 1: Input (L1/L2)
        ax = axes[row_input]
        if input_df is not None and ch['input_col'] in input_df.columns:
            ax.plot(input_df.index, input_df[ch['input_col']], 
                    color=color_input, linewidth=0.8, label='Raw Input (L1/L2)')
        ax.set_ylabel(f"{ch['name']}\\n(Input)\\n{ch['ylabel']}", fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add gap shading to input row
        for gap_start, gap_end in ch_gaps:
            ax.axvspan(gap_start, gap_end, color=color_gap, alpha=0.5, zorder=0)
        
        # Row 2: Reconstruction
        ax = axes[row_recon]
        if recon_df is not None and ch['recon_col'] in recon_df.columns:
            ax.plot(recon_df.index, recon_df[ch['recon_col']], 
                    color=color_recon, linewidth=0.8, label='Reconstruction (Constrained RH)')
        ax.set_ylabel(f"{ch['name']}\\n(Recon)\\n{ch['ylabel']}", fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add gap shading to recon row
        for gap_start, gap_end in ch_gaps:
            ax.axvspan(gap_start, gap_end, color=color_gap, alpha=0.5, zorder=0)
        
        # Row 3: Ground Truth (LM)
        ax = axes[row_lm]
        if lm_df is not None and ch['lm_col'] in lm_df.columns:
            ax.plot(lm_df.index, lm_df[ch['lm_col']], 
                    color=color_lm, linewidth=0.8, label='Ground Truth (LM)')
        ax.set_ylabel(f"{ch['name']}\\n(GT)\\n{ch['ylabel']}", fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add gap shading to GT row
        for gap_start, gap_end in ch_gaps:
            ax.axvspan(gap_start, gap_end, color=color_gap, alpha=0.5, zorder=0)
    
    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Title
    year_str = '-'.join(map(str, years))
    fig.suptitle(
        f'Constrained RH Model Reconstruction - {combo_id} ({year_str})\\n'
        f'(Light red shading indicates gap regions in input data)',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'stacked_with_gaps_constrained_{combo_id}.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def main():
    args = parse_args()
    
    output_dir = ensure_dir(Path(args.output_dir))
    
    print("=" * 80)
    print("Stacked Visualization with Gaps - Constrained RH Model")
    print("=" * 80)
    
    # Parse combination ID
    combo_id = args.combo_id
    site_id, thermo_id, hygro_id, dendro_id = parse_combo_id(combo_id)
    print(f"\nCombination: {combo_id}")
    print(f"  Site: {site_id}, Thermo: {thermo_id}, Hygro: {hygro_id}, Dendro: {dendro_id}")
    print(f"  Years: {args.years}")
    
    # Initialize loaders
    loaders = DataLoaders(
        data_root=Path(args.data_dir),
        meteo_root=Path(args.meteo_root)
    )
    processor = DataProcessor()
    
    # Load raw input data
    print("\nLoading raw input data (L1/L2)...")
    input_df = load_raw_input_data(loaders, processor, thermo_id, hygro_id, dendro_id, args.years)
    if input_df is not None:
        print(f"  Loaded: {len(input_df):,} samples")
        print(f"  Date range: {input_df.index.min()} to {input_df.index.max()}")
    else:
        print("  Warning: No input data loaded!")
    
    # Load LM ground truth
    print("\nLoading LM ground truth...")
    lm_df = load_lm_data(loaders, processor, dendro_id, args.years)
    if lm_df is not None:
        print(f"  Loaded: {len(lm_df):,} samples")
    else:
        print("  Warning: No LM data loaded!")
    
    # Load reconstruction data
    print(f"\nLoading reconstruction data from: {args.recon_path}")
    recon_df = feather.read_feather(args.recon_path)
    if 'ts' in recon_df.columns:
        recon_df = recon_df.set_index('ts')
    if recon_df.index.tz is None:
        recon_df.index = recon_df.index.tz_localize('UTC')
    
    # Filter to years
    recon_df = recon_df[(recon_df.index.year >= min(args.years)) & (recon_df.index.year <= max(args.years))]
    print(f"  Loaded: {len(recon_df):,} samples")
    
    # Resample reconstruction to hourly to match LM
    print("\nResampling reconstruction to hourly...")
    recon_hourly = recon_df.resample('1H').mean()
    print(f"  Hourly samples: {len(recon_hourly):,}")
    
    # Identify gap regions
    print("\nIdentifying gap regions...")
    gap_regions = identify_input_gap_regions(input_df)
    print(f"  Found {len(gap_regions)} gap regions where all inputs are missing")
    
    # Identify per-channel gaps
    per_channel_gaps = {
        'input_T': identify_per_channel_gaps(input_df, 'input_T'),
        'input_RH': identify_per_channel_gaps(input_df, 'input_RH'),
        'input_stem': identify_per_channel_gaps(input_df, 'input_stem')
    }
    for ch, gaps in per_channel_gaps.items():
        print(f"  {ch}: {len(gaps)} gap regions")
    
    # Create visualization
    print("\nCreating stacked visualization...")
    output_path = create_stacked_visualization(
        input_df, recon_hourly, lm_df,
        gap_regions, per_channel_gaps,
        combo_id, args.years, output_dir, args.dpi
    )
    print(f"  Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
