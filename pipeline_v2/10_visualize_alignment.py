#!/usr/bin/env python3
"""
Visualize stem signal alignment for a specific sensor combination.

Shows:
1. Raw input/output stem signals (before alignment)
2. Aligned signals (after shifting to common zero reference)
3. Normalized signals (0-1 range)
"""

import sys
sys.path.insert(0, 'src')

from data.loaders import DataLoaders
from data.processors import DataProcessor
from data.segmentation import Normalizer
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def visualize_alignment(site_id: int, thermo_id: int, hygro_id: int, dendro_id: int,
                       output_dir: Path, year: int = 2023):
    """
    Create visualization of stem signal alignment for a specific combination.
    
    Args:
        site_id: Site ID
        thermo_id: Thermometer series ID
        hygro_id: Hygrometer series ID
        dendro_id: Dendrometer series ID
        output_dir: Directory to save plots
        year: Year to visualize
    """
    # Initialize data loaders
    data_root = Path('/storage/lukovic/Data/FORWARDS/treenet/server_data')
    meteo_root = data_root / 'meteo_data'
    
    loaders = DataLoaders(data_root, meteo_root)
    processor = DataProcessor()
    metadata = loaders.load_metadata()
    
    print(f'Loading site {site_id}, T{thermo_id}_H{hygro_id}_D{dendro_id}...')
    
    # Load all sensors for site
    site_sensors = loaders.load_all_sensors_for_site(site_id, metadata)
    
    # Process sensors
    temp_df = processor.process_sensor_dataframe(site_sensors['thermometer'][thermo_id])
    rh_df = processor.process_sensor_dataframe(site_sensors['hygrometer'][hygro_id])
    stem_df = processor.process_sensor_dataframe(site_sensors['dendrometer_l2'][dendro_id])
    lm_df = processor.process_sensor_dataframe(site_sensors['dendrometer_lm'][dendro_id], keep_all_columns=True)
    meteo_df = loaders.load_meteotest_data(site_id)
    meteo_daily = processor.process_meteo_daily(meteo_df)
    
    # Create input/output arrays
    input_df = processor.merger.create_input_array(temp_df, rh_df, stem_df, meteo_daily)
    output_df = processor.merger.create_target_array(lm_df)
    
    print(f'Input: {input_df.shape}, Output: {output_df.shape}')
    
    # Filter to specific year
    input_year = input_df[input_df.index.year == year].copy()
    output_year = output_df[output_df.index.year == year].copy()
    
    if len(input_year) == 0 or len(output_year) == 0:
        print(f'No data for year {year}')
        # Find available years
        available_years = sorted(set(input_df.index.year) & set(output_df.index.year))
        print(f'Available years with both input and output: {available_years}')
        if available_years:
            year = available_years[-1]  # Use most recent
            print(f'Using year {year} instead')
            input_year = input_df[input_df.index.year == year].copy()
            output_year = output_df[output_df.index.year == year].copy()
        else:
            return
    
    print(f'Year {year}: Input {len(input_year)} rows, Output {len(output_year)} rows')
    
    # Get raw stem values
    input_stem_raw = input_year['stem'].copy()
    output_stem_raw = output_year['stem'].copy()
    
    # Apply alignment
    normalizer = Normalizer(norm_scope='year')
    (aligned_input_stem, aligned_output_stem,
     stem_input_min, stem_input_diff,
     stem_output_min, stem_output_diff) = normalizer.align_stem_signals_yearly(
        input_stem_raw, output_stem_raw
    )
    
    # Normalize to 0-1
    input_stem_norm = (aligned_input_stem - stem_input_min) / stem_input_diff
    output_stem_norm = (aligned_output_stem - stem_output_min) / stem_output_diff
    
    # Find common time range (where both have valid data)
    input_valid = aligned_input_stem.dropna()
    output_valid = aligned_output_stem.dropna()
    input_hourly = input_valid.groupby(input_valid.index.floor('h')).first()
    common_timestamps = input_hourly.index.intersection(output_valid.index)
    if len(common_timestamps) > 0:
        common_start = common_timestamps.min()
        common_end = common_timestamps.max()
    else:
        common_start = common_end = None
    
    # Print statistics
    print(f'\n=== RAW DATA ===')
    print(f'Input stem: {input_stem_raw.min():.1f} to {input_stem_raw.max():.1f} µm (range: {input_stem_raw.max()-input_stem_raw.min():.1f})')
    print(f'Output stem: {output_stem_raw.min():.1f} to {output_stem_raw.max():.1f} µm (range: {output_stem_raw.max()-output_stem_raw.min():.1f})')
    
    print(f'\n=== ALIGNED DATA ===')
    print(f'Input stem: {aligned_input_stem.min():.1f} to {aligned_input_stem.max():.1f} (shifted)')
    print(f'Output stem: {aligned_output_stem.min():.1f} to {aligned_output_stem.max():.1f} (shifted)')
    if common_start and common_end:
        print(f'Common time range: {common_start} to {common_end}')
        print(f'  ({len(common_timestamps)} common hourly timestamps)')
    print(f'Normalization params (from common range only):')
    print(f'  Input:  min={stem_input_min:.1f}, diff={stem_input_diff:.1f}')
    print(f'  Output: min={stem_output_min:.1f}, diff={stem_output_diff:.1f}')
    
    print(f'\n=== NORMALIZED DATA ===')
    print(f'Input stem: {input_stem_norm.min():.4f} to {input_stem_norm.max():.4f}')
    print(f'Output stem: {output_stem_norm.min():.4f} to {output_stem_norm.max():.4f}')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    
    combo_name = f'T{thermo_id}_H{hygro_id}_D{dendro_id}'
    fig.suptitle(f'Stem Signal Alignment - Site {site_id} {combo_name} ({year})', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Raw data
    ax1 = axes[0]
    ax1.plot(input_stem_raw.index, input_stem_raw.values, 
             color='blue', alpha=0.7, linewidth=0.5, label='Input (L2)')
    ax1.plot(output_stem_raw.index, output_stem_raw.values, 
             color='orange', alpha=0.7, linewidth=0.5, label='Output (LM)')
    ax1.set_ylabel('Stem Radius [µm]')
    ax1.set_title('RAW Data (before alignment)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add text with ranges
    raw_input_range = input_stem_raw.max() - input_stem_raw.min()
    raw_output_range = output_stem_raw.max() - output_stem_raw.min()
    ax1.text(0.02, 0.95, 
             f'Input: {input_stem_raw.min():.0f} to {input_stem_raw.max():.0f} µm (Δ={raw_input_range:.0f})\n'
             f'Output: {output_stem_raw.min():.0f} to {output_stem_raw.max():.0f} µm (Δ={raw_output_range:.0f})',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Aligned data (shifted)
    ax2 = axes[1]
    ax2.plot(aligned_input_stem.index, aligned_input_stem.values, 
             color='blue', alpha=0.7, linewidth=0.5, label='Input (L2) - aligned')
    ax2.plot(aligned_output_stem.index, aligned_output_stem.values, 
             color='orange', alpha=0.7, linewidth=0.5, label='Output (LM) - aligned')
    ax2.set_ylabel('Stem Radius [µm] (shifted)')
    ax2.set_title('ALIGNED Data (shifted to common zero at first common timestamp)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add text with alignment info
    common_info = ''
    if common_start and common_end:
        common_info = f'Common range: {common_start.strftime("%Y-%m-%d")} to {common_end.strftime("%Y-%m-%d")}\n'
    ax2.text(0.02, 0.95, 
             f'Both signals shifted to start from 0 at first common valid timestamp\n'
             f'{common_info}'
             f'Normalization from common range only:\n'
             f'  Input: min={stem_input_min:.1f}, diff={stem_input_diff:.1f}\n'
             f'  Output: min={stem_output_min:.1f}, diff={stem_output_diff:.1f}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add shaded region for common range
    if common_start and common_end:
        ax2.axvspan(common_start, common_end, alpha=0.1, color='green', label='Common range')
        ax2.legend(loc='upper left')
    
    # Plot 3: Normalized data (0-1)
    ax3 = axes[2]
    ax3.plot(input_stem_norm.index, input_stem_norm.values, 
             color='blue', alpha=0.7, linewidth=0.5, label='Input (L2) - normalized')
    ax3.plot(output_stem_norm.index, output_stem_norm.values, 
             color='orange', alpha=0.7, linewidth=0.5, label='Output (LM) - normalized')
    ax3.set_ylabel('Normalized Value [0-1]')
    ax3.set_xlabel('Time')
    ax3.set_title('NORMALIZED Data (0-1 range)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    
    # Calculate variation ratio
    input_var = input_stem_norm.max() - input_stem_norm.min()
    output_var = output_stem_norm.max() - output_stem_norm.min()
    ratio = output_var / input_var if input_var > 0 else float('nan')
    
    ax3.text(0.02, 0.95, 
             f'Normalized using common range values only\n'
             f'Input variation: {input_var:.4f}\n'
             f'Output variation: {output_var:.4f}\n'
             f'Ratio (out/in): {ratio:.4f} (target: ~1.0)',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if 0.8 <= ratio <= 1.2 else 'lightyellow', alpha=0.8))
    
    # Add shaded region for common range on normalized plot too
    if common_start and common_end:
        ax3.axvspan(common_start, common_end, alpha=0.1, color='green')
    
    # Format x-axis
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    plt.tight_layout()
    
    # Save
    filename = f'alignment_site{site_id}_{combo_name}_{year}.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'\nSaved: {filepath}')
    return filepath


def main():
    # Default parameters
    site_id = 51
    thermo_id = 640
    hygro_id = 617
    dendro_id = 667
    year = 2023
    
    output_dir = Path('/home/lukovic/data/treenet/visualizations/alignment')
    
    # Check command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Visualize stem signal alignment')
    parser.add_argument('--site', type=int, default=site_id, help='Site ID')
    parser.add_argument('--thermo', type=int, default=thermo_id, help='Thermometer ID')
    parser.add_argument('--hygro', type=int, default=hygro_id, help='Hygrometer ID')
    parser.add_argument('--dendro', type=int, default=dendro_id, help='Dendrometer ID')
    parser.add_argument('--year', type=int, default=year, help='Year to visualize')
    parser.add_argument('--output', type=str, default=str(output_dir), help='Output directory')
    
    args = parser.parse_args()
    
    visualize_alignment(
        site_id=args.site,
        thermo_id=args.thermo,
        hygro_id=args.hygro,
        dendro_id=args.dendro,
        output_dir=Path(args.output),
        year=args.year
    )


if __name__ == '__main__':
    main()
