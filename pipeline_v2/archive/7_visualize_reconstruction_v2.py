#!/usr/bin/env python3
"""
Visualize reconstructed time series from PATH 1 reconstruction.

Creates multi-panel plots showing:
- Original LM data (ground truth where available)
- Reconstructed data
- Comparison for validation periods

Usage:
    python 7_visualize_reconstruction_v2.py \
        --recon-path /home/lukovic/data/treenet/reconstructions/reconstructed_site3_T9_H7_D18.ftr \
        --site-id 3 \
        --thermo-id 9 \
        --hygro-id 7 \
        --dendro-id 18 \
        --years 2023 2024 \
        --output-dir /home/lukovic/data/treenet/visualizations/reconstructions

Author: Lukovic
Date: 2026-01-11
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.feather as feather

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor
from src.utils import ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize reconstructed time series'
    )
    
    parser.add_argument(
        '--recon-path', type=str, required=True,
        help='Path to reconstructed data (.ftr file)'
    )
    parser.add_argument(
        '--site-id', type=int, required=True,
        help='Site ID'
    )
    parser.add_argument(
        '--thermo-id', type=int, required=True,
        help='Thermometer series ID'
    )
    parser.add_argument(
        '--hygro-id', type=int, required=True,
        help='Hygrometer series ID'
    )
    parser.add_argument(
        '--dendro-id', type=int, required=True,
        help='Dendrometer series ID'
    )
    parser.add_argument(
        '--years', type=int, nargs='+', default=None,
        help='Years to visualize (default: all available)'
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data',
        help='Root directory with raw sensor data'
    )
    parser.add_argument(
        '--meteo-root', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data',
        help='Directory with meteo CSV files'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/home/lukovic/data/treenet/visualizations/reconstructions',
        help='Output directory for plots'
    )
    
    return parser.parse_args()


def load_lm_data(loaders: DataLoaders, processor: DataProcessor, dendro_id: int) -> Optional[pd.DataFrame]:
    """Load and process LM data for comparison."""
    lm_raw = loaders.load_dendrometer_lm(dendro_id)
    if lm_raw is None:
        return None
    
    # Process to get local_T, local_RH, stem columns
    lm_processed = processor.process_sensor_dataframe(lm_raw, keep_all_columns=True)
    lm_df = processor.merger.create_target_array(lm_processed)
    
    return lm_df


def plot_year(
    recon_df: pd.DataFrame,
    lm_df: Optional[pd.DataFrame],
    year: int,
    site_id: int,
    sensor_ids: dict,
    output_dir: Path
) -> Path:
    """
    Create a multi-panel plot for one year.
    
    Returns path to saved figure.
    """
    # Filter to year
    recon_year = recon_df[recon_df.index.year == year]
    if lm_df is not None:
        lm_year = lm_df[lm_df.index.year == year]
    else:
        lm_year = None
    
    if len(recon_year) == 0:
        print(f"  No data for year {year}")
        return None
    
    # Create figure with 3 subplots (T, RH, stem)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    channels = ['local_T', 'local_RH', 'stem']
    titles = ['Temperature (°C)', 'Relative Humidity (%)', 'Stem Radius (µm)']
    colors = {'recon': '#2196F3', 'lm': '#4CAF50'}
    
    for ax, channel, title in zip(axes, channels, titles):
        # Plot reconstructed
        ax.plot(
            recon_year.index, 
            recon_year[channel],
            color=colors['recon'],
            alpha=0.8,
            linewidth=0.5,
            label='Reconstructed'
        )
        
        # Plot LM if available
        if lm_year is not None and channel in lm_year.columns:
            ax.plot(
                lm_year.index,
                lm_year[channel],
                color=colors['lm'],
                alpha=0.6,
                linewidth=0.5,
                label='LM (Ground Truth)'
            )
        
        ax.set_ylabel(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Calculate stats for annotation
        if lm_year is not None and channel in lm_year.columns:
            # Find common indices
            common_idx = recon_year.index.intersection(lm_year.index)
            if len(common_idx) > 0:
                recon_vals = recon_year.loc[common_idx, channel]
                lm_vals = lm_year.loc[common_idx, channel]
                valid = ~(recon_vals.isna() | lm_vals.isna())
                if valid.sum() > 0:
                    corr = np.corrcoef(recon_vals[valid], lm_vals[valid])[0, 1]
                    mae = np.abs(recon_vals[valid] - lm_vals[valid]).mean()
                    ax.annotate(
                        f'Corr: {corr:.3f}, MAE: {mae:.2f}',
                        xy=(0.02, 0.95),
                        xycoords='axes fraction',
                        fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
    
    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].set_xlabel(f'{year}')
    
    # Title
    fig.suptitle(
        f'Reconstructed Time Series - Site {site_id} (T={sensor_ids["thermo"]}, '
        f'H={sensor_ids["hygro"]}, D={sensor_ids["dendro"]}) - {year}',
        fontsize=12
    )
    
    plt.tight_layout()
    
    # Save
    filename = f'reconstruction_site{site_id}_T{sensor_ids["thermo"]}_H{sensor_ids["hygro"]}_D{sensor_ids["dendro"]}_{year}.png'
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = ensure_dir(Path(args.output_dir))
    
    print("="*80)
    print("TreeNet AI - Reconstruction Visualization")
    print("="*80)
    
    # Load reconstructed data
    print(f"\nLoading reconstructed data from: {args.recon_path}")
    recon_df = feather.read_feather(args.recon_path)
    
    # Set index if needed
    if 'ts' in recon_df.columns:
        recon_df = recon_df.set_index('ts')
    
    # Ensure timezone-aware
    if recon_df.index.tz is None:
        recon_df.index = recon_df.index.tz_localize('UTC')
    
    print(f"  Loaded: {len(recon_df):,} samples")
    print(f"  Date range: {recon_df.index.min()} to {recon_df.index.max()}")
    
    # Load LM data for comparison
    print("\nLoading LM data for comparison...")
    loaders = DataLoaders(
        data_root=Path(args.data_dir),
        meteo_root=Path(args.meteo_root)
    )
    processor = DataProcessor()
    lm_df = load_lm_data(loaders, processor, args.dendro_id)
    
    if lm_df is not None:
        print(f"  Loaded: {len(lm_df):,} samples")
    else:
        print("  No LM data available for comparison")
    
    # Determine years to plot
    available_years = sorted(recon_df.index.year.unique())
    if args.years:
        years = [y for y in args.years if y in available_years]
    else:
        years = available_years
    
    print(f"\nYears to visualize: {years}")
    
    # Sensor IDs
    sensor_ids = {
        'thermo': args.thermo_id,
        'hygro': args.hygro_id,
        'dendro': args.dendro_id
    }
    
    # Create plots for each year
    print("\nGenerating plots...")
    for year in years:
        print(f"  Year {year}...")
        output_path = plot_year(
            recon_df, lm_df, year, args.site_id, sensor_ids, output_dir
        )
        if output_path:
            print(f"    Saved: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"Visualization complete! Output: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
