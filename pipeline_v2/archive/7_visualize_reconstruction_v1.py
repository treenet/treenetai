#!/usr/bin/env python3
"""
Visualize gap-filling results by comparing original and reconstructed time series.

Creates before/after comparison plots highlighting filled gaps.

Usage:
    python 7_visualize_reconstruction.py \
        --original-path /path/to/original.ftr \
        --reconstructed-path /path/to/reconstructed.ftr \
        --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize gap-filling reconstruction results'
    )
    
    parser.add_argument(
        '--original-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/processed/model_data/intermediate_timeseries',
        help='Directory with original time series .ftr files'
    )
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data',
        help='Directory with raw server data (for gap detection)'
    )
    parser.add_argument(
        '--reconstructed-path',
        type=str,
        required=True,
        help='Path to reconstructed time series .ftr file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lukovic/data/treenet/visualizations/reconstruction',
        help='Output directory for plots (default: /home/lukovic/data/treenet/visualizations/reconstruction)'
    )
    parser.add_argument(
        '--site-id',
        type=int,
        default=3,
        help='Site ID'
    )
    parser.add_argument(
        '--thermo-id',
        type=int,
        default=9,
        help='Thermometer series ID'
    )
    parser.add_argument(
        '--hygro-id',
        type=int,
        default=7,
        help='Hygrometer series ID'
    )
    parser.add_argument(
        '--dendro-id',
        type=int,
        default=18,
        help='Dendrometer series ID'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for visualization (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for visualization (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--show-gaps',
        action='store_true',
        default=True,
        help='Highlight filled gap regions'
    )
    
    return parser.parse_args()


def find_timestamp_gaps(df: pd.DataFrame, expected_freq_minutes: int = 10, min_gap_hours: int = 2) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Find gaps based on missing timestamps (not NaN values).
    
    Args:
        df: DataFrame with DatetimeIndex
        expected_freq_minutes: Expected frequency in minutes
        min_gap_hours: Minimum gap duration in hours
        
    Returns:
        List of (start, end) timestamp tuples for each gap
    """
    expected = pd.Timedelta(minutes=expected_freq_minutes)
    gaps = []
    
    for idx in range(1, len(df)):
        diff = df.index[idx] - df.index[idx - 1]
        if diff > expected * 2:  # More than expected interval
            gap_hours = diff.total_seconds() / 3600
            if gap_hours >= min_gap_hours:
                gaps.append((df.index[idx - 1], df.index[idx]))
    
    return gaps


def load_raw_sensor_data(
    raw_data_dir: Path,
    dendro_id: int
) -> Optional[pd.DataFrame]:
    """
    Load raw dendrometer data to detect actual gaps.
    
    Args:
        raw_data_dir: Path to server_data directory
        dendro_id: Dendrometer series ID
        
    Returns:
        DataFrame with timestamp index, or None if not found
    """
    dendro_path = raw_data_dir / 'dendrometer_l2' / f'dendrometer_l2_series_id_{dendro_id}.ftr'
    if not dendro_path.exists():
        return None
    
    df = pd.read_feather(dendro_path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts')
    return df


def find_gaps(df: pd.DataFrame, column: str, min_gap_hours: int = 2) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Find gaps in a time series column.
    
    Args:
        df: DataFrame with DatetimeIndex
        column: Column name to check for gaps
        min_gap_hours: Minimum gap duration in hours
        
    Returns:
        List of (start, end) timestamp tuples for each gap
    """
    if column not in df.columns:
        return []
    
    # Find NaN values
    is_nan = df[column].isna()
    
    if not is_nan.any():
        return []
    
    # Find gap boundaries
    gap_starts = []
    gap_ends = []
    in_gap = False
    
    for idx in range(len(df)):
        if is_nan.iloc[idx] and not in_gap:
            gap_starts.append(df.index[idx])
            in_gap = True
        elif not is_nan.iloc[idx] and in_gap:
            gap_ends.append(df.index[idx])
            in_gap = False
    
    # Handle gap at end
    if in_gap:
        gap_ends.append(df.index[-1])
    
    # Filter by minimum duration
    gaps = []
    for start, end in zip(gap_starts, gap_ends):
        duration = (end - start).total_seconds() / 3600
        if duration >= min_gap_hours:
            gaps.append((start, end))
    
    return gaps


def create_comparison_plot(
    original_df: pd.DataFrame,
    reconstructed_df: pd.DataFrame,
    column: str,
    title: str,
    output_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_gaps: bool = True,
    ylabel: str = 'Value'
):
    """
    Create a before/after comparison plot.
    
    Args:
        original_df: Original time series with gaps
        reconstructed_df: Reconstructed time series
        column: Column to plot
        title: Plot title
        output_path: Path to save the plot
        start_date: Start date filter
        end_date: End date filter
        show_gaps: Whether to highlight gap regions
        ylabel: Y-axis label
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Filter date range if specified
    orig = original_df.copy()
    recon = reconstructed_df.copy()
    
    if start_date:
        start = pd.Timestamp(start_date, tz='UTC')
        orig = orig[orig.index >= start]
        recon = recon[recon.index >= start]
    
    if end_date:
        end = pd.Timestamp(end_date, tz='UTC')
        orig = orig[orig.index <= end]
        recon = recon[recon.index <= end]
    
    # Find gaps in original
    gaps = find_gaps(orig, column)
    
    # Plot original (top)
    ax1 = axes[0]
    ax1.plot(orig.index, orig[column], 'b-', linewidth=0.5, alpha=0.7, label='Original')
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{title} - Original (with gaps)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Highlight gaps
    if show_gaps:
        for gap_start, gap_end in gaps:
            ax1.axvspan(gap_start, gap_end, color='red', alpha=0.2, label='_nolegend_')
    
    # Plot reconstructed (bottom)
    ax2 = axes[1]
    ax2.plot(recon.index, recon[column], 'g-', linewidth=0.5, alpha=0.7, label='Reconstructed')
    ax2.set_ylabel(ylabel)
    ax2.set_title(f'{title} - Reconstructed (gaps filled)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Highlight filled regions
    if show_gaps:
        for gap_start, gap_end in gaps:
            ax2.axvspan(gap_start, gap_end, color='green', alpha=0.2, label='_nolegend_')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_gap_zoom_plot(
    original_df: pd.DataFrame,
    reconstructed_df: pd.DataFrame,
    column: str,
    gap_idx: int,
    title: str,
    output_path: Path,
    context_days: int = 5,
    ylabel: str = 'Value',
    freq_minutes: int = None
):
    """
    Create a zoomed view of a specific gap.
    
    Args:
        original_df: Original time series
        reconstructed_df: Reconstructed time series
        column: Column to plot
        gap_idx: Index of the gap to zoom into
        title: Plot title
        output_path: Path to save the plot
        context_days: Days of context before/after gap
        ylabel: Y-axis label
        freq_minutes: Expected frequency (None=auto-detect)
    """
    # Auto-detect frequency
    if freq_minutes is None:
        # Check by index density
        time_range = (original_df.index[-1] - original_df.index[0]).total_seconds() / 3600
        freq_minutes = int(time_range / len(original_df) * 60)
        freq_minutes = max(10, min(60, freq_minutes))  # Clamp between 10-60
    
    # Find gaps - try timestamp gaps first, then NaN gaps
    gaps = find_timestamp_gaps(original_df, expected_freq_minutes=freq_minutes, min_gap_hours=2)
    if not gaps:
        gaps = find_gaps(original_df, column)
    
    if gap_idx >= len(gaps):
        print(f"  Warning: Gap index {gap_idx} out of range (found {len(gaps)} gaps)")
        return
    
    gap_start, gap_end = gaps[gap_idx]
    
    # Expand window
    view_start = gap_start - timedelta(days=context_days)
    view_end = gap_end + timedelta(days=context_days)
    
    # Filter data
    orig = original_df[(original_df.index >= view_start) & (original_df.index <= view_end)]
    recon = reconstructed_df[(reconstructed_df.index >= view_start) & (reconstructed_df.index <= view_end)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot original
    ax.plot(orig.index, orig[column], 'b-', linewidth=1.5, alpha=0.7, label='Original')
    
    # Plot reconstructed
    ax.plot(recon.index, recon[column], 'g--', linewidth=1.5, alpha=0.7, label='Reconstructed')
    
    # Highlight gap region
    ax.axvspan(gap_start, gap_end, color='yellow', alpha=0.3, label='Filled gap')
    
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} - Gap {gap_idx + 1} ({gap_start.strftime("%Y-%m-%d")} to {gap_end.strftime("%Y-%m-%d")})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Gap-Filling Visualization")
    print("="*80)
    
    # Load reconstructed data
    print(f"\nLoading reconstructed data: {args.reconstructed_path}")
    recon_df = pd.read_feather(args.reconstructed_path)
    recon_df['ts'] = pd.to_datetime(recon_df['ts'])
    recon_df = recon_df.set_index('ts')
    print(f"  Shape: {recon_df.shape}")
    print(f"  Columns: {recon_df.columns.tolist()}")
    
    # Try to load original output data
    orig_filename = f"test_output_site{args.site_id}_T{args.thermo_id}_H{args.hygro_id}_D{args.dendro_id}.ftr"
    orig_path = Path(args.original_dir) / orig_filename
    
    if not orig_path.exists():
        # Try train prefix
        orig_filename = f"train_output_site{args.site_id}_T{args.thermo_id}_H{args.hygro_id}_D{args.dendro_id}.ftr"
        orig_path = Path(args.original_dir) / orig_filename
    
    if orig_path.exists():
        print(f"\nLoading original data: {orig_path}")
        orig_df = pd.read_feather(orig_path)
        orig_df['ts'] = pd.to_datetime(orig_df['ts'])
        orig_df = orig_df.set_index('ts')
        print(f"  Shape: {orig_df.shape}")
        print(f"  Columns: {orig_df.columns.tolist()}")
    else:
        print(f"\nWarning: Original file not found at {orig_path}")
        print("  Creating synthetic original by adding NaN gaps to reconstruction")
        # Use reconstructed as original (not ideal but shows structure)
        orig_df = recon_df.copy()
    
    # Map column names
    column_mapping = {
        'stem': ('stem', 'Stem Radius Change (μm)'),
        'local_T': ('local_T', 'Temperature (°C)'),
        'local_RH': ('local_RH', 'Relative Humidity (%)'),
    }
    
    print("\n" + "="*80)
    print("Creating comparison plots...")
    print("="*80)
    
    # Create full comparison plots
    for col, (col_name, ylabel) in column_mapping.items():
        if col_name in orig_df.columns and col_name in recon_df.columns:
            output_path = output_dir / f"comparison_{col}_site{args.site_id}.png"
            create_comparison_plot(
                original_df=orig_df,
                reconstructed_df=recon_df,
                column=col_name,
                title=f"Site {args.site_id} - {col_name}",
                output_path=output_path,
                start_date=args.start_date,
                end_date=args.end_date,
                show_gaps=args.show_gaps,
                ylabel=ylabel
            )
    
    # Create zoomed gap plots for stem - use ORIGINAL data to find gaps
    print("\nCreating gap zoom plots...")
    
    # Try timestamp-based gaps first (more reliable)
    stem_gaps = find_timestamp_gaps(orig_df, expected_freq_minutes=60, min_gap_hours=2)
    print(f"  Found {len(stem_gaps)} timestamp gaps in ORIGINAL stem column (hourly)")
    
    if len(stem_gaps) == 0:
        # Try NaN-based gaps
        stem_gaps = find_gaps(orig_df, 'stem')
        print(f"  Found {len(stem_gaps)} NaN gaps in ORIGINAL stem column")
    
    # If no gaps in output data, try loading the input data which has 10-min resolution
    if len(stem_gaps) == 0:
        print("  No gaps in output data. Trying to load raw dendrometer data...")
        
        # First try raw dendrometer data
        raw_df = load_raw_sensor_data(Path(args.raw_data_dir), args.dendro_id)
        if raw_df is not None:
            stem_gaps = find_timestamp_gaps(raw_df, expected_freq_minutes=10, min_gap_hours=2)
            print(f"  Found {len(stem_gaps)} timestamp gaps in RAW dendrometer data")
            orig_df_for_gaps = raw_df
        else:
            # Fall back to intermediate input data
            input_filename = f"test_input_site{args.site_id}_T{args.thermo_id}_H{args.hygro_id}_D{args.dendro_id}.ftr"
            input_path = Path(args.original_dir) / input_filename
            if input_path.exists():
                input_df = pd.read_feather(input_path)
                input_df['ts'] = pd.to_datetime(input_df['ts'])
                input_df = input_df.set_index('ts')
                stem_gaps = find_timestamp_gaps(input_df, expected_freq_minutes=10, min_gap_hours=2)
                print(f"  Found {len(stem_gaps)} timestamp gaps in INPUT data (10-min)")
                orig_df_for_gaps = input_df
            else:
                orig_df_for_gaps = orig_df
    else:
        orig_df_for_gaps = orig_df
    
    for i, (gap_start, gap_end) in enumerate(stem_gaps[:5]):  # First 5 gaps
        gap_duration = (gap_end - gap_start).total_seconds() / 3600
        print(f"  Gap {i+1}: {gap_start} to {gap_end} ({gap_duration:.1f} hours)")
        
        output_path = output_dir / f"gap_zoom_{i+1}_site{args.site_id}.png"
        create_gap_zoom_plot(
            original_df=orig_df_for_gaps,
            reconstructed_df=recon_df,
            column='stem',
            gap_idx=i,
            title=f"Site {args.site_id} - Stem",
            output_path=output_path,
            context_days=5,
            ylabel='Stem Radius Change (μm)'
        )
    
    print("\n" + "="*80)
    print(f"Visualization complete!")
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
