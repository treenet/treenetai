#!/usr/bin/env python3
"""
CLI for visualizing yearly RAW vs normalized time series and segments.

Shows:
1. Raw intermediate time series (pre-normalization) to understand data characteristics
2. Segment-level normalized data (what the model actually sees)

Key features:
- Plots full year of raw data to understand value ranges
- Shows input (10-min) vs output (hourly) for 3 channels
- Highlights normalization misalignment between input/output stems
- Optional: Load actual segments to show normalized values

Usage:
    # Plot raw time series for random samples
    python 9_visualize_yearly_normalization.py --n-samples 5 --split train
    
    # Plot with year filter and show segment comparison
    python 9_visualize_yearly_normalization.py --n-samples 3 --year 2023 --show-segments
    
    # Plot a specific site/sensor combination
    python 9_visualize_yearly_normalization.py --site 3 --thermo 9 --hygro 7 --dendro 18
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import setup_logging


def get_logger():
    """Get logger for this script."""
    setup_logging(verbose=True)
    return logging.getLogger(__name__)


# Constants
INPUT_CHANNELS = ['temp_treenet', 'rh_treenet', 'stem']  # Local sensor channels (0, 1, 2)
OUTPUT_CHANNELS = ['local_T', 'local_RH', 'stem']  # Output channels


class YearlyNormalizationPlotter:
    """Visualize yearly normalized input vs output time series."""
    
    def __init__(self, data_dir: Path, output_dir: Path = None):
        """
        Initialize plotter.
        
        Args:
            data_dir: Path to model_data directory containing intermediate_timeseries
            output_dir: Path for output images (default: /home/lukovic/data/treenet/visualizations)
        """
        self.data_dir = Path(data_dir)
        self.ts_dir = self.data_dir / 'intermediate_timeseries'
        
        if output_dir is None:
            self.output_dir = Path('/home/lukovic/data/treenet/visualizations/yearly_normalization')
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Channel display names
        self.channel_names = {
            'temp_treenet': 'Temperature (Local)',
            'rh_treenet': 'Relative Humidity (Local)',
            'stem': 'Stem Radius',
            'local_T': 'Temperature (Target)',
            'local_RH': 'Relative Humidity (Target)'
        }
        
        # Colors
        self.input_color = '#1f77b4'   # Blue
        self.output_color = '#ff7f0e'  # Orange
    
    def list_available_combinations(self, split: str = 'train') -> list:
        """
        List all available input/output file pairs.
        
        Returns:
            List of tuples (site_id, thermo_id, hygro_id, dendro_id)
        """
        combinations = []
        pattern = f'{split}_input_site*_T*_H*_D*.ftr'
        
        for input_file in self.ts_dir.glob(pattern):
            # Parse filename: train_input_site3_T9_H7_D18.ftr
            parts = input_file.stem.split('_')
            site_id = int(parts[2].replace('site', ''))
            thermo_id = int(parts[3].replace('T', ''))
            hygro_id = int(parts[4].replace('H', ''))
            dendro_id = int(parts[5].replace('D', ''))
            
            # Check output file exists
            output_file = self.ts_dir / f'{split}_output_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}.ftr'
            if output_file.exists():
                combinations.append((site_id, thermo_id, hygro_id, dendro_id))
        
        return combinations
    
    def load_timeseries(self, site_id: int, thermo_id: int, hygro_id: int, 
                        dendro_id: int, split: str = 'train') -> tuple:
        """
        Load input and output time series for a combination.
        
        Returns:
            (input_df, output_df) or (None, None) if files don't exist
        """
        input_file = self.ts_dir / f'{split}_input_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}.ftr'
        output_file = self.ts_dir / f'{split}_output_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}.ftr'
        
        if not input_file.exists() or not output_file.exists():
            return None, None
        
        input_df = pd.read_feather(input_file)
        output_df = pd.read_feather(output_file)
        
        # Ensure 'ts' column is datetime
        input_df['ts'] = pd.to_datetime(input_df['ts'])
        output_df['ts'] = pd.to_datetime(output_df['ts'])
        
        return input_df, output_df
    
    def plot_yearly_comparison(self, input_df: pd.DataFrame, output_df: pd.DataFrame,
                               site_id: int, thermo_id: int, hygro_id: int, 
                               dendro_id: int, year: int = None,
                               save_path: Path = None) -> Path:
        """
        Plot yearly RAW (unnormalized) input vs output for 3 channels.
        
        NOTE: The intermediate time series files contain RAW data before normalization.
        This plot shows the actual sensor values to help identify:
        - Scale differences between input and output (especially stem)
        - Data quality issues
        - Temporal coverage
        
        Args:
            input_df: Input time series (10-min, 11 channels) - RAW
            output_df: Output time series (hourly, 3 channels) - RAW
            site_id, thermo_id, hygro_id, dendro_id: Sensor IDs
            year: Year to filter (if None, uses all data)
            save_path: Output file path (if None, auto-generated)
        
        Returns:
            Path to saved figure
        """
        # Filter by year if specified
        if year is not None:
            input_df = input_df[input_df['ts'].dt.year == year].copy()
            output_df = output_df[output_df['ts'].dt.year == year].copy()
            
            if len(input_df) == 0 or len(output_df) == 0:
                return None
        
        # Determine years in data
        years = sorted(input_df['ts'].dt.year.unique())
        year_str = str(year) if year else f"{min(years)}-{max(years)}"
        
        # Create figure with 3 rows (one per channel)
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(f'RAW (Pre-Normalization) Data: Site {site_id} (T{thermo_id}_H{hygro_id}_D{dendro_id})\n'
                     f'Year: {year_str}\n'
                     f'NOTE: These are RAW values before normalization - values NOT in [0,1]', 
                     fontsize=14, fontweight='bold')
        
        # Channel pairs: (input_col, output_col, title)
        channel_pairs = [
            ('temp_treenet', 'local_T', 'Temperature'),
            ('rh_treenet', 'local_RH', 'Relative Humidity'),
            ('stem', 'stem', 'Stem Radius')
        ]
        
        for ax, (input_col, output_col, title) in zip(axes, channel_pairs):
            # Plot input (10-min)
            ax.plot(input_df['ts'], input_df[input_col], 
                    color=self.input_color, alpha=0.6, linewidth=0.5,
                    label=f'Input (10-min)')
            
            # Plot output (hourly) - thicker line on top
            ax.plot(output_df['ts'], output_df[output_col],
                    color=self.output_color, alpha=0.8, linewidth=1.0,
                    label=f'Target (hourly)')
            
            # Calculate statistics
            input_valid = input_df[input_col].dropna()
            output_valid = output_df[output_col].dropna()
            
            stats_text = (f'Input: min={input_valid.min():.3f}, max={input_valid.max():.3f}, '
                         f'range={input_valid.max() - input_valid.min():.3f}\n'
                         f'Target: min={output_valid.min():.3f}, max={output_valid.max():.3f}, '
                         f'range={output_valid.max() - output_valid.min():.3f}')
            
            ax.set_ylabel(f'{title}\n(raw units)', fontsize=10)
            ax.set_title(f'{title}: Input vs Target (RAW)', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add stats box - show raw statistics
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # For stem channel, highlight the scale difference
            if title == 'Stem Radius':
                input_range = input_valid.max() - input_valid.min()
                output_range = output_valid.max() - output_valid.min()
                scale_ratio = input_range / (output_range + 1e-10)
                warning = f'⚠️ Scale ratio: {scale_ratio:.2f}' if abs(scale_ratio - 1) > 0.2 else '✓ Similar scales'
                ax.text(0.98, 0.95, warning, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', ha='right',
                       bbox=dict(boxstyle='round', 
                                facecolor='red' if abs(scale_ratio - 1) > 0.2 else 'lightgreen', 
                                alpha=0.8))
        
        # Format x-axis
        axes[-1].set_xlabel('Date', fontsize=11)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f'raw_data_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}_{year_str}.png'
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def plot_stem_alignment_detail(self, input_df: pd.DataFrame, output_df: pd.DataFrame,
                                   site_id: int, thermo_id: int, hygro_id: int,
                                   dendro_id: int, year: int = None,
                                   save_path: Path = None) -> Path:
        """
        Create detailed plot focusing on stem signal alignment.
        
        Shows:
        - Full year stem signals (RAW values)
        - Zoomed view of first month (alignment region)
        - Histogram of value distributions
        - Scale statistics
        
        NOTE: Shows RAW data from intermediate files, NOT normalized.
        """
        # Filter by year if specified
        if year is not None:
            input_df = input_df[input_df['ts'].dt.year == year].copy()
            output_df = output_df[output_df['ts'].dt.year == year].copy()
            
            if len(input_df) == 0 or len(output_df) == 0:
                return None
        
        years = sorted(input_df['ts'].dt.year.unique())
        year_str = str(year) if year else f"{min(years)}-{max(years)}"
        
        # Create figure with 2x2 layout
        fig = plt.figure(figsize=(16, 10))
        
        # Full year view
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(input_df['ts'], input_df['stem'], 
                color=self.input_color, alpha=0.6, linewidth=0.5, label='Input stem')
        ax1.plot(output_df['ts'], output_df['stem'],
                color=self.output_color, alpha=0.8, linewidth=1.0, label='Target stem')
        ax1.set_title('Full Year: Stem Radius (RAW VALUES)', fontsize=11)
        ax1.set_ylabel('Raw value (micrometers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Zoomed view of first month
        ax2 = fig.add_subplot(2, 2, 2)
        first_month = input_df['ts'].min() + pd.Timedelta(days=30)
        input_zoom = input_df[input_df['ts'] <= first_month]
        output_zoom = output_df[output_df['ts'] <= first_month]
        
        ax2.plot(input_zoom['ts'], input_zoom['stem'],
                color=self.input_color, alpha=0.7, linewidth=0.8, label='Input stem')
        ax2.plot(output_zoom['ts'], output_zoom['stem'],
                color=self.output_color, alpha=0.9, linewidth=1.2, label='Target stem')
        ax2.set_title('First 30 Days: Stem Values Check', fontsize=11)
        ax2.set_ylabel('Raw value (micrometers)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Histogram comparison
        ax3 = fig.add_subplot(2, 2, 3)
        input_stem = input_df['stem'].dropna()
        output_stem = output_df['stem'].dropna()
        
        bins = np.linspace(min(input_stem.min(), output_stem.min()),
                          max(input_stem.max(), output_stem.max()), 50)
        
        ax3.hist(input_stem, bins=bins, alpha=0.5, label='Input', color=self.input_color)
        ax3.hist(output_stem, bins=bins, alpha=0.5, label='Target', color=self.output_color)
        ax3.set_title('Value Distribution: Stem Radius (RAW)', fontsize=11)
        ax3.set_xlabel('Raw value (micrometers)')
        ax3.set_ylabel('Count')
        ax3.legend()
        
        # Statistics box
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        input_range = input_stem.max() - input_stem.min()
        output_range = output_stem.max() - output_stem.min()
        range_ratio = input_range / (output_range + 1e-10)
        offset_diff = abs(input_stem.iloc[0] - output_stem.iloc[0])
        
        # Color-code the alignment status
        alignment_status = "⚠️ MISALIGNED" if (abs(range_ratio - 1) > 0.3 or offset_diff > 5000) else "✓ ALIGNED"
        alignment_color = 'red' if "MISALIGNED" in alignment_status else 'green'
        
        stats = f"""
RAW Stem Signal Statistics
{'='*40}

INPUT (dendrometer_l2, 10-min):
  Min:   {input_stem.min():.1f} µm
  Max:   {input_stem.max():.1f} µm
  Range: {input_range:.1f} µm
  Mean:  {input_stem.mean():.1f} µm

TARGET (dendrometer_lm, hourly):
  Min:   {output_stem.min():.1f} µm
  Max:   {output_stem.max():.1f} µm
  Range: {output_range:.1f} µm
  Mean:  {output_stem.mean():.1f} µm

ALIGNMENT CHECK:
  Range ratio:  {range_ratio:.2f} (ideal: 1.0)
  Offset diff:  {offset_diff:.1f} µm
  Status: {alignment_status}

NOTE: These are RAW values before normalization.
After normalization, both will be in [0, 1].
The PROBLEM is when they have different offsets
or different ranges, leading to different
normalized values for the same physical state.
"""
        ax4.text(0.05, 0.95, stats, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        fig.suptitle(f'RAW Stem Data Analysis: Site {site_id} (T{thermo_id}_H{hygro_id}_D{dendro_id})\n'
                    f'Year: {year_str}  |  {alignment_status}', 
                    fontsize=14, fontweight='bold', color=alignment_color)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f'stem_raw_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}_{year_str}.png'
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize yearly normalized input vs output time series',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode: specific combination or random samples
    parser.add_argument(
        '--site', type=int, help='Site ID'
    )
    parser.add_argument(
        '--thermo', type=int, help='Thermometer series ID'
    )
    parser.add_argument(
        '--hygro', type=int, help='Hygrometer series ID'
    )
    parser.add_argument(
        '--dendro', type=int, help='Dendrometer series ID'
    )
    parser.add_argument(
        '--n-samples', type=int, default=5,
        help='Number of random samples to plot (default: 5)'
    )
    parser.add_argument(
        '--split', type=str, default='train', choices=['train', 'test'],
        help='Dataset split (default: train)'
    )
    parser.add_argument(
        '--year', type=int, help='Filter to specific year'
    )
    parser.add_argument(
        '--data-dir', type=Path,
        default=Path('/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/model_data'),
        help='Path to model_data directory'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--stem-detail', action='store_true',
        help='Generate additional stem alignment detail plots'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sample selection'
    )
    
    return parser.parse_args()


def main():
    """Main visualization workflow."""
    args = parse_args()
    
    # Setup logging
    logger = get_logger()
    logger.info("Starting yearly normalization visualization")
    
    # Initialize plotter
    plotter = YearlyNormalizationPlotter(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Determine which combinations to plot
    if args.site and args.thermo and args.hygro and args.dendro:
        # Specific combination
        combinations = [(args.site, args.thermo, args.hygro, args.dendro)]
        logger.info(f"Plotting specific combination: site{args.site}_T{args.thermo}_H{args.hygro}_D{args.dendro}")
    else:
        # Random samples
        all_combos = plotter.list_available_combinations(args.split)
        logger.info(f"Found {len(all_combos)} combinations in {args.split} set")
        
        random.seed(args.seed)
        combinations = random.sample(all_combos, min(args.n_samples, len(all_combos)))
        logger.info(f"Selected {len(combinations)} random samples")
    
    # Generate plots
    generated_files = []
    
    for site_id, thermo_id, hygro_id, dendro_id in combinations:
        logger.info(f"Processing site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}")
        
        # Load data
        input_df, output_df = plotter.load_timeseries(
            site_id, thermo_id, hygro_id, dendro_id, args.split
        )
        
        if input_df is None:
            logger.warning(f"  Skipping - files not found")
            continue
        
        # Determine years to process
        if args.year:
            years = [args.year]
        else:
            years = sorted(input_df['ts'].dt.year.unique())
        
        for year in years:
            # Main comparison plot
            path = plotter.plot_yearly_comparison(
                input_df, output_df,
                site_id, thermo_id, hygro_id, dendro_id,
                year=year
            )
            
            if path:
                generated_files.append(path)
                logger.info(f"  Created: {path.name}")
            
            # Stem detail plot (optional)
            if args.stem_detail:
                path = plotter.plot_stem_alignment_detail(
                    input_df, output_df,
                    site_id, thermo_id, hygro_id, dendro_id,
                    year=year
                )
                
                if path:
                    generated_files.append(path)
                    logger.info(f"  Created: {path.name}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Generated {len(generated_files)} plots")
    logger.info(f"Output directory: {plotter.output_dir}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
