#!/usr/bin/env python3
"""
CLI for visualizing processed segments.

Creates plots to validate segment extraction and data quality:
- Individual segment plots (inputs + targets)
- Summary statistics across all segments
- Per-site inspection for specific years

Usage:
    # Plot segments for specific site and year
    python 4_visualize_segments.py --site 1 --year 2021 --split train
    
    # Generate summary statistics
    python 4_visualize_segments.py --summary --split train
    
    # Plot all segments (warning: may create many files)
    python 4_visualize_segments.py --all --split train --max-per-site 5
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PipelineConfig
from visualization.plot_segments import SegmentPlotter
from utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize processed segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data selection
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test'],
        help='Dataset split to visualize (default: train)'
    )
    
    # Plotting modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--site',
        type=int,
        help='Plot segments for specific site ID'
    )
    mode_group.add_argument(
        '--summary',
        action='store_true',
        help='Generate summary statistics only'
    )
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Plot all segments (warning: may create many files)'
    )
    
    # Additional options
    parser.add_argument(
        '--year',
        type=int,
        help='Year to filter (required with --site)'
    )
    parser.add_argument(
        '--max-per-site',
        type=int,
        default=10,
        help='Maximum segments to plot per site (default: 10)'
    )
    parser.add_argument(
        '--no-globals',
        action='store_true',
        help='Do not overlay global channels on input plots'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Override data directory from config'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.site is not None and args.year is None:
        parser.error('--year is required when using --site')
    
    return args


def main():
    """Main visualization workflow."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging('visualization')
    logger.info("Starting segment visualization")
    
    # Load configuration
    config = PipelineConfig()
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = Path(config.paths.model_data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.error("Please run 1_build_segments.py first")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.paths.model_data_dir) / 'visualizations' / args.split
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize plotter
    plotter = SegmentPlotter(local_tz=config.preprocessing.local_timezone)
    
    # Execute visualization mode
    if args.summary:
        logger.info(f"Generating summary statistics for {args.split} set")
        output_path = output_dir / 'summary_statistics.png'
        plotter.plot_summary_stats(
            data_dir=data_dir,
            output_path=output_path,
            split=args.split
        )
        logger.info(f"Summary saved to: {output_path}")
    
    elif args.site is not None:
        logger.info(f"Plotting segments for site {args.site}, year {args.year}")
        site_output_dir = output_dir / f'site_{args.site}' / f'year_{args.year}'
        
        n_plots = plotter.plot_segments_for_site(
            data_dir=data_dir,
            site_id=args.site,
            year=args.year,
            output_dir=site_output_dir,
            split=args.split,
            max_segments=args.max_per_site
        )
        
        if n_plots == 0:
            logger.warning(f"No segments found for site {args.site}, year {args.year}")
        else:
            logger.info(f"Created {n_plots} plots in: {site_output_dir}")
    
    elif args.all:
        logger.info(f"Plotting all segments (max {args.max_per_site} per site)")
        logger.warning("This may create many files!")
        
        # Load metadata to get all sites
        import pickle
        with open(data_dir / f'model_{args.split}_data_combination_ids.pkl', 'rb') as f:
            combo_ids = pickle.load(f)
        
        # Get unique (site, year) pairs
        site_years = set()
        for combo_id, ids_row in combo_ids.items():
            site_id = ids_row['site ID']
            # Will need to check actual segments for years
            site_years.add(site_id)
        
        total_plots = 0
        for site_id in sorted(site_years):
            # Try plotting for common years (2019-2023)
            for year in range(2019, 2024):
                site_output_dir = output_dir / f'site_{site_id}' / f'year_{year}'
                
                n_plots = plotter.plot_segments_for_site(
                    data_dir=data_dir,
                    site_id=site_id,
                    year=year,
                    output_dir=site_output_dir,
                    split=args.split,
                    max_segments=args.max_per_site
                )
                
                total_plots += n_plots
        
        logger.info(f"Created {total_plots} total plots")
    
    logger.info("Visualization complete!")


if __name__ == '__main__':
    main()
