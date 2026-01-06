#!/usr/bin/env python3
"""
CLI for comparing processed segments with raw data.

Denormalizes segments and compares them against original raw files
to validate data integrity throughout the processing pipeline.

Usage:
    # Compare specific segment
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
    
    # Compare multiple segments for a combination
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
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PipelineConfig
from visualization.compare_raw import RawDataComparator
from utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare processed segments with raw data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory with processed segments'
    )
    parser.add_argument(
        '--raw-root',
        type=Path,
        required=True,
        help='Root directory of server_data (with thermometer_l1/, etc.)'
    )
    parser.add_argument(
        '--meteo-root',
        type=Path,
        required=True,
        help='Root directory of meteo_data (with site_*.csv files)'
    )
    
    # Segment selection
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--combo-id',
        type=int,
        required=True,
        help='Combination ID to compare'
    )
    parser.add_argument(
        '--seg-idx',
        type=int,
        help='Specific segment index (single mode)'
    )
    parser.add_argument(
        '--all-segments',
        action='store_true',
        help='Compare all segments for the combination'
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year of the data'
    )
    parser.add_argument(
        '--site',
        type=int,
        required=True,
        help='Site ID'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for comparison plots'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.seg_idx is None and not args.all_segments:
        parser.error('Must specify either --seg-idx or --all-segments')
    
    if args.seg_idx is not None and args.all_segments:
        print('[WARNING] Both --seg-idx and --all-segments specified, using --all-segments')
    
    return args


def main():
    """Main comparison workflow."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging('raw_comparison')
    logger.info("Starting raw data comparison")
    
    # Load configuration
    config = PipelineConfig()
    
    # Validate paths
    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    if not args.raw_root.exists():
        logger.error(f"Raw data root does not exist: {args.raw_root}")
        sys.exit(1)
    
    if not args.meteo_root.exists():
        logger.error(f"Meteo data root does not exist: {args.meteo_root}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.data_dir / 'raw_comparisons' / args.split
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize comparator
    comparator = RawDataComparator(local_tz=config.preprocessing.local_timezone)
    
    # Determine segments to process
    if args.all_segments:
        # Load segment data to get count
        import pickle
        with open(args.data_dir / f'{args.split}_input_segments.pkl', 'rb') as f:
            input_segs = pickle.load(f)
        
        if args.combo_id not in input_segs:
            logger.error(f"Combination {args.combo_id} not found in {args.split} set")
            sys.exit(1)
        
        seg_indices = list(range(len(input_segs[args.combo_id])))
        logger.info(f"Processing {len(seg_indices)} segments for combo {args.combo_id}")
    else:
        seg_indices = [args.seg_idx]
        logger.info(f"Processing single segment: combo {args.combo_id}, seg {args.seg_idx}")
    
    # Process each segment
    for seg_idx in seg_indices:
        try:
            comparator.compare_segment(
                data_dir=args.data_dir,
                raw_root=args.raw_root,
                meteo_root=args.meteo_root,
                split=args.split,
                combo_id=args.combo_id,
                seg_idx=seg_idx,
                year=args.year,
                site_id=args.site,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Failed to compare combo {args.combo_id}, seg {seg_idx}: {e}")
            continue
    
    logger.info(f"\\nComparison complete! Plots saved to: {output_dir}")
    logger.info(f"Segments processed: {seg_indices}")


if __name__ == '__main__':
    main()
