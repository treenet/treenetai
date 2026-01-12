#!/usr/bin/env python3
"""
Batch reconstruct all available sensor combinations for years 2023-2024.

This script:
1. Discovers all valid sensor combinations (T, H, D) from metadata
2. Processes each combination through the trained model
3. Produces clean 1-hour 3-channel output time series
4. Saves reconstructed data to output directory

Uses overlapping 30-day segments with weighted averaging for smooth reconstruction.

Usage:
    python 13_batch_reconstruct.py \
        --model-path /path/to/best_model.keras \
        --year-start 2023 --year-end 2024 \
        --output-dir /path/to/output

Author: Lukovic
Date: 2026-01-11
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding
from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor
from src.data.segmentation import Normalizer
from src.utils import setup_logging, ensure_dir

# Import TimeSeriesReconstructor from 6_reconstruct_timeseries.py
from importlib.util import spec_from_file_location, module_from_spec

def load_reconstructor_module():
    """Dynamically load TimeSeriesReconstructor from 6_reconstruct_timeseries.py"""
    spec = spec_from_file_location(
        "reconstruct_module", 
        Path(__file__).parent / "6_reconstruct_timeseries.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch reconstruct all sensor combinations'
    )
    
    # Model
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model (.keras file)'
    )
    
    # Data paths
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
        default='/storage/lukovic/Data/FORWARDS/treenet/reconstructed_2023_2024',
        help='Output directory for reconstructed time series'
    )
    
    # Year range
    parser.add_argument(
        '--year-start', type=int, default=2023,
        help='Start year for reconstruction'
    )
    parser.add_argument(
        '--year-end', type=int, default=2024,
        help='End year for reconstruction'
    )
    
    # Filtering
    parser.add_argument(
        '--country', type=str, default='Switzerland',
        help='Filter sites by country (default: Switzerland)'
    )
    parser.add_argument(
        '--max-combinations', type=int, default=-1,
        help='Maximum combinations to process (-1 for all)'
    )
    parser.add_argument(
        '--site-ids', type=str, default=None,
        help='Comma-separated list of site IDs to process (default: all)'
    )
    
    # Reconstruction parameters
    parser.add_argument(
        '--overlap-days', type=int, default=5,
        help='Overlap between consecutive segments (days)'
    )
    parser.add_argument(
        '--max-gap-days', type=int, default=12,
        help='Maximum gap length to fill (days)'
    )
    parser.add_argument(
        '--norm-scope', type=str, default='segment',
        choices=['year', 'segment'],
        help='Normalization scope (should match training)'
    )
    parser.add_argument(
        '--output-mode', type=str, default='input_scale',
        choices=['normalized', 'input_scale'],
        help='Output mode'
    )
    
    # Options
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip combinations that already have output files'
    )
    
    return parser.parse_args()


def discover_combinations(
    loaders: DataLoaders,
    country: str = 'Switzerland'
) -> List[Dict]:
    """
    Discover all valid sensor combinations from metadata.
    
    A valid combination has:
    - Thermometer (L1)
    - Hygrometer (L1)
    - Dendrometer (L2)
    - MeteoSwiss data (for Swiss sites)
    
    Returns list of dicts with site_id, thermo_id, hygro_id, dendro_id
    """
    # Load metadata files
    metadata_path = Path(loaders.data_root) / 'metadata_all.pkl'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata_all = pd.read_pickle(metadata_path)
    
    # Filter by country
    if country.lower() != 'all':
        metadata_all = metadata_all[metadata_all['country'] == country]
    
    # Get unique site IDs
    site_ids = metadata_all['site_id'].unique()
    
    print(f"Found {len(site_ids)} sites in {country}")
    
    # Load sensor metadata
    thermo_meta = pd.read_pickle(Path(loaders.data_root) / 'metadata_data_all_l1_temperature.pkl')
    hygro_meta = pd.read_pickle(Path(loaders.data_root) / 'metadata_data_all_l1_humidity.pkl')
    dendro_l2_meta = pd.read_pickle(Path(loaders.data_root) / 'metadata_data_dendro_l2.pkl')
    
    combinations = []
    
    for site_id in site_ids:
        # Get sensors for this site
        site_thermos = thermo_meta[thermo_meta['site_id'] == site_id]['series_id'].unique()
        site_hygros = hygro_meta[hygro_meta['site_id'] == site_id]['series_id'].unique()
        site_dendros = dendro_l2_meta[dendro_l2_meta['site_id'] == site_id]['series_id'].unique()
        
        # Check if meteo data exists
        meteo_path = Path(loaders.meteo_root) / f'meteo_data_site_id_{site_id}.csv'
        if not meteo_path.exists():
            continue
        
        # Create all combinations
        if len(site_thermos) > 0 and len(site_hygros) > 0 and len(site_dendros) > 0:
            for t_id in site_thermos:
                for h_id in site_hygros:
                    for d_id in site_dendros:
                        combinations.append({
                            'site_id': int(site_id),
                            'thermo_id': int(t_id),
                            'hygro_id': int(h_id),
                            'dendro_id': int(d_id)
                        })
    
    return combinations


def process_combination(
    combo: Dict,
    model: tf.keras.Model,
    loaders: DataLoaders,
    processor: DataProcessor,
    normalizer: Normalizer,
    output_dir: Path,
    year_start: int,
    year_end: int,
    overlap_days: int,
    max_gap_days: int,
    output_mode: str,
    verbose: bool,
    reconstructor_class
) -> Tuple[bool, Dict]:
    """
    Process a single sensor combination.
    
    Returns (success, metrics)
    """
    combo_str = f"site{combo['site_id']}_T{combo['thermo_id']}_H{combo['hygro_id']}_D{combo['dendro_id']}"
    
    print(f"\n{'='*60}")
    print(f"Processing: {combo_str}")
    print(f"{'='*60}")
    
    try:
        # Create reconstructor
        reconstructor = reconstructor_class(
            model=model,
            normalizer=normalizer,
            max_gap_days=max_gap_days,
            overlap_days=overlap_days,
            output_mode=output_mode,
            verbose=verbose
        )
        
        # Load and prepare input
        input_df, lm_df = reconstructor.load_and_prepare_input(
            loaders=loaders,
            processor=processor,
            site_id=combo['site_id'],
            thermo_id=combo['thermo_id'],
            hygro_id=combo['hygro_id'],
            dendro_id=combo['dendro_id'],
            year_start=year_start,
            year_end=year_end
        )
        
        if len(input_df) == 0:
            print(f"  No data available for years {year_start}-{year_end}")
            return False, {'error': 'no_data'}
        
        # Run reconstruction
        reconstructed, metrics = reconstructor.reconstruct(input_df, lm_df)
        
        # Save results
        save_path = output_dir / f"reconstructed_{combo_str}.ftr"
        save_df = reconstructed.reset_index()
        save_df.rename(columns={'index': 'ts'}, inplace=True)
        save_df.to_feather(save_path)
        print(f"  Saved: {save_path}")
        
        # Add combo info to metrics
        metrics.update(combo)
        metrics['output_path'] = str(save_path)
        metrics['success'] = True
        
        return True, metrics
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, {'error': str(e), **combo}


def main():
    """Main function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = ensure_dir(Path(args.output_dir))
    
    # Setup logging
    log_file = output_dir / f'batch_reconstruction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_logging(verbose=args.verbose, log_file=log_file)
    
    print("="*80)
    print("TreeNet AI - Batch Time Series Reconstruction")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Years: {args.year_start} - {args.year_end}")
    print(f"Country: {args.country}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\n" + "-"*60)
    print("Loading model...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'TCNBlock': TCNBlock,
            'PositionalEncoding': PositionalEncoding
        }
    )
    print(f"  Model loaded successfully")
    
    # Initialize components
    print("\n" + "-"*60)
    print("Initializing data loaders...")
    loaders = DataLoaders(
        data_root=Path(args.data_dir),
        meteo_root=Path(args.meteo_root)
    )
    processor = DataProcessor()
    normalizer = Normalizer(norm_scope=args.norm_scope)
    
    # Load reconstructor class
    recon_module = load_reconstructor_module()
    TimeSeriesReconstructor = recon_module.TimeSeriesReconstructor
    
    # Discover combinations
    print("\n" + "-"*60)
    print("Discovering sensor combinations...")
    all_combinations = discover_combinations(loaders, args.country)
    print(f"  Found {len(all_combinations)} total combinations")
    
    # Filter by site IDs if specified
    if args.site_ids:
        site_list = [int(s.strip()) for s in args.site_ids.split(',')]
        all_combinations = [c for c in all_combinations if c['site_id'] in site_list]
        print(f"  Filtered to {len(all_combinations)} combinations for sites: {site_list}")
    
    # Limit combinations if specified
    if args.max_combinations > 0 and len(all_combinations) > args.max_combinations:
        all_combinations = all_combinations[:args.max_combinations]
        print(f"  Limited to first {args.max_combinations} combinations")
    
    # Skip existing if requested
    if args.skip_existing:
        filtered = []
        for combo in all_combinations:
            combo_str = f"site{combo['site_id']}_T{combo['thermo_id']}_H{combo['hygro_id']}_D{combo['dendro_id']}"
            output_path = output_dir / f"reconstructed_{combo_str}.ftr"
            if not output_path.exists():
                filtered.append(combo)
        skipped = len(all_combinations) - len(filtered)
        all_combinations = filtered
        print(f"  Skipped {skipped} existing outputs, {len(all_combinations)} remaining")
    
    # Process each combination
    print("\n" + "-"*60)
    print(f"Processing {len(all_combinations)} combinations...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, combo in enumerate(all_combinations):
        print(f"\n[{i+1}/{len(all_combinations)}]", end="")
        
        success, metrics = process_combination(
            combo=combo,
            model=model,
            loaders=loaders,
            processor=processor,
            normalizer=normalizer,
            output_dir=output_dir,
            year_start=args.year_start,
            year_end=args.year_end,
            overlap_days=args.overlap_days,
            max_gap_days=args.max_gap_days,
            output_mode=args.output_mode,
            verbose=args.verbose,
            reconstructor_class=TimeSeriesReconstructor
        )
        
        results.append(metrics)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Save summary
    print("\n" + "="*80)
    print("BATCH RECONSTRUCTION COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(all_combinations)}")
    
    # Save results summary
    summary_path = output_dir / 'batch_results.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(args.model_path),
        'year_range': [args.year_start, args.year_end],
        'country': args.country,
        'successful': successful,
        'failed': failed,
        'combinations': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
