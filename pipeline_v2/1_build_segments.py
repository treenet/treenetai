#!/usr/bin/env python3
"""
Build 30-day segments from raw TreeNet data.

This script:
1. Loads raw sensor data (thermometer, hygrometer, dendrometer)
2. Processes and aligns timestamps to UTC
3. Merges with global meteo data
4. Extracts complete 30-day segments with year-level normalization
5. Splits into train/test sets
6. Saves processed segments

Usage:
    python 1_build_segments.py --config configs/default.yaml
    python 1_build_segments.py --year 2020 --test-ratio 0.2
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, DataPaths, SegmentConfig
from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor, YearGridBuilder
from src.data.segmentation import SegmentBuilder, SegmentMetadata
from src.utils import setup_logging, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build 30-day segments from raw data')
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data',
        help='Root directory with raw sensor data'
    )
    parser.add_argument(
        '--meteo-root',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/meteo_data',
        help='Directory with meteotest CSV files'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed',
        help='Output directory for processed segments'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Ratio of sites to use for testing'
    )
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=10,
        help='Maximum number of random sensor combinations to try per site (default: 10, use -1 for all)'
    )
    parser.add_argument(
        '--max-sites',
        type=int,
        default=-1,
        help='Maximum number of sites to process (default: -1 for all, useful for testing)'
    )
    parser.add_argument(
        '--force-sites',
        type=str,
        default='',
        help='Comma-separated list of specific site IDs to process (e.g., "3,4,10"). Overrides max-sites.'
    )
    parser.add_argument(
        '--force-combination',
        type=str,
        default='',
        help='Force specific sensor combination as "T,H,D" (e.g., "9,7,18"). Only works with single site.'
    )
    parser.add_argument(
        '--segment-days',
        type=int,
        default=30,
        help='Length of each segment in days'
    )
    parser.add_argument(
        '--stride-days',
        type=int,
        default=10,
        help='Stride for overlapping segments in days'
    )
    parser.add_argument(
        '--norm-scope',
        type=str,
        default='year',
        choices=['year', 'segment'],
        help='Normalization scope: year-level (consistent across segments) or segment-level (adapts to local data)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for train/test split'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    output_dir = ensure_dir(Path(args.output_root) / 'processed' / 'model_data')
    log_file = output_dir.parent / 'segment_building.log'
    setup_logging(verbose=args.verbose, log_file=log_file)
    
    print("="*80)
    print("TreeNet AI Pipeline v2 - Segment Building")
    print("="*80)
    print(f"Segment length: {args.segment_days} days, Stride: {args.stride_days} days")
    print(f"Normalization scope: {args.norm_scope}")
    print("="*80)
    
    # Initialize components
    print("\n1. Initializing data loaders...")
    loaders = DataLoaders(
        data_root=Path(args.data_root),
        meteo_root=Path(args.meteo_root)
    )
    
    processor = DataProcessor()
    segment_builder = SegmentBuilder(
        segment_days=args.segment_days,
        stride_days=args.stride_days,
        norm_scope=args.norm_scope
    )
    
    # Load metadata
    print("\n2. Loading metadata...")
    metadata = loaders.load_metadata()
    print(f"   Total sensors: {len(metadata)}")
    
    # Find sites with complete data
    print("\n3. Finding sites with complete sensor coverage...")
    complete_sites = loaders.get_sites_with_complete_data(metadata)
    print(f"   Sites with all 3 sensor types: {len(complete_sites)}")
    print(f"   Site IDs: {sorted(complete_sites)}")
    
    # Split into train/test
    print(f"\n4. Splitting sites (test ratio: {args.test_ratio})...")
    np.random.seed(args.random_seed)
    site_list = sorted(list(complete_sites))
    
    # Use forced sites if specified
    if args.force_sites:
        forced_ids = [int(s.strip()) for s in args.force_sites.split(',')]
        site_list = [s for s in site_list if s in forced_ids]
        print(f"   Forcing specific sites: {site_list}")
    # Otherwise limit number of sites if requested
    elif args.max_sites > 0 and len(site_list) > args.max_sites:
        site_list = [site_list[i] for i in np.random.choice(
            len(site_list), size=args.max_sites, replace=False)]
        print(f"   Randomly selected {args.max_sites} sites for processing")
    
    n_test = max(1, int(len(site_list) * args.test_ratio))
    test_sites = set([site_list[i] for i in np.random.choice(
        len(site_list), size=n_test, replace=False)])
    train_sites = set(site_list) - test_sites
    
    print(f"   Train sites ({len(train_sites)}): {sorted(train_sites)}")
    print(f"   Test sites ({len(test_sites)}): {sorted(test_sites)}")
    
    # Process each split
    for split_name, sites in [('train', train_sites), ('test', test_sites)]:
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*80}")
        
        all_input_segments = {}
        all_output_segments = {}
        all_metadata = []
        all_combo_ids = {}
        
        combo_counter = 0
        
        # Process each site
        for site_id in sorted(sites):
            print(f"\n  Site {site_id}:")
            
            # Load all sensors for this site
            site_sensors = loaders.load_all_sensors_for_site(site_id, metadata)
            
            # Get sensor IDs
            thermo_ids = list(site_sensors['thermometer'].keys())
            hygro_ids = list(site_sensors['hygrometer'].keys())
            dendro_l2_ids = list(site_sensors['dendrometer_l2'].keys())
            dendro_lm_ids = list(site_sensors['dendrometer_lm'].keys())
            
            print(f"    Thermometers: {len(thermo_ids)}")
            print(f"    Hygrometers: {len(hygro_ids)}")
            print(f"    Dendrometers: {len(dendro_l2_ids)}")
            
            # Load meteo data
            meteo_df = loaders.load_meteotest_data(site_id)
            if meteo_df is None:
                print(f"    WARNING: No meteo data for site {site_id}, skipping")
                continue
            
            meteo_daily = processor.process_meteo_daily(meteo_df)
            
            # Create all sensor combinations
            all_combinations = list(product(thermo_ids, hygro_ids, dendro_l2_ids))
            
            # Force specific combination if specified
            if args.force_combination:
                forced_t, forced_h, forced_d = [int(x.strip()) for x in args.force_combination.split(',')]
                if (forced_t in thermo_ids) and (forced_h in hygro_ids) and (forced_d in dendro_l2_ids):
                    all_combinations = [(forced_t, forced_h, forced_d)]
                    print(f"    Forcing combination: T={forced_t}, H={forced_h}, D={forced_d}")
                else:
                    print(f"    WARNING: Forced combination T={forced_t}, H={forced_h}, D={forced_d} not available, using all")
            # Sample combinations if max_combinations is set
            elif args.max_combinations > 0 and len(all_combinations) > args.max_combinations:
                all_combinations = [all_combinations[i] for i in np.random.choice(
                    len(all_combinations), size=args.max_combinations, replace=False)]
                print(f"    Randomly selected {args.max_combinations} of {len(list(product(thermo_ids, hygro_ids, dendro_l2_ids)))} possible combinations")
            
            for thermo_id, hygro_id, dendro_id in all_combinations:
                # Check if LM data exists for this dendrometer
                if dendro_id not in dendro_lm_ids:
                    continue
                
                combo_counter += 1
                print(f"    Combo {combo_counter}: T={thermo_id}, H={hygro_id}, D={dendro_id}")
                
                # Load and process sensor data
                temp_df = processor.process_sensor_dataframe(
                    site_sensors['thermometer'][thermo_id]
                )
                rh_df = processor.process_sensor_dataframe(
                    site_sensors['hygrometer'][hygro_id]
                )
                stem_df = processor.process_sensor_dataframe(
                    site_sensors['dendrometer_l2'][dendro_id]
                )
                lm_df = processor.process_sensor_dataframe(
                    site_sensors['dendrometer_lm'][dendro_id],
                    keep_all_columns=True
                )
                
                # Create input array (11 channels, 10-min) - using all available data
                input_df = processor.merger.create_input_array(
                    temp_df, rh_df, stem_df, meteo_daily
                )
                
                # Create target array (3 channels, hourly) - using all available data
                output_df = processor.merger.create_target_array(lm_df)
                
                # Save intermediate time series as feather files
                intermediate_dir = os.path.join(output_dir, 'intermediate_timeseries')
                os.makedirs(intermediate_dir, exist_ok=True)
                
                input_ts_path = os.path.join(intermediate_dir, 
                    f"{split_name}_input_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}.ftr")
                output_ts_path = os.path.join(intermediate_dir,
                    f"{split_name}_output_site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}.ftr")
                
                # Save with index as column (matching reference format)
                input_save = input_df.reset_index()
                input_save.rename(columns={'index': 'ts'}, inplace=True)
                input_save.to_feather(input_ts_path)
                
                output_save = output_df.reset_index()
                output_save.rename(columns={'index': 'ts'}, inplace=True)
                output_save.to_feather(output_ts_path)
                
                # Build segments from all available data (no year filtering)
                input_segs, output_segs, seg_metadata = segment_builder.build_segments_for_combination(
                    combo_id=combo_counter,
                    site_id=site_id,
                    thermometer_id=thermo_id,
                    hygrometer_id=hygro_id,
                    dendrometer_id=dendro_id,
                    input_df=input_df,
                    output_df=output_df,
                    input_channels=['temp_treenet', 'rh_treenet', 'stem', 
                                   'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy'],
                    target_channels=['local_T', 'local_RH', 'stem']
                )
                
                print(f"      Segments found: {len(input_segs)}")
                
                if len(input_segs) > 0:
                    all_input_segments[combo_counter] = input_segs
                    all_output_segments[combo_counter] = output_segs
                    all_metadata.extend(seg_metadata)
                    
                    # Store combo IDs
                    all_combo_ids[combo_counter] = pd.Series({
                        'site ID': site_id,
                        'thermometer ID': thermo_id,
                        'hygrometer ID': hygro_id,
                        'dendrometer ID': dendro_id
                    })
        
        # Save segments
        print(f"\n  Saving {split_name} segments...")
        print(f"    Total combinations: {len(all_input_segments)}")
        print(f"    Total segments: {len(all_metadata)}")
        
        segment_builder.save_segments(
            output_dir=output_dir,
            split=split_name,
            input_segments=all_input_segments,
            output_segments=all_output_segments,
            metadata=all_metadata,
            combo_ids=all_combo_ids
        )
    
    print("\n" + "="*80)
    print("Segment building complete!")
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
