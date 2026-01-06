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
        '--year',
        type=int,
        default=2020,
        help='Year to process'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Ratio of sites to use for testing'
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
    
    # Initialize components
    print("\n1. Initializing data loaders...")
    loaders = DataLoaders(
        data_root=Path(args.data_root),
        meteo_root=Path(args.meteo_root)
    )
    
    processor = DataProcessor()
    segment_builder = SegmentBuilder(
        segment_days=args.segment_days,
        stride_days=args.stride_days
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
    n_test = max(1, int(len(site_list) * args.test_ratio))
    test_sites = set(np.random.choice(site_list, size=n_test, replace=False))
    train_sites = complete_sites - test_sites
    
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
            for thermo_id, hygro_id, dendro_id in product(thermo_ids, hygro_ids, dendro_l2_ids):
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
                    site_sensors['dendrometer_lm'][dendro_id]
                )
                
                # Create year grids
                grid_10min = YearGridBuilder.create_year_grid_10min(args.year)
                grid_hourly = YearGridBuilder.create_year_grid_hourly(args.year)
                
                # Create input array (11 channels, 10-min)
                input_df = processor.merger.create_input_array(
                    temp_df, rh_df, stem_df, meteo_daily, grid_10min
                )
                
                # Create target array (3 channels, hourly)
                output_df = processor.merger.create_target_array(lm_df, grid_hourly)
                
                # Build segments
                input_segs, output_segs, seg_metadata = segment_builder.build_segments_for_combination(
                    combo_id=combo_counter,
                    site_id=site_id,
                    thermometer_id=thermo_id,
                    hygrometer_id=hygro_id,
                    dendrometer_id=dendro_id,
                    input_df=input_df,
                    output_df=output_df,
                    year=args.year,
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
