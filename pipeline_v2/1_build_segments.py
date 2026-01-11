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
    python 1_build_segments.py --run-name processed_v3 --country Switzerland
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, DataPaths, SegmentConfig
from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor, YearGridBuilder
from src.data.segmentation import SegmentBuilder, SegmentMetadata, FilteredYearInfo
from src.utils import setup_logging, ensure_dir
from src.reporting import BuildReportCollector, save_report


def plot_filtered_year(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    filtered_info: FilteredYearInfo,
    site_id: int,
    thermo_id: int,
    hygro_id: int,
    dendro_id: int,
    output_dir: Path,
    log
):
    """
    Plot and save input/output stem signals for a filtered year.
    
    Creates a 2-panel plot showing the raw input (L2) and output (LM) stem 
    signals for visual inspection of why the year was filtered.
    
    Args:
        input_df: Full input DataFrame with 'stem' column
        output_df: Full output DataFrame with 'stem' column
        filtered_info: FilteredYearInfo with year and reason
        site_id: Site ID
        thermo_id: Thermometer series ID
        hygro_id: Hygrometer series ID  
        dendro_id: Dendrometer series ID
        output_dir: Directory to save plots
        log: Logger instance
    """
    year = filtered_info.year
    combo_str = f"site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}"
    
    # Extract year data
    input_year = input_df[input_df.index.year == year]['stem'] if 'stem' in input_df.columns else None
    output_year = output_df[output_df.index.year == year]['stem'] if 'stem' in output_df.columns else None
    
    if input_year is None or output_year is None or len(input_year) == 0 or len(output_year) == 0:
        log.warning(f"      Cannot plot filtered year {year}: no stem data")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot input (L2) stem
    axes[0].plot(input_year.index, input_year.values, 'b-', linewidth=0.5, alpha=0.8, label='L2 (Input)')
    axes[0].set_ylabel('Stem Radius (µm)')
    axes[0].set_title(f'Input Stem (L2) - Year {year}\nRange: {filtered_info.input_range:.1f} µm')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot output (LM) stem
    axes[1].plot(output_year.index, output_year.values, 'g-', linewidth=0.5, alpha=0.8, label='LM (Output)')
    axes[1].set_ylabel('Stem Radius (µm)')
    axes[1].set_xlabel('Date')
    axes[1].set_title(f'Output Stem (LM) - Year {year}\nRange: {filtered_info.output_range:.1f} µm')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Add overall title with reason
    fig.suptitle(
        f'FILTERED: {combo_str} - Year {year}\nReason: {filtered_info.reason}',
        fontsize=12, fontweight='bold', color='red'
    )
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"filtered_{combo_str}_year{year}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    log.debug(f"      Saved filtered year plot: {plot_filename}")


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
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data',
        help='Directory with meteotest CSV files'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed',
        help='Output directory root for processed segments'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this run (creates output subdirectory). Default: processed_YYYYMMDD_HHMMSS'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='Switzerland',
        help='Filter sites by country (default: Switzerland). Use "all" to include all countries.'
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
    
    # Generate run name if not specified
    if args.run_name is None:
        run_name = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = args.run_name
    
    # Setup output directory structure:
    # output_root/
    #   run_name/
    #     logs/              <- log files
    #       filtered_plots/  <- plots of filtered years
    #     model_data/        <- processed segments
    #       intermediate_timeseries/
    #       segments/
    #       reports/
    run_dir = ensure_dir(Path(args.output_root) / run_name)
    log_dir = ensure_dir(run_dir / 'logs')
    filtered_plots_dir = ensure_dir(log_dir / 'filtered_plots')
    output_dir = ensure_dir(run_dir / 'model_data')
    
    # Setup logging - single log file for all output
    log_file = log_dir / 'build_segments.log'
    log = setup_logging(log_file=log_file, name='build_segments', verbose=args.verbose)
    
    # Determine country filter
    country_filter = None if args.country.lower() == 'all' else args.country
    
    log.info("="*80)
    log.info("TreeNet AI Pipeline v2 - Segment Building")
    log.info("="*80)
    log.info(f"Run name: {run_name}")
    log.info(f"Output directory: {run_dir}")
    log.info(f"Log file: {log_file}")
    log.info(f"Segment length: {args.segment_days} days, Stride: {args.stride_days} days")
    log.info(f"Normalization scope: {args.norm_scope}")
    log.info(f"Country filter: {args.country}")
    log.info("="*80)
    
    # Initialize components
    log.info("\n1. Initializing data loaders...")
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
    
    # Initialize report collector
    report_collector = BuildReportCollector(
        output_root=str(run_dir),
        country_filter=args.country,
        segment_days=args.segment_days,
        stride_days=args.stride_days,
        norm_scope=args.norm_scope,
        random_seed=args.random_seed,
        max_combinations=args.max_combinations
    )
    
    # Load metadata
    log.info("\n2. Loading metadata...")
    metadata = loaders.load_metadata()
    log.info(f"   Total sensors: {len(metadata)}")
    
    # Find sites with complete data (filtered by country)
    log.info(f"\n3. Finding sites with complete sensor coverage (country={args.country})...")
    complete_sites = loaders.get_sites_with_complete_data(metadata, country=country_filter)
    log.info(f"   Sites with all 3 sensor types: {len(complete_sites)}")
    log.info(f"   Site IDs: {sorted(complete_sites)}")
    
    # Record available sites
    report_collector.set_available_sites(list(complete_sites))
    
    # Split into train/test
    log.info(f"\n4. Splitting sites (test ratio: {args.test_ratio})...")
    np.random.seed(args.random_seed)
    site_list = sorted(list(complete_sites))
    
    # Use forced sites if specified
    if args.force_sites:
        forced_ids = [int(s.strip()) for s in args.force_sites.split(',')]
        site_list = [s for s in site_list if s in forced_ids]
        log.info(f"   Forcing specific sites: {site_list}")
    # Otherwise limit number of sites if requested
    elif args.max_sites > 0 and len(site_list) > args.max_sites:
        site_list = [site_list[i] for i in np.random.choice(
            len(site_list), size=args.max_sites, replace=False)]
        log.info(f"   Randomly selected {args.max_sites} sites for processing")
    
    n_test = max(1, int(len(site_list) * args.test_ratio))
    test_sites = set([site_list[i] for i in np.random.choice(
        len(site_list), size=n_test, replace=False)])
    train_sites = set(site_list) - test_sites
    
    log.info(f"   Train sites ({len(train_sites)}): {sorted(train_sites)}")
    log.info(f"   Test sites ({len(test_sites)}): {sorted(test_sites)}")
    
    # Record train/test split in report
    report_collector.set_train_test_split(list(train_sites), list(test_sites))
    
    # Process each split
    total_filtered_years = 0
    for split_name, sites in [('train', train_sites), ('test', test_sites)]:
        log.info(f"\n{'='*80}")
        log.info(f"Processing {split_name.upper()} split")
        log.info(f"{'='*80}")
        
        # Set current split in report collector
        report_collector.set_current_split(split_name)
        
        all_input_segments = {}
        all_output_segments = {}
        all_metadata = []
        all_combo_ids = {}
        split_filtered_years = 0
        
        combo_counter = 0
        
        # Process each site
        for site_id in sorted(sites):
            log.info(f"\n  Site {site_id}:")
            
            # Load all sensors for this site
            site_sensors = loaders.load_all_sensors_for_site(site_id, metadata)
            
            # Get sensor IDs
            thermo_ids = list(site_sensors['thermometer'].keys())
            hygro_ids = list(site_sensors['hygrometer'].keys())
            dendro_l2_ids = list(site_sensors['dendrometer_l2'].keys())
            dendro_lm_ids = list(site_sensors['dendrometer_lm'].keys())
            
            log.info(f"    Thermometers: {len(thermo_ids)}")
            log.info(f"    Hygrometers: {len(hygro_ids)}")
            log.info(f"    Dendrometers: {len(dendro_l2_ids)}")
            
            # Load meteo data
            meteo_df = loaders.load_meteotest_data(site_id)
            has_meteo = meteo_df is not None
            
            # Start site in report collector
            report_collector.start_site(
                site_id=site_id,
                n_thermometers=len(thermo_ids),
                n_hygrometers=len(hygro_ids),
                n_dendrometers=len(dendro_l2_ids),
                has_meteo=has_meteo
            )
            
            if not has_meteo:
                log.warning(f"    No meteo data for site {site_id}, skipping")
                report_collector.finalize_site(site_id)
                continue
            
            meteo_daily = processor.process_meteo_daily(meteo_df)
            
            # Create all sensor combinations
            all_combinations = list(product(thermo_ids, hygro_ids, dendro_l2_ids))
            
            # Force specific combination if specified
            if args.force_combination:
                forced_t, forced_h, forced_d = [int(x.strip()) for x in args.force_combination.split(',')]
                if (forced_t in thermo_ids) and (forced_h in hygro_ids) and (forced_d in dendro_l2_ids):
                    all_combinations = [(forced_t, forced_h, forced_d)]
                    log.info(f"    Forcing combination: T={forced_t}, H={forced_h}, D={forced_d}")
                else:
                    log.warning(f"    Forced combination T={forced_t}, H={forced_h}, D={forced_d} not available, using all")
            # Sample combinations if max_combinations is set
            elif args.max_combinations > 0 and len(all_combinations) > args.max_combinations:
                all_combinations = [all_combinations[i] for i in np.random.choice(
                    len(all_combinations), size=args.max_combinations, replace=False)]
                log.info(f"    Randomly selected {args.max_combinations} of {len(list(product(thermo_ids, hygro_ids, dendro_l2_ids)))} possible combinations")
            
            for thermo_id, hygro_id, dendro_id in all_combinations:
                # Check if LM data exists for this dendrometer
                if dendro_id not in dendro_lm_ids:
                    continue
                
                combo_counter += 1
                log.info(f"    Combo {combo_counter}: T={thermo_id}, H={hygro_id}, D={dendro_id}")
                
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
                # Includes verification to catch NFS silent failures
                try:
                    input_save = input_df.reset_index()
                    input_save.rename(columns={'index': 'ts'}, inplace=True)
                    input_save.to_feather(input_ts_path)
                    
                    output_save = output_df.reset_index()
                    output_save.rename(columns={'index': 'ts'}, inplace=True)
                    output_save.to_feather(output_ts_path)
                    
                    # Verify files were actually written (catches NFS silent failures)
                    input_exists = os.path.exists(input_ts_path)
                    output_exists = os.path.exists(output_ts_path)
                    input_size = os.path.getsize(input_ts_path) if input_exists else 0
                    output_size = os.path.getsize(output_ts_path) if output_exists else 0
                    
                    if not input_exists or not output_exists or input_size == 0 or output_size == 0:
                        log.warning(f"      File verification failed!")
                        log.warning(f"        Input: exists={input_exists}, size={input_size}")
                        log.warning(f"        Output: exists={output_exists}, size={output_size}")
                    elif args.verbose:
                        log.debug(f"        Saved: {os.path.basename(input_ts_path)} ({input_size:,} bytes)")
                        
                except Exception as e:
                    log.error(f"      ERROR saving intermediate files: {e}")
                    log.error(f"        Input path: {input_ts_path}")
                    log.error(f"        Input shape: {input_df.shape if input_df is not None else 'None'}")
                    log.error(f"        Output shape: {output_df.shape if output_df is not None else 'None'}")
                
                # Build segments from all available data (with data quality filtering)
                input_segs, output_segs, seg_metadata, filtered_years_info = segment_builder.build_segments_for_combination(
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
                
                log.info(f"      Segments found: {len(input_segs)}")
                
                # Log and plot filtered years
                if filtered_years_info:
                    log.info(f"      Filtered years: {len(filtered_years_info)}")
                    combo_str = f"site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}"
                    for fyi in filtered_years_info:
                        log.warning(
                            f"      FILTERED: {combo_str} year {fyi.year} - "
                            f"ratio={fyi.ratio:.3f}, input_range={fyi.input_range:.1f}, "
                            f"output_range={fyi.output_range:.1f} - {fyi.reason}"
                        )
                        # Plot filtered year
                        plot_filtered_year(
                            input_df=input_df,
                            output_df=output_df,
                            filtered_info=fyi,
                            site_id=site_id,
                            thermo_id=thermo_id,
                            hygro_id=hygro_id,
                            dendro_id=dendro_id,
                            output_dir=filtered_plots_dir,
                            log=log
                        )
                    split_filtered_years += len(filtered_years_info)
                
                # Record combination in report (including gap analysis)
                report_collector.record_combination(
                    site_id=site_id,
                    thermometer_id=thermo_id,
                    hygrometer_id=hygro_id,
                    dendrometer_id=dendro_id,
                    segment_count=len(input_segs),
                    segment_metadata=seg_metadata,
                    input_df=input_df,
                    output_df=output_df
                )
                
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
            
            # Finalize site in report
            report_collector.finalize_site(site_id)
        
        # Save segments
        log.info(f"\n  Saving {split_name} segments...")
        log.info(f"    Total combinations: {len(all_input_segments)}")
        log.info(f"    Total segments: {len(all_metadata)}")
        if split_filtered_years > 0:
            log.info(f"    Filtered years: {split_filtered_years}")
        
        total_filtered_years += split_filtered_years
        
        segment_builder.save_segments(
            output_dir=output_dir,
            split=split_name,
            input_segments=all_input_segments,
            output_segments=all_output_segments,
            metadata=all_metadata,
            combo_ids=all_combo_ids
        )
    
    # Generate and save build report
    log.info("\n" + "-"*80)
    log.info("Generating build report...")
    report = report_collector.generate_report()
    report_dir = output_dir / "reports"
    save_report(report, report_dir)
    
    log.info("\n" + "="*80)
    log.info("Segment building complete!")
    if total_filtered_years > 0:
        log.info(f"Total filtered years: {total_filtered_years}")
        log.info(f"Filtered year plots saved to: {filtered_plots_dir}")
    log.info(f"Output saved to: {output_dir}")
    log.info(f"Build report saved to: {report_dir}")
    log.info(f"Log file: {log_file}")
    log.info("="*80)


if __name__ == '__main__':
    main()
