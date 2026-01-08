#!/usr/bin/env python3
"""
Reconstruct complete multi-year time series by filling gaps using trained model.

This script:
1. Loads a trained gap-filling model
2. Analyzes test site data to identify gaps ≤ max_gap_days
3. Creates 30-day segments around each gap
4. Uses model to fill gaps with predictions
5. Reconstructs complete time series by patching filled values
6. Saves reconstructed time series with metrics

Usage:
    python 6_reconstruct_timeseries.py \\
        --model-path experiments/20260108_134031/best_model.keras \\
        --test-sites 3,32,43 \\
        --max-gap-days 12 \\
        --norm-scope segment

Author: Lukovic
Date: 2026-01-08
"""

import argparse
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock
from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor
from src.data.segmentation import Normalizer, SegmentExtractor
from src.utils import setup_logging, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reconstruct time series by filling gaps with trained model'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.keras file)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data',
        help='Root directory with raw sensor data'
    )
    parser.add_argument(
        '--meteo-root',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data',
        help='Directory with meteo CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lukovic/data/treenet/pipeline_v2/reconstructions',
        help='Output directory for reconstructed time series'
    )
    parser.add_argument(
        '--test-sites',
        type=str,
        default='3,32,43',
        help='Comma-separated list of test site IDs'
    )
    parser.add_argument(
        '--sensor-combinations',
        type=str,
        default='',
        help='Force specific sensor combinations as "T,H,D;T,H,D" (optional)'
    )
    parser.add_argument(
        '--max-gap-days',
        type=int,
        default=12,
        help='Maximum gap length to fill (in days)'
    )
    parser.add_argument(
        '--segment-days',
        type=int,
        default=30,
        help='Length of segments for gap filling (days)'
    )
    parser.add_argument(
        '--norm-scope',
        type=str,
        default='segment',
        choices=['year', 'segment'],
        help='Normalization scope used during training'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def analyze_gaps(
    df: pd.DataFrame,
    max_gap_days: int = 12,
    expected_freq: str = '10T'
) -> List[Dict]:
    """
    Identify all gaps in time series that are ≤ max_gap_days.
    
    Args:
        df: DataFrame with 'ts' column (timestamps)
        max_gap_days: Maximum gap length to identify (days)
        expected_freq: Expected time frequency ('10T' for 10 min, '1H' for hourly)
    
    Returns:
        List of gap dictionaries with:
        - start_idx: Index before gap
        - end_idx: Index after gap
        - start_time: Timestamp before gap
        - end_time: Timestamp after gap
        - gap_days: Gap length in days
        - gap_samples: Number of missing samples
    """
    if 'ts' not in df.columns or len(df) < 2:
        return []
    
    # Calculate time differences
    df = df.sort_values('ts').reset_index(drop=True)
    time_diffs = df['ts'].diff()
    
    # Expected interval
    if expected_freq == '10T':
        expected = pd.Timedelta('10 minutes')
        tolerance = pd.Timedelta('15 minutes')
    elif expected_freq == '1H':
        expected = pd.Timedelta('1 hour')
        tolerance = pd.Timedelta('75 minutes')
    else:
        raise ValueError(f"Unknown frequency: {expected_freq}")
    
    # Find gaps
    gaps = []
    for idx in range(1, len(df)):
        diff = time_diffs.iloc[idx]
        
        if diff > tolerance:
            gap_days = diff.total_seconds() / (24 * 3600)
            
            if 0 < gap_days <= max_gap_days:
                gap_samples = int(diff / expected) - 1
                
                gaps.append({
                    'start_idx': idx - 1,
                    'end_idx': idx,
                    'start_time': df.iloc[idx - 1]['ts'],
                    'end_time': df.iloc[idx]['ts'],
                    'gap_days': gap_days,
                    'gap_samples': gap_samples
                })
    
    return gaps


def create_segment_around_gap(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    gap: Dict,
    segment_days: int = 30,
    steps_per_hour: int = 6
) -> Optional[Dict]:
    """
    Create a 30-day segment centered on a gap.
    
    Ensures:
    - Segment contains the gap
    - Segment has no other gaps
    - Segment boundaries don't overlap with other gaps
    
    Args:
        input_df: Input DataFrame (10-min resolution)
        output_df: Output DataFrame (hourly resolution)
        gap: Gap dictionary from analyze_gaps()
        segment_days: Length of segment (days)
        steps_per_hour: Steps per hour in input (6 for 10-min)
    
    Returns:
        Dictionary with:
        - input_segment: Input DataFrame segment
        - output_segment: Output DataFrame segment
        - segment_start: Start timestamp
        - segment_end: End timestamp
        - gap_info: Original gap dictionary
        Returns None if segment cannot be created
    """
    gap_start = gap['start_time']
    gap_end = gap['end_time']
    
    # Try centering segment on gap midpoint
    gap_midpoint = gap_start + (gap_end - gap_start) / 2
    seg_start = gap_midpoint - pd.Timedelta(days=segment_days // 2)
    seg_end = seg_start + pd.Timedelta(days=segment_days)
    
    # Extract candidate segment
    input_seg = input_df[
        (input_df['ts'] >= seg_start) & (input_df['ts'] < seg_end)
    ].copy()
    output_seg = output_df[
        (output_df['ts'] >= seg_start) & (output_df['ts'] < seg_end)
    ].copy()
    
    # Check segment validity
    expected_input_samples = segment_days * 24 * steps_per_hour  # 4320 for 30 days
    expected_output_samples = segment_days * 24  # 720 for 30 days
    
    # Must span exactly segment_days
    if len(input_seg) == 0 or len(output_seg) == 0:
        return None
    
    time_span_input = (input_seg['ts'].iloc[-1] - input_seg['ts'].iloc[0]).days
    time_span_output = (output_seg['ts'].iloc[-1] - output_seg['ts'].iloc[0]).days
    
    if time_span_input != segment_days or time_span_output != segment_days:
        return None
    
    # Check for other gaps in this segment (excluding the target gap)
    input_gaps_in_seg = analyze_gaps(input_seg, max_gap_days=999, expected_freq='10T')
    output_gaps_in_seg = analyze_gaps(output_seg, max_gap_days=999, expected_freq='1H')
    
    # Allow only the target gap
    if len(input_gaps_in_seg) > 1 or len(output_gaps_in_seg) > 1:
        return None
    
    # Verify the gap we're filling is within this segment
    gap_in_segment = False
    for g in input_gaps_in_seg:
        if abs((g['start_time'] - gap_start).total_seconds()) < 600:  # Within 10 min
            gap_in_segment = True
            break
    
    if not gap_in_segment and len(input_gaps_in_seg) > 0:
        # Found a different gap, invalid segment
        return None
    
    return {
        'input_segment': input_seg,
        'output_segment': output_seg,
        'segment_start': seg_start,
        'segment_end': seg_end,
        'gap_info': gap
    }


def load_model(model_path: Path) -> tf.keras.Model:
    """Load trained model with custom objects."""
    print(f"\nLoading model from {model_path}...")
    
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'TCNBlock': TCNBlock}
    )
    
    print(f"  Model loaded successfully")
    print(f"  Inputs: {[inp.shape for inp in model.inputs]}")
    print(f"  Outputs: {[out.shape for out in model.outputs]}")
    
    return model


def reconstruct_site(
    model: tf.keras.Model,
    site_id: int,
    thermometer_id: int,
    hygrometer_id: int,
    dendrometer_id: int,
    loaders: DataLoaders,
    processor: DataProcessor,
    normalizer: Normalizer,
    args: argparse.Namespace
) -> Tuple[pd.DataFrame, Dict]:
    """
    Reconstruct time series for one site-sensor combination.
    
    Args:
        model: Trained model
        site_id: Site ID
        thermometer_id: Thermometer series ID
        hygrometer_id: Hygrometer series ID
        dendrometer_id: Dendrometer series ID
        loaders: DataLoaders instance
        processor: DataProcessor instance
        normalizer: Normalizer instance
        args: Command line arguments
    
    Returns:
        Tuple of (reconstructed_df, metrics_dict)
    """
    print(f"\nReconstructing Site {site_id} (T={thermometer_id}, H={hygrometer_id}, D={dendrometer_id})")
    
    # Load raw sensor data
    print("  Loading sensor data...")
    thermo_df = loaders.load_thermometer(thermometer_id)
    hygro_df = loaders.load_hygrometer(hygrometer_id)
    dendro_l2_df = loaders.load_dendrometer_l2(dendrometer_id)
    dendro_lm_df = loaders.load_dendrometer_lm(dendrometer_id)
    meteo_df = loaders.load_meteotest_data(site_id)
    
    if any(df is None for df in [thermo_df, hygro_df, dendro_l2_df, meteo_df]):
        print("  Error: Missing required data files")
        return None, {}
    
    # Process data into input/output format
    print("  Processing data...")
    # TODO: Implement data processing and merging
    # This will combine T, RH, stem (L2), meteo, and doy into input format
    # And create hourly output with T, RH, stem (LM)
    
    # Analyze gaps
    print("  Analyzing gaps...")
    # TODO: Identify gaps in input data
    
    # Create segments around gaps
    print("  Creating gap-filling segments...")
    # TODO: Create 30-day segments around each gap
    
    # Model inference
    print("  Running model inference...")
    # TODO: Normalize segments, predict, denormalize
    
    # Reconstruct time series
    print("  Reconstructing time series...")
    # TODO: Merge predictions back into original data
    
    # Calculate metrics
    metrics = {
        'site_id': site_id,
        'total_gaps': 0,
        'filled_gaps': 0,
        'mae': 0.0,
        'rmse': 0.0
    }
    
    return None, metrics


def main():
    """Main function."""
    args = parse_args()
    
    # Setup
    output_dir = ensure_dir(Path(args.output_dir))
    log_file = output_dir / 'reconstruction.log'
    setup_logging(verbose=args.verbose, log_file=log_file)
    
    print("="*80)
    print("TreeNet AI Pipeline v2 - Time Series Reconstruction")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Max gap length: {args.max_gap_days} days")
    print(f"Segment length: {args.segment_days} days")
    print(f"Normalization: {args.norm_scope}")
    print("="*80)
    
    # Load model
    model = load_model(Path(args.model_path))
    
    # Initialize data components
    print("\nInitializing data loaders...")
    loaders = DataLoaders(
        data_root=Path(args.data_dir),
        meteo_root=Path(args.meteo_root)
    )
    processor = DataProcessor()
    normalizer = Normalizer(norm_scope=args.norm_scope)
    
    # Parse test sites
    test_sites = [int(s.strip()) for s in args.test_sites.split(',')]
    print(f"\nTest sites: {test_sites}")
    
    # Load metadata to find sensor combinations
    metadata = loaders.load_metadata()
    
    # Process each site
    all_metrics = []
    
    for site_id in test_sites:
        # Find available sensor combinations for this site
        site_sensors = metadata[metadata['site_id'] == site_id]
        thermometers = site_sensors[site_sensors['sensor_type'] == 'thermometer']['series_id'].tolist()
        hygrometers = site_sensors[site_sensors['sensor_type'] == 'hygrometer']['series_id'].tolist()
        dendrometers = site_sensors[site_sensors['sensor_type'] == 'dendrometer']['series_id'].tolist()
        
        if not (thermometers and hygrometers and dendrometers):
            print(f"\nSkipping site {site_id}: Missing sensor types")
            continue
        
        # Use first available combination
        # TODO: Allow specifying combinations via --sensor-combinations
        t_id = thermometers[0]
        h_id = hygrometers[0]
        d_id = dendrometers[0]
        
        # Reconstruct
        reconstructed_df, metrics = reconstruct_site(
            model=model,
            site_id=site_id,
            thermometer_id=t_id,
            hygrometer_id=h_id,
            dendrometer_id=d_id,
            loaders=loaders,
            processor=processor,
            normalizer=normalizer,
            args=args
        )
        
        if reconstructed_df is not None:
            # Save reconstructed time series
            save_path = output_dir / f"reconstructed_site{site_id}_T{t_id}_H{h_id}_D{d_id}.ftr"
            reconstructed_df.to_feather(save_path)
            print(f"  Saved: {save_path}")
            
            all_metrics.append(metrics)
    
    # Save summary metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = output_dir / 'reconstruction_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSummary metrics saved: {metrics_path}")
        print("\nReconstruction complete!")
    else:
        print("\nNo sites were successfully reconstructed.")


if __name__ == '__main__':
    main()
