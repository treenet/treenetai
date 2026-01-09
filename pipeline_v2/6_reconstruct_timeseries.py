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
    output_df: Optional[pd.DataFrame],
    gap: Dict,
    segment_days: int = 30,
    steps_per_hour: int = 6
) -> Optional[Dict]:
    """
    Create a 30-day segment centered on a gap.
    
    Ensures:
    - Segment contains the gap
    - Segment has no other large gaps
    - Segment boundaries don't overlap with other gaps
    
    Args:
        input_df: Input DataFrame (10-min resolution) with 'ts' column
        output_df: Output DataFrame (hourly resolution) with 'ts' column, or None
        gap: Gap dictionary from analyze_gaps()
        segment_days: Length of segment (days)
        steps_per_hour: Steps per hour in input (6 for 10-min)
    
    Returns:
        Dictionary with:
        - input_segment: Input DataFrame segment
        - output_segment: Output DataFrame segment (or None)
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
    
    # Align segment start to 10-minute boundary (floor to nearest :00, :10, :20, etc.)
    seg_start = seg_start.floor('10min')
    seg_end = seg_start + pd.Timedelta(days=segment_days)
    
    # Extract candidate input segment
    input_seg = input_df[
        (input_df['ts'] >= seg_start) & (input_df['ts'] < seg_end)
    ].copy()
    
    # Check input segment validity
    expected_input_samples = segment_days * 24 * steps_per_hour  # 4320 for 30 days
    
    if len(input_seg) == 0:
        return None
    
    # Must have reasonable coverage (allow some gaps but not too many)
    coverage = len(input_seg) / expected_input_samples
    if coverage < 0.8:  # Require at least 80% coverage
        return None
    
    time_span_input = (input_seg['ts'].iloc[-1] - input_seg['ts'].iloc[0]).days
    if time_span_input < segment_days - 1:  # Allow 1 day tolerance
        return None
    
    # Handle output segment
    output_seg = None
    if output_df is not None:
        output_seg = output_df[
            (output_df['ts'] >= seg_start) & (output_df['ts'] < seg_end)
        ].copy()
        
        expected_output_samples = segment_days * 24  # 720 for 30 days
        
        if len(output_seg) > 0:
            time_span_output = (output_seg['ts'].iloc[-1] - output_seg['ts'].iloc[0]).days
            if time_span_output < segment_days - 1:
                output_seg = None  # Output not valid, but can still proceed
    
    # Verify the gap we're filling is within this segment
    gap_in_segment = (input_seg['ts'].min() <= gap_start) and (input_seg['ts'].max() >= gap_end)
    
    if not gap_in_segment:
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


def normalize_segment(
    input_seg: pd.DataFrame,
    output_seg: pd.DataFrame,
    normalizer: Normalizer
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Normalize a single segment and return arrays ready for model.
    
    Args:
        input_seg: Input segment DataFrame (10-min resolution)
        output_seg: Output segment DataFrame (hourly resolution)
        normalizer: Normalizer instance
    
    Returns:
        Tuple of (input_array, output_array, input_params, output_params)
    """
    # Compute normalization params for this segment
    input_min, input_diff = normalizer.compute_normalization_params(input_seg)
    output_min, output_diff = normalizer.compute_normalization_params(output_seg)
    
    # Normalize
    input_norm = normalizer.normalize(input_seg, input_min, input_diff)
    output_norm = normalizer.normalize(output_seg, output_min, output_diff)
    
    # Convert to arrays
    input_array = input_norm.values.astype(np.float32)
    output_array = output_norm.values.astype(np.float32)
    
    # Store params
    input_params = {'min': input_min, 'diff': input_diff}
    output_params = {'min': output_min, 'diff': output_diff}
    
    return input_array, output_array, input_params, output_params


def denormalize_predictions(
    predictions: np.ndarray,
    output_params: Dict,
    column_names: List[str]
) -> pd.DataFrame:
    """
    Denormalize model predictions back to original scale.
    
    Args:
        predictions: Normalized predictions (samples, timesteps, channels)
        output_params: Dictionary with 'min' and 'diff' normalization params
        column_names: List of output column names
    
    Returns:
        DataFrame with denormalized predictions
    """
    denorm = np.zeros_like(predictions)
    
    for i, col in enumerate(column_names):
        vmin = output_params['min'].get(col, 0.0)
        vdiff = output_params['diff'].get(col, 1.0)
        denorm[:, :, i] = predictions[:, :, i] * vdiff + vmin
    
    return denorm


def fill_gap_in_timeseries(
    original_df: pd.DataFrame,
    predictions: np.ndarray,
    gap_info: Dict,
    segment_start: pd.Timestamp,
    column_names: List[str]
) -> pd.DataFrame:
    """
    Fill a gap in the original time series with model predictions.
    
    Args:
        original_df: Original time series with gaps
        predictions: Denormalized predictions for the segment
        gap_info: Gap dictionary with start/end timestamps
        segment_start: Start timestamp of the prediction segment
        column_names: List of output column names
    
    Returns:
        Updated DataFrame with gap filled
    """
    gap_start = gap_info['start_time']
    gap_end = gap_info['end_time']
    
    # Create hourly timestamps for predictions
    pred_timestamps = pd.date_range(
        start=segment_start,
        periods=predictions.shape[1],
        freq='1h'
    )
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame(
        predictions[0],  # First (only) sample
        index=pred_timestamps,
        columns=column_names
    )
    
    # Find which prediction indices correspond to the gap
    gap_mask = (pred_df.index > gap_start) & (pred_df.index < gap_end)
    gap_predictions = pred_df.loc[gap_mask]
    
    # Fill gap in original data
    result = original_df.copy()
    for ts, row in gap_predictions.iterrows():
        if ts not in result.index:
            # Insert new row
            result.loc[ts] = row
        else:
            # Update existing row (may have NaN)
            for col in column_names:
                if pd.isna(result.loc[ts, col]):
                    result.loc[ts, col] = row[col]
    
    return result.sort_index()


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
    
    # =========================================================================
    # Step 1: Load raw sensor data
    # =========================================================================
    print("  Loading sensor data...")
    thermo_df = loaders.load_thermometer_l1(thermometer_id)
    hygro_df = loaders.load_hygrometer_l1(hygrometer_id)
    dendro_l2_df = loaders.load_dendrometer_l2(dendrometer_id)
    dendro_lm_df = loaders.load_dendrometer_lm(dendrometer_id)
    meteo_df = loaders.load_meteotest_data(site_id)
    
    if any(df is None for df in [thermo_df, hygro_df, dendro_l2_df, meteo_df]):
        print("  Error: Missing required data files")
        return None, {'site_id': site_id, 'error': 'Missing data'}
    
    print(f"    Thermometer: {len(thermo_df)} samples")
    print(f"    Hygrometer: {len(hygro_df)} samples")
    print(f"    Dendrometer L2: {len(dendro_l2_df)} samples")
    print(f"    Dendrometer LM: {len(dendro_lm_df) if dendro_lm_df is not None else 0} samples")
    
    # =========================================================================
    # Step 2: Process data into input/output format
    # =========================================================================
    print("  Processing data...")
    
    # Process sensor DataFrames to UTC index
    temp_df = processor.process_sensor_dataframe(thermo_df)
    rh_df = processor.process_sensor_dataframe(hygro_df)
    stem_df = processor.process_sensor_dataframe(dendro_l2_df)
    
    # Process meteo to daily civil time
    meteo_daily = processor.process_meteo_daily(meteo_df)
    
    # Create input array (11 channels, 10-min resolution)
    input_df = processor.merger.create_input_array(
        temp_df, rh_df, stem_df, meteo_daily
    )
    
    print(f"    Input array: {len(input_df)} samples, {len(input_df.columns)} channels")
    print(f"    Date range: {input_df.index.min()} to {input_df.index.max()}")
    
    # Create output if LM data available (for validation)
    if dendro_lm_df is not None:
        lm_processed = processor.process_sensor_dataframe(
            dendro_lm_df, keep_all_columns=True
        )
        output_df = processor.merger.create_target_array(lm_processed)
        print(f"    Output array: {len(output_df)} samples (for validation)")
    else:
        output_df = None
    
    # =========================================================================
    # Step 3: Analyze gaps in input data
    # =========================================================================
    print("  Analyzing gaps...")
    
    # Convert to format expected by analyze_gaps (with 'ts' column)
    input_with_ts = input_df.reset_index()
    input_with_ts.rename(columns={'index': 'ts'}, inplace=True)
    
    gaps = analyze_gaps(
        input_with_ts,
        max_gap_days=args.max_gap_days,
        expected_freq='10T'
    )
    
    print(f"    Found {len(gaps)} gaps ≤ {args.max_gap_days} days")
    
    if len(gaps) == 0:
        print("    No gaps to fill!")
        metrics = {
            'site_id': site_id,
            'total_gaps': 0,
            'filled_gaps': 0,
            'skipped_gaps': 0
        }
        return input_df, metrics
    
    # Print gap statistics
    gap_days = [g['gap_days'] for g in gaps]
    print(f"    Gap lengths: min={min(gap_days):.2f}d, max={max(gap_days):.2f}d, mean={np.mean(gap_days):.2f}d")
    
    # =========================================================================
    # Step 4: Create segments around gaps and run model inference
    # =========================================================================
    print("  Creating gap-filling segments...")
    
    filled_gaps = 0
    skipped_gaps = 0
    mae_values = []
    
    # Input channel order (must match training)
    input_channels = ['temp_treenet', 'rh_treenet', 'stem', 
                      'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy']
    # Output channel order
    output_channels = ['local_T', 'local_RH', 'stem']
    
    # Track reconstructed output (start with original output_df if available)
    reconstructed_output = output_df.copy() if output_df is not None else None
    
    # Prepare temporary output DataFrame for segments without LM data
    if output_df is None:
        # Create empty output structure matching input timestamps (hourly)
        hourly_idx = input_df.index[input_df.index.minute == 0]
        reconstructed_output = pd.DataFrame(
            index=hourly_idx,
            columns=output_channels
        )
    
    segment_days = args.segment_days
    steps_per_hour = 6  # 10-min resolution
    
    for gap_idx, gap in enumerate(gaps):
        if args.verbose:
            print(f"\n    Processing gap {gap_idx + 1}/{len(gaps)}: "
                  f"{gap['start_time']} to {gap['end_time']} ({gap['gap_days']:.2f} days)")
        
        # Create segment around this gap
        segment_result = create_segment_around_gap(
            input_df=input_with_ts,
            output_df=output_df.reset_index() if output_df is not None else None,
            gap=gap,
            segment_days=segment_days,
            steps_per_hour=steps_per_hour
        )
        
        if segment_result is None:
            if args.verbose:
                print(f"      Could not create valid segment for this gap")
            skipped_gaps += 1
            continue
        
        input_seg = segment_result['input_segment']
        output_seg = segment_result['output_segment']
        seg_start = segment_result['segment_start']
        seg_end = segment_result['segment_end']
        
        # Set proper index for input/output segments
        input_seg = input_seg.set_index('ts') if 'ts' in input_seg.columns else input_seg
        if output_seg is not None and 'ts' in output_seg.columns:
            output_seg = output_seg.set_index('ts')
        
        # Convert segment start/end to proper timestamps with consistent dtype
        if not isinstance(seg_start, pd.Timestamp):
            seg_start = pd.Timestamp(seg_start, tz='UTC')
        if seg_start.tzinfo is None:
            seg_start = seg_start.tz_localize('UTC')
            
        # Select only required columns
        input_seg_cols = input_seg[[c for c in input_channels if c in input_seg.columns]].copy()
        
        # Create a complete 10-minute index for the segment
        expected_input_length = segment_days * 24 * steps_per_hour  # 4320
        complete_index = pd.date_range(
            start=seg_start,
            periods=expected_input_length,
            freq='10min',
            tz='UTC'
        )
        
        # Ensure index dtype compatibility (convert to same resolution)
        # This fixes reindex mismatch when data uses datetime64[us] vs datetime64[ns]
        input_seg_cols.index = pd.to_datetime(input_seg_cols.index).tz_convert('UTC')
        
        # Reindex to complete grid - this creates NaN for missing timestamps
        input_seg_reindexed = input_seg_cols.reindex(complete_index)
        
        # Create mask BEFORE filling: 1 = valid data, 0 = gap/missing
        input_mask = (~input_seg_reindexed.isna()).astype(np.float32)
        
        # Fill NaN values with interpolation for model input
        input_seg_filled = input_seg_reindexed.interpolate(method='linear', limit_direction='both')
        input_seg_filled = input_seg_filled.ffill().bfill()  # Handle edges
        
        if input_seg_filled.isna().any().any():
            if args.verbose:
                nan_counts = input_seg_filled.isna().sum()
                nan_cols = nan_counts[nan_counts > 0].to_dict()
                print(f"      Segment has unfillable NaNs: {nan_cols}, skipping")
            skipped_gaps += 1
            continue
        
        # Verify length
        if len(input_seg_filled) != expected_input_length:
            if args.verbose:
                print(f"      Segment length {len(input_seg_filled)} != expected {expected_input_length}, skipping")
            skipped_gaps += 1
            continue
        
        # =====================================================================
        # Step 5: Normalize, predict, denormalize
        # =====================================================================
        
        # For segment-level normalization, compute params for this segment
        input_min, input_diff = normalizer.compute_normalization_params(input_seg_filled)
        
        # Create dummy output params (we'll use segment-level for consistency)
        if output_seg is not None and len(output_seg) > 0:
            output_seg_cols = output_seg[[c for c in output_channels if c in output_seg.columns]]
            # Fill NaN in output for param computation only
            output_seg_filled = output_seg_cols.ffill().bfill()
            output_min, output_diff = normalizer.compute_normalization_params(output_seg_filled)
        else:
            # Use input-based approximations for output
            output_min = {'local_T': input_min.get('temp_treenet', 0),
                         'local_RH': input_min.get('rh_treenet', 0),
                         'stem': input_min.get('stem', 0)}
            output_diff = {'local_T': input_diff.get('temp_treenet', 1),
                          'local_RH': input_diff.get('rh_treenet', 1),
                          'stem': input_diff.get('stem', 1)}
        
        # Normalize input
        input_norm = normalizer.normalize(input_seg_filled, input_min, input_diff)
        input_array = input_norm.values.astype(np.float32)
        
        # Reshape for model: (1, timesteps, channels)
        input_array = input_array.reshape(1, -1, len(input_channels))
        
        # Reshape mask to match input shape
        mask_array = input_mask.values.astype(np.float32)
        mask_array = mask_array.reshape(1, -1, len(input_channels))
        
        # Model prediction - TCN expects [input_x, input_mask]
        try:
            predictions = model.predict([input_array, mask_array], verbose=0)
            # Model returns [recon_output, hourly_output]
            # We want the hourly_output (second element)
            hourly_predictions = predictions[1]
        except Exception as e:
            if args.verbose:
                print(f"      Model prediction failed: {e}")
            skipped_gaps += 1
            continue
        
        # Denormalize predictions
        denorm_preds = denormalize_predictions(
            hourly_predictions, 
            {'min': output_min, 'diff': output_diff},
            output_channels
        )
        
        # =====================================================================
        # Step 6: Merge predictions into reconstructed time series
        # =====================================================================
        if reconstructed_output is not None:
            reconstructed_output = fill_gap_in_timeseries(
                original_df=reconstructed_output,
                predictions=denorm_preds,
                gap_info=gap,
                segment_start=seg_start,
                column_names=output_channels
            )
        
        filled_gaps += 1
        
        # Calculate MAE if ground truth available
        if output_seg is not None and len(output_seg) > 0:
            # Compare predictions with actual values in gap region
            gap_start = gap['start_time']
            gap_end = gap['end_time']
            
            pred_ts = pd.date_range(start=seg_start, periods=hourly_predictions.shape[1], freq='1h')
            pred_df = pd.DataFrame(denorm_preds[0], index=pred_ts, columns=output_channels)
            
            # Find overlap with output
            gap_mask = (pred_df.index > gap_start) & (pred_df.index < gap_end)
            gap_preds = pred_df.loc[gap_mask]
            
            if len(gap_preds) > 0 and 'stem' in output_seg.columns:
                actual = output_seg.loc[output_seg.index.intersection(gap_preds.index), 'stem']
                if len(actual) > 0:
                    mae = np.abs(gap_preds.loc[actual.index, 'stem'] - actual).mean()
                    if not np.isnan(mae):
                        mae_values.append(mae)
        
        if (gap_idx + 1) % 50 == 0:
            print(f"    Processed {gap_idx + 1}/{len(gaps)} gaps...")
    
    # =========================================================================
    # Step 7: Calculate metrics
    # =========================================================================
    print(f"  Filled {filled_gaps}/{len(gaps)} gaps ({skipped_gaps} skipped)")
    
    metrics = {
        'site_id': site_id,
        'thermometer_id': thermometer_id,
        'hygrometer_id': hygrometer_id,
        'dendrometer_id': dendrometer_id,
        'total_gaps': len(gaps),
        'filled_gaps': filled_gaps,
        'skipped_gaps': skipped_gaps,
        'fill_rate': filled_gaps / len(gaps) if len(gaps) > 0 else 1.0,
        'mae_stem': np.mean(mae_values) if mae_values else np.nan,
        'mae_count': len(mae_values)
    }
    
    if mae_values:
        print(f"  MAE (stem): {np.mean(mae_values):.2f} μm (from {len(mae_values)} validated gaps)")
    
    return reconstructed_output, metrics


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
        
        # Filter by variable_name (matching loaders.py pattern)
        thermometers = site_sensors[site_sensors['variable_name'] == 'air temperature']['series_id'].tolist()
        hygrometers = site_sensors[site_sensors['variable_name'] == 'relative humidity']['series_id'].tolist()
        dendrometers = site_sensors[site_sensors['variable_name'] == 'tree stem radius change']['series_id'].tolist()
        
        if not (thermometers and hygrometers and dendrometers):
            print(f"\nSkipping site {site_id}: Missing sensor types")
            print(f"  Thermometers: {len(thermometers)}, Hygrometers: {len(hygrometers)}, Dendrometers: {len(dendrometers)}")
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
            # Reset index for feather format
            save_df = reconstructed_df.reset_index()
            save_df.rename(columns={'index': 'ts'}, inplace=True)
            save_df.to_feather(save_path)
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
