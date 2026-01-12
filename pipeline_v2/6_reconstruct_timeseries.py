#!/usr/bin/env python3
"""
Unified time series reconstruction from intermediate files.

This script reconstructs time series using the pre-processed intermediate 
timeseries files created during training data preparation. Supports both
basic reconstruction and scale-aligned reconstruction.

Features:
1. Segment-by-segment sliding window reconstruction
2. Optional scale alignment using Nov-Dec overlap period
3. Validation against LM ground truth (when available)
4. Support for different denormalization modes

Alignment Strategy (--align-stem):
- Starts reconstruction in Nov/Dec of the prior year
- Uses Nov-Dec overlap to compute scale+offset between reconstructed and LM
- Applies transformation to align entire reconstructed stem channel

Usage examples:
    # Basic reconstruction without alignment
    python 6_reconstruct_timeseries.py --model-path /path/to/model.keras \\
        --year-start 2021 --year-end 2022
    
    # With stem scale alignment (recommended)
    python 6_reconstruct_timeseries.py --model-path /path/to/model.keras \\
        --year-start 2021 --year-end 2022 --align-stem

Author: TreeNet AI Pipeline v2
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import timedelta, datetime
import json
from glob import glob
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding
from src.data.segmentation import Normalizer

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
SEGMENT_DAYS = 30
INPUT_STEPS_PER_HOUR = 6
OUTPUT_STEPS_PER_HOUR = 1
INPUT_SAMPLES = SEGMENT_DAYS * 24 * INPUT_STEPS_PER_HOUR  # 4320
OUTPUT_SAMPLES = SEGMENT_DAYS * 24 * OUTPUT_STEPS_PER_HOUR  # 720

INPUT_CHANNELS = [
    'temp_treenet', 'rh_treenet', 'stem',
    'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy'
]
OUTPUT_CHANNELS = ['local_T', 'local_RH', 'stem']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reconstruct time series from intermediate files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction
  python 6_reconstruct_timeseries.py --model-path model.keras --year-start 2021
  
  # With scale alignment (recommended for stem)
  python 6_reconstruct_timeseries.py --model-path model.keras --year-start 2021 --align-stem
        """
    )
    
    # Required
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    
    # Data paths
    parser.add_argument(
        '--intermediate-dir', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data/intermediate_timeseries',
        help='Directory with intermediate timeseries files'
    )
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    # Time range
    parser.add_argument('--year-start', type=int, default=2021, help='Start year')
    parser.add_argument('--year-end', type=int, default=2022, help='End year')
    
    # Alignment options
    parser.add_argument('--align-stem', action='store_true',
                        help='Align stem scale using Nov-Dec overlap period')
    parser.add_argument('--align-start-month', type=int, default=11,
                        help='Month to start alignment period (default: November)')
    
    # Processing options
    parser.add_argument('--stride-hours', type=int, default=24,
                        help='Sliding window stride in hours')
    parser.add_argument('--combo-ids', type=int, nargs='+', default=None,
                        help='Specific combination IDs to process (default: all)')
    
    # Output options
    parser.add_argument('--output-mode', type=str, default='input_scale',
                        choices=['normalized', 'input_scale'],
                        help='Output value scale')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


class IntermediateReconstructor:
    """Reconstruct time series from intermediate files with optional alignment."""
    
    def __init__(self, model_path: str, verbose: bool = False):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'TCNBlock': TCNBlock, 'PositionalEncoding': PositionalEncoding}
        )
        self.verbose = verbose
        if verbose:
            print(f"Model loaded: {model_path}")
    
    def load_intermediate_file(self, file_path: str) -> pd.DataFrame:
        """Load intermediate timeseries file."""
        return pd.read_feather(file_path)
    
    def prepare_segment(self, df_segment: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict, bool]:
        """Prepare input array with segment-level normalization."""
        input_cols = INPUT_CHANNELS
        
        # Check for required columns
        if not all(col in df_segment.columns for col in input_cols):
            return None, None, {}, False
        
        input_data = df_segment[input_cols].values.astype(np.float32)
        
        if len(input_data) != INPUT_SAMPLES:
            return None, None, {}, False
        
        # Segment-level normalization
        norm_params = {}
        for i, col in enumerate(input_cols):
            min_val = np.nanmin(input_data[:, i])
            max_val = np.nanmax(input_data[:, i])
            diff = max_val - min_val
            norm_params[col] = {'min': float(min_val), 'max': float(max_val)}
            
            if diff > 1e-10:
                input_data[:, i] = (input_data[:, i] - min_val) / diff
            else:
                input_data[:, i] = 0.0
        
        # Create mask (1 where valid, 0 where NaN)
        mask = (~np.isnan(input_data)).astype(np.float32)
        input_array = np.nan_to_num(input_data, nan=0.0)
        
        return input_array, mask, norm_params, True
    
    def denormalize_output(self, output: np.ndarray, norm_params: Dict) -> np.ndarray:
        """Denormalize output using input parameters."""
        output_denorm = np.zeros_like(output)
        input_to_output = ['temp_treenet', 'rh_treenet', 'stem']
        
        for i, input_ch in enumerate(input_to_output):
            if input_ch in norm_params:
                min_val = norm_params[input_ch]['min']
                max_val = norm_params[input_ch]['max']
                if max_val - min_val > 1e-10:
                    output_denorm[:, i] = output[:, i] * (max_val - min_val) + min_val
                else:
                    output_denorm[:, i] = min_val
        
        return output_denorm
    
    def reconstruct_combination(
        self, 
        intermediate_file: str,
        year_start: int,
        year_end: int,
        stride_hours: int = 24,
        output_mode: str = 'input_scale',
        align_stem: bool = False,
        align_start_month: int = 11
    ) -> Tuple[pd.DataFrame, Dict]:
        """Reconstruct time series for a single combination."""
        
        # Load data
        df = self.load_intermediate_file(intermediate_file)
        if 'ts' not in df.columns:
            df['ts'] = pd.to_datetime(df.index)
        df['ts'] = pd.to_datetime(df['ts'])
        
        # Determine reconstruction start
        if align_stem:
            # Start from Nov of prior year for alignment
            recon_start = pd.Timestamp(year=year_start - 1, month=align_start_month, day=1, tz='UTC')
            align_end = pd.Timestamp(year=year_start, month=1, day=1, tz='UTC')
        else:
            recon_start = pd.Timestamp(year=year_start, month=1, day=1, tz='UTC')
            align_end = None
        
        recon_end = pd.Timestamp(year=year_end + 1, month=1, day=1, tz='UTC')
        
        # Filter data to reconstruction range
        df_range = df[(df['ts'] >= recon_start) & (df['ts'] < recon_end)]
        
        if len(df_range) < INPUT_SAMPLES:
            return None, {}
        
        # Sliding window reconstruction
        results = []
        stride_samples = stride_hours * INPUT_STEPS_PER_HOUR
        
        n_windows = (len(df_range) - INPUT_SAMPLES) // stride_samples + 1
        
        for i in range(n_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + INPUT_SAMPLES
            
            if end_idx > len(df_range):
                break
            
            segment_df = df_range.iloc[start_idx:end_idx]
            input_arr, mask, norm_params, is_valid = self.prepare_segment(segment_df)
            
            if not is_valid:
                continue
            
            # Predict
            pred = self.model.predict(
                [np.expand_dims(input_arr, 0), np.expand_dims(mask, 0)],
                verbose=0
            )
            pred_hourly = pred[1][0] if isinstance(pred, list) else pred[0]
            
            # Denormalize if needed
            if output_mode == 'input_scale':
                pred_hourly = self.denormalize_output(pred_hourly, norm_params)
            
            # Get timestamps for hourly output
            segment_start = segment_df['ts'].iloc[0]
            hourly_times = pd.date_range(start=segment_start, periods=OUTPUT_SAMPLES, freq='1H')
            
            segment_results = pd.DataFrame({
                'ts': hourly_times,
                'recon_T': pred_hourly[:, 0],
                'recon_RH': pred_hourly[:, 1],
                'recon_stem': pred_hourly[:, 2]
            })
            results.append(segment_results)
        
        if not results:
            return None, {}
        
        # Combine and average overlapping predictions
        all_results = pd.concat(results, ignore_index=True)
        reconstructed = all_results.groupby('ts').agg({
            'recon_T': 'mean',
            'recon_RH': 'mean',
            'recon_stem': 'mean'
        }).reset_index()
        
        # Scale alignment for stem
        metrics = {}
        if align_stem and align_end is not None:
            # Get LM data for alignment period
            df_align = df[(df['ts'] >= recon_start) & (df['ts'] < align_end)]
            if 'stem_lm' in df.columns or 'stem' in df.columns:
                lm_col = 'stem_lm' if 'stem_lm' in df.columns else 'stem'
                
                # Compute alignment using linear regression
                recon_align = reconstructed[(reconstructed['ts'] >= recon_start) & 
                                           (reconstructed['ts'] < align_end)]
                
                if len(recon_align) > 100:
                    # Get corresponding LM values
                    lm_hourly = df.set_index('ts')[lm_col].resample('1H').mean()
                    
                    merged = recon_align.set_index('ts').join(lm_hourly.rename('lm_stem'), how='inner')
                    valid = merged[['recon_stem', 'lm_stem']].dropna()
                    
                    if len(valid) > 10:
                        # Linear fit: lm = scale * recon + offset
                        from scipy import stats
                        slope, intercept, r, p, se = stats.linregress(valid['recon_stem'], valid['lm_stem'])
                        
                        # Apply alignment
                        reconstructed['recon_stem_aligned'] = reconstructed['recon_stem'] * slope + intercept
                        reconstructed['recon_stem'] = reconstructed['recon_stem_aligned']
                        
                        metrics['stem_alignment'] = {
                            'scale': float(slope),
                            'offset': float(intercept),
                            'r': float(r)
                        }
        
        # Filter to requested year range
        final_start = pd.Timestamp(year=year_start, month=1, day=1, tz='UTC')
        final_end = pd.Timestamp(year=year_end + 1, month=1, day=1, tz='UTC')
        reconstructed = reconstructed[(reconstructed['ts'] >= final_start) & 
                                      (reconstructed['ts'] < final_end)]
        
        return reconstructed, metrics


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Time Series Reconstruction")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Years: {args.year_start} - {args.year_end}")
    print(f"Stem alignment: {'enabled' if args.align_stem else 'disabled'}")
    print(f"Output mode: {args.output_mode}")
    
    # Initialize reconstructor
    reconstructor = IntermediateReconstructor(args.model_path, verbose=args.verbose)
    
    # Find intermediate files
    intermediate_files = sorted(glob(f"{args.intermediate_dir}/*.feather"))
    
    if args.combo_ids:
        intermediate_files = [f for f in intermediate_files 
                            if any(f"combo_{cid}_" in f for cid in args.combo_ids)]
    
    print(f"\nFound {len(intermediate_files)} intermediate files")
    
    # Process each combination
    all_metrics = {}
    
    for file_path in intermediate_files:
        combo_id = Path(file_path).stem
        print(f"\nProcessing: {combo_id}")
        
        result, metrics = reconstructor.reconstruct_combination(
            intermediate_file=file_path,
            year_start=args.year_start,
            year_end=args.year_end,
            stride_hours=args.stride_hours,
            output_mode=args.output_mode,
            align_stem=args.align_stem,
            align_start_month=args.align_start_month
        )
        
        if result is not None:
            output_file = output_dir / f"{combo_id}_reconstructed.feather"
            result.to_feather(str(output_file))
            print(f"  Saved: {output_file}")
            print(f"  Rows: {len(result)}")
            
            if metrics:
                all_metrics[combo_id] = metrics
                if 'stem_alignment' in metrics:
                    align = metrics['stem_alignment']
                    print(f"  Stem alignment: scale={align['scale']:.4f}, offset={align['offset']:.1f}, r={align['r']:.4f}")
        else:
            print(f"  SKIPPED - insufficient data")
    
    # Save metrics
    if all_metrics:
        with open(output_dir / 'reconstruction_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Reconstruction complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
