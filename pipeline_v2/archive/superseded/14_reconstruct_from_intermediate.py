#!/usr/bin/env python3
"""
Reconstruct time series from intermediate timeseries files (test set).

This script uses the pre-processed intermediate timeseries files that were
created during the training data preparation phase. These files already have:
- Input: 11-channel 10-min data (temp_treenet, rh_treenet, stem + meteo)
- Output: LM (ground truth) 3-channel 1-hour data

Benefits of using intermediate files:
1. Data is already preprocessed and aligned
2. LM ground truth is available for validation
3. Faster processing (no database queries needed)

Usage:
    python 14_reconstruct_from_intermediate.py \
        --model-path /path/to/best_model.keras \
        --year-start 2021 --year-end 2022 \
        --output-dir /path/to/output

Author: Lukovic
Date: 2026-01-12
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import timedelta
import json
from glob import glob
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding
from src.data.segmentation import Normalizer

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
SEGMENT_DAYS = 30
INPUT_STEPS_PER_HOUR = 6  # 10-min resolution
OUTPUT_STEPS_PER_HOUR = 1  # 1-hour resolution
INPUT_SAMPLES = SEGMENT_DAYS * 24 * INPUT_STEPS_PER_HOUR  # 4320
OUTPUT_SAMPLES = SEGMENT_DAYS * 24 * OUTPUT_STEPS_PER_HOUR  # 720

# Channel definitions
INPUT_CHANNELS = [
    'temp_treenet', 'rh_treenet', 'stem',  # Sensor channels (3)
    'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr',  # Meteo channels (7)
    'doy'  # Time channel (1)
]
OUTPUT_CHANNELS = ['local_T', 'local_RH', 'stem']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reconstruct time series from intermediate files'
    )
    
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model (.keras file)'
    )
    parser.add_argument(
        '--intermediate-dir', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data/intermediate_timeseries',
        help='Directory containing intermediate timeseries files'
    )
    parser.add_argument(
        '--year-start', type=int, default=2021,
        help='Start year for reconstruction'
    )
    parser.add_argument(
        '--year-end', type=int, default=2022,
        help='End year for reconstruction'
    )
    parser.add_argument(
        '--overlap-days', type=int, default=5,
        help='Overlap between segments in days'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to save reconstructed files'
    )
    parser.add_argument(
        '--max-combinations', type=int, default=None,
        help='Maximum number of combinations to process (for testing)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed progress'
    )
    
    return parser.parse_args()


class IntermediateReconstructor:
    """Reconstruct time series from intermediate files."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        overlap_days: int = 5,
        max_gap_days: int = 12
    ):
        self.model = model
        self.overlap_days = overlap_days
        self.max_gap_days = max_gap_days
        self.stride_hours = (30 - overlap_days) * 24  # Hours between segment starts
        
        # For per-segment normalization
        self.normalizer = Normalizer()
    
    def discover_combinations(
        self,
        intermediate_dir: str,
        year_start: int,
        year_end: int
    ) -> List[Dict]:
        """Discover test combinations with data in the specified year range."""
        test_inputs = glob(os.path.join(intermediate_dir, 'test_input_*.ftr'))
        
        combinations = []
        for input_file in test_inputs:
            # Parse combination info from filename
            basename = os.path.basename(input_file)
            combo_str = basename.replace('test_input_', '').replace('.ftr', '')
            
            # Parse site and sensor IDs
            parts = combo_str.split('_')
            site_id = int(parts[0].replace('site', ''))
            thermo_id = int(parts[1].replace('T', ''))
            hygro_id = int(parts[2].replace('H', ''))
            dendro_id = int(parts[3].replace('D', ''))
            
            # Quick check for year range
            df = pd.read_feather(input_file)
            df['ts'] = pd.to_datetime(df['ts'])
            mask = (df['ts'].dt.year >= year_start) & (df['ts'].dt.year <= year_end)
            samples_in_range = mask.sum()
            
            if samples_in_range > INPUT_SAMPLES:  # At least one full segment
                # Check for matching output file
                output_file = input_file.replace('test_input_', 'test_output_')
                if os.path.exists(output_file):
                    combinations.append({
                        'combo_id': combo_str,
                        'site_id': site_id,
                        'thermo_id': thermo_id,
                        'hygro_id': hygro_id,
                        'dendro_id': dendro_id,
                        'input_file': input_file,
                        'output_file': output_file,
                        'samples_in_range': samples_in_range
                    })
        
        return combinations
    
    def create_segment_grid(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Create overlapping segment grid covering the time range."""
        segments = []
        
        # Start at 10-min boundary
        current_start = start_time.floor('10min')
        
        while current_start + pd.Timedelta(days=SEGMENT_DAYS) <= end_time:
            seg_end = current_start + pd.Timedelta(days=SEGMENT_DAYS)
            segments.append((current_start, seg_end))
            
            # Move to next segment (with overlap)
            current_start = current_start + pd.Timedelta(hours=self.stride_hours)
        
        # Add final segment if needed
        if current_start < end_time:
            final_start = end_time - pd.Timedelta(days=SEGMENT_DAYS)
            final_start = final_start.floor('10min')
            if final_start >= start_time and (final_start, end_time) not in segments:
                segments.append((final_start, end_time))
        
        return segments
    
    def prepare_segment(
        self,
        input_df: pd.DataFrame,
        seg_start: pd.Timestamp,
        seg_end: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray, Dict, bool]:
        """
        Prepare a segment for model inference.
        
        Returns:
            (input_array, mask_array, norm_params, is_valid)
        """
        # Create complete 10-min index for segment
        complete_idx = pd.date_range(
            start=seg_start,
            periods=INPUT_SAMPLES,
            freq='10min',
            tz='UTC'
        )
        
        # Extract segment from input
        segment = input_df.loc[
            (input_df.index >= seg_start) & (input_df.index < seg_end)
        ].copy()
        
        # Reindex to complete grid (creates NaN for missing)
        segment_full = segment.reindex(complete_idx)
        
        # Create mask before filling
        mask = (~segment_full.isna()).astype(np.float32)
        
        # Check coverage
        coverage = mask.values.mean()
        if coverage < 0.5:  # Require at least 50% coverage
            return None, None, None, False
        
        # Fill NaN with forward/backward fill for inference
        segment_filled = segment_full.ffill().bfill()
        
        # Per-segment normalization
        values = segment_filled.values
        norm_params = {}
        values_norm = np.zeros_like(values)
        
        for i, col in enumerate(segment_filled.columns):
            col_data = values[:, i]
            min_val = np.nanmin(col_data)
            max_val = np.nanmax(col_data)
            
            norm_params[col] = {'min': min_val, 'max': max_val}
            
            if max_val > min_val:
                values_norm[:, i] = (col_data - min_val) / (max_val - min_val)
            else:
                values_norm[:, i] = 0.5
        
        # Shape for model: (1, 4320, 11)
        input_array = values_norm.reshape(1, INPUT_SAMPLES, -1).astype(np.float32)
        mask_array = mask.values.reshape(1, INPUT_SAMPLES, -1).astype(np.float32)
        
        return input_array, mask_array, norm_params, True
    
    def denormalize_output(
        self,
        output: np.ndarray,
        norm_params: Dict
    ) -> np.ndarray:
        """Denormalize output using input normalization parameters."""
        output_denorm = np.zeros_like(output)
        
        # Map output channels to input channels
        channel_map = {
            0: 'temp_treenet',  # local_T ← temp_treenet
            1: 'rh_treenet',    # local_RH ← rh_treenet
            2: 'stem'           # stem ← stem
        }
        
        for i, input_ch in channel_map.items():
            if input_ch in norm_params:
                min_val = norm_params[input_ch]['min']
                max_val = norm_params[input_ch]['max']
                
                if max_val > min_val:
                    output_denorm[:, i] = output[:, i] * (max_val - min_val) + min_val
                else:
                    output_denorm[:, i] = min_val
            else:
                output_denorm[:, i] = output[:, i]
        
        return output_denorm
    
    def reconstruct_combination(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        year_start: int,
        year_end: int,
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Reconstruct a single combination.
        
        Returns:
            (reconstructed_df, metrics_dict)
        """
        # Filter to year range
        mask_in = (input_df.index.year >= year_start) & (input_df.index.year <= year_end)
        mask_out = (output_df.index.year >= year_start) & (output_df.index.year <= year_end)
        
        input_filtered = input_df[mask_in].copy()
        output_filtered = output_df[mask_out].copy()
        
        if len(input_filtered) < INPUT_SAMPLES:
            raise ValueError(f"Insufficient data: {len(input_filtered)} samples")
        
        # Create segment grid
        start_time = input_filtered.index.min()
        end_time = input_filtered.index.max()
        segments = self.create_segment_grid(start_time, end_time)
        
        if verbose:
            print(f"  Segment grid: {len(segments)} segments")
        
        # Process each segment
        segment_outputs = []
        segments_processed = 0
        segments_skipped = 0
        
        for i, (seg_start, seg_end) in enumerate(segments):
            # Prepare segment
            input_arr, mask_arr, norm_params, is_valid = self.prepare_segment(
                input_filtered, seg_start, seg_end
            )
            
            if not is_valid:
                segments_skipped += 1
                continue
            
            # Run model inference (model expects [input, mask])
            # Model outputs: [reconstructed_input (4320, 11), final_output (720, 3)]
            pred = self.model.predict([input_arr, mask_arr], verbose=0)
            pred_output = pred[1] if isinstance(pred, list) else pred  # Get final output
            pred_squeezed = pred_output[0]  # (720, 3)
            
            # Denormalize
            pred_denorm = self.denormalize_output(pred_squeezed, norm_params)
            
            # Create output timestamps (aligned to hour boundaries)
            # Output starts at the first full hour within the segment
            output_start = seg_start.ceil('1h')
            output_times = pd.date_range(
                start=output_start,
                periods=OUTPUT_SAMPLES,
                freq='1h',
                tz='UTC'
            )
            
            segment_outputs.append({
                'times': output_times,
                'values': pred_denorm,
                'seg_start': seg_start,
                'seg_end': seg_end
            })
            
            segments_processed += 1
        
        if not segment_outputs:
            raise ValueError("No valid segments processed")
        
        if verbose:
            print(f"  Processed: {segments_processed}, Skipped: {segments_skipped}")
        
        # Merge segments with weighted averaging in overlaps
        merged_df = self._merge_segments(segment_outputs)
        
        # Compute metrics against LM ground truth
        metrics = self._compute_metrics(merged_df, output_filtered)
        metrics['segments_processed'] = segments_processed
        metrics['segments_skipped'] = segments_skipped
        
        return merged_df, metrics
    
    def _merge_segments(
        self,
        segment_outputs: List[Dict]
    ) -> pd.DataFrame:
        """Merge overlapping segments using weighted averaging."""
        # Collect all unique timestamps
        all_times = set()
        for seg in segment_outputs:
            all_times.update(seg['times'])
        
        all_times = sorted(all_times)
        
        # Create accumulator arrays
        n_times = len(all_times)
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        
        value_sum = np.zeros((n_times, 3))
        weight_sum = np.zeros((n_times, 3))
        
        # Weighted merging
        for seg in segment_outputs:
            times = seg['times']
            values = seg['values']
            
            # Create weights (higher in center)
            n_samples = len(times)
            weights = np.ones(n_samples)
            
            # Taper edges
            overlap_samples = self.overlap_days * 24
            if n_samples > 2 * overlap_samples:
                for j in range(overlap_samples):
                    w = (j + 1) / (overlap_samples + 1)
                    weights[j] = w
                    weights[-(j+1)] = w
            
            # Accumulate
            for j, t in enumerate(times):
                idx = time_to_idx[t]
                for ch in range(3):
                    val = values[j, ch]
                    if not np.isnan(val):
                        value_sum[idx, ch] += val * weights[j]
                        weight_sum[idx, ch] += weights[j]
        
        # Compute weighted average
        with np.errstate(divide='ignore', invalid='ignore'):
            merged_values = np.where(
                weight_sum > 0,
                value_sum / weight_sum,
                np.nan
            )
        
        # Create output DataFrame
        result_df = pd.DataFrame(
            merged_values,
            index=pd.DatetimeIndex(all_times),
            columns=OUTPUT_CHANNELS
        )
        
        return result_df
    
    def _compute_metrics(
        self,
        reconstructed: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Dict:
        """Compute metrics comparing reconstruction to LM ground truth."""
        metrics = {}
        
        # Align to common timestamps
        common_idx = reconstructed.index.intersection(ground_truth.index)
        
        if len(common_idx) == 0:
            return {'n_common_samples': 0}
        
        recon_aligned = reconstructed.loc[common_idx]
        gt_aligned = ground_truth.loc[common_idx]
        
        metrics['n_common_samples'] = int(len(common_idx))
        
        for ch in OUTPUT_CHANNELS:
            if ch not in gt_aligned.columns:
                continue
            
            recon_vals = recon_aligned[ch].values.astype(float)
            gt_vals = gt_aligned[ch].values.astype(float)
            
            # Only compare where ground truth is valid
            valid_mask = ~np.isnan(gt_vals) & ~np.isnan(recon_vals)
            n_valid = int(valid_mask.sum())
            
            if n_valid < 10:
                continue
            
            recon_valid = recon_vals[valid_mask]
            gt_valid = gt_vals[valid_mask]
            
            # Compute metrics
            mae = float(np.mean(np.abs(recon_valid - gt_valid)))
            rmse = float(np.sqrt(np.mean((recon_valid - gt_valid)**2)))
            
            # Correlation
            if np.std(gt_valid) > 0 and np.std(recon_valid) > 0:
                corr = float(np.corrcoef(recon_valid, gt_valid)[0, 1])
            else:
                corr = float('nan')
            
            # R² score
            ss_res = np.sum((gt_valid - recon_valid)**2)
            ss_tot = np.sum((gt_valid - np.mean(gt_valid))**2)
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float('nan')
            
            # Normalized MAE
            gt_range = float(np.nanmax(gt_valid) - np.nanmin(gt_valid))
            mae_norm = float(mae / gt_range) if gt_range > 0 else float('nan')
            
            metrics[f'{ch}_mae'] = mae
            metrics[f'{ch}_rmse'] = rmse
            metrics[f'{ch}_corr'] = corr
            metrics[f'{ch}_r2'] = r2
            metrics[f'{ch}_mae_norm'] = mae_norm
            metrics[f'{ch}_n_valid'] = n_valid
        
        return metrics


def main():
    args = parse_args()
    
    print("=" * 70)
    print("TreeNet AI - Reconstruction from Intermediate Files")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Years: {args.year_start} - {args.year_end}")
    print(f"Overlap: {args.overlap_days} days")
    print(f"Output: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'TCNBlock': TCNBlock,
            'PositionalEncoding': PositionalEncoding
        }
    )
    print("Model loaded successfully\n")
    
    # Create reconstructor
    reconstructor = IntermediateReconstructor(
        model=model,
        overlap_days=args.overlap_days
    )
    
    # Discover combinations
    print("Discovering combinations...")
    combinations = reconstructor.discover_combinations(
        args.intermediate_dir,
        args.year_start,
        args.year_end
    )
    
    print(f"Found {len(combinations)} combinations with data in {args.year_start}-{args.year_end}")
    
    if args.max_combinations:
        combinations = combinations[:args.max_combinations]
        print(f"Limited to {len(combinations)} combinations")
    
    print()
    
    # Process each combination
    results = []
    for i, combo in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Processing {combo['combo_id']}...")
        
        try:
            # Load intermediate files
            input_df = pd.read_feather(combo['input_file'])
            input_df['ts'] = pd.to_datetime(input_df['ts'])
            input_df.set_index('ts', inplace=True)
            
            output_df = pd.read_feather(combo['output_file'])
            output_df['ts'] = pd.to_datetime(output_df['ts'])
            output_df.set_index('ts', inplace=True)
            
            # Reconstruct
            reconstructed, metrics = reconstructor.reconstruct_combination(
                input_df, output_df,
                args.year_start, args.year_end,
                verbose=args.verbose
            )
            
            # Save reconstructed data
            output_path = output_dir / f"reconstructed_{combo['combo_id']}.ftr"
            save_df = reconstructed.reset_index().rename(columns={'index': 'ts'})
            save_df.to_feather(str(output_path))
            
            # Record result
            result = {
                'combo_id': combo['combo_id'],
                'site_id': combo['site_id'],
                'thermo_id': combo['thermo_id'],
                'hygro_id': combo['hygro_id'],
                'dendro_id': combo['dendro_id'],
                'success': True,
                'output_path': str(output_path),
                **metrics
            }
            results.append(result)
            
            # Print summary
            if 'stem_corr' in metrics and 'local_T_corr' in metrics:
                rh_str = f"r={metrics['local_RH_corr']:.4f}" if 'local_RH_corr' in metrics else "N/A"
                print(f"  ✓ T: r={metrics['local_T_corr']:.4f}, RH: {rh_str}, Stem: r={metrics['stem_corr']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results.append({
                'combo_id': combo['combo_id'],
                'site_id': combo['site_id'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print()
    print("=" * 70)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 70)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        # Compute average metrics
        print("\nAverage metrics (vs LM ground truth):")
        for ch in OUTPUT_CHANNELS:
            corrs = [r[f'{ch}_corr'] for r in successful if f'{ch}_corr' in r and r[f'{ch}_corr'] is not None and not np.isnan(r[f'{ch}_corr'])]
            r2s = [r[f'{ch}_r2'] for r in successful if f'{ch}_r2' in r and r[f'{ch}_r2'] is not None and not np.isnan(r[f'{ch}_r2'])]
            maes = [r[f'{ch}_mae_norm'] for r in successful if f'{ch}_mae_norm' in r and r[f'{ch}_mae_norm'] is not None and not np.isnan(r[f'{ch}_mae_norm'])]
            
            if corrs:
                print(f"  {ch}: r={np.mean(corrs):.4f}±{np.std(corrs):.4f}, "
                      f"R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}, "
                      f"Norm MAE={np.mean(maes):.4f}±{np.std(maes):.4f}")
    
    # Save results
    results_path = output_dir / 'reconstruction_results.json'
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_path': args.model_path,
        'year_range': [args.year_start, args.year_end],
        'overlap_days': args.overlap_days,
        'successful': len(successful),
        'failed': len(failed),
        'results': results
    }
    
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
