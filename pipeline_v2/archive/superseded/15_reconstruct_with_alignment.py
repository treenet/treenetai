#!/usr/bin/env python3
"""
Reconstruct time series from intermediate timeseries files with scale alignment.

Key improvements:
1. Starts reconstruction in Nov/Dec of the prior year for proper initialization
2. Aligns stem channel to LM scale using the overlap period
3. Validates metrics only on the requested year range

The scale alignment strategy:
- Reconstruct starting Nov 1 of (year_start - 1)
- Use Nov-Dec data to compute the offset between reconstructed and LM stem values
- Apply this offset to align the entire reconstructed stem channel

Usage:
    python 15_reconstruct_with_alignment.py \
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
    'temp_treenet', 'rh_treenet', 'stem',
    'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy'
]
OUTPUT_CHANNELS = ['local_T', 'local_RH', 'stem']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reconstruct time series with scale alignment'
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
        help='Start year for reconstruction (will start from Nov of prior year)'
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


class AlignedReconstructor:
    """Reconstruct time series with scale alignment."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        overlap_days: int = 5,
        max_gap_days: int = 12
    ):
        self.model = model
        self.overlap_days = overlap_days
        self.max_gap_days = max_gap_days
        self.stride_hours = (30 - overlap_days) * 24
        
    def discover_combinations(
        self,
        intermediate_dir: str,
        year_start: int,
        year_end: int
    ) -> List[Dict]:
        """Discover test combinations with data including prior year Nov-Dec."""
        test_inputs = glob(os.path.join(intermediate_dir, 'test_input_*.ftr'))
        
        # Need data starting from Nov of (year_start - 1)
        warmup_year = year_start - 1
        
        combinations = []
        for input_file in test_inputs:
            basename = os.path.basename(input_file)
            combo_str = basename.replace('test_input_', '').replace('.ftr', '')
            
            parts = combo_str.split('_')
            site_id = int(parts[0].replace('site', ''))
            thermo_id = int(parts[1].replace('T', ''))
            hygro_id = int(parts[2].replace('H', ''))
            dendro_id = int(parts[3].replace('D', ''))
            
            # Check date range
            df = pd.read_feather(input_file)
            df['ts'] = pd.to_datetime(df['ts'])
            
            # Need data from Nov of warmup_year through year_end
            warmup_start = pd.Timestamp(f'{warmup_year}-11-01', tz='UTC')
            target_end = pd.Timestamp(f'{year_end}-12-31 23:59:59', tz='UTC')
            
            mask = (df['ts'] >= warmup_start) & (df['ts'] <= target_end)
            samples_in_range = mask.sum()
            
            # Also check for warmup data specifically (Nov-Dec of prior year)
            warmup_mask = (df['ts'].dt.year == warmup_year) & (df['ts'].dt.month >= 11)
            warmup_samples = warmup_mask.sum()
            
            if samples_in_range > INPUT_SAMPLES and warmup_samples > 0:
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
                        'samples_in_range': samples_in_range,
                        'warmup_samples': warmup_samples
                    })
        
        return combinations
    
    def create_segment_grid(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Create overlapping segment grid."""
        segments = []
        current_start = start_time.floor('10min')
        
        while current_start + pd.Timedelta(days=SEGMENT_DAYS) <= end_time:
            seg_end = current_start + pd.Timedelta(days=SEGMENT_DAYS)
            segments.append((current_start, seg_end))
            current_start = current_start + pd.Timedelta(hours=self.stride_hours)
        
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
        """Prepare a segment for model inference."""
        complete_idx = pd.date_range(
            start=seg_start, periods=INPUT_SAMPLES, freq='10min', tz='UTC'
        )
        
        segment = input_df.loc[
            (input_df.index >= seg_start) & (input_df.index < seg_end)
        ].copy()
        
        segment_full = segment.reindex(complete_idx)
        mask = (~segment_full.isna()).astype(np.float32)
        
        coverage = mask.values.mean()
        if coverage < 0.5:
            return None, None, None, False
        
        segment_filled = segment_full.ffill().bfill()
        
        values = segment_filled.values
        norm_params = {}
        values_norm = np.zeros_like(values)
        
        for i, col in enumerate(segment_filled.columns):
            col_data = values[:, i]
            min_val = float(np.nanmin(col_data))
            max_val = float(np.nanmax(col_data))
            norm_params[col] = {'min': min_val, 'max': max_val}
            
            if max_val > min_val:
                values_norm[:, i] = (col_data - min_val) / (max_val - min_val)
            else:
                values_norm[:, i] = 0.5
        
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
        channel_map = {0: 'temp_treenet', 1: 'rh_treenet', 2: 'stem'}
        
        for i, input_ch in channel_map.items():
            if input_ch in norm_params:
                min_val = norm_params[input_ch]['min']
                max_val = norm_params[input_ch]['max']
                
                if max_val > min_val:
                    output_denorm[:, i] = output[:, i] * (max_val - min_val) + min_val
                else:
                    output_denorm[:, i] = min_val
        
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
        Reconstruct with scale alignment.
        
        Strategy:
        1. Reconstruct from Nov 1 of (year_start - 1) through year_end
        2. Use Nov-Dec overlap to compute stem offset
        3. Apply offset to align reconstructed stem to LM scale
        4. Return only the requested year range
        """
        warmup_year = year_start - 1
        
        # Filter input to include warmup period
        warmup_start = pd.Timestamp(f'{warmup_year}-11-01', tz='UTC')
        target_end = pd.Timestamp(f'{year_end}-12-31 23:59:59', tz='UTC')
        
        mask_in = (input_df.index >= warmup_start) & (input_df.index <= target_end)
        input_extended = input_df[mask_in].copy()
        
        if len(input_extended) < INPUT_SAMPLES:
            raise ValueError(f"Insufficient data for warmup period")
        
        # Create segment grid for extended period
        start_time = input_extended.index.min()
        end_time = input_extended.index.max()
        segments = self.create_segment_grid(start_time, end_time)
        
        if verbose:
            print(f"  Extended period: {start_time.date()} to {end_time.date()}")
            print(f"  Segment grid: {len(segments)} segments")
        
        # Process segments
        segment_outputs = []
        segments_processed = 0
        segments_skipped = 0
        
        for i, (seg_start, seg_end) in enumerate(segments):
            input_arr, mask_arr, norm_params, is_valid = self.prepare_segment(
                input_extended, seg_start, seg_end
            )
            
            if not is_valid:
                segments_skipped += 1
                continue
            
            pred = self.model.predict([input_arr, mask_arr], verbose=0)
            pred_output = pred[1] if isinstance(pred, list) else pred
            pred_squeezed = pred_output[0]
            
            pred_denorm = self.denormalize_output(pred_squeezed, norm_params)
            
            output_start = seg_start.ceil('1h')
            output_times = pd.date_range(
                start=output_start, periods=OUTPUT_SAMPLES, freq='1h', tz='UTC'
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
        
        # Merge segments
        merged_df = self._merge_segments(segment_outputs)
        
        # Compute stem scale and offset alignment using warmup period
        warmup_mask = (merged_df.index >= warmup_start) & \
                      (merged_df.index < pd.Timestamp(f'{year_start}-01-01', tz='UTC'))
        
        output_warmup_mask = (output_df.index >= warmup_start) & \
                             (output_df.index < pd.Timestamp(f'{year_start}-01-01', tz='UTC'))
        
        alignment_info = {'method': 'none'}
        
        if warmup_mask.any() and output_warmup_mask.any():
            # Get common timestamps in warmup period
            warmup_recon = merged_df[warmup_mask]
            warmup_gt = output_df[output_warmup_mask]
            common_warmup = warmup_recon.index.intersection(warmup_gt.index)
            
            if len(common_warmup) > 100:
                recon_stem = warmup_recon.loc[common_warmup, 'stem'].values
                gt_stem = warmup_gt.loc[common_warmup, 'stem'].values
                
                valid = ~np.isnan(recon_stem) & ~np.isnan(gt_stem)
                n_valid = int(valid.sum())
                
                if n_valid > 50:
                    recon_valid = recon_stem[valid]
                    gt_valid = gt_stem[valid]
                    
                    # Compute scale factor and offset using linear regression
                    # gt = scale * recon + offset
                    recon_mean = np.mean(recon_valid)
                    gt_mean = np.mean(gt_valid)
                    
                    # Check if there's meaningful variation for scale estimation
                    recon_std = np.std(recon_valid)
                    gt_std = np.std(gt_valid)
                    
                    if recon_std > 0.01 and gt_std > 0.01:
                        # Use linear regression for scale + offset
                        # scale = cov(gt, recon) / var(recon)
                        cov_xy = np.mean((recon_valid - recon_mean) * (gt_valid - gt_mean))
                        var_x = np.var(recon_valid)
                        
                        scale = float(cov_xy / var_x) if var_x > 0 else 1.0
                        offset = float(gt_mean - scale * recon_mean)
                        
                        # Apply scale and offset
                        merged_df['stem'] = merged_df['stem'] * scale + offset
                        alignment_info = {
                            'method': 'scale_offset',
                            'scale': scale,
                            'offset': offset,
                            'n_samples': n_valid
                        }
                        
                        if verbose:
                            print(f"  Stem aligned: scale={scale:.4f}, offset={offset:.2f} "
                                  f"(from {n_valid} samples)")
                    else:
                        # Only offset alignment if variation is too low
                        offset = float(gt_mean - recon_mean)
                        merged_df['stem'] = merged_df['stem'] + offset
                        alignment_info = {
                            'method': 'offset_only',
                            'offset': offset,
                            'n_samples': n_valid
                        }
                        
                        if verbose:
                            print(f"  Stem offset: {offset:.2f} (from {n_valid} samples)")
        
        # Filter to requested year range only
        year_mask = (merged_df.index.year >= year_start) & (merged_df.index.year <= year_end)
        result_df = merged_df[year_mask].copy()
        
        # Compute metrics on requested range
        gt_mask = (output_df.index.year >= year_start) & (output_df.index.year <= year_end)
        gt_filtered = output_df[gt_mask]
        metrics = self._compute_metrics(result_df, gt_filtered)
        metrics['segments_processed'] = segments_processed
        metrics['segments_skipped'] = segments_skipped
        
        return result_df, metrics
    
    def _merge_segments(
        self,
        segment_outputs: List[Dict]
    ) -> pd.DataFrame:
        """Merge overlapping segments using weighted averaging."""
        all_times = set()
        for seg in segment_outputs:
            all_times.update(seg['times'])
        
        all_times = sorted(all_times)
        n_times = len(all_times)
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        
        value_sum = np.zeros((n_times, 3))
        weight_sum = np.zeros((n_times, 3))
        
        for seg in segment_outputs:
            times = seg['times']
            values = seg['values']
            
            n_samples = len(times)
            weights = np.ones(n_samples)
            
            overlap_samples = self.overlap_days * 24
            if n_samples > 2 * overlap_samples:
                for j in range(overlap_samples):
                    w = (j + 1) / (overlap_samples + 1)
                    weights[j] = w
                    weights[-(j+1)] = w
            
            for j, t in enumerate(times):
                idx = time_to_idx[t]
                for ch in range(3):
                    val = values[j, ch]
                    if not np.isnan(val):
                        value_sum[idx, ch] += val * weights[j]
                        weight_sum[idx, ch] += weights[j]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            merged_values = np.where(
                weight_sum > 0,
                value_sum / weight_sum,
                np.nan
            )
        
        return pd.DataFrame(
            merged_values,
            index=pd.DatetimeIndex(all_times),
            columns=OUTPUT_CHANNELS
        )
    
    def _compute_metrics(
        self,
        reconstructed: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Dict:
        """Compute metrics comparing reconstruction to LM ground truth."""
        metrics = {}
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
            
            valid_mask = ~np.isnan(gt_vals) & ~np.isnan(recon_vals)
            n_valid = int(valid_mask.sum())
            
            if n_valid < 10:
                continue
            
            recon_valid = recon_vals[valid_mask]
            gt_valid = gt_vals[valid_mask]
            
            mae = float(np.mean(np.abs(recon_valid - gt_valid)))
            rmse = float(np.sqrt(np.mean((recon_valid - gt_valid)**2)))
            
            if np.std(gt_valid) > 0 and np.std(recon_valid) > 0:
                corr = float(np.corrcoef(recon_valid, gt_valid)[0, 1])
            else:
                corr = float('nan')
            
            ss_res = np.sum((gt_valid - recon_valid)**2)
            ss_tot = np.sum((gt_valid - np.mean(gt_valid))**2)
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float('nan')
            
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
    print("TreeNet AI - Reconstruction with Scale Alignment")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Years: {args.year_start} - {args.year_end}")
    print(f"  (Warmup from Nov {args.year_start - 1})")
    print(f"Overlap: {args.overlap_days} days")
    print(f"Output: {args.output_dir}")
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'TCNBlock': TCNBlock,
            'PositionalEncoding': PositionalEncoding
        }
    )
    print("Model loaded successfully\n")
    
    reconstructor = AlignedReconstructor(
        model=model,
        overlap_days=args.overlap_days
    )
    
    print("Discovering combinations...")
    combinations = reconstructor.discover_combinations(
        args.intermediate_dir,
        args.year_start,
        args.year_end
    )
    
    print(f"Found {len(combinations)} combinations with warmup data")
    
    if args.max_combinations:
        combinations = combinations[:args.max_combinations]
        print(f"Limited to {len(combinations)} combinations")
    
    print()
    
    results = []
    for i, combo in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Processing {combo['combo_id']}...")
        
        try:
            input_df = pd.read_feather(combo['input_file'])
            input_df['ts'] = pd.to_datetime(input_df['ts'])
            input_df.set_index('ts', inplace=True)
            
            output_df = pd.read_feather(combo['output_file'])
            output_df['ts'] = pd.to_datetime(output_df['ts'])
            output_df.set_index('ts', inplace=True)
            
            reconstructed, metrics = reconstructor.reconstruct_combination(
                input_df, output_df,
                args.year_start, args.year_end,
                verbose=args.verbose
            )
            
            output_path = output_dir / f"aligned_{combo['combo_id']}.ftr"
            save_df = reconstructed.reset_index().rename(columns={'index': 'ts'})
            save_df.to_feather(str(output_path))
            
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
            
            if 'stem_corr' in metrics and 'local_T_corr' in metrics:
                rh_str = f"r={metrics['local_RH_corr']:.4f}" if 'local_RH_corr' in metrics else "N/A"
                stem_r2 = f"R²={metrics['stem_r2']:.4f}" if 'stem_r2' in metrics else ""
                print(f"  ✓ T: r={metrics['local_T_corr']:.4f}, RH: {rh_str}, "
                      f"Stem: r={metrics['stem_corr']:.4f} {stem_r2}")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            if args.verbose:
                traceback.print_exc()
            results.append({
                'combo_id': combo['combo_id'],
                'site_id': combo['site_id'],
                'success': False,
                'error': str(e)
            })
    
    print()
    print("=" * 70)
    print("RECONSTRUCTION WITH ALIGNMENT COMPLETE")
    print("=" * 70)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nAverage metrics (vs LM ground truth):")
        for ch in OUTPUT_CHANNELS:
            corrs = [r[f'{ch}_corr'] for r in successful 
                     if f'{ch}_corr' in r and r[f'{ch}_corr'] is not None 
                     and not np.isnan(r[f'{ch}_corr'])]
            r2s = [r[f'{ch}_r2'] for r in successful 
                   if f'{ch}_r2' in r and r[f'{ch}_r2'] is not None 
                   and not np.isnan(r[f'{ch}_r2'])]
            maes = [r[f'{ch}_mae_norm'] for r in successful 
                    if f'{ch}_mae_norm' in r and r[f'{ch}_mae_norm'] is not None 
                    and not np.isnan(r[f'{ch}_mae_norm'])]
            
            if corrs:
                print(f"  {ch}: r={np.mean(corrs):.4f}±{np.std(corrs):.4f}, "
                      f"R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}, "
                      f"Norm MAE={np.mean(maes):.4f}±{np.std(maes):.4f}")
    
    results_path = output_dir / 'aligned_results.json'
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_path': args.model_path,
        'year_range': [args.year_start, args.year_end],
        'warmup_year': args.year_start - 1,
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
