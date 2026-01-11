#!/usr/bin/env python3
"""
Comprehensive gap-filling evaluation using synthetic gap injection.

This script:
1. Loads test input/output segment pairs (30-day windows, no gaps)
2. Injects synthetic gaps into the input data
3. Passes gapped input through the model
4. Evaluates prediction quality:
   a) Across the ENTIRE 30-day segment (all timesteps)
   b) ONLY in the gap regions (where model is truly filling gaps)

This is the proper way to evaluate gap-filling capability because:
- We have paired input/output without gaps
- We control exactly where gaps are
- We can compare model predictions to known ground truth

Usage:
    python 9_evaluate_synthetic_gaps.py \
        --model-path <path_to_model.keras> \
        --gap-days 7 \
        --n-samples 100

Author: TreeNet AI Pipeline v2
Date: 2026-01-12
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNModel
from src.gaps.gap_injection import GapInjector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate gap-filling with synthetic gaps on test set'
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
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data',
        help='Directory containing test segment pickle files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lukovic/data/treenet/synthetic_gap_evaluation/',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--gap-days',
        type=int,
        default=7,
        help='Gap length in days for synthetic gaps'
    )
    parser.add_argument(
        '--n-gaps',
        type=int,
        default=2,
        help='Number of gaps to inject per segment'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=-1,
        help='Number of test samples to evaluate (-1 for all)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def compute_metrics(pred: np.ndarray, truth: np.ndarray, 
                   mask: np.ndarray = None) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        pred: Predicted values (N, T, C)
        truth: Ground truth values (N, T, C)
        mask: Optional boolean mask (N, T, C), True = evaluate this point
    
    Returns:
        Dictionary with per-channel metrics
    """
    n_samples, timesteps, n_channels = pred.shape
    
    results = {}
    target_channels = ['local_T', 'local_RH', 'stem']
    
    for i, ch_name in enumerate(target_channels):
        if i >= n_channels:
            break
        
        p = pred[:, :, i].flatten()
        t = truth[:, :, i].flatten()
        
        if mask is not None:
            m = mask[:, :, i].flatten().astype(bool)
            p = p[m]
            t = t[m]
        
        # Remove NaN
        valid = ~(np.isnan(p) | np.isnan(t))
        p = p[valid]
        t = t[valid]
        
        if len(p) == 0:
            results[ch_name] = {
                'mae': np.nan,
                'rmse': np.nan,
                'mse': np.nan,
                'corr': np.nan,
                'r2': np.nan,
                'n_samples': 0
            }
            continue
        
        # MAE
        mae = np.mean(np.abs(p - t))
        
        # MSE and RMSE
        mse = np.mean((p - t) ** 2)
        rmse = np.sqrt(mse)
        
        # Correlation
        if np.std(p) > 1e-10 and np.std(t) > 1e-10:
            corr = np.corrcoef(p, t)[0, 1]
        else:
            corr = np.nan
        
        # R²
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else np.nan
        
        results[ch_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mse': float(mse),
            'corr': float(corr) if not np.isnan(corr) else None,
            'r2': float(r2) if not np.isnan(r2) else None,
            'n_samples': int(len(p))
        }
    
    return results


def create_gap_mask_for_output(input_mask: np.ndarray, 
                               steps_per_hour: int = 6) -> np.ndarray:
    """
    Convert input mask (10-min resolution) to output mask (1-hour resolution).
    
    The output is hourly, so we need to aggregate:
    - If ANY timestep in an hour is gapped, mark that hour as gapped
    
    Args:
        input_mask: Shape (N, T_input, C_input), 1=valid, 0=gap
        steps_per_hour: Timesteps per hour (6 for 10-min data)
    
    Returns:
        output_mask: Shape (N, T_output, C_output), True=gap (to evaluate)
    """
    n_samples, timesteps, n_channels = input_mask.shape
    n_hours = timesteps // steps_per_hour
    
    # Output has 3 channels: local_T, local_RH, stem
    # Corresponding to input channels 0, 1, 2
    output_mask = np.zeros((n_samples, n_hours, 3), dtype=bool)
    
    for h in range(n_hours):
        start_idx = h * steps_per_hour
        end_idx = start_idx + steps_per_hour
        
        for c in range(3):  # Only first 3 channels
            # If any timestep in this hour is gapped (mask=0), mark as gap
            hour_mask = input_mask[:, start_idx:end_idx, c]
            # Gap if any value in the hour is 0
            output_mask[:, h, c] = np.any(hour_mask == 0, axis=1)
    
    return output_mask


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("="*80)
    print("TreeNet AI - Synthetic Gap Injection Evaluation")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Gap length: {args.gap_days} days")
    print(f"Gaps per segment: {args.n_gaps}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = TCNModel.load(args.model_path)
    print("Model loaded successfully")
    
    # Load test data
    data_dir = Path(args.data_dir)
    print(f"\nLoading test data from: {data_dir}")
    
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    # Limit samples if requested
    if args.n_samples > 0 and args.n_samples < len(X_test):
        np.random.seed(args.random_seed)
        indices = np.random.choice(len(X_test), size=args.n_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"Evaluating on {args.n_samples} random samples")
    
    # Create gap injector
    print(f"\nInjecting synthetic gaps ({args.gap_days}-day gaps, {args.n_gaps} per segment)...")
    gap_injector = GapInjector(
        min_gap_days=args.gap_days,
        max_gap_days=args.gap_days,
        min_gaps_per_segment=args.n_gaps,
        max_gaps_per_segment=args.n_gaps,
        gap_channel_prob=1.0,  # Gap all 3 local channels
        random_seed=args.random_seed
    )
    
    # Inject gaps
    X_gapped = []
    masks = []
    for i in range(len(X_test)):
        x_gapped, mask = gap_injector.inject_gaps(X_test[i])
        X_gapped.append(x_gapped)
        masks.append(mask)
    
    X_gapped = np.array(X_gapped)
    masks = np.array(masks)
    
    print(f"Gapped data shape: {X_gapped.shape}")
    print(f"Mask shape: {masks.shape}")
    
    # Count gap statistics
    total_timesteps = X_gapped.shape[0] * X_gapped.shape[1]
    gap_timesteps = np.sum(masks[:, :, :3] == 0)
    gap_pct = 100 * gap_timesteps / (total_timesteps * 3)
    print(f"Gap percentage (local channels): {gap_pct:.2f}%")
    
    # Generate predictions in batches to avoid OOM
    print("\nGenerating predictions...")
    batch_size = 16  # Predict in small batches to avoid OOM
    all_predictions = []
    
    for i in range(0, len(X_gapped), batch_size):
        end_idx = min(i + batch_size, len(X_gapped))
        batch_x = X_gapped[i:end_idx]
        batch_mask = masks[i:end_idx]
        
        batch_pred = model.predict([batch_x, batch_mask], verbose=0)
        
        # Handle different model output formats
        if isinstance(batch_pred, list):
            all_predictions.append(batch_pred[1])  # Hourly prediction output
        else:
            all_predictions.append(batch_pred)
        
        print(f"  Processed {end_idx}/{len(X_gapped)} segments", end='\r')
    
    hourly_pred = np.concatenate(all_predictions, axis=0)
    print(f"\nPrediction shape: {hourly_pred.shape}")
    print(f"Target shape: {y_test.shape}")
    
    # Create output mask for gap regions
    gap_mask = create_gap_mask_for_output(masks)
    non_gap_mask = ~gap_mask
    
    gap_hours = np.sum(gap_mask)
    total_hours = gap_mask.size
    print(f"\nGap hours in output: {gap_hours} ({100*gap_hours/total_hours:.2f}%)")
    print(f"Non-gap hours: {total_hours - gap_hours}")
    
    # Evaluate: ENTIRE 30-day segment
    print("\n" + "="*80)
    print("EVALUATION RESULTS - ENTIRE 30-DAY SEGMENTS")
    print("="*80)
    
    all_metrics = compute_metrics(hourly_pred, y_test)
    
    print(f"\n{'Channel':<12} {'MAE':>10} {'RMSE':>10} {'MSE':>12} {'Corr':>10} {'R²':>10} {'N':>10}")
    print("-"*80)
    for ch, m in all_metrics.items():
        mae = f"{m['mae']:.6f}" if m['mae'] is not None else 'N/A'
        rmse = f"{m['rmse']:.6f}" if m['rmse'] is not None else 'N/A'
        mse = f"{m['mse']:.8f}" if m['mse'] is not None else 'N/A'
        corr = f"{m['corr']:.6f}" if m['corr'] is not None else 'N/A'
        r2 = f"{m['r2']:.6f}" if m['r2'] is not None else 'N/A'
        n = m['n_samples']
        print(f"{ch:<12} {mae:>10} {rmse:>10} {mse:>12} {corr:>10} {r2:>10} {n:>10}")
    
    # Evaluate: GAP REGIONS ONLY
    print("\n" + "="*80)
    print("EVALUATION RESULTS - GAP REGIONS ONLY")
    print("="*80)
    
    gap_metrics = compute_metrics(hourly_pred, y_test, gap_mask)
    
    print(f"\n{'Channel':<12} {'MAE':>10} {'RMSE':>10} {'MSE':>12} {'Corr':>10} {'R²':>10} {'N':>10}")
    print("-"*80)
    for ch, m in gap_metrics.items():
        mae = f"{m['mae']:.6f}" if m['mae'] is not None else 'N/A'
        rmse = f"{m['rmse']:.6f}" if m['rmse'] is not None else 'N/A'
        mse = f"{m['mse']:.8f}" if m['mse'] is not None else 'N/A'
        corr = f"{m['corr']:.6f}" if m['corr'] is not None else 'N/A'
        r2 = f"{m['r2']:.6f}" if m['r2'] is not None else 'N/A'
        n = m['n_samples']
        print(f"{ch:<12} {mae:>10} {rmse:>10} {mse:>12} {corr:>10} {r2:>10} {n:>10}")
    
    # Evaluate: NON-GAP REGIONS (for comparison)
    print("\n" + "="*80)
    print("EVALUATION RESULTS - NON-GAP REGIONS (for comparison)")
    print("="*80)
    
    non_gap_metrics = compute_metrics(hourly_pred, y_test, non_gap_mask)
    
    print(f"\n{'Channel':<12} {'MAE':>10} {'RMSE':>10} {'MSE':>12} {'Corr':>10} {'R²':>10} {'N':>10}")
    print("-"*80)
    for ch, m in non_gap_metrics.items():
        mae = f"{m['mae']:.6f}" if m['mae'] is not None else 'N/A'
        rmse = f"{m['rmse']:.6f}" if m['rmse'] is not None else 'N/A'
        mse = f"{m['mse']:.8f}" if m['mse'] is not None else 'N/A'
        corr = f"{m['corr']:.6f}" if m['corr'] is not None else 'N/A'
        r2 = f"{m['r2']:.6f}" if m['r2'] is not None else 'N/A'
        n = m['n_samples']
        print(f"{ch:<12} {mae:>10} {rmse:>10} {mse:>12} {corr:>10} {r2:>10} {n:>10}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: Gap vs Non-Gap Performance")
    print("="*80)
    print(f"\n{'Channel':<12} {'Gap MAE':>10} {'Non-Gap MAE':>12} {'Gap Corr':>10} {'Non-Gap Corr':>12}")
    print("-"*60)
    for ch in ['local_T', 'local_RH', 'stem']:
        gap_mae = f"{gap_metrics[ch]['mae']:.4f}" if gap_metrics[ch]['mae'] is not None else 'N/A'
        nongap_mae = f"{non_gap_metrics[ch]['mae']:.4f}" if non_gap_metrics[ch]['mae'] is not None else 'N/A'
        gap_corr = f"{gap_metrics[ch]['corr']:.4f}" if gap_metrics[ch]['corr'] is not None else 'N/A'
        nongap_corr = f"{non_gap_metrics[ch]['corr']:.4f}" if non_gap_metrics[ch]['corr'] is not None else 'N/A'
        print(f"{ch:<12} {gap_mae:>10} {nongap_mae:>12} {gap_corr:>10} {nongap_corr:>12}")
    
    # Save results
    results = {
        'config': {
            'model_path': str(args.model_path),
            'data_dir': str(args.data_dir),
            'gap_days': args.gap_days,
            'n_gaps': args.n_gaps,
            'n_samples': len(X_test),
            'random_seed': args.random_seed
        },
        'statistics': {
            'total_segments': len(X_test),
            'gap_percentage': float(gap_pct),
            'gap_hours': int(gap_hours),
            'total_hours': int(total_hours)
        },
        'all_segments': all_metrics,
        'gap_regions': gap_metrics,
        'non_gap_regions': non_gap_metrics
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'synthetic_gap_eval_{args.gap_days}d_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
