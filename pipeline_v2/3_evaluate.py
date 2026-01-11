#!/usr/bin/env python3
"""
Unified evaluation script for TreeNet AI model.

This script combines all evaluation functionality:
1. General model evaluation (metrics on test set)
2. Real gap evaluation (comparing reconstructions to LM data)
3. Synthetic gap evaluation (injecting gaps and evaluating performance)

Modes:
- general: Overall model evaluation with metrics and optional visualizations
- real-gaps: Evaluate performance on real gaps in reconstructed data
- synthetic-gaps: Inject synthetic gaps and evaluate filling performance

Usage:
    # General evaluation
    python 3_evaluate.py --mode general --model-path <model.keras>
    
    # Real gap evaluation
    python 3_evaluate.py --mode real-gaps --recon-path <reconstructions/*.ftr>
    
    # Synthetic gap evaluation
    python 3_evaluate.py --mode synthetic-gaps --model-path <model.keras> --gap-days 7

Author: TreeNet AI Pipeline v2
Date: 2026-01-12
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pickle
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNModel
from src.gaps.metrics import GapFillingMetrics
from src.gaps.gap_injection import GapInjector
from src.utils import setup_logging, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified evaluation script for TreeNet AI model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test segments (30-day windows)
  python 3_evaluate.py --mode segments --model-path experiments/best_model.keras
  
  # Evaluate full reconstructed time series vs LM data
  python 3_evaluate.py --mode reconstruction --recon-path reconstructions/site22.ftr --lm-path site22_lm.ftr
  
  # Synthetic gap injection evaluation
  python 3_evaluate.py --mode synthetic-gaps --model-path experiments/best_model.keras --gap-days 1 7 12
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['segments', 'reconstruction', 'synthetic-gaps'],
        required=True,
        help='Evaluation mode: segments (test set), reconstruction (full timeseries vs LM), synthetic-gaps (inject and evaluate)'
    )
    
    # Model and data paths
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model (.keras file). Required for segments and synthetic-gaps modes.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data',
        help='Directory with processed segment files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for evaluation results'
    )
    
    # Reconstruction evaluation options
    parser.add_argument(
        '--recon-path',
        type=str,
        nargs='+',
        help='Path(s) to reconstruction files (.ftr) for reconstruction mode'
    )
    parser.add_argument(
        '--lm-path',
        type=str,
        nargs='+',
        help='Path(s) to LM (ground truth) files (.ftr) for reconstruction mode'
    )
    parser.add_argument(
        '--lm-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos',
        help='Directory containing LM (ground truth) files (used if --lm-path not provided)'
    )
    
    # Synthetic gap evaluation options
    parser.add_argument(
        '--gap-days',
        type=int,
        nargs='+',
        default=[7],
        help='Gap length(s) in days for synthetic gap injection (can specify multiple)'
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
        default=None,
        help='Number of test samples to use (None for all)'
    )
    
    # General options
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for predictions'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode in ['segments', 'synthetic-gaps'] and not args.model_path:
        parser.error(f"--model-path is required for {args.mode} mode")
    
    if args.mode == 'reconstruction' and not args.recon_path:
        parser.error("--recon-path is required for reconstruction mode")
    
    return args


# ==============================================================================
# SEGMENTS EVALUATION MODE (formerly 'general')
# ==============================================================================

def evaluate_segments(args):
    """
    Evaluate model on test segments (30-day windows).
    
    Computes overall metrics on pre-built test segments.
    This evaluates the model's general reconstruction accuracy with complete input.
    """
    print("="*80)
    print("TreeNet AI - Test Segments Evaluation")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = TCNModel.load(args.model_path)
    
    # Load test data
    data_dir = Path(args.data_dir)
    print(f"Loading test data from: {data_dir}")
    
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    if args.n_samples:
        X_test = X_test[:args.n_samples]
        y_test = y_test[:args.n_samples]
        print(f"Using {args.n_samples} samples")
    
    # Create masks (all ones = no gaps)
    masks = np.ones((len(X_test), X_test.shape[1], 3), dtype=np.float32)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict([X_test, masks], batch_size=args.batch_size, verbose=1)
    
    if isinstance(predictions, list):
        hourly_pred = predictions[1]  # [hourly_output]
    else:
        hourly_pred = predictions
    
    # Compute metrics
    channel_names = ['local_T', 'local_RH', 'stem']
    results = {}
    
    print("\nMetrics per channel:")
    print("-"*50)
    
    for i, ch_name in enumerate(channel_names):
        pred = hourly_pred[:, :, i].flatten()
        truth = y_test[:, :, i].flatten()
        
        valid_mask = ~(np.isnan(pred) | np.isnan(truth))
        pred = pred[valid_mask]
        truth = truth[valid_mask]
        
        mse = np.mean((pred - truth)**2)
        mae = np.mean(np.abs(pred - truth))
        corr = np.corrcoef(pred, truth)[0, 1]
        r2 = 1 - np.sum((pred - truth)**2) / np.sum((truth - np.mean(truth))**2)
        
        results[ch_name] = {
            'MSE': float(mse),
            'MAE': float(mae),
            'Correlation': float(corr),
            'R2': float(r2)
        }
        
        print(f"{ch_name:12} | MSE={mse:.6f} | MAE={mae:.6f} | R={corr:.4f} | R²={r2:.4f}")
    
    # Save results
    results_file = output_dir / 'segments_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'mode': 'segments',
            'model_path': str(args.model_path),
            'n_samples': len(X_test),
            'timestamp': datetime.now().isoformat(),
            'metrics': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


# ==============================================================================
# REAL GAP EVALUATION MODE
# ==============================================================================

def load_reconstruction_data(recon_path: str) -> pd.DataFrame:
    """Load reconstruction file."""
    return pd.read_feather(recon_path)


def load_lm_data_from_path(lm_path: str) -> Optional[pd.DataFrame]:
    """Load LM (ground truth) data from direct path."""
    path = Path(lm_path)
    if path.exists():
        return pd.read_feather(path)
    return None


def load_lm_data_from_dir(lm_dir: str, site_id: int, dendro_id: int) -> Optional[pd.DataFrame]:
    """Load LM (ground truth) data for a site/dendrometer from directory."""
    lm_path = Path(lm_dir) / f'site{site_id}_D{dendro_id}_norm.ftr'
    if lm_path.exists():
        return pd.read_feather(lm_path)
    return None


def evaluate_reconstruction(args):
    """
    Evaluate full reconstructed time series against LM ground truth.
    
    Compares reconstruction outputs to LM data, only where LM is valid (not NaN).
    This evaluates the complete reconstruction pipeline including overlapping
    window averaging.
    """
    print("="*80)
    print("TreeNet AI - Reconstruction Evaluation")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Handle LM paths
    lm_paths = args.lm_path if args.lm_path else [None] * len(args.recon_path)
    if args.lm_path and len(args.lm_path) != len(args.recon_path):
        print("Warning: Number of LM paths doesn't match reconstruction paths.")
        print("         Will try to match by index or use --lm-dir lookup.")
        lm_paths = args.lm_path + [None] * (len(args.recon_path) - len(args.lm_path))
    
    for i, recon_path in enumerate(args.recon_path):
        print(f"\n{'-'*60}")
        print(f"Processing: {recon_path}")
        
        # Load reconstruction
        recon_df = load_reconstruction_data(recon_path)
        
        # Try to load LM data
        lm_df = None
        filename = Path(recon_path).stem
        
        # Option 1: Direct LM path provided
        if i < len(lm_paths) and lm_paths[i]:
            lm_df = load_lm_data_from_path(lm_paths[i])
            if lm_df is not None:
                print(f"  LM from path: {lm_paths[i]}")
        
        # Option 2: Try to find LM in directory based on filename
        if lm_df is None:
            # Extract site and dendro ID from filename
            # Expecting format: reconstructed_site{site}_T*_H*_D{dendro}.ftr
            parts = filename.split('_')
            try:
                site_id = int([p for p in parts if p.startswith('site')][0].replace('site', ''))
                dendro_id = int([p for p in parts if p.startswith('D')][0].replace('D', ''))
                lm_df = load_lm_data_from_dir(args.lm_dir, site_id, dendro_id)
                if lm_df is not None:
                    print(f"  LM from dir: site{site_id}_D{dendro_id}_norm.ftr")
            except (IndexError, ValueError):
                pass
        
        if lm_df is None:
            print(f"  Warning: No LM data found. Skipping.")
            continue
        
        # Merge on timestamp
        merged = pd.merge(recon_df, lm_df, on='ts', suffixes=('_recon', '_lm'))
        print(f"  Merged records: {len(merged)}")
        
        # Evaluate each channel
        channels = ['local_T', 'local_RH', 'stem']
        site_results = {'reconstruction': recon_path, 'channels': {}}
        
        print(f"\n  {'Channel':15} {'N Points':>10} {'MSE':>12} {'MAE':>12} {'Corr':>8} {'R²':>8}")
        print(f"  {'-'*67}")
        
        for ch in channels:
            # Find the reconstruction and LM columns
            recon_col = f'{ch}_recon' if f'{ch}_recon' in merged.columns else ch
            lm_col = f'{ch}_lm' if f'{ch}_lm' in merged.columns else ch
            
            # Also check without suffix (if same column name didn't conflict)
            if recon_col not in merged.columns:
                recon_col = ch
            if lm_col not in merged.columns:
                lm_col = ch
            
            if recon_col not in merged.columns or lm_col not in merged.columns:
                print(f"  {ch:15} Column not found")
                continue
            
            # Only evaluate where LM has valid data
            valid_lm = ~merged[lm_col].isna()
            valid_recon = ~merged[recon_col].isna()
            eval_mask = valid_lm & valid_recon
            
            n_points = eval_mask.sum()
            
            if n_points > 10:
                pred = merged.loc[eval_mask, recon_col].values
                truth = merged.loc[eval_mask, lm_col].values
                
                mse = np.mean((pred - truth)**2)
                mae = np.mean(np.abs(pred - truth))
                corr = np.corrcoef(pred, truth)[0, 1] if len(pred) > 10 else np.nan
                r2 = 1 - np.sum((pred - truth)**2) / np.sum((truth - np.mean(truth))**2) if len(pred) > 10 else np.nan
                
                site_results['channels'][ch] = {
                    'n_points': int(n_points),
                    'MSE': float(mse),
                    'MAE': float(mae),
                    'Correlation': float(corr) if not np.isnan(corr) else None,
                    'R2': float(r2) if not np.isnan(r2) else None
                }
                
                print(f"  {ch:15} {n_points:>10,} {mse:>12.6f} {mae:>12.6f} {corr:>8.4f} {r2:>8.4f}")
            else:
                print(f"  {ch:15} Insufficient valid points ({n_points})")
        
        all_results.append(site_results)
    
    # Save results
    results_file = output_dir / 'reconstruction_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'mode': 'reconstruction',
            'timestamp': datetime.now().isoformat(),
            'reconstructions': all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    
    return all_results


# ==============================================================================
# SYNTHETIC GAP EVALUATION MODE
# ==============================================================================

def create_gap_mask_for_output(input_mask: np.ndarray, 
                               steps_per_hour: int = 6) -> np.ndarray:
    """Convert input mask (10-min) to output mask (1-hour)."""
    n_samples, timesteps, n_channels = input_mask.shape
    n_hours = timesteps // steps_per_hour
    
    output_mask = np.zeros((n_samples, n_hours, 3), dtype=bool)
    
    for h in range(n_hours):
        start_idx = h * steps_per_hour
        end_idx = start_idx + steps_per_hour
        
        for c in range(3):
            hour_mask = input_mask[:, start_idx:end_idx, c]
            output_mask[:, h, c] = np.any(hour_mask == 0, axis=1)
    
    return output_mask


def evaluate_synthetic_gaps(args):
    """
    Evaluate model with synthetic gap injection.
    
    This is the most comprehensive gap-filling evaluation:
    1. Take clean test segments
    2. Inject synthetic gaps
    3. Predict and compare to known ground truth
    """
    print("="*80)
    print("TreeNet AI - Synthetic Gap Evaluation")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = TCNModel.load(args.model_path)
    
    # Load test data
    data_dir = Path(args.data_dir)
    print(f"Loading test data from: {data_dir}")
    
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    if args.n_samples:
        X_test = X_test[:args.n_samples]
        y_test = y_test[:args.n_samples]
        print(f"Using {args.n_samples} samples")
    
    channel_names = ['local_T', 'local_RH', 'stem']
    all_results = {}
    
    # Evaluate each gap length
    for gap_days in args.gap_days:
        print(f"\n{'='*60}")
        print(f"Evaluating {gap_days}-day gaps...")
        print(f"{'='*60}")
        
        # Create gap injector
        gap_injector = GapInjector(
            min_gap_days=gap_days,
            max_gap_days=gap_days,
            min_gaps_per_segment=args.n_gaps,
            max_gaps_per_segment=args.n_gaps,
            gap_channel_prob=1.0,
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
        
        # Generate predictions
        print("Generating predictions...")
        all_predictions = []
        
        for i in range(0, len(X_gapped), args.batch_size):
            end_idx = min(i + args.batch_size, len(X_gapped))
            batch_x = X_gapped[i:end_idx]
            batch_mask = masks[i:end_idx]
            
            batch_pred = model.predict([batch_x, batch_mask], verbose=0)
            
            if isinstance(batch_pred, list):
                all_predictions.append(batch_pred[1])
            else:
                all_predictions.append(batch_pred)
            
            if args.verbose:
                print(f"  Processed {end_idx}/{len(X_gapped)} segments", end='\r')
        
        hourly_pred = np.concatenate(all_predictions, axis=0)
        print(f"\nPrediction shape: {hourly_pred.shape}")
        
        # Create output mask
        gap_mask = create_gap_mask_for_output(masks)
        non_gap_mask = ~gap_mask
        
        # Compute metrics
        results = {
            'gap_days': gap_days,
            'n_samples': len(X_test),
            'n_gaps_per_sample': args.n_gaps,
            'channels': {}
        }
        
        print(f"\n{gap_days}-day gap results:")
        print("-"*60)
        
        for i, ch_name in enumerate(channel_names):
            p = hourly_pred[:, :, i].flatten()
            t = y_test[:, :, i].flatten()
            g = gap_mask[:, :, i].flatten()
            ng = non_gap_mask[:, :, i].flatten()
            
            # Gap metrics
            gap_pred = p[g]
            gap_truth = t[g]
            valid = ~(np.isnan(gap_pred) | np.isnan(gap_truth))
            gap_pred = gap_pred[valid]
            gap_truth = gap_truth[valid]
            
            gap_mse = np.mean((gap_pred - gap_truth)**2)
            gap_mae = np.mean(np.abs(gap_pred - gap_truth))
            gap_corr = np.corrcoef(gap_pred, gap_truth)[0, 1] if len(gap_pred) > 10 else np.nan
            
            # Non-gap metrics
            ng_pred = p[ng]
            ng_truth = t[ng]
            valid = ~(np.isnan(ng_pred) | np.isnan(ng_truth))
            ng_pred = ng_pred[valid]
            ng_truth = ng_truth[valid]
            
            ng_mse = np.mean((ng_pred - ng_truth)**2)
            ng_mae = np.mean(np.abs(ng_pred - ng_truth))
            ng_corr = np.corrcoef(ng_pred, ng_truth)[0, 1] if len(ng_pred) > 10 else np.nan
            
            results['channels'][ch_name] = {
                'gap': {
                    'n_points': len(gap_pred),
                    'MSE': float(gap_mse),
                    'MAE': float(gap_mae),
                    'Correlation': float(gap_corr) if not np.isnan(gap_corr) else None
                },
                'non_gap': {
                    'n_points': len(ng_pred),
                    'MSE': float(ng_mse),
                    'MAE': float(ng_mae),
                    'Correlation': float(ng_corr) if not np.isnan(ng_corr) else None
                }
            }
            
            print(f"{ch_name:12} | Gap: MSE={gap_mse:.6f}, R={gap_corr:.4f} | Non-gap: MSE={ng_mse:.6f}, R={ng_corr:.4f}")
        
        all_results[gap_days] = results
    
    # Save results
    results_file = output_dir / 'synthetic_gap_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'mode': 'synthetic-gaps',
            'model_path': str(args.model_path),
            'timestamp': datetime.now().isoformat(),
            'gap_lengths': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    np.random.seed(args.random_seed)
    
    if args.mode == 'segments':
        return evaluate_segments(args)
    elif args.mode == 'reconstruction':
        return evaluate_reconstruction(args)
    elif args.mode == 'synthetic-gaps':
        return evaluate_synthetic_gaps(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
