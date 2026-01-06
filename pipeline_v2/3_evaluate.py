#!/usr/bin/env python3
"""
Evaluate trained TCN model.

This script:
1. Loads trained model
2. Evaluates on test set
3. Generates predictions
4. Computes metrics per channel
5. (Optional) Creates visualizations

Usage:
    python 3_evaluate.py --model-path experiments/*/best_model.keras
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pickle
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNModel
from src.gaps.metrics import GapFillingMetrics
from src.gaps.gap_injection import GapInjector
from src.utils import setup_logging, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate TCN model')
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.keras file)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/model_data',
        help='Directory with processed segment files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--gap-days',
        type=int,
        default=12,
        help='Gap length for evaluation (days)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of test samples to evaluate (-1 for all)'
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
    
    print("="*80)
    print("TreeNet AI Pipeline v2 - Model Evaluation")
    print("="*80)
    
    # Create output directory
    output_dir = ensure_dir(Path(args.output_dir))
    setup_logging(verbose=args.verbose, log_file=output_dir / 'evaluation.log')
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = TCNModel.load(args.model_path)
    print("Model loaded successfully")
    
    # Load test data
    print(f"\nLoading test data from: {args.data_dir}")
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    # Limit samples if requested
    if args.n_samples > 0 and args.n_samples < len(X_test):
        indices = np.random.choice(len(X_test), size=args.n_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"Evaluating on {args.n_samples} random samples")
    
    # Create gap injector for evaluation
    print(f"\nInjecting {args.gap_days}-day gaps for evaluation...")
    gap_injector = GapInjector(
        min_gap_days=args.gap_days,
        max_gap_days=args.gap_days,
        min_gaps_per_segment=1,
        max_gaps_per_segment=1,
        random_seed=42
    )
    
    X_gapped, masks = gap_injector.inject_gaps_batch(X_test)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict([X_gapped, masks], verbose=1 if args.verbose else 0)
    
    recon_pred = predictions[0]  # Reconstruction
    hourly_pred = predictions[1]  # Hourly prediction
    
    # Compute metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Input channel names
    input_channels = [
        'temp_treenet', 'rh_treenet', 'stem',
        'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy'
    ]
    
    target_channels = ['local_T', 'local_RH', 'stem']
    
    # Metrics for reconstruction (gapped regions only)
    print("\n1. Reconstruction Metrics (gapped regions):")
    print("-" * 80)
    
    recon_metrics = {}
    for i, ch_name in enumerate(input_channels):
        if i >= X_test.shape[-1]:
            break
        
        gap_mask = (masks[..., i] == 0)
        if not np.any(gap_mask):
            continue
        
        mae = GapFillingMetrics.mae(X_test[..., i], recon_pred[..., i], masks[..., i])
        rmse = GapFillingMetrics.rmse(X_test[..., i], recon_pred[..., i], masks[..., i])
        r2 = GapFillingMetrics.r2_score(X_test[..., i], recon_pred[..., i], masks[..., i])
        
        recon_metrics[ch_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"  {ch_name:15s}: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}")
    
    # Metrics for hourly prediction
    print("\n2. Hourly Prediction Metrics:")
    print("-" * 80)
    
    hourly_metrics = {}
    for i, ch_name in enumerate(target_channels):
        if i >= y_test.shape[-1]:
            break
        
        # For hourly, we evaluate on all timesteps (no mask)
        mae = np.mean(np.abs(y_test[..., i] - hourly_pred[..., i]))
        rmse = np.sqrt(np.mean((y_test[..., i] - hourly_pred[..., i]) ** 2))
        
        # R² score
        ss_res = np.sum((y_test[..., i] - hourly_pred[..., i]) ** 2)
        ss_tot = np.sum((y_test[..., i] - np.mean(y_test[..., i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        
        hourly_metrics[ch_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"  {ch_name:15s}: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}")
    
    # Save results
    results = {
        'reconstruction': recon_metrics,
        'hourly_prediction': hourly_metrics,
        'evaluation_config': {
            'model_path': str(args.model_path),
            'gap_days': args.gap_days,
            'n_samples': len(X_test)
        }
    }
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
