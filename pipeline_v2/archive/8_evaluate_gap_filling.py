#!/usr/bin/env python3
"""
Evaluate model performance specifically on gap regions.

This script:
1. Loads reconstruction data (predictions) from test sites
2. Loads LM (ground truth) data
3. Computes metrics ONLY on gap regions where:
   - Input (L1/L2) has a gap
   - LM (ground truth) has valid data
4. Reports gap-filling performance

This evaluates the model's true gap-filling capability: how well does it
reconstruct missing sensor data using only the surrounding context?

Usage:
    python 8_evaluate_gap_filling.py \
        --recon-path reconstructions/test_sites/reconstructed_site22_*.ftr \
        --output-dir gap_evaluation_results/

Author: TreeNet AI Pipeline v2
Date: 2026-01-12
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate model gap-filling performance on test sites'
    )
    
    parser.add_argument(
        '--recon-path',
        type=str,
        required=True,
        help='Path to reconstruction file(s). Can use glob patterns.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lukovic/data/treenet/gap_evaluation/',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--lm-data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data',
        help='Directory containing LM (ground truth) data'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def load_lm_data(site_id: int, thermo_id: int, hygro_id: int, dendro_id: int,
                 lm_data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load LM (ground truth) data for a site combination.
    
    Returns dict with keys: 'temp', 'rh', 'stem'
    """
    lm_data = {}
    
    # Temperature LM (from thermometer L1 - we use L1 as ground truth for T/RH)
    temp_path = lm_data_dir / 'thermometer_l1' / f'thermometer_l1_series_id_{thermo_id}.ftr'
    if temp_path.exists():
        temp_df = pd.read_feather(temp_path)
        temp_df['ts'] = pd.to_datetime(temp_df['ts'], utc=True)
        lm_data['temp'] = temp_df
    else:
        print(f"  Warning: Temperature LM not found at {temp_path}")
        lm_data['temp'] = None
    
    # Humidity LM
    rh_path = lm_data_dir / 'hygrometer_l1' / f'hygrometer_l1_series_id_{hygro_id}.ftr'
    if rh_path.exists():
        rh_df = pd.read_feather(rh_path)
        rh_df['ts'] = pd.to_datetime(rh_df['ts'], utc=True)
        lm_data['rh'] = rh_df
    else:
        print(f"  Warning: Humidity LM not found at {rh_path}")
        lm_data['rh'] = None
    
    # Stem LM
    stem_path = lm_data_dir / 'dendrometer_lm' / f'dendrometer_lm_series_id_{dendro_id}.ftr'
    if stem_path.exists():
        stem_df = pd.read_feather(stem_path)
        stem_df['ts'] = pd.to_datetime(stem_df['ts'], utc=True)
        lm_data['stem'] = stem_df
    else:
        print(f"  Warning: Stem LM not found at {stem_path}")
        lm_data['stem'] = None
    
    return lm_data


def resample_to_hourly(df: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
    """Resample sensor data to hourly means."""
    df = df.set_index('ts')
    hourly = df[value_col].resample('H').mean().reset_index()
    hourly.columns = ['ts', value_col]
    return hourly


def compute_metrics(pred: np.ndarray, truth: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        pred: Predicted values
        truth: Ground truth values
        mask: Boolean mask (True = evaluate this point)
    
    Returns:
        Dictionary with MAE, RMSE, correlation, R², valid count
    """
    if mask is not None:
        pred = pred[mask]
        truth = truth[mask]
    
    # Remove NaN
    valid = ~(np.isnan(pred) | np.isnan(truth))
    pred = pred[valid]
    truth = truth[valid]
    
    if len(pred) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'corr': np.nan,
            'r2': np.nan,
            'n_samples': 0
        }
    
    # MAE
    mae = np.mean(np.abs(pred - truth))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    
    # Correlation
    if np.std(pred) > 1e-10 and np.std(truth) > 1e-10:
        corr = np.corrcoef(pred, truth)[0, 1]
    else:
        corr = np.nan
    
    # R²
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else np.nan
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'corr': float(corr) if not np.isnan(corr) else None,
        'r2': float(r2) if not np.isnan(r2) else None,
        'n_samples': int(len(pred))
    }


def evaluate_gap_filling(recon_df: pd.DataFrame, lm_data: Dict[str, pd.DataFrame],
                        verbose: bool = False) -> Dict:
    """
    Evaluate gap-filling performance.
    
    Computes metrics for:
    1. Gap regions only (where input had gaps but LM has data)
    2. Non-gap regions (for comparison)
    3. All regions
    """
    results = {}
    
    # Channels to evaluate
    channels = [
        ('local_T', 'temp', 'is_gap_T'),
        ('local_RH', 'rh', 'is_gap_RH'),
        ('stem', 'stem', 'is_gap_stem')
    ]
    
    for pred_col, lm_key, gap_col in channels:
        if verbose:
            print(f"\n  Evaluating {pred_col}...")
        
        lm_df = lm_data.get(lm_key)
        if lm_df is None:
            if verbose:
                print(f"    Skipping {pred_col} - no LM data")
            results[pred_col] = {'error': 'No LM data'}
            continue
        
        # Resample LM to hourly
        lm_hourly = resample_to_hourly(lm_df, 'value')
        
        # Merge reconstruction with LM
        merged = pd.merge(
            recon_df[['ts', pred_col, gap_col]],
            lm_hourly.rename(columns={'value': 'lm_value'}),
            on='ts',
            how='inner'
        )
        
        if len(merged) == 0:
            if verbose:
                print(f"    No overlapping data for {pred_col}")
            results[pred_col] = {'error': 'No overlapping timestamps'}
            continue
        
        pred_values = merged[pred_col].values
        lm_values = merged['lm_value'].values
        is_gap = merged[gap_col].values.astype(bool)
        
        # Count gaps with valid LM data
        gap_with_lm = is_gap & ~np.isnan(lm_values)
        non_gap_with_lm = ~is_gap & ~np.isnan(lm_values)
        
        if verbose:
            print(f"    Total samples: {len(merged)}")
            print(f"    Gap samples (with LM): {gap_with_lm.sum()}")
            print(f"    Non-gap samples (with LM): {non_gap_with_lm.sum()}")
        
        # Compute metrics
        gap_metrics = compute_metrics(pred_values, lm_values, gap_with_lm)
        non_gap_metrics = compute_metrics(pred_values, lm_values, non_gap_with_lm)
        all_metrics = compute_metrics(pred_values, lm_values)
        
        results[pred_col] = {
            'gap_regions': gap_metrics,
            'non_gap_regions': non_gap_metrics,
            'all_regions': all_metrics,
            'total_samples': len(merged),
            'gap_samples': int(gap_with_lm.sum()),
            'non_gap_samples': int(non_gap_with_lm.sum())
        }
    
    return results


def extract_site_info(recon_path: Path) -> Tuple[int, int, int, int]:
    """
    Extract site and sensor IDs from reconstruction filename.
    
    Expected format: reconstructed_site{site}_T{thermo}_H{hygro}_D{dendro}.ftr
    """
    name = recon_path.stem
    # Parse: reconstructed_site22_T119_H118_D120
    import re
    match = re.match(r'reconstructed_site(\d+)_T(\d+)_H(\d+)_D(\d+)', name)
    if not match:
        raise ValueError(f"Cannot parse site info from filename: {name}")
    
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Find reconstruction files
    recon_path = Path(args.recon_path)
    if recon_path.is_file():
        recon_files = [recon_path]
    else:
        # Glob pattern
        parent = recon_path.parent
        pattern = recon_path.name
        recon_files = list(parent.glob(pattern))
    
    if not recon_files:
        print(f"No reconstruction files found matching: {args.recon_path}")
        sys.exit(1)
    
    print("="*80)
    print("TreeNet AI - Gap-Filling Evaluation")
    print("="*80)
    print(f"Evaluating {len(recon_files)} reconstruction file(s)")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lm_data_dir = Path(args.lm_data_dir)
    
    all_results = {}
    
    for recon_file in recon_files:
        print(f"\n{'='*60}")
        print(f"Processing: {recon_file.name}")
        print(f"{'='*60}")
        
        # Extract site info
        try:
            site_id, thermo_id, hygro_id, dendro_id = extract_site_info(recon_file)
        except ValueError as e:
            print(f"  Error: {e}")
            continue
        
        print(f"  Site: {site_id}, T={thermo_id}, H={hygro_id}, D={dendro_id}")
        
        # Load reconstruction data
        recon_df = pd.read_feather(recon_file)
        recon_df['ts'] = pd.to_datetime(recon_df['ts'], utc=True)
        
        # Check if gap columns exist
        required_cols = ['is_gap_T', 'is_gap_RH', 'is_gap_stem']
        missing_cols = [c for c in required_cols if c not in recon_df.columns]
        if missing_cols:
            print(f"  Warning: Missing gap columns: {missing_cols}")
            print("  Run reconstruction with per-channel gap tracking first.")
            continue
        
        # Load LM data
        print("  Loading LM (ground truth) data...")
        lm_data = load_lm_data(site_id, thermo_id, hygro_id, dendro_id, lm_data_dir)
        
        # Evaluate
        print("  Evaluating gap-filling performance...")
        results = evaluate_gap_filling(recon_df, lm_data, verbose=args.verbose)
        
        # Store results
        key = f"site{site_id}_T{thermo_id}_H{hygro_id}_D{dendro_id}"
        all_results[key] = results
        
        # Print summary
        print("\n  Results:")
        print("  " + "-"*56)
        print(f"  {'Channel':<12} {'Region':<12} {'MAE':>10} {'RMSE':>10} {'Corr':>10} {'N':>8}")
        print("  " + "-"*56)
        
        for channel, channel_results in results.items():
            if 'error' in channel_results:
                print(f"  {channel:<12} {'ERROR':<12} {channel_results['error']}")
                continue
            
            for region in ['gap_regions', 'non_gap_regions', 'all_regions']:
                metrics = channel_results[region]
                region_short = region.replace('_regions', '').replace('_', '-')
                mae = f"{metrics['mae']:.4f}" if metrics['mae'] is not None else 'N/A'
                rmse = f"{metrics['rmse']:.4f}" if metrics['rmse'] is not None else 'N/A'
                corr = f"{metrics['corr']:.4f}" if metrics['corr'] is not None else 'N/A'
                n = metrics['n_samples']
                print(f"  {channel:<12} {region_short:<12} {mae:>10} {rmse:>10} {corr:>10} {n:>8}")
    
    # Save all results
    output_file = output_dir / f'gap_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Summary across all sites
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("SUMMARY ACROSS ALL SITES (Gap Regions Only)")
        print("="*80)
        
        for channel in ['local_T', 'local_RH', 'stem']:
            maes = []
            rmses = []
            corrs = []
            n_total = 0
            
            for site_key, site_results in all_results.items():
                if channel in site_results and 'gap_regions' in site_results[channel]:
                    m = site_results[channel]['gap_regions']
                    if m['n_samples'] > 0:
                        maes.append(m['mae'])
                        rmses.append(m['rmse'])
                        if m['corr'] is not None:
                            corrs.append(m['corr'])
                        n_total += m['n_samples']
            
            if maes:
                print(f"\n{channel}:")
                print(f"  Mean MAE across sites: {np.mean(maes):.4f}")
                print(f"  Mean RMSE across sites: {np.mean(rmses):.4f}")
                if corrs:
                    print(f"  Mean Correlation: {np.mean(corrs):.4f}")
                print(f"  Total gap samples: {n_total}")


if __name__ == '__main__':
    main()
