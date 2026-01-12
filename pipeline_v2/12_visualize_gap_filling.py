#!/usr/bin/env python3
"""
Unified gap-filling visualization script.

This script visualizes gap-filling results for test segments with multiple
denormalization modes:
1. normalized: Values in [0,1] range (no denormalization)
2. ideal: Uses LM normalization constants (ideal but not operationally feasible)
3. operational: Uses INPUT normalization constants (realistic production scenario)

Usage examples:
    # Default visualization (normalized)
    python 12_visualize_gap_filling.py --data-dir /path/to/data
    
    # Operational denormalization (what would happen in production)
    python 12_visualize_gap_filling.py --data-dir /path/to/data --denorm operational
    
    # Ideal denormalization (for comparison only)
    python 12_visualize_gap_filling.py --data-dir /path/to/data --denorm ideal

Author: TreeNet AI Pipeline v2
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.models.tcn import TCNModel
from src.gaps.gap_injection import GapInjector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize gap-filling on test segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Denormalization modes:
  normalized:  Values in [0,1] range (no physical units)
  ideal:       Uses LM normalization constants (correct scale, but not available in production)
  operational: Uses INPUT normalization constants (realistic production scenario)
        """
    )
    parser.add_argument(
        '--model-path', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260111_152352_segment_norm_attention/best_model.keras',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data',
        help='Directory with test data'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for plots (auto-generated if not specified)'
    )
    parser.add_argument(
        '--denorm', type=str, choices=['normalized', 'ideal', 'operational'], default='normalized',
        help='Denormalization mode'
    )
    parser.add_argument('--gap-days', type=int, default=7, help='Gap length in days')
    parser.add_argument('--n-gaps', type=int, default=2, help='Number of channels with gaps per segment')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output figures')
    
    return parser.parse_args()


def downsample_to_hourly(data_10min, steps_per_hour=6):
    """Downsample 10-minute data to hourly by averaging."""
    n_timesteps, n_channels = data_10min.shape
    n_hours = n_timesteps // steps_per_hour
    reshaped = data_10min[:n_hours * steps_per_hour].reshape(n_hours, steps_per_hour, n_channels)
    return np.nanmean(reshaped, axis=1)


def create_hourly_gap_mask(mask_10min, steps_per_hour=6):
    """Convert 10-minute gap mask to hourly gap mask."""
    n_timesteps, n_channels = mask_10min.shape
    n_hours = n_timesteps // steps_per_hour
    hourly_mask = np.zeros((n_hours, n_channels), dtype=bool)
    
    for h in range(n_hours):
        start_idx = h * steps_per_hour
        end_idx = start_idx + steps_per_hour
        for c in range(n_channels):
            hourly_mask[h, c] = np.any(mask_10min[start_idx:end_idx, c] == 0)
    
    return hourly_mask


def denormalize_data(data_norm, segment_meta, channel_idx, use_output_params=True):
    """
    Denormalize data using segment metadata.
    
    Args:
        data_norm: Normalized data array
        segment_meta: SegmentMetadata object
        channel_idx: 0=temp, 1=rh, 2=stem
        use_output_params: True for LM params (ideal), False for input params (operational)
    """
    if use_output_params:
        # Output (LM) parameters - ideal but not available in production
        output_channels = ['local_T', 'local_RH', 'stem']
        ch = output_channels[channel_idx]
        min_val = segment_meta.output_min[ch]
        diff_val = segment_meta.output_diff[ch]
    else:
        # Input (L1/L2) parameters - operational scenario
        input_channels = ['temp_treenet', 'rh_treenet', 'stem']
        ch = input_channels[channel_idx]
        min_val = segment_meta.input_min[ch]
        diff_val = segment_meta.input_diff[ch]
    
    return data_norm * diff_val + min_val


def get_channel_config(denorm_mode):
    """Get channel configuration based on denormalization mode."""
    if denorm_mode == 'normalized':
        return [
            {'name': 'Temperature', 'unit': '', 'ylabel': 'Temperature (normalized)'},
            {'name': 'Relative Humidity', 'unit': '', 'ylabel': 'RH (normalized)'},
            {'name': 'Stem Radius', 'unit': '', 'ylabel': 'Stem (normalized)'},
        ]
    else:
        return [
            {'name': 'Temperature', 'unit': '°C', 'ylabel': 'Temperature (°C)'},
            {'name': 'Relative Humidity', 'unit': '%', 'ylabel': 'Relative Humidity (%)'},
            {'name': 'Stem Radius', 'unit': 'μm', 'ylabel': 'Stem Radius (μm)'},
        ]


def visualize_sample(x_original, x_gapped, mask, y_pred, y_truth, 
                     segment_meta, sample_idx, gap_days, output_dir, dpi, denorm_mode):
    """Visualize a single sample with gap injection and model predictions."""
    
    # Get hourly input (first 3 channels: temp, rh, stem)
    x_original_hourly = downsample_to_hourly(x_original[:, :3])
    x_gapped_hourly = downsample_to_hourly(x_gapped[:, :3])
    hourly_gap_mask = create_hourly_gap_mask(mask[:, :3])
    
    n_hours = 720
    time_days = np.arange(n_hours) / 24
    channel_config = get_channel_config(denorm_mode)
    
    colors = {
        'raw': '#1f77b4', 'model': '#ff7f0e', 'truth': '#2ca02c', 'truth_lm': '#d62728'
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for ch_idx in range(3):
        ax = axes[ch_idx]
        cfg = channel_config[ch_idx]
        gap_mask = hourly_gap_mask[:, ch_idx]
        
        # Get data based on denormalization mode
        if denorm_mode == 'normalized':
            raw_data = x_original_hourly[:, ch_idx]
            pred_data = y_pred[:, ch_idx]
            truth_data = y_truth[:, ch_idx]
        elif denorm_mode == 'ideal':
            # Use LM params for input (not realistic but correct scale)
            raw_data = denormalize_data(x_original_hourly[:, ch_idx], segment_meta, ch_idx, use_output_params=False)
            pred_data = denormalize_data(y_pred[:, ch_idx], segment_meta, ch_idx, use_output_params=True)
            truth_data = denormalize_data(y_truth[:, ch_idx], segment_meta, ch_idx, use_output_params=True)
        else:  # operational
            # Use input params for everything
            raw_data = denormalize_data(x_original_hourly[:, ch_idx], segment_meta, ch_idx, use_output_params=False)
            pred_data = denormalize_data(y_pred[:, ch_idx], segment_meta, ch_idx, use_output_params=False)
            truth_data = denormalize_data(y_truth[:, ch_idx], segment_meta, ch_idx, use_output_params=False)
            
            # Also show LM scale reference for comparison
            if ch_idx == 2:  # Only for stem
                truth_lm = denormalize_data(y_truth[:, ch_idx], segment_meta, ch_idx, use_output_params=True)
                ax.plot(time_days, truth_lm, color=colors['truth_lm'], alpha=0.4, 
                        linewidth=1, linestyle=':', label='Ground Truth (LM scale)', zorder=0)
        
        # Create gapped raw data
        raw_with_gap = raw_data.copy()
        raw_with_gap[gap_mask] = np.nan
        
        # Plot curves
        ax.plot(time_days, truth_data, color=colors['truth'], alpha=0.7, linewidth=1.5,
                label='Ground Truth', zorder=1)
        ax.plot(time_days, raw_with_gap, color=colors['raw'], alpha=0.8, linewidth=1.5,
                linestyle='--', label='Raw with gap', zorder=2)
        ax.plot(time_days, pred_data, color=colors['model'], alpha=0.9, linewidth=2,
                label='Model Output', zorder=3)
        
        # Shade gap regions
        if np.any(gap_mask):
            diff = np.diff(np.concatenate([[0], gap_mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for i, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan(time_days[start], time_days[min(end, len(time_days)-1)],
                          alpha=0.2, color='red', label='Gap Region' if i == 0 else '')
        
        # Calculate metrics for gap region
        if np.any(gap_mask):
            gap_pred = pred_data[gap_mask]
            gap_truth = truth_data[gap_mask]
            valid = ~(np.isnan(gap_pred) | np.isnan(gap_truth))
            if np.sum(valid) > 10:
                gap_mae = np.mean(np.abs(gap_pred[valid] - gap_truth[valid]))
                gap_corr = np.corrcoef(gap_pred[valid], gap_truth[valid])[0, 1]
                
                if denorm_mode == 'normalized':
                    metrics_text = f'Gap MAE: {gap_mae:.4f}, R: {gap_corr:.3f}'
                elif ch_idx == 0:
                    metrics_text = f'Gap MAE: {gap_mae:.2f}°C, R: {gap_corr:.3f}'
                elif ch_idx == 1:
                    metrics_text = f'Gap MAE: {gap_mae:.1f}%, R: {gap_corr:.3f}'
                else:
                    metrics_text = f'Gap MAE: {gap_mae:.1f}μm, R: {gap_corr:.3f}'
                    if denorm_mode == 'operational':
                        metrics_text += f'\n⚠️ Using INPUT params (different scale than LM)'
            else:
                metrics_text = 'Insufficient gap data'
        else:
            metrics_text = 'No gap in this channel'
        
        ax.set_ylabel(cfg['ylabel'], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if ch_idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Title
    mode_labels = {
        'normalized': 'NORMALIZED [0,1]',
        'ideal': 'IDEAL (LM params)',
        'operational': 'OPERATIONAL (input params)'
    }
    site_info = f"Site {segment_meta.site_id}" if segment_meta else ""
    
    fig.suptitle(f'Gap-Filling Visualization - Sample {sample_idx}\n'
                f'({gap_days}-day gaps, 30-day segment) | Mode: {mode_labels[denorm_mode]}\n'
                f'{site_info}',
                fontsize=12, fontweight='bold', y=1.02)
    
    axes[-1].set_xlabel('Time (days)', fontsize=12)
    plt.tight_layout()
    
    # Save
    suffix = {'normalized': '', 'ideal': '_denorm', 'operational': '_operational'}
    output_path = output_dir / f'gap_filling_sample{sample_idx}_{gap_days}d{suffix[denorm_mode]}.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def main():
    args = parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = {'normalized': '', 'ideal': '_denorm', 'operational': '_operational'}
        output_dir = Path(f'/home/lukovic/data/treenet/gap_filling_visualization{suffix[args.denorm]}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mode_desc = {
        'normalized': 'NORMALIZED [0,1] - no physical units',
        'ideal': 'IDEAL DENORMALIZATION - using LM constants',
        'operational': 'OPERATIONAL DENORMALIZATION - using INPUT constants (realistic)'
    }
    
    print("=" * 70)
    print("Gap-Filling Visualization")
    print("=" * 70)
    print(f"Mode: {mode_desc[args.denorm]}")
    print(f"Model: {args.model_path}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model = TCNModel.load(args.model_path)
    
    # Load data
    data_dir = Path(args.data_dir)
    print(f"Loading data from: {data_dir}")
    
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    # Load metadata for denormalization (if needed)
    segment_meta_list = None
    if args.denorm != 'normalized':
        with open(data_dir / 'test_segment_ids.pkl', 'rb') as f:
            segment_meta_list = pickle.load(f)
        print(f"Segment metadata loaded: {len(segment_meta_list)} segments")
    
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    # Select samples
    np.random.seed(args.random_seed)
    sample_indices = np.random.choice(len(X_test), size=min(args.n_samples, len(X_test)), replace=False)
    print(f"\nSelected samples: {sample_indices}")
    
    # Gap injector
    gap_injector = GapInjector(
        min_gap_days=args.gap_days, max_gap_days=args.gap_days,
        min_gaps_per_segment=args.n_gaps, max_gaps_per_segment=args.n_gaps,
        gap_channel_prob=1.0, random_seed=args.random_seed
    )
    
    print("\nGenerating visualizations...")
    
    for i, idx in enumerate(sample_indices):
        print(f"\n[{i+1}/{len(sample_indices)}] Processing sample {idx}...")
        
        x_original = X_test[idx]
        y_truth = y_test[idx]
        segment_meta = segment_meta_list[idx] if segment_meta_list else None
        
        x_gapped, mask = gap_injector.inject_gaps(x_original.copy())
        
        x_batch = np.expand_dims(x_gapped, axis=0)
        mask_batch = np.expand_dims(mask, axis=0)
        
        pred = model.predict([x_batch, mask_batch], verbose=0)
        y_pred = pred[1][0] if isinstance(pred, list) else pred[0]
        
        visualize_sample(
            x_original, x_gapped, mask, y_pred, y_truth,
            segment_meta, idx, args.gap_days, output_dir, args.dpi, args.denorm
        )
    
    print(f"\n{'=' * 70}")
    print(f"All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
