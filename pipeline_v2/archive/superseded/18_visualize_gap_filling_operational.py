#!/usr/bin/env python3
"""
Visualize gap-filling results for test segments with OPERATIONAL DENORMALIZATION.

This script uses INPUT normalization parameters to denormalize ALL curves (including model output).
This matches what would happen in real deployment when LM data is not available.

Key difference from 17_visualize_gap_filling_denorm.py:
- That script uses LM normalization for model output (ideal but not operationally feasible)
- This script uses INPUT normalization for model output (realistic operational scenario)

For each randomly selected segment, creates a figure with 3 subplots:
1. Temperature (T): raw L1 with gap, model output, ground truth LM - all in °C using INPUT params
2. Relative Humidity (RH): raw L1 with gap, model output, ground truth LM - all in % using INPUT params
3. Stem Radius: raw L2 with gap, model output, ground truth LM - all in μm using INPUT params

Gap regions are shown with shaded background.
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
    parser = argparse.ArgumentParser(description='Visualize gap-filling on test segments (operational denormalization)')
    parser.add_argument(
        '--model-path',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260111_152352_segment_norm_attention/best_model.keras',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data',
        help='Directory with test data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lukovic/data/treenet/gap_filling_visualization_operational',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--gap-days',
        type=int,
        default=7,
        help='Gap length in days'
    )
    parser.add_argument(
        '--n-gaps',
        type=int,
        default=2,
        help='Number of channels with gaps per segment'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output figures'
    )
    return parser.parse_args()


def downsample_to_hourly(data_10min, steps_per_hour=6):
    """
    Downsample 10-minute data to hourly by averaging.
    
    Args:
        data_10min: Array of shape (timesteps, channels) with 10-min data
        steps_per_hour: Number of 10-min steps per hour (default 6)
    
    Returns:
        Array of shape (timesteps//steps_per_hour, channels) with hourly data
    """
    n_timesteps, n_channels = data_10min.shape
    n_hours = n_timesteps // steps_per_hour
    
    # Reshape to (n_hours, steps_per_hour, n_channels)
    reshaped = data_10min[:n_hours * steps_per_hour].reshape(n_hours, steps_per_hour, n_channels)
    
    # Average over the hour (ignoring NaN)
    hourly = np.nanmean(reshaped, axis=1)
    
    return hourly


def create_hourly_gap_mask(mask_10min, steps_per_hour=6):
    """
    Convert 10-minute gap mask to hourly gap mask.
    An hour is considered gapped if ANY of its 10-min steps have a gap.
    
    Args:
        mask_10min: Array of shape (timesteps, channels) with 0=gap, 1=valid
        steps_per_hour: Number of 10-min steps per hour
    
    Returns:
        Array of shape (n_hours, channels) with True=gap, False=valid
    """
    n_timesteps, n_channels = mask_10min.shape
    n_hours = n_timesteps // steps_per_hour
    
    hourly_mask = np.zeros((n_hours, n_channels), dtype=bool)
    
    for h in range(n_hours):
        start_idx = h * steps_per_hour
        end_idx = start_idx + steps_per_hour
        
        for c in range(n_channels):
            # Gap if ANY 10-min step has mask=0
            hourly_mask[h, c] = np.any(mask_10min[start_idx:end_idx, c] == 0)
    
    return hourly_mask


def denormalize_using_input_params(data_normalized, segment_meta, channel_idx):
    """
    Denormalize data using INPUT parameters (operational scenario).
    
    This is what we would use in real deployment when LM data is not available.
    
    Args:
        data_normalized: Normalized data (720,)
        segment_meta: SegmentMetadata object with input_min, input_diff
        channel_idx: 0=temp, 1=rh, 2=stem
    
    Returns:
        Denormalized data in original units
    """
    input_channels = ['temp_treenet', 'rh_treenet', 'stem']
    ch = input_channels[channel_idx]
    
    min_val = segment_meta.input_min[ch]
    diff_val = segment_meta.input_diff[ch]
    
    return data_normalized * diff_val + min_val


def denormalize_using_output_params(data_normalized, segment_meta, channel_idx):
    """
    Denormalize data using OUTPUT (LM) parameters (ideal but not operationally feasible).
    
    This requires having LM data, which we won't have in real deployment.
    
    Args:
        data_normalized: Normalized data (720,)
        segment_meta: SegmentMetadata object with output_min, output_diff
        channel_idx: 0=temp, 1=rh, 2=stem
    
    Returns:
        Denormalized data in original units
    """
    output_channels = ['local_T', 'local_RH', 'stem']
    ch = output_channels[channel_idx]
    
    min_val = segment_meta.output_min[ch]
    diff_val = segment_meta.output_diff[ch]
    
    return data_normalized * diff_val + min_val


def visualize_sample_operational(x_original, x_gapped, mask, y_pred, y_truth, 
                                  segment_meta, sample_idx, gap_days, output_dir, dpi):
    """
    Visualize a single sample using OPERATIONAL (input-based) denormalization.
    
    This shows what the model output would look like in real deployment.
    
    Args:
        x_original: Original input data (4320, 11) - 10-min, normalized
        x_gapped: Data after gap injection (4320, 11) - 10-min, normalized
        mask: Gap mask (4320, 11) - 0 where gap, 1 otherwise
        y_pred: Model prediction (720, 3) - hourly, normalized
        y_truth: Ground truth (720, 3) - hourly, normalized
        segment_meta: SegmentMetadata with min/diff values
        sample_idx: Sample index for title
        gap_days: Gap length in days
        output_dir: Output directory
        dpi: Plot DPI
    """
    # Get hourly input (for display) - first 3 channels: temp, rh, stem
    x_original_hourly = downsample_to_hourly(x_original[:, :3])
    x_gapped_hourly = downsample_to_hourly(x_gapped[:, :3])
    
    # Get hourly gap mask
    hourly_gap_mask = create_hourly_gap_mask(mask[:, :3])
    
    # Time axis in days
    n_hours = 720
    time_days = np.arange(n_hours) / 24
    
    # Channel configuration with units
    channel_config = [
        {
            'name': 'Temperature',
            'unit': '°C',
            'ylabel': 'Temperature (°C)',
        },
        {
            'name': 'Relative Humidity',
            'unit': '%',
            'ylabel': 'Relative Humidity (%)',
        },
        {
            'name': 'Stem Radius',
            'unit': 'μm',
            'ylabel': 'Stem Radius (μm)',
        }
    ]
    
    colors = {
        'raw': '#1f77b4',       # Blue
        'model': '#ff7f0e',     # Orange  
        'truth': '#2ca02c',     # Green
        'truth_lm': '#d62728'   # Red - for LM scale reference
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for ch_idx in range(3):
        ax = axes[ch_idx]
        cfg = channel_config[ch_idx]
        
        # Get normalized data for this channel
        raw_norm = x_original_hourly[:, ch_idx]
        gapped_norm = x_gapped_hourly[:, ch_idx]
        pred_norm = y_pred[:, ch_idx]
        truth_norm = y_truth[:, ch_idx]
        gap_mask = hourly_gap_mask[:, ch_idx]
        
        # Denormalize ALL using INPUT parameters (operational scenario)
        raw_denorm = denormalize_using_input_params(raw_norm, segment_meta, ch_idx)
        pred_denorm = denormalize_using_input_params(pred_norm, segment_meta, ch_idx)
        truth_denorm_input = denormalize_using_input_params(truth_norm, segment_meta, ch_idx)
        
        # Also get ground truth with correct LM denormalization for reference
        truth_denorm_lm = denormalize_using_output_params(truth_norm, segment_meta, ch_idx)
        
        # Create gapped raw data (set gap regions to NaN for plotting)
        raw_with_gap = raw_denorm.copy()
        raw_with_gap[gap_mask] = np.nan
        
        # Plot ground truth with LM scale (dashed, for reference only)
        ax.plot(time_days, truth_denorm_lm, 
                color=colors['truth_lm'], alpha=0.4, linewidth=1, linestyle=':',
                label='Ground Truth (LM scale)', zorder=0)
        
        # Plot ground truth as it would appear with input params
        ax.plot(time_days, truth_denorm_input, 
                color=colors['truth'], alpha=0.7, linewidth=1.5,
                label='Ground Truth (input scale)', zorder=1)
        
        # Plot raw data with gaps
        ax.plot(time_days, raw_with_gap,
                color=colors['raw'], alpha=0.8, linewidth=1.5, linestyle='--',
                label='Raw (L1/L2) with gap', zorder=2)
        
        # Plot model output (denormalized with input params - operational)
        ax.plot(time_days, pred_denorm,
                color=colors['model'], alpha=0.9, linewidth=2,
                label='Model Output (operational)', zorder=3)
        
        # Shade gap regions
        if np.any(gap_mask):
            # Find contiguous gap regions
            diff = np.diff(np.concatenate([[0], gap_mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for i, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan(time_days[start], time_days[min(end, len(time_days)-1)],
                          alpha=0.2, color='red', 
                          label='Gap Region' if i == 0 else '')
        
        # Calculate metrics for gap region (using operational denormalization)
        if np.any(gap_mask):
            gap_pred = pred_denorm[gap_mask]
            gap_truth = truth_denorm_input[gap_mask]
            valid = ~(np.isnan(gap_pred) | np.isnan(gap_truth))
            if np.sum(valid) > 10:
                gap_mae = np.mean(np.abs(gap_pred[valid] - gap_truth[valid]))
                gap_rmse = np.sqrt(np.mean((gap_pred[valid] - gap_truth[valid])**2))
                gap_corr = np.corrcoef(gap_pred[valid], gap_truth[valid])[0, 1]
                
                # Also compute scale difference for stem
                if ch_idx == 2:  # Stem
                    lm_range = (segment_meta.output_min['stem'], 
                               segment_meta.output_min['stem'] + segment_meta.output_diff['stem'])
                    input_range = (segment_meta.input_min['stem'], 
                                  segment_meta.input_min['stem'] + segment_meta.input_diff['stem'])
                    scale_note = f"\n⚠️ LM range: [{lm_range[0]:.0f}, {lm_range[1]:.0f}] vs Input: [{input_range[0]:.0f}, {input_range[1]:.0f}]"
                else:
                    scale_note = ""
                
                # Format metrics with units
                if ch_idx == 0:  # Temperature
                    metrics_text = f'Gap MAE: {gap_mae:.2f}°C, RMSE: {gap_rmse:.2f}°C, R: {gap_corr:.3f}{scale_note}'
                elif ch_idx == 1:  # RH
                    metrics_text = f'Gap MAE: {gap_mae:.1f}%, RMSE: {gap_rmse:.1f}%, R: {gap_corr:.3f}{scale_note}'
                else:  # Stem
                    metrics_text = f'Gap MAE: {gap_mae:.1f}μm, RMSE: {gap_rmse:.1f}μm, R: {gap_corr:.3f}{scale_note}'
            else:
                metrics_text = 'Insufficient gap data'
        else:
            metrics_text = 'No gap in this channel'
        
        ax.set_ylabel(cfg['ylabel'], fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add metrics annotation
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if ch_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Get site info from metadata
    site_info = f"Site {segment_meta.site_id}, Window: {segment_meta.window_start_utc.strftime('%Y-%m-%d')} to {segment_meta.window_end_utc.strftime('%Y-%m-%d')}"
    
    # Overall title
    fig.suptitle(f'Gap-Filling Visualization (OPERATIONAL Denormalization) - Sample {sample_idx}\n'
                f'({gap_days}-day gaps, 30-day segment)\n'
                f'{site_info}\n'
                f'⚠️ All curves use INPUT normalization (what we\'d have in production)',
                fontsize=12, fontweight='bold', y=1.04)
    
    axes[-1].set_xlabel('Time (days)', fontsize=12)
    
    plt.tight_layout()
    
    output_path = output_dir / f'gap_filling_sample{sample_idx}_{gap_days}d_operational.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Gap-Filling Visualization (OPERATIONAL DENORMALIZATION)")
    print("="*70)
    print("⚠️  This uses INPUT parameters for denormalization - realistic scenario!")
    print("⚠️  For stem: expect scale mismatch vs actual LM values")
    print("-"*70)
    print(f"Model: {args.model_path}")
    print(f"Gap length: {args.gap_days} days")
    print(f"Channels with gaps: {args.n_gaps}")
    print(f"Samples to visualize: {args.n_samples}")
    
    # Load model
    print("\nLoading model...")
    model = TCNModel.load(args.model_path)
    
    # Load test data
    data_dir = Path(args.data_dir)
    print(f"Loading test data from: {data_dir}")
    
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    # Load segment metadata (for denormalization)
    with open(data_dir / 'test_segment_ids.pkl', 'rb') as f:
        segment_meta_list = pickle.load(f)
    
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    print(f"Segment metadata loaded: {len(segment_meta_list)} segments")
    
    # Set random seed and select random samples
    np.random.seed(args.random_seed)
    n_total = len(X_test)
    sample_indices = np.random.choice(n_total, size=min(args.n_samples, n_total), replace=False)
    
    print(f"\nSelected samples: {sample_indices}")
    
    # Create gap injector
    gap_injector = GapInjector(
        min_gap_days=args.gap_days,
        max_gap_days=args.gap_days,
        min_gaps_per_segment=args.n_gaps,
        max_gaps_per_segment=args.n_gaps,
        gap_channel_prob=1.0,
        random_seed=args.random_seed
    )
    
    print("\nGenerating visualizations...")
    
    for i, idx in enumerate(sample_indices):
        print(f"\n[{i+1}/{len(sample_indices)}] Processing sample {idx}...")
        
        # Get original data
        x_original = X_test[idx]
        y_truth = y_test[idx]
        segment_meta = segment_meta_list[idx]
        
        # Inject gap
        x_gapped, mask = gap_injector.inject_gaps(x_original.copy())
        
        # Run model prediction
        x_batch = np.expand_dims(x_gapped, axis=0)
        mask_batch = np.expand_dims(mask, axis=0)
        
        pred = model.predict([x_batch, mask_batch], verbose=0)
        
        # Get hourly output (second output of multi-task model)
        if isinstance(pred, list):
            y_pred = pred[1][0]  # Shape: (720, 3)
        else:
            y_pred = pred[0]
        
        # Visualize (operational denormalization)
        visualize_sample_operational(
            x_original=x_original,
            x_gapped=x_gapped,
            mask=mask,
            y_pred=y_pred,
            y_truth=y_truth,
            segment_meta=segment_meta,
            sample_idx=idx,
            gap_days=args.gap_days,
            output_dir=output_dir,
            dpi=args.dpi
        )
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"\nValues are denormalized using INPUT (L1/L2) parameters:")
    print(f"  - Temperature: °C (similar to LM)")
    print(f"  - Relative Humidity: % (similar to LM)")
    print(f"  - Stem Radius: μm (⚠️ DIFFERENT scale than LM!)")
    print(f"\nNote: For stem, the dotted line shows actual LM scale for comparison.")


if __name__ == '__main__':
    main()
