#!/usr/bin/env python3
"""
Create box-and-whisker plots for gap-filling evaluation.

This script:
1. Loads test data and injects synthetic gaps
2. Generates predictions for multiple gap lengths
3. Creates unified box-and-whisker plots with all channels in same plot
4. Shared scale across all channels for direct comparison

Usage:
    python 10_plot_gap_evaluation.py \
        --model-path <path_to_model.keras> \
        --gap-days 1 7 12

Author: TreeNet AI Pipeline v2
Date: 2026-01-12
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNModel
from src.gaps.gap_injection import GapInjector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create box-and-whisker plots for gap-filling evaluation'
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
        default='/home/lukovic/data/treenet/gap_evaluation_plots/',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--gap-days',
        type=int,
        nargs='+',
        default=[7],
        help='Gap length(s) in days for synthetic gaps (can specify multiple)'
    )
    parser.add_argument(
        '--n-gaps',
        type=int,
        default=2,
        help='Number of gaps to inject per segment'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output figures'
    )
    
    return parser.parse_args()


def create_gap_mask_for_output(input_mask: np.ndarray, 
                               steps_per_hour: int = 6) -> np.ndarray:
    """
    Convert input mask (10-min resolution) to output mask (1-hour resolution).
    """
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


def evaluate_gap_length(model, X_test, y_test, gap_days, n_gaps, random_seed):
    """
    Evaluate model performance for a specific gap length.
    
    Returns:
        dict with gap_errors, non_gap_errors, gap_corrs, non_gap_corrs per channel
    """
    channel_names = ['Temperature', 'Rel. Humidity', 'Stem Radius']
    
    # Create gap injector
    gap_injector = GapInjector(
        min_gap_days=gap_days,
        max_gap_days=gap_days,
        min_gaps_per_segment=n_gaps,
        max_gaps_per_segment=n_gaps,
        gap_channel_prob=1.0,
        random_seed=random_seed
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
    
    # Generate predictions in batches
    batch_size = 16
    all_predictions = []
    
    for i in range(0, len(X_gapped), batch_size):
        end_idx = min(i + batch_size, len(X_gapped))
        batch_x = X_gapped[i:end_idx]
        batch_mask = masks[i:end_idx]
        
        batch_pred = model.predict([batch_x, batch_mask], verbose=0)
        
        if isinstance(batch_pred, list):
            all_predictions.append(batch_pred[1])
        else:
            all_predictions.append(batch_pred)
        
        print(f"    Processed {end_idx}/{len(X_gapped)} segments", end='\r')
    
    hourly_pred = np.concatenate(all_predictions, axis=0)
    print()
    
    # Create output mask
    gap_mask = create_gap_mask_for_output(masks)
    non_gap_mask = ~gap_mask
    
    # Collect results
    results = {
        'gap_errors': {},
        'non_gap_errors': {},
        'gap_corrs': {},
        'non_gap_corrs': {}
    }
    
    for i, ch_name in enumerate(channel_names):
        p = hourly_pred[:, :, i].flatten()
        t = y_test[:, :, i].flatten()
        g = gap_mask[:, :, i].flatten()
        ng = non_gap_mask[:, :, i].flatten()
        
        # Gap region errors
        gap_e = np.abs(p[g] - t[g])
        gap_e = gap_e[~np.isnan(gap_e)]
        results['gap_errors'][ch_name] = gap_e
        
        # Non-gap region errors  
        nongap_e = np.abs(p[ng] - t[ng])
        nongap_e = nongap_e[~np.isnan(nongap_e)]
        results['non_gap_errors'][ch_name] = nongap_e
        
        # Per-segment correlations
        gap_corrs = []
        nongap_corrs = []
        
        for seg_idx in range(len(X_test)):
            p_seg = hourly_pred[seg_idx, :, i]
            t_seg = y_test[seg_idx, :, i]
            g_seg = gap_mask[seg_idx, :, i]
            ng_seg = non_gap_mask[seg_idx, :, i]
            
            # Gap correlation for this segment
            if np.sum(g_seg) > 10:
                p_gap = p_seg[g_seg]
                t_gap = t_seg[g_seg]
                valid = ~(np.isnan(p_gap) | np.isnan(t_gap))
                if np.sum(valid) > 10:
                    p_gap = p_gap[valid]
                    t_gap = t_gap[valid]
                    if np.std(p_gap) > 1e-10 and np.std(t_gap) > 1e-10:
                        corr = np.corrcoef(p_gap, t_gap)[0, 1]
                        if not np.isnan(corr):
                            gap_corrs.append(corr)
            
            # Non-gap correlation for this segment
            if np.sum(ng_seg) > 10:
                p_ng = p_seg[ng_seg]
                t_ng = t_seg[ng_seg]
                valid = ~(np.isnan(p_ng) | np.isnan(t_ng))
                if np.sum(valid) > 10:
                    p_ng = p_ng[valid]
                    t_ng = t_ng[valid]
                    if np.std(p_ng) > 1e-10 and np.std(t_ng) > 1e-10:
                        corr = np.corrcoef(p_ng, t_ng)[0, 1]
                        if not np.isnan(corr):
                            nongap_corrs.append(corr)
        
        results['gap_corrs'][ch_name] = gap_corrs
        results['non_gap_corrs'][ch_name] = nongap_corrs
    
    return results


def create_unified_error_plot(all_results, gap_days_list, output_dir, dpi, n_segments):
    """
    Create a unified box plot with all channels in same figure.
    
    Layout: For each gap length, show pairs of (non-gap, gap) for each channel.
    Pairs close together, larger space between channels.
    """
    channel_names = ['Temperature', 'Rel. Humidity', 'Stem Radius']
    channel_short = ['T', 'RH', 'Stem']
    
    # Colors
    color_nongap = '#3498db'  # Blue
    color_gap = '#e74c3c'     # Red
    
    n_gaps = len(gap_days_list)
    n_channels = len(channel_names)
    
    # Create figure
    fig, axes = plt.subplots(1, n_gaps, figsize=(6 * n_gaps, 7), sharey=True)
    
    if n_gaps == 1:
        axes = [axes]
    
    # Collect all errors to determine unified scale
    all_errors = []
    for gap_days in gap_days_list:
        for ch_name in channel_names:
            all_errors.extend(all_results[gap_days]['gap_errors'][ch_name])
            all_errors.extend(all_results[gap_days]['non_gap_errors'][ch_name])
    
    # Determine y-axis limits (use 99th percentile to avoid outlier stretch)
    y_max = np.percentile(all_errors, 99)
    y_min = 0
    
    for ax_idx, gap_days in enumerate(gap_days_list):
        ax = axes[ax_idx]
        results = all_results[gap_days]
        
        # Positions: pairs close together (0.5 apart), channels separated (1.5 apart)
        # Channel 1: pos 1, 1.5 | Channel 2: pos 3.5, 4 | Channel 3: pos 6, 6.5
        positions = []
        data = []
        colors = []
        
        for ch_idx, ch_name in enumerate(channel_names):
            base = 1 + ch_idx * 2.5  # 1, 3.5, 6
            positions.extend([base, base + 0.5])
            data.extend([
                results['non_gap_errors'][ch_name],
                results['gap_errors'][ch_name]
            ])
            colors.extend([color_nongap, color_gap])
        
        # Create box plot
        bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                       widths=0.4, showfliers=False)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # X-axis labels at channel centers
        channel_centers = [1.25, 3.75, 6.25]
        ax.set_xticks(channel_centers)
        ax.set_xticklabels(channel_short, fontsize=12)
        
        # Set limits
        ax.set_xlim(0, 7.5)
        ax.set_ylim(y_min, y_max * 1.05)
        
        # Title and labels
        ax.set_title(f'{gap_days}-Day Gaps', fontsize=14, fontweight='bold')
        
        if ax_idx == 0:
            ax.set_ylabel('Absolute Error (normalized)', fontsize=12)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics below each pair
        for ch_idx, ch_name in enumerate(channel_names):
            ng_median = np.median(results['non_gap_errors'][ch_name])
            g_median = np.median(results['gap_errors'][ch_name])
            
            # Position text below the x-axis
            ax.annotate(f'{ng_median:.4f}', xy=(channel_centers[ch_idx] - 0.25, y_min), 
                       xytext=(channel_centers[ch_idx] - 0.25, -y_max * 0.08),
                       fontsize=8, ha='center', va='top', color=color_nongap,
                       annotation_clip=False)
            ax.annotate(f'{g_median:.4f}', xy=(channel_centers[ch_idx] + 0.25, y_min), 
                       xytext=(channel_centers[ch_idx] + 0.25, -y_max * 0.08),
                       fontsize=8, ha='center', va='top', color=color_gap,
                       annotation_clip=False)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=color_nongap, alpha=0.7, label='Non-Gap Regions'),
        mpatches.Patch(color=color_gap, alpha=0.7, label='Gap Regions')
    ]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Overall title
    gap_str = ', '.join([f'{d}d' for d in gap_days_list])
    fig.suptitle(f'Gap-Filling Performance: Absolute Error Distribution\n'
                f'({n_segments} × 30-day test segments, 2 channels with gaps)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    gap_suffix = '_'.join([str(d) for d in gap_days_list])
    output_file = output_dir / f'gap_evaluation_error_30d_segments_{gap_suffix}d.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_unified_rmse_plot(all_results, gap_days_list, output_dir, dpi, n_segments):
    """
    Create a unified MSE box plot with all channels in same figure.
    
    Layout: Box plot showing squared errors for non-gap and gap regions 
    per channel and gap length, with MSE values labeled.
    """
    channel_names = ['Temperature', 'Rel. Humidity', 'Stem Radius']
    channel_short = ['T', 'RH', 'Stem']
    
    # Colors
    color_nongap = '#3498db'  # Blue
    color_gap = '#e74c3c'     # Red
    
    n_gaps = len(gap_days_list)
    n_channels = len(channel_names)
    
    # Create figure
    fig, axes = plt.subplots(1, n_gaps, figsize=(6 * n_gaps, 7), sharey=True)
    
    if n_gaps == 1:
        axes = [axes]
    
    # Collect all squared errors to determine unified scale
    all_squared_errors = []
    for gap_days in gap_days_list:
        for ch_name in channel_names:
            all_squared_errors.extend(all_results[gap_days]['gap_errors'][ch_name]**2)
            all_squared_errors.extend(all_results[gap_days]['non_gap_errors'][ch_name]**2)
    
    # Determine y-axis limits (use 99th percentile to avoid outlier stretch)
    y_max = np.percentile(all_squared_errors, 99)
    y_min = 0
    
    for ax_idx, gap_days in enumerate(gap_days_list):
        ax = axes[ax_idx]
        results = all_results[gap_days]
        
        # Positions: pairs close together (0.5 apart), channels separated (1.5 apart)
        positions = []
        data = []
        colors = []
        
        for ch_idx, ch_name in enumerate(channel_names):
            base = 1 + ch_idx * 2.5  # 1, 3.5, 6
            positions.extend([base, base + 0.5])
            # Use squared errors for box plot
            data.extend([
                results['non_gap_errors'][ch_name]**2,
                results['gap_errors'][ch_name]**2
            ])
            colors.extend([color_nongap, color_gap])
        
        # Create box plot
        bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                       widths=0.4, showfliers=False)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # X-axis labels at channel centers
        channel_centers = [1.25, 3.75, 6.25]
        ax.set_xticks(channel_centers)
        ax.set_xticklabels(channel_short, fontsize=12)
        
        # Set limits
        ax.set_xlim(0, 7.5)
        ax.set_ylim(y_min, y_max * 1.05)
        
        # Title and labels
        ax.set_title(f'{gap_days}-Day Gaps', fontsize=14, fontweight='bold')
        
        if ax_idx == 0:
            ax.set_ylabel('Squared Error (normalized²)', fontsize=12)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add MSE statistics below each pair
        for ch_idx, ch_name in enumerate(channel_names):
            # MSE = Mean Squared Error
            ng_mse = np.mean(results['non_gap_errors'][ch_name]**2)
            g_mse = np.mean(results['gap_errors'][ch_name]**2)
            
            # Position text below the x-axis (show MSE value)
            ax.annotate(f'MSE:\n{ng_mse:.5f}', xy=(channel_centers[ch_idx] - 0.25, y_min), 
                       xytext=(channel_centers[ch_idx] - 0.25, -y_max * 0.12),
                       fontsize=7, ha='center', va='top', color=color_nongap,
                       annotation_clip=False)
            ax.annotate(f'MSE:\n{g_mse:.5f}', xy=(channel_centers[ch_idx] + 0.25, y_min), 
                       xytext=(channel_centers[ch_idx] + 0.25, -y_max * 0.12),
                       fontsize=7, ha='center', va='top', color=color_gap,
                       annotation_clip=False)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=color_nongap, alpha=0.7, label='Non-Gap Regions'),
        mpatches.Patch(color=color_gap, alpha=0.7, label='Gap Regions')
    ]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Overall title
    fig.suptitle(f'Gap-Filling Performance: MSE Distribution\n'
                f'({n_segments} × 30-day test segments, 2 channels with gaps)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    gap_suffix = '_'.join([str(d) for d in gap_days_list])
    output_file = output_dir / f'gap_evaluation_mse_30d_segments_{gap_suffix}d.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_unified_correlation_plot(all_results, gap_days_list, output_dir, dpi, n_segments):
    """
    Create a unified correlation box plot with all channels in same figure.
    """
    channel_names = ['Temperature', 'Rel. Humidity', 'Stem Radius']
    channel_short = ['T', 'RH', 'Stem']
    
    # Colors
    color_nongap = '#3498db'  # Blue
    color_gap = '#e74c3c'     # Red
    
    n_gaps = len(gap_days_list)
    
    # Create figure
    fig, axes = plt.subplots(1, n_gaps, figsize=(6 * n_gaps, 7), sharey=True)
    
    if n_gaps == 1:
        axes = [axes]
    
    for ax_idx, gap_days in enumerate(gap_days_list):
        ax = axes[ax_idx]
        results = all_results[gap_days]
        
        # Positions: pairs close together, channels separated
        positions = []
        data = []
        colors = []
        
        for ch_idx, ch_name in enumerate(channel_names):
            base = 1 + ch_idx * 2.5
            positions.extend([base, base + 0.5])
            data.extend([
                results['non_gap_corrs'][ch_name],
                results['gap_corrs'][ch_name]
            ])
            colors.extend([color_nongap, color_gap])
        
        # Create box plot
        bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                       widths=0.4, showfliers=False)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # X-axis labels
        channel_centers = [1.25, 3.75, 6.25]
        ax.set_xticks(channel_centers)
        ax.set_xticklabels(channel_short, fontsize=12)
        
        # Set limits
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0.5, 1.02)
        
        # Title and labels
        ax.set_title(f'{gap_days}-Day Gaps', fontsize=14, fontweight='bold')
        
        if ax_idx == 0:
            ax.set_ylabel('Correlation Coefficient', fontsize=12)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics below each pair
        for ch_idx, ch_name in enumerate(channel_names):
            ng_median = np.median(results['non_gap_corrs'][ch_name]) if results['non_gap_corrs'][ch_name] else 0
            g_median = np.median(results['gap_corrs'][ch_name]) if results['gap_corrs'][ch_name] else 0
            
            ax.annotate(f'{ng_median:.3f}', xy=(channel_centers[ch_idx] - 0.25, 0.5), 
                       xytext=(channel_centers[ch_idx] - 0.25, 0.46),
                       fontsize=8, ha='center', va='top', color=color_nongap,
                       annotation_clip=False)
            ax.annotate(f'{g_median:.3f}', xy=(channel_centers[ch_idx] + 0.25, 0.5), 
                       xytext=(channel_centers[ch_idx] + 0.25, 0.46),
                       fontsize=8, ha='center', va='top', color=color_gap,
                       annotation_clip=False)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=color_nongap, alpha=0.7, label='Non-Gap Regions'),
        mpatches.Patch(color=color_gap, alpha=0.7, label='Gap Regions')
    ]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Overall title
    gap_str = ', '.join([f'{d}d' for d in gap_days_list])
    fig.suptitle(f'Gap-Filling Performance: Correlation Distribution (per segment)\n'
                f'({n_segments} × 30-day test segments, 2 channels with gaps)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    gap_suffix = '_'.join([str(d) for d in gap_days_list])
    output_file = output_dir / f'gap_evaluation_correlation_30d_segments_{gap_suffix}d.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*80)
    print("TreeNet AI - Gap-Filling Box-and-Whisker Plots (Unified)")
    print("="*80)
    
    # Create output directory
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
    n_segments = len(X_test)
    
    # Evaluate each gap length
    all_results = {}
    
    for gap_days in args.gap_days:
        print(f"\n{'='*60}")
        print(f"Evaluating {gap_days}-day gaps...")
        print(f"{'='*60}")
        
        results = evaluate_gap_length(
            model=model,
            X_test=X_test,
            y_test=y_test,
            gap_days=gap_days,
            n_gaps=args.n_gaps,
            random_seed=args.random_seed
        )
        all_results[gap_days] = results
        
        # Print summary
        print(f"\n{gap_days}-day gap summary:")
        for ch_name in ['Temperature', 'Rel. Humidity', 'Stem Radius']:
            ng_err = np.median(results['non_gap_errors'][ch_name])
            g_err = np.median(results['gap_errors'][ch_name])
            ng_corr = np.median(results['non_gap_corrs'][ch_name]) if results['non_gap_corrs'][ch_name] else 0
            g_corr = np.median(results['gap_corrs'][ch_name]) if results['gap_corrs'][ch_name] else 0
            print(f"  {ch_name}:")
            print(f"    Error: non-gap={ng_err:.4f}, gap={g_err:.4f}")
            print(f"    Corr:  non-gap={ng_corr:.4f}, gap={g_corr:.4f}")
    
    # Create unified plots
    print(f"\n{'='*60}")
    print("Creating unified plots...")
    print(f"{'='*60}")
    
    create_unified_error_plot(all_results, args.gap_days, output_dir, args.dpi, n_segments)
    create_unified_rmse_plot(all_results, args.gap_days, output_dir, args.dpi, n_segments)
    create_unified_correlation_plot(all_results, args.gap_days, output_dir, args.dpi, n_segments)
    
    print(f"\n{'='*80}")
    print("Plotting complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
