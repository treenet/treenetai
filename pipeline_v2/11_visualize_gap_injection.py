#!/usr/bin/env python3
"""
Visualize gap injection on test segments.

Shows model input (first 3 channels) with ground truth,
highlighting where gaps were injected with shaded background.
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.gaps.gap_injection import GapInjector


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize gap injection')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data',
        help='Directory with test data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lukovic/data/treenet/gap_visualization',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--gap-days',
        type=int,
        default=12,
        help='Gap length in days'
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
        default=3,
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


def visualize_sample(x_original, x_gapped, mask, sample_idx, gap_days, n_gaps, output_dir, dpi):
    """
    Visualize a single sample with gap injection.
    
    Args:
        x_original: Original input data (timesteps, channels)
        x_gapped: Data after gap injection (timesteps, channels)
        mask: Gap mask (timesteps, channels) - 0 where gap, 1 otherwise
        sample_idx: Sample index for title
        gap_days: Gap length in days
        n_gaps: Number of gaps injected
        output_dir: Output directory
        dpi: Plot DPI
    """
    timesteps = x_original.shape[0]
    hours = timesteps // 6  # 10-min data, so 6 steps per hour
    days = hours / 24
    
    # Time axis in days
    time_days = np.arange(timesteps) / (6 * 24)
    
    channel_names = ['Temperature (T)', 'Relative Humidity (RH)', 'Stem Radius']
    colors = ['#d62728', '#2ca02c', '#1f77b4']  # Red, green, blue
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for ch_idx in range(3):
        ax = axes[ch_idx]
        
        # Original data (ground truth)
        ax.plot(time_days, x_original[:, ch_idx], 
                color=colors[ch_idx], alpha=0.3, linewidth=1, 
                label='Ground Truth (hidden from model)')
        
        # Gapped data (model input)
        # Only plot where mask is 1 (valid data)
        gapped_visible = x_gapped[:, ch_idx].copy()
        gapped_visible[mask[:, ch_idx] == 0] = np.nan
        ax.plot(time_days, gapped_visible, 
                color=colors[ch_idx], linewidth=1.5,
                label='Model Input (visible to model)')
        
        # Shade gap regions
        gap_mask = mask[:, ch_idx] == 0
        if np.any(gap_mask):
            # Find contiguous gap regions
            diff = np.diff(np.concatenate([[0], gap_mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                ax.axvspan(time_days[start], time_days[min(end, len(time_days)-1)],
                          alpha=0.3, color='gray', label='Gap Region' if start == starts[0] else '')
        
        ax.set_ylabel(channel_names[ch_idx], fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if ch_idx == 0:
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(f'Sample {sample_idx}: Gap Injection Visualization\n'
                        f'{gap_days}-day gaps × {n_gaps} gaps = up to {gap_days * n_gaps} days missing '
                        f'(but gaps may be in different channels!)',
                        fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel('Time (days)', fontsize=12)
    
    # Add annotation about gap distribution per channel
    total_gaps_per_channel = []
    for ch_idx in range(3):
        gap_mask = mask[:, ch_idx] == 0
        gap_hours = np.sum(gap_mask) / 6  # Convert to hours
        gap_days_ch = gap_hours / 24
        total_gaps_per_channel.append(gap_days_ch)
    
    info_text = f"Gap days per channel: T={total_gaps_per_channel[0]:.1f}d, " \
                f"RH={total_gaps_per_channel[1]:.1f}d, Stem={total_gaps_per_channel[2]:.1f}d"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    output_path = output_dir / f'gap_injection_sample{sample_idx}_{gap_days}d_gaps.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"    {info_text}")


def visualize_gap_distribution(masks, gap_days, n_gaps, n_samples, output_dir, dpi):
    """
    Create a summary showing gap distribution across channels.
    """
    channel_names = ['Temperature', 'Rel. Humidity', 'Stem Radius']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Gap days per channel for each sample
    ax1 = axes[0]
    x_pos = np.arange(n_samples)
    width = 0.25
    
    for ch_idx, ch_name in enumerate(channel_names):
        gap_days_per_sample = []
        for i in range(n_samples):
            gap_mask = masks[i][:, ch_idx] == 0
            gap_hours = np.sum(gap_mask) / 6
            gap_days_per_sample.append(gap_hours / 24)
        
        ax1.bar(x_pos + (ch_idx - 1) * width, gap_days_per_sample, width, 
                label=ch_name, alpha=0.8)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Gap Days')
    ax1.set_title(f'Gap Days per Channel ({n_gaps}×{gap_days}d gaps)')
    ax1.set_xticks(x_pos)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Total data coverage
    ax2 = axes[1]
    segment_days = 30
    
    avg_gap_days = []
    for ch_idx in range(3):
        total = sum(np.sum(masks[i][:, ch_idx] == 0) / (6 * 24) for i in range(n_samples))
        avg_gap_days.append(total / n_samples)
    
    avg_valid_days = [segment_days - g for g in avg_gap_days]
    
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    x_pos = np.arange(3)
    
    bars1 = ax2.bar(x_pos, avg_valid_days, label='Valid Data', color=[c for c in colors], alpha=0.7)
    bars2 = ax2.bar(x_pos, avg_gap_days, bottom=avg_valid_days, label='Gaps', color='gray', alpha=0.5)
    
    ax2.set_ylabel('Days')
    ax2.set_title('Average Data Coverage per Channel')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(channel_names)
    ax2.legend()
    ax2.set_ylim(0, segment_days)
    ax2.axhline(y=segment_days, color='black', linestyle='--', alpha=0.3)
    
    # Add text showing percentages
    for i, (valid, gap) in enumerate(zip(avg_valid_days, avg_gap_days)):
        pct = valid / segment_days * 100
        ax2.text(i, segment_days + 0.5, f'{pct:.0f}% valid', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / f'gap_distribution_summary_{gap_days}d.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {output_path}")


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Gap Injection Visualization")
    print("="*60)
    print(f"Gap length: {args.gap_days} days")
    print(f"Gaps per segment: {args.n_gaps}")
    print(f"Samples to visualize: {args.n_samples}")
    
    # Load test data
    data_dir = Path(args.data_dir)
    print(f"\nLoading test data from: {data_dir}")
    
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    print(f"Test data shape: {X_test.shape}")
    
    # Create gap injector
    gap_injector = GapInjector(
        min_gap_days=args.gap_days,
        max_gap_days=args.gap_days,
        min_gaps_per_segment=args.n_gaps,
        max_gaps_per_segment=args.n_gaps,
        gap_channel_prob=1.0,  # Gap all eligible channels with equal probability
        random_seed=args.random_seed
    )
    
    # Inject gaps and visualize
    print(f"\nGenerating visualizations...")
    
    all_masks = []
    for i in range(args.n_samples):
        x_original = X_test[i]
        x_gapped, mask = gap_injector.inject_gaps(x_original)
        all_masks.append(mask)
        
        visualize_sample(
            x_original=x_original,
            x_gapped=x_gapped,
            mask=mask,
            sample_idx=i,
            gap_days=args.gap_days,
            n_gaps=args.n_gaps,
            output_dir=output_dir,
            dpi=args.dpi
        )
    
    # Create summary plot
    visualize_gap_distribution(
        masks=all_masks,
        gap_days=args.gap_days,
        n_gaps=args.n_gaps,
        n_samples=args.n_samples,
        output_dir=output_dir,
        dpi=args.dpi
    )
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
