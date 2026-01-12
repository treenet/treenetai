#!/usr/bin/env python3
"""
Create box-and-whisker plots for reconstruction evaluation metrics.

This script generates side-by-side boxplots comparing:
- Correlation coefficients across all test combinations
- MAE values across all test combinations
- R² values across all test combinations

For each channel (Temperature, Relative Humidity, Stem).

Usage:
    python visualize_boxplots.py --metrics-file <path> [--output-dir <path>]

Author: TreeNet AI Pipeline
Date: 2026-01-12
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def load_metrics(metrics_file: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def extract_metrics_by_channel(
    all_metrics: Dict,
    metric_name: str = 'correlation'
) -> Dict[str, List[float]]:
    """Extract a specific metric for each channel across all combinations."""
    channels = ['Temperature', 'Relative Humidity', 'Stem']
    result = {ch: [] for ch in channels}
    
    for combo_id, combo_metrics in all_metrics.items():
        for ch in channels:
            if ch in combo_metrics and metric_name in combo_metrics[ch]:
                value = combo_metrics[ch][metric_name]
                if not np.isnan(value):
                    result[ch].append(value)
    
    return result


def create_boxplot_figure(
    metrics_data: Dict[str, Dict[str, List[float]]],
    title: str,
    output_path: Path,
    figsize: tuple = (15, 5)
):
    """Create a figure with 3 subplots (one per metric type)."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Metric configurations
    metric_configs = [
        ('correlation', 'Correlation Coefficient', (-1, 1), 'Correlation'),
        ('mae', 'Mean Absolute Error', None, 'MAE'),
        ('r2', 'R² Score', None, 'R²')
    ]
    
    channels = ['Temperature', 'Relative Humidity', 'Stem']
    colors = ['#2ca02c', '#1f77b4', '#d62728']  # Green, Blue, Red
    
    for ax_idx, (metric_name, ylabel, ylim, title_suffix) in enumerate(metric_configs):
        ax = axes[ax_idx]
        
        # Prepare data for boxplot
        data = []
        labels = []
        
        for ch in channels:
            if ch in metrics_data and metric_name in metrics_data[ch]:
                values = metrics_data[ch][metric_name]
                if values:
                    data.append(values)
                    # Shorten labels
                    short_label = {'Temperature': 'Temp', 'Relative Humidity': 'RH', 'Stem': 'Stem'}[ch]
                    labels.append(f'{short_label}\n(n={len(values)})')
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Style whiskers, caps, medians
            for whisker in bp['whiskers']:
                whisker.set(color='black', linewidth=1.5)
            for cap in bp['caps']:
                cap.set(color='black', linewidth=1.5)
            for median in bp['medians']:
                median.set(color='black', linewidth=2)
            for flier in bp['fliers']:
                flier.set(marker='o', markersize=5, markerfacecolor='gray', alpha=0.5)
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{title_suffix}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if ylim:
            ax.set_ylim(ylim)
        
        # Add horizontal reference lines
        if metric_name == 'correlation':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='Good (0.7)')
            ax.axhline(y=0.9, color='blue', linestyle=':', alpha=0.5, label='Excellent (0.9)')
        elif metric_name == 'r2':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5)
        elif metric_name == 'mae':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def create_detailed_boxplots(
    metrics_data: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    years: str
):
    """Create detailed per-metric boxplots."""
    channels = ['Temperature', 'Relative Humidity', 'Stem']
    colors = {'Temperature': '#2ca02c', 'Relative Humidity': '#1f77b4', 'Stem': '#d62728'}
    
    metric_configs = [
        ('correlation', 'Correlation Coefficient', (-1, 1), 'Higher is better'),
        ('mae', 'Mean Absolute Error', None, 'Lower is better'),
        ('r2', 'R² Score', (-1, 1), 'Higher is better (can be negative)')
    ]
    
    for metric_name, ylabel, ylim, description in metric_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        data = []
        labels = []
        box_colors = []
        
        for ch in channels:
            if ch in metrics_data and metric_name in metrics_data[ch]:
                values = metrics_data[ch][metric_name]
                if values:
                    data.append(values)
                    labels.append(f'{ch}\n(n={len(values)})')
                    box_colors.append(colors[ch])
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            
            # Color boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style
            for whisker in bp['whiskers']:
                whisker.set(color='black', linewidth=1.5)
            for cap in bp['caps']:
                cap.set(color='black', linewidth=1.5)
            for median in bp['medians']:
                median.set(color='white', linewidth=2)
            for flier in bp['fliers']:
                flier.set(marker='o', markersize=6, markerfacecolor='gray', alpha=0.5)
            
            # Add individual points (jittered)
            for i, d in enumerate(data):
                x = np.random.normal(i + 1, 0.04, len(d))
                ax.scatter(x, d, alpha=0.4, s=20, c='black', zorder=3)
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{ylabel} by Channel ({years})\n{description}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if ylim:
            ax.set_ylim(ylim)
        
        # Reference lines
        if metric_name in ['correlation', 'r2']:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            if metric_name == 'correlation':
                ax.axhline(y=0.7, color='darkgreen', linestyle=':', alpha=0.7, linewidth=1.5)
        
        # Add stats annotation
        stats_text = []
        for ch, d in zip(channels, data):
            if d:
                stats_text.append(f'{ch}: median={np.median(d):.3f}, mean={np.mean(d):.3f}')
        
        ax.text(0.02, 0.02, '\n'.join(stats_text), transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_path = output_dir / f'boxplot_{metric_name}_{years.replace("-", "_")}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")


def create_comparison_figure(
    metrics_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    years: str
):
    """Create a comprehensive comparison figure."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    channels = ['Temperature', 'Relative Humidity', 'Stem']
    colors = {'Temperature': '#2ca02c', 'Relative Humidity': '#1f77b4', 'Stem': '#d62728'}
    short_names = {'Temperature': 'Temperature (T)', 'Relative Humidity': 'Relative Humidity (RH)', 'Stem': 'Stem Radius'}
    
    metrics = [
        ('correlation', 'Correlation', (-1, 1)),
        ('r2', 'R²', None),
        ('mae', 'MAE', None)
    ]
    
    # Top row: One boxplot per metric (all channels together)
    for col, (metric_name, metric_label, ylim) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col])
        
        data = []
        labels = []
        box_colors = []
        
        for ch in channels:
            if ch in metrics_data and metric_name in metrics_data[ch]:
                values = metrics_data[ch][metric_name]
                if values:
                    data.append(values)
                    labels.append(ch.replace('Relative Humidity', 'RH'))
                    box_colors.append(colors[ch])
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for median in bp['medians']:
                median.set(color='black', linewidth=2)
        
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if ylim:
            ax.set_ylim(ylim)
        if metric_name in ['correlation', 'r2']:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Bottom row: Summary statistics table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Create summary table
    table_data = []
    row_labels = []
    
    for ch in channels:
        if ch in metrics_data:
            row = []
            for metric_name, _, _ in metrics:
                if metric_name in metrics_data[ch]:
                    values = metrics_data[ch][metric_name]
                    if values:
                        median = np.median(values)
                        mean = np.mean(values)
                        std = np.std(values)
                        row.append(f'{mean:.3f} ± {std:.3f}\n(med: {median:.3f})')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            n_samples = len(metrics_data[ch].get('correlation', []))
            row.append(str(n_samples))
            table_data.append(row)
            row_labels.append(short_names[ch])
    
    col_labels = ['Correlation', 'R²', 'MAE', 'N Combos']
    
    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        rowLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color row labels
    for i, ch in enumerate(channels):
        table[(i+1, -1)].set_facecolor(colors[ch])
        table[(i+1, -1)].set_text_props(color='white', fontweight='bold')
    
    # Title
    fig.suptitle(
        f'Reconstruction Evaluation Summary - Unconstrained Model ({years})\n'
        f'Test Set: {sum(len(metrics_data[ch].get("correlation", [])) for ch in channels if ch in metrics_data)} combination-evaluations',
        fontsize=14, fontweight='bold'
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create box-and-whisker plots for evaluation metrics')
    parser.add_argument('--metrics-file', type=str, 
                        default='/home/lukovic/data/treenet/test_set_evaluation_unconstrained_2021_2022/evaluation_metrics.json',
                        help='Path to metrics JSON file')
    parser.add_argument('--output-dir', type=str,
                        default='/home/lukovic/data/treenet/test_set_evaluation_unconstrained_2021_2022',
                        help='Output directory for plots')
    parser.add_argument('--years', type=str, default='2021-2022',
                        help='Years label for the plots')
    
    args = parser.parse_args()
    
    metrics_file = Path(args.metrics_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from: {metrics_file}")
    all_metrics = load_metrics(metrics_file)
    print(f"Loaded {len(all_metrics)} combinations")
    
    # Extract metrics by channel
    metrics_data = {}
    channels = ['Temperature', 'Relative Humidity', 'Stem']
    
    for ch in channels:
        metrics_data[ch] = {}
        for metric_name in ['correlation', 'mae', 'r2', 'rmse', 'normalized_mae']:
            values = extract_metrics_by_channel(all_metrics, metric_name).get(ch, [])
            if values:
                metrics_data[ch][metric_name] = values
                print(f"  {ch} - {metric_name}: {len(values)} values")
    
    # Create plots
    print("\nGenerating plots...")
    
    # 1. Combined summary figure
    create_boxplot_figure(
        metrics_data,
        f'Reconstruction Metrics - Unconstrained Model ({args.years})',
        output_dir / f'boxplot_summary_{args.years.replace("-", "_")}.png'
    )
    
    # 2. Detailed per-metric plots
    create_detailed_boxplots(metrics_data, output_dir, args.years)
    
    # 3. Comprehensive comparison
    create_comparison_figure(
        metrics_data,
        output_dir / f'evaluation_comparison_{args.years.replace("-", "_")}.png',
        args.years
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
