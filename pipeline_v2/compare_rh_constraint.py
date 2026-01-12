#!/usr/bin/env python3
"""
Compare RH outputs between constrained and unconstrained models.

This script evaluates whether the RH constraint is effective by comparing
the distribution of RH predictions from both models.

Usage:
    python compare_rh_constraint.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding

# Custom loss function for loading constrained model
def constrained_hourly_loss(y_true, y_pred):
    """Placeholder loss function for model loading."""
    import tensorflow as tf
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Model paths
UNCONSTRAINED_MODEL = "/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260111_152352_segment_norm_attention/best_model.keras"
CONSTRAINED_MODEL = "/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260112_190347_constrained_rh_v2_finetune_constrained_rh/best_model.keras"

# Data path
DATA_DIR = "/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data"

# Output path
OUTPUT_DIR = "/home/lukovic/data/treenet/rh_constraint_comparison"

def load_model(model_path: str) -> tf.keras.Model:
    """Load model with custom objects."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            'TCNBlock': TCNBlock, 
            'PositionalEncoding': PositionalEncoding,
            'constrained_hourly_loss': constrained_hourly_loss
        }
    )

def load_test_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data."""
    with open(f"{data_dir}/test_input_segments_numpy.pkl", 'rb') as f:
        X_test = pickle.load(f)
    with open(f"{data_dir}/test_output_segments_numpy.pkl", 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test

def evaluate_rh_bounds(
    model: tf.keras.Model,
    X: np.ndarray,
    n_samples: int = 100
) -> Dict:
    """Evaluate RH predictions and check bounds."""
    
    # Sample random segments
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    
    all_rh_preds = []
    
    for idx in indices:
        # Create mask with same shape as input (11 channels) - all valid
        sample = X[idx:idx+1]
        mask = np.ones_like(sample, dtype=np.float32)
        
        # Get prediction
        pred = model.predict([sample, mask], verbose=0)
        # pred[0] is recon output (4320, 11), pred[1] is hourly output (720, 3)
        hourly_pred = pred[1][0]
        
        # RH is channel 1 in hourly output (local_T=0, local_RH=1, stem=2)
        rh_pred = hourly_pred[:, 1]
        all_rh_preds.extend(rh_pred.tolist())
    
    all_rh_preds = np.array(all_rh_preds)
    
    # Compute statistics
    stats = {
        'mean': float(np.mean(all_rh_preds)),
        'std': float(np.std(all_rh_preds)),
        'min': float(np.min(all_rh_preds)),
        'max': float(np.max(all_rh_preds)),
        'below_0': float(np.sum(all_rh_preds < 0) / len(all_rh_preds) * 100),
        'above_1': float(np.sum(all_rh_preds > 1) / len(all_rh_preds) * 100),
        'in_bounds': float(np.sum((all_rh_preds >= 0) & (all_rh_preds <= 1)) / len(all_rh_preds) * 100),
        'n_samples': len(all_rh_preds),
        'predictions': all_rh_preds
    }
    
    return stats

def plot_comparison(unconstrained_stats: Dict, constrained_stats: Dict, output_dir: str):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RH Constraint Comparison: Unconstrained vs Constrained Model', fontsize=14, fontweight='bold')
    
    # Plot 1: Distribution of RH predictions
    ax1 = axes[0, 0]
    bins = np.linspace(-0.2, 1.3, 50)
    ax1.hist(unconstrained_stats['predictions'], bins=bins, alpha=0.6, label='Unconstrained', color='blue')
    ax1.hist(constrained_stats['predictions'], bins=bins, alpha=0.6, label='Constrained', color='green')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Bounds [0, 1]')
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=2)
    ax1.axvspan(-0.2, 0, alpha=0.2, color='red', label='Out of bounds')
    ax1.axvspan(1, 1.3, alpha=0.2, color='red')
    ax1.set_xlabel('Normalized RH Value')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of RH Predictions')
    ax1.legend()
    
    # Plot 2: Zoom on lower bound violations
    ax2 = axes[0, 1]
    bins_low = np.linspace(-0.2, 0.1, 30)
    ax2.hist(unconstrained_stats['predictions'][unconstrained_stats['predictions'] < 0.1], 
             bins=bins_low, alpha=0.6, label='Unconstrained', color='blue')
    ax2.hist(constrained_stats['predictions'][constrained_stats['predictions'] < 0.1], 
             bins=bins_low, alpha=0.6, label='Constrained', color='green')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvspan(-0.2, 0, alpha=0.3, color='red')
    ax2.set_xlabel('Normalized RH Value')
    ax2.set_ylabel('Count')
    ax2.set_title('Zoom: Lower Bound Violations (RH < 0)')
    ax2.legend()
    
    # Plot 3: Zoom on upper bound violations
    ax3 = axes[1, 0]
    bins_high = np.linspace(0.9, 1.2, 30)
    ax3.hist(unconstrained_stats['predictions'][unconstrained_stats['predictions'] > 0.9], 
             bins=bins_high, alpha=0.6, label='Unconstrained', color='blue')
    ax3.hist(constrained_stats['predictions'][constrained_stats['predictions'] > 0.9], 
             bins=bins_high, alpha=0.6, label='Constrained', color='green')
    ax3.axvline(x=1, color='red', linestyle='--', linewidth=2)
    ax3.axvspan(1, 1.2, alpha=0.3, color='red')
    ax3.set_xlabel('Normalized RH Value')
    ax3.set_ylabel('Count')
    ax3.set_title('Zoom: Upper Bound Violations (RH > 1)')
    ax3.legend()
    
    # Plot 4: Summary statistics bar chart
    ax4 = axes[1, 1]
    metrics = ['Below 0%', 'Above 100%', 'In bounds']
    x = np.arange(len(metrics))
    width = 0.35
    
    unconstrained_vals = [
        unconstrained_stats['below_0'],
        unconstrained_stats['above_1'],
        unconstrained_stats['in_bounds']
    ]
    constrained_vals = [
        constrained_stats['below_0'],
        constrained_stats['above_1'],
        constrained_stats['in_bounds']
    ]
    
    bars1 = ax4.bar(x - width/2, unconstrained_vals, width, label='Unconstrained', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, constrained_vals, width, label='Constrained', color='green', alpha=0.7)
    
    ax4.set_ylabel('Percentage of predictions')
    ax4.set_title('Bound Violation Summary')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'rh_constraint_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def print_summary(unconstrained_stats: Dict, constrained_stats: Dict):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("RH CONSTRAINT COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n{:<30} {:>15} {:>15}".format("Metric", "Unconstrained", "Constrained"))
    print("-" * 60)
    print("{:<30} {:>15.4f} {:>15.4f}".format("Mean RH", unconstrained_stats['mean'], constrained_stats['mean']))
    print("{:<30} {:>15.4f} {:>15.4f}".format("Std RH", unconstrained_stats['std'], constrained_stats['std']))
    print("{:<30} {:>15.4f} {:>15.4f}".format("Min RH", unconstrained_stats['min'], constrained_stats['min']))
    print("{:<30} {:>15.4f} {:>15.4f}".format("Max RH", unconstrained_stats['max'], constrained_stats['max']))
    print("-" * 60)
    print("{:<30} {:>14.2f}% {:>14.2f}%".format("Below 0 (invalid)", unconstrained_stats['below_0'], constrained_stats['below_0']))
    print("{:<30} {:>14.2f}% {:>14.2f}%".format("Above 1 (invalid)", unconstrained_stats['above_1'], constrained_stats['above_1']))
    print("{:<30} {:>14.2f}% {:>14.2f}%".format("In bounds [0,1]", unconstrained_stats['in_bounds'], constrained_stats['in_bounds']))
    print("-" * 60)
    print("{:<30} {:>15d} {:>15d}".format("Total predictions", unconstrained_stats['n_samples'], constrained_stats['n_samples']))
    print("=" * 70)


def main():
    print("=" * 70)
    print("RH Constraint Comparison: Unconstrained vs Constrained Model")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data(DATA_DIR)
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {y_test.shape}")
    
    # Load models
    print("\nLoading unconstrained model...")
    unconstrained_model = load_model(UNCONSTRAINED_MODEL)
    
    print("Loading constrained model...")
    constrained_model = load_model(CONSTRAINED_MODEL)
    
    # Evaluate both models
    n_samples = 200  # Number of test segments to evaluate
    
    print(f"\nEvaluating unconstrained model ({n_samples} segments)...")
    unconstrained_stats = evaluate_rh_bounds(unconstrained_model, X_test, n_samples)
    
    print(f"Evaluating constrained model ({n_samples} segments)...")
    constrained_stats = evaluate_rh_bounds(constrained_model, X_test, n_samples)
    
    # Print summary
    print_summary(unconstrained_stats, constrained_stats)
    
    # Create visualization
    print("\nGenerating comparison plot...")
    plot_path = plot_comparison(unconstrained_stats, constrained_stats, OUTPUT_DIR)
    
    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    
    improvement = unconstrained_stats['above_1'] - constrained_stats['above_1']
    if improvement > 0:
        print(f"✓ Upper bound violations reduced by {improvement:.2f}%")
    else:
        print(f"✗ Upper bound violations increased by {-improvement:.2f}%")
    
    improvement = unconstrained_stats['below_0'] - constrained_stats['below_0']
    if improvement > 0:
        print(f"✓ Lower bound violations reduced by {improvement:.2f}%")
    else:
        print(f"✗ Lower bound violations increased by {-improvement:.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
