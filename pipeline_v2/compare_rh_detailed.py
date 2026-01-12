#!/usr/bin/env python3
"""
Detailed RH output comparison between constrained and unconstrained models.
Analyzes all test segments to get a comprehensive view.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding

# Custom loss function for loading constrained model
def constrained_hourly_loss(y_true, y_pred):
    import tensorflow as tf
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Model paths
UNCONSTRAINED_MODEL = "/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260111_152352_segment_norm_attention/best_model.keras"
CONSTRAINED_MODEL = "/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260112_190347_constrained_rh_v2_finetune_constrained_rh/best_model.keras"

DATA_DIR = "/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data"
OUTPUT_DIR = "/home/lukovic/data/treenet/rh_constraint_comparison"

def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            'TCNBlock': TCNBlock, 
            'PositionalEncoding': PositionalEncoding,
            'constrained_hourly_loss': constrained_hourly_loss
        }
    )

def load_test_data(data_dir: str):
    with open(f"{data_dir}/test_input_segments_numpy.pkl", 'rb') as f:
        X_test = pickle.load(f)
    with open(f"{data_dir}/test_output_segments_numpy.pkl", 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test

def get_all_predictions(model, X, batch_size=4):
    """Get predictions for all samples."""
    all_rh_preds = []
    
    n_batches = (len(X) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(X))
        
        batch_x = X[start:end]
        batch_mask = np.ones_like(batch_x, dtype=np.float32)
        
        pred = model.predict([batch_x, batch_mask], verbose=0)
        hourly_pred = pred[1]  # shape: (batch, 720, 3)
        
        # RH is channel 1
        rh_pred = hourly_pred[:, :, 1]  # shape: (batch, 720)
        all_rh_preds.append(rh_pred)
        
        print(f"  Batch {i+1}/{n_batches} done", end='\r')
    
    print()
    return np.concatenate(all_rh_preds, axis=0)

def main():
    print("=" * 70)
    print("Detailed RH Constraint Comparison (ALL test segments)")
    print("=" * 70)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading test data...")
    X_test, y_test = load_test_data(DATA_DIR)
    print(f"Test samples: {len(X_test)}")
    
    # Ground truth RH
    gt_rh = y_test[:, :, 1]  # (403, 720)
    
    # Load models
    print("\nLoading unconstrained model...")
    unconstrained_model = load_model(UNCONSTRAINED_MODEL)
    
    print("Loading constrained model...")
    constrained_model = load_model(CONSTRAINED_MODEL)
    
    # Get ALL predictions
    print("\nGetting unconstrained predictions (all 403 segments)...")
    unconstrained_rh = get_all_predictions(unconstrained_model, X_test)
    
    print("Getting constrained predictions (all 403 segments)...")
    constrained_rh = get_all_predictions(constrained_model, X_test)
    
    # Flatten for analysis
    gt_flat = gt_rh.flatten()
    unconstrained_flat = unconstrained_rh.flatten()
    constrained_flat = constrained_rh.flatten()
    
    total_preds = len(unconstrained_flat)
    
    # Statistics
    print("\n" + "=" * 70)
    print("COMPLETE STATISTICS (all {} predictions)".format(total_preds))
    print("=" * 70)
    
    print("\n{:<25} {:>12} {:>12} {:>12}".format("Metric", "Ground Truth", "Unconstrained", "Constrained"))
    print("-" * 65)
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format("Mean", gt_flat.mean(), unconstrained_flat.mean(), constrained_flat.mean()))
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format("Std", gt_flat.std(), unconstrained_flat.std(), constrained_flat.std()))
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format("Min", gt_flat.min(), unconstrained_flat.min(), constrained_flat.min()))
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format("Max", gt_flat.max(), unconstrained_flat.max(), constrained_flat.max()))
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format("5th percentile", np.percentile(gt_flat, 5), np.percentile(unconstrained_flat, 5), np.percentile(constrained_flat, 5)))
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format("95th percentile", np.percentile(gt_flat, 95), np.percentile(unconstrained_flat, 95), np.percentile(constrained_flat, 95)))
    
    print("\n--- Bound Violations ---")
    unc_below = (unconstrained_flat < 0).sum()
    unc_above = (unconstrained_flat > 1).sum()
    con_below = (constrained_flat < 0).sum()
    con_above = (constrained_flat > 1).sum()
    
    print("{:<25} {:>12} {:>12}".format("", "Unconstrained", "Constrained"))
    print("-" * 50)
    print("{:<25} {:>12} {:>12}".format("Below 0 (count)", unc_below, con_below))
    print("{:<25} {:>11.4f}% {:>11.4f}%".format("Below 0 (%)", 100*unc_below/total_preds, 100*con_below/total_preds))
    print("{:<25} {:>12} {:>12}".format("Above 1 (count)", unc_above, con_above))
    print("{:<25} {:>11.4f}% {:>11.4f}%".format("Above 1 (%)", 100*unc_above/total_preds, 100*con_above/total_preds))
    
    # MAE with ground truth
    unc_mae = np.abs(unconstrained_flat - gt_flat).mean()
    con_mae = np.abs(constrained_flat - gt_flat).mean()
    print("\n--- Accuracy (vs Ground Truth) ---")
    print("{:<25} {:>12.4f} {:>12.4f}".format("MAE", unc_mae, con_mae))
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Detailed RH Comparison: Ground Truth vs Unconstrained vs Constrained', fontsize=14, fontweight='bold')
    
    # Plot 1: Full distribution
    ax = axes[0, 0]
    bins = np.linspace(-0.1, 1.1, 60)
    ax.hist(gt_flat, bins=bins, alpha=0.5, label='Ground Truth', color='gray', density=True)
    ax.hist(unconstrained_flat, bins=bins, alpha=0.5, label='Unconstrained', color='blue', density=True)
    ax.hist(constrained_flat, bins=bins, alpha=0.5, label='Constrained', color='green', density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(1, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Normalized RH')
    ax.set_ylabel('Density')
    ax.set_title('Full Distribution')
    ax.legend()
    
    # Plot 2: Lower tail zoom
    ax = axes[0, 1]
    lower_mask = (unconstrained_flat < 0.15) | (constrained_flat < 0.15)
    bins = np.linspace(-0.1, 0.2, 40)
    ax.hist(unconstrained_flat[unconstrained_flat < 0.15], bins=bins, alpha=0.6, label='Unconstrained', color='blue')
    ax.hist(constrained_flat[constrained_flat < 0.15], bins=bins, alpha=0.6, label='Constrained', color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Lower bound (0)')
    ax.set_xlabel('Normalized RH')
    ax.set_ylabel('Count')
    ax.set_title('Lower Tail (< 0.15)')
    ax.legend()
    
    # Plot 3: Upper tail zoom  
    ax = axes[0, 2]
    bins = np.linspace(0.85, 1.1, 40)
    ax.hist(unconstrained_flat[unconstrained_flat > 0.85], bins=bins, alpha=0.6, label='Unconstrained', color='blue')
    ax.hist(constrained_flat[constrained_flat > 0.85], bins=bins, alpha=0.6, label='Constrained', color='green')
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Upper bound (1)')
    ax.set_xlabel('Normalized RH')
    ax.set_ylabel('Count')
    ax.set_title('Upper Tail (> 0.85)')
    ax.legend()
    
    # Plot 4: Error distribution
    ax = axes[1, 0]
    unc_error = unconstrained_flat - gt_flat
    con_error = constrained_flat - gt_flat
    bins = np.linspace(-0.3, 0.3, 60)
    ax.hist(unc_error, bins=bins, alpha=0.6, label=f'Unconstrained (MAE={unc_mae:.4f})', color='blue')
    ax.hist(con_error, bins=bins, alpha=0.6, label=f'Constrained (MAE={con_mae:.4f})', color='green')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Prediction - Ground Truth')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # Plot 5: Scatter plot (subsample)
    ax = axes[1, 1]
    idx = np.random.choice(len(gt_flat), min(10000, len(gt_flat)), replace=False)
    ax.scatter(gt_flat[idx], unconstrained_flat[idx], alpha=0.3, s=3, label='Unconstrained', color='blue')
    ax.scatter(gt_flat[idx], constrained_flat[idx], alpha=0.3, s=3, label='Constrained', color='green')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Ground Truth RH')
    ax.set_ylabel('Predicted RH')
    ax.set_title('Predictions vs Ground Truth (10k samples)')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 6: Summary bar chart
    ax = axes[1, 2]
    metrics = ['MAE', 'Below 0\n(count)', 'Above 1\n(count)']
    x = np.arange(len(metrics))
    width = 0.35
    
    unconstrained_vals = [unc_mae * 1000, unc_below, unc_above]  # Scale MAE for visibility
    constrained_vals = [con_mae * 1000, con_below, con_above]
    
    bars1 = ax.bar(x - width/2, unconstrained_vals, width, label='Unconstrained', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, constrained_vals, width, label='Constrained', color='green', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Comparison Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'rh_detailed_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")
    
    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    mae_diff = unc_mae - con_mae
    if mae_diff > 0:
        print(f"✓ Constrained model has {100*mae_diff/unc_mae:.2f}% better MAE")
    else:
        print(f"✗ Constrained model has {100*-mae_diff/unc_mae:.2f}% worse MAE")
    
    below_improvement = unc_below - con_below
    print(f"✓ Below-bound violations: {unc_below} → {con_below} (Δ = {below_improvement})")
    
    above_improvement = unc_above - con_above
    print(f"✓ Above-bound violations: {unc_above} → {con_above} (Δ = {above_improvement})")
    
    if unc_below + unc_above < 100:
        print("\n⚠️  NOTE: The unconstrained model already has very few RH violations.")
        print("   The constraint penalty may not have much effect to demonstrate.")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
