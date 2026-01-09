#!/usr/bin/env python3
"""
Analyze gap filling results by visualizing original segments, 
gaps injected, and model predictions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.gaps.gap_injection import GapInjector
from src.config import GapConfig
import tensorflow as tf


def load_model_and_data(model_path, data_dir):
    """Load trained model and test data."""
    # Import custom layers for model loading
    from src.models.tcn import TCNBlock
    
    # Load model with custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'TCNBlock': TCNBlock}
    )
    
    # Load test data
    test_input = pickle.load(open(f"{data_dir}/test_input_segments_numpy.pkl", "rb"))
    test_output = pickle.load(open(f"{data_dir}/test_output_segments_numpy.pkl", "rb"))
    
    return model, test_input, test_output


def find_gap_regions(mask_1d):
    """
    Find contiguous gap regions from a 1D mask.
    
    Args:
        mask_1d: 1D array where 0 = gap, 1 = valid
        
    Returns:
        List of (start, end) tuples for each gap region
    """
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, val in enumerate(mask_1d):
        if val == 0 and not in_gap:
            # Start of new gap
            in_gap = True
            gap_start = i
        elif val == 1 and in_gap:
            # End of gap
            gaps.append((gap_start, i))
            in_gap = False
    
    # Handle gap at end of array
    if in_gap:
        gaps.append((gap_start, len(mask_1d)))
    
    return gaps


def add_gap_shading(ax, mask_1d, y_min=None, y_max=None, alpha=0.3, color='red', label_first=True):
    """
    Add shaded regions to highlight gaps on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis
        mask_1d: 1D mask where 0 = gap, 1 = valid
        y_min, y_max: Y-axis limits for shading (auto-detected if None)
        alpha: Transparency of shading
        color: Color for gap regions
        label_first: Whether to add label to legend for first gap
    """
    gaps = find_gap_regions(mask_1d)
    
    for i, (start, end) in enumerate(gaps):
        label = 'Gap region' if (i == 0 and label_first) else None
        ax.axvspan(start, end, alpha=alpha, color=color, label=label)


# Note: inject_gaps_for_visualization is no longer needed
# The main GapInjector now only gaps channels 0-2 by default


def visualize_gap_filling(model, X_test, y_test, n_samples=3, save_dir='./gap_analysis'):
    """
    Visualize gap injection and filling for sample segments.
    
    Args:
        model: Trained model
        X_test: Test input data (n_samples, 4320, 11)
        y_test: Test output data (n_samples, 720, 3)
        n_samples: Number of samples to visualize
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Create gap injector (only gaps first 3 channels by default - see gap_injection.py)
    gap_config = GapConfig()
    injector = GapInjector(
        min_gap_days=gap_config.min_gap_days,
        max_gap_days=gap_config.max_gap_days,
        min_gaps_per_segment=gap_config.min_gaps_per_segment,
        max_gaps_per_segment=gap_config.max_gaps_per_segment,
        gap_channel_prob=gap_config.gap_channel_prob,
        random_seed=42
        # gappable_channels defaults to [0, 1, 2] - local sensor channels only
    )
    
    # Channel names
    input_channels = ['temp_treenet', 'rh_treenet', 'stem', 
                     'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy']
    output_channels = ['local_T', 'local_RH', 'stem']
    
    # Select random samples
    np.random.seed(123)  # Different seed to get diverse samples
    indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    
    for idx, sample_idx in enumerate(indices):
        X_orig = X_test[sample_idx:sample_idx+1]
        y_true = y_test[sample_idx:sample_idx+1]
        
        # Inject gaps (now only in channels 0-2 by default)
        X_gapped, mask = injector.inject_gaps_batch(X_orig)
        
        # Get predictions
        predictions = model.predict({'input_x': X_gapped, 'input_mask': mask}, verbose=0)
        recon_pred = predictions[0]  # First output: reconstruction
        hourly_pred = predictions[1]  # Second output: hourly predictions
        
        # Compute combined gap mask for all channels (for hourly plot)
        # A timestep is considered "gapped" if ANY of the first 3 channels has a gap
        combined_mask_10min = np.min(mask[0, :, :3], axis=1)  # 0 if any channel has gap
        
        # Downsample mask to hourly resolution (for hourly prediction plots)
        # 10-min data: 6 timesteps per hour
        n_hourly = 720
        timesteps_per_hour = 6
        combined_mask_hourly = np.zeros(n_hourly)
        for h in range(n_hourly):
            start_idx = h * timesteps_per_hour
            end_idx = start_idx + timesteps_per_hour
            # If any 10-min slot in this hour has a gap, mark hour as gapped
            combined_mask_hourly[h] = np.min(combined_mask_10min[start_idx:end_idx])
        
        # Plot input reconstruction with gap shading
        fig, axes = plt.subplots(4, 3, figsize=(15, 14))
        fig.suptitle(f'Sample {idx+1}: Input Gap Filling (First 3 channels)\nRed shading = Gap regions (synthetically injected)', fontsize=16)
        
        for ch_idx in range(min(3, len(input_channels))):
            ch_mask = mask[0, :, ch_idx]
            
            # Original signal with gap shading
            axes[0, ch_idx].plot(X_orig[0, :, ch_idx], label='Original', linewidth=1, color='blue')
            add_gap_shading(axes[0, ch_idx], ch_mask, alpha=0.25, color='red', label_first=(ch_idx==0))
            axes[0, ch_idx].set_title(f'{input_channels[ch_idx]} - Original')
            axes[0, ch_idx].set_ylabel('Value')
            axes[0, ch_idx].grid(True, alpha=0.3)
            axes[0, ch_idx].legend(loc='upper right')
            
            # Gapped signal with gap shading
            x_plot = X_gapped[0, :, ch_idx].copy()
            x_plot[ch_mask == 0] = np.nan
            axes[1, ch_idx].plot(x_plot, label='With gaps', linewidth=1, color='orange')
            add_gap_shading(axes[1, ch_idx], ch_mask, alpha=0.25, color='red', label_first=False)
            axes[1, ch_idx].set_title(f'{input_channels[ch_idx]} - Gapped')
            axes[1, ch_idx].set_ylabel('Value')
            axes[1, ch_idx].grid(True, alpha=0.3)
            axes[1, ch_idx].legend(loc='upper right')
            
            # Reconstructed signal with gap shading
            axes[2, ch_idx].plot(X_orig[0, :, ch_idx], label='Original', linewidth=1, alpha=0.7, color='blue')
            axes[2, ch_idx].plot(recon_pred[0, :, ch_idx], label='Reconstructed', linewidth=1, color='green')
            add_gap_shading(axes[2, ch_idx], ch_mask, alpha=0.25, color='red', label_first=False)
            axes[2, ch_idx].set_title(f'{input_channels[ch_idx]} - Reconstructed')
            axes[2, ch_idx].set_ylabel('Value')
            axes[2, ch_idx].grid(True, alpha=0.3)
            axes[2, ch_idx].legend(loc='upper right')
            
            # Error (only at gap locations) with gap shading
            error = np.abs(recon_pred[0, :, ch_idx] - X_orig[0, :, ch_idx])
            gap_error = error.copy()
            gap_error[ch_mask == 1] = np.nan
            axes[3, ch_idx].plot(gap_error, label='Gap filling error', linewidth=1, color='red')
            add_gap_shading(axes[3, ch_idx], ch_mask, alpha=0.15, color='gray', label_first=False)
            axes[3, ch_idx].set_title(f'{input_channels[ch_idx]} - Gap Errors')
            axes[3, ch_idx].set_xlabel('Timestep (10-min)')
            axes[3, ch_idx].set_ylabel('MAE')
            axes[3, ch_idx].grid(True, alpha=0.3)
            axes[3, ch_idx].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{idx+1}_input_reconstruction.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot hourly predictions with gap shading
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f'Sample {idx+1}: Hourly Predictions\nRed shading = Hours where input had gaps (model predicts from incomplete data)', fontsize=16)
        
        for ch_idx, ch_name in enumerate(output_channels):
            axes[ch_idx].plot(y_true[0, :, ch_idx], label='True', linewidth=2, alpha=0.8, color='blue')
            axes[ch_idx].plot(hourly_pred[0, :, ch_idx], label='Predicted', linewidth=2, alpha=0.8, color='green')
            # Add gap shading to show where input data had gaps
            add_gap_shading(axes[ch_idx], combined_mask_hourly, alpha=0.25, color='red', label_first=(ch_idx==0))
            axes[ch_idx].set_title(f'{ch_name}')
            axes[ch_idx].set_ylabel('Value')
            axes[ch_idx].set_xlabel('Hour')
            axes[ch_idx].grid(True, alpha=0.3)
            axes[ch_idx].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{idx+1}_hourly_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Compute statistics
        print(f"\nSample {idx+1} Statistics:")
        print(f"{'='*60}")
        
        # Gap statistics
        gap_fraction = 1 - mask[0].mean()
        print(f"Gap fraction: {gap_fraction*100:.2f}%")
        
        # Input reconstruction MAE
        print("\nInput Reconstruction MAE (at gaps only):")
        for ch_idx, ch_name in enumerate(input_channels[:3]):
            gap_mask = mask[0, :, ch_idx] == 0
            if gap_mask.sum() > 0:
                mae = np.abs(recon_pred[0, gap_mask, ch_idx] - X_orig[0, gap_mask, ch_idx]).mean()
                print(f"  {ch_name:15s}: {mae:.4f}")
        
        # Hourly prediction MAE
        print("\nHourly Prediction MAE:")
        for ch_idx, ch_name in enumerate(output_channels):
            mae = np.abs(hourly_pred[0, :, ch_idx] - y_true[0, :, ch_idx]).mean()
            print(f"  {ch_name:15s}: {mae:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Plots saved to: {save_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze gap filling')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-dir', required=True, help='Path to data directory')
    parser.add_argument('--n-samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--output-dir', default='./gap_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    print("Loading model and data...")
    model, X_test, y_test = load_model_and_data(args.model_path, args.data_dir)
    
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    print(f"\nGenerating {args.n_samples} sample visualizations...")
    visualize_gap_filling(model, X_test, y_test, 
                         n_samples=args.n_samples,
                         save_dir=args.output_dir)


if __name__ == '__main__':
    main()
