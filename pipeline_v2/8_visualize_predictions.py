#!/usr/bin/env python3
"""
CLI for visualizing model predictions compared to input and ground truth.

Creates plots to validate model performance:
- Individual segment comparisons (input, prediction, ground truth)
- Per-channel error analysis
- Zoomed views for detailed inspection

Usage:
    # Visualize specific number of samples from an experiment
    python 8_visualize_predictions.py \
        --experiment-dir /path/to/experiment \
        --data-dir /path/to/data \
        --n-samples 5
    
    # Create zoomed views
    python 8_visualize_predictions.py \
        --experiment-dir /path/to/experiment \
        --data-dir /path/to/data \
        --n-samples 3 \
        --zoom-days 7
    
    # Only specific channels
    python 8_visualize_predictions.py \
        --experiment-dir /path/to/experiment \
        --data-dir /path/to/data \
        --channels stem local_T
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pickle
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging


def setup_logging(name: str = 'visualization'):
    """Setup logging and return logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize model predictions vs input and ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required paths
    parser.add_argument(
        '--experiment-dir',
        type=Path,
        required=True,
        help='Path to experiment directory (contains best_model.keras)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Path to data directory (default: parent of experiment-dir)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for plots (default: experiment-dir/visualizations)'
    )
    
    # Sampling options
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'test'],
        help='Dataset split to use (default: test)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample selection (default: 42)'
    )
    
    # View options
    parser.add_argument(
        '--zoom-days',
        type=int,
        help='Create additional zoomed view of first N days'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        default=['local_T', 'local_RH', 'stem'],
        help='Channels to visualize (default: all 3)'
    )
    
    # Plot options
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[16, 10],
        help='Figure size in inches (default: 16 10)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI (default: 150)'
    )
    parser.add_argument(
        '--no-error-band',
        action='store_true',
        help='Do not show error band around predictions'
    )
    
    return parser.parse_args()


class PredictionPlotter:
    """
    Plot model predictions compared to input and ground truth.
    
    Creates comparison visualizations showing:
    - Input signal (raw, aggregated to hourly)
    - Ground truth (cleaned hourly targets)
    - Model predictions
    - Per-channel error metrics
    """
    
    # Channel configuration
    CHANNEL_CONFIG = {
        'local_T': {
            'name': 'Local Temperature',
            'input_col': 'temp_treenet',
            'color_input': '#2ecc71',      # Green
            'color_truth': '#3498db',       # Blue
            'color_pred': '#e74c3c',        # Red
        },
        'local_RH': {
            'name': 'Local Humidity',
            'input_col': 'rh_treenet',
            'color_input': '#27ae60',       # Darker green
            'color_truth': '#2980b9',       # Darker blue
            'color_pred': '#c0392b',        # Darker red
        },
        'stem': {
            'name': 'Stem Radius',
            'input_col': 'stem',
            'color_input': '#1abc9c',       # Teal
            'color_truth': '#9b59b6',       # Purple
            'color_pred': '#e67e22',        # Orange
        }
    }
    
    def __init__(
        self,
        model_path: Path,
        local_tz: str = 'Europe/Zurich',
        figsize: Tuple[float, float] = (16, 10),
        dpi: int = 150
    ):
        """
        Initialize plotter.
        
        Args:
            model_path: Path to trained model (best_model.keras)
            local_tz: Local timezone for date displays
            figsize: Default figure size
            dpi: Default DPI
        """
        self.model_path = Path(model_path)
        self.local_tz = local_tz
        self.figsize = figsize
        self.dpi = dpi
        self.model = None
        
    def load_model(self):
        """Load the trained model."""
        import tensorflow as tf
        
        # Import custom layers for deserialization
        # Need to add the parent directory to ensure proper module resolution
        project_dir = Path(__file__).parent
        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))
        
        # Direct import of the TCN module
        from src.models.tcn import TCNBlock, PositionalEncoding
        
        if self.model is None:
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    'TCNBlock': TCNBlock,
                    'PositionalEncoding': PositionalEncoding
                }
            )
            print(f"  Model loaded successfully")
            
        return self.model
    
    def load_data(
        self,
        data_dir: Path,
        split: str = 'test'
    ) -> tuple:
        """
        Load segment data from processed files.
        
        Args:
            data_dir: Directory with processed segment files (or parent dir with model_data/)
            split: 'train' or 'test'
            
        Returns:
            Tuple of (combo_ids, input_segments, output_segments, segment_metadata)
        """
        base_path = Path(data_dir)
        
        # Check if model_data subdirectory exists
        if (base_path / 'model_data').exists():
            base_path = base_path / 'model_data'
        
        with open(base_path / f'model_{split}_data_combination_ids.pkl', 'rb') as f:
            combo_ids = pickle.load(f)
        
        with open(base_path / f'{split}_input_segments.pkl', 'rb') as f:
            input_segments = pickle.load(f)
        
        with open(base_path / f'{split}_output_segments.pkl', 'rb') as f:
            output_segments = pickle.load(f)
        
        with open(base_path / f'{split}_segment_ids.pkl', 'rb') as f:
            segment_metadata = pickle.load(f)
        
        return combo_ids, input_segments, output_segments, segment_metadata
    
    def prepare_samples(
        self,
        data_dir: Path,
        split: str = 'test',
        n_samples: int = 5,
        seed: int = 42
    ) -> List[dict]:
        """
        Prepare random samples for visualization.
        
        Args:
            data_dir: Path to data directory
            split: Dataset split
            n_samples: Number of samples to prepare
            seed: Random seed
            
        Returns:
            List of sample dictionaries with input, output, metadata
        """
        import pandas as pd
        np.random.seed(seed)
        
        print(f"Loading {split} data from: {data_dir}")
        combo_ids, input_segs, output_segs, seg_meta = self.load_data(data_dir, split)
        
        # Build flat list of all segments
        all_segments = []
        for combo_id, ids_row in combo_ids.items():
            if combo_id not in input_segs:
                continue
            
            for seg_idx, (input_df, output_df) in enumerate(zip(
                input_segs[combo_id], 
                output_segs[combo_id]
            )):
                if len(input_df) == 0 or len(output_df) == 0:
                    continue
                    
                all_segments.append({
                    'combo_id': combo_id,
                    'segment_idx': seg_idx,
                    'site_id': ids_row['site ID'],
                    'input_df': input_df,
                    'output_df': output_df,
                    'ids_row': ids_row
                })
        
        print(f"  Found {len(all_segments)} total segments")
        
        # Random sample
        n_samples = min(n_samples, len(all_segments))
        sample_indices = np.random.choice(len(all_segments), n_samples, replace=False)
        samples = [all_segments[i] for i in sample_indices]
        
        print(f"  Selected {len(samples)} samples")
        
        return samples
    
    def aggregate_input_to_hourly(
        self,
        input_df,
        method: str = 'mean'
    ):
        """
        Aggregate 10-min input data to hourly for comparison.
        
        Args:
            input_df: Input DataFrame at 10-min resolution
            method: Aggregation method ('mean', 'median', 'first')
            
        Returns:
            Hourly aggregated DataFrame
        """
        import pandas as pd
        
        # Resample to hourly
        if method == 'mean':
            hourly = input_df.resample('1h').mean()
        elif method == 'median':
            hourly = input_df.resample('1h').median()
        elif method == 'first':
            hourly = input_df.resample('1h').first()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Remove NaN rows
        hourly = hourly.dropna(how='all')
        
        return hourly
    
    def plot_sample(
        self,
        sample: dict,
        prediction: np.ndarray,
        output_path: Path,
        channels: List[str] = ['local_T', 'local_RH', 'stem'],
        show_error_band: bool = True,
        zoom_hours: Optional[int] = None,
        title_suffix: str = ''
    ):
        """
        Plot a single sample comparison.
        
        Args:
            sample: Sample dictionary with input_df, output_df, metadata
            prediction: Model prediction array (timesteps, channels)
            output_path: Path to save figure
            channels: Channels to plot
            show_error_band: Whether to show error band
            zoom_hours: If set, only show first N hours
            title_suffix: Additional text for title
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        
        input_df = sample['input_df']
        output_df = sample['output_df']
        
        # Aggregate input to hourly
        input_hourly = self.aggregate_input_to_hourly(input_df)
        
        # Create figure with subplots for each channel
        n_channels = len(channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(self.figsize[0], self.figsize[1] * n_channels / 3))
        
        if n_channels == 1:
            axes = [axes]
        
        # Get common time axis
        time_axis = output_df.index
        if zoom_hours:
            time_axis = time_axis[:zoom_hours]
        
        channel_metrics = {}
        
        for ax_idx, channel in enumerate(channels):
            ax = axes[ax_idx]
            config = self.CHANNEL_CONFIG[channel]
            
            # Get data
            input_col = config['input_col']
            
            # Input (aggregated)
            if input_col in input_hourly.columns:
                input_data = input_hourly[input_col]
                if zoom_hours:
                    input_data = input_data.iloc[:zoom_hours] if len(input_data) >= zoom_hours else input_data
                ax.plot(
                    input_data.index[:len(time_axis)],
                    input_data.values[:len(time_axis)],
                    label='Input (raw)',
                    color=config['color_input'],
                    linewidth=1.5,
                    alpha=0.7
                )
            
            # Ground truth
            if channel in output_df.columns:
                truth_data = output_df[channel]
                if zoom_hours:
                    truth_data = truth_data.iloc[:zoom_hours]
                ax.plot(
                    time_axis[:len(truth_data)],
                    truth_data.values,
                    label='Ground Truth',
                    color=config['color_truth'],
                    linewidth=2.0,
                    alpha=0.9
                )
            
            # Prediction
            channel_idx = channels.index(channel)
            if channel_idx < prediction.shape[-1]:
                pred_data = prediction[:, channel_idx]
                if zoom_hours:
                    pred_data = pred_data[:zoom_hours]
                
                ax.plot(
                    time_axis[:len(pred_data)],
                    pred_data,
                    label='Prediction',
                    color=config['color_pred'],
                    linewidth=2.0,
                    linestyle='--',
                    alpha=0.9
                )
                
                # Error band
                if show_error_band and channel in output_df.columns:
                    truth_vals = output_df[channel].values[:len(pred_data)]
                    if zoom_hours:
                        truth_vals = truth_vals[:zoom_hours]
                    
                    error = np.abs(pred_data[:len(truth_vals)] - truth_vals)
                    ax.fill_between(
                        time_axis[:len(error)],
                        pred_data[:len(error)] - error,
                        pred_data[:len(error)] + error,
                        color=config['color_pred'],
                        alpha=0.15,
                        label='_nolegend_'
                    )
                
                # Compute metrics
                if channel in output_df.columns:
                    truth_vals = output_df[channel].values
                    pred_vals = pred_data[:len(truth_vals)]
                    
                    mse = np.mean((pred_vals - truth_vals[:len(pred_vals)])**2)
                    mae = np.mean(np.abs(pred_vals - truth_vals[:len(pred_vals)]))
                    
                    channel_metrics[channel] = {'MSE': mse, 'MAE': mae}
            
            # Formatting
            ax.set_ylabel(config['name'], fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            # Add metrics text
            if channel in channel_metrics:
                metrics = channel_metrics[channel]
                metrics_text = f"MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f}"
                ax.text(
                    0.02, 0.95, metrics_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            # X-axis formatting
            if ax_idx == len(channels) - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                ax.set_xlabel('Date', fontsize=11)
            else:
                ax.set_xticklabels([])
        
        # Title
        start_date = input_df.index[0].strftime('%Y-%m-%d')
        end_date = input_df.index[-1].strftime('%Y-%m-%d')
        
        title = (
            f"Site {sample['site_id']} | Combo {sample['combo_id']} | Segment {sample['segment_idx']}\n"
            f"{start_date} → {end_date}{title_suffix}"
        )
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return channel_metrics
    
    def visualize_samples(
        self,
        data_dir: Path,
        output_dir: Path,
        split: str = 'test',
        n_samples: int = 5,
        seed: int = 42,
        channels: List[str] = ['local_T', 'local_RH', 'stem'],
        zoom_days: Optional[int] = None,
        show_error_band: bool = True
    ) -> dict:
        """
        Visualize multiple samples.
        
        Args:
            data_dir: Path to data directory
            output_dir: Output directory for plots
            split: Dataset split
            n_samples: Number of samples
            seed: Random seed
            channels: Channels to plot
            zoom_days: Create zoomed view of first N days
            show_error_band: Whether to show error band
            
        Returns:
            Dictionary with metrics for all samples
        """
        import tensorflow as tf
        import pandas as pd
        
        # Load model
        model = self.load_model()
        
        # Prepare samples
        samples = self.prepare_samples(data_dir, split, n_samples, seed)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_metrics = {}
        
        for i, sample in enumerate(samples):
            print(f"\nProcessing sample {i+1}/{len(samples)}...")
            print(f"  Site: {sample['site_id']}, Combo: {sample['combo_id']}, Segment: {sample['segment_idx']}")
            
            # Prepare model input
            # Model expects (input_x, input_mask) - two inputs
            input_cols = [
                'temp_treenet', 'rh_treenet', 'stem', 
                'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy'
            ]
            input_array = sample['input_df'][input_cols].values
            input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension
            
            # Create mask (1 = valid data, 0 = missing/gap)
            # For visualization with clean test data, mask is all 1s
            mask_array = np.ones_like(input_array)
            
            # Get prediction
            # Model has 2 outputs: recon_output (4320, 11) and hourly_output (720, 3)
            predictions = model.predict([input_array, mask_array], verbose=0)
            
            # Extract hourly output (the one we compare with targets)
            # predictions is [recon_output, hourly_output]
            if isinstance(predictions, list):
                prediction = predictions[1][0]  # hourly_output, remove batch dim
            else:
                prediction = predictions[0]  # Single output, remove batch dim
            # Full view
            output_path = output_dir / f'prediction_sample_{i}.png'
            metrics = self.plot_sample(
                sample=sample,
                prediction=prediction,
                output_path=output_path,
                channels=channels,
                show_error_band=show_error_band
            )
            print(f"  Saved: {output_path}")
            
            all_metrics[f'sample_{i}'] = {
                'site_id': sample['site_id'],
                'combo_id': sample['combo_id'],
                'segment_idx': sample['segment_idx'],
                'metrics': metrics
            }
            
            # Zoomed view
            if zoom_days:
                zoom_hours = zoom_days * 24
                output_path_zoom = output_dir / f'prediction_sample_{i}_zoomed.png'
                self.plot_sample(
                    sample=sample,
                    prediction=prediction,
                    output_path=output_path_zoom,
                    channels=channels,
                    show_error_band=show_error_band,
                    zoom_hours=zoom_hours,
                    title_suffix=f' (First {zoom_days} days)'
                )
                print(f"  Saved: {output_path_zoom}")
        
        # Save metrics summary
        metrics_path = output_dir / 'prediction_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy floats for JSON serialization
            serializable_metrics = {}
            for key, val in all_metrics.items():
                serializable_metrics[key] = {
                    'site_id': int(val['site_id']),
                    'combo_id': int(val['combo_id']),
                    'segment_idx': int(val['segment_idx']),
                    'metrics': {
                        ch: {m: float(v) for m, v in ch_metrics.items()}
                        for ch, ch_metrics in val['metrics'].items()
                    }
                }
            json.dump(serializable_metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return all_metrics


def main():
    """Main visualization workflow."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging('visualization')
    logger.info("Starting prediction visualization")
    
    # Validate paths
    if not args.experiment_dir.exists():
        logger.error(f"Experiment directory does not exist: {args.experiment_dir}")
        sys.exit(1)
    
    model_path = args.experiment_dir / 'best_model.keras'
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Assume data_dir is parent of experiments/
        data_dir = args.experiment_dir.parent.parent
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Determine output directory
    # Default: /home/lukovic/data/treenet/visualizations/predictions/<experiment_name>/
    DEFAULT_OUTPUT_ROOT = Path('/home/lukovic/data/treenet')
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        experiment_name = args.experiment_dir.name
        output_dir = DEFAULT_OUTPUT_ROOT / 'visualizations' / 'predictions' / experiment_name
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Initialize plotter
    plotter = PredictionPlotter(
        model_path=model_path,
        figsize=tuple(args.figsize),
        dpi=args.dpi
    )
    
    # Visualize
    all_metrics = plotter.visualize_samples(
        data_dir=data_dir,
        output_dir=output_dir,
        split=args.split,
        n_samples=args.n_samples,
        seed=args.seed,
        channels=args.channels,
        zoom_days=args.zoom_days,
        show_error_band=not args.no_error_band
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for channel in args.channels:
        mse_values = []
        mae_values = []
        for sample_data in all_metrics.values():
            if channel in sample_data['metrics']:
                mse_values.append(sample_data['metrics'][channel]['MSE'])
                mae_values.append(sample_data['metrics'][channel]['MAE'])
        
        if mse_values:
            print(f"\n{channel}:")
            print(f"  MSE: {np.mean(mse_values):.4f} ± {np.std(mse_values):.4f}")
            print(f"  MAE: {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
    
    logger.info("Visualization complete!")


if __name__ == '__main__':
    main()
