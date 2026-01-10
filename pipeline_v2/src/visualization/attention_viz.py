"""
Attention visualization utilities for TCN model.

This module provides tools to visualize what the attention mechanism
is focusing on in the time series data.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta


class AttentionExtractor:
    """
    Extract and visualize attention weights from a trained TCN model with attention.
    """
    
    def __init__(self, model: keras.Model):
        """
        Initialize attention extractor.
        
        Args:
            model: Trained Keras model with attention layer
        """
        self.model = model
        self.attention_layer = None
        self.attention_model = None
        
        # Find the attention layer
        self._find_attention_layer()
    
    def _find_attention_layer(self):
        """Find the MultiHeadAttention layer in the model."""
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.MultiHeadAttention):
                self.attention_layer = layer
                print(f"Found attention layer: {layer.name}")
                break
        
        if self.attention_layer is None:
            # Check nested layers
            for layer in self.model.layers:
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if isinstance(sublayer, keras.layers.MultiHeadAttention):
                            self.attention_layer = sublayer
                            print(f"Found attention layer: {sublayer.name}")
                            return
            
            raise ValueError("No MultiHeadAttention layer found in model. "
                           "Make sure the model was trained with use_attention=True")
    
    def get_attention_weights(
        self,
        X: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract attention weights for given input.
        
        Args:
            X: Input data, shape (batch, timesteps, channels)
            mask: Input mask, shape (batch, timesteps, channels)
            
        Returns:
            Attention weights, shape (batch, n_heads, seq_len, seq_len)
        """
        # Create a model that outputs attention scores
        # We need to find the intermediate tensor before attention
        
        # Get the output of the TCN blocks (before attention)
        # Find the downsampled tensor that goes into attention
        
        # Create inputs
        inputs = {
            'input_x': X.astype(np.float32),
            'input_mask': mask.astype(np.float32)
        }
        
        # Build intermediate model to get attention input
        # Find the layer just before attention (attention_downsample)
        downsample_output = None
        for layer in self.model.layers:
            if 'attention_downsample' in layer.name:
                # Create model up to this point
                downsample_output = layer.output
                break
        
        if downsample_output is None:
            raise ValueError("Could not find attention_downsample layer")
        
        # Create model to get the downsampled features
        intermediate_model = keras.Model(
            inputs=self.model.inputs,
            outputs=downsample_output
        )
        
        # Get the features that go into attention
        features = intermediate_model.predict(inputs, verbose=0)
        
        # Now manually compute attention weights
        # Get attention layer's trained weights
        query_dense = self.attention_layer._query_dense
        key_dense = self.attention_layer._key_dense
        
        # Compute queries and keys
        queries = query_dense(features)  # (batch, seq, num_heads, key_dim)
        keys = key_dense(features)       # (batch, seq, num_heads, key_dim)
        
        # Reshape for attention computation
        # queries: (batch, num_heads, seq, key_dim)
        # keys: (batch, num_heads, seq, key_dim)
        queries = tf.transpose(queries, [0, 2, 1, 3])
        keys = tf.transpose(keys, [0, 2, 1, 3])
        
        # Compute attention scores
        scale = tf.math.sqrt(tf.cast(self.attention_layer._key_dim, tf.float32))
        attention_scores = tf.matmul(queries, keys, transpose_b=True) / scale
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        return attention_weights.numpy()
    
    def visualize_attention_weights(
        self,
        attention_weights: np.ndarray,
        sample_idx: int = 0,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Visualize attention weight matrix as heatmap.
        
        Args:
            attention_weights: Attention weights, shape (batch, n_heads, seq_len, seq_len)
            sample_idx: Index of sample to visualize
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        weights = attention_weights[sample_idx]  # (n_heads, seq_len, seq_len)
        n_heads = weights.shape[0]
        seq_len = weights.shape[1]
        
        # Create figure with subplot for each head + average
        n_cols = min(n_heads, 4)
        n_rows = (n_heads + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Plot each head
        for i in range(n_heads):
            im = axes[i].imshow(weights[i], aspect='auto', cmap='viridis')
            axes[i].set_title(f'Head {i+1}')
            axes[i].set_xlabel('Key position (hours)')
            axes[i].set_ylabel('Query position (hours)')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        # Plot average across heads
        if n_heads < len(axes):
            avg_weights = weights.mean(axis=0)
            im = axes[n_heads].imshow(avg_weights, aspect='auto', cmap='viridis')
            axes[n_heads].set_title('Average (all heads)')
            axes[n_heads].set_xlabel('Key position (hours)')
            axes[n_heads].set_ylabel('Query position (hours)')
            plt.colorbar(im, ax=axes[n_heads], fraction=0.046)
        
        # Hide unused subplots
        for i in range(n_heads + 1, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Attention Weights (Hourly Resolution)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention heatmap to: {save_path}")
        
        return fig
    
    def visualize_attention_on_timeseries(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        attention_weights: np.ndarray,
        sample_idx: int = 0,
        query_position: int = 360,  # Middle of 720-hour segment
        channel_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Visualize what the attention focuses on for a specific query position.
        
        Shows the time series with attention weights overlaid as background color.
        
        Args:
            X: Input data (10-min resolution), shape (batch, 4320, n_channels)
            y: Ground truth (hourly), shape (batch, 720, 3)
            y_pred: Predictions (hourly), shape (batch, 720, 3)
            attention_weights: Attention weights, shape (batch, n_heads, 720, 720)
            sample_idx: Index of sample to visualize
            query_position: Which hour (0-719) to show attention from
            channel_names: Names of output channels
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if channel_names is None:
            channel_names = ['Local Temperature', 'Local RH', 'Stem Radius']
        
        # Get data for this sample
        y_true = y[sample_idx]
        y_p = y_pred[sample_idx]
        
        # Average attention across heads
        attn = attention_weights[sample_idx].mean(axis=0)  # (720, 720)
        
        # Get attention weights from the query position
        attn_weights_for_query = attn[query_position, :]  # (720,)
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Time axis (hours)
        hours = np.arange(720)
        
        for i, (ax, name) in enumerate(zip(axes, channel_names)):
            # Plot attention as background
            ax2 = ax.twinx()
            ax2.fill_between(hours, 0, attn_weights_for_query, 
                           alpha=0.3, color='yellow', label='Attention')
            ax2.set_ylabel('Attention', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim(0, attn_weights_for_query.max() * 1.5)
            
            # Plot ground truth and prediction
            ax.plot(hours, y_true[:, i], 'b-', label='Ground Truth', linewidth=1.5)
            ax.plot(hours, y_p[:, i], 'r--', label='Prediction', linewidth=1.5)
            
            # Mark query position
            ax.axvline(x=query_position, color='green', linestyle=':', 
                      linewidth=2, label=f'Query (hour {query_position})')
            
            ax.set_ylabel(name)
            ax.set_title(f'{name}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Hour in segment')
        
        plt.suptitle(f'Attention Focus from Hour {query_position}\n'
                    f'Yellow = where attention looks for context', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to: {save_path}")
        
        return fig
    
    def plot_attention_summary(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Create summary plots of attention patterns.
        
        Shows:
        1. Attention distance histogram (how far does attention look?)
        2. Temporal attention pattern (attention vs relative position)
        
        Args:
            attention_weights: Attention weights, shape (batch, n_heads, seq_len, seq_len)
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Average across all samples and heads
        avg_attn = attention_weights.mean(axis=(0, 1))  # (seq_len, seq_len)
        seq_len = avg_attn.shape[0]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Attention distance distribution
        ax1 = axes[0]
        distances = []
        weights = []
        for i in range(seq_len):
            for j in range(seq_len):
                distances.append(abs(i - j))
                weights.append(avg_attn[i, j])
        
        # Bin by distance
        max_dist = seq_len - 1
        bins = np.arange(0, max_dist + 1)
        binned_weights = np.zeros(len(bins))
        bin_counts = np.zeros(len(bins))
        
        for d, w in zip(distances, weights):
            binned_weights[d] += w
            bin_counts[d] += 1
        
        avg_weight_by_dist = binned_weights / np.maximum(bin_counts, 1)
        
        ax1.bar(bins[:100], avg_weight_by_dist[:100], alpha=0.7, color='steelblue')
        ax1.set_xlabel('Distance (hours)')
        ax1.set_ylabel('Average Attention Weight')
        ax1.set_title('Attention vs Distance\n(How far does attention look?)')
        ax1.axhline(y=1/seq_len, color='r', linestyle='--', 
                   label=f'Uniform ({1/seq_len:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Diagonal pattern (self-attention locality)
        ax2 = axes[1]
        
        # Extract diagonal elements at different offsets
        offsets = list(range(-50, 51))
        diag_means = []
        for offset in offsets:
            diag = np.diagonal(avg_attn, offset=offset)
            diag_means.append(diag.mean())
        
        ax2.plot(offsets, diag_means, 'b-', linewidth=2)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Same position')
        ax2.set_xlabel('Relative Position (hours)')
        ax2.set_ylabel('Average Attention Weight')
        ax2.set_title('Attention Pattern\n(Negative=past, Positive=future)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Attention Summary Statistics', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention summary to: {save_path}")
        
        return fig


def create_attention_report(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    n_samples: int = 5
):
    """
    Generate a complete attention analysis report.
    
    Args:
        model: Trained model with attention
        X_test: Test input data
        y_test: Test ground truth
        y_pred: Model predictions
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Attention Analysis Report")
    print("="*60)
    
    try:
        extractor = AttentionExtractor(model)
    except ValueError as e:
        print(f"Warning: {e}")
        print("Skipping attention visualization.")
        return
    
    # Create masks (all ones for clean data)
    masks = np.ones_like(X_test)
    
    # Get attention weights for test samples
    print("Extracting attention weights...")
    attention_weights = extractor.get_attention_weights(
        X_test[:n_samples], 
        masks[:n_samples]
    )
    
    print(f"Attention shape: {attention_weights.shape}")
    
    # 1. Save attention heatmap for first sample
    print("\nGenerating attention heatmaps...")
    extractor.visualize_attention_weights(
        attention_weights,
        sample_idx=0,
        save_path=output_dir / 'attention_heatmap.png'
    )
    plt.close()
    
    # 2. Save attention summary statistics
    print("Generating attention summary...")
    extractor.plot_attention_summary(
        attention_weights,
        save_path=output_dir / 'attention_summary.png'
    )
    plt.close()
    
    # 3. Visualize attention on time series for multiple positions
    print("Generating attention-on-timeseries visualizations...")
    query_positions = [180, 360, 540]  # Start, middle, end of segment
    
    for i in range(min(n_samples, len(X_test))):
        for q_pos in query_positions:
            extractor.visualize_attention_on_timeseries(
                X_test[i:i+1],
                y_test[i:i+1],
                y_pred[i:i+1],
                attention_weights[i:i+1],
                sample_idx=0,
                query_position=q_pos,
                save_path=output_dir / f'attention_timeseries_sample{i}_hour{q_pos}.png'
            )
            plt.close()
    
    print(f"\nAttention visualizations saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    # Test code
    print("Attention visualization module loaded successfully.")
