#!/usr/bin/env python3
"""
Train TCN model with constrained loss function for physical bounds.

This script adds a penalty-based constraint to prevent RH from exceeding 100%.
The constraint is applied as a soft penalty on the hourly output.

Usage:
    python 16_train_constrained.py --data-dir /path/to/model_data \
        --base-model /path/to/best_model.keras \
        --epochs 30 --rh-penalty-weight 0.1

Author: Lukovic
Date: 2026-01-12
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding
from src.models.training import DataGenerator
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TCN model with physical constraints'
    )
    
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Directory with processed segment files'
    )
    parser.add_argument(
        '--base-model', type=str, required=True,
        help='Path to pre-trained model to fine-tune'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory for experiment outputs'
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (early stopping will end if no improvement)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate (lower for fine-tuning)')
    
    # Constraint
    parser.add_argument('--rh-penalty-weight', type=float, default=0.1,
                        help='Weight for RH boundary penalty')
    
    # Gap injection
    parser.add_argument('--min-gap-days', type=int, default=1, help='Min gap length')
    parser.add_argument('--max-gap-days', type=int, default=12, help='Max gap length')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def create_constrained_loss(rh_penalty_weight: float = 0.1):
    """
    Create a loss function with physical boundary constraints on RH.
    
    The RH channel (index 1) is constrained to [0, 1] in normalized space.
    Values outside this range incur a quadratic penalty.
    
    Args:
        rh_penalty_weight: Weight for the boundary penalty term
        
    Returns:
        Custom loss function
    """
    
    def constrained_hourly_loss(y_true, y_pred):
        """
        MSE loss with penalty for RH exceeding physical bounds.
        
        Args:
            y_true: Ground truth (batch, time, channels)
            y_pred: Predictions (batch, time, channels)
            
        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Extract RH channel (index 1 in output: [local_T, local_RH, stem])
        rh_pred = y_pred[:, :, 1]
        
        # Penalty for values outside [0, 1]
        # Upper bound violation (> 1.0)
        upper_violation = tf.maximum(0.0, rh_pred - 1.0)
        # Lower bound violation (< 0.0)  
        lower_violation = tf.maximum(0.0, -rh_pred)
        
        # Quadratic penalty
        rh_penalty = tf.reduce_mean(
            tf.square(upper_violation) + tf.square(lower_violation)
        )
        
        # Combined loss
        total_loss = mse_loss + rh_penalty_weight * rh_penalty
        
        return total_loss
    
    return constrained_hourly_loss


def main():
    """Main training function."""
    args = parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / f'{timestamp}_constrained_rh'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = exp_dir / 'training.log'
    log = setup_logging(log_file=log_file, name='train_constrained', verbose=args.verbose)
    
    log.info("=" * 70)
    log.info("TreeNet AI - Constrained Training for Physical Bounds")
    log.info("=" * 70)
    log.info(f"Base model: {args.base_model}")
    log.info(f"Output dir: {exp_dir}")
    log.info(f"RH penalty weight: {args.rh_penalty_weight}")
    
    # Load base model
    log.info("\nLoading base model...")
    model = keras.models.load_model(
        args.base_model,
        custom_objects={
            'TCNBlock': TCNBlock,
            'PositionalEncoding': PositionalEncoding
        }
    )
    log.info("Base model loaded successfully")
    
    # Create constrained loss function
    constrained_loss = create_constrained_loss(args.rh_penalty_weight)
    
    # Recompile model with constrained loss
    log.info("\nRecompiling model with constrained loss...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            'recon_output': 'mse',
            'hourly_output': constrained_loss
        },
        loss_weights={
            'recon_output': 1.0,
            'hourly_output': 1.0
        },
        metrics={
            'recon_output': ['mae'],
            'hourly_output': ['mae']
        }
    )
    
    # Load training data
    log.info("\nLoading training data...")
    data_dir = Path(args.data_dir)
    
    import pickle
    with open(data_dir / 'train_input_segments_numpy.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir / 'train_output_segments_numpy.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_val = pickle.load(f)
    
    log.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    log.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    
    # Create gap injector for training augmentation
    from src.gaps.gap_injection import GapInjector
    gap_injector = GapInjector(
        min_gap_days=args.min_gap_days,
        max_gap_days=args.max_gap_days
    )
    
    # Create data generators
    log.info("\nSetting up data generators...")
    train_gen = DataGenerator(
        X=X_train,
        y=y_train,
        batch_size=args.batch_size,
        gap_injector=gap_injector,
        shuffle=True
    )
    
    val_gen = DataGenerator(
        X=X_val,
        y=y_val,
        batch_size=args.batch_size,
        gap_injector=None,  # No gaps for validation
        shuffle=False
    )
    
    log.info(f"Training batches: {len(train_gen)}")
    log.info(f"Validation batches: {len(val_gen)}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_hourly_output_mae',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='min'  # MAE should be minimized
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_hourly_output_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(exp_dir / 'best_model.keras'),
            monitor='val_hourly_output_mae',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(exp_dir / 'training_history.csv')
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(exp_dir / 'tensorboard'),
            histogram_freq=1
        )
    ]
    
    # Train
    log.info(f"\nStarting training for {args.epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = exp_dir / 'final_model.keras'
    model.save(str(final_model_path))
    log.info(f"Final model saved to: {final_model_path}")
    
    # Save config
    config = {
        'base_model': args.base_model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'rh_penalty_weight': args.rh_penalty_weight,
        'min_gap_days': args.min_gap_days,
        'max_gap_days': args.max_gap_days,
        'training_samples': len(train_gen) * args.batch_size,
        'validation_samples': len(val_gen) * args.batch_size
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Summary
    log.info("\n" + "=" * 70)
    log.info("CONSTRAINED TRAINING COMPLETE")
    log.info("=" * 70)
    
    best_val_mae = min(history.history.get('val_hourly_output_mae', [float('inf')]))
    log.info(f"Best validation MAE: {best_val_mae:.6f}")
    log.info(f"Best model saved to: {exp_dir / 'best_model.keras'}")
    log.info(f"Final model saved to: {final_model_path}")
    
    print(f"\nExperiment directory: {exp_dir}")


if __name__ == '__main__':
    main()
