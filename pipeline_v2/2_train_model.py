#!/usr/bin/env python3
"""
Train TCN model for gap filling and hourly prediction.

This unified training script supports:
1. Training from scratch with configurable architecture
2. Fine-tuning pre-trained models
3. Constrained training with physical bounds (e.g., RH in [0,1])

Usage examples:
    # Train from scratch
    python 2_train_model.py --data-dir /path/to/model_data
    
    # Train with attention
    python 2_train_model.py --data-dir /path/to/model_data --use-attention
    
    # Fine-tune with RH constraint
    python 2_train_model.py --data-dir /path/to/model_data \
        --fine-tune /path/to/best_model.keras \
        --constrain-rh --rh-penalty-weight 0.1 --learning-rate 1e-4

Author: TreeNet AI Pipeline v2
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.models.tcn import TCNModel, TCNBlock, PositionalEncoding
from src.models.training import ModelTrainer, DataGenerator, create_experiment_dir
from src.gaps.gap_injection import GapInjector
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TCN model for gap filling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch with attention
  python 2_train_model.py --data-dir /path/to/data --use-attention
  
  # Fine-tune with RH constraint
  python 2_train_model.py --data-dir /path/to/data \\
      --fine-tune /path/to/model.keras --constrain-rh
        """
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data',
        help='Directory with processed segment files'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments',
        help='Base directory for experiment outputs'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Optional experiment name suffix'
    )
    
    # Fine-tuning
    parser.add_argument(
        '--fine-tune', type=str, default=None,
        help='Path to pre-trained model for fine-tuning (skips building new model)'
    )
    
    # Model architecture (ignored if fine-tuning)
    arch_group = parser.add_argument_group('Model architecture (training from scratch)')
    arch_group.add_argument('--n-filters', type=int, default=64, help='TCN filters')
    arch_group.add_argument('--kernel-size', type=int, default=3, help='Kernel size')
    arch_group.add_argument('--n-blocks', type=int, default=4, help='Number of TCN blocks')
    arch_group.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Attention
    arch_group.add_argument('--use-attention', action='store_true', 
                            help='Add attention after TCN encoder')
    arch_group.add_argument('--n-attention-heads', type=int, default=4, 
                            help='Number of attention heads')
    arch_group.add_argument('--attention-key-dim', type=int, default=32, 
                            help='Attention key dimension')
    
    # Training
    train_group = parser.add_argument_group('Training parameters')
    train_group.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_group.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_group.add_argument('--learning-rate', type=float, default=3e-4, 
                             help='Learning rate (use ~1e-4 for fine-tuning)')
    train_group.add_argument('--early-stopping-patience', type=int, default=10,
                             help='Early stopping patience')
    train_group.add_argument('--lr-reduce-patience', type=int, default=5,
                             help='Patience for learning rate reduction')
    
    # Constraints
    constraint_group = parser.add_argument_group('Physical constraints')
    constraint_group.add_argument('--constrain-rh', action='store_true',
                                  help='Add penalty for RH outside [0,1]')
    constraint_group.add_argument('--rh-penalty-weight', type=float, default=0.1,
                                  help='Weight for RH boundary penalty')
    
    # Gap injection
    gap_group = parser.add_argument_group('Gap injection (data augmentation)')
    gap_group.add_argument('--no-gaps', action='store_true', help='Disable gap injection')
    gap_group.add_argument('--min-gap-days', type=int, default=1, help='Min gap length')
    gap_group.add_argument('--max-gap-days', type=int, default=12, help='Max gap length')
    
    # Misc
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def create_constrained_hourly_loss(rh_penalty_weight: float = 0.1):
    """
    Create a loss function with physical boundary constraints on RH.
    
    The RH channel (index 1) is constrained to [0, 1] in normalized space.
    Values outside this range incur a quadratic penalty.
    """
    def constrained_hourly_loss(y_true, y_pred):
        # Standard MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Extract RH channel (index 1: [local_T, local_RH, stem])
        rh_pred = y_pred[:, :, 1]
        
        # Penalty for values outside [0, 1]
        upper_violation = tf.maximum(0.0, rh_pred - 1.0)
        lower_violation = tf.maximum(0.0, -rh_pred)
        
        # Quadratic penalty
        rh_penalty = tf.reduce_mean(
            tf.square(upper_violation) + tf.square(lower_violation)
        )
        
        return mse_loss + rh_penalty_weight * rh_penalty
    
    return constrained_hourly_loss


def build_model_from_config(config, input_shape, output_channels):
    """Build a new TCN model from configuration."""
    model = TCNModel.build(
        input_shape=input_shape,
        output_channels=output_channels,
        n_filters=config.model.n_filters,
        kernel_size=config.model.kernel_size,
        n_blocks=config.model.n_blocks,
        dropout_rate=config.model.dropout_rate,
        use_attention=config.model.use_attention,
        n_attention_heads=config.model.n_attention_heads,
        attention_key_dim=config.model.attention_key_dim
    )
    return model


def load_data(data_dir: Path):
    """Load training and validation data from pickle files."""
    with open(data_dir / 'train_input_segments_numpy.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir / 'train_output_segments_numpy.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
        y_val = pickle.load(f)
    
    return X_train, y_train, X_val, y_val


def main():
    """Main function."""
    args = parse_args()
    
    # Create experiment directory
    suffix = args.experiment_name or ""
    if args.fine_tune:
        suffix = f"{suffix}_finetune" if suffix else "finetune"
    if args.constrain_rh:
        suffix = f"{suffix}_constrained_rh" if suffix else "constrained_rh"
    if args.use_attention and not args.fine_tune:
        suffix = f"{suffix}_attention" if suffix else "attention"
    
    exp_dir = create_experiment_dir(
        base_dir=Path(args.output_dir),
        experiment_name=suffix if suffix else None
    )
    
    # Setup logging
    log_file = exp_dir / 'training.log'
    log = setup_logging(log_file=log_file, name='train_model', verbose=args.verbose)
    
    log.info("=" * 80)
    log.info("TreeNet AI Pipeline v2 - Model Training")
    log.info("=" * 80)
    log.info(f"Experiment directory: {exp_dir}")
    
    # Mode detection
    if args.fine_tune:
        log.info(f"\nMode: FINE-TUNING from {args.fine_tune}")
    else:
        log.info("\nMode: TRAINING FROM SCRATCH")
    
    if args.constrain_rh:
        log.info(f"Constraints: RH bounded to [0,1] with penalty weight {args.rh_penalty_weight}")
    
    # Load data
    log.info("\nLoading data...")
    data_dir = Path(args.data_dir)
    X_train, y_train, X_val, y_val = load_data(data_dir)
    log.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    log.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    
    # Get model (fine-tune existing or build new)
    if args.fine_tune:
        log.info(f"\nLoading pre-trained model: {args.fine_tune}")
        model = keras.models.load_model(
            args.fine_tune,
            custom_objects={
                'TCNBlock': TCNBlock,
                'PositionalEncoding': PositionalEncoding
            }
        )
        log.info("Model loaded successfully")
    else:
        # Build new model from config
        config = PipelineConfig()
        config.model.n_filters = args.n_filters
        config.model.kernel_size = args.kernel_size
        config.model.n_blocks = args.n_blocks
        config.model.dropout_rate = args.dropout
        config.model.use_attention = args.use_attention
        config.model.n_attention_heads = args.n_attention_heads
        config.model.attention_key_dim = args.attention_key_dim
        
        log.info(f"\nBuilding new model:")
        log.info(f"  TCN: {args.n_blocks} blocks, {args.n_filters} filters, kernel={args.kernel_size}")
        if args.use_attention:
            log.info(f"  Attention: {args.n_attention_heads} heads, key_dim={args.attention_key_dim}")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_channels = y_train.shape[2]
        model = build_model_from_config(config, input_shape, output_channels)
    
    # Setup loss functions
    if args.constrain_rh:
        hourly_loss = create_constrained_hourly_loss(args.rh_penalty_weight)
        log.info(f"\nUsing constrained hourly loss (RH penalty weight: {args.rh_penalty_weight})")
    else:
        hourly_loss = 'mse'
    
    # Compile model
    log.info("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            'recon_output': 'mse',
            'hourly_output': hourly_loss
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
    
    # Gap injection
    gap_injector = None
    if not args.no_gaps:
        gap_injector = GapInjector(
            min_gap_days=args.min_gap_days,
            max_gap_days=args.max_gap_days
        )
        log.info(f"Gap injection: {args.min_gap_days}-{args.max_gap_days} days")
    else:
        log.info("Gap injection: disabled")
    
    # Create data generators
    log.info("\nSetting up data generators...")
    train_gen = DataGenerator(
        X=X_train, y=y_train,
        batch_size=args.batch_size,
        gap_injector=gap_injector,
        shuffle=True
    )
    
    val_gen = DataGenerator(
        X=X_val, y=y_val,
        batch_size=args.batch_size,
        gap_injector=None,
        shuffle=False
    )
    
    log.info(f"Training batches: {len(train_gen)}")
    log.info(f"Validation batches: {len(val_gen)}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_hourly_output_mae',
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_hourly_output_mae',
            factor=0.5,
            patience=args.lr_reduce_patience,
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
        keras.callbacks.CSVLogger(str(exp_dir / 'training_history.csv')),
        keras.callbacks.TensorBoard(log_dir=str(exp_dir / 'tensorboard'), histogram_freq=1)
    ]
    
    # Train
    log.info(f"\nStarting training for {args.epochs} epochs...")
    log.info(f"Learning rate: {args.learning_rate}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = exp_dir / 'final_model.keras'
    model.save(str(final_path))
    log.info(f"Final model saved: {final_path}")
    
    # Save config
    config_dict = {
        'mode': 'fine_tune' if args.fine_tune else 'from_scratch',
        'base_model': args.fine_tune,
        'data_dir': str(args.data_dir),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'constrain_rh': args.constrain_rh,
        'rh_penalty_weight': args.rh_penalty_weight if args.constrain_rh else None,
        'gap_injection': not args.no_gaps,
        'min_gap_days': args.min_gap_days,
        'max_gap_days': args.max_gap_days,
        'use_attention': args.use_attention,
        'n_filters': args.n_filters,
        'n_blocks': args.n_blocks,
        'kernel_size': args.kernel_size,
        'dropout': args.dropout,
        'n_attention_heads': args.n_attention_heads,
        'attention_key_dim': args.attention_key_dim,
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'epochs_trained': len(history.history['loss'])
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Summary
    log.info("\n" + "=" * 80)
    log.info("TRAINING COMPLETE")
    log.info("=" * 80)
    
    best_val_mae = min(history.history.get('val_hourly_output_mae', [float('inf')]))
    log.info(f"Best validation MAE: {best_val_mae:.6f}")
    log.info(f"Epochs trained: {len(history.history['loss'])}")
    log.info(f"Best model: {exp_dir / 'best_model.keras'}")
    log.info(f"Final model: {final_path}")
    log.info("=" * 80)


if __name__ == '__main__':
    main()
