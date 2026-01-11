#!/usr/bin/env python3
"""
Train TCN model for gap filling and hourly prediction.

This script:
1. Loads processed 30-day segments
2. Builds TCN model
3. Trains with gap injection for data augmentation
4. Evaluates on test set
5. Saves trained model and metrics

Usage:
    python 2_train_model.py --data-dir /path/to/processed/model_data
    python 2_train_model.py --epochs 100 --batch-size 32 --gap-days 12
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig, ModelConfig, GapConfig
from src.models.training import ModelTrainer, create_experiment_dir
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TCN model')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/model_data',
        help='Directory with processed segment files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/experiments',
        help='Base directory for experiment outputs'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Optional experiment name'
    )
    
    # Model architecture
    parser.add_argument('--n-filters', type=int, default=64, help='TCN filters')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size')
    parser.add_argument('--n-blocks', type=int, default=4, help='Number of TCN blocks')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Attention
    parser.add_argument('--use-attention', action='store_true', help='Add attention after TCN encoder')
    parser.add_argument('--n-attention-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--attention-key-dim', type=int, default=32, help='Attention key dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    
    # Gap injection
    parser.add_argument('--no-gaps', action='store_true', help='Disable gap injection')
    parser.add_argument('--min-gap-days', type=int, default=1, help='Min gap length')
    parser.add_argument('--max-gap-days', type=int, default=12, help='Max gap length')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create experiment directory first
    exp_dir = create_experiment_dir(
        base_dir=Path(args.output_dir),
        experiment_name=args.experiment_name
    )
    
    # Setup logging - single log file for all output
    log_file = exp_dir / 'training.log'
    log = setup_logging(log_file=log_file, name='train_model', verbose=args.verbose)
    
    log.info("="*80)
    log.info("TreeNet AI Pipeline v2 - Model Training")
    log.info("="*80)
    log.info(f"Experiment directory: {exp_dir}")
    log.info(f"Log file: {log_file}")
    
    # Create configuration
    log.info("\nConfiguration:")
    config = PipelineConfig()
    
    # Update model config
    config.model.n_filters = args.n_filters
    config.model.kernel_size = args.kernel_size
    config.model.n_blocks = args.n_blocks
    config.model.dropout_rate = args.dropout
    config.model.epochs = args.epochs
    config.model.batch_size = args.batch_size
    config.model.learning_rate = args.learning_rate
    
    # Attention config
    config.model.use_attention = args.use_attention
    config.model.n_attention_heads = args.n_attention_heads
    config.model.attention_key_dim = args.attention_key_dim
    
    # Update gap config
    config.gap.enabled = not args.no_gaps
    config.gap.min_gap_days = args.min_gap_days
    config.gap.max_gap_days = args.max_gap_days
    
    config.verbose = args.verbose
    
    log.info(f"  Model: TCN with {args.n_blocks} blocks, {args.n_filters} filters")
    if args.use_attention:
        log.info(f"  Attention: {args.n_attention_heads} heads, key_dim={args.attention_key_dim}")
    else:
        log.info(f"  Attention: disabled")
    log.info(f"  Training: {args.epochs} epochs, batch size {args.batch_size}")
    log.info(f"  Gap injection: {'enabled' if config.gap.enabled else 'disabled'}")
    if config.gap.enabled:
        log.info(f"    Gap range: {args.min_gap_days}-{args.max_gap_days} days")
    
    # Initialize trainer
    log.info("\nInitializing trainer...")
    trainer = ModelTrainer(config=config, output_dir=exp_dir)
    
    # Run training pipeline
    log.info("\nStarting training pipeline...")
    metrics = trainer.run_full_pipeline(data_dir=Path(args.data_dir))
    
    log.info("\n" + "="*80)
    log.info("Training complete!")
    log.info(f"Results saved to: {exp_dir}")
    log.info("\nFinal metrics:")
    for key, value in metrics.items():
        log.info(f"  {key}: {value:.6f}")
    log.info("="*80)


if __name__ == '__main__':
    main()
