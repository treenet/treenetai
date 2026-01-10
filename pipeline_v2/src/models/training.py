"""
Training pipeline for TCN model.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from datetime import datetime

from ..gaps.gap_injection import GapInjector
from ..gaps.metrics import GapFillingMetrics


class SegmentNormalizer:
    """
    Normalizes segments with gap-awareness for segment-level normalization.
    
    When gaps are present, normalization parameters are computed from
    non-gap regions only. This reflects real-world gap-filling scenarios
    where we don't know the min/max values in the gap region.
    """
    
    @staticmethod
    def normalize_with_mask(
        segment: np.ndarray,
        mask: np.ndarray,
        method: str = 'minmax'
    ) -> tuple:
        """
        Normalize segment using only non-gap regions.
        
        Args:
            segment: Data array (timesteps, channels)
            mask: Binary mask (1=valid, 0=gap)
            method: 'minmax' for [0,1] scaling
            
        Returns:
            Tuple of (normalized_segment, norm_params)
            norm_params is dict with 'min' and 'diff' per channel
        """
        normalized = segment.copy()
        norm_params = {'min': {}, 'diff': {}}
        
        n_channels = segment.shape[-1]
        
        for ch in range(n_channels):
            # Get valid (non-gap) values for this channel
            valid_mask = mask[:, ch] > 0.5
            valid_values = segment[valid_mask, ch]
            
            if len(valid_values) == 0:
                # No valid values - use defaults
                norm_params['min'][ch] = 0.0
                norm_params['diff'][ch] = 1.0
                continue
            
            if method == 'minmax':
                vmin = float(valid_values.min())
                vmax = float(valid_values.max())
                diff = vmax - vmin
                
                if diff < 1e-8:
                    diff = 1.0
                
                norm_params['min'][ch] = vmin
                norm_params['diff'][ch] = diff
                
                # Normalize entire channel (including gap region)
                # Gap values will be 0 from gap injection, so they'll normalize to -vmin/diff
                normalized[:, ch] = (segment[:, ch] - vmin) / diff
        
        return normalized, norm_params
    
    @staticmethod
    def denormalize(
        segment: np.ndarray,
        norm_params: dict
    ) -> np.ndarray:
        """
        Reverse normalization.
        
        Args:
            segment: Normalized array
            norm_params: Dict with 'min' and 'diff' per channel
            
        Returns:
            Denormalized array
        """
        denormalized = segment.copy()
        
        for ch in range(segment.shape[-1]):
            if ch in norm_params['min'] and ch in norm_params['diff']:
                vmin = norm_params['min'][ch]
                vdiff = norm_params['diff'][ch]
                denormalized[:, ch] = segment[:, ch] * vdiff + vmin
        
        return denormalized


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for training with on-the-fly gap injection.
    
    This generates batches during training with random gaps injected,
    which serves as data augmentation.
    
    Supports two modes:
    1. Pre-normalized data (norm_on_fly=False): Data is already normalized
    2. Gap-aware normalization (norm_on_fly=True): Normalizes each segment
       using only non-gap regions (for segment-level normalization)
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        gap_injector: Optional[GapInjector] = None,
        shuffle: bool = True,
        norm_on_fly: bool = False,
        norm_method: str = 'minmax'
    ):
        """
        Initialize data generator.
        
        Args:
            X: Input data, shape (n_samples, timesteps, channels)
            y: Target data, shape (n_samples, timesteps, channels)
            batch_size: Batch size
            gap_injector: GapInjector instance for augmentation
            shuffle: Whether to shuffle data between epochs
            norm_on_fly: If True, normalize each segment dynamically using
                        only non-gap regions (for segment-level normalization)
            norm_method: Normalization method ('minmax')
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.gap_injector = gap_injector
        self.shuffle = shuffle
        self.norm_on_fly = norm_on_fly
        self.norm_method = norm_method
        
        self.segment_normalizer = SegmentNormalizer() if norm_on_fly else None
        
        self.n_samples = len(X)
        self.indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
        """
        Generate one batch of data.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of (inputs_dict, targets_dict)
        """
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices].copy()
        
        # Inject gaps if configured
        if self.gap_injector is not None:
            X_gapped, masks = self.gap_injector.inject_gaps_batch(X_batch)
        else:
            X_gapped = X_batch.copy()
            masks = np.ones_like(X_batch)
        
        # Apply gap-aware normalization if configured
        if self.norm_on_fly:
            X_normalized = np.zeros_like(X_gapped)
            y_normalized = np.zeros_like(y_batch)
            
            for i in range(len(batch_indices)):
                # Normalize input using only non-gap regions
                X_normalized[i], _ = self.segment_normalizer.normalize_with_mask(
                    X_gapped[i], masks[i], self.norm_method
                )
                
                # For targets, we need to use the same normalization logic
                # But targets don't have gaps, so we use the input's valid mask
                # to determine normalization params, then apply to targets
                
                # For the 3 target channels, find corresponding input channels
                # local_T (target 0) ~ temp_treenet (input 0)
                # local_RH (target 1) ~ rh_treenet (input 1)
                # stem (target 2) ~ stem (input 2)
                
                # Use input mask to compute target normalization
                y_norm_params = {'min': {}, 'diff': {}}
                for ch in range(y_batch.shape[-1]):
                    # Get valid values from input (same mask)
                    # But target has hourly resolution, input has 10-min
                    # So we subsample the mask
                    if X_gapped.shape[1] == 4320 and y_batch.shape[1] == 720:
                        # Subsample mask to hourly (every 6th timestep)
                        hourly_mask = masks[i, ::6, ch] if ch < masks.shape[-1] else np.ones(720)
                    else:
                        hourly_mask = masks[i, :, ch] if ch < masks.shape[-1] else np.ones(y_batch.shape[1])
                    
                    valid_mask = hourly_mask > 0.5
                    valid_values = y_batch[i, valid_mask, ch]
                    
                    if len(valid_values) > 0:
                        vmin = float(valid_values.min())
                        vmax = float(valid_values.max())
                        diff = vmax - vmin if vmax - vmin > 1e-8 else 1.0
                    else:
                        vmin, diff = 0.0, 1.0
                    
                    y_norm_params['min'][ch] = vmin
                    y_norm_params['diff'][ch] = diff
                    y_normalized[i, :, ch] = (y_batch[i, :, ch] - vmin) / diff
            
            X_gapped = X_normalized
            # Keep y_batch as reconstruction target (original unnormalized)
            # But use normalized for hourly output
            X_batch_recon = X_normalized  # For reconstruction
            y_batch_hourly = y_normalized  # For hourly prediction
        else:
            X_batch_recon = X_batch
            y_batch_hourly = y_batch
        
        # Prepare inputs
        inputs = {
            'input_x': X_gapped.astype(np.float32),
            'input_mask': masks.astype(np.float32)
        }
        
        # Prepare targets
        targets = {
            'recon_output': X_batch_recon.astype(np.float32),  # For reconstruction loss
            'hourly_output': y_batch_hourly.astype(np.float32)  # Hourly targets
        }
        
        return inputs, targets
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class ModelTrainer:
    """
    Orchestrates the model training process.
    
    Handles:
    1. Data loading
    2. Model building and compilation
    3. Training with callbacks
    4. Evaluation
    5. Saving results
    """
    
    def __init__(
        self,
        config,
        output_dir: Path
    ):
        """
        Initialize model trainer.
        
        Args:
            config: PipelineConfig object with all settings
            output_dir: Directory to save models and results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.history = None
        self.gap_injector = None
        
        # Create gap injector if enabled
        if config.gap.enabled:
            self.gap_injector = GapInjector(
                min_gap_days=config.gap.min_gap_days,
                max_gap_days=config.gap.max_gap_days,
                min_gaps_per_segment=config.gap.min_gaps_per_segment,
                max_gaps_per_segment=config.gap.max_gaps_per_segment,
                gap_channel_prob=config.gap.gap_channel_prob,
                random_seed=config.gap.random_seed
            )
    
    def load_data(
        self,
        data_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test data from processed segments.
        
        Args:
            data_dir: Directory containing processed segment files
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        print("Loading training data...")
        with open(data_dir / 'train_input_segments_numpy.pkl', 'rb') as f:
            X_train = pickle.load(f)
        
        with open(data_dir / 'train_output_segments_numpy.pkl', 'rb') as f:
            y_train = pickle.load(f)
        
        print("Loading test data...")
        with open(data_dir / 'test_input_segments_numpy.pkl', 'rb') as f:
            X_test = pickle.load(f)
        
        with open(data_dir / 'test_output_segments_numpy.pkl', 'rb') as f:
            y_test = pickle.load(f)
        
        print(f"Train: X={X_train.shape}, y={y_train.shape}")
        print(f"Test:  X={X_test.shape}, y={y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self) -> keras.Model:
        """
        Build and compile the TCN model.
        
        Returns:
            Compiled Keras model
        """
        from ..models.tcn import TCNModel
        
        print("Building TCN model...")
        
        # Check for attention config
        use_attention = getattr(self.config.model, 'use_attention', False)
        n_attention_heads = getattr(self.config.model, 'n_attention_heads', 4)
        attention_key_dim = getattr(self.config.model, 'attention_key_dim', 32)
        
        if use_attention:
            print(f"  Attention: {n_attention_heads} heads, key_dim={attention_key_dim}")
        
        tcn = TCNModel(
            n_input_channels=self.config.data.n_input_channels,
            n_target_channels=self.config.data.n_target_channels,
            n_filters=self.config.model.n_filters,
            kernel_size=self.config.model.kernel_size,
            n_blocks=self.config.model.n_blocks,
            dropout_rate=self.config.model.dropout_rate,
            input_length=self.config.segment.input_steps,
            output_length=self.config.segment.output_steps,
            use_attention=use_attention,
            n_attention_heads=n_attention_heads,
            attention_key_dim=attention_key_dim
        )
        
        model = tcn.build()
        
        tcn.compile_model(
            learning_rate=self.config.model.learning_rate,
            recon_masked_weight=self.config.model.recon_masked_weight,
            recon_unmasked_weight=self.config.model.recon_unmasked_weight,
            hourly_weight=self.config.model.hourly_weight
        )
        
        self.model = model
        
        if self.config.verbose:
            model.summary()
        
        return model
    
    def create_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_path = self.output_dir / 'best_model.keras'
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1 if self.config.verbose else 0
            )
        )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.model.early_stop_patience,
                restore_best_weights=True,
                verbose=1 if self.config.verbose else 0
            )
        )
        
        # Reduce learning rate on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.model.reduce_lr_patience,
                min_lr=self.config.model.min_lr,
                verbose=1 if self.config.verbose else 0
            )
        )
        
        # CSV logger
        csv_path = self.output_dir / 'training_history.csv'
        callbacks.append(
            keras.callbacks.CSVLogger(
                filename=str(csv_path),
                append=False
            )
        )
        
        # TensorBoard (optional)
        tensorboard_dir = self.output_dir / 'tensorboard'
        tensorboard_dir.mkdir(exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True
            )
        )
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Training history object
        """
        if self.model is None:
            self.build_model()
        
        print("\nStarting training...")
        print(f"Epochs: {self.config.model.epochs}")
        print(f"Batch size: {self.config.model.batch_size}")
        print(f"Gap injection: {'enabled' if self.config.gap.enabled else 'disabled'}")
        
        # Check if segment-level normalization is enabled
        # If norm_scope='segment', data is raw and needs on-the-fly normalization
        norm_scope = getattr(self.config.normalization, 'scope', 'year')
        norm_on_fly = (norm_scope == 'segment')
        
        if norm_on_fly:
            print(f"Normalization: on-the-fly (segment-level, gap-aware)")
        else:
            print(f"Normalization: pre-computed (year-level)")
        
        # Create data generators
        train_gen = DataGenerator(
            X_train, y_train,
            batch_size=self.config.model.batch_size,
            gap_injector=self.gap_injector if self.config.gap.enabled else None,
            shuffle=True,
            norm_on_fly=norm_on_fly,
            norm_method=getattr(self.config.normalization, 'method', 'minmax')
        )
        
        val_gen = DataGenerator(
            X_test, y_test,
            batch_size=self.config.model.batch_size,
            gap_injector=None,  # No gaps for validation
            shuffle=False,
            norm_on_fly=norm_on_fly,
            norm_method=getattr(self.config.normalization, 'method', 'minmax')
        )
        
        # Train
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config.model.epochs,
            callbacks=self.create_callbacks(),
            verbose=1 if self.config.verbose else 0
        )
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary of evaluation metrics (MAE, MSE, R²)
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("\nEvaluating model on test set...")
        
        # Generate predictions without gaps
        masks = np.ones_like(X_test)
        predictions = self.model.predict([X_test, masks], verbose=0)
        
        recon_pred = predictions[0]  # Reconstruction output
        hourly_pred = predictions[1]  # Hourly output
        
        def compute_r2(y_true, y_pred):
            """Compute R² (coefficient of determination)."""
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot == 0:
                return 0.0
            return float(1.0 - (ss_res / ss_tot))
        
        # Compute metrics for reconstruction
        recon_metrics = {}
        for i, ch_name in enumerate(self.config.data.input_channels):
            y_true = X_test[..., i].flatten()
            y_pred = recon_pred[..., i].flatten()
            
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            r2 = compute_r2(y_true, y_pred)
            
            recon_metrics[f'recon_{ch_name}_mae'] = float(mae)
            recon_metrics[f'recon_{ch_name}_mse'] = float(mse)
            recon_metrics[f'recon_{ch_name}_r2'] = float(r2)
        
        # Compute metrics for hourly prediction
        hourly_metrics = {}
        for i, ch_name in enumerate(self.config.data.target_channels):
            y_true = y_test[..., i].flatten()
            y_pred = hourly_pred[..., i].flatten()
            
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            r2 = compute_r2(y_true, y_pred)
            
            hourly_metrics[f'hourly_{ch_name}_mae'] = float(mae)
            hourly_metrics[f'hourly_{ch_name}_mse'] = float(mse)
            hourly_metrics[f'hourly_{ch_name}_r2'] = float(r2)
        
        all_metrics = {**recon_metrics, **hourly_metrics}
        
        # Print summary (MAE and R² for brevity)
        print("\nEvaluation Results:")
        print("\nReconstruction Metrics (11 channels):")
        for ch_name in self.config.data.input_channels:
            mae = recon_metrics[f'recon_{ch_name}_mae']
            r2 = recon_metrics[f'recon_{ch_name}_r2']
            print(f"  {ch_name:15s}: MAE={mae:.4f}, R²={r2:.4f}")
        
        print("\nHourly Prediction Metrics (3 channels):")
        for ch_name in self.config.data.target_channels:
            mae = hourly_metrics[f'hourly_{ch_name}_mae']
            r2 = hourly_metrics[f'hourly_{ch_name}_r2']
            print(f"  {ch_name:15s}: MAE={mae:.4f}, R²={r2:.4f}")
        
        return all_metrics
    
    def save_training_history(self):
        """Save training history to JSON file."""
        if self.history is None:
            print("Warning: No training history to save")
            return
        
        history_path = self.output_dir / 'training_history.json'
        
        # Convert numpy types to Python types for JSON serialization
        history_dict = {}
        if hasattr(self.history, 'history'):
            # Keras History object
            history_dict = self.history.history
        elif isinstance(self.history, dict):
            # Already a dictionary
            history_dict = self.history
        
        # Convert numpy arrays to lists
        serializable_history = {}
        for key, value in history_dict.items():
            if hasattr(value, 'tolist'):
                serializable_history[key] = value.tolist()
            elif isinstance(value, list):
                serializable_history[key] = value
            else:
                serializable_history[key] = [value]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
    
    def save_config(self):
        """Save configuration to JSON file."""
        config_path = self.output_dir / 'config.json'
        from dataclasses import asdict
        config_dict = asdict(self.config)
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
    
    def save_results(self, metrics: Dict):
        """
        Save training results and configuration.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        # Save metrics
        metrics_path = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save config using save_config method
        self.save_config()
        
        # Save final model
        final_model_path = self.output_dir / 'final_model.keras'
        self.model.save(str(final_model_path))
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def run_full_pipeline(self, data_dir: Path) -> Dict:
        """
        Run the complete training pipeline.
        
        Args:
            data_dir: Directory with processed segment data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load data
        X_train, y_train, X_test, y_test = self.load_data(data_dir)
        
        # Build model
        self.build_model()
        
        # Train
        self.train(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        
        # Save results
        self.save_results(metrics)
        
        return metrics


def create_experiment_dir(base_dir: Path, experiment_name: Optional[str] = None) -> Path:
    """
    Create a timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment_name:
        exp_dir = base_dir / f"{timestamp}_{experiment_name}"
    else:
        exp_dir = base_dir / timestamp
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir
