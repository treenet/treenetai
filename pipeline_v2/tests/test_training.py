"""
Tests for training pipeline.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle

from src.models.training import DataGenerator, ModelTrainer
from src.gaps.gap_injection import GapInjector


class TestDataGenerator:
    """Test data generator for training."""
    
    def test_initialization(self):
        """Test data generator initialization."""
        X = np.random.randn(100, 720, 11).astype(np.float32)
        y = np.random.randn(100, 120, 3).astype(np.float32)
        
        gen = DataGenerator(
            X=X,
            y=y,
            batch_size=16,
            shuffle=True
        )
        
        assert gen.n_samples == 100
        assert gen.batch_size == 16
        assert len(gen.indices) == 100
    
    def test_length(self):
        """Test generator length calculation."""
        X = np.random.randn(100, 720, 11).astype(np.float32)
        y = np.random.randn(100, 120, 3).astype(np.float32)
        
        gen = DataGenerator(X=X, y=y, batch_size=16)
        
        # Should have ceil(100/16) = 7 batches
        assert len(gen) == 7
    
    def test_batch_generation_without_gaps(self):
        """Test batch generation without gap injection."""
        X = np.random.randn(50, 720, 11).astype(np.float32)
        y = np.random.randn(50, 120, 3).astype(np.float32)
        
        gen = DataGenerator(
            X=X,
            y=y,
            batch_size=8,
            gap_injector=None,
            shuffle=False
        )
        
        # Get first batch
        inputs, targets = gen[0]
        
        # Check structure
        assert 'input_x' in inputs
        assert 'input_mask' in inputs
        assert 'recon_output' in targets
        assert 'hourly_output' in targets
        
        # Check shapes
        assert inputs['input_x'].shape == (8, 720, 11)
        assert inputs['input_mask'].shape == (8, 720, 11)
        assert targets['recon_output'].shape == (8, 720, 11)
        assert targets['hourly_output'].shape == (8, 120, 3)
        
        # Without gap injector, mask should be all ones
        assert np.all(inputs['input_mask'] == 1.0)
    
    def test_batch_generation_with_gaps(self):
        """Test batch generation with gap injection."""
        X = np.random.randn(50, 720, 11).astype(np.float32)
        y = np.random.randn(50, 120, 3).astype(np.float32)
        
        gap_injector = GapInjector(
            min_gap_days=1,
            max_gap_days=3,
            min_gaps_per_segment=1,
            max_gaps_per_segment=2,
            gap_channel_prob=0.5,
            random_seed=42
        )
        
        gen = DataGenerator(
            X=X,
            y=y,
            batch_size=8,
            gap_injector=gap_injector,
            shuffle=False
        )
        
        inputs, targets = gen[0]
        
        # Check that gaps were injected (mask should have zeros)
        assert np.any(inputs['input_mask'] == 0.0)
        
        # Check that gapped input differs from original
        assert not np.allclose(inputs['input_x'], targets['recon_output'])
    
    def test_batch_generation_last_batch(self):
        """Test that last batch handles remaining samples correctly."""
        X = np.random.randn(50, 720, 11).astype(np.float32)
        y = np.random.randn(50, 120, 3).astype(np.float32)
        
        gen = DataGenerator(X=X, y=y, batch_size=16, shuffle=False)
        
        # Last batch should have 50 % 16 = 2 samples
        last_batch_idx = len(gen) - 1
        inputs, targets = gen[last_batch_idx]
        
        assert inputs['input_x'].shape[0] == 2
        assert targets['hourly_output'].shape[0] == 2
    
    def test_shuffle_changes_order(self):
        """Test that shuffle changes batch order between epochs."""
        X = np.random.randn(50, 720, 11).astype(np.float32)
        y = np.random.randn(50, 120, 3).astype(np.float32)
        
        gen = DataGenerator(X=X, y=y, batch_size=8, shuffle=True)
        
        # Get first batch
        inputs1, _ = gen[0]
        first_batch_1 = inputs1['input_x'].copy()
        
        # Trigger epoch end (shuffles indices)
        gen.on_epoch_end()
        
        # Get first batch again
        inputs2, _ = gen[0]
        first_batch_2 = inputs2['input_x'].copy()
        
        # Batches should likely be different after shuffle
        # (Very small chance they're the same by random chance)
        assert not np.allclose(first_batch_1, first_batch_2)
    
    def test_no_shuffle_keeps_order(self):
        """Test that without shuffle, order is maintained."""
        X = np.random.randn(50, 720, 11).astype(np.float32)
        y = np.random.randn(50, 120, 3).astype(np.float32)
        
        gen = DataGenerator(X=X, y=y, batch_size=8, shuffle=False)
        
        # Get first batch
        inputs1, _ = gen[0]
        first_batch_1 = inputs1['input_x'].copy()
        
        # Trigger epoch end
        gen.on_epoch_end()
        
        # Get first batch again
        inputs2, _ = gen[0]
        first_batch_2 = inputs2['input_x'].copy()
        
        # Batches should be identical
        np.testing.assert_allclose(first_batch_1, first_batch_2)
    
    def test_all_samples_covered(self):
        """Test that all samples are covered across batches."""
        X = np.random.randn(50, 720, 11).astype(np.float32)
        y = np.random.randn(50, 120, 3).astype(np.float32)
        
        gen = DataGenerator(X=X, y=y, batch_size=7, shuffle=False)
        
        seen_samples = []
        for i in range(len(gen)):
            inputs, _ = gen[i]
            batch_size = inputs['input_x'].shape[0]
            seen_samples.extend(range(i * 7, i * 7 + batch_size))
        
        # Should have seen all 50 samples
        assert len(seen_samples) == 50
        assert set(seen_samples) == set(range(50))


class TestModelTrainer:
    """Test model trainer orchestration."""
    
    def test_initialization(self, sample_config, tmp_path):
        """Test trainer initialization."""
        trainer = ModelTrainer(
            config=sample_config,
            output_dir=tmp_path / "experiments"
        )
        
        assert trainer.config == sample_config
        assert trainer.output_dir.exists()
        assert trainer.model is None
        assert trainer.history is None
    
    def test_initialization_with_gap_injector(self, sample_config, tmp_path):
        """Test trainer creates gap injector when enabled."""
        sample_config.gap.enabled = True
        
        trainer = ModelTrainer(
            config=sample_config,
            output_dir=tmp_path / "experiments"
        )
        
        assert trainer.gap_injector is not None
        assert isinstance(trainer.gap_injector, GapInjector)
    
    def test_initialization_without_gap_injector(self, sample_config, tmp_path):
        """Test trainer doesn't create gap injector when disabled."""
        sample_config.gap.enabled = False
        
        trainer = ModelTrainer(
            config=sample_config,
            output_dir=tmp_path / "experiments"
        )
        
        assert trainer.gap_injector is None
    
    def test_load_data(self, sample_config, tmp_path):
        """Test data loading from pickle files."""
        # Create dummy data files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        X_train = np.random.randn(50, 720, 11).astype(np.float32)
        y_train = np.random.randn(50, 120, 3).astype(np.float32)
        X_test = np.random.randn(10, 720, 11).astype(np.float32)
        y_test = np.random.randn(10, 120, 3).astype(np.float32)
        
        with open(data_dir / 'train_input_segments_numpy.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(data_dir / 'train_output_segments_numpy.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(data_dir / 'test_input_segments_numpy.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open(data_dir / 'test_output_segments_numpy.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        
        # Load data
        trainer = ModelTrainer(sample_config, tmp_path / "exp")
        X_tr, y_tr, X_te, y_te = trainer.load_data(data_dir)
        
        # Check shapes
        assert X_tr.shape == (50, 720, 11)
        assert y_tr.shape == (50, 120, 3)
        assert X_te.shape == (10, 720, 11)
        assert y_te.shape == (10, 120, 3)
    
    def test_build_model(self, sample_config, tmp_path):
        """Test model building."""
        trainer = ModelTrainer(sample_config, tmp_path / "exp")
        model = trainer.build_model()
        
        assert model is not None
        assert trainer.model is not None
        assert isinstance(model, tf.keras.Model)
    
    def test_create_callbacks(self, sample_config, tmp_path):
        """Test callback creation."""
        trainer = ModelTrainer(sample_config, tmp_path / "exp")
        callbacks = trainer.create_callbacks()
        
        assert len(callbacks) >= 3  # Checkpoint, EarlyStopping, ReduceLROnPlateau
        
        # Check callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'ModelCheckpoint' in callback_types
        assert 'EarlyStopping' in callback_types
        assert 'ReduceLROnPlateau' in callback_types
    
    def test_save_training_history(self, sample_config, tmp_path):
        """Test saving training history."""
        trainer = ModelTrainer(sample_config, tmp_path / "exp")
        
        # Create dummy history
        trainer.history = {
            'loss': [1.0, 0.8, 0.6],
            'val_loss': [1.2, 0.9, 0.7],
            'recon_output_loss': [0.5, 0.4, 0.3],
            'hourly_output_loss': [0.5, 0.4, 0.3]
        }
        
        trainer.save_training_history()
        
        # Check that history file was created
        history_file = tmp_path / "exp" / "training_history.json"
        assert history_file.exists()
        
        # Load and verify
        import json
        with open(history_file) as f:
            loaded_history = json.load(f)
        
        assert loaded_history['loss'] == [1.0, 0.8, 0.6]
        assert loaded_history['val_loss'] == [1.2, 0.9, 0.7]
    
    def test_save_config(self, sample_config, tmp_path):
        """Test saving configuration."""
        trainer = ModelTrainer(sample_config, tmp_path / "exp")
        trainer.save_config()
        
        config_file = tmp_path / "exp" / "config.json"
        assert config_file.exists()
    
    def test_train_single_epoch(self, sample_config, tmp_path):
        """Test training for a single epoch (integration test)."""
        # Create small dataset with correct shape (30 days at 10-min = 4320 timesteps)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # 30 days * 144 steps/day = 4320 timesteps input
        # 30 days * 24 hours/day = 720 timesteps output
        X_train = np.random.randn(8, 4320, 11).astype(np.float32)
        y_train = np.random.randn(8, 720, 3).astype(np.float32)
        X_test = np.random.randn(2, 4320, 11).astype(np.float32)
        y_test = np.random.randn(2, 720, 3).astype(np.float32)
        
        with open(data_dir / 'train_input_segments_numpy.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(data_dir / 'train_output_segments_numpy.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(data_dir / 'test_input_segments_numpy.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open(data_dir / 'test_output_segments_numpy.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        
        # Train for 1 epoch
        sample_config.model.epochs = 1
        sample_config.model.batch_size = 4
        
        trainer = ModelTrainer(sample_config, tmp_path / "exp")
        trainer.load_data(data_dir)
        trainer.build_model()
        
        # Create data generators
        train_gen = DataGenerator(
            X=X_train,
            y=y_train,
            batch_size=sample_config.model.batch_size,
            gap_injector=trainer.gap_injector,
            shuffle=True
        )
        
        test_gen = DataGenerator(
            X=X_test,
            y=y_test,
            batch_size=sample_config.model.batch_size,
            gap_injector=None,
            shuffle=False
        )
        
        # Train
        history = trainer.model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=1,
            callbacks=trainer.create_callbacks(),
            verbose=0
        )
        
        # Check that training completed
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) == 1


class TestTrainingIntegration:
    """Integration tests for full training pipeline."""
    
    def test_full_training_pipeline(self, sample_config, tmp_path):
        """Test complete training workflow."""
        # Setup
        data_dir = tmp_path / "data"
        exp_dir = tmp_path / "experiments"
        data_dir.mkdir()
        
        # Create small dataset
        n_train = 16
        n_test = 4
        timesteps = 30 * 144
        hourly_steps = 30 * 24
        
        X_train = np.random.randn(n_train, timesteps, 11).astype(np.float32)
        y_train = np.random.randn(n_train, hourly_steps, 3).astype(np.float32)
        X_test = np.random.randn(n_test, timesteps, 11).astype(np.float32)
        y_test = np.random.randn(n_test, hourly_steps, 3).astype(np.float32)
        
        # Save data
        with open(data_dir / 'train_input_segments_numpy.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(data_dir / 'train_output_segments_numpy.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(data_dir / 'test_input_segments_numpy.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open(data_dir / 'test_output_segments_numpy.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        
        # Configure for quick test
        sample_config.model.epochs = 2
        sample_config.model.batch_size = 4
        sample_config.gap.enabled = True
        
        # Initialize trainer
        trainer = ModelTrainer(sample_config, exp_dir)
        
        # Load data
        X_tr, y_tr, X_te, y_te = trainer.load_data(data_dir)
        
        # Build model
        model = trainer.build_model()
        
        # Create generators
        train_gen = DataGenerator(
            X=X_tr, y=y_tr,
            batch_size=sample_config.model.batch_size,
            gap_injector=trainer.gap_injector,
            shuffle=True
        )
        test_gen = DataGenerator(
            X=X_te, y=y_te,
            batch_size=sample_config.model.batch_size,
            gap_injector=None,
            shuffle=False
        )
        
        # Train
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=sample_config.model.epochs,
            callbacks=trainer.create_callbacks(),
            verbose=0
        )
        
        trainer.history = history.history
        
        # Save results
        trainer.save_training_history()
        trainer.save_config()
        
        # Verify outputs
        assert (exp_dir / "training_history.json").exists()
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "best_model.keras").exists()
        
        # Verify training occurred
        assert len(history.history['loss']) == 2
        assert all(isinstance(x, (int, float)) for x in history.history['loss'])
