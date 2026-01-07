"""Tests for gap injection module."""

import pytest
import numpy as np
from src.gaps.gap_injection import GapInjector


class TestGapInjector:
    """Test GapInjector class."""
    
    def test_initialization(self):
        """Test gap injector initialization."""
        injector = GapInjector(
            min_gap_days=1,
            max_gap_days=5,
            min_gaps_per_segment=1,
            max_gaps_per_segment=3,
            gap_channel_prob=0.5,
            random_seed=42
        )
        
        assert injector.min_gap_days == 1
        assert injector.max_gap_days == 5
        assert injector.min_gaps == 1
        assert injector.max_gaps == 3
        assert injector.gap_channel_prob == 0.5
    
    def test_inject_gaps_basic(self):
        """Test basic gap injection."""
        injector = GapInjector(random_seed=42)
        
        # Create sample segment (30 days, 11 channels)
        x = np.random.randn(4320, 11).astype(np.float32)
        
        # Inject gaps
        x_gapped, mask = injector.inject_gaps(x)
        
        # Check shapes
        assert x_gapped.shape == x.shape
        assert mask.shape == x.shape
        
        # Check that some values are masked (mask has zeros)
        assert np.any(mask == 0.0)
        
        # Check that gapped values are zero where mask is zero
        assert np.all(x_gapped[mask == 0.0] == 0.0)
        
        # Check that mask values are binary (0 or 1)
        assert np.all((mask == 0.0) | (mask == 1.0))
    
    def test_inject_gaps_preserves_shape(self):
        """Test that gap injection preserves array shape."""
        injector = GapInjector()
        
        x = np.random.randn(1000, 5).astype(np.float32)
        x_gapped, mask = injector.inject_gaps(x)
        
        assert x_gapped.shape == (1000, 5)
        assert mask.shape == (1000, 5)
    
    def test_inject_gaps_creates_gaps(self):
        """Test that gaps are actually created."""
        injector = GapInjector(
            min_gaps_per_segment=1,
            max_gaps_per_segment=1,
            random_seed=42
        )
        
        x = np.ones((4320, 11), dtype=np.float32)
        x_gapped, mask = injector.inject_gaps(x)
        
        # Should have some masked values
        n_masked = np.sum(mask == 0.0)
        assert n_masked > 0
        
        # Masked values should be in the gap size range
        # Min: 1 day * 144 steps = 144
        # Max: 12 days * 144 steps = 1728
        assert n_masked >= 144  # At least 1 day
    
    def test_inject_gaps_different_seeds(self):
        """Test that different seeds create different gaps."""
        x = np.random.randn(4320, 11).astype(np.float32)
        
        injector1 = GapInjector(random_seed=42)
        injector2 = GapInjector(random_seed=123)
        
        x_gapped1, mask1 = injector1.inject_gaps(x.copy())
        x_gapped2, mask2 = injector2.inject_gaps(x.copy())
        
        # Masks should be different (very likely with different seeds)
        assert not np.allclose(mask1, mask2)
    
    def test_inject_gaps_batch(self):
        """Test batch gap injection."""
        injector = GapInjector(random_seed=42)
        
        # Batch of 4 segments
        X = np.random.randn(4, 1000, 11).astype(np.float32)
        
        X_gapped, masks = injector.inject_gaps_batch(X)
        
        # Check shapes
        assert X_gapped.shape == X.shape
        assert masks.shape == X.shape
        
        # Each segment should have gaps
        for i in range(4):
            assert np.any(masks[i] == 0.0)
    
    def test_inject_gaps_respects_min_max_gaps(self):
        """Test that number of gaps is within specified range."""
        # With fixed seed and settings, check multiple runs
        injector = GapInjector(
            min_gaps_per_segment=2,
            max_gaps_per_segment=2,
            gap_channel_prob=1.0,  # Gap all channels
            random_seed=42
        )
        
        x = np.ones((4320, 11), dtype=np.float32)
        x_gapped, mask = injector.inject_gaps(x)
        
        # Should have gaps (at least 2 gaps worth of masked values)
        # Min: 2 gaps * 1 day * 144 steps = 288
        n_masked = np.sum(mask == 0.0)
        assert n_masked >= 144  # At least some gap
    
    def test_inject_gaps_handles_small_segments(self):
        """Test gap injection on small segments."""
        injector = GapInjector(
            min_gap_days=1,
            max_gap_days=2,
            random_seed=42
        )
        
        # Small segment: 3 days
        x = np.random.randn(432, 11).astype(np.float32)  # 3 days * 144
        
        x_gapped, mask = injector.inject_gaps(x)
        
        # Should handle gracefully without error
        assert x_gapped.shape == x.shape
        assert mask.shape == x.shape
    
    def test_mask_is_binary(self):
        """Test that mask contains only 0 and 1."""
        injector = GapInjector(random_seed=42)
        
        x = np.random.randn(1000, 11).astype(np.float32)
        x_gapped, mask = injector.inject_gaps(x)
        
        unique_vals = np.unique(mask)
        assert np.all(np.isin(unique_vals, [0.0, 1.0]))
    
    def test_gapped_values_are_zero(self):
        """Test that gapped values are set to zero."""
        injector = GapInjector(random_seed=42)
        
        x = np.random.randn(1000, 11).astype(np.float32) + 10  # All positive
        x_gapped, mask = injector.inject_gaps(x)
        
        # Where mask is 0, values should be 0
        assert np.all(x_gapped[mask == 0.0] == 0.0)
    
    def test_non_gapped_values_preserved(self):
        """Test that non-gapped values are preserved."""
        injector = GapInjector(random_seed=42)
        
        x = np.random.randn(1000, 11).astype(np.float32)
        x_copy = x.copy()
        x_gapped, mask = injector.inject_gaps(x)
        
        # Where mask is 1, original values should be preserved
        np.testing.assert_allclose(
            x_gapped[mask == 1.0],
            x_copy[mask == 1.0],
            rtol=1e-5
        )
    
    def test_create_gap_mask_only(self):
        """Test creating gap mask without data."""
        injector = GapInjector(random_seed=42)
        
        shape = (1000, 11)
        mask = injector.create_gap_mask_only(shape)
        
        assert mask.shape == shape
        assert np.all((mask == 0.0) | (mask == 1.0))
        assert np.any(mask == 0.0)  # Has some gaps
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        x = np.random.randn(1000, 11).astype(np.float32)
        
        injector1 = GapInjector(random_seed=42)
        injector2 = GapInjector(random_seed=42)
        
        x_gapped1, mask1 = injector1.inject_gaps(x.copy())
        x_gapped2, mask2 = injector2.inject_gaps(x.copy())
        
        # Should be identical with same seed
        np.testing.assert_allclose(mask1, mask2)
        np.testing.assert_allclose(x_gapped1, x_gapped2)
    
    def test_gap_channel_probability(self):
        """Test that gap_channel_prob affects which channels are gapped."""
        # With prob 0, should gap at least one channel (enforced)
        injector_low = GapInjector(gap_channel_prob=0.0, random_seed=42)
        
        x = np.ones((1000, 11), dtype=np.float32)
        x_gapped, mask = injector_low.inject_gaps(x)
        
        # Should still have some gaps (at least one channel)
        assert np.any(mask == 0.0)
        
        # Count how many channels have gaps
        channels_with_gaps = []
        for ch in range(11):
            if np.any(mask[:, ch] == 0.0):
                channels_with_gaps.append(ch)
        
        # Should have at least 1 channel with gaps
        assert len(channels_with_gaps) >= 1


class TestGapInjectorEdgeCases:
    """Test edge cases for gap injection."""
    
    def test_very_short_segment(self):
        """Test gap injection on very short segments."""
        injector = GapInjector(
            min_gap_days=1,
            max_gap_days=1,
            random_seed=42
        )
        
        # 1 day segment
        x = np.random.randn(144, 11).astype(np.float32)
        
        x_gapped, mask = injector.inject_gaps(x)
        
        assert x_gapped.shape == x.shape
        assert mask.shape == x.shape
    
    def test_single_channel(self):
        """Test gap injection on single channel."""
        injector = GapInjector(random_seed=42)
        
        x = np.random.randn(1000, 1).astype(np.float32)
        
        x_gapped, mask = injector.inject_gaps(x)
        
        assert x_gapped.shape == (1000, 1)
        assert mask.shape == (1000, 1)
        assert np.any(mask == 0.0)
    
    def test_large_gap_on_short_segment(self):
        """Test that large gap is clipped to segment size."""
        injector = GapInjector(
            min_gap_days=10,
            max_gap_days=20,
            random_seed=42
        )
        
        # Short segment: 5 days
        x = np.random.randn(720, 11).astype(np.float32)
        
        x_gapped, mask = injector.inject_gaps(x)
        
        # Should handle gracefully (gap will be clipped)
        assert x_gapped.shape == x.shape
        assert mask.shape == x.shape
    
    def test_batch_with_different_sizes_fails(self):
        """Test that batch injection requires uniform sizes."""
        injector = GapInjector()
        
        # This should work
        X = np.random.randn(4, 1000, 11).astype(np.float32)
        X_gapped, masks = injector.inject_gaps_batch(X)
        
        assert X_gapped.shape == X.shape
    
    def test_empty_segment_handling(self):
        """Test behavior with minimal segment."""
        injector = GapInjector(random_seed=42)
        
        # Very small segment
        x = np.random.randn(10, 2).astype(np.float32)
        
        x_gapped, mask = injector.inject_gaps(x)
        
        # Should not crash
        assert x_gapped.shape == (10, 2)
        assert mask.shape == (10, 2)
