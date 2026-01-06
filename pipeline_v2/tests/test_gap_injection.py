"""Tests for gap injection module."""

import pytest
import numpy as np
from src.gaps.gap_injection import GapGenerator, GapInjector
from src.config import GapConfig


class TestGapGenerator:
    """Test GapGenerator class."""
    
    def test_generate_gaps_basic(self):
        """Test basic gap generation."""
        config = GapConfig(
            enabled=True,
            min_gap_days=1,
            max_gap_days=3,
            min_gaps_per_segment=1,
            max_gaps_per_segment=2
        )
        
        generator = GapGenerator(config)
        
        segment_length = 4320  # 30 days @ 10-min
        n_channels = 11
        
        gaps = generator.generate_gaps(
            segment_length=segment_length,
            n_channels=n_channels
        )
        
        # Should generate 1-2 gaps
        assert 1 <= len(gaps) <= 2
        
        # Check gap structure
        for gap in gaps:
            assert 'channel' in gap
            assert 'start_idx' in gap
            assert 'end_idx' in gap
            assert 0 <= gap['channel'] < n_channels
            assert 0 <= gap['start_idx'] < gap['end_idx'] <= segment_length
    
    def test_gap_size_range(self):
        """Test that generated gaps are within specified size range."""
        config = GapConfig(
            min_gap_days=1,
            max_gap_days=5
        )
        
        generator = GapGenerator(config)
        
        gaps = generator.generate_gaps(
            segment_length=4320,
            n_channels=11
        )
        
        # Each gap should be between min and max days
        min_steps = 1 * 24 * 6  # 1 day @ 10-min = 144 steps
        max_steps = 5 * 24 * 6  # 5 days @ 10-min = 720 steps
        
        for gap in gaps:
            gap_length = gap['end_idx'] - gap['start_idx']
            assert min_steps <= gap_length <= max_steps
    
    def test_no_overlapping_gaps_same_channel(self):
        """Test that gaps don't overlap in the same channel."""
        config = GapConfig(
            min_gaps_per_segment=3,
            max_gaps_per_segment=5
        )
        
        generator = GapGenerator(config)
        
        gaps = generator.generate_gaps(
            segment_length=4320,
            n_channels=11
        )
        
        # Group gaps by channel
        channel_gaps = {}
        for gap in gaps:
            ch = gap['channel']
            if ch not in channel_gaps:
                channel_gaps[ch] = []
            channel_gaps[ch].append((gap['start_idx'], gap['end_idx']))
        
        # Check no overlaps within each channel
        for ch, ch_gaps in channel_gaps.items():
            ch_gaps.sort()  # Sort by start index
            for i in range(len(ch_gaps) - 1):
                assert ch_gaps[i][1] <= ch_gaps[i+1][0], f"Gaps overlap in channel {ch}"
    
    def test_disabled_gap_generation(self):
        """Test that gap generation can be disabled."""
        config = GapConfig(enabled=False)
        
        generator = GapGenerator(config)
        
        gaps = generator.generate_gaps(
            segment_length=4320,
            n_channels=11
        )
        
        # Should return empty list when disabled
        assert len(gaps) == 0


class TestGapInjector:
    """Test GapInjector class."""
    
    def test_inject_gaps_basic(self, sample_numpy_segment, sample_gap_spec):
        """Test basic gap injection."""
        injector = GapInjector()
        
        # Inject gaps
        X_gapped = injector.inject_gaps(
            X=sample_numpy_segment.copy(),
            gap_spec=sample_gap_spec
        )
        
        # Check shape is preserved
        assert X_gapped.shape == sample_numpy_segment.shape
        
        # Check that gaps are NaN
        for gap in sample_gap_spec:
            ch = gap['channel']
            start = gap['start_idx']
            end = gap['end_idx']
            
            # Gap region should be NaN
            assert np.all(np.isnan(X_gapped[start:end, ch]))
    
    def test_inject_gaps_preserves_non_gap_values(self, sample_numpy_segment, sample_gap_spec):
        """Test that non-gap values are preserved."""
        injector = GapInjector()
        
        X_original = sample_numpy_segment.copy()
        X_gapped = injector.inject_gaps(X=X_original.copy(), gap_spec=sample_gap_spec)
        
        # Check that non-gap values are unchanged
        for ch in range(X_original.shape[1]):
            # Find all timesteps that are NOT in any gap for this channel
            gap_mask = np.zeros(X_original.shape[0], dtype=bool)
            for gap in sample_gap_spec:
                if gap['channel'] == ch:
                    gap_mask[gap['start_idx']:gap['end_idx']] = True
            
            non_gap_mask = ~gap_mask
            
            # Non-gap values should be identical
            np.testing.assert_array_equal(
                X_gapped[non_gap_mask, ch],
                X_original[non_gap_mask, ch]
            )
    
    def test_inject_empty_gap_spec(self, sample_numpy_segment):
        """Test injection with empty gap specification."""
        injector = GapInjector()
        
        X_original = sample_numpy_segment.copy()
        X_gapped = injector.inject_gaps(X=X_original, gap_spec=[])
        
        # Should be unchanged
        np.testing.assert_array_equal(X_gapped, X_original)
    
    def test_compute_gap_coverage(self, sample_numpy_segment):
        """Test gap coverage calculation."""
        injector = GapInjector()
        
        # Create segment with known gaps
        X = sample_numpy_segment.copy()
        
        # Inject gaps manually
        X[100:200, 0] = np.nan  # 100 steps in channel 0
        X[300:400, 1] = np.nan  # 100 steps in channel 1
        
        # Total: 200 NaN values out of (4320 * 11) = 47,520
        expected_coverage = 200 / (4320 * 11)
        
        coverage = injector.compute_gap_coverage(X)
        
        assert abs(coverage - expected_coverage) < 0.001
    
    def test_multiple_channels_same_timestep(self):
        """Test gaps in multiple channels at same timestep."""
        injector = GapInjector()
        
        X = np.random.rand(1000, 11).astype(np.float32)
        
        # Create gaps in different channels at overlapping times
        gap_spec = [
            {'channel': 0, 'start_idx': 100, 'end_idx': 200},
            {'channel': 1, 'start_idx': 150, 'end_idx': 250},
            {'channel': 2, 'start_idx': 180, 'end_idx': 280}
        ]
        
        X_gapped = injector.inject_gaps(X=X, gap_spec=gap_spec)
        
        # Check overlapping region has gaps in multiple channels
        # At timestep 180-200, all three channels should have gaps
        assert np.all(np.isnan(X_gapped[180:200, 0]))
        assert np.all(np.isnan(X_gapped[180:200, 1]))
        assert np.all(np.isnan(X_gapped[180:200, 2]))


class TestGapGeneratorEdgeCases:
    """Test edge cases for gap generation."""
    
    def test_very_short_segment(self):
        """Test gap generation on very short segment."""
        config = GapConfig(
            min_gap_days=1,
            max_gap_days=2
        )
        
        generator = GapGenerator(config)
        
        # Very short segment (only 3 days)
        segment_length = 3 * 24 * 6  # 432 steps
        
        gaps = generator.generate_gaps(
            segment_length=segment_length,
            n_channels=11
        )
        
        # Should still generate valid gaps
        for gap in gaps:
            assert gap['end_idx'] <= segment_length
    
    def test_single_channel(self):
        """Test gap generation for single channel."""
        config = GapConfig()
        
        generator = GapGenerator(config)
        
        gaps = generator.generate_gaps(
            segment_length=4320,
            n_channels=1  # Only one channel
        )
        
        # All gaps should be in channel 0
        for gap in gaps:
            assert gap['channel'] == 0
    
    def test_large_number_of_gaps(self):
        """Test generation of many gaps."""
        config = GapConfig(
            min_gaps_per_segment=10,
            max_gaps_per_segment=15
        )
        
        generator = GapGenerator(config)
        
        gaps = generator.generate_gaps(
            segment_length=4320,
            n_channels=11
        )
        
        # Should generate 10-15 gaps
        assert 10 <= len(gaps) <= 15


class TestGapInjectorEdgeCases:
    """Test edge cases for gap injection."""
    
    def test_full_segment_gap(self):
        """Test injecting gap covering entire segment."""
        injector = GapInjector()
        
        X = np.random.rand(1000, 3).astype(np.float32)
        
        gap_spec = [
            {'channel': 0, 'start_idx': 0, 'end_idx': 1000}
        ]
        
        X_gapped = injector.inject_gaps(X=X, gap_spec=gap_spec)
        
        # Entire channel 0 should be NaN
        assert np.all(np.isnan(X_gapped[:, 0]))
        
        # Other channels should be unchanged
        np.testing.assert_array_equal(X_gapped[:, 1], X[:, 1])
        np.testing.assert_array_equal(X_gapped[:, 2], X[:, 2])
    
    def test_gap_at_boundaries(self):
        """Test gaps at segment boundaries."""
        injector = GapInjector()
        
        X = np.random.rand(1000, 3).astype(np.float32)
        
        gap_spec = [
            {'channel': 0, 'start_idx': 0, 'end_idx': 100},     # At start
            {'channel': 1, 'start_idx': 900, 'end_idx': 1000}   # At end
        ]
        
        X_gapped = injector.inject_gaps(X=X, gap_spec=gap_spec)
        
        # Check gaps at boundaries
        assert np.all(np.isnan(X_gapped[:100, 0]))
        assert np.all(np.isnan(X_gapped[900:, 1]))
    
    def test_zero_coverage_no_gaps(self):
        """Test coverage calculation with no gaps."""
        injector = GapInjector()
        
        X = np.random.rand(1000, 11).astype(np.float32)
        
        coverage = injector.compute_gap_coverage(X)
        
        # Should be zero (no NaN values)
        assert coverage == 0.0
    
    def test_full_coverage_all_nan(self):
        """Test coverage calculation with all NaN."""
        injector = GapInjector()
        
        X = np.full((1000, 11), np.nan, dtype=np.float32)
        
        coverage = injector.compute_gap_coverage(X)
        
        # Should be 1.0 (100% coverage)
        assert coverage == 1.0
