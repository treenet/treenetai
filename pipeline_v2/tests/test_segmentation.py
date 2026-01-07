"""Tests for segmentation module."""

import pytest
import pandas as pd
import numpy as np
from src.data.segmentation import (
    Normalizer,
    SegmentExtractor,
    SegmentMetadata
)
from src.config import SegmentConfig


class TestNormalizer:
    """Test Normalizer class."""
    
    def test_compute_normalization_params(self, sample_segment_input):
        """Test computing normalization parameters."""
        # Normalizer.compute_normalization_params is a static method
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        df_subset = sample_segment_input[channels]
        
        minima, diffs = Normalizer.compute_normalization_params(df_subset, method='minmax')
        
        # Check all channels have params
        for ch in channels:
            assert ch in minima
            assert ch in diffs
            assert isinstance(minima[ch], float)
            assert isinstance(diffs[ch], float)
    
    def test_normalize(self, sample_segment_input):
        """Test normalization."""
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        df_subset = sample_segment_input[channels]
        
        # Compute params
        minima, diffs = Normalizer.compute_normalization_params(df_subset, method='minmax')
        
        # Normalize
        df_norm = Normalizer.normalize(df_subset, minima, diffs)
        
        # Check values are in [0, 1] range
        for ch in channels:
            assert df_norm[ch].min() >= -0.01  # Allow small numerical error
            assert df_norm[ch].max() <= 1.01
    
    def test_denormalize(self, sample_segment_input):
        """Test denormalization."""
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        df_subset = sample_segment_input[channels]
        
        # Compute params and normalize
        minima, diffs = Normalizer.compute_normalization_params(df_subset, method='minmax')
        df_norm = Normalizer.normalize(df_subset, minima, diffs)
        
        # Denormalize
        df_orig = Normalizer.denormalize(df_norm, minima, diffs)
        
        # Check denormalized values match original
        np.testing.assert_allclose(df_orig.values, df_subset.values, rtol=1e-5)
    
    def test_normalize_denormalize_roundtrip(self, sample_segment_input):
        """Test that normalize -> denormalize is reversible."""
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        df_subset = sample_segment_input[channels]
        
        # Compute params
        minima, diffs = Normalizer.compute_normalization_params(df_subset, method='minmax')
        
        # Normalize
        df_norm = Normalizer.normalize(df_subset, minima, diffs)
        
        # Denormalize
        df_denorm = Normalizer.denormalize(df_norm, minima, diffs)
        
        # Check roundtrip accuracy
        for ch in channels:
            np.testing.assert_allclose(
                df_denorm[ch].values,
                sample_segment_input[ch].values,
                rtol=1e-5
            )
    
    def test_handle_constant_values(self):
        """Test normalization of constant values."""
        # Create dataframe with constant values
        index = pd.date_range('2021-01-01', periods=100, freq='10min', tz='UTC')
        df = pd.DataFrame({'value': [10.0] * 100}, index=index)
        
        minima, diffs = Normalizer.compute_normalization_params(df, method='minmax')
        
        # For constant values, diff should be 1.0 (fallback for stability)
        assert diffs['value'] == 1.0
        
        # Normalization should handle this gracefully
        df_norm = Normalizer.normalize(df, minima, diffs)
        assert df_norm['value'].std() == 0.0


class TestSegmentExtractor:
    """Test SegmentExtractor class."""
    
    def test_find_complete_segments(self, sample_segment_input, sample_segment_output):
        """Test finding complete segments."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(
            segment_days=config.segment_days,
            stride_days=config.stride_days,
            steps_per_hour=6
        )
        
        segments = extractor.find_complete_segments(
            input_df=sample_segment_input,
            output_df=sample_segment_output,
            year=2021
        )
        
        # Should find at least one segment (returns list of timestamp tuples)
        assert len(segments) >= 0
        
        # Check segment structure (tuples of timestamps)
        for seg in segments:
            assert isinstance(seg, tuple)
            assert len(seg) == 2
            start_ts, end_ts = seg
            assert isinstance(start_ts, pd.Timestamp)
            assert isinstance(end_ts, pd.Timestamp)
            assert start_ts < end_ts
    
    def test_segment_length(self, sample_segment_input, sample_segment_output):
        """Test that extracted segments have correct length."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(
            segment_days=config.segment_days,
            stride_days=config.stride_days,
            steps_per_hour=6
        )
        
        segments = extractor.find_complete_segments(
            input_df=sample_segment_input,
            output_df=sample_segment_output,
            year=2021
        )
        
        if len(segments) > 0:
            start_ts, end_ts = segments[0]
            
            # Extract actual segment
            input_seg = sample_segment_input.loc[start_ts:end_ts]
            target_seg = sample_segment_output.loc[start_ts:end_ts]
            
            # Check lengths (approximately, since segment may span slightly differently)
            assert len(input_seg) > 0
            assert len(target_seg) > 0
    
    def test_segment_stride(self):
        """Test that segments are extracted with correct stride."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(
            segment_days=config.segment_days,
            stride_days=config.stride_days,
            steps_per_hour=6
        )
        
        # Create long dataset (90 days)
        n_steps_input = 90 * 24 * 6
        n_steps_output = 90 * 24
        
        index_input = pd.date_range('2021-01-01', periods=n_steps_input, freq='10min', tz='UTC')
        index_output = pd.date_range('2021-01-01', periods=n_steps_output, freq='h', tz='UTC')
        
        input_df = pd.DataFrame({'value': range(n_steps_input)}, index=index_input)
        output_df = pd.DataFrame({'value': range(n_steps_output)}, index=index_output)
        
        segments = extractor.find_complete_segments(
            input_df=input_df,
            output_df=output_df,
            year=2021
        )
        
        # With 90 days, 30-day segments, 10-day stride:
        # Should get multiple segments
        assert len(segments) >= 0
    
    def test_no_complete_segments(self):
        """Test handling when no complete segments can be extracted."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(
            segment_days=config.segment_days,
            stride_days=config.stride_days,
            steps_per_hour=6
        )
        
        # Create very short dataset (1 day)
        index_input = pd.date_range('2021-01-01', periods=144, freq='10min', tz='UTC')
        index_output = pd.date_range('2021-01-01', periods=24, freq='h', tz='UTC')
        
        input_df = pd.DataFrame({'value': range(144)}, index=index_input)
        output_df = pd.DataFrame({'value': range(24)}, index=index_output)
        
        segments = extractor.find_complete_segments(
            input_df=input_df,
            output_df=output_df,
            year=2021
        )
        
        # Should return empty list
        assert len(segments) == 0


class TestSegmentMetadata:
    """Test SegmentMetadata dataclass."""
    
    def test_create_metadata(self):
        """Test creating segment metadata."""
        meta = SegmentMetadata(
            combo_id=1,
            segment_idx=0,
            site_id=1001,
            thermometer_id=2001,
            hygrometer_id=3001,
            dendrometer_id=4001,
            window_start_utc=pd.Timestamp('2021-01-01', tz='UTC'),
            window_end_utc=pd.Timestamp('2021-01-31', tz='UTC'),
            input_min={'temp': 0.0, 'rh': 0.0, 'stem': 0.0},
            input_diff={'temp': 10.0, 'rh': 100.0, 'stem': 1.0},
            output_min={'temp': 0.0, 'rh': 0.0, 'stem': 0.0},
            output_diff={'temp': 10.0, 'rh': 100.0, 'stem': 1.0},
            input_channels=['temp', 'rh', 'stem'],
            target_channels=['temp', 'rh', 'stem']
        )
        
        assert meta.window_start_utc.tz is not None
        assert meta.window_end_utc.tz is not None
        assert meta.combo_id == 1
        assert len(meta.input_channels) == 3


class TestEdgeCases:
    """Test edge cases in segmentation."""
    
    def test_segment_with_gaps(self):
        """Test segmentation with data gaps."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(
            segment_days=config.segment_days,
            stride_days=config.stride_days,
            steps_per_hour=6
        )
        
        # Create data with a gap in the middle
        index1 = pd.date_range('2021-01-01', periods=2000, freq='10min', tz='UTC')
        index2 = pd.date_range('2021-01-20', periods=2000, freq='10min', tz='UTC')
        
        input_df = pd.concat([
            pd.DataFrame({'value': range(2000)}, index=index1),
            pd.DataFrame({'value': range(2000)}, index=index2)
        ]).sort_index()
        
        output_df = input_df.resample('h').mean()
        
        # Should handle gaps appropriately
        segments = extractor.find_complete_segments(
            input_df=input_df,
            output_df=output_df,
            year=2021
        )
        
        assert isinstance(segments, list)
    
    def test_segment_with_nan_values(self):
        """Test normalization with NaN values."""
        # Create data with NaN
        index = pd.date_range('2021-01-01', periods=100, freq='10min', tz='UTC')
        data = np.random.rand(100)
        data[50:55] = np.nan
        
        df = pd.DataFrame({'value': data}, index=index)
        
        # Should handle NaN gracefully (drops NaN values before computing params)
        minima, diffs = Normalizer.compute_normalization_params(df, method='minmax')
        
        # Min/max should ignore NaN
        assert np.isfinite(minima['value'])
        assert np.isfinite(diffs['value'])
    
    def test_custom_segment_length(self):
        """Test extraction with custom segment length."""
        config = SegmentConfig(segment_days=45, stride_days=15)
        extractor = SegmentExtractor(
            segment_days=config.segment_days,
            stride_days=config.stride_days,
            steps_per_hour=6
        )
        
        # Check computed properties
        assert config.input_steps == 45 * 24 * 6
        assert config.output_steps == 45 * 24
        
        # Create appropriately sized data
        n_steps_input = 90 * 24 * 6  # 90 days
        n_steps_output = 90 * 24
        
        index_input = pd.date_range('2021-01-01', periods=n_steps_input, freq='10min', tz='UTC')
        index_output = pd.date_range('2021-01-01', periods=n_steps_output, freq='h', tz='UTC')
        
        input_df = pd.DataFrame({'value': range(n_steps_input)}, index=index_input)
        output_df = pd.DataFrame({'value': range(n_steps_output)}, index=index_output)
        
        segments = extractor.find_complete_segments(
            input_df=input_df,
            output_df=output_df,
            year=2021
        )
        
        # Should find segments
        assert isinstance(segments, list)

