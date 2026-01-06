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
        normalizer = Normalizer()
        
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        params = normalizer.compute_normalization_params(
            df_year=sample_segment_input,
            channels=channels
        )
        
        # Check all channels have params
        for ch in channels:
            assert ch in params
            assert 'min' in params[ch]
            assert 'max' in params[ch]
            assert 'diff' in params[ch]
    
    def test_normalize(self, sample_segment_input, sample_normalization_params):
        """Test normalization."""
        normalizer = Normalizer()
        
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        df_norm = normalizer.normalize(
            df=sample_segment_input,
            params=sample_normalization_params,
            channels=channels
        )
        
        # Check values are in [0, 1] range
        for ch in channels:
            assert df_norm[ch].min() >= -0.01  # Allow small numerical error
            assert df_norm[ch].max() <= 1.01
    
    def test_denormalize(self, sample_normalized_segment, sample_normalization_params):
        """Test denormalization."""
        normalizer = Normalizer()
        
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        df_orig = normalizer.denormalize(
            df_normalized=sample_normalized_segment,
            params=sample_normalization_params,
            channels=channels
        )
        
        # Check denormalized values are in expected range
        assert df_orig['temp_treenet'].min() >= 10.0
        assert df_orig['temp_treenet'].max() <= 20.0
    
    def test_normalize_denormalize_roundtrip(self, sample_segment_input):
        """Test that normalize -> denormalize is reversible."""
        normalizer = Normalizer()
        
        channels = ['temp_treenet', 'rh_treenet', 'stem']
        
        # Compute params
        params = normalizer.compute_normalization_params(
            df_year=sample_segment_input,
            channels=channels
        )
        
        # Normalize
        df_norm = normalizer.normalize(
            df=sample_segment_input,
            params=params,
            channels=channels
        )
        
        # Denormalize
        df_denorm = normalizer.denormalize(
            df_normalized=df_norm,
            params=params,
            channels=channels
        )
        
        # Check roundtrip accuracy
        for ch in channels:
            np.testing.assert_allclose(
                df_denorm[ch].values,
                sample_segment_input[ch].values,
                rtol=1e-5
            )
    
    def test_handle_constant_values(self):
        """Test normalization of constant values."""
        normalizer = Normalizer()
        
        # Create dataframe with constant values
        index = pd.date_range('2021-01-01', periods=100, freq='10min', tz='UTC')
        df = pd.DataFrame({'value': [10.0] * 100}, index=index)
        
        params = normalizer.compute_normalization_params(
            df_year=df,
            channels=['value']
        )
        
        # For constant values, diff should be 0
        assert params['value']['diff'] == 0.0
        
        # Normalization should handle this gracefully
        df_norm = normalizer.normalize(df, params, ['value'])
        assert df_norm['value'].std() == 0.0


class TestSegmentExtractor:
    """Test SegmentExtractor class."""
    
    def test_find_complete_segments(self, sample_segment_input, sample_segment_output):
        """Test finding complete segments."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(config)
        
        segments = extractor.find_complete_segments(
            input_df=sample_segment_input,
            target_df=sample_segment_output
        )
        
        # Should find at least one segment
        assert len(segments) > 0
        
        # Check segment metadata structure
        for seg in segments:
            assert isinstance(seg, SegmentMetadata)
            assert seg.window_start_utc is not None
            assert seg.window_end_utc is not None
            assert seg.input_start_idx >= 0
            assert seg.target_start_idx >= 0
    
    def test_segment_length(self, sample_segment_input, sample_segment_output):
        """Test that extracted segments have correct length."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(config)
        
        segments = extractor.find_complete_segments(
            input_df=sample_segment_input,
            target_df=sample_segment_output
        )
        
        if len(segments) > 0:
            seg = segments[0]
            
            # Extract actual segment
            input_seg = sample_segment_input.iloc[seg.input_start_idx:seg.input_end_idx]
            target_seg = sample_segment_output.iloc[seg.target_start_idx:seg.target_end_idx]
            
            # Check lengths
            assert len(input_seg) == config.input_steps
            assert len(target_seg) == config.output_steps
    
    def test_segment_stride(self):
        """Test that segments are extracted with correct stride."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(config)
        
        # Create long dataset (90 days)
        n_steps_input = 90 * 24 * 6
        n_steps_output = 90 * 24
        
        index_input = pd.date_range('2021-01-01', periods=n_steps_input, freq='10min', tz='UTC')
        index_output = pd.date_range('2021-01-01', periods=n_steps_output, freq='h', tz='UTC')
        
        input_df = pd.DataFrame({'value': range(n_steps_input)}, index=index_input)
        output_df = pd.DataFrame({'value': range(n_steps_output)}, index=index_output)
        
        segments = extractor.find_complete_segments(
            input_df=input_df,
            target_df=output_df
        )
        
        # With 90 days, 30-day segments, 10-day stride:
        # Should get multiple segments
        assert len(segments) > 1
    
    def test_no_complete_segments(self):
        """Test handling when no complete segments can be extracted."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(config)
        
        # Create very short dataset (1 day)
        index_input = pd.date_range('2021-01-01', periods=144, freq='10min', tz='UTC')
        index_output = pd.date_range('2021-01-01', periods=24, freq='h', tz='UTC')
        
        input_df = pd.DataFrame({'value': range(144)}, index=index_input)
        output_df = pd.DataFrame({'value': range(24)}, index=index_output)
        
        segments = extractor.find_complete_segments(
            input_df=input_df,
            target_df=output_df
        )
        
        # Should return empty list or single incomplete segment
        assert isinstance(segments, list)


class TestSegmentMetadata:
    """Test SegmentMetadata dataclass."""
    
    def test_create_metadata(self):
        """Test creating segment metadata."""
        meta = SegmentMetadata(
            window_start_utc=pd.Timestamp('2021-01-01', tz='UTC'),
            window_end_utc=pd.Timestamp('2021-01-31', tz='UTC'),
            input_start_idx=0,
            input_end_idx=4320,
            target_start_idx=0,
            target_end_idx=720
        )
        
        assert meta.window_start_utc.tz.zone == 'UTC'
        assert meta.window_end_utc.tz.zone == 'UTC'
        assert meta.input_end_idx - meta.input_start_idx == 4320
        assert meta.target_end_idx - meta.target_start_idx == 720


class TestEdgeCases:
    """Test edge cases in segmentation."""
    
    def test_segment_with_gaps(self):
        """Test segmentation with data gaps."""
        config = SegmentConfig(segment_days=30, stride_days=10)
        extractor = SegmentExtractor(config)
        
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
            target_df=output_df
        )
        
        assert isinstance(segments, list)
    
    def test_segment_with_nan_values(self):
        """Test normalization with NaN values."""
        normalizer = Normalizer()
        
        # Create data with NaN
        index = pd.date_range('2021-01-01', periods=100, freq='10min', tz='UTC')
        data = np.random.rand(100)
        data[50:55] = np.nan
        
        df = pd.DataFrame({'value': data}, index=index)
        
        # Should handle NaN gracefully
        params = normalizer.compute_normalization_params(
            df_year=df,
            channels=['value']
        )
        
        # Min/max should ignore NaN
        assert np.isfinite(params['value']['min'])
        assert np.isfinite(params['value']['max'])
    
    def test_custom_segment_length(self):
        """Test extraction with custom segment length."""
        config = SegmentConfig(segment_days=45, stride_days=15)
        extractor = SegmentExtractor(config)
        
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
            target_df=output_df
        )
        
        if len(segments) > 0:
            seg = segments[0]
            input_seg = input_df.iloc[seg.input_start_idx:seg.input_end_idx]
            
            # Should have 45-day length
            assert len(input_seg) == 45 * 24 * 6
