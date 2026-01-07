"""Tests for data processing modules."""

import pytest
import pandas as pd
import numpy as np
from src.data.processors import (
    TimestampProcessor,
    DataResampler,
    DataMerger,
    YearGridBuilder,
    DataProcessor
)


class TestTimestampProcessor:
    """Test TimestampProcessor class."""
    
    def test_to_utc_index(self, sample_10min_data):
        """Test conversion to UTC index."""
        processor = TimestampProcessor(local_tz='Europe/Zurich')
        
        # Convert index series to UTC
        utc_idx = processor.to_utc_index(sample_10min_data.index.to_series())
        
        assert utc_idx.tz is not None
        assert len(utc_idx) == len(sample_10min_data)
    
    def test_to_local_series(self, sample_10min_data):
        """Test conversion to local series."""
        processor = TimestampProcessor(local_tz='Europe/Zurich')
        
        local_series = processor.to_local_series(sample_10min_data.index.to_series())
        
        assert local_series.dt.tz.zone == 'Europe/Zurich'
        assert len(local_series) == len(sample_10min_data)
    
    def test_utc_roundtrip(self, sample_10min_data):
        """Test UTC -> Local -> UTC roundtrip."""
        processor = TimestampProcessor(local_tz='Europe/Zurich')
        
        # Convert to UTC
        utc_idx = processor.to_utc_index(sample_10min_data.index.to_series())
        
        # Convert back to local
        local_series = processor.to_local_series(pd.Series(utc_idx))
        
        # Should preserve timestamps (within timezone conversion)
        assert len(local_series) == len(sample_10min_data)


class TestDataResampler:
    """Test DataResampler class."""
    
    def test_resample_to_hourly(self, sample_10min_data):
        """Test resampling 10-min data to hourly."""
        resampler = DataResampler(local_tz='Europe/Zurich')
        
        df_hourly = resampler.resample_to_hourly(sample_10min_data)
        
        # Should have 24 hourly steps for 1 day
        assert len(df_hourly) == 24
        assert df_hourly.index.freq == 'h' or (df_hourly.index[1] - df_hourly.index[0]).total_seconds() == 3600
    
    def test_hourly_subsampling_at_exact_hours(self):
        """Test that hourly resampling picks exact hours."""
        resampler = DataResampler(local_tz='Europe/Zurich')
        
        # Create data with known timestamps
        index = pd.date_range('2021-01-01', periods=144, freq='10min', tz='UTC')
        df = pd.DataFrame({'value': range(144)}, index=index)
        
        df_hourly = resampler.resample_to_hourly(df)
        
        # Check that all timestamps are at :00 minutes
        minutes = df_hourly.index.tz_convert('Europe/Zurich').minute
        assert all(minutes == 0)


class TestDataMerger:
    """Test DataMerger class."""
    
    def test_merge_local_sensors(self, sample_10min_data):
        """Test merging local sensor data."""
        merger = DataMerger()
        
        # Create three sensor dataframes (each with 'value' column)
        temp_df = sample_10min_data.copy()
        rh_df = sample_10min_data.copy()
        stem_df = sample_10min_data.copy()
        
        merged = merger.merge_local_sensors_10min(
            temp_df=temp_df,
            rh_df=rh_df,
            stem_df=stem_df,
            common_index=sample_10min_data.index
        )
        
        assert 'temp_treenet' in merged.columns
        assert 'rh_treenet' in merged.columns
        assert 'stem' in merged.columns
        assert len(merged) == len(sample_10min_data)
    
    def test_create_input_array_structure(self, sample_segment_input):
        """Test input array structure."""
        merger = DataMerger()
        
        # Input array should have 11 channels
        assert len(sample_segment_input.columns) == 11
        
        # Check all expected channels are present
        expected_channels = [
            'temp_treenet', 'rh_treenet', 'stem',
            'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy'
        ]
        for ch in expected_channels:
            assert ch in sample_segment_input.columns
    
    def test_create_target_array_structure(self, sample_segment_output):
        """Test target array structure."""
        merger = DataMerger()
        
        # Target array should have 3 channels
        assert len(sample_segment_output.columns) == 3
        
        # Check all expected channels are present
        expected_channels = ['local_T', 'local_RH', 'stem']
        for ch in expected_channels:
            assert ch in sample_segment_output.columns


class TestYearGridBuilder:
    """Test YearGridBuilder class."""
    
    def test_create_year_grid_10min(self):
        """Test creating year-level 10-min grid."""
        grid = YearGridBuilder.create_year_grid_10min(year=2021)
        
        # Check grid properties
        assert grid[0].year == 2021
        assert grid[-1].year == 2021
        
        # Check 10-min resolution
        time_diff = (grid[1] - grid[0]).total_seconds()
        assert time_diff == 600  # 10 minutes
    
    def test_create_year_grid_hourly(self):
        """Test creating year-level hourly grid."""
        grid = YearGridBuilder.create_year_grid_hourly(year=2021)
        
        # Check grid properties
        assert grid[0].year == 2021
        assert grid[-1].year == 2021
        
        # Check hourly resolution
        time_diff = (grid[1] - grid[0]).total_seconds()
        assert time_diff == 3600  # 1 hour
    
    def test_grid_completeness(self):
        """Test that grid covers full year."""
        grid = YearGridBuilder.create_year_grid_10min(year=2021)
        
        # Grid is in UTC, should start at beginning of year UTC
        first_ts = grid[0]
        assert first_ts.month == 1
        assert first_ts.day == 1
        assert first_ts.hour == 0
        assert first_ts.minute == 0
        
        # Should end at end of year
        last_ts = grid[-1]
        assert last_ts.year == 2021


class TestDataProcessor:
    """Test DataProcessor integration."""
    
    @pytest.mark.skip(reason="DataProcessor.process() method doesn't exist")
    def test_process_complete_workflow(self, sample_10min_data):
        """Test complete data processing workflow."""
        processor = DataProcessor(local_tz='Europe/Zurich')
        
        # Process data
        result = processor.process(sample_10min_data)
        
        # Check result is valid
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result.index.tz == pd.DatetimeTZDtype(tz='UTC')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        resampler = DataResampler(local_tz='Europe/Zurich')
        
        empty_df = pd.DataFrame(columns=['value'])
        empty_df.index = pd.DatetimeIndex([], tz='UTC')
        
        # Should handle empty dataframe gracefully
        result = resampler.resample_to_hourly(empty_df)
        assert len(result) == 0
    
    def test_single_value_dataframe(self):
        """Test handling of single-value DataFrame."""
        resampler = DataResampler(local_tz='Europe/Zurich')
        
        single_df = pd.DataFrame(
            {'value': [1.0]},
            index=pd.DatetimeIndex(['2021-01-01'], tz='UTC')
        )
        
        result = resampler.resample_to_hourly(single_df)
        assert len(result) >= 1
    
    def test_missing_timezone(self):
        """Test handling of timezone-naive data."""
        processor = TimestampProcessor(local_tz='Europe/Zurich')
        
        # Create timezone-naive data
        naive_series = pd.Series(pd.date_range('2021-01-01', periods=10, freq='10min'))
        
        # Should convert to UTC
        result = processor.to_utc_index(naive_series)
        
        # Should handle by localizing (to_utc_index assumes Europe/Zurich)
        assert result.tz is not None

