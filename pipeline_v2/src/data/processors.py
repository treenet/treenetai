"""
Data processing utilities for timestamp conversion, resampling, and merging.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class TimestampProcessor:
    """
    Handles timestamp conversions between local timezone and UTC.
    
    The TreeNet data comes with timestamps in Europe/Zurich timezone,
    but we need everything in UTC for consistent processing.
    """
    
    def __init__(self, local_tz: str = 'Europe/Zurich'):
        """
        Initialize timestamp processor.
        
        Args:
            local_tz: Local timezone string (default: 'Europe/Zurich')
        """
        self.local_tz = local_tz
    
    def to_utc_index(self, ts_series: pd.Series) -> pd.DatetimeIndex:
        """
        Convert a timestamp Series to UTC DatetimeIndex.
        
        This handles both tz-naive and tz-aware inputs safely.
        
        Args:
            ts_series: Series of timestamps
            
        Returns:
            DatetimeIndex in UTC timezone
            
        Example:
            >>> ts = pd.Series(['2020-01-01 12:00:00', '2020-01-01 13:00:00'])
            >>> processor = TimestampProcessor()
            >>> utc_idx = processor.to_utc_index(ts)
            >>> utc_idx.tz
            <UTC>
        """
        # Force to DatetimeIndex first
        dt_index = pd.DatetimeIndex(ts_series)
        
        # Check if already tz-aware
        if getattr(dt_index, 'tz', None) is None:
            # Localize to local timezone
            dt_index = dt_index.tz_localize(
                self.local_tz,
                nonexistent='shift_forward',  # Handle DST gaps
                ambiguous='NaT'               # Handle DST overlaps
            )
        else:
            # Convert to local timezone first, then to UTC
            try:
                dt_index = dt_index.tz_convert(self.local_tz)
            except Exception:
                # If direct conversion fails, hop through UTC
                dt_index = dt_index.tz_convert('UTC').tz_convert(self.local_tz)
        
        # Convert to UTC
        return pd.DatetimeIndex(dt_index.tz_convert('UTC'))
    
    def to_local_series(self, ts_series: pd.Series) -> pd.Series:
        """
        Convert timestamp Series to local timezone.
        
        Args:
            ts_series: Series of timestamps
            
        Returns:
            Series with tz-aware timestamps in local timezone
        """
        dt_index = pd.DatetimeIndex(ts_series)
        
        if getattr(dt_index, 'tz', None) is None:
            dt_index = dt_index.tz_localize(
                self.local_tz,
                nonexistent='shift_forward',
                ambiguous='NaT'
            )
        else:
            try:
                dt_index = dt_index.tz_convert(self.local_tz)
            except Exception:
                dt_index = dt_index.tz_convert('UTC').tz_convert(self.local_tz)
        
        return pd.Series(dt_index)


class DataResampler:
    """
    Resamples time series data to consistent frequencies.
    
    Local sensor data (thermometer, hygrometer, dendrometer) comes in 10-minute resolution.
    Target data (LM) needs to be converted to hourly resolution.
    Global meteo data is daily and needs to be broadcast to 10-minute resolution.
    """
    
    def __init__(self, local_tz: str = 'Europe/Zurich'):
        """
        Initialize data resampler.
        
        Args:
            local_tz: Local timezone for civil time calculations
        """
        self.local_tz = local_tz
        self.ts_processor = TimestampProcessor(local_tz)
    
    def resample_to_10min(
        self, 
        df: pd.DataFrame, 
        value_col: str = 'value'
    ) -> pd.DataFrame:
        """
        Resample data to 10-minute frequency.
        
        Args:
            df: DataFrame with DatetimeIndex and value column
            value_col: Name of the value column
            
        Returns:
            DataFrame resampled to 10-minute frequency
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # Resample to 10 minutes, forward fill within reasonable gaps
        resampled = df[[value_col]].resample('10min').mean()
        return resampled
    
    def resample_to_hourly(
        self, 
        df: pd.DataFrame, 
        method: str = 'subsample'
    ) -> pd.DataFrame:
        """
        Convert 10-minute data to hourly resolution.
        
        Args:
            df: DataFrame with 10-minute data
            method: 'subsample' (take hourly values) or 'mean' (aggregate)
            
        Returns:
            DataFrame with hourly resolution
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        if method == 'subsample':
            # Take only the values at exact hours (xx:00:00)
            # Convert to local time for subsample
            local_idx = df.index.tz_convert(self.local_tz)
            df_local = df.copy()
            df_local.index = local_idx
            
            # Filter for exact hours
            hourly = df_local[
                (df_local.index.minute == 0) & 
                (df_local.index.second == 0)
            ]
            
            # Convert back to UTC
            hourly.index = hourly.index.tz_convert('UTC')
            return hourly
        
        elif method == 'mean':
            # Aggregate to hourly mean
            return df.resample('1h').mean()
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def broadcast_daily_to_10min(
        self, 
        meteo_daily: pd.DataFrame, 
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Broadcast daily meteo data to 10-minute resolution.
        
        Strategy: Map each UTC timestamp to its civil day in local timezone,
        then broadcast the daily values.
        
        Args:
            meteo_daily: DataFrame with daily meteo data (index = civil day)
            target_index: Target 10-minute UTC index
            
        Returns:
            DataFrame with meteo data broadcast to 10-minute resolution
        """
        # Convert target index to local civil days
        civil_days = pd.DatetimeIndex(target_index).tz_convert(
            self.local_tz
        ).normalize()
        
        # Reindex meteo data and forward fill
        broadcast = meteo_daily.reindex(civil_days).ffill()
        
        # Set index to original UTC timestamps
        broadcast.index = target_index
        
        return broadcast


class DataMerger:
    """
    Merges data from multiple sensors and sources into unified input/output arrays.
    
    This handles:
    1. Merging local sensors (temp, RH, stem) at 10-min resolution
    2. Merging global meteo data (broadcast to 10-min)
    3. Adding day-of-year channel
    4. Creating target arrays (hourly resolution)
    """
    
    def __init__(self, local_tz: str = 'Europe/Zurich'):
        """
        Initialize data merger.
        
        Args:
            local_tz: Local timezone for day-of-year calculation
        """
        self.local_tz = local_tz
        self.resampler = DataResampler(local_tz)
    
    def merge_local_sensors_10min(
        self,
        temp_df: pd.DataFrame,
        rh_df: pd.DataFrame,
        stem_df: pd.DataFrame,
        common_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Merge local sensor data to common 10-minute index.
        
        Args:
            temp_df: Temperature data with 'value' column
            rh_df: Relative humidity data with 'value' column
            stem_df: Stem radius change data with 'value' column
            common_index: Common UTC DatetimeIndex to align to
            
        Returns:
            DataFrame with columns: temp_treenet, rh_treenet, stem
        """
        result = pd.DataFrame(index=common_index)
        
        # Reindex each sensor to common index
        if temp_df is not None and len(temp_df) > 0:
            result['temp_treenet'] = temp_df['value'].reindex(common_index)
        else:
            result['temp_treenet'] = np.nan
        
        if rh_df is not None and len(rh_df) > 0:
            result['rh_treenet'] = rh_df['value'].reindex(common_index)
        else:
            result['rh_treenet'] = np.nan
        
        if stem_df is not None and len(stem_df) > 0:
            result['stem'] = stem_df['value'].reindex(common_index)
        else:
            result['stem'] = np.nan
        
        return result
    
    def add_global_meteo(
        self,
        local_df: pd.DataFrame,
        meteo_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add global meteo data to local sensor DataFrame.
        
        Args:
            local_df: DataFrame with local sensors (10-min resolution)
            meteo_daily: Daily meteo data with columns: 
                        tas, tasmax, tasmin, rh, vpd, gh, pr
        
        Returns:
            DataFrame with both local and global channels
        """
        # Broadcast daily meteo to 10-minute resolution
        meteo_10min = self.resampler.broadcast_daily_to_10min(
            meteo_daily, 
            local_df.index
        )
        
        # Merge with local data
        result = pd.concat([local_df, meteo_10min], axis=1)
        
        return result
    
    def add_day_of_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add day-of-year channel based on local timezone.
        
        Args:
            df: DataFrame with UTC DatetimeIndex
            
        Returns:
            DataFrame with additional 'doy' column
        """
        local_idx = df.index.tz_convert(self.local_tz)
        df['doy'] = local_idx.dayofyear
        return df
    
    def create_input_array(
        self,
        temp_df: pd.DataFrame,
        rh_df: pd.DataFrame,
        stem_df: pd.DataFrame,
        meteo_daily: pd.DataFrame,
        common_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Create complete input array with all 11 channels.
        
        Channels (in order):
        0. temp_treenet (local, 10-min)
        1. rh_treenet (local, 10-min)
        2. stem (local, 10-min)
        3. tas (global, daily)
        4. tasmax (global, daily)
        5. tasmin (global, daily)
        6. rh (global, daily)
        7. vpd (global, daily)
        8. gh (global, daily)
        9. pr (global, daily)
        10. doy (day of year)
        
        Args:
            temp_df: Local temperature data
            rh_df: Local relative humidity data
            stem_df: Local stem radius change data
            meteo_daily: Global meteo data (daily)
            common_index: Common 10-minute UTC index
            
        Returns:
            DataFrame with all 11 input channels
        """
        # Merge local sensors
        result = self.merge_local_sensors_10min(
            temp_df, rh_df, stem_df, common_index
        )
        
        # Add global meteo
        result = self.add_global_meteo(result, meteo_daily)
        
        # Add day of year
        result = self.add_day_of_year(result)
        
        return result
    
    def create_target_array(
        self,
        lm_df: pd.DataFrame,
        hourly_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Create target array with 3 channels at hourly resolution.
        
        The LM data contains:
        - value: cleaned stem radius change (10-min)
        - temp: cleaned temperature (hourly, NaN at 10-min)
        - rh: cleaned relative humidity (hourly, NaN at 10-min)
        
        We need to:
        1. Extract hourly temp and rh
        2. Subsample stem to hourly
        
        Channels (in order):
        0. local_T (cleaned temperature, hourly)
        1. local_RH (cleaned relative humidity, hourly)
        2. stem (cleaned stem radius change, hourly)
        
        Args:
            lm_df: LM DataFrame with columns: value, temp, rh
            hourly_index: Target hourly UTC index
            
        Returns:
            DataFrame with 3 target channels at hourly resolution
        """
        result = pd.DataFrame(index=hourly_index)
        
        # Temperature: already hourly in LM data
        if 'temp' in lm_df.columns:
            # Drop NaN values (non-hourly entries)
            temp_hourly = lm_df['temp'].dropna()
            result['local_T'] = temp_hourly.reindex(hourly_index)
        else:
            result['local_T'] = np.nan
        
        # Relative humidity: already hourly in LM data
        if 'rh' in lm_df.columns:
            rh_hourly = lm_df['rh'].dropna()
            result['local_RH'] = rh_hourly.reindex(hourly_index)
        else:
            result['local_RH'] = np.nan
        
        # Stem: subsample from 10-min to hourly
        if 'value' in lm_df.columns:
            stem_hourly = self.resampler.resample_to_hourly(
                lm_df[['value']].rename(columns={'value': 'stem'}),
                method='subsample'
            )
            result['stem'] = stem_hourly['stem'].reindex(hourly_index)
        else:
            result['stem'] = np.nan
        
        return result


class YearGridBuilder:
    """
    Builds complete year grids for alignment and gap detection.
    
    Creates full-year UTC grids at:
    - 10-minute resolution (for inputs)
    - Hourly resolution (for targets)
    """
    
    @staticmethod
    def create_year_grid_10min(year: int) -> pd.DatetimeIndex:
        """
        Create a complete 10-minute grid for a year in UTC.
        
        Args:
            year: Year to create grid for
            
        Returns:
            DatetimeIndex with 10-minute frequency spanning the year
            
        Example:
            >>> grid = YearGridBuilder.create_year_grid_10min(2020)
            >>> len(grid)
            52704  # (366 days * 24 hours * 6 steps/hour for leap year)
        """
        start = pd.Timestamp(f'{year}-01-01 00:00:00', tz='UTC')
        end = pd.Timestamp(f'{year}-12-31 23:50:00', tz='UTC')
        return pd.date_range(start=start, end=end, freq='10min')
    
    @staticmethod
    def create_year_grid_hourly(year: int) -> pd.DatetimeIndex:
        """
        Create a complete hourly grid for a year in UTC.
        
        Args:
            year: Year to create grid for
            
        Returns:
            DatetimeIndex with hourly frequency spanning the year
            
        Example:
            >>> grid = YearGridBuilder.create_year_grid_hourly(2020)
            >>> len(grid)
            8784  # (366 days * 24 hours for leap year)
        """
        start = pd.Timestamp(f'{year}-01-01 00:00:00', tz='UTC')
        end = pd.Timestamp(f'{year}-12-31 23:00:00', tz='UTC')
        return pd.date_range(start=start, end=end, freq='1h')
    
    @staticmethod
    def get_year_range(metadata: pd.DataFrame) -> Tuple[int, int]:
        """
        Get the year range covered by the data.
        
        Args:
            metadata: Metadata DataFrame with series_start and series_stop
            
        Returns:
            Tuple of (min_year, max_year)
        """
        starts = pd.to_datetime(metadata['series_start'], errors='coerce')
        stops = pd.to_datetime(metadata['series_stop'], errors='coerce')
        
        min_year = starts.min().year
        max_year = stops.max().year
        
        return min_year, max_year


class DataProcessor:
    """
    Main data processor combining all processing steps.
    
    This is the high-level interface for converting raw sensor data
    into aligned, merged input/output arrays ready for segmentation.
    """
    
    def __init__(self, local_tz: str = 'Europe/Zurich'):
        """
        Initialize data processor.
        
        Args:
            local_tz: Local timezone for timestamp conversions
        """
        self.local_tz = local_tz
        self.ts_processor = TimestampProcessor(local_tz)
        self.resampler = DataResampler(local_tz)
        self.merger = DataMerger(local_tz)
        self.grid_builder = YearGridBuilder()
    
    def process_sensor_dataframe(
        self, 
        df: pd.DataFrame,
        value_col: str = 'value'
    ) -> pd.DataFrame:
        """
        Process a single sensor DataFrame: convert to UTC and resample.
        
        Args:
            df: Raw sensor DataFrame with 'ts' and value column
            value_col: Name of value column
            
        Returns:
            Processed DataFrame with UTC index and 10-min resolution
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # Convert timestamps to UTC
        utc_index = self.ts_processor.to_utc_index(df['ts'])
        
        # Create DataFrame with UTC index
        result = pd.DataFrame({
            value_col: pd.to_numeric(df[value_col], errors='coerce')
        }, index=utc_index)
        
        # Sort and remove duplicates
        result = result.sort_index()
        result = result[~result.index.duplicated(keep='last')]
        
        return result
    
    def process_meteo_daily(
        self,
        meteo_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process daily meteo data: convert to local civil days.
        
        Args:
            meteo_df: Raw meteo DataFrame with 'ts' column
            
        Returns:
            DataFrame indexed by local civil day
        """
        if meteo_df is None or len(meteo_df) == 0:
            return pd.DataFrame()
        
        # Convert to local timezone
        ts_local = self.ts_processor.to_local_series(meteo_df['ts'])
        
        # Normalize to civil day (midnight)
        civil_day = ts_local.dt.normalize()
        
        # Create result DataFrame
        result = meteo_df.copy()
        result.index = civil_day
        result = result.drop(columns=['ts'])
        
        # Remove duplicates (keep last)
        result = result[~result.index.duplicated(keep='last')]
        result = result.sort_index()
        
        return result
