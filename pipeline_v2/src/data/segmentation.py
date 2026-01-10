"""
Segmentation and normalization for creating 30-day training/test segments.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import pickle


@dataclass
class SegmentMetadata:
    """
    Metadata for a single segment.
    
    Attributes:
        combo_id: Combination ID (unique per site-sensor combination)
        segment_idx: Index of this segment within the combination
        site_id: Site ID
        thermometer_id: Thermometer series ID
        hygrometer_id: Hygrometer series ID
        dendrometer_id: Dendrometer series ID
        window_start_utc: Start timestamp (UTC)
        window_end_utc: End timestamp (UTC)
        input_min: Normalization minima for input channels (dict)
        input_diff: Normalization ranges for input channels (dict)
        output_min: Normalization minima for output channels (dict)
        output_diff: Normalization ranges for output channels (dict)
        input_channels: List of input channel names
        target_channels: List of target channel names
    """
    combo_id: int
    segment_idx: int
    site_id: int
    thermometer_id: int
    hygrometer_id: int
    dendrometer_id: int
    window_start_utc: pd.Timestamp
    window_end_utc: pd.Timestamp
    input_min: Dict[str, float]
    input_diff: Dict[str, float]
    output_min: Dict[str, float]
    output_diff: Dict[str, float]
    input_channels: List[str]
    target_channels: List[str]


class Normalizer:
    """
    Handles normalization for consistent scaling.
    
    Supports two normalization scopes:
    - 'year': Normalization at the year level for consistent scales across segments
    - 'segment': Normalization per-segment for local adaptation (handles jumps better)
    
    For stem radius channel, special handling is applied to align input and output
    scales since they come from different data sources (dendrometer_l2 vs dendrometer_lm).
    """
    
    def __init__(self, norm_scope: str = 'year'):
        """
        Initialize normalizer.
        
        Args:
            norm_scope: 'year' for year-level normalization, 'segment' for segment-level
        """
        self.norm_scope = norm_scope
    
    @staticmethod
    def align_stem_signals_yearly(
        input_stem: pd.Series,
        output_stem: pd.Series
    ) -> tuple:
        """
        Align input and output stem signals for consistent normalization.
        
        This addresses the issue where input stem (from dendrometer_l2) and 
        output stem (from dendrometer_lm) have different baseline values.
        
        Process for each year:
        1. Find first common valid timestamp where both signals have non-NaN values
        2. Shift both signals so they start from zero at this timestamp
        3. Find last common valid timestamp near year end
        4. Compute normalization params from the aligned interval
        
        The starting point (offset) of stem radius is physically irrelevant,
        so this alignment preserves physical meaning while enabling consistent normalization.
        
        Args:
            input_stem: Input stem Series (10-min resolution)
            output_stem: Output stem Series (hourly resolution)
            
        Returns:
            Tuple of (aligned_input_stem, aligned_output_stem, 
                      input_min, input_diff, output_min, output_diff)
        """
        # Group by year
        input_years = input_stem.index.year.unique()
        output_years = output_stem.index.year.unique()
        common_years = sorted(set(input_years) & set(output_years))
        
        if len(common_years) == 0:
            # No overlap - fall back to independent normalization
            return input_stem, output_stem, None, None, None, None
        
        aligned_input = input_stem.copy()
        aligned_output = output_stem.copy()
        
        # Track all shifts for consistency check
        all_input_shifts = []
        all_output_shifts = []
        
        for year in common_years:
            # Get year slices
            input_year_mask = input_stem.index.year == year
            output_year_mask = output_stem.index.year == year
            
            input_year = input_stem[input_year_mask].dropna()
            output_year = output_stem[output_year_mask].dropna()
            
            if len(input_year) == 0 or len(output_year) == 0:
                continue
            
            # Find first common valid timestamp
            # Need to align to hourly for comparison (input is 10-min, output is hourly)
            input_hourly_times = input_year.index.floor('h').unique()
            output_times = output_year.index
            
            common_times = sorted(set(input_hourly_times) & set(output_times))
            
            if len(common_times) == 0:
                continue
            
            first_common = common_times[0]
            last_common = common_times[-1]
            
            # Get values at first common timestamp
            # For input, get the value at the hour mark
            input_at_first = input_year[input_year.index.floor('h') == first_common]
            if len(input_at_first) > 0:
                input_shift = input_at_first.iloc[0]
            else:
                continue
                
            output_at_first = output_year.loc[first_common] if first_common in output_year.index else None
            if output_at_first is None or pd.isna(output_at_first):
                continue
            output_shift = output_at_first
            
            # Apply shifts for this year
            aligned_input.loc[input_year_mask] = input_stem[input_year_mask] - input_shift
            aligned_output.loc[output_year_mask] = output_stem[output_year_mask] - output_shift
            
            all_input_shifts.append(input_shift)
            all_output_shifts.append(output_shift)
        
        # Compute normalization params from aligned signals
        input_vals = aligned_input.dropna()
        output_vals = aligned_output.dropna()
        
        if len(input_vals) == 0 or len(output_vals) == 0:
            return input_stem, output_stem, None, None, None, None
        
        input_min = float(input_vals.min())
        input_max = float(input_vals.max())
        input_diff = input_max - input_min
        if input_diff < 1e-8:
            input_diff = 1.0
        
        output_min = float(output_vals.min())
        output_max = float(output_vals.max())
        output_diff = output_max - output_min
        if output_diff < 1e-8:
            output_diff = 1.0
        
        return aligned_input, aligned_output, input_min, input_diff, output_min, output_diff
    
    @staticmethod
    def compute_normalization_params(
        df: pd.DataFrame,
        method: str = 'minmax'
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute normalization parameters for all columns.
        
        Args:
            df: DataFrame with data to normalize
            method: 'minmax' (scale to [0,1]) or 'zscore' (standardize)
            
        Returns:
            Tuple of (minima_dict, diff_dict) for each column
            - For minmax: min and (max - min)
            - For zscore: mean and std
        """
        minima = {}
        diffs = {}
        
        for col in df.columns:
            values = df[col].dropna()
            
            if len(values) == 0:
                minima[col] = 0.0
                diffs[col] = 1.0
                continue
            
            if method == 'minmax':
                vmin = float(values.min())
                vmax = float(values.max())
                diff = vmax - vmin
                
                if diff < 1e-8:  # Constant column
                    diff = 1.0
                
                minima[col] = vmin
                diffs[col] = diff
            
            elif method == 'zscore':
                mean = float(values.mean())
                std = float(values.std())
                
                if std < 1e-8:  # Constant column
                    std = 1.0
                
                minima[col] = mean
                diffs[col] = std
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        return minima, diffs
    
    @staticmethod
    def normalize(
        df: pd.DataFrame,
        minima: Dict[str, float],
        diffs: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Apply normalization to DataFrame.
        
        Formula: normalized = (value - min) / diff
        
        Args:
            df: DataFrame to normalize
            minima: Dictionary of minima for each column
            diffs: Dictionary of ranges for each column
            
        Returns:
            Normalized DataFrame
        """
        result = df.copy()
        
        for col in df.columns:
            if col in minima and col in diffs:
                vmin = minima[col]
                vdiff = diffs[col]
                
                if np.isfinite(vdiff) and vdiff > 1e-8:
                    result[col] = (df[col] - vmin) / vdiff
                else:
                    result[col] = df[col] - (0.0 if not np.isfinite(vmin) else vmin)
        
        return result
    
    @staticmethod
    def denormalize(
        df: pd.DataFrame,
        minima: Dict[str, float],
        diffs: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Reverse normalization.
        
        Formula: original = normalized * diff + min
        
        Args:
            df: Normalized DataFrame
            minima: Dictionary of minima for each column
            diffs: Dictionary of ranges for each column
            
        Returns:
            Denormalized DataFrame
        """
        result = df.copy()
        
        for col in df.columns:
            if col in minima and col in diffs:
                vmin = minima[col]
                vdiff = diffs[col]
                
                if np.isfinite(vdiff) and vdiff > 1e-8:
                    result[col] = df[col] * vdiff + vmin
                else:
                    result[col] = df[col] + (0.0 if not np.isfinite(vmin) else vmin)
        
        return result


class SegmentExtractor:
    """
    Extracts 30-day segments with strict completeness requirement.
    
    Uses "jump-ahead" logic: if any NaN is found in the candidate window,
    jump to the first timestamp after the last NaN and try again.
    """
    
    def __init__(
        self,
        segment_days: int = 30,
        stride_days: int = 10,
        steps_per_hour: int = 6
    ):
        """
        Initialize segment extractor.
        
        Args:
            segment_days: Length of each segment in days
            stride_days: Overlap between consecutive segments (in days)
            steps_per_hour: Number of 10-minute steps per hour
        """
        self.segment_days = segment_days
        self.stride_days = stride_days
        self.steps_per_hour = steps_per_hour
        
        # Calculate steps
        self.input_steps = segment_days * 24 * steps_per_hour  # 4320 for 30 days (10-min resolution)
        self.output_steps = segment_days * 24  # 720 for 30 days (hourly resolution)
        self.stride_steps_output = stride_days * 24  # 240 for 10 days (hourly resolution)
        self.stride_steps_input = stride_days * 24 * steps_per_hour  # 1440 for 10 days (10-min resolution)
    
    def find_complete_segments(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        verbose: bool = False
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Find all valid 30-day segments with complete coverage from ALL available data.
        
        Algorithm (matching reference notebook):
        1. Drop any NaN rows first
        2. Iterate through output timestamps with stride
        3. Check if segment_length timesteps ahead exists and spans exactly 30 days
        4. Check if start/stop times exist in input
        5. Check if input has exactly segment_length*6 samples between start/stop
        
        Args:
            input_df: Input DataFrame (10-minute resolution)
            output_df: Output DataFrame (hourly resolution)
            verbose: Print debug information
            
        Returns:
            List of (start_timestamp, end_timestamp) tuples for valid segments
        """
        # Drop NaN rows first (matching reference notebook)
        input_clean = input_df.dropna()
        output_clean = output_df.dropna()
        
        if verbose:
            print(f"    Input: {len(input_df)} rows -> {len(input_clean)} after dropna()")
            print(f"    Output: {len(output_df)} rows -> {len(output_clean)} after dropna()")
        
        if len(output_clean) == 0 or len(input_clean) == 0:
            return []
        
        segments = []
        idx = 0
        idx_change = 1  # Initialize idx_change (reference algorithm behavior)
        max_length = len(output_clean)
        stride_length = self.stride_steps_output  # Use OUTPUT stride (240 for 10-day stride with hourly output)
        
        input_times = pd.DatetimeIndex(input_clean.index)
        output_times = pd.DatetimeIndex(output_clean.index)
        
        checked_segments = 0
        time_diff_failed = 0
        input_missing_failed = 0
        sample_count_failed = 0
        
        while (idx + self.output_steps) < max_length:
            start_time = output_clean.iloc[idx].name  # Get timestamp from index
            stop_time = output_clean.iloc[idx + self.output_steps].name
            
            checked_segments += 1
            
            # Check if time difference is exactly segment_days
            if (stop_time - start_time) == pd.Timedelta(days=self.segment_days):
                # Check if start and stop exist in input
                if (start_time in input_times) and (stop_time in input_times):
                    # Find input indices
                    index_a = input_clean.index.get_loc(start_time)
                    index_b = input_clean.index.get_loc(stop_time)
                    
                    actual_samples = index_b - index_a
                    
                    # Check if input segment has exactly the right number of samples
                    if actual_samples == self.input_steps:
                        segments.append((start_time, stop_time))
                        idx_change = stride_length  # Advance by stride (240 steps for 10 days)
                        if verbose and len(segments) <= 3:
                            print(f"      Found segment {len(segments)}: {start_time} to {stop_time}")
                    else:
                        sample_count_failed += 1
                        idx_change = 1  # Reset to single step
                        if verbose and sample_count_failed <= 5:  # Show first 5 failures
                            print(f"      Sample count mismatch: {start_time} to {stop_time}")
                            print(f"        Expected: {self.input_steps}, Got: {actual_samples}")
                else:
                    input_missing_failed += 1
                    idx_change = 1  # Reset to single step
            else:
                time_diff_failed += 1
                idx_change = 1  # Reset to single step
            
            idx += idx_change
        
        if verbose:
            print(f"    Checked {checked_segments} potential segments")
            print(f"    Time diff failed: {time_diff_failed}")
            print(f"    Input missing failed: {input_missing_failed}")
            print(f"    Sample count failed: {sample_count_failed}")
            print(f"    Valid segments: {len(segments)}")
        
        return segments
    
    def extract_segment(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Extract a single segment from DataFrame.
        
        Args:
            df: Source DataFrame
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive - not included in output)
            
        Returns:
            Segment DataFrame with exactly the expected number of samples
        """
        # Use iloc to get exclusive end behavior
        start_idx = df.index.get_loc(start)
        end_idx = df.index.get_loc(end)
        return df.iloc[start_idx:end_idx].copy()


class SegmentBuilder:
    """
    Main segment builder that orchestrates the entire segmentation process.
    
    This combines:
    1. Normalization (year-level or segment-level)
    2. Segment extraction with completeness checking
    3. Metadata tracking
    """
    
    def __init__(
        self,
        segment_days: int = 30,
        stride_days: int = 10,
        norm_method: str = 'minmax',
        norm_scope: str = 'year'
    ):
        """
        Initialize segment builder.
        
        Args:
            segment_days: Length of segments in days
            stride_days: Stride for overlapping segments
            norm_method: Normalization method ('minmax' or 'zscore')
            norm_scope: Normalization scope ('year' or 'segment')
        """
        self.segment_days = segment_days
        self.stride_days = stride_days
        self.norm_method = norm_method
        self.norm_scope = norm_scope
        
        self.normalizer = Normalizer(norm_scope=norm_scope)
        self.extractor = SegmentExtractor(segment_days, stride_days)
    
    def build_segments_for_combination(
        self,
        combo_id: int,
        site_id: int,
        thermometer_id: int,
        hygrometer_id: int,
        dendrometer_id: int,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        input_channels: List[str],
        target_channels: List[str]
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[SegmentMetadata]]:
        """
        Build normalized segments for a single sensor combination.
        
        Process (year-level normalization):
        1. Compute normalization parameters from all available data
        2. Normalize the full data
        3. Find valid segments (complete 30-day windows)
        4. Extract segments
        5. Create metadata for each segment
        
        Process (segment-level normalization):
        1. Find valid segments (complete 30-day windows) from raw data
        2. Extract raw segments
        3. Normalize each segment individually
        4. Store segment-specific normalization parameters in metadata
        
        Args:
            combo_id: Unique combination ID
            site_id: Site ID
            thermometer_id: Thermometer series ID
            hygrometer_id: Hygrometer series ID
            dendrometer_id: Dendrometer series ID
            input_df: Full input DataFrame (10-min, 11 channels)
            output_df: Full output DataFrame (hourly, 3 channels)
            input_channels: List of input channel names
            target_channels: List of target channel names
            
        Returns:
            Tuple of:
            - List of input segment DataFrames
            - List of output segment DataFrames
            - List of SegmentMetadata objects
        """
        input_segments = []
        output_segments = []
        metadata_list = []
        
        if self.norm_scope == 'year':
            # Year-level normalization with aligned stem handling
            
            # 1. Handle stem channel specially - align input and output scales
            if 'stem' in input_df.columns and 'stem' in output_df.columns:
                (aligned_input_stem, aligned_output_stem,
                 stem_input_min, stem_input_diff,
                 stem_output_min, stem_output_diff) = self.normalizer.align_stem_signals_yearly(
                    input_df['stem'], output_df['stem']
                )
                
                # Create working copies with aligned stem
                input_df_aligned = input_df.copy()
                output_df_aligned = output_df.copy()
                input_df_aligned['stem'] = aligned_input_stem
                output_df_aligned['stem'] = aligned_output_stem
            else:
                input_df_aligned = input_df
                output_df_aligned = output_df
                stem_input_min, stem_input_diff = None, None
                stem_output_min, stem_output_diff = None, None
            
            # 2. Compute normalization parameters for all channels
            input_min, input_diff = self.normalizer.compute_normalization_params(
                input_df_aligned, self.norm_method
            )
            output_min, output_diff = self.normalizer.compute_normalization_params(
                output_df_aligned, self.norm_method
            )
            
            # Use aligned stem params if computed
            if stem_input_min is not None:
                input_min['stem'] = stem_input_min
                input_diff['stem'] = stem_input_diff
            if stem_output_min is not None:
                output_min['stem'] = stem_output_min
                output_diff['stem'] = stem_output_diff
            
            # 3. Normalize full data
            input_normalized = self.normalizer.normalize(
                input_df_aligned, input_min, input_diff
            )
            output_normalized = self.normalizer.normalize(
                output_df_aligned, output_min, output_diff
            )
            
            # 3. Find valid segments
            segment_windows = self.extractor.find_complete_segments(
                input_normalized, output_normalized, verbose=False
            )
            
            # 4. Extract segments and create metadata
            for seg_idx, (start, end) in enumerate(segment_windows):
                # Extract segments
                input_seg = self.extractor.extract_segment(
                    input_normalized, start, end
                )
                output_seg = self.extractor.extract_segment(
                    output_normalized, start, end
                )
                
                # Create metadata with year-level normalization params
                metadata = SegmentMetadata(
                    combo_id=combo_id,
                    segment_idx=seg_idx,
                    site_id=site_id,
                    thermometer_id=thermometer_id,
                    hygrometer_id=hygrometer_id,
                    dendrometer_id=dendrometer_id,
                    window_start_utc=start,
                    window_end_utc=end,
                    input_min=input_min,
                    input_diff=input_diff,
                    output_min=output_min,
                    output_diff=output_diff,
                    input_channels=input_channels,
                    target_channels=target_channels
                )
                
                input_segments.append(input_seg)
                output_segments.append(output_seg)
                metadata_list.append(metadata)
        
        elif self.norm_scope == 'segment':
            # Segment-level normalization (new approach)
            # 1. Find valid segments from raw (non-normalized) data
            segment_windows = self.extractor.find_complete_segments(
                input_df, output_df, verbose=False
            )
            
            # 2. Extract and normalize each segment individually
            for seg_idx, (start, end) in enumerate(segment_windows):
                # Extract raw segments
                input_seg_raw = self.extractor.extract_segment(
                    input_df, start, end
                )
                output_seg_raw = self.extractor.extract_segment(
                    output_df, start, end
                )
                
                # Compute segment-specific normalization parameters
                input_min, input_diff = self.normalizer.compute_normalization_params(
                    input_seg_raw, self.norm_method
                )
                output_min, output_diff = self.normalizer.compute_normalization_params(
                    output_seg_raw, self.norm_method
                )
                
                # Normalize this segment
                input_seg = self.normalizer.normalize(
                    input_seg_raw, input_min, input_diff
                )
                output_seg = self.normalizer.normalize(
                    output_seg_raw, output_min, output_diff
                )
                
                # Create metadata with segment-specific normalization params
                metadata = SegmentMetadata(
                    combo_id=combo_id,
                    segment_idx=seg_idx,
                    site_id=site_id,
                    thermometer_id=thermometer_id,
                    hygrometer_id=hygrometer_id,
                    dendrometer_id=dendrometer_id,
                    window_start_utc=start,
                    window_end_utc=end,
                    input_min=input_min,
                    input_diff=input_diff,
                    output_min=output_min,
                    output_diff=output_diff,
                    input_channels=input_channels,
                    target_channels=target_channels
                )
                
                input_segments.append(input_seg)
                output_segments.append(output_seg)
                metadata_list.append(metadata)
        
        else:
            raise ValueError(f"Unknown norm_scope: {self.norm_scope}. Must be 'year' or 'segment'")
        
        return input_segments, output_segments, metadata_list
        
        return input_segments, output_segments, metadata_list
    
    def segments_to_numpy(
        self,
        segment_list: List[pd.DataFrame]
    ) -> np.ndarray:
        """
        Convert list of segment DataFrames to single numpy array.
        
        Args:
            segment_list: List of segment DataFrames
            
        Returns:
            Numpy array of shape (n_segments, timesteps, channels)
        """
        if not segment_list:
            return np.array([])
        
        arrays = [seg.values for seg in segment_list]
        return np.stack(arrays, axis=0)
    
    def save_segments(
        self,
        output_dir: Path,
        split: str,
        input_segments: Dict[int, List[pd.DataFrame]],
        output_segments: Dict[int, List[pd.DataFrame]],
        metadata: List[SegmentMetadata],
        combo_ids: Dict[int, pd.Series]
    ) -> None:
        """
        Save segments and metadata to disk.
        
        Creates:
        - {split}_input_segments.pkl (DataFrames)
        - {split}_output_segments.pkl (DataFrames)
        - {split}_input_segments_numpy.pkl (numpy arrays)
        - {split}_output_segments_numpy.pkl (numpy arrays)
        - {split}_segment_ids.pkl (metadata)
        - model_{split}_data_combination_ids.pkl (combo IDs)
        
        Args:
            output_dir: Output directory
            split: 'train' or 'test'
            input_segments: Dict mapping combo_id -> list of input DataFrames
            output_segments: Dict mapping combo_id -> list of output DataFrames
            metadata: List of SegmentMetadata objects
            combo_ids: Dict mapping combo_id -> Series with sensor IDs
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame versions
        with open(output_dir / f'{split}_input_segments.pkl', 'wb') as f:
            pickle.dump(input_segments, f)
        
        with open(output_dir / f'{split}_output_segments.pkl', 'wb') as f:
            pickle.dump(output_segments, f)
        
        # Convert to numpy and save
        all_input_segs = []
        all_output_segs = []
        
        for combo_id in sorted(input_segments.keys()):
            all_input_segs.extend(input_segments[combo_id])
            all_output_segs.extend(output_segments[combo_id])
        
        if all_input_segs:
            input_numpy = self.segments_to_numpy(all_input_segs)
            with open(output_dir / f'{split}_input_segments_numpy.pkl', 'wb') as f:
                pickle.dump(input_numpy, f)
        
        if all_output_segs:
            output_numpy = self.segments_to_numpy(all_output_segs)
            with open(output_dir / f'{split}_output_segments_numpy.pkl', 'wb') as f:
                pickle.dump(output_numpy, f)
        
        # Save metadata
        with open(output_dir / f'{split}_segment_ids.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save combo IDs
        with open(output_dir / f'model_{split}_data_combination_ids.pkl', 'wb') as f:
            pickle.dump(combo_ids, f)
