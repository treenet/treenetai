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
    Handles year-level normalization for consistent scaling across segments.
    
    Normalization is done at the year level (not per-segment) to ensure
    that all segments from the same year have consistent scales.
    """
    
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
        self.input_steps = segment_days * 24 * steps_per_hour  # 4320 for 30 days
        self.output_steps = segment_days * 24  # 720 for 30 days (hourly)
        self.stride_steps = stride_days * 24 * steps_per_hour  # 1440 for 10 days
    
    def find_complete_segments(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        year: int
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Find all valid 30-day segments with complete coverage.
        
        This implements the "jump-ahead" algorithm:
        1. Start at beginning of year
        2. Check if next 30 days have complete data (no NaN)
        3. If yes: accept segment, advance by stride
        4. If no: find last NaN position, jump to next timestamp, try again
        
        Args:
            input_df: Input DataFrame (10-minute resolution)
            output_df: Output DataFrame (hourly resolution)
            year: Year to extract segments from
            
        Returns:
            List of (start_timestamp, end_timestamp) tuples for valid segments
        """
        # Get year boundaries
        year_start = pd.Timestamp(f'{year}-01-01 00:00:00', tz='UTC')
        year_end = pd.Timestamp(f'{year}-12-31 23:50:00', tz='UTC')
        
        segments = []
        current_start = year_start
        
        while current_start <= year_end:
            # Calculate candidate window end
            candidate_end = current_start + pd.Timedelta(days=self.segment_days)
            
            if candidate_end > year_end:
                break  # Not enough data left for a full segment
            
            # Extract candidate windows
            input_window = input_df.loc[current_start:candidate_end]
            output_window = output_df.loc[current_start:candidate_end]
            
            # Check completeness
            has_input_nans = input_window.isna().any().any()
            has_output_nans = output_window.isna().any().any()
            input_complete = len(input_window) == self.input_steps
            output_complete = len(output_window) == self.output_steps
            
            if (not has_input_nans and not has_output_nans and 
                input_complete and output_complete):
                # Accept this segment
                segments.append((current_start, candidate_end))
                
                # Advance by stride
                current_start = current_start + pd.Timedelta(
                    minutes=self.stride_steps * 10
                )
            else:
                # Find last NaN position in either input or output
                last_nan_idx = None
                
                if has_input_nans:
                    # Find last NaN in input
                    nan_mask = input_window.isna().any(axis=1)
                    if nan_mask.any():
                        last_nan_idx = nan_mask[nan_mask].index[-1]
                
                if has_output_nans:
                    # Find last NaN in output
                    nan_mask = output_window.isna().any(axis=1)
                    if nan_mask.any():
                        output_last_nan = nan_mask[nan_mask].index[-1]
                        if last_nan_idx is None or output_last_nan > last_nan_idx:
                            last_nan_idx = output_last_nan
                
                if last_nan_idx is not None:
                    # Jump to next timestamp after last NaN
                    current_start = last_nan_idx + pd.Timedelta(minutes=10)
                else:
                    # No NaN found, but wrong length - advance by stride
                    current_start = current_start + pd.Timedelta(
                        minutes=self.stride_steps * 10
                    )
        
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
            end: End timestamp (inclusive)
            
        Returns:
            Segment DataFrame
        """
        return df.loc[start:end].copy()


class SegmentBuilder:
    """
    Main segment builder that orchestrates the entire segmentation process.
    
    This combines:
    1. Year-level normalization
    2. Segment extraction with completeness checking
    3. Metadata tracking
    """
    
    def __init__(
        self,
        segment_days: int = 30,
        stride_days: int = 10,
        norm_method: str = 'minmax'
    ):
        """
        Initialize segment builder.
        
        Args:
            segment_days: Length of segments in days
            stride_days: Stride for overlapping segments
            norm_method: Normalization method ('minmax' or 'zscore')
        """
        self.segment_days = segment_days
        self.stride_days = stride_days
        self.norm_method = norm_method
        
        self.normalizer = Normalizer()
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
        year: int,
        input_channels: List[str],
        target_channels: List[str]
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[SegmentMetadata]]:
        """
        Build normalized segments for a single sensor combination.
        
        Process:
        1. Compute year-level normalization parameters
        2. Normalize the full year data
        3. Find valid 30-day windows
        4. Extract segments
        5. Create metadata for each segment
        
        Args:
            combo_id: Unique combination ID
            site_id: Site ID
            thermometer_id: Thermometer series ID
            hygrometer_id: Hygrometer series ID
            dendrometer_id: Dendrometer series ID
            input_df: Full-year input DataFrame (10-min, 11 channels)
            output_df: Full-year output DataFrame (hourly, 3 channels)
            year: Year being processed
            input_channels: List of input channel names
            target_channels: List of target channel names
            
        Returns:
            Tuple of:
            - List of input segment DataFrames
            - List of output segment DataFrames
            - List of SegmentMetadata objects
        """
        # 1. Compute year-level normalization parameters
        input_min, input_diff = self.normalizer.compute_normalization_params(
            input_df, self.norm_method
        )
        output_min, output_diff = self.normalizer.compute_normalization_params(
            output_df, self.norm_method
        )
        
        # 2. Normalize full-year data
        input_normalized = self.normalizer.normalize(
            input_df, input_min, input_diff
        )
        output_normalized = self.normalizer.normalize(
            output_df, output_min, output_diff
        )
        
        # 3. Find valid segments
        segment_windows = self.extractor.find_complete_segments(
            input_normalized, output_normalized, year
        )
        
        # 4. Extract segments and create metadata
        input_segments = []
        output_segments = []
        metadata_list = []
        
        for seg_idx, (start, end) in enumerate(segment_windows):
            # Extract segments
            input_seg = self.extractor.extract_segment(
                input_normalized, start, end
            )
            output_seg = self.extractor.extract_segment(
                output_normalized, start, end
            )
            
            # Create metadata
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
