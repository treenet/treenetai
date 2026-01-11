#!/usr/bin/env python3
"""
Reconstruct complete multi-year time series by filling gaps using trained model.

PATH 1 Implementation:
- Takes any 11-channel 10-min sensor combination (L2/L1 raw data)
- Uses trained TCN model to produce 3-channel 1-hour clean output (LM quality)
- Fills gaps <= max_gap_days in the process

The pipeline:
1. Load raw sensor data for a site/sensor combination
2. Identify gaps in the input data
3. For each gap, create a 30-day segment centered on the gap
4. Run model inference to get clean hourly output
5. Stitch together segments to create complete time series
6. Save reconstructed output

Usage:
    python 6_reconstruct_timeseries_v2.py \
        --model-path /path/to/best_model.keras \
        --site-id 3 \
        --thermo-id 9 \
        --hygro-id 7 \
        --dendro-id 18 \
        --output-dir /home/lukovic/data/treenet/reconstructions

Author: Lukovic
Date: 2026-01-11
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import timedelta
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding
from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor
from src.data.segmentation import Normalizer
from src.utils import setup_logging, ensure_dir

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
SEGMENT_DAYS = 30
INPUT_STEPS_PER_HOUR = 6  # 10-min resolution
OUTPUT_STEPS_PER_HOUR = 1  # 1-hour resolution
INPUT_SAMPLES = SEGMENT_DAYS * 24 * INPUT_STEPS_PER_HOUR  # 4320
OUTPUT_SAMPLES = SEGMENT_DAYS * 24 * OUTPUT_STEPS_PER_HOUR  # 720

# Channel definitions
INPUT_CHANNELS = [
    'temp_treenet', 'rh_treenet', 'stem',  # Sensor channels (3)
    'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr',  # Meteo channels (7)
    'doy'  # Time channel (1)
]
OUTPUT_CHANNELS = ['local_T', 'local_RH', 'stem']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reconstruct time series using trained TCN model'
    )
    
    # Model
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model (.keras file)'
    )
    
    # Site/Sensor specification
    parser.add_argument(
        '--site-id', type=int, required=True,
        help='Site ID'
    )
    parser.add_argument(
        '--thermo-id', type=int, required=True,
        help='Thermometer series ID'
    )
    parser.add_argument(
        '--hygro-id', type=int, required=True,
        help='Hygrometer series ID'
    )
    parser.add_argument(
        '--dendro-id', type=int, required=True,
        help='Dendrometer series ID'
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data',
        help='Root directory with raw sensor data'
    )
    parser.add_argument(
        '--meteo-root', type=str,
        default='/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data',
        help='Directory with meteo CSV files'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/home/lukovic/data/treenet/reconstructions',
        help='Output directory for reconstructed time series'
    )
    
    # Reconstruction parameters
    parser.add_argument(
        '--max-gap-days', type=int, default=12,
        help='Maximum gap length to fill (days)'
    )
    parser.add_argument(
        '--year-start', type=int, default=None,
        help='Start year for reconstruction (default: all available)'
    )
    parser.add_argument(
        '--year-end', type=int, default=None,
        help='End year for reconstruction (default: all available)'
    )
    parser.add_argument(
        '--norm-scope', type=str, default='segment',
        choices=['year', 'segment'],
        help='Normalization scope (should match training)'
    )
    parser.add_argument(
        '--output-mode', type=str, default='input_scale',
        choices=['normalized', 'input_scale'],
        help='Output mode: "normalized" keeps [0,1] range, "input_scale" uses input params for approx denorm'
    )
    
    # Options
    parser.add_argument(
        '--overlap-days', type=int, default=5,
        help='Overlap between consecutive segments (days)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


class TimeSeriesReconstructor:
    """
    Reconstructs clean hourly time series from raw 10-min sensor data.
    
    Uses a trained TCN model to convert 11-channel 10-min input to
    3-channel 1-hour clean output (LM quality).
    
    IMPORTANT LIMITATION:
    The model was trained with per-segment normalization, so outputs are in
    a relative [0,1] scale that cannot be perfectly converted to absolute values
    without the original LM data. Options:
    - 'normalized': Keep output in [0,1] range (relative values)
    - 'input_scale': Use input normalization params (works OK for T, RH; not for stem)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        normalizer: Normalizer,
        max_gap_days: int = 12,
        overlap_days: int = 5,
        output_mode: str = 'input_scale',
        verbose: bool = False
    ):
        self.model = model
        self.normalizer = normalizer
        self.max_gap_days = max_gap_days
        self.overlap_days = overlap_days
        self.output_mode = output_mode
        self.verbose = verbose
        
        # Calculate overlap in samples
        self.overlap_input = overlap_days * 24 * INPUT_STEPS_PER_HOUR
        self.overlap_output = overlap_days * 24 * OUTPUT_STEPS_PER_HOUR
        
        # Stride = segment length - overlap
        self.stride_input = INPUT_SAMPLES - self.overlap_input
        self.stride_output = OUTPUT_SAMPLES - self.overlap_output
    
    def load_and_prepare_input(
        self,
        loaders: DataLoaders,
        processor: DataProcessor,
        site_id: int,
        thermo_id: int,
        hygro_id: int,
        dendro_id: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw sensor data and prepare 11-channel input DataFrame.
        
        Returns:
            Tuple of (input_df, lm_df) where lm_df may be None if no LM data
        """
        print(f"\n{'='*60}")
        print(f"Loading data for Site {site_id}")
        print(f"  Thermometer: {thermo_id}, Hygrometer: {hygro_id}, Dendrometer: {dendro_id}")
        print(f"{'='*60}")
        
        # Load raw sensor data
        thermo_df = loaders.load_thermometer_l1(thermo_id)
        hygro_df = loaders.load_hygrometer_l1(hygro_id)
        dendro_l2_df = loaders.load_dendrometer_l2(dendro_id)
        dendro_lm_df = loaders.load_dendrometer_lm(dendro_id)
        meteo_df = loaders.load_meteotest_data(site_id)
        
        # Check required data
        if thermo_df is None:
            raise ValueError(f"No thermometer data for series {thermo_id}")
        if hygro_df is None:
            raise ValueError(f"No hygrometer data for series {hygro_id}")
        if dendro_l2_df is None:
            raise ValueError(f"No dendrometer L2 data for series {dendro_id}")
        if meteo_df is None:
            raise ValueError(f"No meteo data for site {site_id}")
        
        print(f"  Raw data loaded:")
        print(f"    Thermometer: {len(thermo_df):,} samples")
        print(f"    Hygrometer: {len(hygro_df):,} samples")
        print(f"    Dendrometer L2: {len(dendro_l2_df):,} samples")
        if dendro_lm_df is not None:
            print(f"    Dendrometer LM: {len(dendro_lm_df):,} samples")
        else:
            print(f"    Dendrometer LM: Not available")
        
        # Process sensor data to UTC-indexed format
        temp_df = processor.process_sensor_dataframe(thermo_df)
        rh_df = processor.process_sensor_dataframe(hygro_df)
        stem_df = processor.process_sensor_dataframe(dendro_l2_df)
        
        # Store processed dataframes for per-channel gap analysis
        self._temp_df = temp_df
        self._rh_df = rh_df
        self._stem_df = stem_df
        
        # Process meteo
        meteo_daily = processor.process_meteo_daily(meteo_df)
        
        # Create 11-channel input array
        input_df = processor.merger.create_input_array(
            temp_df, rh_df, stem_df, meteo_daily
        )
        
        # Filter by year range if specified
        if year_start is not None:
            input_df = input_df[input_df.index.year >= year_start]
        if year_end is not None:
            input_df = input_df[input_df.index.year <= year_end]
        
        print(f"\n  Prepared input:")
        print(f"    Shape: {len(input_df):,} samples × {len(input_df.columns)} channels")
        print(f"    Date range: {input_df.index.min()} to {input_df.index.max()}")
        
        # Process LM data if available (for validation)
        lm_df = None
        if dendro_lm_df is not None:
            lm_processed = processor.process_sensor_dataframe(
                dendro_lm_df, keep_all_columns=True
            )
            lm_df = processor.merger.create_target_array(lm_processed)
            
            # Filter by year range
            if year_start is not None:
                lm_df = lm_df[lm_df.index.year >= year_start]
            if year_end is not None:
                lm_df = lm_df[lm_df.index.year <= year_end]
            
            print(f"\n  LM data available for validation:")
            print(f"    Shape: {len(lm_df):,} samples × {len(lm_df.columns)} channels")
        
        return input_df, lm_df
    
    def analyze_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify gaps in time series.
        
        Returns list of gap dictionaries with:
        - start_time, end_time: Gap boundaries
        - gap_hours: Gap length in hours
        - fillable: Whether gap <= max_gap_days
        """
        if len(df) < 2:
            return []
        
        # Ensure sorted index
        df = df.sort_index()
        
        # Calculate time differences
        time_diffs = df.index.to_series().diff()
        
        # Expected interval (10 min) with tolerance (15 min)
        expected = pd.Timedelta('10 minutes')
        tolerance = pd.Timedelta('15 minutes')
        
        gaps = []
        for i in range(1, len(df)):
            diff = time_diffs.iloc[i]
            
            if diff > tolerance:
                gap_hours = diff.total_seconds() / 3600
                gap_days = gap_hours / 24
                
                gaps.append({
                    'start_time': df.index[i-1],
                    'end_time': df.index[i],
                    'gap_hours': gap_hours,
                    'gap_days': gap_days,
                    'fillable': gap_days <= self.max_gap_days
                })
        
        return gaps
    
    def create_segment_grid(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Create overlapping segment grid covering the time range.
        
        Returns list of (segment_start, segment_end) tuples.
        """
        segments = []
        
        # Start at 10-min boundary
        current_start = start_time.floor('10min')
        
        while current_start + pd.Timedelta(days=SEGMENT_DAYS) <= end_time:
            seg_end = current_start + pd.Timedelta(days=SEGMENT_DAYS)
            segments.append((current_start, seg_end))
            
            # Move to next segment (with overlap)
            current_start = current_start + pd.Timedelta(hours=self.stride_output)
        
        # Add final segment if needed
        if current_start < end_time:
            final_start = end_time - pd.Timedelta(days=SEGMENT_DAYS)
            final_start = final_start.floor('10min')
            if final_start >= start_time and (final_start, end_time) not in segments:
                segments.append((final_start, end_time))
        
        return segments
    
    def prepare_segment(
        self,
        input_df: pd.DataFrame,
        seg_start: pd.Timestamp,
        seg_end: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Prepare a segment for model inference.
        
        Returns:
            (input_array, mask_array, is_valid)
            - input_array: (1, 4320, 11) normalized input
            - mask_array: (1, 4320, 11) binary mask (1=valid, 0=missing)
            - is_valid: Whether segment can be processed
        """
        # Create complete 10-min index for segment
        complete_idx = pd.date_range(
            start=seg_start,
            periods=INPUT_SAMPLES,
            freq='10min',
            tz='UTC'
        )
        
        # Extract segment from input
        segment = input_df.loc[
            (input_df.index >= seg_start) & (input_df.index < seg_end)
        ].copy()
        
        # Ensure consistent index dtype
        segment.index = pd.to_datetime(segment.index).tz_convert('UTC')
        
        # Reindex to complete grid (creates NaN for missing)
        segment_full = segment.reindex(complete_idx)
        
        # Create mask before filling
        mask = (~segment_full.isna()).astype(np.float32)
        
        # Check coverage
        coverage = mask.values.mean()
        if coverage < 0.5:  # Require at least 50% coverage
            return None, None, False
        
        # Fill missing values with interpolation
        segment_filled = segment_full.interpolate(method='linear', limit_direction='both')
        segment_filled = segment_filled.ffill().bfill()
        
        # Check for remaining NaN
        if segment_filled.isna().any().any():
            return None, None, False
        
        # Normalize
        input_min, input_diff = self.normalizer.compute_normalization_params(segment_filled)
        segment_norm = self.normalizer.normalize(segment_filled, input_min, input_diff)
        
        # Convert to arrays with correct shape
        input_array = segment_norm.values.astype(np.float32)
        input_array = input_array.reshape(1, INPUT_SAMPLES, -1)
        
        mask_array = mask.values.astype(np.float32)
        mask_array = mask_array.reshape(1, INPUT_SAMPLES, -1)
        
        # Store normalization params for denormalization
        self._current_norm_params = {
            'input_min': input_min,
            'input_diff': input_diff
        }
        
        return input_array, mask_array, True
    
    def run_inference(
        self,
        input_array: np.ndarray,
        mask_array: np.ndarray
    ) -> np.ndarray:
        """
        Run model inference.
        
        Returns:
            Hourly output array (1, 720, 3) in normalized scale
        """
        # Model expects [input, mask]
        predictions = self.model.predict([input_array, mask_array], verbose=0)
        
        # Model returns [recon_output, hourly_output]
        # We want hourly_output (second element)
        hourly_output = predictions[1]
        
        return hourly_output
    
    def denormalize_output(
        self,
        output_norm: np.ndarray,
        seg_start: pd.Timestamp,
        output_mode: str = 'input_scale'
    ) -> pd.DataFrame:
        """
        Denormalize output and create DataFrame with timestamps.
        
        IMPORTANT: The model outputs values in [0,1] range that were normalized
        using EACH OUTPUT SEGMENT'S OWN min/max during training. This means:
        - Without LM data, we cannot perfectly denormalize
        - We can approximate using input normalization params
        
        Args:
            output_norm: Model output array (1, 720, 3) in [0,1] range
            seg_start: Segment start timestamp
            output_mode: How to handle output scale:
                - 'normalized': Keep in [0,1] range
                - 'input_scale': Use input normalization params (approximation)
        
        Returns:
            DataFrame with denormalized (or normalized) output
        """
        # Align to hour boundary (round up to next hour)
        aligned_start = seg_start.ceil('h')
        
        # Create timestamps (hourly)
        timestamps = pd.date_range(
            start=aligned_start,
            periods=OUTPUT_SAMPLES,
            freq='1h',
            tz='UTC'
        )
        
        if output_mode == 'normalized':
            # Keep normalized - output is in [0,1] range
            output_df = pd.DataFrame(
                output_norm[0],
                index=timestamps,
                columns=OUTPUT_CHANNELS
            )
            return output_df
        
        # Default: Use input normalization params as approximation
        # This works well for T and RH (similar scales between sensor and LM)
        # But stem values may differ significantly between L2 and LM
        params = self._current_norm_params
        
        # Map input params to output
        # output channels: local_T, local_RH, stem
        # input channels: temp_treenet, rh_treenet, stem
        output_min = {
            'local_T': params['input_min'].get('temp_treenet', 0),
            'local_RH': params['input_min'].get('rh_treenet', 0),
            'stem': params['input_min'].get('stem', 0)
        }
        output_diff = {
            'local_T': params['input_diff'].get('temp_treenet', 1),
            'local_RH': params['input_diff'].get('rh_treenet', 1),
            'stem': params['input_diff'].get('stem', 1)
        }
        
        # Denormalize
        output_denorm = np.zeros_like(output_norm)
        for i, col in enumerate(OUTPUT_CHANNELS):
            vmin = output_min[col]
            vdiff = output_diff[col]
            output_denorm[:, :, i] = output_norm[:, :, i] * vdiff + vmin
        
        # Create DataFrame
        output_df = pd.DataFrame(
            output_denorm[0],
            index=timestamps,
            columns=OUTPUT_CHANNELS
        )
        
        return output_df
    
    def merge_segments(
        self,
        segment_outputs: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge overlapping segment outputs into single time series.
        
        Uses weighted average in overlap regions (linear blend).
        """
        if not segment_outputs:
            return pd.DataFrame(columns=OUTPUT_CHANNELS)
        
        # Sort by start time
        segment_outputs = sorted(segment_outputs, key=lambda df: df.index.min())
        
        # Get full time range
        all_times = pd.concat([df for df in segment_outputs]).index.unique()
        all_times = all_times.sort_values()
        
        # Initialize result with NaN
        result = pd.DataFrame(
            index=all_times,
            columns=OUTPUT_CHANNELS,
            dtype=np.float64
        )
        counts = pd.DataFrame(
            index=all_times,
            columns=OUTPUT_CHANNELS,
            dtype=np.float64
        )
        result[:] = 0.0
        counts[:] = 0.0
        
        # Add each segment with weighting
        for seg_df in segment_outputs:
            for col in OUTPUT_CHANNELS:
                if col in seg_df.columns:
                    # Simple averaging for overlaps
                    for idx in seg_df.index:
                        if idx in result.index:
                            val = seg_df.loc[idx, col]
                            if not pd.isna(val):
                                result.loc[idx, col] += val
                                counts.loc[idx, col] += 1
        
        # Average overlapping values
        for col in OUTPUT_CHANNELS:
            mask = counts[col] > 0
            result.loc[mask, col] = result.loc[mask, col] / counts.loc[mask, col]
            result.loc[~mask, col] = np.nan
        
        return result
    
    def _create_gap_mask(
        self,
        timestamps: pd.DatetimeIndex,
        gaps: List[Dict]
    ) -> np.ndarray:
        """
        Create boolean mask indicating which timestamps fall within gaps.
        
        Args:
            timestamps: Output timestamps (hourly)
            gaps: List of gap dictionaries from analyze_gaps()
        
        Returns:
            Boolean array where True = timestamp is within a gap
        """
        is_gap = np.zeros(len(timestamps), dtype=bool)
        
        for gap in gaps:
            # Mark hours that fall within this gap
            # A timestamp is in a gap if it's between gap start and gap end
            gap_start = gap['start_time']
            gap_end = gap['end_time']
            
            # Find timestamps within this gap range
            mask = (timestamps > gap_start) & (timestamps < gap_end)
            is_gap |= mask
        
        return is_gap
    
    def _create_per_channel_gap_masks(
        self,
        timestamps: pd.DatetimeIndex
    ) -> Dict[str, np.ndarray]:
        """
        Create per-channel gap masks based on sensor data availability.
        
        An hour is marked as gap for a channel if the underlying sensor data
        has gaps within that hour.
        
        Args:
            timestamps: Output timestamps (hourly)
        
        Returns:
            Dictionary with 'is_gap_T', 'is_gap_RH', 'is_gap_stem' boolean arrays
        """
        result = {}
        
        # Map channels to stored sensor dataframes
        channel_data = {
            'is_gap_T': getattr(self, '_temp_df', None),
            'is_gap_RH': getattr(self, '_rh_df', None),
            'is_gap_stem': getattr(self, '_stem_df', None)
        }
        
        for gap_col, sensor_df in channel_data.items():
            is_gap = np.zeros(len(timestamps), dtype=bool)
            
            if sensor_df is not None and len(sensor_df) > 0:
                # For each output hour, check if sensor has data
                # An hour is a gap if NO sensor samples exist within that hour
                sensor_hours = sensor_df.index.floor('h')
                sensor_hours_set = set(sensor_hours)
                
                for i, ts in enumerate(timestamps):
                    if ts not in sensor_hours_set:
                        is_gap[i] = True
            else:
                # No sensor data - all gaps
                is_gap[:] = True
            
            result[gap_col] = is_gap
        
        return result

    def reconstruct(
        self,
        input_df: pd.DataFrame,
        lm_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Main reconstruction method.
        
        Args:
            input_df: 11-channel input DataFrame (10-min resolution)
            lm_df: Optional LM data for validation
        
        Returns:
            (reconstructed_df, metrics)
            reconstructed_df includes 'is_gap' column indicating gap-filled regions
        """
        print("\n" + "="*60)
        print("Starting reconstruction")
        print("="*60)
        
        # Analyze gaps
        gaps = self.analyze_gaps(input_df)
        fillable_gaps = [g for g in gaps if g['fillable']]
        unfillable_gaps = [g for g in gaps if not g['fillable']]
        
        # Store gaps for later gap marking
        self._gaps = gaps
        
        print(f"\nGap analysis:")
        print(f"  Total gaps found: {len(gaps)}")
        print(f"  Fillable (≤{self.max_gap_days} days): {len(fillable_gaps)}")
        print(f"  Unfillable (>{self.max_gap_days} days): {len(unfillable_gaps)}")
        
        if gaps:
            gap_days = [g['gap_days'] for g in gaps]
            print(f"  Gap lengths: {min(gap_days):.1f} - {max(gap_days):.1f} days")
        
        # Create segment grid
        segments = self.create_segment_grid(
            input_df.index.min(),
            input_df.index.max()
        )
        print(f"\nSegment grid: {len(segments)} segments (with {self.overlap_days}-day overlap)")
        
        # Process each segment
        segment_outputs = []
        processed = 0
        skipped = 0
        
        for i, (seg_start, seg_end) in enumerate(segments):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processing segment {i+1}/{len(segments)}...")
            
            # Prepare segment
            input_array, mask_array, is_valid = self.prepare_segment(
                input_df, seg_start, seg_end
            )
            
            if not is_valid:
                skipped += 1
                continue
            
            # Run inference
            try:
                output_norm = self.run_inference(input_array, mask_array)
            except Exception as e:
                if self.verbose:
                    print(f"    Segment {i+1} inference failed: {e}")
                skipped += 1
                continue
            
            # Denormalize (or keep normalized based on output_mode)
            output_df = self.denormalize_output(output_norm, seg_start, self.output_mode)
            segment_outputs.append(output_df)
            processed += 1
        
        print(f"\nSegment processing:")
        print(f"  Processed: {processed}")
        print(f"  Skipped: {skipped}")
        
        # Merge segments
        print("\nMerging segments...")
        reconstructed = self.merge_segments(segment_outputs)
        
        # Add overall gap mask (from merged input)
        is_gap = self._create_gap_mask(reconstructed.index, self._gaps)
        reconstructed['is_gap'] = is_gap
        
        # Add per-channel gap masks
        per_channel_gaps = self._create_per_channel_gap_masks(reconstructed.index)
        for col_name, gap_array in per_channel_gaps.items():
            reconstructed[col_name] = gap_array
        
        # Print gap statistics
        gap_hours = is_gap.sum()
        gap_T = per_channel_gaps['is_gap_T'].sum()
        gap_RH = per_channel_gaps['is_gap_RH'].sum()
        gap_stem = per_channel_gaps['is_gap_stem'].sum()
        
        print(f"\n  Result: {len(reconstructed):,} hourly samples")
        print(f"  Overall gap-filled hours: {gap_hours:,} ({100*gap_hours/len(reconstructed):.1f}%)")
        print(f"  Per-channel gaps:")
        print(f"    Temperature: {gap_T:,} ({100*gap_T/len(reconstructed):.1f}%)")
        print(f"    Humidity: {gap_RH:,} ({100*gap_RH/len(reconstructed):.1f}%)")
        print(f"    Stem: {gap_stem:,} ({100*gap_stem/len(reconstructed):.1f}%)")
        print(f"  Date range: {reconstructed.index.min()} to {reconstructed.index.max()}")
        
        # Calculate metrics
        metrics = {
            'total_gaps': len(gaps),
            'fillable_gaps': len(fillable_gaps),
            'unfillable_gaps': len(unfillable_gaps),
            'segments_total': len(segments),
            'segments_processed': processed,
            'segments_skipped': skipped,
            'output_samples': len(reconstructed),
            'nan_samples': reconstructed.isna().sum().to_dict()
        }
        
        # Validate against LM if available
        if lm_df is not None:
            print("\nValidating against LM data...")
            print("  NOTE: Comparison is in original (denormalized) space.")
            print("  Stem values may show high error due to L2→LM scale difference.")
            
            # Find common timestamps
            common_idx = reconstructed.index.intersection(lm_df.index)
            
            if len(common_idx) > 0:
                for col in OUTPUT_CHANNELS:
                    if col in reconstructed.columns and col in lm_df.columns:
                        recon_vals = reconstructed.loc[common_idx, col]
                        lm_vals = lm_df.loc[common_idx, col]
                        
                        # Remove NaN
                        valid_mask = ~(recon_vals.isna() | lm_vals.isna())
                        if valid_mask.sum() > 0:
                            # Original scale comparison
                            mae = np.abs(recon_vals[valid_mask] - lm_vals[valid_mask]).mean()
                            rmse = np.sqrt(((recon_vals[valid_mask] - lm_vals[valid_mask])**2).mean())
                            
                            # Correlation (scale-invariant metric)
                            if recon_vals[valid_mask].std() > 0 and lm_vals[valid_mask].std() > 0:
                                corr = np.corrcoef(recon_vals[valid_mask], lm_vals[valid_mask])[0, 1]
                            else:
                                corr = 0.0
                            
                            # Normalized comparison (for fair assessment)
                            # Normalize both to [0,1] using LM min/max
                            lm_min = lm_vals[valid_mask].min()
                            lm_max = lm_vals[valid_mask].max()
                            lm_diff = lm_max - lm_min if lm_max > lm_min else 1.0
                            
                            recon_norm = (recon_vals[valid_mask] - lm_min) / lm_diff
                            lm_norm = (lm_vals[valid_mask] - lm_min) / lm_diff
                            
                            mae_norm = np.abs(recon_norm - lm_norm).mean()
                            rmse_norm = np.sqrt(((recon_norm - lm_norm)**2).mean())
                            
                            metrics[f'{col}_mae'] = mae
                            metrics[f'{col}_rmse'] = rmse
                            metrics[f'{col}_corr'] = corr
                            metrics[f'{col}_mae_normalized'] = mae_norm
                            metrics[f'{col}_rmse_normalized'] = rmse_norm
                            metrics[f'{col}_valid_samples'] = valid_mask.sum()
                            
                            print(f"  {col}: MAE={mae:.3f}, RMSE={rmse:.3f}, Corr={corr:.4f} | Norm MAE={mae_norm:.4f} ({valid_mask.sum():,} samples)")
        
        return reconstructed, metrics


def main():
    """Main function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = ensure_dir(Path(args.output_dir))
    
    # Setup logging
    log_file = output_dir / 'reconstruction.log'
    setup_logging(verbose=args.verbose, log_file=log_file)
    
    print("="*80)
    print("TreeNet AI Pipeline v2 - Time Series Reconstruction (PATH 1)")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Site: {args.site_id}")
    print(f"Sensors: T={args.thermo_id}, H={args.hygro_id}, D={args.dendro_id}")
    print(f"Max gap: {args.max_gap_days} days")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\n" + "-"*60)
    print("Loading model...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'TCNBlock': TCNBlock,
            'PositionalEncoding': PositionalEncoding
        }
    )
    print(f"  Model loaded: {model.name}")
    print(f"  Input shapes: {[inp.shape for inp in model.inputs]}")
    print(f"  Output shapes: {[out.shape for out in model.outputs]}")
    
    # Initialize components
    print("\n" + "-"*60)
    print("Initializing data loaders...")
    loaders = DataLoaders(
        data_root=Path(args.data_dir),
        meteo_root=Path(args.meteo_root)
    )
    processor = DataProcessor()
    normalizer = Normalizer(norm_scope=args.norm_scope)
    
    # Create reconstructor
    reconstructor = TimeSeriesReconstructor(
        model=model,
        normalizer=normalizer,
        max_gap_days=args.max_gap_days,
        overlap_days=args.overlap_days,
        output_mode=args.output_mode,
        verbose=args.verbose
    )
    
    # Load and prepare input data
    input_df, lm_df = reconstructor.load_and_prepare_input(
        loaders=loaders,
        processor=processor,
        site_id=args.site_id,
        thermo_id=args.thermo_id,
        hygro_id=args.hygro_id,
        dendro_id=args.dendro_id,
        year_start=args.year_start,
        year_end=args.year_end
    )
    
    # Run reconstruction
    reconstructed, metrics = reconstructor.reconstruct(input_df, lm_df)
    
    # Save results
    print("\n" + "-"*60)
    print("Saving results...")
    
    # Create combo string for filename
    combo_str = f"site{args.site_id}_T{args.thermo_id}_H{args.hygro_id}_D{args.dendro_id}"
    
    # Save reconstructed time series
    save_path = output_dir / f"reconstructed_{combo_str}.ftr"
    save_df = reconstructed.reset_index()
    save_df.rename(columns={'index': 'ts'}, inplace=True)
    save_df.to_feather(save_path)
    print(f"  Saved: {save_path}")
    
    # Save metrics
    metrics['site_id'] = args.site_id
    metrics['thermo_id'] = args.thermo_id
    metrics['hygro_id'] = args.hygro_id
    metrics['dendro_id'] = args.dendro_id
    
    metrics_path = output_dir / f"metrics_{combo_str}.json"
    import json
    with open(metrics_path, 'w') as f:
        # Convert numpy types to python types for JSON
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_metrics[k] = float(v)
            elif isinstance(v, dict):
                clean_metrics[k] = {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                                   for kk, vv in v.items()}
            else:
                clean_metrics[k] = v
        json.dump(clean_metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")
    
    print("\n" + "="*80)
    print("Reconstruction complete!")
    print("="*80)


if __name__ == '__main__':
    main()
