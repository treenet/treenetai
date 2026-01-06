"""
Data validation utilities.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import warnings


class DataValidator:
    """
    Validates sensor data for completeness, consistency, and quality.
    
    Checks include:
    - Missing values
    - Duplicate timestamps
    - Data coverage
    - Value ranges
    """
    
    @staticmethod
    def validate_sensor_dataframe(
        df: pd.DataFrame,
        series_id: int,
        sensor_type: str,
        required_cols: Optional[List[str]] = None
    ) -> bool:
        """
        Validate a single sensor DataFrame.
        
        Args:
            df: DataFrame to validate
            series_id: Sensor series ID (for error messages)
            sensor_type: Type of sensor (for error messages)
            required_cols: List of required column names
            
        Returns:
            True if valid, False otherwise (warnings are issued)
        """
        if df is None or len(df) == 0:
            warnings.warn(f"{sensor_type} {series_id}: Empty dataframe")
            return False
        
        # Check for required columns
        if required_cols:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                warnings.warn(
                    f"{sensor_type} {series_id}: Missing columns {missing_cols}"
                )
                return False
        
        # Check for duplicate timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.duplicated().any():
                n_dupes = df.index.duplicated().sum()
                warnings.warn(
                    f"{sensor_type} {series_id}: {n_dupes} duplicate timestamps"
                )
        
        # Check for excessive missing values in value column
        if 'value' in df.columns:
            missing_pct = df['value'].isna().sum() / len(df) * 100
            if missing_pct > 50:
                warnings.warn(
                    f"{sensor_type} {series_id}: {missing_pct:.1f}% missing values"
                )
        
        return True
    
    @staticmethod
    def validate_metadata(metadata: pd.DataFrame) -> bool:
        """
        Validate metadata DataFrame.
        
        Args:
            metadata: Metadata DataFrame
            
        Returns:
            True if valid
        """
        required_cols = [
            'series_id', 'site_id', 'variable_name', 
            'series_start', 'series_stop'
        ]
        
        missing_cols = set(required_cols) - set(metadata.columns)
        if missing_cols:
            raise ValueError(f"Metadata missing columns: {missing_cols}")
        
        # Check for duplicate series IDs
        if metadata['series_id'].duplicated().any():
            warnings.warn("Metadata contains duplicate series_id values")
        
        return True
    
    @staticmethod
    def validate_segment(
        input_seg: pd.DataFrame,
        output_seg: pd.DataFrame,
        expected_input_steps: int = 4320,
        expected_output_steps: int = 720
    ) -> bool:
        """
        Validate a segment for completeness.
        
        Args:
            input_seg: Input segment (10-min resolution)
            output_seg: Output segment (hourly resolution)
            expected_input_steps: Expected number of input timesteps
            expected_output_steps: Expected number of output timesteps
            
        Returns:
            True if valid
        """
        # Check dimensions
        if len(input_seg) != expected_input_steps:
            warnings.warn(
                f"Input segment has {len(input_seg)} steps, "
                f"expected {expected_input_steps}"
            )
            return False
        
        if len(output_seg) != expected_output_steps:
            warnings.warn(
                f"Output segment has {len(output_seg)} steps, "
                f"expected {expected_output_steps}"
            )
            return False
        
        # Check for NaN values
        if input_seg.isna().any().any():
            warnings.warn("Input segment contains NaN values")
            return False
        
        if output_seg.isna().any().any():
            warnings.warn("Output segment contains NaN values")
            return False
        
        return True
    
    @staticmethod
    def check_value_ranges(
        df: pd.DataFrame,
        column: str,
        expected_min: float,
        expected_max: float,
        tolerance: float = 0.1
    ) -> bool:
        """
        Check if values are within expected ranges.
        
        Args:
            df: DataFrame to check
            column: Column name to check
            expected_min: Expected minimum value
            expected_max: Expected maximum value
            tolerance: Tolerance factor (0.1 = 10% outside range allowed)
            
        Returns:
            True if mostly within range
        """
        if column not in df.columns:
            return True
        
        values = df[column].dropna()
        if len(values) == 0:
            return True
        
        # Count outliers
        outliers = (values < expected_min) | (values > expected_max)
        outlier_pct = outliers.sum() / len(values)
        
        if outlier_pct > tolerance:
            warnings.warn(
                f"Column '{column}': {outlier_pct*100:.1f}% of values "
                f"outside range [{expected_min}, {expected_max}]"
            )
            return False
        
        return True
    
    @staticmethod
    def validate_site_coverage(
        metadata: pd.DataFrame,
        site_id: int,
        required_vars: List[str]
    ) -> bool:
        """
        Check if a site has all required sensor types.
        
        Args:
            metadata: Metadata DataFrame
            site_id: Site ID to check
            required_vars: List of required variable names
            
        Returns:
            True if site has complete coverage
        """
        site_meta = metadata[metadata['site_id'] == site_id]
        available_vars = set(site_meta['variable_name'].unique())
        missing_vars = set(required_vars) - available_vars
        
        if missing_vars:
            return False
        
        return True
