"""
Data loading utilities for TreeNet sensor data.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import pickle


class DataLoaders:
    """Centralized data loading for all TreeNet data sources."""
    
    def __init__(self, data_root: Path, meteo_root: Path):
        """
        Initialize data loaders.
        
        Args:
            data_root: Root directory containing sensor data
            meteo_root: Directory containing meteotest CSV files
        """
        self.data_root = Path(data_root)
        self.meteo_root = Path(meteo_root)
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        if not self.meteo_root.exists():
            raise FileNotFoundError(f"Meteo root not found: {self.meteo_root}")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata for all sensors.
        
        Returns:
            DataFrame with columns: measure_point, series_id, series_start, 
            series_stop, variable_name, variable_resolution, site_id, etc.
        """
        metadata_path = self.data_root / 'metadata_all.pkl'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        metadata = pd.read_pickle(metadata_path)
        return metadata
    
    def load_thermometer_l1(self, series_id: int) -> Optional[pd.DataFrame]:
        """
        Load raw thermometer data (L1).
        
        Args:
            series_id: Sensor series ID
            
        Returns:
            DataFrame with columns: ts, series, value (temperature in °C)
            Returns None if file not found.
        """
        file_path = self.data_root / 'thermometer_l1' / f'thermometer_l1_series_id_{series_id}.ftr'
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_feather(file_path)
            return df[['ts', 'series', 'value']]
        except Exception as e:
            print(f"Error loading thermometer {series_id}: {e}")
            return None
    
    def load_hygrometer_l1(self, series_id: int) -> Optional[pd.DataFrame]:
        """
        Load raw hygrometer data (L1).
        
        Args:
            series_id: Sensor series ID
            
        Returns:
            DataFrame with columns: ts, series, value (relative humidity in %)
            Returns None if file not found.
        """
        file_path = self.data_root / 'hygrometer_l1' / f'hygrometer_l1_series_id_{series_id}.ftr'
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_feather(file_path)
            return df[['ts', 'series', 'value']]
        except Exception as e:
            print(f"Error loading hygrometer {series_id}: {e}")
            return None
    
    def load_dendrometer_l2(self, series_id: int) -> Optional[pd.DataFrame]:
        """
        Load raw dendrometer data (L2).
        
        Args:
            series_id: Sensor series ID
            
        Returns:
            DataFrame with columns: ts, series, value (stem radius change in μm)
            Returns None if file not found.
        """
        file_path = self.data_root / 'dendrometer_l2' / f'dendrometer_l2_series_id_{series_id}.ftr'
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_feather(file_path)
            # Keep only essential columns
            return df[['ts', 'series', 'value']]
        except Exception as e:
            print(f"Error loading dendrometer L2 {series_id}: {e}")
            return None
    
    def load_dendrometer_lm(self, series_id: int) -> Optional[pd.DataFrame]:
        """
        Load manually cleaned dendrometer data (LM - ground truth).
        
        Args:
            series_id: Sensor series ID
            
        Returns:
            DataFrame with columns: ts, series, value (stem), temp, rh
            Returns None if file not found.
        """
        file_path = self.data_root / 'dendrometer_lm' / f'dendrometer_lm_series_id_{series_id}.ftr'
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_feather(file_path)
            # Keep ground truth columns
            cols = ['ts', 'series', 'value']
            if 'temp' in df.columns:
                cols.append('temp')
            if 'rh' in df.columns:
                cols.append('rh')
            return df[cols]
        except Exception as e:
            print(f"Error loading dendrometer LM {series_id}: {e}")
            return None
    
    def load_meteotest_data(self, site_id: int) -> Optional[pd.DataFrame]:
        """
        Load gridded meteotest weather data for a site.
        
        File naming convention: meteo_data_site_id_{SITE_ID}.csv
        
        Args:
            site_id: Site ID
            
        Returns:
            DataFrame with columns: ts, tas, tasmax, tasmin, rh, vpd, gh, pr
            Returns None if file not found or all values are NaN.
        """
        # Primary naming convention: meteo_data_site_id_{site_id}.csv
        file_path = self.meteo_root / f'meteo_data_site_id_{site_id}.csv'
        
        if not file_path.exists():
            # Fallback: try old naming convention site_{site_id}.csv
            file_path = self.meteo_root / f'site_{site_id}.csv'
            
        if not file_path.exists():
            # Try glob pattern as last resort
            pattern = f'*site*{site_id}*.csv'
            matches = list(self.meteo_root.glob(pattern))
            if not matches:
                return None
            file_path = matches[0]
        
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns exist
            required_cols = ['ts', 'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Check if data is valid (not all NaN)
            # Non-Swiss sites have empty meteo files
            data_cols = [c for c in required_cols if c != 'ts']
            if df[data_cols].isna().all().all():
                return None  # All values are NaN - invalid meteo data
            
            return df[required_cols]
        except Exception as e:
            print(f"Error loading meteotest for site {site_id}: {e}")
            return None
    
    def discover_meteo_files(self) -> Dict[int, Path]:
        """
        Discover all meteotest CSV files and map to site IDs.
        
        Returns:
            Dictionary mapping site_id -> file path
        """
        mapping = {}
        if not self.meteo_root.exists():
            return mapping
        
        for file_path in self.meteo_root.glob('*.csv'):
            # Extract site ID from filename
            matches = re.findall(r'\d+', file_path.name)
            if matches:
                site_id = int(matches[0])
                mapping[site_id] = file_path
        
        return mapping
    
    def discover_sensor_ids(self, sensor_type: str) -> set[int]:
        """
        Discover all available sensor IDs for a given sensor type.
        
        Args:
            sensor_type: One of 'thermometer_l1', 'hygrometer_l1', 
                        'dendrometer_l2', 'dendrometer_lm'
        
        Returns:
            Set of series IDs
        """
        sensor_dir = self.data_root / sensor_type
        if not sensor_dir.exists():
            return set()
        
        ids = set()
        for file_path in sensor_dir.glob('*.ftr'):
            # Extract series ID from filename
            matches = re.findall(r'series_id_(\d+)', file_path.name)
            if matches:
                ids.add(int(matches[0]))
        
        return ids
    
    def load_all_sensors_for_site(
        self, 
        site_id: int, 
        metadata: pd.DataFrame
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Load all available sensors for a specific site.
        
        Args:
            site_id: Site ID to load
            metadata: Metadata DataFrame
            
        Returns:
            Dictionary with keys: 'thermometer', 'hygrometer', 'dendrometer_l2', 'dendrometer_lm'
            Each maps sensor_id -> DataFrame
        """
        site_meta = metadata[metadata['site_id'] == site_id]
        
        result = {
            'thermometer': {},
            'hygrometer': {},
            'dendrometer_l2': {},
            'dendrometer_lm': {}
        }
        
        for _, row in site_meta.iterrows():
            series_id = row['series_id']
            var_name = row['variable_name']
            
            if var_name == 'air temperature':
                df = self.load_thermometer_l1(series_id)
                if df is not None:
                    result['thermometer'][series_id] = df
            
            elif var_name == 'relative humidity':
                df = self.load_hygrometer_l1(series_id)
                if df is not None:
                    result['hygrometer'][series_id] = df
            
            elif var_name == 'tree stem radius change':
                df_l2 = self.load_dendrometer_l2(series_id)
                if df_l2 is not None:
                    result['dendrometer_l2'][series_id] = df_l2
                
                df_lm = self.load_dendrometer_lm(series_id)
                if df_lm is not None:
                    result['dendrometer_lm'][series_id] = df_lm
        
        return result
    
    def get_sites_with_complete_data(
        self, 
        metadata: pd.DataFrame, 
        country: Optional[str] = 'Switzerland'
    ) -> set[int]:
        """
        Find sites that have at least one sensor of each required type.
        
        Args:
            metadata: Metadata DataFrame
            country: Filter to sites in this country. 
                     Default is 'Switzerland' (only Swiss sites have valid meteo data).
                     Use None to include all countries.
            
        Returns:
            Set of site IDs with complete sensor coverage
        """
        complete_sites = set()
        
        for site_id in metadata['site_id'].unique():
            site_meta = metadata[metadata['site_id'] == site_id]
            
            # Filter by country if specified
            if country is not None:
                site_country = site_meta['country'].iloc[0] if 'country' in site_meta.columns else None
                if site_country != country:
                    continue
            
            var_names = set(site_meta['variable_name'].unique())
            
            # Check if site has all three sensor types
            has_temp = 'air temperature' in var_names
            has_humidity = 'relative humidity' in var_names
            has_dendro = 'tree stem radius change' in var_names
            
            if has_temp and has_humidity and has_dendro:
                complete_sites.add(site_id)
        
        return complete_sites
    
    def save_pickle(self, data: any, filepath: Path) -> None:
        """Save data to pickle file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_pickle(self, filepath: Path) -> any:
        """Load data from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
