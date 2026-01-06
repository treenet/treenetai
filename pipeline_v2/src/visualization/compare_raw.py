"""
Compare denormalized segments with raw data sources.

Validates that processed segments match the original raw data files
after denormalization, ensuring no data corruption during processing.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle


class RawDataComparator:
    """
    Compare denormalized segments with raw source files.
    
    For quality assurance, this validates that:
    - Denormalization correctly reverses normalization
    - No data corruption occurred during processing
    - Time windows match exactly
    - Resolution conversions are accurate
    """
    
    def __init__(self, local_tz: str = 'Europe/Zurich'):
        """
        Initialize comparator.
        
        Args:
            local_tz: Local timezone for date displays
        """
        self.local_tz = local_tz
        
        self.input_channels_wo_doy = [
            'temp_treenet', 'rh_treenet', 'stem',
            'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr'
        ]
        self.target_channels = ['local_T', 'local_RH', 'stem']
        self.global_channels = ['tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr']
    
    def load_segment_metadata(
        self,
        data_dir: Path,
        split: str,
        combo_id: int,
        seg_idx: int
    ) -> Dict:
        """
        Load normalization parameters for a specific segment.
        
        Args:
            data_dir: Directory with processed segments
            split: 'train' or 'test'
            combo_id: Combination ID
            seg_idx: Segment index
            
        Returns:
            Dictionary with normalization params and time window
        """
        metadata_path = data_dir / f'{split}_segment_ids.pkl'
        
        with open(metadata_path, 'rb') as f:
            seg_ids = pickle.load(f)
        
        # Search for matching segment
        def find_meta(entry):
            if isinstance(entry, (list, tuple)) and len(entry) >= 9:
                e_combo, e_seg = entry[0], entry[1]
                if int(e_combo) == combo_id and int(e_seg) == seg_idx:
                    return {
                        'ids_row': entry[2],
                        'in_min': entry[3],
                        'in_diff': entry[4],
                        'out_min': entry[5],
                        'out_diff': entry[6],
                        'win_start_utc': pd.to_datetime(entry[7]['window_start_utc']).tz_convert('UTC'),
                        'win_end_utc': pd.to_datetime(entry[7]['window_end_utc']).tz_convert('UTC'),
                    }
            return None
        
        # Search in metadata
        if isinstance(seg_ids, dict):
            bucket = seg_ids.get(combo_id, [])
            for entry in bucket:
                meta = find_meta(entry)
                if meta:
                    return meta
        
        if isinstance(seg_ids, list):
            for entry in seg_ids:
                meta = find_meta(entry)
                if meta:
                    return meta
        
        raise ValueError(f"Metadata not found for combo {combo_id}, segment {seg_idx}")
    
    def denormalize_segment(
        self,
        df_norm: pd.DataFrame,
        mins: Dict,
        diffs: Dict,
        channels: List[str]
    ) -> pd.DataFrame:
        """
        Reverse normalization to original scale.
        
        Args:
            df_norm: Normalized DataFrame
            mins: Minimum values per channel
            diffs: Range values per channel
            channels: Channels to denormalize
            
        Returns:
            Denormalized DataFrame
        """
        result = pd.DataFrame(index=df_norm.index)
        
        for ch in channels:
            if ch not in df_norm.columns:
                continue
            if ch not in mins or ch not in diffs:
                continue
            
            min_val = float(mins[ch]) if mins[ch] is not None else 0.0
            diff_val = float(diffs[ch]) if diffs[ch] is not None else 1.0
            
            if np.isfinite(diff_val) and abs(diff_val) > 1e-8:
                result[ch] = df_norm[ch] * diff_val + min_val
            else:
                result[ch] = df_norm[ch] + min_val
        
        return result
    
    def load_raw_sensor(
        self,
        raw_root: Path,
        sensor_type: str,
        sensor_id: int,
        column: str = 'value'
    ) -> pd.DataFrame:
        """
        Load raw sensor data from feather files.
        
        Args:
            raw_root: Root directory of server_data
            sensor_type: 'thermometer_l1', 'hygrometer_l1', 'dendrometer_l2', 'dendrometer_lm'
            sensor_id: Sensor series ID
            column: Column name to extract
            
        Returns:
            DataFrame with UTC-indexed values
        """
        file_path = raw_root / sensor_type / f'{sensor_type}_series_id_{sensor_id}.ftr'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")
        
        df = pd.read_feather(file_path)
        
        if 'ts' not in df.columns:
            raise ValueError(f"Missing 'ts' column in {file_path}")
        
        # Convert to UTC timestamp
        ts = pd.to_datetime(df['ts'], errors='coerce', utc=True)
        
        # Extract column
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in {file_path}")
        
        result = pd.DataFrame({column: df[column].astype('float64')}, index=ts)
        return result.sort_index()
    
    def load_raw_meteo(
        self,
        meteo_root: Path,
        site_id: int,
        year: int
    ) -> pd.DataFrame:
        """
        Load raw meteo data from CSV files.
        
        Args:
            meteo_root: Root directory of meteo_data
            site_id: Site ID
            year: Year to filter
            
        Returns:
            DataFrame with daily meteo data
        """
        file_path = meteo_root / f'site_{site_id}.csv'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Meteo file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Find timestamp column
        ts_col = 'ts' if 'ts' in df.columns else 'ts_local'
        if ts_col not in df.columns:
            raise ValueError(f"Missing timestamp column in {file_path}")
        
        # Parse timestamp
        ts = pd.to_datetime(df[ts_col], errors='coerce')
        
        # Localize if needed
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(self.local_tz, nonexistent='shift_forward', ambiguous='NaT')
        
        df['ts_local'] = ts
        
        # Filter by year
        df = df[df['ts_local'].dt.year == year]
        
        # Extract global channels
        cols = ['ts_local'] + [c for c in self.global_channels if c in df.columns]
        
        return df[cols].copy()
    
    def compare_segment(
        self,
        data_dir: Path,
        raw_root: Path,
        meteo_root: Path,
        split: str,
        combo_id: int,
        seg_idx: int,
        year: int,
        site_id: int,
        output_dir: Path
    ):
        """
        Compare a single segment with raw data.
        
        Args:
            data_dir: Directory with processed segments
            raw_root: Root directory of server_data
            meteo_root: Root directory of meteo_data
            split: 'train' or 'test'
            combo_id: Combination ID
            seg_idx: Segment index
            year: Year
            site_id: Site ID
            output_dir: Output directory for plots
        """
        print(f"\\nComparing combo {combo_id}, segment {seg_idx}...")
        
        # Load segment data
        with open(data_dir / f'model_{split}_data_combination_ids.pkl', 'rb') as f:
            combo_ids = pickle.load(f)
        
        with open(data_dir / f'{split}_input_segments.pkl', 'rb') as f:
            input_segs = pickle.load(f)
        
        with open(data_dir / f'{split}_output_segments.pkl', 'rb') as f:
            output_segs = pickle.load(f)
        
        # Get sensor IDs
        ids_row = combo_ids[combo_id]
        thermo_id = int(ids_row['thermometer ID'])
        hygro_id = int(ids_row['hygrometer ID'])
        dendro_id = int(ids_row['dendrometer ID'])
        
        # Load metadata
        meta = self.load_segment_metadata(data_dir, split, combo_id, seg_idx)
        
        # Get segments
        input_norm = input_segs[combo_id][seg_idx]
        output_norm = output_segs[combo_id][seg_idx]
        
        # Denormalize
        input_orig = self.denormalize_segment(
            input_norm,
            meta['in_min'],
            meta['in_diff'],
            self.input_channels_wo_doy
        )
        output_orig = self.denormalize_segment(
            output_norm,
            meta['out_min'],
            meta['out_diff'],
            self.target_channels
        )
        
        # Load raw data
        print(f"  Loading raw thermometer {thermo_id}...")
        raw_temp = self.load_raw_sensor(raw_root, 'thermometer_l1', thermo_id)
        
        print(f"  Loading raw hygrometer {hygro_id}...")
        raw_hygro = self.load_raw_sensor(raw_root, 'hygrometer_l1', hygro_id)
        
        print(f"  Loading raw dendrometer {dendro_id}...")
        raw_dendro = self.load_raw_sensor(raw_root, 'dendrometer_l2', dendro_id)
        raw_dendro_lm = self.load_raw_sensor(raw_root, 'dendrometer_lm', dendro_id)
        
        print(f"  Loading raw meteo for site {site_id}...")
        raw_meteo = self.load_raw_meteo(meteo_root, site_id, year)
        
        # Slice to window
        ws_utc = meta['win_start_utc']
        we_utc = meta['win_end_utc']
        
        raw_temp_window = raw_temp[(raw_temp.index >= ws_utc) & (raw_temp.index < we_utc)]
        raw_hygro_window = raw_hygro[(raw_hygro.index >= ws_utc) & (raw_hygro.index < we_utc)]
        raw_dendro_window = raw_dendro[(raw_dendro.index >= ws_utc) & (raw_dendro.index < we_utc)]
        raw_dendro_lm_window = raw_dendro_lm[(raw_dendro_lm.index >= ws_utc) & (raw_dendro_lm.index < we_utc)]
        
        # Create output directory
        segment_dir = output_dir / f'combo_{combo_id}' / f'seg_{seg_idx}'
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot comparisons
        print(f"  Creating comparison plots...")
        
        # Temperature
        if 'temp_treenet' in input_orig.columns and not raw_temp_window.empty:
            self._plot_comparison(
                raw_temp_window.index,
                raw_temp_window['value'],
                'Raw thermometer L1',
                input_orig.index,
                input_orig['temp_treenet'],
                'Denormalized segment',
                'temp_treenet',
                year, site_id, combo_id, seg_idx,
                ws_utc, we_utc,
                segment_dir
            )
        
        # Humidity
        if 'rh_treenet' in input_orig.columns and not raw_hygro_window.empty:
            self._plot_comparison(
                raw_hygro_window.index,
                raw_hygro_window['value'],
                'Raw hygrometer L1',
                input_orig.index,
                input_orig['rh_treenet'],
                'Denormalized segment',
                'rh_treenet',
                year, site_id, combo_id, seg_idx,
                ws_utc, we_utc,
                segment_dir
            )
        
        # Dendrometer input
        if 'stem' in input_orig.columns and not raw_dendro_window.empty:
            self._plot_comparison(
                raw_dendro_window.index,
                raw_dendro_window['value'],
                'Raw dendrometer L2',
                input_orig.index,
                input_orig['stem'],
                'Denormalized segment',
                'stem_input',
                year, site_id, combo_id, seg_idx,
                ws_utc, we_utc,
                segment_dir
            )
        
        # Target temperature
        if 'local_T' in output_orig.columns and not raw_temp_window.empty:
            self._plot_comparison(
                raw_temp_window.index,
                raw_temp_window['value'],
                'Raw L1 temp (10-min)',
                output_orig.index,
                output_orig['local_T'],
                'Denormalized target (hourly)',
                'local_T_target',
                year, site_id, combo_id, seg_idx,
                ws_utc, we_utc,
                segment_dir
            )
        
        # Target humidity
        if 'local_RH' in output_orig.columns and not raw_hygro_window.empty:
            self._plot_comparison(
                raw_hygro_window.index,
                raw_hygro_window['value'],
                'Raw L1 RH (10-min)',
                output_orig.index,
                output_orig['local_RH'],
                'Denormalized target (hourly)',
                'local_RH_target',
                year, site_id, combo_id, seg_idx,
                ws_utc, we_utc,
                segment_dir
            )
        
        # Target dendrometer
        if 'stem' in output_orig.columns and not raw_dendro_lm_window.empty:
            self._plot_comparison(
                raw_dendro_lm_window.index,
                raw_dendro_lm_window['value'],
                'Raw dendrometer LM (10-min)',
                output_orig.index,
                output_orig['stem'],
                'Denormalized target (hourly)',
                'stem_target',
                year, site_id, combo_id, seg_idx,
                ws_utc, we_utc,
                segment_dir
            )
        
        print(f"  Plots saved to: {segment_dir}")
    
    def _plot_comparison(
        self,
        raw_index: pd.DatetimeIndex,
        raw_values: pd.Series,
        raw_label: str,
        seg_index: pd.DatetimeIndex,
        seg_values: pd.Series,
        seg_label: str,
        channel: str,
        year: int,
        site_id: int,
        combo_id: int,
        seg_idx: int,
        ws_utc: pd.Timestamp,
        we_utc: pd.Timestamp,
        output_dir: Path
    ):
        """Create comparison plot for a single channel."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to fractional day of year
        def to_doy_frac(idx):
            loc = idx.tz_convert(self.local_tz)
            seconds = (loc.hour * 3600 + loc.minute * 60 + loc.second) + (loc.microsecond / 1e6)
            frac = seconds / 86400.0
            return loc.dayofyear.to_numpy(dtype=float) + frac
        
        raw_doy = to_doy_frac(raw_index)
        seg_doy = to_doy_frac(seg_index)
        
        # Plot
        ax.plot(raw_doy, raw_values, color='tab:blue', lw=1.2, alpha=0.9, label=raw_label)
        ax.plot(seg_doy, seg_values, color='red', lw=2.0, alpha=0.9, linestyle='--', label=seg_label)
        
        ax.set_xlabel('Day of Year (local)', fontsize=11)
        ax.set_ylabel(f'{channel} (original scale)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.7)
        
        # Set x limits
        all_doy = np.concatenate([raw_doy, seg_doy])
        if len(all_doy) > 0:
            doy_min = max(1, int(all_doy.min()))
            doy_max = min(366, int(all_doy.max()) + 1)
            ax.set_xlim(doy_min, doy_max)
        
        # Title
        ws_local = ws_utc.tz_convert(self.local_tz).strftime('%Y-%m-%d %H:%M')
        we_local = we_utc.tz_convert(self.local_tz).strftime('%Y-%m-%d %H:%M')
        
        title = (
            f'Year {year} • Site {site_id} • Combo {combo_id} • Segment {seg_idx}\\n'
            f'Window: {ws_local} → {we_local} (local)'
        )
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Save
        output_path = output_dir / f'y{year}_site{site_id}_combo{combo_id}_seg{seg_idx}_{channel}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
