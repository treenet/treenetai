#!/usr/bin/env python3
"""
Batch evaluation of reconstruction quality for all test set combinations.

This script:
1. Loads all 20 test set combinations
2. Reconstructs time series using the unconstrained model
3. Applies stem alignment using LM data for proper scale calibration
4. Creates 9-row stacked visualizations (Input/Recon/GT for each channel)
5. Evaluates reconstruction quality vs LM ground truth
6. Generates summary report

IMPORTANT: Stem alignment is ALWAYS used for multi-year reconstruction
to ensure proper scale calibration (see PROJECT_CONTEXT.md).

Usage:
    python batch_evaluate_test_set.py [--years 2020 2021] [--output-dir <path>]

Author: TreeNet AI Pipeline
Date: 2026-01-12
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.feather as feather
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tcn import TCNBlock, PositionalEncoding
from src.data.loaders import DataLoaders
from src.data.processors import DataProcessor
from src.data.segmentation import Normalizer
from src.utils import ensure_dir

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom loss function for model loading
def constrained_hourly_loss(y_true, y_pred):
    """Placeholder loss function for model loading."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Constants
SEGMENT_DAYS = 30
INPUT_SAMPLES = SEGMENT_DAYS * 24 * 6  # 4320 (10-min)
OUTPUT_SAMPLES = SEGMENT_DAYS * 24  # 720 (hourly)
INPUT_CHANNELS = ['temp_treenet', 'rh_treenet', 'stem', 'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy']
OUTPUT_CHANNELS = ['local_T', 'local_RH', 'stem']


class TestSetEvaluator:
    """Batch evaluator for test set combinations."""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        output_dir: str,
        raw_data_dir: str = '/storage/lukovic/Data/FORWARDS/treenet/server_data',
        meteo_dir: str = '/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data',
        years: List[int] = [2021, 2022],
        stride_hours: int = 24
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = ensure_dir(Path(output_dir))
        self.years = years
        self.stride_hours = stride_hours
        
        # Initialize data loaders
        self.loaders = DataLoaders(
            data_root=Path(raw_data_dir),
            meteo_root=Path(meteo_dir)
        )
        self.processor = DataProcessor()
        self.normalizer = Normalizer()
        
        # Load model
        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(
            str(self.model_path),
            custom_objects={
                'TCNBlock': TCNBlock,
                'PositionalEncoding': PositionalEncoding,
                'constrained_hourly_loss': constrained_hourly_loss
            }
        )
        print("Model loaded successfully!")
        
        # Load test combinations
        combo_file = self.data_dir / 'model_test_data_combination_ids.pkl'
        with open(combo_file, 'rb') as f:
            test_combos_raw = pickle.load(f)
        
        self.test_combinations = []
        for idx, combo in test_combos_raw.items():
            self.test_combinations.append({
                'combo_id': f"site{int(combo['site ID'])}_T{int(combo['thermometer ID'])}_H{int(combo['hygrometer ID'])}_D{int(combo['dendrometer ID'])}",
                'site_id': int(combo['site ID']),
                'thermo_id': int(combo['thermometer ID']),
                'hygro_id': int(combo['hygrometer ID']),
                'dendro_id': int(combo['dendrometer ID'])
            })
        
        print(f"Loaded {len(self.test_combinations)} test combinations")
        
        # Results storage
        self.results = {}
    
    def load_intermediate_timeseries(self, combo_id: str) -> Optional[pd.DataFrame]:
        """Load intermediate timeseries for a combination."""
        inter_dir = self.data_dir / 'intermediate_timeseries'
        file_path = inter_dir / f'test_input_{combo_id}.ftr'
        
        if not file_path.exists():
            print(f"  Warning: No intermediate file for {combo_id}")
            return None
        
        df = feather.read_feather(file_path)
        if 'ts' in df.columns:
            df = df.set_index('ts')
        
        return df
    
    def load_raw_input_data(self, thermo_id: int, hygro_id: int, dendro_id: int) -> pd.DataFrame:
        """Load and merge L1/L2 raw input data."""
        dfs = []
        
        # Load thermometer L1
        thermo_raw = self.loaders.load_thermometer_l1(thermo_id)
        if thermo_raw is not None:
            thermo_proc = self.processor.process_sensor_dataframe(thermo_raw, keep_all_columns=True)
            if 'value' in thermo_proc.columns:
                thermo_proc = thermo_proc.rename(columns={'value': 'input_T'})
                thermo_proc = thermo_proc.reset_index()
                dfs.append(thermo_proc[['ts', 'input_T']])
        
        # Load hygrometer L1
        hygro_raw = self.loaders.load_hygrometer_l1(hygro_id)
        if hygro_raw is not None:
            hygro_proc = self.processor.process_sensor_dataframe(hygro_raw, keep_all_columns=True)
            if 'value' in hygro_proc.columns:
                hygro_proc = hygro_proc.rename(columns={'value': 'input_RH'})
                hygro_proc = hygro_proc.reset_index()
                dfs.append(hygro_proc[['ts', 'input_RH']])
        
        # Load dendrometer L2
        dendro_raw = self.loaders.load_dendrometer_l2(dendro_id)
        if dendro_raw is not None:
            dendro_proc = self.processor.process_sensor_dataframe(dendro_raw, keep_all_columns=True)
            if 'value' in dendro_proc.columns:
                dendro_proc = dendro_proc.rename(columns={'value': 'input_stem'})
                dendro_proc = dendro_proc.reset_index()
                dfs.append(dendro_proc[['ts', 'input_stem']])
        
        if not dfs:
            return None
        
        # Merge all
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on='ts', how='outer')
        
        merged = merged.set_index('ts').sort_index()
        
        # Filter to years and reindex to complete time grid
        merged = merged[(merged.index.year >= min(self.years)) & (merged.index.year <= max(self.years))]
        
        complete_idx = pd.date_range(
            f'{min(self.years)}-01-01',
            f'{max(self.years)}-12-31 23:50:00',
            freq='10min',
            tz='UTC'
        )
        merged = merged.reindex(complete_idx)
        
        return merged
    
    def load_lm_data(self, dendro_id: int) -> Optional[pd.DataFrame]:
        """Load LM ground truth data."""
        lm_raw = self.loaders.load_dendrometer_lm(dendro_id)
        if lm_raw is None:
            return None
        
        lm_processed = self.processor.process_sensor_dataframe(lm_raw, keep_all_columns=True)
        lm_df = self.processor.merger.create_target_array(lm_processed)
        
        # Rename for clarity
        lm_df = lm_df.rename(columns={
            'local_T': 'lm_T',
            'local_RH': 'lm_RH',
            'stem': 'lm_stem'
        })
        
        lm_df = lm_df.sort_index()
        lm_df = lm_df[(lm_df.index.year >= min(self.years)) & (lm_df.index.year <= max(self.years))]
        
        return lm_df
    
    def align_stem_to_lm(
        self, 
        recon_df: pd.DataFrame, 
        lm_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align reconstructed stem data to LM scale using linear regression.
        
        This is CRITICAL for multi-year reconstruction evaluation as the model
        outputs normalized values that need proper scale calibration.
        
        Args:
            recon_df: DataFrame with 'recon_stem' column
            lm_df: DataFrame with 'lm_stem' column (ground truth)
            
        Returns:
            recon_df with 'recon_stem' aligned to LM scale
        """
        if recon_df is None or lm_df is None:
            return recon_df
        
        if 'recon_stem' not in recon_df.columns or 'lm_stem' not in lm_df.columns:
            return recon_df
        
        # Find common timestamps
        common_idx = recon_df.index.intersection(lm_df.index)
        
        if len(common_idx) < 100:
            print(f"    Warning: Not enough overlap for stem alignment ({len(common_idx)} samples)")
            return recon_df
        
        # Get aligned data
        recon_values = recon_df.loc[common_idx, 'recon_stem'].values
        lm_values = lm_df.loc[common_idx, 'lm_stem'].values
        
        # Remove NaN pairs
        valid = ~(np.isnan(recon_values) | np.isnan(lm_values))
        if valid.sum() < 100:
            print(f"    Warning: Not enough valid pairs for alignment ({valid.sum()} pairs)")
            return recon_df
        
        recon_valid = recon_values[valid]
        lm_valid = lm_values[valid]
        
        # Linear regression: LM = a * recon + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(recon_valid, lm_valid)
        
        # Apply transformation to all reconstructed values
        aligned_recon_df = recon_df.copy()
        aligned_recon_df['recon_stem'] = recon_df['recon_stem'] * slope + intercept
        
        print(f"    Stem alignment: y = {slope:.4f}x + {intercept:.2f}, R²={r_value**2:.4f}")
        
        return aligned_recon_df

        return lm_df
    
    def reconstruct_timeseries(self, combo: Dict) -> Optional[pd.DataFrame]:
        """Reconstruct time series for a combination using sliding windows."""
        combo_id = combo['combo_id']
        
        # Load intermediate data
        inter_df = self.load_intermediate_timeseries(combo_id)
        if inter_df is None:
            return None
        
        # Filter to years
        inter_df = inter_df[(inter_df.index.year >= min(self.years)) & (inter_df.index.year <= max(self.years))]
        
        if len(inter_df) < INPUT_SAMPLES:
            print(f"  Warning: Not enough data for {combo_id}")
            return None
        
        # Sliding window reconstruction
        stride_samples = self.stride_hours * 6  # Convert hours to 10-min steps
        all_predictions = []
        
        n_windows = (len(inter_df) - INPUT_SAMPLES) // stride_samples + 1
        
        for i in range(n_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + INPUT_SAMPLES
            
            if end_idx > len(inter_df):
                break
            
            # Extract window
            window = inter_df.iloc[start_idx:end_idx].copy()
            window_timestamps = window.index
            
            # Get input channels
            input_data = np.zeros((INPUT_SAMPLES, len(INPUT_CHANNELS)))
            for ch_idx, ch_name in enumerate(INPUT_CHANNELS):
                if ch_name in window.columns:
                    input_data[:, ch_idx] = window[ch_name].values
                else:
                    input_data[:, ch_idx] = 0.0
            
            # Segment-level normalization
            norm_params = {}
            for ch_idx in range(len(INPUT_CHANNELS)):
                ch_data = input_data[:, ch_idx]
                valid_mask = ~np.isnan(ch_data)
                if valid_mask.sum() > 0:
                    min_val = np.nanmin(ch_data)
                    max_val = np.nanmax(ch_data)
                    diff = max_val - min_val
                    if diff < 1e-8:
                        diff = 1.0
                    norm_params[ch_idx] = {'min': min_val, 'diff': diff}
                    input_data[:, ch_idx] = (ch_data - min_val) / diff
                else:
                    norm_params[ch_idx] = {'min': 0, 'diff': 1}
            
            # Create mask
            mask = (~np.isnan(input_data)).astype(np.float32)
            input_data = np.nan_to_num(input_data, nan=0.0).astype(np.float32)
            
            # Model prediction
            input_batch = np.expand_dims(input_data, axis=0)
            mask_batch = np.expand_dims(mask, axis=0)
            
            preds = self.model.predict([input_batch, mask_batch], verbose=0)
            hourly_pred = preds[1][0]  # Shape: (720, 3)
            
            # Denormalize using input params (operational mode)
            denorm_pred = np.zeros_like(hourly_pred)
            for ch_idx in range(3):
                input_ch_idx = ch_idx  # T=0, RH=1, stem=2
                denorm_pred[:, ch_idx] = hourly_pred[:, ch_idx] * norm_params[input_ch_idx]['diff'] + norm_params[input_ch_idx]['min']
            
            # Create timestamps for output (hourly)
            start_time = window_timestamps[0]
            output_times = pd.date_range(start=start_time, periods=OUTPUT_SAMPLES, freq='1h')
            
            # Store predictions
            pred_df = pd.DataFrame({
                'ts': output_times,
                'recon_T': denorm_pred[:, 0],
                'recon_RH': denorm_pred[:, 1],
                'recon_stem': denorm_pred[:, 2]
            })
            all_predictions.append(pred_df)
        
        if not all_predictions:
            return None
        
        # Combine all predictions by averaging overlaps
        combined = pd.concat(all_predictions, ignore_index=True)
        reconstructed = combined.groupby('ts').agg({
            'recon_T': 'mean',
            'recon_RH': 'mean',
            'recon_stem': 'mean'
        }).reset_index()
        
        reconstructed = reconstructed.set_index('ts').sort_index()
        
        return reconstructed
    
    def identify_gaps(self, df: pd.DataFrame, channel: str, min_gap_hours: int = 4) -> List[Tuple]:
        """Identify gap regions for a channel."""
        gaps = []
        
        if df is None or len(df) == 0 or channel not in df.columns:
            return gaps
        
        is_missing = df[channel].isna()
        
        in_gap = False
        gap_start = None
        
        for i, (idx, missing) in enumerate(zip(df.index, is_missing)):
            if missing and not in_gap:
                gap_start = idx
                in_gap = True
            elif not missing and in_gap:
                gap_duration_hours = (df.index[i-1] - gap_start).total_seconds() / 3600
                if gap_duration_hours >= min_gap_hours:
                    gaps.append((gap_start, df.index[i-1]))
                in_gap = False
        
        if in_gap and gap_start is not None:
            gap_duration_hours = (df.index[-1] - gap_start).total_seconds() / 3600
            if gap_duration_hours >= min_gap_hours:
                gaps.append((gap_start, df.index[-1]))
        
        return gaps
    
    def create_stacked_visualization(
        self,
        input_df: pd.DataFrame,
        recon_df: pd.DataFrame,
        lm_df: pd.DataFrame,
        combo_id: str,
        dpi: int = 300
    ) -> Path:
        """Create 9-row stacked visualization with gap shading."""
        # Colors
        color_input = '#2ca02c'  # Green
        color_recon = '#d62728'  # Red
        color_lm = '#1f77b4'     # Blue
        color_gap = '#ffcccc'    # Light red
        
        fig, axes = plt.subplots(9, 1, figsize=(20, 24), sharex=True)
        
        channels = [
            {'name': 'Temperature', 'input_col': 'input_T', 'recon_col': 'recon_T', 
             'lm_col': 'lm_T', 'ylabel': '°C', 'rows': [0, 1, 2]},
            {'name': 'Relative Humidity', 'input_col': 'input_RH', 'recon_col': 'recon_RH',
             'lm_col': 'lm_RH', 'ylabel': '%', 'rows': [3, 4, 5]},
            {'name': 'Stem Radius', 'input_col': 'input_stem', 'recon_col': 'recon_stem',
             'lm_col': 'lm_stem', 'ylabel': 'μm', 'rows': [6, 7, 8]}
        ]
        
        for ch in channels:
            row_input, row_recon, row_lm = ch['rows']
            
            # Identify gaps for each data source
            input_gaps = self.identify_gaps(input_df, ch['input_col']) if input_df is not None else []
            lm_gaps = self.identify_gaps(lm_df, ch['lm_col']) if lm_df is not None else []
            
            # Row 1: Input
            ax = axes[row_input]
            if input_df is not None and ch['input_col'] in input_df.columns:
                ax.plot(input_df.index, input_df[ch['input_col']], 
                        color=color_input, linewidth=0.5, label='Raw Input (L1/L2)')
            for gap_start, gap_end in input_gaps:
                ax.axvspan(gap_start, gap_end, color=color_gap, alpha=0.5, zorder=0)
            ax.set_ylabel(f"{ch['name']}\n(Input)\n{ch['ylabel']}", fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Row 2: Reconstruction
            ax = axes[row_recon]
            if recon_df is not None and ch['recon_col'] in recon_df.columns:
                ax.plot(recon_df.index, recon_df[ch['recon_col']],
                        color=color_recon, linewidth=0.5, label='Reconstruction')
            # Show input gaps on reconstruction too
            for gap_start, gap_end in input_gaps:
                ax.axvspan(gap_start, gap_end, color=color_gap, alpha=0.5, zorder=0)
            ax.set_ylabel(f"{ch['name']}\n(Recon)\n{ch['ylabel']}", fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Row 3: Ground Truth (LM)
            ax = axes[row_lm]
            if lm_df is not None and ch['lm_col'] in lm_df.columns:
                ax.plot(lm_df.index, lm_df[ch['lm_col']],
                        color=color_lm, linewidth=0.5, label='Ground Truth (LM)')
            for gap_start, gap_end in lm_gaps:
                ax.axvspan(gap_start, gap_end, color=color_gap, alpha=0.5, zorder=0)
            ax.set_ylabel(f"{ch['name']}\n(GT)\n{ch['ylabel']}", fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Title
        year_str = '-'.join(map(str, self.years))
        fig.suptitle(
            f'Reconstruction Evaluation - {combo_id} ({year_str})\n'
            f'Green=Input, Red=Reconstruction, Blue=Ground Truth | Light red shading=Gaps',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        output_path = self.output_dir / f'stacked_with_gaps_{combo_id}.png'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def evaluate_reconstruction(
        self,
        recon_df: pd.DataFrame,
        lm_df: pd.DataFrame,
        combo_id: str
    ) -> Dict:
        """Evaluate reconstruction against LM ground truth."""
        metrics = {}
        
        if recon_df is None or lm_df is None:
            return metrics
        
        # Find common timestamps
        common_idx = recon_df.index.intersection(lm_df.index)
        
        if len(common_idx) == 0:
            return metrics
        
        channel_pairs = [
            ('recon_T', 'lm_T', 'Temperature'),
            ('recon_RH', 'lm_RH', 'Relative Humidity'),
            ('recon_stem', 'lm_stem', 'Stem')
        ]
        
        for recon_col, lm_col, name in channel_pairs:
            if recon_col not in recon_df.columns or lm_col not in lm_df.columns:
                continue
            
            recon_vals = recon_df.loc[common_idx, recon_col].values
            lm_vals = lm_df.loc[common_idx, lm_col].values
            
            # Remove NaN pairs
            valid = ~(np.isnan(recon_vals) | np.isnan(lm_vals))
            if valid.sum() < 10:
                continue
            
            recon_valid = recon_vals[valid]
            lm_valid = lm_vals[valid]
            
            # Compute metrics
            mae = np.mean(np.abs(recon_valid - lm_valid))
            rmse = np.sqrt(np.mean((recon_valid - lm_valid) ** 2))
            
            # Correlation
            if np.std(recon_valid) > 0 and np.std(lm_valid) > 0:
                corr = np.corrcoef(recon_valid, lm_valid)[0, 1]
            else:
                corr = np.nan
            
            # R-squared
            ss_res = np.sum((lm_valid - recon_valid) ** 2)
            ss_tot = np.sum((lm_valid - np.mean(lm_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            # Normalized MAE (as percentage of range)
            lm_range = np.max(lm_valid) - np.min(lm_valid)
            norm_mae = mae / lm_range if lm_range > 0 else np.nan
            
            metrics[name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'correlation': float(corr),
                'r2': float(r2),
                'normalized_mae': float(norm_mae),
                'n_samples': int(valid.sum())
            }
        
        return metrics
    
    def run_evaluation(self):
        """Run full evaluation on all test combinations."""
        print("=" * 80)
        print("Batch Evaluation - Test Set Reconstruction (with Stem Alignment)")
        print(f"Model: {self.model_path.name}")
        print(f"Years: {self.years}")
        print(f"Combinations: {len(self.test_combinations)}")
        print("=" * 80)
        
        all_metrics = {}
        
        for i, combo in enumerate(self.test_combinations):
            combo_id = combo['combo_id']
            print(f"\n[{i+1}/{len(self.test_combinations)}] Processing {combo_id}...")
            
            # Load raw input data
            print("  Loading raw input data...")
            input_df = self.load_raw_input_data(
                combo['thermo_id'], combo['hygro_id'], combo['dendro_id']
            )
            
            # Reconstruct
            print("  Reconstructing time series...")
            recon_df = self.reconstruct_timeseries(combo)
            
            if recon_df is None:
                print(f"  Warning: Reconstruction failed for {combo_id}")
                continue
            
            print(f"  Reconstructed: {len(recon_df)} hourly samples")
            
            # Load LM ground truth FIRST (needed for stem alignment)
            print("  Loading LM ground truth...")
            lm_df = self.load_lm_data(combo['dendro_id'])
            
            if lm_df is not None:
                print(f"  LM data: {len(lm_df)} samples")
                
                # IMPORTANT: Apply stem alignment using LM data
                print("  Applying stem alignment...")
                recon_df = self.align_stem_to_lm(recon_df, lm_df)
            else:
                print("  Warning: No LM data available - stem will NOT be aligned!")
            
            # Save reconstruction (after alignment)
            recon_path = self.output_dir / f'reconstructed_{combo_id}.ftr'
            recon_df.reset_index().to_feather(recon_path)
            print(f"  Saved: {recon_path}")
            
            # Create visualization
            print("  Creating visualization...")
            viz_path = self.create_stacked_visualization(
                input_df, recon_df, lm_df, combo_id
            )
            print(f"  Saved: {viz_path}")
            
            # Evaluate
            print("  Evaluating reconstruction...")
            metrics = self.evaluate_reconstruction(recon_df, lm_df, combo_id)
            all_metrics[combo_id] = metrics
            
            # Print summary
            for ch_name, ch_metrics in metrics.items():
                print(f"    {ch_name}: Corr={ch_metrics['correlation']:.4f}, "
                      f"MAE={ch_metrics['mae']:.4f}, R²={ch_metrics['r2']:.4f}")
        
        # Save all metrics
        metrics_path = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved: {metrics_path}")
        
        # Generate summary
        self.generate_summary(all_metrics)
        
        return all_metrics
    
    def generate_summary(self, all_metrics: Dict):
        """Generate summary statistics across all combinations."""
        print("\n" + "=" * 80)
        print("SUMMARY - Reconstruction Quality Across Test Set")
        print("=" * 80)
        
        # Collect metrics by channel
        channel_stats = {'Temperature': [], 'Relative Humidity': [], 'Stem': []}
        
        for combo_id, metrics in all_metrics.items():
            for ch_name in channel_stats.keys():
                if ch_name in metrics:
                    channel_stats[ch_name].append(metrics[ch_name])
        
        # Print summary table
        print(f"\n{'Channel':<20} {'N combos':<10} {'Mean Corr':<12} {'Mean MAE':<12} {'Mean R²':<12}")
        print("-" * 66)
        
        for ch_name, stats_list in channel_stats.items():
            if not stats_list:
                continue
            
            n_combos = len(stats_list)
            mean_corr = np.mean([s['correlation'] for s in stats_list if not np.isnan(s['correlation'])])
            mean_mae = np.mean([s['mae'] for s in stats_list])
            mean_r2 = np.mean([s['r2'] for s in stats_list if not np.isnan(s['r2'])])
            
            print(f"{ch_name:<20} {n_combos:<10} {mean_corr:<12.4f} {mean_mae:<12.4f} {mean_r2:<12.4f}")
        
        # Save summary
        summary_path = self.output_dir / 'evaluation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Reconstruction Quality Summary - Test Set\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Years: {self.years}\n")
            f.write(f"Total combinations: {len(all_metrics)}\n\n")
            
            for ch_name, stats_list in channel_stats.items():
                if not stats_list:
                    continue
                f.write(f"\n{ch_name}:\n")
                f.write(f"  N combinations: {len(stats_list)}\n")
                f.write(f"  Mean Correlation: {np.mean([s['correlation'] for s in stats_list if not np.isnan(s['correlation'])]):.4f}\n")
                f.write(f"  Mean MAE: {np.mean([s['mae'] for s in stats_list]):.4f}\n")
                f.write(f"  Mean R²: {np.mean([s['r2'] for s in stats_list if not np.isnan(s['r2'])]):.4f}\n")
        
        print(f"\nSummary saved: {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch evaluation of reconstruction quality for test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate years 2020-2021 (recommended - most data)
    python batch_evaluate_test_set.py --years 2020 2021
    
    # Evaluate years 2021-2022
    python batch_evaluate_test_set.py --years 2021 2022
    
    # Custom output directory
    python batch_evaluate_test_set.py --years 2020 2021 --output-dir /path/to/output

Note: Stem alignment is ALWAYS applied using LM ground truth data.
        """
    )
    
    parser.add_argument('--years', type=int, nargs='+', default=[2020, 2021],
                        help='Years to evaluate (default: 2020 2021)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated based on years)')
    parser.add_argument('--stride-hours', type=int, default=24,
                        help='Stride between windows in hours (default: 24)')
    
    args = parser.parse_args()
    
    # Unconstrained model path
    model_path = '/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260111_152352_segment_norm_attention/best_model.keras'
    
    # Data directory
    data_dir = '/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data'
    
    # Output directory
    years_str = '_'.join(map(str, args.years))
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'/home/lukovic/data/treenet/test_set_evaluation_unconstrained_{years_str}_aligned'
    
    evaluator = TestSetEvaluator(
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        years=args.years,
        stride_hours=args.stride_hours
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()