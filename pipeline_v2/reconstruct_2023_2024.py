#!/usr/bin/env python3
"""
Reconstruct 2023-2024 for 10 combinations with stem alignment.
"""
import sys
sys.path.insert(0, '.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from src.models.tcn import TCNBlock, PositionalEncoding
import warnings
warnings.filterwarnings('ignore')

# Constants
SEGMENT_DAYS = 30
INPUT_STEPS_PER_HOUR = 6
OUTPUT_STEPS_PER_HOUR = 1
INPUT_SAMPLES = SEGMENT_DAYS * 24 * INPUT_STEPS_PER_HOUR  # 4320
OUTPUT_SAMPLES = SEGMENT_DAYS * 24 * OUTPUT_STEPS_PER_HOUR  # 720

INPUT_CHANNELS = ['temp_treenet', 'rh_treenet', 'stem',
                  'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy']

def constrained_hourly_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

print("Loading model...")
model_path = '/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/experiments/20260111_152352_segment_norm_attention/best_model.keras'
model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        'TCNBlock': TCNBlock,
        'PositionalEncoding': PositionalEncoding,
        'constrained_hourly_loss': constrained_hourly_loss
    }
)
print("Model loaded")

intermediate_dir = Path('/home/lukovic/data/treenet/reconstruction_2023_2024/intermediate_timeseries')
output_dir = Path('/home/lukovic/data/treenet/reconstruction_2023_2024/reconstructions')
output_dir.mkdir(parents=True, exist_ok=True)


def prepare_segment(df_segment):
    """Prepare input array with segment-level normalization."""
    input_data = df_segment[INPUT_CHANNELS].values.astype(np.float32)
    if len(input_data) != INPUT_SAMPLES:
        return None, None, None, False
    
    norm_params = {}
    for i, col in enumerate(INPUT_CHANNELS):
        min_val = np.nanmin(input_data[:, i])
        max_val = np.nanmax(input_data[:, i])
        diff = max_val - min_val
        norm_params[col] = {'min': float(min_val), 'max': float(max_val)}
        if diff > 1e-10:
            input_data[:, i] = (input_data[:, i] - min_val) / diff
        else:
            input_data[:, i] = 0.0
    
    mask = (~np.isnan(input_data)).astype(np.float32)
    input_array = np.nan_to_num(input_data, nan=0.0)
    return input_array, mask, norm_params, True


def denormalize_output(output, norm_params):
    """Denormalize output using input parameters."""
    output_denorm = np.zeros_like(output)
    for i, ch in enumerate(['temp_treenet', 'rh_treenet', 'stem']):
        if ch in norm_params:
            mn = norm_params[ch]['min']
            mx = norm_params[ch]['max']
            if mx - mn > 1e-10:
                output_denorm[:, i] = output[:, i] * (mx - mn) + mn
            else:
                output_denorm[:, i] = mn
    return output_denorm


def reconstruct_combination(df, stride_hours=24):
    """Reconstruct time series for a single combination."""
    stride_samples = stride_hours * INPUT_STEPS_PER_HOUR
    n_windows = (len(df) - INPUT_SAMPLES) // stride_samples + 1
    
    results = []
    for i in range(n_windows):
        start_idx = i * stride_samples
        end_idx = start_idx + INPUT_SAMPLES
        if end_idx > len(df):
            break
        
        segment_df = df.iloc[start_idx:end_idx]
        input_arr, mask, norm_params, valid = prepare_segment(segment_df)
        if not valid:
            continue
        
        pred = model.predict(
            [np.expand_dims(input_arr, 0), np.expand_dims(mask, 0)],
            verbose=0
        )
        
        # Model outputs: [10min_output, hourly_output] - we use hourly
        pred_hourly = pred[1][0] if isinstance(pred, list) else pred[0]
        pred_hourly = denormalize_output(pred_hourly, norm_params)
        
        segment_start = segment_df['ts'].iloc[0]
        hourly_times = pd.date_range(start=segment_start, periods=OUTPUT_SAMPLES, freq='h')
        
        segment_results = pd.DataFrame({
            'ts': hourly_times,
            'temp_pred': pred_hourly[:, 0],
            'rh_pred': pred_hourly[:, 1],
            'stem_pred': pred_hourly[:, 2]
        })
        results.append(segment_results)
    
    if not results:
        return None
    
    all_results = pd.concat(results, ignore_index=True)
    reconstructed = all_results.groupby('ts').agg({
        'temp_pred': 'mean',
        'rh_pred': 'mean',
        'stem_pred': 'mean'
    }).reset_index()
    
    return reconstructed


# Process each file
all_metrics = []
for ftr_file in sorted(intermediate_dir.glob('*.ftr')):
    combo_name = ftr_file.stem
    print(f"\n{'='*80}")
    print(f"Processing: {combo_name}")
    
    data = pd.read_feather(ftr_file)
    data['ts'] = pd.to_datetime(data['ts'])
    print(f"  Data: {len(data):,} rows")
    
    recon = reconstruct_combination(data)
    if recon is None:
        print("  ✗ Failed")
        continue
    
    print(f"  Reconstructed: {len(recon):,} hours")
    
    # Get ground truth (resample to hourly)
    data_hourly = data.set_index('ts').resample('h').mean()
    
    # Align stem using Nov-Dec 2022
    align_start = '2022-11-01'
    align_end = '2023-01-01'
    
    recon_align = recon[(recon['ts'] >= align_start) & (recon['ts'] < align_end)].set_index('ts')
    gt_align = data_hourly.loc[align_start:align_end, 'stem'].dropna()
    
    common_idx = recon_align.index.intersection(gt_align.index)
    if len(common_idx) > 100:
        pred_stem = recon_align.loc[common_idx, 'stem_pred'].values
        gt_stem = gt_align.loc[common_idx].values
        
        offset = np.mean(gt_stem - pred_stem)
        recon['stem_pred_aligned'] = recon['stem_pred'] + offset
        print(f"  Stem offset: {offset:.1f} µm ({len(common_idx):,} samples)")
    else:
        print(f"  Warning: Only {len(common_idx)} alignment samples")
        recon['stem_pred_aligned'] = recon['stem_pred']
    
    # Add ground truth
    recon = recon.set_index('ts')
    recon['temp_gt'] = data_hourly.loc[recon.index, 'temp_treenet']
    recon['rh_gt'] = data_hourly.loc[recon.index, 'rh_treenet']
    recon['stem_gt'] = data_hourly.loc[recon.index, 'stem']
    recon = recon.reset_index()
    
    # Save
    output_file = output_dir / f'{combo_name}_reconstruction.ftr'
    recon.to_feather(output_file)
    print(f"  ✓ Saved")
    
    # Calculate metrics for 2023-2024
    eval_start = '2023-01-01'
    eval_end = '2025-01-01'
    eval_df = recon[(recon['ts'] >= eval_start) & (recon['ts'] < eval_end)]
    
    combo_metrics = {'combo': combo_name}
    print(f"  Metrics for 2023-2024:")
    
    for channel, pred_col, gt_col in [
        ('T', 'temp_pred', 'temp_gt'),
        ('RH', 'rh_pred', 'rh_gt'),
        ('Stem', 'stem_pred_aligned', 'stem_gt')
    ]:
        mask = eval_df[pred_col].notna() & eval_df[gt_col].notna()
        if mask.sum() > 0:
            pred = eval_df.loc[mask, pred_col].values
            gt = eval_df.loc[mask, gt_col].values
            
            mse = np.mean((pred - gt) ** 2)
            corr = np.corrcoef(pred, gt)[0, 1]
            r2 = corr ** 2
            
            combo_metrics[f'{channel}_MSE'] = mse
            combo_metrics[f'{channel}_R2'] = r2
            combo_metrics[f'{channel}_Corr'] = corr
            combo_metrics[f'{channel}_N'] = int(mask.sum())
            
            print(f"    {channel}: MSE={mse:.1f}, R²={r2:.3f}, Corr={corr:.3f}")
    
    all_metrics.append(combo_metrics)

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY METRICS")
print(f"{'='*80}")

metrics_df = pd.DataFrame(all_metrics)
print(metrics_df.to_string())

print(f"\nMean Metrics:")
for ch in ['T', 'RH', 'Stem']:
    mse_col = f'{ch}_MSE'
    r2_col = f'{ch}_R2'
    corr_col = f'{ch}_Corr'
    if mse_col in metrics_df.columns:
        print(f"  {ch}: MSE={metrics_df[mse_col].mean():.1f}, R²={metrics_df[r2_col].mean():.3f}, Corr={metrics_df[corr_col].mean():.3f}")

# Save summary
metrics_df.to_csv(output_dir / 'metrics_summary.csv', index=False)
print(f"\nSaved to: {output_dir / 'metrics_summary.csv'}")
print(f"Output directory: {output_dir}")
