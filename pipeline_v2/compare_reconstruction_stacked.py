#!/usr/bin/env python3
"""
Create stacked visualization comparing unconstrained vs constrained model outputs.
Shows time series with gap annotations for both models side by side.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Tuple

def load_reconstruction(filepath: str) -> pd.DataFrame:
    """Load reconstructed time series."""
    return pd.read_feather(filepath)

def create_stacked_comparison(
    recon_unconstrained: pd.DataFrame,
    recon_constrained: pd.DataFrame,
    intermediate_path: str,
    combo_name: str,
    output_path: str
):
    """Create stacked comparison visualization."""
    
    # Load intermediate for LM ground truth
    df_lm = pd.read_feather(intermediate_path)
    
    # Get time range
    start_date = max(recon_unconstrained['ts'].min(), recon_constrained['ts'].min())
    end_date = min(recon_unconstrained['ts'].max(), recon_constrained['ts'].max())
    
    # Filter to common range
    mask_unc = (recon_unconstrained['ts'] >= start_date) & (recon_unconstrained['ts'] <= end_date)
    mask_con = (recon_constrained['ts'] >= start_date) & (recon_constrained['ts'] <= end_date)
    mask_lm = (df_lm['ts'] >= start_date) & (df_lm['ts'] <= end_date)
    
    unc = recon_unconstrained[mask_unc].copy()
    con = recon_constrained[mask_con].copy()
    lm = df_lm[mask_lm].copy()
    
    # Resample LM to hourly
    lm = lm.set_index('ts')
    lm_hourly = lm.resample('1h').mean().reset_index()
    
    # Create figure with 6 subplots (3 variables x 2 models)
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
    fig.suptitle(f'{combo_name}\nUnconstrained vs Constrained Model Comparison', fontsize=14, fontweight='bold')
    
    # Column mappings (recon_col_unc, recon_col_con, lm_col, ylabel, label, color)
    channels = [
        ('local_T', 'recon_T', 'temp_treenet', 'Temperature (°C)', 'Reconstructed T', 'tab:red'),
        ('local_RH', 'recon_RH', 'rh_treenet', 'Relative Humidity (%)', 'Reconstructed RH', 'tab:green'),
        ('stem', 'recon_stem', 'stem', 'Stem (μm)', 'Reconstructed Stem', 'tab:blue')
    ]
    
    for row, (unc_col, con_col, lm_col, ylabel, label, color) in enumerate(channels):
        # Unconstrained (left)
        ax = axes[row, 0]
        if row == 0:
            ax.set_title('Unconstrained Model', fontweight='bold')
        
        # Plot LM
        if lm_col in lm_hourly.columns:
            ax.plot(lm_hourly['ts'], lm_hourly[lm_col], 'k-', alpha=0.7, linewidth=0.8, label='LM Ground Truth')
        
        # Plot reconstruction
        ax.plot(unc['ts'], unc[unc_col], color=color, alpha=0.8, linewidth=0.5, label=label)
        
        ax.set_ylabel(ylabel)
        if row == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Constrained (right)
        ax = axes[row, 1]
        if row == 0:
            ax.set_title('Constrained RH Model', fontweight='bold')
        
        # Plot LM
        if lm_col in lm_hourly.columns:
            ax.plot(lm_hourly['ts'], lm_hourly[lm_col], 'k-', alpha=0.7, linewidth=0.8, label='LM Ground Truth')
        
        # Plot reconstruction
        ax.plot(con['ts'], con[con_col], color=color, alpha=0.8, linewidth=0.5, label=label)
        
        if row == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Format x-axis
    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths
    combo_name = "site22_T119_H118_D120"
    
    unconstrained_recon = "/home/lukovic/data/treenet/reconstructions_aligned_v2_2021_2022/aligned_site22_T119_H118_D120.ftr"
    constrained_recon = "/home/lukovic/data/treenet/reconstructions_constrained_site22/test_input_site22_T119_H118_D120_reconstructed.feather"
    intermediate_path = "/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_segment_norm_all_combos/model_data/intermediate_timeseries/test_input_site22_T119_H118_D120.ftr"
    
    output_dir = Path("/home/lukovic/data/treenet/rh_constraint_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stacked_comparison_{combo_name}.png"
    
    print(f"Loading unconstrained reconstruction: {unconstrained_recon}")
    recon_unc = load_reconstruction(unconstrained_recon)
    print(f"  Rows: {len(recon_unc)}, Time range: {recon_unc['ts'].min()} to {recon_unc['ts'].max()}")
    
    print(f"Loading constrained reconstruction: {constrained_recon}")
    recon_con = load_reconstruction(constrained_recon)
    print(f"  Rows: {len(recon_con)}, Time range: {recon_con['ts'].min()} to {recon_con['ts'].max()}")
    
    print("Creating comparison visualization...")
    create_stacked_comparison(
        recon_unc, recon_con, intermediate_path, combo_name, str(output_path)
    )


if __name__ == '__main__':
    main()
