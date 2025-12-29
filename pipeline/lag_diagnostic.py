# lag_diagnostic.py
# -*- coding: utf-8 -*-
"""
Automatic Multi-Site Lag Diagnostic Tool
=======================================

Author: M365 Copilot (for Mirko Lukovic)
Date: 2025-12-28

Purpose
-------
Quantify and visualize potential time misalignment (lag) between local 10‑min channels
and global daily drivers across *many sites and segments*. This helps decide, *empirically*,
whether to apply Step 1 (misalignment correction) in your pipeline.

Inputs (expected files in working directory)
--------------------------------------------
- X_train.npy : (N_segments, 4320, 11) normalized per segment
- site_ids_train.npy : (N_segments,) array of site identifiers (str or int)

Optional:
- X_test.npy, site_ids_test.npy (for additional diagnostics)

Channel Map (zero-based)
------------------------
0: local mean temperature (°C), 10-min
1: local relative humidity (%), 10-min
2: local tree stem radius change (µm), 10-min
3: GLOBAL mean temperature (°C), daily cleaned (broadcast in your array)
4: GLOBAL min temperature (°C), daily cleaned
5: GLOBAL max temperature (°C), daily cleaned
6: GLOBAL relative humidity (%), daily cleaned
7: GLOBAL vapor pressure deficit (kPa), daily cleaned
8: GLOBAL precipitation (mm), daily cleaned
9: GLOBAL solar radiation (W m^-2), daily cleaned
10: day of year (DOY), daily

What the script produces
------------------------
- diagnostics/
    - lag_by_segment.csv      (per segment, per pair)
    - lag_by_site.csv         (aggregated stats per site, per pair)
    - recommendation.txt      (human-readable recommendations)
    - plots/
        - correlogram_<pair>_<segmentIdx>.png (optional sampling)
        - violin_<pair>_by_site.png
        - heatmap_significance_<pair>_by_site.png

Pairs diagnosed (default)
-------------------------
- LOCAL_T (ch 0) vs GLOBAL_T_mean (ch 3)
- LOCAL_RH (ch 1) vs GLOBAL_RH (ch 6)
- LOCAL_STEM (ch 2) vs GLOBAL_RADIATION (ch 9)

How to run
----------
$ python lag_diagnostic.py

You can also import and call `run_diagnostics(X, site_ids)` programmatically.

"""

from __future__ import annotations
import os
import math
import typing as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# =====================
# Config & mapping
# =====================
SEQ_LEN_10MIN: int = 4320
HOUR_STEPS: int = 720
STRIDE_PER_HOUR: int = 6
N_CHANNELS: int = 11

CH = {
    'local_T_10m': 0,
    'local_RH_10m': 1,
    'local_stem_10m': 2,
    'global_T_mean_daily': 3,
    'global_RH_daily': 6,
    'global_rad_daily': 9,
}

# Pairs to analyze: (name, local_idx, global_idx, use_abs_corr)
PAIRS = [
    ("T_vs_Tmean", CH['local_T_10m'], CH['global_T_mean_daily'], False),
    ("RH_vs_RHdaily", CH['local_RH_10m'], CH['global_RH_daily'], False),
    ("STEM_vs_RAD", CH['local_stem_10m'], CH['global_rad_daily'], True),
]

MAX_LAG_STEPS: int = 6 * 24  # ±24 hours at 10-min resolution
DETREND: bool = True         # robust detrending before correlation
NORMALIZE: bool = True       # z-score before correlation

# Significance & recommendation thresholds
P_THRESH: float = 0.05
MIN_ABS_LAG_FOR_CORRECTION: int = 6   # ≥6 steps ≈ ≥1 hour
MIN_SIG_FRACTION_FOR_CORRECTION: float = 0.6
MAX_STD_LAG_FOR_CORRECTION: int = 6   # std ≤ 6 steps (≤1 hour), indicates stability

# Output folders
OUT_DIR = os.path.join(os.getcwd(), "diagnostics")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")


# =====================
# Core computations
# =====================

def _detrend_first_diff(x: np.ndarray) -> np.ndarray:
    """Robust detrending via first differences (handles piecewise trends)."""
    return np.diff(x, prepend=x[0]) - np.median(np.diff(x, prepend=x[0]))


def compute_cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int | None,
                              detrend: bool = DETREND, normalize: bool = NORMALIZE) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation for lags in [-max_lag, +max_lag]."""
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if detrend:
        x = _detrend_first_diff(x)
        y = _detrend_first_diff(y)
    if normalize:
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)
    N = len(x)
    corr_full = np.correlate(x, y, mode='full')
    lags_full = np.arange(-N + 1, N)
    if max_lag is None:
        return lags_full, corr_full / N
    idx = np.where((lags_full >= -max_lag) & (lags_full <= max_lag))[0]
    return lags_full[idx], corr_full[idx] / N


def estimate_lag(x: np.ndarray, y: np.ndarray, max_lag: int,
                 use_abs_corr: bool = True) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Estimate lag at which (absolute) correlation peaks."""
    lags, corr = compute_cross_correlation(x, y, max_lag=max_lag)
    if use_abs_corr:
        idx_peak = int(np.argmax(np.abs(corr)))
    else:
        idx_peak = int(np.argmax(corr))
    return int(lags[idx_peak]), float(corr[idx_peak]), lags, corr


def effective_sample_size(x: np.ndarray, y: np.ndarray) -> float:
    """Effective sample size accounting for lag-1 autocorrelation (Bretherton et al. 1999)."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 5:
        return max(5, n)
    # Lag-1 autocorrelation (guard against NaN)
    try:
        r1x = np.corrcoef(x[:-1], x[1:])[0, 1]
        r1y = np.corrcoef(y[:-1], y[1:])[0, 1]
    except Exception:
        r1x, r1y = 0.0, 0.0
    Neff = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    return float(max(5.0, Neff))


def fisher_z_pvalue(r: float, Neff: float) -> tuple[float, float]:
    """Fisher z-transform significance for correlation r with effective N."""
    r = float(r)
    if abs(r) >= 0.999:
        r = math.copysign(0.999, r)
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(max(3.0, Neff - 3.0))
    z_score = z / se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return z_score, p_value


def block_bootstrap_lag(x: np.ndarray, y: np.ndarray, max_lag: int,
                        n_boot: int = 200, block_len: int = 6,
                        use_abs_corr: bool = True) -> tuple[float, float]:
    """Block bootstrap lag CI (95%).
    - block_len in steps (~6 = 1 hour)
    Returns: (ci_low, ci_high) for lag in steps.
    """
    T = len(x)
    if T < block_len:
        return (np.nan, np.nan)
    lags_boot = []
    rng = np.random.default_rng(123)

    # Preprocess once to reduce work
    def preprocess(v):
        v = np.asarray(v, dtype=float)
        if DETREND:
            v = _detrend_first_diff(v)
        if NORMALIZE:
            v = (v - v.mean()) / (v.std() + 1e-8)
        return v

    x0 = preprocess(x)
    y0 = preprocess(y)

    # Build blocks indices
    n_blocks = T // block_len
    starts = np.arange(0, T - block_len + 1, block_len)

    for _ in range(n_boot):
        # Sample blocks with replacement
        sel = rng.choice(starts, size=n_blocks, replace=True)
        xb = np.concatenate([x0[s:s+block_len] for s in sel])
        yb = np.concatenate([y0[s:s+block_len] for s in sel])
        l, r, _, _ = estimate_lag(xb, yb, max_lag=max_lag, use_abs_corr=use_abs_corr)
        lags_boot.append(l)

    lags_boot = np.array(lags_boot)
    return float(np.nanpercentile(lags_boot, 2.5)), float(np.nanpercentile(lags_boot, 97.5))


# =====================
# Plotting helpers
# =====================

def plot_correlogram(lags: np.ndarray, corr: np.ndarray, best_lag: int,
                     title: str, save_path: str | None = None):
    plt.figure(figsize=(9, 4))
    plt.plot(lags, corr, '-k', lw=1.5)
    plt.axvline(best_lag, color='red', linestyle='--', lw=1.5, label=f"best lag = {best_lag}")
    plt.xlabel("Lag (steps)")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_violin_by_site(df: pd.DataFrame, pair_name: str, save_path: str):
    """Violin plot of lag distributions per site for the given pair."""
    plt.figure(figsize=(12, 6))
    # We expect df columns: ['site_id', 'pair', 'lag_steps']
    dsub = df[df['pair'] == pair_name]
    # sort sites by median lag
    order = dsub.groupby('site_id')['lag_steps'].median().sort_values().index.tolist()
    data = [dsub[dsub['site_id'] == sid]['lag_steps'].values for sid in order]
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(order)+1), order, rotation=45, ha='right')
    plt.axhline(0, color='gray', lw=1)
    plt.ylabel("Lag (steps; 6 = 1 hour)")
    plt.title(f"Lag distribution by site — {pair_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_heatmap_significance(df: pd.DataFrame, pair_name: str, save_path: str):
    """Heatmap of fraction significant per site for the pair."""
    dsub = df[df['pair'] == pair_name]
    sig_frac = dsub.groupby('site_id')['is_significant'].mean().to_frame('frac_sig')
    sites = sig_frac.index.tolist()
    vals = sig_frac['frac_sig'].values.reshape(-1, 1)

    plt.figure(figsize=(6, max(3, len(sites)*0.3)))
    plt.imshow(vals, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Fraction significant (p < 0.05)')
    plt.yticks(np.arange(len(sites)), sites)
    plt.xticks([0], [pair_name])
    plt.title(f"Significance by site — {pair_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =====================
# Diagnostics runner
# =====================

def run_diagnostics(X: np.ndarray, site_ids: np.ndarray,
                    sample_plots: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run lag diagnostics across segments and sites.
    Args:
        X: (N, 4320, 11) inputs
        site_ids: (N,) site identifiers (str or int)
        sample_plots: number of random segments for correlogram plots per pair
    Returns:
        df_segments: per-segment results
        df_sites: per-site aggregated stats + recommendation
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    N = X.shape[0]
    rng = np.random.default_rng(123)

    rows = []

    for i in range(N):
        sid = str(site_ids[i])
        for pair_name, li, gi, use_abs in PAIRS:
            x_local = X[i, :, li]
            y_global = X[i, :, gi]
            # Estimate lag
            lag, peak_corr, lags, corr = estimate_lag(x_local, y_global, max_lag=MAX_LAG_STEPS, use_abs_corr=use_abs)
            # Effective N and significance
            Neff = effective_sample_size(x_local, y_global)
            z_score, p_value = fisher_z_pvalue(peak_corr, Neff)
            is_sig = bool(p_value < P_THRESH)
            # CI via block bootstrap
            ci_low, ci_high = block_bootstrap_lag(x_local, y_global, max_lag=MAX_LAG_STEPS, n_boot=200, block_len=6, use_abs_corr=use_abs)

            rows.append({
                'segment_idx': i,
                'site_id': sid,
                'pair': pair_name,
                'lag_steps': lag,
                'lag_hours': lag / STRIDE_PER_HOUR,
                'peak_corr': peak_corr,
                'Neff': Neff,
                'z_score': z_score,
                'p_value': p_value,
                'is_significant': is_sig,
                'ci_low_steps': ci_low,
                'ci_high_steps': ci_high,
            })

    df_segments = pd.DataFrame(rows)

    # Aggregations per site & pair
    def _recommendation(group: pd.DataFrame) -> str:
        median_abs_lag = group['lag_steps'].abs().median()
        std_lag = group['lag_steps'].std()
        frac_sig = group['is_significant'].mean()
        if (median_abs_lag >= MIN_ABS_LAG_FOR_CORRECTION and
            frac_sig >= MIN_SIG_FRACTION_FOR_CORRECTION and
            (std_lag <= MAX_STD_LAG_FOR_CORRECTION or np.isnan(std_lag))):
            return "APPLY_FIXED_LAG_CORRECTION"
        else:
            return "SKIP_CORRECTION"

    agg = df_segments.groupby(['site_id', 'pair']).agg(
        n_segments=('segment_idx', 'count'),
        median_lag_steps=('lag_steps', 'median'),
        mean_lag_steps=('lag_steps', 'mean'),
        std_lag_steps=('lag_steps', 'std'),
        frac_significant=('is_significant', 'mean'),
        median_abs_lag_steps=('lag_steps', lambda s: np.median(np.abs(s))),
    ).reset_index()

    # Apply recommendations per site/pair
    agg['recommendation'] = agg.groupby(['site_id', 'pair']).apply(_recommendation).reset_index(drop=True)

    # Save CSVs
    df_segments.to_csv(os.path.join(OUT_DIR, 'lag_by_segment.csv'), index=False)
    agg.to_csv(os.path.join(OUT_DIR, 'lag_by_site.csv'), index=False)

    # Human-readable recommendations
    with open(os.path.join(OUT_DIR, 'recommendation.txt'), 'w', encoding='utf-8') as f:
        f.write("Lag correction recommendations (per site × pair)\n")
        f.write("Thresholds: abs(median lag) ≥ %d steps; frac significant ≥ %.2f; std ≤ %d steps\n\n" % (
            MIN_ABS_LAG_FOR_CORRECTION, MIN_SIG_FRACTION_FOR_CORRECTION, MAX_STD_LAG_FOR_CORRECTION))
        for _, row in agg.iterrows():
            f.write(f"Site={row['site_id']}, Pair={row['pair']}, n={int(row['n_segments'])}, "
                    f"median_lag={row['median_lag_steps']:.1f} steps, frac_sig={row['frac_significant']:.2f} → "
                    f"{row['recommendation']}\n")

    # Plots: violin per pair, heatmap significance per pair
    for pair_name, _, _, _ in PAIRS:
        plot_violin_by_site(df_segments, pair_name, save_path=os.path.join(PLOTS_DIR, f"violin_{pair_name}_by_site.png"))
        plot_heatmap_significance(df_segments, pair_name, save_path=os.path.join(PLOTS_DIR, f"heatmap_significance_{pair_name}_by_site.png"))

    # Optional: sample correlograms for a few random segments per pair
    if sample_plots > 0:
        for pair_name, li, gi, use_abs in PAIRS:
            sel = rng.choice(np.arange(N), size=min(sample_plots, N), replace=False)
            for i in sel:
                x_local = X[i, :, li]
                y_global = X[i, :, gi]
                lag, peak_corr, lags, corr = estimate_lag(x_local, y_global, max_lag=MAX_LAG_STEPS, use_abs_corr=use_abs)
                title = f"Correlogram seg={i} site={site_ids[i]} pair={pair_name}"
                save_p = os.path.join(PLOTS_DIR, f"correlogram_{pair_name}_seg{i}.png")
                plot_correlogram(lags, corr, lag, title=title, save_path=save_p)

    return df_segments, agg


# =====================
# Script entry point
# =====================
if __name__ == '__main__':
    # Load train arrays and site IDs
    if not all(os.path.exists(p) for p in ['X_train.npy', 'site_ids_train.npy']):
        print("Please provide X_train.npy and site_ids_train.npy in the working directory.")
        print("X_train.npy shape must be (N, 4320, 11); site_ids_train.npy shape must be (N,)")
        raise SystemExit(0)

    X_train = np.load('X_train.npy')
    site_ids_train = np.load('site_ids_train.npy')

    assert X_train.shape[1:] == (SEQ_LEN_10MIN, N_CHANNELS), "X_train shape mismatch"
    assert site_ids_train.shape[0] == X_train.shape[0], "site_ids_train length mismatch"

    print("Running lag diagnostics on training set...")
    df_seg, df_site = run_diagnostics(X_train, site_ids_train, sample_plots=5)
    print("Saved:")
    print(" -", os.path.join(OUT_DIR, 'lag_by_segment.csv'))
    print(" -", os.path.join(OUT_DIR, 'lag_by_site.csv'))
    print(" -", os.path.join(OUT_DIR, 'recommendation.txt'))
    print(" - plots in:", PLOTS_DIR)
