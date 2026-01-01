# simple_builder.py
# -*- coding: utf-8 -*-
"""
Minimal TreeNet → ML preprocessor (single site / single (T,RH,stem) triple)
--------------------------------------------------------------------------

This script builds one calendar-year segment (X @ 10‑min, 11 channels; y @ 1‑h, 3 channels)
for a specified **site_id** and **instrument IDs** (thermometer, hygrometer, dendrometer L2, dendrometer LM).

Design goals:
- **No NaNs** in outputs: robust quantile scaling and fallbacks.
- **Strategy‑B** broadcast of daily civil globals → UTC 10‑min grid.
- Deterministic binning of locals, robust LM hourly resampling.
- Simple parameters and clear prints.

Usage (example for site 3 / 2019 / T=10 / RH=8 / D=18):

python3 simple_builder.py \
  --out_root ./simple_out \
  --site_id 3 \
  --year 2019 \
  --tz Europe/Zurich \
  --thermo_dir /storage/thermometer_l1 \
  --hygro_dir  /storage/hygrometer_l1 \
  --dendro_l2_dir /storage/dendrometer_l2 \
  --dendro_lm_dir /storage/dendrometer_lm \
  --meteo_dir /storage/meteo_daily_civil \
  --t_id 10 --h_id 8 --d_id 18

Outputs:
- simple_out/X.npy (52560, 11)
- simple_out/y.npy (8760, 3)
"""

from __future__ import annotations
import os
import re
import argparse
import typing as t
import numpy as np
import pandas as pd

FREQ_10M = '10min'
FREQ_1H  = '1h'
SEQ_LEN_10MIN = 52560
HOUR_STEPS    = 8760
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad']

# -----------------------------
# Indices & windows
# -----------------------------
def make_year_10m_index_utc(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{year}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_10M)
    # drop Feb 29 if present
    return idx[~((idx.month==2)&(idx.day==29))]

# -----------------------------
# IO helpers
# -----------------------------
def _pick_ts_col(df: pd.DataFrame) -> str:
    for c in ('ts','timestamp','time','date_time','datetime'):
        if c in df.columns:
            return c
    raise ValueError('timestamp column not found')

def _pick_val_col(df: pd.DataFrame, preferred: t.Sequence[str]) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    if 'value' in df.columns:
        return 'value'
    # choose the numeric column with most non‑NaNs
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError('no numeric value column')
    counts = sorted([(df[c].notna().sum(), c) for c in numeric], reverse=True)
    return counts[0][1]

def read_series_utc(series_id: int, dir_path: str, local_tz: str, hint: str) -> pd.Series:
    pat = re.compile(rf'.*series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(dir_path) if pat.match(fn)]
    if not matches:
        raise FileNotFoundError(f'{series_id} not found in {dir_path}')
    fp = os.path.join(dir_path, matches[0])
    df = pd.read_feather(fp)
    ts_col = _pick_ts_col(df)
    if hint == 'thermo':
        val_col = _pick_val_col(df, ('temp','temperature','temperature_mean','value'))
    elif hint == 'hygro':
        val_col = _pick_val_col(df, ('rh','relhum','rh_mean','relative_humidity','value'))
    elif hint == 'dendro_l2':
        val_col = _pick_val_col(df, ('value','radius','rad','l2','stem_radius_change'))
    else:
        val_col = _pick_val_col(df, ())
    ts = pd.to_datetime(df[ts_col], utc=False)
    ts = ts.dt.tz_localize(local_tz) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(local_tz)
    ts = ts.dt.tz_convert('UTC')
    s = pd.Series(df[val_col].to_numpy(), index=ts).sort_index()
    # collapse exact duplicates
    s = s[~s.index.duplicated(keep='mean')]
    return s

def bin_10m(s: pd.Series, how: str='median') -> pd.Series:
    if s.empty:
        return s
    s = s.copy().tz_convert('UTC').sort_index()
    agg = {'median':'median','mean':'mean'}.get(how,'median')
    return s.resample(FREQ_10M, origin='start_day', label='left').agg(agg)

def read_lm_hourly(series_id: int, lm_dir: str, local_tz: str) -> pd.DataFrame:
    pat_hour = re.compile(rf'dendrometer_lm_hourly_series_id_{series_id}\.ftr$')
    pat_raw  = re.compile(rf'dendrometer_lm_series_id_{series_id}\.ftr$')
    m_hour = [fn for fn in os.listdir(lm_dir) if pat_hour.match(fn)]
    if m_hour:
        df = pd.read_feather(os.path.join(lm_dir, m_hour[0]))
        ts_col = _pick_ts_col(df)
        ts = pd.to_datetime(df[ts_col], utc=False)
        ts = ts.dt.tz_localize(local_tz) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(local_tz)
        ts = ts.dt.tz_convert('UTC')
        df = df.set_index(ts)
    else:
        m_raw = [fn for fn in os.listdir(lm_dir) if pat_raw.match(fn)]
        if not m_raw:
            raise FileNotFoundError(f'LM {series_id} not found')
        df = pd.read_feather(os.path.join(lm_dir, m_raw[0]))
        ts_col = _pick_ts_col(df)
        ts = pd.to_datetime(df[ts_col], utc=False)
        ts = ts.dt.tz_localize(local_tz) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(local_tz)
        ts = ts.dt.tz_convert('UTC')
        df = df.set_index(ts)
        # resample to hourly
        for col in ('value','temp','rh'):
            if col not in df.columns:
                df[col] = np.nan
        df = df[['value','temp','rh']].sort_index().resample(FREQ_1H, origin='start_day', label='left').median()
    # ensure canonical names
    out = pd.DataFrame({
        'stem': df['value'] if 'value' in df.columns else df.get('stem', np.nan),
        'local_T': df['temp'] if 'temp' in df.columns else df.get('local_T', np.nan),
        'local_RH': df['rh'] if 'rh' in df.columns else df.get('local_RH', np.nan),
    })
    return out

def read_daily_globals_civil(meteo_dir: str, site_id: int) -> pd.DataFrame:
    # find a CSV with the site id embedded
    mapping = {}
    for fn in os.listdir(meteo_dir):
        if fn.endswith('.csv'):
            m = re.findall(r'\d+', fn)
            if m:
                mapping[int(m[0])] = os.path.join(meteo_dir, fn)
    if site_id not in mapping:
        raise FileNotFoundError('global daily CSV not found for site')
    df = pd.read_csv(mapping[site_id])
    if 'ts' not in df.columns:
        raise ValueError('global CSV must have ts (civil date)')
    idx = pd.to_datetime(df['ts'], utc=False).dt.normalize()
    # required columns
    rename = {'tas':'g_tmean','tasmin':'g_tmin','tasmax':'g_tmax','rh':'g_rh','vpd':'g_vpd','pr':'g_pr','gh':'g_rad'}
    for src,dst in rename.items():
        if src not in df.columns:
            raise ValueError(f'missing {src} in globals CSV')
        df[dst] = df[src]
    out = df.set_index(idx)[list(rename.values())]
    out = out[~out.index.duplicated(keep='last')]
    return out

def strategy_b_broadcast(daily_civil: pd.DataFrame, idx_10m: pd.DatetimeIndex, tz_local: str) -> pd.DataFrame:
    civil_days = idx_10m.tz_convert(tz_local).normalize()
    dc = daily_civil.copy()
    dc.index = pd.DatetimeIndex(pd.to_datetime(dc.index)).normalize()
    out = dc.reindex(civil_days).ffill()
    out.index = idx_10m
    return out

# -----------------------------
# Scaling helpers (robust)
# -----------------------------
def safe_quantiles(arr: np.ndarray, fallback: np.ndarray|None=None, q_low=5.0, q_high=95.0) -> tuple[float,float]:
    """Compute robust quantiles; if arr is all-NaN, use fallback; if still bad, use [0,1]."""
    def q(a):
        a = a.astype(float)
        if np.all(np.isnan(a)):
            return np.nan, np.nan
        q1 = float(np.nanpercentile(a, q_low)); q2 = float(np.nanpercentile(a, q_high))
        if not np.isfinite(q1): q1 = float(np.nanmin(a))
        if not np.isfinite(q2): q2 = float(np.nanmax(a))
        if not np.isfinite(q1) or not np.isfinite(q2):
            return np.nan, np.nan
        if q2 <= q1: q2 = q1 + 1e-6
        return q1, q2
    q1,q2 = q(arr)
    if (not np.isfinite(q1)) or (not np.isfinite(q2)):
        if fallback is not None:
            q1,q2 = q(fallback)
    if (not np.isfinite(q1)) or (not np.isfinite(q2)):
        q1,q2 = 0.0, 1.0
    return q1,q2

def normalize(arr: np.ndarray, q1: float, q2: float) -> np.ndarray:
    out = (arr - q1) / (q2 - q1)
    out = np.clip(out, -0.1, 1.1)
    return out

# -----------------------------
# Main assembly
# -----------------------------
def build_one_segment(
    out_root: str,
    site_id: int,
    year: int,
    tz_local: str,
    thermo_dir: str,
    hygro_dir: str,
    d2_dir: str,
    lm_dir: str,
    meteo_dir: str,
    t_id: int,
    h_id: int,
    d_id: int,
    stem_mode: str='absolute',
    allow_missing_locals: bool=True,
    min_local_coverage: float=0.5,
) -> tuple[np.ndarray, np.ndarray]:
    os.makedirs(out_root, exist_ok=True)

    # Indices
    idx_10m = make_year_10m_index_utc(year)
    # Globals
    daily = read_daily_globals_civil(meteo_dir, site_id)
    glb_10m = strategy_b_broadcast(daily, idx_10m, tz_local)

    # Locals
    sT = bin_10m(read_series_utc(t_id, thermo_dir, tz_local, 'thermo')).reindex(idx_10m)
    sRH= bin_10m(read_series_utc(h_id, hygro_dir,  tz_local, 'hygro')).reindex(idx_10m)
    sST= bin_10m(read_series_utc(d_id, d2_dir,     tz_local, 'dendro_l2')).reindex(idx_10m)
    if stem_mode == 'delta':
        sST = sST.diff()

    # Coverage
    def cov(s: pd.Series) -> float:
        v = s.to_numpy(); n=v.size
        return float(np.sum(~np.isnan(v))/n) if n else 0.0
    covT, covRH, covST = cov(sT), cov(sRH), cov(sST)
    print(f"Coverage T={covT:.3f} RH={covRH:.3f} STEM={covST:.3f}")
    if covST < min_local_coverage:
        raise RuntimeError('Stem coverage below threshold; cannot build segment without stem')

    # Substitution policy (inputs only)
    used_global_T  = allow_missing_locals and (covT  < min_local_coverage)
    used_global_RH = allow_missing_locals and (covRH < min_local_coverage)

    # Scalers (robust):
    # - For T: use sT if not substituting; else use glb_10m['g_tmean'] as fallback
    # - For RH: use sRH or glb_10m['g_rh']
    qT1,qT2 = safe_quantiles(sT.to_numpy(), fallback=glb_10m['g_tmean'].to_numpy())
    qR1,qR2 = safe_quantiles(sRH.to_numpy(), fallback=glb_10m['g_rh'].to_numpy())
    qS1,qS2 = safe_quantiles(sST.to_numpy())

    # Normalize locals
    T_n  = normalize((glb_10m['g_tmean'].to_numpy() if used_global_T else sT.to_numpy()), qT1, qT2)
    RH_n = normalize((glb_10m['g_rh'].to_numpy()    if used_global_RH else sRH.to_numpy()), qR1, qR2)
    ST_n = normalize(sST.to_numpy(), qS1, qS2)

    # Globals normalization
    g_n = {}
    for col in GLOBAL_COLS:
        q1,q2 = safe_quantiles(glb_10m[col].to_numpy())
        g_n[col] = normalize(glb_10m[col].to_numpy(), q1, q2)
    # Day-of-year: fixed scale [1,365]
    g_doy = idx_10m.tz_convert(tz_local).dayofyear.astype(float)
    g_doy_n = (g_doy - 1.0) / (365.0 - 1.0)

    # Compose X
    X = np.column_stack([
        T_n, RH_n, ST_n,
        g_n['g_tmean'], g_n['g_tmin'], g_n['g_tmax'],
        g_n['g_rh'], g_n['g_vpd'], g_n['g_pr'], g_n['g_rad'], g_doy_n,
    ]).astype(np.float32)
    assert X.shape == (SEQ_LEN_10MIN, 11), X.shape

    # LM hourly targets
    df_lm = read_lm_hourly(d_id, lm_dir, tz_local)
    idx_h = pd.date_range(pd.Timestamp(f"{year}-01-01 00:00:00", tz='UTC'), pd.Timestamp(f"{year}-12-31 23:00:00", tz='UTC'), freq=FREQ_1H)
    dfh = df_lm.reindex(idx_h)

    # Normalize targets using the same local scalers
    temp_hr = normalize(dfh['local_T'].to_numpy(), qT1, qT2)
    rh_hr   = normalize(dfh['local_RH'].to_numpy(), qR1, qR2)
    stem_hr = normalize(dfh['stem'].to_numpy(),    qS1, qS2)
    y = np.stack([temp_hr, rh_hr, stem_hr], axis=-1).astype(np.float32)
    assert y.shape == (HOUR_STEPS, 3), y.shape

    # Final NaN protection: replace any NaN with 0.0 and print counts
    def nanfix(A: np.ndarray, name: str):
        n = int(np.isnan(A).sum())
        if n:
            print(f"Warning: {name} contains {n} NaNs → replacing with 0.0")
            A[np.isnan(A)] = 0.0
    nanfix(X, 'X')
    nanfix(y, 'y')

    # Save
    np.save(os.path.join(out_root, 'X.npy'), X)
    np.save(os.path.join(out_root, 'y.npy'), y)
    print('Saved:', os.path.join(out_root, 'X.npy'), X.shape)
    print('Saved:', os.path.join(out_root, 'y.npy'), y.shape)
    print(f"Used globals substitution: T={used_global_T} RH={used_global_RH}")
    return X, y

# -----------------------------
# CLI
# -----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Simple TreeNet preprocessor for one site/year and instrument triple.')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--site_id', type=int, required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--tz', type=str, default='Europe/Zurich')
    ap.add_argument('--thermo_dir', required=True)
    ap.add_argument('--hygro_dir',  required=True)
    ap.add_argument('--dendro_l2_dir', required=True)
    ap.add_argument('--dendro_lm_dir', required=True)
    ap.add_argument('--meteo_dir', required=True)
    ap.add_argument('--t_id', type=int, required=True)
    ap.add_argument('--h_id', type=int, required=True)
    ap.add_argument('--d_id', type=int, required=True)
    ap.add_argument('--stem_mode', type=str, default='absolute', choices=['absolute','delta'])
    ap.add_argument('--allow_missing_locals', type=str, default='true')
    ap.add_argument('--min_local_coverage', type=float, default=0.5)
    args = ap.parse_args()

    build_one_segment(
        out_root=args.out_root,
        site_id=args.site_id,
        year=args.year,
        tz_local=args.tz,
        thermo_dir=args.thermo_dir,
        hygro_dir=args.hygro_dir,
        d2_dir=args.dendro_l2_dir,
        lm_dir=args.dendro_lm_dir,
        meteo_dir=args.meteo_dir,
        t_id=args.t_id,
        h_id=args.h_id,
        d_id=args.d_id,
        stem_mode=args.stem_mode,
        allow_missing_locals=(args.allow_missing_locals.lower()=='true'),
        min_local_coverage=args.min_local_coverage,
    )
