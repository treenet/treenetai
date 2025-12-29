# build_normalized_dataset_treenet.py
# -*- coding: utf-8 -*-
"""
TreeNet → TF Pipeline Pre-processing (Full Module)
=================================================

Build normalized multichannel dataset arrays (X_train, y_train, X_test, y_test)
from TreeNet raw data with many-to-one mapping between inputs (multiple instruments)
and outputs (site-level curated hourly averages).

Key features
------------
- Robust per-site (and optional per-year) **quantile normalization** (q05–q95) for locals/globals
- **Timezone unification** (default: Europe/Zurich)
- **Stem mode**: 'absolute' or 'delta' (10‑min change)
- **Input modes**:
  * 'best'         : choose best-coverage instrument per type (T, RH, STEM) → 1 combo
  * 'combinations' : generate **all** instrument triples per site‑year (Cartesian product) **(default)**
  * 'pooled'       : aggregate (median) across **all** instruments per type (site-level input)
- **Target modes**:
  * 'lm_site_median' (default): site-level hourly median across **all LM dendrometers** (stem, temp, rh)
  * 'lm_per_series'           : target from the **selected LM dendrometer series** only
- **Optional next steps implemented**:
  * Cap number of combinations per site‑year (`--max_combos_per_site_year`)
  * Coverage thresholds for locals (`--min_local_coverage`) and LM (`--min_lm_series`)
  * Comprehensive **diagnostics reports** (CSV) per site‑year: coverage, counts, segments, scalers summaries
  * **Site-level summary report** combining instrument counts and diagnostics

Outputs
-------
- X_train.npy, y_train.npy, site_ids_train.npy (+ test variants)
- normalizers/normalizers_site_<id>.json
- train_identifiers.csv / test_identifiers.csv (instrument mapping per segment window)
- diagnostics/diagnostics_preprocessing.csv (site‑year diagnostic table)
- diagnostics/normalizers_summary.csv (q05–q95 per channel per site‑year)
- diagnostics/site_summary.csv (per-site consolidated summary)

Author: M365 Copilot (for Mirko Lukovic)
Date: 2025-12-29
"""

from __future__ import annotations
import os
import re
import json
import argparse
import typing as t
import numpy as np
import pandas as pd

# =====================
# Config constants
# =====================
SEQ_LEN_10MIN = 4320  # 30 days * 24 h * 6 steps/h
HOUR_STEPS    = 720   # 30 days * 24 h
STRIDE_PER_HR = 6
N_CHANNELS    = 11
N_TARGETS     = 3

LOCAL_COLS = ['local_T', 'local_RH', 'local_stem']
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']

# =====================
# Discover & helper functions
# =====================

def discover_meteo_files(meteo_dir: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for fn in os.listdir(meteo_dir):
        if not fn.endswith('.csv'):
            continue
        m = re.findall(r'\d+', fn)
        if not m:
            continue
        site_id = int(m[0])
        mapping[site_id] = os.path.join(meteo_dir, fn)
    return mapping


def load_global_daily(site_id: int, year: int, meteo_dir: str, tz: str) -> pd.DataFrame:
    site_files = discover_meteo_files(meteo_dir)
    if site_id not in site_files:
        raise FileNotFoundError(f"No meteo CSV found for site {site_id} in {meteo_dir}")
    df = pd.read_csv(site_files[site_id])
    if 'ts' not in df.columns:
        raise ValueError("Global meteo CSV must have 'ts' column")
    df['ts'] = pd.to_datetime(df['ts'], utc=False)
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize(tz)
    else:
        df['ts'] = df['ts'].dt.tz_convert(tz)
    df = df[df['ts'].dt.year == year].copy()
    df['ts_midnight'] = df['ts'].dt.normalize()
    df = df.set_index('ts_midnight')
    rename_map = {
        'tas': 'g_tmean',
        'tasmax': 'g_tmax',
        'tasmin': 'g_tmin',
        'rh': 'g_rh',
        'vpd': 'g_vpd',
        'gh': 'g_rad',
        'pr': 'g_pr',
    }
    for src, dst in rename_map.items():
        if src not in df.columns:
            raise ValueError(f"Global meteo CSV missing column '{src}'")
        df[dst] = df[src]
    df['g_doy'] = df.index.dayofyear
    return df[GLOBAL_COLS]


def read_feather_series(series_id: int, dir_path: str, tz: str, value_col: str = 'value', ts_col: str = 'ts') -> pd.Series:
    pattern = re.compile(rf'.*series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(dir_path) if pattern.match(fn)]
    if not matches:
        raise FileNotFoundError(f"Feather for series_id {series_id} not found in {dir_path}")
    fp = os.path.join(dir_path, matches[0])
    df = pd.read_feather(fp)
    if ts_col not in df.columns:
        for alt in ['timestamp', 'time', 'date_time']:
            if alt in df.columns:
                ts_col = alt
                break
    if ts_col not in df.columns:
        raise ValueError(f"Feather file {fp} missing timestamp column")
    if value_col not in df.columns:
        for alt in ['val', 'value_raw', 'measurement']:
            if alt in df.columns:
                value_col = alt
                break
    if value_col not in df.columns:
        raise ValueError(f"Feather file {fp} missing value column")
    ts = pd.to_datetime(df[ts_col], utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    s = pd.Series(df[value_col].to_numpy(), index=ts)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep='first')]
    return s


def make_year_10m_index(year: int, tz: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz)
    end   = pd.Timestamp(f"{year}-12-31 23:59:59", tz=tz)
    idx = pd.date_range(start=start, end=end, freq='10min')
    return idx


def broadcast_daily_to_10m(daily_df: pd.DataFrame, idx_10m: pd.DatetimeIndex) -> pd.DataFrame:
    daily_df = daily_df.copy()
    daily_df.index = daily_df.index.normalize()
    dates = idx_10m.normalize()
    daily_broadcast = daily_df.reindex(dates).ffill()
    daily_broadcast.index = idx_10m
    return daily_broadcast[GLOBAL_COLS]

# =====================
# Quantile scalers
# =====================

def compute_quantile_scaler(v: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> t.Tuple[float, float]:
    q1 = float(np.nanpercentile(v, q_low))
    q2 = float(np.nanpercentile(v, q_high))
    if not np.isfinite(q1):
        q1 = float(np.nanmin(v))
    if not np.isfinite(q2):
        q2 = float(np.nanmax(v))
    if q2 <= q1:
        q2 = q1 + 1e-6
    return q1, q2


def fit_site_scalers(site_id: int, years: t.Sequence[int], meteo_dir: str,
                     tz: str,
                     per_year: bool = True, q_low: float = 5.0, q_high: float = 95.0) -> dict:
    scalers: dict = {}
    for yr in years:
        idx_10m = make_year_10m_index(yr, tz)
        glb = load_global_daily(site_id, yr, meteo_dir, tz)
        glb_10m = broadcast_daily_to_10m(glb, idx_10m)
        scalers.setdefault(yr if per_year else 'ALL', {})
        for col in GLOBAL_COLS:
            q1, q2 = compute_quantile_scaler(glb_10m[col].to_numpy(), q_low, q_high)
            scalers[yr if per_year else 'ALL'][col] = {'q_low': q1, 'q_high': q2}
    return scalers


def normalize_array(arr: np.ndarray, q1: float, q2: float, clip_low: float = -0.1, clip_high: float = 1.1) -> np.ndarray:
    out = (arr - q1) / (q2 - q1)
    out = np.clip(out, clip_low, clip_high)
    return out

# =====================
# Sensor discovery per site
# =====================

def discover_series_ids(dir_path: str, pattern_prefix: str) -> t.Set[int]:
    ids: set[int] = set()
    regex = re.compile(rf'{re.escape(pattern_prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m:
            ids.add(int(m.group(1)))
    return ids


def series_by_site(metadata_df: pd.DataFrame, series_ids: t.Set[int]) -> dict[int, t.List[int]]:
    df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
    groups = df.groupby('site_id')
    out: dict[int, t.List[int]] = {}
    for site, g in groups:
        out[int(site)] = g['series_id'].tolist()
    return out

# =====================
# Target aggregation (site-level LM)
# =====================

def read_lm_frame(series_id: int, lm_dir: str, tz: str) -> pd.DataFrame:
    pattern = re.compile(rf'dendrometer_lm_series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(lm_dir) if pattern.match(fn)]
    if not matches:
        raise FileNotFoundError(f"LM frame for series_id {series_id} not found in {lm_dir}")
    fp = os.path.join(lm_dir, matches[0])
    df = pd.read_feather(fp)
    if 'ts' not in df.columns:
        raise ValueError(f"LM file {fp} missing 'ts'")
    ts = pd.to_datetime(df['ts'], utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    df = df.set_index(ts)
    # Ensure expected columns exist
    for col in ['value','temp','rh']:
        if col not in df.columns:
            df[col] = np.nan
    return df[['value','temp','rh']]


def to_hourly(df: pd.DataFrame, how: str = 'median') -> pd.DataFrame:
    rule = {'median': 'median', 'mean': 'mean'}[how]
    return df.resample('1H').agg(rule)


def build_site_level_targets(site_id: int, year: int, tz: str,
                             lm_dir: str,
                             dendro_lm_ids_by_site: dict[int, t.List[int]],
                             stem_mode: str = 'absolute',
                             agg: str = 'median') -> pd.DataFrame:
    idx_hour = pd.date_range(pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz), pd.Timestamp(f"{year}-12-31 23:59:59", tz=tz), freq='1H')
    frames = []
    lm_ids = dendro_lm_ids_by_site.get(site_id, [])
    for sid in lm_ids:
        try:
            df = read_lm_frame(sid, lm_dir, tz)
        except Exception:
            continue
        df = df[(df.index.year == year)].copy()
        if df.empty:
            continue
        if stem_mode == 'delta':
            df['value'] = df['value'].diff()
        dfh = to_hourly(df, how=agg)
        dfh.columns = [f'stem_{sid}', f'temp_{sid}', f'rh_{sid}']
        frames.append(dfh)
    if not frames:
        return pd.DataFrame(index=idx_hour, columns=['stem','temp','rh'])
    big = pd.concat(frames, axis=1)
    stem_cols = [c for c in big.columns if c.startswith('stem_')]
    temp_cols = [c for c in big.columns if c.startswith('temp_')]
    rh_cols   = [c for c in big.columns if c.startswith('rh_')]
    agg_func = np.median if agg == 'median' else np.mean
    stem_site = big[stem_cols].apply(agg_func, axis=1)
    temp_site = big[temp_cols].apply(agg_func, axis=1) if temp_cols else pd.Series(np.nan, index=big.index)
    rh_site   = big[rh_cols].apply(agg_func, axis=1) if rh_cols else pd.Series(np.nan, index=big.index)
    out = pd.DataFrame({'stem': stem_site, 'temp': temp_site, 'rh': rh_site})
    out = out.reindex(idx_hour)
    return out

# =====================
# Coverage utilities & diagnostics
# =====================

def coverage_fraction(series: pd.Series) -> float:
    v = series.to_numpy()
    n = v.size
    if n == 0:
        return 0.0
    return float(np.sum(~np.isnan(v)) / n)


def write_diagnostics_row(rows: t.List[dict], **kw):
    rows.append(kw)

# =====================
# Site instrument counts & summary reports
# =====================

def compute_site_instrument_counts(metadata_pickle: str, thermo_dir: str, hygro_dir: str, dendro_l2_dir: str, dendro_lm_dir: str) -> pd.DataFrame:
    metadata_df = pd.read_pickle(metadata_pickle)
    thermo_ids_all = discover_series_ids(thermo_dir, 'thermometer_l1')
    hygro_ids_all  = discover_series_ids(hygro_dir,  'hygrometer_l1')
    dendro_l2_ids_all = discover_series_ids(dendro_l2_dir, 'dendrometer_l2')
    dendro_lm_ids_all = discover_series_ids(dendro_lm_dir, 'dendrometer_lm')

    def counts_by_site(series_ids: t.Set[int]) -> dict[int, int]:
        df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
        return df.groupby('site_id')['series_id'].count().to_dict()

    thermo_counts = counts_by_site(thermo_ids_all)
    hygro_counts  = counts_by_site(hygro_ids_all)
    dendro_l2_counts = counts_by_site(dendro_l2_ids_all)
    dendro_lm_counts = counts_by_site(dendro_lm_ids_all)

    sites = sorted(set(list(thermo_counts.keys()) + list(hygro_counts.keys()) + list(dendro_l2_counts.keys()) + list(dendro_lm_counts.keys())))
    rows = []
    for sid in sites:
        rows.append({
            'site_id': int(sid),
            'n_thermometers': int(thermo_counts.get(sid, 0)),
            'n_hygrometers': int(hygro_counts.get(sid, 0)),
            'n_dendrometers_L2': int(dendro_l2_counts.get(sid, 0)),
            'n_dendrometers_LM': int(dendro_lm_counts.get(sid, 0)),
        })
    return pd.DataFrame(rows)


def write_site_summaries(out_root: str, diagnostics_rows: t.List[dict], instrument_df: pd.DataFrame) -> None:
    diag_df = pd.DataFrame(diagnostics_rows) if diagnostics_rows else pd.DataFrame(columns=['site_id','year','split','combos','windows','segments','lm_series_count','warning'])
    # Aggregate per site
    def agg_split(df, split):
        d = df[df['split'] == split] if 'split' in df.columns else df
        g = d.groupby('site_id').agg(
            years_processed=('year', lambda x: len(set(x))),
            segments_total=('segments', 'sum'),
            combos_total=('combos', 'sum'),
            windows_total=('windows', 'sum'),
            warnings_low_local_combo=('warning', lambda x: int(np.sum(x == 'low_local_coverage_combo'))),
            warnings_low_local_pooled=('warning', lambda x: int(np.sum(x == 'low_local_coverage_pooled'))),
            warnings_insufficient_lm=('warning', lambda x: int(np.sum(x == 'insufficient_lm_series'))),
        ).reset_index()
        g.columns = ['site_id'] + [f'{split}_{c}' for c in g.columns[1:]]
        return g
    site_train = agg_split(diag_df, 'train')
    site_test  = agg_split(diag_df, 'test')
    # Merge instrument counts and splits
    site_summary = instrument_df.merge(site_train, on='site_id', how='left').merge(site_test, on='site_id', how='left')
    # Fill NaNs with zeros for counts
    fill_cols = [c for c in site_summary.columns if c != 'site_id']
    site_summary[fill_cols] = site_summary[fill_cols].fillna(0)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)
    site_summary.to_csv(os.path.join(out_root, 'diagnostics', 'site_summary.csv'), index=False)
    print('Wrote site-level summary:', os.path.join(out_root, 'diagnostics', 'site_summary.csv'))

# =====================
# Build segments per site-year
# =====================

def segment_30d_overlap(idx_10m: pd.DatetimeIndex, overlap_days: int = 10) -> t.List[t.Tuple[pd.Timestamp, pd.Timestamp]]:
    win_days = 30
    stride_days = win_days - overlap_days
    starts = []
    t0 = idx_10m[0].normalize()
    t_end = idx_10m[-1]
    cur = t0
    while cur + pd.Timedelta(days=win_days) <= t_end:
        starts.append(cur)
        cur = cur + pd.Timedelta(days=stride_days)
    return [(s, s + pd.Timedelta(days=win_days)) for s in starts]


def build_segments_for_site_year(site_id: int, year: int, tz: str,
                                 meteo_dir: str,
                                 thermo_dir: str,
                                 hygro_dir: str,
                                 dendro_l2_dir: str,
                                 dendro_lm_dir: str,
                                 metadata_path: str,
                                 scalers: dict,
                                 per_year: bool = True,
                                 require_complete_locals: bool = False,
                                 stem_mode: str = 'absolute',
                                 input_mode: str = 'combinations',
                                 target_mode: str = 'lm_site_median',
                                 max_combos_per_site_year: t.Optional[int] = None,
                                 min_local_coverage: float = 0.7,
                                 min_lm_series: int = 1,
                                 diagnostics_rows: t.Optional[t.List[dict]] = None,
                                 split: str = 'train') -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[dict]]:
    metadata_df = pd.read_pickle(metadata_path)
    idx_10m = make_year_10m_index(year, tz)

    # Globals
    df_glb = load_global_daily(site_id, year, meteo_dir, tz)
    df_glb_10m = broadcast_daily_to_10m(df_glb, idx_10m)

    # Discover series ids per type and by site
    thermo_ids_all = discover_series_ids(thermo_dir, 'thermometer_l1')
    hygro_ids_all  = discover_series_ids(hygro_dir,  'hygrometer_l1')
    dendro_l2_ids_all = discover_series_ids(dendro_l2_dir, 'dendrometer_l2')
    dendro_lm_ids_all = discover_series_ids(dendro_lm_dir, 'dendrometer_lm')

    thermo_by_site = series_by_site(metadata_df, thermo_ids_all)
    hygro_by_site  = series_by_site(metadata_df, hygro_ids_all)
    dendro_l2_by_site = series_by_site(metadata_df, dendro_l2_ids_all)
    dendro_lm_by_site = series_by_site(metadata_df, dendro_lm_ids_all)

    if site_id not in thermo_by_site or site_id not in hygro_by_site or site_id not in dendro_l2_by_site:
        return [], [], []

    thermo_ids = thermo_by_site[site_id]
    hygro_ids  = hygro_by_site[site_id]
    dendro_l2_ids = dendro_l2_by_site[site_id]
    lm_ids_site = dendro_lm_by_site.get(site_id, [])

    # Build site-level LM hourly targets (median across LM series) if requested
    site_targets_hourly = None
    if target_mode == 'lm_site_median':
        site_targets_hourly = build_site_level_targets(site_id, year, tz, dendro_lm_dir, dendro_lm_by_site, stem_mode=stem_mode, agg='median')
        # LM series count check
        if len(lm_ids_site) < min_lm_series:
            if diagnostics_rows is not None:
                write_diagnostics_row(diagnostics_rows, site_id=site_id, year=year, split=split, warning='insufficient_lm_series', lm_series_count=len(lm_ids_site))
            return [], [], []

    # Prepare combinations of instruments according to input_mode
    combos: t.List[tuple[int,int,int]] = []
    if input_mode == 'best':
        def best_id(ids, dirp):
            best, best_count = None, -1
            for sid in ids:
                try:
                    s = read_feather_series(sid, dirp, tz).reindex(idx_10m)
                except Exception:
                    continue
                c = int(np.sum(~np.isnan(s.to_numpy())))
                if c > best_count:
                    best, best_count = sid, c
            return best
        t_id = best_id(thermo_ids, thermo_dir)
        h_id = best_id(hygro_ids, hygro_dir)
        d_id = best_id(dendro_l2_ids, dendro_l2_dir)
        if t_id is None or h_id is None or d_id is None:
            return [], [], []
        combos = [(t_id, h_id, d_id)]
    elif input_mode == 'combinations':
        for t_id in thermo_ids:
            for h_id in hygro_ids:
                for d_id in dendro_l2_ids:
                    combos.append((t_id, h_id, d_id))
        # Cap combinations if requested
        if (max_combos_per_site_year is not None) and (len(combos) > max_combos_per_site_year):
            # deterministic sample: take first K after sorting by ids
            combos.sort()
            combos = combos[:max_combos_per_site_year]
    elif input_mode == 'pooled':
        combos = [(None, None, None)]
    else:
        raise ValueError("input_mode must be one of: best, combinations, pooled")

    key = year if per_year else 'ALL'

    X_list: t.List[np.ndarray] = []
    Y_list: t.List[np.ndarray] = []
    META_list: t.List[dict] = []

    # Precompute pooled series if needed
    if input_mode == 'pooled':
        def pooled_series(ids, dirp):
            arrs = []
            for sid in ids:
                try:
                    arrs.append(read_feather_series(sid, dirp, tz).reindex(idx_10m).to_numpy())
                except Exception:
                    continue
            if not arrs:
                return pd.Series(np.nan, index=idx_10m)
            A = np.vstack(arrs)
            med = np.nanmedian(A, axis=0)
            return pd.Series(med, index=idx_10m)
        s_T_pooled = pooled_series(thermo_ids, thermo_dir)
        s_RH_pooled = pooled_series(hygro_ids, hygro_dir)
        s_STEM_L2_pooled = pooled_series(dendro_l2_ids, dendro_l2_dir)
        s_STEM_L2_pooled = s_STEM_L2_pooled.diff() if stem_mode=='delta' else s_STEM_L2_pooled
        # Ensure local scalers present
        for name, s in [('local_T', s_T_pooled), ('local_RH', s_RH_pooled), ('local_stem', s_STEM_L2_pooled)]:
            if name not in scalers[key]:
                q1, q2 = compute_quantile_scaler(s.to_numpy())
                scalers[key][name] = {'q_low': q1, 'q_high': q2}
        # Coverage check on pooled
        cov_T = coverage_fraction(s_T_pooled)
        cov_RH= coverage_fraction(s_RH_pooled)
        cov_ST= coverage_fraction(s_STEM_L2_pooled)
        if (cov_T < min_local_coverage) or (cov_RH < min_local_coverage) or (cov_ST < min_local_coverage):
            if diagnostics_rows is not None:
                write_diagnostics_row(diagnostics_rows, site_id=site_id, year=year, split=split, warning='low_local_coverage_pooled', cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
            return [], [], []

    windows = segment_30d_overlap(idx_10m, overlap_days=10)
    windows_len = len(windows)
    segments_count = 0

    for (t_id, h_id, d_id) in combos:
        if input_mode != 'pooled':
            try:
                s_T  = read_feather_series(t_id, thermo_dir, tz).reindex(idx_10m)
                s_RH = read_feather_series(h_id, hygro_dir,  tz).reindex(idx_10m)
                s_STEM_L2 = read_feather_series(d_id, dendro_l2_dir, tz).reindex(idx_10m)
            except Exception:
                continue
            s_STEM_L2 = s_STEM_L2.diff() if stem_mode=='delta' else s_STEM_L2
            # Local scalers
            for name, s in [('local_T', s_T), ('local_RH', s_RH), ('local_stem', s_STEM_L2)]:
                if name not in scalers[key]:
                    q1, q2 = compute_quantile_scaler(s.to_numpy())
                    scalers[key][name] = {'q_low': q1, 'q_high': q2}
            # Coverage check per combo
            cov_T = coverage_fraction(s_T)
            cov_RH= coverage_fraction(s_RH)
            cov_ST= coverage_fraction(s_STEM_L2)
            if (cov_T < min_local_coverage) or (cov_RH < min_local_coverage) or (cov_ST < min_local_coverage):
                # Skip this combo but record diagnostic
                if diagnostics_rows is not None:
                    write_diagnostics_row(diagnostics_rows, site_id=site_id, year=year, split=split, warning='low_local_coverage_combo', thermo_id=t_id, hygro_id=h_id, dendro_id=d_id, cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                continue
        else:
            s_T, s_RH, s_STEM_L2 = s_T_pooled, s_RH_pooled, s_STEM_L2_pooled

        # Normalize locals
        def nrm(name, s):
            q1 = scalers[key][name]['q_low']; q2 = scalers[key][name]['q_high']
            return normalize_array(s.to_numpy(), q1, q2)
        T_n    = nrm('local_T', s_T)
        RH_n   = nrm('local_RH', s_RH)
        STEM_n = nrm('local_stem', s_STEM_L2)

        # Normalize globals
        glb_n = {}
        for col in GLOBAL_COLS:
            q1 = scalers[key][col]['q_low']; q2 = scalers[key][col]['q_high']
            glb_n[col] = normalize_array(df_glb_10m[col].to_numpy(), q1, q2)

        # Prepare LM target per combination if needed
        if target_mode == 'lm_per_series':
            try:
                df_lm = read_lm_frame(d_id, dendro_lm_dir, tz)
                if stem_mode == 'delta':
                    df_lm['value'] = df_lm['value'].diff()
                df_lm_h = to_hourly(df_lm, how='median')
            except Exception:
                df_lm_h = pd.DataFrame(index=pd.date_range(pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz), pd.Timestamp(f"{year}-12-31 23:59:59", tz=tz), freq='1H'), columns=['value','temp','rh'])

        for (ws, we) in windows:
            mask = (idx_10m >= ws) & (idx_10m < we)
            if mask.sum() != SEQ_LEN_10MIN:
                continue
            X_seg = np.column_stack([
                T_n[mask], RH_n[mask], STEM_n[mask],
                glb_n['g_tmean'][mask], glb_n['g_tmin'][mask], glb_n['g_tmax'][mask],
                glb_n['g_rh'][mask], glb_n['g_vpd'][mask], glb_n['g_pr'][mask], glb_n['g_rad'][mask], glb_n['g_doy'][mask],
            ]).astype(np.float32)

            if require_complete_locals:
                if np.isnan(s_T[mask]).any() or np.isnan(s_RH[mask]).any() or np.isnan(s_STEM_L2[mask]).any():
                    continue

            # Targets y
            if target_mode == 'lm_site_median' and site_targets_hourly is not None:
                start_h = ws
                end_h   = we - pd.Timedelta(minutes=10)
                idx_h = pd.date_range(start_h.floor('H'), end_h.floor('H'), freq='1H')
                dfh = site_targets_hourly.reindex(idx_h)
                q1s, q2s = scalers[key]['local_stem']['q_low'], scalers[key]['local_stem']['q_high']
                q1t, q2t = scalers[key]['local_T']['q_low'], scalers[key]['local_T']['q_high']
                q1r, q2r = scalers[key]['local_RH']['q_low'], scalers[key]['local_RH']['q_high']
                stem_hr = normalize_array(dfh['stem'].to_numpy(), q1s, q2s)
                temp_hr = normalize_array(dfh['temp'].to_numpy(), q1t, q2t)
                rh_hr   = normalize_array(dfh['rh'].to_numpy(),   q1r, q2r)
                y_seg = np.stack([temp_hr, rh_hr, stem_hr], axis=-1).astype(np.float32)
            else:
                T_hr   = np.median(T_n[mask].reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                RH_hr  = np.median(RH_n[mask].reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                if target_mode == 'lm_per_series':
                    start_h = ws
                    end_h   = we - pd.Timedelta(minutes=10)
                    idx_h = pd.date_range(start_h.floor('H'), end_h.floor('H'), freq='1H')
                    dfh = df_lm_h.reindex(idx_h)
                    q1s, q2s = scalers[key]['local_stem']['q_low'], scalers[key]['local_stem']['q_high']
                    stem_hr = normalize_array(dfh['value'].to_numpy(), q1s, q2s)
                    stem_temp_hr = normalize_array(dfh['temp'].to_numpy(), scalers[key]['local_T']['q_low'], scalers[key]['local_T']['q_high'])
                    stem_rh_hr   = normalize_array(dfh['rh'].to_numpy(),  scalers[key]['local_RH']['q_low'], scalers[key]['local_RH']['q_high'])
                    # Prefer LM temp/rh when available
                    T_hr = np.where(np.isnan(stem_temp_hr), T_hr, stem_temp_hr)
                    RH_hr= np.where(np.isnan(stem_rh_hr),  RH_hr, stem_rh_hr)
                else:
                    STEM_hr = np.median(STEM_n[mask].reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                    stem_hr = STEM_hr
                y_seg = np.stack([T_hr, RH_hr, stem_hr], axis=-1).astype(np.float32)

            X_list.append(X_seg)
            Y_list.append(y_seg)
            META_list.append({
                'site_id': site_id,
                'year': year,
                'input_mode': input_mode,
                'target_mode': target_mode,
                'thermometer_id': int(t_id) if t_id is not None else -1,
                'hygrometer_id': int(h_id) if h_id is not None else -1,
                'dendrometer_id': int(d_id) if d_id is not None else -1,
                'window_start': str(ws),
                'window_end': str(we),
            })
            segments_count += 1

    # diagnostics
    if diagnostics_rows is not None:
        write_diagnostics_row(diagnostics_rows,
            site_id=site_id, year=year, split=split, input_mode=input_mode, target_mode=target_mode,
            combos=len(combos), windows=windows_len, segments=segments_count,
            lm_series_count=len(lm_ids_site))

    return X_list, Y_list, META_list

# =====================
# Build datasets for train/test
# =====================

def build_datasets(out_root: str,
                   meteo_dir: str,
                   thermo_dir: str,
                   hygro_dir: str,
                   dendro_l2_dir: str,
                   dendro_lm_dir: str,
                   metadata_pickle: str,
                   site_ids_train: t.Sequence[int],
                   site_ids_test: t.Optional[t.Sequence[int]],
                   years: t.Sequence[int],
                   per_year: bool = True,
                   tz: str = 'Europe/Zurich',
                   require_complete_locals: bool = False,
                   stem_mode: str = 'absolute',
                   input_mode: str = 'combinations',
                   target_mode: str = 'lm_site_median',
                   max_combos_per_site_year: t.Optional[int] = None,
                   min_local_coverage: float = 0.7,
                   min_lm_series: int = 1) -> None:
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)

    # Fit scalers per site (globals); locals added during building
    site_scalers: dict[int, dict] = {}
    normalizers_summary_rows: t.List[dict] = []

    for sid in site_ids_train:
        scalers = fit_site_scalers(sid, years, meteo_dir, tz, per_year=per_year)
        site_scalers[sid] = scalers
        norm_dir = os.path.join(out_root, 'normalizers'); os.makedirs(norm_dir, exist_ok=True)
        with open(os.path.join(norm_dir, f"normalizers_site_{sid}.json"), 'w', encoding='utf-8') as f:
            json.dump(scalers, f, indent=2)
        # record summary
        for key, chmap in scalers.items():
            for ch, qq in chmap.items():
                normalizers_summary_rows.append({'site_id': sid, 'scope': key, 'channel': ch, 'q_low': qq['q_low'], 'q_high': qq['q_high']})

    # TRAIN
    X_tr_list: t.List[np.ndarray] = []
    y_tr_list: t.List[np.ndarray] = []
    sid_tr_list: t.List[int] = []
    meta_rows: t.List[dict] = []
    diagnostics_rows: t.List[dict] = []

    for sid in site_ids_train:
        scalers = site_scalers[sid]
        for yr in years:
            X_list, y_list, meta_list = build_segments_for_site_year(
                site_id=sid, year=yr, tz=tz,
                meteo_dir=meteo_dir,
                thermo_dir=thermo_dir,
                hygro_dir=hygro_dir,
                dendro_l2_dir=dendro_l2_dir,
                dendro_lm_dir=dendro_lm_dir,
                metadata_path=metadata_pickle,
                scalers=scalers,
                per_year=per_year,
                require_complete_locals=require_complete_locals,
                stem_mode=stem_mode,
                input_mode=input_mode,
                target_mode=target_mode,
                max_combos_per_site_year=max_combos_per_site_year,
                min_local_coverage=min_local_coverage,
                min_lm_series=min_lm_series,
                diagnostics_rows=diagnostics_rows,
                split='train',
            )
            X_tr_list.extend(X_list); y_tr_list.extend(y_list); sid_tr_list.extend([sid]*len(X_list)); meta_rows.extend(meta_list)

    X_train = np.stack(X_tr_list, axis=0) if X_tr_list else np.empty((0, SEQ_LEN_10MIN, N_CHANNELS), dtype=np.float32)
    y_train = np.stack(y_tr_list, axis=0) if y_tr_list else np.empty((0, HOUR_STEPS, N_TARGETS), dtype=np.float32)
    SID_train = np.array(sid_tr_list, dtype=np.int32)

    np.save(os.path.join(out_root, 'X_train.npy'), X_train)
    np.save(os.path.join(out_root, 'y_train.npy'), y_train)
    np.save(os.path.join(out_root, 'site_ids_train.npy'), SID_train)
    print(f"Saved TRAIN arrays: X_train {X_train.shape}, y_train {y_train.shape}, site_ids_train {SID_train.shape}")

    pd.DataFrame(meta_rows).to_csv(os.path.join(out_root, 'train_identifiers.csv'), index=False)

    # TEST
    if site_ids_test is not None:
        for sid in site_ids_test:
            scalers = fit_site_scalers(sid, years, meteo_dir, tz, per_year=per_year)
            site_scalers[sid] = scalers
            norm_dir = os.path.join(out_root, 'normalizers'); os.makedirs(norm_dir, exist_ok=True)
            with open(os.path.join(norm_dir, f"normalizers_site_{sid}.json"), 'w', encoding='utf-8') as f:
                json.dump(scalers, f, indent=2)
            for key, chmap in scalers.items():
                for ch, qq in chmap.items():
                    normalizers_summary_rows.append({'site_id': sid, 'scope': key, 'channel': ch, 'q_low': qq['q_low'], 'q_high': qq['q_high']})

        X_te_list: t.List[np.ndarray] = []
        y_te_list: t.List[np.ndarray] = []
        sid_te_list: t.List[int] = []
        meta_rows_te: t.List[dict] = []

        for sid in site_ids_test:
            scalers = site_scalers[sid]
            for yr in years:
                X_list, y_list, meta_list = build_segments_for_site_year(
                    site_id=sid, year=yr, tz=tz,
                    meteo_dir=meteo_dir,
                    thermo_dir=thermo_dir,
                    hygro_dir=hygro_dir,
                    dendro_l2_dir=dendro_l2_dir,
                    dendro_lm_dir=dendro_lm_dir,
                    metadata_path=metadata_pickle,
                    scalers=scalers,
                    per_year=per_year,
                    require_complete_locals=require_complete_locals,
                    stem_mode=stem_mode,
                    input_mode=input_mode,
                    target_mode=target_mode,
                    max_combos_per_site_year=max_combos_per_site_year,
                    min_local_coverage=min_local_coverage,
                    min_lm_series=min_lm_series,
                    diagnostics_rows=diagnostics_rows,
                    split='test',
                )
                X_te_list.extend(X_list); y_te_list.extend(y_list); sid_te_list.extend([sid]*len(X_list)); meta_rows_te.extend(meta_list)

        X_test = np.stack(X_te_list, axis=0) if X_te_list else np.empty((0, SEQ_LEN_10MIN, N_CHANNELS), dtype=np.float32)
        y_test = np.stack(y_te_list, axis=0) if y_te_list else np.empty((0, HOUR_STEPS, N_TARGETS), dtype=np.float32)
        SID_test = np.array(sid_te_list, dtype=np.int32)

        np.save(os.path.join(out_root, 'X_test.npy'), X_test)
        np.save(os.path.join(out_root, 'y_test.npy'), y_test)
        np.save(os.path.join(out_root, 'site_ids_test.npy'), SID_test)
        pd.DataFrame(meta_rows_te).to_csv(os.path.join(out_root, 'test_identifiers.csv'), index=False)
        print(f"Saved TEST arrays: X_test {X_test.shape}, y_test {y_test.shape}, site_ids_test {SID_test.shape}")

    # Diagnostics & normalizer summaries
    pd.DataFrame(diagnostics_rows).to_csv(os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'), index=False)
    pd.DataFrame(normalizers_summary_rows).to_csv(os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'), index=False)
    print("Wrote diagnostics:")
    print(" -", os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'))
    print(" -", os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'))

    # Site-level summary
    instrument_df = compute_site_instrument_counts(metadata_pickle, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir)
    write_site_summaries(out_root, diagnostics_rows, instrument_df)

# =====================
# CLI
# =====================

def parse_args():
    p = argparse.ArgumentParser(description='TreeNet pre-processing: normalized arrays with many-to-one targets, combos, thresholds, diagnostics, and site summary.')
    p.add_argument('--out_root', required=True, help='Output folder for arrays and normalizers.')
    p.add_argument('--metadata_pickle', required=True, help='Path to metadata_all.pkl (pickle with series_id and site_id).')
    p.add_argument('--meteo_dir', required=True, help='Folder containing meteo CSV files (one per site).')
    p.add_argument('--thermo_dir', required=True, help='Folder containing thermometer L1 feather files.')
    p.add_argument('--hygro_dir', required=True, help='Folder containing hygrometer L1 feather files.')
    p.add_argument('--dendro_l2_dir', required=True, help='Folder containing dendrometer L2 feather files.')
    p.add_argument('--dendro_lm_dir', required=True, help='Folder containing dendrometer LM feather files.')
    p.add_argument('--train_site_ids_csv', required=True, help='CSV with site_ids for train split.')
    p.add_argument('--test_site_ids_csv', required=False, help='CSV with site_ids for test split.')
    p.add_argument('--years', nargs='+', type=int, required=True, help='Years to include (e.g., 2014 2015 2016).')
    p.add_argument('--per_year', type=str, default='true', help='Use per-year scalers (true/false).')
    p.add_argument('--tz', type=str, default='Europe/Zurich', help='Timezone for timestamps.')
    p.add_argument('--require_complete_locals', type=str, default='false', help='Require complete local channels in segments (true/false).')
    p.add_argument('--stem_mode', type=str, default='absolute', choices=['absolute','delta'], help='Use absolute radius or 10-min delta for stem channel and target.')
    p.add_argument('--input_mode', type=str, default='combinations', choices=['best','combinations','pooled'], help='Instrument strategy for inputs (default: combinations).')
    p.add_argument('--target_mode', type=str, default='lm_site_median', choices=['lm_site_median','lm_per_series'], help='Target aggregation strategy.')
    p.add_argument('--max_combos_per_site_year', type=int, default=None, help='Cap number of instrument combinations per site-year (None = no cap).')
    p.add_argument('--min_local_coverage', type=float, default=0.7, help='Minimum fraction of non-NaN local samples per year (default 0.7).')
    p.add_argument('--min_lm_series', type=int, default=1, help='Minimum number of LM series at site for site-level target (default 1).')
    return p.parse_args()


def read_site_ids_csv(path: str) -> t.List[int]:
    df = pd.read_csv(path)
    if 'site_id' not in df.columns:
        raise ValueError('CSV must contain a column named site_id')
    return [int(x) for x in df['site_id'].tolist()]


if __name__ == '__main__':
    args = parse_args()
    per_year = (args.per_year.lower() == 'true')
    require_complete_locals = (args.require_complete_locals.lower() == 'true')

    site_ids_train = read_site_ids_csv(args.train_site_ids_csv)
    site_ids_test = read_site_ids_csv(args.test_site_ids_csv) if args.test_site_ids_csv else None

    build_datasets(
        out_root=args.out_root,
        meteo_dir=args.meteo_dir,
        thermo_dir=args.thermo_dir,
        hygro_dir=args.hygro_dir,
        dendro_l2_dir=args.dendro_l2_dir,
        dendro_lm_dir=args.dendro_lm_dir,
        metadata_pickle=args.metadata_pickle,
        site_ids_train=site_ids_train,
        site_ids_test=site_ids_test,
        years=args.years,
        per_year=per_year,
        tz=args.tz,
        require_complete_locals=require_complete_locals,
        stem_mode=args.stem_mode,
        input_mode=args.input_mode,
        target_mode=args.target_mode,
        max_combos_per_site_year=args.max_combos_per_site_year,
        min_local_coverage=args.min_local_coverage,
        min_lm_series=args.min_lm_series,
    )
