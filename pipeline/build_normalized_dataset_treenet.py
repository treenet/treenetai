# build_normalized_dataset_treenet.py
# -*- coding: utf-8 -*-
"""
TreeNet → TF Pipeline Pre-processing (Year-long Segments, Fully Annotated)
=======================================================================

This module builds normalized multichannel dataset arrays (X_train, y_train, X_test, y_test)
from TreeNet raw data using **year-long segments** (365 days; leap day removed) with configurable
**overlap in days** across a multi-year timeline.

It is designed for masked imputation and drift-detection pipelines where **segment-level normalization**
could destroy long-scale signals. Therefore, normalization is computed **per site** and **per year (or ALL years)**
with robust **quantile scalers**.

Key features
------------
- Year-long windows: fixed input length 52,560 (10-min) and target length 8,760 (hourly)
- Overlap in days across the multi-year index to increase training samples
- Robust per-site/per-year quantile normalization (q05–q95, clipped)
- Input modes: 'best' | 'combinations' | 'pooled'
- Target modes: 'lm_site_median' | 'lm_per_series'
- Stem mode: 'absolute' | 'delta' (10‑min change)
- Thresholds & caps: local coverage, LM minimum series, max combos per site
- Diagnostics CSVs & site summary

Outputs
-------
- Arrays: X_train.npy, y_train.npy (+ optional test variants), site_ids_*.npy
- Normalizers: normalizers/normalizers_site_<id>.json
- Identifiers: train_identifiers.csv / test_identifiers.csv (mapping windows to instruments)
- Diagnostics: diagnostics_preprocessing.csv, normalizers_summary.csv, site_summary.csv

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
# Constants (YEAR-LONG)
# =====================
SEQ_LEN_10MIN = 52560  # 365 days * 24 h * 6 steps/h (we remove Feb 29 to keep fixed length)
HOUR_STEPS    = 8760   # 365 days * 24 h
STRIDE_PER_HR = 6
N_CHANNELS    = 11
N_TARGETS     = 3

LOCAL_COLS = ['local_T', 'local_RH', 'local_stem']
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']

# =====================
# Helpers: meteo, feather, indices
# =====================

def discover_meteo_files(meteo_dir: str) -> dict[int, str]:
    """Return a mapping from **site_id** to the path of its global meteo CSV.

    Assumptions
    -----------
    - One CSV per site, and the *first integer* in the filename is the site ID.
    - Files end with `.csv`.

    Parameters
    ----------
    meteo_dir : str
        Directory containing the meteo CSVs.

    Returns
    -------
    dict[int, str]
        {site_id: filepath}
    """
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


def load_global_daily(site_id: int, meteo_dir: str, tz: str) -> pd.DataFrame:
    """Load and standardize **daily global meteo** for a site across all years.

    The CSV must contain columns: `ts, tas, tasmax, tasmin, rh, vpd, gh, pr`.
    `ts` is parsed to datetime, localized/converted to the chosen timezone.
    A midnight index is set, and we compute `g_doy` (day-of-year). No resampling here.

    Parameters
    ----------
    site_id : int
        Numeric site identifier.
    meteo_dir : str
        Directory with the site's CSV.
    tz : str
        Target timezone (e.g., 'Europe/Zurich').

    Returns
    -------
    pd.DataFrame
        Daily DataFrame indexed by midnight timestamps with columns GLOBAL_COLS.
    """
    site_files = discover_meteo_files(meteo_dir)
    if site_id not in site_files:
        raise FileNotFoundError(f"No meteo CSV found for site {site_id} in {meteo_dir}")
    df = pd.read_csv(site_files[site_id])
    if 'ts' not in df.columns:
        raise ValueError("Global meteo CSV must have 'ts' column")
    df['ts'] = pd.to_datetime(df['ts'], utc=False)
    # Unify timezone: localize if naive, else convert
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize(tz)
    else:
        df['ts'] = df['ts'].dt.tz_convert(tz)
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
    return df[['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']]


def read_feather_series(series_id: int, dir_path: str, tz: str, value_col: str = 'value', ts_col: str = 'ts') -> pd.Series:
    """Read a **single local instrument series** from Feather.

    The filename must contain `series_id_<id>.ftr`. We read columns `ts` and `value` by default,
    unify timezone, sort by time, and drop duplicates.

    Parameters
    ----------
    series_id : int
        Instrument series ID.
    dir_path : str
        Directory containing Feather files of a sensor type.
    tz : str
        Target timezone.
    value_col : str, optional
        The value column name (default 'value').
    ts_col : str, optional
        The timestamp column name (default 'ts').

    Returns
    -------
    pd.Series
        Time-indexed series with tz-aware index, `dtype` float (NaNs allowed).
    """
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


def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove **Feb 29** from a datetime index to ensure **fixed yearly length**.

    This is important when models expect fixed-length inputs. By removing leap days,
    we avoid padding/masking logic and keep tensors rectangular.
    """
    return idx[~((idx.month == 2) & (idx.day == 29))]


def make_multi_year_10m_index(years: t.Sequence[int], tz: str) -> pd.DatetimeIndex:
    """Create a **continuous 10-min multi-year index** from min(years) to max(years), tz-aware.

    Leap days are removed to keep yearly windows at **52,560** steps.
    """
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz=tz)
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz=tz)
    idx = pd.date_range(start=start, end=end, freq='10min')
    idx = strip_leap_days(idx)
    return idx


def make_multi_year_hourly_index(years: t.Sequence[int], tz: str) -> pd.DatetimeIndex:
    """Create a **continuous hourly multi-year index** from min(years) to max(years), tz-aware.

    Leap days are removed to keep yearly windows at **8,760** hours.
    """
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz=tz)
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz=tz)
    idx = pd.date_range(start=start, end=end, freq='1H')
    idx = strip_leap_days(idx)
    return idx


def broadcast_daily_to_10m(daily_df: pd.DataFrame, idx_10m: pd.DatetimeIndex) -> pd.DataFrame:
    """Broadcast **daily** values to **10-min** by mapping each 10-min timestamp to its date.

    We reindex the daily frame to the normalized dates of the 10-min index and **forward-fill**.
    This yields a 10-min DataFrame with the same columns as the input (GLOBAL_COLS).
    """
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
    """Compute robust quantiles `(q_low, q_high)` ignoring NaNs.

    Edge cases handled:
    - If quantiles are not finite (empty/NaN arrays), fall back to nanmin/nanmax.
    - If `q_high <= q_low`, enforce a minimal positive range.
    """
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
    """Fit **global channel** scalers for a site, either **per year** or **ALL years**.

    Locals are added later when building segments (after instrument selection).

    Returns
    -------
    dict
        If per_year=True: `{year: {channel: {'q_low': q, 'q_high': q}}}`;
        else: `{'ALL': {channel: {...}}}`.
    """
    scalers: dict = {}
    glb_all = load_global_daily(site_id, meteo_dir, tz)
    for scope in ([y for y in years] if per_year else ['ALL']):
        if scope == 'ALL':
            df = glb_all[(glb_all.index.year >= min(years)) & (glb_all.index.year <= max(years))]
        else:
            df = glb_all[glb_all.index.year == scope]
        scalers.setdefault(scope, {})
        for col in GLOBAL_COLS:
            q1, q2 = compute_quantile_scaler(df[col].to_numpy())
            scalers[scope][col] = {'q_low': q1, 'q_high': q2}
    return scalers


def normalize_array(arr: np.ndarray, q1: float, q2: float, clip_low: float = -0.1, clip_high: float = 1.1) -> np.ndarray:
    """Normalize `arr` via `(x - q1) / (q2 - q1)` and clip to `[clip_low, clip_high]`.

    Rationale: clipping guards against rare outliers and keeps inputs within model-friendly bounds.
    """
    out = (arr - q1) / (q2 - q1)
    out = np.clip(out, clip_low, clip_high)
    return out

# =====================
# Sensor discovery per site
# =====================

def discover_series_ids(dir_path: str, pattern_prefix: str) -> t.Set[int]:
    """Scan a directory and return **all series IDs** whose filenames match a sensor prefix.

    Filename pattern: `{prefix}_series_id_<id>.ftr`.
    """
    ids: set[int] = set()
    regex = re.compile(rf'{re.escape(pattern_prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m:
            ids.add(int(m.group(1)))
    return ids


def series_by_site(metadata_df: pd.DataFrame, series_ids: t.Set[int]) -> dict[int, t.List[int]]:
    """Group provided `series_ids` **by site** using `metadata_df` (columns: series_id, site_id).

    Returns
    -------
    dict[int, List[int]]
        {site_id: [series_id,...]}
    """
    df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
    groups = df.groupby('site_id')
    out: dict[int, t.List[int]] = {}
    for site, g in groups:
        out[int(site)] = g['series_id'].tolist()
    return out

# =====================
# LM aggregation (multi-year hourly)
# =====================

def read_lm_frame(series_id: int, lm_dir: str, tz: str) -> pd.DataFrame:
    """Read a **LM dendrometer** frame containing `value`, `temp`, `rh` columns.

    - Timestamps are tz-unified and used as the index.
    - Missing `temp` or `rh` columns are created as NaN for safety.
    """
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
    for col in ['value','temp','rh']:
        if col not in df.columns:
            df[col] = np.nan
    return df[['value','temp','rh']]


def to_hourly(df: pd.DataFrame, how: str = 'median') -> pd.DataFrame:
    """Resample to hourly by **median** (default) or **mean**.

    Note: Median is robust to spikes and is often preferred for physiological signals.
    """
    rule = {'median': 'median', 'mean': 'mean'}[how]
    return df.resample('1H').agg(rule)


def build_site_level_targets_multi(site_id: int, years: t.Sequence[int], tz: str,
                                   lm_dir: str,
                                   dendro_lm_ids_by_site: dict[int, t.List[int]],
                                   stem_mode: str = 'absolute',
                                   agg: str = 'median') -> pd.DataFrame:
    """Build **site-level hourly targets** across the multi-year index.

    We aggregate `value` (stem), `temp`, and `rh` across **all LM series** at the site
    using **nanmedian** (or nanmean). If `stem_mode='delta'`, we difference the LM stem.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame indexed by the multi-year hourly index with columns `['stem','temp','rh']`.
    """
    idx_hour = make_multi_year_hourly_index(years, tz)
    frames = []
    lm_ids = dendro_lm_ids_by_site.get(site_id, [])
    for sid in lm_ids:
        try:
            df = read_lm_frame(sid, lm_dir, tz)
        except Exception:
            continue
        df = df[(df.index.year >= min(years)) & (df.index.year <= max(years))].copy()
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
    big = big.reindex(idx_hour)
    stem_cols = [c for c in big.columns if c.startswith('stem_')]
    temp_cols = [c for c in big.columns if c.startswith('temp_')]
    rh_cols   = [c for c in big.columns if c.startswith('rh_')]
    agg_func = np.nanmedian if agg == 'median' else np.nanmean
    stem_site = big[stem_cols].apply(agg_func, axis=1)
    temp_site = big[temp_cols].apply(agg_func, axis=1) if temp_cols else pd.Series(np.nan, index=big.index)
    rh_site   = big[rh_cols].apply(agg_func, axis=1) if rh_cols else pd.Series(np.nan, index=big.index)
    out = pd.DataFrame({'stem': stem_site, 'temp': temp_site, 'rh': rh_site}, index=idx_hour)
    return out

# =====================
# Coverage & diagnostics
# =====================

def coverage_fraction(series: pd.Series) -> float:
    """Compute the **fraction of non-NaN samples** in a series.

    Used to skip windows/combos with insufficient local data.
    """
    v = series.to_numpy()
    n = v.size
    if n == 0:
        return 0.0
    return float(np.sum(~np.isnan(v)) / n)


def write_diagnostics_row(rows: t.List[dict], **kw):
    """Append a **diagnostic row** (dict) to an accumulating list.

    We keep diagnostics lightweight (list of dicts) and serialize at the end.
    """
    rows.append(kw)

# =====================
# Site instrument counts & summary
# =====================

def compute_site_instrument_counts(metadata_pickle: str, thermo_dir: str, hygro_dir: str, dendro_l2_dir: str, dendro_lm_dir: str) -> pd.DataFrame:
    """Compute the **number of instruments per site** for each sensor type.

    Returns a DataFrame with columns:
    `site_id, n_thermometers, n_hygrometers, n_dendrometers_L2, n_dendrometers_LM`.
    """
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
    """Write a **site-level summary** CSV combining instrument counts and diagnostics (train/test).

    The table includes segment/window totals and warning tallies per split.
    """
    diag_df = pd.DataFrame(diagnostics_rows) if diagnostics_rows else pd.DataFrame(columns=['site_id','split','windows','segments','lm_series_count','warning'])
    def agg_split(df, split):
        d = df[df['split'] == split] if 'split' in df.columns else df
        g = d.groupby('site_id').agg(
            segments_total=('segments', 'sum'),
            windows_total=('windows', 'sum'),
            warnings_low_local_combo=('warning', lambda x: int(np.sum(x == 'low_local_coverage_combo'))),
            warnings_low_local_pooled=('warning', lambda x: int(np.sum(x == 'low_local_coverage_pooled'))),
            warnings_insufficient_lm=('warning', lambda x: int(np.sum(x == 'insufficient_lm_series'))),
        ).reset_index()
        g.columns = ['site_id'] + [f'{split}_{c}' for c in g.columns[1:]]
        return g
    site_train = agg_split(diag_df, 'train')
    site_test  = agg_split(diag_df, 'test')
    site_summary = instrument_df.merge(site_train, on='site_id', how='left').merge(site_test, on='site_id', how='left')
    fill_cols = [c for c in site_summary.columns if c != 'site_id']
    site_summary[fill_cols] = site_summary[fill_cols].fillna(0)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)
    site_summary.to_csv(os.path.join(out_root, 'diagnostics', 'site_summary.csv'), index=False)
    print('Wrote site-level summary:', os.path.join(out_root, 'diagnostics', 'site_summary.csv'))

# =====================
# Rolling YEAR windows
# =====================

def rolling_year_windows(idx_10m: pd.DatetimeIndex, overlap_days: int = 10, year_days: int = 365) -> t.List[t.Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate **year-long windows** over a 10-min index with a given day overlap.

    - Window length = `year_days` (default 365)
    - Stride days   = `year_days - overlap_days` (e.g., 365 - 10 = 355)
    - We stop when the end of a window would exceed the last timestamp.

    Returns
    -------
    List[Tuple[pd.Timestamp, pd.Timestamp]]
        List of (start, end) inclusive/exclusive bounds; end is `start + year_days`.
    """
    stride_days = year_days - overlap_days
    starts = []
    cur = idx_10m[0].normalize()
    end_ts = idx_10m[-1]
    while cur + pd.Timedelta(days=year_days) <= end_ts:
        starts.append(cur)
        cur = cur + pd.Timedelta(days=stride_days)
    return [(s, s + pd.Timedelta(days=year_days)) for s in starts]

# =====================
# Build segments (year-long) per site across multi-year index
# =====================

def build_segments_for_site(site_id: int, years: t.Sequence[int], tz: str,
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
                            max_combos_per_site: t.Optional[int] = None,
                            min_local_coverage: float = 0.7,
                            min_lm_series: int = 1,
                            overlap_days: int = 10,
                            diagnostics_rows: t.Optional[t.List[dict]] = None,
                            split: str = 'train') -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[dict]]:
    """Build **year-long window segments** for one site across multiple years.

    Steps (per site):
    1) Create multi-year 10-min and hourly indices (leap days removed).
    2) Load daily globals and broadcast to 10-min.
    3) Discover instruments per site via metadata and directories.
    4) Build **site-level LM hourly targets** across the multi-year index (median across LM series).
    5) Prepare instrument combinations according to `input_mode`.
    6) Slide year-long windows with overlap, perform coverage checks per window/combination.
    7) Normalize globals via scalers for the window's scope (year or ALL) and normalize locals.
    8) Compose X (10-min) and y (hourly) for each valid window and append identifiers.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[dict]]
        X segments (N, 52,560, 11), y segments (N, 8,760, 3), and metadata rows.
    """
    metadata_df = pd.read_pickle(metadata_path)
    idx_10m = make_multi_year_10m_index(years, tz)
    idx_hour = make_multi_year_hourly_index(years, tz)

    # Globals broadcast
    glb_all = load_global_daily(site_id, meteo_dir, tz)
    glb_10m = broadcast_daily_to_10m(glb_all, idx_10m)

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

    # Build site-level LM hourly targets across multi-year timeline
    site_targets_hourly = None
    if target_mode == 'lm_site_median':
        site_targets_hourly = build_site_level_targets_multi(site_id, years, tz, dendro_lm_dir, dendro_lm_by_site, stem_mode=stem_mode, agg='median')
        if len(lm_ids_site) < min_lm_series:
            if diagnostics_rows is not None:
                write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, warning='insufficient_lm_series', lm_series_count=len(lm_ids_site))
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
        if (max_combos_per_site is not None) and (len(combos) > max_combos_per_site):
            combos.sort()
            combos = combos[:max_combos_per_site]
    elif input_mode == 'pooled':
        combos = [(None, None, None)]
    else:
        raise ValueError("input_mode must be one of: best, combinations, pooled")

    # Windows across multi-year index
    windows = rolling_year_windows(idx_10m, overlap_days=overlap_days, year_days=365)
    windows_len = len(windows)

    key_for = (lambda ts: ts.year) if per_year else (lambda ts: 'ALL')

    X_list: t.List[np.ndarray] = []
    Y_list: t.List[np.ndarray] = []
    META_list: t.List[dict] = []
    segments_count = 0

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
        if stem_mode == 'delta':
            s_STEM_L2_pooled = s_STEM_L2_pooled.diff()

    for (ws, we) in windows:
        # Choose scalers by scope
        scope_key = key_for(ws)
        glb_win = glb_10m[(glb_10m.index >= ws) & (glb_10m.index < we)]
        if glb_win.shape[0] != SEQ_LEN_10MIN:
            # Skip incomplete windows (edges)
            continue

        # Normalize globals for the window
        glb_n = {}
        for col in GLOBAL_COLS:
            q1 = scalers[scope_key][col]['q_low']; q2 = scalers[scope_key][col]['q_high']
            glb_n[col] = normalize_array(glb_win[col].to_numpy(), q1, q2)

        # Hourly target slice for this window
        start_h = ws
        end_h   = we - pd.Timedelta(minutes=10)
        idx_h = pd.date_range(start_h.floor('H'), end_h.floor('H'), freq='1H')
        idx_h = strip_leap_days(idx_h)

        # For each combo / pooled
        for (t_id, h_id, d_id) in combos:
            if input_mode != 'pooled':
                try:
                    s_T  = read_feather_series(t_id, thermo_dir, tz).reindex(idx_10m)
                    s_RH = read_feather_series(h_id, hygro_dir,  tz).reindex(idx_10m)
                    s_STEM_L2 = read_feather_series(d_id, dendro_l2_dir, tz).reindex(idx_10m)
                except Exception:
                    continue
                if stem_mode == 'delta':
                    s_STEM_L2 = s_STEM_L2.diff()
                # Slice window
                sT_w  = s_T[(s_T.index >= ws) & (s_T.index < we)]
                sRH_w = s_RH[(s_RH.index >= ws) & (s_RH.index < we)]
                sST_w = s_STEM_L2[(s_STEM_L2.index >= ws) & (s_STEM_L2.index < we)]
                # Coverage check in-window
                cov_T = coverage_fraction(sT_w)
                cov_RH= coverage_fraction(sRH_w)
                cov_ST= coverage_fraction(sST_w)
                if (cov_T < min_local_coverage) or (cov_RH < min_local_coverage) or (cov_ST < min_local_coverage):
                    if diagnostics_rows is not None:
                        write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, warning='low_local_coverage_combo', thermo_id=t_id, hygro_id=h_id, dendro_id=d_id, window_start=str(ws), cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                    continue
                # Ensure local scalers by scope (first time we see locals for this scope)
                for name, s in [('local_T', s_T), ('local_RH', s_RH), ('local_stem', s_STEM_L2)]:
                    if name not in scalers[scope_key]:
                        q1, q2 = compute_quantile_scaler(s.to_numpy())
                        scalers[scope_key][name] = {'q_low': q1, 'q_high': q2}
                # Normalize locals
                T_n    = normalize_array(sT_w.to_numpy(),  scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high'])
                RH_n   = normalize_array(sRH_w.to_numpy(), scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high'])
                STEM_n = normalize_array(sST_w.to_numpy(), scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high'])
            else:
                sT_w  = s_T_pooled[(s_T_pooled.index >= ws) & (s_T_pooled.index < we)]
                sRH_w = s_RH_pooled[(s_RH_pooled.index >= ws) & (s_RH_pooled.index < we)]
                sST_w = s_STEM_L2_pooled[(s_STEM_L2_pooled.index >= ws) & (s_STEM_L2_pooled.index < we)]
                cov_T = coverage_fraction(sT_w)
                cov_RH= coverage_fraction(sRH_w)
                cov_ST= coverage_fraction(sST_w)
                if (cov_T < min_local_coverage) or (cov_RH < min_local_coverage) or (cov_ST < min_local_coverage):
                    if diagnostics_rows is not None:
                        write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, warning='low_local_coverage_pooled', window_start=str(ws), cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                    continue
                for name, s in [('local_T', s_T_pooled), ('local_RH', s_RH_pooled), ('local_stem', s_STEM_L2_pooled)]:
                    if name not in scalers[scope_key]:
                        q1, q2 = compute_quantile_scaler(s.to_numpy())
                        scalers[scope_key][name] = {'q_low': q1, 'q_high': q2}
                T_n    = normalize_array(sT_w.to_numpy(),  scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high'])
                RH_n   = normalize_array(sRH_w.to_numpy(), scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high'])
                STEM_n = normalize_array(sST_w.to_numpy(), scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high'])

            # Compose X (11 channels, 10-min)
            X_seg = np.column_stack([
                T_n, RH_n, STEM_n,
                glb_n['g_tmean'], glb_n['g_tmin'], glb_n['g_tmax'],
                glb_n['g_rh'], glb_n['g_vpd'], glb_n['g_pr'], glb_n['g_rad'], glb_n['g_doy'],
            ]).astype(np.float32)
            if X_seg.shape[0] != SEQ_LEN_10MIN:
                continue

            # Targets y (hourly)
            if target_mode == 'lm_site_median' and (site_targets_hourly is not None):
                dfh = site_targets_hourly.reindex(idx_h)
                q1s, q2s = scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high']
                q1t, q2t = scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high']
                q1r, q2r = scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high']
                stem_hr = normalize_array(dfh['stem'].to_numpy(), q1s, q2s)
                temp_hr = normalize_array(dfh['temp'].to_numpy(), q1t, q2t)
                rh_hr   = normalize_array(dfh['rh'].to_numpy(),   q1r, q2r)
                y_seg = np.stack([temp_hr, rh_hr, stem_hr], axis=-1).astype(np.float32)
            else:
                # Hourly medians from local normalized inputs
                T_hr   = np.median(T_n.reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                RH_hr  = np.median(RH_n.reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                if target_mode == 'lm_per_series':
                    try:
                        df_lm = read_lm_frame(d_id, dendro_lm_dir, tz)
                        if stem_mode == 'delta':
                            df_lm['value'] = df_lm['value'].diff()
                        df_lm_h = to_hourly(df_lm, how='median').reindex(idx_h)
                        q1s, q2s = scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high']
                        stem_hr = normalize_array(df_lm_h['value'].to_numpy(), q1s, q2s)
                        stem_temp_hr = normalize_array(df_lm_h['temp'].to_numpy(), scalers[scope_key]['local_T']['q_low'],  scalers[scope_key]['local_T']['q_high'])
                        stem_rh_hr   = normalize_array(df_lm_h['rh'].to_numpy(),   scalers[scope_key]['local_RH']['q_low'], scalers[scope_key]['local_RH']['q_high'])
                        T_hr = np.where(np.isnan(stem_temp_hr), T_hr, stem_temp_hr)
                        RH_hr= np.where(np.isnan(stem_rh_hr),  RH_hr, stem_rh_hr)
                    except Exception:
                        STEM_hr = np.median(STEM_n.reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                        stem_hr = STEM_hr
                else:
                    STEM_hr = np.median(STEM_n.reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                    stem_hr = STEM_hr
                y_seg = np.stack([T_hr, RH_hr, stem_hr], axis=-1).astype(np.float32)

            if y_seg.shape[0] != HOUR_STEPS:
                continue

            X_list.append(X_seg)
            Y_list.append(y_seg)
            META_list.append({
                'site_id': site_id,
                'years_scope': f"{min(years)}-{max(years)}",
                'window_start': str(ws),
                'window_end': str(we),
                'input_mode': input_mode,
                'target_mode': target_mode,
                'thermometer_id': int(t_id) if t_id is not None else -1,
                'hygrometer_id': int(h_id) if h_id is not None else -1,
                'dendrometer_id': int(d_id) if d_id is not None else -1,
            })
            segments_count += 1

    if diagnostics_rows is not None:
        write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, windows=windows_len, segments=segments_count)

    return X_list, Y_list, META_list

# =====================
# Build datasets (train/test)
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
                   max_combos_per_site: t.Optional[int] = None,
                   min_local_coverage: float = 0.7,
                   min_lm_series: int = 1,
                   overlap_days: int = 10) -> None:
    """High-level orchestration to build TRAIN/TEST arrays and diagnostics.

    It fits site scalers (globals), iterates sites, constructs segments via
    `build_segments_for_site`, concatenates results, writes arrays and CSVs.

    Parameters
    ----------
    out_root : str
        Output folder for arrays/normalizers/diagnostics.
    meteo_dir, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir : str
        Input directories.
    metadata_pickle : str
        Path to `metadata_all.pkl` (must include columns `series_id`, `site_id`).
    site_ids_train, site_ids_test : Sequence[int]
        Train/test site ID lists.
    years : Sequence[int]
        Years to include (e.g., [2014, 2015, 2016]).
    per_year : bool
        If True, use per-year scalers; else a single ALL-years scaler.
    tz : str
        Timezone for timestamps.
    require_complete_locals : bool
        If True, filter windows requiring complete local channels (rarely recommended).
    stem_mode, input_mode, target_mode : str
        Behavior flags described above.
    max_combos_per_site : Optional[int]
        Cap the number of instrument triples per site to control dataset size.
    min_local_coverage : float
        Minimum fraction of observed local samples within a window to accept a combo.
    min_lm_series : int
        Minimum LM series per site required for site-level target aggregation.
    overlap_days : int
        Overlap in days for year-long windows.
    """
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)

    # Fit scalers per site
    site_scalers: dict[int, dict] = {}
    normalizers_summary_rows: t.List[dict] = []

    for sid in site_ids_train:
        scalers = fit_site_scalers(sid, years, meteo_dir, tz, per_year=per_year)
        site_scalers[sid] = scalers
        norm_dir = os.path.join(out_root, 'normalizers'); os.makedirs(norm_dir, exist_ok=True)
        with open(os.path.join(norm_dir, f"normalizers_site_{sid}.json"), 'w', encoding='utf-8') as f:
            json.dump(scalers, f, indent=2)
        for scope, chmap in scalers.items():
            for ch, qq in chmap.items():
                normalizers_summary_rows.append({'site_id': sid, 'scope': scope, 'channel': ch, 'q_low': qq['q_low'], 'q_high': qq['q_high']})

    # TRAIN
    X_tr_list: t.List[np.ndarray] = []
    y_tr_list: t.List[np.ndarray] = []
    sid_tr_list: t.List[int] = []
    meta_rows: t.List[dict] = []
    diagnostics_rows: t.List[dict] = []

    for sid in site_ids_train:
        scalers = site_scalers[sid]
        X_list, y_list, meta_list = build_segments_for_site(
            site_id=sid, years=years, tz=tz,
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
            max_combos_per_site=max_combos_per_site,
            min_local_coverage=min_local_coverage,
            min_lm_series=min_lm_series,
            overlap_days=overlap_days,
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
            for scope, chmap in scalers.items():
                for ch, qq in chmap.items():
                    normalizers_summary_rows.append({'site_id': sid, 'scope': scope, 'channel': ch, 'q_low': qq['q_low'], 'q_high': qq['q_high']})

        X_te_list: t.List[np.ndarray] = []
        y_te_list: t.List[np.ndarray] = []
        sid_te_list: t.List[int] = []
        meta_rows_te: t.List[dict] = []

        for sid in site_ids_test:
            scalers = site_scalers[sid]
            X_list, y_list, meta_list = build_segments_for_site(
                site_id=sid, years=years, tz=tz,
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
                max_combos_per_site=max_combos_per_site,
                min_local_coverage=min_local_coverage,
                min_lm_series=min_lm_series,
                overlap_days=overlap_days,
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

    instrument_df = compute_site_instrument_counts(metadata_pickle, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir)
    write_site_summaries(out_root, diagnostics_rows, instrument_df)

# =====================
# CLI
# =====================

def parse_args():
    """Parse command-line arguments for the pre-processing workflow.

    The defaults favor `combinations` input mode, `lm_site_median` targets, and a 10-day overlap.
    """
    p = argparse.ArgumentParser(description='TreeNet pre-processing with YEAR-LONG segments, overlap, normalization, diagnostics, and site summary.')
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
    p.add_argument('--max_combos_per_site', type=int, default=None, help='Cap number of instrument combinations per site (None = no cap).')
    p.add_argument('--min_local_coverage', type=float, default=0.7, help='Minimum fraction of non-NaN local samples per window (default 0.7).')
    p.add_argument('--min_lm_series', type=int, default=1, help='Minimum number of LM series at site for site-level target (default 1).')
    p.add_argument('--overlap_days', type=int, default=10, help='Overlap in days between year-long windows (default 10).')
    return p.parse_args()


def read_site_ids_csv(path: str) -> t.List[int]:
    """Read a CSV that contains a `site_id` column and return it as a list of ints."""
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
        max_combos_per_site=args.max_combos_per_site,
        min_local_coverage=args.min_local_coverage,
        min_lm_series=args.min_lm_series,
        overlap_days=args.overlap_days,
    )
