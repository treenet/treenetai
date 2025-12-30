# build_normalized_dataset_treenet_utc.py
# -*- coding: utf-8 -*-
"""
TreeNet → TF Pre‑processing — **UTC‑Only Core + Strategy B for Daily Globals**, Fully Annotated
==============================================================================================

This module builds normalized **multichannel year‑long 10‑min input segments** and **hourly targets**
for machine learning, operating entirely on a **UTC timeline** for local instruments and LM targets.

**Global daily climate data** are defined per **civil day in Europe/Zurich**. To merge them with the UTC grid
without losing civil‑day semantics, we use **Strategy B (civil mapping)**:

> For each UTC timestamp, we map it to the **civil day** it belongs to in Europe/Zurich and then look up the
> corresponding daily global row. That daily value is **broadcast across** the UTC 10‑min (or hourly) steps.

Key benefits
------------
- **DST‑proof computations** (everything aligned in UTC for local/LM series),
- **Correct civil semantics** for daily globals (no off‑by‑one on DST boundaries),
- Clean, readable broadcasting logic that works for date‑only daily inputs.

This file includes:
- Robust readers converting **local/LM** time series to **UTC**,
- Strategy‑B helpers to **broadcast daily civil globals** onto the UTC grid,
- Year‑long rolling window segment builder with normalization,
- A small **self‑test** to verify DST fall‑back correctness for Strategy B.
"""

from __future__ import annotations
import os
import re
import json
import argparse
import typing as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================
# Constants
# =============================================================
SEQ_LEN_10MIN = 52560  # 365d * 24h * 6 (leap day removed)
HOUR_STEPS    = 8760   # 365d * 24h
STRIDE_PER_HR = 6
# Channels: 3 locals + 7 globals + g_doy (civil DOY) = 11
N_CHANNELS    = 11
N_TARGETS     = 3  # hourly temp, rh, stem
LOCAL_COLS  = ['local_T', 'local_RH', 'local_stem']
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']

# =============================================================
# UTC axis helpers & indices
# =============================================================

def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove **Feb 29** to keep fixed‑length arrays across years.

    Notes
    -----
    Many downstream array shapes assume 365 days per year.
    If you prefer to retain Feb 29, remove this filter and adjust shapes accordingly.
    """
    return idx[~((idx.month==2) & (idx.day==29))]


def make_multi_year_10m_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    """Create a **UTC 10‑min multi‑year grid** spanning `min(years)`..`max(years)` with leap day removed."""
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq='10min')
    return strip_leap_days(idx)


def make_multi_year_hourly_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    """Create a **UTC hourly multi‑year grid** spanning `min(years)`..`max(years)` with leap day removed."""
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq='1H')
    return strip_leap_days(idx)

# =============================================================
# Readers (locals/LM to UTC) and daily globals (civil dates)
# =============================================================

def discover_meteo_files(meteo_dir: str) -> dict[int, str]:
    """Return `{site_id: path}` mapping for daily global meteo CSVs.

    File naming convention: any file containing an integer site ID will be mapped to that ID.
    If multiple files match a site, the first encountered will be used.
    """
    mapping: dict[int, str] = {}
    for fn in os.listdir(meteo_dir):
        if fn.endswith('.csv'):
            m = re.findall(r'\d+', fn)
            if m:
                mapping[int(m[0])] = os.path.join(meteo_dir, fn)
    return mapping


def load_global_daily_civil(site_id: int, meteo_dir: str) -> pd.DataFrame:
    """Load **daily global meteo** indexed by **civil dates** (Europe/Zurich), no time component.

    Expected input columns (will be copied/renamed):
    - `ts` (date string like 'YYYY-MM-DD' or datetime without tz)
    - `tas` (mean T), `tasmax`, `tasmin`, `rh`, `vpd`, `gh` (global radiation), `pr` (precip)

    Returns
    -------
    DataFrame
        Indexed by **civil date** (tz‑naive midnight), with standardized columns:
        `g_tmean, g_tmin, g_tmax, g_rh, g_vpd, g_pr, g_rad`.
    """
    files = discover_meteo_files(meteo_dir)
    if site_id not in files:
        raise FileNotFoundError(f'No meteo CSV for site {site_id}')
    df = pd.read_csv(files[site_id])
    if 'ts' not in df.columns:
        raise ValueError('Global meteo CSV must have ts column (civil dates)')

    # Parse as date and normalize (keep tz‑naive date index to represent civil date labels)
    idx_civil = pd.to_datetime(df['ts'], utc=False).normalize()

    rename = {
        'tas':'g_tmean', 'tasmax':'g_tmax', 'tasmin':'g_tmin',
        'rh':'g_rh', 'vpd':'g_vpd', 'gh':'g_rad', 'pr':'g_pr'
    }
    for src, dst in rename.items():
        if src not in df.columns:
            raise ValueError(f'Global meteo missing {src}')
        df[dst] = df[src]

    out = df.set_index(idx_civil)[['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad']].copy()
    # Ensure index has unique dates (last wins); you can change policy if needed
    out = out[~out.index.duplicated(keep='last')]
    return out


def _pick_value_column(df: pd.DataFrame, preferred: t.Sequence[str] = ()) -> str:
    """Pick a numeric measurement column from `df`.

    Priority order:
    1) first name in `preferred` present in `df`,
    2) `'value'`,
    3) numeric column with **max non‑null** count.
    """
    for c in preferred:
        if c in df.columns:
            return c
    if 'value' in df.columns:
        return 'value'
    candidates: list[tuple[int, str]] = []
    for c in df.columns:
        if c.lower() in ('ts','timestamp','time','date_time','datetime'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append((int(df[c].notna().sum()), c))
    if not candidates:
        raise ValueError('No numeric value column found')
    candidates.sort(reverse=True)
    return candidates[0][1]


def read_feather_series_utc(series_id: int, dir_path: str, local_tz: str,
                            value_col: str = 'value', ts_col: str = 'ts', sensor_hint: str | None = None) -> pd.Series:
    """Read a **local instrument** feather (`*series_id_<id>.ftr`) and return a **UTC** series.

    Steps:
    - Detect timestamp column and a numeric value column (uses `sensor_hint` to bias selection),
    - Localize naive timestamps to `local_tz` (e.g., Europe/Zurich) to resolve DST,
    - Convert to **UTC** (tz_convert), sort, and deduplicate exact duplicates by mean.
    """
    pat = re.compile(rf'.*series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(dir_path) if pat.match(fn)]
    if not matches:
        raise FileNotFoundError(f'Series {series_id} not found in {dir_path}')
    fp = os.path.join(dir_path, matches[0])
    df = pd.read_feather(fp)

    # Timestamp column detection
    if ts_col not in df.columns:
        for alt in ('timestamp','time','date_time','datetime'):
            if alt in df.columns:
                ts_col = alt; break
    if ts_col not in df.columns:
        raise ValueError(f'Feather {fp} missing timestamp column')

    # Value column heuristic
    if sensor_hint == 'thermometer_l1':
        preferred = ('temp','temperature','value')
    elif sensor_hint == 'hygrometer_l1':
        preferred = ('rh','relhum','value')
    elif sensor_hint == 'dendrometer_l2':
        preferred = ('value','radius','rad','l2')
    else:
        preferred = ()
    if value_col not in df.columns:
        value_col = _pick_value_column(df, preferred)

    ts = pd.to_datetime(df[ts_col], utc=False)
    ts = ts.dt.tz_localize(local_tz) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(local_tz)
    ts = ts.dt.tz_convert('UTC')
    s = pd.Series(df[value_col].to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated(keep='mean')]
    return s


def read_lm_frame_utc(series_id: int, lm_dir: str, local_tz: str) -> pd.DataFrame:
    """Read an LM dendrometer frame and return a **UTC** frame with columns `value`, `temp`, `rh`.

    Missing columns are added as NaN; index is tz‑aware UTC.
    """
    pat = re.compile(rf'dendrometer_lm_series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(lm_dir) if pat.match(fn)]
    if not matches:
        raise FileNotFoundError(f'LM {series_id} missing in {lm_dir}')
    fp = os.path.join(lm_dir, matches[0])
    df = pd.read_feather(fp)

    ts_col = 'ts'
    if ts_col not in df.columns:
        for alt in ('timestamp','time','date_time','datetime'):
            if alt in df.columns:
                ts_col = alt; break
    if ts_col not in df.columns:
        raise ValueError(f'LM {fp} missing timestamp')

    ts = pd.to_datetime(df[ts_col], utc=False)
    ts = ts.dt.tz_localize(local_tz) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(local_tz)
    ts = ts.dt.tz_convert('UTC')

    df = df.set_index(ts)
    for col in ['value','temp','rh']:
        if col not in df.columns:
            df[col] = np.nan
    return df[['value','temp','rh']]

# =============================================================
# Strategy B broadcasting: civil day → UTC grid
# =============================================================

def broadcast_daily_civil_to_utc_grid(daily_civil: pd.DataFrame,
                                      idx_10m_utc: pd.DatetimeIndex,
                                      tz_local: str = "Europe/Zurich") -> pd.DataFrame:
    """Broadcast **daily civil‑time** globals onto a **UTC 10‑min** grid (Strategy **B**).

    Parameters
    ----------
    daily_civil : DataFrame
        Indexed by **civil dates** (tz‑naive) with columns like `g_tmean, g_tmin, g_tmax, g_rh, g_vpd, g_pr, g_rad`.
    idx_10m_utc : DatetimeIndex
        Master **UTC** 10‑min index for your inputs.
    tz_local : str
        Civil timezone that defines daily semantics (default: Europe/Zurich).

    Returns
    -------
    DataFrame
        Same columns as `daily_civil` (plus you can add derived ones), indexed by `idx_10m_utc`.
        For each UTC instant, we **convert** to local civil time and take its **normalized date**;
        that date is used to look up the daily row; values are then **forward‑filled** across the day.

    Notes
    -----
    - This avoids anchoring daily rows to artificial timestamps (e.g., Zurich midnight mapping to 23:00 UTC in summer),
      preventing off‑by‑one surprises near DST boundaries.
    - Ensure the `daily_civil.index` is **unique** and normalized to midnight; duplicates will be resolved upstream.
    """
    if getattr(idx_10m_utc, 'tz', None) is None:
        raise ValueError('idx_10m_utc must be tz‑aware UTC index')

    # Civil day label for each UTC step
    civil_days_for_grid = idx_10m_utc.tz_convert(tz_local).normalize()

    # Guarantee daily index is normalized civil dates
    daily_by_civilday = daily_civil.copy()
    daily_by_civilday.index = pd.to_datetime(daily_by_civilday.index).normalize()

    # Reindex by civil day series; forward-fill to cover any missing day heads
    out = daily_by_civilday.reindex(civil_days_for_grid).ffill()
    out.index = idx_10m_utc  # restore UTC 10‑min index
    return out

# =============================================================
# Axis‑aware (UTC) alignment for locals/LM
# =============================================================

# TODO: this function does not work. It converts all the instrument readings into NaN.
def round_to_10min_utc(s: pd.Series) -> pd.Series:
    """Round to **10‑minute UTC** bins and collapse duplicates by mean.

    This is safe and deterministic because **UTC has no DST transitions**.
    """
    if s.empty:
        return s
    s = s.copy().tz_convert('UTC')
    s.index = s.index.round('10min')
    return s[~s.index.duplicated(keep='mean')]


# TODO: solution
# use 'floor' for speed and 'resample' for robustness
# Recommendation: Start with method='resample', how='median' for robustness; 
# switch to floor if you need speed and your data time stamps are stable.
def bin_to_10min_utc(s: pd.Series, method: str = 'floor', how: str = 'mean') -> pd.Series:
    """
    Bin a tz-aware series to the 10-min UTC grid deterministically.

    method:
      - 'floor': floor timestamps to 10-min bins, collapse duplicates by mean
      - 'resample': resample to explicit 10-min UTC bins, aggregate by `how`
    """
    if method == 'floor':
        s = s.copy().tz_convert('UTC').sort_index()
        s.index = s.index.floor('10min')
        return s[~s.index.duplicated(keep='mean')]
    elif method == 'resample':
        s = s.copy().tz_convert('UTC').sort_index()
        agg = {'mean': 'mean', 'median': 'median'}.get(how, 'mean')
        return s.resample('10min', origin='start_day', label='left').agg(agg)
    else:
        raise ValueError("method must be 'floor' or 'resample'")




# =============================================================
# Quantile scalers
# =============================================================

def compute_quantile_scaler(v: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> t.Tuple[float,float]:
    """Return robust quantiles `(q_low, q_high)`; fallback to min/max if necessary.

    Ensures `q_high > q_low` to avoid near‑zero denominators during normalization.
    """
    q1 = float(np.nanpercentile(v, q_low)); q2 = float(np.nanpercentile(v, q_high))
    if not np.isfinite(q1): q1 = float(np.nanmin(v))
    if not np.isfinite(q2): q2 = float(np.nanmax(v))
    if q2 <= q1: q2 = q1 + 1e-6
    return q1, q2


def fit_site_scalers(site_id: int, years: t.Sequence[int], meteo_dir: str,
                     per_year: bool = True) -> dict:
    """Fit **global channel** quantile scalers per site either per‑year or for **ALL**.

    We fit on the **daily civil** globals (the original daily data). Quantiles are independent
    of the time axis used for alignment and are robust to DST boundaries.
    """
    scalers: dict = {}
    glb_daily = load_global_daily_civil(site_id, meteo_dir)

    scopes = [y for y in years] if per_year else ['ALL']
    for scope in scopes:
        if scope == 'ALL':
            df = glb_daily[(glb_daily.index.year>=min(years)) & (glb_daily.index.year<=max(years))]
        else:
            df = glb_daily[glb_daily.index.year == scope]
        scalers.setdefault(scope, {})
        for col in ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad']:
            q1, q2 = compute_quantile_scaler(df[col].to_numpy())
            scalers[scope][col] = {'q_low': q1, 'q_high': q2}
        # For DOY, we can set [1,365] range directly (or compute empirical quantiles)
        scalers[scope]['g_doy'] = {'q_low': 1.0, 'q_high': 365.0}
    return scalers


def normalize_array(arr: np.ndarray, q1: float, q2: float,
                    clip_low: float = -0.1, clip_high: float = 1.1) -> np.ndarray:
    """Normalize to roughly `[0,1]` via `(arr - q1)/(q2 - q1)` and clip to a safe margin."""
    out = (arr - q1) / (q2 - q1)
    return np.clip(out, clip_low, clip_high)

# =============================================================
# Sensor discovery per site
# =============================================================

def discover_series_ids(dir_path: str, pattern_prefix: str) -> t.Set[int]:
    """Discover instrument files matching `{prefix}_series_id_<id>.ftr` and return the set of IDs."""
    ids: set[int] = set()
    regex = re.compile(rf'{re.escape(pattern_prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m: ids.add(int(m.group(1)))
    return ids


def series_by_site(metadata_df: pd.DataFrame, series_ids: t.Set[int]) -> dict[int, t.List[int]]:
    """Map a set of `series_ids` to `site_id` using columns `series_id` and `site_id` in `metadata_df`."""
    df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
    out: dict[int,t.List[int]] = {}
    for site, g in df.groupby('site_id'):
        out[int(site)] = [int(x) for x in g['series_id'].tolist()]
    return out

# =============================================================
# LM targets (UTC)
# =============================================================

def to_hourly(df: pd.DataFrame, how: str='median') -> pd.DataFrame:
    """Resample a UTC frame to **hourly** by `how` (`median` or `mean`)."""
    return df.resample('1H').agg({'value':how,'temp':how,'rh':how})


def build_site_level_targets_multi_utc(site_id: int, years: t.Sequence[int], local_tz: str, lm_dir: str,
                                       dendro_lm_ids_by_site: dict[int,t.List[int]], stem_mode: str='absolute',
                                       agg: str='median') -> pd.DataFrame:
    """Aggregate all LM series at the site into **site‑level hourly targets** on a **UTC** index.

    Returns a frame with columns `stem`, `temp`, `rh` aligned to the multi‑year UTC hourly index.
    """
    idx_hour = make_multi_year_hourly_index_utc(years)
    frames = []
    lm_ids = dendro_lm_ids_by_site.get(site_id, [])
    for sid in lm_ids:
        try:
            df = read_lm_frame_utc(sid, lm_dir, local_tz)
        except Exception:
            continue
        df = df[(df.index.year>=min(years)) & (df.index.year<=max(years))].copy()
        if df.empty: continue
        if stem_mode=='delta': df['value'] = df['value'].diff()
        dfh = to_hourly(df, how=agg)
        dfh.columns = [f'stem_{sid}', f'temp_{sid}', f'rh_{sid}']
        dfh = dfh.reindex(idx_hour)
        frames.append(dfh)
    if not frames:
        return pd.DataFrame(index=idx_hour, columns=['stem','temp','rh'])
    big = pd.concat(frames, axis=1)
    stem_cols = [c for c in big.columns if c.startswith('stem_')]
    temp_cols = [c for c in big.columns if c.startswith('temp_')]
    rh_cols   = [c for c in big.columns if c.startswith('rh_')]
    agg_func = np.nanmedian if agg=='median' else np.nanmean
    stem_site = big[stem_cols].apply(agg_func, axis=1)
    temp_site = big[temp_cols].apply(agg_func, axis=1) if temp_cols else pd.Series(np.nan, index=big.index)
    rh_site   = big[rh_cols].apply(agg_func, axis=1)   if rh_cols   else pd.Series(np.nan, index=big.index)
    return pd.DataFrame({'stem': stem_site, 'temp': temp_site, 'rh': rh_site}, index=idx_hour)

# =============================================================
# Coverage & diagnostics helpers
# =============================================================

def coverage_fraction(series: pd.Series) -> float:
    """Fraction of non‑NaN entries in `series` (safe for empty)."""
    v = series.to_numpy(); n=v.size
    return float(np.sum(~np.isnan(v))/n) if n else 0.0


def write_diag(rows: t.List[dict], **kw) -> None:
    """Append a diagnostics `kw` dict to the rows list (no‑op if rows is None)."""
    if rows is not None:
        rows.append(kw)

# =============================================================
# Rolling YEAR windows
# =============================================================

def rolling_year_windows(idx_10m: pd.DatetimeIndex, overlap_days: int=10, year_days: int=365) -> t.List[t.Tuple[pd.Timestamp,pd.Timestamp]]:
    """Generate `(start, end)` for **year‑long** windows on a UTC grid with `overlap_days` overlap.

    Example: with `overlap_days=10`, consecutive windows start 355 days apart.
    """
    stride_days = year_days - overlap_days
    starts = []
    cur = idx_10m[0].normalize(); end_ts = idx_10m[-1]
    while cur + pd.Timedelta(days=year_days) <= end_ts:
        starts.append(cur); cur = cur + pd.Timedelta(days=stride_days)
    return [(s, s + pd.Timedelta(days=year_days)) for s in starts]

# =============================================================
# Build segments per site (UTC + Strategy B for globals)
# =============================================================

def build_segments_for_site_utc(
    site_id: int,
    years: t.Sequence[int],
    local_tz: str,
    meteo_dir: str,
    thermo_dir: str,
    hygro_dir: str,
    dendro_l2_dir: str,
    dendro_lm_dir: str,
    metadata_path: str,
    scalers: dict,
    per_year: bool=True,
    require_complete_locals: bool=False,
    stem_mode: str='absolute',
    input_mode: str='combinations',  # 'best' | 'combinations' | 'pooled'
    target_mode: str='lm_site_median',
    max_combos_per_site: t.Optional[int]=None,
    min_local_coverage: float=0.7,
    min_lm_series: int=1,
    overlap_days: int=10,
    diagnostics_rows: t.Optional[t.List[dict]]=None,
    split: str='train',
    allow_missing_locals: bool=False,
    globals_broadcast_strategy: str='civil_map',  # Strategy B only (kept as a flag for explicitness)
) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[dict]]:
    """Construct **UTC year‑long** training segments for a single site using **Strategy B** for globals.

    Workflow
    --------
    1) Build **UTC** multi‑year indices (10‑min & hourly),
    2) Load **daily civil** globals → **broadcast** to UTC 10‑min via Strategy **B**,
    3) Read local series (T/RH/L2) → **UTC**, round to 10‑min, reindex on UTC grid,
    4) Build LM site‑level hourly targets (UTC),
    5) Slide **year‑long** windows, apply coverage thresholds and compose X (11 ch) and y (3 ch).

    Returns
    -------
    X_list, Y_list, META_list : list
        Lists of input segments, target segments, and metadata rows for this site.
    """
    if globals_broadcast_strategy != 'civil_map':
        raise ValueError("Only Strategy B ('civil_map') is supported in this module.")

    metadata_df = pd.read_pickle(metadata_path)
    idx_10m = make_multi_year_10m_index_utc(years)
    idx_hour= make_multi_year_hourly_index_utc(years)

    # 2) Globals: load **civil daily** and broadcast to **UTC 10‑min** (Strategy B)
    glb_daily = load_global_daily_civil(site_id, meteo_dir)
    glb_10m = broadcast_daily_civil_to_utc_grid(glb_daily, idx_10m, tz_local=local_tz)
    # Civil DOY channel derived from UTC index
    glb_10m['g_doy'] = idx_10m.tz_convert(local_tz).dayofyear

    # Series IDs per type
    thermo_ids_all = discover_series_ids(thermo_dir, 'thermometer_l1')
    hygro_ids_all  = discover_series_ids(hygro_dir,  'hygrometer_l1')
    d2_ids_all     = discover_series_ids(dendro_l2_dir, 'dendrometer_l2')
    lm_ids_all     = discover_series_ids(dendro_lm_dir, 'dendrometer_lm')
    thermo_by_site = series_by_site(metadata_df, thermo_ids_all)
    hygro_by_site  = series_by_site(metadata_df, hygro_ids_all)
    d2_by_site     = series_by_site(metadata_df, d2_ids_all)
    lm_by_site     = series_by_site(metadata_df, lm_ids_all)

    if site_id not in thermo_by_site or site_id not in hygro_by_site or site_id not in d2_by_site:
        return [], [], []

    thermo_ids = thermo_by_site[site_id]
    hygro_ids  = hygro_by_site[site_id]
    dendro_l2_ids = d2_by_site[site_id]
    lm_ids_site = lm_by_site.get(site_id, [])

    # 4) Site‑level LM targets (UTC)
    site_targets_hourly = None
    if target_mode == 'lm_site_median':
        site_targets_hourly = build_site_level_targets_multi_utc(site_id, years, local_tz, dendro_lm_dir, lm_by_site, stem_mode=stem_mode, agg='median')
        if len(lm_ids_site) < min_lm_series:
            write_diag(diagnostics_rows, site_id=site_id, split=split, warning='insufficient_lm_series', lm_series_count=len(lm_ids_site))
            return [], [], []

    windows = rolling_year_windows(idx_10m, overlap_days=overlap_days, year_days=365)
    key_for = (lambda ts: ts.year) if per_year else (lambda ts: 'ALL')

    X_list: t.List[np.ndarray] = []
    Y_list: t.List[np.ndarray] = []
    META_list: t.List[dict] = []
    segments_count = 0

    # Optional pooled strategy
    if input_mode == 'pooled':
        def pooled_series(ids, dirp, hint=None):
            arrs = []
            for sid in ids:
                try:
                    s_ser = read_feather_series_utc(sid, dirp, local_tz, sensor_hint=hint)
                    # s_ser = round_to_10min_utc(s_ser)
                    arrs.append(s_ser.reindex(idx_10m).to_numpy())
                except Exception:
                    pass
            if not arrs:
                return pd.Series(np.nan, index=idx_10m)
            A = np.vstack(arrs); med = np.nanmedian(A, axis=0)
            return pd.Series(med, index=idx_10m)
        s_T_pooled   = pooled_series(thermo_ids, thermo_dir, 'thermometer_l1')
        s_RH_pooled  = pooled_series(hygro_ids,  hygro_dir,  'hygrometer_l1')
        s_STEM_pooled= pooled_series(dendro_l2_ids, dendro_l2_dir, 'dendrometer_l2')
        if stem_mode == 'delta': s_STEM_pooled = s_STEM_pooled.diff()

    for (ws, we) in windows:
        scope_key = key_for(ws)
        glb_win = glb_10m[(glb_10m.index>=ws) & (glb_10m.index<we)]
        if glb_win.shape[0] != SEQ_LEN_10MIN:
            continue

        # Normalize global channels for this window
        glb_n = { col: normalize_array(glb_win[col].to_numpy(), scalers[scope_key][col]['q_low'], scalers[scope_key][col]['q_high']) for col in GLOBAL_COLS }

        # Hourly index for targets (UTC)
        start_h = ws; end_h = we - pd.Timedelta(minutes=10)
        idx_h = pd.date_range(start_h.floor('H'), end_h.floor('H'), freq='1H')
        idx_h = strip_leap_days(idx_h)

        def cov_in_window(series_id, dirp, do_diff=False, hint=None):
            try:
                s = read_feather_series_utc(series_id, dirp, local_tz, sensor_hint=hint)
                # s = round_to_10min_utc(s)
                s = s[(s.index>=ws) & (s.index<we)]
                s = s.reindex(idx_10m)
                if do_diff: s = s.diff()
                v = s.to_numpy(); return float((~np.isnan(v)).sum()/v.size) if v.size else 0.0
            except Exception:
                return 0.0

        if input_mode == 'pooled':
            t_candidates=[None]; h_candidates=[None]; d_candidates=[None]
        elif input_mode == 'best':
            best_triplet=None; best_score=-1.0
            for t_id0 in thermo_ids:
                covT=cov_in_window(t_id0, thermo_dir, False, 'thermometer_l1')
                for h_id0 in hygro_ids:
                    covRH=cov_in_window(h_id0, hygro_dir, False, 'hygrometer_l1')
                    for d_id0 in dendro_l2_ids:
                        covST=cov_in_window(d_id0, dendro_l2_dir, stem_mode=='delta', 'dendrometer_l2')
                        score=min(covT,covRH,covST)
                        if score>best_score:
                            best_score=score; best_triplet=(t_id0,h_id0,d_id0)
            if (best_triplet is None) or (best_score < min_local_coverage):
                write_diag(diagnostics_rows, site_id=site_id, split=split, window_start=str(ws), window_end=str(we), warning='no_best_combo_in_window', min_local_coverage=min_local_coverage)
                continue
            t_candidates=[best_triplet[0]]; h_candidates=[best_triplet[1]]; d_candidates=[best_triplet[2]]
        else:
            t_candidates=thermo_ids; h_candidates=hygro_ids; d_candidates=dendro_l2_ids

        for t_id in t_candidates:
            for h_id in h_candidates:
                for d_id in d_candidates:
                    if input_mode != 'pooled':
                        try:
                            s_T  = read_feather_series_utc(t_id, thermo_dir, local_tz, sensor_hint='thermometer_l1')
                            # s_T  = round_to_10min_utc(s_T).reindex(idx_10m)
                            s_RH = read_feather_series_utc(h_id, hygro_dir,  local_tz, sensor_hint='hygrometer_l1')
                            # s_RH = round_to_10min_utc(s_RH).reindex(idx_10m)
                            s_ST = read_feather_series_utc(d_id, dendro_l2_dir, local_tz, sensor_hint='dendrometer_l2')
                            # s_ST = round_to_10min_utc(s_ST).reindex(idx_10m)
                        except Exception:
                            continue
                        if stem_mode=='delta': s_ST = s_ST.diff()
                        sT_w  = s_T[(s_T.index>=ws)&(s_T.index<we)]
                        sRH_w = s_RH[(s_RH.index>=ws)&(s_RH.index<we)]
                        sST_w = s_ST[(s_ST.index>=ws)&(s_ST.index<we)]
                    else:
                        sT_w  = s_T_pooled[(s_T_pooled.index>=ws)&(s_T_pooled.index<we)]
                        sRH_w = s_RH_pooled[(s_RH_pooled.index>=ws)&(s_RH_pooled.index<we)]
                        sST_w = s_STEM_pooled[(s_STEM_pooled.index>=ws)&(s_STEM_pooled.index<we)]
                        if stem_mode=='delta': sST_w = sST_w.diff()

                    cov_T = coverage_fraction(sT_w)
                    cov_RH= coverage_fraction(sRH_w)
                    cov_ST= coverage_fraction(sST_w)

                    used_global_T=False; used_global_RH=False
                    if allow_missing_locals:
                        if cov_T  < min_local_coverage: used_global_T=True
                        if cov_RH < min_local_coverage: used_global_RH=True
                    if (not allow_missing_locals) and ((cov_T<min_local_coverage) or (cov_RH<min_local_coverage) or (cov_ST<min_local_coverage)):
                        write_diag(diagnostics_rows, site_id=site_id, split=split, warning='low_local_coverage_combo', thermo_id=(t_id if t_id is not None else -1), hygro_id=(h_id if h_id is not None else -1), dendro_id=(d_id if d_id is not None else -1), window_start=str(ws), cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                        continue

                    # Lazy quantiles for locals (per scope) if missing
                    for name, sfull in [('local_T', s_T if input_mode!='pooled' else s_T_pooled),
                                        ('local_RH', s_RH if input_mode!='pooled' else s_RH_pooled),
                                        ('local_stem', s_ST if input_mode!='pooled' else s_STEM_pooled)]:
                        if name not in scalers[scope_key]:
                            q1,q2 = compute_quantile_scaler(sfull.to_numpy())
                            scalers[scope_key][name] = {'q_low': q1, 'q_high': q2}

                    # Normalize locals
                    T_n    = normalize_array(sT_w.to_numpy(),  scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high'])
                    RH_n   = normalize_array(sRH_w.to_numpy(), scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high'])
                    STEM_n = normalize_array(sST_w.to_numpy(), scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high'])

                    # Optional global substitution (inputs only)
                    if allow_missing_locals and used_global_T:  T_n = glb_n['g_tmean'].copy()
                    if allow_missing_locals and used_global_RH: RH_n = glb_n['g_rh'].copy()

                    # Compose X segment (11 channels)
                    X_seg = np.column_stack([
                        T_n, RH_n, STEM_n,
                        glb_n['g_tmean'], glb_n['g_tmin'], glb_n['g_tmax'],
                        glb_n['g_rh'], glb_n['g_vpd'], glb_n['g_pr'], glb_n['g_rad'], glb_n['g_doy'],
                    ]).astype(np.float32)
                    if X_seg.shape[0] != SEQ_LEN_10MIN:
                        continue

                    # Compose y targets: LM hourly (preferred) or hourly medians from locals
                    if (target_mode=='lm_site_median') and (site_targets_hourly is not None):
                        dfh = site_targets_hourly.reindex(idx_h)
                        q1s,q2s = scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high']
                        q1t,q2t = scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high']
                        q1r,q2r = scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high']
                        stem_hr = normalize_array(dfh['stem'].to_numpy(), q1s, q2s)
                        temp_hr = normalize_array(dfh['temp'].to_numpy(), q1t, q2t)
                        rh_hr   = normalize_array(dfh['rh'].to_numpy(),   q1r, q2r)
                        y_seg   = np.stack([temp_hr, rh_hr, stem_hr], axis=-1).astype(np.float32)
                    else:
                        T_hr   = np.median(T_n.reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                        RH_hr  = np.median(RH_n.reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                        STEM_hr= np.median(STEM_n.reshape(HOUR_STEPS, STRIDE_PER_HR), axis=1)
                        y_seg  = np.stack([T_hr, RH_hr, STEM_hr], axis=-1).astype(np.float32)

                    if y_seg.shape[0] != HOUR_STEPS:
                        continue

                    write_diag(
                        diagnostics_rows,
                        site_id=site_id, split=split,
                        window_start=str(ws), window_end=str(we),
                        thermo_id=(int(t_id) if t_id is not None else -1),
                        hygro_id=(int(h_id) if h_id is not None else -1),
                        dendro_id=(int(d_id) if d_id is not None else -1),
                        cov_T=float(np.mean(~np.isnan(T_n))),
                        cov_RH=float(np.mean(~np.isnan(RH_n))),
                        cov_ST=float(np.mean(~np.isnan(STEM_n))),
                    )

                    X_list.append(X_seg)
                    Y_list.append(y_seg)
                    META_list.append({
                        'site_id': site_id,
                        'years_scope': f"{min(years)}-{max(years)}",
                        'window_start': str(ws), 'window_end': str(we),
                        'input_mode': input_mode, 'target_mode': target_mode,
                        'thermometer_id': int(t_id) if t_id is not None else -1,
                        'hygrometer_id': int(h_id) if h_id is not None else -1,
                        'dendrometer_id': int(d_id) if d_id is not None else -1,
                    })
                    segments_count += 1

        write_diag(diagnostics_rows, site_id=site_id, split=split, windows=1, segments=segments_count)

    return X_list, Y_list, META_list

# =============================================================
# Build datasets (UTC + Strategy B only)
# =============================================================

def build_datasets_utc(
    out_root: str,
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
    local_tz: str = 'Europe/Zurich',
    require_complete_locals: bool = False,
    stem_mode: str = 'absolute',
    input_mode: str = 'combinations',
    target_mode: str = 'lm_site_median',
    max_combos_per_site: t.Optional[int] = None,
    min_local_coverage: float = 0.7,
    min_lm_series: int = 1,
    overlap_days: int = 10,
    allow_missing_locals: bool = False,
    globals_broadcast_strategy: str = 'civil_map',  # Strategy B only (explicit)
) -> None:
    """Top‑level orchestration to build **UTC** train/test arrays using **Strategy B** for daily globals.

    Saves arrays, identifiers, normalizers, diagnostics, and site‑level summaries under `out_root`.
    """
    if globals_broadcast_strategy != 'civil_map':
        raise ValueError("Only Strategy B ('civil_map') is supported in this module.")

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)

    # Fit global scalers for TRAIN sites (daily civil)
    site_scalers: dict[int, dict] = {}
    normalizers_summary_rows: t.List[dict] = []

    for sid in site_ids_train:
        scalers = fit_site_scalers(sid, years, meteo_dir, per_year=per_year)
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
        X_list, y_list, meta_list = build_segments_for_site_utc(
            site_id=sid, years=years, local_tz=local_tz,
            meteo_dir=meteo_dir, thermo_dir=thermo_dir, hygro_dir=hygro_dir, dendro_l2_dir=dendro_l2_dir, dendro_lm_dir=dendro_lm_dir,
            metadata_path=metadata_pickle, scalers=scalers,
            per_year=per_year, require_complete_locals=require_complete_locals,
            stem_mode=stem_mode, input_mode=input_mode, target_mode=target_mode,
            max_combos_per_site=max_combos_per_site, min_local_coverage=min_local_coverage,
            min_lm_series=min_lm_series, overlap_days=overlap_days,
            diagnostics_rows=diagnostics_rows, split='train',
            allow_missing_locals=allow_missing_locals,
            globals_broadcast_strategy=globals_broadcast_strategy,
        )
        X_tr_list.extend(X_list); y_tr_list.extend(y_list); sid_tr_list.extend([sid]*len(X_list)); meta_rows.extend(meta_list)

    X_train = np.stack(X_tr_list, axis=0) if X_tr_list else np.empty((0, SEQ_LEN_10MIN, N_CHANNELS), dtype=np.float32)
    y_train = np.stack(y_tr_list, axis=0) if y_tr_list else np.empty((0, HOUR_STEPS, N_TARGETS), dtype=np.float32)
    SID_train = np.array(sid_tr_list, dtype=np.int32)

    np.save(os.path.join(out_root, 'X_train.npy'), X_train)
    np.save(os.path.join(out_root, 'y_train.npy'), y_train)
    np.save(os.path.join(out_root, 'site_ids_train.npy'), SID_train)
    pd.DataFrame(meta_rows).to_csv(os.path.join(out_root, 'train_identifiers.csv'), index=False)
    print(f"Saved TRAIN arrays: X_train {X_train.shape}, y_train {y_train.shape}, site_ids_train {SID_train.shape}")

    # TEST
    if site_ids_test is not None:
        for sid in site_ids_test:
            scalers = fit_site_scalers(sid, years, meteo_dir, per_year=per_year)
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
            X_list, y_list, meta_list = build_segments_for_site_utc(
                site_id=sid, years=years, local_tz=local_tz,
                meteo_dir=meteo_dir, thermo_dir=thermo_dir, hygro_dir=hygro_dir, dendro_l2_dir=dendro_l2_dir, dendro_lm_dir=dendro_lm_dir,
                metadata_path=metadata_pickle, scalers=scalers,
                per_year=per_year, require_complete_locals=require_complete_locals,
                stem_mode=stem_mode, input_mode=input_mode, target_mode=target_mode,
                max_combos_per_site=max_combos_per_site, min_local_coverage=min_local_coverage,
                min_lm_series=min_lm_series, overlap_days=overlap_days,
                diagnostics_rows=diagnostics_rows, split='test',
                allow_missing_locals=allow_missing_locals,
                globals_broadcast_strategy=globals_broadcast_strategy,
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

    # Diagnostics and normalizers summary
    pd.DataFrame(diagnostics_rows).to_csv(os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'), index=False)
    pd.DataFrame(normalizers_summary_rows).to_csv(os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'), index=False)
    print("Wrote diagnostics:")
    print(" -", os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'))
    print(" -", os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'))

    # Site‑level summary: instrument counts + diagnostics aggregates
    instrument_df = compute_site_instrument_counts(metadata_pickle, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir)

    diag_df = pd.DataFrame(diagnostics_rows) if diagnostics_rows else pd.DataFrame(columns=['site_id','split','windows','segments','lm_series_count','warning'])
    for col, default in [('site_id', -1), ('split','train'), ('windows',0), ('segments',0), ('warning', None)]:
        if col not in diag_df.columns:
            diag_df[col] = default

    def agg_split(df, split):
        d = df[df['split'] == split] if 'split' in df.columns else df
        g = d.groupby('site_id').agg(
            segments_total=('segments', 'sum'),
            windows_total=('windows', 'sum'),
            warnings_low_local_combo=('warning', lambda x: int(np.sum(x == 'low_local_coverage_combo'))),
            warnings_insufficient_lm=('warning', lambda x: int(np.sum(x == 'insufficient_lm_series'))),
            warnings_no_best=('warning', lambda x: int(np.sum(x == 'no_best_combo_in_window'))),
        ).reset_index()
        g.columns = ['site_id'] + [f'{split}_{c}' for c in g.columns[1:]]
        return g

    site_train = agg_split(diag_df, 'train')
    site_test  = agg_split(diag_df, 'test')

    site_summary = instrument_df.merge(site_train, on='site_id', how='left').merge(site_test, on='site_id', how='left')
    fill_cols = [c for c in site_summary.columns if c != 'site_id']
    site_summary[fill_cols] = site_summary[fill_cols].fillna(0).infer_objects(copy=False)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)
    site_summary.to_csv(os.path.join(out_root, 'diagnostics', 'site_summary.csv'), index=False)
    print('Wrote site-level summary:', os.path.join(out_root, 'diagnostics', 'site_summary.csv'))

# =============================================================
# Instrument counts & civil‑time helpers
# =============================================================

def compute_site_instrument_counts(metadata_pickle: str, thermo_dir: str, hygro_dir: str, d2_dir: str, lm_dir: str) -> pd.DataFrame:
    """Count how many instruments of each type exist per site using metadata mapping."""
    metadata_df = pd.read_pickle(metadata_pickle)
    thermo_ids_all = discover_series_ids(thermo_dir, 'thermometer_l1')
    hygro_ids_all  = discover_series_ids(hygro_dir,  'hygrometer_l1')
    d2_ids_all     = discover_series_ids(d2_dir,     'dendrometer_l2')
    lm_ids_all     = discover_series_ids(lm_dir,     'dendrometer_lm')
    def counts(ids):
        df = metadata_df[metadata_df['series_id'].isin(ids)].copy()
        return df.groupby('site_id')['series_id'].count().to_dict()
    tC = counts(thermo_ids_all); hC = counts(hygro_ids_all); dC = counts(d2_ids_all); lC = counts(lm_ids_all)
    sites = sorted(set(list(tC.keys())+list(hC.keys())+list(dC.keys())+list(lC.keys())))
    return pd.DataFrame([{ 'site_id':int(s), 'n_thermometers':int(tC.get(s,0)), 'n_hygrometers':int(hC.get(s,0)), 'n_dendrometers_L2':int(dC.get(s,0)), 'n_dendrometers_LM':int(lC.get(s,0)) } for s in sites])

# ---- Civil‑time utilities (plots/inspection; arrays remain UTC) ----

def utc_index_to_local(idx_utc: pd.DatetimeIndex, tz: str = 'Europe/Zurich') -> pd.DatetimeIndex:
    """Convert a **UTC** DatetimeIndex to a **civil timezone** for plotting/inspection."""
    if getattr(idx_utc, 'tz', None) is None:
        raise ValueError('utc_index_to_local expects a tz‑aware UTC index')
    return idx_utc.tz_convert(tz)


def series_utc_to_civil(s_utc: pd.Series, tz: str = 'Europe/Zurich') -> pd.Series:
    """Return a **view** of a UTC series in civil local time (useful for human‑readable plots)."""
    if getattr(s_utc.index, 'tz', None) is None:
        raise ValueError('series_utc_to_civil expects a tz‑aware UTC index')
    return s_utc.tz_convert(tz)


def plot_series_civiltime(s_utc: pd.Series, tz: str = 'Europe/Zurich', title: str | None = None,
                          out_png: str | None = None) -> None:
    """Quick helper to plot a UTC series using **civil local time** on the x‑axis.

    If `out_png` is provided, the figure is saved; otherwise it is displayed.
    """
    s_local = series_utc_to_civil(s_utc, tz=tz)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(s_local.index, s_local.values, lw=0.8)
    ax.set_title(title or f'Series in {tz}')
    ax.set_xlabel(f'Time ({tz})'); ax.set_ylabel('Value')
    ax.grid(True, ls=':', alpha=0.6)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    else:
        plt.show()

# =============================================================
# CLI
# =============================================================

def parse_args():
    """Parse CLI args for **UTC‑only** pre‑processing with **Strategy B** globals.

    `--tz` is used only to localize naive local/LM timestamps before converting to UTC.
    Daily globals are provided as civil dates and are NOT converted to UTC timestamps.
    """
    p = argparse.ArgumentParser(description='TreeNet pre‑processing (UTC‑only locals/LM, Strategy B globals).')
    p.add_argument('--out_root', required=True)
    p.add_argument('--metadata_pickle', required=True)
    p.add_argument('--meteo_dir', required=True)
    p.add_argument('--thermo_dir', required=True)
    p.add_argument('--hygro_dir', required=True)
    p.add_argument('--dendro_l2_dir', required=True)
    p.add_argument('--dendro_lm_dir', required=True)
    p.add_argument('--train_site_ids_csv', required=True)
    p.add_argument('--test_site_ids_csv', required=False)
    p.add_argument('--years', nargs='+', type=int, required=True)
    p.add_argument('--per_year', type=str, default='true')
    p.add_argument('--tz', type=str, default='Europe/Zurich')
    p.add_argument('--require_complete_locals', type=str, default='false')
    p.add_argument('--stem_mode', type=str, default='absolute', choices=['absolute','delta'])
    p.add_argument('--input_mode', type=str, default='combinations', choices=['best','combinations','pooled'])
    p.add_argument('--target_mode', type=str, default='lm_site_median', choices=['lm_site_median'])
    p.add_argument('--max_combos_per_site', type=int, default=None)
    p.add_argument('--min_local_coverage', type=float, default=0.7)
    p.add_argument('--min_lm_series', type=int, default=1)
    p.add_argument('--overlap_days', type=int, default=10)
    p.add_argument('--allow_missing_locals', type=str, default='false')
    p.add_argument('--globals_broadcast_strategy', type=str, default='civil_map', choices=['civil_map'],
                   help='Broadcast daily globals by civil day mapping to UTC grid (Strategy B). Only option.')
    p.add_argument('--run_tests', action='store_true', help='Run built‑in Strategy B unit test and exit')
    return p.parse_args()

# =============================================================
# Small Strategy‑B unit test (DST fall‑back day)
# =============================================================

def _test_strategy_b_dst_fallback(tz_local: str = 'Europe/Zurich') -> None:
    """Minimal self‑test to verify Strategy B across a DST fall‑back day (e.g., 2022‑10‑30 in Zurich).

    We create a UTC 10‑min index spanning 2022‑10‑29 .. 2022‑10‑31 and a daily civil table with distinct
    values for the 29th and 30th. We broadcast and then **assert** that each UTC instant maps to the correct
    civil date's value, even through the repeated local hour.
    """
    # Build a UTC grid for two days around fall‑back
    idx_utc = pd.date_range(pd.Timestamp('2022-10-29 00:00:00', tz='UTC'),
                            pd.Timestamp('2022-10-31 00:00:00', tz='UTC'), freq='10min', inclusive='left')

    # Daily civil data for 29th (value=29) and 30th (value=30)
    daily = pd.DataFrame({'ts':['2022-10-29','2022-10-30'], 'g_tmean':[29.0, 30.0]})
    daily.index = pd.to_datetime(daily['ts']).normalize(); daily = daily[['g_tmean']]

    # Broadcast via Strategy B
    out = broadcast_daily_civil_to_utc_grid(daily, idx_utc, tz_local=tz_local)

    # For each UTC instant, compute its civil date and compare values
    civ_dates = idx_utc.tz_convert(tz_local).normalize()
    expected = np.where(civ_dates == pd.Timestamp('2022-10-29'), 29.0,
                        np.where(civ_dates == pd.Timestamp('2022-10-30'), 30.0, np.nan))

    # Assert correctness (allow NaN where beyond provided days)
    mask = ~np.isnan(expected)
    mismatches = np.where(np.abs(out['g_tmean'].to_numpy()[mask] - expected[mask]) > 1e-9)[0]
    if mismatches.size:
        raise AssertionError(f"Strategy B DST test failed at {mismatches.size} positions; example UTC ts: {idx_utc[mask][mismatches[0]]}")
    print('Strategy B DST fall‑back self‑test: OK')

# =============================================================
# Entry point
# =============================================================

def read_site_ids_csv(path: str) -> t.List[int]:
    """Read a CSV containing a `site_id` column and return it as a list of ints."""
    df = pd.read_csv(path)
    if 'site_id' not in df.columns:
        raise ValueError('CSV must contain site_id')
    return [int(x) for x in df['site_id'].tolist()]

if __name__ == '__main__':
    args = parse_args()
    if args.run_tests:
        _test_strategy_b_dst_fallback(tz_local=args.tz)
        raise SystemExit(0)

    per_year = (args.per_year.lower()=='true')
    require_complete_locals = (args.require_complete_locals.lower()=='true')
    site_ids_train = read_site_ids_csv(args.train_site_ids_csv)
    site_ids_test  = read_site_ids_csv(args.test_site_ids_csv) if args.test_site_ids_csv else None

    build_datasets_utc(
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
        local_tz=args.tz,
        require_complete_locals=require_complete_locals,
        stem_mode=args.stem_mode,
        input_mode=args.input_mode,
        target_mode=args.target_mode,
        max_combos_per_site=args.max_combos_per_site,
        min_local_coverage=args.min_local_coverage,
        min_lm_series=args.min_lm_series,
        overlap_days=args.overlap_days,
        allow_missing_locals=(args.allow_missing_locals.lower()=='true'),
        globals_broadcast_strategy=args.globals_broadcast_strategy,
    )
