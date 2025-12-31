# build_normalized_dataset_treenet_utc.py
# -*- coding: utf-8 -*-
"""
TreeNet → TF Pre‑processing — UTC‑Only Core + Strategy B (civil mapping) for Daily Globals
=========================================================================================

This module builds normalized **multichannel year‑long inputs** (10‑min, 11 channels) and
**hourly targets** (1‑hour, 3 channels) for machine learning, operating strictly on a **UTC
timeline** for instrument series.

**Key design choices**
----------------------
- **UTC everywhere for locals/LM**: Naive timestamps are localized to Europe/Zurich only to disambiguate
  DST, then converted to UTC. All binning/resampling/reindexing is done in **UTC**.
- **Strategy B for daily globals**: Daily civil Zurich data are mapped onto the UTC 10‑min grid by
  converting each UTC time to its civil day and reindexing the daily table by those civil‑day labels.
- **Per‑ID pairing**: Inputs and targets are paired by **dendrometer `series_id`**, using the **intersection**
  of dendrometer IDs present in both **L2** (inputs) and **LM** (targets).
- **Targets from LM at whole hours**: For each matched dendrometer ID, targets are taken from the **LM file**
  of the same ID, using **only rows at exact whole hours** (`minute==0 and second==0`).
- **Input combinations at site level**: For each dendrometer ID, we build **all combinations** of local
  **thermometer** and **hygrometer** series available at the **same site**.

Outputs
-------
- `X_train.npy`, `y_train.npy` (and `X_test.npy`, `y_test.npy`), with shapes `(n_segments, 52560, 11)` and
  `(n_segments, 8760, 3)` respectively.
- Identifiers CSVs (`train_identifiers.csv`, `test_identifiers.csv`) describing instrument IDs and windows.
- Per‑site normalizers for global channels (JSON + summary CSV).
- Diagnostics CSV summarizing segment counts and warnings.
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
# Channels: 3 locals + 7 globals + g_doy = 11
N_CHANNELS    = 11
N_TARGETS     = 3      # hourly local_T, local_RH, local_stem
LOCAL_COLS  = ['local_T', 'local_RH', 'local_stem']
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']
FREQ_10M    = '10min'
FREQ_1H     = '1h'

# =============================================================
# UTC axis helpers & indices
# =============================================================

def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove Feb 29 to keep fixed‑length arrays across years."""
    return idx[~((idx.month==2) & (idx.day==29))]


def make_multi_year_10m_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    """Create a UTC 10‑min multi‑year grid spanning min(years)..max(years) with leap day removed."""
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_10M)
    return strip_leap_days(idx)


def make_multi_year_hourly_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    """Create a UTC hourly multi‑year grid spanning min(years)..max(years) with leap day removed."""
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_1H)
    return strip_leap_days(idx)

# =============================================================
# Readers (locals/LM to UTC) and daily globals (civil dates)
# =============================================================

def discover_meteo_files(meteo_dir: str) -> dict[int, str]:
    """Return `{site_id: path}` mapping for daily global meteo CSVs.
    Filenames must contain an integer site ID (first integer found is used).
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
      - `ts` (civil date string 'YYYY-MM-DD' or datetime w/o tz)
      - `tas` (mean T), `tasmax`, `tasmin`, `rh`, `vpd`, `gh` (global radiation), `pr` (precip)

    Returns a DataFrame indexed by civil date with standardized columns:
    `g_tmean, g_tmin, g_tmax, g_rh, g_vpd, g_pr, g_rad`.
    """
    files = discover_meteo_files(meteo_dir)
    if site_id not in files:
        raise FileNotFoundError(f'No meteo CSV for site {site_id}')
    df = pd.read_csv(files[site_id])
    if 'ts' not in df.columns:
        raise ValueError('Global meteo CSV must have ts column (civil dates)')
    # Parse as date and normalize
    idx_civil = pd.to_datetime(df['ts'], utc=False).dt.normalize()
    rename = {
        'tas':'g_tmean', 'tasmax':'g_tmax', 'tasmin':'g_tmin',
        'rh':'g_rh', 'vpd':'g_vpd', 'gh':'g_rad', 'pr':'g_pr'
    }
    for src, dst in rename.items():
        if src not in df.columns:
            raise ValueError(f'Global meteo missing {src}')
        df[dst] = df[src]
    out = df.set_index(idx_civil)[['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad']].copy()
    out = out[~out.index.duplicated(keep='last')]
    return out


def _pick_value_column(df: pd.DataFrame, preferred: t.Sequence[str] = ()) -> str:
    """Pick a numeric measurement column from `df`.
    Priority order: preferred → 'value' → numeric column with max non‑null count.
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
    """Read a local instrument feather (`*series_id_<id>.ftr`) and return a **UTC** series.
    - Detect timestamp column and numeric value column (use `sensor_hint` to bias selection),
    - Localize naive timestamps to `local_tz` to resolve DST, then **convert to UTC**,
    - Sort and deduplicate exact duplicates by mean.
    """
    pat = re.compile(rf'.*series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(dir_path) if pat.match(fn)]
    if not matches:
        raise FileNotFoundError(f'Series {series_id} not found in {dir_path}')
    fp = os.path.join(dir_path, matches[0])
    df = pd.read_feather(fp)

    # Timestamp column
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
    For each UTC instant, convert to local civil time and take its **normalized date**;
    use that date to look up daily row; forward‑fill across the day.
    """
    if getattr(idx_10m_utc, 'tz', None) is None:
        raise ValueError('idx_10m_utc must be tz‑aware UTC index')
    civil_days_for_grid = idx_10m_utc.tz_convert(tz_local).normalize()
    daily_by_civilday = daily_civil.copy()
    daily_by_civilday.index = pd.to_datetime(daily_by_civilday.index).dt.normalize()
    out = daily_by_civilday.reindex(civil_days_for_grid).ffill()
    out.index = idx_10m_utc
    return out

# =============================================================
# Axis‑aware (UTC) deterministic binning for locals/LM
# =============================================================

def bin_to_10min_utc(s: pd.Series, method: str = 'resample', how: str = 'median') -> pd.Series:
    """Bin a tz‑aware series to the 10‑min UTC grid deterministically.
    - `method='floor'`: floor timestamps to 10‑min, collapse duplicates by mean.
    - `method='resample'`: resample to explicit 10‑min UTC bins, aggregate by `how`.
    """
    if s.empty:
        return s
    s = s.copy().tz_convert('UTC').sort_index()
    if method == 'floor':
        s.index = s.index.floor(FREQ_10M)
        return s[~s.index.duplicated(keep='mean')]
    elif method == 'resample':
        agg = {'mean': 'mean', 'median': 'median'}.get(how, 'median')
        return s.resample(FREQ_10M, origin='start_day', label='left').agg(agg)
    else:
        raise ValueError("method must be 'floor' or 'resample'")

# =============================================================
# Quantile scalers (globals per site and scope)
# =============================================================

def compute_quantile_scaler(v: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> t.Tuple[float,float]:
    """Return robust quantiles `(q_low, q_high)`; fallback to min/max; ensure `q_high>q_low`."""
    q1 = float(np.nanpercentile(v, q_low)); q2 = float(np.nanpercentile(v, q_high))
    if not np.isfinite(q1): q1 = float(np.nanmin(v))
    if not np.isfinite(q2): q2 = float(np.nanmax(v))
    if q2 <= q1: q2 = q1 + 1e-6
    return q1, q2


def fit_site_scalers(site_id: int, years: t.Sequence[int], meteo_dir: str,
                     per_year: bool = True) -> dict:
    """Fit **global channel** quantile scalers per site either per‑year or for **ALL**.
    Fitted on the **daily civil** globals (original daily data).
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
# LM targets helper (currently unused but kept for completeness)
# =============================================================

def to_hourly(df: pd.DataFrame, how: str='median') -> pd.DataFrame:
    """Resample a UTC frame to **hourly** by `how` (`median` or `mean`)."""
    return df.resample(FREQ_1H).agg({'value':how,'temp':how,'rh':how})

# =============================================================
# Coverage & diagnostics helpers
# =============================================================

def coverage_fraction(series: pd.Series) -> float:
    """Fraction of non‑NaN entries in `series` (safe for empty)."""
    v = series.to_numpy(); n=v.size
    return float(np.sum(~np.isnan(v))/n) if n else 0.0


def write_diag(rows: t.List[dict] | None, **kw) -> None:
    """Append a diagnostics `kw` dict to the rows list (no‑op if rows is None)."""
    if rows is not None:
        rows.append(kw)

# =============================================================
# Rolling YEAR windows
# =============================================================

def rolling_year_windows(idx_10m: pd.DatetimeIndex, overlap_days: int=10, year_days: int=365) -> t.List[t.Tuple[pd.Timestamp,pd.Timestamp]]:
    """Generate `(start, end)` for **year‑long** windows on a UTC grid with `overlap_days` overlap."""
    stride_days = year_days - overlap_days
    starts = []
    cur = idx_10m[0].normalize(); end_ts = idx_10m[-1]
    while cur + pd.Timedelta(days=year_days) <= end_ts:
        starts.append(cur); cur = cur + pd.Timedelta(days=stride_days)
    return [(s, s + pd.Timedelta(days=year_days)) for s in starts]

# =============================================================
# Build segments per site (UTC + Strategy B + LM-per-ID targets)
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
    max_combos_per_site: t.Optional[int]=None,
    min_local_coverage: float=0.7,
    min_lm_series: int=1,  # retained but not used to reject; rely on intersection
    overlap_days: int=10,
    diagnostics_rows: t.Optional[t.List[dict]]=None,
    split: str='train',
    allow_missing_locals: bool=False,
    globals_broadcast_strategy: str='civil_map',
) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[dict]]:
    """Construct **UTC year‑long** segments with inputs/targets paired by **dendrometer ID intersection**.

    See module docstring for full strategy.
    """
    if globals_broadcast_strategy != 'civil_map':
        raise ValueError("Only Strategy B ('civil_map') is supported in this module.")

    # Metadata, master grids
    metadata_df = pd.read_pickle(metadata_path)
    idx_10m = make_multi_year_10m_index_utc(years)
    idx_hour= make_multi_year_hourly_index_utc(years)

    # Globals (civil daily) → broadcast to UTC 10‑min (Strategy B)
    glb_daily = load_global_daily_civil(site_id, meteo_dir)
    glb_10m = broadcast_daily_civil_to_utc_grid(glb_daily, idx_10m, tz_local=local_tz)
    glb_10m['g_doy'] = idx_10m.tz_convert(local_tz).dayofyear

    # Discover series IDs
    thermo_ids_all = discover_series_ids(thermo_dir, 'thermometer_l1')
    hygro_ids_all  = discover_series_ids(hygro_dir,  'hygrometer_l1')
    d2_ids_all     = discover_series_ids(dendro_l2_dir, 'dendrometer_l2')
    lm_ids_all     = discover_series_ids(dendro_lm_dir, 'dendrometer_lm')

    # Intersection of dendrometers present in both L2 and LM
    dendro_ids_all = sorted(list(set(d2_ids_all) & set(lm_ids_all)))

    # Instrument IDs per site
    thermo_by_site = series_by_site(metadata_df, thermo_ids_all)
    hygro_by_site  = series_by_site(metadata_df, hygro_ids_all)

    X_list: t.List[np.ndarray] = []
    Y_list: t.List[np.ndarray] = []
    META_list: t.List[dict] = []
    segments_count = 0

    # Windows and scope key
    windows = rolling_year_windows(idx_10m, overlap_days=overlap_days, year_days=365)
    key_for = (lambda ts: ts.year) if per_year else (lambda ts: 'ALL')

    # Iterate over matched dendrometer IDs
    for d_id in dendro_ids_all:
        # Find site_id corresponding to this dendro_id
        row = metadata_df[metadata_df['series_id'] == d_id]
        if row.empty:
            write_diag(diagnostics_rows, site_id=site_id, split=split,
                       warning='missing_metadata_for_dendro_id', dendro_id=d_id)
            continue
        site_id_for_d = int(row['site_id'].iloc[0])

        # Collect all thermometers/hygrometers at this site
        t_ids_site = thermo_by_site.get(site_id_for_d, [])
        h_ids_site = hygro_by_site.get(site_id_for_d, [])
        if (not t_ids_site) or (not h_ids_site):
            write_diag(diagnostics_rows, site_id=site_id_for_d, split=split,
                       warning='no_local_T_or_RH_at_site', dendro_id=d_id)
            continue

        # Read L2 stem (same dendro id), bin to 10-min UTC; apply delta if requested
        try:
            s_ST_full = read_feather_series_utc(d_id, dendro_l2_dir, local_tz, sensor_hint='dendrometer_l2')
        except Exception as e:
            write_diag(diagnostics_rows, site_id=site_id_for_d, split=split,
                       warning='read_error_dendro_l2', dendro_id=d_id, error=repr(e))
            continue
        s_ST_full = bin_to_10min_utc(s_ST_full, method='resample', how='median').reindex(idx_10m)
        if stem_mode == 'delta':
            s_ST_full = s_ST_full.diff()

        # Read LM (targets) for this dendro id and keep only **whole-hour rows**
        try:
            df_lm = read_lm_frame_utc(d_id, dendro_lm_dir, local_tz)
        except Exception as e:
            write_diag(diagnostics_rows, site_id=site_id_for_d, split=split,
                       warning='read_error_dendro_lm', dendro_id=d_id, error=repr(e))
            continue
        df_lm_hr = df_lm[(df_lm.index.minute == 0) & (df_lm.index.second == 0)]
        df_lm_hr = df_lm_hr.rename(columns={'value': 'stem', 'temp': 'local_T', 'rh': 'local_RH'})

        # Optional pooled strategy across site instruments
        if input_mode == 'pooled':
            def pooled_series(ids, dirp, hint=None):
                arrs = []
                for sid in ids:
                    try:
                        s_ser = read_feather_series_utc(sid, dirp, local_tz, sensor_hint=hint)
                        s_ser = bin_to_10min_utc(s_ser, method='resample', how='median')
                        arrs.append(s_ser.reindex(idx_10m).to_numpy())
                    except Exception:
                        pass
                if not arrs:
                    return pd.Series(np.nan, index=idx_10m)
                A = np.vstack(arrs); med = np.nanmedian(A, axis=0)
                return pd.Series(med, index=idx_10m)
            s_T_pooled = pooled_series(t_ids_site, thermo_dir, 'thermometer_l1')
            s_RH_pooled= pooled_series(h_ids_site,  hygro_dir,  'hygrometer_l1')

        # Per-window processing
        for (ws, we) in windows:
            scope_key = key_for(ws)

            # Globals slice + normalization
            glb_win = glb_10m[(glb_10m.index>=ws) & (glb_10m.index<we)]
            if glb_win.shape[0] != SEQ_LEN_10MIN:
                continue
            glb_n = { col: normalize_array(glb_win[col].to_numpy(),
                                           scalers[scope_key][col]['q_low'],
                                           scalers[scope_key][col]['q_high'])
                      for col in GLOBAL_COLS }

            # Hourly index for the window
            start_h = ws; end_h = we - pd.Timedelta(minutes=10)
            idx_h = pd.date_range(start_h.floor('h'), end_h.floor('h'), freq=FREQ_1H)
            idx_h = strip_leap_days(idx_h)

            # Targets from LM hourly rows for this **same dendro id**
            dfh = df_lm_hr[(df_lm_hr.index>=ws) & (df_lm_hr.index<=end_h)].reindex(idx_h)

            # Ensure local stem scaler exists
            if 'local_stem' not in scalers[scope_key]:
                q1, q2 = compute_quantile_scaler(s_ST_full.to_numpy())
                scalers[scope_key]['local_stem'] = {'q_low': q1, 'q_high': q2}

            # Build T/RH combinations (or pooled)
            if input_mode == 'pooled':
                combos = [(None, None)]
                preloaded: dict[int, pd.Series] = {}
            elif input_mode == 'best':
                best = None; best_score = -1.0; preloaded = {}
                for t_id0 in t_ids_site:
                    try:
                        sT0 = bin_to_10min_utc(read_feather_series_utc(t_id0, thermo_dir, local_tz, sensor_hint='thermometer_l1'), method='resample', how='median').reindex(idx_10m)
                    except Exception:
                        continue
                    for h_id0 in h_ids_site:
                        try:
                            sRH0 = bin_to_10min_utc(read_feather_series_utc(h_id0, hygro_dir, local_tz, sensor_hint='hygrometer_l1'), method='resample', how='median').reindex(idx_10m)
                        except Exception:
                            continue
                        sT_w  = sT0[(sT0.index>=ws)&(sT0.index<we)]
                        sRH_w = sRH0[(sRH0.index>=ws)&(sRH0.index<we)]
                        sST_w = s_ST_full[(s_ST_full.index>=ws)&(s_ST_full.index<we)]
                        score = min(coverage_fraction(sT_w), coverage_fraction(sRH_w), coverage_fraction(sST_w))
                        if score > best_score:
                            best_score = score; best = (t_id0, h_id0, sT0, sRH0)
                combos = [best[:2]] if best is not None else []
                if best is not None:
                    preloaded = {best[0]: best[2], best[1]: best[3]}
            else:
                combos = [(t_id, h_id) for t_id in t_ids_site for h_id in h_ids_site]
                preloaded = {}

            if max_combos_per_site is not None and len(combos) > max_combos_per_site:
                combos = combos[:max_combos_per_site]

            for (t_id, h_id) in combos:
                # Load or reuse T/RH series
                if input_mode == 'pooled':
                    s_T_full = s_T_pooled
                    s_RH_full= s_RH_pooled
                else:
                    if (input_mode == 'best') and (t_id in preloaded):
                        s_T_full = preloaded[t_id]
                    else:
                        try:
                            s_T_full = read_feather_series_utc(t_id, thermo_dir, local_tz, sensor_hint='thermometer_l1')
                            s_T_full = bin_to_10min_utc(s_T_full, method='resample', how='median').reindex(idx_10m)
                        except Exception as e:
                            write_diag(diagnostics_rows, site_id=site_id_for_d, split=split,
                                       warning='read_error_thermo', thermo_id=t_id, dendro_id=d_id, error=repr(e))
                            continue
                    if (input_mode == 'best') and (h_id in preloaded):
                        s_RH_full = preloaded[h_id]
                    else:
                        try:
                            s_RH_full = read_feather_series_utc(h_id, hygro_dir, local_tz, sensor_hint='hygrometer_l1')
                            s_RH_full = bin_to_10min_utc(s_RH_full, method='resample', how='median').reindex(idx_10m)
                        except Exception as e:
                            write_diag(diagnostics_rows, site_id=site_id_for_d, split=split,
                                       warning='read_error_hygro', hygro_id=h_id, dendro_id=d_id, error=repr(e))
                            continue

                # Window slices
                sT_w  = s_T_full[(s_T_full.index>=ws)&(s_T_full.index<we)]
                sRH_w = s_RH_full[(s_RH_full.index>=ws)&(s_RH_full.index<we)]
                sST_w = s_ST_full[(s_ST_full.index>=ws)&(s_ST_full.index<we)]

                # Coverage gate
                cov_T = coverage_fraction(sT_w)
                cov_RH= coverage_fraction(sRH_w)
                cov_ST= coverage_fraction(sST_w)
                if (cov_T < min_local_coverage) or (cov_RH < min_local_coverage) or (cov_ST < min_local_coverage):
                    write_diag(diagnostics_rows, site_id=site_id_for_d, split=split,
                               warning='low_local_coverage_combo', thermo_id=t_id, hygro_id=h_id, dendro_id=d_id,
                               window_start=str(ws), cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                    continue

                # Ensure local scalers exist (lazy per scope)
                for name, sfull in [('local_T', s_T_full), ('local_RH', s_RH_full)]:
                    if name not in scalers[scope_key]:
                        q1, q2 = compute_quantile_scaler(sfull.to_numpy())
                        scalers[scope_key][name] = {'q_low': q1, 'q_high': q2}

                # Normalize locals in the window
                T_n    = normalize_array(sT_w.to_numpy(),  scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high'])
                RH_n   = normalize_array(sRH_w.to_numpy(), scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high'])
                STEM_n = normalize_array(sST_w.to_numpy(), scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high'])

                # Compose X (11 channels @ 10-min)
                X_seg = np.column_stack([
                    T_n, RH_n, STEM_n,
                    glb_n['g_tmean'], glb_n['g_tmin'], glb_n['g_tmax'],
                    glb_n['g_rh'], glb_n['g_vpd'], glb_n['g_pr'], glb_n['g_rad'], glb_n['g_doy'],
                ]).astype(np.float32)
                if X_seg.shape[0] != SEQ_LEN_10MIN:
                    continue

                # Compose y (3 channels @ 1-hour) from LM per same ID
                q1t,q2t = scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high']
                q1r,q2r = scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high']
                q1s,q2s = scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high']
                temp_hr = normalize_array(dfh['local_T'].to_numpy(), q1t, q2t)
                rh_hr   = normalize_array(dfh['local_RH'].to_numpy(), q1r, q2r)
                stem_hr = normalize_array(dfh['stem'].to_numpy(),    q1s, q2s)
                y_seg   = np.stack([temp_hr, rh_hr, stem_hr], axis=-1).astype(np.float32)
                if y_seg.shape[0] != HOUR_STEPS:
                    continue

                write_diag(
                    diagnostics_rows,
                    site_id=site_id_for_d, split=split,
                    window_start=str(ws), window_end=str(we),
                    thermo_id=int(t_id) if t_id is not None else -1,
                    hygro_id=int(h_id)  if h_id is not None else -1,
                    dendro_id=int(d_id),
                    cov_T=float(np.mean(~np.isnan(T_n))),
                    cov_RH=float(np.mean(~np.isnan(RH_n))),
                    cov_ST=float(np.mean(~np.isnan(STEM_n))),
                    targets_source='LM_same_id'
                )

                X_list.append(X_seg)
                Y_list.append(y_seg)
                META_list.append({
                    'site_id': site_id_for_d,
                    'years_scope': f"{min(years)}-{max(years)}",
                    'window_start': str(ws), 'window_end': str(we),
                    'input_mode': input_mode,
                    'thermometer_id': int(t_id) if t_id is not None else -1,
                    'hygrometer_id': int(h_id)  if h_id is not None else -1,
                    'dendrometer_id': int(d_id),
                })
                segments_count += 1

        write_diag(diagnostics_rows, site_id=site_id_for_d, split=split, windows=1, segments=segments_count)

    return X_list, Y_list, META_list

# =============================================================
# Build datasets (UTC + Strategy B + LM-per-ID targets)
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
    max_combos_per_site: t.Optional[int] = None,
    min_local_coverage: float = 0.7,
    min_lm_series: int = 1,
    overlap_days: int = 10,
    allow_missing_locals: bool = False,
    globals_broadcast_strategy: str = 'civil_map',
) -> None:
    """Top‑level orchestration to build **UTC** train/test arrays using Per‑ID LM targets.
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
            stem_mode=stem_mode, input_mode=input_mode,
            max_combos_per_site=max_combos_per_site, min_local_coverage=min_local_coverage,
            min_lm_series=min_lm_series, overlap_days=overlap_days,
            diagnostics_rows=diagnostics_rows, split='train',
            allow_missing_locals=allow_missing_locals, globals_broadcast_strategy=globals_broadcast_strategy,
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
                stem_mode=stem_mode, input_mode=input_mode,
                max_combos_per_site=max_combos_per_site, min_local_coverage=min_local_coverage,
                min_lm_series=min_lm_series, overlap_days=overlap_days,
                diagnostics_rows=diagnostics_rows, split='test',
                allow_missing_locals=allow_missing_locals, globals_broadcast_strategy=globals_broadcast_strategy,
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
    """Quick helper to plot a UTC series using **civil local time** on the x‑axis."""
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
    """Parse CLI args for UTC‑only pre‑processing with Strategy B globals and LM-per-ID targets.

    `--tz` is used only to localize naive local/LM timestamps before converting to UTC.
    Daily globals are provided as civil dates and are NOT converted to UTC timestamps.
    """
    p = argparse.ArgumentParser(description='TreeNet pre‑processing (UTC‑only locals/LM, Strategy B globals, LM-per-ID targets).')
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
    """Minimal self‑test to verify Strategy B across a DST fall‑back day (e.g., 2022‑10‑30 in Zurich)."""
    idx_utc = pd.date_range(pd.Timestamp('2022-10-29 00:00:00', tz='UTC'),
                            pd.Timestamp('2022-10-31 00:00:00', tz='UTC'), freq=FREQ_10M, inclusive='left')
    daily = pd.DataFrame({'ts':['2022-10-29','2022-10-30'], 'g_tmean':[29.0, 30.0]})
    daily.index = pd.to_datetime(daily['ts']).dt.normalize(); daily = daily[['g_tmean']]
    out = broadcast_daily_civil_to_utc_grid(daily, idx_utc, tz_local=tz_local)
    civ_dates = idx_utc.tz_convert(tz_local).normalize()
    expected = np.where(civ_dates == pd.Timestamp('2022-10-29'), 29.0,
                        np.where(civ_dates == pd.Timestamp('2022-10-30'), 30.0, np.nan))
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
        max_combos_per_site=args.max_combos_per_site,
        min_local_coverage=args.min_local_coverage,
        min_lm_series=args.min_lm_series,
        overlap_days=args.overlap_days,
        allow_missing_locals=(args.allow_missing_locals.lower()=='true'),
        globals_broadcast_strategy=args.globals_broadcast_strategy,
    )
