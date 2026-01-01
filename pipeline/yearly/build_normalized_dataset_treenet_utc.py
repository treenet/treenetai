# build_normalized_dataset_treenet_utc.py
# -*- coding: utf-8 -*-
"""
TreeNet → TF Pre‑processing — UTC‑Only + Strategy B (civil mapping) for Daily Globals

Build normalized **inputs** (10‑min, 11 channels) and **targets** (1‑hour, 3 channels),
pairing **locals** with **LM targets** by **dendrometer series_id** at each site.

Features
--------
- UTC everywhere for locals/LM; Strategy‑B broadcasting for daily globals.
- Deterministic binning to 10‑min via resample (robust to jitter).
- Calendar‑year windows (e.g., 2019-01-01 → 2020-01-01).
- `--input_mode best`: auto‑select best (T,RH) per window.
- `--allow_missing_locals true`: substitute globals for missing locals (T→g_tmean, RH→g_rh) **for inputs only**.
- Stem (L2) must meet coverage; no substitution for stem.

Outputs
-------
- X/y arrays: (n_segments, 52560, 11) and (n_segments, 8760, 3).
- Identifiers CSVs and diagnostics CSVs.
- Per‑site normalizer (global channels) JSON + summary CSV.
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
HOUR_STEPS    = 8760
STRIDE_PER_HR = 6
N_CHANNELS    = 11  # 3 locals + 7 globals + g_doy
N_TARGETS     = 3   # hourly local_T, local_RH, local_stem
LOCAL_COLS  = ['local_T', 'local_RH', 'local_stem']
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']
FREQ_10M    = '10min'
FREQ_1H     = '1h'

# =============================================================
# UTC grid helpers
# =============================================================

def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx[~((idx.month==2) & (idx.day==29))]


def make_multi_year_10m_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_10M)
    return strip_leap_days(idx)


def make_multi_year_hourly_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_1H)
    return strip_leap_days(idx)

# =============================================================
# Readers and globals
# =============================================================

def discover_meteo_files(meteo_dir: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for fn in os.listdir(meteo_dir):
        if fn.endswith('.csv'):
            m = re.findall(r'\d+', fn)
            if m:
                mapping[int(m[0])] = os.path.join(meteo_dir, fn)
    return mapping


def load_global_daily_civil(site_id: int, meteo_dir: str) -> pd.DataFrame:
    files = discover_meteo_files(meteo_dir)
    if site_id not in files:
        raise FileNotFoundError(f'No meteo CSV for site {site_id}')
    df = pd.read_csv(files[site_id])
    if 'ts' not in df.columns:
        raise ValueError('Global meteo CSV must have ts column (civil dates)')
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
    pat = re.compile(rf'.*series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(dir_path) if pat.match(fn)]
    if not matches:
        raise FileNotFoundError(f'Series {series_id} not found in {dir_path}')
    fp = os.path.join(dir_path, matches[0])
    df = pd.read_feather(fp)

    if ts_col not in df.columns:
        for alt in ('timestamp','time','date_time','datetime'):
            if alt in df.columns:
                ts_col = alt; break
    if ts_col not in df.columns:
        raise ValueError(f'Feather {fp} missing timestamp column')

    if sensor_hint == 'thermometer_l1':
        preferred = ('temp','temperature','temperature_mean','value')
    elif sensor_hint == 'hygrometer_l1':
        preferred = ('rh','relhum','rh_mean','relative_humidity','value')
    elif sensor_hint == 'dendrometer_l2':
        preferred = ('value','radius','rad','l2','stem_radius_change')
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



def read_lm_hourly_frame_utc(series_id: int, lm_dir: str, local_tz: str) -> pd.DataFrame:
    """
    Simple & fast LM reader (RAW only) → hourly UTC by selecting exact local HH:00 rows.

    Behavior:
    - Read RAW LM (10-min): 'value' (stem) 10-min grid; 'temp' and 'rh' present at hourly marks.
    - Convert timestamps to local time (e.g., Europe/Zurich) and **keep only rows at HH:00:00**.
    - Reindex to a continuous local hourly index from span start to end (to fill missing hours with NaN).
    - Convert the final local hourly index to UTC and rename columns to canonical ['stem','local_T','local_RH'].
    """
    import os, re
    import numpy as np
    import pandas as pd

    # RAW file only
    pat_raw = re.compile(rf'dendrometer_lm_series_id_{series_id}\.ftr$')
    matches_raw = [fn for fn in os.listdir(lm_dir) if pat_raw.match(fn)]
    if not matches_raw:
        raise FileNotFoundError(f'LM RAW file not found for series {series_id} in {lm_dir}')

    df = pd.read_feather(os.path.join(lm_dir, matches_raw[0]))

    # Timestamp column
    ts_col = 'ts' if 'ts' in df.columns else ('timestamp' if 'timestamp' in df.columns else None)
    if ts_col is None:
        raise ValueError('LM raw file missing timestamp column')

    # Localize to site timezone
    ts_local = pd.to_datetime(df[ts_col], utc=False)
    ts_local = ts_local.dt.tz_localize(local_tz) if getattr(ts_local.dt, 'tz', None) is None else ts_local.dt.tz_convert(local_tz)

    df_local = df.copy()
    df_local.index = ts_local

    # Ensure columns exist
    for col in ['value', 'temp', 'rh']:
        if col not in df_local.columns:
            df_local[col] = np.nan

    # Keep only exact local HH:00 rows
    hh00_mask = (df_local.index.minute == 0) & (df_local.index.second == 0)
    df_hourly_local = df_local.loc[hh00_mask, ['value', 'temp', 'rh']].sort_index()

    if df_hourly_local.empty:
        # Return empty hourly frame; builder will handle reindex/diagnostics
        return pd.DataFrame(columns=['stem','local_T','local_RH'])

    # Continuous local hourly index
    start_local = df_hourly_local.index.min()
    end_local   = df_hourly_local.index.max()
    idx_hour_local = pd.date_range(start_local, end_local, freq='1h')

    # Reindex to continuous local hours
    df_hourly_local = df_hourly_local.reindex(idx_hour_local)

    # Convert to UTC
    idx_hour_utc = df_hourly_local.index.tz_convert('UTC')
    out = pd.DataFrame({
        'stem':    df_hourly_local['value'].to_numpy(),
        'local_T': df_hourly_local['temp'].to_numpy(),
        'local_RH':df_hourly_local['rh'].to_numpy(),
    }, index=idx_hour_utc).sort_index()

    return out


def broadcast_daily_civil_to_utc_grid(daily_civil: pd.DataFrame,
                                      idx_10m_utc: pd.DatetimeIndex,
                                      tz_local: str = "Europe/Zurich") -> pd.DataFrame:
    if getattr(idx_10m_utc, 'tz', None) is None:
        raise ValueError('idx_10m_utc must be tz‑aware UTC index')
    civil_days_for_grid = idx_10m_utc.tz_convert(tz_local).normalize()
    daily_by_civilday = daily_civil.copy()
    daily_by_civilday.index = pd.DatetimeIndex(pd.to_datetime(daily_by_civilday.index)).normalize()
    out = daily_by_civilday.reindex(civil_days_for_grid).ffill()
    out.index = idx_10m_utc
    return out


def bin_to_10min_utc(s: pd.Series, method: str = 'resample', how: str = 'median') -> pd.Series:
    if s.empty:
        return s
    s = s.copy().tz_convert('UTC').sort_index()
    if method == 'floor':
        s.index = s.index.floor(FREQ_10M)
        return s[~s.index.duplicated(keep='mean')]
    agg = {'mean': 'mean', 'median': 'median'}.get(how, 'median')
    return s.resample(FREQ_10M, origin='start_day', label='left').agg(agg)

# =============================================================
# Normalizers (globals)
# =============================================================

def compute_quantile_scaler(v: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> t.Tuple[float,float]:
    q1 = float(np.nanpercentile(v, q_low)); q2 = float(np.nanpercentile(v, q_high))
    if not np.isfinite(q1): q1 = float(np.nanmin(v))
    if not np.isfinite(q2): q2 = float(np.nanmax(v))
    if q2 <= q1: q2 = q1 + 1e-6
    return q1, q2


def fit_site_scalers(site_id: int, years: t.Sequence[int], meteo_dir: str,
                     per_year: bool = True) -> dict:
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
    out = (arr - q1) / (q2 - q1)
    return np.clip(out, clip_low, clip_high)

# =============================================================
# Discovery by directory & metadata
# =============================================================

def discover_series_ids(dir_path: str, pattern_prefix: str) -> t.Set[int]:
    ids: set[int] = set()
    regex = re.compile(rf'{re.escape(pattern_prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m: ids.add(int(m.group(1)))
    return ids


def series_by_site(metadata_df: pd.DataFrame, series_ids: t.Set[int]) -> dict[int, t.List[int]]:
    df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
    out: dict[int,t.List[int]] = {}
    for site, g in df.groupby('site_id'):
        out[int(site)] = [int(x) for x in g['series_id'].tolist()]
    return out


def get_site_instrument_ids_by_metadata(metadata_df: pd.DataFrame, site_id: int) -> tuple[list[int], list[int], list[int]]:
    rows = metadata_df[metadata_df['site_id'] == site_id].copy()
    if rows.empty:
        return [], [], []
    var = rows['variable_name'].astype(str).str.strip().str.lower()
    thermo_mask = var.str.contains('air temperature', na=False) | var.str.contains('temperature', na=False) | var.str.contains('temp', na=False)
    hygro_mask  = var.str.contains('relative humidity', na=False) | var.str.contains('humidity', na=False) | var.str.contains('rh', na=False)
    dendro_mask = var.str.contains('tree stem radius change', na=False) | var.str.contains('stem radius', na=False) | var.str.contains('radius', na=False) | var.str.contains('dendrometer', na=False)
    thermo_ids = rows.loc[thermo_mask, 'series_id'].astype(int).tolist()
    hygro_ids  = rows.loc[hygro_mask,  'series_id'].astype(int).tolist()
    dendro_ids = rows.loc[dendro_mask, 'series_id'].astype(int).tolist()
    return thermo_ids, hygro_ids, dendro_ids

# =============================================================
# Diagnostics helpers
# =============================================================

def coverage_fraction(series: pd.Series) -> float:
    v = series.to_numpy(); n=v.size
    return float(np.sum(~np.isnan(v))/n) if n else 0.0


def write_diag(rows: t.List[dict] | None, **kw) -> None:
    if rows is not None:
        rows.append(kw)

# =============================================================
# Calendar-year windows
# =============================================================

def rolling_year_windows(idx_10m: pd.DatetimeIndex,
                         overlap_days: int = 10,   # unused here
                         year_days: int = 365) -> t.List[t.Tuple[pd.Timestamp,pd.Timestamp]]:
    if len(idx_10m) == 0:
        return []
    step = (idx_10m[1] - idx_10m[0]) if len(idx_10m) > 1 else pd.Timedelta(minutes=10)
    span_start = idx_10m[0].normalize()
    span_end_excl = idx_10m[-1] + step
    years_in_index = sorted(set(idx_10m.year))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for y in years_in_index:
        ws = pd.Timestamp(f"{y}-01-01 00:00:00", tz='UTC')
        we = pd.Timestamp(f"{y+1}-01-01 00:00:00", tz='UTC')
        if ws >= span_start and we <= span_end_excl:
            windows.append((ws, we))
    return windows

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
    min_lm_series: int=1,
    overlap_days: int=10,
    diagnostics_rows: t.Optional[t.List[dict]]=None,
    split: str='train',
    allow_missing_locals: bool=False,
    globals_broadcast_strategy: str='civil_map',
) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[dict]]:
    if globals_broadcast_strategy != 'civil_map':
        raise ValueError("Only Strategy B ('civil_map') is supported in this module.")

    metadata_df = pd.read_pickle(metadata_path)
    idx_10m = make_multi_year_10m_index_utc(years)
    idx_hour= make_multi_year_hourly_index_utc(years)

    glb_daily = load_global_daily_civil(site_id, meteo_dir)
    glb_10m = broadcast_daily_civil_to_utc_grid(glb_daily, idx_10m, tz_local=local_tz)
    glb_10m['g_doy'] = idx_10m.tz_convert(local_tz).dayofyear

    thermo_ids_site, hygro_ids_site, dendro_l2_ids_site = get_site_instrument_ids_by_metadata(metadata_df, site_id)
    thermo_ids_by_site = series_by_site(metadata_df, discover_series_ids(thermo_dir, 'thermometer_l1'))
    hygro_ids_by_site  = series_by_site(metadata_df, discover_series_ids(hygro_dir,  'hygrometer_l1'))
    d2_ids_by_site     = series_by_site(metadata_df, discover_series_ids(dendro_l2_dir, 'dendrometer_l2'))

    thermo_ids_site = sorted(set(thermo_ids_site) | set(thermo_ids_by_site.get(site_id, [])))
    hygro_ids_site  = sorted(set(hygro_ids_site)  | set(hygro_ids_by_site.get(site_id, [])))
    dendro_l2_ids_site = sorted(set(dendro_l2_ids_site) | set(d2_ids_by_site.get(site_id, [])))

    if (not thermo_ids_site) or (not hygro_ids_site) or (not dendro_l2_ids_site):
        write_diag(diagnostics_rows, site_id=site_id, split=split, warning='missing_instruments_at_site',
                   detail='Require at least one T, one RH, and one L2 dendrometer')
        return [], [], []

    lm_ids_hourly = set(discover_series_ids(dendro_lm_dir, 'dendrometer_lm_hourly'))
    lm_ids_raw    = set(discover_series_ids(dendro_lm_dir, 'dendrometer_lm'))
    lm_ids_all    = lm_ids_hourly | lm_ids_raw
    dendro_ids_all = [d for d in dendro_l2_ids_site if d in lm_ids_all]
    if not dendro_ids_all:
        write_diag(diagnostics_rows, site_id=site_id, split=split, warning='no_dendro_id_intersection_for_site',
                   detail='No L2∩LM dendrometers for this site')
        return [], [], []

    if diagnostics_rows is not None:
        for d_id in dendro_l2_ids_site:
            diagnostics_rows.append({
                'site_id': site_id, 'split': split, 'debug_type': 'site_inspection',
                'thermo_ids_count': len(thermo_ids_site), 'hygro_ids_count': len(hygro_ids_site),
                'dendro_l2_ids_count': len(dendro_l2_ids_site), 'dendro_id': int(d_id),
                'lm_available': bool(d_id in lm_ids_all),
            })

    windows = rolling_year_windows(idx_10m, overlap_days=overlap_days, year_days=365)
    key_for = (lambda ts: ts.year) if per_year else (lambda ts: 'ALL')

    X_list: t.List[np.ndarray] = []
    Y_list: t.List[np.ndarray] = []
    META_list: t.List[dict] = []

    for d_id in dendro_ids_all:
        scope_ids_T = thermo_ids_site[:]
        scope_ids_RH= hygro_ids_site[:]

        try:
            s_ST_full = read_feather_series_utc(d_id, dendro_l2_dir, local_tz, sensor_hint='dendrometer_l2')
        except Exception as e:
            write_diag(diagnostics_rows, site_id=site_id, split=split, warning='read_error_dendro_l2', dendro_id=d_id, error=repr(e))
            continue
        s_ST_full = bin_to_10min_utc(s_ST_full, method='resample', how='median').reindex(idx_10m)
        if stem_mode == 'delta':
            s_ST_full = s_ST_full.diff()

        try:
            df_lm_hr = read_lm_hourly_frame_utc(d_id, dendro_lm_dir, local_tz)
        except Exception as e:
            write_diag(diagnostics_rows, site_id=site_id, split=split, warning='read_error_dendro_lm_hourly', dendro_id=d_id, error=repr(e))
            continue

        for (ws, we) in windows:
            scope_key = key_for(ws)
            glb_win = glb_10m[(glb_10m.index>=ws) & (glb_10m.index<we)]
            if glb_win.shape[0] != SEQ_LEN_10MIN:
                continue

            start_h = ws; end_h = we - pd.Timedelta(minutes=10)
            idx_h = pd.date_range(start_h.floor('h'), end_h.floor('h'), freq=FREQ_1H)
            idx_h = strip_leap_days(idx_h)
            dfh = df_lm_hr[(df_lm_hr.index>=ws) & (df_lm_hr.index<=end_h)].reindex(idx_h)
            if dfh[['stem','local_T','local_RH']].isna().all().all():
                write_diag(diagnostics_rows, site_id=site_id, split=split, warning='lm_hourly_targets_all_nan_after_reindex', dendro_id=int(d_id), window_start=str(ws))
                continue

            glb_n = {
                col: normalize_array(glb_win[col].to_numpy(),
                                     scalers[scope_key][col]['q_low'],
                                     scalers[scope_key][col]['q_high'])
                for col in GLOBAL_COLS
            }

            if 'local_stem' not in scalers[scope_key]:
                q1, q2 = compute_quantile_scaler(s_ST_full.to_numpy())
                scalers[scope_key]['local_stem'] = {'q_low': q1, 'q_high': q2}

            # ----- Input selection: best or combinations -----
            preloaded: dict[int, pd.Series] = {}
            combos: list[tuple[int,int]] = []

            if input_mode == 'best':
                best = None
                best_score = -1.0
                for t_id0 in scope_ids_T:
                    try:
                        sT0 = bin_to_10min_utc(read_feather_series_utc(t_id0, thermo_dir, local_tz, sensor_hint='thermometer_l1'),
                                               method='resample', how='median').reindex(idx_10m)
                    except Exception:
                        continue
                    for h_id0 in scope_ids_RH:
                        try:
                            sRH0 = bin_to_10min_utc(read_feather_series_utc(h_id0, hygro_dir, local_tz, sensor_hint='hygrometer_l1'),
                                                    method='resample', how='median').reindex(idx_10m)
                        except Exception:
                            continue

                        sT_w  = sT0[(sT0.index>=ws)&(sT0.index<we)]
                        sRH_w = sRH0[(sRH0.index>=ws)&(sRH0.index<we)]
                        sST_w = s_ST_full[(s_ST_full.index>=ws)&(s_ST_full.index<we)]

                        covT, covRH, covST = coverage_fraction(sT_w), coverage_fraction(sRH_w), coverage_fraction(sST_w)

                        if not allow_missing_locals:
                            # STRICT: all three locals must meet coverage
                            if (covT < min_local_coverage) or (covRH < min_local_coverage) or (covST < min_local_coverage):
                                continue
                        else:
                            # SUBSTITUTION: stem must meet coverage; T/RH may be substituted
                            if covST < min_local_coverage:
                                continue

                        score = min(covT, covRH, covST)
                        if score > best_score:
                            best_score = score
                            best = (t_id0, h_id0, sT0, sRH0)

                if best is None:
                    write_diag(
                        diagnostics_rows,
                        site_id=site_id, split=split,
                        warning=('no_best_combo_in_window_strict' if not allow_missing_locals else 'no_best_combo_in_window'),
                        window_start=str(ws), min_local_coverage=min_local_coverage
                    )
                    continue

                combos = [(best[0], best[1])]
                preloaded = {best[0]: best[2], best[1]: best[3]}
            else:
                combos = [(t_id, h_id) for t_id in scope_ids_T for h_id in scope_ids_RH]
                if max_combos_per_site is not None and len(combos) > max_combos_per_site:
                    combos = combos[:max_combos_per_site]

            # ----- Build segments -----
            for (t_id, h_id) in combos:
                if input_mode == 'best' and (t_id in preloaded):
                    s_T_full = preloaded[t_id]
                else:
                    try:
                        s_T_full = read_feather_series_utc(t_id, thermo_dir, local_tz, sensor_hint='thermometer_l1')
                        s_T_full = bin_to_10min_utc(s_T_full, method='resample', how='median').reindex(idx_10m)
                    except Exception as e:
                        write_diag(diagnostics_rows, site_id=site_id, split=split, warning='read_error_thermo', thermo_id=t_id, dendro_id=d_id, error=repr(e))
                        continue
                if input_mode == 'best' and (h_id in preloaded):
                    s_RH_full = preloaded[h_id]
                else:
                    try:
                        s_RH_full = read_feather_series_utc(h_id, hygro_dir, local_tz, sensor_hint='hygrometer_l1')
                        s_RH_full = bin_to_10min_utc(s_RH_full, method='resample', how='median').reindex(idx_10m)
                    except Exception as e:
                        write_diag(diagnostics_rows, site_id=site_id, split=split, warning='read_error_hygro', hygro_id=h_id, dendro_id=d_id, error=repr(e))
                        continue

                sT_w  = s_T_full[(s_T_full.index>=ws)&(s_T_full.index<we)]
                sRH_w = s_RH_full[(s_RH_full.index>=ws)&(s_RH_full.index<we)]
                sST_w = s_ST_full[(s_ST_full.index>=ws)&(s_ST_full.index<we)]

                cov_T  = coverage_fraction(sT_w)
                cov_RH = coverage_fraction(sRH_w)
                cov_ST = coverage_fraction(sST_w)

                used_global_T  = allow_missing_locals and (cov_T  < min_local_coverage)
                used_global_RH = allow_missing_locals and (cov_RH < min_local_coverage)

                # Stem must meet coverage regardless of policy
                if cov_ST < min_local_coverage:
                    write_diag(diagnostics_rows, site_id=site_id, split=split,
                               warning='low_local_coverage_combo', thermo_id=t_id, hygro_id=h_id, dendro_id=d_id,
                               window_start=str(ws), cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                    continue

                # In strict mode, T and RH must also meet coverage
                if not allow_missing_locals and (cov_T < min_local_coverage or cov_RH < min_local_coverage):
                    write_diag(diagnostics_rows, site_id=site_id, split=split,
                               warning='low_local_coverage_combo', thermo_id=t_id, hygro_id=h_id, dendro_id=d_id,
                               window_start=str(ws), cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                    continue

                for name, sfull in [('local_T', s_T_full), ('local_RH', s_RH_full)]:
                    if name not in scalers[scope_key]:
                        q1, q2 = compute_quantile_scaler(sfull.to_numpy())
                        scalers[scope_key][name] = {'q_low': q1, 'q_high': q2}

                T_n    = normalize_array(sT_w.to_numpy(),  scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high'])
                RH_n   = normalize_array(sRH_w.to_numpy(), scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high'])
                STEM_n = normalize_array(sST_w.to_numpy(), scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high'])

                if used_global_T:
                    T_n = normalize_array(glb_win['g_tmean'].to_numpy(),
                                          scalers[scope_key]['local_T']['q_low'], scalers[scope_key]['local_T']['q_high'])
                if used_global_RH:
                    RH_n = normalize_array(glb_win['g_rh'].to_numpy(),
                                           scalers[scope_key]['local_RH']['q_low'], scalers[scope_key]['local_RH']['q_high'])

                X_seg = np.column_stack([
                    T_n, RH_n, STEM_n,
                    glb_n['g_tmean'], glb_n['g_tmin'], glb_n['g_tmax'],
                    glb_n['g_rh'], glb_n['g_vpd'], glb_n['g_pr'], glb_n['g_rad'], glb_n['g_doy'],
                ]).astype(np.float32)
                if X_seg.shape[0] != SEQ_LEN_10MIN:
                    continue

                q1t,q2t = scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high']
                q1r,q2r = scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high']
                q1s,q2s = scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high']
                temp_hr = normalize_array(dfh['local_T'].to_numpy(), q1t, q2t)
                rh_hr   = normalize_array(dfh['local_RH'].to_numpy(), q1r, q2r)
                stem_hr = normalize_array(dfh['stem'].to_numpy(),    q1s, q2s)
                y_seg   = np.stack([temp_hr, rh_hr, stem_hr], axis=-1).astype(np.float32)
                if np.isnan(y_seg).all():
                    write_diag(diagnostics_rows, site_id=site_id, split=split,
                               warning='lm_targets_all_nan_in_window', dendro_id=int(d_id), window_start=str(ws))
                    continue
                if y_seg.shape[0] != HOUR_STEPS:
                    continue

                write_diag(diagnostics_rows, site_id=site_id, split=split,
                           window_start=str(ws), window_end=str(we),
                           thermo_id=int(t_id), hygro_id=int(h_id), dendro_id=int(d_id),
                           cov_T=float(np.mean(~np.isnan(T_n))), cov_RH=float(np.mean(~np.isnan(RH_n))),
                           cov_ST=float(np.mean(~np.isnan(STEM_n))),
                           used_global_T=bool(used_global_T), used_global_RH=bool(used_global_RH),
                           targets_source='LM_same_id')

                X_list.append(X_seg)
                Y_list.append(y_seg)
                META_list.append({
                    'site_id': site_id,
                    'years_scope': f"{min(years)}-{max(years)}",
                    'window_start': str(ws), 'window_end': str(we),
                    'input_mode': input_mode,
                    'thermometer_id': int(t_id), 'hygrometer_id': int(h_id), 'dendrometer_id': int(d_id),
                })

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
    if globals_broadcast_strategy != 'civil_map':
        raise ValueError("Only Strategy B ('civil_map') is supported in this module.")

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)

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

    suffix = '365d'
    x_train_fn = os.path.join(out_root, f'X_train_{suffix}.npy')
    y_train_fn = os.path.join(out_root, f'y_train_{suffix}.npy')
    sid_train_fn = os.path.join(out_root, f'site_ids_train_{suffix}.npy')
    ids_train_fn = os.path.join(out_root, f'train_identifiers_{suffix}.csv')
    np.save(x_train_fn, X_train)
    np.save(y_train_fn, y_train)
    np.save(sid_train_fn, SID_train)
    pd.DataFrame(meta_rows).to_csv(ids_train_fn, index=False)
    print(f"Saved TRAIN arrays: {x_train_fn} {X_train.shape}, {y_train_fn} {y_train.shape}, {sid_train_fn} {SID_train.shape}")

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

        suffix = '365d'
        x_test_fn = os.path.join(out_root, f'X_test_{suffix}.npy')
        y_test_fn = os.path.join(out_root, f'y_test_{suffix}.npy')
        sid_test_fn = os.path.join(out_root, f'site_ids_test_{suffix}.npy')
        ids_test_fn = os.path.join(out_root, f'test_identifiers_{suffix}.csv')
        np.save(x_test_fn, X_test)
        np.save(y_test_fn, y_test)
        np.save(sid_test_fn, SID_test)
        pd.DataFrame(meta_rows_te).to_csv(ids_test_fn, index=False)
        print(f"Saved TEST arrays: {x_test_fn} {X_test.shape}, {y_test_fn} {y_test.shape}, {sid_test_fn} {SID_test.shape}")

    pd.DataFrame(diagnostics_rows).to_csv(os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'), index=False)
    pd.DataFrame(normalizers_summary_rows).to_csv(os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'), index=False)
    print("Wrote diagnostics:")
    print(" -", os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'))
    print(" -", os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'))

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
            warnings_no_best_strict=('warning', lambda x: int(np.sum(x == 'no_best_combo_in_window_strict'))),
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
    return pd.DataFrame([{ 'site_id':int(s), 'n_thermometers':int(tC.get(s,0)), 'n_hygrometers':int(hC.get(s,0)),
                           'n_dendrometers_L2':int(dC.get(s,0)), 'n_dendrometers_LM':int(lC.get(s,0)) } for s in sites])

# ---- Civil-time utilities (plots/inspection; arrays remain UTC) ----

def utc_index_to_local(idx_utc: pd.DatetimeIndex, tz: str = 'Europe/Zurich') -> pd.DatetimeIndex:
    if getattr(idx_utc, 'tz', None) is None:
        raise ValueError('utc_index_to_local expects a tz‑aware UTC index')
    return idx_utc.tz_convert(tz)


def series_utc_to_civil(s_utc: pd.Series, tz: str = 'Europe/Zurich') -> pd.Series:
    if getattr(s_utc.index, 'tz', None) is None:
        raise ValueError('series_utc_to_civil expects a tz‑aware UTC index')
    return s_utc.tz_convert(tz)


def plot_series_civiltime(s_utc: pd.Series, tz: str = 'Europe/Zurich', title: str | None = None,
                          out_png: str | None = None) -> None:
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
# Strategy-B unit test
# =============================================================

def _test_strategy_b_dst_fallback(tz_local: str = 'Europe/Zurich') -> None:
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
