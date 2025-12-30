# build_normalized_dataset_treenet.py
# -*- coding: utf-8 -*-
"""
TreeNet → TF Pre-processing (Year-long Segments, Axis-Aware, DST‑Safe, Fully Annotated)
=====================================================================================

This module builds normalized datasets for ML from TreeNet data with **year‑long 10‑min segments**
and **hourly targets**. You can select a **time axis** for alignment and indexing:

- `local_dst`   → Europe/Zurich (civil local time with DST)
- `fixed_winter`→ CET (UTC+01:00, *no* DST) — “winter time only”
- `utc`         → UTC (no DST)

All local series are **converted and aligned** on the selected axis. For `local_dst`, alignment is
DST‑safe by rounding in UTC and converting back, resolving the repeated fall‑back hour using `--fold_rule`.

Outputs
-------
- Arrays: X_train.npy, y_train.npy (and X_test.npy, y_test.npy if test sites provided)
- Identifiers: train_identifiers.csv, test_identifiers.csv
- Normalizers per site/year: normalizers_site_<sid>.json; summary CSV
- Diagnostics CSVs: coverage warnings, counts, and per‑site summaries
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
# Constants
# =====================
SEQ_LEN_10MIN = 52560  # 365d * 24h * 6
HOUR_STEPS    = 8760   # 365d * 24h
STRIDE_PER_HR = 6
N_CHANNELS    = 11     # local T/RH/STEM + 8 global
N_TARGETS     = 3      # hourly temp, rh, stem
LOCAL_COLS  = ['local_T', 'local_RH', 'local_stem']
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']

# =====================
# Axis helpers & indices
# =====================

def tz_for_axis(time_axis: str, local_tz: str = 'Europe/Zurich') -> str:
    """Return timezone name for a given `time_axis` choice."""
    if time_axis == 'local_dst':
        return local_tz
    elif time_axis == 'fixed_winter':
        return 'CET'
    elif time_axis == 'utc':
        return 'UTC'
    else:
        raise ValueError(f"Unknown time_axis: {time_axis}")


def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx[~((idx.month==2)&(idx.day==29))]


def make_multi_year_10m_index_axis(years: t.Sequence[int], time_axis: str, local_tz: str = 'Europe/Zurich') -> pd.DatetimeIndex:
    """Create a **multi‑year 10‑min grid** spanning min(years)..max(years) in selected axis tz."""
    tz = tz_for_axis(time_axis, local_tz)
    y0,y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz=tz)
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz=tz)
    idx = pd.date_range(start=start, end=end, freq='10min')
    return strip_leap_days(idx)


def make_multi_year_hourly_index_axis(years: t.Sequence[int], time_axis: str, local_tz: str = 'Europe/Zurich') -> pd.DatetimeIndex:
    """Create a **multi‑year hourly grid** spanning min(years)..max(years) in selected axis tz."""
    tz = tz_for_axis(time_axis, local_tz)
    y0,y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz=tz)
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz=tz)
    idx = pd.date_range(start=start, end=end, freq='1H')
    return strip_leap_days(idx)

# =====================
# Meteo & readers
# =====================

def discover_meteo_files(meteo_dir: str) -> dict[int,str]:
    """Return `{site_id: csv_path}` mapping for daily global meteo files.
    File names must contain an integer site ID.
    """
    mapping: dict[int,str] = {}
    for fn in os.listdir(meteo_dir):
        if fn.endswith('.csv'):
            m = re.findall(r'\d+', fn)
            if m:
                mapping[int(m[0])] = os.path.join(meteo_dir, fn)
    return mapping


def load_global_daily(site_id: int, meteo_dir: str, tz_local: str) -> pd.DataFrame:
    """Load daily global meteo for `site_id` and standardize column names.
    Index at midnight in `tz_local`.
    """
    files = discover_meteo_files(meteo_dir)
    if site_id not in files:
        raise FileNotFoundError(f'No meteo CSV for site {site_id}')
    df = pd.read_csv(files[site_id])
    if 'ts' not in df.columns:
        raise ValueError('Global meteo CSV must have ts column (midnight timestamps).')
    ts = pd.to_datetime(df['ts'], utc=False)
    ts = ts.dt.tz_localize(tz_local) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(tz_local)
    df = df.set_index(ts.dt.normalize())
    rename = {'tas':'g_tmean','tasmax':'g_tmax','tasmin':'g_tmin','rh':'g_rh','vpd':'g_vpd','gh':'g_rad','pr':'g_pr'}
    for a,b in rename.items():
        if a not in df.columns:
            raise ValueError(f'Global meteo missing {a}')
        df[b] = df[a]
    df['g_doy'] = df.index.dayofyear
    return df[['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']]


def read_feather_series(series_id: int, dir_path: str, tz_local: str, value_col: str = 'value', ts_col: str = 'ts') -> pd.Series:
    """Read a **local instrument** series (`thermometer_l1`, `hygrometer_l1`, `dendrometer_l2`) and normalize to `tz_local`."""
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
        raise ValueError(f'Feather {fp} missing timestamp')
    if value_col not in df.columns:
        for alt in ('temp','temperature','rh','relhum','value_raw','measurement','radius','rad'):
            if alt in df.columns:
                value_col = alt; break
    if value_col not in df.columns:
        raise ValueError(f'Feather {fp} missing value column')
    ts = pd.to_datetime(df[ts_col], utc=False)
    ts = ts.dt.tz_localize(tz_local) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(tz_local)
    s = pd.Series(df[value_col].to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated(keep='mean')]
    return s

# =====================
# Axis‑aware alignment
# =====================

def align_to_grid_10min_axis(s: pd.Series,
                             time_axis: str,
                             fold_rule: str = 'mean',
                             local_tz: str = 'Europe/Zurich') -> pd.Series:
    """Align a 10‑min series to the selected axis (see coverage module for detailed behavior)."""
    if s.empty:
        return s
    s = s.copy()
    if time_axis == 'utc':
        # Ensure UTC
        idx = s.index
        if getattr(idx,'tz',None) is None:
            idx = pd.to_datetime(idx, utc=False).tz_localize('UTC')
        else:
            idx = idx.tz_convert('UTC')
        s.index = idx
        s.index = s.index.round('10min')
        s = s[~s.index.duplicated(keep='mean')]
        return s
    if time_axis == 'fixed_winter':
        # Convert to CET
        idx = s.index
        if getattr(idx,'tz',None) is None:
            idx = pd.to_datetime(idx, utc=False).tz_localize(local_tz, ambiguous='infer', nonexistent='shift_forward')
        else:
            idx = idx.tz_convert(local_tz)
        s.index = idx.tz_convert('CET')
        s.index = s.index.round('10min')
        s = s[~s.index.duplicated(keep='mean')]
        return s
    if time_axis == 'local_dst':
        idx = s.index
        if getattr(idx,'tz',None) is None:
            idx = pd.to_datetime(idx, utc=False).tz_localize(local_tz, ambiguous='infer', nonexistent='shift_forward')
        else:
            idx = idx.tz_convert(local_tz)
        s.index = idx
        s_utc = s.tz_convert('UTC'); s_utc.index = s_utc.index.round('10min')
        s_local = s_utc.tz_convert(local_tz)
        if fold_rule in ('earliest','first','latest','last'):
            keep = 'first' if fold_rule in ('earliest','first') else 'last'
            s_local = s_local[~s_local.index.duplicated(keep=keep)]
        elif fold_rule in ('mean','sum','min','max'):
            df = s_local.to_frame('v'); g = df.groupby(df.index.tz_localize(None))
            reducer = {'mean':np.mean,'sum':np.sum,'min':np.min,'max':np.max}[fold_rule]
            reduced = g.agg(reducer)
            reduced.index = reduced.index.tz_localize(local_tz, ambiguous='earliest', nonexistent='shift_forward')
            s_local = reduced['v']
        else:
            s_local = s_local[~s_local.index.duplicated(keep='first')]
        return s_local
    raise ValueError(f"Unexpected time_axis: {time_axis}")

# =====================
# Quantile scalers
# =====================

def compute_quantile_scaler(v: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> t.Tuple[float,float]:
    q1 = float(np.nanpercentile(v, q_low)); q2 = float(np.nanpercentile(v, q_high))
    if not np.isfinite(q1): q1 = float(np.nanmin(v))
    if not np.isfinite(q2): q2 = float(np.nanmax(v))
    if q2 <= q1: q2 = q1 + 1e-6
    return q1,q2


def fit_site_scalers(site_id: int, years: t.Sequence[int], meteo_dir: str, tz_local: str, per_year: bool=True) -> dict:
    """Fit **global channel** quantile scalers per site per year (or ALL)."""
    scalers: dict = {}
    glb_all = load_global_daily(site_id, meteo_dir, tz_local)
    for scope in ([y for y in years] if per_year else ['ALL']):
        if scope=='ALL':
            df = glb_all[(glb_all.index.year>=min(years))&(glb_all.index.year<=max(years))]
        else:
            df = glb_all[glb_all.index.year==scope]
        scalers.setdefault(scope,{})
        for col in GLOBAL_COLS:
            q1,q2 = compute_quantile_scaler(df[col].to_numpy())
            scalers[scope][col]={'q_low':q1,'q_high':q2}
    return scalers


def normalize_array(arr: np.ndarray, q1: float, q2: float, clip_low: float=-0.1, clip_high: float=1.1) -> np.ndarray:
    out = (arr - q1) / (q2 - q1)
    return np.clip(out, clip_low, clip_high)

# =====================
# Sensor discovery per site
# =====================

def discover_series_ids(dir_path: str, pattern_prefix: str) -> t.Set[int]:
    ids: set[int] = set()
    regex = re.compile(rf'{re.escape(pattern_prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m: ids.add(int(m.group(1)))
    return ids


def series_by_site(metadata_df: pd.DataFrame, series_ids: t.Set[int]) -> dict[int, t.List[int]]:
    df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
    groups = df.groupby('site_id')
    out: dict[int,t.List[int]] = {}
    for site, g in groups:
        out[int(site)] = g['series_id'].tolist()
    return out

# =====================
# LM targets
# =====================

def read_lm_frame(series_id: int, lm_dir: str, tz_local: str) -> pd.DataFrame:
    pat = re.compile(rf'dendrometer_lm_series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(lm_dir) if pat.match(fn)]
    if not matches: raise FileNotFoundError(f'LM {series_id} missing')
    fp = os.path.join(lm_dir, matches[0])
    df = pd.read_feather(fp)
    if 'ts' not in df.columns:
        for c in ('timestamp','time','date_time','datetime'):
            if c in df.columns: df = df.rename(columns={c:'ts'}); break
    if 'ts' not in df.columns: raise ValueError(f'LM {fp} missing ts')
    ts = pd.to_datetime(df['ts'], utc=False)
    ts = ts.dt.tz_localize(tz_local) if getattr(ts.dt,'tz',None) is None else ts.dt.tz_convert(tz_local)
    df = df.set_index(ts)
    for col in ['value','temp','rh']:
        if col not in df.columns: df[col] = np.nan
    return df[['value','temp','rh']]


def to_hourly(df: pd.DataFrame, how: str='median') -> pd.DataFrame:
    return df.resample('1H').agg({'value':how,'temp':how,'rh':how})


def build_site_level_targets_multi(site_id: int, years: t.Sequence[int], tz_local: str, lm_dir: str, dendro_lm_ids_by_site: dict[int,t.List[int]], stem_mode: str='absolute', agg: str='median', time_axis: str='local_dst') -> pd.DataFrame:
    """Aggregate LM series per site into **hourly site targets** on the selected axis.
    Returns a DataFrame with columns `stem`, `temp`, `rh` on the multi‑year hourly index.
    """
    idx_hour = make_multi_year_hourly_index_axis(years, time_axis, tz_local)
    frames = []
    lm_ids = dendro_lm_ids_by_site.get(site_id, [])
    for sid in lm_ids:
        try:
            df = read_lm_frame(sid, lm_dir, tz_local)
        except Exception:
            continue
        df = df[(df.index.year>=min(years))&(df.index.year<=max(years))].copy()
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

# =====================
# Coverage & diagnostics
# =====================

def coverage_fraction(series: pd.Series) -> float:
    v = series.to_numpy(); n=v.size
    return float(np.sum(~np.isnan(v))/n) if n else 0.0


def write_diagnostics_row(rows: t.List[dict], **kw): rows.append(kw)

# =====================
# Rolling windows
# =====================

def rolling_year_windows(idx_10m: pd.DatetimeIndex, overlap_days: int=10, year_days: int=365) -> t.List[t.Tuple[pd.Timestamp,pd.Timestamp]]:
    """Return (start, end) for year‑long windows with `overlap_days` between consecutive windows."""
    stride_days = year_days - overlap_days
    starts = []
    cur = idx_10m[0].normalize(); end_ts = idx_10m[-1]
    while cur + pd.Timedelta(days=year_days) <= end_ts:
        starts.append(cur); cur = cur + pd.Timedelta(days=stride_days)
    return [(s, s + pd.Timedelta(days=year_days)) for s in starts]

# =====================
# Build segments per site
# =====================

def build_segments_for_site(site_id: int, years: t.Sequence[int], tz_local: str,
                            meteo_dir: str, thermo_dir: str, hygro_dir: str, dendro_l2_dir: str, dendro_lm_dir: str,
                            metadata_path: str, scalers: dict,
                            per_year: bool=True, require_complete_locals: bool=False,
                            stem_mode: str='absolute', input_mode: str='combinations', target_mode: str='lm_site_median',
                            max_combos_per_site: t.Optional[int]=None, min_local_coverage: float=0.7, min_lm_series: int=1,
                            overlap_days: int=10, diagnostics_rows: t.Optional[t.List[dict]]=None, split: str='train',
                            allow_missing_locals: bool=False, fold_rule: str='mean', time_axis: str='local_dst') -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[dict]]:
    """Construct year‑long segments for one site in the selected `time_axis`.
    Returns lists of input segments (X), target segments (Y), and metadata rows.
    """
    metadata_df = pd.read_pickle(metadata_path)
    idx_10m = make_multi_year_10m_index_axis(years, time_axis, tz_local)
    idx_hour= make_multi_year_hourly_index_axis(years, time_axis, tz_local)

    glb_all = load_global_daily(site_id, meteo_dir, tz_local)
    glb_10m = (lambda df, idx: (df.copy().rename(columns=lambda c: c))[GLOBAL_COLS])(glb_all, idx_10m)
    # Broadcast daily to 10-min
    daily = glb_all.copy(); daily.index = daily.index.normalize()
    dates = idx_10m.normalize()
    glb_10m = daily.reindex(dates).ffill(); glb_10m.index = idx_10m

    # Discover series per type
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

    site_targets_hourly = None
    if target_mode == 'lm_site_median':
        site_targets_hourly = build_site_level_targets_multi(site_id, years, tz_local, dendro_lm_dir, lm_by_site, stem_mode=stem_mode, agg='median', time_axis=time_axis)
        if len(lm_ids_site) < min_lm_series:
            if diagnostics_rows is not None:
                write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, warning='insufficient_lm_series', lm_series_count=len(lm_ids_site))
            return [], [], []

    windows = rolling_year_windows(idx_10m, overlap_days=overlap_days, year_days=365)
    key_for = (lambda ts: ts.year) if per_year else (lambda ts: 'ALL')

    X_list: t.List[np.ndarray] = []
    Y_list: t.List[np.ndarray] = []
    META_list: t.List[dict] = []
    segments_count = 0

    # Pooled strategy (median across instruments per step)
    if input_mode == 'pooled':
        def pooled_series(ids, dirp):
            arrs = []
            for sid in ids:
                try:
                    s_ser = read_feather_series(sid, dirp, tz_local)
                    s_ser = align_to_grid_10min_axis(s_ser, time_axis=time_axis, fold_rule=fold_rule, local_tz=tz_local)
                    arrs.append(s_ser.reindex(idx_10m).to_numpy())
                except Exception:
                    pass
            if not arrs:
                return pd.Series(np.nan, index=idx_10m)
            A = np.vstack(arrs); med = np.nanmedian(A, axis=0)
            return pd.Series(med, index=idx_10m)
        s_T_pooled  = pooled_series(thermo_ids, thermo_dir)
        s_RH_pooled = pooled_series(hygro_ids,  hygro_dir)
        s_STEM_pooled = pooled_series(dendro_l2_ids, dendro_l2_dir)
        if stem_mode == 'delta': s_STEM_pooled = s_STEM_pooled.diff()

    for (ws,we) in windows:
        scope_key = key_for(ws)
        glb_win = glb_10m[(glb_10m.index>=ws)&(glb_10m.index<we)]
        if glb_win.shape[0] != SEQ_LEN_10MIN:
            continue
        glb_n = { col: normalize_array(glb_win[col].to_numpy(), scalers[scope_key][col]['q_low'], scalers[scope_key][col]['q_high']) for col in GLOBAL_COLS }
        start_h = ws; end_h = we - pd.Timedelta(minutes=10)
        idx_h = pd.date_range(start_h.floor('H'), end_h.floor('H'), freq='1H')
        idx_h = strip_leap_days(idx_h)

        def cov_in_window(series_id, dirp, do_diff=False):
            try:
                s = read_feather_series(series_id, dirp, tz_local)
                s = align_to_grid_10min_axis(s, time_axis=time_axis, fold_rule=fold_rule, local_tz=tz_local)
                s = s[(s.index>=ws)&(s.index<we)]
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
                covT=cov_in_window(t_id0, thermo_dir, False)
                for h_id0 in hygro_ids:
                    covRH=cov_in_window(h_id0, hygro_dir, False)
                    for d_id0 in dendro_l2_ids:
                        covST=cov_in_window(d_id0, dendro_l2_dir, stem_mode=='delta')
                        score=min(covT,covRH,covST)
                        if score>best_score:
                            best_score=score; best_triplet=(t_id0,h_id0,d_id0)
            if (best_triplet is None) or (best_score < min_local_coverage):
                if diagnostics_rows is not None:
                    write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, window_start=str(ws), window_end=str(we), warning='no_best_combo_in_window', min_local_coverage=min_local_coverage)
                continue
            t_candidates=[best_triplet[0]]; h_candidates=[best_triplet[1]]; d_candidates=[best_triplet[2]]
        else:
            t_candidates=thermo_ids; h_candidates=hygro_ids; d_candidates=dendro_l2_ids

        for t_id in t_candidates:
            for h_id in h_candidates:
                for d_id in d_candidates:
                    if input_mode != 'pooled':
                        try:
                            s_T  = read_feather_series(t_id, thermo_dir, tz_local)
                            s_T  = align_to_grid_10min_axis(s_T, time_axis=time_axis, fold_rule=fold_rule, local_tz=tz_local).reindex(idx_10m)
                            s_RH = read_feather_series(h_id, hygro_dir,  tz_local)
                            s_RH = align_to_grid_10min_axis(s_RH, time_axis=time_axis, fold_rule=fold_rule, local_tz=tz_local).reindex(idx_10m)
                            s_ST = read_feather_series(d_id, dendro_l2_dir, tz_local)
                            s_ST = align_to_grid_10min_axis(s_ST, time_axis=time_axis, fold_rule=fold_rule, local_tz=tz_local).reindex(idx_10m)
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

                    cov_T = coverage_fraction(sT_w); cov_RH = coverage_fraction(sRH_w); cov_ST = coverage_fraction(sST_w)
                    used_global_T=False; used_global_RH=False
                    if allow_missing_locals:
                        if cov_T  < min_local_coverage: used_global_T=True
                        if cov_RH < min_local_coverage: used_global_RH=True
                    if (not allow_missing_locals) and ((cov_T<min_local_coverage) or (cov_RH<min_local_coverage) or (cov_ST<min_local_coverage)):
                        if diagnostics_rows is not None:
                            write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, warning='low_local_coverage_combo', thermo_id=(t_id if t_id is not None else -1), hygro_id=(h_id if h_id is not None else -1), dendro_id=(d_id if d_id is not None else -1), window_start=str(ws), cov_T=cov_T, cov_RH=cov_RH, cov_ST=cov_ST)
                        continue

                    # Lazy quantiles for locals (per scope) if missing
                    for name, s in [('local_T', s_T if input_mode!='pooled' else s_T_pooled),('local_RH', s_RH if input_mode!='pooled' else s_RH_pooled),('local_stem', s_ST if input_mode!='pooled' else s_STEM_pooled)]:
                        if name not in scalers[scope_key]:
                            q1,q2 = compute_quantile_scaler(s.to_numpy())
                            scalers[scope_key][name]={'q_low':q1,'q_high':q2}

                    # Normalize locals
                    T_n    = normalize_array(sT_w.to_numpy(),  scalers[scope_key]['local_T']['q_low'],    scalers[scope_key]['local_T']['q_high'])
                    RH_n   = normalize_array(sRH_w.to_numpy(), scalers[scope_key]['local_RH']['q_low'],   scalers[scope_key]['local_RH']['q_high'])
                    STEM_n = normalize_array(sST_w.to_numpy(), scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high'])

                    # Global substitution (inputs only)
                    if allow_missing_locals and used_global_T:  T_n = glb_n['g_tmean'].copy()
                    if allow_missing_locals and used_global_RH: RH_n = glb_n['g_rh'].copy()

                    X_seg = np.column_stack([
                        T_n, RH_n, STEM_n,
                        glb_n['g_tmean'], glb_n['g_tmin'], glb_n['g_tmax'],
                        glb_n['g_rh'], glb_n['g_vpd'], glb_n['g_pr'], glb_n['g_rad'], glb_n['g_doy'],
                    ]).astype(np.float32)
                    if X_seg.shape[0] != SEQ_LEN_10MIN:
                        continue

                    # Targets: LM site median if available, else hourly medians from locals
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

                    if diagnostics_rows is not None:
                        try:
                            cov_T_seg = float(np.mean(~np.isnan(T_n)))
                            cov_RH_seg= float(np.mean(~np.isnan(RH_n)))
                            cov_ST_seg= float(np.mean(~np.isnan(STEM_n)))
                        except Exception:
                            cov_T_seg = cov_RH_seg = cov_ST_seg = np.nan
                        write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, window_start=str(ws), window_end=str(we), thermo_id=(int(t_id) if t_id is not None else -1), hygro_id=(int(h_id) if h_id is not None else -1), dendro_id=(int(d_id) if d_id is not None else -1), cov_T=cov_T_seg, cov_RH=cov_RH_seg, cov_ST=cov_ST_seg, used_global_T=int(used_global_T), used_global_RH=int(used_global_RH))

                    X_list.append(X_seg); Y_list.append(y_seg); META_list.append({
                        'site_id': site_id,
                        'years_scope': f"{min(years)}-{max(years)}",
                        'window_start': str(ws), 'window_end': str(we),
                        'input_mode': input_mode, 'target_mode': target_mode,
                        'thermometer_id': int(t_id) if t_id is not None else -1,
                        'hygrometer_id': int(h_id) if h_id is not None else -1,
                        'dendrometer_id': int(d_id) if d_id is not None else -1,
                        'used_global_T': int(used_global_T), 'used_global_RH': int(used_global_RH),
                    })
                    segments_count += 1
        if diagnostics_rows is not None:
            write_diagnostics_row(diagnostics_rows, site_id=site_id, split=split, windows=1, segments=segments_count)

    return X_list, Y_list, META_list

# =====================
# Datasets builder
# =====================

def build_datasets(out_root: str, meteo_dir: str, thermo_dir: str, hygro_dir: str, dendro_l2_dir: str, dendro_lm_dir: str,
                   metadata_pickle: str, site_ids_train: t.Sequence[int], site_ids_test: t.Optional[t.Sequence[int]],
                   years: t.Sequence[int], per_year: bool=True, tz_local: str='Europe/Zurich', require_complete_locals: bool=False,
                   stem_mode: str='absolute', input_mode: str='combinations', target_mode: str='lm_site_median',
                   max_combos_per_site: t.Optional[int]=None, min_local_coverage: float=0.7, min_lm_series: int=1,
                   overlap_days: int=10, allow_missing_locals: bool=False, fold_rule: str='mean', time_axis: str='local_dst') -> None:
    """Top‑level orchestration to build train/test arrays in the selected `time_axis`."""
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root,'diagnostics'), exist_ok=True)

    site_scalers: dict[int,dict] = {}; normalizers_summary_rows: t.List[dict] = []

    # Fit global scalers for TRAIN sites
    for sid in site_ids_train:
        scalers = fit_site_scalers(sid, years, meteo_dir, tz_local, per_year=per_year)
        site_scalers[sid]=scalers
        norm_dir=os.path.join(out_root,'normalizers'); os.makedirs(norm_dir, exist_ok=True)
        with open(os.path.join(norm_dir, f'normalizers_site_{sid}.json'),'w',encoding='utf-8') as f: json.dump(scalers,f,indent=2)
        for scope,chmap in scalers.items():
            for ch,qq in chmap.items(): normalizers_summary_rows.append({'site_id':sid,'scope':scope,'channel':ch,'q_low':qq['q_low'],'q_high':qq['q_high']})

    # TRAIN
    X_tr_list=[]; y_tr_list=[]; sid_tr_list=[]; meta_rows=[]; diagnostics_rows=[]
    for sid in site_ids_train:
        scalers = site_scalers[sid]
        X_list,y_list,meta_list = build_segments_for_site(sid, years, tz_local, meteo_dir, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir, metadata_pickle, scalers, per_year, require_complete_locals, stem_mode, input_mode, target_mode, max_combos_per_site, min_local_coverage, min_lm_series, overlap_days, diagnostics_rows, 'train', allow_missing_locals, fold_rule, time_axis)
        X_tr_list.extend(X_list); y_tr_list.extend(y_list); sid_tr_list.extend([sid]*len(X_list)); meta_rows.extend(meta_list)
    X_train = np.stack(X_tr_list, axis=0) if X_tr_list else np.empty((0,SEQ_LEN_10MIN,N_CHANNELS),dtype=np.float32)
    y_train = np.stack(y_tr_list, axis=0) if y_tr_list else np.empty((0,HOUR_STEPS,N_TARGETS),dtype=np.float32)
    SID_train = np.array(sid_tr_list,dtype=np.int32)
    np.save(os.path.join(out_root,'X_train.npy'), X_train)
    np.save(os.path.join(out_root,'y_train.npy'), y_train)
    np.save(os.path.join(out_root,'site_ids_train.npy'), SID_train)
    pd.DataFrame(meta_rows).to_csv(os.path.join(out_root,'train_identifiers.csv'), index=False)
    print(f'Saved TRAIN arrays: X_train {X_train.shape}, y_train {y_train.shape}, site_ids_train {SID_train.shape}')

    # TEST
    if site_ids_test is not None:
        for sid in site_ids_test:
            scalers = fit_site_scalers(sid, years, meteo_dir, tz_local, per_year=per_year)
            site_scalers[sid]=scalers
            norm_dir=os.path.join(out_root,'normalizers'); os.makedirs(norm_dir, exist_ok=True)
            with open(os.path.join(norm_dir, f'normalizers_site_{sid}.json'),'w',encoding='utf-8') as f: json.dump(scalers,f,indent=2)
            for scope,chmap in scalers.items():
                for ch,qq in chmap.items(): normalizers_summary_rows.append({'site_id':sid,'scope':scope,'channel':ch,'q_low':qq['q_low'],'q_high':qq['q_high']})
        X_te_list=[]; y_te_list=[]; sid_te_list=[]; meta_rows_te=[]
        for sid in site_ids_test:
            scalers = site_scalers[sid]
            X_list,y_list,meta_list = build_segments_for_site(sid, years, tz_local, meteo_dir, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir, metadata_pickle, scalers, per_year, require_complete_locals, stem_mode, input_mode, target_mode, max_combos_per_site, min_local_coverage, min_lm_series, overlap_days, diagnostics_rows, 'test', allow_missing_locals, fold_rule, time_axis)
            X_te_list.extend(X_list); y_te_list.extend(y_list); sid_te_list.extend([sid]*len(X_list)); meta_rows_te.extend(meta_list)
        X_test = np.stack(X_te_list, axis=0) if X_te_list else np.empty((0,SEQ_LEN_10MIN,N_CHANNELS),dtype=np.float32)
        y_test = np.stack(y_te_list, axis=0) if y_te_list else np.empty((0,HOUR_STEPS,N_TARGETS),dtype=np.float32)
        SID_test = np.array(sid_te_list,dtype=np.int32)
        np.save(os.path.join(out_root,'X_test.npy'), X_test)
        np.save(os.path.join(out_root,'y_test.npy'), y_test)
        np.save(os.path.join(out_root,'site_ids_test.npy'), SID_test)
        pd.DataFrame(meta_rows_te).to_csv(os.path.join(out_root,'test_identifiers.csv'), index=False)
        print(f'Saved TEST arrays: X_test {X_test.shape}, y_test {y_test.shape}, site_ids_test {SID_test.shape}')

    # Diagnostics and normalizers summary
    pd.DataFrame(diagnostics_rows).to_csv(os.path.join(out_root,'diagnostics','diagnostics_preprocessing.csv'), index=False)
    pd.DataFrame(normalizers_summary_rows).to_csv(os.path.join(out_root,'diagnostics','normalizers_summary.csv'), index=False)
    print('Wrote diagnostics:')
    print(' -', os.path.join(out_root,'diagnostics','diagnostics_preprocessing.csv'))
    print(' -', os.path.join(out_root,'diagnostics','normalizers_summary.csv'))

    # Site summary (instrument counts + diagnostics aggregates)
    instrument_df = compute_site_instrument_counts(metadata_pickle, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir)
    diag_df = pd.DataFrame(diagnostics_rows) if diagnostics_rows else pd.DataFrame(columns=['site_id','split','windows','segments','lm_series_count','warning'])
    for col,default in [('site_id',-1),('split','train'),('windows',0),('segments',0),('warning',None)]:
        if col not in diag_df.columns: diag_df[col]=default
    def agg_split(df, split):
        d = df[df['split']==split] if 'split' in df.columns else df
        g = d.groupby('site_id').agg(
            segments_total=('segments','sum'),
            windows_total=('windows','sum'),
            warnings_low_local_combo=('warning', lambda x: int(np.sum(x=='low_local_coverage_combo'))),
            warnings_low_local_pooled=('warning', lambda x: int(np.sum(x=='low_local_coverage_pooled'))),
            warnings_insufficient_lm=('warning', lambda x: int(np.sum(x=='insufficient_lm_series'))),
            warnings_no_best=('warning', lambda x: int(np.sum(x=='no_best_combo_in_window'))),
        ).reset_index()
        g.columns = ['site_id'] + [f'{split}_{c}' for c in g.columns[1:]]; return g
    site_train = agg_split(diag_df,'train'); site_test = agg_split(diag_df,'test')
    site_summary = instrument_df.merge(site_train, on='site_id', how='left').merge(site_test, on='site_id', how='left')
    fill_cols = [c for c in site_summary.columns if c!='site_id']
    site_summary[fill_cols] = site_summary[fill_cols].fillna(0).infer_objects(copy=False)
    os.makedirs(os.path.join(out_root,'diagnostics'), exist_ok=True)
    site_summary.to_csv(os.path.join(out_root,'diagnostics','site_summary.csv'), index=False)
    print('Wrote site-level summary:', os.path.join(out_root,'diagnostics','site_summary.csv'))

# =====================
# CLI
# =====================

def parse_args():
    """Parse CLI args for pre‑processing.
    Includes `--time_axis` and `--fold_rule` controlling alignment of local series.
    """
    p = argparse.ArgumentParser(description='TreeNet pre-processing (axis-aware, DST-safe, robust).')
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
    p.add_argument('--target_mode', type=str, default='lm_site_median', choices=['lm_site_median','lm_per_series'])
    p.add_argument('--max_combos_per_site', type=int, default=None)
    p.add_argument('--min_local_coverage', type=float, default=0.7)
    p.add_argument('--min_lm_series', type=int, default=1)
    p.add_argument('--overlap_days', type=int, default=10)
    p.add_argument('--allow_missing_locals', type=str, default='false')
    p.add_argument('--fold_rule', type=str, default='mean', choices=['mean','earliest','latest','first','last','min','max','sum'])
    p.add_argument('--time_axis', type=str, default='local_dst', choices=['local_dst','fixed_winter','utc'])
    return p.parse_args()


def read_site_ids_csv(path: str) -> t.List[int]:
    df = pd.read_csv(path)
    if 'site_id' not in df.columns: raise ValueError('CSV must contain site_id')
    return [int(x) for x in df['site_id'].tolist()]

if __name__ == '__main__':
    args = parse_args()
    per_year = (args.per_year.lower()=='true')
    require_complete_locals = (args.require_complete_locals.lower()=='true')
    site_ids_train = read_site_ids_csv(args.train_site_ids_csv)
    site_ids_test = read_site_ids_csv(args.test_site_ids_csv) if args.test_site_ids_csv else None
    build_datasets(out_root=args.out_root, meteo_dir=args.meteo_dir, thermo_dir=args.thermo_dir, hygro_dir=args.hygro_dir, dendro_l2_dir=args.dendro_l2_dir, dendro_lm_dir=args.dendro_lm_dir, metadata_pickle=args.metadata_pickle, site_ids_train=site_ids_train, site_ids_test=site_ids_test, years=args.years, per_year=per_year, tz_local=args.tz, require_complete_locals=require_complete_locals, stem_mode=args.stem_mode, input_mode=args.input_mode, target_mode=args.target_mode, max_combos_per_site=args.max_combos_per_site, min_local_coverage=args.min_local_coverage, min_lm_series=args.min_lm_series, overlap_days=args.overlap_days, allow_missing_locals=(args.allow_missing_locals.lower()=='true'), fold_rule=args.fold_rule, time_axis=args.time_axis)
