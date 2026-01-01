# build_30day_dataset_treenet_utc.py
# -*- coding: utf-8 -*-

"""
TreeNet TF Preprocessing - 30-day segments (UTC) with Strategy-B globals.
Accepts the same CLI superset as the yearly script (per_year, stem_mode, input_mode, max_combos_per_site, overlap_days, etc.).
Prints per-site summaries and final array shapes. Writes diagnostics.
"""

from __future__ import annotations
import os
import re
import json
import argparse
import typing as t
import numpy as np
import pandas as pd

FREQ_10M = '10min'
FREQ_1H  = '1h'
GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']

def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx[~(((idx.month==2) & (idx.day==29)))]

def make_multi_year_10m_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_10M)
    return strip_leap_days(idx)

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
        raise ValueError('Global meteo CSV must have ts column')
    idx_civil = pd.to_datetime(df['ts'], utc=False).dt.normalize()
    rename = {'tas':'g_tmean','tasmax':'g_tmax','tasmin':'g_tmin','rh':'g_rh','vpd':'g_vpd','gh':'g_rad','pr':'g_pr'}
    for src, dst in rename.items():
        df[dst] = df[src]
    out = df.set_index(idx_civil)[['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad']].copy()
    out = out[~out.index.duplicated(keep='last')]
    return out

def broadcast_daily_civil_to_utc_grid(daily_civil: pd.DataFrame,
                                      idx_10m_utc: pd.DatetimeIndex,
                                      tz_local: str = 'Europe/Zurich') -> pd.DataFrame:
    if getattr(idx_10m_utc, 'tz', None) is None:
        raise ValueError('idx_10m_utc must be tz-aware UTC index')
    civil_days_for_grid = idx_10m_utc.tz_convert(tz_local).normalize()
    daily_by_civilday = daily_civil.copy()
    daily_by_civilday.index = pd.to_datetime(daily_by_civilday.index).normalize()
    out = daily_by_civilday.reindex(civil_days_for_grid).ffill()
    out.index = idx_10m_utc
    return out

def _pick_value_column(df: pd.DataFrame, preferred: t.Sequence[str] = ()) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    if 'value' in df.columns:
        return 'value'
    candidates: list[tuple[int,str]] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ('ts','timestamp','time','date_time','datetime'):
            candidates.append((int(df[c].notna().sum()), c))
    candidates.sort(reverse=True)
    if not candidates:
        raise ValueError('No numeric column found')
    return candidates[0][1]

def read_feather_series_utc(series_id: int, dir_path: str, local_tz: str,
                             value_col: str = 'value', ts_col: str = 'ts', sensor_hint: t.Optional[str] = None) -> pd.Series:
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
        raise ValueError('Missing timestamp column')
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
    pat_raw = re.compile(rf'dendrometer_lm_series_id_{series_id}\.ftr$')
    matches_raw = [fn for fn in os.listdir(lm_dir) if pat_raw.match(fn)]
    if not matches_raw:
        raise FileNotFoundError(f'LM RAW file not found for series {series_id}')
    df = pd.read_feather(os.path.join(lm_dir, matches_raw[0]))
    ts_col = 'ts' if 'ts' in df.columns else ('timestamp' if 'timestamp' in df.columns else None)
    if ts_col is None:
        raise ValueError('LM RAW missing timestamp')
    ts_local = pd.to_datetime(df[ts_col], utc=False)
    ts_local = ts_local.dt.tz_localize(local_tz) if getattr(ts_local.dt,'tz',None) is None else ts_local.dt.tz_convert(local_tz)
    df_local = df.copy(); df_local.index = ts_local
    for col in ['value','temp','rh']:
        if col not in df_local.columns:
            df_local[col] = np.nan
    hh_mask = (df_local.index.minute == 0) & (df_local.index.second == 0)
    df_hourly_local = df_local.loc[hh_mask, ['value','temp','rh']].sort_index()
    if df_hourly_local.empty:
        return pd.DataFrame(columns=['stem','local_T','local_RH'])
    start_local = df_hourly_local.index.min(); end_local = df_hourly_local.index.max()
    idx_hour_local = pd.date_range(start_local, end_local, freq=FREQ_1H)
    df_hourly_local = df_hourly_local.reindex(idx_hour_local)
    idx_hour_utc = df_hourly_local.index.tz_convert('UTC')
    out = pd.DataFrame({
        'stem': df_hourly_local['value'].to_numpy(),
        'local_T': df_hourly_local['temp'].to_numpy(),
        'local_RH': df_hourly_local['rh'].to_numpy(),
    }, index=idx_hour_utc).sort_index()
    return out

def compute_quantile_scaler(v: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> t.Tuple[float,float]:
    q1 = float(np.nanpercentile(v, q_low)); q2 = float(np.nanpercentile(v, q_high))
    if not np.isfinite(q1): q1 = float(np.nanmin(v))
    if not np.isfinite(q2): q2 = float(np.nanmax(v))
    if q2 <= q1: q2 = q1 + 1e-6
    return q1, q2

def fit_site_scalers(site_id: int, years: t.Sequence[int], meteo_dir: str,
                     per_year: bool = True) -> dict:
    glb_daily = load_global_daily_civil(site_id, meteo_dir)
    scopes = [y for y in years] if per_year else ['ALL']
    scalers: dict = {}
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
    thermo_mask = var.str.contains('temperature', na=False) | var.str.contains('temp', na=False)
    hygro_mask  = var.str.contains('humidity', na=False)   | var.str.contains('rh', na=False)
    dendro_mask = var.str.contains('dendrometer', na=False) | var.str.contains('stem', na=False) | var.str.contains('radius', na=False)
    thermo_ids = rows.loc[thermo_mask, 'series_id'].astype(int).tolist()
    hygro_ids  = rows.loc[hygro_mask,  'series_id'].astype(int).tolist()
    dendro_ids = rows.loc[dendro_mask, 'series_id'].astype(int).tolist()
    return thermo_ids, hygro_ids, dendro_ids

def coverage_fraction(series: pd.Series) -> float:
    v = series.to_numpy(); n=v.size
    return float(np.sum(~np.isnan(v))/n) if n else 0.0

def rolling_day_windows(idx_10m: pd.DatetimeIndex, window_days: int, stride_days: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(idx_10m) == 0:
        return []
    W = pd.Timedelta(days=window_days)
    stride = pd.Timedelta(days=stride_days)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    t0 = idx_10m[0].floor('10min')
    step = pd.Timedelta(minutes=10)
    end_excl = idx_10m[-1] + step
    t = t0
    while t + W <= end_excl:
        ws = t; we = t + W
        windows.append((ws, we))
        t = t + stride
    return windows

def build_segments_for_site_30d(
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
    min_local_coverage: float = 0.5,
    allow_missing_locals: bool = False,
    complete_only: bool = True,
    input_mode: str = 'combinations',
    max_combos_per_site: t.Optional[int] = None,
    window_days: int = 30,
    window_stride_days: int = 1,
) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[dict]]:
    metadata_df = pd.read_pickle(metadata_path)
    idx_10m = make_multi_year_10m_index_utc(years)
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
    lm_ids_raw = set(discover_series_ids(dendro_lm_dir, 'dendrometer_lm'))
    dendro_ids_all = [d for d in dendro_l2_ids_site if d in lm_ids_raw]
    windows = rolling_day_windows(idx_10m, window_days=window_days, stride_days=window_stride_days)

    X_list: t.List[np.ndarray] = []
    Y_list: t.List[np.ndarray] = []
    META_list: t.List[dict] = []

    for d_id in dendro_ids_all:
        try:
            s_ST_full = read_feather_series_utc(d_id, dendro_l2_dir, local_tz, sensor_hint='dendrometer_l2')
        except Exception:
            continue
        s_ST_full = s_ST_full.tz_convert('UTC').sort_index()
        s_ST_full = s_ST_full.resample(FREQ_10M, origin='start_day', label='left').median().reindex(idx_10m)
        try:
            df_lm_hr = read_lm_hourly_frame_utc(d_id, dendro_lm_dir, local_tz)
        except Exception:
            continue
        for (ws, we) in windows:
            glb_win = glb_10m[(glb_10m.index>=ws)&(glb_10m.index<we)]
            if glb_win.shape[0] == 0:
                continue
            start_h = ws; end_h = we - pd.Timedelta(minutes=10)
            idx_h = pd.date_range(start_h.floor('h'), end_h.floor('h'), freq=FREQ_1H)
            idx_h = strip_leap_days(idx_h)
            dfh = df_lm_hr[(df_lm_hr.index>=ws)&(df_lm_hr.index<=end_h)].reindex(idx_h)
            if complete_only and dfh[['stem','local_T','local_RH']].isna().any().any():
                continue
            scope_key = ws.year
            if scope_key not in scalers:
                scalers[scope_key] = {}
            if 'local_stem' not in scalers[scope_key]:
                q1, q2 = compute_quantile_scaler(s_ST_full.to_numpy())
                scalers[scope_key]['local_stem'] = {'q_low': q1, 'q_high': q2}
            generated = 0
            for t_id in thermo_ids_site:
                try:
                    s_T_full = read_feather_series_utc(t_id, thermo_dir, local_tz, sensor_hint='thermometer_l1')
                    s_T_full = s_T_full.tz_convert('UTC').sort_index()
                    s_T_full = s_T_full.resample(FREQ_10M, origin='start_day', label='left').median().reindex(idx_10m)
                except Exception:
                    continue
                for h_id in hygro_ids_site:
                    try:
                        s_RH_full = read_feather_series_utc(h_id, hygro_dir, local_tz, sensor_hint='hygrometer_l1')
                        s_RH_full = s_RH_full.tz_convert('UTC').sort_index()
                        s_RH_full = s_RH_full.resample(FREQ_10M, origin='start_day', label='left').median().reindex(idx_10m)
                    except Exception:
                        continue
                    sT_w = s_T_full[(s_T_full.index>=ws)&(s_T_full.index<we)]
                    sRH_w= s_RH_full[(s_RH_full.index>=ws)&(s_RH_full.index<we)]
                    sST_w= s_ST_full[(s_ST_full.index>=ws)&(s_ST_full.index<we)]
                    covT, covRH, covST = coverage_fraction(sT_w), coverage_fraction(sRH_w), coverage_fraction(sST_w)
                    eff_min = 1.0 if complete_only else min_local_coverage
                    if not allow_missing_locals:
                        if (covT < eff_min) or (covRH < eff_min) or (covST < eff_min):
                            continue
                    else:
                        if covST < eff_min:
                            continue
                    for name, sfull in [('local_T', s_T_full), ('local_RH', s_RH_full)]:
                        if name not in scalers[scope_key]:
                            q1, q2 = compute_quantile_scaler(sfull.to_numpy())
                            scalers[scope_key][name] = {'q_low': q1, 'q_high': q2}
                    T_n   = normalize_array(sT_w.to_numpy(),  scalers[scope_key]['local_T']['q_low'],   scalers[scope_key]['local_T']['q_high'])
                    RH_n  = normalize_array(sRH_w.to_numpy(),  scalers[scope_key]['local_RH']['q_low'], scalers[scope_key]['local_RH']['q_high'])
                    STEM_n= normalize_array(sST_w.to_numpy(),  scalers[scope_key]['local_stem']['q_low'],scalers[scope_key]['local_stem']['q_high'])
                    glb_n = {col: normalize_array(glb_win[col].to_numpy(), scalers[scope_key][col]['q_low'], scalers[scope_key][col]['q_high']) for col in ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad']}
                    glb_n['g_doy'] = normalize_array(glb_win['g_doy'].to_numpy(), 1.0, 365.0)
                    X_seg = np.column_stack([
                        T_n, RH_n, STEM_n,
                        glb_n['g_tmean'], glb_n['g_tmin'], glb_n['g_tmax'],
                        glb_n['g_rh'], glb_n['g_vpd'], glb_n['g_pr'], glb_n['g_rad'], glb_n['g_doy']
                    ]).astype(np.float32)
                    temp_hr = normalize_array(dfh['local_T'].to_numpy(), scalers[scope_key]['local_T']['q_low'], scalers[scope_key]['local_T']['q_high'])
                    rh_hr   = normalize_array(dfh['local_RH'].to_numpy(), scalers[scope_key]['local_RH']['q_low'], scalers[scope_key]['local_RH']['q_high'])
                    stem_hr = normalize_array(dfh['stem'].to_numpy(),     scalers[scope_key]['local_stem']['q_low'], scalers[scope_key]['local_stem']['q_high'])
                    y_seg = np.stack([temp_hr, rh_hr, stem_hr], axis=-1).astype(np.float32)
                    if complete_only and (np.isnan(X_seg).any() or np.isnan(y_seg).any()):
                        continue
                    X_list.append(X_seg)
                    Y_list.append(y_seg)
                    META_list.append({'site_id': site_id, 'window_start': str(ws), 'window_end': str(we), 'thermometer_id': int(t_id), 'hygrometer_id': int(h_id), 'dendrometer_id': int(d_id)})
                    generated += 1
                    if (input_mode == 'best' and generated >= 1) or ((max_combos_per_site is not None) and (generated >= max_combos_per_site)):
                        break
                if (input_mode == 'best' and generated >= 1):
                    break
    return X_list, Y_list, META_list

def build_datasets_30d(
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
    local_tz: str = 'Europe/Zurich',
    min_local_coverage: float = 0.5,
    allow_missing_locals: bool = False,
    complete_only: bool = True,
    per_year: bool = True,
    input_mode: str = 'combinations',
    max_combos_per_site: t.Optional[int] = None,
    window_days: int = 30,
    window_stride_days: int = 1,
) -> None:
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'diagnostics'), exist_ok=True)
    site_scalers: dict[int,dict] = {}
    normalizers_rows: t.List[dict] = []
    for sid in site_ids_train:
        scalers = fit_site_scalers(sid, years, meteo_dir, per_year=per_year)
        site_scalers[sid] = scalers
        norm_dir = os.path.join(out_root, 'normalizers'); os.makedirs(norm_dir, exist_ok=True)
        with open(os.path.join(norm_dir, f"normalizers_site_{sid}.json"), 'w', encoding='utf-8') as f:
            json.dump(scalers, f, indent=2)
        for scope, chmap in scalers.items():
            for ch, qq in chmap.items():
                normalizers_rows.append({'site_id': sid, 'scope': scope, 'channel': ch, 'q_low': qq['q_low'], 'q_high': qq['q_high']})

    X_tr_list: t.List[np.ndarray] = []
    y_tr_list: t.List[np.ndarray] = []
    sid_tr_list: t.List[int] = []
    meta_rows: t.List[dict] = []
    diagnostics_rows: t.List[dict] = []
    for sid in site_ids_train:
        X_list, Y_list, META_list = build_segments_for_site_30d(
            site_id=sid, years=years, local_tz=local_tz,
            meteo_dir=meteo_dir, thermo_dir=thermo_dir, hygro_dir=hygro_dir, dendro_l2_dir=dendro_l2_dir, dendro_lm_dir=dendro_lm_dir,
            metadata_path=metadata_pickle, scalers=site_scalers[sid],
            min_local_coverage=min_local_coverage, allow_missing_locals=allow_missing_locals,
            complete_only=complete_only, input_mode=input_mode, max_combos_per_site=max_combos_per_site,
            window_days=window_days, window_stride_days=window_stride_days,
        )
        print(f"Site {sid} (TRAIN): kept {len(X_list)} segments")
        X_tr_list.extend(X_list); y_tr_list.extend(Y_list); sid_tr_list.extend([sid]*len(X_list)); meta_rows.extend(META_list)
    if X_tr_list:
        first_len = X_tr_list[0].shape[0]
        assert all(x.shape[0] == first_len for x in X_tr_list), 'Mixed segment lengths; run one window length per execution.'
        X_train = np.stack(X_tr_list, axis=0)
        y_train = np.stack(y_tr_list, axis=0)
    else:
        X_train = np.empty((0, 0, len(GLOBAL_COLS)+3), dtype=np.float32)
        y_train = np.empty((0, 0, 3), dtype=np.float32)
    SID_train = np.array(sid_tr_list, dtype=np.int32)

    suffix = f"{window_days}d"
    x_train_fn = os.path.join(out_root, f'X_train_{suffix}.npy')
    y_train_fn = os.path.join(out_root, f'y_train_{suffix}.npy')
    sid_train_fn = os.path.join(out_root, f'site_ids_train_{suffix}.npy')
    ids_train_fn = os.path.join(out_root, f'train_identifiers_{suffix}.csv')
    np.save(x_train_fn, X_train)
    np.save(y_train_fn, y_train)
    np.save(sid_train_fn, SID_train)
    pd.DataFrame(meta_rows).to_csv(ids_train_fn, index=False)
    print(f"Saved TRAIN arrays: {x_train_fn} {X_train.shape}, {y_train_fn} {y_train.shape}, {sid_train_fn} {SID_train.shape}")
    if X_train.size == 0:
        print('WARNING: No TRAIN segments produced. Consider lowering --min_local_coverage or disabling --complete_only.')

    if site_ids_test is not None:
        X_te_list: t.List[np.ndarray] = []
        y_te_list: t.List[np.ndarray] = []
        sid_te_list: t.List[int] = []
        meta_rows_te: t.List[dict] = []
        for sid in site_ids_test:
            scalers_te = fit_site_scalers(sid, years, meteo_dir, per_year=per_year)
            X_list, Y_list, META_list = build_segments_for_site_30d(
                site_id=sid, years=years, local_tz=local_tz,
                meteo_dir=meteo_dir, thermo_dir=thermo_dir, hygro_dir=hygro_dir, dendro_l2_dir=dendro_l2_dir, dendro_lm_dir=dendro_lm_dir,
                metadata_path=metadata_pickle, scalers=scalers_te,
                min_local_coverage=min_local_coverage, allow_missing_locals=allow_missing_locals,
                complete_only=complete_only, input_mode=input_mode, max_combos_per_site=max_combos_per_site,
                window_days=window_days, window_stride_days=window_stride_days,
            )
            print(f"Site {sid} (TEST): kept {len(X_list)} segments")
            X_te_list.extend(X_list); y_te_list.extend(Y_list); sid_te_list.extend([sid]*len(X_list)); meta_rows_te.extend(META_list)
        if X_te_list:
            first_len = X_te_list[0].shape[0]
            assert all(x.shape[0] == first_len for x in X_te_list)
            X_test = np.stack(X_te_list, axis=0)
            y_test = np.stack(y_te_list, axis=0)
        else:
            X_test = np.empty((0, 0, len(GLOBAL_COLS)+3), dtype=np.float32)
            y_test = np.empty((0, 0, 3), dtype=np.float32)
        SID_test = np.array(sid_te_list, dtype=np.int32)
        x_test_fn = os.path.join(out_root, f'X_test_{suffix}.npy')
        y_test_fn = os.path.join(out_root, f'y_test_{suffix}.npy')
        sid_test_fn = os.path.join(out_root, f'site_ids_test_{suffix}.npy')
        ids_test_fn = os.path.join(out_root, f'test_identifiers_{suffix}.csv')
        np.save(x_test_fn, X_test)
        np.save(y_test_fn, y_test)
        np.save(sid_test_fn, SID_test)
        pd.DataFrame(meta_rows_te).to_csv(ids_test_fn, index=False)
        print(f"Saved TEST arrays: {x_test_fn} {X_test.shape}, {y_test_fn} {y_test.shape}, {sid_test_fn} {SID_test.shape}")
        if X_test.size == 0:
            print('WARNING: No TEST segments produced. Consider lowering --min_local_coverage or disabling --complete_only.')

    pd.DataFrame(normalizers_rows).to_csv(os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'), index=False)
    pd.DataFrame(diagnostics_rows).to_csv(os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'), index=False)
    print("Wrote diagnostics:")
    print(" -", os.path.join(out_root, 'diagnostics', 'diagnostics_preprocessing.csv'))
    print(" -", os.path.join(out_root, 'diagnostics', 'normalizers_summary.csv'))
    print("Build completed.")

def parse_args():
    p = argparse.ArgumentParser(description='TreeNet 30-day preprocessing (UTC-only locals/LM, Strategy-B globals, RAW LM only).')
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
    p.add_argument('--min_local_coverage', type=float, default=0.5)
    p.add_argument('--min_lm_series', type=int, default=1)
    p.add_argument('--overlap_days', type=int, default=10)
    p.add_argument('--allow_missing_locals', type=str, default='false')
    p.add_argument('--globals_broadcast_strategy', type=str, default='civil_map', choices=['civil_map'])
    p.add_argument('--run_tests', action='store_true', help='Run built-in Strategy B unit test and exit')
    p.add_argument('--window_days', type=int, default=30)
    p.add_argument('--window_stride_days', type=int, default=1)
    p.add_argument('--complete_only', type=str, default='true')
    return p.parse_args()

def read_site_ids_csv(path: str) -> t.List[int]:
    df = pd.read_csv(path)
    if 'site_id' not in df.columns:
        raise ValueError('CSV must contain site_id')
    return [int(x) for x in df['site_id'].tolist()]

if __name__ == '__main__':
    args = parse_args()
    site_ids_train = read_site_ids_csv(args.train_site_ids_csv)
    site_ids_test = read_site_ids_csv(args.test_site_ids_csv) if args.test_site_ids_csv else None
    build_datasets_30d(
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
        local_tz=args.tz,
        min_local_coverage=args.min_local_coverage,
        allow_missing_locals=(args.allow_missing_locals.lower()=='true'),
        complete_only=(args.complete_only.lower()=='true'),
        per_year=(args.per_year.lower()=='true'),
        input_mode=args.input_mode,
        max_combos_per_site=args.max_combos_per_site,
        window_days=args.window_days,
        window_stride_days=args.window_stride_days,
    )
