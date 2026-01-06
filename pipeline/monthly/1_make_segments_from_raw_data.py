
# build_30day_dropin_utc.py
# -*- coding: utf-8 -*-

"""
Drop-in 30-day segment builder (UTC-safe)

- Locals (10-min): Europe/Zurich -> UTC, resample to 10-min, align to full-year 10-min UTC grid
- LM (hourly): convert to local civil time, floor to local hour, aggregate, convert to UTC, align to full-year hourly grid
- Globals (daily): Strategy-B by local civil day -> broadcast to 10-min UTC grid
- Segmentation: strict 30-day completeness using "jump past last NaN"
- Normalization: computed on the entire year, then applied to segments
- Traceability: same intermediate files + segment_ids.pkl with minima/diffs and channel origins
"""

from __future__ import annotations
import os
import re
import argparse
import typing as t
import numpy as np
import pandas as pd
import pickle

# -----------------------------------------------------------------------------
# Constants & channel sets
# -----------------------------------------------------------------------------
FREQ_10M = '10min'
FREQ_1H  = '1h'
LOCAL_TZ = 'Europe/Zurich'

INPUT_CHANNELS_10M = [
    'temp_treenet','rh_treenet','stem',
    'tas','tasmax','tasmin','rh','vpd','gh','pr','doy'
]
TARGET_CHANNELS_1H = ['local_T','local_RH','stem']

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# -----------------------------------------------------------------------------
# Timestamp helpers
# -----------------------------------------------------------------------------
def to_utc_index(series_ts: pd.Series, local_tz: str) -> pd.DatetimeIndex:
    """
    Convert a ts Series to a tz-aware UTC DatetimeIndex via local_tz.
    Works for both tz-naive and tz-aware inputs.
    """
    dt_index = pd.DatetimeIndex(series_ts)  # force DatetimeIndex first
    if getattr(dt_index, 'tz', None) is None:
        dt_index = dt_index.tz_localize(local_tz, nonexistent='shift_forward', ambiguous='NaT')
    else:
        try:
            dt_index = dt_index.tz_convert(local_tz)
        except Exception:
            # Safe hop through UTC if direct convert fails
            dt_index = dt_index.tz_convert('UTC').tz_convert(local_tz)
    return pd.DatetimeIndex(dt_index.tz_convert('UTC'))

def to_local_series(series_ts: pd.Series, local_tz: str) -> pd.Series:
    """
    Returns a tz-aware Series in local_tz.
    Handles tz-aware (e.g., Etc/GMT-1) and tz-naive inputs safely.
    """
    dt_index = pd.DatetimeIndex(series_ts)  # force DatetimeIndex
    if getattr(dt_index, 'tz', None) is None:
        dt_index = dt_index.tz_localize(local_tz, nonexistent='shift_forward', ambiguous='NaT')
    else:
        try:
            dt_index = dt_index.tz_convert(local_tz)
        except Exception:
            dt_index = dt_index.tz_convert('UTC').tz_convert(local_tz)
    return pd.Series(dt_index)

# -----------------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------------
def read_feather_local_to_utc(path: str, value_col: str, rename_to: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    ts_col = 'ts' if 'ts' in df.columns else next((c for c in ('timestamp','time','date_time','datetime') if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"Missing timestamp column in {path}")
    ts_utc = to_utc_index(df[ts_col], LOCAL_TZ)
    out = pd.DataFrame({'ts': ts_utc, rename_to: pd.to_numeric(df[value_col], errors='coerce')}).sort_values('ts')
    return out

# -----------------------------------------------------------------------------
# Globals (daily) Strategy-B: civil day -> broadcast to UTC 10-min grid
# -----------------------------------------------------------------------------
def discover_meteo_files(meteo_dir: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not os.path.isdir(meteo_dir):
        return mapping
    for fn in os.listdir(meteo_dir):
        if fn.endswith('.csv'):
            m = re.findall(r'\d+', fn)
            if m:
                mapping[int(m[0])] = os.path.join(meteo_dir, fn)
    return mapping

def load_global_daily(site_id: int, meteo_dir: str) -> pd.DataFrame:
    files = discover_meteo_files(meteo_dir)
    if site_id not in files:
        raise FileNotFoundError(f'No meteo CSV for site {site_id} in {meteo_dir}')
    df = pd.read_csv(files[site_id])
    if 'ts' not in df.columns:
        raise ValueError('Global meteo CSV must have ts column')

    ts_local = to_local_series(df['ts'], LOCAL_TZ)  # local tz-aware series
    civil_day = ts_local.dt.normalize()
    df = df.assign(civil_day=civil_day)

    for src in ['tas','tasmax','tasmin','rh','vpd','gh','pr']:
        if src not in df.columns:
            df[src] = np.nan

    out = (df[['civil_day','tas','tasmax','tasmin','rh','vpd','gh','pr']]
           .dropna(subset=['civil_day'])
           .drop_duplicates(subset=['civil_day'], keep='last')
           .set_index('civil_day')
           .sort_index())
    return out

def broadcast_daily_to_10m_utc(meteo_daily: pd.DataFrame, idx_10m_utc: pd.DatetimeIndex) -> pd.DataFrame:
    civil_for_grid = pd.DatetimeIndex(idx_10m_utc).tz_convert(LOCAL_TZ).normalize()
    out = meteo_daily.reindex(civil_for_grid).ffill()
    out.index = idx_10m_utc
    return out

# -----------------------------------------------------------------------------
# LM hourly (targets): local hour floor -> aggregate -> UTC -> hourly grid
# -----------------------------------------------------------------------------
def lm_hourly_local_to_utc(df_lm_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robust hourly builder for tz-aware inputs like 'Etc/GMT-1':
      - Convert to local civil time (Europe/Zurich) with a UTC hop if needed
      - Floor to local hour and aggregate means (prefer resample, fallback groupby)
      - Convert hourly index to UTC, guarantee DatetimeIndex
    """
    if 'ts' not in df_lm_raw.columns:
        raise ValueError('LM raw frame must contain ts')

    ts_local = to_local_series(df_lm_raw['ts'], LOCAL_TZ)

    # Numeric columns
    df = pd.DataFrame({
        'ts_local': ts_local,
        'stem':    pd.to_numeric(df_lm_raw.get('value', pd.Series([np.nan]*len(df_lm_raw))), errors='coerce'),
        'local_T': pd.to_numeric(df_lm_raw.get('temp',  pd.Series([np.nan]*len(df_lm_raw))), errors='coerce'),
        'local_RH':pd.to_numeric(df_lm_raw.get('rh',    pd.Series([np.nan]*len(df_lm_raw))), errors='coerce'),
    }).dropna(subset=['ts_local'])

    df_idx = df.set_index('ts_local')
    # Try resample (fast & clean). If it fails, fallback to groupby on floored hour.
    try:
        hourly = df_idx.resample('1h')[['stem','local_T','local_RH']].mean()
    except Exception:
        df['ts_local_hour'] = df['ts_local'].dt.floor('h')
        hourly = df.groupby('ts_local_hour')[['stem','local_T','local_RH']].mean()

    # Ensure DatetimeIndex + tz-aware local (if lost), then convert to UTC
    if not isinstance(hourly.index, pd.DatetimeIndex):
        hourly.index = pd.to_datetime(hourly.index, errors='coerce')
    if getattr(hourly.index, 'tz', None) is None:
        hourly.index = hourly.index.tz_localize(LOCAL_TZ, nonexistent='shift_forward', ambiguous='NaT')
    hourly.index = hourly.index.tz_convert('UTC')

    return hourly.sort_index()

# -----------------------------------------------------------------------------
# Year index builders & feature columns
# -----------------------------------------------------------------------------
def make_year_10m_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    return pd.date_range(start=start, end=end, freq=FREQ_10M)

def make_year_hourly_index_utc(years: t.Sequence[int]) -> pd.DatetimeIndex:
    y0, y1 = min(years), max(years)
    start = pd.Timestamp(f"{y0}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{y1}-12-31 23:59:59", tz='UTC')
    return pd.date_range(start=start, end=end, freq=FREQ_1H)

def add_local_civil_columns_10m(idx_10m_utc: pd.DatetimeIndex) -> pd.DataFrame:
    local = idx_10m_utc.tz_convert(LOCAL_TZ)
    df = pd.DataFrame(index=idx_10m_utc)
    df['doy']   = local.dayofyear
    df['month'] = local.month
    df['hour']  = local.hour
    return df

# -----------------------------------------------------------------------------
# Normalization (year-level)
# -----------------------------------------------------------------------------
def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    if len(df.shape) != 2:
        raise Exception(f'normalize_dataframe: expected 2D dataframe, got shape {df.shape}')
    numeric_cols = df.select_dtypes(include='number').columns
    minima: dict = {}
    diffs: dict = {}
    out = df.copy()
    for col in numeric_cols:
        if col == 'hour':
            out[col] = df[col] / 24.0
            minima[col] = 0.0
            diffs[col] = 24.0
        elif col == 'doy':
            out[col] = df[col] / 365.0
            minima[col] = 1.0
            diffs[col] = 365.0
        elif col == 'month':
            out[col] = df[col] / 12.0
            minima[col] = 0.0
            diffs[col] = 12.0
        else:
            series = pd.to_numeric(df[col], errors='coerce')
            min_val = series.min(skipna=True)
            max_val = series.max(skipna=True)
            minima[col] = float(min_val) if pd.notna(min_val) else np.nan
            if pd.isna(min_val) or pd.isna(max_val):
                out[col] = series
                diffs[col] = np.nan
            else:
                diff = max_val - min_val
                if abs(diff) > 1e-4:
                    out[col] = (series - min_val) / diff
                    diffs[col] = float(diff)
                else:
                    out[col] = (series - min_val)
                    diffs[col] = 1.0
    return out, minima, diffs

# -----------------------------------------------------------------------------
# Segmentation: jump past the last NaN (strict completeness)
# -----------------------------------------------------------------------------
def find_complete_30d_windows_jump(
    idx_10m: pd.DatetimeIndex,
    inputs_ok_10m: np.ndarray,
    idx_1h: pd.DatetimeIndex,
    targets_ok_1h: np.ndarray,
    window_days: int = 30,
    stride_days_after_accept: int = 10,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(idx_10m) == 0 or len(idx_1h) == 0:
        return []
    step_10m = pd.Timedelta(minutes=10)
    stride   = pd.Timedelta(days=stride_days_after_accept)
    W        = pd.Timedelta(days=window_days)

    ws = idx_10m[0].floor('10min')
    end_excl_10m = idx_10m[-1] + step_10m
    wins: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    while ws + W <= end_excl_10m:
        we = ws + W
        in_mask = (idx_10m >= ws) & (idx_10m < we)
        hr_mask = (idx_1h  >= ws) & (idx_1h  < we)
        if not in_mask.any() or not hr_mask.any():
            ws += step_10m
            continue
        ok_in = inputs_ok_10m[in_mask]
        ok_tg = targets_ok_1h[hr_mask]
        if ok_in.all() and ok_tg.all():
            wins.append((ws, we))
            ws = ws + stride
            continue
        last_bad = []
        if (~ok_in).any():
            last_bad.append(idx_10m[in_mask][np.flatnonzero(~ok_in)[-1]])
        if (~ok_tg).any():
            last_bad.append(idx_1h[hr_mask][np.flatnonzero(~ok_tg)[-1]])
        ws = (max(last_bad) + step_10m) if last_bad else (ws + step_10m)
    return wins

# -----------------------------------------------------------------------------
# Robust locals -> 10-min -> year grid
# -----------------------------------------------------------------------------
def to_full_10m(df: pd.DataFrame, col: str, idx_10m_year: pd.DatetimeIndex) -> pd.Series:
    s = df.set_index('ts')[col]

    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.DatetimeIndex(s.index)
        except Exception as e:
            raise RuntimeError(f'Failed to build DatetimeIndex for {col}: {e}')

    try:
        if getattr(s.index, 'tz', None) is None:
            s.index = s.index.tz_localize('UTC')
        else:
            s.index = s.index.tz_convert('UTC')
    except Exception:
        s.index = pd.to_datetime(s.index, utc=True)

    s = s.sort_index()

    try:
        s10 = s.resample(FREQ_10M).median()
    except Exception:
        s10 = s.copy()
        try:
            s10.index = s10.index.tz_convert('UTC')
        except Exception:
            pass
        s10.index = pd.DatetimeIndex(s10.index.tz_localize(None))
        s10 = s10.resample(FREQ_10M).median()
        s10.index = s10.index.tz_localize('UTC')

    s10 = s10.reindex(idx_10m_year)
    return s10

# -----------------------------------------------------------------------------
# Discovery of series IDs by directory
# -----------------------------------------------------------------------------
def discover_series_ids(dir_path: str, prefix: str) -> set[int]:
    ids: set[int] = set()
    if not os.path.isdir(dir_path):
        return ids
    regex = re.compile(rf'{re.escape(prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m:
            ids.add(int(m.group(1)))
    return ids

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def build_30d_segments(
    out_root: str,
    metadata_pickle: str,
    meteo_dir: str,
    thermo_dir: str,
    hygro_dir: str,
    dendro_l2_dir: str,
    dendro_lm_dir: str,
    train_site_ids_csv: str,
    test_site_ids_csv: str | None,
    years: t.Sequence[int],
    window_days: int = 30,
    stride_days_after_accept: int = 10,
) -> None:
    proc_dir = os.path.join(out_root, 'processed', 'model_data')
    diag_dir = os.path.join(out_root, 'processed', 'diagnostics')
    for d in (proc_dir, diag_dir):
        ensure_dir(d)

    metadata = pd.read_pickle(metadata_pickle)
    train_sites = pd.read_csv(train_site_ids_csv)['site_id'].astype(int).tolist()
    test_sites  = pd.read_csv(test_site_ids_csv)['site_id'].astype(int).tolist() if test_site_ids_csv else []

    d2_ids = discover_series_ids(dendro_l2_dir, 'dendrometer_l2')
    lm_ids = discover_series_ids(dendro_lm_dir, 'dendrometer_lm')
    T_ids  = discover_series_ids(thermo_dir,      'thermometer_l1')
    RH_ids = discover_series_ids(hygro_dir,       'hygrometer_l1')

    def ids_by_site(ids: set[int]) -> dict[int, list[int]]:
        rows = metadata[metadata['series_id'].isin(ids)]
        out: dict[int, list[int]] = {}
        for site, g in rows.groupby('site_id'):
            out[int(site)] = [int(x) for x in g['series_id'].tolist()]
        return out

    d2_by_site = ids_by_site(d2_ids)
    lm_by_site = ids_by_site(lm_ids)
    T_by_site  = ids_by_site(T_ids)
    RH_by_site = ids_by_site(RH_ids)

    idx_10m_year = make_year_10m_index_utc(years)
    idx_1h_year  = make_year_hourly_index_utc(years)
    aux_10m_cols = add_local_civil_columns_10m(idx_10m_year)

    diag_rows: list[dict] = []

    def process_sites(sites: list[int], split_tag: str) -> None:
        combo_ids = {}
        combo_counter = 0

        seg_inputs_pandas: dict[int, list[pd.DataFrame]] = {}
        seg_outputs_pandas: dict[int, list[pd.DataFrame]] = {}
        numpy_inputs: list[np.ndarray] = []
        numpy_outputs: list[np.ndarray] = []
        segment_ids: dict[int, list] = {}

        for site in sites:
            T_list  = T_by_site.get(site, [])
            RH_list = RH_by_site.get(site, [])
            d2_list = d2_by_site.get(site, [])
            lm_list = lm_by_site.get(site, [])
            dendro_ids = [d for d in d2_list if d in lm_list]
            if (not T_list) or (not RH_list) or (not dendro_ids):
                diag_rows.append({'site_id': site, 'split': split_tag, 'reason': 'missing_instruments'})
                print(f"Site {site} ({split_tag}): skipped due to missing instruments")
                continue

            kept_segments_for_site = 0

            for t_id in T_list:
                for h_id in RH_list:
                    for d_id in dendro_ids:
                        try:
                            df_T  = read_feather_local_to_utc(os.path.join(thermo_dir, f'thermometer_l1_series_id_{t_id}.ftr'), 'value', 'temp_treenet')
                            df_RH = read_feather_local_to_utc(os.path.join(hygro_dir,  f'hygrometer_l1_series_id_{h_id}.ftr'), 'value', 'rh_treenet')
                            df_ST = read_feather_local_to_utc(os.path.join(dendro_l2_dir,f'dendrometer_l2_series_id_{d_id}.ftr'), 'value', 'stem')
                        except Exception as e:
                            diag_rows.append({'site_id': site, 'split': split_tag, 'reason': 'read_error_locals', 'details': str(e)})
                            continue

                        try:
                            sT  = to_full_10m(df_T,  'temp_treenet', idx_10m_year)
                            sRH = to_full_10m(df_RH, 'rh_treenet',   idx_10m_year)
                            sST = to_full_10m(df_ST, 'stem',         idx_10m_year)
                        except Exception as e:
                            diag_rows.append({'site_id': site, 'split': split_tag, 'reason': 'resample_error_locals', 'details': str(e)})
                            continue

                        try:
                            meteo_daily = load_global_daily(site, meteo_dir)
                            meteo_10m = broadcast_daily_to_10m_utc(meteo_daily, idx_10m_year)
                        except Exception as e:
                            diag_rows.append({'site_id': site, 'split': split_tag, 'reason': 'meteo_error', 'details': str(e)})
                            continue

                        input_df = pd.DataFrame(index=idx_10m_year)
                        input_df['temp_treenet'] = sT
                        input_df['rh_treenet']   = sRH
                        input_df['stem']         = sST
                        for col in ['tas','tasmax','tasmin','rh','vpd','gh','pr']:
                            input_df[col] = pd.to_numeric(meteo_10m[col], errors='coerce')
                        input_df['doy'] = aux_10m_cols['doy']

                        # LM hourly robust
                        try:
                            df_lm_raw = pd.read_feather(os.path.join(dendro_lm_dir, f'dendrometer_lm_series_id_{d_id}.ftr'))
                            lm_hourly = lm_hourly_local_to_utc(df_lm_raw)
                            if not isinstance(lm_hourly.index, pd.DatetimeIndex):
                                lm_hourly.index = pd.to_datetime(lm_hourly.index, errors='coerce')
                            if getattr(lm_hourly.index, 'tz', None) is None:
                                lm_hourly.index = lm_hourly.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
                            lm_hourly = lm_hourly.sort_index().reindex(idx_1h_year)
                        except Exception as e:
                            diag_rows.append({'site_id': site, 'split': split_tag, 'reason': 'lm_error', 'details': str(e)})
                            continue

                        # Save intermediate combination files
                        combo_id = combo_counter
                        combo_ids[combo_id] = pd.DataFrame(
                            [[site, t_id, h_id, d_id]],
                            columns=['site ID','thermometer ID','hygrometer ID','dendrometer ID']
                        ).loc[0]
                        if split_tag == 'TRAIN':
                            input_df.reset_index().rename(columns={'index':'ts'}).to_feather(os.path.join(proc_dir, f'train_input_combination_{combo_id}.ftr'))
                            lm_hourly.reset_index().rename(columns={'index':'ts'}).to_feather(os.path.join(proc_dir, f'train_output_combination_{combo_id}.ftr'))
                        else:
                            input_df.reset_index().rename(columns={'index':'ts'}).to_feather(os.path.join(proc_dir, f'test_input_combination_{combo_id}.ftr'))
                            lm_hourly.reset_index().rename(columns={'index':'ts'}).to_feather(os.path.join(proc_dir, f'test_output_combination_{combo_id}.ftr'))

                        # Normalize on entire year
                        norm_input_df,  in_minima,  in_diffs  = normalize_dataframe(input_df)
                        norm_output_df, out_minima, out_diffs = normalize_dataframe(lm_hourly)

                        inputs_ok = (~norm_input_df['temp_treenet'].isna()).to_numpy() \
                                &  (~norm_input_df['rh_treenet'].isna()).to_numpy()   \
                                &  (~norm_input_df['stem'].isna()).to_numpy()         \
                                &  (~norm_input_df['tas'].isna()).to_numpy()          \
                                &  (~norm_input_df['tasmax'].isna()).to_numpy()       \
                                &  (~norm_input_df['tasmin'].isna()).to_numpy()       \
                                &  (~norm_input_df['rh'].isna()).to_numpy()           \
                                &  (~norm_input_df['vpd'].isna()).to_numpy()          \
                                &  (~norm_input_df['gh'].isna()).to_numpy()           \
                                &  (~norm_input_df['pr'].isna()).to_numpy()           \
                                &  (~norm_input_df['doy'].isna()).to_numpy()

                        targets_ok = (~norm_output_df[['local_T','local_RH','stem']].isna().any(axis=1)).to_numpy()

                        windows = find_complete_30d_windows_jump(
                            idx_10m=idx_10m_year,
                            inputs_ok_10m=inputs_ok,
                            idx_1h=idx_1h_year,
                            targets_ok_1h=targets_ok,
                            window_days=30,
                            stride_days_after_accept=stride_days_after_accept,
                        )

                        if not windows:
                            diag_rows.append({'site_id': site, 'split': split_tag, 'combo_id': combo_id, 'reason': 'no_complete_windows'})
                            combo_counter += 1
                            continue

                        seg_inputs_pandas[combo_id] = []
                        seg_outputs_pandas[combo_id] = []
                        segment_ids[combo_id] = []

                        for seg_idx, (ws, we) in enumerate(windows):
                            X_seg = norm_input_df[(norm_input_df.index >= ws) & (norm_input_df.index < we)][INPUT_CHANNELS_10M]
                            Y_seg = norm_output_df[(norm_output_df.index >= ws) & (norm_output_df.index < we)][TARGET_CHANNELS_1H]

                            if (X_seg.shape[0] != 30*24*6) or (Y_seg.shape[0] != 30*24):
                                diag_rows.append({
                                    'site_id': site, 'split': split_tag, 'combo_id': combo_id,
                                    'reason': 'length_mismatch', 'ws': str(ws), 'we': str(we),
                                    'len_X': int(X_seg.shape[0]), 'len_Y': int(Y_seg.shape[0])
                                })
                                continue

                            seg_inputs_pandas[combo_id].append(X_seg.copy())
                            seg_outputs_pandas[combo_id].append(Y_seg.copy())
                            numpy_inputs.append(np.array(X_seg.fillna(-1.0), dtype=np.float32))
                            numpy_outputs.append(np.array(Y_seg.fillna(-1.0), dtype=np.float32))

                            segment_ids[combo_id].append([
                                combo_id,
                                seg_idx,
                                combo_ids[combo_id],
                                in_minima,
                                in_diffs,
                                out_minima,
                                out_diffs,
                                {'window_start_utc': ws.isoformat(), 'window_end_utc': we.isoformat()},
                                {'input_channels': INPUT_CHANNELS_10M, 'target_channels': TARGET_CHANNELS_1H}
                            ])
                            kept_segments_for_site += 1

                        combo_counter += 1

            print(f"Site {site} ({split_tag}): kept {kept_segments_for_site} segments")

        with open(os.path.join(proc_dir, f'model_{split_tag.lower()}_data_combination_ids.pkl'), 'wb') as f:
            pickle.dump(combo_ids, f)
        with open(os.path.join(proc_dir, f'{split_tag.lower()}_input_segments.pkl'), 'wb') as f:
            pickle.dump(seg_inputs_pandas, f)
        with open(os.path.join(proc_dir, f'{split_tag.lower()}_output_segments.pkl'), 'wb') as f:
            pickle.dump(seg_outputs_pandas, f)

        X_np = np.stack(numpy_inputs, axis=0) if numpy_inputs else np.empty((0, 30*24*6, len(INPUT_CHANNELS_10M)), dtype=np.float32)
        Y_np = np.stack(numpy_outputs, axis=0) if numpy_outputs else np.empty((0, 30*24, len(TARGET_CHANNELS_1H)), dtype=np.float32)
        with open(os.path.join(proc_dir, f'{split_tag.lower()}_input_segments_numpy.pkl'), 'wb') as f:
            pickle.dump(X_np, f)
        with open(os.path.join(proc_dir, f'{split_tag.lower()}_output_segments_numpy.pkl'), 'wb') as f:
            pickle.dump(Y_np, f)
        with open(os.path.join(proc_dir, f'{split_tag.lower()}_segment_ids.pkl'), 'wb') as f:
            pickle.dump(segment_ids, f)

        print(f"Saved {split_tag} numpy arrays: X {X_np.shape}, Y {Y_np.shape}")

    if train_sites:
        process_sites(train_sites, 'TRAIN')
    if test_sites:
        process_sites(test_sites, 'TEST')

    pd.DataFrame(diag_rows).to_csv(os.path.join(diag_dir, 'diagnostics_preprocessing.csv'), index=False)
    print("Wrote diagnostics:")
    print(" -", os.path.join(diag_dir, 'diagnostics_preprocessing.csv'))
    print("Build completed.")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Drop-in 30-day segment builder with UTC-safe pipeline and year-level normalization.')
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
    p.add_argument('--window_days', type=int, default=30)
    p.add_argument('--stride_days_after_accept', type=int, default=10)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    build_30d_segments(
        out_root=args.out_root,
        metadata_pickle=args.metadata_pickle,
        meteo_dir=args.meteo_dir,
        thermo_dir=args.thermo_dir,
        hygro_dir=args.hygro_dir,
        dendro_l2_dir=args.dendro_l2_dir,
        dendro_lm_dir=args.dendro_lm_dir,
        train_site_ids_csv=args.train_site_ids_csv,
        test_site_ids_csv=args.test_site_ids_csv,
        years=args.years,
        window_days=args.window_days,
        stride_days_after_accept=args.stride_days_after_accept,
    )
