# build_coverage_report_treenet.py
# -*- coding: utf-8 -*-
"""
TreeNet Data Coverage Report (DST‑Safe, Time‑Axis Configurable, Fully Annotated)
==============================================================================

This module computes **per‑site, per‑instrument, per‑year coverage** for TreeNet sensors.
You can choose the **time axis** used for all alignment and indexing via `--time_axis`:

- `local_dst`   → Europe/Zurich (civil local time **with** DST)  
- `fixed_winter`→ CET (UTC+01:00, **no DST**) — “winter time only”  
- `utc`         → UTC (no DST), best for internal computation  

Key points
---------
- For `local_dst`, we round **in UTC** and convert back, resolving the fall‑back hour via `--fold_rule`.
- For `fixed_winter` (CET) and `utc`, we round directly in that zone — no DST ambiguity ever occurs.
- All yearly grids (10‑min and hourly) are generated **in the selected time axis**.

Outputs
-------
- coverage/instruments_available.csv
- coverage/coverage_by_instrument_year.csv
- coverage/coverage_by_site_year.csv
- coverage/plots/site_<id>_coverage.png
- coverage/debug/debug_first_last.csv (when --debug true)
"""

from __future__ import annotations
import os
import re
import argparse
import typing as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Time‑axis helpers
# =====================

def tz_for_axis(time_axis: str, local_tz: str = 'Europe/Zurich') -> str:
    """Return the timezone string used for indexing given a `time_axis`.
    - 'local_dst'   → local_tz (Europe/Zurich)
    - 'fixed_winter'→ 'CET' (UTC+01:00, no DST)
    - 'utc'         → 'UTC'
    """
    if time_axis == 'local_dst':
        return local_tz
    elif time_axis == 'fixed_winter':
        return 'CET'
    elif time_axis == 'utc':
        return 'UTC'
    else:
        raise ValueError(f"Unknown time_axis: {time_axis}")

# =====================
# Index helpers
# =====================

def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove Feb 29 to keep fixed 365‑day grids (useful for year‑long arrays)."""
    return idx[~((idx.month == 2) & (idx.day == 29))]


def make_year_10m_index_axis(year: int, time_axis: str, local_tz: str = 'Europe/Zurich') -> pd.DatetimeIndex:
    """Create a **10‑minute grid** for a given year in the **selected time axis**."""
    tz = tz_for_axis(time_axis, local_tz)
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz)
    end   = pd.Timestamp(f"{year}-12-31 23:59:59", tz=tz)
    idx = pd.date_range(start=start, end=end, freq='10min')
    return strip_leap_days(idx)


def make_year_hourly_index_axis(year: int, time_axis: str, local_tz: str = 'Europe/Zurich') -> pd.DatetimeIndex:
    """Create an **hourly grid** for a given year in the **selected time axis**."""
    tz = tz_for_axis(time_axis, local_tz)
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz)
    end   = pd.Timestamp(f"{year}-12-31 23:59:59", tz=tz)
    idx = pd.date_range(start=start, end=end, freq='1H')
    return strip_leap_days(idx)

# =====================
# Discovery helpers
# =====================

def discover_series_ids(dir_path: str, pattern_prefix: str) -> t.Set[int]:
    """Scan `dir_path` for files `{pattern_prefix}_series_id_<id>.ftr` and return the set of IDs."""
    ids: set[int] = set()
    regex = re.compile(rf'{re.escape(pattern_prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m:
            ids.add(int(m.group(1)))
    return ids


def series_by_site(metadata_df: pd.DataFrame, series_ids: t.Set[int]) -> dict[int, t.List[int]]:
    """Map a set of `series_ids` to `site_id` using metadata (assumes columns: `series_id`, `site_id`)."""
    df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
    out: dict[int, t.List[int]] = {}
    for site, g in df.groupby('site_id'):
        out[int(site)] = [int(x) for x in g['series_id'].tolist()]
    return out

# =====================
# Robust readers (flexible value selection)
# =====================

def _pick_value_column(df: pd.DataFrame, preferred: t.Sequence[str] = ()) -> str:
    """Choose a numeric measurement column in `df`.
    Priority: any in `preferred` → 'value' → numeric column with max non‑null count.
    """
    for c in preferred:
        if c in df.columns:
            return c
    if 'value' in df.columns:
        return 'value'
    candidates: list[tuple[int,str]] = []
    for c in df.columns:
        if c.lower() in ('ts','time','timestamp','date_time','datetime'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append((int(df[c].notna().sum()), c))
    if not candidates:
        raise ValueError('No numeric value column found')
    candidates.sort(reverse=True)
    return candidates[0][1]


def read_feather_series_flexible(series_id: int, dir_path: str, tz: str, sensor_hint: str | None = None) -> pd.Series:
    """Read a single **local instrument** series with flexible value selection and TZ normalization to `tz`."""
    pattern = re.compile(rf'.*series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(dir_path) if pattern.match(fn)]
    if not matches:
        raise FileNotFoundError(f"Series {series_id} not found in {dir_path}")
    fp = os.path.join(dir_path, matches[0])
    df = pd.read_feather(fp)

    ts_col = None
    for c in ('ts','timestamp','time','date_time','datetime'):
        if c in df.columns:
            ts_col = c; break
    if ts_col is None:
        raise ValueError(f"{fp} has no timestamp column")

    preferred: tuple[str,...] = ()
    if sensor_hint == 'thermometer_l1':
        preferred = ('temp','temperature','value')
    elif sensor_hint == 'hygrometer_l1':
        preferred = ('rh','relhum','value')
    elif sensor_hint == 'dendrometer_l2':
        preferred = ('value','radius','rad','l2')
    val_col = _pick_value_column(df, preferred)

    ts = pd.to_datetime(df[ts_col], utc=False)
    ts = ts.dt.tz_localize(tz) if getattr(ts.dt, 'tz', None) is None else ts.dt.tz_convert(tz)
    s = pd.Series(df[val_col].to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated(keep='mean')]
    return s


def read_lm_frame(series_id: int, lm_dir: str, tz: str) -> pd.DataFrame:
    """Read an LM dendrometer frame with tz‑aware index and columns `value`, `temp`, `rh`.
    Accepts common timestamp aliases and renames them to `ts`.
    """
    pattern = re.compile(rf'dendrometer_lm_series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(lm_dir) if pattern.match(fn)]
    if not matches:
        raise FileNotFoundError(f"LM series {series_id} not found in {lm_dir}")
    fp = os.path.join(lm_dir, matches[0])
    df = pd.read_feather(fp)

    if 'ts' not in df.columns:
        for c in ('timestamp','time','date_time','datetime'):
            if c in df.columns:
                df = df.rename(columns={c:'ts'})
                break
    if 'ts' not in df.columns:
        raise ValueError(f"LM file {fp} missing 'ts'")

    ts = pd.to_datetime(df['ts'], utc=False)
    ts = ts.dt.tz_localize(tz) if getattr(ts.dt, 'tz', None) is None else ts.dt.tz_convert(tz)
    df = df.set_index(ts)

    for col in ['value','temp','rh']:
        if col not in df.columns:
            df[col] = np.nan

    return df[['value','temp','rh']]

# =====================
# Alignment helpers (DST‑safe and axis‑aware)
# =====================

def align_to_grid_10min_axis(s: pd.Series,
                             time_axis: str,
                             fold_rule: str = 'mean',
                             local_tz: str = 'Europe/Zurich') -> pd.Series:
    """Align a 10‑min series to the selected `time_axis`.

    Behavior per axis:
    - 'utc':
        Ensure UTC index → round('10min') → deduplicate by mean.
    - 'fixed_winter' (CET):
        Ensure CET index → round('10min') → deduplicate by mean.
    - 'local_dst':
        Ensure Europe/Zurich index → convert to UTC → round('10min') → convert back →
        resolve duplicates (fall‑back) using `fold_rule`.

    `fold_rule` for 'local_dst':
    - 'earliest'/'first' → keep DST instance
    - 'latest'/'last'    → keep standard‑time instance
    - 'mean'/'sum'/'min'/'max' → reduce duplicates (and attach the label using 'earliest' by default)
    """
    if s.empty:
        return s
    s = s.copy()

    # Normalize to desired axis tz
    axis_tz = tz_for_axis(time_axis, local_tz)

    if time_axis == 'utc':
        # Ensure UTC tz
        idx = s.index
        if getattr(idx, 'tz', None) is None:
            idx = pd.to_datetime(idx, utc=False).tz_localize('UTC')
        else:
            idx = idx.tz_convert('UTC')
        s.index = idx
        # Round directly in UTC
        s.index = s.index.round('10min')
        s = s[~s.index.duplicated(keep='mean')]
        return s

    if time_axis == 'fixed_winter':
        # Convert everything to CET (no DST)
        idx = s.index
        if getattr(idx, 'tz', None) is None:
            idx = pd.to_datetime(idx, utc=False).tz_localize(local_tz, ambiguous='infer', nonexistent='shift_forward')
        else:
            idx = idx.tz_convert(local_tz)
        s.index = idx.tz_convert('CET')
        # Round in CET
        s.index = s.index.round('10min')
        s = s[~s.index.duplicated(keep='mean')]
        return s

    if time_axis == 'local_dst':
        # Ensure Europe/Zurich tz
        idx = s.index
        if getattr(idx, 'tz', None) is None:
            idx = pd.to_datetime(idx, utc=False).tz_localize(local_tz, ambiguous='infer', nonexistent='shift_forward')
        else:
            idx = idx.tz_convert(local_tz)
        s.index = idx
        # Round in UTC, convert back to local, resolve duplicates
        s_utc = s.tz_convert('UTC')
        s_utc.index = s_utc.index.round('10min')
        s_local = s_utc.tz_convert(local_tz)
        if fold_rule in ('earliest','first','latest','last'):
            keep = 'first' if fold_rule in ('earliest','first') else 'last'
            s_local = s_local[~s_local.index.duplicated(keep=keep)]
        elif fold_rule in ('mean','sum','min','max'):
            df = s_local.to_frame('v')
            g = df.groupby(df.index.tz_localize(None))
            reducer = {'mean': np.mean, 'sum': np.sum, 'min': np.min, 'max': np.max}[fold_rule]
            reduced = g.agg(reducer)
            # Re‑localize naive keys with explicit ambiguous resolution; default to 'earliest'
            reduced.index = reduced.index.tz_localize(local_tz, ambiguous='earliest', nonexistent='shift_forward')
            s_local = reduced['v']
        else:
            s_local = s_local[~s_local.index.duplicated(keep='first')]
        return s_local

    raise ValueError(f"Unexpected time_axis: {time_axis}")

# =====================
# Coverage computation
# =====================

def coverage_fraction_arr(v: np.ndarray) -> float:
    """Coverage = fraction of non‑NaN entries in a numeric array `v`."""
    if v.size == 0:
        return 0.0
    return float(np.sum(~np.isnan(v)) / v.size)


def compute_coverage_for_site(site_id: int,
                              years: t.Sequence[int],
                              time_axis: str,
                              local_tz: str,
                              metadata_df: pd.DataFrame,
                              thermo_dir: str, hygro_dir: str, d2_dir: str, lm_dir: str,
                              stem_mode_delta: bool = True,
                              debug_rows: t.Optional[list] = None,
                              fold_rule: str = 'mean') -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per‑instrument **per‑year coverage** for a single site.

    Parameters
    ----------
    time_axis : {'local_dst','fixed_winter','utc'}
        Controls timezone for alignment and indexing.
    local_tz : str
        Civil local timezone (e.g., 'Europe/Zurich') used when `time_axis='local_dst'`.
    fold_rule : str
        Duplicate resolution for fall‑back hour under `local_dst`.
    """
    # Discover instrument IDs
    thermo_ids_all = discover_series_ids(thermo_dir, 'thermometer_l1')
    hygro_ids_all  = discover_series_ids(hygro_dir,  'hygrometer_l1')
    d2_ids_all     = discover_series_ids(d2_dir,     'dendrometer_l2')
    lm_ids_all     = discover_series_ids(lm_dir,     'dendrometer_lm')

    thermo_by_site = series_by_site(metadata_df, thermo_ids_all)
    hygro_by_site  = series_by_site(metadata_df, hygro_ids_all)
    d2_by_site     = series_by_site(metadata_df, d2_ids_all)
    lm_by_site     = series_by_site(metadata_df, lm_ids_all)

    thermo_ids = thermo_by_site.get(site_id, [])
    hygro_ids  = hygro_by_site.get(site_id, [])
    d2_ids     = d2_by_site.get(site_id, [])
    lm_ids     = lm_by_site.get(site_id, [])

    rows_instr: t.List[dict] = []
    rows_site:  t.List[dict] = []

    for year in years:
        idx10 = make_year_10m_index_axis(year, time_axis, local_tz)
        idxH  = make_year_hourly_index_axis(year, time_axis, local_tz)

        # Thermometer L1
        for sid in thermo_ids:
            try:
                s = read_feather_series_flexible(sid, thermo_dir, local_tz, sensor_hint='thermometer_l1')
                s = align_to_grid_10min_axis(s, time_axis=time_axis, fold_rule=fold_rule, local_tz=local_tz)
                s = s.reindex(idx10)
                v = s.to_numpy(); cov = coverage_fraction_arr(v)
                rows_instr.append({'site_id': site_id, 'sensor_type': 'thermometer_l1', 'series_id': sid,
                                   'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                                   'samples_expected': int(idx10.size), 'coverage': cov})
                if debug_rows is not None and len(debug_rows) < 500:
                    debug_rows.append({'site_id': site_id, 'series_id': sid, 'sensor_type':'thermometer_l1', 'year': year,
                                       'first_ts': str(s.first_valid_index()), 'last_ts': str(s.last_valid_index()),
                                       'non_null': int(np.sum(~np.isnan(v)))})
            except Exception:
                rows_instr.append({'site_id': site_id, 'sensor_type': 'thermometer_l1', 'series_id': sid,
                                   'year': year, 'samples_present': 0, 'samples_expected': int(idx10.size), 'coverage': 0.0})

        # Hygrometer L1
        for sid in hygro_ids:
            try:
                s = read_feather_series_flexible(sid, hygro_dir, local_tz, sensor_hint='hygrometer_l1')
                s = align_to_grid_10min_axis(s, time_axis=time_axis, fold_rule=fold_rule, local_tz=local_tz)
                s = s.reindex(idx10)
                v = s.to_numpy(); cov = coverage_fraction_arr(v)
                rows_instr.append({'site_id': site_id, 'sensor_type': 'hygrometer_l1', 'series_id': sid,
                                   'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                                   'samples_expected': int(idx10.size), 'coverage': cov})
                if debug_rows is not None and len(debug_rows) < 500:
                    debug_rows.append({'site_id': site_id, 'series_id': sid, 'sensor_type':'hygrometer_l1', 'year': year,
                                       'first_ts': str(s.first_valid_index()), 'last_ts': str(s.last_valid_index()),
                                       'non_null': int(np.sum(~np.isnan(v)))})
            except Exception:
                rows_instr.append({'site_id': site_id, 'sensor_type': 'hygrometer_l1', 'series_id': sid,
                                   'year': year, 'samples_present': 0, 'samples_expected': int(idx10.size), 'coverage': 0.0})

        # Dendrometer L2
        for sid in d2_ids:
            try:
                s = read_feather_series_flexible(sid, d2_dir, local_tz, sensor_hint='dendrometer_l2')
                s = align_to_grid_10min_axis(s, time_axis=time_axis, fold_rule=fold_rule, local_tz=local_tz)
                if stem_mode_delta:
                    s = s.diff()
                s = s.reindex(idx10)
                v = s.to_numpy(); cov = coverage_fraction_arr(v)
                rows_instr.append({'site_id': site_id, 'sensor_type': 'dendrometer_l2', 'series_id': sid,
                                   'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                                   'samples_expected': int(idx10.size), 'coverage': cov})
                if debug_rows is not None and len(debug_rows) < 500:
                    debug_rows.append({'site_id': site_id, 'series_id': sid, 'sensor_type':'dendrometer_l2', 'year': year,
                                       'first_ts': str(s.first_valid_index()), 'last_ts': str(s.last_valid_index()),
                                       'non_null': int(np.sum(~np.isnan(v)))})
            except Exception:
                rows_instr.append({'site_id': site_id, 'sensor_type': 'dendrometer_l2', 'series_id': sid,
                                   'year': year, 'samples_present': 0, 'samples_expected': int(idx10.size), 'coverage': 0.0})

        # LM outputs (hourly)
        for sid in lm_ids:
            try:
                df = read_lm_frame(sid, lm_dir, local_tz)
                if stem_mode_delta:
                    df['value'] = df['value'].diff()
                dfH = df.resample('1H').median().reindex(idxH)
                for col, label in [('temp','lm_temp'), ('rh','lm_rh'), ('value','lm_stem')]:
                    v = dfH[col].to_numpy(); cov = coverage_fraction_arr(v)
                    rows_instr.append({'site_id': site_id, 'sensor_type': label, 'series_id': sid,
                                       'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                                       'samples_expected': int(idxH.size), 'coverage': cov})
                if debug_rows is not None and len(debug_rows) < 500:
                    debug_rows.append({'site_id': site_id, 'series_id': sid, 'sensor_type':'dendrometer_lm', 'year': year,
                                       'first_ts': str(dfH.first_valid_index()), 'last_ts': str(dfH.last_valid_index()),
                                       'non_null': int(dfH.notna().any(axis=1).sum())})
            except Exception:
                for label in ['lm_temp','lm_rh','lm_stem']:
                    rows_instr.append({'site_id': site_id, 'sensor_type': label, 'series_id': sid,
                                       'year': year, 'samples_present': 0, 'samples_expected': int(idxH.size), 'coverage': 0.0})

        # Aggregated site‑level coverage (median per sensor type)
        df_instr = pd.DataFrame([r for r in rows_instr if (r['site_id']==site_id and r['year']==year)])
        def med(sensor_type: str) -> float:
            d = df_instr[df_instr['sensor_type'] == sensor_type]
            return float(np.nanmedian(d['coverage'].to_numpy())) if not d.empty else np.nan
        rows_site.append({
            'site_id': site_id,
            'year': year,
            'thermometer_l1_cov': med('thermometer_l1'),
            'hygrometer_l1_cov':  med('hygrometer_l1'),
            'dendrometer_l2_cov': med('dendrometer_l2'),
            'lm_temp_cov':        med('lm_temp'),
            'lm_rh_cov':          med('lm_rh'),
            'lm_stem_cov':        med('lm_stem'),
        })

    coverage_instr_df = pd.DataFrame(rows_instr)
    coverage_site_df  = pd.DataFrame(rows_site)
    return coverage_instr_df, coverage_site_df

# =====================
# Instruments availability
# =====================

def instruments_available_table(metadata_pickle: str,
                                thermo_dir: str, hygro_dir: str, d2_dir: str, lm_dir: str) -> pd.DataFrame:
    """List instrument presence per site along with counts and explicit `series_ids` lists."""
    md = pd.read_pickle(metadata_pickle)
    thermo_ids_all = discover_series_ids(thermo_dir, 'thermometer_l1')
    hygro_ids_all  = discover_series_ids(hygro_dir,  'hygrometer_l1')
    d2_ids_all     = discover_series_ids(d2_dir,     'dendrometer_l2')
    lm_ids_all     = discover_series_ids(lm_dir,     'dendrometer_lm')

    def list_by_site(ids: t.Set[int]) -> pd.DataFrame:
        df = md[md['series_id'].isin(ids)].copy()
        return df.groupby('site_id')['series_id'].apply(lambda s: sorted([int(x) for x in s.tolist()])).reset_index(name='series_ids')

    t_df = list_by_site(thermo_ids_all); t_df['sensor_type'] = 'thermometer_l1'
    h_df = list_by_site(hygro_ids_all);  h_df['sensor_type'] = 'hygrometer_l1'
    d_df = list_by_site(d2_ids_all);     d_df['sensor_type'] = 'dendrometer_l2'
    l_df = list_by_site(lm_ids_all);     l_df['sensor_type'] = 'dendrometer_lm'

    all_df = pd.concat([t_df, h_df, d_df, l_df], axis=0, ignore_index=True)
    all_df['count'] = all_df['series_ids'].apply(lambda xs: len(xs))
    return all_df.sort_values(['site_id','sensor_type'])

# =====================
# Visualization
# =====================

def plot_site_coverage(site_id: int, coverage_site_df: pd.DataFrame, out_png: str) -> None:
    """Bar plots: locals (T/RH/L2) and LM outputs (temp/rh/stem) coverage per year."""
    df = coverage_site_df[coverage_site_df['site_id'] == site_id].copy()
    if df.empty:
        return
    years = df['year'].astype(int).tolist()
    bars_local = [df['thermometer_l1_cov'].to_numpy(), df['hygrometer_l1_cov'].to_numpy(), df['dendrometer_l2_cov'].to_numpy()]
    bars_lm    = [df['lm_temp_cov'].to_numpy(), df['lm_rh_cov'].to_numpy(), df['lm_stem_cov'].to_numpy()]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    labels_local = ['T (L1)', 'RH (L1)', 'Stem (L2)']
    labels_lm    = ['LM Temp', 'LM RH', 'LM Stem']

    x = np.arange(len(years)); width = 0.25
    for i, arr in enumerate(bars_local):
        ax[0].bar(x + (i-1)*width, arr, width=width, label=labels_local[i])
    ax[0].set_ylim(0, 1.0)
    ax[0].set_ylabel('Coverage (fraction)')
    ax[0].set_title(f'Site {site_id} — Local Inputs Coverage by Year')
    ax[0].legend(loc='upper right')

    for i, arr in enumerate(bars_lm):
        ax[1].bar(x + (i-1)*width, arr, width=width, label=labels_lm[i])
    ax[1].set_ylim(0, 1.0)
    ax[1].set_ylabel('Coverage (fraction)')
    ax[1].set_title(f'Site {site_id} — LM Outputs Coverage by Year')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(years, rotation=45)
    ax[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# =====================
# Notebook‑friendly parameters API
# =====================
from dataclasses import dataclass

@dataclass
class CoverageParams:
    """Parameters container to run coverage analysis from a notebook.
    Use `params_to_args(p)` to obtain an argparse.Namespace for `main(args)`.
    """
    out_root: str
    metadata_pickle: str
    thermo_dir: str
    hygro_dir: str
    dendro_l2_dir: str
    dendro_lm_dir: str
    years: list[int]
    tz: str = 'Europe/Zurich'
    stem_mode: str = 'delta'      # 'absolute' | 'delta'
    sites_csv: str | None = None
    plots: bool = True
    debug: bool = False
    fold_rule: str = 'mean'
    time_axis: str = 'local_dst'  # 'local_dst' | 'fixed_winter' | 'utc'


def params_to_args(p: CoverageParams) -> argparse.Namespace:
    """Convert a `CoverageParams` instance to an argparse.Namespace."""
    return argparse.Namespace(
        out_root=p.out_root,
        metadata_pickle=p.metadata_pickle,
        thermo_dir=p.thermo_dir,
        hygro_dir=p.hygro_dir,
        dendro_l2_dir=p.dendro_l2_dir,
        dendro_lm_dir=p.dendro_lm_dir,
        years=[int(y) for y in p.years],
        tz=p.tz,
        stem_mode=p.stem_mode,
        sites_csv=p.sites_csv,
        plots='true' if p.plots else 'false',
        debug='true' if p.debug else 'false',
        fold_rule=p.fold_rule,
        time_axis=p.time_axis,
    )


def dict_to_args(d: dict) -> argparse.Namespace:
    """Convert a Python dict of parameters to an argparse.Namespace.
    Normalizes booleans and ensures `years` is a list of int.
    """
    d = d.copy()
    if 'plots' in d and isinstance(d['plots'], bool):
        d['plots'] = 'true' if d['plots'] else 'false'
    if 'debug' in d and isinstance(d['debug'], bool):
        d['debug'] = 'true' if d['debug'] else 'false'
    if 'years' in d:
        d['years'] = [int(y) for y in d['years']]
    d.setdefault('fold_rule', 'mean')
    d.setdefault('time_axis', 'local_dst')
    return argparse.Namespace(**d)

# =====================
# CLI & main orchestration
# =====================

def parse_args():
    """Parse CLI args for coverage analysis.
    Flags include `--time_axis` to select the timeline used for alignment and indexing.
    """
    p = argparse.ArgumentParser(description='Coverage report with axis-aware alignment and quick plots.')
    p.add_argument('--out_root', required=True)
    p.add_argument('--metadata_pickle', required=True)
    p.add_argument('--thermo_dir', required=True)
    p.add_argument('--hygro_dir', required=True)
    p.add_argument('--dendro_l2_dir', required=True)
    p.add_argument('--dendro_lm_dir', required=True)
    p.add_argument('--years', nargs='+', type=int, required=True)
    p.add_argument('--tz', type=str, default='Europe/Zurich')
    p.add_argument('--stem_mode', type=str, default='delta', choices=['absolute','delta'])
    p.add_argument('--sites_csv', type=str, default=None)
    p.add_argument('--plots', type=str, default='true')
    p.add_argument('--debug', type=str, default='false')
    p.add_argument('--fold_rule', type=str, default='mean', choices=['mean','earliest','latest','first','last','min','max','sum'])
    p.add_argument('--time_axis', type=str, default='local_dst', choices=['local_dst','fixed_winter','utc'])
    return p.parse_args()


def main(args: argparse.Namespace | None = None):
    """Run coverage analysis from CLI or notebook. Handles `time_axis` and `fold_rule`."""
    if args is None:
        args = parse_args()

    local_tz = args.tz
    years = args.years
    stem_mode_delta = (args.stem_mode == 'delta')
    make_plots = (args.plots.lower() == 'true')
    debug = (args.debug.lower() == 'true')
    fold_rule = args.fold_rule
    time_axis = args.time_axis

    # Output folders
    os.makedirs(args.out_root, exist_ok=True)
    cov_dir  = os.path.join(args.out_root, 'coverage'); os.makedirs(cov_dir, exist_ok=True)
    plot_dir = os.path.join(cov_dir, 'plots'); os.makedirs(plot_dir, exist_ok=True)
    debug_dir= os.path.join(cov_dir, 'debug'); os.makedirs(debug_dir, exist_ok=True)

    # Metadata and availability
    metadata_df = pd.read_pickle(args.metadata_pickle)
    avail_df = instruments_available_table(args.metadata_pickle, args.thermo_dir, args.hygro_dir, args.dendro_l2_dir, args.dendro_lm_dir)
    avail_df.to_csv(os.path.join(cov_dir, 'instruments_available.csv'), index=False)

    # Sites to process
    if args.sites_csv:
        sites = pd.read_csv(args.sites_csv)['site_id'].astype(int).tolist()
    else:
        sites = sorted(metadata_df['site_id'].astype(int).unique().tolist())

    all_instr_rows: t.List[pd.DataFrame] = []
    all_site_rows:  t.List[pd.DataFrame] = []
    debug_rows: list = [] if debug else None

    for sid in sites:
        try:
            instr_df, site_df = compute_coverage_for_site(
                site_id=sid, years=years, time_axis=time_axis, local_tz=local_tz,
                metadata_df=metadata_df,
                thermo_dir=args.thermo_dir, hygro_dir=args.hygro_dir,
                d2_dir=args.dendro_l2_dir, lm_dir=args.dendro_lm_dir,
                stem_mode_delta=stem_mode_delta,
                debug_rows=debug_rows,
                fold_rule=fold_rule,
            )
        except Exception:
            instr_df = pd.DataFrame(columns=['site_id','sensor_type','series_id','year','samples_present','samples_expected','coverage'])
            site_df  = pd.DataFrame(columns=['site_id','year','thermometer_l1_cov','hygrometer_l1_cov','dendrometer_l2_cov','lm_temp_cov','lm_rh_cov','lm_stem_cov'])
        all_instr_rows.append(instr_df)
        all_site_rows.append(site_df)
        if make_plots:
            try:
                plot_site_coverage(sid, site_df, os.path.join(plot_dir, f'site_{sid}_coverage.png'))
            except Exception:
                pass

    coverage_instr_df = pd.concat(all_instr_rows, axis=0, ignore_index=True) if all_instr_rows else pd.DataFrame()
    coverage_site_df  = pd.concat(all_site_rows,  axis=0, ignore_index=True) if all_site_rows else pd.DataFrame()

    coverage_instr_df.to_csv(os.path.join(cov_dir, 'coverage_by_instrument_year.csv'), index=False)
    coverage_site_df.to_csv(os.path.join(cov_dir, 'coverage_by_site_year.csv'), index=False)

    if debug and debug_rows:
        pd.DataFrame(debug_rows).to_csv(os.path.join(debug_dir, 'debug_first_last.csv'), index=False)

    print('Wrote coverage tables:')
    print(' -', os.path.join(cov_dir, 'instruments_available.csv'))
    print(' -', os.path.join(cov_dir, 'coverage_by_instrument_year.csv'))
    print(' -', os.path.join(cov_dir, 'coverage_by_site_year.csv'))
    if make_plots:
        print('Wrote site plots to:', plot_dir)
    if debug:
        print('Wrote debug to:', os.path.join(debug_dir, 'debug_first_last.csv'))

if __name__ == '__main__':
    main()
