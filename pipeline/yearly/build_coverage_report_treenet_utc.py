# build_coverage_report_treenet_utc.py
# -*- coding: utf-8 -*-
"""
TreeNet Data Coverage Report — **UTC‑Only, DST‑Proof**, Fully Annotated
=====================================================================

This module computes **per‑site, per‑instrument, per‑year coverage** for TreeNet sensors, after
converting **all timestamps to UTC**. By doing every alignment (rounding, resampling, reindexing)
strictly in UTC, we remove all daylight‑saving (DST) ambiguity (no repeated or missing hours).

Highlights
----------
- **UTC everywhere**: local or naive timestamps are localized to Europe/Zurich *only to disambiguate*,
  then immediately converted to UTC.
- **Rounding in UTC**: `10‑minute` binning and `hourly` resampling are deterministic.
- **Yearly UTC grids**: Fixed‑length (leap day removed) for robust arrays.
- **Civil‑time helpers**: small utilities to plot any UTC series in a civil timezone
  (e.g., Europe/Zurich) **without modifying** the underlying UTC arrays.

Outputs
-------
- coverage/instruments_available.csv
- coverage/coverage_by_instrument_year.csv
- coverage/coverage_by_site_year.csv
- coverage/plots/site_<id>_coverage.png  (if --plots true)
- coverage/debug/debug_first_last.csv     (if --debug true)
"""

from __future__ import annotations
import os
import re
import argparse
import typing as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================================================
# Constants
# =============================================================
FREQ_10MIN = '10min'
FREQ_HOURLY = '1h'


# =============================================================
# Create lists of sites to consider for training and testing 
# =============================================================

def create_site_lists():

    # Path to your input CSV
    INPUT_CSV = "sites.csv"

    # Output paths
    TRAIN_OUT = "train_sites.csv"
    TEST_OUT = "test_sites.csv"

    # Load the CSV
    df = pd.read_csv(INPUT_CSV)

    # Filter only country == "Switzerland"
    df_ch = df[df["country"] == "Switzerland"].copy()

    # Random 80/20 split (stratification optional if countries differ)
    train_df, test_df = train_test_split(
        df_ch,
        test_size=0.2,
        random_state=42,  # for reproducibility
        shuffle=True
    )

    # Save only the site_id column
    train_df[["site_id", "site_name", "site_xcor", "site_ycor", "site_altitude"]].to_csv(TRAIN_OUT, index=False)
    test_df[["site_id", "site_name", "site_xcor", "site_ycor", "site_altitude"]].to_csv(TEST_OUT, index=False)

    print(f"Train sites saved to: {TRAIN_OUT} ({len(train_df)} rows)")
    print(f"Test sites saved to:  {TEST_OUT} ({len(test_df)} rows)")



# =============================================================
# UTC core helpers
# =============================================================

def strip_leap_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return `idx` with **Feb 29** removed to keep a fixed 365‑day year.
    This avoids ragged arrays later when comparing across years.
    """
    return idx[~((idx.month == 2) & (idx.day == 29))]


def ensure_utc_series(s: pd.Series, local_tz: str = 'Europe/Zurich') -> pd.Series:
    """Ensure a **UTC tz‑aware** series.

    - If `s` has a naive index, we **localize** to `local_tz` (civil time) using
      `ambiguous='infer'` and `nonexistent='shift_forward'` to resolve DST boundaries.
    - Then we **convert to UTC** to remove DST ambiguity entirely.
    """
    if s.empty:
        return s
    s = s.copy()
    idx = s.index
    if getattr(idx, 'tz', None) is None:
        idx = pd.to_datetime(idx, utc=False).tz_localize(
            local_tz, ambiguous='infer', nonexistent='shift_forward'
        )
    else:
        idx = idx.tz_convert(local_tz)
    s.index = idx.tz_convert('UTC')
    return s


# TODO: this function does not work. It converts all the instrument readings into NaN.
def round_to_10min_utc(s: pd.Series) -> pd.Series:
    """Round the series to **10‑minute bins in UTC** and deduplicate by mean.
    This is safe and deterministic because UTC has no DST transitions.
    """
    if s.empty:
        return s
    s = s.copy()
    if getattr(s.index, 'tz', None) is None:
        raise ValueError('round_to_10min_utc expects a tz‑aware (UTC) index')
    s = s.tz_convert('UTC')
    s.index = s.index.round(FREQ_10MIN)
    # If multiple points fall onto the same rounded bin, collapse by mean
    s = s[~s.index.duplicated(keep='mean')]
    return s

# TODO: solution
# use 'floor' for speed and 'resample' for robustness
#Recommendation: Start with method='resample', how='median' for robustness; 
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
        s.index = s.index.floor(FREQ_10MIN)
        return s[~s.index.duplicated(keep='mean')]
    elif method == 'resample':
        s = s.copy().tz_convert('UTC').sort_index()
        agg = {'mean': 'mean', 'median': 'median'}.get(how, 'mean')
        return s.resample(FREQ_10MIN, origin='start_day', label='left').agg(agg)
    else:
        raise ValueError("method must be 'floor' or 'resample'")


def make_year_10m_index_utc(year: int) -> pd.DatetimeIndex:
    """Make a **UTC 10‑min grid** for a calendar year with leap day removed.
    Length = 365 × 24 × 6 = 52,560.
    """
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{year}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_10MIN)
    return strip_leap_days(idx)


def make_year_hourly_index_utc(year: int) -> pd.DatetimeIndex:
    """Make a **UTC hourly grid** for a calendar year (leap day removed).
    Length = 365 × 24 = 8,760.
    """
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz='UTC')
    end   = pd.Timestamp(f"{year}-12-31 23:59:59", tz='UTC')
    idx = pd.date_range(start=start, end=end, freq=FREQ_HOURLY)
    return strip_leap_days(idx)

# =============================================================
# Discovery helpers
# =============================================================

def discover_series_ids(dir_path: str, pattern_prefix: str) -> t.Set[int]:
    """Return the set of `series_id` recognized by files named
    `{pattern_prefix}_series_id_<id>.ftr` in `dir_path`.
    """
    ids: set[int] = set()
    regex = re.compile(rf'{re.escape(pattern_prefix)}_series_id_(\d+)\.ftr$')
    for fn in os.listdir(dir_path):
        m = regex.match(fn)
        if m:
            ids.add(int(m.group(1)))
    return ids


def series_by_site(metadata_df: pd.DataFrame, series_ids: t.Set[int]) -> dict[int, t.List[int]]:
    """Group a given set of `series_ids` by `site_id` using the mapping in `metadata_df`.
    Assumes columns `series_id` and `site_id` are present.
    """
    df = metadata_df[metadata_df['series_id'].isin(series_ids)].copy()
    out: dict[int, t.List[int]] = {}
    for site, g in df.groupby('site_id'):
        out[int(site)] = [int(x) for x in g['series_id'].tolist()]
    return out

# =============================================================
# Robust readers (to UTC)
# =============================================================

def _pick_value_column(df: pd.DataFrame, preferred: t.Sequence[str] = ()) -> str:
    """Pick a numeric measurement column from `df`.
    Priority: any name in `preferred` → `'value'` → numeric column with **max non‑null count**.
    """
    for c in preferred:
        if c in df.columns:
            return c
    if 'value' in df.columns:
        return 'value'
    cand: list[tuple[int, str]] = []
    for c in df.columns:
        if c.lower() in ('ts', 'timestamp', 'time', 'date_time', 'datetime'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cand.append((int(df[c].notna().sum()), c))
    if not cand:
        raise ValueError('No numeric value column found')
    cand.sort(reverse=True)
    return cand[0][1]


def read_feather_series_flexible_utc(series_id: int, dir_path: str, local_tz: str,
                                     sensor_hint: str | None = None) -> pd.Series:
    """Read a **local instrument** series and return a **UTC** series.

    Steps
    -----
    1) Read feather file `*series_id_<id>.ftr` and detect timestamp column.
    2) Pick a numeric value column (prefer sensor‑specific names; else `'value'`; else most populated numeric).
    3) Localize timestamps to `local_tz` (Europe/Zurich) if naive, then **convert to UTC**.
    4) Sort and deduplicate timestamps (keep='mean' for exact duplicates).
    """
    pattern = re.compile(rf'.*series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(dir_path) if pattern.match(fn)]
    if not matches:
        raise FileNotFoundError(f"Series {series_id} not found in {dir_path}")
    fp = os.path.join(dir_path, matches[0])
    df = pd.read_feather(fp)

    ts_col = None
    for c in ('ts', 'timestamp', 'time', 'date_time', 'datetime'):
        if c in df.columns:
            ts_col = c; break
    if ts_col is None:
        raise ValueError(f"{fp} has no timestamp column")

    preferred: tuple[str, ...] = ()
    if sensor_hint == 'thermometer_l1':
        preferred = ('temp', 'temperature', 'value')
    elif sensor_hint == 'hygrometer_l1':
        preferred = ('rh', 'relhum', 'value')
    elif sensor_hint == 'dendrometer_l2':
        preferred = ('value', 'radius', 'rad', 'l2')
    val_col = _pick_value_column(df, preferred)

    ts = pd.to_datetime(df[ts_col], utc=False)
    # Localize to civil time only to disambiguate, then convert to UTC
    ts = ts.dt.tz_localize(local_tz) if getattr(ts.dt, 'tz', None) is None else ts.dt.tz_convert(local_tz)
    ts = ts.dt.tz_convert('UTC')

    s = pd.Series(df[val_col].to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated(keep='mean')]
    return s


def read_lm_frame_utc(series_id: int, lm_dir: str, local_tz: str) -> pd.DataFrame:
    """Read a **LM dendrometer** frame and return a **UTC** frame with columns `value`, `temp`, `rh`.
    Missing columns are added as NaN if absent.
    """
    pattern = re.compile(rf'dendrometer_lm_hourly_series_id_{series_id}\.ftr$')
    matches = [fn for fn in os.listdir(lm_dir) if pattern.match(fn)]
    if not matches:
        raise FileNotFoundError(f"LM series {series_id} not found in {lm_dir}")
    fp = os.path.join(lm_dir, matches[0])
    df = pd.read_feather(fp)

    if 'ts' not in df.columns:
        for c in ('timestamp', 'time', 'date_time', 'datetime'):
            if c in df.columns:
                df = df.rename(columns={c: 'ts'}); break
    if 'ts' not in df.columns:
        raise ValueError(f"LM file {fp} missing 'ts'")

    ts = pd.to_datetime(df['ts'], utc=False)
    ts = ts.dt.tz_localize(local_tz) if getattr(ts.dt, 'tz', None) is None else ts.dt.tz_convert(local_tz)
    ts = ts.dt.tz_convert('UTC')
    df = df.set_index(ts)

    for col in ['value', 'temp', 'rh']:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure column order
    return df[['value', 'temp', 'rh']]

# =============================================================
# Coverage computation (UTC)
# =============================================================

def coverage_fraction_arr(v: np.ndarray) -> float:
    """Return fraction of non‑NaN entries in numeric array `v`."""
    if v.size == 0:
        return 0.0
    return float(np.sum(~np.isnan(v)) / v.size)


def compute_coverage_for_site_utc(
    site_id: int,
    years: t.Sequence[int],
    local_tz: str,
    metadata_df: pd.DataFrame,
    thermo_dir: str,
    hygro_dir: str,
    d2_dir: str,
    lm_dir: str,
    stem_mode_delta: bool = True,
    debug_rows: t.Optional[list] = None,
    error_rows: t.Optional[list] = None,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute **per‑year coverage** for all instruments of a site using a **UTC timeline**.

    Returns
    -------
    coverage_instr_df : DataFrame
        Detailed rows per instrument×year (counts & coverage fraction).
    coverage_site_df : DataFrame
        Per‑site×year medians across sensor types (useful high‑level summary).
    """
    # Discover instrument IDs and map to site
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
        idx10 = make_year_10m_index_utc(year)
        idxH  = make_year_hourly_index_utc(year)

        # Thermometer L1
        for sid in thermo_ids:
            try:
                s = read_feather_series_flexible_utc(sid, thermo_dir, local_tz, sensor_hint='thermometer_l1')
                # s = round_to_10min_utc(s).reindex(idx10)
                v = s.to_numpy(); cov = coverage_fraction_arr(v)
                rows_instr.append({
                    'site_id': site_id, 'sensor_type': 'thermometer_l1', 'series_id': sid,
                    'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                    'samples_expected': int(idx10.size), 'coverage': cov
                })
                if debug_rows is not None and len(debug_rows) < 50000:
                    debug_rows.append({
                        'site_id': site_id, 'series_id': sid, 'sensor_type': 'thermometer_l1', 'year': year,
                        'first_ts': str(s.first_valid_index()), 'last_ts': str(s.last_valid_index()),
                        'non_null': int(np.sum(~np.isnan(v)))
                    })
            except Exception as e:
                if error_rows is not None:
                    error_rows.append({
                        'site_id': site_id,
                        'sensor_type': 'thermometer_l1',
                        'series_id': sid,
                        'year': year,
                        'step': 'read_or_align',          # rough stage
                        'error_type': type(e).__name__,
                        'error': repr(e),
                    })
                print('thermometer exception')
                rows_instr.append({
                    'site_id': site_id, 'sensor_type': 'thermometer_l1', 'series_id': sid,
                    'year': year, 'samples_present': 0, 'samples_expected': int(idx10.size), 'coverage': 0.0
                })

        # Hygrometer L1
        for sid in hygro_ids:
            try:
                s = read_feather_series_flexible_utc(sid, hygro_dir, local_tz, sensor_hint='hygrometer_l1')
                # s = round_to_10min_utc(s).reindex(idx10)
                v = s.to_numpy(); cov = coverage_fraction_arr(v)
                rows_instr.append({
                    'site_id': site_id, 'sensor_type': 'hygrometer_l1', 'series_id': sid,
                    'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                    'samples_expected': int(idx10.size), 'coverage': cov
                })
                if debug_rows is not None and len(debug_rows) < 50000:
                    debug_rows.append({
                        'site_id': site_id, 'series_id': sid, 'sensor_type': 'hygrometer_l1', 'year': year,
                        'first_ts': str(s.first_valid_index()), 'last_ts': str(s.last_valid_index()),
                        'non_null': int(np.sum(~np.isnan(v)))
                    })
            except Exception as e:
                if error_rows is not None:
                    error_rows.append({
                        'site_id': site_id,
                        'sensor_type': 'hygrometer_l1',
                        'series_id': sid,
                        'year': year,
                        'step': 'read_or_align',          # rough stage
                        'error_type': type(e).__name__,
                        'error': repr(e),
                    })
                print('hygrometer exception')
                rows_instr.append({
                    'site_id': site_id, 'sensor_type': 'hygrometer_l1', 'series_id': sid,
                    'year': year, 'samples_present': 0, 'samples_expected': int(idx10.size), 'coverage': 0.0
                })

        # Dendrometer L2
        for sid in d2_ids:
            try:
                s = read_feather_series_flexible_utc(sid, d2_dir, local_tz, sensor_hint='dendrometer_l2')
                # s = round_to_10min_utc(s)
                if stem_mode_delta:
                    s = s.diff()
                s = s.reindex(idx10)
                v = s.to_numpy(); cov = coverage_fraction_arr(v)
                rows_instr.append({
                    'site_id': site_id, 'sensor_type': 'dendrometer_l2', 'series_id': sid,
                    'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                    'samples_expected': int(idx10.size), 'coverage': cov
                })
                if debug_rows is not None and len(debug_rows) < 50000:
                    debug_rows.append({
                        'site_id': site_id, 'series_id': sid, 'sensor_type': 'dendrometer_l2', 'year': year,
                        'first_ts': str(s.first_valid_index()), 'last_ts': str(s.last_valid_index()),
                        'non_null': int(np.sum(~np.isnan(v)))
                    })
            except Exception as e:
                if error_rows is not None:
                    error_rows.append({
                        'site_id': site_id,
                        'sensor_type': 'dendrometer_l2',
                        'series_id': sid,
                        'year': year,
                        'step': 'read_or_align',          # rough stage
                        'error_type': type(e).__name__,
                        'error': repr(e),
                    })
                print('dendrometer l2 exception')
                rows_instr.append({
                    'site_id': site_id, 'sensor_type': 'dendrometer_l2', 'series_id': sid,
                    'year': year, 'samples_present': 0, 'samples_expected': int(idx10.size), 'coverage': 0.0
                })

        # LM (hourly outputs)
        for sid in lm_ids:
            try:
                df = read_lm_frame_utc(sid, lm_dir, local_tz)
                if stem_mode_delta:
                    df['value'] = df['value'].diff()
                dfH = df.resample(FREQ_HOURLY).median().reindex(idxH)
                for col, label in [('temp', 'lm_temp'), ('rh', 'lm_rh'), ('value', 'lm_stem')]:
                    v = dfH[col].to_numpy(); cov = coverage_fraction_arr(v)
                    rows_instr.append({
                        'site_id': site_id, 'sensor_type': label, 'series_id': sid,
                        'year': year, 'samples_present': int(np.sum(~np.isnan(v))),
                        'samples_expected': int(idxH.size), 'coverage': cov
                    })
                if debug_rows is not None and len(debug_rows) < 50000:
                    debug_rows.append({
                        'site_id': site_id, 'series_id': sid, 'sensor_type': 'dendrometer_lm', 'year': year,
                        'first_ts': str(dfH.first_valid_index()), 'last_ts': str(dfH.last_valid_index()),
                        'non_null': int(dfH.notna().any(axis=1).sum())
                    })
            except Exception as e:
                if error_rows is not None:
                    for label in ['lm_temp', 'lm_rh', 'lm_stem']:
                        error_rows.append({
                            'site_id': site_id,
                            'sensor_type': label,
                            'series_id': sid,
                            'year': year,
                            'step': 'read_or_align',          # rough stage
                            'error_type': type(e).__name__,
                            'error': repr(e),
                        })
                print('dendrometer lm exception')
                for label in ['lm_temp', 'lm_rh', 'lm_stem']:
                    rows_instr.append({
                        'site_id': site_id, 'sensor_type': label, 'series_id': sid,
                        'year': year, 'samples_present': 0,
                        'samples_expected': int(idxH.size), 'coverage': 0.0
                    })

        # Per‑site medians for this year
        df_instr = pd.DataFrame([r for r in rows_instr if (r['site_id'] == site_id and r['year'] == year)])
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

    return pd.DataFrame(rows_instr), pd.DataFrame(rows_site)

# =============================================================
# Instruments availability
# =============================================================

def instruments_available_table(metadata_pickle: str, thermo_dir: str, hygro_dir: str, d2_dir: str, lm_dir: str) -> pd.DataFrame:
    """List **which instruments exist** per site, with counts and explicit ID lists.
    Useful to sanity‑check coverage results and data presence before processing.
    """
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
    return all_df.sort_values(['site_id', 'sensor_type'])

# =============================================================
# Visualization & civil‑time helpers
# =============================================================

def plot_site_coverage(site_id: int, coverage_site_df: pd.DataFrame, out_png: str) -> None:
    """Render two bar charts per site (locals & LM outputs) showing **yearly coverage fractions**.
    This plot is unaffected by time zone per se because it uses per‑year aggregates.
    """
    df = coverage_site_df[coverage_site_df['site_id'] == site_id].copy()
    if df.empty:
        return
    years = df['year'].astype(int).tolist()
    bars_local = [df['thermometer_l1_cov'].to_numpy(), df['hygrometer_l1_cov'].to_numpy(), df['dendrometer_l2_cov'].to_numpy()]
    bars_lm    = [df['lm_temp_cov'].to_numpy(),        df['lm_rh_cov'].to_numpy(),        df['lm_stem_cov'].to_numpy()]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    labels_local = ['T (L1)', 'RH (L1)', 'Stem (L2)']
    labels_lm    = ['LM Temp', 'LM RH', 'LM Stem']

    x = np.arange(len(years)); width = 0.25
    for i, arr in enumerate(bars_local):
        ax[0].bar(x + (i - 1) * width, arr, width=width, label=labels_local[i])
    ax[0].set_ylim(0, 1.0); ax[0].set_ylabel('Coverage (fraction)')
    ax[0].set_title(f'Site {site_id} — Local Inputs Coverage by Year'); ax[0].legend(loc='upper right')

    for i, arr in enumerate(bars_lm):
        ax[1].bar(x + (i - 1) * width, arr, width=width, label=labels_lm[i])
    ax[1].set_ylim(0, 1.0); ax[1].set_ylabel('Coverage (fraction)')
    ax[1].set_title(f'Site {site_id} — LM Outputs Coverage by Year')
    ax[1].set_xticks(x); ax[1].set_xticklabels(years, rotation=45); ax[1].legend(loc='upper right')

    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)

# ---- Civil‑time utilities (do not change data arrays) ----

def utc_index_to_local(idx_utc: pd.DatetimeIndex, tz: str = 'Europe/Zurich') -> pd.DatetimeIndex:
    """Convert a **UTC DatetimeIndex** to civil local time (e.g., Europe/Zurich) for plotting.
    The data remain in UTC elsewhere; this is only for presentation.
    """
    if getattr(idx_utc, 'tz', None) is None:
        raise ValueError('utc_index_to_local expects a tz‑aware UTC index')
    return idx_utc.tz_convert(tz)


def series_utc_to_civil(s_utc: pd.Series, tz: str = 'Europe/Zurich') -> pd.Series:
    """Return a **view** of the UTC series in civil local time (for plotting/inspection only)."""
    if getattr(s_utc.index, 'tz', None) is None:
        raise ValueError('series_utc_to_civil expects a tz‑aware UTC index')
    return s_utc.tz_convert(tz)


def plot_series_civiltime(s_utc: pd.Series, tz: str = 'Europe/Zurich', title: str | None = None,
                          out_png: str | None = None) -> None:
    """Quick helper to plot a UTC series using **civil local time** on the x‑axis.

    Parameters
    ----------
    s_utc : Series
        UTC series to plot.
    tz : str
        Civil timezone for the plot (default: Europe/Zurich).
    title : str | None
        Optional plot title.
    out_png : str | None
        If provided, save to this path; otherwise show the figure.
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
# CLI & main
# =============================================================

def parse_args():
    """Parse CLI args for **UTC‑only** coverage analysis.
    `--tz` is still used *only* to correctly localize **naive** timestamps before converting to UTC.
    """
    p = argparse.ArgumentParser(description='TreeNet coverage (UTC‑only, DST‑proof).')
    p.add_argument('--out_root', required=True)
    p.add_argument('--metadata_pickle', required=True)
    p.add_argument('--thermo_dir', required=True)
    p.add_argument('--hygro_dir', required=True)
    p.add_argument('--dendro_l2_dir', required=True)
    p.add_argument('--dendro_lm_dir', required=True)
    p.add_argument('--years', nargs='+', type=int, required=True)
    p.add_argument('--tz', type=str, default='Europe/Zurich', help='Civil timezone to localize naive timestamps before converting to UTC')
    p.add_argument('--sites_csv', type=str, default=None)
    p.add_argument('--stem_mode', type=str, default='delta', choices=['absolute','delta'])
    p.add_argument('--plots', type=str, default='true')
    p.add_argument('--debug', type=str, default='false')
    return p.parse_args()


def main(args: argparse.Namespace | None = None):
    """Run coverage analysis on a **UTC** time axis.
    Writes CSV tables and (optionally) per‑site coverage plots.
    """
    if args is None:
        args = parse_args()

    years = args.years
    local_tz = args.tz
    make_plots = (args.plots.lower() == 'true')
    debug = (args.debug.lower() == 'true')
    stem_mode_delta = (args.stem_mode == 'delta')

    # Output folders
    os.makedirs(args.out_root, exist_ok=True)
    cov_dir  = os.path.join(args.out_root, 'coverage'); os.makedirs(cov_dir, exist_ok=True)
    plot_dir = os.path.join(cov_dir, 'plots'); os.makedirs(plot_dir, exist_ok=True)
    debug_dir= os.path.join(cov_dir, 'debug'); os.makedirs(debug_dir, exist_ok=True)

    metadata_df = pd.read_pickle(args.metadata_pickle)
    avail_df = instruments_available_table(args.metadata_pickle, args.thermo_dir, args.hygro_dir, args.dendro_l2_dir, args.dendro_lm_dir)
    avail_df.to_csv(os.path.join(cov_dir, 'instruments_available.csv'), index=False)

    # Which sites
    if args.sites_csv:
        sites = pd.read_csv(args.sites_csv)['site_id'].astype(int).tolist()
    else:
        sites = sorted(metadata_df['site_id'].astype(int).unique().tolist())

    all_instr_rows: t.List[pd.DataFrame] = []
    all_site_rows:  t.List[pd.DataFrame] = []
    debug_rows: list = [] if debug else None
    error_rows: list = [] if debug else None

    for sid in sites:
        try:
            instr_df, site_df = compute_coverage_for_site_utc(
                site_id=sid, years=years, local_tz=local_tz,
                metadata_df=metadata_df,
                thermo_dir=args.thermo_dir, hygro_dir=args.hygro_dir,
                d2_dir=args.dendro_l2_dir, lm_dir=args.dendro_lm_dir,
                stem_mode_delta=stem_mode_delta,
                debug_rows=debug_rows,
                error_rows=error_rows,
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

    if debug and debug_rows and len(debug_rows) > 0:
        pd.DataFrame(debug_rows).to_csv(os.path.join(debug_dir, 'debug_first_last.csv'), index=False)
        print('Wrote debug to:', os.path.join(debug_dir, 'debug_first_last.csv'))
    else:
        if debug:
            print('No per-series debug rows captured (debug_first_last.csv not written).')

    if debug and error_rows and len(error_rows) > 0:
        pd.DataFrame(error_rows).to_csv(os.path.join(debug_dir, 'debug_errors.csv'), index=False)
        print('Wrote error log to:', os.path.join(debug_dir, 'debug_errors.csv'))
    else:
        if debug:
            print('No errors encountered (debug_errors.csv not written).')


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
