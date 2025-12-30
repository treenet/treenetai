# build_normalized_dataset_treenet.py — API Documentation


> Generated on demand from docstrings and module introspection.


## Module Overview

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


## Constants and Configuration

- **SEQ_LEN_10MIN** = `52560`

- **HOUR_STEPS** = `8760`

- **STRIDE_PER_HR** = `6`

- **N_CHANNELS** = `11`

- **N_TARGETS** = `3`

- **LOCAL_COLS** = `['local_T', 'local_RH', 'local_stem']`

- **GLOBAL_COLS** = `['g_tmean', 'g_tmin', 'g_tmax', 'g_rh', 'g_vpd', 'g_pr', 'g_rad', 'g_doy']`


## Functions

### `discover_meteo_files(meteo_dir)`

Return a mapping from **site_id** to the path of its global meteo CSV.

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

### `load_global_daily(site_id, meteo_dir, tz)`

Load and standardize **daily global meteo** for a site across all years.

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

### `read_feather_series(series_id, dir_path, tz, value_col='value', ts_col='ts')`

Read a **single local instrument series** from Feather.

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

### `strip_leap_days(idx)`

Remove **Feb 29** from a datetime index to ensure **fixed yearly length**.

This is important when models expect fixed-length inputs. By removing leap days,
we avoid padding/masking logic and keep tensors rectangular.

### `make_multi_year_10m_index(years, tz)`

Create a **continuous 10-min multi-year index** from min(years) to max(years), tz-aware.

Leap days are removed to keep yearly windows at **52,560** steps.

### `make_multi_year_hourly_index(years, tz)`

Create a **continuous hourly multi-year index** from min(years) to max(years), tz-aware.

Leap days are removed to keep yearly windows at **8,760** hours.

### `broadcast_daily_to_10m(daily_df, idx_10m)`

Broadcast **daily** values to **10-min** by mapping each 10-min timestamp to its date.

We reindex the daily frame to the normalized dates of the 10-min index and **forward-fill**.
This yields a 10-min DataFrame with the same columns as the input (GLOBAL_COLS).

### `compute_quantile_scaler(v, q_low=5.0, q_high=95.0)`

Compute robust quantiles `(q_low, q_high)` ignoring NaNs.

Edge cases handled:
- If quantiles are not finite (empty/NaN arrays), fall back to nanmin/nanmax.
- If `q_high <= q_low`, enforce a minimal positive range.

### `fit_site_scalers(site_id, years, meteo_dir, tz, per_year=True, q_low=5.0, q_high=95.0)`

Fit **global channel** scalers for a site, either **per year** or **ALL years**.

Locals are added later when building segments (after instrument selection).

Returns
-------
dict
    If per_year=True: `{year: {channel: {'q_low': q, 'q_high': q}}}`;
    else: `{'ALL': {channel: {...}}}`.

### `normalize_array(arr, q1, q2, clip_low=-0.1, clip_high=1.1)`

Normalize `arr` via `(x - q1) / (q2 - q1)` and clip to `[clip_low, clip_high]`.

Rationale: clipping guards against rare outliers and keeps inputs within model-friendly bounds.

### `discover_series_ids(dir_path, pattern_prefix)`

Scan a directory and return **all series IDs** whose filenames match a sensor prefix.

Filename pattern: `{prefix}_series_id_<id>.ftr`.

### `series_by_site(metadata_df, series_ids)`

Group provided `series_ids` **by site** using `metadata_df` (columns: series_id, site_id).

Returns
-------
dict[int, List[int]]
    {site_id: [series_id,...]}

### `read_lm_frame(series_id, lm_dir, tz)`

Read a **LM dendrometer** frame containing `value`, `temp`, `rh` columns.

- Timestamps are tz-unified and used as the index.
- Missing `temp` or `rh` columns are created as NaN for safety.

### `to_hourly(df, how='median')`

Resample to hourly by **median** (default) or **mean**.

Note: Median is robust to spikes and is often preferred for physiological signals.

### `build_site_level_targets_multi(site_id, years, tz, lm_dir, dendro_lm_ids_by_site, stem_mode='absolute', agg='median')`

Build **site-level hourly targets** across the multi-year index.

We aggregate `value` (stem), `temp`, and `rh` across **all LM series** at the site
using **nanmedian** (or nanmean). If `stem_mode='delta'`, we difference the LM stem.

Returns
-------
pd.DataFrame
    Hourly DataFrame indexed by the multi-year hourly index with columns `['stem','temp','rh']`.

### `coverage_fraction(series)`

Compute the **fraction of non-NaN samples** in a series.

Used to skip windows/combos with insufficient local data.

### `write_diagnostics_row(rows, **kw)`

Append a **diagnostic row** (dict) to an accumulating list.

We keep diagnostics lightweight (list of dicts) and serialize at the end.

### `compute_site_instrument_counts(metadata_pickle, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir)`

Compute the **number of instruments per site** for each sensor type.

Returns a DataFrame with columns:
`site_id, n_thermometers, n_hygrometers, n_dendrometers_L2, n_dendrometers_LM`.

### `write_site_summaries(out_root, diagnostics_rows, instrument_df)`

Write a **site-level summary** CSV combining instrument counts and diagnostics (train/test).

The table includes segment/window totals and warning tallies per split.

### `rolling_year_windows(idx_10m, overlap_days=10, year_days=365)`

Generate **year-long windows** over a 10-min index with a given day overlap.

- Window length = `year_days` (default 365)
- Stride days   = `year_days - overlap_days` (e.g., 365 - 10 = 355)
- We stop when the end of a window would exceed the last timestamp.

Returns
-------
List[Tuple[pd.Timestamp, pd.Timestamp]]
    List of (start, end) inclusive/exclusive bounds; end is `start + year_days`.

### `build_segments_for_site(site_id, years, tz, meteo_dir, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir, metadata_path, scalers, per_year=True, require_complete_locals=False, stem_mode='absolute', input_mode='combinations', target_mode='lm_site_median', max_combos_per_site=None, min_local_coverage=0.7, min_lm_series=1, overlap_days=10, diagnostics_rows=None, split='train')`

Build **year-long window segments** for one site across multiple years.

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

### `build_datasets(out_root, meteo_dir, thermo_dir, hygro_dir, dendro_l2_dir, dendro_lm_dir, metadata_pickle, site_ids_train, site_ids_test, years, per_year=True, tz='Europe/Zurich', require_complete_locals=False, stem_mode='absolute', input_mode='combinations', target_mode='lm_site_median', max_combos_per_site=None, min_local_coverage=0.7, min_lm_series=1, overlap_days=10)`

High-level orchestration to build TRAIN/TEST arrays and diagnostics.

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

### `parse_args()`

Parse command-line arguments for the pre-processing workflow.

The defaults favor `combinations` input mode, `lm_site_median` targets, and a 10-day overlap.

### `read_site_ids_csv(path)`

Read a CSV that contains a `site_id` column and return it as a list of ints.
