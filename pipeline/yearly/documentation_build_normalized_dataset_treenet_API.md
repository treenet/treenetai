# build_normalized_dataset_treenet_utc.py — API Documentation (Updated)
> UTC-only pipeline (Strategy‑B globals, LM‑per‑ID pairing) — **Dec 31, 2025 update**

## Module Overview

This module builds normalized **multichannel dataset arrays** from TreeNet raw data using **calendar‑year segments** (leap day removed), a **UTC** timeline for locals and LM, **Strategy‑B** mapping of **daily civil** globals to the **UTC 10‑min** grid, and **LM targets paired by dendrometer `series_id`**. It supports per‑window **best instrument selection** and **optional substitution** of missing local inputs (T/RH) with globals (inputs only).

**Key characteristics**
- **UTC everywhere (locals & LM)**; robust to DST.
- **Strategy‑B** broadcast: civil daily → UTC 10‑min (with `g_doy`).
- **Calendar‑year windows**: `YYYY‑01‑01 00:00 UTC → (YYYY+1)‑01‑01 00:00 UTC`.
- **Per‑ID LM pairing**: each L2 dendrometer pairs only with the **same** LM ID.
- **Input selection**: `best` (recommended) or `combinations`.
- **Substitution policy** (`--allow_missing_locals true`): T→`g_tmean`, RH→`g_rh` in **inputs only**; stem must meet coverage; LM targets remain ground truth.

---

## Constants and Configuration
- `SEQ_LEN_10MIN = 52560` — 365 days × 24 h × 6 (leap day removed).
- `HOUR_STEPS = 8760` — 365 × 24.
- `N_CHANNELS = 11` — 3 locals + 7 globals + `g_doy`.
- `N_TARGETS = 3` — hourly `local_T`, `local_RH`, `local_stem` (LM).
- `LOCAL_COLS = ['local_T', 'local_RH', 'local_stem']`.
- `GLOBAL_COLS = ['g_tmean','g_tmin','g_tmax','g_rh','g_vpd','g_pr','g_rad','g_doy']`.
- Frequencies: `FREQ_10M = '10min'`, `FREQ_1H = '1h'`.

---

## Function Reference (current script)

### Indices & UTC grid
- **`strip_leap_days(idx)`** → remove Feb 29 to keep fixed‑length arrays.
- **`make_multi_year_10m_index_utc(years)`** → UTC 10‑min master index spanning requested years.
- **`make_multi_year_hourly_index_utc(years)`** → UTC hourly master index spanning requested years.

### Global meteo (daily civil) & broadcast
- **`discover_meteo_files(meteo_dir)`** → map `site_id → csv` by capturing integer from filename.
- **`load_global_daily_civil(site_id, meteo_dir)`** → load & standardize civil‑daily globals (`tas`, `tasmin`, `tasmax`, `rh`, `vpd`, `gh`, `pr` → `g_*`).
- **`broadcast_daily_civil_to_utc_grid(daily_civil, idx_10m_utc, tz_local)`** → Strategy‑B mapping from civil day to UTC 10‑min grid; adds `g_doy` in the builder.

### Local series readers & binning
- **`_pick_value_column(df, preferred)`** → choose correct numeric value column.
- **`read_feather_series_utc(series_id, dir_path, local_tz, ..., sensor_hint)`** → read local series; localize to `local_tz`, convert to UTC; deduplicate.
- **`bin_to_10min_utc(s, method='resample', how='median')`** → 10‑min binning.

### LM targets (hourly) — **updated behavior**
- **`read_lm_hourly_frame_utc(series_id, lm_dir, local_tz)`**  
  **RAW LM only** (we **ignore** `dendrometer_lm_hourly_series_id_*.ftr`). The RAW file contains:
  - `value` (stem) at **10‑min** resolution,
  - `temp` and `rh` at **1‑hour** resolution (local hour marks).

  **Current policy (strict equality):**
  1. Convert RAW timestamps to **local time** (`local_tz`).
  2. **Select only rows at exact local `HH:00:00`** (minute==0 & second==0).
  3. Reindex to a **continuous local hourly** index across the data span, leaving **NaN** where exact hour rows are missing.
  4. Convert the local hourly index to **UTC**.
  5. Return canonical columns: `['stem','local_T','local_RH']` with UTC hourly index.

  **Rationale:** This guarantees that the hourly LM frame is composed of *actual* local hour marks without interpolation or nearest matching. It reflects your requirement that the dendrometer target (`value`) used in the model should be the exact hour sample.

  **Known trade‑off:** If the RAW `value` grid occasionally drifts (e.g., `HH:00:10`), those hours become **NaN** under strict equality and may reduce usable windows unless substitution or tolerance is introduced elsewhere.

### Normalization
- **`compute_quantile_scaler(v, q_low, q_high)`** → robust bounds.
- **`fit_site_scalers(site_id, years, meteo_dir, per_year)`** → fit **global** channel scalers per site/scope.
- **`normalize_array(arr, q1, q2, clip_low=-0.1, clip_high=1.1)`** → affine normalization with clipping.

### Instrument discovery (union strategy)
- **`discover_series_ids(dir, prefix)`** → collect present `series_id`s from directory.
- **`series_by_site(metadata_df, series_ids)`** → map discovered IDs back to `site_id` using metadata.
- **`get_site_instrument_ids_by_metadata(metadata_df, site_id)`** → semantic discovery via `variable_name` tokens (T/RH/L2).

### Diagnostics & coverage
- **`coverage_fraction(series)`** → fraction of non‑NaNs.
- **`write_diag(rows, **kw)`** → append a diagnostics event.

### Windows
- **`rolling_year_windows(idx_10m, overlap_days, year_days)`** → calendar‑year windows present in the 10‑min index.

### Builders
- **`build_segments_for_site_utc(...)`** → per‑site builder: Strategy‑B globals; instrument discovery (union); L2∩LM intersection; input selection (`best`/`combinations`) under coverage/substitution policy; normalization; segment accumulation.
- **`build_datasets_utc(...)`** → orchestration over train/test sites; fit scalers; write arrays/identifiers/diagnostics.

### Counts & inspection
- **`compute_site_instrument_counts(...)`** → instrument counts per site.
- **`utc_index_to_local(idx_utc, tz)`**, **`series_utc_to_civil(s_utc, tz)`**, **`plot_series_civiltime(...)`** → inspection helpers.

### CLI & test
- **`parse_args()`**, **`_test_strategy_b_dst_fallback(tz_local)`**, **`read_site_ids_csv(path)`**, **`__main__`**.

---

## Parameter Semantics (Key Flags)
- **`--stem_mode`**: `absolute` / `delta` (inputs only; targets never differenced).
- **`--input_mode`**: `best` / `combinations`.
- **`--allow_missing_locals`**:
  - `false` (strict) → T,RH,stem must satisfy `--min_local_coverage` (no substitution).
  - `true` (substitution) → stem must satisfy coverage; T/RH may be substituted in **inputs** using globals.
- **`--min_local_coverage`**: coverage gate in window.

---

## Current LM handling vs. potential future improvements

### Current (strict equality at local hour marks)
- Pros: exact hour fidelity; transparent provenance; no interpolation.
- Cons: any slight drift of the 10‑min `value` grid away from `HH:00:00` results in **NaN** at that hour.

### Proposed improvements (optional, configurable)
1. **Nearest‑within tolerance** (e.g., `±10 min` for `value`, `±20 min` for `temp/rh`):
   - Select the sample closest to the local hour if exact `HH:00:00` is missing.
   - Pros: fewer NaNs, more usable windows.
   - Cons: introduces a small alignment tolerance; should be reported in diagnostics (e.g., `lm_hour_nearest_used=true`).
   - Implementation sketch:
     - Add CLI `--lm_hour_match_mode {exact, nearest}` and `--lm_hour_tolerance_value`, `--lm_hour_tolerance_temp_rh`.
     - Use `pandas.merge_asof` on local timestamps (ensure identical dtypes); convert to UTC after alignment.

2. **Rounding to nearest hour** for `value` only:
   - Round timestamps with seconds >=30 to the next hour; otherwise to the previous hour.
   - Pros: deterministic and very fast; no merge.
   - Cons: can bias towards earlier/later samples near boundaries; document behavior.
   - Implementation sketch:
     - Compute `rounded = ts_local.floor('h')` or `ceil('h')` based on seconds threshold; groupby the rounded times and pick median or first.

3. **Hourly resampling (median) with pre‑filtering**:
   - Pre‑filter `value` to keep samples within `±15 min` around each hour, then aggregate by hour (`median`).
   - Pros: robust to spikes, transparent filter window.
   - Cons: aggregates rather than selecting a single sample; document as an alternative target definition.

For each approach, record an explicit **diagnostic flag** when non‑exact alignment is used, and expose tolerances in identifiers CSV.

---

## Usage Examples

### Site 3 — strict, high‑quality (best selection)
```bash
python3 build_normalized_dataset_treenet_utc.py   --out_root ./out_site3_strict   --metadata_pickle /storage/metadata_all.pkl   --meteo_dir /storage/meteo_daily_civil   --thermo_dir /storage/thermometer_l1   --hygro_dir  /storage/hygrometer_l1   --dendro_l2_dir /storage/dendrometer_l2   --dendro_lm_dir /storage/dendrometer_lm   --train_site_ids_csv site3.csv   --years 2019   --stem_mode absolute   --input_mode best   --min_local_coverage 0.60   --allow_missing_locals false
```

### Site 3 — robust with substitution (best selection)
```bash
python3 build_normalized_dataset_treenet_utc.py   --out_root ./out_site3_sub   --metadata_pickle /storage/metadata_all.pkl   --meteo_dir /storage/meteo_daily_civil   --thermo_dir /storage/thermometer_l1   --hygro_dir  /storage/hygrometer_l1   --dendro_l2_dir /storage/dendrometer_l2   --dendro_lm_dir /storage/dendrometer_lm   --train_site_ids_csv site3.csv   --years 2019   --stem_mode absolute   --input_mode best   --min_local_coverage 0.50   --allow_missing_locals true
```

### Site 10 — explore all combinations
```bash
python3 build_normalized_dataset_treenet_utc.py   --out_root ./out_site10_allpairs   --metadata_pickle /storage/metadata_all.pkl   --meteo_dir /storage/meteo_daily_civil   --thermo_dir /storage/thermometer_l1   --hygro_dir  /storage/hygrometer_l1   --dendro_l2_dir /storage/dendrometer_l2   --dendro_lm_dir /storage/dendrometer_lm   --train_site_ids_csv site10.csv   --years 2019   --input_mode combinations   --max_combos_per_site 3   --allow_missing_locals true
```

---

## Troubleshooting (diagnostics to check)
- `missing_instruments_at_site` — discovery yielded empty T/RH/L2.
- `no_dendro_id_intersection_for_site` — no L2∩LM for the site.
- `no_best_combo_in_window_strict` / `no_best_combo_in_window` — best‑mode found no eligible pair.
- `low_local_coverage_combo` — coverage gate failure (see `cov_T`, `cov_RH`, `cov_ST`).
- `lm_hourly_targets_all_nan_after_reindex` — LM window has all NaNs; segment skipped.

---

## Notes
- Per‑ID pairing is **strict**: inputs for dendrometer ID `d` are paired only with LM targets for **the same ID**.
- Globals are normalized from daily civil data per site/year; locals are normalized lazily per scope.
- Use lowercase frequencies (`'10min'`, `'1h'`) with modern pandas.
