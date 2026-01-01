# TreeNet UTC Preprocessing Pipeline
**Strategy‑B globals + LM‑per‑ID target pairing**

This repository provides a **UTC‑based preprocessing pipeline** that converts TreeNet multi‑sensor data (thermometers, hygrometers, dendrometers L2, and LM targets) into deep‑learning‑ready arrays:

- **X**: 10‑minute inputs `(52560 × 11 channels)`
- **y**: 1‑hour targets `(8760 × 3 channels)`
- Per‑site normalization stats
- Segment identifiers
- Full diagnostics

It implements:

- **UTC timeline only** for locals and LM (robust to DST)
- **Strategy‑B** broadcast of **daily civil** globals → **UTC 10‑min** grid
- **Per‑ID pairing** (each L2 dendrometer is paired only with LM of the **same `series_id`**)
- **Calendar‑year windows** (e.g., `2019‑01‑01 → 2020‑01‑01` in UTC)
- Deterministic 10‑minute binning / hourly resampling for LM
- Best‑instrument selection and/or global substitution (optional)

---

## 🗺️ Dataflow Diagram

```mermaid
flowchart LR
  subgraph Inputs
    MET[Daily globals (civil): tas, tasmin, tasmax, rh, vpd, pr, gh]
    TH[thermometer_l1_*]
    HY[hygrometer_l1_*]
    D2[dendrometer_l2_*]
    LM[ or dendrometer_lm_*]
    META[metadata_all.pkl]
  end
  subgraph Processing
    A[Localize → TZ (Europe/Zurich)\nthen convert → UTC]
    B[Bin locals to 10‑min UTC (median)]
    SB[Strategy‑B broadcast civil daily → UTC 10‑min]
    YW[Calendar‑year window (UTC)]
    SEL[Select (T,RH) by input_mode\n(best/combinations)\n& policy allow_missing_locals]
    SUB[[Substitute inputs from globals\n(T→g_tmean, RH→g_rh) when allowed]]
    NORM[Per‑site quantile normalization\n(globals prefit; locals lazy)]
    LMH[LM → hourly UTC (prefer hourly file; else resample)]
    PAIR[Per‑ID pairing (L2 ↔ LM same series_id)]
  end
  subgraph Outputs
    X[X.npy (10‑min, 11 ch)]
    Y[y.npy (1‑h, 3 ch)]
    IDS[train/test_identifiers.csv]
    NOR[normalizers_site_*.json]
    DIAG[diagnostics/*.csv]
  end

  TH --> A --> B
  HY --> A --> B
  D2 --> A --> B
  LM --> A --> LMH
  MET --> SB
  META --> SEL

  B -->|T,RH,STEM| YW --> SEL
  SB --> YW
  SEL -->|if locals sparse and allowed| SUB --> NORM
  SEL -->|otherwise| NORM
  LMH --> PAIR
  NORM --> PAIR --> X
  LMH --> Y
  NORM --> IDS
  PAIR --> IDS
  NORM --> NOR
  SEL --> DIAG
  LMH --> DIAG
```

---

## 📦 Outputs

Under `--out_root` you will get:

- **Arrays**
  - `X_train.npy` — `(n_segments, 52560, 11)`
  - `y_train.npy` — `(n_segments, 8760, 3)`
  - `X_test.npy`, `y_test.npy` (when `--test_site_ids_csv` is provided)
- **Identifiers**
  - `train_identifiers.csv`, `test_identifiers.csv` — instrument IDs, windows, flags
- **Normalizers**
  - `normalizers/normalizers_site_<site>.json` — per‑site quantile scalers (globals)
  - `diagnostics/normalizers_summary.csv`
- **Diagnostics**
  - `diagnostics/diagnostics_preprocessing.csv` — per‑event reasons for skips/choices
  - `diagnostics/site_summary.csv` — aggregated per‑site stats

---

## 🛠️ CLI — Minimal Working Example (Site 3, Year 2019)

```bash
python3 build_normalized_dataset_treenet_utc.py \
  --out_root ./outputs \
  --metadata_pickle /storage/metadata_all.pkl \
  --meteo_dir /storage/meteo_daily_civil \
  --thermo_dir /storage/thermometer_l1 \
  --hygro_dir  /storage/hygrometer_l1 \
  --dendro_l2_dir /storage/dendrometer_l2 \
  --dendro_lm_dir /storage/dendrometer_lm \
  --train_site_ids_csv train_sites.csv \
  --test_site_ids_csv  test_sites.csv \
  --years 2019 \
  --stem_mode absolute \
  --input_mode best \
  --allow_missing_locals true \
  --min_local_coverage 0.50 \
  --tz Europe/Zurich
```

---

## 🎚️ Key Parameters and Behavior

### `--stem_mode` (default: `absolute`)
Controls how **L2 dendrometer** input is formed **before windowing** (10‑min grid):

- `absolute` → use raw binned L2 stem values.
- `delta` → use **first differences** (`s.diff()`), emphasizing *change* over *level*.

**Choosing between absolute vs delta**
- **absolute** if absolute position is meaningful and baselines are stable (e.g., instrument offsets are calibrated).
- **delta** if you want to remove baseline drift and emphasize short‑term dynamics (e.g., water‑related elastic response). Targets (LM) are **never differenced**.

---

### `--input_mode` (default: `combinations`)
Defines how thermometer/hygrometer inputs are chosen per site & window.

- `best` (recommended)
  1) Evaluate coverage for **every (T,RH,L2)** triple.  
  2) Pick the **single (T,RH)** with the **highest minimum coverage** across `(T, RH, stem)` under the policy in `--allow_missing_locals`.
  3) Deterministic, yields fewer but higher‑quality segments.
  
  Diagnostics: `no_best_combo_in_window_strict` (strict mode) or `no_best_combo_in_window` (substitution mode).

- `combinations`  
  Use **all (T,RH)** pairs at the site (bounded by `--max_combos_per_site`).

- `pooled`  
  Pooled median across site instruments (not typical for strict LM‑per‑ID pairing cases).

---

### `--allow_missing_locals` (default: `false`)
**Coverage policy** for local inputs and optional substitution from globals.

- `false` → **strict mode**  
  Only build segments where **T**, **RH**, and **stem** each satisfy:
  
  `coverage ≥ --min_local_coverage` (e.g., 0.50)
  
  No substitution. If nothing qualifies → skip window (`no_best_combo_in_window_strict`).

- `true` → **substitution mode**  
  **Stem** must satisfy the coverage threshold. **T**/**RH** may be substituted in the **inputs** by **globals** when local coverage is low:
  
  `T → g_tmean` (normalized to local T scale)  
  `RH → g_rh` (normalized to local RH scale)
  
  LM targets remain **ground truth** (no substitution).  
  Diagnostics include `used_global_T=true` / `used_global_RH=true` when applied.

---

## 📏 Coverage and Gating

Coverage is computed as the fraction of **non‑NaN** entries in the window.

Thresholds (`--min_local_coverage`) are enforced as follows:

| Mode                         | Stem (L2) | Thermometer | Hygrometer |
|------------------------------|-----------|-------------|------------|
| allow_missing_locals=false   | must pass | must pass   | must pass  |
| allow_missing_locals=true    | must pass | may substitute | may substitute |

If **stem** coverage is below threshold, the window is always rejected.

Diagnostics:
- `low_local_coverage_combo` — rejected; includes `cov_T`, `cov_RH`, `cov_ST`.
- `no_best_combo_in_window_strict` — no valid triple in strict mode.

---

## 📅 Calendar‑Year Windows

For each `YYYY` in `--years`, the window is:

```
[YYYY‑01‑01 00:00 UTC, (YYYY+1)‑01‑01 00:00 UTC)
```

- Inputs X: 52560 × 10‑min steps
- Targets y: 8760 × 1‑hour steps
- Leap day removed

---

## 🎯 LM Target Pairing (per‑dendrometer‑ID)

- LM series ID must match the L2 dendrometer `series_id`.
- Preferred hourly file: `_series_id_<id>.ftr`.
- Fallback: `dendrometer_lm_series_id_<id>.ftr` (10‑min) → resampled to 1h UTC.

LM target channels: `['stem', 'local_T', 'local_RH']`.

Diagnostics:
- `lm_hourly_targets_all_nan_after_reindex` — LM missing in the window.
- `lm_targets_all_nan_in_window` — rare edge case after normalization.

---

## 🔎 Usage Examples per Site

### Site 3 — strict, high‑quality (best selection)
```bash
python3 build_normalized_dataset_treenet_utc.py \
  --out_root ./out_site3_strict \
  --metadata_pickle /storage/metadata_all.pkl \
  --meteo_dir /storage/meteo_daily_civil \
  --thermo_dir /storage/thermometer_l1 \
  --hygro_dir  /storage/hygrometer_l1 \
  --dendro_l2_dir /storage/dendrometer_l2 \
  --dendro_lm_dir /storage/dendrometer_lm \
  --train_site_ids_csv site3.csv \
  --years 2019 \
  --stem_mode absolute \
  --input_mode best \
  --min_local_coverage 0.60 \
  --allow_missing_locals false
```

### Site 3 — robust with substitution (best selection)
```bash
python3 build_normalized_dataset_treenet_utc.py \
  --out_root ./out_site3_sub \
  --metadata_pickle /storage/metadata_all.pkl \
  --meteo_dir /storage/meteo_daily_civil \
  --thermo_dir /storage/thermometer_l1 \
  --hygro_dir  /storage/hygrometer_l1 \
  --dendro_l2_dir /storage/dendrometer_l2 \
  --dendro_lm_dir /storage/dendrometer_lm \
  --train_site_ids_csv site3.csv \
  --years 2019 \
  --stem_mode absolute \
  --input_mode best \
  --min_local_coverage 0.50 \
  --allow_missing_locals true
```

### Site 36 — many instruments (best selection, strict)
```bash
--input_mode best \
--allow_missing_locals false \
--min_local_coverage 0.70
```

### Site 10 — explore all combinations
```bash
--input_mode combinations \
--max_combos_per_site 3 \
--allow_missing_locals true
```

---

## 🧪 Troubleshooting

1) **Empty arrays** → check `diagnostics/diagnostics_preprocessing.csv` for:
   - `missing_instruments_at_site`
   - `no_dendro_id_intersection_for_site`
   - `no_best_combo_in_window_strict` / `no_best_combo_in_window`
   - `low_local_coverage_combo`
   - `lm_hourly_targets_all_nan_after_reindex`

2) **No windows** → sanity‑check the calendar window creation:
```python
from build_normalized_dataset_treenet_utc import make_multi_year_10m_index_utc, rolling_year_windows
idx = make_multi_year_10m_index_utc([2019])
print(rolling_year_windows(idx))  # expect one window for 2019
```

3) **LM availability** (per ID/year):
```python
from build_normalized_dataset_treenet_utc import read_lm_hourly_frame_utc
lm = read_lm_hourly_frame_utc(<dendro_id>, "/storage/dendrometer_lm", "Europe/Zurich")
print(len(lm['2019-01-01':'2019-12-31']))
```

---

## 📝 Notes
- Per‑ID pairing is **strict**: inputs for dendrometer ID `d` are paired only with LM targets for **the same ID**.
- Globals are normalized from daily civil data per site/year; locals are normalized lazily per scope.
- Use lowercase frequencies (`'10min'`, `'1h'`) to satisfy modern pandas.

If you want a paper‑quality Strategy‑B figure or per‑site/year best‑ID caching to accelerate selection, open an issue or ping the maintainer.


---

## ⚙️ CLI Parameters (Complete)

Below is the full list of command-line parameters supported by `build_normalized_dataset_treenet_utc.py`, their types, defaults, and usage.

- `--out_root` *(str, required)*: Root output directory where arrays, identifiers, normalizers, and diagnostics are written.
- `--metadata_pickle` *(str, required)*: Path to the metadata pickle containing instrument/site information.
- `--meteo_dir` *(str, required)*: Directory with daily civil global meteorology CSVs (`tas`, `tasmin`, `tasmax`, `rh`, `vpd`, `gh`, `pr`).
- `--thermo_dir` *(str, required)*: Directory containing thermometer local series (`thermometer_l1_series_id_<id>.ftr`).
- `--hygro_dir` *(str, required)*: Directory containing hygrometer local series (`hygrometer_l1_series_id_<id>.ftr`).
- `--dendro_l2_dir` *(str, required)*: Directory containing dendrometer L2 local series (`dendrometer_l2_series_id_<id>.ftr`).
- `--dendro_lm_dir` *(str, required)*: Directory containing **RAW LM** files (`dendrometer_lm_series_id_<id>.ftr`).
- `--train_site_ids_csv` *(str, required)*: CSV with a `site_id` column, listing sites for training.
- `--test_site_ids_csv` *(str, optional)*: CSV with a `site_id` column, listing sites for testing.
- `--years` *(int list, required)*: One or more years (e.g., `2019 2020`) forming the master UTC grid.
- `--per_year` *(str, default: `true`)*: If `true`, fit global scalers per year; if `false`, fit once for all years.
- `--tz` *(str, default: `Europe/Zurich`)*: Local timezone used for civil-day mapping and LM HH:00 selection.
- `--require_complete_locals` *(str, default: `false`)*: If `true`, enforce complete locals in discovery/selection.
- `--stem_mode` *(str, default: `absolute`)*: `absolute` or `delta`, applied to L2 stem inputs (targets are never differenced).
- `--input_mode` *(str, default: `combinations`)*: `best`, `combinations`, or `pooled` (site-instrument selection strategy).
- `--max_combos_per_site` *(int, default: None)*: Caps the number of (T,RH) pairs per site in `combinations` mode.
- `--min_local_coverage` *(float, default: `0.7`)*: Coverage gate for locals (fraction of non-NaNs in window).
- `--min_lm_series` *(int, default: `1`)*: Minimum LM series per site (kept for compatibility; LM reading is per-ID).
- `--overlap_days` *(int, default: `10`)*: Overlap used only for calendar-year windows (kept for compatibility).
- `--allow_missing_locals` *(str, default: `false`)*: If `true`, allow substituting **inputs** T/RH from globals; stem must pass.
- `--globals_broadcast_strategy` *(str, default: `civil_map`)*: Strategy-B civil-day mapping to UTC 10-min grid (only option).
- `--run_tests` *(flag)*: Runs a small Strategy-B self-test and exits.
- `--window_days` *(int, default: `365`)*: Window length in days. Use `30` for 30-day segments.
- `--window_stride_days` *(int, default: `1`)*: Stride in days for sliding windows (e.g., 1-day stride for 30-day windows).
- `--complete_only` *(str, default: `false`)*: If `true` **and** `window_days < 365`, enforce strict local completeness (no NaNs in inputs/targets). Year-long windows ignore this flag.

### Output file naming (by window length)
When `--window_days` is used, array and identifier filenames include the window length suffix:
- Train arrays: `X_train_<window_days>d.npy`, `y_train_<window_days>d.npy`, `site_ids_train_<window_days>d.npy`
- Train identifiers: `train_identifiers_<window_days>d.csv`
- Test arrays: `X_test_<window_days>d.npy`, `y_test_<window_days>d.npy`, `site_ids_test_<window_days>d.npy`
- Test identifiers: `test_identifiers_<window_days>d.csv`

Examples: `X_train_30d.npy`, `y_train_365d.npy`.
