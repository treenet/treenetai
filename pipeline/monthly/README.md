
# TreeNet 30‑day Segment Builder (UTC‑safe)

This repository contains a **drop‑in pipeline** that builds **strictly uninterrupted 30‑day segments** from TreeNet data, using a **UTC‑homogenized** workflow and Mirko's **jump‑ahead** segmentation logic. The builder produces both **intermediate files** (for traceability) and **final segment arrays** (NumPy pickles), together with a diagnostics CSV.

> **Key idea**: Only segments with **complete coverage** in **both** the 10‑minute **inputs** (11 channels) and the hourly **targets** (3 channels) are kept.

---

## 1. What the pipeline does (high‑level)

1. **Read & homogenize timestamps**
   - **Locals (10‑min)**: dendrometer L2 (stem), thermometer L1 (temp), hygrometer L1 (RH)
     - Convert timestamps to **Europe/Zurich**, then **UTC**.
     - Resample to **10‑min** and reindex to a **full‑year 10‑min UTC grid**.
   - **LM targets (hourly)**: dendrometer LM stem, LM temp, LM RH
     - Convert to **local civil time**; floor to **local hour** and aggregate.
     - Convert the hourly index to **UTC** and reindex to a **full‑year hourly UTC grid**.
   - **Globals (daily)**: meteotest daily variables (tas, tasmax, tasmin, rh, vpd, gh, pr)
     - Map by **local civil day** (Strategy‑B) and broadcast onto the **10‑min UTC grid**.

2. **Year‑level normalization (per combination)**
   - Compute normalization **across the entire selected year(s)** for inputs and targets.
   - Apply the same normalization to every segment (consistent scales across the year).

3. **Strict segmentation (30 days)**
   - Scan from the start of the year using the **jump‑ahead rule**:
     - Candidate window = **30 days**.
     - If any NaN occurs within the window (inputs or targets), **advance to the first timestamp after the last NaN** found in either granularity (10‑min or hourly), then try again.
     - Accept only windows with **full coverage** → 10‑min inputs = 4320 rows × 11 channels; hourly targets = 720 rows × 3 channels.
   - After a window is accepted, advance by an **accept stride** (default = **10 days**) and keep scanning.

4. **Write outputs**
   - Intermediate **combination files** (Feather) for inputs and targets.
   - Final **segment arrays** (NumPy pickles) for train/test.
   - Per‑segment **traceability** pickle, with normalization coefficients and channel provenance.
   - **Diagnostics** CSV with skip reasons per site/combo.

---

## 2. File outputs & their meaning

All outputs are written under:
```
<out_root>/processed/
```

### 2.1. Intermediate (per combination)
```
model_data/
  ├── train_input_combination_<combo_id>.ftr
  ├── train_output_combination_<combo_id>.ftr
  ├── test_input_combination_<combo_id>.ftr
  └── test_output_combination_<combo_id>.ftr
```
Each **combo** is a unique triple at a site: **(thermometer L1 ID, hygrometer L1 ID, dendrometer L2 ID)** paired with the corresponding **LM** target data. These files contain the **full‑year** normalized matrices aligned on UTC grids (10‑min for inputs, hourly for targets).

### 2.2. Segment arrays (NumPy pickles)
```
model_data/
  ├── train_input_segments_numpy.pkl    # shape (N_segments, 4320, 11)
  ├── train_output_segments_numpy.pkl   # shape (N_segments, 720, 3)
  ├── test_input_segments_numpy.pkl     # shape (N_segments, 4320, 11)
  └── test_output_segments_numpy.pkl    # shape (N_segments, 720, 3)
```
These pickles store only **accepted segments** (strict completeness). The shapes are fixed by design: 30 days × 24 h × 6 = **4320** steps for inputs, and 30 days × 24 h = **720** steps for targets.

### 2.3. Segment lists (pandas) — optional inspection
```
model_data/
  ├── train_input_segments.pkl     # dict: combo_id → list[pd.DataFrame] (normalized input segments)
  ├── train_output_segments.pkl    # dict: combo_id → list[pd.DataFrame] (normalized target segments)
  ├── test_input_segments.pkl
  └── test_output_segments.pkl
```
Same content as the NumPy pickles, but kept as **pandas frames** for easy inspection.

### 2.4. Combination identifiers
```
model_data/
  ├── model_train_data_combination_ids.pkl
  └── model_test_data_combination_ids.pkl
```
Mapping **combo_id → IDs row** with columns: `['site ID','thermometer ID','hygrometer ID','dendrometer ID']`.

### 2.5. Per‑segment traceability (normalization + provenance)
```
model_data/
  ├── train_segment_ids.pkl
  └── test_segment_ids.pkl
```
Each entry corresponds to **one accepted segment** and contains:
- **combo_id** and **segment index**,
- the **IDs row** (site ID, instrument IDs),
- **normalization minima & differences** for input and target (year‑level),
- **UTC window bounds** (`window_start_utc`, `window_end_utc`),
- **channel lists** (input = 11 names; target = 3 names).

> **Invariant:** If a segment appears in `*_segment_ids.pkl`, it also appears in the corresponding `*_input_segments_numpy.pkl` and `*_output_segments_numpy.pkl` pickles.

### 2.6. Diagnostics
```
diagnostics/
  └── diagnostics_preprocessing.csv
```
A row per **skip** with one of these reasons:
- `missing_instruments`: the site lacks one or more required sensors.
- `read_error_locals`: failure reading local sensor files.
- `resample_error_locals`: failure resampling locals to 10‑min.
- `meteo_error`: failure processing daily meteo.
- `lm_error`: failure building LM hourly.
- `no_complete_windows`: no strictly complete 30‑day window found for that **combo**.
- `length_mismatch`: a sliced window did not have the expected number of rows.

> You can have **accepted segments** and **`no_complete_windows`** simultaneously: some combos at a site may meet strict completeness, others may not.

---

## 3. CLI parameters

```bash
python3 build_30day_dropin_utc.py \
  --out_root /path/to/outputs_30d \
  --metadata_pickle /path/to/metadata_all.pkl \
  --meteo_dir /path/to/meteo_daily_csvs \
  --thermo_dir /path/to/thermometer_l1 \
  --hygro_dir  /path/to/hygrometer_l1 \
  --dendro_l2_dir /path/to/dendrometer_l2 \
  --dendro_lm_dir /path/to/dendrometer_lm \
  --train_site_ids_csv train_sites.csv \
  --test_site_ids_csv  test_sites.csv \
  --years 2019 \
  --window_days 30 \
  --stride_days_after_accept 10
```

- `--out_root` : Root directory where outputs are written.
- `--metadata_pickle` : Pickle with instrument metadata (`series_id`, `site_id`).
- `--meteo_dir` : Folder with **daily** meteo CSVs (one per site). Column `ts` must be a date string (e.g., `YYYY-MM-DD`).
- `--thermo_dir`, `--hygro_dir`, `--dendro_l2_dir`, `--dendro_lm_dir` : Folders with Feather files named like `thermometer_l1_series_id_<ID>.ftr`, etc., with `ts` tz‑aware (`Etc/GMT-1`).
- `--train_site_ids_csv`, `--test_site_ids_csv` : CSVs with a `site_id` column listing sites for each split.
- `--years` : One or more years to build; the builder aligns entire year grids (10‑min and 1‑h UTC) and normalizes **across the provided years**.
- `--window_days` : Segment window length in days (default: 30).
- `--stride_days_after_accept` : How many days to advance the start pointer after an accepted window (default: 10). The jump‑ahead logic always uses "(last NaN + 10 min)" when a candidate window is incomplete.

---

## 4. Normalization: how it works

**Scope:** Normalization is computed **year‑wide** (across all timestamps in the selected year(s)) for each combo, then applied to all segments derived from that combo.

**Method (column‑wise):**
- Special numeric columns:
  - `hour` → `hour / 24`
  - `doy`  → `doy / 365`
  - `month`→ `month / 12`
- All other numeric channels:
  - Compute **min** and **max** with `skipna=True`.
  - If both min and max are finite and `|max - min| > 1e-4` → scale to **[0, 1]**: `(x - min) / (max - min)`.
  - If the range is tiny (`≤ 1e-4`) → **shift only**: `x - min` (avoid division by near‑zero).
  - If a column is **entirely NaN** → leave as NaN; record **`diff = NaN`** in the normalization metadata.

**Missing values:**
- **Ignored** when computing **min** and **max** (i.e., they don’t affect the statistics), thanks to `skipna=True`.
- **Preserved** during normalization (NaNs remain NaNs). In strict segmentation, **windows with NaNs are rejected**; otherwise, NaNs may appear in intermediate matrices but never in accepted segments.

**Traceability:** For every accepted segment, the builder stores the **normalization minima and differences** used (both for input and target) in `train_segment_ids.pkl` / `test_segment_ids.pkl`. This allows full **reproducibility** and **inverse scaling** later on.

---

## 5. Interpreting diagnostics (`diagnostics_preprocessing.csv`)

Examples:
- `no_complete_windows` with `combo_id = 7` means that the specific thermo/hygro/dendro triple at the site had **no fully complete** 30‑day span in the selected year(s), even after jump‑ahead scanning.
- `lm_error` often indicates issues converting LM timestamps or aggregating hourly; check the LM files (`dendrometer_lm_series_id_<ID>.ftr`).
- `resample_error_locals` flags problems resampling 10‑min locals; verify the local feather file formats and that `ts` is tz‑aware.

> You can still get accepted segments for other combos at the same site while some combos emit `no_complete_windows`.

---

## 6. Loading the produced arrays

```python
import pickle
base = '/home/lukovic/data/treenet/outputs_30d/processed/model_data'
X_train = pickle.load(open(f'{base}/train_input_segments_numpy.pkl', 'rb'))
y_train = pickle.load(open(f'{base}/train_output_segments_numpy.pkl','rb'))
X_test  = pickle.load(open(f'{base}/test_input_segments_numpy.pkl',  'rb'))
y_test  = pickle.load(open(f'{base}/test_output_segments_numpy.pkl', 'rb'))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# Expected: (N, 4320, 11) and (N, 720, 3)
```

If you need `.npy` files for downstream tooling:
```python
import numpy as np, pickle, os
base = '/home/lukovic/data/treenet/outputs_30d/processed/model_data'
out  = '/home/lukovic/data/treenet/outputs_30d'
os.makedirs(out, exist_ok=True)
np.save(f'{out}/X_train_30d.npy', pickle.load(open(f'{base}/train_input_segments_numpy.pkl','rb')))
np.save(f'{out}/y_train_30d.npy', pickle.load(open(f'{base}/train_output_segments_numpy.pkl','rb')))
np.save(f'{out}/X_test_30d.npy',  pickle.load(open(f'{base}/test_input_segments_numpy.pkl','rb')))
np.save(f'{out}/y_test_30d.npy',  pickle.load(open(f'{base}/test_output_segments_numpy.pkl','rb')))
```

---

## 7. Notes & future options

- If you want more segments and can tolerate slight gaps in LM, add a parameter like `--min_target_coverage 0.95` (to accept windows with ≤ 5% missing hourly targets) and log missing counts per window.
- If you plan to pool normalization across **multiple years** (e.g., site‑wide scaling for 2018–2020), we can add `--per_year false` and store aggregated normalization metadata alongside per‑year records.
- If you’d like the builder to also emit `.npy` by default, we can add a small block after saving the pickles to write `X_train_30d.npy`, `y_train_30d.npy`, etc.

---

## 8. Troubleshooting checklist

- **Empty arrays**: Make sure you’re loading the correct **30‑day output** folder (`outputs_30d/processed/model_data`) and the correct format (pickles vs `.npy`).
- **`no_complete_windows`**: Inspect year coverage for LM and locals (10‑min). Some combos may be sparse in the selected year.
- **Timestamp errors**: Verify that feather `ts` is tz‑aware (`Etc/GMT-1`) and that meteo `ts` is a date string (`YYYY-MM-DD`).

---

**Contact / Maintainer**: Mirko Lukovic

