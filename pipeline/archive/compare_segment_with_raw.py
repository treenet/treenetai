#!/usr/bin/env python3
# compare_segment_with_raw.py (v2)
#
# PURPOSE
#   Compare denormalized 30-day segment data (original scale) to **raw source files**
#   for strict QA. Supports **batch processing of all segments** for a given combo.
#
# INPUT CHANNELS (10): temp_treenet, rh_treenet, stem(L2), tas, tasmax, tasmin, rh, vpd, gh, pr
# TARGET CHANNELS (3): local_T (hourly), local_RH (hourly), stem(LM hourly)
#
# RAW FILES (under --raw_root):
#   - temperature_l1_series_id_<thermo_id>.ftr    (10-min)
#   - hygrometer_l1_series_id_<hygro_id>.ftr      (10-min)
#   - dendrometer_l2_series_id_<dendro_id>.ftr    (10-min)
#   - dendrometer_lm_series_id_<lm_dendro_id>.ftr (hourly)
#   - site_<site_id>.csv                          (daily globals)
#
# USAGE EXAMPLES:
#   # Single segment
#   python3 compare_segment_with_raw.py \
#     --out_root /home/lukovic/data/treenet/outputs_30d \
#     --raw_root /storage/lukovic/Data/FORWARDS/treenet/raw \
#     --year 2019 --site_id 3 --split TRAIN --combo_id 22 --seg_idx 0 \
#     --fig_out /home/lukovic/data/treenet/outputs_30d/processed/figures_compare_raw
#
#   # Batch: all segments for combo_id=22
#   python3 compare_segment_with_raw.py \
#     --out_root /home/lukovic/data/treenet/outputs_30d \
#     --raw_root /storage/lukovic/Data/FORWARDS/treenet/raw \
#     --year 2019 --site_id 3 --split TRAIN --combo_id 22 --all_segments \
#     --fig_out /home/lukovic/data/treenet/outputs_30d/processed/figures_compare_raw
#
import os
import argparse
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOCAL_TZ = 'Europe/Zurich'
INPUT_CHANNELS = ['temp_treenet','rh_treenet','stem','tas','tasmax','tasmin','rh','vpd','gh','pr','doy']
TARGET_CHANNELS = ['local_T','local_RH','stem']
INPUT_CHANNELS_WO_DOY = [c for c in INPUT_CHANNELS if c != 'doy']
GLOBAL_CHANNELS = ['tas','tasmax','tasmin','rh','vpd','gh','pr']

# ---------------- IO helpers ----------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_pickles(out_root: str, split: str):
    base = os.path.join(out_root, 'processed', 'model_data')
    combo_ids = pickle.load(open(os.path.join(base, f'model_{split.lower()}_data_combination_ids.pkl'), 'rb'))
    in_segs = pickle.load(open(os.path.join(base, f'{split.lower()}_input_segments.pkl'), 'rb'))
    out_segs = pickle.load(open(os.path.join(base, f'{split.lower()}_output_segments.pkl'), 'rb'))
    seg_ids  = pickle.load(open(os.path.join(base, f'{split.lower()}_segment_ids.pkl'), 'rb'))
    return base, combo_ids, in_segs, out_segs, seg_ids

# ------------- seg_ids parsing -------------

def find_segment_meta(seg_ids_obj, combo_id: int, seg_idx: int):
    def _normalize_entry(entry):
        if isinstance(entry, (list, tuple)) and len(entry) >= 9:
            e_combo = entry[0]; e_seg = entry[1]
            if int(e_combo) == combo_id and int(e_seg) == seg_idx:
                ids_row = entry[2]
                in_min  = entry[3]; in_diff = entry[4]
                out_min = entry[5]; out_diff = entry[6]
                win     = entry[7]
                chans   = entry[8]
                try:
                    ws = pd.to_datetime(win.get('window_start_utc')).tz_convert('UTC')
                    we = pd.to_datetime(win.get('window_end_utc')).tz_convert('UTC')
                except Exception:
                    ws = pd.to_datetime(win.get('window_start_utc')).tz_localize('UTC')
                    we = pd.to_datetime(win.get('window_end_utc')).tz_localize('UTC')
                return {
                    'ids_row': ids_row,
                    'in_min': in_min, 'in_diff': in_diff,
                    'out_min': out_min, 'out_diff': out_diff,
                    'win_start_utc': ws, 'win_end_utc': we,
                    'input_channels': chans.get('input_channels', INPUT_CHANNELS),
                    'target_channels': chans.get('target_channels', TARGET_CHANNELS)
                }
        return None
    if isinstance(seg_ids_obj, dict):
        bucket = seg_ids_obj.get(combo_id, None)
        if bucket is not None:
            for entry in bucket:
                meta = _normalize_entry(entry)
                if meta is not None:
                    return meta
        for _, lst in seg_ids_obj.items():
            try:
                for entry in lst:
                    meta = _normalize_entry(entry)
                    if meta is not None:
                        return meta
            except Exception:
                continue
    if isinstance(seg_ids_obj, list):
        for entry in seg_ids_obj:
            meta = _normalize_entry(entry)
            if meta is not None:
                return meta
    raise RuntimeError(f"Segment meta not found for combo_id={combo_id}, seg_idx={seg_idx}")

# ------------- denormalization -------------

def inv_normalize(series: pd.Series, mn: float, diff: float) -> pd.Series:
    try:
        d = float(diff) if diff is not None else np.nan
    except Exception:
        d = np.nan
    try:
        m = float(mn) if mn is not None else np.nan
    except Exception:
        m = np.nan
    if np.isfinite(d) and d > 1e-8:
        return series * d + m
    else:
        return series + (0.0 if not np.isfinite(m) else m)

# ---------------- raw loaders -----------------

def read_ftr_one_value(raw_path: str) -> pd.DataFrame:
    if not os.path.isfile(raw_path):
        raise RuntimeError(f"Raw file not found: {raw_path}")
    df = pd.read_feather(raw_path)
    if 'ts' not in df.columns:
        raise RuntimeError(f"Raw feather missing 'ts': {raw_path}")
    ts = pd.to_datetime(df['ts'], errors='coerce', utc=True)
    df = df.drop(columns=['ts'])
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        raise RuntimeError(f"No numeric column in {raw_path}")
    val_col = num_cols[-1]
    out = pd.DataFrame({val_col: df[val_col].astype('float64')})
    out.index = ts
    out = out.sort_index()
    return out.rename(columns={val_col: 'value'})

def read_site_csv(raw_path: str, year: int) -> pd.DataFrame:
    if not os.path.isfile(raw_path):
        raise RuntimeError(f"Site CSV not found: {raw_path}")
    df = pd.read_csv(raw_path)
    ts_col = 'ts' if 'ts' in df.columns else ('ts_local' if 'ts_local' in df.columns else None)
    if ts_col is None:
        raise RuntimeError(f"Site CSV missing 'ts'/'ts_local': {raw_path}")
    ts = pd.to_datetime(df[ts_col], errors='coerce')
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(LOCAL_TZ, nonexistent='shift_forward', ambiguous='NaT')
    df = df.assign(ts_local=ts)
    df = df[df['ts_local'].dt.year == year]
    cols = [c for c in GLOBAL_CHANNELS if c in df.columns]
    return df[['ts_local'] + cols].copy()

# ---------------- plotting helpers -----------------

def doy_from_index(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.tz_convert(LOCAL_TZ).dayofyear.to_numpy()

# ---------------- core comparison -----------------

def compare_one_segment(
    year: int,
    site_id: int,
    combo_id: int,
    seg_idx: int,
    ids_row,
    in_min: Dict,
    in_diff: Dict,
    out_min: Dict,
    out_diff: Dict,
    ws_utc: pd.Timestamp,
    we_utc: pd.Timestamp,
    Xseg_norm: pd.DataFrame,
    Yseg_norm: pd.DataFrame,
    raw_temp: pd.DataFrame,
    raw_hygro: pd.DataFrame,
    raw_l2: pd.DataFrame,
    raw_lm: pd.DataFrame,
    raw_site: pd.DataFrame,
    fig_out_dir: str,
):
    # Denorm helpers
    def denorm_df(df_norm: pd.DataFrame, mins: Dict, diffs: Dict, cols: List[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df_norm.index)
        for c in cols:
            if c in df_norm.columns and c in mins and c in diffs:
                out[c] = inv_normalize(df_norm[c].astype('float64'), mins[c], diffs[c])
        return out

    xin_cols = [c for c in INPUT_CHANNELS_WO_DOY if c in Xseg_norm.columns]
    y_cols   = [c for c in TARGET_CHANNELS if c in Yseg_norm.columns]

    Xseg_orig = denorm_df(Xseg_norm, in_min, in_diff, xin_cols)
    Yseg_orig = denorm_df(Yseg_norm, out_min, out_diff, y_cols)

    # Slice raw window
    def slice_utc(df: pd.DataFrame) -> pd.DataFrame:
        ix = df.index
        if ix.tz is None:
            df = df.copy(); df.index = df.index.tz_localize('UTC')
        return df.loc[(df.index>=ws_utc) & (df.index<we_utc)]

    df_temp_w = slice_utc(raw_temp)
    df_hygro_w = slice_utc(raw_hygro)
    df_l2_w = slice_utc(raw_l2)
    df_lm_w = slice_utc(raw_lm)

    df_site_w = raw_site[(raw_site['ts_local']>=ws_utc.tz_convert(LOCAL_TZ)) & (raw_site['ts_local']<we_utc.tz_convert(LOCAL_TZ))]

    # Resampling for targets
    def to_hourly_mean(df: pd.DataFrame) -> pd.DataFrame:
        df_loc = df.copy(); df_loc.index = df_loc.index.tz_convert(LOCAL_TZ)
        hourly = df_loc.resample('H').mean()
        hourly.index = hourly.index.tz_convert('UTC')
        return hourly

    df_temp_hour = to_hourly_mean(df_temp_w)
    df_hygro_hour = to_hourly_mean(df_hygro_w)

    # Globals: daily mean of denorm segment
    def seg_daily_mean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        use = [c for c in cols if c in df.columns]
        if not use:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
        loc = df.copy(); loc.index = loc.index.tz_convert(LOCAL_TZ)
        daily = loc[use].resample('D').mean()
        return daily

    Xseg_daily_globals = seg_daily_mean(Xseg_orig, GLOBAL_CHANNELS)

    # Plotting
    out_dir = os.path.join(fig_out_dir, f"year_{year}", f"site_{site_id}", f"combo_{combo_id}", f"seg_{seg_idx}")
    ensure_dir(out_dir)

    def plot_two_lines(x1_idx: pd.DatetimeIndex, y1: np.ndarray, label1: str,
                       x2_idx: pd.DatetimeIndex, y2: np.ndarray, label2: str,
                       ch: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(doy_from_index(x1_idx), y1, color='tab:blue', lw=1.2, alpha=0.95, label=label1)
        ax.plot(doy_from_index(x2_idx), y2, color='black', lw=2.0, alpha=0.9, label=label2)
        ax.set_xlabel('Day of year (local)')
        ax.set_ylabel(f"{ch} (original scale)")
        ax.grid(True, alpha=0.3)
        all_doy = np.concatenate([doy_from_index(x1_idx), doy_from_index(x2_idx)])
        if len(all_doy)>0:
            ax.set_xlim(max(1, int(all_doy.min())), min(366, int(all_doy.max())))
        ax.legend(loc='best', fontsize=9, framealpha=0.6)
        fn = os.path.join(out_dir, f"y{year}_site{site_id}_combo{combo_id}_seg{seg_idx}_{ch}.png")
        title = (f"Year {year} • Site {site_id} • Combo {combo_id} • Segment {seg_idx}\n"
                 f"Channel: {ch} • Window: {ws_utc.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')} → {we_utc.tz_convert(LOCAL_TZ).strftime('%Y-%m-%d %H:%M')} (local)")
        fig.suptitle(title, fontsize=11)
        fig.savefig(fn, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Inputs
    if 'temp_treenet' in xin_cols:
        plot_two_lines(df_temp_w.index, df_temp_w['value'].to_numpy(), 'raw temperature L1',
                       Xseg_orig.index, Xseg_orig['temp_treenet'].to_numpy(), 'segment denorm', 'temp_treenet')
    if 'rh_treenet' in xin_cols:
        plot_two_lines(df_hygro_w.index, df_hygro_w['value'].to_numpy(), 'raw hygrometer L1',
                       Xseg_orig.index, Xseg_orig['rh_treenet'].to_numpy(), 'segment denorm', 'rh_treenet')
    if 'stem' in xin_cols:
        plot_two_lines(df_l2_w.index, df_l2_w['value'].to_numpy(), 'raw dendrometer L2',
                       Xseg_orig.index, Xseg_orig['stem'].to_numpy(), 'segment denorm', 'stem_input_L2')
    if not Xseg_daily_globals.empty:
        for gch in [c for c in GLOBAL_CHANNELS if c in Xseg_daily_globals.columns and c in df_site_w.columns]:
            plot_two_lines(df_site_w['ts_local'], df_site_w[gch].to_numpy(), f'raw site daily {gch}',
                           Xseg_daily_globals.index, Xseg_daily_globals[gch].to_numpy(), 'segment daily mean', gch)

    # Targets
    if 'local_T' in y_cols:
        plot_two_lines(df_temp_hour.index, df_temp_hour['value'].to_numpy(), 'raw L1 temp hourly mean',
                       Yseg_orig.index, Yseg_orig['local_T'].to_numpy(), 'segment denorm', 'local_T')
    if 'local_RH' in y_cols:
        plot_two_lines(df_hygro_hour.index, df_hygro_hour['value'].to_numpy(), 'raw L1 RH hourly mean',
                       Yseg_orig.index, Yseg_orig['local_RH'].to_numpy(), 'segment denorm', 'local_RH')
    if 'stem' in y_cols:
        plot_two_lines(df_lm_w.index, df_lm_w['value'].to_numpy(), 'raw dendrometer LM hourly',
                       Yseg_orig.index, Yseg_orig['stem'].to_numpy(), 'segment denorm', 'stem_target_LM')

# ------------------ main -------------------

def main():
    p = argparse.ArgumentParser(description='Compare denormalized segment(s) vs raw files on original scale.')
    p.add_argument('--out_root', required=True)
    p.add_argument('--raw_root', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--site_id', type=int, required=True)
    p.add_argument('--split', choices=['TRAIN','TEST'], required=True)
    p.add_argument('--combo_id', type=int, required=True)
    p.add_argument('--seg_idx', type=int, default=None, help='Segment index to process (use with single mode)')
    p.add_argument('--all_segments', action='store_true', help='Process all segments for the given combo_id')
    p.add_argument('--fig_out', required=True)
    p.add_argument('--lm_dendro_id', type=int, default=None, help='Override LM dendrometer ID if different from L2')
    args = p.parse_args()

    if (args.seg_idx is None) and (not args.all_segments):
        raise SystemExit("Provide either --seg_idx or --all_segments")
    if (args.seg_idx is not None) and args.all_segments:
        print('[WARN] Both --seg_idx and --all_segments provided; proceeding with --all_segments.')

    base, combo_ids, in_segs, out_segs, seg_ids = load_pickles(args.out_root, args.split)

    # IDs
    ids_row = combo_ids.get(args.combo_id, None)
    if ids_row is None:
        raise RuntimeError(f"combo_id={args.combo_id} not found in {args.split} combo file")
    try:
        site_check = int(ids_row['site ID'])
        thermo_id = int(ids_row['thermometer ID'])
        hygro_id  = int(ids_row['hygrometer ID'])
        dendro_id = int(ids_row['dendrometer ID'])
    except Exception:
        site_check = int(ids_row.iloc[0])
        thermo_id = int(ids_row.iloc[1])
        hygro_id  = int(ids_row.iloc[2])
        dendro_id = int(ids_row.iloc[3])
    lm_dendro_id = args.lm_dendro_id if args.lm_dendro_id is not None else dendro_id

    if site_check != args.site_id:
        print(f"[WARN] combo_id site ({site_check}) != requested site_id ({args.site_id}).")

    # Preload RAW once per batch
    raw_temp = read_ftr_one_value(os.path.join(args.raw_root, f"temperature_l1_series_id_{thermo_id}.ftr"))
    raw_hygro = read_ftr_one_value(os.path.join(args.raw_root, f"hygrometer_l1_series_id_{hygro_id}.ftr"))
    raw_l2    = read_ftr_one_value(os.path.join(args.raw_root, f"dendrometer_l2_series_id_{dendro_id}.ftr"))
    raw_lm    = read_ftr_one_value(os.path.join(args.raw_root, f"dendrometer_lm_series_id_{lm_dendro_id}.ftr"))
    raw_site  = read_site_csv(os.path.join(args.raw_root, f"site_{args.site_id}.csv"), args.year)

    # Collect segments list
    X_list = in_segs.get(args.combo_id, [])
    Y_list = out_segs.get(args.combo_id, [])
    if not X_list or not Y_list:
        raise RuntimeError(f"No segments for combo {args.combo_id}")

    seg_indices = list(range(min(len(X_list), len(Y_list)))) if args.all_segments else [args.seg_idx]

    for seg_idx in seg_indices:
        # Meta per segment (distinct min/diff/window)
        meta = find_segment_meta(seg_ids, args.combo_id, seg_idx)
        in_min, in_diff = meta['in_min'], meta['in_diff']
        out_min, out_diff = meta['out_min'], meta['out_diff']
        ws_utc, we_utc = meta['win_start_utc'], meta['win_end_utc']
        Xseg_norm = X_list[seg_idx]
        Yseg_norm = Y_list[seg_idx]

        compare_one_segment(
            year=args.year,
            site_id=args.site_id,
            combo_id=args.combo_id,
            seg_idx=seg_idx,
            ids_row=ids_row,
            in_min=in_min,
            in_diff=in_diff,
            out_min=out_min,
            out_diff=out_diff,
            ws_utc=ws_utc,
            we_utc=we_utc,
            Xseg_norm=Xseg_norm,
            Yseg_norm=Yseg_norm,
            raw_temp=raw_temp,
            raw_hygro=raw_hygro,
            raw_l2=raw_l2,
            raw_lm=raw_lm,
            raw_site=raw_site,
            fig_out_dir=args.fig_out,
        )

    print(f"Wrote comparison figures under: {os.path.join(args.fig_out, f'year_{args.year}', f'site_{args.site_id}', f'combo_{args.combo_id}')}\nSegments processed: {seg_indices}")

if __name__ == '__main__':
    main()
