#!/usr/bin/env python3
# inspect_segments.py (v8)
# Changes:
#  - Add legends for target panels (all modes)
#  - Plot **lines** instead of points for all channels (inputs, targets, globals)
#  - For **combo/year** figures, show **full DOY range (1..366)** for both inputs and targets
#  - For **per-segment** figures, keep dynamic DOY range (only days with data) as before
#  - Keep normalized globals overlay with --plot_globals_norm and --globals_agg {noon,daily_mean}

import os
import argparse
import pickle
from typing import Dict, List, Tuple, Sequence, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CHANNELS = [
    "temp_treenet","rh_treenet","stem","tas","tasmax","tasmin",
    "rh","vpd","gh","pr","doy"
]
TARGET_CHANNELS = ["local_T","local_RH","stem"]
GLOBAL_CHANNELS = ["tas","tasmax","tasmin","rh","vpd","gh","pr"]
LOCAL_TZ = "Europe/Zurich"

CONSISTENCY_INPUTS  = ["temp_treenet","rh_treenet","stem"]
CONSISTENCY_TARGETS = ["local_T","local_RH","stem"]

# ------------------------------- IO helpers ----------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_split_pickles(out_root: str, split: str):
    base = os.path.join(out_root, "processed", "model_data")
    combo_ids_path = os.path.join(base, f"model_{split.lower()}_data_combination_ids.pkl")
    in_segs_path   = os.path.join(base, f"{split.lower()}_input_segments.pkl")
    out_segs_path  = os.path.join(base, f"{split.lower()}_output_segments.pkl")
    seg_ids_path   = os.path.join(base, f"{split.lower()}_segment_ids.pkl")

    combo_ids = pickle.load(open(combo_ids_path, "rb"))
    in_segs   = pickle.load(open(in_segs_path,   "rb"))
    out_segs  = pickle.load(open(out_segs_path,  "rb"))
    seg_ids   = pickle.load(open(seg_ids_path,   "rb"))
    return combo_ids, in_segs, out_segs, seg_ids

# ---------------------------- Filtering helpers ------------------------------

def _extract_site_id(ids_row: pd.Series) -> int:
    try:
        return int(ids_row["site ID"])
    except Exception:
        return int(ids_row.iloc[0])

def _extract_combo_instrument_ids(ids_row: pd.Series) -> Tuple[int,int,int]:
    try:
        return int(ids_row["thermometer ID"]), int(ids_row["hygrometer ID"]), int(ids_row["dendrometer ID"])
    except Exception:
        return int(ids_row.iloc[1]), int(ids_row.iloc[2]), int(ids_row.iloc[3])

def filter_segments_by_year_site(
    combo_ids: Dict[int, pd.Series],
    in_segs: Dict[int, List[pd.DataFrame]],
    out_segs: Dict[int, List[pd.DataFrame]],
    year: int,
    site_id: int,
) -> List[Tuple[int,int,pd.DataFrame,pd.DataFrame,pd.Series]]:
    results: List[Tuple[int,int,pd.DataFrame,pd.DataFrame,pd.Series]] = []
    for cid, ids_row in combo_ids.items():
        if _extract_site_id(ids_row) != site_id:
            continue
        X_list = in_segs.get(cid, [])
        Y_list = out_segs.get(cid, [])
        for i, (X_df, Y_df) in enumerate(zip(X_list, Y_list)):
            if len(X_df) == 0 or len(Y_df) == 0:
                continue
            local_years = X_df.index.tz_convert(LOCAL_TZ).year
            if np.any(local_years == year):
                results.append((cid, i, X_df, Y_df, ids_row))
    return results

# ----------------------------- Plotting helpers ------------------------------

def make_colors(names: Sequence[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(names)}

def doy_from_index(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.tz_convert(LOCAL_TZ).dayofyear.to_numpy()

def format_interval(idx: pd.DatetimeIndex) -> str:
    loc = idx.tz_convert(LOCAL_TZ)
    start = pd.Timestamp(loc.min()).strftime("%Y-%m-%d %H:%M")
    end   = pd.Timestamp(loc.max()).strftime("%Y-%m-%d %H:%M")
    return f"{start} → {end} (local)"

# Coverage union

def union_coverage_pct(indices: List[pd.DatetimeIndex], full_index: pd.DatetimeIndex) -> float:
    if not indices or len(full_index) == 0:
        return 0.0
    union_idx = pd.DatetimeIndex([], tz=getattr(full_index, "tz", None))
    for idx in indices:
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
            union_idx = union_idx.union(idx)
    if len(union_idx) == 0:
        return 0.0
    return 100.0 * (len(union_idx) / float(len(full_index)))

# Year reference indices (UTC)

def make_year_indices_utc(year: int):
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    end   = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
    idx10 = pd.date_range(start=start, end=end, freq="10min")
    idx1h = pd.date_range(start=start, end=end, freq="1h")
    return idx10, idx1h

# ---------------------- Aggregation helpers ----------------------------------

def sample_noon_inputs(X_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if X_df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
    use_cols = [c for c in cols if c in X_df.columns]
    if not use_cols:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
    local_idx = X_df.index.tz_convert(LOCAL_TZ)
    df_loc = X_df[use_cols].copy()
    df_loc.index = local_idx
    rows = []
    for day, g in df_loc.groupby(df_loc.index.date):
        noon_ts = pd.Timestamp(pd.Timestamp(day).strftime("%Y-%m-%d") + " 12:00:00", tz=LOCAL_TZ)
        diffs = np.abs((g.index - noon_ts).to_series().dt.total_seconds()).values
        if len(diffs) == 0:
            continue
        i_min = int(np.argmin(diffs))
        if diffs[i_min] <= 20 * 60:
            row = g.iloc[i_min]
            row.name = noon_ts
            rows.append(row)
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
    out = pd.DataFrame(rows)
    return out

def sample_noon_targets(Y_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if Y_df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
    use_cols = [c for c in cols if c in Y_df.columns]
    if not use_cols:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
    local_idx = Y_df.index.tz_convert(LOCAL_TZ)
    df_loc = Y_df[use_cols].copy()
    df_loc.index = local_idx
    df_noon = df_loc[(df_loc.index.hour == 12) & (df_loc.index.minute == 0)]
    return df_noon

def daily_mean_local(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=LOCAL_TZ))
    df_loc = df[use_cols].copy()
    df_loc.index = df.index.tz_convert(LOCAL_TZ)
    daily = df_loc.resample('D').mean()
    return daily

def local_time_of_day_hours(idx: pd.DatetimeIndex) -> np.ndarray:
    loc = idx.tz_convert(LOCAL_TZ)
    return loc.hour + loc.minute/60.0

# -------------------- Figure type 1: per-segment ------------------------------

def plot_segment(
    year: int,
    site_id: int,
    combo_id: int,
    seg_idx: int,
    ids_row: pd.Series,
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    fig_out_dir: str,
    input_cols: Sequence[str],
    seg_cov_in_pct: float = 100.0,
    seg_cov_tg_pct: float = 100.0,
    noon_daily: bool = False,
    native_points: bool = False,
    plot_globals_norm: bool = False,
    globals_agg: str = 'noon',
) -> None:
    ensure_dir(fig_out_dir)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], hspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    if noon_daily:
        in_cols = [c for c in CONSISTENCY_INPUTS if c in X_df.columns]
        tgt_cols = [c for c in CONSISTENCY_TARGETS if c in Y_df.columns]
        X_plot = sample_noon_inputs(X_df, in_cols)
        Y_plot = sample_noon_targets(Y_df, tgt_cols)

        in_colors = make_colors(in_cols)
        x_doy = doy_from_index(X_plot.index)
        for col in in_cols:
            ax1.plot(x_doy, X_plot[col].to_numpy(), label=col, color=in_colors[col], lw=1.2)
        ax1.set_ylabel("Normalized (inputs @ 12:00)")
        ax1.grid(True, alpha=0.3)

        if plot_globals_norm:
            g_cols = [c for c in GLOBAL_CHANNELS if c in X_df.columns]
            if g_cols:
                ax1b = ax1.twinx()
                if globals_agg == 'daily_mean':
                    Xg_day = daily_mean_local(X_df, g_cols)
                    g_doy = doy_from_index(Xg_day.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_day[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=1.0)
                else:
                    Xg_noon = sample_noon_inputs(X_df, g_cols)
                    g_doy = doy_from_index(Xg_noon.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_noon[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=1.0)
                ax1b.set_ylabel("Globals (normalized)")
                h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax1b.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right", ncol=2, fontsize=9, framealpha=0.5)

        tgt_colors = make_colors(tgt_cols)
        y_doy = doy_from_index(Y_plot.index)
        for col in tgt_cols:
            ax2.plot(y_doy, Y_plot[col].to_numpy(), label=col, color=tgt_colors[col], lw=1.2)
        ax2.set_ylabel("Normalized (targets @ 12:00)")
        ax2.set_xlabel("Day of year")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.5)

        all_doy = np.concatenate([x_doy, y_doy]) if len(y_doy) > 0 else x_doy
        if len(all_doy) > 0:
            xmin, xmax = int(all_doy.min()), int(all_doy.max())
            ax1.set_xlim(max(1, xmin), min(366, xmax))
            ax2.set_xlim(max(1, xmin), min(366, xmax))

    elif native_points:
        # Now plot **lines** vs time-of-day
        in_cols = [c for c in input_cols if c in X_df.columns]
        tgt_cols = [c for c in CONSISTENCY_TARGETS if c in Y_df.columns]
        in_colors = make_colors(in_cols)
        x_tod = local_time_of_day_hours(X_df.index)
        for col in in_cols:
            ax1.plot(x_tod, X_df[col].to_numpy(), label=col, color=in_colors[col], lw=0.8)
        ax1.set_ylabel("Normalized (inputs)")
        ax1.set_xlabel("Time of day (hours)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 24)

        if plot_globals_norm:
            g_cols = [c for c in GLOBAL_CHANNELS if c in X_df.columns]
            if g_cols:
                ax1b = ax1.twinx()
                if globals_agg == 'daily_mean':
                    Xg_day = daily_mean_local(X_df, g_cols)
                    g_doy = doy_from_index(Xg_day.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_day[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                else:
                    Xg_noon = sample_noon_inputs(X_df, g_cols)
                    g_doy = doy_from_index(Xg_noon.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_noon[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                ax1b.set_ylabel("Globals (normalized)")
                h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax1b.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right", ncol=2, fontsize=9, framealpha=0.5)

        tgt_colors = make_colors(tgt_cols)
        y_tod = local_time_of_day_hours(Y_df.index)
        for col in tgt_cols:
            ax2.plot(y_tod, Y_df[col].to_numpy(), label=col, color=tgt_colors[col], lw=0.9)
        ax2.set_ylabel("Normalized (targets)")
        ax2.set_xlabel("Time of day (hours)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        ax2.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.5)

    else:
        in_cols = [c for c in input_cols if c in X_df.columns]
        in_colors = make_colors(in_cols)
        x_doy = doy_from_index(X_df.index)
        for col in in_cols:
            ax1.plot(x_doy, X_df[col].to_numpy(), label=col, color=in_colors[col], lw=1.0)
        ax1.set_ylabel("Normalized (inputs)")
        ax1.grid(True, alpha=0.3)

        if plot_globals_norm:
            g_cols = [c for c in GLOBAL_CHANNELS if c in X_df.columns]
            if g_cols:
                ax1b = ax1.twinx()
                if globals_agg == 'daily_mean':
                    Xg_day = daily_mean_local(X_df, g_cols)
                    g_doy = doy_from_index(Xg_day.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_day[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                else:
                    Xg_noon = sample_noon_inputs(X_df, g_cols)
                    g_doy = doy_from_index(Xg_noon.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_noon[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                ax1b.set_ylabel("Globals (normalized)")
                h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax1b.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right", ncol=2, fontsize=9, framealpha=0.5)

        tgt_cols = [c for c in TARGET_CHANNELS if c in Y_df.columns]
        tgt_colors = make_colors(tgt_cols)
        y_doy = doy_from_index(Y_df.index)
        for col in tgt_cols:
            ax2.plot(y_doy, Y_df[col].to_numpy(), label=col, color=tgt_colors[col], lw=1.2)
        ax2.set_ylabel("Normalized (targets)")
        ax2.set_xlabel("Day of year")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.5)

        all_doy = np.concatenate([x_doy, y_doy]) if len(y_doy) > 0 else x_doy
        if len(all_doy) > 0:
            xmin, xmax = int(all_doy.min()), int(all_doy.max())
            ax1.set_xlim(max(1, xmin), min(366, xmax))
            ax2.set_xlim(max(1, xmin), min(366, xmax))

    # Title
    therm_id, hygro_id, dendro_id = _extract_combo_instrument_ids(ids_row)
    interval = format_interval(X_df.index)
    title = (
        f"Year {year} • Site {site_id} • Combo {combo_id} • Segment {seg_idx}\n"
        f"Thermo {therm_id} • Hygro {hygro_id} • Dendro {dendro_id} • Interval: {interval}"
        f" • cov_in={seg_cov_in_pct:.1f}% • cov_tg={seg_cov_tg_pct:.1f}%"
    )
    fig.suptitle(title, fontsize=12)
    if not (noon_daily or native_points):
        ax1.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.5)

    fn = os.path.join(fig_out_dir, f"segment_y{year}_site{site_id}_combo{combo_id}_seg{seg_idx}.png")
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ------------------- Figure type 2: combo/year overlay -----------------------

def plot_combo_year(
    year: int,
    site_id: int,
    combo_id: int,
    ids_row: pd.Series,
    X_list: List[pd.DataFrame],
    Y_list: List[pd.DataFrame],
    fig_out_dir: str,
    input_cols: Sequence[str],
    combo_cov_in_pct: float,
    combo_cov_tg_pct: float,
    noon_daily: bool = False,
    native_points: bool = False,
    plot_globals_norm: bool = False,
    globals_agg: str = 'noon',
) -> None:
    ensure_dir(fig_out_dir)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], hspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    if noon_daily:
        in_cols = [c for c in CONSISTENCY_INPUTS]
        tgt_cols = [c for c in CONSISTENCY_TARGETS]
        in_colors = make_colors(in_cols)
        for seg_i, X_df in enumerate(X_list):
            X_noon = sample_noon_inputs(X_df, in_cols)
            x_doy = doy_from_index(X_noon.index)
            for col in in_cols:
                ax1.plot(x_doy, X_noon[col].to_numpy(), label=col if seg_i==0 else None,
                         color=in_colors[col], lw=0.9)
        ax1.set_ylabel("Normalized (inputs @ 12:00)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 366)  # full year

        if plot_globals_norm and len(X_list)>0:
            g_cols = [c for c in GLOBAL_CHANNELS if c in X_list[0].columns]
            if g_cols:
                ax1b = ax1.twinx()
                if globals_agg == 'daily_mean':
                    Xg_day = daily_mean_local(X_list[0], g_cols)
                    g_doy = doy_from_index(Xg_day.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_day[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                else:
                    Xg_noon = sample_noon_inputs(X_list[0], g_cols)
                    g_doy = doy_from_index(Xg_noon.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_noon[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                ax1b.set_ylabel("Globals (normalized)")
                h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax1b.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right", ncol=2, fontsize=9, framealpha=0.5)

        tgt_colors = make_colors(tgt_cols)
        for seg_i, Y_df in enumerate(Y_list):
            Y_noon = sample_noon_targets(Y_df, tgt_cols)
            y_doy = doy_from_index(Y_noon.index)
            for col in tgt_cols:
                ax2.plot(y_doy, Y_noon[col].to_numpy(), label=col if seg_i==0 else None,
                         color=tgt_colors[col], lw=1.0)
        ax2.set_ylabel("Normalized (targets @ 12:00)")
        ax2.set_xlabel("Day of year")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, 366)  # full year
        ax2.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.5)

    elif native_points:
        in_cols = [c for c in input_cols]
        tgt_cols = [c for c in CONSISTENCY_TARGETS]
        in_colors = make_colors(in_cols)
        for seg_i, X_df in enumerate(X_list):
            x_tod = local_time_of_day_hours(X_df.index)
            for col in in_cols:
                ax1.plot(x_tod, X_df[col].to_numpy(), label=col if seg_i==0 else None,
                         color=in_colors[col], lw=0.8)
        ax1.set_ylabel("Normalized (inputs)")
        ax1.set_xlabel("Time of day (hours)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 24)

        if plot_globals_norm and len(X_list)>0:
            g_cols = [c for c in GLOBAL_CHANNELS if c in X_list[0].columns]
            if g_cols:
                ax1b = ax1.twinx()
                if globals_agg == 'daily_mean':
                    Xg_day = daily_mean_local(X_list[0], g_cols)
                    g_doy = doy_from_index(Xg_day.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_day[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                else:
                    Xg_noon = sample_noon_inputs(X_list[0], g_cols)
                    g_doy = doy_from_index(Xg_noon.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_noon[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                ax1b.set_ylabel("Globals (normalized)")
                h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax1b.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right", ncol=2, fontsize=9, framealpha=0.5)

        tgt_colors = make_colors(tgt_cols)
        for seg_i, Y_df in enumerate(Y_list):
            y_tod = local_time_of_day_hours(Y_df.index)
            for col in tgt_cols:
                ax2.plot(y_tod, Y_df[col].to_numpy(), label=col if seg_i==0 else None,
                         color=tgt_colors[col], lw=0.9)
        ax2.set_ylabel("Normalized (targets)")
        ax2.set_xlabel("Time of day (hours)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        ax2.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.5)

    else:
        in_cols = [c for c in input_cols if len(X_list)>0 and c in X_list[0].columns]
        in_colors = make_colors(in_cols)
        for seg_i, X_df in enumerate(X_list):
            x_doy = doy_from_index(X_df.index)
            for col in in_cols:
                ax1.plot(x_doy, X_df[col].to_numpy(), label=col if seg_i==0 else None,
                         color=in_colors[col], lw=0.9)
        ax1.set_ylabel("Normalized (inputs)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 366)  # full year

        if plot_globals_norm and len(X_list)>0:
            g_cols = [c for c in GLOBAL_CHANNELS if c in X_list[0].columns]
            if g_cols:
                ax1b = ax1.twinx()
                if globals_agg == 'daily_mean':
                    Xg_day = daily_mean_local(X_list[0], g_cols)
                    g_doy = doy_from_index(Xg_day.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_day[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                else:
                    Xg_noon = sample_noon_inputs(X_list[0], g_cols)
                    g_doy = doy_from_index(Xg_noon.index)
                    g_colors = make_colors(g_cols)
                    for gc in g_cols:
                        ax1b.plot(g_doy, Xg_noon[gc].to_numpy(), label=f"global_norm:{gc}", color=g_colors[gc], lw=0.9)
                ax1b.set_ylabel("Globals (normalized)")
                h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax1b.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc="upper right", ncol=2, fontsize=9, framealpha=0.5)

        tgt_cols = [c for c in TARGET_CHANNELS if len(Y_list)>0 and c in Y_list[0].columns]
        tgt_colors = make_colors(tgt_cols)
        for seg_i, Y_df in enumerate(Y_list):
            y_doy = doy_from_index(Y_df.index)
            for col in tgt_cols:
                ax2.plot(y_doy, Y_df[col].to_numpy(), label=col if seg_i==0 else None,
                         color=tgt_colors[col], lw=1.0)
        ax2.set_ylabel("Normalized (targets)")
        ax2.set_xlabel("Day of year")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, 366)  # full year
        ax2.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.5)

    therm_id, hygro_id, dendro_id = _extract_combo_instrument_ids(ids_row)
    title = (
        f"Year {year} • Site {site_id} • Combo {combo_id}\n"
        f"Thermo {therm_id} • Hygro {hygro_id} • Dendro {dendro_id}"
        f" • cov_in={combo_cov_in_pct:.1f}% • cov_tg={combo_cov_tg_pct:.1f}%"
    )
    fig.suptitle(title, fontsize=12)

    fn = os.path.join(fig_out_dir, f"combo_year_y{year}_site{site_id}_combo{combo_id}.png")
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------- Main -------------------------------------

def parse_inputs_filter(raw: str) -> List[str]:
    if not raw:
        return INPUT_CHANNELS.copy()
    wanted = [s.strip() for s in raw.split(',') if s.strip()]
    valid = [c for c in wanted if c in INPUT_CHANNELS]
    invalid = sorted(set(wanted) - set(valid))
    if invalid:
        print(f"[WARN] Ignoring invalid input channels: {', '.join(invalid)}")
    return valid if valid else INPUT_CHANNELS.copy()

def main():
    p = argparse.ArgumentParser(description="Cross-inspection figures for TreeNet 30-day segments.")
    p.add_argument("--out_root", required=True, help="Output root used by the builder (e.g., /home/.../outputs_30d)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--site_id", type=int, required=True)
    p.add_argument("--split", choices=["TRAIN","TEST","both"], default="both")
    p.add_argument("--fig_out", required=True, help="Directory to write figures")
    p.add_argument("--inputs_filter", default="", help="Comma-separated subset of input channels to plot (default: all)")
    p.add_argument("--noon_daily", action="store_true", help="Plot only 3 consistency channels sampled at local noon (12:00) vs DOY")
    p.add_argument("--native_points", action="store_true", help="Plot native resolution as lines with x-axis = local time-of-day (hours)")
    p.add_argument("--plot_globals_norm", action="store_true", help="Overlay normalized global channels from processed inputs")
    p.add_argument("--globals_agg", choices=['noon','daily_mean'], default='noon', help="Aggregation for normalized globals when overlayed")
    args = p.parse_args()

    if args.noon_daily and args.native_points:
        print("[WARN] --noon_daily and --native_points both set; defaulting to --noon_daily.")
        args.native_points = False

    input_cols = parse_inputs_filter(args.inputs_filter)
    if args.noon_daily:
        input_cols = CONSISTENCY_INPUTS.copy()
        print("[INFO] Using noon sampling for channels:", ', '.join(CONSISTENCY_INPUTS + CONSISTENCY_TARGETS))
    elif input_cols != INPUT_CHANNELS:
        print(f"[INFO] Plotting input channels: {', '.join(input_cols)}")

    splits = ["TRAIN","TEST"] if args.split == "both" else [args.split]
    idx10_year, idx1h_year = make_year_indices_utc(args.year)

    for split in splits:
        combo_ids, in_segs, out_segs, seg_ids = load_split_pickles(args.out_root, split)
        filtered = filter_segments_by_year_site(combo_ids, in_segs, out_segs, args.year, args.site_id)
        if not filtered:
            print(f"No segments found for year {args.year}, site {args.site_id} in {split}.")
            continue

        seg_out_dir = os.path.join(args.fig_out, f"year_{args.year}", f"site_{args.site_id}", split, "segments")
        ensure_dir(seg_out_dir)
        for cid, seg_idx, X_df, Y_df, ids_row in filtered:
            combo_dir = os.path.join(seg_out_dir, f"combo_{cid}")
            ensure_dir(combo_dir)
            plot_segment(
                args.year, args.site_id, cid, seg_idx, ids_row, X_df, Y_df,
                combo_dir, input_cols, seg_cov_in_pct=100.0, seg_cov_tg_pct=100.0,
                noon_daily=args.noon_daily, native_points=args.native_points,
                plot_globals_norm=args.plot_globals_norm, globals_agg=args.globals_agg,
            )

        combo_out_dir = os.path.join(args.fig_out, f"year_{args.year}", f"site_{args.site_id}", split, "combos_year")
        ensure_dir(combo_out_dir)
        by_combo: Dict[int, Tuple[List[pd.DataFrame], List[pd.DataFrame], pd.Series]] = {}
        for cid, seg_idx, X_df, Y_df, ids_row in filtered:
            if cid not in by_combo:
                by_combo[cid] = ([], [], ids_row)
            by_combo[cid][0].append(X_df)
            by_combo[cid][1].append(Y_df)

        for cid, (X_list, Y_list, ids_row) in by_combo.items():
            in_indices = [X.index for X in X_list if len(X) > 0]
            tg_indices = [Y.index for Y in Y_list if len(Y) > 0]
            cov_in = union_coverage_pct(in_indices, idx10_year)
            cov_tg = union_coverage_pct(tg_indices, idx1h_year)
            plot_combo_year(
                args.year, args.site_id, cid, ids_row, X_list, Y_list,
                combo_out_dir, input_cols, cov_in, cov_tg,
                noon_daily=args.noon_daily, native_points=args.native_points,
                plot_globals_norm=args.plot_globals_norm, globals_agg=args.globals_agg,
            )

        print(
            f"Saved figures for {split}:\n"
            f" - segments   → {seg_out_dir}\n"
            f" - combos/year→ {combo_out_dir}"
        )

if __name__ == "__main__":
    main()
