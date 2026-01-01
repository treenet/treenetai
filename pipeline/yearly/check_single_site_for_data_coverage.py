
import os, re
import numpy as np
import pandas as pd

site_id = 22
tz = "Europe/Zurich"

thermo_dir = "/storage/lukovic/Data/FORWARDS/treenet/server_data/thermometer_l1"
hygro_dir  = "/storage/lukovic/Data/FORWARDS/treenet/server_data/hygrometer_l1"
d2_dir     = "/storage/lukovic/Data/FORWARDS/treenet/server_data/dendrometer_l2"

# Year-long window for 2014
ws = pd.Timestamp("2014-01-01 00:00:00", tz=tz)
we = pd.Timestamp("2015-01-01 00:00:00", tz=tz)

def strip_leap_days(idx):
    return idx[~((idx.month == 2) & (idx.day == 29))]

# 10-min grid (strict)
idx_10m = pd.date_range(ws, we - pd.Timedelta(minutes=10), freq="10min", tz=tz)
idx_10m = strip_leap_days(idx_10m)

# Load metadata
meta = pd.read_pickle("/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_all.pkl")[["series_id","site_id"]]

def ids_in_dir(dirp, prefix):
    pat = re.compile(rf"{re.escape(prefix)}_series_id_(\d+)\.ftr$")
    ids = []
    for fn in os.listdir(dirp):
        m = pat.match(fn)
        if m: ids.append(int(m.group(1)))
    return ids

thermo_ids = meta.loc[(meta["site_id"]==site_id) &
                      meta["series_id"].isin(ids_in_dir(thermo_dir, "thermometer_l1")), "series_id"].tolist()
hygro_ids  = meta.loc[(meta["site_id"]==site_id) &
                      meta["series_id"].isin(ids_in_dir(hygro_dir, "hygrometer_l1")), "series_id"].tolist()
d2_ids     = meta.loc[(meta["site_id"]==site_id) &
                      meta["series_id"].isin(ids_in_dir(d2_dir, "dendrometer_l2")), "series_id"].tolist()

def read_series(series_id, dirp):
    fp = None
    for fn in os.listdir(dirp):
        if fn.endswith(f"series_id_{series_id}.ftr"):
            fp = os.path.join(dirp, fn)
            break
    if fp is None:
        return pd.Series(dtype=float)
    df = pd.read_feather(fp)
    ts = pd.to_datetime(df["ts"], utc=False)
    ts = ts.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    s = pd.Series(df["value"].to_numpy(), index=ts).sort_index()
    s = s[(s.index >= ws) & (s.index < we)]
    # Reindex to strict grid (this is what the builder does)
    s = s.reindex(idx_10m)
    return s

def coverage(s):
    v = s.to_numpy()
    return float(np.sum(~np.isnan(v)) / v.size) if v.size else 0.0

print(f"Site {site_id} — 2014 window [{ws} .. {we})")
print("Thermometers:")
for sid in thermo_ids:
    c = coverage(read_series(sid, thermo_dir))
    print(f"  id={sid}: coverage={c:.3f}")

print("Hygrometers:")
for sid in hygro_ids:
    c = coverage(read_series(sid, hygro_dir))
    print(f"  id={sid}: coverage={c:.3f}")

print("Dendrometer L2 (stem):")
for sid in d2_ids:
    s = read_series(sid, d2_dir)
    # If you use stem_mode=delta, apply diff like the builder:
    s = s.diff()
    c = coverage(s)
    print(f"  id={sid}: coverage={c:.3f}")
