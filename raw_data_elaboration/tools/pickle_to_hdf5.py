
#!/usr/bin/env python3
"""
Convert ONE pickle file (containing a single large dict of time series) into HDF5,
storing each series as its own datasets so you can read one at a time.

USAGE:
    python pickle_bigdict_to_hdf5.py INPUT.pkl OUTPUT.h5
Options:
    --group /series                    HDF5 group where series are stored
    --compression gzip                 gzip | lzf | none
    --compression-level 4              gzip level 0-9
    --no-shuffle                       disable shuffle filter
    --no-fletcher32                    disable fletcher32 checksum
    --chunk-len 262144                 chunk length for 1D datasets
    --cast-float32                     cast float64 to float32 to save space
"""

import argparse
import gc
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import h5py

# Optional pandas support (for DatetimeIndex or ISO time parsing)
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# ---------- Helpers ----------

def sanitize_key(key: str) -> str:
    """Make an HDF5-safe path component from an arbitrary identifier."""
    safe = str(key).strip().replace("/", "∕")
    if safe == "":
        safe = "_"
    return safe

def to_epoch_ns(dt64: np.ndarray) -> np.ndarray:
    """Convert numpy datetime64[*] array to int64 epoch nanoseconds."""
    if not np.issubdtype(dt64.dtype, np.datetime64):
        raise TypeError("Index is not datetime64; cannot convert to epoch ns.")
    return dt64.astype("datetime64[ns]").view("int64")

def normalize_series(value: Any, cast_float32: bool = False) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Normalize various time series representations into:
      (time_ns: Optional[int64 array], values: np.ndarray, attrs: dict)

    Supported:
      - numpy.ndarray (values only)
      - pandas.Series (DatetimeIndex) or DataFrame
      - tuple/list: (time_like, values_like)
      - list of (time, value) pairs
    """
    attrs: Dict[str, Any] = {}
    time_ns: Optional[np.ndarray] = None

    # Pandas objects
    if HAS_PANDAS:
        if isinstance(value, pd.Series):
            vals = value.to_numpy()
            idx = value.index
            if np.issubdtype(idx.dtype, np.datetime64):
                time_ns = to_epoch_ns(idx.values)
                attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
            else:
                attrs["index_dtype"] = str(idx.dtype)
                attrs["index_note"] = "Non-datetime index not stored."
        elif isinstance(value, pd.DataFrame):
            if value.shape[1] == 1:
                vals = value.iloc[:, 0].to_numpy()
                idx = value.index
                if np.issubdtype(idx.dtype, np.datetime64):
                    time_ns = to_epoch_ns(idx.values)
                    attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
                else:
                    attrs["index_dtype"] = str(idx.dtype)
                    attrs["index_note"] = "Non-datetime index not stored."
            elif {"time", "value"}.issubset(set(value.columns)):
                vals = value["value"].to_numpy()
                tcol = value["time"].to_numpy()
                if np.issubdtype(tcol.dtype, np.datetime64):
                    time_ns = to_epoch_ns(tcol)
                    attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
                else:
                    # Try parsing strings if necessary
                    if tcol.dtype.kind in ("U", "S", "O"):
                        time_ns = pd.to_datetime(tcol).values.astype("datetime64[ns]").view("int64")
                        attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
                    else:
                        raise TypeError("Unsupported 'time' dtype; convert to datetime64[ns] first.")
            else:
                raise TypeError("Unsupported DataFrame. Use single column with DatetimeIndex or columns {'time','value'}.")
        else:
            vals = None  # fall through

    if 'vals' not in locals() or vals is None:
        # Tuple/list (time, values)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            t, v = value
            vals = np.asarray(v)
            t_arr = np.asarray(t)
            if np.issubdtype(t_arr.dtype, np.datetime64):
                time_ns = to_epoch_ns(t_arr)
                attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
            elif np.issubdtype(t_arr.dtype, np.integer):
                time_ns = t_arr.astype(np.int64)
                attrs["time_unit"] = "user-supplied integer units"
            elif t_arr.dtype.kind in ("U", "S", "O"):
                if HAS_PANDAS:
                    time_ns = pd.to_datetime(t_arr).values.astype("datetime64[ns]").view("int64")
                    attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
                else:
                    raise TypeError("String timestamps require pandas; install pandas or pre-convert.")
            else:
                raise TypeError(f"Unsupported time dtype: {t_arr.dtype}")
        # List of (time, value) pairs
        elif isinstance(value, list) and value and isinstance(value[0], (tuple, list)) and len(value[0]) == 2:
            t_list, v_list = zip(*value)
            vals = np.asarray(v_list)
            t_arr = np.asarray(t_list)
            if np.issubdtype(t_arr.dtype, np.datetime64):
                time_ns = to_epoch_ns(t_arr)
                attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
            else:
                if HAS_PANDAS:
                    time_ns = pd.to_datetime(t_arr).values.astype("datetime64[ns]").view("int64")
                    attrs["time_unit"] = "ns since 1970-01-01 00:00:00"
                else:
                    raise TypeError("String/other timestamps require pandas.")
        else:
            # Numpy array or numeric-like: values only
            vals = np.asarray(value)

    if cast_float32 and np.issubdtype(vals.dtype, np.floating) and vals.dtype == np.float64:
        vals = vals.astype(np.float32, copy=False)
        attrs["cast_float32"] = True

    attrs["value_dtype"] = str(vals.dtype)
    return time_ns, vals, attrs


def write_one_series(h5f: h5py.File, group_path: str, key: str, value: Any,
                     compression: Optional[str], compression_opts: Optional[int],
                     shuffle: bool, fletcher32: bool, chunk_len: int, cast_float32: bool) -> None:
    """Write a single series into h5 under /series/<sanitized_key>/{values, time}."""
    g_series = h5f.require_group(group_path)
    safe = sanitize_key(key)
    if safe in g_series:
        del g_series[safe]
    g = g_series.create_group(safe)
    g.attrs["original_id"] = str(key)

    time_ns, vals, attrs = normalize_series(value, cast_float32=cast_float32)
    n = int(vals.shape[0])
    chunk = (min(chunk_len, n),) if n > 0 else None

    d_values = g.create_dataset(
        "values",
        data=vals,
        chunks=chunk,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=shuffle,
        fletcher32=fletcher32,
    )
    for ak, av in attrs.items():
        d_values.attrs[ak] = av

    if time_ns is not None:
        d_time = g.create_dataset(
            "time",
            data=time_ns.astype(np.int64, copy=False),
            chunks=chunk,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            fletcher32=fletcher32,
        )
        d_time.attrs["time_unit"] = attrs.get("time_unit", "ns since 1970-01-01 00:00:00")

    # Make sure big arrays can be GC’d quickly
    del vals
    if time_ns is not None:
        del time_ns
    gc.collect()


def convert_pickle_bigdict_to_hdf5(pickle_path: str, h5_path: str, group: str = "/series",
                                   compression: Optional[str] = "gzip", compression_level: int = 4,
                                   shuffle: bool = True, fletcher32: bool = True,
                                   chunk_len: int = 262144, cast_float32: bool = False) -> int:
    """Load ONE big dict from pickle, write to HDF5, freeing entries as we go. Returns count of series."""
    comp_opts = compression_level if compression == "gzip" else None

    with open(pickle_path, "rb") as pf:
        big = pickle.load(pf)  # NOTE: requires enough RAM once, as you confirmed.

    if not isinstance(big, dict):
        raise TypeError(f"Expected a dict in {pickle_path}, got {type(big)}")

    count = 0
    with h5py.File(h5_path, "w") as h5f:
        h5f.attrs["creator"] = "pickle_bigdict_to_hdf5.py"
        h5f.attrs["source_pickle"] = pickle_path
        h5f.attrs["compression"] = str(compression)
        if comp_opts is not None:
            h5f.attrs["compression_opts"] = comp_opts

        # Pop items to release memory progressively
        while big:
            k, v = big.popitem()
            write_one_series(h5f, group, k, v, compression, comp_opts, shuffle, fletcher32, chunk_len, cast_float32)
            count += 1
            # Optional: add a small GC every N items if series are large
            if (count % 50) == 0:
                gc.collect()

    # After loop, 'big' is empty and memory footprint is minimal again
    gc.collect()
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a big-dict pickle of time series into HDF5.")
    p.add_argument("pickle_path", help="Input pickle file (single big dict)")
    p.add_argument("h5_path", help="Output HDF5 file")
    p.add_argument("--group", default="/series", help="HDF5 group path (default: /series)")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"],
                   help="Compression algorithm (default: gzip)")
    p.add_argument("--compression-level", type=int, default=4, help="gzip level 0-9 (default: 4)")
    p.add_argument("--no-shuffle", action="store_true", help="Disable shuffle filter")
    p.add_argument("--no-fletcher32", action="store_true", help="Disable fletcher32 checksum")
    p.add_argument("--chunk-len", type=int, default=262144, help="Chunk length for 1D datasets")
    p.add_argument("--cast-float32", action="store_true", help="Cast float64 values to float32 to save space")
    return p.parse_args()


def main():
    args = parse_args()
    compression = None if args.compression.lower() == "none" else args.compression
    n = convert_pickle_bigdict_to_hdf5(
        pickle_path=args.pickle_path,
        h5_path=args.h5_path,
        group=args.group,
        compression=compression,
        compression_level=args.compression_level,
        shuffle=(not args.no_shuffle),
        fletcher32=(not args.no_fletcher32),
        chunk_len=args.chunk_len,
        cast_float32=args.cast_float32,
    )
    print(f"Converted {n} series to {args.h5_path} under group '{args.group}'.")


if __name__ == "__main__":
    main()
