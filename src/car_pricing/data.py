"""Data loading, cleaning, and multi-format I/O (CSV / Parquet / Feather).

The format helpers back both the format-benchmark notebook (03) and the ability
to cache a cleaned, columnar copy of the data for fast reloads during modelling.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from . import config


# --- Format-agnostic read / write ------------------------------------------
def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path`, choosing the format from the file extension."""
    path = Path(path)
    suffixes = path.suffixes
    if path.suffix == ".csv" or suffixes[-2:] == [".csv", ".gz"]:
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix in (".feather", ".arrow"):
        df.to_feather(path)
    else:
        raise ValueError(f"Unsupported output format: {path.name}")


def read_dataframe(path: Path) -> pd.DataFrame:
    """Read a DataFrame from `path`, choosing the reader from the extension.

    pandas auto-decompresses .gz/.bz2/.xz CSVs, so no manual unzip is needed.
    """
    path = Path(path)
    if path.suffix in (".parquet",):
        return pd.read_parquet(path)
    if path.suffix in (".feather", ".arrow"):
        return pd.read_feather(path)
    return pd.read_csv(path)   # handles .csv and compressed .csv.*


def benchmark_formats(df: pd.DataFrame, out_dir: Path,
                      reads: int = 5) -> Dict[str, Dict[str, float]]:
    """Write `df` in several formats and time size + read for each.

    Returns {label: {"bytes", "write_s", "read_s"}}. Used by notebook 03 and by
    docs/FORMAT_BENCHMARKS.md — measured, not estimated.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _time_read(reader: Callable, path: Path) -> float:
        """Return the fastest of `reads` timed calls to `reader(path)`, in seconds."""
        best = float("inf")
        for _ in range(reads):
            t = time.perf_counter()
            reader(path)
            best = min(best, time.perf_counter() - t)
        return best

    targets = [
        ("csv", "bench.csv", lambda p: df.to_csv(p, index=False), pd.read_csv),
        ("csv_gzip", "bench.csv.gz", lambda p: df.to_csv(p, index=False, compression="gzip"), pd.read_csv),
        ("csv_bz2", "bench.csv.bz2", lambda p: df.to_csv(p, index=False, compression="bz2"), pd.read_csv),
        ("csv_xz", "bench.csv.xz", lambda p: df.to_csv(p, index=False, compression="xz"), pd.read_csv),
        ("parquet_snappy", "bench.snappy.parquet", lambda p: df.to_parquet(p, compression="snappy"), pd.read_parquet),
        ("parquet_zstd", "bench.zstd.parquet", lambda p: df.to_parquet(p, compression="zstd"), pd.read_parquet),
        ("feather", "bench.feather", lambda p: df.to_feather(p), pd.read_feather),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for label, fname, writer, reader in targets:
        path = out_dir / fname
        t = time.perf_counter()
        writer(path)
        write_s = time.perf_counter() - t
        results[label] = {
            "bytes": float(path.stat().st_size),
            "write_s": write_s,
            "read_s": _time_read(reader, path),
        }
    return results


# --- Cleaning --------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Read the raw (gzip-compressed) dataset into a DataFrame."""
    return pd.read_csv(config.DATA_RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the awkward seat columns, drop invalid rows, normalise text."""
    df = df.rename(columns={"5": "Seats_5", ">5": "Seats_Above_5"})

    before = len(df)
    df = df.dropna(subset=[config.TARGET])
    df = df[df[config.TARGET] > 0]
    df = df.dropna()
    for col in config.TARGET_ENCODE:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df = df.reset_index(drop=True)
    df.attrs["dropped_rows"] = before - len(df)
    return df
