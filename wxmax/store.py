"""Tiny Parquet persistence for the obs / forecast / feature tables.

Uses DuckDB (already a core dep) for Parquet IO so we don't drag in pyarrow.
Tables are plain files under the config data dirs; the feature store is just a
directory of Parquet partitions read back with DuckDB when modeling.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("_df", df)
        con.execute(f"COPY _df TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        ).df()
    finally:
        con.close()
