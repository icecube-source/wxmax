"""Assemble the per-(station, date) design matrix for modeling.

Columns: one per model's forecast daily-max, plus the cross-model ensemble
mean/spread (a cheap uncertainty proxy — the models disagree more when the
atmosphere is less predictable), plus the observed truth. This is the table the
MOS blend (P4) and EMOS head (P5) train on.

Reuses the ingest layer (Open-Meteo forecasts + ASOS truth).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import DEFAULT_MODELS, Config, load_config
from ..ingest.forecasts_openmeteo import fetch_model_tmax_history
from ..ingest.obs_asos import observed_daily_max
from ..stations import load_stations


def assemble_training_frame(
    station_ids: list[str], start: date, end: date,
    cfg: Config | None = None, models: tuple[str, ...] = DEFAULT_MODELS,
) -> pd.DataFrame:
    """Build [station, date, <model cols>, ens_mean, ens_spread, truth].

    Rows missing any model forecast or the truth are dropped so the design
    matrix is dense (models with sparse history, e.g. AIFS pre-2025, are simply
    excluded by `present_models` — see the returned attrs)."""
    cfg = cfg or load_config()
    registry = load_stations(cfg.stations_path)
    frames = []
    for sid in station_ids:
        st = registry[sid]
        fc = fetch_model_tmax_history(st.lat, st.lon, st.tz, start, end, models,
                                      api_key=cfg.openmeteo_api_key)
        wide = fc.pivot(index="date", columns="model", values="tmax")
        truth = observed_daily_max(st, start, end).set_index("date")["tmax_f"]
        wide["truth"] = truth
        wide.insert(0, "station", sid)
        frames.append(wide.reset_index())
    df = pd.concat(frames, ignore_index=True)

    # Keep only models with full coverage across the assembled rows.
    present = [m for m in models if m in df.columns and df[m].notna().mean() > 0.5]
    df = df.dropna(subset=present + ["truth"]).reset_index(drop=True)
    df["ens_mean"] = df[present].mean(axis=1)
    df["ens_spread"] = df[present].std(axis=1, ddof=0)
    df.attrs["model_cols"] = present
    return df


def design_matrix(df: pd.DataFrame, model_cols: list[str] | None = None):
    """Return (F, y, spread) numpy arrays from an assembled frame."""
    cols = model_cols or df.attrs.get("model_cols")
    F = df[cols].to_numpy(dtype=float)
    y = df["truth"].to_numpy(dtype=float)
    spread = df["ens_spread"].to_numpy(dtype=float)
    return F, y, spread
