"""Baseline back-test: score each single model against station truth.

Establishes the bar the calibrated ensemble (P4-P6) must beat. Scoring is split
from fetching so it is unit-testable offline: `score_by_model` takes an already
joined long frame; `run_baseline_backtest` is the live convenience wrapper.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..config import DEFAULT_MODELS, Config, load_config
from ..ingest.forecasts_openmeteo import fetch_model_tmax_history
from ..ingest.obs_asos import observed_daily_max
from ..stations import load_stations
from .metrics import bias, mae, rmse


def join_forecast_truth(model_long: pd.DataFrame, truth_long: pd.DataFrame) -> pd.DataFrame:
    """Inner-join per-model forecasts to observed truth on [station, date].

    model_long: [station, date, model, tmax]; truth_long: [station, date, tmax_f].
    """
    truth = truth_long.rename(columns={"tmax_f": "truth"})[["station", "date", "truth"]]
    return model_long.merge(truth, on=["station", "date"], how="inner")


def score_by_model(joined: pd.DataFrame, group_extra: list[str] | None = None) -> pd.DataFrame:
    """Per-model MAE/RMSE/bias over rows with both forecast and truth present.

    `group_extra` adds grouping columns (e.g. ["station"], ["region", "season"]).
    """
    keys = (group_extra or []) + ["model"]
    valid = joined.dropna(subset=["tmax", "truth"])
    rows = []
    for key_vals, d in valid.groupby(keys):
        rec = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        rec.update(n=len(d), mae=mae(d["truth"], d["tmax"]),
                   rmse=rmse(d["truth"], d["tmax"]), bias=bias(d["truth"], d["tmax"]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values((group_extra or []) + ["mae"]).reset_index(drop=True)


def run_baseline_backtest(
    station_ids: list[str], start: date, end: date,
    cfg: Config | None = None, models: tuple[str, ...] = DEFAULT_MODELS,
) -> dict[str, pd.DataFrame]:
    """Fetch truth + per-model forecasts for stations and score the baselines.

    Returns {"joined", "overall", "by_station"}.
    """
    cfg = cfg or load_config()
    registry = load_stations(cfg.stations_path)
    model_parts, truth_parts = [], []
    for sid in station_ids:
        st = registry[sid]
        truth_parts.append(observed_daily_max(st, start, end))
        fc = fetch_model_tmax_history(st.lat, st.lon, st.tz, start, end, models,
                                      api_key=cfg.openmeteo_api_key)
        fc.insert(0, "station", sid)
        model_parts.append(fc)
    model_long = pd.concat(model_parts, ignore_index=True)
    truth_long = pd.concat(truth_parts, ignore_index=True)
    joined = join_forecast_truth(model_long, truth_long)
    return {
        "joined": joined,
        "overall": score_by_model(joined),
        "by_station": score_by_model(joined, group_extra=["station"]),
    }
