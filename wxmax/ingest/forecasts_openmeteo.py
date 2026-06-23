"""Forecast ingestion from Open-Meteo (JSON, no GRIB).

Open-Meteo gives `temperature_2m_max` per model directly, plus an ensemble API
(member-resolved) and — crucially for the back-test — a Historical-Forecast API
that archives past forecast runs. That archive is the forecast-vs-actual fuel.

When `models=` is supplied, daily keys are SUFFIXED with the model id, e.g.
`temperature_2m_max_gfs_seamless`. The ensemble API suffixes with the member,
e.g. `temperature_2m_max_member01`. `_parse_daily` handles both (and the
no-suffix single-series case) uniformly.

Hosts: the free tier (api.open-meteo.com / *-api.open-meteo.com) is
NON-COMMERCIAL. For the commercial fund product, pass an api_key and we route to
customer-api.open-meteo.com.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ._http import get_json

FREE_HOSTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "ensemble": "https://ensemble-api.open-meteo.com/v1/ensemble",
}
COMMERCIAL_HOSTS = {
    "forecast": "https://customer-api.open-meteo.com/v1/forecast",
    "historical_forecast": "https://customer-historical-forecast-api.open-meteo.com/v1/forecast",
    "previous_runs": "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
    "archive": "https://customer-archive-api.open-meteo.com/v1/archive",
    "ensemble": "https://customer-ensemble-api.open-meteo.com/v1/ensemble",
}

BASE_VAR = "temperature_2m_max"


def _hosts(api_key: str | None) -> dict:
    return COMMERCIAL_HOSTS if api_key else FREE_HOSTS


def _parse_daily(payload: dict, base: str = BASE_VAR) -> pd.DataFrame:
    """Long DataFrame [date, series, value] from an Open-Meteo `daily` block.

    `series` is the key suffix after `base` (model id or member, "" if none).
    """
    daily = payload.get("daily", {})
    if "time" not in daily:
        return pd.DataFrame(columns=["date", "series", "value"])
    dates = pd.to_datetime(daily["time"]).date
    rows = []
    for key, vals in daily.items():
        if key == "time" or not key.startswith(base):
            continue
        series = key[len(base):].lstrip("_")  # "" | "gfs_seamless" | "member01"
        for d, v in zip(dates, vals):
            rows.append((d, series, v))
    out = pd.DataFrame(rows, columns=["date", "series", "value"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def fetch_model_tmax_history(
    lat: float, lon: float, tz: str, start: date, end: date,
    models: tuple[str, ...], unit: str = "fahrenheit", api_key: str | None = None,
) -> pd.DataFrame:
    """Per-model forecast daily-max over [start, end] from the archive.

    Returns long: [date, model, tmax]. These are the multi-model predictor
    columns for the feature store.
    """
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": BASE_VAR, "models": ",".join(models),
        "temperature_unit": unit,
    }
    if api_key:
        params["apikey"] = api_key
    payload = get_json(_hosts(api_key)["historical_forecast"], params=params)
    df = _parse_daily(payload)
    # Single-model responses may omit the suffix; backfill the model name.
    if len(models) == 1:
        df.loc[df["series"] == "", "series"] = models[0]
    return df.rename(columns={"series": "model", "value": "tmax"})


def fetch_era5_tmax(
    lat: float, lon: float, tz: str, start: date, end: date,
    unit: str = "fahrenheit", api_key: str | None = None,
) -> pd.DataFrame:
    """ERA5 reanalysis daily-max — a convenient gridded truth/feature.

    NOTE: reanalysis, not station obs. Use obs_asos/obs_ghcn for verification
    truth; ERA5 is a fallback feature and a sanity reference. Returns [date, tmax].
    """
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": BASE_VAR, "temperature_unit": unit, "models": "era5",
    }
    if api_key:
        params["apikey"] = api_key
    payload = get_json(_hosts(api_key)["archive"], params=params)
    df = _parse_daily(payload)
    return df[["date", "value"]].rename(columns={"value": "tmax"})


def fetch_ensemble_tmax(
    lat: float, lon: float, tz: str, start: date, end: date,
    model: str, unit: str = "fahrenheit", api_key: str | None = None,
) -> pd.DataFrame:
    """Ensemble member daily-max for one ensemble system.

    Returns long [date, member, tmax]; members feed ensemble spread features
    and CRPS scoring. `past_days`-style ranges are expressed via start/end.
    """
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": BASE_VAR, "models": model, "temperature_unit": unit,
    }
    if api_key:
        params["apikey"] = api_key
    payload = get_json(_hosts(api_key)["ensemble"], params=params)
    df = _parse_daily(payload)
    return df.rename(columns={"series": "member", "value": "tmax"})
