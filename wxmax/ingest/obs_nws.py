"""Free, commercial-clean forecasts + observations from the NWS API.

- `fetch_nbm_daily_max`: the gridpoint daytime `maxTemperature` (NBM-based) for a
  local day -> one forecast EXPERT. Public domain, no key, JSON.
- `fetch_latest_obs` / `fetch_obs_series`: live station temperature for the
  intraday nowcast (obs_max_so_far, trend, current temp).

All temperatures are returned in degrees F. The `/points` -> grid mapping is
stable per station, so it's cached.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from ..stations import Station
from ..timeutil import to_local
from ._http import get_json

NWS = "https://api.weather.gov"
_grid_cache: dict[str, tuple[str, int, int]] = {}


def _c_to_f(c: float | None) -> float | None:
    return None if c is None else c * 9.0 / 5.0 + 32.0


def gridpoint(st: Station) -> tuple[str, int, int]:
    """(wfo, gridX, gridY) for a station; cached (the mapping is stable)."""
    if st.id not in _grid_cache:
        pr = get_json(f"{NWS}/points/{st.lat},{st.lon}")["properties"]
        _grid_cache[st.id] = (pr["gridId"], int(pr["gridX"]), int(pr["gridY"]))
    return _grid_cache[st.id]


def fetch_nbm_daily_max(st: Station, target_day: date) -> float | None:
    """NWS gridpoint daytime max (°F) whose period falls on `target_day` local."""
    wfo, x, y = gridpoint(st)
    props = get_json(f"{NWS}/gridpoints/{wfo}/{x},{y}")["properties"]
    for v in props.get("maxTemperature", {}).get("values", []):
        start = v["validTime"].split("/")[0]
        if to_local([start], st.tz)[0].date() == target_day:
            return _c_to_f(v.get("value"))
    return None


def fetch_latest_obs(st: Station) -> tuple[datetime, float] | None:
    """(timestamp, temp °F) of the station's latest observation, or None."""
    pr = get_json(f"{NWS}/stations/{st.id}/observations/latest")["properties"]
    t = pr.get("temperature", {}).get("value")
    ts = pr.get("timestamp")
    if t is None or ts is None:
        return None
    return (pd.to_datetime(ts), _c_to_f(t))


def fetch_hourly_forecast(st: Station) -> pd.DataFrame:
    """NWS hourly forecast temperatures (°F) -> columns valid(UTC), tmpf.

    Used to estimate the model's expected *remaining* rise for the nowcast.
    """
    wfo, x, y = gridpoint(st)
    periods = get_json(f"{NWS}/gridpoints/{wfo}/{x},{y}/forecast/hourly")["properties"]["periods"]
    rows = []
    for p in periods:
        if p.get("temperature") is not None and p.get("startTime"):
            t = float(p["temperature"])
            if p.get("temperatureUnit") == "C":
                t = t * 9.0 / 5.0 + 32.0
            rows.append((pd.to_datetime(p["startTime"], utc=True), t))
    df = pd.DataFrame(rows, columns=["valid", "tmpf"])
    return df.sort_values("valid").reset_index(drop=True) if len(df) else df


def fetch_obs_series(st: Station, start: date, end: date) -> pd.DataFrame:
    """Observation time series (UTC) over [start, end] -> columns valid, tmpf (°F)."""
    params = {
        "start": f"{start.isoformat()}T00:00:00Z",
        "end": f"{end.isoformat()}T23:59:59Z",
    }
    feats = get_json(f"{NWS}/stations/{st.id}/observations", params=params).get("features", [])
    rows = []
    for f in feats:
        p = f.get("properties", {})
        t = p.get("temperature", {}).get("value")
        ts = p.get("timestamp")
        if t is not None and ts is not None:
            rows.append((pd.to_datetime(ts, utc=True), _c_to_f(t)))
    df = pd.DataFrame(rows, columns=["valid", "tmpf"])
    return df.sort_values("valid").reset_index(drop=True) if len(df) else df
