"""Forecast experts from ECMWF Open Data (free, CC-BY): IFS and AIFS.

We pull 2 m temperature (`2t`) for steps covering the next ~2 days as a single
global GRIB per step-set, then sample every panel station from that one download
(efficient: 11 stations share the same fields). Daily max is the local-day max of
the sampled sub-daily series. The open IFS set has no `mx2t`, so we aggregate the
instantaneous `2t` series ourselves (3-hourly sampling can clip the true peak by a
degree or two — the online learner accounts for each expert's bias).

Attribution: "Generated using Copernicus / ECMWF open data (CC BY 4.0)."
"""
from __future__ import annotations

import os
import tempfile
from datetime import date

import pandas as pd
import xarray as xr

from ..stations import Station
from ..timeutil import daily_max

DEFAULT_STEPS = list(range(0, 49, 3))  # hours; covers today + tomorrow local


def _kelvin_to_f(k):
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def _lon_normalizer(grid_lon_max: float):
    """Match a station longitude to the grid's convention (0..360 vs -180..180)."""
    if grid_lon_max > 180.0:
        return lambda lon: lon % 360.0           # grid is 0..360
    return lambda lon: ((lon + 180.0) % 360.0) - 180.0  # grid is -180..180


def fetch_series(
    stations: list[Station], model: str = "ifs", steps: list[int] | None = None,
    source: str = "ecmwf",
) -> dict[str, pd.DataFrame]:
    """Sample 2m-temp (°F) at each station from one ECMWF download per step-set.

    `model`: "ifs" (deterministic / ENS control) or "aifs-single" (AI). Returns
    {station_id: DataFrame[valid(UTC), tmpf]}.
    """
    from ecmwf.opendata import Client  # local import; only needed for GRIB experts

    steps = steps or DEFAULT_STEPS
    client = Client(source=source, model=model)
    target = os.path.join(tempfile.gettempdir(), f"wxmax_ec_{model}.grib2")
    # The auto-"latest" run can be unresolvable or 404 (notably AIFS before its
    # post) -> walk back to recent prior runs until one resolves.
    last_exc: Exception | None = None
    for date_kw in ({}, {"date": -1}, {"date": -2}):
        try:
            client.retrieve(type="fc", param="2t", step=steps, target=target, **date_kw)
            last_exc = None
            break
        except Exception as e:  # noqa: PERF203
            last_exc = e
    if last_exc is not None:
        raise last_exc
    ds = xr.open_dataset(target, engine="cfgrib")
    da = ds["t2m"]
    valid = pd.to_datetime(da["valid_time"].values)
    norm_lon = _lon_normalizer(float(da.longitude.max()))
    out: dict[str, pd.DataFrame] = {}
    for st in stations:
        s = da.sel(latitude=st.lat, longitude=norm_lon(st.lon), method="nearest")
        out[st.id] = pd.DataFrame({"valid": valid, "tmpf": _kelvin_to_f(s.values)})
    return out


def fetch_daily_max(
    stations: list[Station], target_day: date, model: str = "ifs",
    steps: list[int] | None = None, source: str = "ecmwf",
) -> dict[str, float | None]:
    """Local-day max forecast (°F) per station for `target_day`."""
    series = fetch_series(stations, model=model, steps=steps, source=source)
    res: dict[str, float | None] = {}
    for st in stations:
        df = series[st.id]
        dm = daily_max(df["valid"], df["tmpf"], tz=st.tz, min_obs_per_day=1)
        res[st.id] = float(dm.loc[target_day, "tmax"]) if target_day in dm.index else None
    return res
