"""Ground truth: observed daily-max temperature from ASOS via IEM.

We pull hourly METAR air temperature (tmpf) from the Iowa Environmental Mesonet
(IEM) — public domain — and reduce it to the station's LOCAL-day maximum using
`timeutil.daily_max`. This is the canonical near-real-time truth source; GHCN
(see obs_ghcn.py) is the QC'd historical authority but lags ~45-60 days.

Why derive max from hourly rather than trust a daily field: the NWS
`maxTemperatureLast24Hours` style fields are frequently null, and IEM's hourly
series is dense and reliable.
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from ..stations import Station
from ..timeutil import daily_max
from ._http import get_text

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def iem_id(icao: str) -> str:
    """IEM uses the 3-letter id (no leading K) for US ASOS sites."""
    return icao[1:] if len(icao) == 4 and icao.upper().startswith("K") else icao


def fetch_asos_hourly(station_icao: str, start: date, end: date) -> pd.DataFrame:
    """Hourly air temperature (°F) in UTC for [start, end] inclusive.

    Returns columns: valid (UTC datetime), tmpf (float, NaN for missing).
    """
    params = {
        "station": iem_id(station_icao),
        "data": "tmpf",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "report_type": "3",
    }
    text = get_text(IEM_ASOS_URL, params=params)
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return pd.DataFrame(columns=["valid", "tmpf"])
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")  # "M"/"T" -> NaN
    return df[["valid", "tmpf"]].dropna(subset=["valid"])


def observed_daily_max(
    station: Station, start: date, end: date, min_obs_per_day: int = 12
) -> pd.DataFrame:
    """Observed local-day max temperature (°F) for a station.

    Columns: station, date, tmax_f, n_obs, source. `min_obs_per_day` guards
    against partial days producing a spuriously low "max" (ASOS reports ~hourly,
    so a full day is ~24 obs; 12 is a conservative floor).
    """
    hourly = fetch_asos_hourly(station.id, start, end)
    if hourly.empty:
        return pd.DataFrame(columns=["station", "date", "tmax_f", "n_obs", "source"])
    dm = daily_max(hourly["valid"], hourly["tmpf"], tz=station.tz,
                   min_obs_per_day=min_obs_per_day)
    dm = dm.reset_index().rename(columns={"tmax": "tmax_f"})
    dm.insert(0, "station", station.id)
    dm["source"] = "asos_iem"
    return dm[["station", "date", "tmax_f", "n_obs", "source"]]
