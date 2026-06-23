"""Time-zone-aware daily-max bucketing.

The target is the daily maximum temperature over the station's LOCAL calendar
day. Models/obs arrive in UTC (or mixed); bucketing in UTC would smear the peak
across two local days. Everything funnels through `daily_max`, which converts to
the station tz before grouping.
"""
from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def to_local(times: pd.Series | pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Coerce a timestamp series to a tz-aware DatetimeIndex in `tz`.

    Naive timestamps are assumed UTC (Open-Meteo returns UTC when we request
    `timezone=GMT`). Aware timestamps are converted.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=False))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(ZoneInfo(tz))


def daily_max(
    times,
    values,
    tz: str,
    min_obs_per_day: int = 1,
) -> pd.DataFrame:
    """Reduce a sub-daily temperature series to per-local-day maxima.

    Returns a DataFrame indexed by local `date` with columns:
      tmax      — maximum value over the local day
      n_obs     — number of samples contributing to that day

    Days with fewer than `min_obs_per_day` samples are dropped (guards against
    a "daily max" computed from one stray hour — important when deriving max
    from coarse 6-hourly model output that may clip the true peak).
    """
    local = to_local(times, tz)
    vals = np.asarray(values, dtype="float64")
    df = pd.DataFrame({"value": vals}, index=local)
    df = df.dropna()
    df["day"] = df.index.date
    grp = df.groupby("day")["value"]
    out = pd.DataFrame({"tmax": grp.max(), "n_obs": grp.size()})
    out = out[out["n_obs"] >= min_obs_per_day]
    out.index = pd.Index([d for d in out.index], name="date")
    return out


def observed_max_for_day(times, values, tz: str, target_day: date) -> float | None:
    """Convenience: the observed daily max for one specific local date.

    Returns None if no samples fall on that local day.
    """
    dm = daily_max(times, values, tz)
    if target_day in dm.index:
        return float(dm.loc[target_day, "tmax"])
    return None
