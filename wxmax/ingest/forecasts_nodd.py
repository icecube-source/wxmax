"""Forecast experts from NOAA NODD via Herbie (free, public domain): GFS (+HRRR).

Herbie byte-range-subsets the GRIB to just the 2 m temperature field, so each
step is a small download. We loop the forecast hours covering the next ~2 days,
sample every station, and reduce to the local-day max. GFS uses a regular 0..360
lat/lon grid (1-D coords -> fast `sel(method="nearest")`).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ..stations import Station
from ..timeutil import daily_max

DEFAULT_FXX = list(range(0, 49, 3))


def latest_cycle(lag_hours: int = 5, cycle_hours: int = 6) -> str:
    """Most recent model cycle (UTC) likely to be published, as 'YYYY-MM-DD HH:00'."""
    now = datetime.now(timezone.utc) - timedelta(hours=lag_hours)
    h = (now.hour // cycle_hours) * cycle_hours
    return now.replace(hour=h, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:00")


def fetch_gfs_daily_max(
    stations: list[Station], target_day: date,
    run: str | None = None, fxx_list: list[int] | None = None,
) -> dict[str, float | None]:
    """GFS local-day max (°F) per station for `target_day`."""
    from herbie import Herbie  # local import; only needed for GRIB experts

    run = run or latest_cycle()
    fxx_list = fxx_list or DEFAULT_FXX
    series: dict[str, list] = {st.id: [] for st in stations}
    for fx in fxx_list:
        try:
            ds = Herbie(run, model="gfs", fxx=fx).xarray("TMP:2 m above ground")
        except Exception:
            continue  # missing step -> skip
        da = ds["t2m"]
        vt = pd.Timestamp(da.valid_time.values)
        for st in stations:
            v = da.sel(latitude=st.lat, longitude=st.lon % 360, method="nearest").item()
            series[st.id].append((vt, (v - 273.15) * 9.0 / 5.0 + 32.0))
    res: dict[str, float | None] = {}
    for st in stations:
        rows = series[st.id]
        if not rows:
            res[st.id] = None
            continue
        df = pd.DataFrame(rows, columns=["valid", "tmpf"])
        dm = daily_max(df["valid"], df["tmpf"], tz=st.tz, min_obs_per_day=1)
        res[st.id] = float(dm.loc[target_day, "tmax"]) if target_day in dm.index else None
    return res
