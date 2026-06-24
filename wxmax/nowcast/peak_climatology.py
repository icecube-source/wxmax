"""Per-station empirical climatology of the hour-of-daily-maximum.

The crude solar-noon+fixed-lag "peak window" mislocked Miami at 1:30 PM. Instead
we learn, from real hourly observations, *when each city actually peaks* and turn
that into P_clim(now) = the empirical CDF of the peak hour evaluated at the
current local time = "probability the daily max has already occurred, by
climatology alone."

Two pooled windows (free IEM ASOS hourly -> true hourly, no time-of-observation
bias):
  Recent (R):   last `recent_days` (default 10) -> the current airmass/regime.
  Baseline (C): same calendar window +/- `pm_days` from the prior `prior_years`.

p_passed blends the two ECDFs, weighting Recent by w_r (shrunk toward Baseline
when Recent is small). Cached per station per day under panel_data/climatology/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingest.obs_asos import fetch_asos_hourly
from ..stations import Station
from ..timeutil import to_local


def peak_hour_per_day(hourly: pd.DataFrame, tz: str, min_obs_per_day: int = 12) -> pd.DataFrame:
    """Local decimal hour of the daily max for each day with enough obs.

    Returns columns: date, peak_hour, tmax, n_obs.
    """
    if hourly.empty:
        return pd.DataFrame(columns=["date", "peak_hour", "tmax", "n_obs"])
    loc = to_local(hourly["valid"], tz)
    df = pd.DataFrame({"tmpf": np.asarray(hourly["tmpf"], dtype=float)}, index=loc).dropna()
    df["day"] = df.index.date
    df["hr"] = df.index.hour + df.index.minute / 60.0
    rows = []
    for day, g in df.groupby("day"):
        if len(g) < min_obs_per_day:
            continue
        pos = int(np.argmax(g["tmpf"].values))  # first occurrence of the max
        rows.append((day, float(g["hr"].iloc[pos]), float(g["tmpf"].max()), len(g)))
    return pd.DataFrame(rows, columns=["date", "peak_hour", "tmax", "n_obs"])


@dataclass
class PeakClimatology:
    station: str
    built_date: str
    recent: list           # peak hours, recent window (current regime)
    baseline: list         # peak hours, prior-year same-window
    coastal: bool = False

    @property
    def _all(self) -> np.ndarray:
        return np.array(list(self.recent) + list(self.baseline), dtype=float)

    @property
    def n(self) -> int:
        return len(self.recent) + len(self.baseline)

    def _w_recent(self, base: float = 0.5, target: int = 8) -> float:
        if not self.recent:
            return 0.0
        if not self.baseline:
            return 1.0
        return base * min(1.0, len(self.recent) / target)  # shrink R toward C when sparse

    def p_passed(self, now_hour: float) -> float | None:
        """Climatological P(daily max already occurred by `now_hour`)."""
        if self.n == 0:
            return None

        def ecdf(arr) -> float:
            a = np.asarray(arr, dtype=float)
            return float((a <= now_hour).mean()) if len(a) else 0.0

        if self.recent and self.baseline:
            w = self._w_recent()
            return w * ecdf(self.recent) + (1.0 - w) * ecdf(self.baseline)
        return ecdf(self._all)

    def pct(self, q: float) -> float:
        a = self._all
        return float(np.percentile(a, q)) if len(a) else float("nan")

    def stats(self) -> dict:
        return {"n": self.n, "median": self.pct(50), "p90": self.pct(90),
                "p95": self.pct(95), "p99": self.pct(99),
                "iqr": self.pct(75) - self.pct(25)}


def build_climatology(station: Station, today: date, recent_days: int = 10,
                      prior_years=(1, 2), pm_days: int = 10) -> PeakClimatology:
    def hours_for(start: date, end: date) -> list:
        try:
            h = fetch_asos_hourly(station.id, start, end)
        except Exception:
            return []
        return peak_hour_per_day(h, station.tz)["peak_hour"].tolist()

    recent = hours_for(today - timedelta(days=recent_days), today - timedelta(days=1))
    baseline: list = []
    for py in prior_years:
        try:
            center = today.replace(year=today.year - py)
        except ValueError:                      # Feb 29 in a non-leap prior year
            center = today.replace(year=today.year - py, day=28)
        baseline += hours_for(center - timedelta(days=pm_days), center + timedelta(days=pm_days))
    return PeakClimatology(station.id, today.isoformat(), recent, baseline, station.coastal)


def _cache_path(cfg: Config, station_id: str) -> Path:
    return cfg.panel_dir / "climatology" / f"{station_id}.json"


def load_or_build(station: Station, today: date, cfg: Config) -> PeakClimatology:
    """Return today's cached climatology for a station, building+caching if stale."""
    p = _cache_path(cfg, station.id)
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("built_date") == today.isoformat():
            return PeakClimatology(**d)
    clim = build_climatology(station, today)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clim.__dict__))
    return clim
