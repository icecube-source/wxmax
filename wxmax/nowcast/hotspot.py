"""Detect when the hottest part of the day has passed.

Three independent signals; we flip to HIGH conviction when >=2 fire:
  1. past_peak    -- local clock time is past the climatological peak-hour window
  2. trend_flat   -- the last few observations are flat/declining (slope <= 0.5 F/hr)
  3. model_done   -- the model's hourly shape shows < 0.2 F of rise remaining

The peak-hour window is derived from real geometry: local solar noon (from the
station longitude + the date's UTC offset, so DST is handled) plus a seasonal
lag (~2.5 h summer, ~2 h winter). This generalizes the Midway example (plateaued
72 F, past the ~2 pm peak, model showed ~0 further rise -> HIGH conviction).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from ..stations import Station


def utc_offset_hours(tz: str, d: date) -> float:
    off = datetime(d.year, d.month, d.day, 12, tzinfo=ZoneInfo(tz)).utcoffset()
    return off.total_seconds() / 3600.0 if off else 0.0


def climatological_peak_hour(station: Station, d: date) -> tuple[float, float]:
    """(start, end) local decimal-hour window of the climatological temperature peak."""
    off = utc_offset_hours(station.tz, d)
    solar_noon = 12.0 - station.lon / 15.0 + off          # local clock hour of solar noon
    lag = 2.5 if 4 <= d.month <= 9 else 2.0                # season-dependent thermal lag
    peak = solar_noon + lag
    return (peak - 1.0, peak + 1.0)


def observed_trend_declining(hours, temps, window: int = 3) -> tuple[bool, float]:
    """Is the recent obs trend flat/declining? Returns (flat_or_declining, slope F/hr)."""
    h = np.asarray(hours, dtype=float)[-window:]
    t = np.asarray(temps, dtype=float)[-window:]
    if len(h) < 2:
        return (False, float("nan"))
    slope = float(np.polyfit(h, t, 1)[0])
    return (slope <= 0.5, slope)


def model_remaining_rise(hours, temps, now_hour: float) -> float | None:
    """From a model hourly series, max(future temps) - temp at/just-before now (>=0)."""
    h = np.asarray(hours, dtype=float)
    t = np.asarray(temps, dtype=float)
    future = h >= now_hour
    if not future.any():
        return None
    at_now = t[h <= now_hour]
    base = at_now[-1] if len(at_now) else t[future][0]
    return max(0.0, float(t[future].max() - base))


@dataclass(frozen=True)
class Conviction:
    high_conviction: bool
    n_signals: int
    past_peak: bool
    trend_flat: bool
    model_done: bool
    slope: float
    peak_window: tuple[float, float]


def detect_high_conviction(
    station: Station, d: date, now_hour: float,
    obs_hours, obs_temps, model_rise: float | None,
) -> Conviction:
    window = climatological_peak_hour(station, d)
    past_peak = now_hour > window[1]
    trend_flat, slope = observed_trend_declining(obs_hours, obs_temps)
    model_done = (model_rise is not None) and (model_rise < 0.2)
    n = int(past_peak) + int(trend_flat) + int(model_done)
    return Conviction(n >= 2, n, past_peak, trend_flat, model_done, slope, window)
