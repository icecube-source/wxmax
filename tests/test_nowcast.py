"""Tests for hotspot detection + obs-anchored nowcast."""
from datetime import date
from pathlib import Path

from wxmax.nowcast.hotspot import (
    climatological_peak_hour,
    detect_high_conviction,
    model_remaining_rise,
    observed_trend_declining,
)
from wxmax.nowcast.obs_nowcast import intraday_nowcast, obs_anchored_high
from wxmax.stations import load_stations

STATIONS = load_stations(Path(__file__).resolve().parent.parent / "config" / "stations.yaml")


def test_peak_window_summer_afternoon():
    # LAX, late June -> solar noon ~12.9 local + 2.5h lag -> peak ~15.4.
    start, end = climatological_peak_hour(STATIONS["KLAX"], date(2026, 6, 23))
    assert 14.0 < start < 15.0
    assert 16.0 < end < 17.0


def test_trend_flat_vs_rising():
    flat, slope = observed_trend_declining([13, 14, 15], [72, 72, 72])
    assert flat and abs(slope) < 1e-9
    rising, slope2 = observed_trend_declining([9, 10, 11], [66, 69, 72])
    assert not rising and slope2 > 1.0


def test_model_remaining_rise():
    hrs = list(range(10, 21))
    temps = [66, 69, 71, 72, 72, 71, 70, 69, 68, 67, 66]  # peak at 14:00
    assert model_remaining_rise(hrs, temps, now_hour=15) == 0.0
    assert model_remaining_rise(hrs, temps, now_hour=11) > 0.0


def test_midway_plateau_flips_high_conviction():
    # The worked example: 3 pm, plateaued at 72, model shows 0 remaining rise.
    kmdw = STATIONS["KMDW"]
    conv = detect_high_conviction(
        kmdw, date(2026, 6, 23), now_hour=15.0,
        obs_hours=[12, 13, 14, 15], obs_temps=[71, 72, 72, 72], model_rise=0.0,
    )
    assert conv.high_conviction  # trend_flat + model_done fire
    assert conv.trend_flat and conv.model_done


def test_morning_rising_is_not_high_conviction():
    kmdw = STATIONS["KMDW"]
    conv = detect_high_conviction(
        kmdw, date(2026, 6, 23), now_hour=9.0,
        obs_hours=[7, 8, 9], obs_temps=[63, 66, 69], model_rise=8.0,
    )
    assert not conv.high_conviction


def test_obs_anchored_high_enforces_floor():
    # Even if current temp + tiny rise is below what we've already seen, high >= obs_max.
    assert obs_anchored_high(obs_max_so_far=72.0, obs_now=70.0, model_remaining_rise=0.0) == 72.0
    assert obs_anchored_high(72.0, 71.0, 3.0) == 74.0


def test_interval_collapses_under_high_conviction():
    locked = intraday_nowcast(72.0, 72.0, 0.0, high_conviction=True)
    assert locked.high == 72.0
    assert (locked.hi - locked.lo) <= 1.0  # near-point band
    assert locked.lo >= 72.0 - 1e-9        # floored at obs_max
    open_ = intraday_nowcast(72.0, 72.0, 0.0, high_conviction=False)
    assert (open_.hi - open_.lo) > (locked.hi - locked.lo)  # wider when unresolved
