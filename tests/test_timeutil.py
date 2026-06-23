"""Tests for tz-aware daily-max bucketing."""
from datetime import date

import numpy as np
import pandas as pd

from wxmax.timeutil import daily_max, observed_max_for_day, to_local


def _utc(hours):
    """Build a UTC hourly index for the given ISO date prefix list."""
    return pd.to_datetime(hours, utc=True)


def test_local_bucketing_splits_across_utc_midnight():
    # LA is UTC-7 in summer. A peak at 2026-06-23T22:00Z is 15:00 local on the
    # 23rd; a value at 2026-06-24T06:00Z is 23:00 local still on the 23rd.
    times = _utc([
        "2026-06-23T20:00Z",  # 13:00 PDT 23rd
        "2026-06-23T22:00Z",  # 15:00 PDT 23rd  <- daytime peak
        "2026-06-24T06:00Z",  # 23:00 PDT 23rd
        "2026-06-24T10:00Z",  # 03:00 PDT 24th
    ])
    vals = [70.0, 85.0, 66.0, 60.0]
    dm = daily_max(times, vals, tz="America/Los_Angeles")
    # All but the last sample belong to the local 23rd.
    assert dm.loc[date(2026, 6, 23), "tmax"] == 85.0
    assert dm.loc[date(2026, 6, 23), "n_obs"] == 3
    assert dm.loc[date(2026, 6, 24), "tmax"] == 60.0


def test_utc_bucketing_would_differ():
    # Same data grouped naively by UTC date would put the 06:00Z sample on the
    # 24th — confirming why local bucketing matters.
    times = _utc(["2026-06-23T22:00Z", "2026-06-24T06:00Z"])
    vals = [85.0, 66.0]
    local = to_local(times, "America/Los_Angeles")
    assert [d.date() for d in local] == [date(2026, 6, 23), date(2026, 6, 23)]


def test_min_obs_filter_drops_thin_days():
    times = _utc(["2026-06-23T20:00Z", "2026-06-24T10:00Z"])
    vals = [80.0, 60.0]
    dm = daily_max(times, vals, tz="America/Los_Angeles", min_obs_per_day=2)
    # Each local day has only 1 sample -> all dropped.
    assert dm.empty


def test_nan_values_ignored():
    times = _utc(["2026-06-23T20:00Z", "2026-06-23T21:00Z"])
    vals = [np.nan, 77.0]
    dm = daily_max(times, vals, tz="America/Los_Angeles")
    assert dm.loc[date(2026, 6, 23), "tmax"] == 77.0
    assert dm.loc[date(2026, 6, 23), "n_obs"] == 1


def test_observed_max_for_day():
    times = _utc(["2026-06-23T20:00Z", "2026-06-23T22:00Z"])
    vals = [70.0, 88.0]
    assert observed_max_for_day(times, vals, "America/Los_Angeles", date(2026, 6, 23)) == 88.0
    assert observed_max_for_day(times, vals, "America/Los_Angeles", date(2026, 6, 24)) is None


def test_phoenix_has_no_dst():
    # America/Phoenix stays UTC-7 year round; a Jan timestamp must use -7, not -6.
    times = _utc(["2026-01-15T20:00Z"])  # 13:00 MST
    local = to_local(times, "America/Phoenix")
    assert local[0].hour == 13
