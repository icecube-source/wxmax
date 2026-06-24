"""Obs-anchored daily-max nowcast + interval that collapses as the day resolves.

The daily max can never be below what's already been observed, so we anchor to
the running observed max and add only the model's expected *remaining* rise.
Once HIGH conviction fires, the interval collapses to a near-point band; before
that it stays wide (the day is unresolved).
"""
from __future__ import annotations

from dataclasses import dataclass


def obs_anchored_high(obs_max_so_far: float, obs_now: float, model_remaining_rise: float | None) -> float:
    """max the day can still reach = max(seen so far, current + remaining model rise)."""
    rise = max(0.0, model_remaining_rise or 0.0)
    return max(obs_max_so_far, obs_now + rise)


@dataclass(frozen=True)
class Nowcast:
    high: float
    lo: float
    hi: float
    high_conviction: bool


def intraday_nowcast(
    obs_max_so_far: float, obs_now: float, model_remaining_rise: float | None,
    high_conviction: bool, locked_half: float = 0.5, open_half: float = 2.0,
) -> Nowcast:
    """Obs-anchored high + interval. Floor the low bound at obs_max_so_far (the max
    can't be less than what we've already seen)."""
    high = obs_anchored_high(obs_max_so_far, obs_now, model_remaining_rise)
    half = locked_half if high_conviction else open_half
    lo = max(obs_max_so_far, high - half)
    hi = high + half
    return Nowcast(high=high, lo=lo, hi=hi, high_conviction=high_conviction)
