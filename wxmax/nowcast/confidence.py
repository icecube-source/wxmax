"""Calibrated P(daily max has already occurred) = P_clim x P_traj x P_model.

  P_clim  : climatological ECDF of the peak hour at `now` (peak_climatology).
  P_traj  : has the temperature actually stopped rising? -- from `dwell` (minutes
            since the running max was last beaten) and `drop` (max_so_far - now).
  P_model : ~1 when the model expects no further rise, -> 0 while more heating is
            coming (kills sea-breeze / transient-cloud false flats).

All three must agree (product) to reach high confidence. A lock additionally
requires HARD guards (never before the back-tested safe hour `p_lock`, a minimum
sustained `dwell`, and the model showing essentially no remaining rise), so a
short mid-day flat alone can never lock. Sigmoid constants are literature-based
defaults; the back-test calibrates the per-station guards (`p_lock`, `dwell_min`)
to hold the locked-too-early rate <= 1%.
"""
from __future__ import annotations

import math

DWELL0 = 90.0     # minutes of "stopped rising" at which P_traj crosses 0.5
DROP0 = 0.5       # °F drop from the running max at which P_traj crosses 0.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def p_traj(dwell_min: float, drop_f: float, a: float = 0.05, b: float = 1.6) -> float:
    """P(temperature has actually stopped rising)."""
    return _sigmoid(a * (dwell_min - DWELL0) + b * (drop_f - DROP0))


def p_model(remaining_rise: float | None, c: float = 5.0, offset: float = 6.0) -> float:
    """~1 when the model expects no more rise; drops sharply as expected rise grows."""
    r = 0.0 if remaining_rise is None else max(0.0, remaining_rise)
    return _sigmoid(offset - c * r)


def peak_passed_confidence(p_clim: float | None, dwell_min: float, drop_f: float,
                           remaining_rise: float | None) -> float | None:
    if p_clim is None:
        return None
    return p_clim * p_traj(dwell_min, drop_f) * p_model(remaining_rise)


def decide_lock(conf: float | None, now_hour: float, p_lock: float, dwell_min: float,
                remaining_rise: float | None, threshold: float = 0.99,
                dwell_floor: float = 90.0, rise_eps: float = 0.2) -> bool:
    """Lock only at >= threshold confidence AND all hard guards pass."""
    if conf is None:
        return False
    guards = (
        now_hour >= p_lock
        and dwell_min >= dwell_floor
        and remaining_rise is not None and remaining_rise < rise_eps
    )
    return bool(conf >= threshold and guards)
