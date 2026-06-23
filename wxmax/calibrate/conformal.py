"""Conformalized Quantile Regression (CQR) — the reported confidence interval.

The EMOS Gaussian gives a sharp base interval [mu - z*sigma, mu + z*sigma], but
its coverage is only as good as the Gaussian assumption. Split-conformal CQR
adds a distribution-free, finite-sample coverage guarantee: on a held-out
calibration set we measure how far the truth fell outside the base interval, and
widen (or shrink) every interval by that empirical margin.

Guarantee: with exchangeable data, P(Y in calibrated interval) >= 1 - alpha.

Reference: Romano, Patterson & Candès, "Conformalized Quantile Regression"
(NeurIPS 2019). For seasonal drift, swap in Adaptive Conformal Inference (online
alpha update) — left as the P5 stretch.
"""
from __future__ import annotations

import numpy as np

# z = Phi^{-1}(1 - alpha/2) for common central levels (avoids needing erfinv).
Z_FOR_LEVEL = {0.80: 1.2815515, 0.90: 1.6448536, 0.95: 1.9599640}


def gaussian_interval(mu, sigma, level: float):
    """Central predictive interval [lo, hi] at the given coverage `level`."""
    if level not in Z_FOR_LEVEL:
        raise ValueError(f"level must be one of {sorted(Z_FOR_LEVEL)}")
    z = Z_FOR_LEVEL[level]
    mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    return mu - z * sigma, mu + z * sigma


class ConformalInterval:
    """Split-conformal calibrator for a base [lo, hi] interval at a fixed alpha."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.qhat_: float | None = None

    def fit(self, lo_cal, hi_cal, y_cal) -> "ConformalInterval":
        lo = np.asarray(lo_cal, float); hi = np.asarray(hi_cal, float)
        y = np.asarray(y_cal, float)
        # CQR nonconformity score: signed distance outside the base interval.
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        # finite-sample-adjusted (1 - alpha) empirical quantile
        rank = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat_ = float(np.quantile(scores, rank, method="higher"))
        return self

    def transform(self, lo, hi):
        if self.qhat_ is None:
            raise RuntimeError("ConformalInterval not fitted")
        return np.asarray(lo, float) - self.qhat_, np.asarray(hi, float) + self.qhat_
