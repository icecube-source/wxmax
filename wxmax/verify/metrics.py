"""Verification metrics for point and probabilistic daily-max forecasts.

Pure NumPy — no scoring library needed. Conventions:
  y       observed daily max (1-D, length n)
  members ensemble matrix (n, m)
  q_pred  quantile forecasts (n, k) aligned to a quantile grid `taus` (k,)

Deterministic: mae, rmse, bias.
Probabilistic:  crps_ensemble, crps_gaussian, pinball_loss.
Calibration:    pit_gaussian, pit_ensemble, rank_histogram,
                interval_coverage, interval_width, spread_skill_ratio.
"""
from __future__ import annotations

from math import erf, pi, sqrt

import numpy as np

_SQRT2 = sqrt(2.0)
_INV_SQRTPI = 1.0 / sqrt(pi)
_erf_vec = np.vectorize(erf)


def _phi(z):  # standard normal pdf
    return np.exp(-0.5 * np.asarray(z) ** 2) / sqrt(2 * pi)


def _Phi(z):  # standard normal cdf
    return 0.5 * (1.0 + _erf_vec(np.asarray(z) / _SQRT2))


# ---- deterministic -------------------------------------------------------
def mae(y, yhat) -> float:
    return float(np.nanmean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    d = np.asarray(y) - np.asarray(yhat)
    return float(np.sqrt(np.nanmean(d ** 2)))


def bias(y, yhat) -> float:
    return float(np.nanmean(np.asarray(yhat) - np.asarray(y)))


# ---- probabilistic -------------------------------------------------------
def pinball_loss(y, q_pred, tau: float) -> float:
    """Quantile (pinball) loss for a single quantile level tau."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q_pred, dtype=float)
    d = y - q
    return float(np.nanmean(np.maximum(tau * d, (tau - 1.0) * d)))


def crps_ensemble(y, members) -> float:
    """Mean CRPS for an ensemble, empirical estimator.

    CRPS_i = mean_j|x_ij - y_i| - 0.5 * mean_{j,k}|x_ij - x_ik|.
    Reduces to |x - y| for a degenerate (point) ensemble.
    """
    y = np.asarray(y, dtype=float)
    M = np.atleast_2d(np.asarray(members, dtype=float))
    if M.shape[0] != y.shape[0]:
        M = M.T
    term1 = np.nanmean(np.abs(M - y[:, None]), axis=1)
    diff = np.abs(M[:, :, None] - M[:, None, :])
    term2 = 0.5 * np.nanmean(diff, axis=(1, 2))
    return float(np.nanmean(term1 - term2))


def crps_gaussian(y, mu, sigma) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = (y - mu) / sigma
    crps = sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - _INV_SQRTPI)
    return float(np.nanmean(crps))


# ---- calibration ---------------------------------------------------------
def pit_gaussian(y, mu, sigma) -> np.ndarray:
    """Probability integral transform under a Gaussian forecast; ~U(0,1) if calibrated."""
    return _Phi((np.asarray(y) - np.asarray(mu)) / np.asarray(sigma))


def pit_ensemble(y, members) -> np.ndarray:
    """Empirical PIT = fraction of members <= obs (ensemble analogue)."""
    y = np.asarray(y, dtype=float)
    M = np.atleast_2d(np.asarray(members, dtype=float))
    if M.shape[0] != y.shape[0]:
        M = M.T
    return np.mean(M <= y[:, None], axis=1)


def rank_histogram(y, members) -> np.ndarray:
    """Counts of the obs rank among m members (length m+1). Flat = calibrated."""
    y = np.asarray(y, dtype=float)
    M = np.atleast_2d(np.asarray(members, dtype=float))
    if M.shape[0] != y.shape[0]:
        M = M.T
    m = M.shape[1]
    ranks = np.sum(M < y[:, None], axis=1)  # 0..m
    return np.bincount(ranks, minlength=m + 1)


def interval_coverage(y, lo, hi) -> float:
    """Empirical coverage: fraction of obs within [lo, hi]."""
    y = np.asarray(y, dtype=float)
    return float(np.nanmean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def interval_width(lo, hi) -> float:
    return float(np.nanmean(np.asarray(hi) - np.asarray(lo)))


def spread_skill_ratio(members, y) -> float:
    """Ensemble spread / RMSE(ensemble mean). ~1.0 when well dispersed."""
    y = np.asarray(y, dtype=float)
    M = np.atleast_2d(np.asarray(members, dtype=float))
    if M.shape[0] != y.shape[0]:
        M = M.T
    spread = np.sqrt(np.nanmean(np.nanvar(M, axis=1, ddof=1)))
    skill = rmse(y, np.nanmean(M, axis=1))
    return float(spread / skill) if skill > 0 else float("nan")


def sort_quantiles(q_pred: np.ndarray) -> np.ndarray:
    """Enforce monotone quantiles by sorting along the quantile axis (anti-crossing)."""
    return np.sort(np.asarray(q_pred, dtype=float), axis=-1)
