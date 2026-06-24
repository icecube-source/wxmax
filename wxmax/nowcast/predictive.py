"""Predictive distribution for today's max + the calibrated confidence.

Today's max is Y = max(M_t, R_t):
  M_t = observed max so far (known; sensor noise sigma_obs ~ 0.5 F)
  R_t ~ N(mu_R, sigma_R^2) = remaining-day max, decaying with P_peak(t):
    mu_R    = M_t + (1 - P_peak) * max(mu_F - M_t, delta_climo)
    sigma_R^2 = (1 - P_peak) * sigma_F^2 + P_peak * sigma_obs^2
  (mu_F, sigma_F^2) = de-biased BMA forecast from the online learner.

Y is max(known, Gaussian) -> a Gaussian with an atom at M_t, NOT a plain Gaussian.
Its mean (the real-time best estimate) and variance use rectified-Gaussian moments;
confidence = P(|Y - yhat| <= 1 F) read off the max-structure CDF. This is
forecast-driven (non-zero) in the morning and -> ~0.99 once the peak is locked.

A dispersion multiplier `lam` (from confidence calibration) inflates sigma_F to fix
over/under-dispersion; an isotonic map (also from calibration) maps raw -> reliable.
"""
from __future__ import annotations

import math

SIGMA_OBS = 0.5            # ASOS sensor / whole-degree floor (°F)
_Z = {0.80: 1.2815515, 0.90: 1.6448536}


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predictive_max(mu_F: float, sigma_F2: float, m_t: float, p_peak: float,
                   sigma_obs: float = SIGMA_OBS, delta_climo: float = 0.0,
                   lam: float = 1.0) -> tuple[float, float, float, float]:
    """Return (yhat, var, mu_R, sigma_R) for Y = max(m_t, R_t)."""
    sigma_F2 = max(sigma_F2, 0.0) * lam * lam
    mu_R = m_t + (1.0 - p_peak) * max(mu_F - m_t, delta_climo)
    sigma_R2 = (1.0 - p_peak) * sigma_F2 + p_peak * sigma_obs ** 2
    sigma_R = math.sqrt(max(sigma_R2, sigma_obs ** 2))
    z = (m_t - mu_R) / sigma_R
    Phiz, phiz = _Phi(z), _phi(z)
    yhat = m_t + (mu_R - m_t) * (1.0 - Phiz) + sigma_R * phiz
    e_y2 = (m_t ** 2) * Phiz + (mu_R ** 2 + sigma_R ** 2) * (1.0 - Phiz) \
        + sigma_R * (mu_R + m_t) * phiz
    var = max(0.0, e_y2 - yhat ** 2)
    return yhat, var, mu_R, sigma_R


def cdf_Y(y: float, m_t: float, mu_R: float, sigma_R: float) -> float:
    """CDF of Y = max(m_t, R_t): 0 below the observed-max floor, Gaussian above."""
    if y < m_t:
        return 0.0
    return _Phi((y - mu_R) / sigma_R)


def confidence_pm1(yhat: float, m_t: float, mu_R: float, sigma_R: float,
                   tau: float = 1.0) -> float:
    """P(|Y - yhat| <= tau) under the predictive law (raw, pre-isotonic)."""
    return cdf_Y(yhat + tau, m_t, mu_R, sigma_R) - cdf_Y(yhat - tau, m_t, mu_R, sigma_R)


def real_time_estimate(mu_F: float, sigma_F2: float, m_t: float, p_peak: float,
                       sigma_obs: float = SIGMA_OBS, delta_climo: float = 0.0,
                       lam: float = 1.0, level: float = 0.80, tau: float = 1.0) -> dict:
    """Full real-time summary: best estimate, central interval, and confidence.

    The interval is the central `level` credible interval of Y (floored at m_t,
    since the max can't be below what's been observed)."""
    yhat, var, mu_R, sigma_R = predictive_max(mu_F, sigma_F2, m_t, p_peak, sigma_obs, delta_climo, lam)
    atom = _Phi((m_t - mu_R) / sigma_R)          # P(Y == m_t)
    p_lo, p_hi = (1 - level) / 2, 1 - (1 - level) / 2
    zl = _Z.get(level, 1.2815515)
    lo = m_t if p_lo <= atom else mu_R - sigma_R * zl
    hi = m_t if p_hi <= atom else mu_R + sigma_R * zl
    lo = max(m_t, lo)
    hi = max(lo, hi)
    conf = confidence_pm1(yhat, m_t, mu_R, sigma_R, tau)
    return {"yhat": yhat, "var": var, "lo": lo, "hi": hi, "confidence": conf}
