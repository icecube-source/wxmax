"""Gaussian EMOS / Nonhomogeneous Gaussian Regression head (P5).

Predictive distribution N(mu, sigma) where:
  mu    = bias-corrected linear blend of the model forecasts (the MOS mean), and
  sigma = sqrt(a + b * ensemble_variance)  — variance grows with cross-model
          disagreement, the NGR insight that spread should track uncertainty.

We fit mu by OLS (the MOS mean) and (a, b) by regressing squared residuals on
ensemble variance. This is a lightweight, closed-form stand-in for the
minimum-CRPS fit; it yields a sharp, spread-aware Gaussian that the conformal
layer then calibrates to exact coverage.
"""
from __future__ import annotations

import numpy as np

from .mos_blend import LinearMOS


class GaussianEMOS:
    def __init__(self, var_floor: float = 0.25) -> None:
        self.mos = LinearMOS()
        self.var_coef_: np.ndarray | None = None  # [a, b]
        self.var_floor = var_floor

    def fit(self, F: np.ndarray, y: np.ndarray, spread: np.ndarray) -> "GaussianEMOS":
        F = np.asarray(F, float); y = np.asarray(y, float)
        spread = np.asarray(spread, float)
        self.mos.fit(F, y)
        resid = y - self.mos.predict(F)
        # var ~ a + b * spread^2  (regress squared error on ensemble variance)
        Z = np.column_stack([np.ones(len(spread)), spread ** 2])
        self.var_coef_, *_ = np.linalg.lstsq(Z, resid ** 2, rcond=None)
        return self

    def predict(self, F: np.ndarray, spread: np.ndarray):
        """Return (mu, sigma) for each row."""
        if self.var_coef_ is None:
            raise RuntimeError("GaussianEMOS not fitted")
        mu = self.mos.predict(F)
        spread = np.asarray(spread, float)
        var = self.var_coef_[0] + self.var_coef_[1] * spread ** 2
        sigma = np.sqrt(np.maximum(var, self.var_floor))
        return mu, sigma
