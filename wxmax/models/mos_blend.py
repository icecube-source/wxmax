"""MOS bias-correction / multi-model blend (P4).

Classic Model Output Statistics: regress the observed station daily-max on the
multi-model forecast vector. A plain least-squares linear blend already removes
the per-model, per-station bias we saw (ECMWF +2°F at coastal sites) and beats
any single raw model. LightGBM is the drop-in nonlinear upgrade (P4 stretch);
the interface (`fit`/`predict`) is identical so the eval harness is unchanged.
"""
from __future__ import annotations

import numpy as np


class LinearMOS:
    """Ordinary-least-squares blend with intercept: y ~ 1 + F."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None  # shape (k+1,), [intercept, w...]

    def fit(self, F: np.ndarray, y: np.ndarray) -> "LinearMOS":
        F = np.asarray(F, dtype=float)
        y = np.asarray(y, dtype=float)
        X = np.column_stack([np.ones(len(F)), F])
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LinearMOS not fitted")
        F = np.asarray(F, dtype=float)
        X = np.column_stack([np.ones(len(F)), F])
        return X @ self.coef_
