"""Offline tests for MOS blend, EMOS, and conformal calibration."""
import numpy as np

from wxmax.calibrate.conformal import ConformalInterval, gaussian_interval
from wxmax.models.emos import GaussianEMOS
from wxmax.models.mos_blend import LinearMOS
from wxmax.verify.metrics import interval_coverage


def test_linear_mos_recovers_coefficients():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(2000, 2))
    y = 2.0 + 1.5 * F[:, 0] - 0.5 * F[:, 1]  # exact linear, no noise
    mos = LinearMOS().fit(F, y)
    assert np.allclose(mos.coef_, [2.0, 1.5, -0.5], atol=1e-6)
    assert np.allclose(mos.predict(F), y, atol=1e-6)


def test_emos_sigma_grows_with_spread():
    rng = np.random.default_rng(1)
    n = 6000
    F = rng.normal(size=(n, 3))
    spread = rng.uniform(0.5, 3.0, n)
    mu_true = 1.0 + F.sum(axis=1)
    y = mu_true + rng.normal(0, spread)  # heteroscedastic in spread
    emos = GaussianEMOS().fit(F, y, spread)
    _, sigma = emos.predict(F, spread)
    # sigma should be monotinically related to spread (positive correlation)
    assert np.corrcoef(spread, sigma)[0, 1] > 0.9


def test_conformal_restores_undercovered_interval():
    rng = np.random.default_rng(2)
    y_cal = rng.normal(0, 1, 4000)
    y_te = rng.normal(0, 1, 4000)
    # Base predictive sigma is too small -> 90% interval undercovers badly.
    mu = np.zeros(4000)
    sigma = np.full(4000, 0.5)
    lo_ca, hi_ca = gaussian_interval(mu, sigma, 0.90)
    raw_cov = interval_coverage(y_te, *gaussian_interval(mu, sigma, 0.90))
    assert raw_cov < 0.75  # demonstrably undercovered

    conf = ConformalInterval(alpha=0.10).fit(lo_ca, hi_ca, y_cal)
    clo, chi = conf.transform(*gaussian_interval(mu, sigma, 0.90))
    cqr_cov = interval_coverage(y_te, clo, chi)
    assert cqr_cov >= 0.88  # restored to ~nominal
    assert cqr_cov > raw_cov


def test_conformal_quantile_rank_clips_to_one():
    # Tiny calibration set: rank must clip to 1.0 (can't exceed the max score).
    conf = ConformalInterval(alpha=0.10).fit([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 2.0, 0.5])
    assert conf.qhat_ is not None and conf.qhat_ >= 1.0
