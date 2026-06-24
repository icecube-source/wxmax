"""Tests for the predictive max distribution + confidence."""
from wxmax.nowcast.predictive import (
    cdf_Y,
    confidence_pm1,
    predictive_max,
    real_time_estimate,
)


def test_morning_is_forecast_driven_nonzero():
    # p_peak=0: yhat ~ mu_F, var ~ sigma_F^2, confidence forecast-driven (NOT 0)
    yhat, var, mu_R, sigma_R = predictive_max(mu_F=90, sigma_F2=4, m_t=80, p_peak=0.0)
    assert abs(yhat - 90) < 1e-3 and abs(var - 4) < 1e-3
    conf = confidence_pm1(yhat, 80, mu_R, sigma_R)
    assert 0.30 < conf < 0.45                 # ~38% with sigma=2°F -> meaningful, not 0


def test_tighter_forecast_more_confident():
    c_tight = real_time_estimate(85, 1.0, 80, 0.0)["confidence"]   # sigma_F=1
    c_wide = real_time_estimate(85, 16.0, 80, 0.0)["confidence"]   # sigma_F=4
    assert c_tight > c_wide > 0


def test_locked_collapses_to_observed_max():
    yhat, var, mu_R, sigma_R = predictive_max(mu_F=96, sigma_F2=9, m_t=92, p_peak=1.0)
    assert abs(yhat - 92) < 0.3 and var < 0.5     # -> observed max, tiny variance
    conf = confidence_pm1(yhat, 92, mu_R, sigma_R)
    assert conf > 0.95                            # ~99% once locked


def test_cdf_floored_at_observed_max_and_monotone():
    _, _, mu_R, sigma_R = predictive_max(90, 4, 80, 0.3)
    assert cdf_Y(79.0, 80, mu_R, sigma_R) == 0.0          # below the floor
    assert cdf_Y(85, 80, mu_R, sigma_R) <= cdf_Y(95, 80, mu_R, sigma_R)


def test_real_time_estimate_interval_floored():
    r = real_time_estimate(mu_F=90, sigma_F2=9, m_t=85, p_peak=0.3)
    assert r["yhat"] >= 85 - 1e-9                 # max can't be below observed
    assert r["lo"] >= 85 - 1e-9 and r["hi"] >= r["lo"]
    assert 0.0 <= r["confidence"] <= 1.0
