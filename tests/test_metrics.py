"""Tests for verification metrics (closed-form / synthetic checks)."""
import numpy as np

from wxmax.verify import metrics as M


def test_point_metrics():
    y = np.array([10.0, 20.0, 30.0])
    yhat = np.array([12.0, 18.0, 30.0])
    assert M.mae(y, yhat) == (2 + 2 + 0) / 3
    assert abs(M.rmse(y, yhat) - np.sqrt((4 + 4 + 0) / 3)) < 1e-12
    assert M.bias(y, yhat) == (2 - 2 + 0) / 3


def test_pinball_median_is_half_mae():
    y = np.array([10.0, 20.0, 30.0])
    q = np.array([12.0, 18.0, 33.0])
    assert abs(M.pinball_loss(y, q, 0.5) - 0.5 * M.mae(y, q)) < 1e-12


def test_crps_ensemble_degenerate_equals_abs_error():
    # All members identical -> CRPS = |member - y|.
    y = np.array([10.0, 20.0])
    members = np.array([[12.0, 12.0, 12.0], [18.0, 18.0, 18.0]])
    assert abs(M.crps_ensemble(y, members) - np.mean([2.0, 2.0])) < 1e-9


def test_crps_gaussian_known_value():
    # mu=y, sigma=1 -> CRPS = 2*phi(0) - 1/sqrt(pi) ~= 0.23369.
    crps = M.crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert abs(crps - 0.233695) < 1e-4


def test_gaussian_interval_coverage_is_near_nominal():
    rng = np.random.default_rng(0)
    n = 20000
    mu = np.zeros(n)
    sigma = np.ones(n)
    y = rng.normal(mu, sigma)
    # 90% central interval under the (correct) forecast distribution.
    lo, hi = mu - 1.6448536 * sigma, mu + 1.6448536 * sigma
    cov = M.interval_coverage(y, lo, hi)
    assert abs(cov - 0.90) < 0.02


def test_spread_skill_ratio_near_one_when_calibrated():
    rng = np.random.default_rng(1)
    n, m = 4000, 30
    # Forecast signal mu; truth and members both scatter around mu with the
    # SAME error std s. Then spread ~= s and RMSE(ens mean) ~= s -> ratio ~= 1.
    mu = rng.normal(0, 1, n)
    s = 1.0
    truth = mu + rng.normal(0, s, n)
    members = mu[:, None] + rng.normal(0, s, size=(n, m))
    r = M.spread_skill_ratio(members, truth)
    assert 0.85 < r < 1.15


def test_rank_histogram_shape_and_total():
    rng = np.random.default_rng(2)
    n, m = 500, 20
    members = rng.normal(0, 1, size=(n, m))
    y = rng.normal(0, 1, n)
    h = M.rank_histogram(y, members)
    assert h.shape == (m + 1,)
    assert h.sum() == n


def test_sort_quantiles_monotone():
    q = np.array([[5.0, 3.0, 9.0, 1.0]])
    s = M.sort_quantiles(q)
    assert list(s[0]) == [1.0, 3.0, 5.0, 9.0]
