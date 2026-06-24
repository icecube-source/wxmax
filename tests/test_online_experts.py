"""Tests for the variance-aware BMA expert blend."""
import numpy as np

from wxmax.models.online_experts import OnlineExpertBlend

EXPERTS = ["good", "biased", "noisy"]


def _train(T=300, seed=0):
    rng = np.random.default_rng(seed)
    blend = OnlineExpertBlend(EXPERTS, priors={e: 9.0 for e in EXPERTS})
    for _ in range(T):
        truth = rng.uniform(60, 100)
        vals = {
            "good": truth + rng.normal(0, 1.0),         # unbiased, tight
            "biased": truth + 6.0 + rng.normal(0, 1.0),  # +6°F bias, tight after de-bias
            "noisy": truth + rng.normal(0, 4.0),         # unbiased, wide
        }
        blend.update("K", vals, truth)
    return blend


def test_weights_favor_low_variance_experts():
    blend = _train()
    w = blend.weights("K")
    assert min(w, key=w.get) == "noisy"       # widest -> least weight
    assert w["good"] > w["noisy"] and w["biased"] > w["noisy"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_bias_is_learned_and_removed():
    blend = _train()
    reg = blend._region("K")
    assert abs(reg["biased"]["b"] - 6.0) < 1.0          # ~+6°F bias captured
    # de-biased blend prediction is ~unbiased: feed the experts' raw values for a known truth
    truth = 80.0
    mu, var = blend.predict_dist("K", {"good": 80.0, "biased": 86.0, "noisy": 80.0})
    assert abs(mu - truth) < 1.5 and var > 0


def test_missing_expert_renormalizes():
    blend = _train()
    mu = blend.predict("K", {"good": 80.0})       # only one present
    assert abs(mu - 80.0) < 1.5                   # ~ de-biased good
    mu2, var2 = blend.predict_dist("K", {"good": 80.0, "noisy": 90.0})
    assert 79 <= mu2 <= 91 and var2 > 0


def test_predictive_variance_positive_and_betweenterm():
    blend = OnlineExpertBlend(EXPERTS, priors={e: 4.0 for e in EXPERTS})
    # experts disagree a lot -> between-expert term inflates variance
    _, var_spread = blend.predict_dist("K", {"good": 70.0, "biased": 90.0, "noisy": 80.0})
    _, var_agree = blend.predict_dist("K", {"good": 80.0, "biased": 80.0, "noisy": 80.0})
    assert var_spread > var_agree > 0


def test_state_roundtrip():
    blend = _train(T=40)
    clone = OnlineExpertBlend.from_state(blend.to_state())
    assert clone.weights("K") == blend.weights("K")
    assert clone._region("K")["biased"]["b"] == blend._region("K")["biased"]["b"]
