"""Tests for the Hedge + Fixed-Share online expert blend."""
import numpy as np

from wxmax.models.online_experts import OnlineExpertBlend

EXPERTS = ["good", "biased", "noisy"]


def _simulate(T=400, seed=0):
    rng = np.random.default_rng(seed)
    blend = OnlineExpertBlend(EXPERTS, horizon=T, alpha=0.02)
    eq_loss = 0.0
    for _ in range(T):
        truth = rng.uniform(60, 100)
        vals = {
            "good": truth + rng.normal(0, 1.0),    # ~unbiased, tight
            "biased": truth + 6.0,                  # always 6°F hot
            "noisy": truth + rng.normal(0, 4.0),    # unbiased, wide
        }
        blend.update("KXXX", vals, truth)
        eq = np.mean([vals[e] for e in EXPERTS])
        eq_loss += min(1.0, abs(eq - truth) / blend.l_max)
    return blend, eq_loss


def test_weights_concentrate_on_best_expert():
    blend, _ = _simulate()
    w = blend.weights("KXXX")
    assert max(w, key=w.get) == "good"
    assert w["good"] > w["noisy"] > w["biased"]


def test_weights_sum_to_one_and_fixed_share_floor():
    blend, _ = _simulate()
    w = np.array(list(blend.weights("KXXX").values()))
    assert abs(w.sum() - 1.0) < 1e-9
    # fixed-share guarantees every weight >= alpha / N each round
    assert w.min() >= blend.alpha / blend.n - 1e-9


def test_regret_is_sublinear_and_beats_equal_weight():
    T = 400
    blend, eq_loss = _simulate(T=T)
    # regret vs best fixed expert stays well sub-linear (bound ~ sqrt(T ln N))
    assert blend.regret("KXXX") < 0.1 * T
    # the adaptive blend beats naive equal-weight (which carries the biased expert)
    assert blend._state["KXXX"]["cum_blend"] < eq_loss


def test_missing_expert_is_ignored_and_renormalized():
    blend = OnlineExpertBlend(EXPERTS, horizon=100)
    # only two experts present -> combo uses just those, weights renormalized
    pred = blend.predict("K", {"good": 80.0, "noisy": 90.0})
    assert 80.0 <= pred <= 90.0
    # NaN / None treated as absent
    assert blend.predict("K", {"good": 80.0, "biased": float("nan")}) == 80.0
    # update with a missing expert doesn't crash and leaves its weight finite
    blend.update("K", {"good": 80.0}, 79.0)
    assert abs(sum(blend.weights("K").values()) - 1.0) < 1e-9


def test_state_roundtrip():
    blend, _ = _simulate(T=50)
    state = blend.to_state()
    clone = OnlineExpertBlend.from_state(state)
    assert clone.weights("KXXX") == blend.weights("KXXX")
    assert clone.eta == blend.eta
