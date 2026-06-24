"""Online source selection as variance-aware Bayesian Model Averaging.

Each forecast source is an "expert". Per region we track, online, each expert's
running BIAS and ERROR VARIANCE (EWMA with forgetting, so it adapts to seasons /
model upgrades). We DE-BIAS each expert and combine by INVERSE-VARIANCE weights
(a fixed-share floor keeps the Hedge-style tracking property so a recovering
expert isn't locked out). This yields both the blend weights AND a calibrated
predictive variance (BMA within + between-expert), which the confidence layer
needs — plain Hedge gave weights but no variance, so confidence wasn't derivable.

Update (after CLI truth y), for each present expert i:
    e   = f_i - y
    b_i <- b_i + bias_lr * (e - b_i)                  # running bias
    r   = e - b_i
    s2_i<- max((1-var_lr)*s2_i + var_lr*r^2, s2_floor) # running error variance
Blend (over present experts):
    g_i = f_i - b_i                                   # de-biased forecast
    w_i ∝ 1/s2_i ;  w <- (1-alpha)w + alpha/N         # inverse-variance + fixed-share
    mu_F  = Σ w_i g_i
    s2_F  = kappa * Σ w_i (s2_i + (g_i - mu_F)^2)      # within + between (BMA), corr-inflated

Refs: Raftery et al. 2005 (BMA), Fisher inverse-variance pooling, Herbster-Warmuth
fixed-share. State is JSON-serializable for daily persistence.
"""
from __future__ import annotations

import math

# Informed cold-start priors on each expert's error variance (°F^2). NWS-NBM and
# GFS-seamless were tighter in R&D; ECMWF coarser (esp. coastal). Tunable.
DEFAULT_PRIOR_S2 = {"nws_nbm": 4.0, "gfs": 4.0, "ecmwf_ifs": 9.0, "ecmwf_aifs": 9.0}
PRIOR_FALLBACK = 6.25  # (2.5 °F)^2


def _is_present(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


class OnlineExpertBlend:
    def __init__(self, experts: list[str], alpha: float = 0.02, bias_lr: float = 0.08,
                 var_lr: float = 0.06, s2_floor: float = 0.25, kappa: float = 1.0,
                 priors: dict | None = None) -> None:
        if not experts:
            raise ValueError("need at least one expert")
        self.experts = list(experts)
        self.n = len(self.experts)
        self.alpha, self.bias_lr, self.var_lr = alpha, bias_lr, var_lr
        self.s2_floor, self.kappa = s2_floor, kappa
        self.priors = priors or {e: DEFAULT_PRIOR_S2.get(e, PRIOR_FALLBACK) for e in self.experts}
        # per-region: {expert: {"b":bias, "s2":var, "n":count}}
        self._state: dict[str, dict] = {}

    def _region(self, key: str) -> dict:
        if key not in self._state:
            self._state[key] = {e: {"b": 0.0, "s2": float(self.priors.get(e, PRIOR_FALLBACK)),
                                    "n": 0.0} for e in self.experts}
        return self._state[key]

    def _present(self, values: dict) -> list[str]:
        return [e for e in self.experts if e in values and _is_present(values[e])]

    # ---- prediction --------------------------------------------------------
    def predict_dist(self, key: str, values: dict) -> tuple[float | None, float | None]:
        """De-biased BMA blend -> (mu_F, sigma_F^2) over present experts."""
        reg = self._region(key)
        present = self._present(values)
        if not present:
            return (None, None)
        inv = {e: 1.0 / max(reg[e]["s2"], self.s2_floor) for e in present}
        z = sum(inv.values())
        w = {e: inv[e] / z for e in present}
        w = {e: (1 - self.alpha) * w[e] + self.alpha / len(present) for e in present}  # fixed-share
        g = {e: values[e] - reg[e]["b"] for e in present}                              # de-bias
        mu = sum(w[e] * g[e] for e in present)
        var = self.kappa * sum(w[e] * (reg[e]["s2"] + (g[e] - mu) ** 2) for e in present)
        return (mu, var)

    def predict(self, key: str, values: dict) -> float | None:
        return self.predict_dist(key, values)[0]

    def weights(self, key: str) -> dict[str, float]:
        """Display weights (inverse-variance over ALL experts at current variances)."""
        reg = self._region(key)
        inv = {e: 1.0 / max(reg[e]["s2"], self.s2_floor) for e in self.experts}
        z = sum(inv.values())
        return {e: inv[e] / z for e in self.experts}

    # ---- update ------------------------------------------------------------
    def update(self, key: str, values: dict, truth: float) -> None:
        reg = self._region(key)
        for e in self._present(values):
            err = values[e] - truth
            st = reg[e]
            st["b"] += self.bias_lr * (err - st["b"])
            resid = err - st["b"]
            st["s2"] = max((1 - self.var_lr) * st["s2"] + self.var_lr * resid ** 2, self.s2_floor)
            st["n"] += 1.0

    # ---- persistence -------------------------------------------------------
    def to_state(self) -> dict:
        return {"schema": "bma1", "experts": self.experts, "alpha": self.alpha,
                "bias_lr": self.bias_lr, "var_lr": self.var_lr, "s2_floor": self.s2_floor,
                "kappa": self.kappa, "priors": self.priors, "regions": self._state}

    @classmethod
    def from_state(cls, state: dict) -> "OnlineExpertBlend":
        if state.get("schema") != "bma1":          # legacy (Hedge) state -> start fresh
            raise ValueError("incompatible state schema")
        obj = cls(state["experts"], alpha=state["alpha"], bias_lr=state["bias_lr"],
                  var_lr=state["var_lr"], s2_floor=state["s2_floor"], kappa=state["kappa"],
                  priors=state.get("priors"))
        obj._state = {k: {e: dict(v) for e, v in reg.items()} for k, reg in state.get("regions", {}).items()}
        return obj
