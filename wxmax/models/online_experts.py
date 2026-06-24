"""Online source selection as prediction from expert advice.

Each forecast source is an "expert". Each day, per region, we predict a weighted
convex combination of the experts (the "ensemble ratio"), then -- once the NWS
CLI official max is observed -- pay a loss and update the weights. This is the
classic Hedge / Multiplicative-Weights algorithm; with a Fixed-Share step it
*tracks the best expert over time*, so the weighting adapts to seasons and model
upgrades instead of locking onto a stale historical winner.

Guarantee (Hedge): regret vs. the best fixed expert is O(sqrt(T ln N)) -- the
weighted blend provably converges toward the best source(s) per region, and in
practice beats the naive equal-weight average within days.

Update (after truth y_t), for experts present on round t:
    loss_i  = clip(|f_i - y_t| / L_max, 0, 1)
    w_i    <- w_i * exp(-eta * loss_i)          # multiplicative weights
    w      <- w / sum(w)                         # renormalize
    w_i    <- (1 - alpha) * w_i + alpha / N      # fixed-share (drift)
with eta = sqrt(8 ln N / horizon).

State is plain dict (JSON/Parquet serializable) so the panel can persist and
reload per-region weights across daily runs.
"""
from __future__ import annotations

import math

import numpy as np


class OnlineExpertBlend:
    def __init__(
        self,
        experts: list[str],
        horizon: int = 365,
        alpha: float = 0.02,
        l_max: float = 20.0,
        eta: float | None = None,
    ) -> None:
        if len(experts) < 1:
            raise ValueError("need at least one expert")
        self.experts = list(experts)
        self.n = len(self.experts)
        self.alpha = alpha
        self.l_max = l_max
        # theory-optimal learning rate for N experts over `horizon` rounds
        self.eta = eta if eta is not None else math.sqrt(8.0 * math.log(max(self.n, 2)) / horizon)
        # per-region state: key -> {"w": np.array, "n": int, "cum_loss": np.array, "cum_blend": float}
        self._state: dict[str, dict] = {}

    # ---- per-region weight vectors ----------------------------------------
    def _region(self, key: str) -> dict:
        if key not in self._state:
            self._state[key] = {
                "w": np.full(self.n, 1.0 / self.n),
                "n": 0,
                "cum_loss": np.zeros(self.n),  # per-expert cumulative normalized loss
                "cum_blend": 0.0,              # blend cumulative normalized loss
            }
        return self._state[key]

    def weights(self, key: str) -> dict[str, float]:
        w = self._region(key)["w"]
        return dict(zip(self.experts, w.tolist()))

    # ---- prediction --------------------------------------------------------
    def _present_mask(self, values: dict[str, float]) -> np.ndarray:
        return np.array([
            e in values and values[e] is not None and not (isinstance(values[e], float) and math.isnan(values[e]))
            for e in self.experts
        ])

    def predict(self, key: str, values: dict[str, float]) -> float | None:
        """Weighted combo over the experts present this round (renormalized)."""
        w = self._region(key)["w"]
        mask = self._present_mask(values)
        if not mask.any():
            return None
        wm = w * mask
        wm = wm / wm.sum()
        f = np.array([values.get(e, 0.0) if mask[i] else 0.0 for i, e in enumerate(self.experts)])
        return float((wm * f).sum())

    # ---- update ------------------------------------------------------------
    def update(self, key: str, values: dict[str, float], truth: float) -> None:
        """Pay loss for present experts and update the region's weights."""
        st = self._region(key)
        w = st["w"].copy()
        mask = self._present_mask(values)
        if not mask.any():
            return
        losses = np.zeros(self.n)
        for i, e in enumerate(self.experts):
            if mask[i]:
                losses[i] = min(1.0, abs(values[e] - truth) / self.l_max)
        # blend loss (for regret accounting), using the weights actually used
        blend = self.predict(key, values)
        if blend is not None:
            st["cum_blend"] += min(1.0, abs(blend - truth) / self.l_max)
        st["cum_loss"] += np.where(mask, losses, 0.0)
        # multiplicative-weights step on present experts only
        w[mask] *= np.exp(-self.eta * losses[mask])
        w = w / w.sum()
        # fixed-share: leak a little mass back to uniform so stale experts recover
        w = (1.0 - self.alpha) * w + self.alpha / self.n
        st["w"] = w
        st["n"] += 1

    def regret(self, key: str) -> float:
        """Blend cumulative loss minus the best single expert's -- should stay small/negative."""
        st = self._region(key)
        return float(st["cum_blend"] - st["cum_loss"].min())

    # ---- persistence -------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "experts": self.experts, "eta": self.eta, "alpha": self.alpha, "l_max": self.l_max,
            "regions": {
                k: {"w": v["w"].tolist(), "n": v["n"],
                    "cum_loss": v["cum_loss"].tolist(), "cum_blend": v["cum_blend"]}
                for k, v in self._state.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict) -> "OnlineExpertBlend":
        obj = cls(state["experts"], alpha=state["alpha"], l_max=state["l_max"], eta=state["eta"])
        for k, v in state.get("regions", {}).items():
            obj._state[k] = {
                "w": np.array(v["w"], dtype=float), "n": int(v["n"]),
                "cum_loss": np.array(v["cum_loss"], dtype=float), "cum_blend": float(v["cum_blend"]),
            }
        return obj
