"""Confidence calibration: fit the predictive-dispersion multiplier lambda.

The raw predictive variance can be over/under-confident (notably because the
experts are correlated). We fit one multiplier lambda so the standardized
forecast errors z=(truth-mu_F)/(lambda*sigma_F) have unit variance (PIT
dispersion fix). It activates once enough (forecast, truth) pairs accumulate;
until then lambda defaults to 1.0. (A full per-phase isotonic reliability map is
the next refinement once more history exists.)
"""
from __future__ import annotations

import json

import numpy as np

from .. import store
from ..config import Config, load_config

MIN_PAIRS = 20


def fit_lambda(mu, sigma, truth) -> float:
    mu, sigma, truth = np.asarray(mu, float), np.asarray(sigma, float), np.asarray(truth, float)
    ok = sigma > 1e-6
    z = (truth[ok] - mu[ok]) / sigma[ok]
    return float(np.sqrt(np.mean(z ** 2))) if z.size >= MIN_PAIRS else 1.0


def load_conf_lambda(cfg: Config) -> float:
    p = cfg.panel_dir / "climatology" / "conf_calib.json"
    if p.exists():
        try:
            return float(json.loads(p.read_text()).get("lambda", cfg.conf_lambda))
        except Exception:
            pass
    return cfg.conf_lambda


def calibrate_all(cfg: Config | None = None) -> float:
    cfg = cfg or load_config()
    est_p, truth_p = cfg.panel_dir / "estimates.parquet", cfg.panel_dir / "truth.parquet"
    if not (est_p.exists() and truth_p.exists()):
        print("conf calibration: insufficient data")
        return cfg.conf_lambda
    est = store.read_parquet(est_p)
    truth = store.read_parquet(truth_p)
    cols = [c for c in ("date", "station", "fc_mu", "fc_sigma") if c in est.columns]
    if "fc_mu" not in cols:
        print("conf calibration: no forecast-distribution history yet")
        return cfg.conf_lambda
    e = est[est.conviction == "ESTIMATE"][cols].dropna()
    t = truth.dropna(subset=["cli_max_f"])[["date", "station", "cli_max_f"]]
    m = e.merge(t, on=["date", "station"])
    lam = fit_lambda(m["fc_mu"], m["fc_sigma"], m["cli_max_f"]) if len(m) else cfg.conf_lambda
    out = cfg.panel_dir / "climatology" / "conf_calib.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lambda": round(lam, 3), "n": int(len(m))}))
    print(f"conf lambda = {lam:.3f} from n={len(m)} (forecast, truth) pairs -> {out}")
    return lam
