"""End-to-end calibrated-ensemble evaluation (P6).

Chronological split: fit MOS+EMOS on TRAIN, fit conformal on CALIB, evaluate on
TEST. Reports the headline numbers:
  - point accuracy of the blend vs the best single model (must beat the bar),
  - CRPS of the predictive Gaussian,
  - interval coverage vs nominal + width, raw EMOS vs conformal-calibrated.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..calibrate.conformal import ConformalInterval, gaussian_interval
from ..config import Config
from ..features.build_features import assemble_training_frame, design_matrix
from ..models.emos import GaussianEMOS
from .metrics import crps_gaussian, interval_coverage, interval_width, mae, rmse


def chronological_split(df: pd.DataFrame, frac_train=0.6, frac_calib=0.2):
    df = df.sort_values(["date", "station"]).reset_index(drop=True)
    n = len(df)
    a, b = int(n * frac_train), int(n * (frac_train + frac_calib))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def run_calibrated_eval(
    station_ids: list[str], start: date, end: date,
    cfg: Config | None = None, levels=(0.80, 0.90),
) -> dict:
    df = assemble_training_frame(station_ids, start, end, cfg=cfg)
    cols = df.attrs["model_cols"]
    tr, ca, te = chronological_split(df)

    Ftr, ytr, sptr = design_matrix(tr, cols)
    emos = GaussianEMOS().fit(Ftr, ytr, sptr)

    mu_ca, sig_ca = emos.predict(*design_matrix(ca, cols)[::2])  # F, spread
    Fte, yte, spte = design_matrix(te, cols)
    mu_te, sig_te = emos.predict(Fte, spte)
    yca = ca["truth"].to_numpy(float)

    # point: blend vs best single model (on TEST)
    per_model_mae = {c: mae(yte, te[c].to_numpy(float)) for c in cols}
    best_model = min(per_model_mae, key=per_model_mae.get)

    interval_rows = []
    for lv in levels:
        lo_ca, hi_ca = gaussian_interval(mu_ca, sig_ca, lv)
        conf = ConformalInterval(alpha=1 - lv).fit(lo_ca, hi_ca, yca)
        lo_te, hi_te = gaussian_interval(mu_te, sig_te, lv)
        clo, chi = conf.transform(lo_te, hi_te)
        interval_rows.append(dict(
            level=lv,
            raw_coverage=interval_coverage(yte, lo_te, hi_te),
            raw_width=interval_width(lo_te, hi_te),
            cqr_coverage=interval_coverage(yte, clo, chi),
            cqr_width=interval_width(clo, chi),
        ))

    return {
        "model_cols": cols,
        "n_train": len(tr), "n_calib": len(ca), "n_test": len(te),
        "best_single_model": best_model,
        "best_single_mae": per_model_mae[best_model],
        "blend_mae": mae(yte, mu_te),
        "blend_rmse": rmse(yte, mu_te),
        "blend_crps": crps_gaussian(yte, mu_te, sig_te),
        "intervals": pd.DataFrame(interval_rows),
    }
