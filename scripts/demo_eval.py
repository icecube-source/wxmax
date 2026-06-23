"""Reproducible demo: baseline bar + calibrated-ensemble evaluation.

Usage:
    python scripts/demo_eval.py

Hits live Open-Meteo (forecasts/ERA5) + IEM ASOS (truth). Free tier is
non-commercial; set OPENMETEO_API_KEY for the commercial endpoint.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from wxmax.config import load_config
from wxmax.verify.backtest import run_baseline_backtest
from wxmax.verify.evaluate import run_calibrated_eval

pd.set_option("display.width", 140)

STATIONS = ["KLAX", "KMDW", "KPHX", "KDEN", "KMIA", "KBOS"]


def main() -> None:
    cfg = load_config(openmeteo_api_key=os.environ.get("OPENMETEO_API_KEY"))

    print("### 1) Single-model baseline bar (summer 2025) ###")
    base = run_baseline_backtest(STATIONS, date(2025, 7, 1), date(2025, 8, 31), cfg=cfg)
    print(base["overall"].round(2).to_string(index=False))

    print("\n### 2) Calibrated ensemble (16-month chronological split) ###")
    r = run_calibrated_eval(STATIONS, date(2024, 6, 1), date(2025, 9, 30), cfg=cfg)
    print(f"blended models : {r['model_cols']}")
    print(f"best single ({r['best_single_model']}): MAE={r['best_single_mae']:.2f} °F")
    print(f"calibrated blend          : MAE={r['blend_mae']:.2f}  "
          f"RMSE={r['blend_rmse']:.2f}  CRPS={r['blend_crps']:.2f} °F")
    print("\ninterval calibration (coverage should match level):")
    print(r["intervals"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
