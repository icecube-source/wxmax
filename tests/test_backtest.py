"""Offline tests for the baseline back-test scoring (no network)."""
from datetime import date

import pandas as pd

from wxmax.verify.backtest import join_forecast_truth, score_by_model


def _joined():
    # Two models, two stations, two days. modelA is perfect; modelB is +5 hot.
    model_long = pd.DataFrame([
        ("KLAX", date(2026, 6, 23), "A", 80.0),
        ("KLAX", date(2026, 6, 24), "A", 82.0),
        ("KLAX", date(2026, 6, 23), "B", 85.0),
        ("KLAX", date(2026, 6, 24), "B", 87.0),
        ("KMDW", date(2026, 6, 23), "A", 70.0),
        ("KMDW", date(2026, 6, 23), "B", 75.0),
    ], columns=["station", "date", "model", "tmax"])
    truth_long = pd.DataFrame([
        ("KLAX", date(2026, 6, 23), 80.0),
        ("KLAX", date(2026, 6, 24), 82.0),
        ("KMDW", date(2026, 6, 23), 70.0),
    ], columns=["station", "date", "tmax_f"])
    return join_forecast_truth(model_long, truth_long)


def test_join_inner_keeps_matching_rows():
    j = _joined()
    # 6 model rows but truth only covers 3 (station,date) pairs -> 6 joined
    # (2 models x 3 pairs).
    assert len(j) == 6
    assert set(j.columns) >= {"station", "date", "model", "tmax", "truth"}


def test_score_by_model_orders_by_mae():
    overall = score_by_model(_joined())
    assert list(overall["model"]) == ["A", "B"]  # A perfect -> lower MAE first
    a = overall[overall.model == "A"].iloc[0]
    b = overall[overall.model == "B"].iloc[0]
    assert a["mae"] == 0.0
    assert b["mae"] == 5.0 and b["bias"] == 5.0


def test_score_by_station_grouping():
    by_st = score_by_model(_joined(), group_extra=["station"])
    assert set(by_st["station"]) == {"KLAX", "KMDW"}
    assert {"station", "model", "mae", "n"} <= set(by_st.columns)
