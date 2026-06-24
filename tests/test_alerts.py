"""Tests for NWS heat-alert ingestion + Tier-1 heat-regime adjustment."""
from datetime import date

import wxmax.ingest.alerts_nws as al
from wxmax.config import load_config
from wxmax.ingest.alerts_nws import HeatAlert, fetch_active_heat
from wxmax.panel.panel import Panel, apply_heat_regime
from wxmax.stations import Station


def _st():
    return Station("KDCA", "DC", 38.85, -77.04, 5.0, "America/New_York", "mid_atlantic", False)


def test_fetch_active_heat_detects(monkeypatch):
    payload = {"features": [
        {"properties": {"event": "Flood Watch"}},
        {"properties": {"event": "Excessive Heat Warning", "severity": "Severe",
                        "onset": "a", "ends": "b"}},
    ]}
    monkeypatch.setattr(al, "get_json", lambda url, params=None: payload)
    h = fetch_active_heat(_st())
    assert h is not None and h.event == "Excessive Heat Warning" and h.severity == "Severe"


def test_fetch_active_heat_none_when_no_heat(monkeypatch):
    monkeypatch.setattr(al, "get_json",
                        lambda url, params=None: {"features": [{"properties": {"event": "Flood Warning"}}]})
    assert fetch_active_heat(_st()) is None
    monkeypatch.setattr(al, "get_json", lambda url, params=None: {"features": []})
    assert fetch_active_heat(_st()) is None


def test_apply_heat_regime_skews_up_and_caps():
    point, lo, hi, conf = apply_heat_regime(95.0, half=2.0, conf=88)
    assert point == 96.5                  # +1.5°F upward bias
    assert (hi - point) > (point - lo)    # asymmetric: fatter upper tail
    assert conf == 70                     # capped


def test_start_of_day_applies_heat_regime(tmp_path, monkeypatch):
    cfg = load_config(panel_dir=tmp_path / "panel_data", docs_dir=tmp_path / "docs")
    p = Panel(cfg=cfg, station_ids=("KDCA", "KLAX"))
    monkeypatch.setattr(Panel, "gather_experts",
                        lambda self, d: {"KDCA": {"nws_nbm": 99.0, "gfs": 98.0},
                                         "KLAX": {"nws_nbm": 75.0, "gfs": 74.0}})
    monkeypatch.setattr(p, "_gather_heat",
                        lambda: {"KDCA": HeatAlert("Excessive Heat Warning", "Severe", None, None)})
    df = p.start_of_day(date(2026, 6, 24))
    dca = df[df.station == "KDCA"].iloc[0]
    lax = df[df.station == "KLAX"].iloc[0]
    assert dca["regime"] == "heat" and lax["regime"] == "normal"
    assert dca["alert_event"] == "Excessive Heat Warning"
    assert dca["confidence"] <= 70
    assert (dca["hi"] - dca["estimate"]) > (dca["estimate"] - dca["lo"])  # upper-skewed
