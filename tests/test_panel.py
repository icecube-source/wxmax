"""Offline lifecycle test for the panel (experts + learning loop mocked)."""
from datetime import date, timedelta

import wxmax.panel.panel as pp
from wxmax.config import load_config
from wxmax.panel.panel import Panel


class _FakeCli:
    def __init__(self, v):
        self.cli_max_f = v
        self.high_time = "300 PM"


def test_panel_lifecycle_learns_to_downweight_biased_expert(tmp_path, monkeypatch):
    cfg = load_config(panel_dir=tmp_path / "panel_data", docs_dir=tmp_path / "docs")
    panel = Panel(cfg=cfg, station_ids=("KLAX", "KPHX"))

    # nbm ~truth, gfs ~truth+1, ecmwf_ifs always 8°F hot (the biased expert)
    experts = {
        "KLAX": {"nws_nbm": 72.0, "ecmwf_ifs": 80.0, "gfs": 71.0},
        "KPHX": {"nws_nbm": 109.0, "ecmwf_ifs": 117.0, "gfs": 108.0},
    }
    truth = {"KLAX": 72.0, "KPHX": 109.0}
    monkeypatch.setattr(Panel, "gather_experts", lambda self, d: experts)
    monkeypatch.setattr(pp, "fetch_cli_max", lambda sid, d: _FakeCli(truth[sid]))

    df = panel.start_of_day(date(2026, 6, 3))
    assert set(df["station"]) == {"KLAX", "KPHX"}
    assert (df["conviction"] == "ESTIMATE").all()
    # cold start: blend ~ simple average of the 3 experts
    klax_est = df[df.station == "KLAX"]["estimate"].iloc[0]
    assert abs(klax_est - (72 + 80 + 71) / 3) < 0.2
    assert "confidence" in df.columns
    assert df["confidence"].between(30, 98).all()

    # run the daily loop forward
    for i in range(25):
        d = date(2026, 6, 3) + timedelta(days=i)
        panel.start_of_day(d)
        panel.end_of_day(d)

    w = panel.blend.weights("KLAX")
    assert w["ecmwf_ifs"] < w["nws_nbm"]   # biased expert demoted
    assert w["ecmwf_ifs"] < w["gfs"]
    assert (tmp_path / "panel_data" / "weights.json").exists()
    assert (tmp_path / "panel_data" / "truth.parquet").exists()

    # a fresh Panel reloads the learned weights from disk
    reloaded = Panel(cfg=cfg, station_ids=("KLAX", "KPHX"))
    assert reloaded.blend.weights("KLAX")["ecmwf_ifs"] == w["ecmwf_ifs"]


def test_end_of_day_idempotent_and_rebuild(tmp_path, monkeypatch):
    cfg = load_config(panel_dir=tmp_path / "panel_data", docs_dir=tmp_path / "docs")
    p = Panel(cfg=cfg, station_ids=("KLAX",))
    monkeypatch.setattr(Panel, "gather_experts",
                        lambda self, d: {"KLAX": {"nws_nbm": 72.0, "ecmwf_ifs": 80.0, "gfs": 71.0}})
    monkeypatch.setattr(pp, "fetch_cli_max", lambda sid, d: _FakeCli(72.0))
    d = date(2026, 6, 10)
    p.start_of_day(d)
    p.end_of_day(d)
    w1 = dict(p.blend.weights("KLAX"))
    p.end_of_day(d)                       # re-running the same day must not double-count
    assert dict(p.blend.weights("KLAX")) == w1
    p.rebuild_weights()                   # replay history -> same single-update weights
    w3 = p.blend.weights("KLAX")
    assert all(abs(w1[k] - w3[k]) < 1e-9 for k in w1)
