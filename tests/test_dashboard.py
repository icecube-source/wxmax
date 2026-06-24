"""Offline test that the static dashboard renders from the panel store."""
from datetime import date, timedelta

import wxmax.panel.panel as pp
from wxmax.config import load_config
from wxmax.panel import dashboard
from wxmax.panel.panel import Panel


class _FakeCli:
    def __init__(self, v):
        self.cli_max_f, self.high_time = v, "300 PM"


def test_dashboard_renders(tmp_path, monkeypatch):
    cfg = load_config(panel_dir=tmp_path / "panel_data", docs_dir=tmp_path / "docs")
    panel = Panel(cfg=cfg, station_ids=("KLAX", "KPHX"))
    experts = {
        "KLAX": {"nws_nbm": 72.0, "ecmwf_ifs": 80.0, "gfs": 71.0},
        "KPHX": {"nws_nbm": 110.0, "ecmwf_ifs": 117.0, "gfs": 108.0},
    }
    truth = {"KLAX": 72.0, "KPHX": 109.0}
    monkeypatch.setattr(Panel, "gather_experts", lambda self, d: experts)
    monkeypatch.setattr(pp, "fetch_cli_max", lambda sid, d: _FakeCli(truth[sid]))
    for i in range(5):
        d = date(2026, 6, 3) + timedelta(days=i)
        panel.start_of_day(d)
        panel.end_of_day(d)

    out = dashboard.render(cfg, generated_at="TEST-TS")
    html = out.read_text()
    assert out.exists() and out.name == "index.html"
    assert "Los Angeles Intl" in html and "Phoenix Sky Harbor Intl" in html
    assert "TEST-TS" in html
    assert "MAE" in html                       # accuracy section populated
    assert "expert weights" in html.lower()
    # new features:
    assert "#f7f8fa" in html                   # light theme
    assert "toggleW" in html                   # collapsible weights
    assert "Realized max temperature" in html  # realized-over-time section
    assert "June" in html                      # today's date rendered in header
    assert "Confidence" in html                # confidence column
    assert "mean absolute error" in html       # MAE caption
    assert 'http-equiv="refresh"' in html      # auto-refresh
    assert "Morning est" in html               # static morning column
    assert "Real-time best" in html            # live real-time column
