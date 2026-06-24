"""Tests for the alerter + watcher lock-detection (offline)."""
import pandas as pd

from wxmax.config import load_config
from wxmax.panel import notify
from wxmax.panel.watch import load_alerted, new_locks, save_alerted


def _high(station, est):
    return {"date": "2026-06-24", "station": station, "conviction": "HIGH",
            "estimate": est, "lo": est - 0.5, "hi": est + 0.5}


def test_new_locks_only_returns_unalerted_high():
    df = pd.DataFrame([_high("KPHX", 110), _high("KLAX", 72)])
    locks = new_locks(df, alerted={("2026-06-24", "KLAX")})
    assert len(locks) == 1 and locks[0]["station"] == "KPHX"
    # already-alerted + empty inputs
    assert new_locks(df, alerted={("2026-06-24", "KPHX"), ("2026-06-24", "KLAX")}) == []
    assert new_locks(pd.DataFrame(), set()) == []


def test_send_alert_dispatches_and_falls_back(monkeypatch, capsys):
    notify.send_alert("Subj", "Body", backend="log")
    out = capsys.readouterr().out
    assert "Subj" in out and "Body" in out

    def boom(s, b):
        raise RuntimeError("smtp down")
    monkeypatch.setitem(notify.BACKENDS, "email", boom)
    notify.send_alert("S2", "B2", backend="email")
    out = capsys.readouterr().out
    assert "ALERT-FALLBACK" in out and "[ALERT]" in out   # fell back to log


def test_alerted_roundtrip(tmp_path):
    cfg = load_config(panel_dir=tmp_path / "panel_data", docs_dir=tmp_path / "docs")
    cfg.panel_dir.mkdir(parents=True, exist_ok=True)
    s = {("2026-06-24", "KPHX"), ("2026-06-24", "KLAX")}
    save_alerted(cfg, s)
    assert load_alerted(cfg) == s
