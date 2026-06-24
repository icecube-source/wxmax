"""Offline tests for CLI ground-truth parsing (no network)."""
from datetime import date

import wxmax.ingest.truth_cli as tc


def _fake_year(records):
    return lambda station, year: records


def test_fetch_cli_max_finds_date(monkeypatch):
    recs = [
        {"valid": "2026-06-21", "high": 70, "high_time": "200 PM", "wfo": "LOX",
         "product": "x-CLILAX"},
        {"valid": "2026-06-22", "high": 72, "high_time": "307 PM", "wfo": "LOX",
         "product": "y-CLILAX"},
    ]
    monkeypatch.setattr(tc, "_records_for_year", _fake_year(recs))
    c = tc.fetch_cli_max("KLAX", date(2026, 6, 22))
    assert c is not None
    assert c.cli_max_f == 72.0
    assert c.high_time == "307 PM"
    assert c.wfo == "LOX"


def test_missing_date_returns_none(monkeypatch):
    monkeypatch.setattr(tc, "_records_for_year", _fake_year([{"valid": "2026-06-21", "high": 70}]))
    assert tc.fetch_cli_max("KLAX", date(2026, 6, 22)) is None


def test_null_high_returns_none(monkeypatch):
    # CLI exists but max not yet reported (high is null).
    monkeypatch.setattr(tc, "_records_for_year",
                        _fake_year([{"valid": "2026-06-22", "high": None}]))
    assert tc.fetch_cli_max("KLAX", date(2026, 6, 22)) is None
