"""Offline tests for ingest parsing + persistence (no network)."""
from datetime import date

from wxmax.ingest.forecasts_openmeteo import _parse_daily
from wxmax.ingest.obs_asos import iem_id
from wxmax.store import read_parquet, write_parquet


def test_iem_id_strips_k_prefix():
    assert iem_id("KLAX") == "LAX"
    assert iem_id("KMDW") == "MDW"
    # Non-K / non-4-char ids pass through unchanged.
    assert iem_id("PHNL") == "PHNL"
    assert iem_id("LAX") == "LAX"


def test_parse_daily_multi_model_suffix():
    payload = {
        "daily": {
            "time": ["2026-06-23", "2026-06-24"],
            "temperature_2m_max_gfs_seamless": [70.5, 71.5],
            "temperature_2m_max_ecmwf_ifs025": [80.3, 83.9],
        }
    }
    df = _parse_daily(payload)
    assert set(df["series"]) == {"gfs_seamless", "ecmwf_ifs025"}
    gfs = df[(df.series == "gfs_seamless") & (df.date == date(2026, 6, 23))]
    assert gfs["value"].iloc[0] == 70.5


def test_parse_daily_member_suffix_and_no_suffix():
    member = _parse_daily({"daily": {"time": ["2026-06-23"],
                                      "temperature_2m_max_member01": [88.0]}})
    assert member["series"].iloc[0] == "member01"
    plain = _parse_daily({"daily": {"time": ["2026-06-23"],
                                     "temperature_2m_max": [88.0]}})
    assert plain["series"].iloc[0] == ""


def test_parse_daily_empty():
    assert _parse_daily({}).empty


def test_store_roundtrip(tmp_path):
    df = _parse_daily({"daily": {"time": ["2026-06-23", "2026-06-24"],
                                 "temperature_2m_max_gfs_seamless": [70.5, 71.5]}})
    p = write_parquet(df, tmp_path / "x.parquet")
    back = read_parquet(p)
    assert len(back) == 2
    assert "value" in back.columns
