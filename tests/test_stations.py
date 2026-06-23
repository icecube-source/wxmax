"""Tests for the station registry and arbitrary-point snapping."""
from pathlib import Path

import pytest

from wxmax.config import load_config
from wxmax.stations import haversine_km, load_stations, nearest_station

STATIONS_PATH = Path(__file__).resolve().parent.parent / "config" / "stations.yaml"


@pytest.fixture(scope="module")
def stations():
    return load_stations(STATIONS_PATH)


def test_registry_loads_and_has_anchors(stations):
    assert "KLAX" in stations and "KMDW" in stations
    klax = stations["KLAX"]
    assert klax.tz == "America/Los_Angeles"
    assert klax.coastal is True
    assert 30 < klax.lat < 35


def test_no_duplicate_ids(stations):
    # load_stations raises on dupes; reaching here with >10 stations is enough.
    assert len(stations) >= 12


def test_haversine_known_distance():
    # KLAX -> KJFK is ~3970 km.
    d = haversine_km(33.9425, -118.4081, 40.6413, -73.7781)
    assert 3900 < d < 4050


def test_nearest_station_snaps_downtown_la_to_klax(stations):
    # Downtown LA (34.05, -118.24) — nearest panel station should be KLAX.
    st, dist = nearest_station(34.0522, -118.2437, stations)
    assert st.id == "KLAX"
    assert dist < 30  # km


def test_config_paths_resolve():
    cfg = load_config()
    assert cfg.stations_path.exists()
    assert cfg.obs_dir.name == "obs"
    assert 0.5 in cfg.quantiles
