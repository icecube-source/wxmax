"""Station registry + arbitrary-point snapping.

Named stations are the training/verification backbone; arbitrary lat/lon are
served by snapping to the nearest station (the "Both" requirement). Verification
is only rigorous at stations, so `nearest_station` carries the distance so
callers can flag low-confidence snaps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    tz: str
    region: str
    coastal: bool


def load_stations(path: str | Path) -> dict[str, Station]:
    """Load the station registry keyed by station id."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    out: dict[str, Station] = {}
    for rec in raw["stations"]:
        st = Station(
            id=rec["id"],
            name=rec["name"],
            lat=float(rec["lat"]),
            lon=float(rec["lon"]),
            elevation_m=float(rec["elevation_m"]),
            tz=rec["tz"],
            region=rec["region"],
            coastal=bool(rec.get("coastal", False)),
        )
        if st.id in out:
            raise ValueError(f"duplicate station id: {st.id}")
        out[st.id] = st
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_station(
    lat: float, lon: float, stations: dict[str, Station]
) -> tuple[Station, float]:
    """Nearest station to an arbitrary point and its distance in km."""
    if not stations:
        raise ValueError("empty station registry")
    best: tuple[Station, float] | None = None
    for st in stations.values():
        d = haversine_km(lat, lon, st.lat, st.lon)
        if best is None or d < best[1]:
            best = (st, d)
    assert best is not None
    return best
