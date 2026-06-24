"""Real-time NWS active alerts (free, public domain) -> heat-regime flag.

Hits api.weather.gov/alerts/active?point=lat,lon and detects heat-related events
(Excessive/Extreme Heat Warning or Watch, Heat Advisory). When one is active we
are in an extreme-tail regime where the models systematically under-predict the
peak, so the panel widens the upper tail, nudges the point up, and caps the
confidence (see panel.apply_heat_regime).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..stations import Station
from ._http import get_json

NWS_ALERTS = "https://api.weather.gov/alerts/active"
HEAT_KEYWORDS = ("excessive heat", "extreme heat", "heat advisory")


@dataclass(frozen=True)
class HeatAlert:
    event: str
    severity: str | None
    onset: str | None
    ends: str | None


def fetch_active_heat(station: Station) -> HeatAlert | None:
    """Return the active heat alert for a station's point, or None."""
    data = get_json(NWS_ALERTS, params={"point": f"{station.lat},{station.lon}"})
    for f in data.get("features", []):
        p = f.get("properties", {})
        event = p.get("event") or ""
        if any(k in event.lower() for k in HEAT_KEYWORDS):
            return HeatAlert(
                event=event,
                severity=p.get("severity"),
                onset=p.get("onset"),
                ends=p.get("ends") or p.get("expires"),
            )
    return None
