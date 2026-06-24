"""Ground truth: the NWS Climatological Report (CLI / F-6) official daily max.

This is the authoritative number the panel is scored against -- whole degrees F,
issued by the local WFO from quality-controlled data (so it can differ from raw
ASOS, notably at Central Park / KNYC). We read it station-keyed from the Iowa
Environmental Mesonet JSON endpoint, which exposes the parsed CLI `high` value
directly (no text parsing). The NWS api.weather.gov CLI text product is the
fallback but is keyed by WFO (which issues CLI for several sites), so IEM's
station-keyed feed is preferred.

Issuance: the *final* CLI posts ~01:30 LST the following morning; a same-day
preliminary may exist (esp. KNYC). Fetch end-of-day with retries; treat a
missing `high` as "not yet reported".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ._http import get_json

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py"


@dataclass(frozen=True)
class CliMax:
    station: str
    date: date
    cli_max_f: float        # official daily max, whole degrees F
    high_time: str | None   # local time of occurrence, e.g. "244 PM"
    wfo: str | None
    product: str | None     # CLI product id, e.g. ...-CLILAX


def _records_for_year(station: str, year: int) -> list[dict]:
    """IEM returns the station's full-year list of CLI daily records."""
    data = get_json(IEM_CLI_URL, params={"station": station, "year": year, "fmt": "json"})
    if isinstance(data, list):
        return data
    # tolerate {"results": [...]} or {"data": [...]} shapes
    return data.get("results") or data.get("data") or []


def fetch_cli_max(station: str, d: date) -> CliMax | None:
    """Official CLI daily max for one station/date, or None if not yet reported."""
    for rec in _records_for_year(station, d.year):
        if rec.get("valid") == d.isoformat():
            high = rec.get("high")
            if high is None:
                return None
            return CliMax(
                station=station,
                date=d,
                cli_max_f=float(high),
                high_time=rec.get("high_time"),
                wfo=rec.get("wfo"),
                product=rec.get("product"),
            )
    return None
