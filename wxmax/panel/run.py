"""CLI entry the scheduler invokes. Time-gated phases; idempotent per station/day.

  python -m wxmax.panel.run --phases start,poll,end

Phases:
  start  -- start_of_day(today): gather experts -> blended ESTIMATE per city
  poll   -- hourly_poll(today): flip cities past their peak to HIGH conviction
  end    -- end_of_day(yesterday): record official CLI max + update expert weights

`today` defaults to the current UTC date (override with --date). `end` always
processes the day before `today`, by which time all CLI reports have posted.
After the phases run, the static dashboard is regenerated unless --no-dashboard.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

from ..config import load_config
from . import dashboard
from .panel import Panel


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the wxmax daily panel.")
    ap.add_argument("--phases", default="start,poll,end",
                    help="comma list of: start,poll,end")
    ap.add_argument("--date", help="reference 'today' as YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--no-dashboard", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config()
    today = date.fromisoformat(a.date) if a.date else datetime.now(timezone.utc).date()
    phases = [p.strip() for p in a.phases.split(",") if p.strip()]
    panel = Panel(cfg=cfg)

    if "start" in phases:
        print(f"== start_of_day {today} ==")
        print(panel.start_of_day(today).to_string(index=False))
    if "poll" in phases:
        print(f"== hourly_poll {today} ==")
        hp = panel.hourly_poll(today)
        print(hp.to_string(index=False) if len(hp) else "(no stations past peak)")
    if "end" in phases:
        y = today - timedelta(days=1)
        print(f"== end_of_day {y} (record CLI truth + update weights) ==")
        print(panel.end_of_day(y).to_string(index=False))

    if not a.no_dashboard:
        out = dashboard.render(cfg)
        print(f"dashboard -> {out}")


if __name__ == "__main__":
    main()
