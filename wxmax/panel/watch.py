"""Continuous watcher: refresh every N minutes, alert the moment a city locks.

Loop each tick:
  - first tick of a new (UTC) day -> start_of_day (morning estimates)
  - every tick -> hourly_poll; any city that NEWLY flips to HIGH conviction
    (hottest part of the day has passed -> max locked) fires an immediate alert
  - after ~14 UTC -> end_of_day(yesterday): record CLI truth + update weights
  - regenerate the dashboard (with an auto-refresh meta tag)

`alerted.json` (per (date, station)) makes alerts fire once per city per day and
survive restarts. Optionally serves the dashboard over HTTP (--serve PORT).

Run:
  WXMAX_ALERT=desktop python -m wxmax.panel.watch --serve 8000          # on your Mac
  WXMAX_ALERT=ntfy WXMAX_NTFY_TOPIC=my-wxmax python -m wxmax.panel.watch # on a VPS
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from ..config import load_config
from ..stations import load_stations
from . import dashboard, notify
from .panel import Panel


def _alerted_path(cfg):
    return cfg.panel_dir / "alerted.json"


def load_alerted(cfg) -> set[tuple[str, str]]:
    p = _alerted_path(cfg)
    return {tuple(x) for x in json.loads(p.read_text())} if p.exists() else set()


def save_alerted(cfg, alerted: set) -> None:
    _alerted_path(cfg).write_text(json.dumps(sorted(list(k) for k in alerted)))


def new_locks(high_df, alerted: set) -> list[dict]:
    """HIGH-conviction rows not yet alerted, as dicts."""
    if high_df is None or len(high_df) == 0:
        return []
    out = []
    for _, r in high_df.iterrows():
        if r.get("conviction") == "HIGH" and (r["date"], r["station"]) not in alerted:
            out.append(r.to_dict())
    return out


def _serve(cfg, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(cfg.docs_dir))
    srv = ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"dashboard served at http://localhost:{port}/  (auto-refreshes)", flush=True)


def tick(panel: Panel, names: dict, alerted: set, interval: int, alert_backend, state: dict) -> None:
    now = datetime.now(timezone.utc)
    today = now.date()
    if state.get("day") != today:
        state.update(day=today, started=False, ended=False)
    if not state["started"]:
        panel.start_of_day(today)
        state["started"] = True
        print(f"[{now:%Y-%m-%d %H:%MZ}] start_of_day {today}", flush=True)

    hp = panel.hourly_poll(today)
    for r in new_locks(hp, alerted):
        name = names.get(r["station"], r["station"])
        subj = f"\U0001f525 Max LOCKED: {name} {r['estimate']:.0f}°F"
        body = (f"{name} ({r['station']}) — {r['date']}\n"
                f"Daily max locked at {r['estimate']:.0f}°F: the hottest part of the day "
                f"has passed (high conviction).\nInterval [{r['lo']:.0f}, {r['hi']:.0f}] °F.")
        notify.send_alert(subj, body, backend=alert_backend)
        alerted.add((r["date"], r["station"]))
        save_alerted(panel.cfg, alerted)
        print(f"  ALERT {r['station']} {r['estimate']:.0f}F", flush=True)

    if now.hour >= 14 and not state["ended"]:
        panel.end_of_day(today - timedelta(days=1))
        state["ended"] = True
        print(f"[{now:%H:%MZ}] end_of_day {today - timedelta(days=1)} (CLI truth + weights)", flush=True)

    dashboard.render(panel.cfg, refresh_seconds=interval)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Continuous wxmax panel watcher + alerter.")
    ap.add_argument("--interval", type=int, default=900, help="seconds between ticks (default 900 = 15 min)")
    ap.add_argument("--serve", type=int, default=0, help="serve dashboard on this port (0 = off)")
    ap.add_argument("--alert", default=None, help="override WXMAX_ALERT backend (log/desktop/email/ntfy/slack)")
    ap.add_argument("--once", action="store_true", help="run a single tick and exit (for testing)")
    a = ap.parse_args(argv)

    cfg = load_config()
    names = {sid: st.name for sid, st in load_stations(cfg.stations_path).items()}
    panel = Panel(cfg=cfg)
    alerted = load_alerted(cfg)
    if a.serve:
        _serve(cfg, a.serve)

    state: dict = {}
    print(f"watcher up: every {a.interval}s, alert backend = {a.alert or 'env/log'}", flush=True)
    while True:
        try:
            tick(panel, names, alerted, a.interval, a.alert, state)
        except Exception as e:  # a transient fetch error shouldn't kill the loop
            print(f"[tick-error] {type(e).__name__}: {e}", flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
