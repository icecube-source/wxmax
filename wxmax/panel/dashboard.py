"""Render a self-contained static HTML dashboard from the panel's Parquet store.

Three sections:
  1. Today's panel  -- per city: current call (HIGH conviction if the peak has
     passed, else morning ESTIMATE), value, and interval.
  2. Forecast accuracy vs NWS CLI -- MAE of the morning ESTIMATE against the
     official daily max, per station and overall (the real forecast skill;
     HIGH-conviction values are excluded since they're obs-anchored).
  3. Per-region expert weights -- the online learner's current source mix.

No external assets; writes cfg.docs_dir/index.html (servable via GitHub Pages).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import store
from ..config import PANEL_STATIONS, Config, load_config
from ..stations import load_stations

EXPERT_COLORS = {
    "nws_nbm": "#4c78a8", "ecmwf_ifs": "#f58518",
    "ecmwf_aifs": "#54a24b", "gfs": "#e45756",
}


def _read(path: Path) -> pd.DataFrame:
    return store.read_parquet(path) if path.exists() else pd.DataFrame()


def _today_panel_rows(est: pd.DataFrame, names: dict) -> str:
    if est.empty:
        return "<tr><td colspan='4'>no panel data yet</td></tr>"
    day = est["date"].max()
    rows = ""
    for sid in [s for s in names]:
        sub = est[(est.date == day) & (est.station == sid)]
        if sub.empty:
            continue
        pick = sub[sub.conviction == "HIGH"]
        pick = pick.iloc[0] if len(pick) else sub.iloc[0]
        high_conv = pick["conviction"] == "HIGH"
        badge = (f"<span class='badge {'high' if high_conv else 'est'}'>"
                 f"{'HIGH CONVICTION' if high_conv else 'estimate'}</span>")
        val = pick.get("estimate")
        rng = (f"[{pick['lo']:.0f}, {pick['hi']:.0f}]"
               if pd.notna(pick.get("lo")) else "")
        rows += (f"<tr><td>{names[sid]}</td><td class='num big'>"
                 f"{val:.0f}&deg;F</td><td class='num'>{rng}</td><td>{badge}</td></tr>")
    return rows or "<tr><td colspan='4'>no rows for latest day</td></tr>"


def _accuracy_rows(est: pd.DataFrame, truth: pd.DataFrame, names: dict) -> str:
    if est.empty or truth.empty:
        return "<tr><td colspan='3'>accuracy populates once CLI truth is recorded</td></tr>"
    fc = est[est.conviction == "ESTIMATE"][["date", "station", "estimate"]]
    t = truth.dropna(subset=["cli_max_f"])[["date", "station", "cli_max_f"]]
    m = fc.merge(t, on=["date", "station"])
    if m.empty:
        return "<tr><td colspan='3'>accuracy populates once forecasts & truth overlap</td></tr>"
    m["abs_err"] = (m["estimate"] - m["cli_max_f"]).abs()
    g = m.groupby("station")["abs_err"].agg(["mean", "count"]).sort_values("mean")
    rows = ""
    for sid, r in g.iterrows():
        rows += (f"<tr><td>{names.get(sid, sid)}</td><td class='num'>{r['mean']:.2f}</td>"
                 f"<td class='num'>{int(r['count'])}</td></tr>")
    rows += (f"<tr class='total'><td>ALL</td><td class='num'>{m['abs_err'].mean():.2f}</td>"
             f"<td class='num'>{len(m)}</td></tr>")
    return rows


def _weight_bars(weights: dict, names: dict) -> str:
    regions = weights.get("regions", {})
    experts = weights.get("experts", [])
    if not regions:
        return "<p class='muted'>weights start uniform and adapt as CLI truth arrives.</p>"
    out = ""
    for sid in names:
        r = regions.get(sid)
        if not r:
            continue
        segs = ""
        for e, w in zip(experts, r["w"]):
            c = EXPERT_COLORS.get(e, "#888")
            segs += (f"<div class='seg' style='width:{w*100:.1f}%;background:{c}' "
                     f"title='{e}: {w:.2f}'></div>")
        out += (f"<div class='wrow'><div class='wlabel'>{names[sid]}"
                f"<span class='muted'> &middot; {r['n']} updates</span></div>"
                f"<div class='bar'>{segs}</div></div>")
    legend = " ".join(
        f"<span class='lg'><i style='background:{EXPERT_COLORS.get(e,'#888')}'></i>{e}</span>"
        for e in experts)
    return out + f"<div class='legend'>{legend}</div>"


def render(cfg: Config | None = None, generated_at: str | None = None) -> Path:
    cfg = cfg or load_config()
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
    reg = load_stations(cfg.stations_path)
    names = {sid: reg[sid].name for sid in PANEL_STATIONS}  # ordered as in the panel
    est = _read(cfg.panel_dir / "estimates.parquet")
    truth = _read(cfg.panel_dir / "truth.parquet")
    wpath = cfg.panel_dir / "weights.json"
    weights = json.loads(wpath.read_text()) if wpath.exists() else {}
    gen = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    day = est["date"].max() if not est.empty else "—"

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wxmax daily max-temp panel</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background:#0f1115; color:#e6e6e6; }}
.wrap {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 22px; margin: 0 0 2px; }}
.sub {{ color:#9aa0aa; font-size: 13px; margin-bottom: 24px; }}
h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing:.06em; color:#9aa0aa; margin: 28px 0 8px; }}
table {{ width:100%; border-collapse: collapse; font-size: 14px; }}
td, th {{ padding: 7px 10px; border-bottom: 1px solid #232733; text-align:left; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.big {{ font-size: 17px; font-weight: 600; }}
.total td {{ font-weight: 700; border-top: 2px solid #333; }}
.badge {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight:600; }}
.badge.high {{ background:#0b3d1e; color:#3fb950; }}
.badge.est {{ background:#3a2f08; color:#d4a72c; }}
.wrow {{ margin: 6px 0; }}
.wlabel {{ font-size: 13px; margin-bottom: 3px; }}
.bar {{ display:flex; height: 14px; border-radius: 4px; overflow:hidden; background:#232733; }}
.seg {{ height:100%; }}
.muted {{ color:#9aa0aa; font-weight: 400; }}
.legend {{ margin-top: 12px; font-size: 12px; color:#9aa0aa; }}
.lg {{ margin-right: 14px; }} .lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }}
.foot {{ margin-top: 40px; font-size: 12px; color:#6b7280; }}
</style></head><body><div class="wrap">
<h1>wxmax &mdash; daily maximum temperature panel</h1>
<div class="sub">11 US cities &middot; ground truth = NWS Climatological Report (CLI) &middot;
latest day <b>{day}</b> &middot; generated {gen}</div>

<h2>Today's panel</h2>
<table><tr><th>City</th><th class="num">Max &deg;F</th><th class="num">Interval</th><th>Call</th></tr>
{_today_panel_rows(est, names)}</table>

<h2>Forecast accuracy vs NWS CLI (MAE &deg;F)</h2>
<table><tr><th>City</th><th class="num">MAE</th><th class="num">n days</th></tr>
{_accuracy_rows(est, truth, names)}</table>

<h2>Per-region expert weights (online learner)</h2>
{_weight_bars(weights, names)}

<div class="foot">Free, public-domain data only (NWS API, NOAA NODD, ECMWF Open Data, IEM).
Online source selection via Hedge + Fixed-Share. &copy; built with wxmax.</div>
</div></body></html>"""
    out = cfg.docs_dir / "index.html"
    out.write_text(html)
    return out
