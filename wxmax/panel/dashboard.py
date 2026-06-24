"""Render a self-contained static HTML dashboard from the panel's Parquet store.

Sections:
  1. Today's panel        -- per city: current call (HIGH conviction if the peak
     has passed, else morning ESTIMATE), value, interval.
  2. Realized max over time -- the recorded NWS CLI official daily max per city
     across the available history, with a sparkline trend.
  3. Forecast accuracy vs CLI -- MAE of the morning ESTIMATE per station/overall.
  4. Per-region expert weights -- the online learner's current source mix
     (collapsible).

Light themed; today's date in the header. No external assets; writes
cfg.docs_dir/index.html (servable via GitHub Pages).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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


def _conf_badge(c) -> str:
    if c is None or (isinstance(c, float) and pd.isna(c)):
        return ""
    c = int(c)
    cls = "high" if c >= 80 else ("mid" if c >= 60 else "low")
    return f"<span class='conf {cls}'>{c}%</span>"


def _load_clim_stats(cfg: Config) -> dict:
    """Per-station climatological peak-hour median/P90 from the cached JSONs."""
    out, d = {}, cfg.panel_dir / "climatology"
    for sid in PANEL_STATIONS:
        p = d / f"{sid}.json"
        if not p.exists():
            continue
        try:
            j = json.loads(p.read_text())
            arr = np.array((j.get("recent") or []) + (j.get("baseline") or []), dtype=float)
            if len(arr):
                out[sid] = {"median": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}
        except Exception:
            pass
    return out


def _fmt_date(daystr: str) -> str:
    try:
        dt = datetime.strptime(daystr, "%Y-%m-%d")
        return f"{dt:%A, %B} {dt.day}, {dt.year}"
    except Exception:
        return daystr


def _sparkline(vals: list[float], w: int = 90, h: int = 22) -> str:
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pts = " ".join(f"{i/(n-1)*w:.1f},{h-2-(v-lo)/rng*(h-4):.1f}" for i, v in enumerate(vals))
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<polyline points='{pts}' fill='none' stroke='#d9480f' stroke-width='1.5'/></svg>")


def _val_cell(row) -> str:
    if row is None or pd.isna(row.get("estimate")):
        return "&mdash;"
    rng = (f" <span class='rng'>[{row['lo']:.0f}, {row['hi']:.0f}]</span>"
           if pd.notna(row.get("lo")) else "")
    return f"<b>{row['estimate']:.0f}&deg;F</b>{rng}"


def _today_panel_rows(est: pd.DataFrame, names: dict, clim_stats: dict) -> str:
    if est.empty:
        return "<tr><td colspan='6'>no panel data yet</td></tr>"
    day = est["date"].max()
    rows = ""
    for sid in names:
        sub = est[(est.date == day) & (est.station == sid)]
        if sub.empty:
            continue
        m = sub[sub.conviction == "ESTIMATE"]              # static morning estimate
        m = m.iloc[0] if len(m) else None
        intr = sub[sub.conviction.isin(["HIGH", "TRACKING"])]  # live real-time best
        if len(intr):
            intr = intr.assign(_o=intr["conviction"].map({"HIGH": 0, "TRACKING": 1})).sort_values("_o")
            rt = intr.iloc[0]
        else:
            rt = None
        if rt is not None:
            locked = rt["conviction"] == "HIGH"
            rt_cell = _val_cell(rt) + (" <span class='lock'>&#128274;</span>" if locked else "")
            conf = _conf_badge(rt.get("confidence"))
            label = ("high", "LOCKED") if locked else ("track", "tracking")
        else:
            rt_cell = "&mdash;"
            conf = _conf_badge(m.get("confidence")) if m is not None else ""
            label = ("est", "estimate")
        badge = f"<span class='badge {label[0]}'>{label[1]}</span>"
        is_heat = bool("regime" in sub.columns and (sub["regime"] == "heat").any())
        heat = " <span class='heat' title='NWS heat alert active'>&#9888; HEAT</span>" if is_heat else ""
        cs = clim_stats.get(sid)
        peak = f"{cs['median']:.0f}&ndash;{cs['p90']:.0f}h" if cs else "&mdash;"
        rows += (f"<tr><td>{names[sid]}{heat}</td><td>{_val_cell(m)}</td><td>{rt_cell}</td>"
                 f"<td>{conf}</td><td class='num'>{peak}</td><td>{badge}</td></tr>")
    return rows or "<tr><td colspan='6'>no rows for latest day</td></tr>"


def _realized_section(truth: pd.DataFrame, names: dict) -> str:
    t = truth.dropna(subset=["cli_max_f"]) if not truth.empty else truth
    if t.empty:
        return "<p class='muted'>realized maxima populate as the official CLI reports are recorded.</p>"
    piv = t.pivot_table(index="station", columns="date", values="cli_max_f", aggfunc="last")
    dates = sorted(piv.columns)[-10:]  # most recent ~10 days
    head = "".join(f"<th class='num'>{d[5:]}</th>" for d in dates)  # MM-DD
    body = ""
    for sid in names:
        if sid not in piv.index:
            continue
        vals = [piv.loc[sid, d] if d in piv.columns else float("nan") for d in dates]
        present = [v for v in vals if pd.notna(v)]
        cells = "".join(
            f"<td class='num'>{v:.0f}</td>" if pd.notna(v) else "<td class='num muted'>&middot;</td>"
            for v in vals)
        body += f"<tr><td>{names[sid]}</td><td>{_sparkline(present)}</td>{cells}</tr>"
    return (f"<table><tr><th>City</th><th>Trend</th>{head}</tr>{body}</table>")


def _accuracy_rows(est: pd.DataFrame, truth: pd.DataFrame, names: dict) -> str:
    if est.empty or truth.empty:
        return "<tr><td colspan='3'>accuracy populates once CLI truth is recorded</td></tr>"
    fc = est[est.conviction == "ESTIMATE"][["date", "station", "estimate"]]
    t = truth.dropna(subset=["cli_max_f"])[["date", "station", "cli_max_f"]]
    m = fc.merge(t, on=["date", "station"])
    if m.empty:
        return "<tr><td colspan='3'>accuracy populates once forecasts &amp; truth overlap</td></tr>"
    m["abs_err"] = (m["estimate"] - m["cli_max_f"]).abs()
    g = m.groupby("station")["abs_err"].agg(["mean", "count"]).sort_values("mean")
    rows = "".join(
        f"<tr><td>{names.get(sid, sid)}</td><td class='num'>{r['mean']:.2f}</td>"
        f"<td class='num'>{int(r['count'])}</td></tr>" for sid, r in g.iterrows())
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
        r = regions.get(sid)                       # {expert: {b, s2, n}}
        if not r:
            continue
        inv = {e: 1.0 / max(r[e]["s2"], 0.25) for e in experts if e in r}
        z = sum(inv.values()) or 1.0
        w = {e: inv[e] / z for e in inv}
        nupd = int(max((r[e]["n"] for e in r), default=0))
        segs = ""
        for e in experts:
            if e not in r:
                continue
            we, be, sde = w.get(e, 0.0), r[e]["b"], r[e]["s2"] ** 0.5
            segs += (f"<div class='seg' style='width:{we*100:.1f}%;background:{EXPERT_COLORS.get(e,'#888')}' "
                     f"title='{e}: w={we:.2f}, bias {be:+.1f}F, sd {sde:.1f}F'></div>")
        out += (f"<div class='wrow'><div class='wlabel'>{names[sid]}"
                f"<span class='muted'> &middot; {nupd} updates</span></div>"
                f"<div class='bar'>{segs}</div></div>")
    legend = " ".join(
        f"<span class='lg'><i style='background:{EXPERT_COLORS.get(e,'#888')}'></i>{e}</span>"
        for e in experts)
    return out + f"<div class='legend'>{legend}</div>"


def render(cfg: Config | None = None, generated_at: str | None = None,
           refresh_seconds: int = 900) -> Path:
    cfg = cfg or load_config()
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
    reg = load_stations(cfg.stations_path)
    names = {sid: reg[sid].name for sid in PANEL_STATIONS}  # ordered as in the panel
    est = _read(cfg.panel_dir / "estimates.parquet")
    truth = _read(cfg.panel_dir / "truth.parquet")
    wpath = cfg.panel_dir / "weights.json"
    weights = json.loads(wpath.read_text()) if wpath.exists() else {}
    clim_stats = _load_clim_stats(cfg)
    gen = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    day = est["date"].max() if not est.empty else None
    date_str = _fmt_date(day) if day else "—"

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh_seconds}">
<title>wxmax daily max-temp panel</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background:#f7f8fa; color:#1a1d21; }}
.wrap {{ max-width: 880px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 22px; margin: 0 0 2px; }}
.today {{ font-size: 15px; font-weight: 600; color:#0b3d91; margin: 2px 0; }}
.sub {{ color:#6b7280; font-size: 13px; margin-bottom: 20px; }}
h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing:.06em; color:#6b7280;
     margin: 30px 0 8px; display:flex; align-items:center; gap:10px; }}
table {{ width:100%; border-collapse: collapse; font-size: 14px; background:#fff;
        border:1px solid #e3e6ea; border-radius:8px; overflow:hidden; }}
td, th {{ padding: 7px 10px; border-bottom: 1px solid #eef0f3; text-align:left; }}
th {{ background:#f1f3f6; color:#444; font-weight:600; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.big {{ font-size: 17px; font-weight: 700; }}
.total td {{ font-weight: 700; background:#f1f3f6; }}
.badge {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight:600; }}
.badge.high {{ background:#e4f5e9; color:#1a7f37; }}
.badge.est {{ background:#fdf3d8; color:#9a6700; }}
.badge.track {{ background:#eef2ff; color:#3949ab; }}
.rng {{ color:#6b7280; font-size:12px; font-weight:400; }}
.lock {{ color:#1a7f37; }}
.conf {{ font-size:12px; font-weight:700; padding:2px 8px; border-radius:10px; }}
.conf.high {{ background:#e4f5e9; color:#1a7f37; }}
.conf.mid {{ background:#fdf3d8; color:#9a6700; }}
.conf.low {{ background:#fde8e8; color:#b42318; }}
.heat {{ font-size:11px; font-weight:700; padding:2px 6px; border-radius:8px; background:#fde8e8; color:#b42318; }}
.cap {{ font-size:12px; color:#6b7280; margin:6px 2px 0; }}
.wrow {{ margin: 6px 0; }}
.wlabel {{ font-size: 13px; margin-bottom: 3px; }}
.bar {{ display:flex; height: 14px; border-radius: 4px; overflow:hidden; background:#e3e6ea; }}
.seg {{ height:100%; }}
.muted {{ color:#9aa0aa; font-weight: 400; }}
.legend {{ margin-top: 12px; font-size: 12px; color:#6b7280; }}
.lg {{ margin-right: 14px; }} .lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }}
.toggle {{ font-size: 11px; text-transform:none; letter-spacing:0; padding:2px 10px;
          border:1px solid #c7ccd3; border-radius:12px; background:#fff; color:#374151; cursor:pointer; }}
.foot {{ margin-top: 40px; font-size: 12px; color:#9aa0aa; }}
</style></head><body><div class="wrap">
<h1>wxmax &mdash; daily maximum temperature panel</h1>
<div class="today">Forecast day: {date_str} <span class="muted" style="font-weight:400">(US local)</span></div>
<div class="sub">12 US cities &middot; ground truth = NWS Climatological Report (CLI) &middot; generated {gen}</div>

<h2>Today's panel</h2>
<table><tr><th>City</th><th>Morning est</th><th>Real-time best</th><th>Confidence</th><th class="num">Clim peak</th><th>Call</th></tr>
{_today_panel_rows(est, names, clim_stats)}</table>
<p class="cap"><b>Morning est</b> = the static begin-of-day forecast (fixed all day).
<b>Real-time best</b> = the live best estimate of today's max &mdash; it updates every few minutes and
converges to the realized max as the peak passes; <span class="lock">&#128274;</span> = LOCKED.
<b>Confidence</b> = P(today's official max within &plusmn;1&deg;F of our estimate) &mdash; forecast-driven in the
morning (expert agreement + skill), sharpening as obs arrive, ~99% once locked. The LOCK fires separately when P(peak passed) &ge; 85%.
<b>Clim peak</b> = this city's typical peak-hour window (median&ndash;P90, local) from its own history.
<span class="heat">&#9888; HEAT</span> = active NWS heat alert.</p>

<h2>Realized max temperature (NWS CLI) over time</h2>
{_realized_section(truth, names)}

<h2>Forecast accuracy vs NWS CLI (MAE &deg;F)</h2>
<table><tr><th>City</th><th class="num">MAE</th><th class="num">n days</th></tr>
{_accuracy_rows(est, truth, names)}</table>
<p class="cap">MAE = mean absolute error: the average of |forecast &minus; official CLI max|, in &deg;F.
Lower is better (e.g. MAE 1.5 = off by 1.5&deg;F on average).</p>

<h2>Per-region expert weights <button id="wbtn" class="toggle" onclick="toggleW()">hide</button></h2>
<div id="weights">
{_weight_bars(weights, names)}
</div>

<div class="foot">Free, public-domain data only (NWS API, NOAA NODD, ECMWF Open Data, IEM).
Online source selection via Hedge + Fixed-Share. Built with wxmax.</div>
<script>
function toggleW() {{
  var w = document.getElementById('weights'), b = document.getElementById('wbtn');
  var hidden = w.style.display === 'none';
  w.style.display = hidden ? '' : 'none';
  b.textContent = hidden ? 'hide' : 'show';
}}
</script>
</div></body></html>"""
    out = cfg.docs_dir / "index.html"
    out.write_text(html)
    return out
