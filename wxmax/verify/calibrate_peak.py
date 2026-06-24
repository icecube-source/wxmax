"""Back-test calibration of the per-station lock guards (p_lock, dwell_min).

Replays historical days: for each day, simulate the live confidence trajectory
(P_clim x P_traj, with P_model held neutral since we have no historical model
run — live P_model only adds delay, never an earlier lock, so this is
conservative). Grid-search (p_lock, dwell_min) for the combo that locks as early
as possible while keeping the "locked before the true peak" rate <= 1% at the
0.99 threshold. Emits a reliability table and writes per-station params to
panel_data/climatology/calibration.json (loaded by the panel; falls back to the
climatological P99 hour + 90-min dwell when a station has no entry).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..ingest.obs_asos import fetch_asos_hourly
from ..nowcast import peak_climatology
from ..nowcast.confidence import p_model, p_traj
from ..stations import load_stations
from ..timeutil import to_local

P_LOCK_GRID = [13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0]
DWELL_GRID = [30.0, 40.0, 45.0, 60.0, 90.0]
THRESHOLD = 0.85
EARLY_BUDGET = 0.06


def _days(hourly: pd.DataFrame, tz: str, min_obs: int = 12):
    loc = to_local(hourly["valid"], tz)
    df = pd.DataFrame({"tmpf": np.asarray(hourly["tmpf"], float)}, index=loc).dropna()
    df["day"] = df.index.date
    df["hr"] = df.index.hour + df.index.minute / 60.0
    for day, g in df.groupby("day"):
        g = g.sort_index()
        if len(g) >= min_obs:
            yield day, g["hr"].values, g["tmpf"].values


def _simulate(hrs, temps, clim):
    """Confidence trajectory for one day + the true peak hour."""
    true_peak = float(hrs[int(np.argmax(temps))])
    run_max, max_hr, traj = -1e9, hrs[0], []
    for hr, t in zip(hrs, temps):
        if t > run_max:
            run_max, max_hr = t, hr
        dwell = (hr - max_hr) * 60.0
        pc = clim.p_passed(float(hr)) or 0.0
        conf = pc * p_traj(dwell, run_max - t) * p_model(0.0)
        traj.append((float(hr), conf, dwell, bool(hr >= true_peak)))
    return traj, true_peak


def calibrate_station(station, today: date, cfg: Config, days: int = 45):
    clim = peak_climatology.load_or_build(station, today, cfg)
    try:
        h = fetch_asos_hourly(station.id, today - timedelta(days=days), today - timedelta(days=1))
    except Exception:
        h = pd.DataFrame()
    sims, rel = [], []
    if not h.empty:
        for _, hrs, temps in _days(h, station.tz):
            traj, tp = _simulate(hrs, temps, clim)
            sims.append((traj, tp))
            rel += [(c, p) for _, c, _, p in traj]

    best = None
    for p_lock in P_LOCK_GRID:
        for dwell_min in DWELL_GRID:
            n = early = 0
            delays = []
            for traj, tp in sims:
                lock = next((hr for hr, c, dw, _ in traj
                             if c >= THRESHOLD and hr >= p_lock and dw >= dwell_min), None)
                if lock is None:
                    continue
                n += 1
                early += int(lock < tp)
                if lock >= tp:
                    delays.append(lock - tp)
            if n == 0:
                continue
            rate = early / n
            med = float(np.median(delays)) if delays else 99.0
            key = (0, med, p_lock) if rate <= EARLY_BUDGET else (1, rate, p_lock)
            cand = {"p_lock": p_lock, "dwell_min": dwell_min, "n_days": len(sims),
                    "n_locked": n, "early_rate": round(rate, 4), "median_delay_h": round(med, 2)}
            if best is None or key < best[0]:
                best = (key, cand)

    if best is None:  # no usable history -> climatological fallback
        p99 = clim.pct(99)
        return {"p_lock": float(p99) if p99 == p99 else 16.0, "dwell_min": 90.0,
                "n_days": 0, "n_locked": 0, "early_rate": None, "fallback": True}, rel
    return best[1], rel


def reliability_table(rel) -> str:
    if not rel:
        return "  (no data)"
    arr = np.array(rel, dtype=float)
    out = ["  conf-bin      n     P(peak actually passed)"]
    for lo, hi in [(0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0001)]:
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        n = int(m.sum())
        frac = float(arr[m, 1].mean()) if n else float("nan")
        out.append(f"  [{lo:.2f},{hi:.2f})  {n:6d}   {frac:.3f}")
    return "\n".join(out)


def load_calibration(cfg: Config) -> dict:
    p = cfg.panel_dir / "climatology" / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}


def calibrate_all(station_ids, today: date, cfg: Config | None = None, days: int = 45) -> dict:
    cfg = cfg or load_config()
    reg = load_stations(cfg.stations_path)
    out, rel_all = {}, []
    for sid in station_ids:
        params, rel = calibrate_station(reg[sid], today, cfg, days=days)
        out[sid] = params
        rel_all += rel
        print(f"  {sid}: p_lock={params['p_lock']:.1f}h dwell={params['dwell_min']:.0f}m "
              f"early={params.get('early_rate')} delay={params.get('median_delay_h')}h "
              f"(n_days={params['n_days']}, locked={params['n_locked']})")
    print("\nreliability (pooled across stations):")
    print(reliability_table(rel_all))
    path = cfg.panel_dir / "climatology" / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return out
