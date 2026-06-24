"""The daily panel orchestrator.

Lifecycle (each method idempotent per station/day; safe to re-run):
  start_of_day(d)  -> gather expert forecasts, blend (online weights) into an
                      ESTIMATE per station, persist.
  hourly_poll(d)   -> pull live obs; once the hotspot detector fires, replace the
                      estimate with a HIGH-conviction obs-anchored max.
  end_of_day(d)    -> fetch the official NWS CLI max, record it, update each
                      region's online expert weights (Hedge + Fixed-Share).

Experts (all free / commercial-clean): NWS NBM (JSON), ECMWF IFS + AIFS (open
data GRIB), GFS (NODD GRIB). Missing experts on a given run are simply absent --
the online blend renormalizes over whoever showed up. Weights are keyed per
station (each location is its own expert-advice problem).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .. import store
from ..config import PANEL_STATIONS, Config, load_config
from ..ingest import forecasts_ecmwf, forecasts_nodd, obs_nws
from ..ingest.truth_cli import fetch_cli_max
from ..models.online_experts import OnlineExpertBlend
from ..nowcast.hotspot import detect_high_conviction, model_remaining_rise
from ..nowcast.obs_nowcast import intraday_nowcast
from ..stations import Station, load_stations
from ..timeutil import to_local

EXPERTS = ["nws_nbm", "ecmwf_ifs", "ecmwf_aifs", "gfs"]


def confidence_score(half_width: float) -> int:
    """0-100 confidence from the forecast interval half-width (°F).

    A tighter band = more confident. This unifies both signals: in the morning
    ESTIMATE phase the half-width grows with expert disagreement (low confidence
    when models diverge); once a city flips to HIGH conviction the band collapses
    to ~0.5°F, so it naturally scores near-certain. NOT a calibrated probability
    -- it's a monotone, interpretable tightness score (capped 30-98).
    """
    return int(round(max(30.0, min(98.0, 100.0 - 7.5 * half_width))))


class Panel:
    def __init__(self, cfg: Config | None = None, station_ids=PANEL_STATIONS) -> None:
        self.cfg = cfg or load_config()
        reg = load_stations(self.cfg.stations_path)
        self.stations = [reg[s] for s in station_ids]
        self.dir = self.cfg.panel_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.blend = self._load_blend()

    # ---- persistence -------------------------------------------------------
    @property
    def _weights_path(self) -> Path:
        return self.dir / "weights.json"

    def _load_blend(self) -> OnlineExpertBlend:
        if self._weights_path.exists():
            return OnlineExpertBlend.from_state(json.loads(self._weights_path.read_text()))
        return OnlineExpertBlend(EXPERTS)

    def _save_blend(self) -> None:
        self._weights_path.write_text(json.dumps(self.blend.to_state(), indent=2))

    def _append(self, name: str, rows: list[dict], keys: list[str]) -> pd.DataFrame:
        path = self.dir / f"{name}.parquet"
        new = pd.DataFrame(rows)
        if path.exists():
            old = store.read_parquet(path)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
        store.write_parquet(new, path)
        return new

    # ---- expert gathering (best-effort per source) -------------------------
    def gather_experts(self, d: date) -> dict[str, dict[str, float]]:
        vals: dict[str, dict[str, float]] = {st.id: {} for st in self.stations}

        def safe(fn, expert):
            try:
                res = fn()
            except Exception as e:  # one bad source never sinks the panel
                print(f"  [warn] expert {expert} failed: {type(e).__name__}: {str(e)[:80]}")
                return
            for sid, v in res.items():
                if v is not None:
                    vals[sid][expert] = float(v)

        for st in self.stations:
            try:
                v = obs_nws.fetch_nbm_daily_max(st, d)
                if v is not None:
                    vals[st.id]["nws_nbm"] = float(v)
            except Exception:
                pass
        safe(lambda: forecasts_ecmwf.fetch_daily_max(self.stations, d, model="ifs"), "ecmwf_ifs")
        safe(lambda: forecasts_ecmwf.fetch_daily_max(self.stations, d, model="aifs-single"), "ecmwf_aifs")
        safe(lambda: forecasts_nodd.fetch_gfs_daily_max(self.stations, d), "gfs")
        return vals

    # ---- lifecycle ---------------------------------------------------------
    def start_of_day(self, d: date) -> pd.DataFrame:
        experts = self.gather_experts(d)
        rows = []
        for st in self.stations:
            ev = experts[st.id]
            est = self.blend.predict(st.id, ev)
            w = self.blend.weights(st.id)
            # interval half-width grows with cross-expert disagreement (spread)
            vals = list(ev.values())
            spread = float(np.std(vals)) if len(vals) >= 2 else 0.0
            half = min(10.0, max(1.5, 1.0 + 1.25 * spread))
            rows.append({
                "date": d.isoformat(), "station": st.id, "conviction": "ESTIMATE",
                "estimate": round(est, 1) if est is not None else None,
                "lo": round(est - half, 1) if est is not None else None,
                "hi": round(est + half, 1) if est is not None else None,
                "confidence": confidence_score(half) if est is not None else None,
                "spread": round(spread, 1),
                "n_experts": len(ev),
                **{f"x_{k}": round(v, 1) for k, v in ev.items()},
                "weights": json.dumps({k: round(w[k], 3) for k in ev}),
            })
        self._append("estimates", rows, keys=["date", "station", "conviction"])
        return pd.DataFrame(rows)

    def hourly_poll(self, d: date) -> pd.DataFrame:
        rows = []
        for st in self.stations:
            try:
                obs = obs_nws.fetch_obs_series(st, d, d)
            except Exception:
                continue
            if obs.empty:
                continue
            local = to_local(obs["valid"], st.tz)
            obs = obs.assign(hr=[t.hour + t.minute / 60 for t in local], day=[t.date() for t in local])
            obs = obs[obs["day"] == d]
            if obs.empty:
                continue
            obs_max, obs_now = float(obs["tmpf"].max()), float(obs["tmpf"].iloc[-1])
            now_hour = float(obs["hr"].iloc[-1])
            try:
                hf = obs_nws.fetch_hourly_forecast(st)
                hl = to_local(hf["valid"], st.tz)
                # Restrict to TODAY's local hours only -- otherwise "hours >= now"
                # wrongly sweeps in tomorrow's peak and inflates the remaining rise.
                today = [(t.hour + t.minute / 60, v) for t, v in zip(hl, hf["tmpf"]) if t.date() == d]
                rise = (model_remaining_rise([h for h, _ in today], [v for _, v in today], now_hour)
                        if today else None)
            except Exception:
                rise = None
            conv = detect_high_conviction(st, d, now_hour, obs["hr"].tolist(), obs["tmpf"].tolist(), rise)
            if not conv.high_conviction:
                continue
            nc = intraday_nowcast(obs_max, obs_now, rise, high_conviction=True)
            rows.append({
                "date": d.isoformat(), "station": st.id, "conviction": "HIGH",
                "estimate": round(nc.high, 1), "lo": round(nc.lo, 1), "hi": round(nc.hi, 1),
                "confidence": confidence_score((nc.hi - nc.lo) / 2.0),
                "obs_max_so_far": round(obs_max, 1), "n_signals": conv.n_signals,
            })
        if rows:
            self._append("estimates", rows, keys=["date", "station", "conviction"])
        return pd.DataFrame(rows)

    def end_of_day(self, d: date, expert_values: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
        """Record official CLI truth and update each station's expert weights."""
        if expert_values is None:
            expert_values = self._expert_values_from_estimates(d)
        scored = self._scored_stations(d)  # already counted -> don't double-update
        rows = []
        for st in self.stations:
            cli = fetch_cli_max(st.id, d)
            if cli is None:
                rows.append({"date": d.isoformat(), "station": st.id, "cli_max_f": None})
                continue
            ev = expert_values.get(st.id, {})
            if ev and st.id not in scored:
                self.blend.update(st.id, ev, cli.cli_max_f)
            rows.append({"date": d.isoformat(), "station": st.id, "cli_max_f": cli.cli_max_f,
                         "high_time": cli.high_time})
        self._save_blend()
        self._append("truth", rows, keys=["date", "station"])
        return pd.DataFrame(rows)

    def _scored_stations(self, d: date) -> set[str]:
        """Stations whose truth for day `d` is already recorded (weights applied)."""
        path = self.dir / "truth.parquet"
        if not path.exists():
            return set()
        t = store.read_parquet(path)
        t = t[(t["date"] == d.isoformat()) & t["cli_max_f"].notna()]
        return set(t["station"])

    def rebuild_weights(self) -> None:
        """Deterministically replay the committed (estimates, truth) history into a
        fresh blend -- one update per (station, day). Makes the learned weights
        reproducible and repairs any prior double-counting."""
        self.blend = OnlineExpertBlend(EXPERTS)
        path = self.dir / "truth.parquet"
        if path.exists():
            truth = store.read_parquet(path).dropna(subset=["cli_max_f"])
            for d_iso in sorted(truth["date"].unique()):
                ev_by = self._expert_values_from_estimates(date.fromisoformat(d_iso))
                for _, r in truth[truth.date == d_iso].iterrows():
                    ev = ev_by.get(r["station"])
                    if ev:
                        self.blend.update(r["station"], ev, float(r["cli_max_f"]))
        self._save_blend()

    def _expert_values_from_estimates(self, d: date) -> dict[str, dict[str, float]]:
        path = self.dir / "estimates.parquet"
        out: dict[str, dict[str, float]] = {}
        if not path.exists():
            return out
        df = store.read_parquet(path)
        df = df[df["date"] == d.isoformat()]
        xcols = [c for c in df.columns if c.startswith("x_")]
        for _, r in df.iterrows():
            ev = {c[2:]: float(r[c]) for c in xcols if c in r and pd.notna(r[c])}
            if ev:  # only the ESTIMATE row carries expert values (HIGH rows don't)
                out[r["station"]] = ev
        return out
