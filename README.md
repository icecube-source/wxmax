# wxmax — calibrated "highest temperature today at X" (US)

A multi-model ensemble that forecasts the **daily maximum temperature** at a US
location and reports a **calibrated confidence interval** (an 80%/90% interval
that actually contains the realized max ~80%/90% of the time).

Why not just call one weather API? Every single model — AI or physics — carries
a systematic, location-specific bias for near-surface daily max. Measured at
coastal **KLAX** over a week in June 2024, raw model daily-max vs ASOS truth:

| model | MAE °F | bias |
|---|---|---|
| GFS | 0.78 | +0.4 |
| GEM | 4.4 | +4.4 |
| ICON | 5.2 | +5.2 |
| **ECMWF IFS** | **10.7** | **+10.7** |
| ERA5 (reanalysis) | 6.7 | — |

ECMWF's grid cell averages in hot inland air; ERA5 reanalysis is itself 6.7°F
off — which is why **station observations, not reanalysis, are the truth.**

## Status (prototype back-test)

Implemented and tested end-to-end on **live** data (no GPU, no GRIB — all JSON):

- **P0** package, config, station registry (15 diverse US sites), tz-aware
  local-day max bucketing.
- **P1** ground truth: ASOS hourly (IEM) → observed local-day max.
- **P2** forecasts: Open-Meteo Historical-Forecast archive (GFS, ECMWF IFS,
  AIFS, ICON, GEM), ensemble members, ERA5.
- **P3** verification metrics (MAE/RMSE/CRPS/pinball/PIT/rank-hist/coverage/
  width/spread-skill) + single-model baseline back-test.
- **P4/P5** MOS linear blend, Gaussian **EMOS/NGR** head (variance grows with
  cross-model spread), **conformalized quantile (CQR)** interval calibration.

### Headline result (6 stations, 16-month chronological split)

```
best single model (GFS)  : MAE 1.07 °F
calibrated blend         : MAE 1.05 °F   CRPS 0.78 °F
interval coverage  (test): 80% level -> 85.4%   |   90% level -> 91.6%
```

**Honest read:** at well-sited US airport ASOS stations, raw GFS daily-max is
already ~1°F MAE, so a linear blend adds little point accuracy. The system's
demonstrated value here is the **reliable interval** (90%→91.6% coverage), and
bias correction for badly-sited/biased members (the ECMWF +10°F case). Bigger
point-accuracy gains need the documented upgrades below and pay off most at hard
locations (coastal microclimate, complex terrain).

## Quickstart

```bash
pip install -r requirements.txt        # core backbone (P0-P5)
python -m pytest -q                    # 31 offline tests
python scripts/demo_eval.py            # live baseline + calibrated eval
```

```python
from datetime import date
from wxmax.verify.evaluate import run_calibrated_eval
r = run_calibrated_eval(["KLAX","KMDW"], date(2024,6,1), date(2025,9,30))
print(r["intervals"])
```

## Layout

```
wxmax/ingest/      obs_asos (truth) · forecasts_openmeteo (models/ensembles/ERA5)
wxmax/features/    build_features  -> per-(station,date) design matrix
wxmax/models/      mos_blend (LinearMOS) · emos (GaussianEMOS/NGR)
wxmax/calibrate/   conformal (CQR + gaussian_interval)
wxmax/verify/      metrics · backtest (the bar) · evaluate (end-to-end)
config/stations.yaml   diverse US station panel
scripts/demo_eval.py   reproducible demo
```

## Documented upgrades (not yet built)

- **P4 nonlinear MOS**: LightGBM with season/day-of-year/recent-bias/terrain
  features + more years/stations (drop-in `fit`/`predict`, eval unchanged).
- **P5 quantile head + MAPIE**: gradient-boosted quantile regression and
  Adaptive Conformal Inference (ACI) for seasonal drift.
- **P6 point optimizer**: pinball-optimal point instead of the blend mean.
- **P7 serving**: live GRIB ingest (Herbie/ECMWF Open Data/WeatherNext
  BigQuery), FastAPI endpoint, arbitrary-point snapping + terrain adjustment.
- **Truth**: add GHCN-Daily TMAX (CC0) as the QC'd historical authority.

## Continuous hosting + instant alerts

GitHub Actions cron is fine for a once/twice-daily refresh, but **not** for a
15-minute "alert the moment a city locks" loop (its schedule is best-effort, and
high-frequency private-repo runs blow the free minutes). For that, run the
always-on **watcher**:

```bash
pip install -e ".[panel]"
# refresh every 15 min, serve the dashboard locally, pick an alert channel:
WXMAX_ALERT=desktop python -m wxmax.panel.watch --serve 8000                       # macOS pop-ups
WXMAX_ALERT=ntfy WXMAX_NTFY_TOPIC=my-secret-topic python -m wxmax.panel.watch      # phone/desktop push
WXMAX_ALERT=email SMTP_HOST=smtp.gmail.com SMTP_USER=you@gmail.com \
  SMTP_PASS=app-password ALERT_TO=you@example.com python -m wxmax.panel.watch
```

Each tick (default `--interval 900` = 15 min) it: runs the morning estimate once
per day, polls observations, fires **one alert per city per day the instant its
max locks** (hottest part of the day has passed → high conviction), records the
official CLI max + updates weights after ~14 UTC, and regenerates the
auto-refreshing dashboard. `--serve PORT` exposes it at `http://localhost:PORT/`.

**Where to run it (pick one):**
- **Your Mac, always on** — `WXMAX_ALERT=desktop`, $0, native pop-ups. Keep alive
  at login with a launchd agent.
- **A $5/mo VPS** (Hetzner/Fly.io/DigitalOcean) — `WXMAX_ALERT=ntfy` (free push)
  or `email`; keep alive with systemd. Comfortably within a $10–20 budget.

Minimal systemd unit (VPS):
```ini
[Service]
WorkingDirectory=/opt/wxmax
Environment=WXMAX_ALERT=ntfy WXMAX_NTFY_TOPIC=my-secret-topic
ExecStart=/opt/wxmax/.venv/bin/python -m wxmax.panel.watch --serve 8080
Restart=always
[Install]
WantedBy=multi-user.target
```

Alert backends: `log` (default), `desktop` (macOS), `email` (SMTP), `ntfy`
(ntfy.sh push), `slack` (webhook). Data stays $0/key-free; only the alert channel
may need one credential (an SMTP app password — or none for ntfy/desktop).

## Licensing (commercial-clean)

NOAA = public domain; ECMWF/DWD = CC-BY; GHCN = CC0. Open-Meteo **free tier is
non-commercial** — set `OPENMETEO_API_KEY` for the commercial endpoint.
Meteostat is intentionally excluded (data is CC-BY-NC).
