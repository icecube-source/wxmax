"""Central configuration: paths, model list, quantile grid.

Kept tiny and dependency-free so every module can import it. Paths are resolved
relative to the repo root (the parent of this package) unless overridden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Forecast models we pull from Open-Meteo (JSON). Identifiers are Open-Meteo's
# `models=` values; see ingest/forecasts_openmeteo.py. AIFS = ECMWF's AI model.
DEFAULT_MODELS: tuple[str, ...] = (
    "gfs_seamless",        # NOAA GFS
    "ecmwf_ifs025",        # ECMWF IFS (ENS control / ex-HRES)
    "ecmwf_aifs025",       # ECMWF AIFS (AI)
    "icon_seamless",       # DWD ICON
    "gem_seamless",        # Environment Canada GEM
)

# Ensemble systems (member-resolving) on the Open-Meteo Ensemble API.
DEFAULT_ENSEMBLES: tuple[str, ...] = (
    "gfs_seamless",        # GEFS, 31 members
    "icon_seamless",       # ICON-EPS, 40/20 members
    "ecmwf_ifs025",        # IFS ENS, 51 members
)

# Quantile grid for the probabilistic head (P5). Symmetric, includes the median.
DEFAULT_QUANTILES: tuple[float, ...] = (
    0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
)

# The production panel's fixed universe (the ONLY locations it serves).
# Ground truth for each is the NWS Climatological Report (CLI) daily max.
PANEL_STATIONS: tuple[str, ...] = (
    "KSEA",  # Seattle
    "KSFO",  # San Francisco
    "KPHX",  # Phoenix
    "KLAX",  # Los Angeles
    "KNYC",  # New York Central Park (climate station, not an airport)
    "KLAS",  # Las Vegas
    "KMIA",  # Miami
    "KAUS",  # Austin-Bergstrom
    "KDCA",  # Washington DC (Reagan National — official DC site)
    "KMSY",  # New Orleans
    "KDEN",  # Denver
    "KMDW",  # Chicago Midway
)


@dataclass(frozen=True)
class Config:
    data_dir: Path = REPO_ROOT / "data"
    stations_path: Path = REPO_ROOT / "config" / "stations.yaml"
    models: tuple[str, ...] = DEFAULT_MODELS
    ensembles: tuple[str, ...] = DEFAULT_ENSEMBLES
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    # Open-Meteo: free (non-commercial) host by default. For the commercial
    # fund product, set base host to customer-api.open-meteo.com + api key.
    openmeteo_api_key: str | None = None
    # Committed (version-controlled) panel state + published dashboard. GitHub
    # Actions runners are ephemeral, so the panel's weights/history live in the
    # repo, not in the gitignored R&D `data/` dir.
    panel_dir: Path = REPO_ROOT / "panel_data"
    docs_dir: Path = REPO_ROOT / "docs"

    @property
    def obs_dir(self) -> Path:
        return self.data_dir / "obs"

    @property
    def forecast_dir(self) -> Path:
        return self.data_dir / "forecast"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.obs_dir, self.forecast_dir, self.panel_dir, self.docs_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(**overrides) -> Config:
    """Build a Config, applying keyword overrides (e.g. data_dir=...)."""
    return Config(**overrides)
