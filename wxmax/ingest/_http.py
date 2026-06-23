"""Shared HTTP helper: a session with a polite User-Agent and simple backoff.

IEM and NWS both expect a descriptive User-Agent; Open-Meteo rate-limits the
free tier, so we retry 429/5xx with exponential backoff.
"""
from __future__ import annotations

import time

import requests

USER_AGENT = "wxmax/0.1 (https://plpfunds.com; data@plpfunds.com)"

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _session = s
    return _session


def _get(url: str, params: dict | None, timeout: float, retries: int):
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = session().get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} from {url}")
            resp.raise_for_status()
            return resp
        except (requests.RequestException,) as exc:  # noqa: PERF203
            last_exc = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GET failed after {retries + 1} attempts: {url}") from last_exc


def get_json(url: str, params: dict | None = None, timeout: float = 60, retries: int = 3) -> dict:
    return _get(url, params, timeout, retries).json()


def get_text(url: str, params: dict | None = None, timeout: float = 120, retries: int = 3) -> str:
    return _get(url, params, timeout, retries).text
