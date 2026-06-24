"""Pluggable alert delivery for "max locked" events.

Backend selected by `WXMAX_ALERT` env (or the --alert flag): one of
  log      -- print to stdout (default; no setup)
  desktop  -- macOS pop-up via osascript (use when the watcher runs on your Mac)
  email    -- SMTP (env: SMTP_HOST, SMTP_PORT=587, SMTP_USER, SMTP_PASS,
              ALERT_TO, ALERT_FROM); works with a Gmail/Fastmail app password
  ntfy     -- push to ntfy.sh (env: WXMAX_NTFY_TOPIC) -> phone + desktop, free
  slack    -- Slack incoming webhook (env: WXMAX_SLACK_WEBHOOK)

Any backend failure falls back to `log` so a delivery hiccup never crashes the
watcher.
"""
from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from email.message import EmailMessage

import requests


def _log(subject: str, body: str) -> None:
    print(f"[ALERT] {subject}\n{body}", flush=True)


def _desktop(subject: str, body: str) -> None:
    if sys.platform == "darwin":
        safe = body.replace('"', "'").replace("\n", " ")
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe}" with title "{subject}"'],
            check=False)
    else:
        _log(subject, body)


def _email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("ALERT_FROM", os.environ["SMTP_USER"])
    msg["To"] = os.environ.get("ALERT_TO", os.environ["SMTP_USER"])
    msg.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587"))) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.send_message(msg)


def _ntfy(subject: str, body: str) -> None:
    topic = os.environ["WXMAX_NTFY_TOPIC"]
    requests.post(f"https://ntfy.sh/{topic}", data=body.encode(),
                  headers={"Title": subject, "Tags": "fire"}, timeout=15)


def _slack(subject: str, body: str) -> None:
    requests.post(os.environ["WXMAX_SLACK_WEBHOOK"],
                  json={"text": f"*{subject}*\n{body}"}, timeout=15)


BACKENDS = {"log": _log, "desktop": _desktop, "email": _email, "ntfy": _ntfy, "slack": _slack}


def send_alert(subject: str, body: str, backend: str | None = None) -> None:
    backend = backend or os.environ.get("WXMAX_ALERT", "log")
    fn = BACKENDS.get(backend, _log)
    try:
        fn(subject, body)
    except Exception as e:  # never let a delivery failure stop the watcher
        print(f"[ALERT-FALLBACK] backend={backend} failed: {type(e).__name__}: {e}", flush=True)
        _log(subject, body)
