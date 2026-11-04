"""Thin HTTP client for the local daemon (``pfnstudio.daemon``).

``pfnstudio run`` and friends are clients of the daemon, never model hosts:

    from pfnstudio import client
    client.ensure_running()                 # spawn `python -m pfnstudio.daemon` if needed
    info = client.load("cdfm-base")   # or "slug:tag"
    out = client.predict(info["slug"], {"task": "discover", "inputs": {...}})

Only ``requests`` is imported at module load — importing this module is cheap.

Env: PFNSTUDIO_HOST (daemon URL, default http://127.0.0.1:11440),
PFNSTUDIO_SERVE_TOKEN (sent as Bearer when set), PFNSTUDIO_DAEMON_LOG (where a
spawned daemon writes its output).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:11440"
HEALTH_TIMEOUT = 1.0  # seconds per /healthz probe
START_POLL = 0.25  # seconds between probes while the daemon boots
START_TIMEOUT = 120.0  # seconds to wait for a spawned daemon's /healthz (env PFNSTUDIO_DAEMON_START_TIMEOUT); a cold torch import can take well over 30 s
_LOOPBACK = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


class DaemonError(RuntimeError):
    """A non-2xx reply from the daemon; ``status`` and ``detail`` carry the body."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"daemon replied {status}: {detail}")
        self.status = status
        self.detail = detail


def base_url() -> str:
    """Daemon URL from PFNSTUDIO_HOST; a bare ``host:port`` gets ``http://``."""
    raw = os.environ.get("PFNSTUDIO_HOST", "").strip() or DEFAULT_BASE_URL
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _headers() -> dict[str, str]:
    tok = os.environ.get("PFNSTUDIO_SERVE_TOKEN", "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _unwrap(resp: requests.Response) -> Any:
    if resp.ok:
        return resp.json() if resp.content else {}
    try:
        detail = resp.json().get("detail", resp.json())
    except ValueError:
        detail = resp.text
    raise DaemonError(resp.status_code, detail)


def _get(path: str, **kw: Any) -> Any:
    return _unwrap(requests.get(f"{base_url()}{path}", headers=_headers(), **kw))


def _post(path: str, body: dict[str, Any], **kw: Any) -> Any:
    return _unwrap(requests.post(f"{base_url()}{path}", json=body, headers=_headers(), **kw))


def is_running() -> bool:
    """True when ``GET /healthz`` answers 200 within a short timeout."""
    try:
        return requests.get(f"{base_url()}/healthz", timeout=HEALTH_TIMEOUT).status_code == 200
    except requests.RequestException:
        return False


def log_path() -> Path:
    return Path(
        os.environ.get(
            "PFNSTUDIO_DAEMON_LOG",
            str(Path.home() / ".cache" / "pfnstudio" / "daemon" / "daemon.log"),
        )
    )


def _spawn_daemon(host: str, port: int) -> subprocess.Popen:
    """Start ``python -m pfnstudio.daemon`` detached; output goes to log_path()."""
    log = log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    out = open(log, "ab")  # noqa: SIM115 — handed to the child, closed with it
    return subprocess.Popen(
        [sys.executable, "-m", "pfnstudio.daemon", "--host", host, "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )


def ensure_running() -> None:
    """Make sure a daemon answers at base_url(); spawn one when it does not.

    Only a loopback PFNSTUDIO_HOST can be auto-started — a remote daemon that
    is down is an error, not something to start here.
    """
    if is_running():
        return
    parts = urlsplit(base_url())
    host, port = parts.hostname or "127.0.0.1", parts.port or 80
    if host not in _LOOPBACK:
        raise RuntimeError(
            f"no daemon at {base_url()} and it is not local — start `pfnstudio serve` there"
        )
    proc = _spawn_daemon(host, port)
    deadline = time.monotonic() + float(
        os.environ.get("PFNSTUDIO_DAEMON_START_TIMEOUT", START_TIMEOUT)
    )
    while time.monotonic() < deadline:
        if is_running():
            return
        if proc.poll() is not None:
            raise RuntimeError(f"daemon exited with code {proc.returncode} — see {log_path()}")
        time.sleep(START_POLL)
    raise RuntimeError(
        f"daemon did not answer at {base_url()} within the start timeout — see {log_path()}"
    )


def load(ref: str) -> dict:
    """``POST /load`` — resolve, pull and mount ``slug`` or ``slug:tag``.
    Blocks until the model is ready (downloads + weights load); no timeout."""
    return _post("/load", {"ref": ref}, timeout=None)


def unload(slug: str) -> dict:
    return _post("/unload", {"slug": slug})


def ps() -> list[dict]:
    out = _get("/ps")
    return out if isinstance(out, list) else []


def predict(slug: str, payload: dict) -> dict:
    """``POST /models/<slug>/predict`` — the container's predict route."""
    return _post(f"/models/{slug}/predict", payload, timeout=None)


def capabilities(slug: str) -> dict:
    """``GET /models/<slug>/capabilities`` — the container's capability manifest."""
    return _get(f"/models/{slug}/capabilities")
