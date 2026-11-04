"""Optional Bearer auth for self-hosted container HTTP + MCP."""

from __future__ import annotations

import os
from typing import Any


def serve_token() -> str | None:
    raw = os.environ.get("PFNSTUDIO_SERVE_TOKEN", "").strip()
    return raw or None


def extract_bearer(header: str | None) -> str | None:
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    tok = parts[1].strip()
    return tok or None


def auth_required() -> bool:
    return serve_token() is not None


def check_auth(authorization: str | None) -> dict[str, Any] | None:
    """Return an error body when auth fails, else None."""
    expected = serve_token()
    if not expected:
        return None
    presented = extract_bearer(authorization)
    if not presented or presented != expected:
        return {"error": "unauthorized", "detail": "Missing or invalid Bearer token"}
    return None
