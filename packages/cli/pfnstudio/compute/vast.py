"""Vast.ai compute adapter.

End-to-end flow when `submit()` is called:

  1. Read config from env (api key, gpu filter, cost caps, ssh key path).
  2. Ensure our SSH public key is registered with the Vast account
     (idempotent — Vast dedupes by key text).
  3. Search Vast's offer market for the cheapest instance matching
     gpu_type + num_gpus + max_hourly_cost. Emit `vast.no_offers` and
     return early if nothing matches the cost cap.
  4. Provision the chosen offer with our Docker image.
  5. Poll instance state until SSH is reachable. Emit `vast.ssh_ready`.
  6. Tar the project root locally; scp it to the remote.
  7. SSH into the instance and run the local adapter inside it via the
     bundled pfnstudio CLI. Pipe its stdout back to ours line-by-line
     so the API sees the same JSON-line events it already understands.
  8. Pull results.json back; emit `vast.training_done`.
  9. Destroy the instance (always — finally block guarantees teardown
     even on Ctrl+C, SSH dropout, or remote crash).

Why this lives in the CLI rather than the API: the SSH/scp orchestration
is much easier to write in Python with the actual project layout in
hand. The API just spawns this with the right env vars and parses the
JSON-line stream like it already does for the local adapter.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from .base import ComputeAdapter

VAST_API_BASE = "https://console.vast.ai/api/v0"


# ─────────────────────────────────────────────────────────────────────
# Event emission — same JSON-line protocol the local adapter uses, so
# the API runner doesn't need to learn anything new. Vast-specific events
# are namespaced with `vast.` so callers can filter for them.
# ─────────────────────────────────────────────────────────────────────


def _emit(event: str, **fields: Any) -> None:
    """Write one event line to stdout if PFNSTUDIO_JSON_PROGRESS=1; else
    write a human-readable equivalent to stderr."""
    if os.environ.get("PFNSTUDIO_JSON_PROGRESS") == "1":
        sys.stdout.write(json.dumps({"event": event, **fields, "ts": time.time()}) + "\n")
        sys.stdout.flush()
    else:
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        sys.stderr.write(f"[vast] {event} {kv}\n".rstrip() + "\n")


def _emit_log(message: str) -> None:
    _emit("log", line=f"[vast] {message}")


# ─────────────────────────────────────────────────────────────────────
# Config — read from env so the API can pass per-run overrides without
# us having to read the org's vastConfig blob directly.
# ─────────────────────────────────────────────────────────────────────


def _config_from_env() -> dict[str, Any]:
    # GPU types are a set the user is OK with; the CLI searches across
    # all of them and picks the cheapest. Accept either:
    #   - VAST_GPU_TYPES="RTX_3090,RTX_4090" (preferred)
    #   - VAST_GPU_TYPE="RTX_3090"           (legacy single, still works)
    # Strip / uppercase / dedupe so downstream code can rely on canonical
    # shape. Falls back to the multi-GPU default if neither env is set.
    raw = os.environ.get("VAST_GPU_TYPES") or os.environ.get("VAST_GPU_TYPE", "RTX_3090,RTX_4090")
    gpu_types = [g.strip().upper() for g in raw.split(",") if g.strip()]
    # de-dupe preserving insertion order
    seen: set[str] = set()
    gpu_types = [g for g in gpu_types if not (g in seen or seen.add(g))]

    # VAST_OFFER_ID: when the user pre-selected a specific offer in the
    # UI, we skip the per-dispatch search entirely and provision exactly
    # that one. Parsed lazily because the empty-string env case (set but
    # cleared) should be treated as "not set".
    raw_offer_id = (os.environ.get("VAST_OFFER_ID") or "").strip()
    offer_id: int | None = None
    if raw_offer_id:
        try:
            offer_id = int(raw_offer_id)
        except ValueError:
            offer_id = None

    return {
        "api_key": os.environ.get("VAST_API_KEY"),
        "gpu_types": gpu_types,
        "num_gpus": int(os.environ.get("VAST_NUM_GPUS", "1")),
        "image": os.environ.get(
            "VAST_IMAGE",
            "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        ),
        "max_hourly_cost": float(os.environ.get("VAST_MAX_HOURLY_COST", "0.50")),
        "max_minutes": int(os.environ.get("VAST_MAX_MINUTES", "120")),
        "ssh_key_path": os.environ.get(
            "VAST_SSH_KEY_PATH",
            str(Path.home() / ".ssh" / "id_rsa"),
        ),
        "disk_gb": int(os.environ.get("VAST_DISK_GB", "20")),
        "offer_id": offer_id,
    }


# ─────────────────────────────────────────────────────────────────────
# Vast REST API helpers
# ─────────────────────────────────────────────────────────────────────


def _vast_get(path: str, api_key: str, **params: Any) -> Any:
    import requests

    r = requests.get(
        f"{VAST_API_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _vast_get_raw(path: str, api_key: str, **params: Any) -> tuple[int, Any, str]:
    """GET that doesn't raise on non-2xx and returns (status_code, parsed_body_or_None, raw_text_tail).
    Used for diagnostic tracing — we want the response body even on errors,
    and a short tail of raw text in case the body isn't valid JSON."""
    import requests

    r = requests.get(
        f"{VAST_API_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=30,
    )
    body: Any = None
    try:
        body = r.json()
    except Exception:
        body = None
    return r.status_code, body, r.text[:500]


def _vast_put(path: str, api_key: str, body: dict | None = None) -> Any:
    import requests

    r = requests.put(
        f"{VAST_API_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body or {},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _vast_post(path: str, api_key: str, body: dict | None = None) -> Any:
    import requests

    r = requests.post(
        f"{VAST_API_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body or {},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _vast_delete(path: str, api_key: str) -> None:
    import requests

    r = requests.delete(
        f"{VAST_API_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    # Vast returns 200 on success; tolerate 404 (already gone).
    if r.status_code not in (200, 404):
        r.raise_for_status()


def _ensure_ssh_key_registered(api_key: str, ssh_key_path: str) -> None:
    """POST our public key to /users/current/ssh-keys/. Vast dedupes by
    text content, so calling this on every run is safe and cheap."""
    pubkey_path = ssh_key_path + ".pub"
    if not os.path.exists(pubkey_path):
        raise RuntimeError(
            f"SSH public key not found at {pubkey_path}. Generate one with "
            f"`ssh-keygen -t ed25519 -f {ssh_key_path}` or set VAST_SSH_KEY_PATH."
        )
    pubkey = Path(pubkey_path).read_text().strip()
    try:
        _vast_post("/users/current/ssh-keys/", api_key, {"ssh_key": pubkey})
    except Exception as e:
        # 409 (already registered) or duplicate is fine; anything else we
        # treat as best-effort — the user may have manually added the key.
        _emit_log(f"ssh key registration: {e} (continuing — key may already be in account)")


def _baseline_market_check(api_key: str) -> dict[str, Any]:
    """Hit /bundles/ with NO filters at all. Used as a last-resort sanity
    check when every filtered search returned empty — confirms whether
    the endpoint is reachable and the auth + URL path are even right.

    Returns a dict with `{ok: bool, offer_count: int, hint: str}` so the
    error message can tell the user "Vast returned 1234 offers in the
    unfiltered baseline, but none match your filter shape — probably a
    query-format issue" versus "Vast returned zero unfiltered — auth or
    endpoint is broken"."""
    try:
        body = _vast_get("/bundles/", api_key)
    except Exception as e:
        return {"ok": False, "offer_count": 0, "hint": f"baseline GET failed: {e}"}
    if not isinstance(body, dict):
        return {
            "ok": False,
            "offer_count": 0,
            "hint": f"response was {type(body).__name__}, expected dict",
        }
    offers = body.get("offers", [])
    if not isinstance(offers, list):
        return {
            "ok": False,
            "offer_count": 0,
            "hint": f"'offers' key was {type(offers).__name__} not list — Vast API may have changed shape",
        }
    return {"ok": True, "offer_count": len(offers), "hint": "ok"}


def _gpu_name_matches(offer_name: str | None, gpu_type: str) -> bool:
    """Case- and separator-insensitive substring match between Vast's
    `gpu_name` field and our requested gpu_type. Vast's catalog uses
    inconsistent spellings (e.g. `RTX 4090`, `RTX_4090`, sometimes
    `GeForce RTX 4090`), so we normalise both sides to uppercase with
    underscores collapsed to spaces before substring testing."""
    if not offer_name:
        return False

    def _norm(s: str) -> str:
        return s.upper().replace("_", " ").replace("-", " ").strip()

    return _norm(gpu_type) in _norm(offer_name)


def _search_offers(
    api_key: str,
    cfg: dict[str, Any],
    *,
    include_over_cap: bool = False,
    per_type_limit: int = 5,
    permissive: bool = False,
    emit_debug: bool = False,
) -> list[dict[str, Any]]:
    """Fetch the Vast market once and filter client-side.

    Why we don't push the filter into Vast's `q` parameter:
      - Vast's /bundles/?q={...} is finicky. `{"like": "%X%"}` on
        `gpu_name` returns HTTP 400; `{"eq": "RTX_4090"}` returns 0
        offers because the catalog spelling varies (`RTX 4090`,
        `GeForce RTX 4090`, etc). The unfiltered baseline reliably
        returns the whole market, so we fetch that and filter in
        Python. Cheaper than guessing Vast's evolving filter grammar.

    Filter modes:
      - `permissive=False, include_over_cap=False` (default): rentable
            + on-demand + under cost cap. Provisioning path. We don't
            require `verified` because that flag is reserved for a few
            datacenter operators — every consumer GPU host on Vast is
            unverified, so requiring it would block ~99% of supply.
      - `permissive=False, include_over_cap=True`: same, no cost cap.
            Used to show "raise the cap to $X" diagnostics.
      - `permissive=True`: drops rentable / type filters so we can see
            what's in the market when the strict path returns empty.

    Each offer is tagged with `__matched_type` so callers know which
    selected GPU type it satisfied."""
    if emit_debug:
        status, body, raw_tail = _vast_get_raw("/bundles/", api_key)
        if status >= 400 or not isinstance(body, dict):
            _emit(
                "vast.debug",
                stage="search",
                http_status=status,
                offer_count=0,
                raw_tail=raw_tail,
                note="baseline GET failed or returned non-dict",
            )
            return []
    else:
        try:
            body = _vast_get("/bundles/", api_key)
        except Exception as e:
            _emit_log(f"offer search failed: {e}")
            return []

    offers = body.get("offers", []) if isinstance(body, dict) else []
    if not isinstance(offers, list):
        offers = []

    def _is_truthy_flag(v: Any) -> bool:
        # Vast occasionally returns booleans as 0/1 ints depending on
        # endpoint version. Treat truthy/1/'true' as True.
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return False

    def _strict_fail_reason(offer: dict[str, Any]) -> str | None:
        """Return None if the offer passes the strict tier, else a short
        human-readable label explaining which flag failed. Used to
        annotate permissive-tier offers in the no-offers diagnostic so
        the user sees exactly why each was rejected (rentable=false vs
        type=interruptible vs etc) instead of guessing."""
        if not _is_truthy_flag(offer.get("rentable", True)):
            return "rentable=false"
        offer_type = (offer.get("type") or offer.get("rental_type") or "").lower()
        if offer_type and offer_type not in ("on-demand", "reserved"):
            return f"type={offer_type}"
        return None

    def _passes(offer: dict[str, Any], gpu_type: str) -> bool:
        if not _gpu_name_matches(offer.get("gpu_name"), gpu_type):
            return False
        try:
            n_gpus = int(offer.get("num_gpus", 0))
        except (TypeError, ValueError):
            return False
        if n_gpus != cfg["num_gpus"]:
            return False
        if not permissive and _strict_fail_reason(offer) is not None:
            return False
        if not include_over_cap:
            try:
                dph = float(offer.get("dph_total", 9e9))
            except (TypeError, ValueError):
                return False
            if dph > cfg["max_hourly_cost"]:
                return False
        return True

    candidates: list[dict[str, Any]] = []
    per_type_counts: dict[str, int] = {}
    for gpu_type in cfg["gpu_types"]:
        matched = [o for o in offers if _passes(o, gpu_type)]
        matched.sort(key=lambda o: float(o.get("dph_total", 1e9)))
        per_type_counts[gpu_type] = len(matched)
        for o in matched[:per_type_limit]:
            o["__matched_type"] = gpu_type
            # When this offer came in via the permissive tier, tag it
            # with the reason it would have failed strict — so the
            # caller's error-rendering can show "$X RTX 4090 (rentable=false)".
            if permissive:
                o["__strict_fail"] = _strict_fail_reason(o)
            candidates.append(o)

    if emit_debug:
        # Inventory by gpu_name+num_gpus tells the user what's actually
        # in the market right now, so they can pick a GPU type that has
        # supply without guessing Vast's catalog spelling.
        from collections import Counter

        inv = Counter(f"{o.get('gpu_name', '?')} ×{o.get('num_gpus', '?')}" for o in offers)
        _emit(
            "vast.debug",
            stage="search",
            total_offers=len(offers),
            asked_for=cfg["gpu_types"],
            matches_per_type=per_type_counts,
            permissive=permissive,
            include_over_cap=include_over_cap,
            top_inventory=dict(inv.most_common(10)),
        )

    candidates.sort(key=lambda o: float(o.get("dph_total", 1e9)))
    return candidates


def _find_cheapest_offer(api_key: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Cheapest offer that matches the filter AND stays under the cost cap.
    Returns None if nothing's available under-cap."""
    offers = _search_offers(api_key, cfg, include_over_cap=False, per_type_limit=1)
    return offers[0] if offers else None


def _fetch_offer_by_id(api_key: str, offer_id: int) -> dict[str, Any] | None:
    """Look up a specific offer by id. Used when the user pre-selected
    one in the UI; we skip the search and provision exactly this offer.
    Returns None if the offer no longer exists in the market (someone
    else rented it, host went offline, etc)."""
    try:
        body = _vast_get("/bundles/", api_key)
    except Exception as e:
        _emit_log(f"offer lookup for id={offer_id} failed: {e}")
        return None
    offers = body.get("offers", []) if isinstance(body, dict) else []
    for o in offers:
        if int(o.get("id") or o.get("ask_contract_id") or 0) == offer_id:
            return o
    return None


# ISO 3166-1 alpha-2 → human-readable continent. Vast's `geolocation`
# field is usually a 2-letter country code; this mapping covers the
# countries Vast typically has hosts in (data centers cluster in a small
# number of regions). Anything not listed falls back to the raw code.
_COUNTRY_TO_REGION: dict[str, str] = {
    # North America
    "US": "North America",
    "CA": "North America",
    "MX": "North America",
    # South America
    "BR": "South America",
    "AR": "South America",
    "CL": "South America",
    # Europe
    "DE": "Europe",
    "NL": "Europe",
    "FR": "Europe",
    "GB": "Europe",
    "UK": "Europe",
    "PL": "Europe",
    "RO": "Europe",
    "IT": "Europe",
    "ES": "Europe",
    "FI": "Europe",
    "SE": "Europe",
    "NO": "Europe",
    "DK": "Europe",
    "IE": "Europe",
    "PT": "Europe",
    "BE": "Europe",
    "AT": "Europe",
    "CH": "Europe",
    "CZ": "Europe",
    "HU": "Europe",
    "GR": "Europe",
    "RU": "Europe",
    "UA": "Europe",
    "EE": "Europe",
    "LV": "Europe",
    "LT": "Europe",
    "BG": "Europe",
    "HR": "Europe",
    "SK": "Europe",
    "SI": "Europe",
    # Asia
    "CN": "Asia",
    "JP": "Asia",
    "KR": "Asia",
    "TW": "Asia",
    "HK": "Asia",
    "IN": "Asia",
    "SG": "Asia",
    "TH": "Asia",
    "ID": "Asia",
    "MY": "Asia",
    "VN": "Asia",
    "PH": "Asia",
    "AE": "Asia",
    "IL": "Asia",
    "TR": "Asia",
    # Oceania
    "AU": "Oceania",
    "NZ": "Oceania",
    # Africa
    "ZA": "Africa",
    "EG": "Africa",
    "NG": "Africa",
    "KE": "Africa",
}


def _format_region(loc: str | None) -> str:
    """Pretty-print Vast's geolocation field. Country code stays primary
    (it's what Vast reports and what shows up in their dashboard), the
    continent is appended in parens so users scanning for "any in
    Europe" can match by eye without mapping ISO codes mentally."""
    if not loc:
        return "?"
    code = loc.strip().upper()
    region = _COUNTRY_TO_REGION.get(code)
    return f"{code} · {region}" if region else code


def _format_offers_table(offers: list[dict[str, Any]]) -> str:
    """Compact text rendering of the cheapest N offers. Embedded into the
    error message so the user sees what the market is offering right now
    and can pick a realistic cost cap. Plain text (no ANSI codes) so it
    renders cleanly in both the live event stream and the run-detail UI.

    When an offer carries `__strict_fail` (set by the search when it came
    in via the permissive tier), append the rejection reason so the user
    sees `$0.136/hr ... — rejected: rentable=false`."""
    if not offers:
        return ""
    rows: list[str] = []
    for o in offers:
        dph = float(o.get("dph_total", 0))
        gpu = (o.get("gpu_name") or o.get("__matched_type") or "?").strip()
        n_gpu = int(o.get("num_gpus", 1))
        ram = o.get("gpu_ram") or o.get("gpu_total_ram")
        ram_str = f" · {round(float(ram) / 1024)}GB" if ram else ""
        loc_str = _format_region(o.get("geolocation"))
        strict_fail = o.get("__strict_fail")
        fail_str = f"  — rejected: {strict_fail}" if strict_fail else ""
        rows.append(f"  ${dph:.3f}/hr · {gpu}{ram_str} · ×{n_gpu} · {loc_str}{fail_str}")
    return "\n".join(rows)


def _provision_instance(api_key: str, offer_id: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Provision the chosen offer. Returns the new instance descriptor
    (which includes the assigned instance_id we use for follow-ups)."""
    template_hash_id = (
        os.environ.get("VAST_TEMPLATE_HASH_ID") or cfg.get("template_hash_id") or ""
    ).strip()
    if template_hash_id:
        body = {
            "client_id": "me",
            "template_hash_id": template_hash_id,
            "disk": cfg["disk_gb"],
        }
    else:
        body = {
            "client_id": "me",
            "image": cfg["image"],
            "disk": cfg["disk_gb"],
            # `onstart_cmd` runs once when the container first boots. We don't
            # use it for the real work (SSH-exec is easier to monitor), but a
            # no-op pause keeps the container alive past image entrypoint.
            "onstart_cmd": "tail -f /dev/null",
            "runtype": "ssh",
        }
    return _vast_put(f"/asks/{offer_id}/", api_key, body)


def vast_proxy_ssh_port(raw_ssh_port: int) -> int:
    """Vast quirk: the `ssh_port` field in /instances/{id}/ is the proxy
    SSH port minus 1. Confirmed empirically — Vast's UI shows
    "Proxy ssh connect: ssh -p N root@sshX.vast.ai" while the API returns
    `ssh_port: N-1`; TCP-connect tests confirmed port N-1 closes the
    connection mid-handshake while port N reaches the auth phase. Kept in
    sync with the same adjustment in organizations.service.ts."""
    return raw_ssh_port + 1


def _wait_for_ssh(api_key: str, instance_id: int, timeout_s: int = 600) -> dict[str, Any]:
    """Poll until the instance is running and has an SSH endpoint. Returns
    the instance descriptor with `ssh_host` and `ssh_port` populated
    (`ssh_port` already adjusted to the real proxy port — see
    `vast_proxy_ssh_port`)."""
    deadline = time.time() + timeout_s
    backoff = 5
    while time.time() < deadline:
        info = _vast_get(f"/instances/{instance_id}/", api_key)
        inst = info.get("instances") if isinstance(info, dict) else None
        if inst:
            status = inst.get("actual_status") or inst.get("intended_status")
            ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
            ssh_port = inst.get("ssh_port")
            if status == "running" and ssh_host and ssh_port:
                inst["ssh_port"] = vast_proxy_ssh_port(int(ssh_port))
                return inst
        time.sleep(backoff)
        backoff = min(backoff + 2, 15)
    raise TimeoutError(f"Instance {instance_id} did not become SSH-ready within {timeout_s}s")


def _destroy_instance(api_key: str, instance_id: int) -> None:
    """Best-effort tear-down. Never raises — we always want to release
    the instance even when called from a finally block during an error."""
    try:
        _vast_delete(f"/instances/{instance_id}/", api_key)
    except Exception as e:
        _emit_log(
            f"failed to destroy instance {instance_id}: {e} — DESTROY MANUALLY in cloud.vast.ai"
        )


# ─────────────────────────────────────────────────────────────────────
# SSH / scp orchestration
# ─────────────────────────────────────────────────────────────────────


def _ssh_base_args(ssh_key_path: str, ssh_host: str, ssh_port: int) -> list[str]:
    return [
        "-i",
        ssh_key_path,
        "-p",
        str(ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ServerAliveInterval=20",
        f"root@{ssh_host}",
    ]


def _wait_for_ssh_connection(
    ssh_key_path: str,
    ssh_host: str,
    ssh_port: int,
    timeout_s: int = 300,
    *,
    ssh_user: str = "root",
) -> None:
    """Vast reports `running` before sshd is actually accepting connections.
    Retry a no-op `true` until it succeeds.

    On final failure we emit the FULL ssh -v trace as a `vast.debug`
    event line — collapsing 30+ lines of OpenSSH diagnostic into a
    single error string was making it impossible to tell apart "key
    permissions wrong" / "Vast proxy ACL miss" / "TCP refused" without
    re-running locally. Now the trace flows into the run's event log
    where the user can read it inline.

    Also pre-validates the private key file mode — if it's group/world-
    readable, OpenSSH silently refuses to use it and you get "Connection
    closed" without ever attempting auth."""
    # Pre-flight: key permissions. OpenSSH ignores keys with mode
    # >0600 and prints "WARNING: UNPROTECTED PRIVATE KEY FILE" which
    # users miss because we'd previously suppressed verbose output.
    try:
        mode = os.stat(ssh_key_path).st_mode & 0o777
        if mode & 0o077:
            _emit_log(
                f"WARN: SSH private key {ssh_key_path} has mode {oct(mode)} — "
                f"OpenSSH may refuse it. Run: chmod 600 {ssh_key_path}"
            )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"SSH private key {ssh_key_path} not found on the API host. "
            f"Run ./run.sh install to bootstrap ~/.ssh/vast_id."
        ) from e

    _emit_log(
        f"ssh probe: {ssh_user}@{ssh_host}:{ssh_port} key={ssh_key_path} timeout={timeout_s}s [v6]"
    )

    deadline = time.time() + timeout_s
    last_output = ""
    last_rc = -1
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        probe_cmd = [
            "ssh",
            "-v",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-i",
            ssh_key_path,
            "-p",
            str(ssh_port),
            f"{ssh_user}@{ssh_host}",
            "true",
        ]
        r = subprocess.run(probe_cmd, capture_output=True, timeout=15)
        if r.returncode == 0:
            return
        last_rc = r.returncode
        # ssh -v writes everything to stderr; stdout is normally empty.
        last_output = (
            r.stderr.decode(errors="replace") + r.stdout.decode(errors="replace")
        ).strip()
        time.sleep(5)

    # Emit the full trace as its own event so the run-detail event log
    # shows it inline. Truncate to 4 KB to avoid overflowing the log
    # ring buffer if the user has very chatty ssh banners.
    trace = last_output[:4000] if last_output else "(no output)"
    for line in trace.split("\n")[-30:]:  # last 30 lines is plenty
        if line.strip():
            _emit("log", line=f"[ssh] {line}")

    # Pick a tight summary line for the raised exception.
    diag = "(no output)"
    if last_output:
        signal_phrases = (
            "Permission denied",
            "Connection refused",
            "Connection timed out",
            "Connection closed",
            "no matching",
            "Could not resolve",
            "Host key verification",
            "UNPROTECTED PRIVATE KEY",
            "Authentication failed",
        )
        lines = [ln for ln in last_output.split("\n") if ln.strip()]
        match = next((ln for ln in lines if any(p in ln for p in signal_phrases)), None)
        diag = (match or lines[-1] if lines else "(no output)")[:300]
    raise TimeoutError(
        f"SSH never became reachable on {ssh_host}:{ssh_port} after {timeout_s}s "
        f"({attempts} attempts, last exit={last_rc}, last signal: {diag}). "
        f"Full -v trace above in run log."
    )


def _make_project_tarball(project_root: Path) -> Path:
    """Bundle the project root plus the pfnstudio packages it needs.

    We ship the local pfnstudio-core + pfnstudio CLI packages alongside
    the project so the remote doesn't have to pip-install from a registry
    (pfnstudio isn't on PyPI yet). Skips heavy dirs we don't need on the
    remote (node_modules, .venv, dist, .git)."""
    tmp = Path(tempfile.mkstemp(suffix=".tgz", prefix="pfnstudio-vast-")[1])
    tmp.unlink()  # mkstemp creates an empty file; we want tarfile to make a fresh one

    # pfnstudio-cloud root is six levels up from this file:
    #   .../pfnstudio-cloud/packages/cli/pfnstudio/compute/vast.py
    cli_root = Path(__file__).resolve().parents[4]
    pkg_core = cli_root / "packages" / "core"
    pkg_cli = cli_root / "packages" / "cli"

    skip_dirs = {
        "node_modules",
        ".venv",
        "dist",
        ".git",
        "__pycache__",
        ".angular",
        ".turbo",
        "logs",
    }

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if any(p in skip_dirs for p in parts):
            return None
        return info

    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(project_root, arcname="project", filter=_filter)
        if pkg_core.exists():
            tf.add(pkg_core, arcname="pfnstudio-core", filter=_filter)
        if pkg_cli.exists():
            tf.add(pkg_cli, arcname="pfnstudio-cli", filter=_filter)
    return tmp


def _scp_to_remote(
    ssh_key_path: str,
    ssh_host: str,
    ssh_port: int,
    local: Path,
    remote: str,
) -> None:
    cmd = [
        "scp",
        "-i",
        ssh_key_path,
        "-P",
        str(ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        str(local),
        f"root@{ssh_host}:{remote}",
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed: {r.stderr.decode(errors='replace')}")


def _scp_from_remote(
    ssh_key_path: str,
    ssh_host: str,
    ssh_port: int,
    remote: str,
    local: Path,
) -> bool:
    """Pull a file back from the remote. Returns True on success, False if
    the file isn't there. Other errors raise."""
    cmd = [
        "scp",
        "-i",
        ssh_key_path,
        "-P",
        str(ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        f"root@{ssh_host}:{remote}",
        str(local),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode == 0:
        return True
    err = r.stderr.decode(errors="replace")
    if "No such file" in err or "not a regular file" in err:
        return False
    raise RuntimeError(f"scp from remote failed: {err}")


def _ssh_stream(
    ssh_key_path: str,
    ssh_host: str,
    ssh_port: int,
    command: str,
    forward_events: bool = False,
) -> int:
    """Run a command over SSH, streaming stdout/stderr line-by-line back
    to our own stdout. When `forward_events` is True, lines that parse as
    valid JSON events are re-emitted as-is (so progress events from the
    remote pfnstudio run flow to the API runner unchanged); other lines
    are wrapped in a `log` event. Returns the remote exit code."""
    cmd = ["ssh", *_ssh_base_args(ssh_key_path, ssh_host, ssh_port), command]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            if forward_events:
                # Try to parse as a JSON event; if it parses and has an
                # `event` key, forward verbatim. Otherwise treat as a log
                # line. This preserves the API's existing event pipeline.
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict) and "event" in parsed:
                        sys.stdout.write(line + "\n")
                        sys.stdout.flush()
                        continue
                except Exception:
                    pass
                _emit("log", line=line)
            else:
                _emit("log", line=line)
    finally:
        proc.wait()
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────


class VastAdapter(ComputeAdapter):
    name = "vast"

    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        cfg = _config_from_env()
        if not cfg["api_key"]:
            return {
                "status": "error",
                "reason": "VAST_API_KEY not set. Configure your org's Vast key at /orgs/<slug>/settings/vast, or set the env var directly.",
            }

        # Pre-flight checks before we spend any money.
        if shutil.which("ssh") is None or shutil.which("scp") is None:
            return {
                "status": "error",
                "reason": "ssh/scp not found on PATH. Required for the Vast adapter.",
            }
        if not os.path.exists(cfg["ssh_key_path"]):
            return {
                "status": "error",
                "reason": (
                    f"SSH key not found at {cfg['ssh_key_path']}. Set VAST_SSH_KEY_PATH "
                    f"or generate one with `ssh-keygen -t ed25519 -f {cfg['ssh_key_path']}`."
                ),
            }

        _emit(
            "vast.starting",
            gpu_types=cfg["gpu_types"],
            num_gpus=cfg["num_gpus"],
            max_hourly_cost=cfg["max_hourly_cost"],
        )

        try:
            _ensure_ssh_key_registered(cfg["api_key"], cfg["ssh_key_path"])
        except Exception as e:
            return {"status": "error", "reason": f"could not register SSH key with Vast: {e}"}

        # If the UI pre-selected a specific offer, provision that one
        # and skip the search + cascading-diagnostic path entirely. The
        # user has already seen the full market and chosen; we honour
        # their choice even if rentable=false (the provision call will
        # surface a clean error if the host has since gone away).
        if cfg.get("offer_id"):
            _emit("vast.using_preselected_offer", offer_id=cfg["offer_id"])
            offer = _fetch_offer_by_id(cfg["api_key"], int(cfg["offer_id"]))
            if offer is None:
                return {
                    "status": "error",
                    "reason": (
                        f"Pre-selected Vast offer id={cfg['offer_id']} is no longer "
                        f"available — host may have gone offline or been rented out. "
                        f"Open the run composer and pick another GPU."
                    ),
                }
        else:
            _emit("vast.searching")
            try:
                offer = _find_cheapest_offer(cfg["api_key"], cfg)
            except Exception as e:
                return {"status": "error", "reason": f"Vast offer search failed: {e}"}

        if offer is None:
            # The strict under-cap search came back empty. Escalate
            # through progressively looser searches so the user gets
            # actionable info instead of "nothing's available":
            #
            #   1. Strict + cap removed   — same filters, just no cost
            #      ceiling. Tells the user "raise the cap to X".
            #   2. Permissive + cap kept  — drop verified / rentable /
            #      on-demand. Surfaces interruptible / unverified hosts
            #      under the cap. Tells the user "your filter is too
            #      strict, but cheap hosts exist".
            #   3. Permissive + no cap    — last-resort look at the
            #      entire market. Tells the user "the market has X
            #      machines that match your GPU + count, here's the
            #      cheapest, but they're all gated by safety filters
            #      and/or above the cap".
            #
            # We never auto-provision from these relaxed queries —
            # they're read-only diagnostics. Provisioning still uses the
            # strict path.
            def _safe_search(**kwargs: Any) -> list[dict[str, Any]]:
                try:
                    return _search_offers(cfg["api_key"], cfg, per_type_limit=5, **kwargs)
                except Exception as e:  # pragma: no cover — diagnostics shouldn't kill the run
                    _emit_log(f"market overview lookup failed: {e}")
                    return []

            strict_no_cap = _safe_search(include_over_cap=True)
            if strict_no_cap:
                table = _format_offers_table(strict_no_cap[:8])
                reason = (
                    f"No Vast offers under ${cfg['max_hourly_cost']:.2f}/hr for "
                    f"gpu_types={cfg['gpu_types']} (num_gpus={cfg['num_gpus']}).\n\n"
                    f"Cheapest available right now (sorted by $/hr):\n{table}\n\n"
                    f"Raise the cost cap to at least ${float(strict_no_cap[0].get('dph_total', 0)):.3f}/hr "
                    f"in /settings/vast, or widen the GPU set to include more types."
                )
                return {"status": "error", "reason": reason}

            # Strict-no-cap was empty too — drop the safety filters.
            permissive_capped = _safe_search(permissive=True, include_over_cap=False)
            if permissive_capped:
                table = _format_offers_table(permissive_capped[:8])
                reason = (
                    f"No rentable on-demand offers under ${cfg['max_hourly_cost']:.2f}/hr for "
                    f"gpu_types={cfg['gpu_types']}, but the broader market has matches "
                    f"(some may be interruptible or not currently accepting rentals):\n\n"
                    f"{table}"
                )
                return {"status": "error", "reason": reason}

            permissive_no_cap = _safe_search(permissive=True, include_over_cap=True)
            if permissive_no_cap:
                cheapest = float(permissive_no_cap[0].get("dph_total", 0))
                table = _format_offers_table(permissive_no_cap[:8])
                reason = (
                    f"No usable Vast offers for gpu_types={cfg['gpu_types']} (num_gpus={cfg['num_gpus']}) "
                    f"under your ${cfg['max_hourly_cost']:.2f}/hr cap. The broader market has these "
                    f"(any price, any rentable state):\n\n"
                    f"{table}\n\n"
                    f"Raise the cost cap to at least ${cheapest:.3f}/hr in /settings/vast to allow "
                    f"these offers, or widen the GPU set."
                )
                return {"status": "error", "reason": reason}

            # Genuinely empty across every relaxation — Vast has zero
            # matching machines worldwide, OR something's broken with
            # the API call (auth, region, rate limit). Run two extra
            # diagnostics so the error message tells the user which:
            #
            #   - baseline /bundles/ with NO filters: confirms endpoint
            #     reachability and that the response shape we expect
            #     still exists. If this also returns zero, the issue is
            #     auth/endpoint, not our filters.
            #   - re-run the permissive+no-cap search with emit_debug=1:
            #     fires `vast.debug` events with the exact query string,
            #     HTTP status, and body keys so the user can share what
            #     Vast is actually returning.
            baseline = _baseline_market_check(cfg["api_key"])
            try:
                _search_offers(
                    cfg["api_key"],
                    cfg,
                    permissive=True,
                    include_over_cap=True,
                    per_type_limit=1,
                    emit_debug=True,
                )
            except Exception as e:
                _emit_log(f"debug re-search failed: {e}")

            if baseline["ok"] and baseline["offer_count"] > 0:
                diag = (
                    f"Baseline /bundles/ (no filters) returned "
                    f"{baseline['offer_count']} offers — endpoint and auth are fine. "
                    f"Our filter shape is excluding everything. See `vast.debug` "
                    f"events above for the exact query string + HTTP status returned "
                    f"by Vast — that tells us whether the gpu_name didn't match "
                    f"Vast's catalog spelling, or whether a filter operator was rejected."
                )
            elif baseline["ok"]:
                diag = (
                    "Baseline /bundles/ returned 0 offers — Vast's entire market "
                    "appears empty from your account. Likely a region/quota restriction. "
                    "Try logging into cloud.vast.ai directly to check account status."
                )
            else:
                diag = (
                    f"Baseline /bundles/ check failed: {baseline['hint']}. "
                    f"Likely an auth or endpoint problem — verify your VAST_API_KEY "
                    f"at /settings/vast and click 'Test connection'."
                )

            reason = (
                f"Vast returned zero offers for gpu_types={cfg['gpu_types']} "
                f"(num_gpus={cfg['num_gpus']}) even with all filters relaxed.\n\n"
                f"{diag}"
            )
            return {"status": "error", "reason": reason}

        offer_id = offer.get("id") or offer.get("ask_contract_id")
        dph = float(offer.get("dph_total", 0))
        gpu_name = offer.get("gpu_name", "?")
        location = offer.get("geolocation", "?")
        _emit(
            "vast.offer_selected",
            offer_id=offer_id,
            gpu=gpu_name,
            dollars_per_hour=dph,
            location=location,
        )

        try:
            provisioned = _provision_instance(cfg["api_key"], offer_id, cfg)
        except Exception as e:
            return {"status": "error", "reason": f"Vast provision call failed: {e}"}

        instance_id = (
            provisioned.get("new_contract")
            or provisioned.get("instance_id")
            or provisioned.get("id")
        )
        if not instance_id:
            return {
                "status": "error",
                "reason": f"Vast provision response missing instance id: {provisioned}",
            }

        _emit("vast.provisioned", instance_id=instance_id, dollars_per_hour=dph)
        start_time = time.time()

        # Install a signal handler so Ctrl-C / SIGTERM from the API still
        # triggers the teardown. We re-raise after cleanup so the calling
        # frame's `finally` block runs and the API marks the run cancelled.
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _on_term(signum: int, _frame: Any) -> None:
            _emit_log(f"signal {signum} received — destroying instance and exiting")
            _destroy_instance(cfg["api_key"], int(instance_id))
            os._exit(143)

        signal.signal(signal.SIGTERM, _on_term)

        results: dict[str, Any] = {"status": "error", "reason": "did not complete"}
        try:
            inst = _wait_for_ssh(cfg["api_key"], int(instance_id))
            ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
            ssh_port = int(inst.get("ssh_port"))
            _emit("vast.ssh_ready", host=ssh_host, port=ssh_port)

            _wait_for_ssh_connection(cfg["ssh_key_path"], ssh_host, ssh_port)

            _emit("vast.uploading_project")
            tarball = _make_project_tarball(project_root)
            try:
                _scp_to_remote(cfg["ssh_key_path"], ssh_host, ssh_port, tarball, "/workspace.tgz")
            finally:
                try:
                    tarball.unlink()
                except Exception:
                    pass

            # Untar + install pfnstudio + project deps. The remote runs
            # the local adapter with PFNSTUDIO_JSON_PROGRESS=1 so its
            # progress events flow to us unchanged.
            #
            # Blackwell self-heal: before installing OUR packages we
            # check whether the image's PyTorch has a CUDA kernel for
            # the rented card's compute capability. The image we ship
            # (pytorch/pytorch:2.4.0-cuda12.1) was built before
            # Blackwell (sm_100, sm_120) existed; renting an RTX 5070
            # / 5080 / 5090 / B100 / B200 against that image gets the
            # operator a 'CUDA error: no kernel image is available'
            # crash inside model.to(device) on the first training
            # step. The fix is a torch upgrade to the cu130 nightly
            # which includes Blackwell kernels.
            #
            # We don't always upgrade — that would add ~60s + 2 GB of
            # bandwidth to every Vast dispatch regardless of card.
            # The check is cheap: grep torch.cuda.get_arch_list() for
            # the rented card's sm_XX tag. On 3090/4090/A100/H100 the
            # tag is already in the list and we skip; on Blackwell+
            # we upgrade. SSH non-interactive sessions don't auto-
            # activate conda, so we explicitly target the system
            # python3 with --break-system-packages — which is what
            # the rest of install_cmd uses too.
            run_rel = run_yaml.relative_to(project_root)
            install_cmd = (
                "set -euo pipefail; "
                "mkdir -p /workspace && cd /workspace && tar -xzf /workspace.tgz && "
                # ── Blackwell self-heal: detect missing CUDA kernel ────────
                # The python3 -c command prints the card's sm_XX; the
                # grep checks if torch was built for it. Both short-
                # circuit if torch isn't importable (degenerate
                # image — let the install below fail loudly).
                "CARD_CAP=$(python3 -c 'import torch; c=torch.cuda.get_device_capability(0); print(f\"sm_{c[0]}{c[1]}\")' 2>/dev/null || echo ''); "
                'if [ -n "$CARD_CAP" ] && ! python3 -c \'import torch; print(torch.cuda.get_arch_list())\' 2>/dev/null | grep -q "$CARD_CAP"; then '
                '  echo "[vast] card needs $CARD_CAP, system torch missing it — upgrading torch (Blackwell+)."; '
                "  python3 -m pip install --quiet --upgrade --break-system-packages --pre --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu130; "
                "  echo \"[vast] torch upgrade done: $(python3 -c 'import torch; print(torch.__version__)')\"; "
                "fi; "
                # ── End Blackwell self-heal ──────────────────────────────
                "python3 -m pip install --quiet -e ./pfnstudio-core -e ./pfnstudio-cli && "
                "if [ -f ./project/requirements.txt ]; then python3 -m pip install --quiet -r ./project/requirements.txt; fi"
            )
            _emit("vast.installing")
            rc = _ssh_stream(cfg["ssh_key_path"], ssh_host, ssh_port, install_cmd)
            if rc != 0:
                results = {
                    "status": "error",
                    "reason": f"remote install failed with exit code {rc}",
                }
                return results

            run_cmd = (
                "set -euo pipefail; cd /workspace/project && "
                "PFNSTUDIO_JSON_PROGRESS=1 "
                f"python3 -m pfnstudio run '{run_rel}' --target local"
            )
            _emit("vast.training_started")
            rc = _ssh_stream(cfg["ssh_key_path"], ssh_host, ssh_port, run_cmd, forward_events=True)
            if rc != 0:
                results = {
                    "status": "error",
                    "reason": f"remote training failed with exit code {rc}",
                }
                return results

            # Pull results back. The local adapter writes results into the
            # run YAML or as a sibling .json — try both common locations.
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
                local_results = Path(fh.name)
            try:
                got = _scp_from_remote(
                    cfg["ssh_key_path"],
                    ssh_host,
                    ssh_port,
                    f"/workspace/project/runs/{run_rel.stem}.results.json",
                    local_results,
                )
                if got:
                    results = json.loads(local_results.read_text())
                else:
                    # Training succeeded but no results file — the local
                    # adapter may have returned results directly via the
                    # `done` event which we've already forwarded.
                    results = {
                        "status": "completed",
                        "reason": "remote training finished; no results.json on disk",
                    }
            finally:
                try:
                    local_results.unlink()
                except Exception:
                    pass

            elapsed = time.time() - start_time
            cost = (elapsed / 3600.0) * dph
            _emit(
                "vast.training_done", elapsed_seconds=elapsed, estimated_cost_dollars=round(cost, 4)
            )
            results.setdefault("compute", {})
            if isinstance(results.get("compute"), dict):
                results["compute"].update(
                    {
                        "provider": "vast",
                        "instance_id": instance_id,
                        "gpu": gpu_name,
                        "dollars_per_hour": dph,
                        "elapsed_seconds": round(elapsed, 2),
                        "estimated_cost_dollars": round(cost, 4),
                    }
                )
            return results
        finally:
            signal.signal(signal.SIGTERM, original_sigterm)
            _emit("vast.destroying", instance_id=instance_id)
            _destroy_instance(cfg["api_key"], int(instance_id))
            _emit("vast.destroyed", instance_id=instance_id)
