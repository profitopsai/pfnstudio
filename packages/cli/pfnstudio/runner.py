"""Self-hosted runner — pull jobs from PFN Studio cloud and run them locally.

The runner pattern mirrors GitHub Actions' self-hosted runner: outbound-
only long-poll, no inbound port opens on the user's machine. Works
behind NAT/firewall, no SSH key dance, registers once and stays
addressable across reboots.

Wire protocol:

  POST /runner/capabilities                       on startup: torch/CUDA/disk
  GET  /runner/jobs/next                          long-poll for next claimable
                                                  job (+ pendingPublishes)
  GET  /runner/jobs/:runId/bundle                 tar.gz of the project source
  POST /runner/jobs/:runId/heartbeat              every HEARTBEAT_S seconds
  POST /runner/jobs/:runId/events                 stream JSON events as they
                                                  happen
  POST /runner/jobs/:runId/artifact               checkpoint tar.gz on success
                                                  OR publish-from-runner
  POST /runner/jobs/:runId/result                 final upload (status+events)

  GET  /runner/serve-targets                      Phase 2: deployments this
                                                  runner should be serving
  POST /runner/deployments/:id/serve-ready        local serve subprocess up
  GET  /runner/predict-requests                   short-poll for pending
                                                  predict requests
  POST /runner/predict-requests/:id/response      post local serve's response

Auth header: ``Authorization: Token psr_<48-hex>``. Token issued exactly
once when a user clicks "Add a runner" in /settings/runners; stored at
``~/.pfnstudio/runner.json`` with chmod 0600.

Scope of this module:
  * register / doctor / status / artifacts / forget / start subcommands
  * Long-poll loop with SIGINT / SIGTERM graceful shutdown
  * Per-job heartbeat thread
  * Per-job bundle download → extract to a temp dir → subprocess
    ``pfnstudio run`` in it → MOVE checkpoint to local persistent
    home → tear down the temp dir → post final result with
    artifactRef={kind:'local-runner',...} so cloud knows where it is
    without ever seeing the bytes
  * JSON-event piping back to the cloud as Run.events
  * Live state file at ``~/.pfnstudio/runner-state.json`` (pid,
    last poll, current job + bundle dir + trainer pid) consumed
    by ``runner status`` for at-a-glance "what's it doing" output
  * Local artifact registry at ``~/.pfnstudio/runs/<runId>/`` —
    `runner artifacts` lists, `runner forget` deletes. Auto-upload
    on training-success was removed in 0.8.6: the model stays on
    the runner until the operator clicks "Publish from runner →"
    in Studio (Phase 1.5b adds the publish-request poll path)
  * Phase 2 (0.8.8) runner-served endpoints: two background threads
    serve_manager (every 5 s reconciles local pfnstudio-serve
    subprocesses with the cloud's serve-targets list) and
    predict_poller (every 250 ms forwards predict requests to the
    matching local serve, posts the response back). Inference data
    never leaves the runner box.

Still out of scope:
  * Per-job venv isolation (we run in the runner's own Python env).
  * Docker executor mode.
"""

from __future__ import annotations

import io
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import __version__

console = Console()


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN, +/-Inf) with None.

    JSON has no representation for NaN/Inf, so ``requests`` raises
    ``InvalidJSONError`` if any slips into a payload — which would
    discard an ENTIRE run report (events + final metrics) over a single
    bad metric. A trainer/eval producing NaN is a real signal, so we
    keep it visible: null renders as "no value" in Studio rather than
    losing the whole result. Applied at event ingestion so both the live
    stream and the final result post are safe.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# Where the bearer token + cloud URL live. Permissions are 0600 so other
# users on the same machine can't steal the token.
CONFIG_PATH = Path.home() / ".pfnstudio" / "runner.json"

# Live state of the long-poll loop, used by `pfnstudio runner status`
# so the operator can see what the runner is doing without having to
# ssh in and grep `ps`. Updated at significant events: boot, after
# each poll, on job claim, on job finish, on shutdown. 0600 like the
# config — contains nothing secret but matching perms keeps both
# files invisible to other UIDs on shared machines.
STATE_PATH = Path.home() / ".pfnstudio" / "runner-state.json"

# Where trained checkpoints land. The runner keeps them locally by
# default — Studio cloud only ever sees a {kind:'local-runner',...}
# reference in Run.artifactRef until the user explicitly clicks
# "Publish from runner →", at which point the runner uploads via the
# existing /artifact endpoint. This is the data-residency story: the
# model never leaves this box unless the operator says so.
#
# Layout per completed run:
#   ~/.pfnstudio/runs/<runId>/checkpoint/    model.pt, topology.json, …
#   ~/.pfnstudio/runs/<runId>/meta.json      {runSlug, completedAt, sizeBytes, cloudUrl}
RUNS_ROOT = Path.home() / ".pfnstudio" / "runs"

# How often we tell the cloud "yes I'm still running this job". Has to
# be shorter than the cloud's HEARTBEAT_STALE_S threshold (90s) so the
# job's claim doesn't expire mid-training.
HEARTBEAT_INTERVAL_S = 30

# Long-poll request timeout. The cloud responds within ~25s; we give it
# 35s of slack to cover network latency + reverse-proxy buffering.
POLL_TIMEOUT_S = 35

# Backoff schedule when the cloud is unreachable / 5xx. Capped so a
# transient blip doesn't push the next attempt arbitrarily far.
BACKOFF_LADDER_S = (2, 5, 15, 30, 60)

# How often the serve manager reconciles its running subprocesses with
# cloud's serve-targets list. 5 s is fine — endpoint create/delete is
# a deliberate user action, not high-frequency.
SERVE_TARGETS_POLL_S = 5

# How often the predict poller asks for pending requests. Tight (250 ms)
# because end-user-facing latency is the sum of this + local serve
# inference + posting back. Postgres queries are indexed.
PREDICT_POLL_S = 0.25

# Per-predict-request timeout when forwarding to the local serve
# subprocess. Most PFN inference is sub-second; 60 s catches CPU
# warm-up + the rare big batch.
PREDICT_FORWARD_TIMEOUT_S = 60

runner_app = typer.Typer(
    name="runner",
    help="Self-hosted runner — register this machine and pull jobs from PFN Studio.",
    no_args_is_help=True,
)


# ── Config helpers ────────────────────────────────────────────────────


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        console.print(
            "[red]Not registered.[/red] Run [bold]pfnstudio runner register --token psr_...[/bold]"
        )
        raise typer.Exit(2)
    try:
        return json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Corrupt config at {CONFIG_PATH}:[/red] {e}")
        raise typer.Exit(2) from e


def _save_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    # Token must NOT be world-readable — chmod after write so the file
    # never briefly has loose perms.
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        # On Windows the chmod is best-effort; the token is still in the
        # user's profile directory which has reasonable defaults.
        pass


# ── State file helpers ────────────────────────────────────────────────


def _read_state() -> dict[str, Any]:
    """Best-effort read of the state file. Returns {} if absent or
    corrupt — `runner status` interprets missing fields as 'unknown'."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _update_state(**fields: Any) -> None:
    """Merge fields into the state file. NEVER raises — state is
    informational; a permissions blip or full disk shouldn't take down
    the long-poll loop. Caller passes keys to set; values of None
    explicitly null out fields (currentJob=None on job finish, etc.)."""
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        cur = _read_state()
        cur.update(fields)
        STATE_PATH.write_text(json.dumps(cur, indent=2, default=str))
        try:
            STATE_PATH.chmod(0o600)
        except OSError:
            pass
    except OSError:
        pass


def _persist_checkpoint(
    bundle_dir: Path, run_id: str, run_slug: str, cloud_url: str
) -> tuple[Path, int] | None:
    """Move the trained checkpoint from the temp bundle dir to its
    durable home under RUNS_ROOT/<runId>/. Returns (path, sizeBytes)
    on success; None if there's nothing to persist (no checkpoint dir
    was written — likely a skipped or failed run).

    Uses shutil.move so an existing local dir from a prior re-run gets
    cleanly replaced. A previous attempt's leftovers (rare — same runId
    re-running on the same box) would otherwise grow stale.
    """
    src = bundle_dir / "checkpoint"
    if not src.is_dir():
        return None
    dst_root = RUNS_ROOT / run_id
    dst_ckpt = dst_root / "checkpoint"
    try:
        RUNS_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        if dst_ckpt.exists():
            shutil.rmtree(dst_ckpt, ignore_errors=True)
        dst_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_ckpt))
    except OSError as e:
        console.print(f"[yellow]Could not persist checkpoint locally:[/yellow] {e}")
        return None

    size = 0
    for p in dst_ckpt.rglob("*"):
        try:
            if p.is_file():
                size += p.stat().st_size
        except OSError:
            pass

    # Persist the entire project tree from the bundle alongside the
    # checkpoint. ModelLoader reads priors/<id>/prior.yaml + prior.py,
    # models/<id>.yaml, evals/<id>.yaml at load time to reconstruct
    # the model — without these, serve fails with
    # "FileNotFoundError: model.yaml not found at <cwd>/models/...".
    #
    # The cloud's streamRunnerBundle already writes the canonical
    # project layout into the bundle root with normalized refs; just
    # copy the project dirs across before rmtree wipes the bundle.
    # run.yaml is also stashed at the runs/<slug>.yaml path the serve
    # CLI's project-auto-detect logic expects (parent.parent of the
    # manifest when its parent is named 'runs').
    # adapters/ + blocks/ matter for serving too: an adapter base model (TCPFN,
    # …) serves through adapters/<slug>.py — discover_in_project must find it in
    # the persisted run dir, or serve fails at load with
    # "No module named 'pfnstudio_core.training.adapters.<name>'".
    for sub in ("priors", "models", "evals", "runs", "adapters", "blocks"):
        src = bundle_dir / sub
        if not src.is_dir():
            continue
        dst = dst_root / sub
        try:
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(str(src), str(dst))
        except OSError as e:
            console.print(f"[yellow]Could not persist {sub}/ locally:[/yellow] {e}")
    # Also keep run.yaml at the top level for compatibility with the
    # `pfnstudio serve run.yaml` invocation in _spawn_serve_for.
    src_yaml = bundle_dir / "runs" / f"{run_slug}.yaml"
    if src_yaml.exists() and not (dst_root / "run.yaml").exists():
        try:
            shutil.copy2(str(src_yaml), str(dst_root / "run.yaml"))
        except OSError:
            pass

    # Stash a meta.json so `runner artifacts` can list them without
    # needing the cloud, and `runner forget` can age-out old runs.
    meta = {
        "runId": run_id,
        "runSlug": run_slug,
        "completedAt": _ts(),
        "sizeBytes": size,
        "cloudUrl": cloud_url,
    }
    try:
        (dst_root / "meta.json").write_text(json.dumps(meta, indent=2))
    except OSError:
        pass

    return dst_ckpt, size


# ── Phase 2: serve-lifecycle + predict polling ──────────────────────


def _spawn_serve_for(run_id: str, run_slug: str) -> tuple[subprocess.Popen[str], int] | None:
    """Boot a ``pfnstudio serve`` subprocess for one run on a random
    loopback port. Returns (proc, port) on success; None if the local
    checkpoint is missing or the subprocess fails to declare a port
    within 15 s (which generally means torch / dep load failed).

    The serve binary writes a single JSON line to stdout once the HTTP
    server is up: ``{"event":"ready","port":N,...}``. We block on that
    line so the registry only contains servers actually accepting
    requests.
    """
    run_dir = RUNS_ROOT / run_id
    ckpt = run_dir / "checkpoint"
    if not ckpt.is_dir():
        console.print(
            f"[yellow]Cannot serve {run_slug}: local checkpoint missing at "
            f"{ckpt}. Operator may have run `runner forget`. Re-train to "
            f"reproduce it.[/yellow]"
        )
        return None

    # `pfnstudio serve` needs a real run.yaml — the manifest the
    # trainer used, carrying prior/model/evals references. It gets
    # copied here by _persist_checkpoint when training succeeds (the
    # cloud's bundle ships runs/<slug>.yaml). Older runs persisted
    # before this fix landed won't have it; fall back to a stub but
    # log loudly so the operator knows to re-train.
    manifest = run_dir / "run.yaml"
    if not manifest.exists():
        console.print(
            f"[yellow]Cannot serve {run_slug}: no run.yaml at {manifest}. "
            f"This means the run was trained on a pre-0.8.11 runner that "
            f"didn't persist the manifest. Re-run training to regenerate.[/yellow]"
        )
        return None

    # Drain subprocess output in real time so we can:
    #   (a) detect the {event:'ready',port:N} line and unblock
    #   (b) capture stderr / non-JSON lines for the failure path
    #       (which previously got silently swallowed — the "giving
    #       up after 15s" message had no diagnostics attached).
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pfnstudio",
            "serve",
            str(manifest),
            "--checkpoint",
            str(ckpt),
            "--project",
            # Pin project_root to the persisted dir. Without --project,
            # serve falls back to Path.cwd() (which is wherever the
            # operator started 'pfnstudio runner start' from — often
            # /home/user or some random path) and ModelLoader looks
            # for models/<id>.yaml THERE, not next to the checkpoint.
            str(run_dir),
            "--port",
            "0",  # let OS pick; serve emits the port on the ready event.
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    port: int | None = None
    captured: list[str] = []
    deadline = time.time() + 15
    assert proc.stdout is not None
    while time.time() < deadline and port is None:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break  # subprocess exited
            continue
        captured.append(line.rstrip("\n"))
        try:
            ev = json.loads(line)
            if ev.get("event") == "ready" and isinstance(ev.get("port"), int):
                port = ev["port"]
                break
            if ev.get("event") == "error":
                # Loader emitted a structured error — surface immediately.
                console.print(
                    f"[red]Serve for {run_slug} reported an error at "
                    f"stage={ev.get('stage', '?')}:[/red] {ev.get('message', line)}"
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                return None
        except (json.JSONDecodeError, TypeError):
            continue
    if port is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        exit_code = proc.poll()
        # Dump the captured output so the operator can see WHY the
        # subprocess didn't reach ready. Trim to last ~30 lines — full
        # tracebacks are noisy but the relevant lines are always near
        # the tail.
        tail = "\n  ".join(captured[-30:]) if captured else "(no output captured)"
        console.print(
            f"[yellow]Serve for {run_slug} didn't emit a ready event within 15 s "
            f"(exit_code={exit_code}). Subprocess output:[/yellow]\n  {tail}"
        )
        return None
    # Drain the rest of stdout in a daemon thread so the pipe buffer
    # doesn't block the subprocess once it's serving.
    threading.Thread(
        target=lambda: [_ for _ in iter(proc.stdout.readline, "")],  # type: ignore[union-attr]
        daemon=True,
    ).start()
    return proc, port


def _serve_manager_loop(
    base: str,
    headers: dict[str, str],
    registry: dict[str, dict[str, Any]],
    lock: threading.Lock,
    stop_flag: threading.Event,
) -> None:
    """Reconcile local pfnstudio-serve subprocesses with the cloud's
    serve-targets list. Registry shape:
        {deploymentId: {runId, runSlug, proc, port, ready}}

    Per-deployment failure backoff: a deployment whose spawn fails
    gets retried with exponentially-spaced cooldowns (5s, 30s, 5min,
    capped) so a broken artifact doesn't flood the terminal with the
    same loader error every reconcile.
    """
    # deploymentId → {fails, nextAttemptAt} for the backoff tracker
    failures: dict[str, dict[str, float]] = {}
    while not stop_flag.is_set():
        try:
            r = requests.get(
                f"{base}/runner/serve-targets",
                headers=headers,
                timeout=10,
            )
            if r.ok:
                targets = r.json().get("targets") or []
                target_ids = {t["deploymentId"] for t in targets}
                # Drop backoff state for deployments that have gone
                # away — covers the operator "delete the failing
                # deployment + create a new one with a different id".
                for stale in [d for d in failures if d not in target_ids]:
                    failures.pop(stale, None)
                # Start any new targets we don't have a serve for yet.
                now = time.time()
                for t in targets:
                    dep_id = t["deploymentId"]
                    run_id = t["runId"]
                    if dep_id in registry:
                        continue
                    fail_state = failures.get(dep_id)
                    if fail_state and now < fail_state["nextAttemptAt"]:
                        continue  # in cooldown
                    if dep_id in registry:
                        continue
                    # Read the persisted run.yaml's `id` for a human-
                    # friendly slug, falling back to the runId tail if
                    # the persistence is missing. run_id[:8] would
                    # collide across runs that share a CUID prefix —
                    # surprisingly common when a re-train reuses the
                    # same time-bucketed id space.
                    persisted_yaml = RUNS_ROOT / run_id / "run.yaml"
                    run_slug = run_id[-8:]
                    if persisted_yaml.exists():
                        try:
                            doc = yaml.safe_load(persisted_yaml.read_text())
                            if isinstance(doc, dict) and isinstance(doc.get("id"), str):
                                run_slug = doc["id"]
                        except (OSError, yaml.YAMLError):
                            pass
                    spawned = _spawn_serve_for(run_id, run_slug)
                    if not spawned:
                        # Exponential cooldown so a broken artifact
                        # doesn't reprint the same loader error every
                        # 5s. 5s → 30s → 5min → 5min ...
                        prev = failures.get(dep_id, {"fails": 0})
                        n = int(prev.get("fails", 0)) + 1
                        backoff_s = (5, 30, 300, 300, 300)[min(n - 1, 4)]
                        failures[dep_id] = {
                            "fails": n,
                            "nextAttemptAt": time.time() + backoff_s,
                        }
                        if n == 1:
                            console.print(
                                f"[dim]Backing off serve retries for "
                                f"{run_slug} (next attempt in {backoff_s}s).[/dim]"
                            )
                        continue
                    failures.pop(dep_id, None)
                    proc, port = spawned
                    with lock:
                        registry[dep_id] = {
                            "runId": run_id,
                            "proc": proc,
                            "port": port,
                            "ready": False,
                        }
                    # Tell the cloud the serve is up so its Deployment
                    # flips to status='ready' and the predict UI unlocks.
                    try:
                        ack = requests.post(
                            f"{base}/runner/deployments/{dep_id}/serve-ready",
                            headers=headers,
                            timeout=10,
                        )
                        if ack.ok:
                            with lock:
                                registry[dep_id]["ready"] = True
                            console.print(
                                f"[green]✓[/green] serving deployment {dep_id[:12]}… "
                                f"on localhost:{port}"
                            )
                    except requests.RequestException:
                        pass
                # Stop any serves whose deployments were deleted.
                with lock:
                    to_kill = [d for d in registry if d not in target_ids]
                for d in to_kill:
                    with lock:
                        entry = registry.pop(d, None)
                    if entry:
                        try:
                            entry["proc"].terminate()
                        except ProcessLookupError:
                            pass
                        console.print(f"[dim]stopped serve for deployment {d[:12]}…[/dim]")
        except requests.RequestException:
            pass
        if stop_flag.wait(SERVE_TARGETS_POLL_S):
            break
    # Shutdown — kill any running serves.
    with lock:
        for entry in registry.values():
            try:
                entry["proc"].terminate()
            except ProcessLookupError:
                pass


def _predict_poll_loop(
    base: str,
    headers: dict[str, str],
    registry: dict[str, dict[str, Any]],
    lock: threading.Lock,
    stop_flag: threading.Event,
) -> None:
    """Short-poll for pending predict requests, forward each to its
    deployment's local serve subprocess, post the response back."""
    while not stop_flag.is_set():
        try:
            r = requests.get(
                f"{base}/runner/predict-requests",
                headers=headers,
                timeout=10,
            )
            if r.ok:
                for req in r.json().get("requests") or []:
                    _forward_predict(base, headers, registry, lock, req)
        except requests.RequestException:
            pass
        if stop_flag.wait(PREDICT_POLL_S):
            break


def _forward_predict(
    base: str,
    headers: dict[str, str],
    registry: dict[str, dict[str, Any]],
    lock: threading.Lock,
    req: dict[str, Any],
) -> None:
    """Forward one claimed predict request to the local serve, post
    the response back to cloud. Best-effort: errors get reported back
    as responseError so the caller sees something coherent."""
    req_id = req.get("id")
    dep_id = req.get("deploymentId")
    payload = req.get("payload") or {}
    if not isinstance(req_id, str) or not isinstance(dep_id, str):
        return

    with lock:
        entry = registry.get(dep_id)
    if not entry:
        _post_predict_response(
            base,
            headers,
            req_id,
            error="no local serve for this deployment (race with shutdown?)",
            status=503,
        )
        return

    port = entry["port"]
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/predict",
            json=payload,
            timeout=PREDICT_FORWARD_TIMEOUT_S,
        )
        _post_predict_response(
            base,
            headers,
            req_id,
            status=r.status_code,
            payload=_safe_json(r),
        )
    except requests.RequestException as e:
        _post_predict_response(
            base,
            headers,
            req_id,
            error=f"{type(e).__name__}: {e}",
            status=502,
        )


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text[:4000]}


def _post_predict_response(
    base: str,
    headers: dict[str, str],
    req_id: str,
    *,
    status: int = 200,
    payload: Any = None,
    error: str | None = None,
) -> None:
    body: dict[str, Any] = {"status": status}
    if payload is not None:
        # Model outputs are the most likely source of NaN/Inf — scrub so
        # a non-finite prediction can't fail the whole response post.
        body["payload"] = _json_safe(payload)
    if error is not None:
        body["error"] = error
    try:
        requests.post(
            f"{base}/runner/predict-requests/{req_id}/response",
            headers=headers,
            json=body,
            timeout=15,
        )
    except requests.RequestException:
        # Cloud's 30 s wait will time out; nothing we can do here.
        pass


def _pid_alive(pid: int) -> bool:
    """Cross-platform 'is this PID still running' check. POSIX uses
    signal 0 (kernel checks the process exists without delivering
    anything); Windows' os.kill rejects signal 0 but raises a
    different error if the PID is gone."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by another user / under a
        # systemd-protected scope. From our POV "alive enough".
        return True
    except OSError:
        return False


# ── Capability reporting ──────────────────────────────────────────────


def _core_version() -> str | None:
    """Installed pfnstudio-core version (the training engine). Drifts
    independently of this CLI — a runner can be on a current CLI but a
    stale core (or vice-versa), which is exactly the mismatch that's
    invisible when only the CLI version is reported. None if not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("pfnstudio-core")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        import pfnstudio_core  # type: ignore

        return getattr(pfnstudio_core, "__version__", None)
    except Exception:
        return None


def _capabilities() -> dict[str, Any]:
    """Self-report what this machine can run. Posted on register + start
    so the cloud's job dispatcher can gate incompatible jobs."""
    caps: dict[str, Any] = {
        "osArch": f"{platform.system().lower()}/{platform.machine()}",
        "pythonVersion": platform.python_version(),
        "pfnstudioVersion": __version__,
        "pfnstudioCoreVersion": _core_version(),
        "hostname": socket.gethostname(),
    }
    # torch is optional at install time (`pip install pfnstudio[torch]`).
    # Missing torch means training will fail; we still register so the
    # operator can see "this runner reported no torch" in /settings.
    try:
        import torch  # type: ignore

        caps["torchVersion"] = torch.__version__
        caps["cudaAvailable"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                caps["cudaVersion"] = torch.version.cuda
                caps["gpu"] = torch.cuda.get_device_name(0)
                caps["gpuMemoryGb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    1,
                )
            except Exception:
                # Any of these can raise on weird CUDA setups; capability
                # reporting must never crash the runner.
                pass
    except ImportError:
        caps["torchVersion"] = None
        caps["cudaAvailable"] = False
    # Free disk on the workspace partition (assume $HOME for now).
    try:
        free = shutil.disk_usage(Path.home()).free
        caps["diskFreeGb"] = round(free / (1024**3), 1)
    except OSError:
        pass
    return caps


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ── register ─────────────────────────────────────────────────────────


@runner_app.command()
def register(
    token: str = typer.Option(..., "--token", "-t", help="Runner token (starts with psr_)."),
    cloud_url: str = typer.Option(
        "https://api.pfnstudio.com",
        "--cloud-url",
        help="Base URL of the PFN Studio API (override for staging or self-hosted).",
    ),
) -> None:
    """Save the runner token + cloud URL and verify connectivity.

    The token is issued in /settings/runners → Add a runner → copy the
    token from the modal. Stored at ~/.pfnstudio/runner.json with 0600.
    Verifies the token by posting current capabilities to the cloud.
    """
    if not token.startswith("psr_"):
        console.print(
            "[red]Bad token format.[/red] Runner tokens start with 'psr_'. "
            "Did you copy a user API token (ps_…) by mistake?"
        )
        raise typer.Exit(2)

    cloud_url = cloud_url.rstrip("/")
    caps = _capabilities()
    try:
        r = requests.post(
            f"{cloud_url}/runner/capabilities",
            json={"capabilities": caps},
            headers={"Authorization": f"Token {token}"},
            timeout=15,
        )
    except requests.RequestException as e:
        console.print(f"[red]Could not reach {cloud_url}:[/red] {e}")
        raise typer.Exit(2) from e

    if r.status_code == 401:
        console.print(
            "[red]Token rejected.[/red] Either the runner was revoked, or this "
            "token is a user API token (ps_…) not a runner token (psr_…)."
        )
        raise typer.Exit(2)
    if not r.ok:
        console.print(f"[red]Cloud returned {r.status_code}:[/red] {r.text[:200]}")
        raise typer.Exit(2)

    _save_config({"token": token, "cloud_url": cloud_url})
    console.print(f"[green]✓ Registered.[/green] Token saved to [dim]{CONFIG_PATH}[/dim]")
    console.print("[dim]Reported capabilities:[/dim]")
    for k, v in caps.items():
        console.print(f"  [dim]{k}[/dim] = {v}")
    console.print()
    console.print("Start the long-poll loop with: [bold]pfnstudio runner start[/bold]")


# ── doctor ───────────────────────────────────────────────────────────


@runner_app.command()
def doctor() -> None:
    """Preflight checks — torch, CUDA, disk, network reachability."""
    caps = _capabilities()
    table = Table(title="Runner environment", show_header=False, box=None)
    table.add_column("key", style="dim")
    table.add_column("value")
    for k, v in caps.items():
        table.add_row(k, str(v))
    console.print(table)

    problems: list[str] = []
    if not caps.get("torchVersion"):
        problems.append(
            "torch not installed. Install with: [bold]pip install 'pfnstudio[torch]'[/bold]"
        )
    if not caps.get("cudaAvailable"):
        problems.append(
            "[yellow]No CUDA[/yellow] — runner can only execute CPU-feasible runs. "
            "GPU jobs dispatched to this runner will fail at training start."
        )
    disk_gb = caps.get("diskFreeGb")
    if isinstance(disk_gb, (int, float)) and disk_gb < 10:
        problems.append(
            f"Low disk: {disk_gb} GB free. Training writes checkpoints (often "
            "hundreds of MB); 10+ GB recommended."
        )

    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except Exception:
            cfg = None
        if cfg and cfg.get("cloud_url"):
            console.print(f"\n[dim]Pinging cloud at[/dim] {cfg['cloud_url']}…")
            try:
                # /health is unauth so we don't even need the token here.
                r = requests.get(f"{cfg['cloud_url']}/health", timeout=10)
                if r.ok:
                    console.print(f"[green]✓ Cloud reachable[/green] ({r.status_code})")
                else:
                    problems.append(f"Cloud responded {r.status_code} on /health")
            except requests.RequestException as e:
                problems.append(f"Cloud unreachable: {e}")
    else:
        console.print(
            f"\n[dim]No config at {CONFIG_PATH}.[/dim] "
            "Run [bold]pfnstudio runner register --token ...[/bold] first."
        )

    if problems:
        console.print()
        for p in problems:
            console.print(f"  [yellow]⚠[/yellow] {p}")
        raise typer.Exit(1 if any("cuda" not in p.lower() for p in problems) else 0)
    console.print("\n[green]All checks passed.[/green]")


# ── status ──────────────────────────────────────────────────────────


def _fmt_age(iso_str: str | None) -> str:
    """Render an ISO timestamp as a relative "Xs/m/h/d ago" string.
    Returns '?' for None / unparseable input."""
    if not iso_str:
        return "?"
    try:
        when = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "?"
    age = (datetime.now(timezone.utc) - when).total_seconds()
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age / 60)}m ago"
    if age < 86400:
        return f"{int(age / 3600)}h ago"
    return f"{int(age / 86400)}d ago"


def _dir_size_bytes(p: Path) -> int:
    """Quick recursive size of a directory tree. Returns 0 on any
    permission / not-found error — status is informational."""
    total = 0
    try:
        for entry in p.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n = n / 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


@runner_app.command()
def status() -> None:
    """Report whether the runner is healthy, what job it's running, and
    where to look on disk. Reads ``~/.pfnstudio/runner-state.json``
    (written by ``runner start``) + pings the cloud for liveness."""
    # ── Config + identity ──
    cfg: dict[str, Any] | None = None
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            cfg = None
    if cfg:
        console.print(f"[bold]Runner[/bold]          : {socket.gethostname()}")
        console.print(f"[bold]Cloud[/bold]           : {cfg.get('cloud_url', '?')}")
        console.print(f"[bold]Config[/bold]          : {CONFIG_PATH}")
        # Surface installed-vs-running version drift. If the operator
        # ran `pip install -U pfnstudio` while the long-poll loop was
        # going, the running process is still on the OLD version it
        # imported at boot. State file's pfnstudioVersionAtStart is
        # written by `start()` on first state-update; mismatch =>
        # restart needed.
        state_pre = _read_state()
        running_v = state_pre.get("pfnstudioVersionAtStart")
        if running_v and running_v != __version__:
            console.print(
                f"[bold]CLI[/bold]             : installed [bold]{__version__}[/bold] · "
                f"running [yellow]{running_v}[/yellow] "
                f"[yellow](restart the loop to pick up {__version__})[/yellow]"
            )
        else:
            console.print(f"[bold]CLI[/bold]             : {__version__}")
        # Core (training engine) + Python — reported so the operator (and
        # the cloud) can spot a CLI/core drift without guessing.
        core_v = _core_version()
        console.print(
            "[bold]Core[/bold]            : "
            + (core_v if core_v else "[yellow]not installed[/yellow] — training will fail")
        )
        console.print(f"[bold]Python[/bold]          : {platform.python_version()}")
    else:
        console.print(
            f"[bold]Config[/bold]          : [yellow]{CONFIG_PATH} missing[/yellow] — "
            "this runner isn't registered.\n"
            "                  Run [bold]pfnstudio runner register --token psr_…[/bold]"
        )
        return

    state = _read_state()

    # ── Long-poll loop liveness ──
    pid = state.get("pid")
    started_at = state.get("startedAt")
    last_poll_at = state.get("lastPollAt")
    stopped_at = state.get("stoppedAt")
    if isinstance(pid, int) and _pid_alive(pid) and not stopped_at:
        console.print(
            f"[bold]Long-poll loop[/bold]  : [green]✓ RUNNING[/green]  "
            f"(pid {pid}, up {_fmt_age(started_at)}, last poll {_fmt_age(last_poll_at)})"
        )
    elif isinstance(pid, int):
        console.print(
            f"[bold]Long-poll loop[/bold]  : [red]✗ NOT RUNNING[/red]\n"
            f"                  Last seen     : {_fmt_age(stopped_at or last_poll_at)} (pid {pid})\n"
            f"                  Start it with : [bold]pfnstudio runner start[/bold]"
        )
    else:
        console.print(
            "[bold]Long-poll loop[/bold]  : [red]✗ NOT RUNNING[/red]\n"
            "                  No state file from a previous run — never started?\n"
            "                  Start it with : [bold]pfnstudio runner start[/bold]"
        )

    # ── Cloud reachable ──
    if cfg.get("cloud_url"):
        try:
            r = requests.get(f"{cfg['cloud_url']}/health", timeout=5)
            if r.ok:
                console.print(
                    f"[bold]Cloud reachable[/bold] : [green]✓ HTTP {r.status_code}[/green]"
                )
            else:
                console.print(
                    f"[bold]Cloud reachable[/bold] : [yellow]HTTP {r.status_code}[/yellow]"
                )
        except requests.RequestException as e:
            console.print(f"[bold]Cloud reachable[/bold] : [red]✗ {e}[/red]")

    # ── Current job (or idle) ──
    job = state.get("currentJob") if isinstance(state.get("currentJob"), dict) else None
    if job:
        console.print(
            f"\n[bold]Current job[/bold]     : {job.get('runSlug', '?')} "
            f"({job.get('runId', '?')[:16]}…)"
        )
        console.print(f"  Started        : {_fmt_age(job.get('claimedAt'))}")
        bundle_dir = job.get("bundleDir")
        if bundle_dir:
            bp = Path(bundle_dir)
            if bp.exists():
                size = _dir_size_bytes(bp)
                console.print(f"  Bundle dir     : {bundle_dir}  ({_fmt_bytes(size)})")
                ckpt = bp / "checkpoint"
                if ckpt.is_dir():
                    files = sorted(p.name for p in ckpt.iterdir() if p.is_file())
                    size = _dir_size_bytes(ckpt)
                    flist = ", ".join(files[:3]) + (" …" if len(files) > 3 else "")
                    console.print(
                        f"  Checkpoint     : {_fmt_bytes(size)} ({flist or 'no files yet'})"
                    )
                else:
                    console.print("  Checkpoint     : not yet written")
            else:
                console.print(
                    f"  Bundle dir     : {bundle_dir}  [yellow](missing — stale state?)[/yellow]"
                )
        trainer_pid = job.get("trainerPid")
        if isinstance(trainer_pid, int):
            alive_str = "[green]alive[/green]" if _pid_alive(trainer_pid) else "[red]dead[/red]"
            console.print(f"  Trainer pid    : {trainer_pid} ({alive_str})")
    else:
        console.print(
            "\n[bold]Current job[/bold]     : none — runner idle, waiting for cloud to dispatch"
        )

    # ── Capabilities snapshot ──
    caps = _capabilities()
    cap_parts: list[str] = []
    if caps.get("torchVersion"):
        cap_parts.append(f"torch {caps['torchVersion']}")
    if caps.get("cudaAvailable"):
        gpu = caps.get("gpu", "")
        cv = caps.get("cudaVersion")
        cap_parts.append(f"cuda {cv or 'True'}{(' ' + gpu) if gpu else ''}")
    else:
        cap_parts.append("cuda False")
    if isinstance(caps.get("diskFreeGb"), (int, float)):
        cap_parts.append(f"{caps['diskFreeGb']} GB free")
    console.print(f"\n[bold]Capabilities[/bold]    : {', '.join(cap_parts) or '(none)'}")

    # ── Local artifacts summary ──
    local = _scan_local_artifacts()
    if local:
        total = sum(it["sizeBytes"] for it in local)
        console.print(
            f"[bold]Local artifacts[/bold] : {len(local)} run(s), {_fmt_bytes(total)}  "
            f"[dim](see `pfnstudio runner artifacts`)[/dim]"
        )


# ── artifacts / forget ──────────────────────────────────────────────


def _scan_local_artifacts() -> list[dict[str, Any]]:
    """Walk RUNS_ROOT for one directory per persisted run. Returns a
    list sorted newest-first, each entry containing what `runner
    artifacts` needs to display + what `runner forget` needs to
    confirm the target."""
    out: list[dict[str, Any]] = []
    if not RUNS_ROOT.is_dir():
        return out
    for entry in RUNS_ROOT.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
        ckpt = entry / "checkpoint"
        size = 0
        files = 0
        if ckpt.is_dir():
            for p in ckpt.rglob("*"):
                try:
                    if p.is_file():
                        size += p.stat().st_size
                        files += 1
                except OSError:
                    pass
        out.append(
            {
                "runId": meta.get("runId", entry.name),
                "runSlug": meta.get("runSlug", "?"),
                "completedAt": meta.get("completedAt"),
                "sizeBytes": size,
                "files": files,
                "path": str(entry),
            }
        )
    out.sort(key=lambda e: e.get("completedAt") or "", reverse=True)
    return out


@runner_app.command()
def artifacts() -> None:
    """List trained checkpoints stored locally on this runner.

    These are the runs the cloud knows about as
    ``artifactRef.kind = 'local-runner'`` — model weights that haven't
    been uploaded to Studio. Click "Publish from runner →" in Studio
    to upload one; use ``runner forget <runId>`` to delete locally.
    """
    items = _scan_local_artifacts()
    if not items:
        console.print(
            f"[dim]No local artifacts at {RUNS_ROOT}. Trained runs will appear here\n"
            f"once `pfnstudio runner start` lands a successful job.[/dim]"
        )
        return
    table = Table(title=f"Local artifacts at {RUNS_ROOT}", show_lines=False)
    table.add_column("Run slug", style="bold")
    table.add_column("Run ID", style="dim")
    table.add_column("Completed", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Files", justify="right")
    total = 0
    for it in items:
        total += it["sizeBytes"]
        completed = it.get("completedAt") or "?"
        rel = _fmt_age(completed) if completed != "?" else "?"
        table.add_row(
            str(it["runSlug"]),
            str(it["runId"])[:16] + "…",
            rel,
            _fmt_bytes(it["sizeBytes"]),
            str(it["files"]),
        )
    console.print(table)
    console.print(f"[dim]Total: {len(items)} run(s), {_fmt_bytes(total)}.[/dim]")


@runner_app.command()
def forget(
    run_id: str = typer.Argument(
        ..., help="Run ID (or substring matching a unique run) to delete locally."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete a locally-stored checkpoint to free disk.

    Doesn't affect anything cloud-side — the run record stays, just
    its ``artifactRef`` becomes stale (the UI shows "missing from
    runner"). If you also want to publish it first, do that BEFORE
    calling forget.
    """
    items = _scan_local_artifacts()
    matches = [
        i
        for i in items
        if i["runId"] == run_id or run_id in str(i["runId"]) or run_id in str(i["runSlug"])
    ]
    if not matches:
        console.print(f"[red]No local artifact matches '{run_id}'.[/red]")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[red]Ambiguous — '{run_id}' matches {len(matches)} runs:[/red]")
        for m in matches:
            console.print(f"  - {m['runSlug']}  ({m['runId']})")
        console.print("[dim]Re-run with a longer unique prefix.[/dim]")
        raise typer.Exit(1)
    target = matches[0]
    if not yes:
        console.print(
            f"About to delete [bold]{target['runSlug']}[/bold] ({target['runId']}, "
            f"{_fmt_bytes(target['sizeBytes'])}). [yellow]This can't be undone.[/yellow]"
        )
        confirm = typer.confirm("Proceed?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)
    try:
        shutil.rmtree(target["path"])
    except OSError as e:
        console.print(f"[red]Delete failed:[/red] {e}")
        raise typer.Exit(1) from e
    console.print(f"[green]✓[/green] Removed {target['path']}.")


# ── start ────────────────────────────────────────────────────────────


@runner_app.command()
def start(
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        help=(
            "Fallback project root, used only when the cloud's per-job "
            "bundle endpoint isn't available (older cloud, network "
            "issue). Normally the runner fetches the project as a "
            "tar.gz per job and extracts to a temp dir, so this is "
            "rarely consulted. Default: current dir."
        ),
    ),
    poll_idle_log_every_s: int = typer.Option(
        300,
        "--idle-log-every",
        help=(
            "Print 'still polling, nothing to do' every N seconds when the "
            "queue is empty. Set to 0 to suppress idle logs entirely."
        ),
    ),
) -> None:
    """Long-poll the cloud for jobs and execute them locally.

    Press Ctrl-C to stop gracefully (current job finishes; next poll
    skipped). SIGTERM (sent by systemd / docker stop) behaves the same.
    """
    cfg = _load_config()
    base = cfg["cloud_url"]
    headers = {"Authorization": f"Token {cfg['token']}"}

    caps = _capabilities()
    # Warn loudly at startup if torch isn't installed — every job will
    # otherwise come back "skipped: tabular_embedder requires torch" and
    # the operator has to dig through the cloud's run page to find out
    # why. Don't refuse to start (operator may be debugging the runner's
    # plumbing without intending to train).
    if not caps.get("torchVersion"):
        console.print(
            "[bold yellow]⚠ torch is not installed.[/bold yellow] Every "
            "training job will skip until you install it.\n"
            "  Fix: [bold]pip install -U 'pfnstudio[runner]'[/bold]"
        )

    # Bump capabilities on start so a runner that was offline for a
    # week and just got an OS update reports the new state.
    try:
        requests.post(
            f"{base}/runner/capabilities",
            json={"capabilities": caps},
            headers=headers,
            timeout=15,
        )
    except requests.RequestException as e:
        console.print(f"[yellow]Could not post capabilities at startup:[/yellow] {e}")

    stop_flag = threading.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        if not stop_flag.is_set():
            console.print(
                f"\n[yellow]Received {sig_name} — finishing current job, then stopping.[/yellow]"
            )
            stop_flag.set()
        else:
            # Second signal — exit hard.
            console.print(f"\n[red]Second {sig_name} — exiting immediately.[/red]")
            sys.exit(1)

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    console.print(
        f"[green]✓ Runner online.[/green] Polling [bold]{base}[/bold] for jobs.\n"
        f"[dim]Project root: {project_root}  ·  Ctrl-C to stop.[/dim]"
    )

    # Drop a fresh state file so `pfnstudio runner status` reports
    # this loop, not the previous one. stoppedAt=None signals "currently
    # running" — flipped to a timestamp on graceful shutdown below.
    # pfnstudioVersionAtStart lets `runner status` detect "you upgraded
    # pip but haven't restarted me" — common operator footgun.
    _update_state(
        pid=os.getpid(),
        startedAt=_ts(),
        lastPollAt=_ts(),
        cloudUrl=base,
        currentJob=None,
        stoppedAt=None,
        pfnstudioVersionAtStart=__version__,
    )

    # Phase 2: runner-served endpoints. Two background threads handle
    # the inference side while the main loop keeps polling for jobs:
    #
    #   serve_manager — every SERVE_TARGETS_POLL_S seconds, ask cloud
    #     which deployments this runner should be running, start /
    #     stop pfnstudio-serve subprocesses to match.
    #
    #   predict_poller — every PREDICT_POLL_S (~250 ms), pull pending
    #     predict requests, forward them to the matching local serve
    #     subprocess, post the response back.
    serve_registry: dict[str, dict[str, Any]] = {}
    serve_lock = threading.Lock()
    serve_thread = threading.Thread(
        target=_serve_manager_loop,
        args=(base, headers, serve_registry, serve_lock, stop_flag),
        daemon=True,
    )
    predict_thread = threading.Thread(
        target=_predict_poll_loop,
        args=(base, headers, serve_registry, serve_lock, stop_flag),
        daemon=True,
    )
    serve_thread.start()
    predict_thread.start()

    backoff_idx = 0
    last_idle_log = time.time()

    while not stop_flag.is_set():
        try:
            r = requests.get(
                f"{base}/runner/jobs/next",
                headers=headers,
                timeout=POLL_TIMEOUT_S,
            )
        except requests.RequestException as e:
            wait = BACKOFF_LADDER_S[min(backoff_idx, len(BACKOFF_LADDER_S) - 1)]
            console.print(f"[yellow]Poll failed:[/yellow] {e} — retrying in {wait}s")
            backoff_idx += 1
            if stop_flag.wait(wait):
                break
            continue
        backoff_idx = 0
        _update_state(lastPollAt=_ts())

        if r.status_code == 401:
            console.print(
                "[red]Auth failed (401).[/red] This runner was revoked — "
                "re-register from /settings/runners."
            )
            break
        if r.status_code >= 500:
            wait = BACKOFF_LADDER_S[min(backoff_idx, len(BACKOFF_LADDER_S) - 1)]
            console.print(f"[yellow]Cloud returned {r.status_code}; backing off {wait}s[/yellow]")
            backoff_idx += 1
            if stop_flag.wait(wait):
                break
            continue
        if not r.ok:
            console.print(f"[red]Cloud returned {r.status_code}:[/red] {r.text[:200]}")
            break

        data = r.json()

        # Publish requests are independent of jobs — handle them first
        # so they don't get starved when a training run is queued. Cap
        # is 10 per response from the cloud; we process all of them in
        # this iteration before moving on.
        for pub in data.get("pendingPublishes") or []:
            _process_publish_request(base, headers, pub)

        if data.get("status") == "idle":
            if poll_idle_log_every_s > 0 and (time.time() - last_idle_log) >= poll_idle_log_every_s:
                console.print("[dim]· still polling, no jobs queued[/dim]")
                last_idle_log = time.time()
            continue

        job = data.get("job") or {}
        if not job.get("runId"):
            console.print(f"[red]Malformed job payload:[/red] {data}")
            continue

        # Never let one job kill the runner. _run_job reports its own
        # result on the normal + expected-failure paths; this catches
        # anything that escapes (a bug in the executor, an unexpected
        # exception) so the daemon logs it, best-effort marks the job
        # failed, and keeps polling for the next one.
        try:
            _run_job(base, headers, job, project_root, stop_flag)
        except Exception as exc:
            run_id = job.get("runId")
            console.print(
                f"[red]Job {run_id} crashed the executor — failing it and "
                f"continuing:[/red] {type(exc).__name__}: {exc}"
            )
            try:
                requests.post(
                    f"{base}/runner/jobs/{run_id}/result",
                    json={
                        "status": "error",
                        "events": [],
                        "error": f"runner executor error: {type(exc).__name__}: {exc}",
                    },
                    headers=headers,
                    timeout=30,
                )
            except requests.RequestException:
                pass
            _update_state(currentJob=None)
        last_idle_log = time.time()

    _update_state(stoppedAt=_ts(), currentJob=None)
    console.print("[green]Runner stopped.[/green]")


# ── Job execution ────────────────────────────────────────────────────


def _sync_core_if_configured() -> None:
    """Refresh ``pfnstudio-core`` before running a job, if opted in.

    A self-hosted runner's installed core can lag the version a job's model
    needs — a model referencing ``grid_preprocessor`` (the axial-attention
    library, added in core 0.9.0) fails with "No block registered" on an older
    core. Opt in with either env var, set when you launch the runner:

      PFNSTUDIO_RUNNER_SYNC_CORE=1        # upgrade `pfnstudio-core` from PyPI
      PFNSTUDIO_RUNNER_CORE_SPEC=<spec>   # upgrade that exact pip spec instead
                                          # e.g. pfnstudio-core==0.9.0, or a
                                          # git URL for an unreleased version

    ``--no-deps`` keeps it quick: torch/numpy are already installed on a
    runner, so only the pure-Python core code is refreshed. A failed sync is
    logged and non-fatal — the job proceeds on the installed core.
    """
    spec = os.environ.get("PFNSTUDIO_RUNNER_CORE_SPEC", "").strip()
    if not spec and os.environ.get("PFNSTUDIO_RUNNER_SYNC_CORE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        spec = "pfnstudio-core"
    if not spec:
        return
    console.print(f"[dim]runner: syncing {spec} before job…[/dim]")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "--no-deps", spec],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        console.print(f"[dim]runner: {spec} up to date[/dim]")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        console.print(
            f"[yellow]runner: core sync failed, continuing with the installed "
            f"core: {str(detail)[-300:]}[/yellow]"
        )


def _materialize_job_workspace(
    base: str,
    headers: dict[str, str],
    run_id: str,
    run_slug: str,
    job: dict[str, Any],
    fallback_project_root: Path,
) -> tuple[Path | None, Path, str]:
    """Return (bundle_dir_to_cleanup, work_root, run_yaml_rel_path).

    Two paths in priority order:

    1. **Bundle download (default).** Fetch ``GET /runner/jobs/<id>/bundle``
       and extract its tar.gz into a fresh temp dir. The bundle contains
       the full project tree the trainer needs (priors/, models/, evals/,
       runs/<slug>.yaml). work_root = the temp dir, run_yaml_rel =
       ``runs/<slug>.yaml``. bundle_dir_to_cleanup = the temp dir so the
       caller can rmtree it on exit.

    2. **Fallback to --project-root.** If the cloud returns 404 (older
       cloud without the bundle endpoint) or download fails, write the
       job's spec dict to a temp run.yaml inside the operator's
       --project-root and run there. work_root = fallback_project_root,
       run_yaml_rel = ``<tmpname>.yaml``. bundle_dir_to_cleanup = None.

    Network errors (timeout, 5xx) raise so the caller can mark the job
    failed with a clear message. Only 404 falls through silently —
    that's the "older cloud" case where the fallback is expected.
    """
    try:
        resp = requests.get(
            f"{base}/runner/jobs/{run_id}/bundle",
            headers=headers,
            timeout=120,
            stream=True,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"bundle download failed: {e}") from e

    if resp.status_code == 404:
        # Older cloud without the bundle endpoint OR job already
        # released. Fall back to operator-supplied project_root.
        console.print(
            "[dim]Bundle endpoint not available — falling back to "
            f"--project-root ({fallback_project_root}). Update the "
            "cloud for bundle download support.[/dim]"
        )
        spec = {
            "id": run_slug,
            "prior": job.get("priorRef") or {},
            "model": job.get("modelRef") or {},
            "evals": job.get("evalRefs") or [],
            "hyperparams": job.get("hyperparams") or {},
            "compute": {"target": "local"},
        }
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=f"-{run_slug}.yaml",
            delete=False,
            dir=str(fallback_project_root),
        ) as f:
            yaml.safe_dump(spec, f)
            tmp_yaml = Path(f.name)
        return None, fallback_project_root, tmp_yaml.name

    if not resp.ok:
        raise RuntimeError(f"bundle download returned {resp.status_code}: {resp.text[:200]}")

    # Stream the tar.gz body into memory then extract. Project bundles
    # are small (YAML + Python source, no datasets), so we don't bother
    # spooling to disk first.
    bundle_dir = Path(tempfile.mkdtemp(prefix=f"pfnstudio-runner-{run_slug}-"))
    try:
        buf = io.BytesIO(resp.content)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            # filter='data' refuses absolute paths, symlinks, and
            # parent-relative escapes — defense against a malicious
            # archive (Python 3.12+; older versions fall back to
            # default behavior with a warning).
            try:
                tar.extractall(path=bundle_dir, filter="data")
            except TypeError:
                tar.extractall(path=bundle_dir)
    except (tarfile.TarError, OSError) as e:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise RuntimeError(f"bundle extract failed: {e}") from e

    run_yaml_rel = f"runs/{run_slug}.yaml"
    if not (bundle_dir / run_yaml_rel).exists():
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise RuntimeError(f"bundle is missing {run_yaml_rel} — cloud sent an incomplete bundle.")
    console.print(f"[dim]Bundle extracted to {bundle_dir}[/dim]")
    return bundle_dir, bundle_dir, run_yaml_rel


def _process_publish_request(base: str, headers: dict[str, str], pub: dict[str, Any]) -> None:
    """Tar.gz a locally-stored checkpoint and POST it to the cloud.
    Triggered when the cloud's poll response includes a pendingPublishes
    entry — the operator clicked 'Publish from runner →' in Studio
    after training had moved on.

    Best-effort: a missing checkpoint (operator already ran `runner
    forget`) prints a warning and moves on. The cloud's
    publishRequestedAt stays set, so the next poll re-presents this
    request — clears once the operator either uploads it (re-creating
    the local copy via a re-run) or revokes the publish request via UI.
    """
    run_id = pub.get("runId")
    run_slug = pub.get("runSlug", run_id)
    if not isinstance(run_id, str):
        return
    local_root = RUNS_ROOT / run_id
    ckpt = local_root / "checkpoint"
    if not ckpt.is_dir():
        console.print(
            f"[yellow]Publish requested for {run_slug} but local checkpoint is "
            f"missing.[/yellow] Was it `runner forget`'d? Re-run training to "
            f"reproduce it, or have the operator click 'Cancel publish' in Studio."
        )
        return

    console.print(f"[blue]→[/blue] uploading checkpoint for [bold]{run_slug}[/bold]…")
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
            tar_path = Path(tf.name)
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(ckpt, arcname="checkpoint")
        with open(tar_path, "rb") as fh:
            r = requests.post(
                f"{base}/runner/jobs/{run_id}/artifact",
                headers=headers,
                files={"file": ("checkpoint.tar.gz", fh, "application/gzip")},
                timeout=600,
            )
        tar_path.unlink(missing_ok=True)
        if r.ok:
            size = r.json().get("bytes", 0)
            console.print(f"[green]✓[/green] published {run_slug} ({size:,} bytes uploaded).")
        else:
            console.print(
                f"[yellow]Cloud rejected publish upload ({r.status_code}):[/yellow] {r.text[:200]}"
            )
    except (OSError, requests.RequestException) as e:
        console.print(
            f"[yellow]Publish upload for {run_slug} failed:[/yellow] {type(e).__name__}: {e}"
        )


def _run_job(
    base: str,
    headers: dict[str, str],
    job: dict[str, Any],
    project_root: Path,
    stop_flag: threading.Event,
) -> None:
    """Execute one claimed job — subprocess the local trainer, stream
    its JSON events back via /events, post final result."""
    run_id = str(job["runId"])
    run_slug = job.get("runSlug", run_id)
    console.print(f"\n[bold blue]→ claimed[/bold blue] {run_slug} [dim](run_id={run_id})[/dim]")

    # Mark "we have a current job" early so `runner status` can show it
    # even if bundle download is taking a while. trainerPid + bundleDir
    # filled in later as they become known.
    _update_state(
        currentJob={
            "runId": run_id,
            "runSlug": run_slug,
            "claimedAt": _ts(),
            "bundleDir": None,
            "trainerPid": None,
        }
    )

    # Heartbeat thread keeps the cloud's claim fresh while training runs.
    alive = threading.Event()
    alive.set()

    def _heartbeat() -> None:
        while alive.is_set():
            try:
                requests.post(
                    f"{base}/runner/jobs/{run_id}/heartbeat",
                    headers=headers,
                    timeout=10,
                )
            except requests.RequestException:
                # Transient — keep ticking; the cloud will reclaim the job
                # if heartbeats lapse beyond HEARTBEAT_STALE_S.
                pass
            for _ in range(HEARTBEAT_INTERVAL_S):
                if not alive.is_set():
                    return
                time.sleep(1)

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    events_acc: list[dict[str, Any]] = []
    final_status = "failed"
    final_error: dict[str, Any] | None = None
    proc: subprocess.Popen[str] | None = None
    bundle_dir: Path | None = None
    tmp_yaml: Path | None = None

    try:
        # Resolve which directory the trainer will run from. Bundle
        # download is the modern path (cloud streams the project
        # source per job); fallback to operator's --project-root if
        # the cloud lacks the endpoint or the download fails. Either
        # way the subprocess CWD is `work_root`.
        bundle_dir, work_root, run_yaml_rel = _materialize_job_workspace(
            base, headers, run_id, run_slug, job, project_root
        )
        # tmp_yaml is only set in the fallback path; tracked here so
        # finally: cleanup knows what to unlink.
        if bundle_dir is None:
            tmp_yaml = work_root / run_yaml_rel
        # Surface where the bundle landed so `runner status` can show
        # disk usage + the path the operator would `ls` to inspect.
        _update_state(
            currentJob={
                "runId": run_id,
                "runSlug": run_slug,
                "claimedAt": _ts(),
                "bundleDir": str(bundle_dir) if bundle_dir else str(work_root),
                "trainerPid": None,
            }
        )

        # Optionally refresh pfnstudio-core before the job so a self-hosted
        # runner picks up newer core blocks (e.g. the axial-attention library)
        # — otherwise a model referencing a block the installed core lacks
        # fails with "No block registered under '<type>'".
        _sync_core_if_configured()

        # PFNSTUDIO_JSON_PROGRESS=1 tells the local adapter to emit
        # JSON-line events on stdout, which we pipe up to /events.
        env = os.environ.copy()
        env["PFNSTUDIO_JSON_PROGRESS"] = "1"

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pfnstudio",
                "run",
                run_yaml_rel,
                "--target",
                "local",
            ],
            cwd=str(work_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        # Record the trainer PID so `runner status` can tell the user
        # "yes, a python -m pfnstudio run is alive" instead of making
        # them grep ps.
        _update_state(
            currentJob={
                "runId": run_id,
                "runSlug": run_slug,
                "claimedAt": _ts(),
                "bundleDir": str(bundle_dir) if bundle_dir else str(work_root),
                "trainerPid": proc.pid,
            }
        )

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            # Try to parse a JSON event; otherwise treat as a plain log line.
            try:
                ev = json.loads(line)
                if not isinstance(ev, dict) or "event" not in ev:
                    raise ValueError("not an event")
            except (json.JSONDecodeError, ValueError):
                ev = {"event": "log", "line": line, "ts": _ts()}
            # Scrub NaN/Inf so a single bad metric can't make the event
            # (or the final result post that re-sends events_acc) fail to
            # serialize and lose the whole run report.
            ev = _json_safe(ev)
            events_acc.append(ev)

            # Best-effort stream — don't block the trainer if the cloud
            # is briefly unreachable.
            try:
                requests.post(
                    f"{base}/runner/jobs/{run_id}/events",
                    json={"events": [ev]},
                    headers=headers,
                    timeout=5,
                )
            except requests.RequestException:
                pass

            if ev.get("event") == "finished":
                final_status = ev.get("status", "completed")

            if stop_flag.is_set():
                # User asked the runner to stop; kill this job and
                # report cancelled.
                console.print("[yellow]Stop requested — terminating in-flight training.[/yellow]")
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                final_status = "cancelled"
                break

        proc.wait()
        # If the trainer exited cleanly but didn't emit a 'finished'
        # event, infer success from the return code.
        if proc.returncode == 0 and final_status == "failed":
            final_status = "completed"
        elif proc.returncode != 0 and final_status not in ("cancelled", "failed"):
            final_status = "failed"
            final_error = {"message": f"trainer exited with code {proc.returncode}"}

    except Exception as e:
        final_error = {"message": f"runner-side error: {type(e).__name__}: {e}"}
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    finally:
        alive.clear()
        # Clear the currentJob marker so `runner status` reports idle
        # again. Done eagerly here so a stuck artifact upload below
        # doesn't make the status command keep showing this run as
        # the current job forever.
        _update_state(currentJob=None)

        # Move the trained checkpoint to its durable local home BEFORE
        # rmtree wipes the bundle. Data-residency story: the model
        # stays on this runner unless the operator clicks "Publish
        # from runner →" in Studio later, which triggers an upload
        # via the long-poll publish-request channel.
        artifact_ref: dict[str, Any] | None = None
        if final_status == "completed" and bundle_dir is not None:
            persisted = _persist_checkpoint(bundle_dir, run_id, run_slug, base)
            if persisted is not None:
                local_path, size_bytes = persisted
                artifact_ref = {
                    "kind": "local-runner",
                    "runId": run_id,
                    "sizeBytes": size_bytes,
                    "localPath": str(local_path),
                }
                size_mb = size_bytes / (1024 * 1024)
                console.print(
                    f"[dim]Checkpoint kept on this runner ({size_mb:.1f} MB) at "
                    f"{local_path} — click Publish in Studio to upload.[/dim]"
                )

        # Best-effort cleanup. Two cases:
        #   - bundle path: nuke the whole temp extract dir
        #   - fallback path: just unlink the temp run.yaml we wrote
        #     inside the operator's --project-root (don't touch the
        #     operator's project files).
        if bundle_dir is not None:
            try:
                shutil.rmtree(bundle_dir, ignore_errors=True)
            except Exception:
                pass
        elif tmp_yaml is not None:
            try:
                tmp_yaml.unlink(missing_ok=True)
            except Exception:
                pass

        # Post the final state. Retry once on transient errors — losing
        # this report would leave the cloud with the run stuck in
        # 'running' until the next requeue sweep.
        result_body: dict[str, Any] = {
            "status": final_status,
            "events": events_acc,
            "error": final_error,
        }
        if artifact_ref is not None:
            result_body["artifactRef"] = artifact_ref
        result_body = _json_safe(result_body)
        for attempt in range(2):
            try:
                requests.post(
                    f"{base}/runner/jobs/{run_id}/result",
                    json=result_body,
                    headers=headers,
                    timeout=30,
                )
                break
            except requests.RequestException as e:
                if attempt == 0:
                    console.print(f"[yellow]Failed to post result, retrying:[/yellow] {e}")
                    time.sleep(2)
                else:
                    console.print(f"[red]Could not post result for {run_id}:[/red] {e}")

        marker = (
            "[green]✓[/green]"
            if final_status == "completed"
            else "[yellow]⚠[/yellow]"
            if final_status == "cancelled"
            else "[red]✗[/red]"
        )
        console.print(f"{marker} run {run_slug} → [bold]{final_status}[/bold]")
