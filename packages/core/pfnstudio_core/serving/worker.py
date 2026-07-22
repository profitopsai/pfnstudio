"""HTTP worker that holds one trained checkpoint in memory and serves
predictions over a loopback port. Used by Phase 1 of the deployments
system: Studio spawns one process per Deployment, polls /healthz until
'ready', and forwards user predict() calls via /predict.

Event-stream protocol:
    Each line printed to stdout is a JSON object the parent (NestJS)
    consumes. Events:

      {"event":"starting","port":N}
      {"event":"ready",   "port":N}
      {"event":"error",   "stage":"load"|"serve","message":"..."}
      {"event":"shutdown","reason":"sigterm"|"sigint"|"exit"}

    This mirrors the runner.service.ts event-line protocol used for
    training runs — same parser on the NestJS side, same UX surface
    for failures.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _emit(event: str, **fields: Any) -> None:
    """Write one JSON event line to stdout and flush.

    NestJS reads the spawned worker's stdout line-by-line; flushing
    means health-poll latency is bounded by request RTT, not by Python's
    default output buffering.
    """
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _make_handler(loader: Any) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class closed over a single ModelLoader.

    Each connection gets its own handler instance (stdlib semantics),
    but they all share the loader — so the model is loaded once at
    process start and reused across requests.
    """

    class _Handler(BaseHTTPRequestHandler):
        # Silence the default per-request stdout log line. The parent
        # tails stdout for the event-stream protocol above; HTTP access
        # logs would interleave and confuse the parser. Errors still
        # surface via log_error → stderr (kept for debugging).
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                # Worker is fully constructed before do_GET can be
                # invoked (server isn't started until __init__ done),
                # so reaching here means the loader is ready.
                self._send_json(200, {"status": "ready"})
                return
            self._send_json(404, {"error": "not_found", "path": self.path})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send_json(404, {"error": "not_found", "path": self.path})
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                self._send_json(400, {"error": "empty_body"})
                return
            try:
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._send_json(400, {"error": "invalid_json", "detail": str(e)})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "payload_must_be_object"})
                return

            # Tag + detect extraction mirrors the CLI predict command
            # so the worker's contract matches every other entry point.
            tag = payload.pop("tag", None)
            detect_flag = bool(payload.pop("detect", False))

            try:
                out = loader.predict(payload, tag=tag, detect=detect_flag)
            except ValueError as e:
                # Payload-shape rejections are user errors → 400.
                self._send_json(400, {"error": "invalid_payload", "detail": str(e)})
                return
            except Exception as e:  # pragma: no cover — defensive
                # Model-side failures are server-side → 500 + stderr trace
                # so operators have something to debug from logs.
                traceback.print_exc(file=sys.stderr)
                self._send_json(500, {"error": "inference_failed", "detail": str(e)})
                return

            self._send_json(200, out)

    return _Handler


class _AdapterServeLoader:
    """Warm, adapter-backed serve loader. Resolves the run's adapter, loads its
    model ONCE, and answers per request through ``adapter.predict()`` /
    ``adapter.capabilities()`` — the persistent analogue of
    ``predict._run_adapter_inference`` (which loads per call). This is what makes
    ``pfnstudio serve`` work for adapter base models (TCPFN, …); the default
    ``ModelLoader`` only understands the studio block model and rejects the
    ``{task, inputs}`` envelope with ``invalid_payload``. Exposes ``predict``
    with the same ``(payload, tag, detect)`` signature the handler calls."""

    def __init__(
        self,
        adapter_name: str,
        manifest_path: Path,
        project_root: Path,
        checkpoint_dir: Path,
    ) -> None:
        import torch
        import yaml

        from ..registry import discover_in_project
        from ..training.adapters import resolve_adapter

        self._name = adapter_name
        discover_in_project(project_root)
        try:
            data = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            data = {}
        hp = (data.get("hyperparams") or {}) if isinstance(data, dict) else {}
        self._hp: dict[str, Any] = hp if isinstance(hp, dict) else {}
        self._adapter = resolve_adapter(adapter_name, self._hp.get("adapterModule"))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._adapter.load_model(Path(checkpoint_dir), self._hp, self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def predict(self, payload: dict, tag: Any = None, detect: bool = False) -> dict:
        import torch

        if isinstance(payload, dict) and payload.get("__capabilities__"):
            cap_fn = getattr(self._adapter, "capabilities", None)
            caps = cap_fn() if callable(cap_fn) else {}
            return {
                "engine": self._name,
                "command": "capabilities",
                "result": {"capabilities": caps or {}},
            }
        predict_fn = getattr(self._adapter, "predict", None)
        if predict_fn is None:
            return {
                "engine": self._name,
                "command": "predict",
                "result": {"error": f"the '{self._name}' adapter does not implement predict()."},
            }
        with torch.no_grad():
            result = predict_fn(self._model, payload or {}, self._hp, self._device)
        return {"engine": self._name, "command": "predict", "result": result}


def serve(
    *,
    manifest_path: Path,
    project_root: Path,
    checkpoint_dir: Path,
    host: str = "127.0.0.1",
    port: int = 0,
) -> None:
    """Block forever serving predictions for one checkpoint.

    ``port=0`` lets the OS pick a free port; the actual port is reported
    in the {"event":"ready",...} line so the parent can record it.

    Returns when the server is shut down via SIGTERM/SIGINT — emits a
    shutdown event before unwinding. Any failure during loader
    construction is emitted as a load-stage error event with exit 1
    so the parent can surface the reason on the Deployment row.
    """
    from ..training.predict import ModelLoader, _peek_run_adapter

    _emit("starting", host=host, port=port, run_dir=str(project_root))

    try:
        adapter_name = _peek_run_adapter(manifest_path)
        if adapter_name:
            # Adapter base model (TCPFN, …): serve through the adapter's own
            # predict()/capabilities(), not the block-model ModelLoader (which
            # 400s 'invalid_payload' on the {task, inputs} envelope).
            loader = _AdapterServeLoader(
                adapter_name, manifest_path, project_root, checkpoint_dir
            )
        else:
            loader = ModelLoader(
                manifest_path=manifest_path,
                project_root=project_root,
                checkpoint_dir=checkpoint_dir,
            )
    except BaseException as e:
        traceback.print_exc(file=sys.stderr)
        _emit("error", stage="load", message=f"{type(e).__name__}: {e}")
        sys.exit(1)

    handler_cls = _make_handler(loader)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    actual_port = int(httpd.server_port)

    shutdown_reason = {"reason": "exit"}
    shutdown_event = threading.Event()

    def _request_shutdown(reason: str) -> None:
        if shutdown_event.is_set():
            return
        shutdown_reason["reason"] = reason
        shutdown_event.set()
        # shutdown() blocks until serve_forever() returns; run in a
        # thread so the signal handler can return immediately.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    def _on_signal(signum: int, _frame: Any) -> None:
        name = "sigterm" if signum == signal.SIGTERM else "sigint"
        _request_shutdown(name)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    _emit("ready", host=host, port=actual_port)
    try:
        httpd.serve_forever()
    except BaseException as e:  # pragma: no cover — defensive
        traceback.print_exc(file=sys.stderr)
        _emit("error", stage="serve", message=f"{type(e).__name__}: {e}")
        _emit("shutdown", reason=shutdown_reason["reason"])
        loader.close()
        sys.exit(1)

    _emit("shutdown", reason=shutdown_reason["reason"])
    loader.close()
    httpd.server_close()
