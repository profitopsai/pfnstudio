"""FastAPI runtime for self-hosted PFN model containers.

Exposes capability-driven HTTP routes, OpenAPI/Swagger, and an embedded MCP
endpoint. Used by ``packages/container/`` entrypoint; the stdlib worker in
``worker.py`` remains for NestJS deployment spawns.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .auth import check_auth
from .docs import build_mcp_docs_html, build_scalar_html
from .loader import create_serve_loader, fetch_capabilities
from .mcp_http import handle_mcp_request
from .openapi import build_openapi_spec


def _unwrap_capabilities(body: dict[str, Any]) -> dict[str, Any]:
    result = body.get("result") if isinstance(body.get("result"), dict) else body
    caps = result.get("capabilities") if isinstance(result, dict) else None
    return caps if isinstance(caps, dict) else {}


def _lock_digest(project_root: Path) -> str | None:
    """The model lock's digest when the image was built with one — lock.json at
    the app root (project_root, or APP_ROOT when serving from a separate app directory)."""
    import os

    roots = [project_root]
    app_root = os.environ.get("APP_ROOT")
    if app_root:
        roots.append(Path(app_root))
    for root in roots:
        p = root / "lock.json"
        if not p.is_file():
            continue
        try:
            lock = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        digest = lock.get("digest") if isinstance(lock, dict) else None
        return digest if isinstance(digest, str) and digest else None
    return None


def _normalise_task_body(body: dict[str, Any], task: str) -> dict[str, Any]:
    payload = dict(body)
    if task == "predict":
        return payload
    from_body = payload.pop("task", None)
    resolved = from_body if task == "predict" else task
    if "inputs" in payload:
        return {"task": resolved, "inputs": payload["inputs"]}
    return {"task": resolved, "inputs": payload}


def create_app(
    *,
    manifest_path: Path,
    project_root: Path,
    checkpoint_dir: Path,
    title: str = "PFN Studio model container",
) -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.openapi.docs import get_swagger_ui_html
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fastapi is required for container serving — pip install fastapi uvicorn"
        ) from e

    loader = create_serve_loader(
        manifest_path=manifest_path,
        project_root=project_root,
        checkpoint_dir=checkpoint_dir,
    )

    def _predict(payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        if body.get("__capabilities__"):
            return fetch_capabilities(
                loader=loader,
                manifest_path=manifest_path,
                project_root=project_root,
            )
        tag = body.pop("tag", None)
        detect = bool(body.pop("detect", False))
        return loader.predict(body, tag=tag, detect=detect)

    caps_body = fetch_capabilities(
        loader=loader,
        manifest_path=manifest_path,
        project_root=project_root,
    )
    capabilities = _unwrap_capabilities(caps_body)
    openapi_spec = build_openapi_spec(capabilities=capabilities, title=title)
    lock_digest = _lock_digest(project_root)
    caps_response = {**caps_body, "lock_digest": lock_digest} if lock_digest else caps_body

    app = FastAPI(title=title, docs_url=None, redoc_url=None, openapi_url=None)

    async def _require_auth(authorization: str | None = Header(default=None)) -> None:
        err = check_auth(authorization)
        if err:
            raise HTTPException(status_code=401, detail=err)

    @app.get("/health")
    @app.get("/healthz")  # alias — prefer /health
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/capabilities", dependencies=[Depends(_require_auth)])
    async def capabilities() -> dict[str, Any]:
        return caps_response

    @app.get("/openapi.json")
    async def openapi_json() -> dict[str, Any]:
        return openapi_spec

    @app.get("/docs")
    async def docs() -> Any:
        return HTMLResponse(build_mcp_docs_html(title=title, caps_body=caps_body))

    @app.get("/api/docs")
    async def api_docs() -> Any:
        return HTMLResponse(build_scalar_html(title=title, openapi_url="/openapi.json"))

    @app.get("/docs/swagger")
    async def docs_swagger() -> Any:
        return get_swagger_ui_html(openapi_url="/openapi.json", title=title)

    @app.post("/predict", dependencies=[Depends(_require_auth)])
    async def predict_route(body: dict[str, Any]) -> dict[str, Any]:
        try:
            return _predict(dict(body))
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail={"error": "invalid_payload", "detail": str(e)}
            ) from e
        except Exception as e:  # pragma: no cover
            traceback.print_exc(file=sys.stderr)
            raise HTTPException(
                status_code=500, detail={"error": "inference_failed", "detail": str(e)}
            ) from e

    @app.post("/mcp", dependencies=[Depends(_require_auth)])
    async def mcp_route(req: dict[str, Any]) -> JSONResponse:
        if not isinstance(req, dict):
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32600, "message": "invalid request"}},
                status_code=400,
            )
        resp = handle_mcp_request(req, caps_body=caps_body, predict=_predict)
        return JSONResponse(resp)

    @app.post("/{task}", dependencies=[Depends(_require_auth)])
    async def task_route(task: str, body: dict[str, Any]) -> dict[str, Any]:
        if task in (
            "health",
            "healthz",
            "capabilities",
            "openapi.json",
            "docs",
            "mcp",
            "predict",
            "api",
        ):
            raise HTTPException(status_code=404, detail={"error": "not_found", "path": task})
        payload = _normalise_task_body(body, task)
        try:
            return _predict(payload)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail={"error": "invalid_payload", "detail": str(e)}
            ) from e
        except Exception as e:  # pragma: no cover
            traceback.print_exc(file=sys.stderr)
            raise HTTPException(
                status_code=500, detail={"error": "inference_failed", "detail": str(e)}
            ) from e

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        close = getattr(loader, "close", None)
        if callable(close):
            close()

    # Emit the same stdout events as pfnstudio serve so operators can reuse log parsers.
    print(
        json.dumps({"event": "starting", "host": "0.0.0.0", "run_dir": str(project_root)}),
        flush=True,
    )
    print(json.dumps({"event": "ready", "host": "0.0.0.0", "port": "8080"}), flush=True)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a trained PFN model over FastAPI.")
    parser.add_argument("manifest", type=Path, help="Path to run YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    manifest = args.manifest.resolve()
    project = args.project.resolve() if args.project else manifest.parent.parent
    checkpoint = args.checkpoint.resolve()

    app = create_app(
        manifest_path=manifest,
        project_root=project,
        checkpoint_dir=checkpoint,
        title=f"PFN model — {manifest.stem}",
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
