"""Local multi-model daemon — the container's server, hosted on your machine.

``pfnstudio serve`` runs this in the foreground; ``pfnstudio run`` auto-starts
it when it is unreachable (``client.ensure_running``) and talks to it over
HTTP, the way ``ollama run`` is a thin client of ``ollama serve``.

One process hosts many base models. Each loaded model is the SAME FastAPI app
the self-hostable container runs (``pfnstudio_core.serving.fastapi_app.create_app``),
mounted under ``/models/<slug>/`` — so ``/models/<slug>/predict``,
``/capabilities``, ``/mcp``, ``/openapi.json``, ``/docs`` and ``POST /{task}``
behave exactly as they do in the container. The host app adds management:

    GET  /healthz           liveness
    GET  /ps                loaded models (slug, fmProject, digest, loaded_at, ...)
    POST /load   {"ref"}    resolve listing → bundle + checkpoint → mount
    POST /unload {"slug"}   unmount + close the loader

Models stay loaded until unloaded or the daemon exits. ``keep_alive`` (an idle
unload) is an optional setting with no default — see ``build_host_app``.

Env: PFNSTUDIO_API_URL (marketplace), PFNSTUDIO_BASEMODELS_CACHE /
PFNSTUDIO_BASEMODELS_DIR / PFNSTUDIO_CHECKPOINT_CACHE (local store),
PFNSTUDIO_SERVE_TOKEN (optional Bearer auth — applied to management routes and,
by the container app, to every per-model route).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11440
MOUNT_PREFIX = "/models"
DEFAULT_API_URL = "https://cloud.pfnstudio.com/api"
TIERS = ("native", "adapter")


# ── pure helpers (no fastapi / torch / network; unit-tested) ──────────────────


def api_base() -> str:
    """Marketplace API base — same default and env var as ``pfnstudio run``."""
    return os.environ.get("PFNSTUDIO_API_URL", DEFAULT_API_URL).rstrip("/")


def basemodels_cache() -> Path:
    return Path(
        os.environ.get(
            "PFNSTUDIO_BASEMODELS_CACHE",
            str(Path.home() / ".cache" / "pfnstudio" / "basemodels"),
        )
    )


def is_safe_slug(slug: str) -> bool:
    """A slug becomes a URL mount segment and a file stem — keep it to the
    marketplace's own alphabet."""
    return (
        bool(slug) and all(ch.isalnum() or ch in "._-" for ch in slug) and slug not in (".", "..")
    )


def mount_path(slug: str) -> str:
    """Where a model's container app lives on the host: ``/models/<slug>``."""
    if not is_safe_slug(slug):
        raise ValueError(f"invalid model slug: {slug!r}")
    return f"{MOUNT_PREFIX}/{slug}"


def parse_ref(ref: str) -> tuple[str, str | None]:
    """``slug`` or ``slug:tag`` → ``(slug, tag)``. Delegates to
    ``lock.parse_model_ref`` when that module is present (it owns the grammar);
    otherwise splits on the last colon."""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("empty model ref")
    try:
        from .lock import parse_model_ref  # another story ships this module
    except ImportError:
        parse_model_ref = None
    if parse_model_ref is not None:
        slug, tag = parse_model_ref(ref)
        return str(slug), (str(tag) if tag else None)
    slug, _, tag = ref.rpartition(":") if ":" in ref else (ref, "", "")
    return slug, (tag or None)


def read_digest(bundle_dir: Path | None) -> str | None:
    """The bundle's lock.json digest when ``lock.read_lock`` exists and the
    bundle carries one; None otherwise."""
    if bundle_dir is None:
        return None
    try:
        from .lock import read_lock
    except ImportError:
        return None
    try:
        lock = read_lock(Path(bundle_dir))
    except Exception:
        return None
    if not isinstance(lock, dict):
        return None
    digest = lock.get("digest")
    return str(digest) if digest else None


def normalize_tier(value: Any) -> str | None:
    """Listing ``tier`` field ("native" | "adapter"); None when absent/unknown."""
    tier = str(value).strip().lower() if value is not None else ""
    return tier if tier in TIERS else None


def manifest_text(adapter_name: str) -> str:
    """The minimal run manifest ``pfnstudio run`` synthesizes so the loader
    routes through the base model's own adapter (``hyperparams.adapter`` /
    ``baseAdapter``)."""
    return (
        "description: serve\n"
        "hyperparams:\n"
        f"  adapter: {adapter_name}\n"
        f"  baseAdapter: {adapter_name}\n"
    )


def write_run_manifest(state_dir: Path, slug: str, adapter_name: str) -> Path:
    """``<state_dir>/runs/<slug>.yaml`` — kept for the daemon's lifetime because
    the loader re-reads the manifest when it builds capabilities."""
    path = Path(state_dir) / "runs" / f"{slug}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_text(adapter_name), encoding="utf-8")
    return path


# ── registry ─────────────────────────────────────────────────────────────────


@dataclass
class LoadedModel:
    slug: str
    fm_project: str
    adapter: str
    checkpoint_dir: Path
    bundle_dir: Path
    manifest_path: Path
    tier: str | None = None
    tag: str | None = None
    digest: str | None = None
    loaded_at: str = ""
    app: Any = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "fmProject": self.fm_project,
            "adapter": self.adapter,
            "tier": self.tier,
            "tag": self.tag,
            "digest": self.digest,
            "loaded_at": self.loaded_at,
            "path": mount_path(self.slug),
            "checkpoint_dir": str(self.checkpoint_dir),
            "bundle_dir": str(self.bundle_dir),
        }


class Registry:
    """In-memory table of loaded models; the single source for ``/ps``."""

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self.lock = threading.Lock()

    def get(self, slug: str) -> LoadedModel | None:
        return self._models.get(slug)

    def add(self, model: LoadedModel) -> None:
        self._models[model.slug] = model

    def remove(self, slug: str) -> LoadedModel | None:
        return self._models.pop(slug, None)

    def ps(self) -> list[dict[str, Any]]:
        return [m.summary() for m in sorted(self._models.values(), key=lambda m: m.loaded_at)]

    def __iter__(self):
        return iter(list(self._models.values()))

    def __len__(self) -> int:
        return len(self._models)


# ── resolution (network + disk) ──────────────────────────────────────────────


class ModelNotFound(LookupError):  # noqa: N818 — reads as the condition it signals
    pass


def resolve_listing(slug: str, api: str | None = None) -> dict[str, Any]:
    """The marketplace listing for ``slug`` (type == basemodel)."""
    import requests

    base = api or api_base()
    resp = requests.get(f"{base}/marketplace/listings", timeout=20)
    resp.raise_for_status()
    listings = resp.json()
    base_models = [
        x
        for x in (listings if isinstance(listings, list) else [])
        if isinstance(x, dict) and x.get("type") == "basemodel"
    ]
    listing = next((x for x in base_models if x.get("slug") == slug), None)
    if listing is None:
        avail = sorted(str(x.get("slug")) for x in base_models)
        raise ModelNotFound(
            f"no base model with slug {slug!r}; available: {', '.join(avail) or 'none'}"
        )
    return listing


def ensure_bundle(api: str, slug: str, fm_project: str) -> Path:
    """The adapter bundle dir — same precedence as ``pfnstudio run``: a
    checkout / PFNSTUDIO_BASEMODELS_DIR, then the local cache, else pull from
    the marketplace into the cache."""
    from .cli import _fetch_basemodel_bundle, _find_basemodels_dir

    override = os.environ.get("PFNSTUDIO_BASEMODELS_DIR")
    found = _find_basemodels_dir(fm_project, Path(override) if override else None)
    if found is not None:
        return found
    cached = basemodels_cache() / fm_project
    if (cached / "adapters").is_dir():
        return cached
    return _fetch_basemodel_bundle(api, slug, fm_project)


def ensure_checkpoint(checkpoint_ref: Any) -> Path:
    """Materialize the listing's checkpointRef (manifest dict, hf:// URI, URL
    or local path) into the checkpoint cache."""
    from pfnstudio_core.serving.hf import resolve_checkpoint

    ref_arg = (
        json.dumps(checkpoint_ref) if isinstance(checkpoint_ref, dict) else str(checkpoint_ref)
    )
    return resolve_checkpoint(ref_arg)


def load_model(ref: str, state_dir: Path, api: str | None = None) -> LoadedModel:
    """Resolve ``ref`` end to end and build its container app (not mounted)."""
    slug, tag = parse_ref(ref)
    if not is_safe_slug(slug):
        raise ValueError(f"invalid model slug: {slug!r}")
    base = api or api_base()
    listing = resolve_listing(slug, base)
    fm_project = listing.get("fmProject")
    adapter_name = listing.get("baseAdapter")
    checkpoint_ref = listing.get("checkpointRef")
    if not fm_project or not adapter_name:
        raise ValueError(f"listing {slug!r} has no fmProject/baseAdapter — cannot host it")
    if not checkpoint_ref:
        raise ValueError(f"listing {slug!r} has no checkpointRef — nothing to pull")
    tier = normalize_tier(listing.get("tier"))

    # Planned: route tier-2 adapters to their per-model container.
    # The tier field distinguishes models whose adapter runs inside the studio
    # process ("native") from those packaged with their own runtime
    # ("adapter"). Once that routing exists, a tier-2 model is NOT loaded here:
    # the daemon starts (or attaches to) that model's container and mounts a
    # reverse proxy at mount_path(slug) so the client-facing API is unchanged.
    # v1 hosts both tiers in-process via create_app; `tier` is only reported.

    bundle_dir = ensure_bundle(base, slug, str(fm_project))
    checkpoint_dir = ensure_checkpoint(checkpoint_ref)
    manifest_path = write_run_manifest(state_dir, slug, str(adapter_name))

    from pfnstudio_core.serving.fastapi_app import create_app

    app = create_app(
        manifest_path=manifest_path,
        project_root=bundle_dir,
        checkpoint_dir=checkpoint_dir,
        title=f"PFN model — {slug}",
    )
    return LoadedModel(
        slug=slug,
        fm_project=str(fm_project),
        adapter=str(adapter_name),
        checkpoint_dir=checkpoint_dir,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        tier=tier,
        tag=tag,
        digest=read_digest(bundle_dir),
        loaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        app=app,
    )


# ── host app ─────────────────────────────────────────────────────────────────


def build_host_app(
    registry: Registry | None = None,
    state_dir: Path | None = None,
    *,
    keep_alive: float | None = None,
) -> Any:
    """The management app that mounts one container app per loaded model.

    ``keep_alive``: seconds a model may sit idle before it is unloaded. The
    default (None) never unloads — models stay until ``/unload`` or exit.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from starlette.routing import Mount
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fastapi is required for `pfnstudio serve` — pip install fastapi uvicorn"
        ) from e

    from pfnstudio_core.serving.auth import check_auth

    registry = registry if registry is not None else Registry()
    owns_state = state_dir is None
    state = (
        Path(state_dir)
        if state_dir is not None
        else Path(tempfile.mkdtemp(prefix="pfnstudio-daemon-"))
    )
    last_used: dict[str, float] = {}

    app = FastAPI(title="PFN Studio local daemon", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.registry = registry
    app.state.state_dir = state

    async def _require_auth(authorization: str | None = Header(default=None)) -> None:
        err = check_auth(authorization)
        if err:
            raise HTTPException(status_code=401, detail=err)

    def _touching(slug: str, inner: Any) -> Any:
        # ASGI shim: every request to the model counts as use (for keep_alive).
        async def asgi(scope: dict, receive: Any, send: Any) -> None:
            if scope.get("type") == "http":
                last_used[slug] = time.monotonic()
            await inner(scope, receive, send)

        return asgi

    def _mount(model: LoadedModel) -> None:
        app.mount(mount_path(model.slug), _touching(model.slug, model.app), name=model.slug)

    def _unmount(slug: str) -> None:
        path = mount_path(slug)
        app.router.routes[:] = [
            r for r in app.router.routes if not (isinstance(r, Mount) and r.path == path)
        ]

    async def _close(model: LoadedModel) -> None:
        # Starlette does not forward lifespan to mounted apps; run the container
        # app's shutdown handlers ourselves so the loader is closed.
        try:
            await model.app.router.shutdown()
        except Exception:
            pass
        model.app = None

    @app.get("/healthz")
    @app.get("/health")
    async def healthz() -> dict[str, Any]:
        return {"status": "ready", "models": len(registry)}

    @app.get("/ps", dependencies=[Depends(_require_auth)])
    async def ps() -> list[dict[str, Any]]:
        return registry.ps()

    @app.post("/load", dependencies=[Depends(_require_auth)])
    def load(body: dict[str, Any]) -> dict[str, Any]:  # sync → threadpool, keeps /healthz live
        ref = body.get("ref") or body.get("slug") or body.get("model")
        if not isinstance(ref, str) or not ref.strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_payload", "detail": "body needs a model ref"},
            )
        try:
            slug, _tag = parse_ref(ref)
            mount_path(slug)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail={"error": "invalid_ref", "detail": str(e)}
            ) from e
        with registry.lock:  # one load at a time; repeat loads are idempotent
            existing = registry.get(slug)
            if existing is not None:
                last_used[slug] = time.monotonic()
                return {**existing.summary(), "already_loaded": True}
            try:
                model = load_model(ref, state)
            except ModelNotFound as e:
                raise HTTPException(
                    status_code=404, detail={"error": "not_found", "detail": str(e)}
                ) from e
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail={"error": "invalid_ref", "detail": str(e)}
                ) from e
            except ModuleNotFoundError as e:
                raise HTTPException(
                    status_code=424,
                    detail={
                        "error": "missing_dependency",
                        "detail": f"the adapter needs a package that isn't installed: {e.name or e}",
                        "hint": "pip install -r <bundle_dir>/requirements.txt",
                    },
                ) from e
            except Exception as e:
                import traceback

                traceback.print_exc(file=sys.stderr)
                raise HTTPException(
                    status_code=500,
                    detail={"error": "load_failed", "detail": f"{type(e).__name__}: {e}"},
                ) from e
            _mount(model)
            registry.add(model)
            last_used[slug] = time.monotonic()
            return {**model.summary(), "already_loaded": False}

    @app.post("/unload", dependencies=[Depends(_require_auth)])
    async def unload(body: dict[str, Any]) -> dict[str, Any]:
        slug = body.get("slug") or body.get("ref") or body.get("model")
        if not isinstance(slug, str) or not slug.strip():
            raise HTTPException(
                status_code=400, detail={"error": "invalid_payload", "detail": "body needs a slug"}
            )
        slug = parse_ref(slug)[0]
        with registry.lock:
            model = registry.remove(slug)
            if model is None:
                raise HTTPException(status_code=404, detail={"error": "not_loaded", "slug": slug})
            _unmount(slug)
            last_used.pop(slug, None)
        await _close(model)
        return {"slug": slug, "unloaded": True}

    if keep_alive is not None:
        # Opt-in idle unload; idleness = no request to the model (see _touching).
        async def _reaper() -> None:
            import asyncio

            while True:
                await asyncio.sleep(keep_alive)
                now = time.monotonic()
                for slug, t in list(last_used.items()):
                    if now - t >= keep_alive:
                        with registry.lock:
                            model = registry.remove(slug)
                            if model is None:
                                continue
                            _unmount(slug)
                            last_used.pop(slug, None)
                        await _close(model)

        @app.on_event("startup")
        async def _start_reaper() -> None:
            import asyncio

            asyncio.get_running_loop().create_task(_reaper())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        for model in registry:
            registry.remove(model.slug)
            await _close(model)
        if owns_state:
            shutil.rmtree(state, ignore_errors=True)

    return app


def main(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, keep_alive: float | None = None
) -> None:
    """Run the daemon in the foreground until SIGINT/SIGTERM."""
    app = build_host_app(keep_alive=keep_alive)
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "uvicorn is required for `pfnstudio serve` — pip install fastapi uvicorn"
        ) from e
    print(json.dumps({"event": "starting", "host": host, "port": port}), flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


def _cli(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pfnstudio.daemon", description="PFN Studio local model daemon."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--keep-alive",
        type=float,
        default=None,
        help="Seconds a model may sit idle before it is unloaded (default: never).",
    )
    args = parser.parse_args(argv)
    main(args.host, args.port, keep_alive=args.keep_alive)


if __name__ == "__main__":
    _cli()
