"""Shared warm loader construction for stdlib and FastAPI serve paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_serve_loader(
    *,
    manifest_path: Path,
    project_root: Path,
    checkpoint_dir: Path,
) -> Any:
    """Load a checkpoint once and return a loader with ``predict(payload, ...)``.

    Uses the adapter path when the run declares one; otherwise falls back to
    :class:`pfnstudio_core.training.predict.ModelLoader`.
    """
    from ..training.predict import ModelLoader, _peek_run_adapter
    from .worker import _AdapterServeLoader

    adapter_name = _peek_run_adapter(manifest_path)
    if adapter_name:
        return _AdapterServeLoader(adapter_name, manifest_path, project_root, checkpoint_dir)
    return ModelLoader(
        manifest_path=manifest_path,
        project_root=project_root,
        checkpoint_dir=checkpoint_dir,
    )


def fetch_capabilities(
    *,
    loader: Any,
    manifest_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Return the capability manifest without running inference.

    Native studio models answer from declarative YAML (no weights read).
    Adapter-backed runs delegate to the warm loader's capabilities().
    """
    from ..training.predict import _native_capabilities, _peek_run_adapter

    if _peek_run_adapter(manifest_path):
        return loader.predict({"__capabilities__": True})
    return _native_capabilities(manifest_path=manifest_path, project_root=project_root)
