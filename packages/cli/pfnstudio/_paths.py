"""Locate bundled schemas and starters.

When running from source: look up the tree for `schemas/` and `starters/`.
When pip-installed: look in `pfnstudio/_bundled/` (populated by
`scripts/sync-bundled.sh` before build).
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _asset_root(name: str) -> Path:
    """Resolve an asset directory, bundled first, then from the source tree.

    The source-tree walk only accepts a parent that also holds a
    `pyproject.toml`, so it cannot wander past the repo root and pick up a
    same-named directory elsewhere on the machine.
    """
    bundled = _HERE / "_bundled" / name
    if bundled.is_dir():
        return bundled
    for parent in _HERE.parents:
        candidate = parent / name
        if candidate.is_dir() and (parent / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {name}/ directory.")


def schemas_root() -> Path:
    return _asset_root("schemas")


def starters_root() -> Path:
    return _asset_root("starters")


def fm_project_template() -> Path:
    return starters_root() / "fm-project"
