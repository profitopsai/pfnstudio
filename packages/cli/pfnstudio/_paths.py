"""Locate bundled schemas and templates.

When running from source: look up the tree for `schemas/` and `templates/`.
When pip-installed: look in `priorstudio/_bundled/` (populated by hatch at build time).
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent


def schemas_root() -> Path:
    bundled = _HERE / "_bundled" / "schemas"
    if bundled.exists():
        return bundled
    for parent in _HERE.parents:
        candidate = parent / "schemas"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate schemas/ directory.")


def templates_root() -> Path:
    bundled = _HERE / "_bundled" / "templates"
    if (bundled / "fm-project").is_dir():
        return bundled
    for parent in _HERE.parents:
        for directory in ("templates", "starters"):
            candidate = parent / directory
            if (candidate / "fm-project").is_dir():
                return candidate
    raise FileNotFoundError("Could not locate templates/fm-project or starters/fm-project.")


def fm_project_template() -> Path:
    return templates_root() / "fm-project"
