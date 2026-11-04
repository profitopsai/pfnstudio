"""Sample inputs for the marketplace base models, shipped with pfnstudio.

``samples/<slug>/<task>.json`` is a ready ``--input`` file for
``pfnstudio run <slug>``: the ``{task, inputs}`` request that runs the model on
the sample its adapter declares. ``pfnstudio run <slug> --input sample`` uses
it and prints its path, so a developer can copy it and swap in their own data.

Each file is the adapter's declared sample converted by
``pfnstudio_core.serving.samples.sample_to_inputs``. ``./run.sh verify``
compares every file with the sample the running model declares, so a changed
adapter sample shows up before a release instead of drifting.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_INPUT = "sample"
"""The ``--input`` value that selects the shipped sample."""

_DIR = Path(__file__).resolve().parent


def bundled_sample_tasks(slug: str) -> list[str]:
    """The tasks pfnstudio ships a sample input for, for ``slug``."""
    d = _DIR / slug
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def bundled_sample(slug: str, task: str | None) -> Path | None:
    """The shipped sample input for ``slug``'s ``task`` (or its only one when
    ``task`` is None), or None when there is none."""
    tasks = bundled_sample_tasks(slug)
    if task is None:
        task = tasks[0] if len(tasks) == 1 else None
    if task is None or task not in tasks:
        return None
    return _DIR / slug / f"{task}.json"
