"""Weights & Biases tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tracker


class WandbTracker(Tracker):
    name = "wandb"

    def __init__(self) -> None:
        self._run = None

    def start(self, run_id: str, config: dict[str, Any], project_root: Path) -> None:
        try:
            import wandb
        except ImportError:
            self._run = None
            return
        project = config.get("tracking", {}).get("wandb_project") or "pfnstudio"
        self._run = wandb.init(project=project, name=run_id, config=config, reinit=True)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run is None:
            return
        self._run.log(metrics, step=step)

    def finish(self, results: dict[str, Any]) -> None:
        if self._run is None:
            return
        self._run.summary.update(
            {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
        )
        self._run.finish()
