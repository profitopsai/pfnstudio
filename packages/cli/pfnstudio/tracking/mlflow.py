"""MLflow tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tracker


class MLflowTracker(Tracker):
    name = "mlflow"

    def __init__(self) -> None:
        self._active = False

    def start(self, run_id: str, config: dict[str, Any], project_root: Path) -> None:
        try:
            import mlflow
        except ImportError:
            return
        experiment = config.get("tracking", {}).get("mlflow_experiment") or "pfnstudio"
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_id)
        flat = _flatten(config)
        for k, v in flat.items():
            mlflow.log_param(k, v)
        self._active = True

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        import mlflow

        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def finish(self, results: dict[str, Any]) -> None:
        if not self._active:
            return
        import mlflow

        for k, v in results.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"final_{k}", v)
        mlflow.end_run()


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
