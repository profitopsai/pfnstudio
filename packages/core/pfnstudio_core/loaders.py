"""Load YAML artifacts into typed pydantic objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from .eval import EvalSpec
from .eval_normalize import normalize_eval_raw
from .model import ModelSpec
from .model_blocks import normalize_model_blocks
from .prior import PriorSpec
from .run import RunSpec


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prior(path: Path) -> PriorSpec:
    return PriorSpec.model_validate(_read_yaml(path))


def load_model(path: Path) -> ModelSpec:
    raw = _read_yaml(path)
    if isinstance(raw.get("blocks"), list):
        heads = raw.get("output_heads") if isinstance(raw.get("output_heads"), list) else None
        raw["blocks"] = normalize_model_blocks(raw["blocks"], output_heads=heads)
    return ModelSpec.model_validate(raw)


def load_eval(path: Path) -> EvalSpec:
    raw = _read_yaml(path)
    if isinstance(raw, dict):
        raw = normalize_eval_raw(raw)
    return EvalSpec.model_validate(raw)


def load_run(path: Path) -> RunSpec:
    return RunSpec.model_validate(_read_yaml(path))


def load_project(project_root: Path) -> dict[str, list]:
    """Discover every artifact under a project root."""
    root = Path(project_root)
    return {
        "priors": [load_prior(p) for p in (root / "priors").rglob("prior.yaml")],
        "models": [load_model(p) for p in (root / "models").glob("*.yaml")],
        "evals": [load_eval(p) for p in (root / "evals").glob("*.yaml")],
        "runs": [load_run(p) for p in (root / "runs").glob("*.yaml")],
    }
