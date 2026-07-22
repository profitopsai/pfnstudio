"""Decorator-based registries for priors, architecture blocks, scorers, and
model adapters (the plug-in seam for continued-pretraining external base
models like TCPFN through their OWN architecture + loss)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_PRIORS: dict[str, type] = {}
_BLOCKS: dict[str, type] = {}
_SCORERS: dict[str, type] = {}
_ADAPTERS: dict[str, type] = {}


def _register(table: dict[str, type], name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in table and table[name] is not cls:
            raise ValueError(f"Already registered under '{name}': {table[name]!r}")
        table[name] = cls
        return cls

    return decorator


def register_prior(name: str) -> Callable[[type[T]], type[T]]:
    """Register a Prior subclass so YAML specs can reference it by `id`."""
    return _register(_PRIORS, name)


def register_block(name: str) -> Callable[[type[T]], type[T]]:
    """Register an architecture block so model YAML can reference it by `type`."""
    return _register(_BLOCKS, name)


def register_scorer(name: str) -> Callable[[type[T]], type[T]]:
    """Register a DatasetScorer subclass so an eval slug resolves to it.

    Ship a scorer in a template beside its eval spec — `evals/<slug>.py`
    with `@register_scorer("<slug>")` — and `discover_in_project` imports it
    at run time. Paper-specific scorers live in the template, not core.
    """
    return _register(_SCORERS, name)


def register_adapter(name: str) -> Callable[[type[T]], type[T]]:
    """Register a model Adapter so a run's `hyperparams.adapter` resolves to it.

    An adapter is the plug-in seam for continued-pretraining an EXTERNAL base
    model through its own architecture + loss (instead of the studio's
    block-composed model). Ship it as `adapters/<name>.py` with
    `@register_adapter("<name>")` — `discover_in_project` imports it at run time
    — or in the model's own installed package.
    """
    return _register(_ADAPTERS, name)


def get_prior(name: str) -> type:
    # Loose match: tolerate `-` ↔ `_` swaps. The DB slugifier converts `_` to
    # `-` (URL-safe), but Python decorators register with idiomatic snake_case.
    # Without this, a prior registered as "linear_regression" can't be found
    # under its YAML id "linear-regression".
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        if candidate in _PRIORS:
            return _PRIORS[candidate]
    raise KeyError(f"No prior registered under '{name}'. Available: {sorted(_PRIORS)}")


def get_block(name: str) -> type:
    # Tolerate `-` ↔ `_` swaps, same as get_prior: the DB slugifier uses `-`
    # but @register_block names are idiomatic snake_case.
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        if candidate in _BLOCKS:
            return _BLOCKS[candidate]
    raise KeyError(f"No block registered under '{name}'. Available: {sorted(_BLOCKS)}")


def get_scorer(name: str) -> type:
    # Tolerate `-` ↔ `_` swaps, same as get_prior: the DB slugifier uses `-`
    # but decorators register idiomatic snake_case.
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        if candidate in _SCORERS:
            return _SCORERS[candidate]
    raise KeyError(f"No scorer registered under '{name}'. Available: {sorted(_SCORERS)}")


def get_adapter(name: str) -> type:
    # Tolerate `-` ↔ `_` swaps, same as get_prior: the DB slugifier uses `-`
    # but decorators register idiomatic snake_case.
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        if candidate in _ADAPTERS:
            return _ADAPTERS[candidate]
    raise KeyError(f"No adapter registered under '{name}'. Available: {sorted(_ADAPTERS)}")


def list_priors() -> list[str]:
    return sorted(_PRIORS)


def list_blocks() -> list[str]:
    return sorted(_BLOCKS)


def list_scorers() -> list[str]:
    return sorted(_SCORERS)


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)


def _clear_for_tests() -> None:
    _PRIORS.clear()
    _BLOCKS.clear()
    _SCORERS.clear()
    _ADAPTERS.clear()


def discover_in_project(project_root: Any) -> None:
    """Import every Python module under priors/, evals/, models/, blocks/ and
    adapters/ so decorators register (priors, scorers, blocks, adapters).

    project_root: pathlib.Path — typed loosely to avoid an import cycle.
    """
    import importlib.util
    from pathlib import Path

    root = Path(project_root)
    for sub in ("priors", "evals", "models", "blocks", "adapters"):
        d = root / sub
        if not d.exists():
            continue
        for py in d.rglob("*.py"):
            spec = importlib.util.spec_from_file_location(
                f"_ps_dyn_{py.stem}_{abs(hash(str(py)))}", py
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
