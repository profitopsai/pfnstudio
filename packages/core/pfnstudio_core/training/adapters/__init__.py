"""Model adapters — the plug-in seam for continued-pretraining EXTERNAL base
models (TCPFN, Do-PFN, ...) through their OWN architecture + loss, instead of
the studio's block-composed model.

Weights alone can't be continued — you need the architecture that produced
them. An Adapter is how a base model brings that architecture: it knows how to
build the model + load the checkpoint, sample the (counterfactual) prior the
model's loss needs, run one train step, and save the result. The studio core
runs a single generic loop over these five methods, so adding a new base model
is "author an adapter + declare `adapter`" — never a core change.

An adapter is authored exactly like a prior: a Python file with
`@register_adapter("<name>")` under `adapters/` (discovered by
`discover_in_project`), or shipped in the model's own package and declared via
the base model's `adapterModule`. This `__init__` deliberately does NOT import
the concrete adapters, so registration stays lazy (import-on-use).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    """Contract a base model implements to be continue-pretrainable in the
    studio. Tensors live on `device`; the generic loop owns the step loop,
    backward, grad-clip, optimizer.step, progress + checkpoint IO."""

    def load_model(self, checkpoint_dir: Path, hp: dict[str, Any], device: Any) -> Any:
        """Build the architecture and load the base weights from
        `checkpoint_dir` (the runner staged them at `<ws>/init_checkpoint/`).
        Return an nn.Module on `device`, in train() mode."""
        ...

    def make_optimizer(self, model: Any, hp: dict[str, Any]) -> Any:
        """Return a torch optimizer over the model's parameters."""
        ...

    def make_sampler(
        self, hp: dict[str, Any], device: Any, prior: Any = None
    ) -> Callable[[int], dict[str, Any]]:
        """Return a callable(step) -> batch. The batch is whatever `train_step`
        consumes — for causal models, the model's own counterfactual prior.

        ``prior`` is the run's selected, registered prior instance (a first-class
        ``priors/<id>/prior.py`` object), or None. Adapters that want the
        operator-editable prior to be the source of truth should call
        ``prior.sample(...)`` instead of hardcoding a sampler. Declaring the
        ``prior`` parameter is opt-in: adapters that keep the 2-arg signature
        are still called the old way.
        """
        ...

    def train_step(self, model: Any, batch: dict[str, Any], step: int) -> Any:
        """One forward pass returning a scalar loss tensor (before backward)."""
        ...

    def save(self, model: Any, out_dir: Path, hp: dict[str, Any], step: int) -> None:
        """Write the trained checkpoint into `out_dir` — the runner copies every
        file from here to the run's checkpointPath. Must be re-loadable by
        `load_model` for further continued pretraining."""
        ...

    def predict(self, model: Any, payload: dict[str, Any], hp: dict[str, Any], device: Any) -> dict[str, Any]:
        """OPTIONAL. Run inference through the base model's own architecture +
        real weights (the `model` returned by `load_model`). This is what makes
        "install a base model → Predict" use the actual pretrained weights
        instead of the studio placeholder model.

        `payload` is the predict request body (may be empty). With no user data,
        implementations sample a task from the model's own prior and return
        predicted-vs-ground-truth, so the result both USES the real weights and
        SHOWS they work. Return a JSON-serialisable dict (the run-detail page
        renders it). Adapters without this method fall back to a clear error."""
        ...

    def capabilities(self) -> dict[str, Any]:
        """OPTIONAL. Declare the inference tasks this adapter supports and each
        task's declarative I/O schema, as
            { "<task>": {"label", "description", "input_schema",
                         "output_schema", "sample"?} }

        This is the generic Try-it / Inference contract: the Studio renders a
        schema-driven form + result per task from these schemas — across BOTH the
        run-detail Try-it panel AND the Inference tab — and ``predict`` dispatches
        on a ``{task, inputs}`` envelope. ``input_schema.kind`` selects a form
        template (e.g. 'context_query', 'table', 'table_with_target');
        ``output_schema.kind`` selects a result renderer (e.g. 'per_row_scalar',
        'graph', 'ranked_list'). Adapters without this method fall back to the
        single opaque predict() (legacy, task-inferred-from-eval behaviour).

        ``sample`` (OPTIONAL, per task) — a runnable example the Studio pre-fills
        into the EMPTY form fields so a task lands one click away from a result.
        This is where each model ships ITS OWN domain-appropriate example data:
        the generic UI owns none of it. Keys map to the form fields of the task's
        ``input_schema.kind``; values are strings (textarea/input contents,
        exactly what a user would type or upload as CSV):
            input_schema.kind      sample keys
            ─────────────────      ─────────────────────────────────────────
            'context_query'        context, query
            'table'                rows, columns, (target)
            'table_with_target'    rows, columns, target, (event_time)
        Only non-empty fields are seeded, so a sample never clobbers user input.
        Ship a sample that actually returns a clean result on YOUR weights —
        it is the first thing a prospect sees."""
        ...


def resolve_adapter(name: str, adapter_module: str | None) -> Adapter:
    """Resolve the adapter for `name`. If not already registered (e.g. by
    `discover_in_project`), import the declared `adapter_module` (which
    self-registers via @register_adapter), else fall back to the bundled
    reference adapter `...training.adapters.<name>`."""
    from pfnstudio_core.registry import get_adapter

    try:
        return get_adapter(name)()
    except KeyError:
        pass
    module = adapter_module or f"pfnstudio_core.training.adapters.{name.replace('-', '_')}"
    importlib.import_module(module)  # triggers @register_adapter on import
    return get_adapter(name)()


def run_adapter_training(
    *,
    adapter_name: str,
    adapter_module: str | None,
    run: Any,
    workspace_dir: Path,
    emit_event: Callable[..., None],
    emit_log: Callable[[str], None],
) -> dict[str, Any]:
    """Generic continued-pretraining loop over an Adapter: load the base model,
    sample the model's prior, step N times (backward + grad-clip +
    optimizer.step), write a checkpoint, and return a results dict carrying
    `checkpoint_dir` so the runner persists it."""
    import torch

    hp: dict[str, Any] = dict(getattr(run, "hyperparams", {}) or {})
    steps = int(hp.get("steps", 1000))
    seed = int(hp.get("seed", 42))
    grad_clip = float(hp.get("grad_clip", 1.0))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        adapter = resolve_adapter(adapter_name, adapter_module)
    except (KeyError, ImportError, ModuleNotFoundError) as e:
        return {
            "status": "error",
            "reason": (
                f"adapter '{adapter_name}' is not available ({type(e).__name__}: {e}). "
                "The model's package/adapter must be installed in this environment."
            ),
        }

    ckpt_dir = Path(workspace_dir) / "init_checkpoint"
    emit_log(f"the '{adapter_name}' adapter: loading base model on {device} …")
    try:
        model = adapter.load_model(ckpt_dir, hp, device)
        model.train()
    except Exception as e:  # noqa: BLE001 — surface as a run error, not a crash
        # Disambiguate the adapter's NAME from a missing dependency PACKAGE:
        # e.g. the 'tcpfn' adapter needing the (separate) 'tcpfn' Python
        # package. Without this, "adapter tcpfn: No module named 'tcpfn'"
        # reads as one thing when it's two.
        hint = ""
        if isinstance(e, ModuleNotFoundError):
            hint = (
                f" — the '{adapter_name}' adapter needs the '{e.name}' Python "
                f"package installed in this environment (it is a dependency of "
                f"the adapter, not the adapter itself). Install it into the "
                f"studio venv (e.g. pip install {e.name}) and re-run."
            )
        return {
            "status": "error",
            "reason": (
                f"the '{adapter_name}' adapter: failed to load its base model — "
                f"{type(e).__name__}: {e}{hint}"
            ),
        }

    optimizer = adapter.make_optimizer(model, hp)

    # Resolve the run's selected prior so the adapter can sample from a
    # first-class, operator-editable prior OBJECT (priors/<id>/prior.py) rather
    # than hardcoding one. discover_in_project() already ran in the caller, so
    # the prior is registered. Passed only to adapters whose make_sampler opts
    # in via a `prior` parameter — older 2-arg adapters keep working unchanged.
    prior_instance = None
    try:
        from pfnstudio_core.registry import get_prior

        _prior_id = getattr(getattr(run, "prior", None), "id", None)
        if _prior_id:
            prior_instance = get_prior(_prior_id)()
    except Exception:  # noqa: BLE001 — prior is optional; adapter may not use it
        prior_instance = None

    import inspect as _inspect

    if "prior" in _inspect.signature(adapter.make_sampler).parameters:
        sampler = adapter.make_sampler(hp, device, prior=prior_instance)
    else:
        sampler = adapter.make_sampler(hp, device)

    emit_log(f"the '{adapter_name}' adapter: continuing pretraining for {steps} steps (seed {seed}) …")
    emit_event("start", steps=steps, adapter=adapter_name)

    losses: list[float] = []
    log_every = max(1, steps // 100)
    try:
        for step in range(steps):
            batch = sampler(step)
            optimizer.zero_grad(set_to_none=True)
            loss = adapter.train_step(model, batch, step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lv = float(loss.detach().to("cpu"))
            losses.append(lv)
            if step % log_every == 0 or step == steps - 1:
                emit_event("step", step=step, loss=lv)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "error",
            "reason": f"the '{adapter_name}' adapter: training step failed at step {len(losses)} — {type(e).__name__}: {e}",
        }

    out_dir = Path(workspace_dir) / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        adapter.save(model, out_dir, hp, steps)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "reason": f"the '{adapter_name}' adapter: checkpoint save failed — {type(e).__name__}: {e}"}

    tail = losses[-max(1, steps // 10):] if losses else []
    results = {
        "status": "ok",
        "adapter": adapter_name,
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last_10pct": (sum(tail) / len(tail)) if tail else None,
        "checkpoint_dir": str(out_dir),
    }
    emit_event("done", **results)
    return results
