"""Local compute adapter — runs the PFN training loop in-process."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .base import ComputeAdapter


def _emit_event(event: str, **fields: Any) -> None:
    """Emit one JSON line event on stdout when run via the PFN Studio API
    (PFNSTUDIO_JSON_PROGRESS=1). Falls back to no-op when interactive."""
    if os.environ.get("PFNSTUDIO_JSON_PROGRESS") != "1":
        return
    sys.stdout.write(json.dumps({"event": event, **fields, "ts": time.time()}) + "\n")
    sys.stdout.flush()


def _emit_log(message: str) -> None:
    """Convenience wrapper for the common 'log' event."""
    if os.environ.get("PFNSTUDIO_JSON_PROGRESS") == "1":
        _emit_event("log", line=message)
    else:
        sys.stderr.write(f"{message}\n")


class LocalAdapter(ComputeAdapter):
    name = "local"

    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        from pfnstudio_core.datasets import RegistryDatasetLoader
        from pfnstudio_core.loaders import load_model, load_prior, load_run
        from pfnstudio_core.model import Model
        from pfnstudio_core.registry import discover_in_project, get_prior
        from pfnstudio_core.training import train_pfn

        sys.path.insert(0, str(project_root))
        try:
            discover_in_project(project_root)

            # Surface the registry datasets the API resolved for this run.
            # Available to scorers via `loader.load_table(...)`. Emits one
            # log line per loaded dataset so the live run UI shows the data
            # actually flowing into the subprocess, not just being declared.
            ds_loader = RegistryDatasetLoader.from_project(project_root)
            for key in ds_loader.available:
                ds = ds_loader._index[key]
                _emit_log(f"dataset {key} loaded from {ds.dir / ds.filename}")

            run = load_run(run_yaml)

            # ── Adapter dispatch ─────────────────────────────────────────
            # A run may train through a FOREIGN base model's own architecture +
            # loss (continued-pretraining an external model like TCPFN) instead
            # of the studio's block-composed model. This is the single generic
            # branch — each model is a registered adapter, never a per-model
            # branch here. No `hyperparams.adapter` = the native path below.
            # `discover_in_project` above already imported any project-local
            # adapters/<name>.py, so their @register_adapter has fired.
            adapter_name = (run.hyperparams or {}).get("adapter")
            if adapter_name:
                from pfnstudio_core.training.adapters import run_adapter_training

                return run_adapter_training(
                    adapter_name=str(adapter_name),
                    adapter_module=(run.hyperparams or {}).get("adapterModule"),
                    run=run,
                    workspace_dir=project_root,
                    emit_event=_emit_event,
                    emit_log=_emit_log,
                )

            prior_yaml = project_root / "priors" / run.prior.id / "prior.yaml"
            if not prior_yaml.exists():
                return {"status": "error", "reason": f"prior.yaml not found at {prior_yaml}"}
            prior_spec = load_prior(prior_yaml)

            try:
                prior_cls = get_prior(run.prior.id)
            except KeyError:
                return {
                    "status": "error",
                    "reason": (
                        f"prior '{run.prior.id}' has a YAML spec but no Python class "
                        f'registered via @register_prior("{run.prior.id}"). '
                        "See examples/tcpfn for the registration pattern."
                    ),
                }

            prior = prior_cls()
            prior.spec = prior_spec

            model_yaml = project_root / "models" / f"{run.model.id}.yaml"
            if not model_yaml.exists():
                return {"status": "error", "reason": f"model.yaml not found at {model_yaml}"}
            model_spec = load_model(model_yaml)

            # Seed torch BEFORE constructing the model so every block's
            # default-init (nn.Linear / nn.TransformerEncoder weights, etc.)
            # is deterministic per the run's `hyperparams.seed`. train_pfn
            # re-seeds again inside its own scope for LazyLinear / any later
            # stochastic ops — these two seedings are independent and both
            # needed for end-to-end determinism on the same machine.
            try:
                import torch as _torch

                _torch.manual_seed(int(run.hyperparams.get("seed", 42)))
            except ImportError:
                pass

            try:
                model = Model(model_spec)
            except KeyError as e:
                return {"status": "error", "reason": f"unregistered block: {e}"}
            except ImportError as e:
                return {"status": "skipped", "reason": str(e)}

            # Periodic-eval callback for the trainer. We close over the
            # adapter's own scorer dispatch, so each mid-training cycle
            # uses the same logic as the end-of-run eval — just at a
            # different step. The trainer flushes one `eval` event per
            # slug per cycle; the UI's live scorecard aggregates these
            # by slug+metric to draw the convergence trajectory.
            #
            # `step` is passed through purely for logging / debugging;
            # the trainer is the source of truth for which step it ran at.
            def _periodic_eval(at_step: int) -> dict[str, Any]:
                return self._run_dataset_scorers(
                    model=model,
                    run=run,
                    project_root=project_root,
                    loader=ds_loader,
                )

            results = train_pfn(
                model=model,
                prior=prior,
                run=run,
                eval_fn=_periodic_eval,
            )

            # Run any registry-dataset scorers after training. Each eval in
            # the run's evalRefs is matched against BUILTIN_SCORERS by slug;
            # matched scorers compute real-data metrics from the loaded
            # registry datasets, unmatched ones are skipped (synthetic-only).
            eval_outputs = self._run_dataset_scorers(
                model=model,
                run=run,
                project_root=project_root,
                loader=ds_loader,
            )
            if eval_outputs:
                results["evals"] = eval_outputs
                for slug, payload in eval_outputs.items():
                    if payload.get("skipped"):
                        _emit_log(f"eval {slug} skipped: {payload.get('skip_reason')}")
                    else:
                        metric_summary = ", ".join(
                            f"{k}={v:.4g}" for k, v in payload.get("metrics", {}).items()
                        )
                        _emit_log(f"eval {slug} scored: {metric_summary}")

                # Re-emit `done` so the API's lastDone capture sees the eval
                # results. train_pfn already emitted its own done with the
                # training fields; this augments those with the new `evals`
                # block. The API persists the most recent done into
                # run.results, so this is what shows up on the UI.
                _emit_event("done", **results)

            return results
        finally:
            if str(project_root) in sys.path:
                sys.path.remove(str(project_root))

    def _run_dataset_scorers(
        self,
        *,
        model: Any,
        run: Any,
        project_root: Path,
        loader: Any,
    ) -> dict[str, dict[str, Any]]:
        """For each eval in the run, look up a built-in scorer by slug and
        execute it. Returns slug → serialisable result dict suitable for
        merging into the run's `results.evals` JSON column."""
        from pfnstudio_core.loaders import load_eval
        from pfnstudio_core.registry import get_scorer
        from pfnstudio_core.scorers import BUILTIN_SCORERS

        out: dict[str, dict[str, Any]] = {}
        for ref in getattr(run, "evals", []) or []:
            eval_id = getattr(ref, "id", None) or getattr(ref, "slug", None)
            if not eval_id:
                continue
            eval_yaml = project_root / "evals" / f"{eval_id}.yaml"
            if not eval_yaml.exists():
                continue
            try:
                eval_spec = load_eval(eval_yaml)
            except Exception as e:
                _emit_log(f"eval {eval_id} skipped: could not load spec ({e})")
                continue

            # Resolve the scorer: a template-shipped @register_scorer (loaded by
            # discover_in_project in submit()) wins, so paper-specific scorers
            # live in the template; core's BUILTIN_SCORERS are the generic
            # fallback. This is the single scoring path — there is no separate
            # Eval.score contract anymore.
            scorer: Any = None
            for key in (eval_spec.id, eval_id):
                try:
                    scorer = get_scorer(key)()
                    break
                except KeyError:
                    continue
            if scorer is None:
                scorer = BUILTIN_SCORERS.get(eval_spec.id) or BUILTIN_SCORERS.get(eval_id)
            if scorer is None:
                # No scorer for this slug — fine, the eval is synthetic-only
                # (metadata card with no real-data scoring step).
                continue

            try:
                result = scorer.score(
                    model=model,
                    eval_spec=eval_spec,
                    loader=loader,
                    run_spec=run,
                )
                out[eval_id] = {
                    "metrics": result.metrics,
                    "meta": result.meta,
                    "skipped": result.skipped,
                    "skip_reason": result.skip_reason,
                }
            except Exception as e:
                out[eval_id] = {
                    "metrics": {},
                    "meta": {},
                    "skipped": True,
                    "skip_reason": f"scorer raised: {type(e).__name__}: {e}",
                }
                _emit_log(f"eval {eval_id} crashed: {type(e).__name__}: {e}")
        return out
