"""Do-PFN CID and CATE recovery scorer for PFN Studio.

Paste this entire file into ``evals/cid_recovery.py``.

This version supports the corrected project blocks:

* ``dopfn_input_encoder`` with a dynamically located masked Y column;
* ``bar_distribution_head`` whose decoder and borders live on ``decoder``;
* CUDA training/evaluation without mixing CPU inputs and GPU parameters.

It runs both the general random-DAG recovery check and the paper's six fixed
case studies: Observed Confounder, Observed Mediator, Confounder + Mediator,
Unobserved Confounder, Back-Door Criterion, and Front-Door Criterion.
"""

from __future__ import annotations

import json
from typing import Any

from pfnstudio_core.registry import register_scorer
from pfnstudio_core.scorers.base import DatasetScorer, ScorerResult

NUM_TASKS = 50
SCORER_POINTS = 256
BASE_SEED = 900_000
RIDGE_LAMBDA = 1.0
ORACLE_MC = 64
PICP_MASS = 0.90
_EPS = 1.0e-9

# Paper Section 4.1 / Appendix D.1: 100 independently sampled datasets for
# each of six fixed causal structures. The released evaluator uses 200
# observational rows as in-context evidence; 64 held-out interventional rows
# keep repeated in-training evaluation tractable while still scoring every
# structure on 6,400 query subjects.
CASE_TASKS_PER_STUDY = 100
CASE_CONTEXT_ROWS = 200
CASE_QUERY_ROWS = 64
CASE_BASE_SEED = 1_200_000
CASE_MAX_RETRIES = 20


CASE_STUDIES = (
    {
        "slug": "observed_confounder",
        "label": "Observed Confounder",
        "order": ("C", "T", "Y"),
        "parents": {"C": (), "T": ("C",), "Y": ("C", "T")},
        "observed": ("C",),
    },
    {
        "slug": "observed_mediator",
        "label": "Observed Mediator",
        "order": ("T", "M", "Y"),
        "parents": {"T": (), "M": ("T",), "Y": ("M", "T")},
        "observed": ("M",),
        "randomized_treatment": True,
    },
    {
        "slug": "confounder_mediator",
        "label": "Confounder + Mediator",
        "order": ("C", "T", "M", "Y"),
        "parents": {
            "C": (),
            "T": ("C",),
            "M": ("T",),
            "Y": ("C", "M", "T"),
        },
        "observed": ("M", "C"),
    },
    {
        "slug": "unobserved_confounder",
        "label": "Unobserved Confounder",
        "order": ("U", "C", "T", "Y"),
        "parents": {
            "U": (),
            "C": (),
            "T": ("U", "C"),
            "Y": ("U", "C", "T"),
        },
        "observed": ("C",),
    },
    {
        "slug": "backdoor",
        "label": "Back-Door Criterion",
        "order": ("C", "T", "M", "Y"),
        "parents": {
            "C": (),
            "T": ("C",),
            "M": ("T",),
            "Y": ("C", "M"),
        },
        "observed": ("C",),
    },
    {
        "slug": "frontdoor",
        "label": "Front-Door Criterion",
        "order": ("U", "T", "M", "Y"),
        "parents": {
            "U": (),
            "T": ("U",),
            "M": ("T",),
            "Y": ("U", "M"),
        },
        "observed": ("M",),
    },
)


def _paper_case_activation(name: str, value: Any, np: Any) -> Any:
    """Appendix D.1 activation pool: square, tanh, and ReLU."""
    if name == "square":
        return np.square(value)
    if name == "tanh":
        return np.tanh(value)
    if name == "relu":
        return np.maximum(value, 0.0)
    raise ValueError(f"Unknown case-study activation: {name!r}")


def _sample_case_mechanisms(spec: dict[str, Any], rng: Any, np: Any) -> dict[str, Any]:
    """Sample Kaiming-uniform edge weights and one paper activation/node."""
    mechanisms: dict[str, Any] = {}
    activation_names = np.asarray(["square", "tanh", "relu"], dtype=object)
    for node in spec["order"]:
        parents = tuple(spec["parents"][node])
        if not parents:
            continue
        bound = 1.0 / np.sqrt(float(len(parents)))
        mechanisms[node] = {
            "parents": parents,
            "weights": rng.uniform(-bound, bound, size=len(parents)),
            "activation": str(rng.choice(activation_names)),
        }
    return mechanisms


def _sample_case_noise(
    spec: dict[str, Any],
    rows: int,
    sigma_exo: float,
    sigma_eps: float,
    rng: Any,
) -> dict[str, Any]:
    noise: dict[str, Any] = {}
    for node in spec["order"]:
        is_root = len(spec["parents"][node]) == 0
        if node == "T" and is_root and spec.get("randomized_treatment", False):
            noise[node] = rng.integers(0, 2, size=rows).astype("float64")
        else:
            scale = sigma_exo if is_root else sigma_eps
            noise[node] = rng.normal(0.0, scale, size=rows)
    return noise


def _case_forward(
    spec: dict[str, Any],
    mechanisms: dict[str, Any],
    noise: dict[str, Any],
    np: Any,
    treatment_override: Any = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in spec["order"]:
        if node == "T" and treatment_override is not None:
            values[node] = np.asarray(treatment_override, dtype=np.float64)
            continue

        parents = tuple(spec["parents"][node])
        if not parents:
            values[node] = np.asarray(noise[node], dtype=np.float64).copy()
            continue

        mechanism = mechanisms[node]
        parent_matrix = np.column_stack([values[parent] for parent in parents])
        linear = parent_matrix @ mechanism["weights"]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            generated = (
                _paper_case_activation(mechanism["activation"], linear, np)
                + noise[node]
            )

        # Released Do-PFN binarizes a generated treatment at its batch mean.
        # Root treatment in Observed Mediator is already Bernoulli.
        if node == "T":
            generated = (generated < generated.mean()).astype(np.float64)
        values[node] = generated
    return values


def _generate_structured_case_task(
    spec: dict[str, Any],
    seed: int,
    np: Any,
) -> dict[str, Any]:
    """Generate one fixed-graph case-study dataset and exact paired CATE."""
    for retry in range(CASE_MAX_RETRIES):
        rng = np.random.default_rng(int(seed) + retry * 1_000_003)
        sigma_exo = float(rng.uniform(1.0, 3.0))
        sigma_eps = float(0.3 * rng.beta(1.0, 5.0))
        mechanisms = _sample_case_mechanisms(spec, rng, np)

        context_noise = _sample_case_noise(
            spec, CASE_CONTEXT_ROWS, sigma_exo, sigma_eps, rng
        )
        query_noise = _sample_case_noise(
            spec, CASE_QUERY_ROWS, sigma_exo, sigma_eps, rng
        )
        context = _case_forward(spec, mechanisms, context_noise, np)

        treatment_query = rng.integers(0, 2, size=CASE_QUERY_ROWS).astype(np.float64)
        query_interventional = _case_forward(
            spec,
            mechanisms,
            query_noise,
            np,
            treatment_override=treatment_query,
        )
        query_do0 = _case_forward(
            spec,
            mechanisms,
            query_noise,
            np,
            treatment_override=np.zeros(CASE_QUERY_ROWS, dtype=np.float64),
        )
        query_do1 = _case_forward(
            spec,
            mechanisms,
            query_noise,
            np,
            treatment_override=np.ones(CASE_QUERY_ROWS, dtype=np.float64),
        )

        context_columns = [context["T"]]
        query_columns = [query_interventional["T"]]
        for covariate in spec["observed"]:
            context_columns.append(context[covariate])
            query_columns.append(query_interventional[covariate])
        context_columns.append(context["Y"])
        query_columns.append(np.full(CASE_QUERY_ROWS, np.nan, dtype=np.float64))
        sequence = np.concatenate(
            [np.column_stack(context_columns), np.column_stack(query_columns)],
            axis=0,
        )
        y_query = np.asarray(query_interventional["Y"], dtype=np.float64)
        cate_true = np.asarray(query_do1["Y"] - query_do0["Y"], dtype=np.float64)

        finite_parts = [
            sequence[:CASE_CONTEXT_ROWS],
            sequence[CASE_CONTEXT_ROWS:, :-1],
            y_query,
            cate_true,
        ]
        if all(np.isfinite(part).all() for part in finite_parts):
            return {
                "X": sequence,
                "y": y_query,
                "cate_true": cate_true,
                "n_ctx": CASE_CONTEXT_ROWS,
                "sigma_exo": sigma_exo,
                "sigma_eps": sigma_eps,
                "retry": retry,
            }

    raise FloatingPointError(
        f"Could not generate a finite {spec['label']} task after "
        f"{CASE_MAX_RETRIES} attempts (seed={seed})."
    )


def _ridge_fit(x: Any, y: Any, lam: float, np: Any) -> Any:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    matrix = xb.T @ xb + float(lam) * np.eye(xb.shape[1])
    return np.linalg.solve(matrix, xb.T @ y)


def _ridge_pred(weights: Any, x: Any, np: Any) -> Any:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return xb @ weights


def _log(level: str, message: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "event": "cid_recovery",
                "level": level,
                "message": message,
                **fields,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


@register_scorer("cid_recovery")
class DoPfnCidRecoveryScorer(DatasetScorer):
    """Measure interventional-outcome and CATE recovery."""

    def score(
        self, *, model: Any, eval_spec: Any, loader: Any, run_spec: Any
    ) -> ScorerResult:
        try:
            import numpy as np
            import torch
            import torch.nn as nn
        except ImportError as exc:
            return ScorerResult(
                metrics={},
                meta={"dependency_missing": str(exc)},
                skipped=True,
                skip_reason=f"missing dependency: {exc}",
            )

        from pfnstudio_core.registry import get_prior
        from pfnstudio_core.training.loop import _split_encoder_heads

        try:
            prior_cls = get_prior(run_spec.prior.id)
        except KeyError:
            return ScorerResult(
                metrics={},
                meta={"prior_id": run_spec.prior.id},
                skipped=True,
                skip_reason=f"prior '{run_spec.prior.id}' is not registered",
            )
        prior = prior_cls()

        modules = list(model.modules)
        encoder, heads = _split_encoder_heads(modules)
        if not heads:
            return ScorerResult(
                metrics={},
                meta={},
                skipped=True,
                skip_reason="Model has no output head.",
            )
        head = heads[0]

        def block_submodules(block: Any) -> list[Any]:
            """Find nn.Modules held by either a built-in or plain project block."""
            if isinstance(block, nn.Module):
                return [block]
            found: list[Any] = []
            for value in vars(block).values():
                if isinstance(value, nn.Module) and all(
                    value is not old for old in found
                ):
                    found.append(value)
            return found

        # Training moves each block's torch submodules to CUDA. Discover the
        # actual device generically instead of assuming the head attribute is
        # named ``proj`` (the corrected BAR head uses ``decoder``).
        model_device = torch.device("cpu")
        device_source = "fallback_cpu"
        for block in [*encoder, *heads]:
            located = False
            for submodule in block_submodules(block):
                parameter = next(submodule.parameters(), None)
                if parameter is not None:
                    model_device = parameter.device
                    device_source = (
                        f"{type(block).__name__}.{type(submodule).__name__}.parameter"
                    )
                    located = True
                    break
                buffer = next(submodule.buffers(), None)
                if buffer is not None:
                    model_device = buffer.device
                    device_source = (
                        f"{type(block).__name__}.{type(submodule).__name__}.buffer"
                    )
                    located = True
                    break
            if located:
                break

        reduce_fn = None
        for _, block in modules:
            candidate = getattr(block, "to_prediction", None)
            if callable(candidate):
                reduce_fn = candidate
                break

        # Support the corrected ``decoder.bar_borders`` and older
        # ``proj.bar_borders`` checkpoints.
        bar_borders = None
        border_source = None
        for _, block in modules:
            for attribute in ("decoder", "proj"):
                carrier = getattr(block, attribute, None)
                borders = getattr(carrier, "bar_borders", None)
                if borders is not None:
                    bar_borders = borders.detach().cpu().numpy().astype(np.float64)
                    border_source = f"{type(block).__name__}.{attribute}.bar_borders"
                    break
            if bar_borders is not None:
                break

        _log(
            "info",
            "evaluation_started",
            device=str(model_device),
            device_source=device_source,
            encoder_blocks=len(encoder),
            head_class=type(head).__name__,
            has_distribution_reducer=reduce_fn is not None,
            border_source=border_source,
            num_tasks=NUM_TASKS,
            points_per_task=SCORER_POINTS,
            oracle_mc=ORACLE_MC,
            structured_cases=len(CASE_STUDIES),
            structured_tasks_per_case=CASE_TASKS_PER_STUDY,
        )

        def run_head_raw(sequence: Any, n_ctx: int) -> Any:
            """Run the same sequential encoder/head path used by training."""
            with torch.no_grad():
                encoded = (
                    torch.from_numpy(np.asarray(sequence, dtype=np.float32))
                    .unsqueeze(0)
                    .to(model_device)
                )
                for block_index, block in enumerate(encoder):
                    if n_ctx > 0 and getattr(block, "needs_single_eval_pos", False):
                        result = block(encoded, single_eval_pos=n_ctx)
                    else:
                        result = block(encoded)
                    encoded = result[0] if isinstance(result, tuple) else result
                    if not torch.isfinite(encoded).all():
                        raise FloatingPointError(
                            "Non-finite encoder output during CID eval at "
                            f"block {block_index} ({type(block).__name__})."
                        )
                output = head(encoded)
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Non-finite head output during CID eval.")
                return output[:, n_ctx:, :]

        def to_scalar(raw: Any) -> Any:
            reduced = reduce_fn(raw) if reduce_fn is not None else raw
            return reduced[0, :, 0].detach().cpu().numpy().astype(np.float64)

        def run_model(sequence: Any, n_ctx: int) -> Any:
            return to_scalar(run_head_raw(sequence, n_ctx))

        def locate_y_column(sequence: Any, n_ctx: int) -> int:
            context = sequence[:n_ctx]
            query = sequence[n_ctx:]
            candidates = np.isfinite(context).any(axis=0) & (~np.isfinite(query)).all(
                axis=0
            )
            indices = np.flatnonzero(candidates)
            if indices.size != 1:
                raise ValueError(
                    "Expected one masked Y column; found "
                    f"{indices.tolist()} in shape {sequence.shape}."
                )
            return int(indices[0])

        def context_target_scale(
            sequence: Any,
            n_ctx: int,
            y_column: int,
        ) -> tuple[float, float]:
            context_y = np.asarray(sequence[:n_ctx, y_column], dtype=np.float64)
            finite = context_y[np.isfinite(context_y)]
            if finite.size == 0:
                raise FloatingPointError(
                    "Cannot normalize evaluation Y without finite context values."
                )
            mean = float(finite.mean())
            std = 1.0 if finite.size <= 1 else float(finite.std(ddof=1)) + 1.0e-6
            if not np.isfinite(std) or std < 1.0e-6:
                std = 1.0e-6
            return mean, std

        def normalize_sequence_target(
            sequence: Any,
            n_ctx: int,
            y_column: int,
        ) -> tuple[Any, float, float]:
            mean, std = context_target_scale(sequence, n_ctx, y_column)
            normalized = np.asarray(sequence, dtype=np.float64).copy()
            finite = np.isfinite(normalized[:, y_column])
            normalized[finite, y_column] = np.clip(
                (normalized[finite, y_column] - mean) / std,
                -100.0,
                100.0,
            )
            return normalized, mean, std

        def picp_hits(
            raw: Any,
            y_true: Any,
            target_mean: float = 0.0,
            target_std: float = 1.0,
        ) -> tuple[int, int] | None:
            if bar_borders is None:
                return None
            logits = raw[0].detach().cpu().numpy().astype(np.float64)
            if logits.shape[-1] < 2 or logits.shape[-1] + 1 != bar_borders.shape[0]:
                return None
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(probabilities, axis=1)
            tail = 0.5 * (1.0 - PICP_MASS)
            low_index = (cumulative >= tail).argmax(axis=1)
            high_index = (cumulative >= 1.0 - tail).argmax(axis=1)
            last = bar_borders.shape[0] - 1
            low = bar_borders[low_index] * float(target_std) + float(target_mean)
            high = bar_borders[np.minimum(high_index + 1, last)] * float(
                target_std
            ) + float(target_mean)
            covered = (y_true >= low) & (y_true <= high)
            return int(covered.sum()), int(covered.shape[0])

        sse_cid = 0.0
        n_cid = 0
        nmse_cid_sum = 0.0
        nmse_cid_tasks = 0
        sse_cate_pfn = 0.0
        sse_cate_naive = 0.0
        n_cate = 0
        naive_rows = 0
        nmse_cate_sum = 0.0
        nmse_cate_tasks = 0
        sum_cate_pfn = 0.0
        sum_cate_true = 0.0
        picp_hit = 0
        picp_n = 0
        tasks_scored = 0

        for task_index in range(NUM_TASKS):
            task = prior.sample(
                seed=BASE_SEED + task_index,
                num_samples=SCORER_POINTS,
                oracle_mc=ORACLE_MC,
            )
            if task.get("is_valid", True) is False:
                _log("warning", "invalid_eval_task_skipped", task=task_index)
                continue

            sequence = task.get("X")
            y_query = task.get("y")
            n_ctx = task.get("n_ctx")
            cate_true = task.get("cate_true")
            if (
                sequence is None
                or y_query is None
                or n_ctx is None
                or cate_true is None
            ):
                return ScorerResult(
                    metrics={},
                    meta={"prior_keys": sorted(task.keys())},
                    skipped=True,
                    skip_reason="Prior must emit X, y, n_ctx and cate_true.",
                )

            sequence = np.asarray(sequence, dtype=np.float64)
            n_ctx = int(n_ctx)
            if sequence.ndim != 2 or not 0 < n_ctx < sequence.shape[0]:
                return ScorerResult(
                    metrics={},
                    meta={"sequence_shape": tuple(sequence.shape), "n_ctx": n_ctx},
                    skipped=True,
                    skip_reason="Invalid Do-PFN evaluation sequence.",
                )
            y_column = locate_y_column(sequence, n_ctx)
            num_covariate_columns = y_column - 1
            if num_covariate_columns < 1:
                return ScorerResult(
                    metrics={},
                    meta={"y_column": y_column},
                    skipped=True,
                    skip_reason="Do-PFN eval requires treatment plus covariates before Y.",
                )

            # The corrected prior emits model-facing normalized Y plus raw
            # values for paper metrics.  Older priors remain supported with
            # the identity scale.
            task_meta = task.get("task_meta") or {}
            target_mean = float(task_meta.get("target_mean", 0.0))
            target_std = float(task_meta.get("target_std", 1.0))
            y_query_model = np.asarray(y_query, dtype=np.float64).reshape(-1)
            y_query = np.asarray(
                task.get("y_raw", y_query_model), dtype=np.float64
            ).reshape(-1)
            cate_model_all = np.asarray(cate_true, dtype=np.float64).reshape(-1)
            cate_all = np.asarray(
                task.get("cate_true_raw", cate_model_all), dtype=np.float64
            ).reshape(-1)
            cate_query = (
                cate_all[n_ctx:] if cate_all.shape[0] == sequence.shape[0] else cate_all
            )
            n_query = sequence.shape[0] - n_ctx
            if y_query.shape[0] != n_query or cate_query.shape[0] != n_query:
                raise ValueError(
                    "Eval target length mismatch: "
                    f"queries={n_query}, y={y_query.shape[0]}, cate={cate_query.shape[0]}."
                )

            raw_original = run_head_raw(sequence, n_ctx)
            prediction_original = to_scalar(raw_original) * target_std + target_mean
            cid_squared_error = float(np.sum((prediction_original - y_query) ** 2))
            sse_cid += cid_squared_error
            n_cid += n_query

            coverage = picp_hits(
                raw_original,
                y_query,
                target_mean,
                target_std,
            )
            if coverage is not None:
                picp_hit += coverage[0]
                picp_n += coverage[1]

            outcome_range = float(np.max(y_query) - np.min(y_query))
            if outcome_range > _EPS:
                nmse_cid_sum += (cid_squared_error / n_query) / (outcome_range**2)
                nmse_cid_tasks += 1

            sequence_do0 = sequence.copy()
            sequence_do1 = sequence.copy()
            sequence_do0[n_ctx:, 0] = 0.0
            sequence_do1[n_ctx:, 0] = 1.0
            cate_prediction = (
                run_model(sequence_do1, n_ctx) - run_model(sequence_do0, n_ctx)
            ) * target_std
            cate_squared_error = float(np.sum((cate_prediction - cate_query) ** 2))
            sse_cate_pfn += cate_squared_error
            n_cate += n_query
            sum_cate_pfn += float(np.sum(cate_prediction))
            sum_cate_true += float(np.sum(cate_query))

            cate_range = float(np.max(cate_query) - np.min(cate_query))
            if cate_range > _EPS:
                nmse_cate_sum += (cate_squared_error / n_query) / (cate_range**2)
                nmse_cate_tasks += 1

            x_context = sequence[:n_ctx, 1:y_column]
            treatment_context = sequence[:n_ctx, 0]
            y_context = sequence[:n_ctx, y_column] * target_std + target_mean
            x_query = sequence[n_ctx:, 1:y_column]
            treated = treatment_context > 0.5
            untreated = ~treated
            required_rows = num_covariate_columns + 1
            if (
                int(treated.sum()) >= required_rows
                and int(untreated.sum()) >= required_rows
            ):
                weights1 = _ridge_fit(
                    x_context[treated], y_context[treated], RIDGE_LAMBDA, np
                )
                weights0 = _ridge_fit(
                    x_context[untreated], y_context[untreated], RIDGE_LAMBDA, np
                )
                cate_naive = _ridge_pred(weights1, x_query, np) - _ridge_pred(
                    weights0, x_query, np
                )
                sse_cate_naive += float(np.sum((cate_naive - cate_query) ** 2))
                naive_rows += n_query

            tasks_scored += 1
            if tasks_scored == 1 or tasks_scored % 10 == 0:
                _log(
                    "info",
                    "evaluation_progress",
                    tasks_scored=tasks_scored,
                    tasks_requested=NUM_TASKS,
                    y_column=y_column,
                    covariates=num_covariate_columns,
                    cid_mse_so_far=sse_cid / max(n_cid, 1),
                    cate_mse_so_far=sse_cate_pfn / max(n_cate, 1),
                )

        if n_cate == 0:
            return ScorerResult(
                metrics={},
                meta={"tasks_scored": tasks_scored},
                skipped=True,
                skip_reason="No query positions were scored.",
            )

        cid_mse = sse_cid / max(n_cid, 1)
        cate_mse = sse_cate_pfn / n_cate
        naive_cate_mse = sse_cate_naive / naive_rows if naive_rows else float("nan")
        cid_nmse = nmse_cid_sum / nmse_cid_tasks if nmse_cid_tasks else float("nan")
        cate_nmse = nmse_cate_sum / nmse_cate_tasks if nmse_cate_tasks else float("nan")
        ate_error = abs(sum_cate_pfn / n_cate - sum_cate_true / n_cate)
        picp_90 = picp_hit / picp_n if picp_n else float("nan")
        picp_90_gap = abs(picp_90 - PICP_MASS) if picp_n else float("nan")

        metrics = {
            "cid_mse": cid_mse,
            "cid_nmse": cid_nmse,
            "cate_mse": cate_mse,
            "cate_nmse": cate_nmse,
            "naive_cate_mse": naive_cate_mse,
            "oracle_cate_mse": 0.0,
            "ratio_vs_naive_cate": (
                cate_mse / naive_cate_mse
                if naive_cate_mse > _EPS
                else (1.0 if cate_mse <= _EPS else float("inf"))
            ),
            "ratio_vs_oracle_cate": cate_mse,
            "ate_error": ate_error,
            "picp_90": picp_90,
            "picp_90_gap": picp_90_gap,
        }
        meta = {
            "num_tasks_requested": NUM_TASKS,
            "num_tasks_scored": tasks_scored,
            "points_per_task": SCORER_POINTS,
            "oracle_mc": ORACLE_MC,
            "device": str(model_device),
            "device_source": device_source,
            "border_source": border_source,
            "naive_rows_scored": naive_rows,
            "picp_rows_scored": picp_n,
            "nmse_cid_tasks": nmse_cid_tasks,
            "nmse_cate_tasks": nmse_cate_tasks,
            "cate_method": "paired do(0)/do(1) query treatment intervention",
            "ratio_vs_naive_cate_note": "PFN CATE MSE / naive CATE MSE; below 1 is better",
            "cate_true": (
                "Monte-Carlo oracle E[Y|do(1),X]-E[Y|do(0),X] " f"(n_mc={ORACLE_MC})"
            ),
            "target_transform": (
                "context Y standardization with sample std + 1e-6 and "
                "[-100,100] clipping; predictions converted back to raw units"
            ),
            "picp_note": "central 90% predictive interval of BAR posterior",
        }

        # Paper Section 4.1: evaluate the six fixed causal structures in
        # addition to the random-DAG recovery check above. Each task samples
        # fresh mechanisms/noise but keeps the case graph fixed.
        structured_cid_sse = 0.0
        structured_cate_sse = 0.0
        structured_rows = 0
        structured_cid_nmse_sum = 0.0
        structured_cid_nmse_tasks = 0
        structured_cate_nmse_sum = 0.0
        structured_cate_nmse_tasks = 0
        structured_cate_pred_sum = 0.0
        structured_cate_true_sum = 0.0
        structured_picp_hits = 0
        structured_picp_rows = 0
        structured_retries = 0
        case_summaries: dict[str, Any] = {}

        for case_index, case_spec in enumerate(CASE_STUDIES):
            case_cid_sse = 0.0
            case_cate_sse = 0.0
            case_rows = 0
            case_cid_nmse_sum = 0.0
            case_cid_nmse_tasks = 0
            case_cate_nmse_sum = 0.0
            case_cate_nmse_tasks = 0
            case_cate_pred_sum = 0.0
            case_cate_true_sum = 0.0
            case_picp_hits = 0
            case_picp_rows = 0
            case_retries = 0

            for case_task_index in range(CASE_TASKS_PER_STUDY):
                case_seed = CASE_BASE_SEED + case_index * 100_000 + case_task_index
                task = _generate_structured_case_task(
                    case_spec,
                    case_seed,
                    np,
                )
                case_retries += int(task["retry"])
                sequence = np.asarray(task["X"], dtype=np.float64)
                y_query = np.asarray(task["y"], dtype=np.float64).reshape(-1)
                cate_query = np.asarray(task["cate_true"], dtype=np.float64).reshape(-1)
                n_ctx_case = int(task["n_ctx"])
                n_query_case = int(y_query.shape[0])

                y_column_case = locate_y_column(sequence, n_ctx_case)
                sequence_model, target_mean_case, target_std_case = (
                    normalize_sequence_target(
                        sequence,
                        n_ctx_case,
                        y_column_case,
                    )
                )

                raw_case = run_head_raw(sequence_model, n_ctx_case)
                prediction_case = (
                    to_scalar(raw_case) * target_std_case + target_mean_case
                )
                cid_error = float(np.sum((prediction_case - y_query) ** 2))

                sequence_do0 = sequence_model.copy()
                sequence_do1 = sequence_model.copy()
                sequence_do0[n_ctx_case:, 0] = 0.0
                sequence_do1[n_ctx_case:, 0] = 1.0
                cate_prediction = (
                    run_model(sequence_do1, n_ctx_case)
                    - run_model(sequence_do0, n_ctx_case)
                ) * target_std_case
                cate_error = float(np.sum((cate_prediction - cate_query) ** 2))

                case_cid_sse += cid_error
                case_cate_sse += cate_error
                case_rows += n_query_case
                case_cate_pred_sum += float(np.sum(cate_prediction))
                case_cate_true_sum += float(np.sum(cate_query))

                outcome_range = float(np.max(y_query) - np.min(y_query))
                if outcome_range > _EPS:
                    case_cid_nmse_sum += (cid_error / n_query_case) / (outcome_range**2)
                    case_cid_nmse_tasks += 1

                cate_range = float(np.max(cate_query) - np.min(cate_query))
                if cate_range > _EPS:
                    case_cate_nmse_sum += (cate_error / n_query_case) / (cate_range**2)
                    case_cate_nmse_tasks += 1

                coverage = picp_hits(
                    raw_case,
                    y_query,
                    target_mean_case,
                    target_std_case,
                )
                if coverage is not None:
                    case_picp_hits += coverage[0]
                    case_picp_rows += coverage[1]

            case_slug = str(case_spec["slug"])
            case_cid_mse = case_cid_sse / max(case_rows, 1)
            case_cate_mse = case_cate_sse / max(case_rows, 1)
            case_cid_nmse = (
                case_cid_nmse_sum / case_cid_nmse_tasks
                if case_cid_nmse_tasks
                else float("nan")
            )
            case_cate_nmse = (
                case_cate_nmse_sum / case_cate_nmse_tasks
                if case_cate_nmse_tasks
                else float("nan")
            )
            case_ate_error = abs(
                case_cate_pred_sum / max(case_rows, 1)
                - case_cate_true_sum / max(case_rows, 1)
            )
            case_picp_90 = (
                case_picp_hits / case_picp_rows if case_picp_rows else float("nan")
            )

            prefix = f"case_{case_slug}"
            metrics.update(
                {
                    f"{prefix}_cid_mse": case_cid_mse,
                    f"{prefix}_cid_nmse": case_cid_nmse,
                    f"{prefix}_cate_mse": case_cate_mse,
                    f"{prefix}_cate_nmse": case_cate_nmse,
                    f"{prefix}_ate_error": case_ate_error,
                    f"{prefix}_picp_90": case_picp_90,
                }
            )
            case_summaries[case_slug] = {
                "label": case_spec["label"],
                "tasks": CASE_TASKS_PER_STUDY,
                "rows": case_rows,
                "observed_covariates": list(case_spec["observed"]),
                "generator_retries": case_retries,
                "cid_nmse_tasks": case_cid_nmse_tasks,
                "cate_nmse_tasks": case_cate_nmse_tasks,
            }

            structured_cid_sse += case_cid_sse
            structured_cate_sse += case_cate_sse
            structured_rows += case_rows
            structured_cid_nmse_sum += case_cid_nmse_sum
            structured_cid_nmse_tasks += case_cid_nmse_tasks
            structured_cate_nmse_sum += case_cate_nmse_sum
            structured_cate_nmse_tasks += case_cate_nmse_tasks
            structured_cate_pred_sum += case_cate_pred_sum
            structured_cate_true_sum += case_cate_true_sum
            structured_picp_hits += case_picp_hits
            structured_picp_rows += case_picp_rows
            structured_retries += case_retries

            _log(
                "info",
                "structured_case_complete",
                case=case_slug,
                label=case_spec["label"],
                tasks=CASE_TASKS_PER_STUDY,
                cid_mse=case_cid_mse,
                cid_nmse=case_cid_nmse,
                cate_mse=case_cate_mse,
                cate_nmse=case_cate_nmse,
                ate_error=case_ate_error,
                picp_90=case_picp_90,
                generator_retries=case_retries,
            )

        metrics.update(
            {
                "case6_cid_mse": structured_cid_sse / max(structured_rows, 1),
                "case6_cid_nmse": structured_cid_nmse_sum
                / max(structured_cid_nmse_tasks, 1),
                "case6_cate_mse": structured_cate_sse / max(structured_rows, 1),
                "case6_cate_nmse": structured_cate_nmse_sum
                / max(structured_cate_nmse_tasks, 1),
                "case6_ate_error": abs(
                    structured_cate_pred_sum / max(structured_rows, 1)
                    - structured_cate_true_sum / max(structured_rows, 1)
                ),
                "case6_picp_90": structured_picp_hits / max(structured_picp_rows, 1),
                "case6_picp_90_gap": abs(
                    structured_picp_hits / max(structured_picp_rows, 1) - PICP_MASS
                ),
            }
        )
        meta.update(
            {
                "structured_case_studies": case_summaries,
                "structured_case_tasks_per_study": CASE_TASKS_PER_STUDY,
                "structured_case_context_rows": CASE_CONTEXT_ROWS,
                "structured_case_query_rows": CASE_QUERY_ROWS,
                "structured_case_total_tasks": (
                    len(CASE_STUDIES) * CASE_TASKS_PER_STUDY
                ),
                "structured_case_total_rows": structured_rows,
                "structured_case_generator_retries": structured_retries,
                "structured_case_nonlinearities": [
                    "square",
                    "tanh",
                    "relu",
                ],
                "structured_case_cate_true": (
                    "paired potential-outcome Y(do(1))-Y(do(0)) with shared "
                    "exogenous noise; descendants of treatment are recomputed"
                ),
            }
        )
        _log("info", "evaluation_complete", metrics=metrics, meta=meta)
        return ScorerResult(metrics=metrics, meta=meta)
