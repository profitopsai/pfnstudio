"""Do-PFN CID + CATE recovery scorer — ships in the template, not core.

This is the executable half of `evals/cid_recovery.yaml`. It is registered by
slug via `@register_scorer("cid_recovery")` and discovered at run time by
`pfnstudio_core.registry.discover_in_project` (which imports `evals/*.py`), so
the Do-PFN template owns its own paper-specific scoring — exactly like its
`prior.py`. Nothing paper-specific lives in pfnstudio-core.

Bound to the random-DAG `do_pfn_scm` prior, whose packed token layout is
`[t, x_1..x_d, y]` (width d+2): column 0 is the binary treatment, columns 1..d
the covariates, column d+1 the outcome (real on context rows, NaN on query
rows — the in-context learning marker). There is no separate is_context column,
so the do(0)/do(1) contrast is formed by editing the query treatment column
directly — the same way you'd query the trained model for a CATE at inference.

What it computes (all numbers are measured, none are placeholders):

  For NUM_TASKS fresh SCMs drawn from the run's own prior, each scored with a
  proper Monte-Carlo oracle CATE (the prior is asked with oracle_mc>0, so
  `cate_true` is the real E[Y|do(1),X]−E[Y|do(0),X], not a single-draw
  contrast):
    • CID   — the model's predicted interventional outcome (query rows carry
              the SCM's own do(T=t) intervention) vs the prior's oracle Y_int.
    • CATE  — query the model twice on the SAME context, forcing every query
              treatment to 0 then to 1, and take (pred_do1 − pred_do0). Scored
              against the oracle `cate_true`.
    • naive — a per-task T-learner ridge (fit E[Y|X] on the observed T=1 and
              T=0 context arms separately, difference on the query X). Biased
              under the unobserved confounder — the baseline Do-PFN beats.
    • PICP  — the paper's uncertainty metric. The bar head emits a full
              posterior over the interventional outcome; we take its central
              90% predictive interval and measure how often the true Y_int
              lands inside. Well-calibrated ⇒ ≈0.90.

The model's final block is a distributional (bar) head, so its raw output is
per-bucket logits; the block's generic `to_prediction` reduces them to the
distribution mean for the point-estimate metrics, while PICP reads the raw
per-bucket logits + the head's bucket borders directly. A plain scalar_head
(no `to_prediction`, no borders) passes through and simply skips PICP.

Metrics returned match `cid_recovery.yaml`:
  cid_mse, cid_nmse (range-normalized, the paper's primary metric),
  cate_mse, cate_nmse, naive_cate_mse, oracle_cate_mse (≡0 floor),
  ratio_vs_naive_cate (naive/pfn, >1 = model beats naive),
  ratio_vs_oracle_cate (absolute CATE MSE vs the oracle; 0 = perfect recovery),
  ate_error (|mean model CATE − mean oracle CATE|),
  picp_90 (central-90% interval coverage; target 0.90),
  picp_90_gap (|picp_90 − 0.90|; lower = better calibrated).
"""

from __future__ import annotations

# Absolute imports — this module is loaded by discover_in_project via
# exec_module (no package context), so relative imports would fail.
from pfnstudio_core.registry import register_scorer
from pfnstudio_core.scorers.base import DatasetScorer, ScorerResult

# 50 fresh DGPs per the eval spec. Points-per-task kept small so scoring is
# quick even for the 12-layer axial model; seeds are large + fixed so scoring
# is reproducible and disjoint from the training seed schedule.
NUM_TASKS = 50
SCORER_POINTS = 256
BASE_SEED = 900_000
RIDGE_LAMBDA = 1.0
# Monte-Carlo draws for the prior's oracle CATE. 64 matches the DGP's internal
# noise integration; enough to make cate_true a stable ground truth.
ORACLE_MC = 64
# Central predictive-interval mass for PICP (paper reports 90% coverage).
PICP_MASS = 0.90
_EPS = 1e-9


def _ridge_fit(x, y, lam, np):
    """Closed-form ridge with a bias term. x: (n, d) → weights (d+1,)."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    a = xb.T @ xb + lam * np.eye(xb.shape[1])
    return np.linalg.solve(a, xb.T @ y)


def _ridge_pred(w, x, np):
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return xb @ w


@register_scorer("cid_recovery")
class DoPfnCidRecoveryScorer(DatasetScorer):
    """Measure interventional-outcome (CID) and CATE recovery vs baselines."""

    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult:
        try:
            import numpy as np
            import torch
        except ImportError as e:
            return ScorerResult(
                metrics={}, meta={"dependency_missing": str(e)},
                skipped=True, skip_reason=f"missing dependency: {e}",
            )

        from pfnstudio_core.registry import get_prior
        from pfnstudio_core.training.loop import _split_encoder_heads

        try:
            prior_cls = get_prior(run_spec.prior.id)
        except KeyError:
            return ScorerResult(
                metrics={}, meta={"prior_id": run_spec.prior.id},
                skipped=True,
                skip_reason=f"prior '{run_spec.prior.id}' not registered in this project",
            )
        prior = prior_cls()

        modules = list(model.modules)
        encoder, heads = _split_encoder_heads(modules)
        # The device the model lives on — the trainer may have moved it to CUDA,
        # so forward inputs must land there too (outputs are pulled back to CPU
        # via .cpu() before numpy).
        model_device = None
        _head_proj = getattr(heads[0], "proj", None) if heads else None
        if _head_proj is not None:
            _p = next(_head_proj.parameters(), None)
            if _p is not None:
                model_device = _p.device
        # The distributional head's logits→mean reducer, if any.
        reduce_fn = None
        for _, mod in modules:
            fn = getattr(mod, "to_prediction", None)
            if callable(fn):
                reduce_fn = fn
                break
        # Bucket borders of the bar head (for PICP). None ⇒ head is a plain
        # scalar_head and calibration is skipped.
        bar_borders = None
        for _, mod in modules:
            b = getattr(getattr(mod, "proj", None), "bar_borders", None)
            if b is not None:
                bar_borders = b.detach().cpu().numpy().astype(np.float64)
                break

        def run_head_raw(seq: np.ndarray, n_ctx: int):
            """Forward exactly as the trainer does — n_ctx-aware blocks get
            single_eval_pos, tuple outputs unwrapped — and return the RAW head
            output at the query positions: (n_qry, C) per-bucket logits for a
            bar head, or (n_qry, 1) for a scalar head."""
            with torch.no_grad():
                enc = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
                if model_device is not None:
                    enc = enc.to(model_device)
                for mod in encoder:
                    if n_ctx > 0 and getattr(mod, "needs_single_eval_pos", False):
                        r = mod(enc, single_eval_pos=n_ctx)
                    else:
                        r = mod(enc)
                    enc = r[0] if isinstance(r, tuple) else r
                out = heads[0](enc) if heads else enc
                return out[:, n_ctx:, :]  # keep as (1, n_qry, C) torch tensor

        def to_scalar(raw) -> np.ndarray:
            """Reduce a raw head output to one number per query position."""
            out = reduce_fn(raw) if reduce_fn is not None else raw
            return out[0, :, 0].cpu().numpy().astype(np.float64)

        def run_model(seq: np.ndarray, n_ctx: int) -> np.ndarray:
            return to_scalar(run_head_raw(seq, n_ctx))

        def picp_hits(raw, y_true: np.ndarray):
            """How many query targets land inside the central PICP_MASS interval
            of the predicted bar distribution. Returns (hits, count) or None if
            the head is not distributional."""
            if bar_borders is None:
                return None
            logits = raw[0].cpu().numpy().astype(np.float64)  # (n_qry, C)
            if logits.shape[-1] < 2 or logits.shape[-1] + 1 != bar_borders.shape[0]:
                return None
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            cdf = np.cumsum(p, axis=1)
            tail = 0.5 * (1.0 - PICP_MASS)
            lo_idx = (cdf >= tail).argmax(axis=1)
            hi_idx = (cdf >= 1.0 - tail).argmax(axis=1)
            last = bar_borders.shape[0] - 1
            lo = bar_borders[lo_idx]
            hi = bar_borders[np.minimum(hi_idx + 1, last)]
            covered = (y_true >= lo) & (y_true <= hi)
            return int(covered.sum()), int(covered.shape[0])

        sse_cid = 0.0          # predicted Y_int vs oracle Y_int
        n_cid = 0
        nmse_cid_sum = 0.0     # per-task range-normalized CID MSE
        nmse_cid_tasks = 0     # tasks with a non-degenerate CID range
        nmse_cate_sum = 0.0    # per-task range-normalized CATE MSE
        nmse_cate_tasks = 0    # tasks with a non-degenerate CATE range
        sse_cate_pfn = 0.0     # (pred1-pred0) vs cate_true
        sse_cate_naive = 0.0   # T-learner ridge vs cate_true
        n_cate = 0
        naive_rows = 0
        sum_cate_pfn = 0.0
        sum_cate_true = 0.0
        picp_hit = 0
        picp_n = 0

        for k in range(NUM_TASKS):
            task = prior.sample(
                seed=BASE_SEED + k, num_samples=SCORER_POINTS, oracle_mc=ORACLE_MC
            )
            seq = task.get("X")
            y_q = task.get("y")
            n_ctx = task.get("n_ctx")
            cate_true = task.get("cate_true")
            if seq is None or y_q is None or n_ctx is None or cate_true is None:
                return ScorerResult(
                    metrics={}, meta={"prior_keys": sorted(task.keys())},
                    skipped=True,
                    skip_reason=(
                        "Prior didn't emit the Do-PFN shape (need X, y, n_ctx, cate_true)."
                    ),
                )

            seq = np.asarray(seq, dtype=np.float64)
            n_ctx = int(n_ctx)
            width = seq.shape[1]
            d = width - 2  # [t, x_1..x_d, y]
            if seq.ndim != 2 or d < 1:
                return ScorerResult(
                    metrics={}, meta={"token_width": int(width)},
                    skipped=True, skip_reason="Token width < 3 — not the Do-PFN packing.",
                )

            y_q = np.asarray(y_q, dtype=np.float64)
            cate_q = np.asarray(cate_true, dtype=np.float64)[n_ctx:]

            # CID: the model on the query as generated (its own do(T=t_int)).
            raw_orig = run_head_raw(seq, n_ctx)
            pred_orig = to_scalar(raw_orig)
            se_cid = np.sum((pred_orig - y_q) ** 2)
            sse_cid += float(se_cid)
            n_cid += int(y_q.shape[0])

            # PICP: does the true interventional Y fall in the head's central
            # 90% predictive interval? (paper's uncertainty-calibration metric)
            hits = picp_hits(raw_orig, y_q)
            if hits is not None:
                picp_hit += hits[0]
                picp_n += hits[1]

            # Range-normalized CID MSE (paper's primary metric): divide the
            # squared error by the target's range² so tasks are comparable.
            rng_y = float(np.max(y_q) - np.min(y_q))
            if rng_y > _EPS:
                nmse_cid_sum += float(se_cid / y_q.shape[0]) / (rng_y ** 2)
                nmse_cid_tasks += 1

            # CATE: same context, force every query treatment to 0 then 1.
            seq0 = seq.copy(); seq0[n_ctx:, 0] = 0.0
            seq1 = seq.copy(); seq1[n_ctx:, 0] = 1.0
            cate_pfn = run_model(seq1, n_ctx) - run_model(seq0, n_ctx)
            se_cate = np.sum((cate_pfn - cate_q) ** 2)
            sse_cate_pfn += float(se_cate)
            n_cate += int(cate_q.shape[0])
            sum_cate_pfn += float(np.sum(cate_pfn))
            sum_cate_true += float(np.sum(cate_q))

            rng_c = float(np.max(cate_q) - np.min(cate_q))
            if rng_c > _EPS:
                nmse_cate_sum += float(se_cate / cate_q.shape[0]) / (rng_c ** 2)
                nmse_cate_tasks += 1

            # Naive T-learner ridge on the OBSERVED (confounded) context.
            x_ctx = seq[:n_ctx, 1 : width - 1]
            t_ctx = seq[:n_ctx, 0]
            y_ctx = seq[:n_ctx, width - 1]
            x_q = seq[n_ctx:, 1 : width - 1]
            m1 = t_ctx > 0.5
            m0 = ~m1
            if int(m1.sum()) >= d + 1 and int(m0.sum()) >= d + 1:
                w1 = _ridge_fit(x_ctx[m1], y_ctx[m1], RIDGE_LAMBDA, np)
                w0 = _ridge_fit(x_ctx[m0], y_ctx[m0], RIDGE_LAMBDA, np)
                cate_naive = _ridge_pred(w1, x_q, np) - _ridge_pred(w0, x_q, np)
                sse_cate_naive += float(np.sum((cate_naive - cate_q) ** 2))
                naive_rows += int(cate_q.shape[0])

        if n_cate == 0:
            return ScorerResult(
                metrics={}, meta={}, skipped=True,
                skip_reason="No query positions scored.",
            )

        cid_mse = sse_cid / max(n_cid, 1)
        cate_mse = sse_cate_pfn / n_cate
        naive_cate_mse = (sse_cate_naive / naive_rows) if naive_rows else float("nan")
        oracle_cate_mse = 0.0  # oracle predicts its own cate_true → floor is 0
        ate_error = abs(sum_cate_pfn / n_cate - sum_cate_true / n_cate)
        cid_nmse = (nmse_cid_sum / nmse_cid_tasks) if nmse_cid_tasks else float("nan")
        cate_nmse = (nmse_cate_sum / nmse_cate_tasks) if nmse_cate_tasks else float("nan")
        picp_90 = (picp_hit / picp_n) if picp_n else float("nan")
        picp_90_gap = abs(picp_90 - PICP_MASS) if picp_n else float("nan")

        metrics = {
            "cid_mse": cid_mse,
            "cid_nmse": cid_nmse,
            "cate_mse": cate_mse,
            "cate_nmse": cate_nmse,
            "naive_cate_mse": naive_cate_mse,
            "oracle_cate_mse": oracle_cate_mse,
            # >1 means the model's CATE beats the confounded observational ridge.
            "ratio_vs_naive_cate": (naive_cate_mse / cate_mse) if cate_mse > _EPS else float("inf"),
            # Absolute CATE MSE against the MC oracle (0 = perfect recovery).
            "ratio_vs_oracle_cate": cate_mse,
            "ate_error": ate_error,
            "picp_90": picp_90,
            "picp_90_gap": picp_90_gap,
        }
        meta = {
            "num_tasks": NUM_TASKS,
            "points_per_task": SCORER_POINTS,
            "oracle_mc": ORACLE_MC,
            "naive_rows_scored": naive_rows,
            "picp_rows_scored": picp_n,
            # Transparency: normalized MSE skips tasks whose target range is
            # ~0 (undefined normalization). On random DAGs a sizeable fraction
            # of tasks have a near-zero oracle CATE (the treatment barely moves
            # the outcome, or mediators are pinned), so cate_nmse is reported
            # over a SUBSET — surfaced here so it isn't a silent drop. The
            # structured case studies (v0.2) give stronger, non-degenerate
            # signal.
            "nmse_cid_tasks": nmse_cid_tasks,
            "nmse_cate_tasks": nmse_cate_tasks,
            "cate_method": "paired do(0)/do(1) by forcing the query treatment column",
            "cate_true": f"Monte-Carlo oracle E[Y|do(1),X]-E[Y|do(0),X] (n_mc={ORACLE_MC})",
            "picp_note": "central 90% predictive interval of the bar-head posterior",
        }
        return ScorerResult(metrics=metrics, meta=meta)
