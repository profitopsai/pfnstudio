"""Cascade structure-discovery scorer — model-based adjacency recovery.

Ships in the study, not core. Registered by slug via
`@register_scorer("cascade_recovery")` and discovered at run time by
`pfnstudio_core.registry.discover_in_project` (which imports `evals/*.py`).

Unlike a correlation baseline, this scores the TRAINED model directly: it
forwards the discovery stack (tabular_embedder -> transformer_encoder x3 ->
causal_attention_pool -> discovery_head) on fresh SCM tasks from the run's
own prior, reads the `discovery_head`'s (V, V) adjacency logits, and scores
them against the prior's ground-truth `A`.

The discovery head is aligned to a fixed number of variables V at build time
(`num_variables` in the model YAML). This scorer reads that V off the head and
samples the prior with `d = V` so every task's ground-truth `A` is (V, V) and
matches the model's output — the same alignment the trainer relies on.

Metrics match `cascade_recovery.yaml`: auroc, shd, f1, precision, recall, and
`chance` (the 0.5 AUROC floor).
"""

from __future__ import annotations

from pfnstudio_core.registry import register_scorer
from pfnstudio_core.scorers.base import DatasetScorer, ScorerResult

NUM_TASKS = 30
BASE_SEED = 50_000


@register_scorer("cascade_recovery")
class CascadeDiscoveryScorer(DatasetScorer):
    """Adjacency AUROC / SHD / F1 of the trained discovery head vs ground truth."""

    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult:
        try:
            import numpy as np
            import torch
        except ImportError as e:
            return ScorerResult(
                metrics={},
                meta={"dependency_missing": str(e)},
                skipped=True,
                skip_reason=f"missing dependency: {e}",
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
                skip_reason=f"prior '{run_spec.prior.id}' not registered in this project",
            )
        prior = prior_cls()

        encoder, heads = _split_encoder_heads(list(model.modules))
        # The discovery head — the one exposing num_variables (a (V, V) head).
        head = next((h for h in heads if hasattr(h, "num_variables")), None)
        if head is None:
            return ScorerResult(
                metrics={},
                meta={"heads": [type(h).__name__ for h in heads]},
                skipped=True,
                skip_reason=(
                    "Model has no discovery head (a block with num_variables). "
                    "This scorer needs a (V, V) adjacency-producing head."
                ),
            )
        V = int(head.num_variables)

        def run_model(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                enc = torch.from_numpy(X.astype(np.float32)).unsqueeze(0)  # (1, N, V)
                for mod in encoder:
                    r = mod(enc)
                    enc = r[0] if isinstance(r, tuple) else r
                logits = head(enc)  # (..., V, V)
                probs = torch.sigmoid(logits).reshape(-1, V, V)[0]
                return probs.cpu().numpy().astype(np.float64)

        flat_true: list[float] = []
        flat_pred: list[float] = []
        tasks_used = 0
        skipped_no_A = 0

        for k in range(NUM_TASKS):
            # Force the prior's variable count to the head's V so A is (V, V).
            task = prior.sample(seed=BASE_SEED + k, d=V)
            A_true = task.get("A") if isinstance(task, dict) else None
            X = task.get("X") if isinstance(task, dict) else None
            if A_true is None or X is None:
                skipped_no_A += 1
                continue

            A_true = np.asarray(A_true, dtype=np.float32)
            if A_true.ndim == 3:
                A_true = A_true.max(axis=-1)
            X = np.asarray(X, dtype=np.float32)
            if A_true.shape != (V, V) or X.ndim != 2 or X.shape[1] != V:
                skipped_no_A += 1
                continue

            probs = run_model(X)
            if probs.shape != (V, V):
                skipped_no_A += 1
                continue

            mask = ~np.eye(V, dtype=bool)  # off-diagonal edges only
            flat_true.extend((A_true[mask] > 0.5).astype(np.float32).tolist())
            flat_pred.extend(probs[mask].astype(np.float32).tolist())
            tasks_used += 1

        if tasks_used == 0:
            return ScorerResult(
                metrics={},
                meta={"tasks_attempted": NUM_TASKS, "skipped_no_A": skipped_no_A, "V": V},
                skipped=True,
                skip_reason=(
                    "No task produced a usable (A, X) pair matching the head's "
                    f"V={V}. Prior must emit {{'A': (V, V), 'X': (N, V)}}."
                ),
            )

        metrics = self._compute_metrics(
            np.asarray(flat_true, dtype=np.float32),
            np.asarray(flat_pred, dtype=np.float32),
        )
        return ScorerResult(
            metrics=metrics,
            meta={
                "tasks_attempted": NUM_TASKS,
                "tasks_used": tasks_used,
                "skipped_no_A": skipped_no_A,
                "num_variables": V,
                "edge_score_source": "trained discovery_head (V, V) logits",
            },
        )

    def _compute_metrics(self, true_arr, pred_arr) -> dict:
        """AUROC / SHD / F1 / precision / recall — standard discovery set."""
        import numpy as np

        n_pos = int(true_arr.sum())
        n_neg = int(true_arr.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            auroc = 0.5
        else:
            # AUROC via Mann-Whitney U statistic — no sklearn dependency.
            order = np.argsort(-pred_arr, kind="mergesort")
            ranked = true_arr[order]
            rank_pos = np.where(ranked > 0.5)[0] + 1
            sum_ranks = float(rank_pos.sum())
            auroc = (n_pos * (n_pos + 1) / 2.0 + n_pos * n_neg - sum_ranks) / (n_pos * n_neg)

        # Threshold at the predicted-score median so predicted edge density
        # matches ground-truth density on average.
        thr = float(np.median(pred_arr)) if pred_arr.size else 0.0
        bin_pred = (pred_arr >= thr).astype(np.float32)
        bin_true = (true_arr > 0.5).astype(np.float32)
        tp = float(((bin_pred == 1) & (bin_true == 1)).sum())
        fp = float(((bin_pred == 1) & (bin_true == 0)).sum())
        fn = float(((bin_pred == 0) & (bin_true == 1)).sum())
        shd = fp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "auroc": float(auroc),
            "shd": float(shd),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "chance": 0.5,
        }
