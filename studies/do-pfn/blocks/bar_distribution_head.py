"""Bar-distribution (Riemann) output head — PROJECT block.

Lives in the project, exactly like a prior or eval; discovered at run time
(discover_in_project imports blocks/*.py) and referenced from the model by
its registered type `bar_distribution_head`. Nothing is baked into
pfnstudio-core — core only provides the generic hooks this block plugs into.

Instead of regressing a single number, this head predicts a full distribution
over the outcome by discretizing it into `num_buckets` buckets and emitting a
logit per bucket (Müller et al. 2022, "Transformers Can Do Bayesian
Inference"; the same head TabPFN and Do-PFN use). It trains with the proper
bucketized negative-log-likelihood and, at inference, reports the distribution
mean as the point estimate.

It opts into three generic, duck-typed hooks the trainer / predict loop call
when present — core carries no bar-distribution-specific knowledge:

  - `is_head = True`        → the trainer treats this as an output head (a
                              project head can't add its class to core's
                              allowlist, so it declares itself one).
  - `setup(*, prior, ...)`  → before the first forward pass, pool outcome
                              samples from the prior and set equal-mass bucket
                              borders (so buckets carry ~equal probability).
  - `loss(logits, target)`  → own the loss: bucketized NLL, not MSE.
  - `to_prediction(output)` → collapse per-bucket logits to the distribution
                              mean so the predict path gets a scalar.

Authoring conventions (copy for your own head): a plain class decorated with
@register_block; build torch submodules as attributes; import torch INSIDE the
methods so discovery never fails when torch is absent; accept **_ so unknown
config is ignored. The bucket borders are stored as a registered buffer on the
projection so they travel with the checkpoint and are restored at inference.
"""

from __future__ import annotations

from typing import Any

from pfnstudio_core.registry import register_block

# How many prior tasks to pool when fitting bucket borders in setup(), and how
# many points to draw per task (small — this is a one-time border fit, not
# training). ~40 × 256 gives enough samples to place 100 equal-mass borders.
_SETUP_TASKS: int = 40
_SETUP_POINTS: int = 256
_SETUP_BASE_SEED: int = 987_654


@register_block("bar_distribution_head")
class BarDistributionHead:
    """Distributional regression head over `num_buckets` outcome buckets."""

    # Opt into the trainer's head fan-out without editing core's allowlist.
    is_head = True

    def __init__(self, d_model: int = 256, num_buckets: int = 100, **_: Any) -> None:
        import torch
        import torch.nn as nn

        self.num_buckets = int(num_buckets)
        self.proj = nn.Linear(d_model, self.num_buckets)
        # Bucket borders (num_buckets + 1 boundaries). Registered as a buffer
        # ON proj so it (a) moves with .to(device), (b) is saved in proj's
        # state_dict and restored at inference. Placeholder until setup() fits
        # it to the prior; a wide linspace keeps predict sane even if a
        # checkpoint predates setup.
        self.proj.register_buffer(
            "bar_borders", torch.linspace(-3.0, 3.0, self.num_buckets + 1)
        )

    def __call__(self, x: Any) -> Any:
        # (B, N, d_model) -> (B, N, num_buckets) per-bucket logits.
        return self.proj(x)

    # ── generic pre-training setup hook ────────────────────────────────────
    def setup(self, *, prior: Any, hp: dict | None = None, device: Any = None, **_: Any) -> None:
        import numpy as np
        import torch

        pooled: list[np.ndarray] = []
        for i in range(_SETUP_TASKS):
            try:
                task = prior.sample(seed=_SETUP_BASE_SEED + i, num_samples=_SETUP_POINTS)
            except TypeError:
                # Prior.sample without a num_samples kwarg — fall back to defaults.
                task = prior.sample(seed=_SETUP_BASE_SEED + i)
            # Query targets (the loss targets) …
            y = np.asarray(task.get("y", []), dtype=np.float32).ravel()
            pooled.append(y)
            # … plus context outcomes carried in X's y-column (col d+1),
            # finite rows only (query rows are NaN there).
            X = np.asarray(task.get("X", []), dtype=np.float32)
            if X.ndim == 2 and X.shape[1] >= 1:
                ycol = X[:, -1]
                pooled.append(ycol[np.isfinite(ycol)])

        ally = np.concatenate([p for p in pooled if p.size]) if pooled else np.zeros(1, np.float32)
        ally = ally[np.isfinite(ally)]
        if ally.size < self.num_buckets + 1:
            # Degenerate prior draw — keep the linspace placeholder.
            return

        # Equal-mass borders: quantiles at evenly spaced probabilities. Nudge
        # to strictly increasing so bucketize + widths stay well-defined even
        # when the outcome has point masses (e.g. binary-ish treatments).
        qs = np.linspace(0.0, 1.0, self.num_buckets + 1)
        borders = np.quantile(ally, qs).astype(np.float64)
        eps = 1e-6 * (float(ally.std()) + 1.0)
        for j in range(1, borders.size):
            if borders[j] <= borders[j - 1]:
                borders[j] = borders[j - 1] + eps

        borders_t = torch.as_tensor(borders, dtype=self.proj.bar_borders.dtype)
        if device is not None:
            borders_t = borders_t.to(device)
        self.proj.bar_borders.copy_(borders_t)

    # ── generic custom-loss hook (bucketized NLL) ──────────────────────────
    def loss(self, logits: Any, target: Any) -> Any:
        import torch
        import torch.nn.functional as F

        borders = self.proj.bar_borders
        nb = borders.numel() - 1
        target = target.reshape(-1).to(logits.dtype)

        # Bucket index containing each target, clamped to the valid range so
        # targets beyond the outer borders land in the edge buckets.
        idx = torch.bucketize(target, borders, right=False) - 1
        idx = idx.clamp(0, nb - 1)

        logp = F.log_softmax(logits, dim=-1)  # (n_qry, nb)
        widths = (borders[1:] - borders[:-1]).clamp_min(1e-6)  # (nb,)
        chosen = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # (n_qry,)
        # Continuous NLL: −log( p_bucket / bucket_width ) — density, not mass,
        # so the head is penalized for spreading probability over wide buckets.
        nll = -(chosen - torch.log(widths[idx]))
        return nll.mean()

    # ── generic prediction-reduction hook (distribution mean) ──────────────
    def to_prediction(self, output: Any) -> Any:
        import torch.nn.functional as F

        borders = self.proj.bar_borders
        centers = 0.5 * (borders[1:] + borders[:-1])  # (nb,)
        probs = F.softmax(output, dim=-1)  # (..., nb)
        return (probs * centers).sum(dim=-1, keepdim=True)  # (..., 1)
