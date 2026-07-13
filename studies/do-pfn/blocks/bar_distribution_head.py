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
_SETUP_TASKS: int = 10_000
_SETUP_POINTS: int = 256
_SETUP_BASE_SEED: int = 987_654


def _bar_tensor_stats(name: str, tensor: Any) -> dict[str, Any]:
    import torch

    t = tensor.detach()
    finite = torch.isfinite(t)
    out: dict[str, Any] = {
        "name": name,
        "shape": tuple(t.shape),
        "numel": int(t.numel()),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(t).sum().item()),
        "posinf": int(torch.isposinf(t).sum().item()),
        "neginf": int(torch.isneginf(t).sum().item()),
    }
    if finite.any():
        vals = t[finite]
        out["min"] = float(vals.min().item())
        out["max"] = float(vals.max().item())
        out["max_abs"] = float(vals.abs().max().item())
    return out


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
    def setup(
        self, *, prior: Any, hp: dict | None = None, device: Any = None, **_: Any
    ) -> None:
        import numpy as np
        import torch

        pooled: list[np.ndarray] = []
        for i in range(_SETUP_TASKS):
            try:
                task = prior.sample(
                    seed=_SETUP_BASE_SEED + i, num_samples=_SETUP_POINTS
                )
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

        ally = (
            np.concatenate([p for p in pooled if p.size])
            if pooled
            else np.zeros(1, np.float32)
        )
        ally = ally[np.isfinite(ally)]
        print(
            "[bar_distribution_head] setup pooled outcomes "
            f"tasks={_SETUP_TASKS} "
            f"points_per_task={_SETUP_POINTS} "
            f"finite_values={int(ally.size)} "
            f"min={float(ally.min()) if ally.size else None} "
            f"max={float(ally.max()) if ally.size else None} "
            f"std={float(ally.std()) if ally.size else None}",
            flush=True,
        )
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
        print(
            "[bar_distribution_head] setup borders "
            f"num_buckets={self.num_buckets} "
            f"border_min={float(borders_t.min().item())} "
            f"border_max={float(borders_t.max().item())} "
            f"min_width={float((borders_t[1:] - borders_t[:-1]).min().item())} "
            f"max_width={float((borders_t[1:] - borders_t[:-1]).max().item())}",
            flush=True,
        )

    @staticmethod
    def _halfnormal_with_p_weight_before(range_max: Any, p: float = 0.5) -> Any:
        import torch

        p_t = torch.tensor(p, device=range_max.device, dtype=range_max.dtype)
        one = torch.tensor(1.0, device=range_max.device, dtype=range_max.dtype)
        unit = torch.distributions.HalfNormal(one)
        scale = range_max / unit.icdf(p_t).clamp_min(1e-8)
        return torch.distributions.HalfNormal(scale.clamp_min(1e-8))

    # ── generic custom-loss hook (bucketized NLL) ──────────────────────────
    def loss(self, logits: Any, target: Any) -> Any:
        import torch
        import torch.nn.functional as F  # noqa: N812

        borders = self.proj.bar_borders.to(device=logits.device, dtype=logits.dtype)
        nb = borders.numel() - 1

        logits = logits.reshape(-1, nb)
        target = target.reshape(-1).to(device=logits.device, dtype=logits.dtype)

        if not torch.isfinite(borders).all():
            print(
                "[bar_distribution_head] non-finite bucket borders "
                f"borders={_bar_tensor_stats('borders', borders)}",
                flush=True,
            )
            raise RuntimeError("BAR head has non-finite bucket borders.")

        widths = borders[1:] - borders[:-1]
        if not torch.isfinite(widths).all() or (widths <= 0).any():
            print(
                "[bar_distribution_head] invalid bucket widths "
                f"borders={_bar_tensor_stats('borders', borders)} "
                f"widths={_bar_tensor_stats('widths', widths)}",
                flush=True,
            )
            raise RuntimeError("BAR head has invalid bucket widths.")

        finite_target = torch.isfinite(target)
        finite_logits = torch.isfinite(logits).all(dim=-1)
        valid = finite_target & finite_logits

        if valid.sum() == 0:
            print(
                "[bar_distribution_head] no valid target/logit rows "
                f"target={_bar_tensor_stats('target', target)} "
                f"logits={_bar_tensor_stats('logits', logits)} "
                f"borders={_bar_tensor_stats('borders', borders)} "
                f"widths={_bar_tensor_stats('widths', widths)}",
                flush=True,
            )
            raise RuntimeError("BAR head got no finite target/logit rows.")

        if (~finite_logits & finite_target).any():
            print(
                "[bar_distribution_head] dropping non-finite logit rows "
                f"finite_targets={int(finite_target.sum().item())} "
                f"finite_logits={int(finite_logits.sum().item())} "
                f"valid={int(valid.sum().item())} "
                f"target={_bar_tensor_stats('target', target)} "
                f"logits={_bar_tensor_stats('logits', logits)}",
                flush=True,
            )

        logits = logits[valid].clamp(-60.0, 60.0)
        target = target[valid]

        # Stable BAR training path:
        # map each continuous target to one bucket, then use cross entropy.
        # This avoids fragile density/tail math that can create NaN gradients.
        target = target.clamp(
            min=borders[0].detach(),
            max=borders[-1].detach(),
        )

        bucket_idx = torch.bucketize(target, borders, right=False) - 1
        bucket_idx = bucket_idx.clamp(0, nb - 1).long()

        loss = F.cross_entropy(logits, bucket_idx)

        if not torch.isfinite(loss):
            print(
                "[bar_distribution_head] non-finite CE loss "
                f"loss={float(loss.detach().item()) if loss.numel() else None} "
                f"target={_bar_tensor_stats('target', target)} "
                f"logits={_bar_tensor_stats('logits', logits)} "
                f"bucket_idx={_bar_tensor_stats('bucket_idx', bucket_idx.float())} "
                f"borders={_bar_tensor_stats('borders', borders)} "
                f"widths={_bar_tensor_stats('widths', widths)}",
                flush=True,
            )
            raise RuntimeError("BAR head produced non-finite cross-entropy loss.")

        return loss

    # ── generic prediction-reduction hook (distribution mean) ──────────────
    def to_prediction(self, output: Any) -> Any:
        import torch.nn.functional as F  # noqa: N812

        borders = self.proj.bar_borders
        centers = 0.5 * (borders[1:] + borders[:-1])  # (nb,)
        probs = F.softmax(output, dim=-1)  # (..., nb)
        return (probs * centers).sum(dim=-1, keepdim=True)  # (..., 1)
