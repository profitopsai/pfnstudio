"""Paper/checkpoint-faithful Do-PFN bar-distribution head for PFN Studio.

Paste this entire file into the project's existing
``blocks/bar_distribution_head.py``.

The released Do-PFN checkpoint uses:

* decoder: Linear(192, 768) -> GELU -> Linear(768, 100)
* 100 equal-mass buckets estimated from prior outcomes
* FullSupportBarDistribution negative log density
* half-normal tails for the first and last buckets

PFN Studio discovers the generic ``setup``, ``loss`` and ``to_prediction``
hooks used below; no pfnstudio-core modification is required.
"""

from __future__ import annotations

import json
from typing import Any

from pfnstudio_core.registry import register_block


@register_block("bar_distribution_head")
class BarDistributionHead:
    """Do-PFN decoder plus full-support bar-distribution criterion."""

    # Project-defined output heads opt into PFN Studio's parallel head path.
    is_head: bool = True

    def __init__(
        self,
        d_model: int = 192,
        hidden_dim: int = 768,
        num_buckets: int = 100,
        setup_tasks: int = 100,
        setup_points: int = 256,
        setup_seed: int = 987_654,
        tail_mass_within_edge: float = 0.5,
        min_bucket_width: float = 1.0e-6,
        log_first_n: int = 3,
        seed: int = 42,
        **_: Any,
    ) -> None:
        import torch
        import torch.nn as nn

        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.num_buckets = int(num_buckets)
        self.setup_tasks = int(setup_tasks)
        self.setup_points = int(setup_points)
        self.setup_seed = int(setup_seed)
        self.tail_mass_within_edge = float(tail_mass_within_edge)
        self.min_bucket_width = float(min_bucket_width)
        self.log_first_n = max(0, int(log_first_n))
        self._loss_calls = 0

        if self.d_model <= 0 or self.hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive.")
        if self.num_buckets < 2:
            raise ValueError("num_buckets must be at least 2.")
        if self.setup_tasks <= 0 or self.setup_points <= 1:
            raise ValueError("setup_tasks must be > 0 and setup_points must be > 1.")
        if not 0.0 < self.tail_mass_within_edge < 1.0:
            raise ValueError("tail_mass_within_edge must be between 0 and 1.")
        if self.min_bucket_width <= 0.0:
            raise ValueError("min_bucket_width must be positive.")

        # Isolate initialization so this project block does not change the RNG
        # used to initialize subsequent modules.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.decoder = nn.Sequential(
                nn.Linear(self.d_model, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.num_buckets),
            )

        # setup() replaces these before a fresh run. Keeping borders on the
        # decoder makes them move, save and restore with this block's weights.
        self.decoder.register_buffer(
            "bar_borders",
            torch.linspace(-10.0, 10.0, self.num_buckets + 1),
        )
        self.decoder.register_buffer(
            "bar_borders_ready",
            torch.tensor(False, dtype=torch.bool),
        )

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def forward(self, x: Any) -> Any:
        import torch

        if not torch.is_tensor(x) or x.shape[-1] != self.d_model:
            shape = tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
            raise ValueError(
                "bar_distribution_head expects (..., d_model) input with "
                f"d_model={self.d_model}; got {shape}."
            )
        logits = self.decoder(x)
        if not torch.isfinite(logits).all():
            self._log("error", "non_finite_logits", **self._stats(logits))
            raise FloatingPointError("bar_distribution_head produced NaN/Inf logits.")
        return logits

    def setup(
        self,
        *,
        prior: Any,
        hp: dict[str, Any] | None = None,
        device: Any = None,
        **_: Any,
    ) -> None:
        """Estimate equal-mass borders from the same prior used for training.

        The released checkpoint configuration uses ``bar_dist_init_batches=100``;
        ``setup_tasks=100`` mirrors that one-time initialization scale.
        """
        import numpy as np
        import torch

        pooled: list[np.ndarray] = []
        failed = 0
        invalid = 0

        for index in range(self.setup_tasks):
            try:
                try:
                    task = prior.sample(
                        seed=self.setup_seed + index,
                        num_samples=self.setup_points,
                    )
                except TypeError:
                    task = prior.sample(seed=self.setup_seed + index)
            except Exception as exc:
                failed += 1
                if failed <= 3:
                    self._log(
                        "warning",
                        "border_sample_failed",
                        sample=index,
                        exception_type=type(exc).__name__,
                        reason=str(exc),
                    )
                continue

            if task.get("is_valid", True) is False:
                invalid += 1
                continue

            # Query interventional outcomes used as loss targets.
            query_y = np.asarray(task.get("y", []), dtype=np.float32).reshape(-1)
            if query_y.size:
                pooled.append(query_y[np.isfinite(query_y)])

            # Observed context outcomes stored in the raw grid's final column.
            grid = np.asarray(task.get("X", []), dtype=np.float32)
            if grid.ndim == 2 and grid.shape[1] >= 1:
                context_y = grid[:, -1]
                pooled.append(context_y[np.isfinite(context_y)])

        nonempty = [values for values in pooled if values.size]
        if not nonempty:
            self._log(
                "error",
                "border_setup_no_values",
                setup_tasks=self.setup_tasks,
                failed=failed,
                invalid=invalid,
            )
            raise RuntimeError("Could not collect any finite Y values for BAR borders.")

        values_np = np.concatenate(nonempty).astype(np.float32, copy=False)
        values_np = values_np[np.isfinite(values_np)]
        if values_np.size <= self.num_buckets:
            raise RuntimeError(
                "BAR border setup needs more finite outcomes than buckets: "
                f"values={values_np.size}, buckets={self.num_buckets}."
            )

        values = torch.as_tensor(values_np, dtype=torch.float32)
        borders = self._equal_mass_borders(values)
        if device is not None:
            borders = borders.to(device)
        borders = borders.to(
            device=self.decoder.bar_borders.device,
            dtype=self.decoder.bar_borders.dtype,
        )
        self.decoder.bar_borders.copy_(borders)
        self.decoder.bar_borders_ready.fill_(True)

        widths = borders[1:] - borders[:-1]
        self._log(
            "info",
            "border_setup_complete",
            setup_tasks=self.setup_tasks,
            failed=failed,
            invalid=invalid,
            finite_values=int(values_np.size),
            outcome_min=float(values_np.min()),
            outcome_max=float(values_np.max()),
            outcome_mean=float(values_np.mean()),
            outcome_std=float(values_np.std()),
            buckets=self.num_buckets,
            border_min=float(borders[0].item()),
            border_max=float(borders[-1].item()),
            min_bucket_width=float(widths.min().item()),
            max_bucket_width=float(widths.max().item()),
        )

    def loss(self, logits: Any, target: Any) -> Any:
        """Mean FullSupportBarDistribution negative log density."""
        import torch
        import torch.nn.functional as functional

        if logits.shape[-1] != self.num_buckets:
            raise ValueError(f"Expected {self.num_buckets} logits; got {logits.shape[-1]}.")
        if not torch.isfinite(logits).all():
            self._log("error", "non_finite_logits_in_loss", **self._stats(logits))
            raise FloatingPointError("BAR loss received NaN/Inf logits.")

        borders = self.decoder.bar_borders.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        widths = borders[1:] - borders[:-1]
        if not torch.isfinite(borders).all() or not torch.isfinite(widths).all():
            raise FloatingPointError("BAR borders contain NaN/Inf.")
        if (widths <= 0).any():
            raise ValueError("BAR borders must be strictly increasing.")

        flat_logits = logits.reshape(-1, self.num_buckets)
        flat_target = target.to(
            device=logits.device,
            dtype=logits.dtype,
        ).reshape(-1)
        if flat_logits.shape[0] != flat_target.shape[0]:
            raise ValueError(
                f"BAR logits/target row mismatch: {flat_logits.shape[0]} vs {flat_target.shape[0]}."
            )

        valid = torch.isfinite(flat_target)
        if not valid.any():
            self._log("error", "no_finite_targets", **self._stats(flat_target))
            raise FloatingPointError("BAR loss received no finite targets.")
        logits_valid = flat_logits[valid]
        y = flat_target[valid]

        # The bucket probability is converted to a density by dividing by its
        # width. Plain cross-entropy omits this term and is not BAR NLL.
        bucket_idx = torch.searchsorted(borders, y) - 1
        bucket_idx = bucket_idx.clamp(0, self.num_buckets - 1).long()
        scaled_log_probs = functional.log_softmax(logits_valid, dim=-1) - widths.log()
        log_density = scaled_log_probs.gather(1, bucket_idx[:, None]).squeeze(1)

        # Full support: replace the uniform density shape in the two edge bars
        # with half-normal tails extending to -inf/+inf. This is the released
        # FullSupportBarDistribution.forward calculation.
        left_mask = bucket_idx == 0
        right_mask = bucket_idx == self.num_buckets - 1
        left_tail = self._edge_halfnormal(widths[0])
        right_tail = self._edge_halfnormal(widths[-1])

        if left_mask.any():
            left_distance = (borders[1] - y[left_mask]).clamp_min(1.0e-8)
            log_density[left_mask] = (
                log_density[left_mask] + left_tail.log_prob(left_distance) + widths[0].log()
            )
        if right_mask.any():
            right_distance = (y[right_mask] - borders[-2]).clamp_min(1.0e-8)
            log_density[right_mask] = (
                log_density[right_mask] + right_tail.log_prob(right_distance) + widths[-1].log()
            )

        nll = -log_density
        result = nll.mean()
        if not torch.isfinite(result):
            self._log(
                "error",
                "non_finite_bar_nll",
                logits=self._stats(logits_valid),
                targets=self._stats(y),
                borders=self._stats(borders),
                nll=self._stats(nll),
            )
            raise FloatingPointError("Full-support BAR NLL became NaN/Inf.")

        self._loss_calls += 1
        if self._loss_calls <= self.log_first_n:
            counts = torch.bincount(bucket_idx, minlength=self.num_buckets)
            self._log(
                "info",
                "loss_summary",
                call=self._loss_calls,
                loss=float(result.detach().item()),
                valid_targets=int(valid.sum().item()),
                ignored_targets=int((~valid).sum().item()),
                left_tail_targets=int(left_mask.sum().item()),
                right_tail_targets=int(right_mask.sum().item()),
                occupied_buckets=int((counts > 0).sum().item()),
                target_min=float(y.min().item()),
                target_max=float(y.max().item()),
                logits_max_abs=float(logits_valid.detach().abs().max().item()),
            )
        return result

    def to_prediction(self, output: Any) -> Any:
        """Return the full-support distribution mean as a scalar prediction."""
        import torch
        import torch.nn.functional as functional

        borders = self.decoder.bar_borders.to(
            device=output.device,
            dtype=output.dtype,
        )
        widths = borders[1:] - borders[:-1]
        bucket_means = borders[:-1] + widths / 2.0

        left_tail = self._edge_halfnormal(widths[0])
        right_tail = self._edge_halfnormal(widths[-1])
        bucket_means = bucket_means.clone()
        bucket_means[0] = borders[1] - left_tail.mean
        bucket_means[-1] = borders[-2] + right_tail.mean

        probs = functional.softmax(output, dim=-1)
        mean = (probs * bucket_means).sum(dim=-1, keepdim=True)
        if not torch.isfinite(mean).all():
            self._log("error", "non_finite_prediction_mean", **self._stats(mean))
            raise FloatingPointError("BAR distribution mean became NaN/Inf.")
        return mean

    def _equal_mass_borders(self, values: Any) -> Any:
        """Match Do-PFN's get_bucket_limits equal-count midpoint algorithm."""
        import torch

        values = values.flatten()
        values = values[torch.isfinite(values)]
        usable = int(values.numel()) - (int(values.numel()) % self.num_buckets)
        values = values[:usable]
        sorted_values = values.sort().values
        per_bucket = usable // self.num_buckets

        inner = (
            sorted_values[per_bucket - 1 :: per_bucket][:-1] + sorted_values[per_bucket::per_bucket]
        ) / 2.0
        borders = torch.cat([sorted_values[:1], inner, sorted_values[-1:]])

        # Continuous Do-PFN outcomes normally make these strict already. The
        # nudge protects the density calculation if a custom prior has atoms.
        scale = max(float(sorted_values.std(unbiased=False).item()), 1.0)
        min_width = max(self.min_bucket_width, self.min_bucket_width * scale)
        borders = borders.clone()
        for index in range(1, borders.numel()):
            if borders[index] <= borders[index - 1]:
                borders[index] = borders[index - 1] + min_width
        return borders

    def _edge_halfnormal(self, edge_width: Any) -> Any:
        import torch

        one = edge_width.new_tensor(1.0)
        probability = edge_width.new_tensor(self.tail_mass_within_edge)
        unit = torch.distributions.HalfNormal(one)
        scale = edge_width / unit.icdf(probability).clamp_min(1.0e-8)
        return torch.distributions.HalfNormal(scale.clamp_min(1.0e-8))

    @staticmethod
    def _stats(value: Any) -> dict[str, Any]:
        import torch

        detached = value.detach()
        finite = torch.isfinite(detached)
        result: dict[str, Any] = {
            "shape": list(detached.shape),
            "numel": int(detached.numel()),
            "finite": int(finite.sum().item()),
            "nan": int(torch.isnan(detached).sum().item()),
            "posinf": int(torch.isposinf(detached).sum().item()),
            "neginf": int(torch.isneginf(detached).sum().item()),
        }
        if finite.any():
            vals = detached[finite]
            result.update(
                min=float(vals.min().item()),
                max=float(vals.max().item()),
                max_abs=float(vals.abs().max().item()),
            )
        return result

    @staticmethod
    def _log(level: str, message: str, **fields: Any) -> None:
        print(
            json.dumps(
                {
                    "event": "bar_distribution_head",
                    "level": level,
                    "message": message,
                    **fields,
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
