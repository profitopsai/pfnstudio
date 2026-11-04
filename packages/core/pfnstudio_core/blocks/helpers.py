"""Simple helper blocks — normalization and feed-forward mixers.

These are the small structural pieces that sit between bigger blocks
in a brain's chain. None of them do heavy lifting on their own, but
they are common enough that the wizard treats them as first-class
helpers under their parent slots (rms_norm under THINKING, mlp under
SKILL).
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block


@register_block("rms_norm")
class RMSNorm:
    """Root-mean-square layer norm — cheaper than LayerNorm because it
    skips the mean centering. Common between transformer layers or
    right after the encoder before a head reads it.
    """

    def __init__(self, dim: int = 256, eps: float = 1e-6):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("rms_norm requires torch.") from e

        # nn.RMSNorm is available in torch ≥ 2.4. The wizard's basic
        # mode targets up-to-date torch, but fall back to a manual
        # implementation if the user is on an older build — keeps the
        # block usable across the supported range.
        if hasattr(nn, "RMSNorm"):
            self._norm: Any = nn.RMSNorm(dim, eps=eps)
        else:
            import torch

            class _Manual(nn.Module):
                def __init__(self, d: int, e: float):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(d))
                    self.eps = e

                def forward(self, x: Any) -> Any:
                    rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                    return x * rms * self.weight

            self._norm = _Manual(dim, eps)
        self.dim = dim

    def __call__(self, x: Any) -> Any:
        return self._norm(x)


@register_block("mlp")
class MLP:
    """Feed-forward mixer — `n_layers` of (Linear → activation), then a
    final projection. Often sits between the encoder and a head to give
    the model one more chance to shape its representation before the
    output projection.
    """

    def __init__(
        self,
        d_in: int = 256,
        d_hidden: int | None = None,
        d_out: int | None = None,
        n_layers: int = 2,
        activation: str = "gelu",
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("mlp requires torch.") from e

        d_hidden = d_hidden if d_hidden is not None else d_in * 4
        d_out = d_out if d_out is not None else d_in

        act_cls: Any
        if activation == "gelu":
            act_cls = nn.GELU
        elif activation == "relu":
            act_cls = nn.ReLU
        elif activation == "silu":
            act_cls = nn.SiLU
        else:
            raise ValueError(
                f"mlp: unsupported activation '{activation}'. Use 'gelu', 'relu', or 'silu'."
            )

        # n_layers controls the count of hidden Linear+activation pairs
        # before the final projection. n_layers=2 → Linear → act →
        # Linear → act → Linear(out). n_layers=1 → Linear → act →
        # Linear(out). Always at least one hidden + one output layer.
        layers: list[Any] = []
        in_dim = d_in
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(in_dim, d_hidden))
            layers.append(act_cls())
            in_dim = d_hidden
        layers.append(nn.Linear(in_dim, d_out))
        self._seq = nn.Sequential(*layers)
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

    def __call__(self, x: Any) -> Any:
        return self._seq(x)
