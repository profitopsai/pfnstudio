"""Time-aware positional encodings.

These blocks add positional/timing information to a sequence so a
downstream encoder can reason about *when* each token sits in time —
useful for forecasting brains (the ``time_series`` speciality).

Both variants take an input ``(B, N, d_model)`` and add an encoding
of the same shape, producing ``(B, N, d_model)``. The user picks a
short- or long-range variant depending on whether they're modelling
minute/hour delays or hour/day/week delays.

The two variants differ only in the scale of frequencies they cover:

  * ``short_range_time_encoding`` — sinusoidal positional encoding
    tuned for sequences up to a few hundred steps. Cheap and
    deterministic; no learnable parameters.
  * ``long_range_time_encoding`` — log-spaced sinusoidal frequencies
    so the encoding stays distinguishable across very long sequences
    (hours → weeks).

A future per-instance schema could expose ``max_seq_len`` as a
tunable; today both default to a reasonable per-variant cap.
"""

from __future__ import annotations

import math
from typing import Any

from ..registry import register_block


def _sinusoidal_table(seq_len: int, d_model: int, base: float) -> Any:
    """Standard sin/cos positional encoding table — `(seq_len, d_model)`.

    Frequencies span 1 down to 1/base^((d_model-2)/d_model). The
    short-range variant uses base=10000 (transformer paper default);
    long-range uses a larger base to stretch the spectrum.
    """
    import torch

    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model + 1) // 2 - (d_model % 2)])
    return pe


@register_block("short_range_time_encoding")
class ShortRangeTimeEncoding:
    """Sinusoidal positional encoding tuned for short sequences (≤ a
    few hundred steps). Adds the encoding to the input — no learnable
    parameters."""

    def __init__(self, d_model: int = 256, max_seq_len: int = 512):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("short_range_time_encoding requires torch.") from e

        pe = _sinusoidal_table(max_seq_len, d_model, base=10000.0)
        # Register as a buffer so device-moves carry it along, but it
        # isn't a learnable parameter — wrap a small Module to give us
        # state-dict mechanics without subclassing nn.Module ourselves
        # (keeps consistency with the other block classes here that
        # delegate to a wrapped submodule).
        self._table = nn.Parameter(pe, requires_grad=False)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def __call__(self, x: Any) -> Any:
        seq_len = x.shape[1]
        return x + self._table[:seq_len].unsqueeze(0)


@register_block("long_range_time_encoding")
class LongRangeTimeEncoding:
    """Log-spaced sinusoidal positional encoding — keeps frequencies
    distinguishable across very long sequences (hours → weeks).

    Same forward signature as the short-range variant; only the base
    frequency differs."""

    def __init__(self, d_model: int = 256, max_seq_len: int = 8192):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("long_range_time_encoding requires torch.") from e

        # Larger base = lower frequencies in the high half, so the
        # encoding still varies after 10k+ steps. Picked 1e5 — a
        # superset of the short-range spectrum for safe stacking.
        pe = _sinusoidal_table(max_seq_len, d_model, base=100000.0)
        self._table = nn.Parameter(pe, requires_grad=False)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def __call__(self, x: Any) -> Any:
        seq_len = x.shape[1]
        return x + self._table[:seq_len].unsqueeze(0)
