"""Cross-attention encoder variants.

In the pipeline the brain sees a single sequence ``(B, N, d_model)`` —
labeled context and query positions are mixed. These blocks structure
a forward pass as one or more layers of pre-norm cross-attention,
which the wizard surfaces as alternatives to the standard transformer
encoder:

  * ``cross_attention_layer`` — one pass. Lightweight; queries don't
    refine across iterations.
  * ``cross_attention_encoder`` — n stacked passes. The thinking core
    of in-context brains that want repeated study of the examples.

Both expose the same forward signature as ``transformer_encoder`` so
they slot into the chain in the same position without a contract
change. Internally they use ``nn.TransformerEncoderLayer`` (pre-norm)
— the layer's self-attention degenerates to "every token attends to
every other" which, for an interleaved context+query sequence, is the
same wiring as a cross-attention layer that treats the whole sequence
as both keys and queries.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block


def _encoder_layer(d_model: int, n_heads: int, dropout: float, ff_mult: int) -> Any:
    """Pre-norm encoder layer factory — used by both the single-layer
    and stacked variants below so they stay consistent."""
    import torch.nn as nn

    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * ff_mult,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )


@register_block("cross_attention_layer")
class CrossAttentionLayer:
    """One pre-norm cross-attention layer. embed_dim + num_heads
    match the wizard's extraBlockConfig defaults."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ):
        try:
            import torch.nn as nn  # noqa: F401
        except ImportError as e:
            raise ImportError("cross_attention_layer requires torch.") from e

        self._layer = _encoder_layer(embed_dim, num_heads, dropout, ff_mult)
        self.d_model = embed_dim

    def __call__(self, x: Any) -> Any:
        return self._layer(x)


@register_block("cross_attention_encoder")
class CrossAttentionEncoder:
    """A stack of cross-attention layers — n_layers passes over the
    same context+query sequence. Param names mirror ``transformer_
    encoder`` so the wizard's recipe-bound depth/width/heads sliders
    feed it without translation."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("cross_attention_encoder requires torch.") from e

        self.module = nn.TransformerEncoder(
            _encoder_layer(d_model, n_heads, dropout, ff_mult),
            num_layers=n_layers,
            # Pre-norm encoder layers don't benefit from nested-tensor
            # optimization — silence the warning by opting out.
            enable_nested_tensor=False,
        )
        self.d_model = d_model

    def __call__(self, x: Any) -> Any:
        return self.module(x)
