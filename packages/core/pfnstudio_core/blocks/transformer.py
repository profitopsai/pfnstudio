"""Transformer encoder block.

Implementation requires PyTorch. Importing this module without torch installed
will raise on first instantiation, not on import — so projects that use a
non-PyTorch framework can still load this package.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block


@register_block("transformer_encoder")
class TransformerEncoder:
    """Standard pre-norm transformer encoder."""

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
            raise ImportError(
                "transformer_encoder block requires torch. "
                "Install with: pip install pfnstudio-core[torch]"
            ) from e

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is silently disabled when norm_first=True
        # (pre-LayerNorm). Pass False explicitly so PyTorch doesn't warn —
        # we know the optimization can't apply to the pre-norm variant.
        self.module = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.d_model = d_model

    def __call__(self, x: Any) -> Any:
        return self.module(x)


@register_block("tag_aware_transformer_encoder")
class TagAwareTransformerEncoder:
    """Transformer encoder with an optional prepended tag token.

    Identical to :class:`TransformerEncoder` when ``tag_dim == 0`` or
    when ``tag`` is not passed to ``__call__`` — including state-dict
    layout when ``tag_dim == 0``. With a non-zero ``tag_dim``, the
    block carries an extra ``nn.Linear(tag_dim, d_model)`` that
    projects the tag vector to one token, prepends it to the input
    sequence, runs the encoder, and strips the tag token from the
    output (so the block's API stays ``(B, N, d_model) → (B, N, d_model)``).

    Back-compat:
    - A model spec that uses ``transformer_encoder`` is unchanged.
    - A model spec that uses ``tag_aware_transformer_encoder`` with
      ``tag_dim=0`` has the same parameter set as ``transformer_encoder``
      (no ``tag_embedder`` weights), so checkpoints round-trip cleanly.
    - Passing ``tag=None`` to a tag-aware block skips the prepend and
      runs the regular encoder forward — useful at inference when the
      caller has no constraints to inject.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        ff_mult: int = 4,
        tag_dim: int = 0,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "tag_aware_transformer_encoder block requires torch. "
                "Install with: pip install pfnstudio-core[torch]"
            ) from e

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is silently disabled when norm_first=True
        # (pre-LayerNorm). Pass False explicitly so PyTorch doesn't warn —
        # we know the optimization can't apply to the pre-norm variant.
        self.module = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.d_model = d_model
        self.tag_dim = tag_dim
        # Only allocate the tag embedder when tag_dim > 0. Zero-dim tag
        # means "this block is wired for tag awareness but has no axes
        # yet" — keep the state dict identical to a regular encoder so
        # the same checkpoint file works for both block types.
        self.tag_embedder = nn.Linear(tag_dim, d_model) if tag_dim > 0 else None

    def __call__(self, x: Any, tag: Any = None) -> Any:
        if self.tag_embedder is not None and tag is not None:
            # tag: (B, tag_dim) → (B, 1, d_model)
            tag_emb = self.tag_embedder(tag).unsqueeze(1)
            # Concat as the first token in the sequence.
            x_with_tag = __import__("torch").cat([tag_emb, x], dim=1)
            out_with_tag = self.module(x_with_tag)
            # Strip the tag token so the output shape matches input.
            return out_with_tag[:, 1:, :]
        return self.module(x)


@register_block("causal_attention_pool")
class CausalAttentionPool:
    """Pools a sequence to a single token via attention with a learned query."""

    def __init__(self, d_model: int = 256):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("causal_attention_pool requires torch.") from e

        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def __call__(self, x: Any) -> Any:
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)
