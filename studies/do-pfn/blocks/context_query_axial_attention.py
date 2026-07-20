"""Released-Do-PFN-compatible axial transformer layer for PFN Studio.

Paste this entire file into a project block and register/use the block as
``dopfn_axial_attention_block``. Replace every built-in
``axial_attention_block`` in the Do-PFN model with this block.

Input/output shape: ``(B, rows, feature_tokens, d_model)``.

The important difference from PFN Studio's generic axial block is the
query-to-context attention. Do-PFN uses ordinary six-head attention: every
query head reads the matching K/V head from the context. Query rows never
attend to other query rows.
"""

from __future__ import annotations

import json
from typing import Any

from pfnstudio_core.registry import register_block


@register_block("context_query_axial_attention")
class DoPFNAxialAttentionBlock:
    """One post-norm PerFeatureEncoderLayer matching released Do-PFN."""

    needs_single_eval_pos: bool = True
    _instance_counter: int = 0

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 6,
        ff_mult: int = 4,
        layer_norm_eps: float = 1.0e-5,
        log_first_n: int = 1,
        **_: Any,
    ) -> None:
        import torch.nn as nn

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.ff_mult = int(ff_mult)
        self.layer_norm_eps = float(layer_norm_eps)
        self.log_first_n = max(0, int(log_first_n))
        self._forward_calls = 0
        self.instance_index = type(self)._instance_counter
        type(self)._instance_counter += 1

        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive; got {self.d_model}.")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(f"n_heads={self.n_heads} must divide d_model={self.d_model}.")
        if self.ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive; got {self.ff_mult}.")
        if self.layer_norm_eps <= 0.0:
            raise ValueError(f"layer_norm_eps must be positive; got {self.layer_norm_eps}.")

        # These are standard MultiheadAttention modules, as used by the
        # released PerFeatureEncoderLayer. bias=False comes from its checkpoint
        # configuration. Each head retains its own Q/K/V representation.
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.item_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )

        # The released checkpoint uses parameter-free post-LayerNorm.
        self.norm_features = nn.LayerNorm(
            self.d_model,
            eps=self.layer_norm_eps,
            elementwise_affine=False,
        )
        self.norm_items = nn.LayerNorm(
            self.d_model,
            eps=self.layer_norm_eps,
            elementwise_affine=False,
        )
        self.norm_mlp = nn.LayerNorm(
            self.d_model,
            eps=self.layer_norm_eps,
            elementwise_affine=False,
        )

        hidden = self.ff_mult * self.d_model
        self.linear1 = nn.Linear(self.d_model, hidden, bias=False)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden, self.d_model, bias=False)

        # Do-PFN/TabPFN initialization: residual branches initially contribute
        # zero, while Q/K/V and the first MLP projection retain their normal
        # PyTorch initialization. Do not reset the RNG per block: all 12 layers
        # must receive distinct parameters under the run's global seed.
        nn.init.zeros_(self.feature_attention.out_proj.weight)
        nn.init.zeros_(self.item_attention.out_proj.weight)
        nn.init.zeros_(self.linear2.weight)

    def __call__(
        self,
        x: Any,
        *,
        single_eval_pos: int | None = None,
    ) -> Any:
        return self.forward(x, single_eval_pos=single_eval_pos)

    def forward(
        self,
        x: Any,
        *,
        single_eval_pos: int | None = None,
    ) -> Any:
        import torch

        if not torch.is_tensor(x):
            raise TypeError(
                f"dopfn_axial_attention_block expects a torch.Tensor; got {type(x).__name__}."
            )
        if x.dim() != 4:
            raise ValueError(
                f"dopfn_axial_attention_block expects shape (B, R, C, E); got {tuple(x.shape)}."
            )

        batch_size, rows, feature_tokens, embedding = (int(value) for value in x.shape)
        if embedding != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}; got final dimension {embedding}.")
        if single_eval_pos is None:
            raise ValueError(
                "dopfn_axial_attention_block requires single_eval_pos from the prior's n_ctx value."
            )
        n_ctx = int(single_eval_pos)
        if not 0 < n_ctx <= rows:
            raise ValueError(f"single_eval_pos must be in [1, {rows}]; got {n_ctx}.")
        if not torch.isfinite(x).all():
            self._log("error", "non_finite_input", **self._stats(x))
            raise FloatingPointError("dopfn_axial_attention_block received NaN/Inf.")

        # 1. Attention between feature tokens, independently for every row.
        feature_input = x.reshape(
            batch_size * rows,
            feature_tokens,
            self.d_model,
        )
        feature_output, _ = self.feature_attention(
            feature_input,
            feature_input,
            feature_input,
            need_weights=False,
        )
        feature_output = feature_output.reshape(
            batch_size,
            rows,
            feature_tokens,
            self.d_model,
        )
        x = self.norm_features(x + feature_output)

        # 2. Attention between rows, independently for every feature token.
        # Context rows attend to context rows. Query rows use all six matching
        # context K/V heads and cannot see themselves or other query rows.
        item_input = (
            x.transpose(1, 2).contiguous().reshape(batch_size * feature_tokens, rows, self.d_model)
        )
        context = item_input[:, :n_ctx]
        context_output, _ = self.item_attention(
            context,
            context,
            context,
            need_weights=False,
        )

        if n_ctx < rows:
            query = item_input[:, n_ctx:]
            query_output, _ = self.item_attention(
                query,
                context,
                context,
                need_weights=False,
            )
            item_output = torch.cat([context_output, query_output], dim=1)
        else:
            item_output = context_output

        item_output = (
            item_output.reshape(
                batch_size,
                feature_tokens,
                rows,
                self.d_model,
            )
            .transpose(1, 2)
            .contiguous()
        )
        x = self.norm_items(x + item_output)

        # 3. Position-wise GELU MLP and post-norm residual.
        mlp_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm_mlp(x + mlp_output)

        if not torch.isfinite(x).all():
            self._log("error", "non_finite_output", **self._stats(x))
            raise FloatingPointError("dopfn_axial_attention_block produced NaN/Inf.")

        self._forward_calls += 1
        if self._forward_calls <= self.log_first_n:
            self._log(
                "info",
                "forward_summary",
                input_shape=[batch_size, rows, feature_tokens, embedding],
                output_shape=list(x.shape),
                n_ctx=n_ctx,
                n_query=rows - n_ctx,
                d_model=self.d_model,
                n_heads=self.n_heads,
                head_dim=self.d_model // self.n_heads,
                ff_hidden=self.ff_mult * self.d_model,
                query_attention="full_multi_head_context_kv",
                max_abs_output=float(x.detach().abs().max().item()),
            )

        return x

    @staticmethod
    def _stats(value: Any) -> dict[str, Any]:
        import torch

        finite = torch.isfinite(value)
        return {
            "shape": list(value.shape),
            "finite": int(finite.sum().item()),
            "nan": int(torch.isnan(value).sum().item()),
            "posinf": int(torch.isposinf(value).sum().item()),
            "neginf": int(torch.isneginf(value).sum().item()),
        }

    def _log(self, level: str, message: str, **fields: Any) -> None:
        print(
            json.dumps(
                {
                    "event": "dopfn_axial_attention_block",
                    "level": level,
                    "message": message,
                    "layer_index": self.instance_index,
                    **fields,
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
