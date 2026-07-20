"""Paper/checkpoint-faithful Do-PFN input encoder for PFN Studio.

Paste this entire file into the project block registered as
``dopfn_input_encoder``.  It replaces BOTH ``grid_preprocessor`` and
``tabular_cell_embedder``.

Expected raw grid
-----------------
``x`` has shape ``(B, R, C)`` and columns ``[treatment, covariates..., y]``.
The first ``single_eval_pos`` rows are context rows.  Query-row ``y`` values
must be NaN so that the answer cannot leak into the model.

Output
------
``(B, R, G + 1, d_model)`` where ``G`` is the number of X feature groups and
the LAST token is the separately encoded Y token.  With the released Do-PFN
setting ``features_per_group=85`` and this prior's <= 7 X columns, ``G=1`` and
the output is ``(B, R, 2, 192)``.

The implementation follows the encoder serialized in ``dopfn_model.pkl``:

* X: NaN/Inf indicator -> context-mean imputation -> context-only
  normalization and [-100, 100] clipping -> pad/group to 85 -> concatenate
  value and missingness channels -> Linear(170, 192).
* Y: externally context-standardized by the prior/inference wrapper because
  the checkpoint has ``transform_target=True``; then NaN/Inf indicator ->
  context-mean imputation -> concatenate value and missingness channels ->
  Linear(2, 192).
* The encoded Y is appended as the last feature token.

The serialized encoder contains a ColumnMarkerEncoderStep, but its final
LinearInputEncoderStep consumes only ``main`` and ``nan_indicators``.  Adding
learned treatment/covariate/outcome role embeddings would therefore introduce
parameters that are not present in the released checkpoint.  Treatment is
kept in the first fixed X slot, matching the released input order.
"""

from __future__ import annotations

import json
from typing import Any

from pfnstudio_core.registry import register_block


@register_block("separate_xy_input_encoder")
class DoPFNInputEncoder:
    """Separate X/Y encoder used before the Do-PFN axial transformer stack."""

    # PFN Studio threads batch['n_ctx'] to blocks that opt into this contract.
    needs_single_eval_pos: bool = True

    NAN_INDICATOR = -2.0
    POS_INF_INDICATOR = 2.0
    NEG_INF_INDICATOR = 4.0

    def __init__(
        self,
        d_model: int = 192,
        num_cols: int = 8,
        features_per_group: int = 85,
        normalize_by_used_features: bool = True,
        clip_value: float = 100.0,
        eps: float = 1.0e-6,
        bias: bool = True,
        seed: int = 42,
        log_first_n: int = 3,
        **_: Any,
    ) -> None:
        import torch
        import torch.nn as nn

        self.d_model = int(d_model)
        self.num_cols = int(num_cols)
        self.features_per_group = int(features_per_group)
        self.normalize_by_used_features = bool(normalize_by_used_features)
        self.clip_value = float(clip_value)
        self.eps = float(eps)
        self.log_first_n = max(0, int(log_first_n))
        self._forward_calls = 0

        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive; got {self.d_model}.")
        if self.features_per_group <= 0:
            raise ValueError(f"features_per_group must be positive; got {self.features_per_group}.")
        if self.clip_value <= 0.0:
            raise ValueError(f"clip_value must be positive; got {self.clip_value}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive; got {self.eps}.")

        # Use a private RNG scope: deterministic initialization without changing
        # the seed used by blocks constructed after this project block.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.x_encoder = nn.Linear(
                2 * self.features_per_group,
                self.d_model,
                bias=bool(bias),
            )
            self.y_encoder = nn.Linear(2, self.d_model, bias=bool(bias))

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
                "dopfn_input_encoder expects a torch.Tensor with shape "
                f"(B, R, C); got {type(x).__name__}."
            )
        if x.dim() != 3:
            raise ValueError(
                "dopfn_input_encoder must receive the RAW grid (B, R, C). "
                "Remove grid_preprocessor and tabular_cell_embedder from the "
                f"model. Received shape {tuple(x.shape)}."
            )

        batch_size, rows, cols = (int(v) for v in x.shape)
        if cols < 2:
            raise ValueError(
                "Do-PFN input needs at least one X column and one final Y "
                f"column; received C={cols}."
            )
        if single_eval_pos is None:
            raise ValueError(
                "dopfn_input_encoder requires single_eval_pos. Ensure the "
                "prior returns batch['n_ctx']."
            )
        n_ctx = int(single_eval_pos)
        if not 0 < n_ctx < rows:
            raise ValueError(f"single_eval_pos must be in [1, R-1]; got {n_ctx} for R={rows}.")
        if not x.is_floating_point():
            x = x.float()

        # The prior emits Y as its final column, but PFN Studio may right-pad a
        # later, narrower variable-feature batch to a width seen on an earlier
        # step. Those zero padding columns then appear *after* Y. Locate Y by
        # the Do-PFN masking contract instead: it is the one column that is
        # finite in context and non-finite for every query row.
        y_col = self._locate_y_column(x, n_ctx)
        if y_col <= 0:
            raise ValueError(
                f"Detected Y at column {y_col}; at least one treatment/X column must precede it."
            )
        x_raw = x[:, :, :y_col]
        y_raw = x[:, :, y_col : y_col + 1]
        right_padding_cols = cols - y_col - 1

        # A finite query Y would leak the answer.  The released model appends
        # missing Y rows internally when only context Y is supplied; Studio's
        # packed grid represents the same thing with NaNs in the final column.
        query_y_finite = torch.isfinite(y_raw[:, n_ctx:]).sum()
        if int(query_y_finite.item()) != 0:
            self._log(
                "error",
                "query_target_leakage",
                query_finite_y=int(query_y_finite.item()),
                n_ctx=n_ctx,
                rows=rows,
            )
            raise ValueError(
                "Query-row Y values must be NaN. Finite query targets would "
                "leak the answers into the transformer."
            )

        # PerFeatureTransformer groups/pads X before running its sequential
        # encoder.  Do the same and fold each feature group into the batch axis
        # while fitting row statistics.
        x_feature_count = int(x_raw.shape[-1])
        group_size = self.features_per_group
        pad = (-x_feature_count) % group_size
        if pad:
            x_raw = torch.cat([x_raw, x_raw.new_zeros(batch_size, rows, pad)], dim=-1)
        n_groups = int(x_raw.shape[-1] // group_size)
        x_grouped = (
            x_raw.reshape(batch_size, rows, n_groups, group_size)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * n_groups, rows, group_size)
        )

        x_indicator = self._nan_indicator(x_grouped)
        x_clean, x_all_missing = self._mean_impute(x_grouped, n_ctx)
        x_scaled_unclipped = self._normalize_x(x_clean, n_ctx)
        x_clip_count = int((x_scaled_unclipped.abs() > self.clip_value).sum().item())
        x_scaled = x_scaled_unclipped.clamp(
            min=-self.clip_value,
            max=self.clip_value,
        )

        # Checkpoint setting: normalize_by_used_features=True.  A feature is
        # "used" if any row differs from row zero. Padded zero features are not
        # counted. This is VariableNumFeaturesEncoderStep's exact purpose.
        if self.normalize_by_used_features:
            used = (x_scaled[:, 1:, :] != x_scaled[:, :1, :]).any(dim=1)
            used_count = used.sum(dim=-1).clamp(min=1).to(x_scaled.dtype)
            scale = (float(group_size) / used_count).sqrt().view(-1, 1, 1)
            x_scaled = x_scaled * scale

        x_encoder_input = torch.cat([x_scaled, x_indicator], dim=-1)
        embedded_x = self.x_encoder(x_encoder_input)
        embedded_x = (
            embedded_x.reshape(batch_size, n_groups, rows, self.d_model)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # The released y_encoder is NanHandlingEncoderStep followed directly
        # by LinearInputEncoderStep(2, 192); it does not normalize Y here.
        y_indicator = self._nan_indicator(y_raw)
        y_clean, y_all_missing = self._mean_impute(y_raw, n_ctx)
        y_encoder_input = torch.cat([y_clean, y_indicator], dim=-1)
        embedded_y = self.y_encoder(y_encoder_input).unsqueeze(2)

        out = torch.cat([embedded_x, embedded_y], dim=2)
        if not torch.isfinite(out).all():
            finite = torch.isfinite(out)
            self._log(
                "error",
                "non_finite_encoder_output",
                shape=list(out.shape),
                nan=int(torch.isnan(out).sum().item()),
                posinf=int(torch.isposinf(out).sum().item()),
                neginf=int(torch.isneginf(out).sum().item()),
                finite=int(finite.sum().item()),
            )
            raise FloatingPointError(
                "dopfn_input_encoder produced NaN/Inf; inspect the emitted "
                "non_finite_encoder_output event."
            )

        self._forward_calls += 1
        if self._forward_calls <= self.log_first_n:
            self._log(
                "info",
                "forward_summary",
                call=self._forward_calls,
                input_shape=[batch_size, rows, cols],
                expected_num_cols=self.num_cols,
                detected_y_column=y_col,
                ignored_right_padding_columns=right_padding_cols,
                x_features=x_feature_count,
                feature_groups=n_groups,
                output_shape=list(out.shape),
                n_ctx=n_ctx,
                n_query=rows - n_ctx,
                x_nonfinite=int((~torch.isfinite(x_grouped)).sum().item()),
                y_nonfinite=int((~torch.isfinite(y_raw)).sum().item()),
                x_all_missing_context_features=int(x_all_missing.sum().item()),
                y_all_missing_context_features=int(y_all_missing.sum().item()),
                standardized_values_clipped=x_clip_count,
                max_abs_standardized_before_clip=self._finite_abs_max(x_scaled_unclipped),
                max_abs_encoder_output=self._finite_abs_max(out),
            )

        return out

    def _locate_y_column(self, value: Any, n_ctx: int) -> int:
        import torch

        context = value[:, :n_ctx, :]
        query = value[:, n_ctx:, :]

        # Y has observed context values and is masked for all query rows.
        candidates = torch.isfinite(context).any(dim=1) & (~torch.isfinite(query)).all(dim=1)
        counts = candidates.sum(dim=1)
        if not (counts == 1).all():
            self._log(
                "error",
                "cannot_locate_y_column",
                input_shape=list(value.shape),
                n_ctx=n_ctx,
                candidate_counts=[int(v) for v in counts.detach().cpu().tolist()],
                candidate_columns=[
                    torch.nonzero(row, as_tuple=False).flatten().detach().cpu().tolist()
                    for row in candidates
                ],
            )
            raise ValueError(
                "Could not uniquely locate the masked Y column. Expected "
                "exactly one column that is finite in context and non-finite "
                "for every query row."
            )

        indices = candidates.to(torch.int64).argmax(dim=1)
        if not (indices == indices[0]).all():
            detected = [int(v) for v in indices.detach().cpu().tolist()]
            self._log(
                "error",
                "inconsistent_y_columns_in_batch",
                detected_columns=detected,
            )
            raise ValueError(
                "Samples in one batch have Y in different columns: "
                f"{detected}. The prior batch must share one feature width."
            )
        return int(indices[0].item())

    def _nan_indicator(self, value: Any) -> Any:
        import torch

        indicator = torch.zeros_like(value)
        indicator = torch.where(
            torch.isnan(value),
            indicator.new_full((), self.NAN_INDICATOR),
            indicator,
        )
        indicator = torch.where(
            torch.isposinf(value),
            indicator.new_full((), self.POS_INF_INDICATOR),
            indicator,
        )
        indicator = torch.where(
            torch.isneginf(value),
            indicator.new_full((), self.NEG_INF_INDICATOR),
            indicator,
        )
        return indicator

    def _mean_impute(self, value: Any, n_ctx: int) -> tuple[Any, Any]:
        import torch

        context = value[:, :n_ctx]
        finite = torch.isfinite(context)
        counts = finite.sum(dim=1)
        all_missing = counts == 0
        safe = torch.where(finite, context, torch.zeros_like(context))
        means = safe.sum(dim=1) / counts.clamp(min=1).to(value.dtype)
        means = torch.where(all_missing, torch.zeros_like(means), means)
        cleaned = torch.where(torch.isfinite(value), value, means.unsqueeze(1).expand_as(value))
        return cleaned, all_missing

    def _normalize_x(self, value: Any, n_ctx: int) -> Any:
        context = value[:, :n_ctx]
        mean = context.mean(dim=1, keepdim=True)
        if n_ctx == 1:
            std = context.new_ones(context.shape[0], 1, context.shape[2])
        else:
            # Do-PFN's torch_nanstd uses the sample standard deviation (N-1)
            # and normalize_data adds 1e-6 afterward.
            std = context.std(dim=1, unbiased=True, keepdim=True) + self.eps
        std = std.clamp_min(self.eps)
        return (value - mean) / std

    @staticmethod
    def _finite_abs_max(value: Any) -> float | None:
        import torch

        finite = torch.isfinite(value)
        if not finite.any():
            return None
        return float(value[finite].abs().max().item())

    @staticmethod
    def _log(level: str, message: str, **fields: Any) -> None:
        # One JSON object per line integrates cleanly with PFN Studio's run log.
        print(
            json.dumps(
                {
                    "event": "dopfn_input_encoder",
                    "level": level,
                    "message": message,
                    **fields,
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
