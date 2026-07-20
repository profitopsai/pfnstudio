"""Cell-grid preprocessing pipeline — bridges raw tabular data to the
cell_embedder's input contract.

The ``tabular_cell_embedder`` block (see
``packages/core/pfnstudio_core/blocks/grid.py``) expects
``(B, R, C, 2)`` input where the last dim is
``[preprocessed_value, nan_indicator]``. This module produces that
shape from raw ``(B, R, C)`` tabular input via a fitted preprocessing
pipeline.

Pipeline (the live, testable version of upstream's ``_embed_features``
preprocessing — see tabpfn_v2.py lines 638–711):

  1. Extract NaN/+Inf/−Inf indicators (BEFORE imputation, so the
     model knows which cells were originally missing).
  2. Replace NaN/Inf with the per-feature mean of train rows (the
     "imputation" step — the cell embedder downstream doesn't see NaN).
  3. Standard-scale per-feature using train-row mean + std.
  4. Stack value + indicator → ``(B, R, C, 2)``.

Stats are fit on the train rows (the first ``n_train`` rows along R)
and applied to the whole tensor. Supports the in-context learning
contract: train rows fit + transform together; test rows are
transformed with the train-fitted stats.

What's NOT in this module yet (deferred from upstream parity):
  - Constant-feature removal (the column-selection mask).
  - Per-group normalization (``_normalize_feature_groups``).
  - The fitted-cache reuse path for inference (covered when the
    ModelLoader integration lands in a later phase).

These extras affect numerical agreement with the upstream checkpoint
but don't gate the prior → preprocessor → cell_embedder chain working
end-to-end, which is what Phase 3 needs to establish.

The shipped version (when this lands in a project) doesn't need to
mirror this string anywhere — preprocessing is studio-side
infrastructure, not project-shipped Python.
"""

from __future__ import annotations

# Lazy torch import — module loads on python-only environments; the
# functions raise clearly on first use without torch.
try:
    import torch

    _Tensor = torch.Tensor
    _Module = torch.nn.Module
except ImportError:  # pragma: no cover — torch is a hard requirement here
    torch = None  # type: ignore[assignment]
    _Tensor = object  # type: ignore[assignment]
    _Module = object  # type: ignore[assignment]


# ── Indicator constants — must match the cell embedder ────────────────
# These are the values the ``tabular_cell_embedder`` reads through its
# 2nd input channel (see ``blocks/tabpfn.py`` constants of the same
# name). Drifting these breaks every paper-pinned model silently — the
# constants are re-exported here so importers can use one canonical
# source.
NAN_INDICATOR: float = -2.0
INFINITY_INDICATOR: float = 2.0
NEG_INFINITY_INDICATOR: float = 4.0


def extract_nan_indicator(x):
    """Per-cell NaN / +Inf / −Inf indicator tensor.

    Args
    ----
    x : torch.Tensor of shape ``(B, R, C)``.

    Returns
    -------
    torch.Tensor of the same shape, where each cell is:
      - ``NAN_INDICATOR`` (−2.0) if the cell was NaN
      - ``INFINITY_INDICATOR`` (2.0) if the cell was +Inf
      - ``NEG_INFINITY_INDICATOR`` (4.0) if the cell was −Inf
      - 0.0 if the cell was finite
    """
    if torch is None:  # pragma: no cover
        raise ImportError(
            "extract_nan_indicator requires torch. Install with: pip install pfnstudio-core[torch]"
        )
    indicator = torch.zeros_like(x)
    indicator[torch.isnan(x)] = NAN_INDICATOR
    indicator[torch.isposinf(x)] = INFINITY_INDICATOR
    indicator[torch.isneginf(x)] = NEG_INFINITY_INDICATOR
    return indicator


def mean_impute(x, n_train: int):
    """Replace NaN/Inf with the per-feature mean of train rows.

    Args
    ----
    x : torch.Tensor of shape ``(B, R, C)``, possibly with NaN/Inf.
    n_train : int
        Number of train rows along the R axis (first n_train).

    Returns
    -------
    cleaned : torch.Tensor, same shape as x, all finite.
    feature_means : torch.Tensor of shape ``(B, C)`` — per-batch,
        per-feature train-row mean used for imputation. Cached for
        later reuse on test-only rows.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("mean_impute requires torch.")
    if x.dim() != 3:
        raise ValueError(f"mean_impute expects 3-D input (B, R, C); got shape {tuple(x.shape)}.")
    if not (0 < n_train <= x.shape[1]):
        raise ValueError(f"n_train must be in (0, R={x.shape[1]}]; got {n_train}.")

    # Compute per-feature means from the train rows, ignoring non-finite
    # cells. We replace non-finite cells with zero, sum, and divide by
    # the count of finite cells per feature.
    train_x = x[:, :n_train]  # (B, n_train, C)
    finite_mask = torch.isfinite(train_x)
    safe_train = torch.where(finite_mask, train_x, torch.zeros_like(train_x))
    counts = finite_mask.sum(dim=1).to(x.dtype).clamp(min=1.0)  # (B, C)
    feature_means = safe_train.sum(dim=1) / counts  # (B, C)

    # Replace any non-finite cell anywhere in x with its feature's mean.
    means_broadcast = feature_means.unsqueeze(1).expand_as(x)
    cleaned = torch.where(torch.isfinite(x), x, means_broadcast)
    return cleaned, feature_means


def standard_scale(x, n_train: int):
    """Standard-scale per-feature using train-row statistics.

    Args
    ----
    x : torch.Tensor of shape ``(B, R, C)``, assumed already finite
        (call ``mean_impute`` first).
    n_train : int
        Number of train rows along the R axis.

    Returns
    -------
    scaled : torch.Tensor, same shape as x, train rows have mean ≈ 0
        and std ≈ 1 per feature per batch.
    stats : dict
        ``{"mean": (B, C), "std": (B, C)}`` — fitted statistics for
        later reuse on test-only rows.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("standard_scale requires torch.")
    if x.dim() != 3:
        raise ValueError(f"standard_scale expects 3-D input (B, R, C); got shape {tuple(x.shape)}.")
    if not (0 < n_train <= x.shape[1]):
        raise ValueError(f"n_train must be in (0, R={x.shape[1]}]; got {n_train}.")

    train_x = x[:, :n_train]
    mean = train_x.mean(dim=1)  # (B, C)
    # Clamp std lower bound to avoid divide-by-zero on constant features.
    # Constant features would emit NaN otherwise.
    std = train_x.std(dim=1, unbiased=False).clamp(min=1e-6)  # (B, C)
    scaled = (x - mean.unsqueeze(1)) / std.unsqueeze(1)
    return scaled, {"mean": mean, "std": std}


class TabularPreprocessor:
    """Composes the three preprocessing steps into a stateful pipeline.

    Usage
    -----
    ::

        preprocessor = TabularPreprocessor()
        # Fit on train rows + transform all rows in one shot.
        out = preprocessor.fit_transform(x_BRC, n_train=...)
        # out: (B, R, C, 2) ready for tabular_cell_embedder.

        # Later, with new test-only rows (e.g. cached inference reuse):
        new_out = preprocessor.transform_test_only(x_test_BRC)

    State (None before ``fit_transform`` is called):
      - ``feature_means``: (B, C) — per-feature train-row means.
      - ``scaler_stats``: ``{"mean": (B, C), "std": (B, C)}``.

    Reusable across forward passes only if the train set is the same.
    For a different train set, instantiate a fresh preprocessor or call
    ``fit_transform`` again (overwrites stats).
    """

    def __init__(self):
        self.feature_means = None  # (B, C) — set by fit_transform
        self.scaler_stats = None  # {"mean", "std"} — set by fit_transform
        self._fitted = False

    def fit_transform(self, x, n_train: int):
        """Fit on train rows + transform all rows.

        Args
        ----
        x : torch.Tensor of shape ``(B, R, C)``.
        n_train : int — number of train rows along R.

        Returns
        -------
        torch.Tensor of shape ``(B, R, C, 2)`` — last dim is
        ``[preprocessed_value, nan_indicator]``. This is the exact
        input contract of the Phase-1 ``tabular_cell_embedder`` block.
        """
        if torch is None:  # pragma: no cover
            raise ImportError("TabularPreprocessor requires torch.")
        if x.dim() != 3:
            raise ValueError(f"fit_transform expects 3-D input (B, R, C); got {tuple(x.shape)}.")

        # 1. Extract NaN/Inf indicators BEFORE any cell is replaced.
        indicator = extract_nan_indicator(x)
        # 2. Mean-impute NaN/Inf using train-row means.
        cleaned, self.feature_means = mean_impute(x, n_train)
        # 3. Standard-scale per-feature using train-row stats.
        scaled, self.scaler_stats = standard_scale(cleaned, n_train)
        self._fitted = True

        # 4. Stack value + indicator along a new last dim → (B, R, C, 2).
        return torch.stack([scaled, indicator], dim=-1)

    def transform_test_only(self, x_test):
        """Apply previously-fitted stats to new test rows.

        This is what the in-context inference path calls when reusing
        a cached training context (matching upstream's ``feature_cache``
        in ``_embed_features``).

        Args
        ----
        x_test : torch.Tensor of shape ``(B, R_test, C)``.

        Returns
        -------
        torch.Tensor of shape ``(B, R_test, C, 2)``.
        """
        if torch is None:  # pragma: no cover
            raise ImportError("TabularPreprocessor requires torch.")
        if not self._fitted:
            raise RuntimeError(
                "transform_test_only called before fit_transform — preprocessor "
                "has no fitted stats to apply."
            )
        if x_test.dim() != 3:
            raise ValueError(f"transform_test_only expects 3-D input; got {tuple(x_test.shape)}.")

        indicator = extract_nan_indicator(x_test)
        # Use cached train-row means for imputation.
        means_broadcast = self.feature_means.unsqueeze(1).expand_as(x_test)
        cleaned = torch.where(torch.isfinite(x_test), x_test, means_broadcast)
        # Use cached train-row stats for scaling.
        scaled = (cleaned - self.scaler_stats["mean"].unsqueeze(1)) / self.scaler_stats[
            "std"
        ].unsqueeze(1)
        return torch.stack([scaled, indicator], dim=-1)
