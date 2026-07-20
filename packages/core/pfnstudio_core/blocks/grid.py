"""Cell-grid primitive blocks for tabular in-context learning models.

Reusable studio blocks that operate on a 2D cell-table representation
(rows × columns of cells, each cell carrying a value and a NaN
indicator). The primitives compose into TabPFN-v2-style architectures
(Hollmann et al., *Nature* 2025) and Do-PFN-style architectures
(Robertson et al., NeurIPS 2025) — same axial-attention pattern,
different prior + head. Naming is paper-agnostic so future axial-on-
cells papers can compose these blocks without adding a paper-specific
mega-block to the registry.

Reference implementation for the architecture pattern:
``PriorLabs/TabPFN/src/tabpfn/architectures/tabpfn_v2.py`` (Prior Labs
License — read-only reference; this file is original).

Primitives this module hosts:

  1. ``grid_preprocessor``        — raw (B, R, C) → (B, R, C, 2)
  2. ``tabular_cell_embedder``    — embedding stage
  3. ``along_row_attention``      — feature-axis attention
  4. ``along_column_attention``   — row-axis attention + KV
  5. ``axial_attention_block``    — composed transformer layer
  6. ``row_pool_for_head``        — cell → row collapse

Each primitive lands with a unit test (``tests/core/test_grid_blocks.py``)
that asserts shape + determinism + (eventually) numerical agreement with
the upstream reference class on a fixed-seed input.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block

# Lazy torch import — module-level guard so the rest of the package can
# load on python-only environments. Block instantiation fails clearly
# (see each __init__) if torch isn't installed.
try:
    import torch.nn as nn

    _Module = nn.Module
except ImportError:  # pragma: no cover — torch is a required dep for these blocks
    _Module = object  # type: ignore[assignment]


def _sdpa_no_flash(
    q: Any,
    k: Any,
    v: Any,
    /,
    **kwargs: Any,
) -> Any:
    """``scaled_dot_product_attention`` that survives huge effective batches.

    Background: CUDA caps ``gridDim.x`` at 65,535. Both Flash and the
    memory-efficient backend launch one block per (batch * heads)
    slot, so ``B * H`` past 65k crashes the kernel with::

        RuntimeError: CUDA error: invalid configuration argument

    Grid-style row attention folds samples × rows together
    (``BR = B * R``) before reaching SDPA, so at Marathon batch
    sizes BR * H sails past 65k on every GPU we rent — Flash AND
    mem-efficient — and the error is not card-dependent (A100 80GB
    sees it the same as RTX 5070 12GB).

    Strategy:

      1. If ``B * H <= 60_000`` (safety margin under 65k), call SDPA
         directly with the EFFICIENT_ATTENTION backend. This is the
         fast path — covers sanity / standard preset sizes.
      2. Otherwise, chunk the batch dim into pieces small enough to
         fit, call SDPA per chunk under EFFICIENT_ATTENTION + MATH,
         concat results. Adds a small Python-loop overhead but is
         numerically identical (each chunk's attention is over the
         SAME K/V, since attention doesn't cross batch members).
      3. If even a single-row chunk doesn't fit (extreme H counts),
         force MATH backend only. MATH uses plain bmm/softmax/bmm
         with no gridDim ceiling — slower but always works.

    Other blocks (transformer_encoder, etc.) keep PyTorch's full
    auto-picker because their dims are bounded by seq_len, not BR —
    Flash is still the right pick there.

    Backward-compatible across PyTorch 2.0–2.4+: prefers the modern
    ``torch.nn.attention.sdpa_kernel``, falls back to the deprecated
    ``torch.backends.cuda.sdp_kernel`` for older versions.
    """
    import torch
    import torch.nn.functional as F

    # CUDA gridDim.x ceiling minus a safety margin for kernel-internal
    # block multipliers (some op variants launch 1.2-1.5x blocks per
    # batch slot). Empirically 60k is the largest value that's been
    # observed to NOT crash; 65535 itself sometimes still trips.
    GRID_MAX = 60_000

    # SDPA input shape contract is (B, H, S, D) — batch and head dims
    # are positions 0 and 1. q.shape[0] * q.shape[1] is the kernel
    # launch's gridDim.x equivalent. k/v can have B=1 broadcast (e.g.
    # GQA enable_gqa=True path); we chunk against q's batch dim and
    # broadcast k/v to match per-chunk.
    if q.dim() < 4:
        # Defensive: if a caller hands us a 3-D tensor (unusual for
        # SDPA), don't try to chunk — just dispatch and let SDPA
        # raise if the shape's wrong.
        return _sdpa_call(F, q, k, v, **kwargs)

    B, H = int(q.shape[0]), int(q.shape[1])
    total = B * H

    if total <= GRID_MAX:
        # Fast path — fits in one kernel launch.
        return _sdpa_call(F, q, k, v, **kwargs)

    # Chunked path — split q's batch dim. K/V might be the same batch
    # size as q, or broadcast (B=1), or smaller (uncommon but possible
    # under certain GQA variants).
    chunk = max(1, GRID_MAX // max(H, 1))
    parts = []
    for i in range(0, B, chunk):
        q_i = q[i : i + chunk]
        # Broadcast-aware k/v slicing: if k/v already match q's batch
        # we slice them; if they're broadcast (B=1) we pass through.
        k_i = k[i : i + chunk] if k.shape[0] == B else k
        v_i = v[i : i + chunk] if v.shape[0] == B else v
        parts.append(_sdpa_call(F, q_i, k_i, v_i, **kwargs))
    return torch.cat(parts, dim=0)


def _sdpa_call(F: Any, q: Any, k: Any, v: Any, **kwargs: Any) -> Any:
    """Inner SDPA dispatch with backend hint. Used by _sdpa_no_flash's
    fast path AND each chunk in the chunked path. Single source of
    truth for the backend selection."""
    import torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore[attr-defined]

        # EFFICIENT_ATTENTION first (fast), MATH as fallback if the
        # shape/dtype combo isn't supported by EFFICIENT. The chunked
        # caller ensures B * H <= GRID_MAX per call so neither backend
        # hits the gridDim ceiling.
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    except ImportError:  # PyTorch < 2.3
        with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
            enable_flash=False,
            enable_mem_efficient=True,
            enable_math=True,
        ):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)


# NaN/Inf indicator values matching upstream constants (lines 47-49 of
# tabpfn_v2.py). Exposed at module scope so the ModelLoader's
# preprocessing pipeline can produce indicators with the same encoding
# the cell embedder expects.
NAN_INDICATOR: float = -2.0
INFINITY_INDICATOR: float = 2.0
NEG_INFINITY_INDICATOR: float = 4.0

# The cell input is concatenated [value, nan_indicator] before the
# linear projection — see upstream's ENCODING_SIZE_MULTIPLIER=2.
# Doubling factor for the per-feature representation entering the
# feature-group linear projection.
ENCODING_SIZE_MULTIPLIER: int = 2


@register_block("tabular_cell_embedder")
class TabularCellEmbedder(_Module):
    """Per-cell embedding stage for the TabPFN-v2 architecture.

    Input contract
    --------------
    ``x``: ``(B, R, C, 2)`` float tensor. The last dim is the pair
    ``[preprocessed_value, nan_indicator]`` where ``nan_indicator`` is one of
    ``{0.0, NAN_INDICATOR, INFINITY_INDICATOR, NEG_INFINITY_INDICATOR}``.

    The ModelLoader's preprocessing pipeline is responsible for producing
    this:

      1. ``constant_feature_removal`` (fit on train rows)
      2. ``nan_indicator_extraction``  (capture BEFORE imputation)
      3. ``mean_imputation``           (fit on train rows)
      4. ``standard_scaling``          (fit on train rows)
      5. ``feature_group_normalization`` (fit over all rows)
      6. concat ``[value, nan_indicator]`` along a new last dim

    Steps 1-5 are dataset-statistic fitting and live outside the block
    library. See ``docs/tabpfn-primitives-spec.md`` for why.

    Output
    ------
    ``(B, R, num_feature_groups, d_model)`` where
    ``num_feature_groups = ceil(C / features_per_group)``. C is padded
    with zero-valued (value=0, nan_indicator=0) cells if not divisible
    by ``features_per_group`` so the grouping stays uniform.

    What the block does
    -------------------
    1. **Pad + reshape**: groups every ``features_per_group`` consecutive
       features into one "feature group" along the C axis. Upstream
       rationale: attention scales badly in C, so reducing C by 2× (the
       default group size) is a free win.
    2. **Linear project**: ``(features_per_group × ENCODING_SIZE_MULTIPLIER
       → d_model)``, no bias. Xavier-uniform init matching upstream's
       attention-projection init style.
    3. **Per-column positional embedding**: random source vectors of
       size ``d_model // 4`` are generated per-forward from a seeded
       ``torch.Generator`` (so the PE is deterministic given the seed
       but adapts to a variable column count), then projected to
       ``d_model`` via a learned ``Linear``. Added to each cell.

    Notes
    -----
    - The PE projection (``self._pe_projector``) IS learnable. The random
      source vectors are seed-deterministic but not parameters of the
      model.
    - ``seed=0`` matches ``TabPFNV2Config.seed``; the production
      checkpoint upstream uses ``seed=42`` — pass that explicitly when
      loading a paper-pinned checkpoint.
    - ``d_model`` must be divisible by 4 for the PE subspace projection.
    """

    def __init__(
        self,
        d_model: int = 192,
        features_per_group: int = 2,
        seed: int = 0,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "tabular_cell_embedder requires torch. "
                "Install with: pip install pfnstudio-core[torch]"
            ) from e
        super().__init__()

        if d_model % 4 != 0:
            raise ValueError(
                "d_model must be divisible by 4 for the PE subspace "
                f"projection (d_model // 4 → d_model); got d_model={d_model}."
            )
        if features_per_group < 1:
            raise ValueError(f"features_per_group must be ≥ 1; got {features_per_group}.")

        self.d_model = d_model
        self.features_per_group = features_per_group
        self.seed = seed

        # Linear projecting [v, nan_ind] concat per group to d_model.
        # ENCODING_SIZE_MULTIPLIER=2 is the (value, indicator) pair per
        # feature; features_per_group is how many features in a group.
        # No bias — matches upstream feature_group_embedder (line 526).
        self._feature_group_embedder = nn.Linear(
            ENCODING_SIZE_MULTIPLIER * features_per_group,
            d_model,
            bias=False,
        )
        nn.init.xavier_uniform_(self._feature_group_embedder.weight)

        # Per-column PE projector — (d_model // 4) random source → d_model.
        # IS learnable; matches upstream's feature_positional_embedding_embeddings
        # (line 548-550). Default torch init is fine — upstream doesn't
        # override it.
        self._pe_projector = nn.Linear(d_model // 4, d_model)

    def forward(self, x: Any) -> Any:
        """Forward pass.

        Args
        ----
        x : torch.Tensor of shape ``(B, R, C, 2)``
            Preprocessed (value, NaN/Inf indicator) per cell. See
            "Input contract" in the class docstring for the
            preprocessing pipeline that produces this.

        Returns
        -------
        torch.Tensor of shape ``(B, R, num_feature_groups, d_model)``.
        """
        import torch

        if x.dim() != 4:
            raise ValueError(
                f"tabular_cell_embedder expects 4-D input (B, R, C, 2); got shape {tuple(x.shape)}."
            )
        B, R, C, two = x.shape
        if two != ENCODING_SIZE_MULTIPLIER:
            raise ValueError(
                "tabular_cell_embedder expects last dim = "
                f"{ENCODING_SIZE_MULTIPLIER} (value + nan_indicator); "
                f"got {two}."
            )

        # Pad along C if not divisible by features_per_group. Zeros are
        # safe to pad with because (value=0, nan_indicator=0) is the
        # imputed/clean cell encoding and the matching column embedding
        # is also added uniformly — padded groups become a constant
        # background the rest of the network can attend over without
        # introducing leakage.
        F = self.features_per_group
        rem = C % F
        if rem:
            pad = F - rem
            zeros = torch.zeros(
                B,
                R,
                pad,
                ENCODING_SIZE_MULTIPLIER,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, zeros], dim=2)
            C_padded = C + pad
        else:
            C_padded = C
        num_groups = C_padded // F

        # Group: (B, R, C_padded, 2) → (B, R, num_groups, 2*F).
        x_grouped = x.reshape(B, R, num_groups, F * ENCODING_SIZE_MULTIPLIER)

        # Linear project to d_model: (B, R, num_groups, d_model). Cast
        # to the projector's parameter dtype so half-precision setups
        # don't error on a dtype mismatch.
        target_dtype = self._feature_group_embedder.weight.dtype
        emb = self._feature_group_embedder(x_grouped.to(target_dtype))

        # Per-column PE: regenerate deterministically per forward.
        # Why per-forward and not at __init__:
        #   - num_groups depends on the runtime input's C, which can
        #     vary per dataset, so the PE table size is not known until
        #     forward time.
        #   - The Generator(seed) makes this deterministic; modulo the
        #     device the same seed yields the same source vectors.
        gen = torch.Generator(device=x.device).manual_seed(self.seed)
        pe_subspace = torch.randn(
            (num_groups, self.d_model // 4),
            device=x.device,
            dtype=target_dtype,
            generator=gen,
        )
        pe = self._pe_projector(pe_subspace)  # (num_groups, d_model)

        # Broadcast-add: (B, R, num_groups, d_model) + (num_groups, d_model)
        # via implicit (1, 1, num_groups, d_model).
        return emb + pe[None, None]


@register_block("along_row_attention")
class AlongRowAttention(_Module):
    """Multi-head self-attention along the **feature-group axis** of a
    (B, R, C', E) tensor — every row's feature groups attend to each
    other. Used inside ``tabpfn_block`` as the first attention sub-block.

    Mirrors upstream ``tabpfn_v2.py::AlongRowAttention`` (lines 119–148).
    There's no train/test masking and no KV cache for this axis because
    rows are independent — each row's row-attention contribution is
    computed in isolation. The B and R dims are folded into a single
    batch for the attention call.

    Input contract
    --------------
    ``x``: ``(B, R, C', E)`` — typically produced by
    ``tabular_cell_embedder``. Internally flattens to ``(B*R, C', E)``
    for the attention, then un-flattens before returning.

    Output
    ------
    ``(B, R, C', E)`` — same shape.

    Initialization (matches upstream lines 113–116)
    -----------------------------------------------
    - Q, K, V projections: Xavier-uniform, **no bias**.
    - Output projection: **zero-initialized**, **no bias**.

    The zero-init on the output projection is load-bearing: at
    initialization the whole attention contribution is exactly zero,
    so the residual path in ``tabpfn_block`` dominates and the model
    starts as the identity. This is a residual-friendly init —
    breaks symmetry through training rather than at init.

    Notes
    -----
    - Uses PyTorch's ``torch.nn.functional.scaled_dot_product_attention``
      directly (which expects ``(B, H, S, D)``), not upstream's
      ``scaled_dot_product_attention`` wrapper. Same math, simpler
      dependency surface. Modern PyTorch's SDPA handles GQA / MQA
      natively if/when we need it later.
    - ``head_dim`` defaults to ``d_model // n_heads``. Explicit
      override is supported for the cases where d_model isn't evenly
      divisible.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 6,
        head_dim: int | None = None,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "along_row_attention requires torch. "
                "Install with: pip install pfnstudio-core[torch]"
            ) from e
        super().__init__()

        if n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {n_heads}.")
        if head_dim is None:
            if d_model % n_heads != 0:
                raise ValueError(
                    f"d_model={d_model} not divisible by n_heads={n_heads}; "
                    "pass head_dim explicitly if you need a different ratio."
                )
            head_dim = d_model // n_heads
        if head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {head_dim}.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        # No bias on any of the four projections — matches upstream
        # device_and_dtype_no_bias on lines 97–110.
        self._q_projection = nn.Linear(d_model, inner, bias=False)
        self._k_projection = nn.Linear(d_model, inner, bias=False)
        self._v_projection = nn.Linear(d_model, inner, bias=False)
        self._out_projection = nn.Linear(inner, d_model, bias=False)

        # Q, K, V: Xavier-uniform (upstream lines 113–115). Output:
        # zero-init (line 116) — residual-friendly start.
        nn.init.xavier_uniform_(self._q_projection.weight)
        nn.init.xavier_uniform_(self._k_projection.weight)
        nn.init.xavier_uniform_(self._v_projection.weight)
        nn.init.zeros_(self._out_projection.weight)

    def forward(self, x: Any) -> Any:
        """Forward.

        Args
        ----
        x : torch.Tensor of shape ``(B, R, C', E)``.

        Returns
        -------
        torch.Tensor of shape ``(B, R, C', E)``.
        """
        import torch  # noqa: F401  — used implicitly via the .reshape methods

        if x.dim() != 4:
            raise ValueError(
                f"along_row_attention expects 4-D input (B, R, C', E); got shape {tuple(x.shape)}."
            )
        B, R, Cp, E = x.shape
        if self.d_model != E:
            raise ValueError(
                f"along_row_attention expects last dim = d_model = {self.d_model}; got {E}."
            )

        H, D = self.n_heads, self.head_dim
        # Fold (B, R) → BR for attention. Each row's feature groups
        # attend to each other independently of other rows.
        BR = B * R
        x_flat = x.reshape(BR, Cp, E)

        # Project to Q, K, V — each (BR, C', H*D).
        q = self._q_projection(x_flat)
        k = self._k_projection(x_flat)
        v = self._v_projection(x_flat)

        # Reshape for SDPA: (BR, C', H, D) → (BR, H, C', D).
        # PyTorch's F.scaled_dot_product_attention expects (B, H, S, D).
        q = q.view(BR, Cp, H, D).transpose(1, 2)
        k = k.view(BR, Cp, H, D).transpose(1, 2)
        v = v.view(BR, Cp, H, D).transpose(1, 2)

        # No mask, no causal — all features attend to each other.
        # SDPA returns (BR, H, C', D). _sdpa_no_flash avoids the
        # Flash Attention backend whose 65k gridDim.x ceiling we hit
        # at Marathon's batch_size=256 (see helper docstring).
        attn = _sdpa_no_flash(q, k, v, attn_mask=None, is_causal=False)

        # Back to (BR, C', H*D) and project.
        attn = attn.transpose(1, 2).contiguous().view(BR, Cp, H * D)
        out_flat = self._out_projection(attn)

        return out_flat.view(B, R, Cp, E)


# ── Cache type alias ─────────────────────────────────────────────────
# A KV cache is a (K, V) tuple of tensors, each of shape
# (B, C', N_train, 1, D), where the last-but-one dim is 1 because
# AlongColumnAttention caches ONLY the first attention head (multi-query
# attention pattern — test rows share a single K/V head). Stored as a
# plain tuple to avoid coupling the block library to a custom dataclass.
ColumnKVCache = tuple  # actually tuple[torch.Tensor, torch.Tensor] at runtime


@register_block("along_column_attention")
class AlongColumnAttention(_Module):
    """Multi-head self-attention along the **row axis** of a (B, C', R, E)
    tensor, with three mechanisms beyond standard self-attention:

      1. **Train/test mask via ``single_eval_pos``.** Train rows
         (positions ``[0, single_eval_pos)``) attend to each other.
         Test rows (positions ``[single_eval_pos, R)``) attend ONLY to
         train rows — never to themselves, never to other test rows.
         This is the in-context learning contract: test rows query the
         train context.

      2. **Multi-query attention for test rows.** Train rows use full
         multi-head attention (``H`` independent K/V heads). Test rows
         share a SINGLE K/V head (the first one), letting many test
         queries reuse the same projected keys/values. This is what
         keeps the KV cache small enough to be useful.

      3. **KV cache.** When ``return_kv=True`` is passed and no
         ``cached_kv`` is provided, the block returns the projected
         (K, V) of the train rows (first head only). When ``cached_kv``
         is provided, the K/V projections are skipped entirely — every
         input row is treated as a test row attending to the cached
         train context.

    Mirrors upstream ``tabpfn_v2.py::AlongColumnAttention`` (lines
    151–245). The Q/K/V/output projections share initialization with
    ``AlongRowAttention`` (Xavier on Q/K/V, zero on output).

    Input contract
    --------------
    ``x``: ``(B, C', R, E)``. The R axis is the train+test row stack.
    Internally flattens to ``(B*C', R, E)`` for attention.

    Output
    ------
    ``(out, kv_entry)`` tuple:
      - ``out``: ``(B, C', R, E)`` — same shape as input.
      - ``kv_entry``: ``None`` unless ``return_kv=True`` AND
        ``cached_kv`` was not provided, in which case it's a
        ``(K, V)`` tuple of two tensors, each of shape
        ``(B, C', N_train, 1, D)`` — only the first attention head.

    Three execution paths
    ---------------------
    a. ``cached_kv is not None``: every input row is a test row.
       Q from all rows; K/V come from the cache. SDPA in MQA mode
       (single K/V head, multiple Q heads — torch's ``enable_gqa=True``).

    b. ``single_eval_pos is None`` or ``single_eval_pos == R``: every
       input row is a train row. Standard multi-head self-attention.

    c. ``single_eval_pos`` strictly between 0 and R: mixed. Train
       queries do full MHA over train K/V; test queries do MQA over
       the first K/V head of train. Concat along R for the output.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 6,
        head_dim: int | None = None,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "along_column_attention requires torch. "
                "Install with: pip install pfnstudio-core[torch]"
            ) from e
        super().__init__()

        if n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {n_heads}.")
        if head_dim is None:
            if d_model % n_heads != 0:
                raise ValueError(
                    f"d_model={d_model} not divisible by n_heads={n_heads}; "
                    "pass head_dim explicitly if you need a different ratio."
                )
            head_dim = d_model // n_heads
        if head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {head_dim}.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        # Same projection shape + init as AlongRowAttention — both
        # inherit from upstream's ``Attention`` base class. No bias on
        # any of the four projections.
        self._q_projection = nn.Linear(d_model, inner, bias=False)
        self._k_projection = nn.Linear(d_model, inner, bias=False)
        self._v_projection = nn.Linear(d_model, inner, bias=False)
        self._out_projection = nn.Linear(inner, d_model, bias=False)

        nn.init.xavier_uniform_(self._q_projection.weight)
        nn.init.xavier_uniform_(self._k_projection.weight)
        nn.init.xavier_uniform_(self._v_projection.weight)
        nn.init.zeros_(self._out_projection.weight)

    def forward(
        self,
        x: Any,
        *,
        single_eval_pos: int | None = None,
        cached_kv: Any = None,
        return_kv: bool = False,
    ) -> Any:
        """Forward.

        Args
        ----
        x : torch.Tensor of shape ``(B, C', R, E)``.
        single_eval_pos : int | None
            Boundary between train rows (positions ``[0, single_eval_pos)``)
            and test rows (positions ``[single_eval_pos, R)``). If None,
            every row is treated as a train row. Ignored when
            ``cached_kv`` is provided (every row is then a test row).
        cached_kv : tuple[Tensor, Tensor] | None
            Pre-computed (K, V) of train rows from a previous forward
            pass. When provided: K/V projections are skipped; ``x``
            is treated as test-only rows querying the cache.
        return_kv : bool
            If True (and no cache was passed), also return the (K, V)
            of train rows for later cache use.

        Returns
        -------
        ``(out, kv_entry)`` where ``out`` has shape ``(B, C', R, E)`` and
        ``kv_entry`` is ``None`` unless ``return_kv`` triggered cache
        construction.
        """
        import torch

        if x.dim() != 4:
            raise ValueError(
                f"along_column_attention expects 4-D input (B, C', R, E); "
                f"got shape {tuple(x.shape)}."
            )
        B, Cp, R, E = x.shape
        if self.d_model != E:
            raise ValueError(
                f"along_column_attention expects last dim = d_model = {self.d_model}; got {E}."
            )

        H, D = self.n_heads, self.head_dim
        Bc = B * Cp
        x_flat = x.reshape(Bc, R, E)

        # Q is always projected from all rows in the input.
        q = self._q_projection(x_flat).view(Bc, R, H, D)
        # PyTorch SDPA expects (B, H, S, D); we transpose once.
        q_BHSD = q.transpose(1, 2)

        kv_entry = None

        if cached_kv is not None:
            # ── Path (a) — cached inference: every input row is a test
            # row attending to the cached train K/V via single-head MQA.
            k_BCNHD, v_BCNHD = cached_kv  # each (B, C', N_train, 1, D)
            # Fold (B, C') → Bc to match q_BHSD's batch.
            k_BcNHD = k_BCNHD.reshape(Bc, k_BCNHD.shape[2], 1, D)
            v_BcNHD = v_BCNHD.reshape(Bc, v_BCNHD.shape[2], 1, D)
            k_BHSD = k_BcNHD.transpose(1, 2)  # (Bc, 1, N_train, D)
            v_BHSD = v_BcNHD.transpose(1, 2)
            # GQA: q has H heads, k/v has 1 — PyTorch broadcasts internally.
            attn_BHSD = _sdpa_no_flash(
                q_BHSD,
                k_BHSD,
                v_BHSD,
                enable_gqa=True,
            )
            attn_BSHD = attn_BHSD.transpose(1, 2)  # (Bc, R, H, D)

        else:
            # K and V are projected from train rows only.
            N = R if single_eval_pos is None else single_eval_pos
            if not (0 <= N <= R):
                raise ValueError(f"single_eval_pos must be in [0, R={R}]; got {single_eval_pos}.")

            x_train = x_flat[:, :N]
            k_train = self._k_projection(x_train).view(Bc, N, H, D)
            v_train = self._v_projection(x_train).view(Bc, N, H, D)

            if single_eval_pos is None or single_eval_pos == R:
                # ── Path (b) — all train rows: standard multi-head
                # self-attention. No mask needed; all rows attend.
                k_BHSD = k_train.transpose(1, 2)  # (Bc, H, N, D)
                v_BHSD = v_train.transpose(1, 2)
                attn_BHSD = _sdpa_no_flash(
                    q_BHSD,
                    k_BHSD,
                    v_BHSD,
                )
                attn_BSHD = attn_BHSD.transpose(1, 2)

            else:
                # ── Path (c) — mixed train+test. Train queries (first
                # N positions) do full MHA over train K/V; test queries
                # (last M=R-N positions) do single-head MQA over the
                # first K/V head of train. Concat along the R axis.

                # Train queries → full MHA.
                attn_train_BHSD = _sdpa_no_flash(
                    q_BHSD[:, :, :N],
                    k_train.transpose(1, 2),
                    v_train.transpose(1, 2),
                )  # (Bc, H, N, D)

                # Test queries → MQA against the FIRST K/V head only.
                # k_train[:, :, :1] keeps shape (Bc, N, 1, D) → after
                # transpose: (Bc, 1, N, D). enable_gqa lets H query heads
                # share the single K/V head.
                attn_test_BHSD = _sdpa_no_flash(
                    q_BHSD[:, :, N:],
                    k_train[:, :, :1].transpose(1, 2),
                    v_train[:, :, :1].transpose(1, 2),
                    enable_gqa=True,
                )  # (Bc, H, R-N, D)

                attn_BHSD = torch.cat([attn_train_BHSD, attn_test_BHSD], dim=2)
                attn_BSHD = attn_BHSD.transpose(1, 2)

            if return_kv:
                # Cache only the first head — that's what test rows can
                # attend to. Reshape back to (B, C', N, 1, D) so the
                # caller gets a shape that matches the block's I/O
                # convention.
                k_cache = k_train[:, :, :1].contiguous().detach().view(B, Cp, N, 1, D)
                v_cache = v_train[:, :, :1].contiguous().detach().view(B, Cp, N, 1, D)
                kv_entry = (k_cache, v_cache)

        # Reshape (Bc, R, H, D) → (Bc, R, H*D) and project.
        attn_flat = attn_BSHD.contiguous().view(Bc, R, H * D)
        out_flat = self._out_projection(attn_flat)

        return out_flat.view(B, Cp, R, E), kv_entry


@register_block("axial_attention_block")
class AxialAttentionBlock(_Module):
    """Composed transformer-style layer that applies attention along
    one axis at a time (row-axis, then column-axis), followed by an
    MLP. Pattern known as **axial attention** (Ho et al. 2019, *Axial
    Attention in Multidimensional Transformers*).

    Used as the main per-layer building block of TabPFN-v2 — a stack
    of ~12 of these is the transformer trunk. The TabPFN-v2 paper
    naming was ``TabPFNBlock``; we call it ``axial_attention_block``
    because the *pattern* is generic. Future axial-attention papers
    can compose this same block, configured differently.

    Sub-modules
    -----------
    - ``_row_attn``  : :class:`AlongRowAttention` — attention along C'
    - ``_col_attn``  : :class:`AlongColumnAttention` — attention along R
    - ``_ln_row``    : LayerNorm after the row attention
    - ``_ln_col``    : LayerNorm after the column attention
    - ``_ln_mlp``    : LayerNorm after the MLP
    - ``_mlp``       : 2-layer position-wise feedforward

    Forward (post-norm, residual)
    -----------------------------
    For input ``x`` of shape ``(B, R, C', E)``::

        # Row attention sub-block (no train/test mask, no cache).
        x = LayerNorm(x + row_attn(x))

        # Column attention sub-block (passes single_eval_pos, cached_kv,
        # return_kv straight through). Requires transpose to (B, C', R, E).
        x_BCRE = x.transpose(1, 2).contiguous()
        col_out, kv_entry = col_attn(x_BCRE, ...)
        x_BCRE = LayerNorm(x_BCRE + col_out)
        x = x_BCRE.transpose(1, 2).contiguous()

        # Position-wise MLP sub-block.
        x = LayerNorm(x + mlp(x))

    Config knobs (all default to the TabPFN-v2 paper's exact settings)
    -----------------------------------------------------------------
    - ``norm_position``: ``'post'`` (TabPFN-v2 default) | ``'pre'``.
      Pre-norm wraps the sub-block input in LayerNorm BEFORE the
      sub-block call; post-norm wraps the residual sum AFTER.
    - ``norm_affine``: ``False`` (TabPFN-v2 default) | ``True``.
      Whether LayerNorm carries learnable scale/bias parameters. The
      TabPFN-v2 ``LowerPrecisionLayerNorm`` uses ``False``.
    - ``mlp_init``: ``'zero_second_linear'`` (TabPFN-v2 default) |
      ``'default'``. Zero-initializing the MLP's second linear is a
      residual-friendly trick that makes the block start as the
      identity — gradients break symmetry during training.
    - ``mlp_activation``: ``'gelu'`` (TabPFN-v2 default) | ``'relu'``.
    - ``ff_mult``: ``4`` (TabPFN-v2 default). MLP hidden width =
      ``ff_mult * d_model``.

    Notes
    -----
    - The block returns ``(out, kv_entry)`` matching
      :class:`AlongColumnAttention`'s API. The row attention and MLP
      don't have a KV-cache concept; only the column attention does.
    - ``norm_position='post'`` matches TabPFN-v2; the existing
      ``transformer_encoder`` block uses pre-norm (``norm_first=True``).
      Same word, different default — this block is post-norm by default.
    """

    # Trainer/predict-loop opt-in: any block declaring this attribute as
    # True receives ``single_eval_pos=n_ctx`` as a kwarg when composed in
    # a Model. Mirrors the existing ``tag_embedder`` precedent in
    # training/loop.py — declarative dispatch, no global kwargs.
    needs_single_eval_pos: bool = True

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 6,
        head_dim: int | None = None,
        ff_mult: int = 4,
        norm_position: str = "post",
        norm_affine: bool = False,
        mlp_init: str = "zero_second_linear",
        mlp_activation: str = "gelu",
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "axial_attention_block requires torch. "
                "Install with: pip install pfnstudio-core[torch]"
            ) from e
        super().__init__()

        if norm_position not in ("pre", "post"):
            raise ValueError(f"norm_position must be 'pre' or 'post'; got {norm_position!r}.")
        if mlp_init not in ("default", "zero_second_linear"):
            raise ValueError(
                f"mlp_init must be 'default' or 'zero_second_linear'; got {mlp_init!r}."
            )
        if mlp_activation not in ("gelu", "relu"):
            raise ValueError(f"mlp_activation must be 'gelu' or 'relu'; got {mlp_activation!r}.")
        if ff_mult < 1:
            raise ValueError(f"ff_mult must be ≥ 1; got {ff_mult}.")

        self.d_model = d_model
        self.norm_position = norm_position

        # Attention sub-blocks — reuse the primitives we already
        # tested individually. As nn.Module subclasses they nest
        # cleanly into this block's parameter discovery.
        self._row_attn = AlongRowAttention(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        self._col_attn = AlongColumnAttention(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
        )

        # Three post- (or pre-) norms. ``elementwise_affine=False``
        # matches upstream's LowerPrecisionLayerNorm (line 308). We
        # don't reproduce the lower-precision casting detail — it's
        # an fp16/bf16 stability trick that's invisible in fp32 and
        # adds complexity not relevant to the architecture's identity.
        self._ln_row = nn.LayerNorm(d_model, elementwise_affine=norm_affine)
        self._ln_col = nn.LayerNorm(d_model, elementwise_affine=norm_affine)
        self._ln_mlp = nn.LayerNorm(d_model, elementwise_affine=norm_affine)

        # Position-wise MLP — 2-layer, no bias, configurable activation.
        # Upstream sets bias=False on both linears (line 314-316).
        ff_dim = ff_mult * d_model
        activation: Any
        if mlp_activation == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.ReLU()
        self._mlp_linear1 = nn.Linear(d_model, ff_dim, bias=False)
        self._mlp_act = activation
        self._mlp_linear2 = nn.Linear(ff_dim, d_model, bias=False)

        if mlp_init == "zero_second_linear":
            # The second linear is zero-initialized → MLP contributes
            # exactly zero at init → block behaves as identity until
            # trained. Same residual-friendly trick as the attention
            # out_projections.
            nn.init.zeros_(self._mlp_linear2.weight)

    def forward(
        self,
        x: Any,
        *,
        single_eval_pos: int | None = None,
        cached_kv: Any = None,
        return_kv: bool = False,
    ) -> Any:
        """Forward.

        Args
        ----
        x : torch.Tensor of shape ``(B, R, C', E)``.
        single_eval_pos, cached_kv, return_kv :
            Passed through to the column attention sub-block. See
            :class:`AlongColumnAttention` for semantics.

        Returns
        -------
        ``(out, kv_entry)`` where ``out`` has shape ``(B, R, C', E)``
        and ``kv_entry`` is whatever the column attention sub-block
        returned.
        """
        # ── 1. Row attention sub-block.
        if self.norm_position == "pre":
            row_out = self._row_attn(self._ln_row(x))
            x = x + row_out
        else:  # post
            row_out = self._row_attn(x)
            x = self._ln_row(x + row_out)

        # ── 2. Column attention sub-block. Operates on (B, C', R, E),
        # so transpose before the call and back after.
        x_BCRE = x.transpose(1, 2).contiguous()
        if self.norm_position == "pre":
            col_in = self._ln_col(x_BCRE)
            col_out, kv_entry = self._col_attn(
                col_in,
                single_eval_pos=single_eval_pos,
                cached_kv=cached_kv,
                return_kv=return_kv,
            )
            x_BCRE = x_BCRE + col_out
        else:  # post
            col_out, kv_entry = self._col_attn(
                x_BCRE,
                single_eval_pos=single_eval_pos,
                cached_kv=cached_kv,
                return_kv=return_kv,
            )
            x_BCRE = self._ln_col(x_BCRE + col_out)
        x = x_BCRE.transpose(1, 2).contiguous()

        # ── 3. Position-wise MLP sub-block.
        if self.norm_position == "pre":
            mlp_out = self._mlp_linear2(self._mlp_act(self._mlp_linear1(self._ln_mlp(x))))
            x = x + mlp_out
        else:  # post
            mlp_out = self._mlp_linear2(self._mlp_act(self._mlp_linear1(x)))
            x = self._ln_mlp(x + mlp_out)

        return x, kv_entry


@register_block("row_pool_for_head")
class RowPoolForHead(_Module):
    """Collapse the cell-table representation ``(B, R, C', E)`` down to
    a per-row representation ``(B, R, E)`` by selecting a single
    feature-group column.

    The last block in the TabPFN-v2 architecture before the output
    head. The output head wants per-row embeddings, not per-cell; this
    block picks one specific column's E-dim representation as that
    per-row embedding.

    Why "select", not "mean" or "max": for Do-PFN, the model is trained
    to put the predicted outcome distribution into the **treatment
    column's cell** (column 0 per the paper's "first column is
    treatment" convention). Pooling by mean or max would dilute the
    signal across columns; the trained model encodes its prediction
    in one specific position.

    Other pooling strategies (mean across all columns, max-pool,
    attention-pool) are out of scope for this block — they'd be
    different primitives (``row_mean_pool``, etc.).

    Input contract
    --------------
    ``x``: ``(B, R, C', E)``. Typically the output of the final
    ``axial_attention_block``.

    Output
    ------
    ``(B, R, E)`` — per-row embedding.

    Config
    ------
    ``target_col``: which column index (along C') to select. Default
    ``0`` matches Do-PFN's treatment-column convention. Out-of-range
    indices raise at forward time so the model spec is debuggable.
    """

    def __init__(self, target_col: int = 0):
        try:
            import torch  # noqa: F401  — required for nn.Module super
        except ImportError as e:
            raise ImportError(
                "row_pool_for_head requires torch. Install with: pip install pfnstudio-core[torch]"
            ) from e
        super().__init__()

        if target_col < 0:
            raise ValueError(
                f"target_col must be ≥ 0; got {target_col}. Negative "
                "indexing is intentionally rejected so the model spec "
                "is unambiguous."
            )
        self.target_col = target_col

    def forward(self, x: Any) -> Any:
        """Forward.

        Args
        ----
        x : torch.Tensor of shape ``(B, R, C', E)``.

        Returns
        -------
        torch.Tensor of shape ``(B, R, E)``.
        """
        if x.dim() != 4:
            raise ValueError(
                f"row_pool_for_head expects 4-D input (B, R, C', E); got shape {tuple(x.shape)}."
            )
        Cp = x.shape[2]
        if self.target_col >= Cp:
            raise ValueError(
                f"target_col={self.target_col} is out of range for input "
                f"with C'={Cp} feature groups. Check the model spec."
            )
        # Select the target column's cell stack: (B, R, E).
        return x[:, :, self.target_col, :]


@register_block("grid_preprocessor")
class GridPreprocessor(_Module):
    """Adapter block: raw ``(B, R, C)`` → cell-embedder-ready ``(B, R, C, 2)``.

    Wraps :class:`pfnstudio_core._grid_preprocessing.TabularPreprocessor`
    in ``nn.Module`` form so it can sit at the front of a composed-by-blocks
    TabPFN-family model. Stateless across forward passes — each call fits
    its own train-row stats and applies them, matching the in-context
    learning contract (train rows fit the scaler, test rows transform with
    the same stats).

    Trainer integration
    -------------------
    Declares ``needs_single_eval_pos = True``; the trainer's forward loop
    threads ``single_eval_pos=n_ctx`` into the block's forward kwargs.
    See ``packages/core/pfnstudio_core/training/loop.py`` for the dispatch.

    Why this is a block
    -------------------
    Without it, every TabPFN-family template would need to hand-build a
    ``(B, R, C, 2)`` tensor in its prior or a one-off model wrapper. The
    block makes preprocessing first-class in the composed-from-blocks
    pipeline — drop it at the front of any model.yaml that feeds raw
    floats into a TabPFN-style cell embedder.

    Pipeline
    --------
    1. Extract NaN / +Inf / −Inf indicators (BEFORE imputation).
    2. Mean-impute NaN cells using the train-row mean.
    3. Standard-scale per-feature using train-row mean + std.
    4. Stack ``[value, indicator]`` along a new last dim.

    Input / output
    --------------
    - Input  ``x``: ``(B, R, C)`` float tensor. May contain NaN / Inf.
    - Output:      ``(B, R, C, 2)`` — last dim is
                   ``[preprocessed_value, nan_indicator]``.

    The output shape exactly matches the input contract of
    :class:`TabularCellEmbedder` so the two compose without glue code.
    """

    # Trainer/predict-loop opt-in — see AxialAttentionBlock's comment.
    needs_single_eval_pos: bool = True

    def __init__(self) -> None:
        try:
            import torch.nn as nn  # noqa: F401 -- verifies the optional dependency
        except ImportError as e:  # pragma: no cover — torch is a required dep
            raise ImportError(
                "grid_preprocessor requires torch. Install with: pip install pfnstudio-core[torch]"
            ) from e
        super().__init__()

    def forward(self, x: Any, *, single_eval_pos: int | None = None) -> Any:
        """Run the preprocessing pipeline.

        Args
        ----
        x : ``(B, R, C)`` torch.Tensor. May contain NaN / Inf.
        single_eval_pos : int | None
            Boundary index — rows ``[0, single_eval_pos)`` are train
            (used to fit the scaler), rows ``[single_eval_pos, R)`` are
            test (transformed only). Threaded by the trainer/predict
            loop when this block declares ``needs_single_eval_pos``.

        Returns
        -------
        ``(B, R, C, 2)`` torch.Tensor ready for ``tabular_cell_embedder``.
        """
        # Import here so module-level loading doesn't require torch.
        from .._grid_preprocessing import TabularPreprocessor

        if single_eval_pos is None or single_eval_pos <= 0:
            raise ValueError(
                "grid_preprocessor requires single_eval_pos > 0 — the "
                "boundary between train and test rows. The trainer threads "
                "this from batch['n_ctx']; if you're seeing this error, "
                "either the prior didn't emit n_ctx or the model is being "
                "invoked outside the trainer/predict path."
            )
        preprocessor = TabularPreprocessor()
        return preprocessor.fit_transform(x, n_train=int(single_eval_pos))
