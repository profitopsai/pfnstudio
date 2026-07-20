"""Minimal PFN training loop.

This is *the* canonical PFN loop: each step, sample a task from the prior,
split into context / query, predict the query from the context.

It's intentionally framework-agnostic at the boundary: the model is anything
callable; the prior returns numpy arrays; the loss is supplied by the caller.
The default loss assumes a discovery-style adjacency target and uses BCE.

JSON-line progress
──────────────────
When the env var ``PFNSTUDIO_JSON_PROGRESS=1`` is set, every step prints one
line of JSON to stdout (line-buffered). The PFN Studio API parses these lines
to drive the live training UI; running in a normal terminal still works (the
output just looks a bit verbose).

Event schema (one JSON object per line):
    { "event": "start",  "steps": int, "batch_size": int, "lr": float, ... }
    { "event": "step",   "step": int, "loss": float, "elapsed_s": float }
    { "event": "done",   "status": "ok"|"skipped"|"failed", "final_loss": float, ... }
    { "event": "error",  "message": str }
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from ..axes import encode_tag, get_axis, sample_tag
from ..axes.detector import AxisDetector, train_detector
from ..axes.honoring import compute_axis_honoring
from ..prior import Prior
from ..run import RunSpec

_EMIT_JSON = os.environ.get("PFNSTUDIO_JSON_PROGRESS") == "1"

# Reserved key the trainer uses to stash the encoded tag vector into
# the first sample of each batch. step_fn pulls it out and forwards
# the tensor to any encoder block that declares a tag_embedder. Kept
# as a module-level constant so step_fn and the loop body agree on
# the name without an extra import path.
_BATCH_TAG_KEY = "__pfnstudio_batch_tag__"

# The trainer stashes the resolved torch.device here on the hyperparams dict so
# _default_step can move batch tensors to the same device the model is on.
_HP_DEVICE_KEY = "__pfnstudio_device__"


def _emit(event: str, **fields: Any) -> None:
    """Print one JSON-progress line to stdout, line-buffered, when enabled."""
    if not _EMIT_JSON:
        return
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


# Block-type names that act as output heads — they consume the
# encoder's pooled representation and emit a task-specific tensor.
# Multi-head models stack one of these per skill the user picked, all
# fed from the same encoder output. The training step splits modules
# into encoder + heads using this set so two scalar_heads (e.g.
# `number` + `classification` skills) don't crash on dim mismatch.
#
# Keep in sync with pfnstudio_core.blocks.heads — adding a new head
# block requires adding its registered name here.
_HEAD_BLOCK_TYPE_NAMES: set[str] = {"discovery_head", "estimation_head", "scalar_head"}


def _is_head_module(mod: Any) -> bool:
    """True iff `mod` is an output head (vs encoder block).

    Identified by a STRICT class-name allowlist plus the ``is_head = True``
    duck-type (which a project head declares). An earlier
    ``__name__.endswith("head")`` heuristic over-matched: it classified
    pooling stages like ``RowPoolForHead`` — which feed INTO a head but are
    NOT heads — as parallel-fan-out heads, so a model with a pool before its
    head assembled wrong (the head ran on the pre-pool encoder output,
    surfacing as tensor-size mismatches).
    """
    return type(mod).__name__ in {"DiscoveryHead", "EstimationHead", "ScalarHead"} or bool(
        getattr(mod, "is_head", False)
    )


def _split_encoder_heads(modules: list[tuple[str, Any]]) -> tuple[list[Any], list[Any]]:
    """Split the module chain at the first head block.

    Returns (encoder_blocks, head_blocks). Heads-at-end is the
    convention (the wizard's buildModel + every paper-pinned template
    follows it); blocks after the first head are still treated as
    heads and run in parallel on the encoder output.
    """
    encoder: list[Any] = []
    heads: list[Any] = []
    in_heads = False
    for _, mod in modules:
        if in_heads or _is_head_module(mod):
            in_heads = True
            heads.append(mod)
        else:
            encoder.append(mod)
    return encoder, heads


def _model_tabular_d_in(model: Any) -> int | None:
    """Input width from a tabular embedder's ``_linear.in_features``.

    Only inspects blocks whose class name is ``TabularEmbedder``.
    Other blocks (e.g. an MLP) also expose a ``d_in`` attribute for
    *their* internal width — scanning those produced false positives
    in some model topologies.
    """
    widths: list[int] = []
    for _, mod in getattr(model, "modules", []):
        if type(mod).__name__ != "TabularEmbedder":
            continue
        linear = getattr(mod, "_linear", None)
        if linear is None:
            continue
        in_features = getattr(linear, "in_features", None)
        if in_features is not None and int(in_features) > 0:
            widths.append(int(in_features))
    return min(widths) if widths else None


def _default_step(model: Any, batch: list[dict], hp: dict) -> Any:
    """Default step. Auto-detects the prior's task by inspecting batch
    keys and picks an appropriate loss:

    - ``A`` present       → discovery / structure: BCE-with-logits
                            against the ground-truth adjacency.
    - ``labels`` present  → classification: BCE-with-logits against
                            0/1 labels (binary). The model emits one
                            logit per point.
    - ``y`` present       → regression: MSE between model output and
                            ``y`` (broadcast to the model's output
                            shape).
    - none of the above   → no-op skip; the loop reports this as a
                            structural mismatch so the operator can
                            fix the prior or provide a custom step_fn.

    Multi-head models (e.g. wizard brains where the user picked two
    skills that map to the same head type) are handled by running the
    encoder once and then each head in parallel on the encoder
    output. We pick whichever head's output shape matches the
    target's expected shape and compute the loss against it; unused
    heads still see gradient through their parameters because they
    run on the same encoder graph, but their loss contribution is
    zero — so the optimiser focuses on the chosen-head path.

    Falls back to a no-op if torch is unavailable.
    """
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812 — F is the canonical alias for torch.nn.functional
    except ImportError:
        return None

    if not batch:
        return None
    sample0 = batch[0]
    has_A = "A" in sample0
    has_labels = "labels" in sample0
    has_y = "y" in sample0
    if not has_A and not has_labels and not has_y:
        # Prior didn't emit a recognised target. The loop turns this
        # into a {"status": "skipped", "reason": "step_fn returned None"}
        # which the run-detail page surfaces — caller can write a
        # project-specific step_fn for novel target shapes.
        return None

    # The trainer stashed its resolved device on hp; move every tensor we
    # build here onto it so the forward runs on the GPU when there is one.
    device = hp.get(_HP_DEVICE_KEY) if isinstance(hp, dict) else None

    X = torch.stack([torch.from_numpy(b["X"]).float() for b in batch])
    if device is not None:
        X = X.to(device)

    # Promptable training: if the trainer stashed an encoded tag in
    # the batch (see _BATCH_TAG_KEY at the bottom of this file), turn
    # it into a (B, tag_dim) tensor and pass it to any encoder block
    # that knows about tags. Blocks identify themselves by having a
    # non-None ``tag_embedder`` attribute — keeps the dispatch loose
    # and avoids forcing every block to accept a ``tag`` kwarg.
    tag_np = batch[0].get(_BATCH_TAG_KEY)
    if tag_np is not None:
        tag_tensor = torch.from_numpy(tag_np).float().unsqueeze(0).repeat(len(batch), 1)
        if device is not None:
            tag_tensor = tag_tensor.to(device)
    else:
        tag_tensor = None

    # Split into encoder + heads so multi-skill brains don't chain
    # head outputs (which crashes because head_2's Linear(d_model, 1)
    # gets head_1's (..., 1) output instead of the encoder's
    # (..., d_model) output).
    modules = list(getattr(model, "modules", []))
    encoder, heads = _split_encoder_heads(modules)

    # Run the encoder forward sequentially. Tag-aware blocks (those with a
    # tag_embedder) receive the tag; n_ctx-aware blocks (grid_preprocessor,
    # axial_attention_block — those declaring ``needs_single_eval_pos``)
    # receive single_eval_pos, the train/test boundary from the prior's
    # n_ctx. Some blocks return (out, kv_entry) tuples for KV-cache
    # compatibility — unwrap the tensor for sequential composition.
    n_ctx_fwd = int(sample0.get("n_ctx") or 0)
    enc_out = X
    for mod in encoder:
        kwargs: dict[str, Any] = {}
        if tag_tensor is not None and getattr(mod, "tag_embedder", None) is not None:
            kwargs["tag"] = tag_tensor
        if n_ctx_fwd > 0 and getattr(mod, "needs_single_eval_pos", False):
            kwargs["single_eval_pos"] = n_ctx_fwd
        result = mod(enc_out, **kwargs) if kwargs else mod(enc_out)
        enc_out = result[0] if isinstance(result, tuple) else result

    # Run each head on the encoder output. If there are no head blocks
    # at all (older wizard brains without a skill head), treat the
    # encoder output itself as the only candidate "head output".
    head_outputs: list[Any] = [head(enc_out) for head in heads] if heads else [enc_out]

    def _pick_head_for_shape(target_shape_tail: tuple[int, ...]) -> Any | None:
        """Pick the first head output whose trailing dims match the
        target. Returns None if no head produces the right shape."""
        for ho in head_outputs:
            if ho.shape[-len(target_shape_tail) :] == target_shape_tail:
                return ho
        return None

    if has_A:
        A = torch.stack([torch.from_numpy(b["A"]).float() for b in batch])
        if device is not None:
            A = A.to(device)
        # Discovery target is (B, V, V) — pick the head whose trailing
        # dims match. For a wizard brain with a stray scalar_head, the
        # discovery_head's (V, V) output is the right pick.
        pred = _pick_head_for_shape(tuple(A.shape[-2:]))
        if pred is None:
            return None
        return F.binary_cross_entropy_with_logits(pred, A)

    if has_labels:
        # Classification branch. Targets are 0/1 per point; a
        # scalar_head with d_out=1 produces (B, N, 1) logits which we
        # squeeze to (B, N) before scoring.
        labels = torch.stack([torch.from_numpy(b["labels"]).float() for b in batch])
        if device is not None:
            labels = labels.to(device)
        n_ctx_c = sample0.get("n_ctx")
        # Find the first head whose output, after the optional context
        # slice + trailing-1 squeeze, lines up with the labels shape.
        for ho in head_outputs:
            pred = ho[:, int(n_ctx_c) :, :] if n_ctx_c is not None else ho
            if pred.dim() == labels.dim() + 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            if pred.shape == labels.shape:
                return F.binary_cross_entropy_with_logits(pred, labels)
        return None

    # Regression branch
    y = torch.stack([torch.from_numpy(b["y"]).float() for b in batch])
    if device is not None:
        y = y.to(device)
    n_ctx = sample0.get("n_ctx")

    # Generic custom-loss hook. A head that trains with its own objective
    # (e.g. a bar-distribution head's bucketized NLL over per-bucket logits)
    # exposes loss(query_output, target). It gets its query-span output and
    # the targets, flattened across the batch (the head reshapes rows
    # internally). Duck-typed + OPTIONAL: heads without it fall through to MSE.
    nq = int(n_ctx) if n_ctx is not None else 0
    for head, ho in zip(heads, head_outputs, strict=False):
        _loss = getattr(head, "loss", None)
        if callable(_loss):
            q = ho[:, nq:, :]  # (B, n_qry, out_dim)
            return _loss(q.reshape(-1, q.shape[-1]), y.reshape(-1))

    for ho in head_outputs:
        pred = ho[:, int(n_ctx) :, :] if n_ctx is not None else ho
        if pred.dim() == y.dim() + 1 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if pred.shape == y.shape:
            return F.mse_loss(pred, y)
    return None


def train_pfn(
    model: Any,
    prior: Prior,
    run: RunSpec,
    *,
    step_fn: Callable[[Any, list[dict], dict], Any] | None = None,
    on_step: Callable[[int, float], None] | None = None,
    eval_fn: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the PFN training loop in-process. Returns a results dict.

    Set ``PFNSTUDIO_JSON_PROGRESS=1`` in the env to emit one JSON line per
    step to stdout (consumed by the PFN Studio API to drive the live UI).

    Periodic eval: when ``eval_fn`` is provided AND ``hyperparams.eval_every``
    is a positive integer, the trainer calls ``eval_fn(step)`` every N steps
    (and once before training starts as a step-0 baseline). The callback
    should return a dict of ``slug → {metrics, meta, skipped, skip_reason}``;
    the trainer emits one ``eval`` event per slug per cycle. Callback runs
    in-process — eval cost adds to wall clock — so callers should set a
    cadence that's coarser than the total step budget.
    """
    try:
        import torch
    except ImportError:
        result = {
            "status": "skipped",
            "reason": "torch not installed; install pfnstudio-core[torch] to train",
            "steps": 0,
        }
        _emit("done", **result)
        return result

    hp = run.hyperparams

    # The training loop's canonical hyperparam is `steps` — total gradient
    # updates. But the run-composer presets express the same idea as
    # `epochs` (with `batch_size` already a separate field). Honour both:
    # if the user gave only `epochs`, treat each "epoch" as one training
    # step. This isn't true epoch semantics (we don't iterate a dataset),
    # but for PFN-style on-the-fly sampling it's the same wall-clock unit
    # — and it matches what the UI shows ("running 200 epochs" lands as
    # 200 progress bar ticks instead of silently degrading to 100).
    #
    # If the user gave both, `steps` wins (more explicit). If neither,
    # default to 100 to match the historical behaviour.
    if "steps" in hp:
        steps = int(hp["steps"])
    elif "epochs" in hp:
        steps = int(hp["epochs"])
    else:
        steps = 100
    batch_size = int(hp.get("batch_size", 8))
    lr = float(hp.get("lr", 1e-4))
    seed = int(hp.get("seed", 42))
    # Periodic eval cadence — 0 disables (only end-of-run eval, the
    # historic behaviour). When > 0, the adapter's `eval_fn` is called
    # every N steps; results stream as `eval` events for the live UI.
    eval_every = int(hp.get("eval_every", 0))

    # Seed *all* sources of randomness as early as possible. This makes
    # same-machine, same-version runs byte-identical:
    #   - Python's built-in `random` — some libraries (e.g. dataloader
    #     shuffles, scorer sampling) reach for it transparently.
    #   - torch    → controls Linear / TransformerEncoder default init
    #                AND LazyLinear's first-forward init AND any other
    #                stochastic torch ops.
    #   - numpy    → priors use np.random.default_rng(seed) per-call
    #                (already deterministic), but seeding global numpy
    #                here protects any scorer / user code that uses
    #                global np.random.
    #   - PYTHONHASHSEED — set in the parent process before fork so
    #     dict iteration / set ordering is stable across runs.
    # Cross-machine / cross-version determinism is *not* guaranteed —
    # see study READMEs' "Reproducibility" section.
    import os
    import random as _random

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:
        pass

    # Pin deterministic CPU/GPU kernels for paper-reproducible runs.
    # `use_deterministic_algorithms(True)` makes torch raise when it
    # would otherwise dispatch to a non-deterministic kernel; pair
    # with `warn_only=True` so a kernel that lacks a deterministic
    # path doesn't kill the run — we warn and continue. The cudnn
    # flags pin GPU-side determinism for kernels that do have one.
    # All wrapped in try/except so older torch versions (where these
    # APIs vary) don't break training.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass

    _emit(
        "start",
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        prior_id=run.prior.id,
        model_id=run.model.id,
    )

    # Collect trainable parameters across every block. Blocks have three
    # legitimate shapes today:
    #   1. block is itself an nn.Module        → mod.parameters()
    #   2. block holds an nn.Module at .module → mod.module.parameters()
    #   3. block holds one or more nn.Modules at other attribute names
    #      (e.g. TabularEmbedder._linear, ScalarHead.proj). Walk the
    #      instance __dict__ to find them. Without this, those weights
    #      stay frozen at random init and only the transformer trains —
    #      which silently works for some easy regression cases and fails
    #      hard on classification.
    nn_mod = getattr(torch, "nn", None)
    nn_module_cls = getattr(nn_mod, "Module", None) if nn_mod else None

    def _block_nn_modules(b: Any) -> list[Any]:
        if nn_module_cls is not None and isinstance(b, nn_module_cls):
            return [b]
        seen: list[Any] = []
        attached = getattr(b, "module", None)
        if nn_module_cls is not None and isinstance(attached, nn_module_cls):
            seen.append(attached)
        for v in vars(b).values():
            if nn_module_cls is not None and isinstance(v, nn_module_cls) and v not in seen:
                seen.append(v)
        return seen

    def _block_nn_named_modules(b: Any) -> list:
        """(attr_name, nn.Module) pairs — the named companion of
        ``_block_nn_modules``. A block with >1 nn.Module needs its submodules
        namespaced by attribute in the checkpoint, otherwise colliding param
        keys (two Sequentials each with "0.weight") overwrite each other at
        save time. Single-module blocks keep the legacy flat keying.
        """
        if nn_module_cls is not None and isinstance(b, nn_module_cls):
            return [("", b)]
        out: list = []
        attached = getattr(b, "module", None)
        if nn_module_cls is not None and isinstance(attached, nn_module_cls):
            out.append(("module", attached))
        for attr, v in vars(b).items():
            if (
                nn_module_cls is not None
                and isinstance(v, nn_module_cls)
                and all(v is not m for _, m in out)
            ):
                out.append((attr, v))
        return out

    # Pick the training device BEFORE collecting parameters — moving submodules
    # after the optimizer captured their tensors would strand its state on the
    # old device. Resolution: PFNSTUDIO_DEVICE env override → CUDA if available
    # → CPU. On a GPU box this is what lets a heavy model (e.g. a 12-layer
    # axial-attention transformer) train in VRAM instead of exhausting host RAM.
    device_override = os.environ.get("PFNSTUDIO_DEVICE", "").strip()
    if device_override:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    for _, _dev_mod in getattr(model, "modules", []):
        for _sub in _block_nn_modules(_dev_mod):
            _sub.to(device)
    _emit("log", line=f"training device: {device}")
    if isinstance(hp, dict):
        hp[_HP_DEVICE_KEY] = device

    params = []
    for _, mod in getattr(model, "modules", []):
        for sub in _block_nn_modules(mod):
            params.extend(sub.parameters())
    if not params:
        result = {"status": "skipped", "reason": "no trainable parameters found", "steps": 0}
        _emit("done", **result)
        return result

    optim = torch.optim.AdamW(params, lr=lr)
    step_fn = step_fn or _default_step

    # Generic pre-training setup hook. A block that must prepare before the
    # first forward pass — e.g. a bar-distribution head computing bucket
    # borders from the prior — exposes setup(*, prior, hp, device). Duck-typed
    # and OPTIONAL: blocks without it are untouched (training on CPU here, so
    # device is None; the block keeps its tensors on CPU).
    for _, _setup_mod in getattr(model, "modules", []):
        _setup = getattr(_setup_mod, "setup", None)
        if callable(_setup):
            _setup(prior=prior, hp=hp, device=device)

    # Local closure: invoke the adapter's eval callback if one was passed
    # and we're configured to eval periodically. Stays a no-op when
    # `eval_fn` is None (CLI usage, tests) or when eval_every <= 0
    # (default — no periodic eval).
    #
    # We swallow exceptions per-eval so a buggy scorer can't kill the
    # training run mid-way. Each crash surfaces as a `skipped` payload
    # the UI can render.
    def _emit_eval_cycle(at_step: int) -> None:
        if eval_fn is None or eval_every <= 0:
            return
        try:
            cycle_results = eval_fn(at_step) or {}
        except Exception as e:  # pragma: no cover — defensive
            _emit("log", line=f"eval_fn raised at step {at_step}: {type(e).__name__}: {e}")
            return
        for slug, payload in cycle_results.items():
            # Each payload is the ScorerResult-shaped dict the adapter
            # already builds for end-of-run evals; pass it through.
            _emit(
                "eval",
                slug=slug,
                step=at_step,
                metrics=payload.get("metrics") or {},
                meta=payload.get("meta") or {},
                skipped=bool(payload.get("skipped")),
                skip_reason=payload.get("skip_reason"),
            )

    losses: list[float] = []
    t0 = time.time()
    prior_params = {**run.prior.overrides}

    # Promptable-priors tag sampling. **Opt-in** via the run's
    # ``promptable_training`` hyperparam — without it, the trainer
    # passes ``tag=None`` every batch, which Prior.sample_batch ignores
    # for axis-less priors and treats as UNKNOWN for axis-aware ones
    # (byte-identical to the pre-axis path either way).
    #
    # Why opt-in: a prior that declares axes still trains today on the
    # marginal data distribution (mixture across tag values). The
    # existing model can't condition on a tag — so feeding it
    # axis-conditioned data only shifts the marginal it learns,
    # silently regressing benchmarks calibrated against the pre-axis
    # distribution. Opt-in is enabled in session 4 alongside the
    # tag-input model pathway; until then the safe default is None.
    promptable = bool(hp.get("promptable_training", False))
    axis_objs = []
    if promptable:
        for axis_name in getattr(prior, "axes", []) or []:
            try:
                axis_objs.append(get_axis(axis_name))
            except KeyError:
                # An unknown axis name on the prior is a configuration
                # error worth flagging — surface as a log event but
                # don't kill the run; the prior may still function
                # with tag=None.
                _emit(
                    "log",
                    line=f"prior declares unknown axis {axis_name!r}; skipping",
                )

    import numpy as _np_for_tags  # local import keeps the top-level import light

    _tag_rng = _np_for_tags.random.default_rng(seed)

    def _sample_batch_tag() -> dict[str, Any] | None:
        """Sample a tag dict for the current batch, honoring each axis's
        unknown_mass. Returns None when promptable training is disabled
        or the prior has no axes — call sites unconditionally pass the
        result and existing call paths stay byte-identical."""
        if not axis_objs:
            return None
        return sample_tag(axis_objs, _tag_rng)

    # Step-0 baseline: score the untrained (random-init) model so the
    # live scorecard can show "before training" → "after step N" → …
    # without a gap. Cheap to run; especially useful as a comparison
    # point when checking whether training is actually helping.
    if eval_every > 0 and eval_fn is not None:
        _emit_eval_cycle(0)

    for step in range(steps):
        # Disjoint seed range per step. The previous formula `seed=seed+step`
        # made consecutive steps share batch_size-1 seeds (sample_batch
        # iterates seed+0..seed+batch_size-1), so each task appeared in
        # ~batch_size consecutive batches and then never again. Recency
        # overfitting followed: late-step seeds fit, early-step seeds
        # forgotten. Multiplying by batch_size makes each task seen exactly
        # once across training, which is what the PFN setup assumes.
        # Sample a fresh tag per batch. Returns None when the prior
        # declares no axes — in which case `sample_batch(tag=None, ...)`
        # is byte-identical to `sample_batch(...)` (Prior.sample_batch's
        # default ignores tag for axis-less priors).
        batch_tag = _sample_batch_tag()
        batch = prior.sample_batch(
            batch_size=batch_size,
            seed=seed + step * batch_size,
            tag=batch_tag,
            **prior_params,
        )
        # Encode the sampled tag into a fixed-length vector and stash
        # it on batch[0] for step_fn to read. Skipped when the tag is
        # None — step_fn falls back to the regular (non-tag) forward
        # pass, identical to the pre-axis training path.
        if batch_tag is not None and axis_objs and batch:
            batch[0][_BATCH_TAG_KEY] = encode_tag(batch_tag, axis_objs)
        loss = step_fn(model, batch, hp)
        if loss is None:
            result = {"status": "skipped", "reason": "step_fn returned None", "steps": step}
            _emit("done", **result)
            return result

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        elapsed = time.time() - t0
        _emit("step", step=step, loss=loss_val, elapsed_s=elapsed)
        if on_step:
            on_step(step, loss_val)

        # Periodic eval: after every Nth step. We use `(step + 1)`
        # because `step` is 0-indexed — the user mental model is
        # "eval every 100 steps" meaning at steps 100, 200, etc.,
        # not at 99, 199. Skip when step+1 hits `steps` (the final
        # step) because the end-of-run eval in the adapter will cover
        # that case with no extra cost.
        if eval_every > 0 and eval_fn is not None:
            done = step + 1
            if done % eval_every == 0 and done < steps:
                _emit_eval_cycle(done)

    # Persist the trained model so /runs/:id/predict can load it.
    # Two files, both under <cwd>/checkpoint/:
    #   model.pt      — state_dict from the trainable modules
    #   topology.json — the block list + per-block init config, so
    #                   the inference path can reconstruct the
    #                   model object before loading state_dict
    # This is per-run-cwd; the API runner copies the directory out
    # to stable storage after the CLI exits.
    checkpoint_dir: str | None = None
    checkpoint_save_error: str | None = None
    try:
        import json
        import os

        ckpt_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save weights — flat dict keyed by "<block_name>.<param>"
        # so the inference loader knows which weights go where even
        # if blocks change order. Uses the same walker as the optimizer
        # so anything that trains gets persisted.
        sd: dict[str, Any] = {}
        param_count = 0
        for name, mod in getattr(model, "modules", []):
            named = _block_nn_named_modules(mod)
            multi = len(named) > 1
            for attr, sub in named:
                # Namespace each submodule by attr ONLY when the block has more
                # than one nn.Module — keeps single-module blocks on the exact
                # legacy keying so existing checkpoints still bind.
                prefix = f"{name}.{attr}." if (multi and attr) else f"{name}."
                for k, v in sub.state_dict().items():
                    sd[f"{prefix}{k}"] = v
                    param_count += 1
        if param_count == 0:
            # Empty state_dict means _block_nn_modules didn't find any
            # nn.Module inside the blocks. A loaded checkpoint with no
            # weights would silently produce a model with random init,
            # so refuse to claim success.
            raise RuntimeError(
                "no nn.Module weights found across model blocks — "
                "_block_nn_modules walker returned empty. Blocks may be "
                "wrapping their torch modules in an attribute the walker "
                "doesn't recognise."
            )
        torch.save(sd, os.path.join(ckpt_dir, "model.pt"))
        # Topology: the same list the trainer iterated, in order.
        # Stored as JSON so non-Python tooling can introspect it.
        topology = {
            "model_id": run.model.id,
            "prior_id": run.prior.id,
            "blocks": [
                {"name": name, "type": type(mod).__name__}
                for name, mod in getattr(model, "modules", [])
            ],
        }
        with open(os.path.join(ckpt_dir, "topology.json"), "w") as fh:
            json.dump(topology, fh, indent=2)
        checkpoint_dir = ckpt_dir
    except Exception as e:
        # Don't fail the run if checkpoint save fails — the metrics
        # are still useful. Surface the reason on the done event so the
        # brain page can show *why* Try-it is disabled rather than the
        # generic "no usable checkpoint" message.
        checkpoint_dir = None
        checkpoint_save_error = f"{type(e).__name__}: {e}"
        _emit("log", line=f"checkpoint save failed: {checkpoint_save_error}")

    # Detect the model's input feature count from the first 2-D weight
    # tensor (e.g. a TabularEmbedder._linear with weight shape
    # (d_model, d_in)). Surfaced on the done event so the API can
    # store it on Run.results and the Try-it form can pre-size its
    # textarea inputs — otherwise the user has to discover d_in by
    # triggering a predict failure that reports the dimension.
    d_in: int | None = None
    for _, mod in getattr(model, "modules", []):
        if d_in is not None:
            break
        for sub in _block_nn_modules(mod):
            w = getattr(sub, "weight", None)
            if w is not None and w.dim() >= 2:
                d_in = int(w.shape[-1])
                break

    # Emit both first-10pct and last-10pct loss means so the brain page
    # can grade *how well it learnt* (first / last = a layman loss-drop
    # ratio) separately from *how well it predicts* (the Test-your-brain
    # chart's accuracy grade). One number on its own ("final loss
    # 0.4582") doesn't tell the user whether training did anything.
    result = {
        "status": "ok",
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_first_10pct": (
            sum(losses[: max(1, steps // 10)]) / max(1, steps // 10) if losses else None
        ),
        "mean_loss_last_10pct": (
            sum(losses[-max(1, steps // 10) :]) / max(1, steps // 10) if losses else None
        ),
        "wall_time_s": time.time() - t0,
        "checkpoint_dir": checkpoint_dir,
        "d_in": d_in,
    }
    if checkpoint_save_error is not None:
        result["checkpoint_save_error"] = checkpoint_save_error

    # Promptable training: at end-of-training, measure whether the
    # model actually honors each axis. This is the load-bearing
    # check that catches the silent-failure mode where the model
    # ignores the tag (predictions are unchanged when the chip flips).
    # Skipped cleanly when promptable mode is off or the model has
    # no tag-aware blocks — no extra cost on regular runs.
    if axis_objs:
        try:
            honoring = compute_axis_honoring(
                model=model,
                prior=prior,
                axes=axis_objs,
                seed=seed,
            )
            if honoring:
                result["axis_honoring"] = honoring
                for axis_name, axis_score in honoring.items():
                    _emit(
                        "axis_honoring",
                        axis=axis_name,
                        divergence_mean=axis_score["divergence_mean"],
                        divergence_max=axis_score["divergence_max"],
                        values_compared=list(axis_score["values_compared"]),
                    )
        except Exception as e:  # pragma: no cover — defensive
            _emit(
                "log",
                line=f"honoring computation failed: {type(e).__name__}: {e}",
            )

        # Companion detector — small, fast (≤ ~30s on CPU). Lets the
        # Brain page pre-fill chips from the user's data instead of
        # leaving them blank. Saved beside the main checkpoint as
        # detector.pt; if training fails the brain still works (the
        # user just doesn't get pre-fill).
        if checkpoint_dir is not None:
            try:
                detector_result = _train_companion_detector(
                    prior=prior,
                    axes=axis_objs,
                    checkpoint_dir=checkpoint_dir,
                    seed=seed,
                )
                if detector_result is not None:
                    result["detector"] = detector_result
                    _emit("detector_trained", **detector_result)
            except Exception as e:  # pragma: no cover — defensive
                _emit(
                    "log",
                    line=f"companion detector training failed: {type(e).__name__}: {e}",
                )

    _emit("done", **result)
    return result


def _train_companion_detector(
    *,
    prior: Any,
    axes: list[Any],
    checkpoint_dir: str,
    seed: int,
) -> dict[str, Any] | None:
    """Train a small AxisDetector on the same prior and persist its
    weights to ``checkpoint_dir/detector.pt``. Returns a summary dict
    with status, final loss, and last-batch accuracy per axis — or
    None when the detector has no scored axes (nothing to predict).

    Sized small on purpose: 32-d × 2 layers × 300 steps so the extra
    train time is minimal on top of the main run. Calibration polish
    (more steps, better head) is a future-session lever."""
    try:
        import os as _os

        import torch as _torch
    except ImportError:
        return None

    detector = AxisDetector(axes=axes, d_model=32, n_heads=4, n_layers=2)
    if not detector.scored_axes:
        return None

    train_result = train_detector(
        detector=detector,
        prior=prior,
        steps=300,
        batch_size=16,
        lr=1e-3,
        seed=seed + 1,  # different seed-stream than the brain
    )
    if train_result.get("status") != "ok":
        return train_result

    detector_path = _os.path.join(checkpoint_dir, "detector.pt")
    # Persist the detector's state-dict in a self-describing layout
    # so the predict path can reconstruct the AxisDetector before
    # loading the weights.
    payload = {
        "axes": [a.name for a in detector.scored_axes],
        "d_model": detector.d_model,
        "state": {
            "embedder": detector.embedder.state_dict(),
            "encoder": detector.encoder.state_dict(),
            "heads": detector.heads.state_dict(),
        },
    }
    _torch.save(payload, detector_path)
    return {
        "status": "ok",
        "path": detector_path,
        "steps": train_result.get("steps"),
        "final_loss": train_result.get("final_loss"),
        "last_batch_accuracy": train_result.get("last_batch_accuracy"),
    }
