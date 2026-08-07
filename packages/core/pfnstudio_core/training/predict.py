"""Inference path for trained PFN checkpoints.

Mirrors the local adapter's submit() shape — load the project from
its YAML files, instantiate the same Model object the trainer used,
restore weights from the checkpoint, then run a forward pass on the
operator-supplied input.

The input payload shape is prior-task-dependent. We auto-detect
based on the keys present:

- Regression PFN (linear-regression style):
    payload = {
      "context": {"x": [[..],..], "y": [..]},
      "query":   {"x": [[..],..]}
    }
    output  = {"predictions": [..], "task": "regression"}

- Discovery PFN (adjacency target):
    payload = {"X": [[..],..]}
    output  = {"adjacency": [[..],..], "task": "discovery"}

Anything outside these shapes returns ``{"error": "..."}`` rather
than a guess — the operator is the one closest to the prior's data
contract and should be the one to surface a clear failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _checkpoint_d_in(checkpoint_dir: Path) -> int | None:
    """Read persisted tabular width from topology.json (newer checkpoints)."""
    topo = checkpoint_dir / "topology.json"
    if not topo.is_file():
        return None
    try:
        data = json.loads(topo.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    d = data.get("d_in")
    return int(d) if isinstance(d, int) and d > 0 else None


def _block_nn_modules(b: Any) -> list:
    """Walk a block to find every torch.nn.Module sub-module attached to it.

    Different blocks expose their nn.Module under different attr names
    (``module``, ``_linear``, ...). Hard-coding ``.module`` broke
    TabularEmbedder (its module is ``._linear``), surfacing as
    ``AttributeError: 'TabularEmbedder' object has no attribute
    'load_state_dict'``. Walking ``vars()`` finds every nn.Module inside
    the block; ``load_state_dict(strict=False)`` then matches whatever
    keys belong to each.
    """
    import torch.nn as _nn

    if isinstance(b, _nn.Module):
        return [b]
    seen: list = []
    attached = getattr(b, "module", None)
    if isinstance(attached, _nn.Module):
        seen.append(attached)
    for v in vars(b).values():
        if isinstance(v, _nn.Module) and v not in seen:
            seen.append(v)
    return seen


def _block_nn_named_modules(b: Any) -> list:
    """Like ``_block_nn_modules`` but pairs each nn.Module with the attribute
    name it is stored under.

    A block that holds MORE THAN ONE nn.Module — e.g. a gated residual with
    both a ``mlp`` and a ``gate`` ``nn.Sequential`` — needs its submodules
    namespaced in the checkpoint. Both Sequentials expose a ``0.weight`` key,
    so keyed under only the bare block name they collide and silently
    overwrite each other at save time (the reload then feeds ``gate``'s
    ``0.weight`` into ``mlp`` → a size-mismatch RuntimeError). The attribute
    name is that per-submodule namespace. Single-module blocks (every
    built-in) keep the legacy flat keying, so no existing checkpoint changes.
    """
    import torch.nn as _nn

    if isinstance(b, _nn.Module):
        return [("", b)]
    out: list = []
    attached = getattr(b, "module", None)
    if isinstance(attached, _nn.Module):
        out.append(("module", attached))
    for attr, v in vars(b).items():
        if isinstance(v, _nn.Module) and all(v is not m for _, m in out):
            out.append((attr, v))
    return out


class ModelLoader:
    """Hold a trained PFN checkpoint in memory, ready to serve many
    ``predict()`` calls without re-reading the run / model yaml /
    weights from disk on every call.

    Two usage patterns:

    1. **One-shot** (the existing ``/runs/:id/predict`` path) — used as
       a context manager so sys.path mutation is cleaned up::

           with ModelLoader(
               manifest_path=…, project_root=…, checkpoint_dir=…
           ) as loader:
               result = loader.predict(payload, tag=tag, detect=False)

    2. **Long-lived** (the forthcoming ``pfnstudio serve`` worker, Phase 1)
       — instantiate once at process start, call ``predict`` per request,
       call ``close()`` at shutdown::

           loader = ModelLoader(...)
           try:
               for req in incoming:
                   yield loader.predict(req.payload, tag=req.tag)
           finally:
               loader.close()

    The class owns ``sys.path`` mutation for the duration of its life —
    the project root has to stay importable for the prior/model classes
    to resolve. ``close()`` removes the entry; ``__init__`` cleans up on
    construction failure as well, so a half-built loader can't leak path
    state.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        project_root: Path,
        checkpoint_dir: Path,
    ) -> None:
        import torch

        from pfnstudio_core.loaders import load_model, load_run
        from pfnstudio_core.model import Model
        from pfnstudio_core.registry import discover_in_project

        from .loop import _model_tabular_d_in

        # Same project bootstrap as the local trainer adapter.
        self._project_root_str: str | None = str(project_root)
        # Kept for native schema-driven execution ({task, inputs}) + the
        # capabilities ping: both read the declarative model/prior manifest.
        self._manifest_path: Path = manifest_path
        self._project_root: Path = project_root
        sys.path.insert(0, self._project_root_str)

        # If anything below raises, restore sys.path before propagating —
        # otherwise the caller's interpreter is left with the project
        # root permanently on the path.
        try:
            discover_in_project(project_root)
            self.run = load_run(manifest_path)

            model_yaml = project_root / "models" / f"{self.run.model.id}.yaml"
            if not model_yaml.exists():
                raise FileNotFoundError(f"model.yaml not found at {model_yaml}")
            model_spec = load_model(model_yaml)
            self.model = Model(model_spec)

            # Restore weights. Trainer wrote a flat dict keyed by
            # "<block_name>.<param>"; redistribute back to each block's
            # state_dict so blocks added/reordered since training still
            # bind correctly.
            ckpt_path = checkpoint_dir / "model.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")
            flat = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            for name, mod in getattr(self.model, "modules", []):
                block_sd = {
                    k[len(name) + 1 :]: v for k, v in flat.items() if k.startswith(name + ".")
                }
                if not block_sd:
                    continue
                named = _block_nn_named_modules(mod)
                multi = len(named) > 1
                for attr, sub in named:
                    if multi and attr:
                        # Namespaced checkpoint: pull only this submodule's
                        # keys (e.g. "mlp.*") so a sibling's colliding
                        # "0.weight" can't be loaded into the wrong module.
                        sub_sd = {
                            k[len(attr) + 1 :]: v
                            for k, v in block_sd.items()
                            if k.startswith(attr + ".")
                        }
                        if not sub_sd and any(True for _ in sub.parameters()):
                            raise RuntimeError(
                                f"Checkpoint for block '{name}' predates the "
                                f"multi-submodule fix — no namespaced weights "
                                f"for '{attr}'. Re-run training to produce a "
                                f"compatible checkpoint."
                            )
                    else:
                        sub_sd = block_sd
                    # strict=False ignores keys that don't belong to this
                    # sub-module — supports blocks with multiple nn.Modules.
                    sub.load_state_dict(sub_sd, strict=False)
                    sub.eval()

            # Tabular embedder width (packed-token d_in), not an
            # arbitrary encoder weight matrix — scanning the first 2-D
            # weight picks d_model and breaks for some prior+model
            # combinations whose first weight isn't the input projection.
            d_in: int | None = _model_tabular_d_in(self.model)
            if d_in is None:
                d_in = _checkpoint_d_in(checkpoint_dir)
            self.d_in: int | None = d_in
            self.checkpoint_dir: Path = checkpoint_dir
        except BaseException:
            self.close()
            raise

    def predict(
        self,
        payload: dict,
        *,
        tag: dict[str, Any] | None = None,
        detect: bool = False,
    ) -> dict:
        """Run one forward pass on the loaded model.

        ``tag`` (optional) is a dict ``{axis_name: value}`` for promptable
        priors. When the prior declares axes and tag is provided, the
        tag is encoded into a fixed-length vector and routed to any
        tag-aware block in the model. Brains trained without
        ``promptable_training`` silently ignore the tag.

        ``detect`` (optional, default False) — when True and the run's
        checkpoint dir contains a ``detector.pt`` from a companion-trained
        AxisDetector, the predict response includes a ``detected_tag``
        field with the detector's per-axis proposal + confidence.

        Raises on payload shape mismatches; callers (the CLI predict
        command, the ``/runs/:id/predict`` endpoint, the forthcoming
        worker process) wrap into their preferred error envelope.
        """
        # Native capabilities ping — a cheap manifest read from the model/prior
        # YAML that never touches the weights. Lets a NATIVE studio model (no
        # adapter) self-describe its Try-it/Inference contract, same as the
        # adapter path does.
        if isinstance(payload, dict) and payload.get("__capabilities__"):
            return _native_capabilities(
                manifest_path=self._manifest_path, project_root=self._project_root
            )

        # Native schema-driven task ({task, inputs}) — the envelope the Try-it /
        # Inference surfaces post. Handled here so BOTH the CLI run_inference
        # path and the runner's serve handler execute custom tasks (e.g. RCA)
        # identically. The legacy raw {context,query}/{X} payloads fall through.
        if isinstance(payload, dict) and payload.get("task") and "inputs" in payload:
            return self._predict_native_task(payload)

        if self.d_in is not None:
            _validate_input_dim(payload, self.d_in)

        # Resolve tag → encoded tensor lazily. We need the prior's
        # declared axes; look them up via the registry. If the prior
        # has no axes (or tag is None) we skip the encoding and the
        # dispatch helpers run their non-tag path.
        tag_tensor = _resolve_tag_tensor(self.run, tag)

        out = _dispatch_inference(
            self.model, payload, expected_d_in=self.d_in, tag_tensor=tag_tensor
        )

        # Auto-detector proposal — only if the caller asked for it
        # AND the run's checkpoint actually ships a detector. The
        # response gets a ``detected_tag`` field shaped as
        # ``{axis_name: {value, confidence, probs}}``.
        if detect:
            try:
                detected = _run_detector(
                    run=self.run,
                    checkpoint_dir=self.checkpoint_dir,
                    payload=payload,
                    expected_d_in=self.d_in,
                )
                if detected:
                    out["detected_tag"] = detected
            except Exception as e:  # pragma: no cover — defensive
                out["detect_warning"] = f"{type(e).__name__}: {e}"

        return out

    def _predict_native_task(self, payload: dict) -> dict:
        """Execute a schema-driven {task, inputs} predict against this loaded
        native studio model.

        Resolves the declared task spec from the model/prior capabilities
        manifest, asks the run's prior to pack `inputs` into the model tensor
        (``prior.pack_inference`` — the model-specific bit stays project-owned),
        runs the encoder→head forward, and shapes the head output per the
        task's ``output_schema.kind``. Returns the raw result dict (or
        ``{"error": ...}``); never raises for user-data problems so both the
        CLI and the serve handler surface a clean message.
        """
        import numpy as np
        import torch

        from ..registry import get_prior

        task = payload.get("task")
        inputs = payload.get("inputs")
        if not isinstance(task, str) or not task:
            return {"error": "predict envelope is missing a 'task'."}
        if not isinstance(inputs, dict):
            return {"error": "predict envelope is missing an 'inputs' object."}

        caps = _resolve_native_manifest(self._manifest_path, self._project_root)
        spec = caps.get(task) if isinstance(caps, dict) else None
        if not isinstance(spec, dict):
            avail = sorted(caps) if isinstance(caps, dict) else []
            return {"error": f"this model declares no task '{task}'. Available: {avail}."}

        try:
            prior_cls = get_prior(self.run.prior.id)
        except KeyError:
            return {"error": "the run's prior isn't available, so native inference can't pack its input."}
        packer = getattr(prior_cls(), "pack_inference", None)
        if not callable(packer):
            return {
                "error": (
                    f"the '{self.run.prior.id}' prior declares no pack_inference(), so the "
                    f"'{task}' task can't run natively. Add pack_inference() to the prior, or "
                    "serve this model through an adapter."
                )
            }

        # The model's trained node capacity (the encoder asserts cols ==
        # 3*k_max); read it off the loaded model so the packer can't drift
        # from the weights. Fall back to d_in/3.
        k_max = next(
            (int(getattr(m, "k_max")) for _, m in getattr(self.model, "modules", []) if hasattr(m, "k_max")),
            None,
        )
        if k_max is None and self.d_in:
            k_max = int(self.d_in) // 3

        try:
            packed = packer(task=task, inputs=inputs, k_max=k_max, spec=spec)
        except Exception as e:  # noqa: BLE001 — user-data shape errors are expected
            return {"error": f"couldn't build the model input for '{task}' — {type(e).__name__}: {e}."}

        X = np.asarray(packed["X"], dtype=np.float32)
        sep = int(packed.get("single_eval_pos") or 0)
        node_names = list(packed.get("node_names") or [])

        xt = torch.from_numpy(X).float()
        if xt.dim() == 2:
            xt = xt.unsqueeze(0)
        with torch.no_grad():
            head_out = _forward_encoder_and_pick_head(
                self.model, xt, (), single_eval_pos=sep
            )
            if head_out is None:
                return {"error": "the model produced no head output for this task."}
            to_pred = _head_to_prediction(self.model)
            pred = to_pred(head_out) if to_pred else head_out

        result = _shape_native_output(spec, pred, node_names)
        result["task"] = task
        return result

    def close(self) -> None:
        """Remove this loader's project root from ``sys.path``. Idempotent —
        safe to call twice (no-ops the second time) and from ``__init__``
        failure paths where the entry may or may not have been inserted.
        """
        root = getattr(self, "_project_root_str", None)
        if root and root in sys.path:
            sys.path.remove(root)
        self._project_root_str = None

    def __enter__(self) -> ModelLoader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def run_inference(
    *,
    manifest_path: Path,
    project_root: Path,
    checkpoint_dir: Path,
    payload: dict,
    tag: dict[str, Any] | None = None,
    detect: bool = False,
) -> dict:
    """One-shot load + predict + close. Back-compat entry point.

    Kept as the call site for the existing ``/runs/:id/predict`` route
    and the CLI ``predict`` command — they pay the small re-load cost
    on every call. Long-lived serving paths (the ``pfnstudio serve``
    worker, which holds one checkpoint in memory across many predict
    requests) instantiate :class:`ModelLoader` directly and reuse it.
    """
    # Adapter dispatch: a run that continued-pretrained an external base model
    # (hyperparams.adapter, or baseAdapter on the installed base run) predicts
    # through THAT model's own architecture + real weights, not the studio
    # block model. Generic — one branch for every adapter. Also answers the
    # __capabilities__ manifest cheaply, before any weights load.
    adapter_name = _peek_run_adapter(manifest_path)
    if adapter_name:
        return _run_adapter_inference(
            adapter_name=adapter_name,
            manifest_path=manifest_path,
            project_root=project_root,
            checkpoint_dir=checkpoint_dir,
            payload=payload,
        )

    with ModelLoader(
        manifest_path=manifest_path,
        project_root=project_root,
        checkpoint_dir=checkpoint_dir,
    ) as loader:
        return loader.predict(payload, tag=tag, detect=detect)


def _peek_run_adapter(manifest_path: Path) -> str | None:
    """Read the run YAML's `hyperparams.adapter` (continued-pretrain run) or
    `baseAdapter` (the installed base run). Returns the adapter name to route
    inference through its own architecture, or None for a studio-model run.
    YAML-only — no project bootstrap, no RunSpec validation (base runs have
    empty prior/model refs and would fail it)."""
    try:
        import yaml
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — any read/parse issue → not an adapter run
        return None
    if not isinstance(data, dict):
        return None
    hp = data.get("hyperparams") or {}
    if not isinstance(hp, dict):
        return None
    name = hp.get("adapter") or hp.get("baseAdapter")
    return str(name) if isinstance(name, str) and name else None


def _run_adapter_inference(
    *,
    adapter_name: str,
    manifest_path: Path,
    project_root: Path,
    checkpoint_dir: Path,
    payload: dict,
) -> dict:
    """Predict through a base model's OWN architecture + real weights.

    Discovers the project's `adapters/<slug>.py`, resolves the adapter, loads
    the real checkpoint (the same `load_model` continued-pretraining uses), and
    calls the adapter's `predict()`. For a base model with no user data, the
    adapter samples a task from the model's own prior and returns predicted-vs-
    ground-truth — which both uses the real weights AND shows they work.
    """
    import torch
    import yaml

    from pfnstudio_core.registry import discover_in_project
    from pfnstudio_core.training.adapters import resolve_adapter

    discover_in_project(project_root)
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        data = {}
    hp = (data.get("hyperparams") or {}) if isinstance(data, dict) else {}
    if not isinstance(hp, dict):
        hp = {}

    try:
        adapter = resolve_adapter(adapter_name, hp.get("adapterModule"))
    except Exception as e:  # noqa: BLE001
        return {
            "engine": adapter_name,
            "command": "predict",
            "result": {"error": f"the '{adapter_name}' adapter is not available ({type(e).__name__}: {e})."},
        }

    # Capability manifest — static, weights not needed. Answered BEFORE the
    # (expensive) model load so the Try-it / Inference surfaces can fetch the
    # task schema cheaply. Adapters without capabilities() report {}.
    if isinstance(payload, dict) and payload.get("__capabilities__"):
        cap_fn = getattr(adapter, "capabilities", None)
        try:
            caps = cap_fn() if callable(cap_fn) else {}
        except Exception as e:  # noqa: BLE001
            return {
                "engine": adapter_name,
                "command": "capabilities",
                "result": {"error": f"capabilities() failed — {type(e).__name__}: {e}"},
            }
        return {
            "engine": adapter_name,
            "command": "capabilities",
            "result": {"capabilities": caps or {}},
        }

    predict_fn = getattr(adapter, "predict", None)
    if predict_fn is None:
        return {
            "engine": adapter_name,
            "command": "predict",
            "result": {"error": f"the '{adapter_name}' adapter does not implement predict() yet."},
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = adapter.load_model(Path(checkpoint_dir), hp, device)
        model.eval()
        with torch.no_grad():
            result = predict_fn(model, payload or {}, hp, device)
    except Exception as e:  # noqa: BLE001 — surface as a structured error, not a crash
        return {
            "engine": adapter_name,
            "command": "predict",
            "result": {"error": f"the '{adapter_name}' adapter failed during inference — {type(e).__name__}: {e}"},
        }

    return {"engine": adapter_name, "command": "predict", "result": result}


def _run_detector(
    *,
    run: Any,
    checkpoint_dir: Path,
    payload: dict,
    expected_d_in: int | None,
) -> dict[str, Any]:
    """Load the companion detector (if present) and run ``detect()`` on
    the payload's context. Returns the proposal dict, or empty dict
    when no detector was shipped with this run."""
    detector_path = checkpoint_dir / "detector.pt"
    if not detector_path.exists():
        return {}

    try:
        import torch

        from ..axes import AxisDetector, detect, get_axis
    except ImportError:
        return {}

    saved = torch.load(detector_path, map_location="cpu", weights_only=False)
    axis_names = list(saved.get("axes", []))
    axes = [get_axis(n) for n in axis_names]
    if not axes:
        return {}

    detector = AxisDetector(axes=axes, d_model=int(saved.get("d_model", 32)), n_heads=4, n_layers=2)
    state = saved["state"]
    detector.embedder.load_state_dict(state["embedder"])
    detector.encoder.load_state_dict(state["encoder"])
    detector.heads.load_state_dict(state["heads"])

    # Build a context tensor from the payload. For regression /
    # classification payloads, pack the context's (x, y/labels) rows
    # the same way the trainer did. For discovery, use X directly.
    ctx_tensor = _detector_context_tensor(payload, expected_d_in=expected_d_in)
    if ctx_tensor is None:
        return {}

    return detect(detector=detector, context=ctx_tensor)


def _detector_context_tensor(payload: dict, *, expected_d_in: int | None) -> Any:
    """Convert the payload's context section into a tensor the detector
    can read. Returns None when no context is available (e.g. pure
    discovery without context/query semantics)."""
    import numpy as np
    import torch

    if "context" in payload:
        ctx = payload["context"] or {}
        x_ctx = np.asarray(ctx.get("x", []), dtype=np.float32)
        if x_ctx.size == 0:
            return None
        if x_ctx.ndim == 1:
            x_ctx = x_ctx.reshape(-1, 1)
        # Build the same packed-token shape the trainer uses. Tag
        # detection doesn't need the query tokens; pass the context
        # only with the trailing 1.0 is_context_flag.
        if "y" in ctx:
            y_ctx = np.asarray(ctx["y"], dtype=np.float32)
        elif "labels" in ctx:
            y_ctx = np.asarray(ctx["labels"], dtype=np.float32)
        else:
            y_ctx = np.zeros(x_ctx.shape[0], dtype=np.float32)
        packed = np.concatenate(
            [
                x_ctx,
                y_ctx.reshape(-1, 1),
                np.ones((x_ctx.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        return torch.from_numpy(packed).float().unsqueeze(0)

    if "X" in payload:
        X = np.asarray(payload["X"], dtype=np.float32)
        return torch.from_numpy(X).float().unsqueeze(0)

    return None


def _resolve_tag_tensor(run: Any, tag: dict[str, Any] | None) -> Any:
    """Look up the prior's axes from the registry, encode the tag, and
    return a (1, tag_dim) tensor — or None if the prior has no axes
    or no tag was provided.

    Returning a (1, ...) tensor instead of (B, ...) lets the dispatch
    helpers tile it to whatever batch size the request uses (1 for
    most predict calls, but eg. honoring runs may pass larger batches)."""
    if tag is None:
        return None
    try:
        import torch

        from ..axes import encode_tag, get_axis
        from ..registry import get_prior
    except ImportError:
        return None

    prior_cls = None
    try:
        prior_cls = get_prior(run.prior.id)
    except KeyError:
        return None

    axis_names = list(getattr(prior_cls, "axes", []) or [])
    if not axis_names:
        return None
    try:
        axis_objs = [get_axis(name) for name in axis_names]
    except KeyError:
        return None

    encoded = encode_tag(tag, axis_objs)
    if encoded.size == 0:
        return None
    return torch.from_numpy(encoded).float().unsqueeze(0)  # (1, tag_dim)


def _validate_input_dim(payload: dict, expected: int) -> None:
    """Reject payloads whose `context.x` / `query.x` rows don't match
    the model's expected feature count, with a message that tells the
    user exactly what to paste. Without this check the model would
    raise a generic LazyLinear AssertionError ("in_features inferred
    from input: M is not equal to in_features from self.weight: N")
    several frames deep, which is not actionable.

    The model's expected dim depends on the prior's packing convention:

    - Packed-token regression/classification: model expects F + 2
      (features + y_or_zero + is_context_flag). Payload's `context.x`
      / `query.x` rows are F-wide; the dispatcher adds the 2 marker
      columns internally. Accept `F == expected - 2`.
    - Discovery (unpacked): model expects F = features. Payload's `X`
      rows are F-wide. Accept `F == expected`.
    """
    import numpy as np

    ctx = payload.get("context") or {}
    qry = payload.get("query") or {}
    for label, arr_raw in (("context.x", ctx.get("x")), ("query.x", qry.get("x"))):
        if arr_raw is None:
            continue
        arr = np.asarray(arr_raw, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            got = 1
        else:
            got = int(arr.shape[-1])
        if got == expected or got + 2 == expected:
            continue
        # Compute the user-facing payload width (subtract 2 if packed).
        payload_width = expected - 2 if expected >= 3 else expected
        raise ValueError(
            f"Model expects {payload_width} feature(s) per row, but {label} "
            f"has {got}. Each row should be {payload_width} numbers separated "
            f"by spaces or commas — e.g. "
            f"`{', '.join(['0.0'] * min(payload_width, 8))}"
            + (", ..." if payload_width > 8 else "")
            + "`."
        )


def _forward_encoder_and_pick_head(
    model: Any,
    x: Any,
    expected_shape_tail: tuple,
    tag_tensor: Any = None,
    single_eval_pos: int | None = None,
) -> Any:
    """Run the model's encoder blocks once, then run each head block
    in parallel on the encoder output and return the first head whose
    trailing dimensions match `expected_shape_tail`.

    Mirrors the trainer's `_split_encoder_heads` + per-branch shape
    pick in pfnstudio_core.training.loop. Without this split,
    multi-head models (e.g. a model with both regression and
    classification heads, which both append `scalar_head` blocks)
    crash at the second head — the first head's `(..., 1)` output is
    fed to a `Linear(64, 1)` that expects `(..., 64)`.

    When ``tag_tensor`` is provided, it's tiled to the input's batch
    size and routed to any encoder block exposing a non-None
    ``tag_embedder`` attribute. Same dispatch rule the training loop
    uses, so train-time and inference-time tag routing stay in sync.

    Returns None when no head produces a matching shape — caller
    surfaces that as a clear error rather than rendering garbage.
    """
    from .loop import _split_encoder_heads  # type: ignore

    modules = list(getattr(model, "modules", []))
    encoder, heads = _split_encoder_heads(modules)
    batch_tag = _tile_tag_to_batch(tag_tensor, x)
    enc_out = x
    for mod in encoder:
        kwargs: dict[str, Any] = {}
        if batch_tag is not None and getattr(mod, "tag_embedder", None) is not None:
            kwargs["tag"] = batch_tag
        # n_ctx-aware blocks (e.g. mace_delta_pool, grid_preprocessor) receive
        # the context/query boundary — same dispatch the training loop uses.
        if (
            single_eval_pos is not None
            and single_eval_pos > 0
            and getattr(mod, "needs_single_eval_pos", False)
        ):
            kwargs["single_eval_pos"] = single_eval_pos
        result = mod(enc_out, **kwargs) if kwargs else mod(enc_out)
        enc_out = result[0] if isinstance(result, tuple) else result
    head_outputs = [head(enc_out) for head in heads] if heads else [enc_out]
    if not expected_shape_tail:
        return head_outputs[0] if head_outputs else None
    for ho in head_outputs:
        if tuple(ho.shape[-len(expected_shape_tail) :]) == tuple(expected_shape_tail):
            return ho
    # No exact match — first head is the best guess (caller can
    # decide whether to surface it). Most single-head models land here.
    return head_outputs[0] if head_outputs else None


def _tile_tag_to_batch(tag_tensor: Any, x: Any) -> Any:
    """Tile a (1, tag_dim) tag tensor to (B, tag_dim) where B is the
    batch size of x. Returns None for None input so callers can use
    the same code path either way."""
    if tag_tensor is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.is_tensor(tag_tensor):
        return None
    batch_size = int(x.shape[0])
    if tag_tensor.shape[0] == batch_size:
        return tag_tensor
    return tag_tensor.repeat(batch_size, 1)


def _pack_context_query(
    x_ctx: Any,
    targets_ctx: Any,
    x_qry: Any,
) -> Any:
    """Build the packed-token sequence the trainer expects.

    Each token is `(features..., target_or_zero, is_context_flag)`:
      - context tokens: real target value, is_context = 1
      - query tokens:   target = 0,          is_context = 0

    Same packing convention pfns_classification uses + the breast_cancer
    scorer builds manually. Total per-token dim = F + 2.
    """
    import numpy as np

    n_ctx = int(x_ctx.shape[0])
    n_qry = int(x_qry.shape[0])
    ctx_tok = np.concatenate(
        [x_ctx, targets_ctx.reshape(-1, 1), np.ones((n_ctx, 1), dtype=np.float32)],
        axis=1,
    )
    q_tok = np.concatenate(
        [
            x_qry,
            np.zeros((n_qry, 1), dtype=np.float32),
            np.zeros((n_qry, 1), dtype=np.float32),
        ],
        axis=1,
    )
    return np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)


def _head_to_prediction(model: Any):
    """Return the model's head-level ``to_prediction`` reducer, or None.

    A distributional head (e.g. a bar-distribution head that outputs
    per-bucket logits) collapses its output to a point estimate via a
    ``to_prediction(output) -> (..., 1)`` method. This finds the first
    block exposing one so the regression predict path can apply it before
    slicing the scalar channel. Generic: core names no specific head.
    """
    for _, mod in getattr(model, "modules", []):
        fn = getattr(mod, "to_prediction", None)
        if callable(fn):
            return fn
    return None


def _forward_full_sequence(
    model: Any, seq: Any, tag_tensor: Any = None, single_eval_pos: int | None = None
) -> Any:
    """Run the model on a packed sequence, returning the head output.

    Mirrors the training-time forward pass: encoder once, then each
    head on the encoder output (single-head models are the common case;
    multi-head models pick the first head whose trailing dim is 1).

    When ``tag_tensor`` is provided, it's tiled to the input's batch
    size and routed to any encoder block exposing a non-None
    ``tag_embedder`` attribute. ``single_eval_pos`` (the context/query
    boundary) is routed to n_ctx-aware blocks (grid_preprocessor,
    axial_attention_block) — same dispatch as the training loop.
    """
    from .loop import _split_encoder_heads  # type: ignore

    modules = list(getattr(model, "modules", []))
    encoder, heads = _split_encoder_heads(modules)
    batch_tag = _tile_tag_to_batch(tag_tensor, seq)
    enc_out = seq
    for mod in encoder:
        kwargs: dict[str, Any] = {}
        if batch_tag is not None and getattr(mod, "tag_embedder", None) is not None:
            kwargs["tag"] = batch_tag
        if (
            single_eval_pos is not None
            and single_eval_pos > 0
            and getattr(mod, "needs_single_eval_pos", False)
        ):
            kwargs["single_eval_pos"] = single_eval_pos
        result = mod(enc_out, **kwargs) if kwargs else mod(enc_out)
        enc_out = result[0] if isinstance(result, tuple) else result
    head_outputs = [head(enc_out) for head in heads] if heads else [enc_out]
    # Single-output regressor/classifier — pick the first head whose
    # trailing dim is 1. Most catalog templates are single-head.
    for ho in head_outputs:
        if ho.dim() >= 3 and ho.shape[-1] == 1:
            return ho
    return head_outputs[0] if head_outputs else None


def _read_declared_capabilities(yaml_path: Path) -> dict:
    """Read a `capabilities:` manifest from a model.yaml / prior.yaml, YAML-only.

    Returns {} when the file is missing, unparseable, or declares no manifest.
    Deliberately does NOT go through load_model / load_prior (which require the
    project importable + blocks resolvable) — a capabilities ping must stay
    cheap and must never depend on the checkpoint, which may live on a remote
    runner. Same philosophy as the `_peek_*` helpers above.
    """
    import yaml

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — any read/parse issue → no manifest
        return {}
    if not isinstance(data, dict):
        return {}
    caps = data.get("capabilities")
    return caps if isinstance(caps, dict) else {}


def _resolve_native_manifest(manifest_path: Path, project_root: Path) -> dict:
    """The declared capabilities manifest for a native studio-model run.

    The model's `capabilities:` wins; the prior's is the fallback, so a prior
    can own the task schema for the data shape it produces. Returns {} when
    neither declares one. Shared by the capabilities ping and native execution
    so both read the SAME manifest.
    """
    import yaml

    try:
        run_yaml = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        run_yaml = {}

    def _ref_id(key: str) -> str | None:
        ref = run_yaml.get(key) if isinstance(run_yaml, dict) else None
        rid = ref.get("id") if isinstance(ref, dict) else None
        return rid if isinstance(rid, str) and rid else None

    caps: dict = {}
    model_id = _ref_id("model")
    if model_id:
        caps = _read_declared_capabilities(project_root / "models" / f"{model_id}.yaml")
    if not caps:
        prior_id = _ref_id("prior")
        if prior_id:
            caps = _read_declared_capabilities(
                project_root / "priors" / prior_id / "prior.yaml"
            )
    return caps if isinstance(caps, dict) else {}


def _native_capabilities(*, manifest_path: Path, project_root: Path) -> dict:
    """Answer the capability manifest for a native studio-model run (no adapter)
    from its declarative model/prior YAML — no weights, no adapter. Returns the
    generic capabilities envelope every Try-it surface understands; an empty
    manifest simply leaves the host on its legacy form.
    """
    caps = _resolve_native_manifest(manifest_path, project_root)
    return {
        "engine": "pfnstudio",
        "command": "capabilities",
        "result": {"capabilities": caps or {}},
    }


def _shape_native_output(spec: dict, pred: Any, node_names: list) -> dict:
    """Project a native model's head output into the result shape declared by
    the task's `output_schema.kind`. Currently handles `ranked_list` (the shape
    the schema-Try-it's bar renderer expects); anything else returns the raw
    array so the caller can still inspect it.
    """
    import numpy as np
    import torch

    os_ = spec.get("output_schema") if isinstance(spec, dict) else None
    os_ = os_ if isinstance(os_, dict) else {}
    kind = os_.get("kind")

    arr = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)

    if kind == "ranked_list":
        # Head output is (B, R, K) (per-row, all rows identical for a broadcast
        # per-episode answer) or (B, K). Read batch 0's last (query) row.
        a = arr[0] if arr.ndim >= 2 else arr
        row = a[-1] if a.ndim == 2 else a  # (K,)
        field = os_.get("field", "root_causes")
        name_key = os_.get("label_field", "variable")
        score_key = os_.get("score_field", "score")
        limit = int(row.shape[-1])
        n = min(len(node_names), limit) if node_names else limit
        items = [
            {
                name_key: (node_names[i] if i < len(node_names) else f"x{i + 1}"),
                score_key: float(row[i]),
            }
            for i in range(n)
        ]
        items.sort(key=lambda d: -d[score_key])
        return {field: items}

    if kind in ("classification", "class_probabilities", "probabilities"):
        # Honor a DECLARED classification task: shape the head output into
        # class probabilities instead of letting it fall through to a raw /
        # regression-looking array. Binary (scalar head → 1 logit/row) →
        # sigmoid; multi-class (K>1 logits) → softmax over the class axis.
        a = arr[0] if arr.ndim >= 2 else arr  # drop batch → (R, K) or (R,)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        field = os_.get("field", "predictions")
        if a.shape[-1] == 1:
            p1 = 1.0 / (1.0 + np.exp(-a[:, 0]))
            return {
                field: [int(p >= 0.5) for p in p1],
                "probabilities": [[float(1.0 - p), float(p)] for p in p1],
            }
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        sm = e / e.sum(axis=-1, keepdims=True)
        return {
            field: [int(row.argmax()) for row in sm],
            "probabilities": [[float(v) for v in row] for row in sm],
        }

    return {"raw": arr.tolist()}


def _dispatch_inference(
    model: Any,
    payload: dict,
    *,
    expected_d_in: int | None = None,
    tag_tensor: Any = None,
) -> dict:
    """Route the payload to the regression / classification / discovery
    path based on which keys are present. Returns the inference output dict.

    Regression + classification now use the packed-token PFN convention
    introduced with the regression-ICL refactor (see
    docs/refactor-regression-icl.md). The model sees the FULL sequence —
    context tokens (with observed targets) plus query tokens (masked) —
    and we slice predictions out at query positions. Earlier versions
    of this file ran the model on query x's alone, which silently
    threw away the context; that's the bug this commit closes.

    `expected_d_in`: the model's actual in_features (from its weight
    matrix). When the payload's per-token width already equals
    `expected_d_in` we skip _pack_context_query — those models were
    trained un-packed (e.g. forecast priors that emit the full
    engineered-feature row directly) and double-packing would crash
    the model.     Uses the packed path only when the payload width is
    `expected_d_in - 2` (the canonical packed convention). When width
    already equals `expected_d_in`, rows are concatenated un-packed.
    When expected_d_in is unknown, un-packed is assumed if width matches
    the model embedder (see _checkpoint_d_in / topology.json).
    """
    import numpy as np
    import torch

    # Classification: {"context": {"x": ..., "labels": [...]}, "query": {"x": ...}}
    if "context" in payload and "query" in payload and "labels" in (payload.get("context") or {}):
        ctx = payload["context"]
        qry = payload["query"]
        x_ctx = np.asarray(ctx.get("x", []), dtype=np.float32)
        lbl_ctx = np.asarray(ctx.get("labels", []), dtype=np.float32)
        x_qry = np.asarray(qry.get("x", []), dtype=np.float32)
        if x_ctx.ndim == 1:
            x_ctx = x_ctx.reshape(-1, 1)
        if x_qry.ndim == 1:
            x_qry = x_qry.reshape(-1, 1)

        packed_input = expected_d_in is not None and int(x_ctx.shape[-1]) + 2 == expected_d_in
        if packed_input:
            seq = _pack_context_query(x_ctx, lbl_ctx, x_qry)
        else:
            # Un-packed regime: trainer fed the prior's emitted rows
            # straight to the model (no marker columns). Concatenate
            # context + query rows in the same order without adding
            # any extra columns.
            seq = np.concatenate([x_ctx, x_qry], axis=0).astype(np.float32)
        n_ctx = int(x_ctx.shape[0])

        with torch.no_grad():
            inp = torch.from_numpy(seq).float().unsqueeze(0)  # (1, N, F+2)
            out = _forward_full_sequence(model, inp, tag_tensor=tag_tensor, single_eval_pos=n_ctx)
            if out is None:
                raise ValueError(
                    "Model has no head producing a scalar logit per token — expected a scalar_head."
                )
            logits = out[0, n_ctx:, 0]  # query positions only
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            preds = [1 if p >= 0.5 else 0 for p in probs]
        return {
            "task": "classification",
            "predictions": preds,
            "probabilities": probs,
            "context_size": int(x_ctx.shape[0]),
            "query_size": int(x_qry.shape[0]),
        }

    # Regression: {"context": {"x": ..., "y": ...}, "query": {"x": ...}}
    if "context" in payload and "query" in payload:
        ctx = payload["context"]
        qry = payload["query"]
        x_ctx = np.asarray(ctx.get("x", []), dtype=np.float32)
        y_ctx = np.asarray(ctx.get("y", []), dtype=np.float32)
        x_qry = np.asarray(qry.get("x", []), dtype=np.float32)
        if x_ctx.ndim == 1:
            x_ctx = x_ctx.reshape(-1, 1)
        if x_qry.ndim == 1:
            x_qry = x_qry.reshape(-1, 1)

        packed_input = expected_d_in is not None and int(x_ctx.shape[-1]) + 2 == expected_d_in
        if packed_input:
            seq = _pack_context_query(x_ctx, y_ctx, x_qry)
        else:
            # Un-packed regime — forecast models (ar2, tabpfn-ts) train
            # on the prior's engineered rows directly; the trainer's
            # regression branch forwards the full sequence without adding
            # marker columns. Mirror that here so the predict path
            # matches the train-time tensor shape exactly.
            seq = np.concatenate([x_ctx, x_qry], axis=0).astype(np.float32)
        n_ctx = int(x_ctx.shape[0])

        with torch.no_grad():
            inp = torch.from_numpy(seq).float().unsqueeze(0)  # (1, N, F+2)
            out = _forward_full_sequence(model, inp, tag_tensor=tag_tensor, single_eval_pos=n_ctx)
            if out is None:
                raise ValueError(
                    "Model has no head producing scalar predictions — expected a scalar_head."
                )
            # Generic distributional-head reduction. A head that emits a
            # distribution over buckets (e.g. a bar-distribution head, output
            # shape (..., num_buckets)) exposes `to_prediction(output) ->
            # (..., 1)` to collapse its logits to a point estimate. Duck-typed
            # + OPTIONAL: a plain scalar_head has no such method, so `out`
            # passes through unchanged and the channel-0 slice is the scalar.
            reduce_fn = _head_to_prediction(model)
            if reduce_fn is not None:
                out = reduce_fn(out)
            preds = out[0, n_ctx:, 0].cpu().numpy().tolist()  # query positions only
        return {
            "task": "regression",
            "predictions": preds,
            "context_size": int(x_ctx.shape[0]),
            "query_size": int(x_qry.shape[0]),
        }

    # Discovery: {"X": [[...]]} → adjacency matrix (unpacked path,
    # since discovery doesn't use context/query semantics).
    if "X" in payload and "y" not in payload:
        X = np.asarray(payload["X"], dtype=np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(X).float().unsqueeze(0)
            out = _forward_encoder_and_pick_head(model, xt, (), tag_tensor=tag_tensor)
            if out is None:
                raise ValueError("Model has no head — can't run discovery inference.")
            adj = out.squeeze(0).cpu().numpy().tolist()
        return {"task": "discovery", "adjacency": adj}

    raise ValueError(
        "unrecognised inference payload shape — expected either "
        "{'context':{'x','y'},'query':{'x'}} (regression) or "
        "{'X':[[...]]} (discovery)."
    )
