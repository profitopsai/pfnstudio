"""Normalize legacy model block types before registry lookup."""

from __future__ import annotations

import inspect
import re
from typing import Any

from .registry import get_block


def parse_marketplace_block_descriptor(raw: str) -> tuple[str, dict[str, Any]]:
    """Parse marketplace catalog strings like ``tabular_embedder (d=64)``."""
    s = raw.strip()
    m = re.match(r"^tabular_embedder(?:\s*\(d=(\d+)\))?$", s, re.I)
    if m:
        cfg: dict[str, Any] = {}
        if m.group(1):
            cfg["d_model"] = int(m.group(1))
        return "tabular_embedder", cfg
    m = re.match(
        r"^transformer_encoder\s*[×x]\s*(\d+)(?:\s*\(heads=(\d+)\))?$",
        s,
        re.I,
    )
    if m:
        cfg = {"n_layers": int(m.group(1))}
        if m.group(2):
            cfg["n_heads"] = int(m.group(2))
        return "transformer_encoder", cfg
    m = re.match(r"^scalar_head(?:\s*\(d_out=(\d+)\))?$", s, re.I)
    if m:
        cfg = {}
        if m.group(1):
            cfg["d_out"] = int(m.group(1))
        return "scalar_head", cfg
    for bare in (
        "causal_attention_pool",
        "discovery_head",
        "estimation_head",
        "scalar_head",
        "tabular_embedder",
        "transformer_encoder",
    ):
        if s == bare:
            return bare, {}
    return s, {}


_RUNTIME_BLOCK_TYPES = frozenset(
    {
        "tabular_embedder",
        "transformer_encoder",
        "causal_attention_pool",
        "scalar_head",
        "discovery_head",
        "estimation_head",
        "rms_norm",
        "mlp",
        "time_encoding",
        "long_horizon_time_encoding",
    }
)


def filter_block_config(block_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Drop designer-only keys (e.g. ``note``) before ``cls(**config)``."""
    cls = get_block(block_type)
    allowed = {p for p in inspect.signature(cls.__init__).parameters if p != "self"}
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key not in allowed:
            continue
        # UI uses 0 for "infer d_in at first batch" — omit so LazyLinear is used.
        if key == "d_in" and value in (None, 0):
            continue
        out[key] = value
    return out


_HEAD_TYPE_BY_TASK: dict[str, str] = {
    "discovery": "discovery_head",
    "regression": "scalar_head",
    "forecast": "scalar_head",
    "classification": "scalar_head",
    "estimation": "estimation_head",
}

# Prior.sample() defaults for common discovery priors (linear_scm, chain_scm).
_DISCOVERY_PRIOR_DEFAULT_D = 5


def infer_discovery_num_variables(
    prior_spec: Any,
    prior_overrides: dict[str, Any] | None = None,
    prior_instance: Any | None = None,
) -> int | None:
    """Resolve D for discovery_head from run prior overrides / sample defaults.

      Marketplace discovery-encoder models historically defaulted to
    num_variables=11 (Sachs). linear_scm uses d=5 by default — mismatch
      caused every training step to skip with a shape error.
    """
    import inspect

    overrides = dict(prior_overrides or {})
    for key in ("d", "num_variables", "num_vars", "n_variables"):
        if key in overrides and overrides[key] is not None:
            return int(overrides[key])

    if prior_instance is not None:
        try:
            sig = inspect.signature(prior_instance.sample)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            for key in ("d", "num_variables"):
                if key not in sig.parameters:
                    continue
                default = sig.parameters[key].default
                if default is not inspect.Parameter.empty and default is not None:
                    return int(default)

    params = getattr(prior_spec, "parameters", None) or {}
    if isinstance(params, dict):
        for key in ("d", "num_variables"):
            spec = params.get(key)
            if not isinstance(spec, dict):
                continue
            if "default" in spec and spec["default"] is not None:
                return int(spec["default"])

    # Discovery priors without an explicit override use fixed d in sample().
    if isinstance(params, dict) and "d" in params:
        return _DISCOVERY_PRIOR_DEFAULT_D
    if isinstance(params, dict) and "num_variables" in params:
        spec = params["num_variables"]
        if isinstance(spec, dict) and isinstance(spec.get("range"), list):
            r = spec["range"]
            if len(r) >= 2:
                return round((float(r[0]) + float(r[1])) / 2)
    return None


def align_discovery_head_num_variables(
    blocks: list[dict[str, Any]],
    num_variables: int,
) -> list[dict[str, Any]]:
    """Patch discovery_head.config.num_variables on a normalized block list."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        if b.get("type") != "discovery_head":
            out.append(b)
            continue
        cfg = dict(b.get("config") or {})
        cfg["num_variables"] = int(num_variables)
        out.append({**b, "config": cfg})
    return out


def _ensure_output_head_blocks(
    out: list[dict[str, Any]],
    output_heads: list[dict[str, Any]] | None,
    last_d_model: int,
    discovery_num_variables: int | None = None,
) -> list[dict[str, Any]]:
    if not output_heads:
        return out
    types = {b["type"] for b in out}
    result = list(out)
    for head in output_heads:
        task = head.get("task", "")
        head_type = _HEAD_TYPE_BY_TASK.get(str(task))
        if not head_type or head_type in types:
            continue
        if head_type == "discovery_head":
            cfg: dict[str, Any] = {"d_model": last_d_model}
            if discovery_num_variables is not None:
                cfg["num_variables"] = discovery_num_variables
        else:
            cfg = {"d_model": last_d_model, "d_out": 1}
        try:
            cfg = filter_block_config(head_type, cfg)
        except KeyError:
            pass
        result.append({"type": head_type, "name": head.get("name"), "config": cfg})
        types.add(head_type)
    return result


def normalize_model_blocks(
    blocks: list[dict[str, Any]],
    *,
    output_heads: list[dict[str, Any]] | None = None,
    discovery_num_variables: int | None = None,
) -> list[dict[str, Any]]:
    """Map bundle/designer block types to pfnstudio_core registry names."""
    last_d_model = 256
    out: list[dict[str, Any]] = []

    for b in blocks:
        type_ = b.get("type", "")
        config: dict[str, Any] = dict(b.get("config") or {})
        name = b.get("name")

        if type_ not in _RUNTIME_BLOCK_TYPES and re.search(r"[×x(]", type_):
            parsed_type, parsed_cfg = parse_marketplace_block_descriptor(type_)
            type_ = parsed_type
            config = {**parsed_cfg, **config}

        if type_ in ("custom", "conv1d"):
            continue

        if type_ == "transformer":
            type_ = "transformer_encoder"
        elif type_ == "attention_pool":
            type_ = "causal_attention_pool"
            q = config.pop("query_dim", None)
            if isinstance(q, (int, float)):
                config["d_model"] = int(q)
        elif type_ == "linear":
            type_ = "scalar_head"
            out_features = config.pop("out_features", None)
            config["d_model"] = last_d_model
            config["d_out"] = int(out_features) if isinstance(out_features, (int, float)) else 1
        elif type_ == "embedding":
            type_ = "tabular_embedder"
            dim = config.pop("dim", None)
            config.pop("vocab_size", None)
            if isinstance(dim, (int, float)):
                config["d_model"] = int(dim)
        elif type_ == "layer_norm":
            type_ = "rms_norm"
        elif type_ == "mlp" and "hidden_dim" in config:
            hidden = config.pop("hidden_dim", None)
            config.pop("dropout", None)
            if isinstance(hidden, (int, float)):
                config["d_hidden"] = int(hidden)
            if config.get("d_in") is None:
                config["d_in"] = last_d_model

        if (
            type_
            in (
                "transformer_encoder",
                "causal_attention_pool",
                "discovery_head",
                "estimation_head",
                "scalar_head",
            )
            and config.get("d_model") is None
        ):
            config["d_model"] = last_d_model

        if type_ in ("tabular_embedder", "transformer_encoder"):
            d = config.get("d_model")
            if isinstance(d, (int, float)):
                last_d_model = int(d)
        if type_ == "scalar_head":
            if config.get("d_model") is None:
                config["d_model"] = last_d_model
            d_out = config.get("d_out")
            d_model = config.get("d_model", last_d_model)
            if (
                isinstance(d_out, (int, float))
                and isinstance(d_model, (int, float))
                and (int(d_out) == int(d_model) or int(d_out) > 64)
            ):
                config["d_out"] = 1

        if (
            not out
            and type_ == "transformer_encoder"
            and not any(x.get("type") in ("tabular_embedder", "embedding") for x in blocks)
        ):
            out.append(
                {
                    "type": "tabular_embedder",
                    "name": "embed",
                    "config": {"d_model": config.get("d_model", last_d_model)},
                }
            )

        if type_ == "discovery_head" and discovery_num_variables is not None:
            config["num_variables"] = discovery_num_variables

        try:
            clean = filter_block_config(type_, config)
        except KeyError:
            clean = {k: v for k, v in config.items() if k != "note"}
        out.append({"type": type_, "name": name, "config": clean})

    return _ensure_output_head_blocks(out, output_heads, last_d_model, discovery_num_variables)
