"""Normalize legacy / bundle eval YAML before EvalSpec validation."""

from __future__ import annotations

from typing import Any


def normalize_eval_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Some bundles use dataset.notes + registry_version without dataset.name."""
    out = dict(raw)
    ds = out.get("dataset")
    if not isinstance(ds, dict):
        out["dataset"] = {"name": str(out.get("id", "eval")), "source": None}
        return out

    ds_norm = dict(ds)
    if not ds_norm.get("name"):
        ds_norm["name"] = str(out.get("id", "eval"))
    if not ds_norm.get("source"):
        if ds_norm.get("registry_version"):
            ds_norm["source"] = f"registry:{ds_norm['name']}"
        elif isinstance(ds_norm.get("notes"), str) and "sachs" in ds_norm["notes"].lower():
            ds_norm["source"] = "sachs:build_ground_truth"
    # Bundle-style evals ship registry_version without a studio catalog
    # entry yet — training should proceed; scorers skip when data is absent.
    if ds_norm.get("registry_version") and ds_norm.get("optional") is not True:
        ds_norm["optional"] = True
    out["dataset"] = ds_norm
    return out
