"""Device selection and optional torch.compile for container serving."""

from __future__ import annotations

import os
from typing import Any


def resolve_serve_device() -> Any:
    """Pick inference device from PFNSTUDIO_DEVICE or CUDA availability."""
    import torch

    override = os.environ.get("PFNSTUDIO_DEVICE", "").strip()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_compile_enabled() -> bool:
    flag = os.environ.get("PFNSTUDIO_TORCH_COMPILE", "").strip().lower()
    return flag in ("1", "true", "yes")


def maybe_torch_compile(model: Any) -> Any:
    """Apply torch.compile when PFNSTUDIO_TORCH_COMPILE is set."""
    if not torch_compile_enabled():
        return model
    import torch

    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)
    except Exception:
        return model
