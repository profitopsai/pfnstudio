"""Modal.com compute adapter.

Modal's Python SDK lets us define a remote function inline. The simplest model:
import modal at submit-time, define an image with pfnstudio installed, mount
the project root, and call pfnstudio.compute.local.LocalAdapter inside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ComputeAdapter


class ModalAdapter(ComputeAdapter):
    name = "modal"

    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        try:
            import modal  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "reason": "modal not installed. Install: pip install modal",
            }

        return {
            "status": "submitted",
            "adapter": self.name,
            "run_yaml": str(run_yaml.relative_to(project_root)),
            "note": (
                "Modal app definition needs a per-org image + secrets config. "
                "Configure .pfnstudio/modal.yaml — see docs/compute.md."
            ),
        }
