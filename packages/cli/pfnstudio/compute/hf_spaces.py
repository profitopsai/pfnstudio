"""HuggingFace Spaces compute adapter.

Spaces aren't typically used for batch training, but they work for short
fine-tunes. This adapter pushes the project root + a Dockerfile to a Space and
streams logs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import ComputeAdapter


class HFSpacesAdapter(ComputeAdapter):
    name = "hf_spaces"

    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        if not os.environ.get("HF_TOKEN"):
            return {
                "status": "error",
                "reason": "HF_TOKEN not set. Create one at https://huggingface.co/settings/tokens",
            }

        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "reason": "huggingface_hub not installed. Install: pip install huggingface_hub",
            }

        return {
            "status": "submitted",
            "adapter": self.name,
            "run_yaml": str(run_yaml.relative_to(project_root)),
            "note": (
                "Space repo and SDK type need per-org config. "
                "Configure .pfnstudio/hf_spaces.yaml — see docs/compute.md."
            ),
        }
