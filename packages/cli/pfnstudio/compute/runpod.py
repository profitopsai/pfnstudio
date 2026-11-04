"""RunPod compute adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import ComputeAdapter


class RunPodAdapter(ComputeAdapter):
    name = "runpod"

    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        if not os.environ.get("RUNPOD_API_KEY"):
            return {
                "status": "error",
                "reason": "RUNPOD_API_KEY not set. Get one at https://www.runpod.io/console/user/settings",
            }

        try:
            import runpod  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "reason": "runpod not installed. Install: pip install runpod",
            }

        return {
            "status": "submitted",
            "adapter": self.name,
            "run_yaml": str(run_yaml.relative_to(project_root)),
            "note": (
                "RunPod pod template + on-start script need per-org config. "
                "Configure .pfnstudio/runpod.yaml — see docs/compute.md."
            ),
        }
