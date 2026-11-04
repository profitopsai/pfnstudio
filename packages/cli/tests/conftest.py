"""Make `pfnstudio` importable from a plain checkout (no editable install)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
