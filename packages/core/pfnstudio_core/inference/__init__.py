"""Inference repos: a published model's inference-only files, run without its source.

PFN Studio's *publish-inference* writes a repo that holds exactly:

* ``manifest.json``: what the model reads and answers. Its ``format`` names the runtime here
  that reads it; the rest lists the inputs with their ranges, the catalog rows it can be given,
  the outputs, the hidden factors and the release's own scores;
* ``model.pt2``: the trained network as a ``torch.export`` graph, its normalization inside;
* ``README.md``: the model card.

No priors, simulators, training code or resume states: nothing here needs them. ``load_inference``
reads an inference repo (a folder, or ``hf://<org>/<repo>``) and returns the runtime its format
names. On the command line: ``pfnstudio run hf://<org>/<repo> --task predict --input req.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
GRAPH = "model.pt2"
# The files an inference repo may hold: the exporter refuses to publish anything else.
ALLOWED_FILES = frozenset({MANIFEST, GRAPH, "README.md", ".gitattributes"})
FORMATS = {"pfnstudio.inference.world": 1}


class InferenceError(ValueError):
    """The folder or repo is not an inference repo this version can read; the message says why."""


def is_inference_ref(ref: str | Path) -> bool:
    """``hf://…`` refs, and folders holding an inference manifest."""
    s = str(ref)
    if s.startswith("hf://"):
        return True
    p = Path(s)
    if not (p / MANIFEST).is_file():
        return False
    try:
        return json.loads((p / MANIFEST).read_text()).get("format") in FORMATS
    except (OSError, ValueError):
        return False


def resolve(ref: str | Path, *, revision: str | None = None, token: str | None = None) -> Path:
    """A local folder for ``ref``: the folder itself, or ``hf://<org>/<repo>[@revision]``
    downloaded into the Hugging Face cache (a private repo needs ``hf auth login`` or HF_TOKEN).
    """
    s = str(ref)
    if not s.startswith("hf://"):
        return Path(s)
    repo = s[len("hf://") :].strip("/")
    if "@" in repo:
        repo, at = repo.split("@", 1)
        revision = revision or at
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise InferenceError(
            "reading an hf:// inference repo needs huggingface_hub: pip install huggingface_hub"
        ) from e
    return Path(
        snapshot_download(
            repo, revision=revision or "main", token=token, allow_patterns=sorted(ALLOWED_FILES)
        )
    )


def read_manifest(root: Path) -> dict[str, Any]:
    f = Path(root) / MANIFEST
    if not f.is_file():
        raise InferenceError(f"{root} holds no {MANIFEST}: it is not an inference repo")
    m = json.loads(f.read_text())
    fmt, ver = m.get("format"), m.get("format_version")
    if fmt not in FORMATS:
        raise InferenceError(f"inference format '{fmt}' is not one this pfnstudio-core reads")
    if not isinstance(ver, int) or ver > FORMATS[fmt]:
        raise InferenceError(
            f"{fmt} version {ver} is newer than this pfnstudio-core reads ({FORMATS[fmt]}): "
            "pip install -U pfnstudio-core"
        )
    return m


def load_inference(
    ref: str | Path, *, revision: str | None = None, token: str | None = None, device: str = "cpu"
):
    """The runtime for an inference repo: today a :class:`~pfnstudio_core.inference.world.WorldInference`."""
    root = resolve(ref, revision=revision, token=token)
    m = read_manifest(root)
    if m["format"] == "pfnstudio.inference.world":
        from .world import WorldInference

        return WorldInference(root, m, device=device)
    raise InferenceError(f"no runtime for {m['format']}")  # pragma: no cover — FORMATS guards it


__all__ = [
    "ALLOWED_FILES",
    "GRAPH",
    "MANIFEST",
    "InferenceError",
    "is_inference_ref",
    "load_inference",
    "read_manifest",
    "resolve",
]
