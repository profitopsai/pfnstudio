"""HuggingFace push / pull helpers for the Deployments flow.

Two operations:

  - push_folder(local_dir, repo_id, token, revision='main', private=True)
      Mirror the contents of `local_dir` into `repo_id` on HF Hub. Used
      by NestJS at promote time to publish the trained run's project
      tree + checkpoint to the user's HF account.

  - pull_snapshot(repo_id, dest_dir, token, revision='main')
      Materialise a copy of `repo_id` at `revision` into `dest_dir`. Used
      by the worker at boot time when artifactRef.kind == 'hf'.

Both go through huggingface_hub so authentication, chunking, LFS
handling, and resume-on-error all work without us reimplementing them.

The CLI wraps each in a subcommand. NestJS invokes the CLI rather than
embedding Python — same pattern as `pfnstudio serve`.
"""

from __future__ import annotations

from pathlib import Path


def _ensure_model_card_metadata(local_dir: Path) -> None:
    """Give the repo card the YAML frontmatter the Hub expects.

    HF renders README.md as the model card and shows a "YAML Metadata
    Warning: empty or missing yaml metadata in repo card" banner when the
    file has no leading ``---`` block. Studio projects author a plain
    markdown README (it doubles as the project front page in the studio,
    where frontmatter would be noise), so the metadata is added here — at
    the HF boundary, on the staged copy about to be uploaded — rather than
    stored on the project.

    Rules:
      - A README that already opens with its own frontmatter is the
        author's — left untouched.
      - Only generic facts are written (library name + tags). No license
        is invented and nothing project-specific is inferred.
      - A missing README gets a metadata-only card so the banner never
        shows on a bare repo.
    """
    readme = local_dir / "README.md"
    try:
        body = readme.read_text(encoding="utf-8") if readme.exists() else ""
    except OSError:
        return
    if body.lstrip().startswith("---"):
        return
    meta = (
        "---\n"
        "library_name: pfnstudio\n"
        "tags:\n"
        "- pfnstudio\n"
        "- prior-fitted-network\n"
        "- tabular\n"
        "---\n\n"
    )
    readme.write_text(meta + body, encoding="utf-8")


def push_folder(
    *,
    local_dir: Path,
    repo_id: str,
    token: str,
    revision: str = "main",
    private: bool = True,
    commit_message: str | None = None,
) -> dict[str, str]:
    """Push every file under ``local_dir`` to ``repo_id``.

    Returns ``{"commit_sha": "...", "url": "..."}``.

    The repo must already exist; the NestJS side ensures that via the
    Hub's create-repo endpoint before calling here so we keep this helper
    focused. We do still pass ``create_pr=False`` so a freshly-created
    empty repo accepts the upload as a normal commit on the branch.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _ensure_model_card_metadata(Path(local_dir))
    # exist_ok=True so re-running the push (e.g. user clicks Promote
    # twice without deleting the deployment) doesn't blow up.
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message or "pfnstudio: publish trained run",
    )
    # CommitInfo has .commit_url and .oid (the commit hash). Fall back
    # to string-only if a future hub release changes the shape.
    commit_sha = getattr(result, "oid", "") or ""
    url = getattr(result, "commit_url", "") or f"https://huggingface.co/{repo_id}"
    return {"commit_sha": str(commit_sha), "url": str(url)}


def pull_snapshot(
    *,
    repo_id: str,
    dest_dir: Path,
    token: str,
    revision: str = "main",
) -> Path:
    """Download ``repo_id`` at ``revision`` into ``dest_dir``.

    Uses ``snapshot_download`` with ``local_dir=dest_dir`` so files
    end up as real files (not symlinks into the global hub cache).
    Returns ``dest_dir`` so callers can pipe straight into the worker.
    """
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
        local_dir=str(dest_dir),
    )
    return dest_dir


# ─────────────────────────────────────────────────────────────────────────
# Checkpoint URI resolver
# ─────────────────────────────────────────────────────────────────────────
#
# The studio's predict path takes ``--checkpoint <dir>`` and expects to
# find ``model.pt`` + ``topology.json`` (and any future companion files)
# there. The catalog's Solution templates carry a ``checkpoint_ref``
# describing where those files live.
#
# Two valid forms for ``checkpoint_ref``:
#
#   1. **Local path** (legacy / dev): a plain filesystem path that
#      already contains the checkpoint files. Returned unchanged.
#
#   2. **Manifest** (production): an explicit mapping of local filename
#      → URI. Each URI is resolved independently; the resulting files
#      are materialised in one local directory using the keys as
#      filenames. No directory-listing on HF, no guessing which ``.pt``
#      to use — operators make the file roles explicit at publish time.
#
#      Example manifest::
#
#          {
#              "model.pt":      "hf://<org>/<repo>/v0_1/model.pt",
#              "topology.json": "hf://<org>/<repo>/v0_1/topology.json"
#          }
#
#      URIs accept two forms:
#        - ``hf://OWNER/REPO/PATH/TO/FILE`` — compact form (preferred)
#        - ``https://huggingface.co/OWNER/REPO/blob/REV/PATH`` — full URL
#          (the one HuggingFace's "Copy file URL" button emits)
#
#      ``revision`` defaults to ``main`` when not encoded in the URI.
#
# The CLI's predict command stringly accepts whichever form the run YAML
# carried (the API json-stringifies a manifest dict before passing it on
# the command line). ``resolve_checkpoint`` is the single entrypoint that
# detects which form was passed and returns a usable directory.

# Recognized checkpoint file extensions used for plain-path detection.
# A URI ending in one of these is treated as a single-file reference;
# everything else is treated as a directory or manifest reference.
_CHECKPOINT_FILE_EXTS = {".pt", ".safetensors", ".bin", ".json"}


def _parse_hf_uri(uri: str) -> tuple[str, str, str]:
    """Parse an HF URI into ``(repo_id, path_in_repo, revision)``.

    Accepts both forms documented above. Raises ``ValueError`` on any
    string that doesn't match a recognised HF URL shape — caller is
    responsible for detecting "plain local path" before calling this.
    """
    s = uri.strip()
    if s.startswith("hf://"):
        rest = s[len("hf://") :]
        parts = rest.split("/", 2)
        if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
            raise ValueError(f"hf:// URI must have the shape hf://OWNER/REPO/PATH; got: {uri!r}")
        owner, repo, path = parts[0], parts[1], parts[2]
        return f"{owner}/{repo}", path, "main"

    if s.startswith("https://huggingface.co/"):
        rest = s[len("https://huggingface.co/") :]
        parts = rest.split("/")
        # The HF URL grammar is:
        #   {owner}/{repo}/{blob|tree|resolve}/{revision}/{...path}
        # We accept all three middle segments — they're interchangeable
        # for downloading a single file. ``blob`` is the GUI viewer URL;
        # ``resolve`` is the raw-file URL; ``tree`` is the dir browser.
        if len(parts) < 5 or parts[2] not in {"blob", "tree", "resolve"}:
            raise ValueError(
                "huggingface.co URL must have the shape "
                "https://huggingface.co/OWNER/REPO/{blob,tree,resolve}/REV/PATH; "
                f"got: {uri!r}"
            )
        owner, repo, _segment, revision = parts[0], parts[1], parts[2], parts[3]
        path = "/".join(parts[4:])
        if not owner or not repo or not path:
            raise ValueError(f"huggingface.co URL is missing owner/repo/path; got: {uri!r}")
        return f"{owner}/{repo}", path, revision

    raise ValueError(
        f"unrecognised checkpoint URI scheme — expected hf:// or "
        f"https://huggingface.co/...; got: {uri!r}"
    )


def _looks_like_hf_uri(s: str) -> bool:
    """Quick discriminator — does this string look like an HF URI vs a
    plain filesystem path? Used so callers can branch early without
    paying the parse cost or its error-message overhead on simple
    /local/paths/like/this."""
    return s.startswith("hf://") or s.startswith("https://huggingface.co/")


def resolve_checkpoint_uri(
    uri: str,
    *,
    dest_dir: Path,
    filename: str,
    token: str | None = None,
) -> Path:
    """Materialise a single HF URI into ``dest_dir`` under ``filename``.

    Uses ``hf_hub_download`` so caching, ETag re-validation, chunking,
    and resume-on-error are HF-Hub-managed. The downloaded file ends up
    as ``dest_dir / filename`` regardless of the URI's path inside the
    repo — this lets manifest entries rename files at materialisation
    (e.g. publish as ``final.pt``, present locally as ``model.pt``).

    Args
    ----
    uri : str
        ``hf://OWNER/REPO/PATH`` or ``https://huggingface.co/...``.
    dest_dir : Path
        Local directory the file is dropped into.
    filename : str
        Local name within ``dest_dir`` (typically the manifest key).
    token : str | None
        Optional HF auth token. If None, reads ``HF_TOKEN`` from env.
        Public repos work without a token.

    Returns
    -------
    Path : Absolute path to the materialised file.
    """
    from huggingface_hub import hf_hub_download

    repo_id, path_in_repo, revision = _parse_hf_uri(uri)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        revision=revision,
        token=token,
        local_dir=str(dest_dir),
        repo_type="model",
    )
    materialised = Path(local_path).resolve()
    # If the HF path-in-repo is nested (e.g. v0_1/model.pt), hf_hub_download
    # preserves the path under local_dir (dest_dir/v0_1/model.pt). The
    # predict path wants files flat at the manifest filename, so move
    # into place if needed.
    desired = (dest_dir / filename).resolve()
    if materialised != desired:
        desired.parent.mkdir(parents=True, exist_ok=True)
        if desired.exists():
            desired.unlink()
        materialised.rename(desired)
    return desired


def _github_blob_to_raw(url: str) -> str:
    """Normalise a GitHub *blob* page URL (which serves HTML) to its raw file
    URL. Non-GitHub URLs and already-raw URLs pass through unchanged.

    https://github.com/OWNER/REPO/blob/REF/PATH
      → https://raw.githubusercontent.com/OWNER/REPO/REF/PATH
    """
    import re

    m = re.match(r"^https://github\.com/([^/]+)/([^/]+)/blob/(.+)$", url)
    if m:
        owner, repo, rest = m.group(1), m.group(2), m.group(3)
        rest = rest.split("?", 1)[0]  # drop ?raw=true etc.
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"
    return url


def resolve_http_uri(url: str, *, dest_dir: Path, filename: str) -> Path:
    """Download a single file from a generic HTTP(S) URL into ``dest_dir``.

    Covers base-model checkpoints hosted anywhere that isn't HuggingFace — a
    GitHub raw/blob file, a GitHub Release asset, S3, a CDN. GitHub blob pages
    are normalised to their raw URL first. Streams to disk so large weights
    don't buffer in memory.
    """
    import shutil as _shutil
    import urllib.request

    raw = _github_blob_to_raw(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    req = urllib.request.Request(raw, headers={"User-Agent": "pfnstudio"})
    with urllib.request.urlopen(req) as resp, open(target, "wb") as out:
        _shutil.copyfileobj(resp, out)
    return target


def resolve_checkpoint(
    ref: str,
    *,
    cache_root: Path | None = None,
    token: str | None = None,
) -> Path:
    """Resolve a ``checkpoint_ref`` (in any supported form) to a local
    checkpoint directory the CLI's predict path can consume.

    Form detection (in order):

      1. **Plain local path** — if the string parses as a JSON-encoded
         manifest, fall through to (3); otherwise, if it doesn't look
         like an HF URI, return ``Path(ref).resolve()`` unchanged. This
         keeps existing dev workflows (local checkpoint dirs) intact.

      2. **Single HF URI** — ``hf://OWNER/REPO/PATH`` or full
         ``https://huggingface.co/...`` URL pointing at a single file.
         Resolves into a cache dir named after the URI's repo + path
         and returns that dir. The downloaded filename is taken from
         the URI's basename. Useful for legacy single-bundle ``.pt``
         checkpoints where topology is embedded in the state dict.

      3. **Manifest dict (JSON-encoded)** — ``{"local-name": "hf://..."}``
         mapping. Each entry resolves into the same dir, with the
         resulting filename being the key (not the URI's basename).
         This is the production form for studio-trained checkpoints
         (``model.pt`` + ``topology.json`` + future companions).

    Cache layout::

        <cache_root>/<repo>--<rev>--<path-hash>/
            model.pt
            topology.json
            ...

    Args
    ----
    ref : str
        The ``checkpoint_ref`` as stored in Run.checkpointPath. May be a
        local path, a single HF URI, or a JSON-encoded manifest dict.
    cache_root : Path | None
        Root of the studio's checkpoint cache. Defaults to
        ``~/.cache/pfnstudio/checkpoints``.
    token : str | None
        Optional HF auth token. None → reads ``HF_TOKEN`` from env.

    Returns
    -------
    Path : Local directory containing the checkpoint files.
    """
    import hashlib
    import json
    import os

    if not ref:
        raise ValueError("checkpoint ref is empty")

    if cache_root is None:
        cache_root = Path(
            os.environ.get(
                "PFNSTUDIO_CHECKPOINT_CACHE",
                str(Path.home() / ".cache" / "pfnstudio" / "checkpoints"),
            )
        )
    if token is None:
        token = os.environ.get("HF_TOKEN", "").strip() or None

    # Try manifest form first: ref might be a JSON object literal.
    manifest: dict[str, str] | None = None
    if ref.startswith("{"):
        try:
            parsed = json.loads(ref)
            if isinstance(parsed, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
            ):
                manifest = parsed
        except json.JSONDecodeError:
            manifest = None

    if manifest is not None:
        if not manifest:
            raise ValueError("checkpoint manifest is empty")
        # Cache key: stable hash of the sorted manifest items, so the
        # same manifest always lands in the same dir regardless of key
        # order. Re-resolving uses hf_hub_download's ETag check to skip
        # the network round-trip when the file is already there.
        digest = hashlib.sha256(json.dumps(sorted(manifest.items())).encode("utf-8")).hexdigest()[
            :16
        ]
        # Build a human-skimmable dir name from the first manifest URI's repo,
        # so operators inspecting the cache can tell at a glance which
        # checkpoint is which. The sample may be a non-HF URL (a GitHub raw
        # .cpkt, a CDN asset), so only parse it as HF when it looks like one.
        sample_uri = next(iter(manifest.values()))
        if _looks_like_hf_uri(sample_uri):
            repo_id, _path, revision = _parse_hf_uri(sample_uri)
            dest = cache_root / f"{repo_id.replace('/', '--')}--{revision}--{digest}"
        else:
            dest = cache_root / f"manifest--{digest}"
        # Route each entry by scheme — a manifest can mix HF URIs, generic
        # HTTP(S) URLs (GitHub raw / release / CDN), and local paths. Mirrors
        # the single-URI branch below so a manifest is never HF-only.
        for fname, uri in manifest.items():
            if _looks_like_hf_uri(uri):
                resolve_checkpoint_uri(uri, dest_dir=dest, filename=fname, token=token)
            elif uri.startswith("http://") or uri.startswith("https://"):
                resolve_http_uri(_github_blob_to_raw(uri), dest_dir=dest, filename=fname)
            else:
                import shutil as _shutil

                dest.mkdir(parents=True, exist_ok=True)
                _shutil.copyfile(Path(uri).expanduser().resolve(), dest / fname)
        return dest

    # Single-URI form.
    if _looks_like_hf_uri(ref):
        repo_id, path_in_repo, revision = _parse_hf_uri(ref)
        owner_repo = repo_id.replace("/", "--")
        path_safe = path_in_repo.replace("/", "--")
        dest = cache_root / f"{owner_repo}--{revision}--{path_safe}"
        # Use the URI's basename as the local filename. Caller can
        # rename downstream if needed.
        filename = path_in_repo.rsplit("/", 1)[-1]
        resolve_checkpoint_uri(
            ref,
            dest_dir=dest,
            filename=filename,
            token=token,
        )
        return dest

    # Generic HTTP(S) URL — a base model hosted anywhere that isn't HF: a
    # GitHub raw/blob file, a GitHub Release asset, S3, a CDN. (HF https URLs
    # were already caught by _looks_like_hf_uri above.)
    if ref.startswith("http://") or ref.startswith("https://"):
        raw = _github_blob_to_raw(ref)
        filename = raw.rsplit("/", 1)[-1] or "model.pt"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        dest = cache_root / f"url--{digest}"
        resolve_http_uri(raw, dest_dir=dest, filename=filename)
        return dest

    # Plain local path — return as-is, no fetch.
    p = Path(ref).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    return p
