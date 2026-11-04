"""Pfnfile — compose a new PFN model from a marketplace base model.

The Pfnfile is to PFN Studio what a Modelfile is to Ollama: a small, line-
oriented recipe that names a base checkpoint and (optionally) how to continue
training it. Grammar (instructions are case-insensitive, any order, `#` starts
a comment, blank lines are ignored)::

    FROM      <base-slug[:tag]>     required, exactly once — marketplace base model
    NAME      <default name>        optional, once — used when the CLI gets no name
    PRIOR     <slug-or-path>        optional, once — prior to train on
    MODEL     <slug-or-path>        optional, once — studio model artifact
    EVAL      <slug-or-path>        optional, repeatable — evals to attach
    PARAMETER <key> <value>         repeatable — typed (int / float / bool / str),
                                    passed through as run hyperparameters
    LICENSE   <text> | \"\"\"...\"\"\"     optional — repeated lines join; triple quotes
                                    open a multi-line block
    REQUIRES  <min-studio-version>  optional, once — minimum pfnstudio version

Example::

    FROM <base-slug>:latest
    PRIOR my-domain-prior
    PARAMETER steps 2000
    PARAMETER lr 3e-5
    LICENSE Apache-2.0

Pure module: no network, no heavy imports — `create_cli` does the talking.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_TRIPLE = '"""'


class ParseError(ValueError):
    """A Pfnfile could not be parsed. Carries the 1-based line number."""

    def __init__(self, line: int, reason: str, source: str | None = None):
        self.line, self.reason, self.source = line, reason, source
        where = f"{source}:{line}" if source else f"line {line}"
        super().__init__(f"{where}: {reason}")


@dataclass
class Pfnfile:
    from_slug: str
    from_tag: str | None = None
    name: str | None = None
    prior: str | None = None
    model: str | None = None
    evals: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    license: str | None = None
    requires: str | None = None
    source: str | None = None

    @property
    def from_ref(self) -> str:
        return f"{self.from_slug}:{self.from_tag}" if self.from_tag else self.from_slug

    @property
    def has_training_spec(self) -> bool:
        """True when the file asks for a continued-pretraining / fine-tune run
        on top of the base (anything beyond FROM / NAME / LICENSE / REQUIRES)."""
        return bool(self.prior or self.model or self.evals or self.parameters)


def coerce_value(raw: str) -> Any:
    """PARAMETER value typing: quoted → str verbatim; int; float; true/false → bool; else str."""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    if _INT_RE.match(raw):
        return int(raw)
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return float(raw)
    except ValueError:
        return raw


def _split_from(value: str, line: int, source: str | None) -> tuple[str, str | None]:
    slug, sep, tag = value.partition(":")
    if not _SLUG_RE.match(slug):
        raise ParseError(
            line, f"FROM: invalid base slug {slug!r} (lowercase letters, digits, . _ -)", source
        )
    if sep and not _SLUG_RE.match(tag):
        raise ParseError(line, f"FROM: invalid tag {tag!r}", source)
    return slug, (tag or None)


def parse_text(text: str, source: str | None = None) -> Pfnfile:
    """Parse Pfnfile text. Raises ParseError with line number + reason."""
    lines = text.splitlines()
    seen: dict[str, int] = {}
    from_slug: str | None = None
    out = Pfnfile(from_slug="", source=source)
    license_parts: list[str] = []
    i = 0
    while i < len(lines):
        lineno = i + 1
        stripped = lines[i].strip()
        i += 1
        if not stripped or stripped.startswith("#"):
            continue
        head, _, rest = stripped.partition(" ")
        key, value = head.upper(), rest.strip()
        # Inline comments only after whitespace, so values like "a#b" survive.
        if key != "LICENSE":
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()

        if key in ("FROM", "NAME", "PRIOR", "MODEL", "REQUIRES"):
            if key in seen:
                raise ParseError(lineno, f"duplicate {key} (first at line {seen[key]})", source)
            seen[key] = lineno
            if not value:
                raise ParseError(lineno, f"{key} needs a value", source)

        if key == "FROM":
            from_slug, out.from_tag = _split_from(value, lineno, source)
        elif key == "NAME":
            out.name = value
        elif key == "PRIOR":
            out.prior = value
        elif key == "MODEL":
            out.model = value
        elif key == "EVAL":
            if not value:
                raise ParseError(lineno, "EVAL needs a slug or path", source)
            out.evals.append(value)
        elif key == "PARAMETER":
            try:
                parts = shlex.split(value, posix=True)
            except ValueError as e:
                raise ParseError(lineno, f"PARAMETER: {e}", source) from e
            if len(parts) < 2:
                raise ParseError(lineno, "PARAMETER needs <key> <value>", source)
            pkey = parts[0]
            if pkey in out.parameters:
                raise ParseError(lineno, f"duplicate PARAMETER {pkey!r}", source)
            # Re-read the raw value (after the key) so quoting decides the type.
            raw = value[len(pkey) :].strip() if value.startswith(pkey) else " ".join(parts[1:])
            out.parameters[pkey] = coerce_value(raw)
        elif key == "LICENSE":
            if value.startswith(_TRIPLE):
                body = value[len(_TRIPLE) :]
                if body.rstrip().endswith(_TRIPLE) and len(body.rstrip()) >= len(_TRIPLE):
                    license_parts.append(body.rstrip()[: -len(_TRIPLE)].strip())
                else:
                    block, closed = [body] if body else [], False
                    while i < len(lines):
                        ln = lines[i]
                        i += 1
                        if ln.rstrip().endswith(_TRIPLE):
                            block.append(ln.rstrip()[: -len(_TRIPLE)])
                            closed = True
                            break
                        block.append(ln)
                    if not closed:
                        raise ParseError(lineno, 'LICENSE: unterminated """ block', source)
                    license_parts.append("\n".join(block).strip("\n"))
            elif value:
                license_parts.append(value)
            else:
                raise ParseError(lineno, "LICENSE needs text", source)
        elif key == "REQUIRES":
            if not _VERSION_RE.match(value):
                raise ParseError(
                    lineno, f"REQUIRES: expected a dotted version like 0.9.0, got {value!r}", source
                )
            out.requires = value
        else:
            raise ParseError(lineno, f"unknown instruction {head!r}", source)

    if not from_slug:
        raise ParseError(len(lines) or 1, "missing FROM <base-slug[:tag]> (required)", source)
    out.from_slug = from_slug
    out.license = "\n".join(license_parts) if license_parts else None
    return out


def parse_file(path: str | Path) -> Pfnfile:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ParseError(0, f"cannot read {p}: {e.strerror or e}", str(p)) from e
    return parse_text(text, source=str(p))


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s) or "model"


def version_tuple(v: str) -> tuple[int, ...]:
    """'0.8.16+dev' / '0.9.0rc1' → (0, 8, 16); non-numeric tails are dropped."""
    nums: list[int] = []
    for part in re.split(r"[.+-]", v):
        m = re.match(r"\d+", part)
        if not m:
            break
        nums.append(int(m.group()))
    return tuple(nums)


def requires_satisfied(pf: Pfnfile, installed: str | None) -> bool | None:
    """True/False against the installed pfnstudio version; None if unknown."""
    if not pf.requires:
        return True
    if not installed:
        return None
    return version_tuple(installed) >= version_tuple(pf.requires)


def to_request(
    pf: Pfnfile,
    name: str,
    *,
    base_run_id: str | None = None,
    prior_slug: str | None = None,
    model_slug: str | None = None,
    eval_slugs: list[str] | None = None,
    digest: str | None = None,
) -> dict[str, Any]:
    """Body for POST /projects/{projectId}/runs (CreateRunDto) — the continued-
    pretraining run that warm-starts from the installed base run.

    Refs default to the Pfnfile's PRIOR/MODEL/EVAL values; the CLI passes the
    resolved slugs when those were paths or were left for the project default.
    Lineage (FROM, digest, LICENSE, REQUIRES) rides along under
    hyperparams.pfnfile so it is recorded server-side next to the run.
    """
    evals = eval_slugs if eval_slugs is not None else list(pf.evals)
    body: dict[str, Any] = {
        "slug": slugify(name),
        "description": f"{name} — FROM {pf.from_ref}",
        "priorRef": {"slug": prior_slug or pf.prior},
        "modelRef": {"slug": model_slug or pf.model},
        "evalRefs": [{"slug": s} for s in evals],
        "hyperparams": {
            **pf.parameters,
            "pfnfile": {
                "from": {"slug": pf.from_slug, "tag": pf.from_tag or "latest", "digest": digest},
                **({"license": pf.license} if pf.license else {}),
                **({"requires": pf.requires} if pf.requires else {}),
            },
        },
    }
    if base_run_id:
        body["initWeightsFromRunId"] = base_run_id
    return body
