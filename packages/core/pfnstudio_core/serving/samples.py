"""Try-it sample data: the shape contract, and how samples reach a manifest.

A "sample" is runnable example data for ONE inference task, keyed by the form
fields that task's ``input_schema.kind`` renders — it is literally what a user
would otherwise type or upload. Samples come from two places:

  1. the adapter, via ``capabilities()[task]["sample"]`` (one) or ``["samples"]``
     (several), and
  2. the project's own ``samples/*.json`` files, which a user uploads and binds
     to a task with a ``"task"`` key inside the file.

Both are normalised into ``capabilities()[task]["samples"]`` — a list — so every
client (Try-it, MCP, CLI) sees one shape regardless of origin.

A sample may also carry ground truth::

    {"task": "root_cause", "observations": [...], "anomalies": [...],
     "expected": {"top": "conveyor_speed"}}

``expected`` is keyed by the task's ``output_schema.kind`` (see
``_EXPECTED_FIELDS``) rather than its input kind, because "correct" means a
different thing for a ranked list than for a graph. It is nested rather than
spread across the top level so the ground-truth namespace can never collide
with an input field of the same name.

Why validation lives here rather than in a client
-------------------------------------------------
A sample nested under the right task can still be shaped for the WRONG kind.
That is not hypothetical: a deployed adapter declared ``kind="context_query"``
with ``{context: {X, labels}, query: {X}}``, the task later moved to
``table_with_target``, and the Try-it form — which seeds ``rows``/``columns``/
``target`` — silently rendered nothing. The form could not distinguish "adapter
shipped no sample" from "adapter shipped a sample I cannot read", so a broken
manifest sat in production unnoticed.

Validating at the ``capabilities()`` boundary fixes that once for every surface
instead of once per client, and a bad sample is REPORTED (see ``issues``) rather
than dropped, because dropping is what hid it in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Field contract per input_schema.kind: the sample keys each kind's form binds.
# This mirrors what the studio's schema-driven form reads for that kind; keep
# the two in step when adding a kind.
#
#   required — absent means the form cannot seed the task at all
#   optional — seeded when present, ignored when not
_KIND_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "context_query": (frozenset({"context", "query"}), frozenset()),
    "table": (frozenset({"rows"}), frozenset({"columns", "target", "threshold"})),
    "table_with_target": (
        frozenset({"rows", "columns", "target"}),
        frozenset({"event_time"}),
    ),
    "rca": (
        frozenset({"observations", "anomalies"}),
        frozenset({"columns", "symptom"}),
    ),
    "rdb": (
        frozenset({"tables", "target_table", "entity_id", "target"}),
        frozenset({"relationships", "max_depth", "time_column"}),
    ),
}

# Ground-truth contract per output_schema.kind — the mirror of _KIND_FIELDS on
# the OUTPUT side. A sample may carry an ``expected`` block holding the known
# answer for its data, which the Try-it form shows beside what the model
# actually returned, so "did this work?" is answerable without the reader
# holding the right answer in their head.
#
# This is deliberately NOT a score. A sample is hand-picked demo data, often
# three rows; an accuracy computed from it would be quoted as model accuracy,
# which is what evals (held-out sets, tracked over time) are for. What lives
# here is a worked answer, and the surfaces that read it show agreement and
# disagreement rather than a headline number.
#
#   required — absent means there is nothing to compare
#   optional — refines the comparison (e.g. a tolerance for float equality)
_EXPECTED_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "per_row_scalar": (frozenset({"values"}), frozenset({"tolerance"})),
    "class_probabilities": (frozenset({"labels"}), frozenset()),
    "graph": (frozenset({"edges"}), frozenset()),
    # Either form answers "did the right cause come out on top?": `top` names
    # the one variable that should rank first, `order` gives the full expected
    # ranking. Neither is individually required, so the check is bespoke below.
    "ranked_list": (frozenset(), frozenset({"top", "order"})),
}

# Present on any sample, carrying no INPUT field data. `expected` is here
# because it is the answer, not an input: counting it as a field would make it
# look like an unrecognised one to validate_sample.
_META_KEYS = frozenset({"name", "description", "task", "source", "expected"})


def known_kinds() -> frozenset[str]:
    return frozenset(_KIND_FIELDS)


def known_output_kinds() -> frozenset[str]:
    return frozenset(_EXPECTED_FIELDS)


def validate_expected(expected: Any, out_kind: str | None, *, label: str) -> str | None:
    """Return a problem with a sample's ``expected`` block for a task whose
    ``output_schema.kind`` is ``out_kind``, or None when it is usable.

    Same reasoning as validate_sample: ground truth shaped for the wrong output
    kind renders nothing, which is indistinguishable from a sample that simply
    has no ground truth. Checking it where the contract is defined means the
    mismatch is REPORTED once, for every client, instead of quietly ignored.
    """
    if expected is None:
        return None
    if not isinstance(expected, dict):
        return (
            f"{label}: 'expected' must be an object of ground-truth fields, "
            f"got {type(expected).__name__}"
        )
    if not expected:
        return f"{label}: 'expected' is empty — omit it, or state the known answer."
    if not out_kind or out_kind not in _EXPECTED_FIELDS:
        return None

    required, optional = _EXPECTED_FIELDS[out_kind]
    present = {k for k, v in expected.items() if v is not None}

    # ranked_list accepts either `top` or `order`; every other kind has a
    # fixed required set.
    if out_kind == "ranked_list" and not (present & optional):
        return (
            f"{label}: 'expected' for a ranked_list needs 'top' (the variable that "
            f"should rank first) or 'order' (the full expected ranking)"
        )

    missing = sorted(required - present)
    unknown = sorted(present - required - optional)
    if not missing and not unknown:
        return None

    # Report both halves in one message. Ground truth under the wrong key is the
    # likeliest mistake by far — saying only "missing values" to someone who
    # supplied `labels` describes the symptom and hides the cause.
    detail = (
        f"{label}: output kind '{out_kind}' compares on {', '.join(sorted(required | optional))}"
    )
    if missing:
        detail += f" but 'expected' is missing {', '.join(missing)}"
    if unknown:
        joiner = " and" if missing else " but 'expected'"
        detail += (
            f"{joiner} has {', '.join(unknown)}, "
            "which it does not compare — that would be silently ignored"
        )
    return detail


def validate_sample(
    sample: Any,
    kind: str | None,
    *,
    label: str,
    out_kind: str | None = None,
) -> str | None:
    """Return a human-readable problem with ``sample`` for a task of ``kind``,
    or None when it is usable.

    An unknown kind is not an error here: adapters may declare kinds this core
    does not know about (the form falls back to its own handling), so we only
    check the ones whose contract we own. ``out_kind`` is the task's
    ``output_schema.kind``, used to check any ``expected`` ground truth.
    """
    if not isinstance(sample, dict):
        return f"{label}: expected an object of form fields, got {type(sample).__name__}"

    problem = validate_expected(sample.get("expected"), out_kind, label=label)
    if problem:
        return problem

    if not kind or kind not in _KIND_FIELDS:
        return None

    required, optional = _KIND_FIELDS[kind]
    present = {k for k, v in sample.items() if k not in _META_KEYS and v is not None}
    missing = sorted(required - present)
    if missing:
        extra = sorted(present - required - optional)
        detail = f"{label}: task kind '{kind}' needs {', '.join(sorted(required))}"
        detail += f" but the sample is missing {', '.join(missing)}"
        if extra:
            detail += f" (it has unrecognised {', '.join(extra)})"
        return detail
    return None


def normalise_task_samples(spec: Any, *, task: str) -> list[str]:
    """Fold a task spec's ``sample`` / ``samples`` into a validated ``samples``
    list, IN PLACE. Returns the problems found.

    ``sample`` (singular) stays supported — it is what every existing adapter
    ships — and becomes the first entry. Invalid entries are dropped from the
    list but reported, so a client renders the usable ones and can still tell
    the operator what was wrong with the rest.
    """
    if not isinstance(spec, dict):
        return []

    kind = None
    schema = spec.get("input_schema")
    if isinstance(schema, dict):
        k = schema.get("kind")
        kind = k if isinstance(k, str) else None

    out_kind = None
    out_schema = spec.get("output_schema")
    if isinstance(out_schema, dict):
        k = out_schema.get("kind")
        out_kind = k if isinstance(k, str) else None

    raw: list[Any] = []
    single = spec.get("sample")
    if single is not None:
        raw.append(single)
    many = spec.get("samples")
    if isinstance(many, list):
        raw.extend(many)
    elif many is not None:
        return [f"{task}: 'samples' must be a list, got {type(many).__name__}"]

    issues: list[str] = []
    valid: list[dict[str, Any]] = []
    for i, s in enumerate(raw):
        label = f"{task} sample {i + 1}"
        if isinstance(s, dict) and isinstance(s.get("name"), str):
            label = f"{task} sample '{s['name']}'"
        problem = validate_sample(s, kind, label=label, out_kind=out_kind)
        if problem:
            issues.append(problem)
            continue
        entry = dict(s)
        entry.setdefault("name", f"sample {i + 1}")
        entry.setdefault("source", "adapter")
        valid.append(entry)

    spec["samples"] = valid
    # Keep `sample` as the first valid entry so older clients that only read the
    # singular key keep working.
    if valid:
        spec["sample"] = valid[0]
    else:
        spec.pop("sample", None)
    return issues


def load_uploaded_samples(project_root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Read ``<project_root>/samples/*.json`` into {task: [sample, ...]}.

    Each file binds itself to a task with a top-level ``"task"`` key. Binding by
    content rather than by folder means renaming or moving the file cannot
    silently re-bind it, and a typo names a task we can check against the
    manifest instead of becoming an orphan nobody reports.

    Returns the samples and any problems (unreadable file, missing/!string
    ``task``). Whether the named task EXISTS is checked by the caller, which is
    the only place that knows the manifest.
    """
    samples: dict[str, list[dict[str, Any]]] = {}
    issues: list[str] = []

    d = Path(project_root) / "samples"
    if not d.is_dir():
        return samples, issues

    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(f"samples/{f.name}: not readable as JSON ({type(e).__name__}: {e})")
            continue
        if not isinstance(data, dict):
            issues.append(f"samples/{f.name}: expected a JSON object, got {type(data).__name__}")
            continue
        task = data.get("task")
        if not isinstance(task, str) or not task:
            issues.append(
                f'samples/{f.name}: needs a top-level "task" naming the capability it is for'
            )
            continue
        entry = {k: v for k, v in data.items() if k != "task"}
        entry.setdefault("name", f.name)
        entry["source"] = "uploaded"
        samples.setdefault(task, []).append(entry)

    return samples, issues


def apply_samples(caps: Any, project_root: Path | None = None) -> list[str]:
    """Normalise every task's samples and fold in the project's uploaded ones.

    Mutates ``caps`` in place and returns all problems found, for the caller to
    attach to the manifest response.
    """
    if not isinstance(caps, dict):
        return []

    uploaded: dict[str, list[dict[str, Any]]] = {}
    issues: list[str] = []
    if project_root is not None:
        uploaded, issues = load_uploaded_samples(project_root)

    for task, spec in caps.items():
        if not isinstance(spec, dict):
            continue
        for s in uploaded.pop(task, []):
            spec.setdefault("samples", [])
            if isinstance(spec["samples"], list):
                spec["samples"].append(s)
        issues.extend(normalise_task_samples(spec, task=task))

    # Anything left names a task the adapter does not expose — the exact case a
    # folder-based binding would have swallowed.
    for task, orphans in uploaded.items():
        for s in orphans:
            issues.append(
                f"samples/{s.get('name')}: task '{task}' is not one of this model's "
                f"capabilities ({', '.join(sorted(caps)) or 'none'})"
            )
    return issues
