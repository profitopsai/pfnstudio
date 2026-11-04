"""Cross-reference lint checks for an FM project.

Schema validation lives in validate.py. This module checks the things schemas
can't: that names referenced in one artifact actually exist as another artifact.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _safe_load(path: Path) -> dict:
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def lint_project(project: Path) -> list[str]:
    errors: list[str] = []

    prior_ids: dict[str, str] = {}
    for prior_yaml in (project / "priors").rglob("prior.yaml"):
        spec = _safe_load(prior_yaml)
        if "id" in spec:
            prior_ids[spec["id"]] = spec.get("version", "")
        impl = prior_yaml.parent / spec.get("implementation", "prior.py")
        if not impl.exists():
            errors.append(f"{prior_yaml}: implementation file not found: {impl}")
        if prior_yaml.parent.name != spec.get("id"):
            errors.append(
                f"{prior_yaml}: id '{spec.get('id')}' does not match directory name "
                f"'{prior_yaml.parent.name}'"
            )

    model_ids: dict[str, str] = {}
    for model_yaml in (project / "models").glob("*.yaml"):
        spec = _safe_load(model_yaml)
        if "id" in spec:
            model_ids[spec["id"]] = spec.get("version", "")

    eval_ids: dict[str, str] = {}
    for eval_yaml in (project / "evals").glob("*.yaml"):
        spec = _safe_load(eval_yaml)
        if "id" in spec:
            eval_ids[spec["id"]] = spec.get("version", "")
        # An eval that declares metrics needs code to compute them. The
        # scoring code is a sibling `evals/<slug>.py` with @register_scorer.
        # If that file exists but registers no scorer (empty / forgot the
        # decorator), the declared metrics silently never get computed — the
        # run reports nothing with no clue why. Flag it at author time.
        # (A *missing* .py may be covered by a core built-in scorer, so we
        # don't flag that here; the run-time path surfaces it with a reason.)
        declared_metrics = spec.get("metrics") or []
        if declared_metrics:
            scorer_py = eval_yaml.with_suffix(".py")
            if scorer_py.exists():
                try:
                    code = scorer_py.read_text()
                except Exception:
                    code = ""
                if "register_scorer" not in code:
                    metric_names = ", ".join(
                        m.get("name", str(m)) if isinstance(m, dict) else str(m)
                        for m in declared_metrics
                    )
                    errors.append(
                        f"{eval_yaml}: declares metrics ({metric_names}) but "
                        f"{scorer_py.name} has no @register_scorer — nothing will "
                        f"compute those metrics. Add a DatasetScorer decorated "
                        f'with @register_scorer("{spec.get("id", eval_yaml.stem)}").'
                    )

    for run_yaml in (project / "runs").glob("*.yaml"):
        spec = _safe_load(run_yaml)
        prior_ref = (spec.get("prior") or {}).get("id")
        if prior_ref and prior_ref not in prior_ids:
            errors.append(f"{run_yaml}: references unknown prior '{prior_ref}'")
        model_ref = (spec.get("model") or {}).get("id")
        if model_ref and model_ref not in model_ids:
            errors.append(f"{run_yaml}: references unknown model '{model_ref}'")
        for ev in spec.get("evals") or []:
            ev_id = ev.get("id") if isinstance(ev, dict) else None
            if ev_id and ev_id not in eval_ids:
                errors.append(f"{run_yaml}: references unknown eval '{ev_id}'")

    bib_keys: set[str] = set()
    bib = project / "literature" / "references.bib"
    if bib.exists():
        for line in bib.read_text().splitlines():
            line = line.strip()
            if line.startswith("@") and "{" in line:
                key = line.split("{", 1)[1].rstrip(",").strip()
                bib_keys.add(key)

    summaries_dir = project / "literature" / "summaries"
    if summaries_dir.exists():
        for md in summaries_dir.glob("*.md"):
            text = md.read_text()
            if "bibkey:" in text:
                for line in text.splitlines():
                    if line.strip().startswith("bibkey:"):
                        key = line.split(":", 1)[1].strip()
                        if key and key not in bib_keys:
                            errors.append(
                                f"{md}: references bibkey '{key}' not present in references.bib"
                            )

    for yaml_path in (
        list((project / "priors").rglob("prior.yaml"))
        + list((project / "evals").glob("*.yaml"))
        + list((project / "models").glob("*.yaml"))
    ):
        spec = _safe_load(yaml_path)
        for key in spec.get("citations") or []:
            if key not in bib_keys:
                errors.append(f"{yaml_path}: cites '{key}' which is not in references.bib")

    return errors
