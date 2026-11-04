"""The sample inputs shipped in pfnstudio/samples/ (`run <slug> --input sample`)."""

from __future__ import annotations

import json

from pfnstudio.samples import _DIR, bundled_sample, bundled_sample_tasks


def test_every_shipped_sample_is_a_task_inputs_request() -> None:
    files = sorted(_DIR.glob("*/*.json"))
    assert files, "pfnstudio ships no sample inputs"
    for f in files:
        body = json.loads(f.read_text(encoding="utf-8"))
        assert set(body) == {"task", "inputs"}, f
        assert body["task"] == f.stem, f
        assert isinstance(body["inputs"], dict) and body["inputs"], f


def test_lookup_by_task_and_single_task_default() -> None:
    slug = next(p.parent.name for p in sorted(_DIR.glob("*/*.json")))
    tasks = bundled_sample_tasks(slug)
    assert bundled_sample(slug, tasks[0]) == _DIR / slug / f"{tasks[0]}.json"
    if len(tasks) == 1:
        assert bundled_sample(slug, None) == _DIR / slug / f"{tasks[0]}.json"
    assert bundled_sample(slug, "no-such-task") is None
    assert bundled_sample("no-such-model", None) is None
