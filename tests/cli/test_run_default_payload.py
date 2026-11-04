"""What `pfnstudio run <slug>` sends when no --input is given.

A task with a table-shaped declared sample runs ON that sample (minus the
sample's own labels); anything else sends the bare {task} so the adapter runs
its own self-check. Shapes below are the ones the core's samples
normalisation serves.
"""

from __future__ import annotations

from pfnstudio.cli import _default_payload

TABLE_SAMPLE = {
    "rows": [[0.1, 0.2], [0.3, 0.4]],
    "columns": ["x0", "x1"],
    "name": "sample 1",
    "source": "adapter",
}


def test_table_task_runs_on_its_declared_sample_without_the_labels() -> None:
    spec = {"input_schema": {"kind": "table"}, "sample": TABLE_SAMPLE}
    assert _default_payload("discover", spec) == {
        "task": "discover",
        "inputs": {"rows": [[0.1, 0.2], [0.3, 0.4]], "columns": ["x0", "x1"]},
    }


def test_table_with_target_keeps_its_target() -> None:
    sample = {**TABLE_SAMPLE, "target": "x1", "expected": {"accuracy": 1.0}}
    spec = {"input_schema": {"kind": "table_with_target"}, "sample": sample}
    out = _default_payload("classify", spec)
    assert out["inputs"]["target"] == "x1"
    assert "expected" not in out["inputs"] and "name" not in out["inputs"]


def test_form_shaped_sample_is_not_sent_as_input() -> None:
    # context_query samples are shaped for the studio form, not for predict.
    spec = {
        "input_schema": {"kind": "context_query"},
        "sample": {"context": [[1, 2]], "query": [[3]]},
    }
    assert _default_payload("estimate", spec) == {"task": "estimate"}


def test_no_sample_falls_back_to_the_adapters_self_check() -> None:
    assert _default_payload("forecast", {"input_schema": {"kind": "series"}}) == {
        "task": "forecast"
    }
    assert _default_payload("forecast", None) == {"task": "forecast"}


def test_no_task_sends_nothing() -> None:
    assert _default_payload(None, {"sample": TABLE_SAMPLE}) == {}
