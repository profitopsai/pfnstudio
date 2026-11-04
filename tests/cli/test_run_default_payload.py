"""What `pfnstudio run <slug>` sends when no --input is given.

A task with a declared sample runs ON that sample, converted to predict
inputs the way the studio's "Try with sample" converts it; a task without one
sends the bare {task} so the adapter runs its own self-check. Shapes below are
the ones the core's samples normalisation serves.
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


def test_context_query_text_sample_becomes_t_x_y() -> None:
    # Do-PFN / CausalPFN ship the form's text: treatment, covariates…, outcome.
    spec = {
        "input_schema": {"kind": "context_query"},
        "sample": {"context": "1, 0.5, 0.2, 3.0\n0, -1, 0.4, 1.5", "query": "0.1, 0.2"},
    }
    assert _default_payload("estimate", spec) == {
        "task": "estimate",
        "inputs": {
            "context": {"t": [1.0, 0.0], "X": [[0.5, 0.2], [-1.0, 0.4]], "y": [3.0, 1.5]},
            "query": {"X": [[0.1, 0.2]]},
        },
    }


def test_text_rows_and_columns_string_are_parsed() -> None:
    spec = {
        "input_schema": {"kind": "table_with_target"},
        "sample": {"rows": "1,2\n3,4", "columns": "a, b", "target": "b"},
    }
    assert _default_payload("rank_causes", spec)["inputs"] == {
        "rows": [[1.0, 2.0], [3.0, 4.0]],
        "columns": ["a", "b"],
        "target": "b",
    }


def test_no_sample_falls_back_to_the_adapters_self_check() -> None:
    assert _default_payload("forecast", {"input_schema": {"kind": "table"}}) == {"task": "forecast"}
    assert _default_payload("forecast", None) == {"task": "forecast"}


def test_no_task_sends_nothing() -> None:
    assert _default_payload(None, {"sample": TABLE_SAMPLE}) == {}
