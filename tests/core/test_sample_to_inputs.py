"""sample_to_inputs: a declared sample (form fields) → the predict inputs
Try-it sends for it. Used by `pfnstudio run` with no --input."""

from __future__ import annotations

from pfnstudio_core.serving.samples import sample_to_inputs


def test_list_rows_and_meta_keys() -> None:
    s = {"rows": [[1, 2]], "columns": ["a", "b"], "name": "n", "expected": {"edges": []}}
    assert sample_to_inputs("table", s) == {"rows": [[1, 2]], "columns": ["a", "b"]}


def test_context_query_already_structured_passes_through() -> None:
    s = {"context": {"X": [[1]], "t": [0], "y": [2]}, "query": {"X": [[1]]}}
    assert sample_to_inputs("context_query", s) == s


def test_context_query_text_splits_treatment_covariates_outcome() -> None:
    out = sample_to_inputs("context_query", {"context": "1 2 3 4\n0 5 6 7", "query": "9 8"})
    assert out == {
        "context": {"t": [1.0, 0.0], "X": [[2.0, 3.0], [5.0, 6.0]], "y": [4.0, 7.0]},
        "query": {"X": [[9.0, 8.0]]},
    }


def test_non_numeric_cells_stay_strings() -> None:
    assert sample_to_inputs("table", {"rows": "a,1\nb,2"})["rows"] == [["a", 1.0], ["b", 2.0]]


def test_rdb_sample_passes_through() -> None:
    s = {"tables": {"t": [{"id": 1}]}, "target_table": "t", "entity_id": "id", "target": "y"}
    assert sample_to_inputs("rdb", s) == s
