"""The world-model inference runtime: an inference repo (manifest.json + a torch.export graph) answers predict,
recommend and diagnose, refuses what the world does not cover, and pads and masks small row
counts without changing answers.

The graph here is a toy with the inference graph contract's signature (the real one is the released
network, written by PFN Studio's publish-inference): each output's median is the mean of the
measured first output over the valid context rows, plus the query's first and third inputs
(bandwidth, batch); so a padding
row that leaked into the answer would show.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pfnstudio_core.inference import (
    InferenceError,
    is_inference_ref,
    load_inference,
    read_manifest,
)
from pfnstudio_core.inference.world import Refusal, diagnose, expand_candidates

INPUTS = ["gpu_bw", "price", "batch", "tp"]
OUTPUTS = ["tok_s", "fits"]
UNKNOWNS = ["eff"]


class Toy(torch.nn.Module):
    def forward(self, ctx_x, ctx_sweep, ctx_y, ctx_valid, qry_x, qry_sweep):
        y0 = ctx_y[..., 0]
        m = torch.isfinite(y0) & ctx_valid
        mean = torch.where(m, y0, torch.zeros_like(y0)).sum(-1) / m.sum(-1).clamp(min=1)
        med = (mean[:, None] + qry_x[..., 0] + qry_x[..., 2] + 100.0 * qry_sweep)[..., None]
        med = med.expand(-1, -1, 2)
        u = (0.25 + 0.5 * ctx_valid.double().mean(-1))[:, None]
        return med, med - 1.0, med + 1.0, u, u - 0.1, u + 0.1


def _repo(tmp_path: Path, zero_shot: bool = True) -> Path:
    from torch.export import Dim

    tmp_path.mkdir(parents=True, exist_ok=True)

    ex = (
        torch.zeros(1, 3, 4, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, 2, dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 5, 4, dtype=torch.float64),
        torch.zeros(1, 5, dtype=torch.float64),
    )
    c, q = Dim("c", min=2), Dim("q", min=2)
    dyn = {
        "ctx_x": {1: c},
        "ctx_sweep": {1: c},
        "ctx_y": {1: c},
        "ctx_valid": {1: c},
        "qry_x": {1: q},
        "qry_sweep": {1: q},
    }
    torch.export.save(
        torch.export.export(Toy(), ex, dynamic_shapes=dyn), str(tmp_path / "model.pt2")
    )
    manifest = {
        "format": "pfnstudio.inference.world",
        "format_version": 1,
        "name": "Toy",
        "torch": "test",
        "graph": {"file": "model.pt2", "min_rows": 2},
        "network": {
            "inputs": INPUTS,
            "outputs": OUTPUTS,
            "unknowns": UNKNOWNS,
            "unknown_ranges": {"eff": [0.0, 1.0]},
            "zero_shot": zero_shot,
        },
        "inputs": [{"name": "price", "lo": 0.1, "hi": 10}, {"name": "batch", "lo": 1, "hi": 64}],
        "facts": [
            {
                "name": "hardware",
                "role": "environment",
                "key": "id",
                "columns": ["gpu_bw"],
                "rows": [{"id": "fast", "gpu_bw": 2000}, {"id": "slow", "gpu_bw": 500}],
            },
            {
                "name": "parallelism",
                "role": "intervention",
                "key": "id",
                "columns": ["tp"],
                "rows": [{"id": "tp1", "tp": 1}, {"id": "tp2", "tp": 2}],
            },
        ],
        "engine_picks": {"input": "batch", "maximizes": ["tok_s"], "ceiling": 64},
        "examples": {
            "predict": {
                "question": "predict",
                "environment": {"hardware": "slow", "price": 1},
                "observations": [],
                "queries": [{"batch": 4, "tp": 1}],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


ENV = {"hardware": "fast", "price": 2.0}


def test_reads_an_inference_folder(tmp_path):
    root = _repo(tmp_path)
    assert (
        is_inference_ref(root)
        and is_inference_ref("hf://org/repo")
        and not is_inference_ref(tmp_path / "x")
    )
    m = read_manifest(root)
    m["format_version"] = 99
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(InferenceError, match="newer"):
        load_inference(root)


def test_predict_masks_padding(tmp_path):
    c = load_inference(_repo(tmp_path))
    obs = [{"settings": {"batch": 8, "parallelism": "tp1"}, "measured": {"tok_s": 10.0}}]
    a = c.ask(
        {
            "question": "predict",
            "environment": ENV,
            "observations": obs,
            "queries": [{"batch": 16, "tp": 2}],
        }
    )
    p = a["answer"]["predictions"][0]["outputs"]["tok_s"]
    # one context row (padded to two) and one query (padded to two): only the real row counts
    assert p["p50"] == pytest.approx(10.0 + 2000.0 + 16)
    assert (p["p5"], p["p95"]) == (pytest.approx(2025.0), pytest.approx(2027.0))
    assert a["posterior"]["fit"][0]["fitted"] == pytest.approx(10.0 + 2000.0 + 8)
    assert a["posterior"]["not_reached"] is None


def test_zero_shot_and_its_refusal(tmp_path):
    q = {
        "question": "predict",
        "environment": ENV,
        "observations": [],
        "queries": [{"batch": 1, "tp": 1}],
    }
    a = load_inference(_repo(tmp_path / "a")).ask(q)
    assert a["answer"]["predictions"][0]["outputs"]["tok_s"]["p50"] == pytest.approx(2001.0)
    with pytest.raises(Refusal, match="no data"):
        load_inference(_repo(tmp_path / "b", zero_shot=False)).ask(q)


@pytest.mark.parametrize(
    "bad, why",
    [
        ({"batch": 100, "tp": 1}, "outside what this world covers"),
        ({"batch": 4, "tp": 3}, "no row of the parallelism table"),
        ({"batch": 4, "parallelism": "tp9"}, "no row 'tp9'"),
        ({"batch": 4}, "not given"),
        ({"batch": 4, "tp": 1, "bacth": 2}, "not inputs of this world"),
    ],
)
def test_refusals(tmp_path, bad, why):
    c = load_inference(_repo(tmp_path))
    with pytest.raises(Refusal, match=why):
        c.ask({"question": "predict", "environment": ENV, "observations": [], "queries": [bad]})


def test_sweep_row_takes_the_ceiling(tmp_path):
    c = load_inference(_repo(tmp_path))
    row, sweep = c.row(ENV, {"tp": 1, "sweep": True})
    assert sweep == 1.0 and row["batch"] == 64
    a = c.ask(
        {
            "question": "predict",
            "environment": ENV,
            "observations": [],
            "queries": [{"tp": 1, "sweep": True}],
        }
    )
    assert "sweep row" in a["answer"]["predictions"][0]["note"]


def test_recommend_ranks_feasible_first(tmp_path):
    c = load_inference(_repo(tmp_path))
    a = c.ask(
        {
            "question": "recommend",
            "environment": ENV,
            "observations": [],
            "candidates": {"grid": {"batch": [1, 8]}, "fixed": {"tp": 1}},
            "objective": "tok_s",
            "direction": "max",
            "constraints": [{"output": "fits", "op": "<=", "value": 2005}],
        }
    )["answer"]
    # batch 8 (2008) scores higher but breaks the constraint, so batch 1 (2001) ranks first
    assert [r["settings"]["batch"] for r in a["ranked"]] == [1, 8]
    assert a["ranked"][0]["feasible"] and not a["ranked"][1]["feasible"]
    assert expand_candidates([{"a": 1}]) == [{"a": 1}]


def test_diagnose_and_effect(tmp_path):
    c = load_inference(_repo(tmp_path))
    obs = [{"settings": {"batch": 8, "tp": 1}, "measured": {"tok_s": 5.0}}]
    d = c.ask({"question": "diagnose", "environment": ENV, "observations": obs})["answer"]
    assert d["unknowns"][0]["unknown"] == "eff" and d["not_reached"] is None
    assert diagnose({"unknowns": {}, "fit": [], "warnings": []})["unknowns"] == []
    with pytest.raises(Refusal, match="effect is not in this inference repo"):
        c.ask({"question": "effect", "environment": ENV, "observations": obs})


def test_selfcheck_scores_a_measurement_set(tmp_path):
    c = load_inference(_repo(tmp_path))
    ms = {
        "outputs": ["tok_s"],
        "group_by": ["g"],
        "rows": [
            {"g": "a", "gpu_bw": 2000, "price": 1, "batch": 4, "tp": 1, "tok_s": 2004.0},
            {"g": "a", "gpu_bw": 2000, "price": 1, "batch": 4, "tp": 1, "tok_s": 2004.0},
            {"g": "b", "gpu_bw": 500, "price": 1, "_maximize_over": "batch", "tp": 1, "tok_s": 1.0},
        ],
    }
    s = c.selfcheck(ms)["tok_s"]
    assert s["unpredicted"] == 1 and s["groups"] == 2
    # each row of group a from the other: 2004 + 2000 + 4 against 2004 measured
    assert s["network"] == pytest.approx((4008.0 - 2004.0) / 2004.0)
    assert np.isfinite(s["network_zero_shot"])
