"""Pfnfile parser unit tests — pure, no network."""

from __future__ import annotations

import pytest
from pfnstudio.pfnfile import (
    ParseError,
    coerce_value,
    parse_text,
    requires_satisfied,
    slugify,
    to_request,
    version_tuple,
)

HAPPY = """
# A composed model
FROM <your-org>-base:v2
NAME demo-model
PRIOR priors/my-prior
MODEL my-base-xs
EVAL eval-a
EVAL evals/eval-b.yaml
PARAMETER steps 2000
PARAMETER lr 3e-5
PARAMETER cross_sectional true
PARAMETER tag "42"
PARAMETER note hello world   # trailing comment
LICENSE Apache-2.0
REQUIRES 0.8.0
""".replace("<your-org>-base", "org-base")


def test_happy_path():
    pf = parse_text(HAPPY, source="Pfnfile")
    assert pf.from_slug == "org-base" and pf.from_tag == "v2" and pf.from_ref == "org-base:v2"
    assert pf.name == "demo-model"
    assert pf.prior == "priors/my-prior" and pf.model == "my-base-xs"
    assert pf.evals == ["eval-a", "evals/eval-b.yaml"]
    assert pf.parameters == {
        "steps": 2000,
        "lr": 3e-5,
        "cross_sectional": True,
        "tag": "42",
        "note": "hello world",
    }
    assert isinstance(pf.parameters["steps"], int) and isinstance(pf.parameters["lr"], float)
    assert pf.license == "Apache-2.0" and pf.requires == "0.8.0"
    assert pf.has_training_spec


def test_missing_from():
    with pytest.raises(ParseError) as ei:
        parse_text("PARAMETER steps 10\n")
    assert "missing FROM" in str(ei.value)


def test_duplicate_from_reports_line():
    with pytest.raises(ParseError) as ei:
        parse_text("FROM a\n\nFROM b\n")
    assert ei.value.line == 3 and "duplicate FROM" in ei.value.reason


def test_case_insensitive_any_order_comments_blank_lines():
    pf = parse_text("\n# c\n  parameter steps 5\n\nfrom base-x   # inline\n\n")
    assert pf.from_slug == "base-x" and pf.from_tag is None and pf.parameters == {"steps": 5}
    assert not pf.license and pf.name is None


def test_from_only_has_no_training_spec():
    pf = parse_text("FROM base-x\nLICENSE MIT\n")
    assert not pf.has_training_spec


def test_typed_parameter_coercion():
    assert coerce_value("7") == 7 and coerce_value("-3") == -3
    assert coerce_value("0.5") == 0.5 and coerce_value("1e-4") == 1e-4
    assert coerce_value("TRUE") is True and coerce_value("false") is False
    assert coerce_value('"7"') == "7" and coerce_value("'x y'") == "x y"
    assert coerce_value("adam") == "adam"


def test_duplicate_parameter_and_unknown_instruction():
    with pytest.raises(ParseError, match="duplicate PARAMETER"):
        parse_text("FROM a\nPARAMETER k 1\nPARAMETER k 2\n")
    with pytest.raises(ParseError, match="unknown instruction"):
        parse_text("FROM a\nBOGUS x\n")
    with pytest.raises(ParseError, match="PARAMETER needs"):
        parse_text("FROM a\nPARAMETER onlykey\n")


def test_invalid_from_and_requires():
    with pytest.raises(ParseError, match="invalid base slug"):
        parse_text("FROM Bad/Slug\n")
    with pytest.raises(ParseError, match="REQUIRES"):
        parse_text("FROM a\nREQUIRES latest\n")


def test_license_multiline_block_and_joined_lines():
    pf = parse_text('FROM a\nLICENSE """\nline one\nline two\n"""\n')
    assert pf.license == "line one\nline two"
    pf2 = parse_text("FROM a\nLICENSE first\nLICENSE second\n")
    assert pf2.license == "first\nsecond"
    with pytest.raises(ParseError, match="unterminated"):
        parse_text('FROM a\nLICENSE """\nnever closed\n')


def test_to_request_shape():
    pf = parse_text(HAPPY)
    body = to_request(
        pf,
        "My Model",
        base_run_id="run_1",
        prior_slug="my-prior",
        eval_slugs=["eval-a", "eval-b"],
        digest="sha256:abc",
    )
    assert body["slug"] == "my-model"
    assert body["initWeightsFromRunId"] == "run_1"
    assert body["priorRef"] == {"slug": "my-prior"}
    assert body["modelRef"] == {"slug": "my-base-xs"}
    assert body["evalRefs"] == [{"slug": "eval-a"}, {"slug": "eval-b"}]
    hp = body["hyperparams"]
    assert hp["steps"] == 2000 and hp["lr"] == 3e-5
    assert hp["pfnfile"] == {
        "from": {"slug": "org-base", "tag": "v2", "digest": "sha256:abc"},
        "license": "Apache-2.0",
        "requires": "0.8.0",
    }
    assert "FROM org-base:v2" in body["description"]
    # Without a base run id the warm-start edge is simply absent.
    assert "initWeightsFromRunId" not in to_request(pf, "x")
    assert (
        to_request(parse_text("FROM a\n"), "x")["hyperparams"]["pfnfile"]["from"]["tag"] == "latest"
    )


def test_version_helpers():
    assert version_tuple("0.8.16+dev") == (0, 8, 16) and version_tuple("1.2rc1") == (1, 2)
    assert slugify("Hello  World!") == "hello-world"
    pf = parse_text("FROM a\nREQUIRES 0.9.0\n")
    assert requires_satisfied(pf, "0.9.1") is True
    assert requires_satisfied(pf, "0.8.16") is False
    assert requires_satisfied(pf, None) is None
    assert requires_satisfied(parse_text("FROM a\n"), None) is True
