"""Loop plumbing, on CPU with FakeWriter -- routing, round cap, passthrough."""

from __future__ import annotations

import pytest

from medsumverify.graph.build import build_graph, run_condition
from medsumverify.graph.nodes import parse_critique, should_continue
from medsumverify.graph.state import CONDITIONS, initial_state
from medsumverify.models.factcheck import FakeFactChecker
from medsumverify.models.retriever import FakeRetriever
from medsumverify.models.writer import DEFAULT_FAKE_DRAFT, FakeWriter
from medsumverify.verify.verifier import Verifier

from .fixtures.distortions import FAKE_SOURCE

SOURCE = [FAKE_SOURCE]


@pytest.fixture
def verifier():
    return Verifier(factchecker=FakeFactChecker(), retriever=FakeRetriever())


def _run(condition, verifier=None, draft=DEFAULT_FAKE_DRAFT, max_rounds=2):
    return run_condition(
        condition, "CD0001", SOURCE, FakeWriter(), verifier, draft, max_rounds
    )


def test_plain_returns_the_draft_untouched(verifier):
    out = _run("plain")
    assert out["summary"] == DEFAULT_FAKE_DRAFT
    assert out["round"] == 0


def test_grounded_revises_and_rechecks(verifier):
    out = _run("grounded", verifier)
    assert out["summary"] != DEFAULT_FAKE_DRAFT, "nothing was revised"
    # One report per critique pass: initial check plus a re-check after revising.
    assert len(out["reports"]) >= 2, "final summary was never re-verified"


def test_round_cap_is_respected(verifier):
    out = _run("grounded", verifier, max_rounds=2)
    assert out["round"] <= 2


@pytest.mark.parametrize("max_rounds", [1, 2, 3])
def test_round_cap_is_configurable(verifier, max_rounds):
    out = _run("grounded", verifier, max_rounds=max_rounds)
    assert out["round"] <= max_rounds


def test_unflagged_claims_pass_through_verbatim(verifier):
    """The Reviser must touch only what was flagged."""
    out = _run("grounded", verifier)
    before = out["history"][0]["summary"]
    from medsumverify.verify.segment import segment

    orig = segment(before)
    final = segment(out["summary"])
    assert len(orig) == len(final), "revision changed the number of claims"
    # The study-count sentence carries no claim and must be byte-identical.
    assert orig[0] == final[0], (orig[0], final[0])


def test_all_three_conditions_share_one_draft(verifier):
    """The paired design collapses if the conditions start from different text."""
    outs = {c: _run(c, verifier) for c in CONDITIONS}
    drafts = {c: o["history"][0]["summary"] for c, o in outs.items()}
    assert len(set(drafts.values())) == 1, drafts


def test_selfcritique_runs_without_a_verifier():
    out = _run("selfcritique", verifier=None)
    assert out["critiques"], "the control condition produced no critique"


def test_grounded_requires_a_verifier():
    with pytest.raises(ValueError):
        build_graph("grounded", FakeWriter(), None)


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        build_graph("nonsense", FakeWriter())


def test_reviser_gets_evidence_only_in_the_grounded_condition(verifier):
    grounded = _run("grounded", verifier)
    flags = [f for rep in grounded["reports"] for f in rep["claims"] if f["flags"]]
    assert any(f["evidence"] for f in flags), "grounded flags carried no evidence"

    critique = _run("selfcritique")
    # Self-critique flags are parsed from text and carry no retrieved evidence.
    parsed = parse_critique("CLAIM 1: too strong", ["The drug reduces mortality."])
    assert parsed[0]["evidence"] == ""


def test_routing_stops_when_nothing_is_flagged():
    state = initial_state("x", SOURCE, "grounded", "draft", 2)
    state["flags"] = []
    assert should_continue(state) == "end"
    state["flags"] = [{"index": 0, "claim": "c", "reason": "r", "evidence": "e"}]
    assert should_continue(state) == "revise"
    state["round"] = 2
    assert should_continue(state) == "end", "round cap must beat outstanding flags"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("NONE", 0),
        ("CLAIM 1: overstated\nCLAIM 2: causal language", 2),
        ("CLAIM 1: a\nCLAIM 1: duplicate", 1),
        ("CLAIM 99: out of range", 0),
        ("unparseable rambling", 0),
        ("", 0),
    ],
)
def test_parse_critique_is_forgiving(text, expected):
    claims = ["one", "two", "three"]
    assert len(parse_critique(text, claims)) == expected


def test_empty_draft_does_not_crash(verifier):
    out = run_condition("grounded", "CD0002", SOURCE, FakeWriter(), verifier, " ", 2)
    assert "summary" in out
