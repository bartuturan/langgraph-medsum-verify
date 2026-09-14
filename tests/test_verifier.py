"""Verifier behaviour, exercised on CPU with the fake fact-checker and retriever."""

from __future__ import annotations

import pytest

from medsumverify.models.factcheck import FakeFactChecker
from medsumverify.models.retriever import FakeRetriever, split_sentences
from medsumverify.verify.verifier import (
    DIRECTION_FLIP,
    OVERCLAIM,
    Verifier,
)

from .fixtures.distortions import CASES, FAITHFUL, FAKE_SOURCE, STRENGTHENED


@pytest.fixture(scope="module")
def verifier():
    return Verifier(factchecker=FakeFactChecker(), retriever=FakeRetriever())


@pytest.mark.parametrize("case", STRENGTHENED, ids=lambda c: c.name)
def test_strengthened_claims_are_flagged(verifier, case):
    rep = verifier.verify(case.claim, [case.source])
    assert rep.any_flagged, f"missed: {case.claim!r} vs {case.source!r}"
    assert OVERCLAIM in rep.claims[0].flags or DIRECTION_FLIP in rep.claims[0].flags


@pytest.mark.parametrize("case", FAITHFUL, ids=lambda c: c.name)
def test_faithful_claims_are_not_overclaim_flagged(verifier, case):
    rep = verifier.verify(case.claim, [case.source])
    assert OVERCLAIM not in rep.claims[0].flags, rep.claims[0]


def test_descriptive_sentences_are_never_flagged(verifier):
    rep = verifier.verify("Twelve trials involving 3400 participants were included.", [FAKE_SOURCE])
    assert rep.claims[0].claim_type == "descriptive"
    assert not rep.claims[0].flagged


def test_multi_claim_summary_flags_only_the_bad_ones(verifier):
    from medsumverify.models.writer import DEFAULT_FAKE_DRAFT

    rep = verifier.verify(DEFAULT_FAKE_DRAFT, [FAKE_SOURCE])
    assert rep.n_claims == 4
    flagged = {c.index for c in rep.flagged}
    # "The intervention reduces mortality" overstates a hedged source.
    assert 1 in flagged
    # The study-count sentence must not be flagged.
    assert 0 not in flagged


def test_report_carries_evidence_and_a_reason(verifier):
    rep = verifier.verify("The intervention reduces mortality.", [FAKE_SOURCE])
    c = rep.claims[0]
    assert c.evidence, "reviser needs the retrieved evidence to work from"
    assert c.reason(), "every flag must explain itself"
    assert c.evidence_sentences


def test_empty_summary_is_handled(verifier):
    rep = verifier.verify("", [FAKE_SOURCE])
    assert rep.n_claims == 0 and not rep.any_flagged


def test_empty_source_does_not_crash(verifier):
    rep = verifier.verify("The intervention reduces mortality.", [])
    assert rep.n_claims == 1


def test_report_is_json_serialisable(verifier):
    import json

    rep = verifier.verify("The intervention reduces mortality.", [FAKE_SOURCE])
    assert json.loads(json.dumps(rep.to_dict()))["claims"][0]["claim"]


def test_split_sentences_handles_abbreviations():
    sents = split_sentences("Patients received 5 mg vs. placebo. Outcomes were measured at 12 weeks.")
    assert len(sents) == 2, sents


def test_overall_discrimination_on_fixtures(verifier):
    """The verifier must separate planted distortions from faithful controls."""
    flagged_bad = sum(
        1 for c in CASES if c.overclaim and verifier.verify(c.claim, [c.source]).any_flagged
    )
    n_bad = sum(1 for c in CASES if c.overclaim)
    false_alarms = sum(
        1 for c in CASES
        if c.kind == "faithful"
        and OVERCLAIM in (verifier.verify(c.claim, [c.source]).claims or [None])[0].flags
    )
    assert flagged_bad == n_bad, f"caught only {flagged_bad}/{n_bad} planted distortions"
    assert false_alarms == 0, f"{false_alarms} faithful claims wrongly called overclaims"


# --------------------------------------------------------------------------
# Retrieval scores are recorded so the relevance cutoffs can be re-fitted
# offline instead of re-running retrieval on a GPU per candidate value.
# --------------------------------------------------------------------------


def test_every_evidence_sentence_carries_its_score(verifier):
    report = verifier.verify("Aspirin reduces mortality.", [FAKE_SOURCE])
    for c in report.claims:
        assert len(c.evidence_scores) == len(c.evidence_sentences)
        assert all(isinstance(s, float) for s in c.evidence_scores)


def test_scores_are_in_retrieval_order(verifier):
    """Descending, so `evidence_scores[0]` is the top match."""
    report = verifier.verify("Aspirin reduces mortality.", [FAKE_SOURCE])
    for c in report.claims:
        assert c.evidence_scores == sorted(c.evidence_scores, reverse=True)


def test_scores_survive_the_json_round_trip(verifier):
    """They are only useful if they reach results/experiment.jsonl."""
    import json

    report = verifier.verify("Aspirin reduces mortality.", [FAKE_SOURCE])
    back = json.loads(json.dumps(report.to_dict()))
    for c in back["claims"]:
        assert len(c["evidence_scores"]) == len(c["evidence_sentences"])


def test_a_recorded_report_can_be_reswept_offline(verifier):
    """The point of the field: replay `relevant_sentences` from a finished run.

    Rebuilding `ranked` from the two recorded lists must reproduce exactly what
    the verifier computed at the recorded thresholds -- otherwise an offline
    sweep would be calibrating against something the run never did.
    """
    from medsumverify.verify.verifier import relevant_sentences

    report = verifier.verify("Aspirin reduces mortality.", [FAKE_SOURCE])
    th = verifier.thresholds
    for c in report.claims:
        ranked = list(zip(c.evidence_sentences, c.evidence_scores))
        replayed = relevant_sentences(ranked, th.relevance_floor, th.max_evidence_sentences)
        assert replayed == verifier._relevant(ranked)


def test_a_tighter_floor_keeps_no_more_than_a_looser_one():
    """What a sweep is for: the cutoff has to actually bite, monotonically."""
    from medsumverify.verify.verifier import relevant_sentences

    ranked = [("a", 0.90), ("b", 0.80), ("c", 0.55), ("d", 0.40), ("e", 0.10)]
    prev = None
    for floor in (0.0, 0.5, 0.7, 0.9, 1.0):
        keep = relevant_sentences(ranked, floor, max_sentences=5)
        if prev is not None:
            assert len(keep) <= len(prev)
        prev = keep
    assert relevant_sentences(ranked, 1.0, 5) == ["a"]


def test_the_fallback_returns_the_top_match_when_nothing_clears_the_floor():
    """Documents the gap: abstention is not currently possible.

    The floor is relative, so an all-junk retrieval still yields a sentence.
    Pinned so that adding an absolute floor later is a deliberate change.
    """
    from medsumverify.verify.verifier import relevant_sentences

    assert relevant_sentences([("junk", 0.02), ("worse", 0.01)], 0.6, 3) == ["junk"]
    assert relevant_sentences([("a", 0.0), ("b", 0.0)], 0.6, 3) == ["a"]
    assert relevant_sentences([], 0.6, 3) == []
