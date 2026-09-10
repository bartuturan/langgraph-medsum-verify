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
