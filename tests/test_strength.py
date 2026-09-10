"""The strength scorer is the project's measuring instrument -- test it hard."""

from __future__ import annotations

import pytest

from medsumverify.verify.strength import (
    ClaimType,
    aggregate,
    classify,
    score_strength,
)

from .fixtures.distortions import CASES, FAITHFUL, STRENGTHENED, WEAKENED

LADDER = [
    "There is insufficient evidence to determine whether aspirin reduces mortality.",
    "Aspirin may be associated with reduced mortality.",
    "Aspirin is associated with reduced mortality.",
    "Aspirin reduces mortality.",
    "Aspirin significantly reduces mortality.",
]


def test_ladder_levels():
    assert [score_strength(s).level for s in LADDER] == [0, 1, 2, 3, 3]


def test_ladder_is_strictly_monotone():
    """Continuous scores must order the ladder, not just bucket it."""
    scores = [score_strength(s).score for s in LADDER]
    assert scores == sorted(scores), scores
    assert len(set(scores)) == len(scores), f"ties break rank correlation: {scores}"


def test_continuous_score_stays_inside_its_band():
    for s in LADDER:
        sc = score_strength(s)
        assert abs(sc.score - sc.level) <= 0.49, sc


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Twelve trials involving 3400 participants were included.", ClaimType.DESCRIPTIVE),
        ("Further well-designed trials are needed.", ClaimType.SUFFICIENCY),
        ("Clinicians should consider offering this treatment.", ClaimType.RECOMMENDATION),
        ("There was no significant difference between groups.", ClaimType.FINDING),
        ("The drug reduces mortality.", ClaimType.FINDING),
    ],
)
def test_claim_typing(sentence, expected):
    assert classify(sentence) is expected


def test_null_result_is_a_confident_finding():
    """A clean null result is a claim, not a description, and is not hedged."""
    sc = score_strength("There was no significant difference between groups.")
    assert sc.claim_type is ClaimType.FINDING
    assert sc.level == 3
    # "significant" inside "no significant" must not be logged as a booster.
    assert "significant" not in sc.cues


def test_trailing_research_caveat_does_not_zero_the_claim():
    """'X reduces mortality, but more trials are needed' is still a strong claim."""
    assert score_strength("X reduces mortality, but more trials are needed.").level == 3
    # ...while a claim that only exists inside the caveat is not.
    assert score_strength(
        "More trials are needed to determine whether X reduces mortality."
    ).level == 0


def test_subordinate_clause_stays_governed_by_the_hedge():
    assert score_strength("It is unclear whether the treatment improves survival.").level == 0


def test_stacked_hedges_read_weaker_than_one():
    one = score_strength("The intervention may improve outcomes.")
    two = score_strength("The intervention may possibly improve outcomes.")
    assert two.score < one.score
    assert one.level == two.level == 1


def test_bare_assertion_scores_below_explicit_booster():
    bare = score_strength("Aspirin reduces mortality.")
    boosted = score_strength("Aspirin significantly reduces mortality.")
    assert bare.level == boosted.level == 3
    assert bare.score < boosted.score


def test_grade_certainty_maps_onto_the_scale():
    levels = [
        score_strength("Very low certainty evidence for an effect on mortality.").level,
        score_strength("Low certainty evidence of reduced mortality.").level,
        score_strength("Moderate certainty evidence of reduced mortality.").level,
        score_strength("High certainty evidence of reduced mortality.").level,
    ]
    assert levels == [0, 1, 2, 3]


@pytest.mark.parametrize("case", STRENGTHENED, ids=lambda c: c.name)
def test_strengthened_cases_score_above_their_source(case):
    assert score_strength(case.claim).score > score_strength(case.source).score


@pytest.mark.parametrize("case", FAITHFUL, ids=lambda c: c.name)
def test_faithful_cases_do_not_score_above_their_source(case):
    claim, source = score_strength(case.claim), score_strength(case.source)
    assert claim.level <= source.level, (claim, source)


@pytest.mark.parametrize("case", WEAKENED, ids=lambda c: c.name)
def test_weakened_cases_score_below_their_source(case):
    """Over-hedging is the other failure mode and must be visible too."""
    assert score_strength(case.claim).score < score_strength(case.source).score


def test_aggregate_skips_non_claims():
    sents = [
        "Twelve trials involving 3400 participants were included.",
        "The intervention may reduce mortality.",
    ]
    assert aggregate(sents, "max") == score_strength(sents[1]).score


def test_aggregate_rejects_unknown_mode():
    with pytest.raises(ValueError):
        aggregate(["The drug reduces mortality."], "nonsense")


def test_empty_and_junk_input():
    for junk in ("", "   ", "\n"):
        assert score_strength(junk).level == 0
    assert aggregate([]) == 0.0


def test_every_fixture_case_is_scoreable():
    for case in CASES:
        assert 0 <= score_strength(case.claim).level <= 3
        assert 0 <= score_strength(case.source).level <= 3
