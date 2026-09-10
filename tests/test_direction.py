"""Effect direction, on the same -1/0/1/2 scale as the human `ed_*` labels."""

from __future__ import annotations

import re

import pytest

from medsumverify.verify import direction as D
from medsumverify.verify.direction import contradicts, detect_direction
from medsumverify.verify.strength import EFFECT

from .fixtures.distortions import DIRECTION_FLIPS


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("The drug reduces mortality.", D.POSITIVE),
        ("The drug increases survival.", D.POSITIVE),
        ("The drug increases mortality.", D.NEGATIVE),
        ("The drug reduces bone growth.", D.NEGATIVE),
        ("There was no significant difference in mortality.", D.NO_EFFECT),
        ("The drug did not reduce mortality.", D.NO_EFFECT),
        ("Outcomes were similar between groups.", D.NO_EFFECT),
        ("The treatment is effective.", D.POSITIVE),
        ("The higher dose increased serious adverse events.", D.NEGATIVE),
        ("Twelve trials were included in this review.", D.NA),
    ],
)
def test_direction_cases(sentence, expected):
    got = detect_direction(sentence)
    assert got.value == expected, f"{sentence!r} -> {got}"


def test_null_result_beats_the_verb():
    """'no significant difference in mortality' must not read as a benefit."""
    assert detect_direction(
        "There was no significant difference in mortality between groups."
    ).value == D.NO_EFFECT


def test_negated_verb_is_a_null_result_not_a_harm():
    assert detect_direction("The drug did not reduce mortality.").value == D.NO_EFFECT


@pytest.mark.parametrize("case", DIRECTION_FLIPS, ids=lambda c: c.name)
def test_planted_flips_are_contradictory(case):
    claim, source = detect_direction(case.claim), detect_direction(case.source)
    assert contradicts(claim, source), f"claim={claim} source={source}"


def test_omission_is_not_a_contradiction():
    """A summary that states no direction has not contradicted anything.

    45% of raw mismatches in the human data are omissions, so conflating the
    two would make the flip detector look like it fires constantly.
    """
    stated = detect_direction("The drug reduces mortality.")
    silent = detect_direction("Twelve trials were included.")
    assert silent.value == D.NA
    assert not contradicts(silent, stated)
    assert not contradicts(stated, silent)


def test_effect_lexicons_stay_in_sync():
    """A verb known to direction.py but not to strength.py silently kills a claim.

    That exact gap ("shorten") demoted a real finding to "descriptive", so it
    never reached the fact-checker. Guard it.
    """
    probes = [
        "shortens hospital stay", "prolongs survival", "promotes healing",
        "relieves pain", "alleviates symptoms", "prevents relapse",
        "reduces mortality", "improves function", "increases survival",
        "minimises complications", "eliminates infection", "delays progression",
    ]
    missing = [p for p in probes if not EFFECT.search(p)]
    assert not missing, f"direction verbs unknown to the strength lexicon: {missing}"


def test_direction_label_roundtrip():
    for v in (D.NA, D.NEGATIVE, D.NO_EFFECT, D.POSITIVE):
        assert D.Direction(v).label
