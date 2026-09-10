"""Effect direction: does the claim say the intervention helped, harmed, or did nothing?

Encoded on the MSLR `ed_*` scale so it lines up with the human labels:

    -1  N/A        no direction asserted
     0  Negative   intervention looks worse
     1  No effect  no difference found
     2  Positive   intervention looks better

The hard part is that direction is not readable off the verb alone: "reduces
mortality" is positive but "reduces bone density" is negative. Direction is the
verb combined with whether the outcome is something you want more or less of,
so the lexicon pairs an up/down verb with an outcome-valence noun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Direction", "detect_direction", "contradicts"]

NA, NEGATIVE, NO_EFFECT, POSITIVE = -1, 0, 1, 2

_WORD = re.compile(r"[a-z]+")

# Outcomes you want less of.
_BAD = {
    "mortality", "death", "deaths", "died", "morbidity", "pain", "relapse",
    "recurrence", "infection", "infections", "complication", "complications",
    "adverse", "bleeding", "stroke", "admission", "admissions", "readmission",
    "hospitalisation", "hospitalization", "symptoms", "symptom", "risk",
    "failure", "dropout", "withdrawals", "toxicity", "nausea", "vomiting",
    "preterm", "prematurity", "disability", "fracture", "fractures", "seizures",
    "depression", "anxiety", "bleeding", "thrombosis", "injury", "harm", "harms",
    "exacerbations", "duration", "length",
}
# Outcomes you want more of.
_GOOD = {
    "survival", "recovery", "remission", "cure", "healing", "function",
    "functioning", "mobility", "continence", "satisfaction", "adherence",
    "compliance", "cognition", "growth", "weight", "birthweight", "quality",
    "wellbeing", "independence", "strength", "performance", "success",
    "effectiveness", "response", "improvement", "outcomes", "outcome",
}

_DOWN = re.compile(r"\b(?:reduc\w*|lower\w*|decreas\w*|diminish\w*|prevent\w*|"
                   r"fewer|less|shorten\w*|minimis\w*|minimiz\w*)\b", re.I)
_UP = re.compile(r"\b(?:increas\w*|rais\w*|improv\w*|enhanc\w*|greater|higher|"
                 r"more|prolong\w*|promot\w*)\b", re.I)

_NO_EFFECT = re.compile(
    r"\b(?:no\s+(?:statistically\s+)?(?:significant\s+)?"
    r"(?:difference|effect|benefit|change|association|advantage|evidence\s+of\s+(?:a\s+)?(?:difference|effect))|"
    r"did\s+not\s+(?:differ|improve|reduce|increase|affect|change)|"
    r"(?:was|were)\s+not\s+(?:different|superior|inferior|effective)|"
    r"similar\s+(?:to|in|between|across)|comparable\s+(?:to|with)|equivalent\s+to|"
    r"little\s+or\s+no\s+(?:difference|effect))\b",
    re.I,
)
_POSITIVE_BARE = re.compile(
    r"\b(?:(?:is|are|was|were|appears?|seems?)\s+(?:to\s+be\s+)?"
    r"(?:effective|efficacious|beneficial|superior|safe)|"
    r"benefit\w*|favour\w*|favor\w*)\b", re.I,
)
# Harm must be *asserted*, not merely mentioned: "insufficient evidence
# regarding long-term harms" names a topic, it does not claim the drug harms.
_NEGATIVE_BARE = re.compile(
    r"\b(?:(?:is|are|was|were)\s+(?:harmful|unsafe|inferior|ineffective|toxic)|"
    r"caus\w*\s+(?:harm|significant\s+harm)|resulted\s+in\s+harm|"
    r"(?:increased|more|greater)\s+(?:harm|adverse\s+(?:effects?|events?)|toxicity))\b",
    re.I,
)
_NEGATED = re.compile(r"\b(?:not|no|never|neither|nor|failed\s+to|did\s+not)\b", re.I)

_WINDOW_AFTER, _WINDOW_BEFORE = 8, 4


@dataclass(frozen=True)
class Direction:
    value: int
    evidence: str = ""

    @property
    def label(self) -> str:
        return {NA: "N/A", NEGATIVE: "Negative", NO_EFFECT: "NoEffect", POSITIVE: "Positive"}[self.value]

    def __repr__(self) -> str:
        return f"<dir {self.label}{' ' + self.evidence if self.evidence else ''}>"


def _valence_near(tokens: list[str], i: int) -> tuple[str | None, str | None]:
    """Nearest outcome noun to the verb at position i, and its valence."""
    lo, hi = max(0, i - _WINDOW_BEFORE), min(len(tokens), i + _WINDOW_AFTER + 1)
    best: tuple[int, str, str] | None = None
    for j in range(lo, hi):
        tok = tokens[j]
        val = "bad" if tok in _BAD else "good" if tok in _GOOD else None
        if val is None:
            continue
        dist = abs(j - i)
        if best is None or dist < best[0]:
            best = (dist, val, tok)
    return (best[1], best[2]) if best else (None, None)


def detect_direction(sentence: str) -> Direction:
    text = (sentence or "").strip()
    if not text:
        return Direction(NA)

    # An explicit null result wins: "no significant difference in mortality"
    # must not be read as a downward effect on a bad outcome.
    m = _NO_EFFECT.search(text)
    if m:
        return Direction(NO_EFFECT, m.group(0).lower())

    tokens = _WORD.findall(text.lower())

    for pattern, verb_dir in ((_DOWN, "down"), (_UP, "up")):
        for m in pattern.finditer(text):
            word = m.group(0).lower()
            try:
                i = tokens.index(word)
            except ValueError:
                continue
            # "did not reduce mortality" is a null result, not a benefit.
            prefix = text[max(0, m.start() - 30): m.start()]
            negated = bool(_NEGATED.search(prefix))
            valence, noun = _valence_near(tokens, i)
            if valence is None:
                continue
            if negated:
                return Direction(NO_EFFECT, f"not {word} {noun}")
            good = (verb_dir == "down" and valence == "bad") or (
                verb_dir == "up" and valence == "good"
            )
            return Direction(POSITIVE if good else NEGATIVE, f"{word} {noun}")

    m = _NEGATIVE_BARE.search(text)
    if m and not _NEGATED.search(text[max(0, m.start() - 20): m.start()]):
        return Direction(NEGATIVE, m.group(0).lower())
    m = _POSITIVE_BARE.search(text)
    if m:
        if _NEGATED.search(text[max(0, m.start() - 20): m.start()]):
            return Direction(NO_EFFECT, f"not {m.group(0).lower()}")
        return Direction(POSITIVE, m.group(0).lower())

    # A bare up/down verb with no identifiable outcome still signals improvement.
    if _UP.search(text) and re.search(r"\bimprov\w*", text, re.I):
        return Direction(POSITIVE, "improve")
    return Direction(NA)


def contradicts(claim_dir: Direction, evidence_dir: Direction) -> bool:
    """True only for a genuine flip -- omission on either side is not a contradiction."""
    if claim_dir.value == NA or evidence_dir.value == NA:
        return False
    return claim_dir.value != evidence_dir.value
