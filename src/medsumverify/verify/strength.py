"""Claim-strength scorer, graded onto the human 0-3 scale.

The scale mirrors the MSLR annotation scheme so our numbers are directly
comparable to `strength_target` / `strength_generated`:

    0  Insufficient  "there is insufficient evidence to determine..."
    1  Weak          "may be associated with reduced mortality"
    2  Moderate      "is associated with reduced mortality"
    3  Strong        "reduces mortality"

Cues are matched by precedence, not by count: an insufficiency cue beats a
hedge, a hedge beats a moderate marker, and a moderate marker beats a booster.
That ordering is what makes "may be associated with" score 1 rather than 2 --
the hedge scopes over the whole relation. A sentence that asserts an effect
with no cue at all is a bare assertion and scores 3, which is precisely the
distortion this project hunts.

Deliberately unsupervised: nothing here is fitted on the human annotations it
gets validated against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

__all__ = ["ClaimType", "StrengthScore", "score_strength", "aggregate", "is_effect_claim"]


class ClaimType(str, Enum):
    FINDING = "finding"          # asserts something about an effect
    SUFFICIENCY = "sufficiency"  # comments on the state of the evidence itself
    RECOMMENDATION = "recommendation"
    DESCRIPTIVE = "descriptive"  # trial counts, populations -- carries no claim


def _rx(*alts: str) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)


# --- level 0: the evidence is described as inadequate -----------------------
INSUFFICIENCY = _rx(
    r"insufficient\s+(?:evidence|data|information)",
    r"(?:no|lack\s+of|little|scant|absence\s+of)\s+(?:good\s+|reliable\s+|robust\s+|clear\s+)?evidence",
    r"no\s+(?:firm|clear|reliable|robust|definite)\s+conclusions?",
    r"no\s+conclusions?\s+can\s+be\s+drawn",
    r"cannot\s+be\s+(?:drawn|determined|established|assessed)",
    r"could\s+not\s+be\s+(?:determined|established|assessed)",
    r"(?:is|are|remains?)\s+(?:currently\s+)?(?:inconclusive|unclear|uncertain|unknown)",
    r"inconclusive",
    r"unclear",
    r"not\s+clear",
    r"too\s+(?:few|small|limited|little|sparse)",
    # "harms were not assessed in any trial" describes a gap in the evidence
    # base. Without this it parses as a bare assertion and scores 3, which lets
    # an irrelevant sentence license an overclaim elsewhere in the summary.
    r"(?:was|were|is|are)\s+not\s+(?:assessed|measured|reported|evaluated|investigated|studied)",
    r"(?:no|not\s+enough)\s+(?:data|trials?|studies)\s+(?:were|was|are|is)?\s*"
    r"(?:available|reported|identified)?",
    r"did\s+not\s+(?:assess|measure|report|evaluate|investigate)",
    r"(?:further|more|additional|larger|better|higher[\s-]quality)\s+"
    r"(?:research|trials?|studies|evidence|work|data)\s+(?:is|are)\s+"
    r"(?:needed|required|warranted)",
    r"very[\s-]low[\s-](?:quality|certainty)",  # GRADE: very low -> 0
)

# --- level 1: asserted, but hedged ------------------------------------------
HEDGE = _rx(
    r"may", r"might", r"could", r"possibly", r"perhaps", r"potentially",
    r"suggests?", r"suggested", r"suggesting", r"suggestive",
    r"seems?", r"seemed", r"appears?", r"appeared",
    r"(?:some|limited|weak|sparse|preliminary|tentative)\s+evidence",
    r"low[\s-](?:quality|certainty)",  # GRADE: low -> 1
    r"it\s+is\s+possible",
    r"trends?\s+(?:towards?|toward)",
    r"non[\s-]significant",
    r"not\s+statistically\s+significant",
)

# --- level 2: committed but non-causal / explicitly probabilistic ------------
MODERATE = _rx(
    r"(?:is|are|was|were|being)\s+associated\s+with",
    r"associations?\s+(?:with|between)",
    r"correlated\s+with", r"correlation", r"linked\s+to",
    r"probably", r"likely",
    r"moderate[\s-](?:quality|certainty)",  # GRADE: moderate -> 2
    r"indicates?", r"indicated",
)

# --- level 3: explicit boosters ---------------------------------------------
BOOSTER = _rx(
    r"significantly", r"statistically\s+significant", r"significant",
    r"clearly", r"strongly", r"markedly", r"substantially", r"considerably",
    r"demonstrates?", r"demonstrated", r"shows?", r"showed", r"shown",
    r"confirms?", r"confirmed", r"establishes?", r"established",
    r"proves?", r"proven", r"conclusive", r"definitive",
    r"(?:strong|robust|compelling|convincing)\s+evidence",
    r"high[\s-](?:quality|certainty)",  # GRADE: high -> 3
    r"(?:is|are|was|were)\s+(?:effective|efficacious|beneficial|safe)",
)

# --- does this sentence assert an effect at all? ----------------------------
EFFECT = _rx(
    r"reduc\w*", r"lower\w*", r"decreas\w*", r"diminish\w*", r"prevent\w*",
    r"increas\w*", r"rais\w*", r"improv\w*", r"enhanc\w*", r"worsen\w*",
    r"benefit\w*", r"harm\w*", r"effective", r"efficacy", r"effects?",
    r"superior", r"inferior", r"associated",
    # Clinical effect verbs. These must stay in step with direction.py's
    # _UP/_DOWN sets -- a verb known there but unknown here silently demotes a
    # real claim to "descriptive" and it never gets checked at all.
    r"shorten\w*", r"prolong\w*", r"promot\w*", r"delay\w*", r"accelerat\w*",
    r"alleviat\w*", r"reliev\w*", r"resolv\w*", r"suppress\w*", r"eliminat\w*",
    r"minimis\w*", r"minimiz\w*", r"cures?", r"cured", r"heal\w*", r"fewer",
    # A clean null result is a finding, not a description of the study set.
    r"no\s+(?:statistically\s+)?(?:significant\s+)?"
    r"(?:difference|effect|benefit|change|association|advantage)",
    r"did\s+not\s+(?:differ|improve|reduce|increase|affect)",
    r"(?:was|were)\s+not\s+(?:different|superior|inferior)",
    r"similar\s+to", r"comparable", r"equivalent", r"better", r"worse",
)

# "no significant difference" is a confident null finding, not a booster claim.
# Python's re has no variable-width lookbehind, so negated boosters are matched
# and then filtered out by inspecting the preceding text.
_NEGATOR_BEFORE = re.compile(r"(?:\bno\b|\bnot\b|\bnon[\s-]?|\bwithout\b)\s*$", re.IGNORECASE)

SUFFICIENCY_ONLY = _rx(
    r"(?:further|more|additional|larger|better|higher[\s-]quality)\s+"
    r"(?:research|trials?|studies|evidence|work|data)",
    r"insufficient\s+(?:evidence|data)",
    r"no\s+conclusions?\s+can\s+be\s+drawn",
    r"well[\s-]designed\s+(?:trials?|studies)",
)

RECOMMENDATION = _rx(
    r"should\s+(?:be|consider|not)", r"we\s+recommend", r"is\s+recommended",
    r"clinicians?\s+should", r"practitioners?\s+should",
)

# Sentences that only report study inventory carry no claim.
DESCRIPTIVE = _rx(
    r"(?:we\s+)?(?:included|identified|retrieved|screened)\s+\d+",
    r"\d+\s+(?:trials?|studies|participants?|patients?)\s+(?:were|was)\s+"
    r"(?:included|identified|eligible)",
)


def _finding_clauses(sentence: str) -> list[str]:
    """Clauses that assert an effect in their own right.

    "There is insufficient evidence to determine whether X helps" -> []
    "X reduces mortality, but more trials are needed."            -> ["X reduces mortality"]

    Clauses that only talk about the evidence base are dropped, so a trailing
    "more trials are needed" cannot drag a confident claim down to level 0.
    """
    parts = re.split(r"[;,]|\bbut\b|\bhowever\b|\balthough\b", sentence, flags=re.IGNORECASE)
    out = []
    for part in parts:
        if INSUFFICIENCY.search(part) or SUFFICIENCY_ONLY.search(part):
            continue
        if EFFECT.search(part):
            # A subordinate "whether/if" clause is still governed by the hedge
            # in the matrix clause ("unclear whether X improves survival").
            if re.search(r"\b(?:whether|if)\b", part, re.IGNORECASE):
                continue
            out.append(part.strip())
    return out


def _has_standalone_finding(sentence: str) -> bool:
    return bool(_finding_clauses(sentence))


def classify(sentence: str) -> ClaimType:
    """Sentence type. Order matters: sufficiency talk outranks a bare effect verb."""
    if INSUFFICIENCY.search(sentence) or SUFFICIENCY_ONLY.search(sentence):
        # "more trials are needed to establish whether X reduces Y" is a
        # statement about the evidence base, not a finding.
        if not _has_standalone_finding(sentence):
            return ClaimType.SUFFICIENCY
    if RECOMMENDATION.search(sentence) and not EFFECT.search(sentence):
        return ClaimType.RECOMMENDATION
    if EFFECT.search(sentence):
        return ClaimType.FINDING
    return ClaimType.DESCRIPTIVE


def is_effect_claim(sentence: str) -> bool:
    return classify(sentence) is ClaimType.FINDING


@dataclass(frozen=True)
class StrengthScore:
    level: int          # 0-3, directly comparable to the human annotation
    score: float        # continuous, stays inside [level-0.49, level+0.49]
    claim_type: ClaimType
    cues: tuple[str, ...]

    def __repr__(self) -> str:  # keeps traces readable
        return (
            f"<strength {self.level} ({self.score:.2f}) "
            f"{self.claim_type.value} {list(self.cues)}>"
        )


def _found(pattern: re.Pattern, text: str, drop_negated: bool = False) -> list[str]:
    hits = []
    for m in pattern.finditer(text):
        if drop_negated and _NEGATOR_BEFORE.search(text[max(0, m.start() - 12): m.start()]):
            continue  # "no significant difference" is not a booster
        hits.append(m.group(0).lower())
    return hits


# A hedge that frames a whole clause -- "the evidence suggests that X reduces
# Y" -- softens that clause by one step rather than flattening it to "weak".
# Under plain precedence "suggests" outranked everything after it, so an
# overclaim wrapped in the frame ("suggests that X significantly increases Y")
# scored 1, while the same claim without the frame scored 3. Qwen opens most
# drafts with exactly this frame, so the blind spot sat where the experiment's
# overclaims are. A bare "suggests" with no "that" clause is still a hedge.
FRAME_HEDGE = re.compile(
    r"(?<!\w)(?:(?:the|this|these|our|current|available|pooled|overall)\s+)?"
    r"(?:evidence|data|results?|findings|studies|trials?|review|"
    r"analys[ie]s|meta-analys[ie]s)\s+"
    r"(?:\w+ly\s+)?(?:suggests?|suggested|appears?|appeared|seems?|seemed)\s+"
    r"(?:to\s+(?:show|indicate|suggest)\s+)?that(?!\w)"
    r"|(?<!\w)it\s+(?:appears|appeared|seems|seemed)\s+that(?!\w)",
    re.IGNORECASE,
)


def _framed(text: str) -> "StrengthScore | None":
    """Score 'FRAME that CLAUSE' as the clause, one step softer. None if unframed."""
    frame = FRAME_HEDGE.search(text)
    if not frame:
        return None
    inner_text = text[frame.end():].strip(" ,;:")
    if not inner_text:
        return None
    inner = score_strength(inner_text)
    if inner.claim_type not in (ClaimType.FINDING, ClaimType.SUFFICIENCY):
        return None
    if inner.level >= 2:
        level, score = inner.level - 1, inner.score - 1.0
    else:
        # Already weak or insufficient: the frame stacks like a second hedge
        # but cannot push a weak claim down to "insufficient evidence".
        level, score = inner.level, inner.score - (0.12 if inner.level == 1 else 0.0)
    score = max(level - 0.49, min(level + 0.49, score))
    return StrengthScore(level, round(score, 3), inner.claim_type,
                         (frame.group(0).lower(), *inner.cues))


def score_strength(sentence: str) -> StrengthScore:
    """Score one sentence. Precedence: insufficiency > hedge > moderate > booster.

    A clause-framing hedge ("the evidence suggests that ...") is handled first:
    the clause is scored on its own and then softened by one step.
    """
    text = (sentence or "").strip()
    if not text:
        return StrengthScore(0, 0.0, ClaimType.DESCRIPTIVE, ())

    framed = _framed(text)
    if framed is not None:
        return framed

    ctype = classify(text)

    # When a confident claim shares a sentence with an "and more research is
    # needed" aside, score only the clauses that carry the claim.
    span = text
    if ctype is ClaimType.FINDING:
        clauses = _finding_clauses(text)
        if clauses and (INSUFFICIENCY.search(text) or SUFFICIENCY_ONLY.search(text)):
            span = " ; ".join(clauses)

    insuf = _found(INSUFFICIENCY, span)
    hedge = _found(HEDGE, span)
    mod = _found(MODERATE, span)
    boost = _found(BOOSTER, span, drop_negated=True)

    if ctype in (ClaimType.DESCRIPTIVE, ClaimType.RECOMMENDATION):
        # No claim about an effect -- report the type and let aggregation skip it.
        return StrengthScore(0, 0.0, ctype, ())

    if insuf or ctype is ClaimType.SUFFICIENCY:
        level, cues = 0, tuple(insuf)
    elif hedge:
        level, cues = 1, tuple(hedge)
    elif mod:
        level, cues = 2, tuple(mod)
    else:
        # Booster or bare assertion. Both are level 3; the bare assertion is
        # the interesting case -- "reduces mortality" with nothing softening it.
        level, cues = 3, tuple(boost)

    # Within-band nudge so the score is usable for rank correlation and AUROC
    # instead of collapsing ~600 summaries onto four tied values.
    adj = 0.0
    if level == 1:
        adj -= 0.12 * (len(hedge) - 1)          # stacked hedges read weaker
        if insuf:
            adj -= 0.2
    elif level == 3:
        adj += 0.12 * len(boost)                # piled-on boosters read stronger
        if not boost:
            adj -= 0.15                         # bare assertion < explicit booster
    elif level == 2 and boost:
        adj += 0.15
    score = max(level - 0.49, min(level + 0.49, level + adj))
    return StrengthScore(level, round(score, 3), ctype, cues)


AGGREGATIONS = ("max", "mean", "last", "findings_max")


def aggregate(sentences: Iterable[str], how: str = "max") -> float:
    """Summary-level strength over claim-bearing sentences.

    `how` is chosen on the 50 calibration reviews and then frozen; see
    eval/validate_verifier.py. Descriptive and recommendation sentences are
    excluded because they carry no claim to be strong or weak about.
    """
    scores = [score_strength(s) for s in sentences]
    claims = [s for s in scores if s.claim_type in (ClaimType.FINDING, ClaimType.SUFFICIENCY)]
    if not claims:
        return 0.0
    vals = [c.score for c in claims]
    if how == "max":
        return max(vals)
    if how == "mean":
        return sum(vals) / len(vals)
    if how == "last":  # Cochrane conclusions often put the headline claim last
        return vals[-1]
    if how == "findings_max":
        f = [c.score for c in claims if c.claim_type is ClaimType.FINDING]
        return max(f) if f else max(vals)
    raise ValueError(f"unknown aggregation {how!r}")
