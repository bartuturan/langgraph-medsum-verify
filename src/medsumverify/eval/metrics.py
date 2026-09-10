"""Per-summary metrics, including the guardrails against degenerate hedging.

The primary metric is deliberately *not* anything the grounded loop optimized.
The loop maximizes MiniCheck support and minimizes its own strength delta
against the retrieved abstracts; scoring it on those would be circular. Instead
every summary is scored against the Cochrane reviewer's own conclusion, which
no condition ever saw, using the scorer validated against human labels.

The guardrails exist because the primary metric has an obvious cheat: a reviser
that prepends "may" to every sentence drives overclaiming to zero while
destroying the summary. A win on strength only counts if direction accuracy and
content overlap hold up.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from ..verify.direction import detect_direction
from ..verify.segment import segment
from ..verify.strength import HEDGE, ClaimType, aggregate, score_strength

__all__ = ["SummaryMetrics", "score_summary", "AGGREGATION"]

# Chosen on the calibration folds in eval/validate_strength.py, then frozen.
AGGREGATION = "mean"

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of and or in on for to is are was were be been being with by that "
    "this these those it its as at from not no there their they which but".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 2}


def content_f1(summary: str, target: str) -> float:
    """Token-level F1 against the reference. A crude but honest coverage check.

    Guardrail, not a quality metric: it only has to notice a summary collapsing
    into contentless hedging.
    """
    s, t = _content_tokens(summary), _content_tokens(target)
    if not s or not t:
        return 0.0
    overlap = len(s & t)
    if not overlap:
        return 0.0
    p, r = overlap / len(s), overlap / len(t)
    return 2 * p * r / (p + r)


@dataclass
class SummaryMetrics:
    review_id: str
    condition: str

    # --- primary: miscalibration against the unseen human conclusion --------
    strength_summary: float
    strength_target: float
    delta_strength: float      # >0 = stated more strongly than the reviewer did
    abs_delta: float           # miscalibration in either direction
    overclaims: bool

    # --- direction ----------------------------------------------------------
    direction_summary: str
    direction_target: str
    direction_match: bool

    # --- guardrails ---------------------------------------------------------
    n_words: int
    n_claims: int
    n_findings: int
    hedge_density: float
    content_f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def score_summary(summary: str, target: str, review_id: str = "", condition: str = "") -> SummaryMetrics:
    claims = segment(summary)
    scores = [score_strength(c) for c in claims]
    findings = [s for s in scores if s.claim_type is ClaimType.FINDING]

    s_sum = aggregate(claims, AGGREGATION)
    s_tgt = aggregate(segment(target), AGGREGATION)
    delta = s_sum - s_tgt

    d_sum = detect_direction(summary)
    d_tgt = detect_direction(target)

    n_hedges = len(HEDGE.findall(summary or ""))
    words = len(_TOKEN.findall(summary or ""))

    return SummaryMetrics(
        review_id=review_id,
        condition=condition,
        strength_summary=round(s_sum, 4),
        strength_target=round(s_tgt, 4),
        delta_strength=round(delta, 4),
        abs_delta=round(abs(delta), 4),
        overclaims=delta > 0,
        direction_summary=d_sum.label,
        direction_target=d_tgt.label,
        direction_match=d_sum.value == d_tgt.value,
        n_words=words,
        n_claims=len(claims),
        n_findings=len(findings),
        hedge_density=round(n_hedges / max(len(claims), 1), 4),
        content_f1=round(content_f1(summary, target), 4),
    )
