"""The Verifier: break a summary into claims and check each one with real tools.

Per claim:

  1. segment      pysbd splits the summary; each sentence is a claim
  2. retrieve     top-k abstract sentences that bear on it
  3. support      MiniCheck(evidence, claim) -> P(supported)
  4. strength     how strongly the claim is *stated*        (0-3)
  5. evidence     how strongly the evidence *supports* it   (0-3, same scorer)

A claim is flagged when it is stated more strongly than its evidence warrants,
or when the fact-checker cannot find support for it, or when it asserts the
opposite direction from the evidence. Using the identical scorer on both sides
is what makes the strength delta mean anything.

Nothing here asks a language model whether the summary looks right.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..config import (
    DEFAULT_THRESHOLDS,
    EVIDENCE_CHAR_CAP,
    RETRIEVE_TOP_K,
    Thresholds,
)
from ..models.factcheck import FactChecker
from ..models.retriever import Retriever, build_evidence, split_sentences
from .direction import Direction, contradicts, detect_direction
from .segment import segment
from .strength import ClaimType, StrengthScore, score_strength

__all__ = ["ClaimReport", "VerificationReport", "Verifier", "relevant_sentences"]

# Flag reasons, kept as constants so the Reviser prompt and the analysis code
# cannot drift apart on spelling.
OVERCLAIM = "overclaim"
UNSUPPORTED = "unsupported"
DIRECTION_FLIP = "direction_flip"


def relevant_sentences(
    ranked: Sequence[tuple[str, float]],
    relevance_floor: float,
    max_sentences: int,
) -> list[str]:
    """Retrieved sentences close enough to the best match to speak to the claim.

    Free of `self` and of `Thresholds` so a finished run can be swept offline:
    rebuild `ranked` as `zip(c["evidence_sentences"], c["evidence_scores"])`
    from a recorded ClaimReport and call this with any candidate cutoffs.

    Note the floor is *relative* to the best score in `ranked`, so this cannot
    tell "three good matches" from "five equally bad ones" -- and the trailing
    fallback guarantees a sentence comes back even when nothing is on topic.
    Detecting that case needs an absolute floor or a per-claim background
    comparison, neither of which is implemented here.
    """
    if not ranked:
        return []
    top = max(score for _, score in ranked)
    if top <= 0:
        return [ranked[0][0]]
    floor = relevance_floor * top
    keep = [s for s, score in ranked if score >= floor]
    return keep[:max_sentences] or [ranked[0][0]]


@dataclass
class ClaimReport:
    index: int
    claim: str
    claim_type: str
    claim_strength: float
    claim_level: int
    evidence_strength: float
    evidence_level: int
    support_prob: float
    strength_delta: float
    claim_direction: str
    evidence_direction: str
    evidence: str
    evidence_sentences: list[str] = field(default_factory=list)
    # Retrieval score per sentence in `evidence_sentences`, same order. Recorded
    # so the relevance cutoffs can be re-fitted offline: `relevant_sentences()`
    # is a pure function of these two lists plus the thresholds, so a sweep over
    # `relevance_floor` / `max_evidence_sentences` replays from a finished run
    # instead of re-running retrieval on a GPU per candidate value. Kept beside
    # `evidence_sentences` rather than folded into it so runs recorded before
    # this field existed still load.
    evidence_scores: list[float] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    def reason(self) -> str:
        """Human-readable explanation, also fed to the Reviser."""
        bits = []
        if OVERCLAIM in self.flags:
            bits.append(
                f"stated at strength {self.claim_level}/3 but the evidence only "
                f"supports {self.evidence_level}/3"
            )
        if UNSUPPORTED in self.flags:
            bits.append(f"the source does not clearly support it (p={self.support_prob:.2f})")
        if DIRECTION_FLIP in self.flags:
            bits.append(
                f"claims a {self.claim_direction} effect where the evidence "
                f"shows {self.evidence_direction}"
            )
        return "; ".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationReport:
    claims: list[ClaimReport] = field(default_factory=list)

    @property
    def flagged(self) -> list[ClaimReport]:
        return [c for c in self.claims if c.flagged]

    @property
    def any_flagged(self) -> bool:
        return bool(self.flagged)

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    def summary_line(self) -> str:
        kinds: dict[str, int] = {}
        for c in self.flagged:
            for f in c.flags:
                kinds[f] = kinds.get(f, 0) + 1
        return f"{len(self.flagged)}/{self.n_claims} claims flagged {kinds or ''}".strip()

    def to_dict(self) -> dict:
        return {"claims": [c.to_dict() for c in self.claims]}


@dataclass
class Verifier:
    factchecker: FactChecker
    retriever: Retriever
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    top_k: int = RETRIEVE_TOP_K
    char_cap: int = EVIDENCE_CHAR_CAP

    def verify(self, summary: str, source_documents: Sequence[str]) -> VerificationReport:
        claims = segment(summary)
        if not claims:
            return VerificationReport()

        source_sentences: list[str] = []
        for doc in source_documents:
            source_sentences.extend(split_sentences(doc))

        evidences, used = [], []
        for claim in claims:
            passage, ranked = build_evidence(
                claim, source_sentences, self.retriever, self.top_k, self.char_cap
            )
            evidences.append(passage)
            used.append(ranked)

        # One batched fact-check call for the whole summary.
        probs = self.factchecker.score(evidences, claims) if evidences else []

        reports: list[ClaimReport] = []
        for i, (claim, passage, ranked) in enumerate(zip(claims, evidences, used)):
            reports.append(
                self._assess(i, claim, passage, ranked, probs[i] if i < len(probs) else 0.0)
            )
        return VerificationReport(reports)

    def _assess(self, index, claim, passage, ranked, support_prob) -> ClaimReport:
        cs: StrengthScore = score_strength(claim)
        ev_sentences = [s for s, _ in ranked]
        ev_sentence_scores = [round(float(sc), 4) for _, sc in ranked]

        # Evidence strength is read only off the sentences that are actually
        # about this claim. Taking max over the whole top-k lets an unrelated
        # retrieved sentence ("harms were not assessed") supply a level-3
        # reading and silently mask a real overclaim, so the pool is first
        # trimmed to sentences close to the best retrieval score.
        relevant = self._relevant(ranked)
        ev_scores = [score_strength(s) for s in relevant]
        ev_claims = [e for e in ev_scores if e.claim_type in (ClaimType.FINDING, ClaimType.SUFFICIENCY)]
        ev_best = max(ev_claims, key=lambda e: e.score) if ev_claims else None
        ev_strength = ev_best.score if ev_best else 0.0
        ev_level = ev_best.level if ev_best else 0

        claim_dir = detect_direction(claim)
        ev_dir = self._evidence_direction(relevant)

        delta = cs.score - ev_strength
        flags: list[str] = []
        # Only claim-bearing sentences can overclaim; "12 trials were included"
        # is not an overstatement of anything.
        checkable = cs.claim_type in (ClaimType.FINDING, ClaimType.SUFFICIENCY)
        if checkable:
            if (cs.level - ev_level) >= self.thresholds.delta_level:
                flags.append(OVERCLAIM)
            if support_prob < self.thresholds.tau_support:
                flags.append(UNSUPPORTED)
            # A claim that asserts no direction cannot flip one, and neither can
            # a statement about the evidence base ("we don't know about harms").
            if cs.claim_type is ClaimType.FINDING and contradicts(claim_dir, ev_dir):
                flags.append(DIRECTION_FLIP)

        return ClaimReport(
            index=index,
            claim=claim,
            claim_type=cs.claim_type.value,
            claim_strength=round(cs.score, 3),
            claim_level=cs.level,
            evidence_strength=round(ev_strength, 3),
            evidence_level=ev_level,
            support_prob=round(float(support_prob), 4),
            strength_delta=round(delta, 3),
            claim_direction=claim_dir.label,
            evidence_direction=ev_dir.label,
            evidence=passage,
            evidence_sentences=ev_sentences,
            evidence_scores=ev_sentence_scores,
            flags=flags,
            cues=list(cs.cues),
        )

    def _relevant(self, ranked: Sequence[tuple[str, float]]) -> list[str]:
        return relevant_sentences(
            ranked,
            self.thresholds.relevance_floor,
            self.thresholds.max_evidence_sentences,
        )

    @staticmethod
    def _evidence_direction(sentences: Sequence[str]) -> Direction:
        """Majority direction across retrieved evidence, ignoring silent sentences."""
        votes: dict[int, int] = {}
        witness: dict[int, str] = {}
        for s in sentences:
            d = detect_direction(s)
            if d.value == -1:
                continue
            votes[d.value] = votes.get(d.value, 0) + 1
            witness.setdefault(d.value, d.evidence)
        if not votes:
            return Direction(-1)
        best = max(votes, key=lambda v: votes[v])
        return Direction(best, witness.get(best, ""))
