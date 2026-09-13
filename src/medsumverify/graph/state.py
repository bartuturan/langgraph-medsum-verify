"""Shared state for the three-role loop."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Condition = Literal["plain", "selfcritique", "grounded"]

CONDITIONS: tuple[str, ...] = ("plain", "selfcritique", "grounded")


class Flag(TypedDict):
    """One claim the Reviser is asked to fix.

    Produced by the Verifier (grounded) or by the model critiquing itself
    (selfcritique). Identical shape either way, so the Reviser cannot tell
    which condition it is in except by whether `evidence` is populated.
    """

    index: int
    claim: str
    reason: str
    evidence: str  # "" in the self-critique condition -- that is the manipulation


class LoopState(TypedDict, total=False):
    review_id: str
    condition: str
    source_documents: list[str]

    draft: str        # round-0 draft, identical across all three conditions
    summary: str      # current text
    claims: list[str]

    round: int
    max_rounds: int

    flags: list[Flag]
    reports: list[dict[str, Any]]   # verifier output, one per round
    critiques: list[str]            # raw self-critique text, one per round
    history: list[dict[str, Any]]   # {round, summary, n_flagged, n_claims}
    # One entry per flag the Reviser acted on: what it proposed, what the gate
    # did about it, and the strength either side. `outcome` is accepted,
    # retried or rejected -- so "how often did the gate save us" is a number
    # that can be read straight off the results.
    repairs: list[dict[str, Any]]

    n_llm_calls: int
    error: str


def initial_state(
    review_id: str,
    source_documents: list[str],
    condition: str,
    draft: str = "",
    max_rounds: int = 2,
) -> LoopState:
    return LoopState(
        review_id=review_id,
        condition=condition,
        source_documents=list(source_documents),
        draft=draft,
        summary=draft,
        claims=[],
        round=0,
        max_rounds=max_rounds,
        flags=[],
        reports=[],
        critiques=[],
        history=[],
        repairs=[],
        n_llm_calls=0,
    )
