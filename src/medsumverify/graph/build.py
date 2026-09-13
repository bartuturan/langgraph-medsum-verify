"""The three conditions, as LangGraph state graphs.

    plain          draft -> END
    selfcritique   draft -> critic   -> reviser -> critic   -> ... (2 rounds max)
    grounded       draft -> verifier -> reviser -> verifier -> ... (2 rounds max)

The last two are the same graph with one node swapped. That is the experiment:
the critique step either comes from the model's own opinion or from real
checks, and everything downstream of it is held constant.
"""

from __future__ import annotations

from typing import Callable

from ..models.writer import Writer
from ..verify.verifier import Verifier
from .nodes import (
    make_drafter,
    make_reviser,
    make_self_critic,
    make_verifier_node,
    should_continue,
)
from .state import CONDITIONS, LoopState, initial_state

__all__ = ["build_graph", "run_condition", "CONDITIONS"]


def build_graph(
    condition: str,
    writer: Writer,
    verifier: Verifier | None = None,
    gate: bool = True,
):
    """`gate` is passed straight to the reviser and so applies to both revising
    conditions; gating only the grounded one would confound the gate with the
    grounding it is supposed to be helping."""
    from langgraph.graph import END, START, StateGraph

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if condition == "grounded" and verifier is None:
        raise ValueError("the grounded condition needs a verifier")

    g = StateGraph(LoopState)
    g.add_node("draft", make_drafter(writer))

    if condition == "plain":
        g.add_edge(START, "draft")
        g.add_edge("draft", END)
        return g.compile()

    critique_node = (
        make_verifier_node(verifier) if condition == "grounded" else make_self_critic(writer)
    )
    g.add_node("critique", critique_node)
    g.add_node("revise", make_reviser(writer, gate=gate))

    g.add_edge(START, "draft")
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", should_continue, {"revise": "revise", "end": END})
    # Back to the critique step so the final summary is always re-checked --
    # that is what makes "was it fixed?" answerable.
    g.add_edge("revise", "critique")
    return g.compile()


def run_condition(
    condition: str,
    review_id: str,
    source_documents: list[str],
    writer: Writer,
    verifier: Verifier | None = None,
    draft: str = "",
    max_rounds: int = 2,
    gate: bool = True,
) -> LoopState:
    """Run one condition on one review. `draft` pins the shared Round-0 text."""
    graph = build_graph(condition, writer, verifier, gate=gate)
    state = initial_state(review_id, source_documents, condition, draft, max_rounds)
    # recursion_limit guards against a routing bug looping forever; the real
    # stopping rule is should_continue.
    return graph.invoke(state, {"recursion_limit": 4 * max_rounds + 8})
