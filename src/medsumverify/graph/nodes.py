"""The three roles, as pure functions.

No LangGraph imports here on purpose: the roles are ordinary callables that map
state to state, so they can be unit-tested on CPU without the framework, and
graph/build.py wraps them in a StateGraph.
"""

from __future__ import annotations

import re
from typing import Callable

from ..models.writer import Writer
from ..verify.segment import clean_summary, segment
from ..verify.verifier import Verifier
from . import prompts
from .state import Flag, LoopState

__all__ = ["make_drafter", "make_verifier_node", "make_self_critic", "make_reviser", "should_continue"]


# --------------------------------------------------------------------------
# Drafter
# --------------------------------------------------------------------------


# 20k chars is roughly 5k tokens, which fits ~85% of the selected dev reviews
# whole. Median source is 11.2k chars; the 12k default was silently truncating
# half of them, which would have meant the drafter never saw evidence the
# verifier was later checking its claims against.
def make_drafter(writer: Writer, source_chars: int = 20000) -> Callable[[LoopState], dict]:
    def drafter(state: LoopState) -> dict:
        # The paired design depends on all three conditions starting from the
        # same Round-0 text, so a pre-supplied draft is used as-is.
        if state.get("draft"):
            text = state["draft"]
            calls = 0
        else:
            source = "\n\n".join(state["source_documents"])[:source_chars]
            text = writer.chat(
                prompts.DRAFTER_SYSTEM,
                prompts.DRAFTER_USER.format(source=source),
                max_new_tokens=320,
                role="draft",
            )
            text = clean_summary(text)
            calls = 1
        claims = segment(text)
        return {
            "draft": text,
            "summary": text,
            "claims": claims,
            "n_llm_calls": state.get("n_llm_calls", 0) + calls,
            "history": [*state.get("history", []),
                        {"round": 0, "summary": text, "n_claims": len(claims), "n_flagged": 0}],
        }

    return drafter


# --------------------------------------------------------------------------
# Verifier (grounded condition): flags come from real checks
# --------------------------------------------------------------------------


def make_verifier_node(verifier: Verifier) -> Callable[[LoopState], dict]:
    def verify(state: LoopState) -> dict:
        report = verifier.verify(state["summary"], state["source_documents"])
        flags: list[Flag] = [
            Flag(index=c.index, claim=c.claim, reason=c.reason(), evidence=c.evidence)
            for c in report.flagged
        ]
        return {
            "claims": [c.claim for c in report.claims],
            "flags": flags,
            "reports": [*state.get("reports", []), report.to_dict()],
        }

    return verify


# --------------------------------------------------------------------------
# Self-critic (control condition): flags come from the model's own opinion
# --------------------------------------------------------------------------

_CLAIM_LINE = re.compile(r"CLAIM\s*(\d+)\s*[:.\-]\s*(.+)", re.IGNORECASE)


def parse_critique(text: str, claims: list[str]) -> list[Flag]:
    """Turn free-text critique into the same Flag shape the Verifier emits.

    Deliberately forgiving: a 3B model will not always honour the format, and
    silently dropping its output would hand the grounded condition an unfair
    win by making the control do nothing.
    """
    if not text or re.search(r"\bNONE\b", text.strip()[:40], re.IGNORECASE):
        return []
    flags: list[Flag] = []
    seen: set[int] = set()
    for line in text.splitlines():
        m = _CLAIM_LINE.search(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1  # prompt numbers from 1
        if idx < 0 or idx >= len(claims) or idx in seen:
            continue
        seen.add(idx)
        flags.append(
            Flag(index=idx, claim=claims[idx], reason=m.group(2).strip(), evidence="")
        )
    return flags


def make_self_critic(writer: Writer) -> Callable[[LoopState], dict]:
    def critic(state: LoopState) -> dict:
        claims = state.get("claims") or segment(state["summary"])
        if not claims:
            return {"flags": [], "critiques": [*state.get("critiques", []), ""]}
        text = writer.chat(
            prompts.CRITIC_SYSTEM,
            prompts.CRITIC_USER.format(numbered_claims=prompts.numbered(claims)),
            max_new_tokens=256,
            role="critique",
        )
        return {
            "claims": claims,
            "flags": parse_critique(text, claims),
            "critiques": [*state.get("critiques", []), text],
            "n_llm_calls": state.get("n_llm_calls", 0) + 1,
        }

    return critic


# --------------------------------------------------------------------------
# Reviser: shared by both revising conditions
# --------------------------------------------------------------------------


def make_reviser(writer: Writer) -> Callable[[LoopState], dict]:
    def reviser(state: LoopState) -> dict:
        claims = list(state.get("claims") or segment(state["summary"]))
        flags = state.get("flags", [])
        if not claims or not flags:
            return {"round": state.get("round", 0) + 1}

        calls = 0
        for flag in flags:
            i = flag["index"]
            if not (0 <= i < len(claims)):
                continue
            block = (
                prompts.EVIDENCE_BLOCK.format(evidence=flag["evidence"])
                if flag.get("evidence") else ""
            )
            rewritten = writer.chat(
                prompts.REVISER_SYSTEM,
                prompts.REVISER_USER.format(
                    claim=flag["claim"], reason=flag["reason"] or "stated too strongly",
                    evidence_block=block,
                ),
                max_new_tokens=120,
                role="revise",
            )
            calls += 1
            rewritten = _first_sentence(clean_summary(rewritten))
            if rewritten:
                claims[i] = rewritten

        summary = " ".join(claims).strip()
        rnd = state.get("round", 0) + 1
        return {
            "summary": summary,
            "claims": claims,
            "round": rnd,
            "flags": [],
            "n_llm_calls": state.get("n_llm_calls", 0) + calls,
            "history": [*state.get("history", []),
                        {"round": rnd, "summary": summary,
                         "n_claims": len(claims), "n_flagged": len(flags)}],
        }

    return reviser


def _first_sentence(text: str) -> str:
    """Keep the model's rewrite to one sentence; it sometimes adds commentary."""
    text = (text or "").strip().strip('"').strip()
    if not text:
        return ""
    sents = segment(text, min_chars=1)
    return sents[0].strip() if sents else text


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def should_continue(state: LoopState) -> str:
    """Revise while something is flagged and the round budget is not spent."""
    if state.get("round", 0) >= state.get("max_rounds", 2):
        return "end"
    return "revise" if state.get("flags") else "end"
