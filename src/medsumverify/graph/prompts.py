"""Every prompt in the system, in one file so the writeup can quote them.

The three conditions share the drafter and the reviser verbatim. The only
difference between the self-critique and grounded conditions is where the list
of flagged claims comes from -- a language model's own opinion, or the
verifier's tool output. Keeping the reviser identical is what makes the
comparison about grounding rather than about prompt luck.
"""

from __future__ import annotations

DRAFTER_SYSTEM = (
    "You are a medical evidence summarizer. You write the conclusions section "
    "of a systematic review based only on the trial abstracts you are given. "
    "Match the strength of your language to the strength of the evidence: if "
    "the trials are small, inconsistent, or report non-significant results, say "
    "so. Do not overstate findings."
)

DRAFTER_USER = """Below are the abstracts of the trials included in a systematic review.

{source}

Write the authors' conclusions for this review in 2-4 sentences. State what the \
evidence shows about the intervention's effect. Output only the conclusions, with \
no heading, preamble, or bullet points."""


CRITIC_SYSTEM = (
    "You are a critical reviewer checking a draft summary of medical evidence "
    "for overstatement. You have no tools and no access to the source trials -- "
    "judge from the summary itself."
)

CRITIC_USER = """Here is a draft summary of a systematic review, split into numbered claims.

{numbered_claims}

Identify the claims that are stated more strongly than a cautious reviewer would \
state them: causal language where only an association is warranted, confident \
assertions that should be hedged, or effect directions that look overstated.

Reply with one line per problematic claim, in exactly this format:
CLAIM <number>: <short reason>

If no claim is problematic, reply with the single word NONE."""


# The reviser is shared by both revising conditions. `evidence_block` is filled
# in for the grounded condition and left empty for self-critique -- that
# absence is the experimental manipulation.
REVISER_SYSTEM = (
    "You revise individual claims in medical evidence summaries so that the "
    "strength of the language matches the strength of the evidence. You make "
    "the smallest change that fixes the problem. You never add findings that "
    "are not already present, and you never delete the claim entirely."
)

REVISER_USER = """A claim from a systematic review summary has been flagged.

CLAIM:
{claim}

PROBLEM:
{reason}
{evidence_block}
Rewrite this one claim so its strength matches what the evidence actually supports. \
Keep the same subject and the same outcome. Do not turn it into a statement about \
needing more research unless the evidence genuinely shows nothing. Do not hedge \
further than the evidence requires.

Output only the rewritten claim as a single sentence, with no preamble."""

EVIDENCE_BLOCK = """
WHAT THE SOURCE TRIALS ACTUALLY SAY:
{evidence}
"""


def numbered(claims: list[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
