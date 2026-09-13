"""What the no-strengthen gate did, and what it would have done.

Two things live here, and they answer different questions.

`report` reads the `repairs` recorded by graph.nodes.make_reviser during a
*gated* run: how many rewrites were accepted, fixed on the retry, or thrown
away, split by condition and by flag kind. That is the "how often did the gate
save us" number, and it is also the diagnostic that tells you which of two
things you are looking at -- a Reviser that overshoots, or flags that are wrong
in the first place. A gate that rejects almost everything is evidence for the
second.

`replay` works on an *ungated* run. It walks the recorded rounds, reverts every
rewrite that came out stronger than the claim it was meant to fix, and rescores
the result. It exists so the implementation can be checked against the analysis
that motivated the gate: on the 50-review run in `results/` it reproduces 110
grounded rewrites of which 50 are rejected, and abs_delta 1.327 -> 1.240.

It also shows the part that is easy to miss. The same gate applied to the
self-critique arm makes it *worse* (1.113 -> 1.169), because 30 of the 50 plain
drafts already sit below the reviewer's strength and self-critique's
strengthening was moving some of them the right way. The gate is not a free
win; it is a constraint that helps where the loop overshoots and costs where it
was undershooting.

An estimate, not a result -- with the gate in place the second round would have
been revising different text -- so it never writes anything.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from ..config import paths
from ..experiment.run import load_results
from ..graph.state import CONDITIONS
from ..verify.segment import segment
from ..verify.strength import score_strength
from .metrics import score_summary

__all__ = ["report", "replay", "flag_kind"]

OUTCOMES = ("accepted", "retried", "rejected")
KINDS = ("overclaim", "unsupported", "direction", "other")


def flag_kind(reason: str) -> str:
    """Coarse flag type, read off the reason string the Verifier wrote.

    The grounded condition's reasons come from ClaimReport.reason(); the
    self-critique condition's are the model's own words and land in "other".
    Reading the kind here rather than carrying it on the Flag keeps `evidence`
    the only thing that distinguishes the two revising conditions.
    """
    r = (reason or "").lower()
    if "the evidence only supports" in r:
        return "overclaim"
    if "does not clearly support" in r:
        return "unsupported"
    if "effect where the evidence" in r:
        return "direction"
    return "other"


def _rows(results_dir: Path | None = None) -> list[tuple[str, dict]]:
    return [
        (r["condition"], rep)
        for r in load_results(results_dir)
        if not r.get("error") and r.get("condition") in CONDITIONS
        for rep in r.get("repairs", [])
    ]


def report(results_dir: Path | None = None, verbose: bool = True) -> dict:
    out_dir = Path(results_dir or paths().results)
    rows = _rows(out_dir)
    if not rows:
        print(
            f"[gate] no repair records in {out_dir}. Either the run predates the gate "
            "or nothing was ever flagged; try `replay` for a pre-gate run."
        )
        return {"n_repairs": 0}

    by_cond: dict[str, Counter] = {c: Counter() for c in CONDITIONS}
    by_kind: dict[str, Counter] = {k: Counter() for k in KINDS}
    for condition, rep in rows:
        by_cond[condition][rep["outcome"]] += 1
        by_kind[flag_kind(rep.get("reason", ""))][rep["outcome"]] += 1

    blocked = [r for _, r in rows if r["outcome"] in ("retried", "rejected")]
    softened = sum(
        1 for _, r in rows
        if r.get("score_final") is not None and r["score_final"] < r["score_before"]
    )
    out = {
        "n_repairs": len(rows),
        "by_condition": {c: dict(v) for c, v in by_cond.items() if v},
        "by_flag_kind": {k: dict(v) for k, v in by_kind.items() if v},
        "n_blocked": len(blocked),
        "n_softened": softened,
    }

    if verbose:
        L = ["=" * 70, f"REVISER GATE   {len(rows)} repair(s) across {out_dir}", "=" * 70,
             "", f"  {'':<16}" + "".join(f"{o:>12}" for o in OUTCOMES)]
        for label, table in (("by condition", by_cond), ("by flag kind", by_kind)):
            L.append(f"  -- {label}")
            for name, counts in table.items():
                if not counts:
                    continue
                L.append(f"  {name:<16}" + "".join(f"{counts[o]:>12}" for o in OUTCOMES))
        L += ["",
              f"  strengthening blocked ... {len(blocked)} of {len(rows)} "
              f"({100 * len(blocked) / len(rows):.0f}%)",
              f"  rewrites that softened .. {softened}",
              "  a high rejection rate is evidence the flags are wrong, not the Reviser",
              "=" * 70]
        print("\n".join(L))
    return out


# --------------------------------------------------------------------------
# Replaying an ungated run
# --------------------------------------------------------------------------


def _rounds(record: dict) -> list[list[str]]:
    """Per-round claim lists. Exact for grounded, re-segmented for the rest."""
    if record.get("reports"):
        return [[c["claim"] for c in rep["claims"]] for rep in record["reports"]]
    return [segment(h["summary"]) for h in record.get("history", [])]


def replay(results_dir: Path | None = None, verbose: bool = True) -> dict:
    """Estimate what the gate would have done to a run made without it."""
    out_dir = Path(results_dir or paths().results)
    records = [r for r in load_results(out_dir) if not r.get("error")]
    if any(r.get("repairs") for r in records):
        print("[gate] these records already come from a gated run; use `report` instead")

    by_review: dict[str, dict[str, dict]] = {}
    for r in records:
        if r.get("condition") in CONDITIONS and r.get("summary") and r.get("target"):
            by_review.setdefault(r["review_id"], {})[r["condition"]] = r
    reviews = sorted(k for k, v in by_review.items() if all(c in v for c in CONDITIONS))

    n_rewrites = n_stronger = 0
    per_condition = {c: Counter() for c in CONDITIONS}
    guarded: dict[str, dict[str, str]] = {}
    for rid in reviews:
        for cond in CONDITIONS:
            rec = by_review[rid][cond]
            rounds = _rounds(rec)
            if len(rounds) < 2:
                guarded.setdefault(rid, {})[cond] = rec["summary"]
                continue
            current = list(rounds[0])
            for a, b in zip(rounds, rounds[1:]):
                if len(a) != len(b) or len(a) != len(current):
                    current = list(b)  # segmentation drifted; take the run's own text
                    continue
                for j, (before, after) in enumerate(zip(a, b)):
                    if before == after:
                        continue
                    n_rewrites += 1
                    per_condition[cond]["rewrites"] += 1
                    # The real gate compares the proposal with the claim it was
                    # handed, which in a gated run is whatever survived the
                    # earlier rounds -- not the ungated trace's text.
                    if score_strength(after).score > score_strength(current[j]).score:
                        n_stronger += 1
                        per_condition[cond]["rejected"] += 1
                    else:
                        current[j] = after
            guarded.setdefault(rid, {})[cond] = " ".join(current).strip()

    def mean_abs(pick) -> dict[str, float]:
        return {
            c: sum(
                score_summary(pick(rid, c), by_review[rid][c]["target"], rid, c).abs_delta
                for rid in reviews
            ) / len(reviews)
            for c in CONDITIONS
        } if reviews else {}

    actual = mean_abs(lambda rid, c: by_review[rid][c]["summary"])
    gated = mean_abs(lambda rid, c: guarded[rid][c])
    out = {
        "n_reviews": len(reviews),
        "n_rewrites": n_rewrites,
        "n_would_reject": n_stronger,
        "per_condition": {c: dict(v) for c, v in per_condition.items() if v},
        "abs_delta_actual": actual,
        "abs_delta_if_gated": gated,
    }

    if verbose:
        L = ["=" * 70,
             f"GATE REPLAY (estimate)   {len(reviews)} reviews, {n_rewrites} rewrites",
             "=" * 70, "",
             f"  {'condition':<16}{'rewrites':>10}{'rejected':>10}"
             f"{'ungated':>12}{'if gated':>12}"]
        for c in CONDITIONS:
            L.append(f"  {c:<16}{per_condition[c]['rewrites']:>10}"
                     f"{per_condition[c]['rejected']:>10}"
                     f"{actual.get(c, float('nan')):>12.3f}"
                     f"{gated.get(c, float('nan')):>12.3f}")
        L += ["",
              "  abs_delta, lower is better. An estimate only: with the gate in place",
              "  the second round would have been revising different text.",
              "=" * 70]
        print("\n".join(L))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    mode = args[0] if args and args[0] in ("report", "replay") else "report"
    where = Path(args[-1]) if len(args) > 1 else None
    (replay if mode == "replay" else report)(where)
