"""Build a condition-blinded sheet of claims for a human to label.

Everything else in the evaluation is automatic, and every automatic measure
shares the same lexicon. This is the one check that does not: a person reads
the claim next to the evidence and says whether it overstates it.

The sheet is shuffled and carries no condition label. The key is written to a
separate file so the sheet can be filled in without seeing which condition
produced which claim.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from ..config import SEED, paths
from ..data.cochrane import load_review_index
from ..experiment.run import load_results
from ..graph.state import CONDITIONS
from ..verify.segment import segment
from ..verify.strength import ClaimType, score_strength

__all__ = ["build_audit_sheet"]

SHEET_NAME = "audit_sheet.csv"
KEY_NAME = "audit_key.json"
INSTRUCTIONS = """HOW TO FILL THIS IN

For each row, compare the CLAIM against the EVIDENCE from the source trials.

Put one of these in the `verdict` column:
  overclaim  - the claim is stated more strongly than the evidence supports
               (causal language for an association, a confident assertion where
                the trials were inconclusive, a firm number from a wide interval)
  faithful   - the strength of the claim matches the strength of the evidence
  underclaim - the claim is markedly weaker/vaguer than the evidence supports
  unclear    - you cannot tell from the evidence shown

Judge only the strength match. Ignore grammar, style, and whether the claim is
the most interesting one that could have been made.

Rows are shuffled and the condition that produced each claim is deliberately
hidden. Do not look at audit_key.json until the sheet is finished.
"""


def build_audit_sheet(
    n_claims: int = 60,
    seed: int = SEED,
    split: str = "dev",
    results_dir: Path | None = None,
) -> Path:
    out_dir = Path(results_dir or paths().results)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        r for r in load_results(out_dir)
        if not r.get("error") and r.get("summary") and r.get("condition") in CONDITIONS
    ]
    if not records:
        raise RuntimeError("no experiment results to audit -- run the experiment first")

    reviews = load_review_index(split)
    rng = random.Random(seed)

    rows = []
    for rec in records:
        review = reviews.get(rec["review_id"])
        if review is None:
            continue
        for i, claim in enumerate(segment(rec["summary"])):
            sc = score_strength(claim)
            # Only claim-bearing sentences are auditable; "12 trials were
            # included" has no strength to be wrong about.
            if sc.claim_type not in (ClaimType.FINDING, ClaimType.SUFFICIENCY):
                continue
            rows.append({
                "review_id": rec["review_id"],
                "condition": rec["condition"],
                "claim_index": i,
                "claim": claim,
            })

    # Sample evenly across conditions so no condition dominates the sheet.
    per_condition = max(n_claims // len(CONDITIONS), 1)
    picked = []
    for condition in CONDITIONS:
        pool = [r for r in rows if r["condition"] == condition]
        rng.shuffle(pool)
        picked.extend(pool[:per_condition])
    rng.shuffle(picked)

    # Attach evidence with the same retriever the verifier uses, so the human
    # sees what the machine saw.
    from ..models.retriever import FakeRetriever, build_evidence, split_sentences

    retriever = FakeRetriever()
    sheet_rows = []
    key = {}
    for n, row in enumerate(picked, 1):
        review = reviews[row["review_id"]]
        sentences: list[str] = []
        for doc in review.documents():
            sentences.extend(split_sentences(doc))
        evidence, _ = build_evidence(row["claim"], sentences, retriever, k=5)
        item_id = f"A{n:03d}"
        sheet_rows.append({
            "id": item_id,
            "claim": row["claim"],
            "evidence": evidence,
            "verdict": "",
            "notes": "",
        })
        key[item_id] = row

    sheet = out_dir / SHEET_NAME
    with open(sheet, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "claim", "evidence", "verdict", "notes"])
        w.writeheader()
        w.writerows(sheet_rows)

    (out_dir / KEY_NAME).write_text(json.dumps(key, indent=2), encoding="utf-8")
    (out_dir / "audit_instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")

    print(f"[write] {sheet}  ({len(sheet_rows)} claims, condition hidden)")
    print(f"[write] {out_dir / KEY_NAME}  (do not open until the sheet is filled in)")
    return sheet


if __name__ == "__main__":
    build_audit_sheet()
