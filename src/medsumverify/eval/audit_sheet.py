"""Build a condition-blinded sheet of claims for a human to label, then score it.

Everything else in the evaluation is automatic, and every automatic measure
shares the same lexicon. This is the one check that does not: a person reads
the claim next to the evidence and says whether it overstates it.

The sheet is shuffled and carries no condition label. The key is written to a
separate file so the sheet can be filled in without seeing which condition
produced which claim.

    python -m medsumverify.eval.audit_sheet          # build (refuses to clobber labels)
    python -m medsumverify.eval.audit_sheet score    # score a filled-in sheet
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
from pathlib import Path

from ..config import SEED, paths
from ..data.cochrane import load_review_index
from ..experiment.run import load_results
from ..graph.state import CONDITIONS
from ..verify.segment import segment
from ..verify.strength import ClaimType, score_strength

__all__ = ["build_audit_sheet", "score_audit", "read_sheet"]

SHEET_NAME = "audit_sheet.csv"
KEY_NAME = "audit_key.json"
SCORED_NAME = "audit_scored.json"
VERDICTS = ("overclaim", "faithful", "underclaim", "unclear")
DECIDED = ("overclaim", "faithful", "underclaim")

# Spellings a person is likely to type. Anything else is reported back as
# invalid rather than guessed at.
_ALIASES = {
    "overclaim": "overclaim", "over": "overclaim", "oc": "overclaim",
    "faithful": "faithful", "ok": "faithful", "f": "faithful",
    "underclaim": "underclaim", "under": "underclaim", "uc": "underclaim",
    "unclear": "unclear", "?": "unclear", "skip": "unclear",
}

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

Saving from Excel is fine, including with semicolons or a byte-order mark.
When you are done, score it with:

    python -m medsumverify.eval.audit_sheet score
"""


def read_sheet(path: Path) -> list[dict]:
    """Read the sheet the way a spreadsheet program is likely to have saved it.

    Excel adds a byte-order mark, and in many European locales it saves CSV
    with semicolons -- sometimes in the Windows codepage rather than UTF-8.
    All of that is handled here rather than making the labeller re-save.
    """
    raw = path.read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()
    header = lines[0] if lines else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    rows = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [
        {(k or "").strip().lower(): (v or "") for k, v in row.items()}
        for row in rows
    ]


def build_audit_sheet(
    n_claims: int = 60,
    seed: int = SEED,
    split: str = "dev",
    results_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    out_dir = Path(results_dir or paths().results)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / SHEET_NAME

    # A filled-in sheet is an hour of someone's time. Never clobber one by
    # accident -- re-running the notebook cell must not wipe the labels.
    if sheet.exists() and not overwrite:
        filled = [r for r in read_sheet(sheet) if r.get("verdict", "").strip()]
        if filled:
            raise FileExistsError(
                f"{sheet} already has {len(filled)} verdicts filled in; "
                "pass overwrite=True (or --overwrite) to replace it"
            )

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

    # Evidence comes from the same kind of sentence retrieval the verifier
    # uses, so the human judges roughly what the machine judged. TF-IDF here,
    # not the dense retriever, so the sheet can be rebuilt on CPU.
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

    with open(sheet, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "claim", "evidence", "verdict", "notes"])
        w.writeheader()
        w.writerows(sheet_rows)

    (out_dir / KEY_NAME).write_text(json.dumps(key, indent=2), encoding="utf-8")
    (out_dir / "audit_instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")

    print(f"[write] {sheet}  ({len(sheet_rows)} claims, condition hidden)")
    print(f"[write] {out_dir / KEY_NAME}  (do not open until the sheet is filled in)")
    return sheet


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval -- behaves at k=0 and k=n, unlike the normal one."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _auto_verdict(verifier, claim: str, evidence: str) -> str | None:
    """What the strength comparison says about the same claim and evidence.

    Uses only the strength levels, not the fact-checker, so it is exactly the
    automatic instrument the human is being compared with.
    """
    report = verifier.verify(claim, [evidence])
    checkable = [
        c for c in report.claims if c.claim_type in (ClaimType.FINDING.value, ClaimType.SUFFICIENCY.value)
    ]
    if not checkable:
        return None
    gap = checkable[0].claim_level - checkable[0].evidence_level
    return "overclaim" if gap >= 1 else "underclaim" if gap <= -1 else "faithful"


def score_audit(results_dir: Path | None = None, verbose: bool = True) -> dict:
    """Unblind a filled-in sheet and report overclaim rates per condition."""
    from scipy import stats

    from ..models.factcheck import FakeFactChecker
    from ..models.retriever import FakeRetriever
    from ..verify.verifier import Verifier
    from .compare import CONTRASTS, _holm

    out_dir = Path(results_dir or paths().results)
    sheet, key_path = out_dir / SHEET_NAME, out_dir / KEY_NAME
    for p in (sheet, key_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} -- build the audit sheet first")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    rows = read_sheet(sheet)

    labelled, blank, invalid, unknown = [], [], [], []
    for row in rows:
        item = row.get("id", "").strip()
        raw = row.get("verdict", "").strip().lower()
        if item not in key:
            unknown.append(item)
            continue
        if not raw:
            blank.append(item)
            continue
        verdict = _ALIASES.get(raw)
        if verdict is None:
            invalid.append({"id": item, "verdict": raw})
            continue
        labelled.append({
            "id": item,
            "condition": key[item]["condition"],
            "verdict": verdict,
            "claim": row.get("claim", ""),
            "evidence": row.get("evidence", ""),
        })
    if not labelled:
        raise RuntimeError("no verdicts filled in yet -- nothing to score")

    per_condition = {}
    for c in CONDITIONS:
        mine = [r for r in labelled if r["condition"] == c]
        counts = {v: sum(1 for r in mine if r["verdict"] == v) for v in VERDICTS}
        decided = sum(counts[v] for v in DECIDED)
        lo, hi = _wilson(counts["overclaim"], decided)
        per_condition[c] = {
            "counts": counts,
            "n_decided": decided,
            "overclaim_rate": counts["overclaim"] / decided if decided else None,
            "overclaim_ci95": [lo, hi],
        }

    # Claims are sampled independently per condition, not paired, so this is
    # Fisher's exact test on the 2x2 of overclaim vs. not.
    tests = []
    for a, b in CONTRASTS:
        ca, cb = per_condition[a], per_condition[b]
        table = [
            [ca["counts"]["overclaim"], ca["n_decided"] - ca["counts"]["overclaim"]],
            [cb["counts"]["overclaim"], cb["n_decided"] - cb["counts"]["overclaim"]],
        ]
        if ca["n_decided"] and cb["n_decided"]:
            odds, p = stats.fisher_exact(table)
        else:
            odds, p = float("nan"), 1.0
        tests.append({"a": a, "b": b, "table": table,
                      "odds_ratio": _finite(odds), "p_raw": float(p)})
    for t, p_adj in zip(tests, _holm([t["p_raw"] for t in tests])):
        t["p_holm"] = float(p_adj)

    # Does the automatic instrument agree with the person, on Qwen's own text?
    verifier = Verifier(FakeFactChecker(), FakeRetriever())
    human, auto = [], []
    for r in labelled:
        if r["verdict"] not in DECIDED:
            continue
        a = _auto_verdict(verifier, r["claim"], r["evidence"])
        if a is None:
            continue
        human.append(r["verdict"])
        auto.append(a)
    agreement = None
    if len(human) >= 5:
        from sklearn.metrics import cohen_kappa_score

        kappa = None
        if len(set(human) | set(auto)) > 1:
            kappa = _finite(cohen_kappa_score(human, auto, labels=list(DECIDED)))
        hb = [h == "overclaim" for h in human]
        ab = [a == "overclaim" for a in auto]
        tp = sum(h and a for h, a in zip(hb, ab))
        fp = sum((not h) and a for h, a in zip(hb, ab))
        fn = sum(h and (not a) for h, a in zip(hb, ab))
        agreement = {
            "n": len(human),
            "exact": sum(h == a for h, a in zip(human, auto)) / len(human),
            "kappa": kappa,
            "overclaim_precision": tp / (tp + fp) if tp + fp else None,
            "overclaim_recall": tp / (tp + fn) if tp + fn else None,
        }

    out = {
        "n_rows": len(rows),
        "n_labelled": len(labelled),
        "blank": blank,
        "invalid": invalid,
        "unknown_ids": unknown,
        "per_condition": per_condition,
        "tests": tests,
        "agreement_with_automatic": agreement,
    }
    if verbose:
        print(_audit_report(out))
    dest = out_dir / SCORED_NAME
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[write] {dest}")
    return out


def _fmt(x: float | None, spec: str = ".2f") -> str:
    return "  n/a" if x is None else format(x, spec)


def _audit_report(out: dict) -> str:
    L = ["=" * 78,
         f"BLIND AUDIT   {out['n_labelled']}/{out['n_rows']} claims labelled"
         f"   (blank {len(out['blank'])}, invalid {len(out['invalid'])})",
         "=" * 78, "",
         f"  {'condition':<14}{'over':>6}{'faith':>7}{'under':>7}{'unclr':>7}"
         f"   overclaim rate [95% CI]"]
    for c in CONDITIONS:
        pc = out["per_condition"][c]
        k = pc["counts"]
        lo, hi = pc["overclaim_ci95"]
        L.append(
            f"  {c:<14}{k['overclaim']:>6}{k['faithful']:>7}{k['underclaim']:>7}{k['unclear']:>7}"
            f"   {_fmt(pc['overclaim_rate'])} [{_fmt(lo)}, {_fmt(hi)}]"
        )
    L += ["", "  overclaim rate, Fisher exact, Holm-corrected"]
    for t in out["tests"]:
        star = "*" if t["p_holm"] < 0.05 else " "
        L.append(f"  {star} {t['a']:>13} vs {t['b']:<13} OR={_fmt(t['odds_ratio'])}  p={t['p_holm']:.4f}")
    ag = out["agreement_with_automatic"]
    L += ["", "  human vs automatic strength check (same claim, same evidence)"]
    if ag is None:
        L.append("    too few decided claims to compare")
    else:
        L.append(
            f"    n={ag['n']}  exact={ag['exact']:.2f}  kappa={_fmt(ag['kappa'])}  "
            f"overclaim precision={_fmt(ag['overclaim_precision'])} "
            f"recall={_fmt(ag['overclaim_recall'])}"
        )
    if out["invalid"]:
        L += ["", "  unrecognised verdicts (fix and re-run): "
              + ", ".join(f"{r['id']}={r['verdict']!r}" for r in out["invalid"])]
    per = [out["per_condition"][c]["n_decided"] for c in CONDITIONS]
    L += ["",
          f"  {min(per)}-{max(per)} decided claims per condition: only large differences "
          "can reach significance.",
          "  Read this as a human sanity check on the automatic metrics.",
          "=" * 78]
    return "\n".join(L)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score_audit()
    else:
        build_audit_sheet(overwrite="--overwrite" in sys.argv)
