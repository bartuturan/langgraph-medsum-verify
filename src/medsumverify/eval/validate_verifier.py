"""Result #1: does what the verifier flags correspond to what humans called distorted?

This runs the complete verifier -- retrieval, MiniCheck, strength, direction --
over the 597 annotated (review, system) pairs, using the same Cochrane test
abstracts as source, and asks whether its flags line up with the human labels.

It is the project's first real result and it stands on its own: if the loop
never worked, this would still say something about whether cheap tool-grounded
checks track expert judgements of claim distortion.

Three questions, kept separate because they are separate constructs:

  V1  strength   rank correlation with human claim strength (see
                 validate_strength.py, which needs no GPU)
  V2  overclaim  can the flags detect strength_generated > strength_target?
                 Positives are ~9% of rows -- AUPRC and lift matter more than
                 AUROC, which flatters at that prevalence.
  V3  direction  can the direction flag detect a genuine flip? Validated
                 against true flips only (n=55); the 269 "omissions", where the
                 summary states no direction at all, are a different failure and
                 are reported separately rather than folded in.

Thresholds are fitted on training folds and applied out-of-fold, so no number
here comes from a threshold tuned on the row it scores.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import DEFAULT_THRESHOLDS, Thresholds, paths
from ..data.annotations import cv_folds, load_annotated
from ..data.cochrane import load_review_index
from ..verify.verifier import DIRECTION_FLIP, OVERCLAIM, UNSUPPORTED, Verifier

__all__ = ["validate_verifier"]


@dataclass
class VerifierValidation:
    n_pairs: int
    n_reviews: int
    base_rate_overclaim: float
    auroc_overclaim: float
    auprc_overclaim: float
    auprc_baseline: float
    precision: float
    recall: float
    f1: float
    flag_rate: float
    chosen_thresholds: list[dict]
    n_true_flips: int
    n_omissions: int
    auroc_direction: float | None
    precision_direction: float | None
    recall_direction: float | None
    per_flag_rates: dict = field(default_factory=dict)
    gate_passed: bool = False

    def report(self) -> str:
        L = [
            "=" * 72,
            "RESULT 1  full verifier vs human annotations (out-of-fold)",
            "=" * 72,
            f"  {self.n_pairs} summaries over {self.n_reviews} Cochrane test reviews",
            "",
            "  V2  overclaim detection",
            f"      base rate     {self.base_rate_overclaim:.1%}   "
            f"(the verifier flags {self.flag_rate:.1%} of summaries)",
            f"      AUROC         {self.auroc_overclaim:.3f}",
            f"      AUPRC         {self.auprc_overclaim:.3f}  "
            f"(baseline {self.auprc_baseline:.3f}, "
            f"lift {self.auprc_overclaim / max(self.auprc_baseline, 1e-9):.2f}x)",
            f"      precision     {self.precision:.3f}",
            f"      recall        {self.recall:.3f}",
            f"      F1            {self.f1:.3f}",
            "",
            "  V3  effect direction",
            f"      true flips    {self.n_true_flips}   omissions {self.n_omissions} "
            "(reported separately -- not the same failure)",
        ]
        if self.auroc_direction is not None:
            L += [
                f"      AUROC         {self.auroc_direction:.3f}",
                f"      precision     {self.precision_direction:.3f}   "
                f"recall {self.recall_direction:.3f}",
            ]
        else:
            L.append("      too few flips to score")
        L += [
            "",
            f"  flag mix: {self.per_flag_rates}",
            "",
            f"  GATE (AUROC >= 0.65): {'PASSED' if self.gate_passed else 'FAILED'}",
            "=" * 72,
        ]
        return "\n".join(L)


def _fit_thresholds(scores: np.ndarray, y: np.ndarray) -> Thresholds:
    """Pick the level gap that maximises F1 on the training folds."""
    best, best_f1 = 1, -1.0
    for level in (1, 2):
        pred = scores >= level
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best, best_f1 = level, f1
    return Thresholds(
        delta_level=best,
        tau_support=DEFAULT_THRESHOLDS.tau_support,
        relevance_floor=DEFAULT_THRESHOLDS.relevance_floor,
        max_evidence_sentences=DEFAULT_THRESHOLDS.max_evidence_sentences,
    )


def validate_verifier(fake: bool = False, n_splits: int = 5, limit: int | None = None) -> VerifierValidation:
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ..models.factcheck import load_factchecker
    from ..models.retriever import load_retriever

    rows = list(load_annotated())
    if limit:
        rows = rows[:limit]
    reviews = load_review_index("test")

    verifier = Verifier(load_factchecker(fake=fake), load_retriever(fake=fake))

    # Run the verifier once per pair; thresholds are applied afterwards so the
    # expensive part is not repeated per fold.
    max_gap = np.full(len(rows), np.nan)      # claim_level - evidence_level
    min_support = np.full(len(rows), np.nan)
    any_flip = np.zeros(len(rows), dtype=bool)
    flag_counts: dict[str, int] = {}

    for i, row in enumerate(rows):
        review = reviews.get(row.review_id)
        if review is None or not row.summary:
            continue
        report = verifier.verify(row.summary, review.documents())
        if not report.claims:
            continue
        gaps = [c.claim_level - c.evidence_level for c in report.claims]
        max_gap[i] = max(gaps)
        min_support[i] = min(c.support_prob for c in report.claims)
        any_flip[i] = any(DIRECTION_FLIP in c.flags for c in report.claims)
        for c in report.claims:
            for f in c.flags:
                flag_counts[f] = flag_counts.get(f, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  verified {i + 1}/{len(rows)}")

    y = np.array([np.nan if r.is_overclaim is None else float(r.is_overclaim) for r in rows])
    usable = ~(np.isnan(max_gap) | np.isnan(y))

    # Out-of-fold thresholded predictions.
    pred = np.zeros(len(rows), dtype=bool)
    chosen = []
    for train_idx, test_idx in cv_folds(n_splits=n_splits):
        tr = train_idx[usable[train_idx]]
        te = test_idx[usable[test_idx]]
        if len(tr) < 10 or len(te) == 0:
            continue
        th = _fit_thresholds(max_gap[tr], y[tr])
        chosen.append({"delta_level": th.delta_level})
        pred[te] = max_gap[te] >= th.delta_level

    yy, gg, pp = y[usable], max_gap[usable], pred[usable]
    tp = float((pp & (yy == 1)).sum())
    fp = float((pp & (yy == 0)).sum())
    fn = float((~pp & (yy == 1)).sum())
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)

    # V3: true flips only.
    rel = [r.direction_relation for r in rows]
    flip_mask = np.array([x == "flip" for x in rel])
    agree_mask = np.array([x == "agree" for x in rel])
    dir_eval = flip_mask | agree_mask
    auroc_dir = prec_dir = rec_dir = None
    if flip_mask.sum() >= 10 and agree_mask.sum() >= 10:
        y_dir = flip_mask[dir_eval].astype(float)
        s_dir = any_flip[dir_eval].astype(float)
        if len(set(s_dir)) > 1:
            auroc_dir = float(roc_auc_score(y_dir, s_dir))
        dtp = float((s_dir.astype(bool) & (y_dir == 1)).sum())
        dfp = float((s_dir.astype(bool) & (y_dir == 0)).sum())
        dfn = float((~s_dir.astype(bool) & (y_dir == 1)).sum())
        prec_dir = dtp / max(dtp + dfp, 1e-9)
        rec_dir = dtp / max(dtp + dfn, 1e-9)

    auroc = float(roc_auc_score(yy, gg)) if len(set(yy)) > 1 else float("nan")
    out = VerifierValidation(
        n_pairs=int(usable.sum()),
        n_reviews=len({r.review_id for r in rows}),
        base_rate_overclaim=float(yy.mean()),
        auroc_overclaim=auroc,
        auprc_overclaim=float(average_precision_score(yy, gg)),
        auprc_baseline=float(yy.mean()),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-9),
        flag_rate=float(pp.mean()),
        chosen_thresholds=chosen,
        n_true_flips=int(flip_mask.sum()),
        n_omissions=int(sum(1 for x in rel if x == "omission")),
        auroc_direction=auroc_dir,
        precision_direction=prec_dir,
        recall_direction=rec_dir,
        per_flag_rates=flag_counts,
        gate_passed=bool(auroc >= 0.65),
    )
    print(out.report())
    dest = paths().results / ("validation_verifier_fake.json" if fake else "validation_verifier.json")
    dest.write_text(json.dumps(asdict(out), indent=2), encoding="utf-8")
    print(f"[write] {dest}")
    return out


if __name__ == "__main__":
    import sys

    validate_verifier(fake="--fake" in sys.argv)
