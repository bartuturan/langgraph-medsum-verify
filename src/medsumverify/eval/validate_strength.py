"""Result #1a: does the strength scorer measure what humans called claim strength?

Runs on CPU with no model downloads -- the scorer is pure lexicon -- so this is
the cheapest possible check of the project's central instrument, and it stands
on its own even if everything downstream breaks.

Protocol
--------
Aggregation ("max" / "mean" / "last" / "findings_max") is a free choice, so it
is fitted on the training folds of a grouped, stratified 5-fold split and
applied to the held-out fold. Every reported number is out-of-fold. Grouping is
by review_id so one review's six system summaries never straddle a fold.

Two things are measured separately:
  V1  Can it rank claim strength at all?  Spearman rho against the human 0-3
      labels, on generated summaries and on reference conclusions.
  V2  Can the *delta* detect overclaiming?  AUROC and AUPRC for
      strength(generated) - strength(reference) against the human judgement
      that the summary overstates its reference. Positives are ~9% of rows, so
      AUPRC and the prevalence baseline matter more than AUROC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from ..config import paths
from ..data.annotations import cv_folds, load_annotated
from ..verify.segment import segment
from ..verify.strength import AGGREGATIONS, aggregate

__all__ = ["validate_strength", "StrengthValidation"]


def _summary_strength(text: str, how: str) -> float:
    return aggregate(segment(text), how)


@dataclass
class StrengthValidation:
    n_pairs: int
    n_reviews: int
    chosen_aggregation: dict[str, int]
    rho_generated: float
    p_generated: float
    rho_target: float
    p_target: float
    rho_delta: float
    p_delta: float
    n_overclaim: int
    base_rate: float
    auroc: float
    auprc: float
    auprc_baseline: float
    human_ceiling_rho: float | None
    n_double_annotated: int

    def report(self) -> str:
        lines = [
            "=" * 68,
            "RESULT 1a  strength scorer vs human annotations (out-of-fold)",
            "=" * 68,
            f"  pairs={self.n_pairs}  reviews={self.n_reviews}  "
            f"aggregation chosen per fold: {self.chosen_aggregation}",
            "",
            "  V1  rank correlation with human claim strength",
            f"      generated summaries   rho = {self.rho_generated:+.3f}  (p={self.p_generated:.2g})",
            f"      reference conclusions rho = {self.rho_target:+.3f}  (p={self.p_target:.2g})",
            f"      strength delta        rho = {self.rho_delta:+.3f}  (p={self.p_delta:.2g})",
        ]
        if self.human_ceiling_rho is not None:
            lines.append(
                f"      human-human ceiling   rho = {self.human_ceiling_rho:+.3f}  "
                f"(n={self.n_double_annotated} double-annotated)"
            )
        lines += [
            "",
            "  V2  detecting overclaiming from the strength delta alone",
            f"      positives  {self.n_overclaim}/{self.n_pairs}  (base rate {self.base_rate:.1%})",
            f"      AUROC      {self.auroc:.3f}",
            f"      AUPRC      {self.auprc:.3f}   (prevalence baseline {self.auprc_baseline:.3f}, "
            f"lift {self.auprc / max(self.auprc_baseline, 1e-9):.2f}x)",
            "=" * 68,
        ]
        return "\n".join(lines)


def _human_ceiling(rows) -> tuple[float | None, int]:
    """Agreement between the two annotators on doubly-annotated pairs."""
    a, b = [], []
    for r in rows:
        if r.n_annotators < 2:
            continue
        vals = [x.get("strength_generated") for x in r.raw]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            a.append(vals[0])
            b.append(vals[1])
    if len(a) < 5:
        return None, len(a)
    rho, _ = stats.spearmanr(a, b)
    return float(rho), len(a)


def validate_strength(n_splits: int = 5, verbose: bool = True) -> StrengthValidation:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = load_annotated()

    # Precompute every aggregation once; the scorer is deterministic.
    cache = {
        how: (
            np.array([_summary_strength(r.summary, how) for r in rows]),
            np.array([_summary_strength(r.target, how) for r in rows]),
        )
        for how in AGGREGATIONS
    }

    human_gen = np.array(
        [r.label("strength_generated") if r.label("strength_generated") is not None else np.nan
         for r in rows]
    )
    human_tgt = np.array(
        [r.label("strength_target") if r.label("strength_target") is not None else np.nan
         for r in rows]
    )
    y_over = np.array([np.nan if r.is_overclaim is None else float(r.is_overclaim) for r in rows])

    oof_delta = np.full(len(rows), np.nan)
    oof_gen = np.full(len(rows), np.nan)
    oof_tgt = np.full(len(rows), np.nan)
    chosen: dict[str, int] = {}

    for train_idx, test_idx in cv_folds(n_splits=n_splits):
        # Pick the aggregation that best ranks human strength on the training folds.
        best_how, best_rho = AGGREGATIONS[0], -np.inf
        for how in AGGREGATIONS:
            gen, _ = cache[how]
            mask = train_idx[~np.isnan(human_gen[train_idx])]
            if len(mask) < 10:
                continue
            rho, _ = stats.spearmanr(gen[mask], human_gen[mask])
            if np.isfinite(rho) and rho > best_rho:
                best_how, best_rho = how, rho
        chosen[best_how] = chosen.get(best_how, 0) + 1

        gen, tgt = cache[best_how]
        oof_gen[test_idx] = gen[test_idx]
        oof_tgt[test_idx] = tgt[test_idx]
        oof_delta[test_idx] = gen[test_idx] - tgt[test_idx]

    def _rho(x, y):
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 5:
            return float("nan"), float("nan")
        r, p = stats.spearmanr(x[m], y[m])
        return float(r), float(p)

    rho_gen, p_gen = _rho(oof_gen, human_gen)
    rho_tgt, p_tgt = _rho(oof_tgt, human_tgt)
    human_delta = human_gen - human_tgt
    rho_del, p_del = _rho(oof_delta, human_delta)

    m = ~(np.isnan(oof_delta) | np.isnan(y_over))
    y, s = y_over[m], oof_delta[m]
    base = float(y.mean())
    ceiling, n_double = _human_ceiling(rows)

    out = StrengthValidation(
        n_pairs=int(m.sum()),
        n_reviews=len({r.review_id for r in rows}),
        chosen_aggregation=chosen,
        rho_generated=rho_gen, p_generated=p_gen,
        rho_target=rho_tgt, p_target=p_tgt,
        rho_delta=rho_del, p_delta=p_del,
        n_overclaim=int(y.sum()),
        base_rate=base,
        auroc=float(roc_auc_score(y, s)),
        auprc=float(average_precision_score(y, s)),
        auprc_baseline=base,
        human_ceiling_rho=ceiling,
        n_double_annotated=n_double,
    )
    if verbose:
        print(out.report())
    dest = paths().results / "validation_strength.json"
    dest.write_text(json.dumps(asdict(out), indent=2), encoding="utf-8")
    print(f"[write] {dest}")
    return out


if __name__ == "__main__":
    validate_strength()
