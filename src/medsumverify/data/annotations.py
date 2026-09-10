"""Human annotations of claim strength and effect direction.

Source: allenai/mslr-annotated-dataset, data_with_overlap_scores.json -- which
is JSONL despite the extension. What is actually in it (measured, not assumed):

  * 470 lines, one per Cochrane *test* review, each carrying `target`.
    So test-split targets do exist here even though test-targets.csv does not.
  * 10 systems produced predictions; 6 of them were annotated.
  * 636 annotation rows over 597 unique (review, system) pairs spanning
    274 review_ids. 39 pairs are double-annotated -> the human-agreement ceiling.
  * Label scales: strength_* in 0-3, ed_* in {-1 N/A, 0 Negative, 1 No effect,
    2 Positive}. Some are null and must be dropped per-metric.

Two base rates that shape the whole evaluation:

  * Overclaiming (strength_generated > strength_target) occurs in only 9.1% of
    rows; underclaiming occurs in 56.6%. These 2022-era fine-tuned seq2seq
    systems hedge into vagueness rather than overstate. Detection is therefore
    an imbalanced problem -- report AUPRC next to AUROC.
  * 45% of effect-direction "mismatches" are the generated summary stating no
    direction at all (omission), not a genuine flip. Only ~9% are true flips.
    Validating a flip detector against raw mismatch would measure the wrong
    thing, so the two are kept separate here.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

from ..config import SEED, paths
from .download import ensure_annotations

__all__ = [
    "AnnotatedSummary",
    "ED_LABELS",
    "load_annotated",
    "load_test_targets",
    "calibration_validation_split",
    "direction_relation",
]

ED_LABELS = {-1: "N/A", 0: "Negative", 1: "NoEffect", 2: "Positive"}

_ORDINAL = ("fluency", "population", "intervention", "outcome",
            "ed_target", "ed_generated", "strength_target", "strength_generated")


def _mean_or_none(vals: list) -> float | None:
    keep = [v for v in vals if v is not None]
    return statistics.fmean(keep) if keep else None


@dataclass
class AnnotatedSummary:
    """One (review, system) pair plus the human judgement(s) of it."""

    review_id: str
    system: str
    summary: str          # the system's generated text
    target: str           # the Cochrane reviewer's conclusion
    raw: list[dict] = field(default_factory=list)   # per-annotator rows

    def label(self, name: str) -> float | None:
        """Mean over annotators; None when nobody supplied it."""
        return _mean_or_none([a.get(name) for a in self.raw])

    def label_int(self, name: str) -> int | None:
        v = self.label(name)
        return None if v is None else int(round(v))

    @property
    def n_annotators(self) -> int:
        return len(self.raw)

    @property
    def strength_delta(self) -> float | None:
        """Human-judged over/under-claiming. Positive = stated too strongly."""
        g, t = self.label("strength_generated"), self.label("strength_target")
        return None if g is None or t is None else g - t

    @property
    def is_overclaim(self) -> bool | None:
        d = self.strength_delta
        return None if d is None else d > 0

    @property
    def direction_relation(self) -> str | None:
        return direction_relation(
            self.label_int("ed_generated"), self.label_int("ed_target")
        )


def direction_relation(generated: int | None, target: int | None) -> str | None:
    """Classify how a summary's effect direction relates to the reference.

    Separating 'omission' from 'flip' matters: 45% of raw mismatches are the
    summary simply not committing to a direction, which is a different failure
    from asserting the opposite one.
    """
    if generated is None or target is None:
        return None
    if generated == target:
        return "agree"
    if generated == -1:
        return "omission"      # summary states no direction
    if target == -1:
        return "target_na"     # reference states none; nothing to contradict
    return "flip"              # genuinely contradictory direction


@lru_cache(maxsize=2)
def _load_raw() -> tuple[dict, ...]:
    path = ensure_annotations()
    with open(path, encoding="utf-8") as fh:
        return tuple(json.loads(line) for line in fh if line.strip())


@lru_cache(maxsize=2)
def load_annotated(annotated_only: bool = True) -> tuple[AnnotatedSummary, ...]:
    """All (review, system) pairs. With annotated_only, just the judged ones."""
    out: list[AnnotatedSummary] = []
    for rec in _load_raw():
        for pred in rec.get("predictions", []):
            anns = pred.get("annotations") or []
            if annotated_only and not anns:
                continue
            out.append(
                AnnotatedSummary(
                    review_id=str(rec["review_id"]),
                    system=str(pred.get("exp_short", "?")),
                    summary=(pred.get("prediction") or "").strip(),
                    target=(rec.get("target") or "").strip(),
                    raw=list(anns),
                )
            )
    return tuple(out)


@lru_cache(maxsize=2)
def load_test_targets() -> dict[str, str]:
    """review_id -> reference conclusion, for all 470 Cochrane test reviews."""
    return {
        str(r["review_id"]): (r.get("target") or "").strip()
        for r in _load_raw()
        if r.get("target")
    }


def calibration_validation_split(seed: int = SEED) -> tuple[frozenset[str], frozenset[str]]:
    """Split annotated review_ids in half, deterministically.

    Kept for quick inspection only. With ~48 overclaim positives in total, a
    single 50/50 split lands base rates as far apart as 5.9% vs 12.5%, so the
    reported protocol is out-of-fold CV via `cv_folds()` instead.
    """
    import random

    ids = sorted({a.review_id for a in load_annotated()})
    rng = random.Random(seed)
    rng.shuffle(ids)
    half = len(ids) // 2
    return frozenset(ids[:half]), frozenset(ids[half:])


def cv_folds(n_splits: int = 5, seed: int = SEED):
    """Out-of-fold splits over annotated pairs: grouped by review, stratified by outcome.

    Thresholds and the aggregation choice are fitted on the training folds and
    applied to the held-out fold, so every reported number is out-of-fold. The
    grouping keeps one review's six system summaries on the same side, and the
    stratification stops a 9%-prevalence positive class from clustering into
    one fold. Yields (train_idx, test_idx) over `load_annotated()` order.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    rows = load_annotated()
    idx = [i for i, a in enumerate(rows) if a.is_overclaim is not None]
    y = np.array([int(rows[i].is_overclaim) for i in idx])
    groups = np.array([rows[i].review_id for i in idx])
    idx = np.array(idx)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(idx, y, groups):
        yield idx[train], idx[test]
