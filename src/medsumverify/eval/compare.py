"""Paired comparison of the three conditions.

All three conditions run on the same reviews from the same Round-0 draft, so
every comparison is within-review. Paired Wilcoxon signed-rank on the review
level, Holm-corrected across the three contrasts, with bootstrap CIs on the
paired differences.

The reading of the result is fixed in advance:

  * grounded beats plain            -> the loop does something
  * grounded beats selfcritique     -> the something is the grounding, not the
                                       second lap round the loop
  * selfcritique beats plain, and
    grounded does not beat it       -> looping is what helps; the tools are not
                                       earning their keep

and any strength win is discounted if the guardrails move the wrong way.

The held-out NLI judge is a secondary check from a different model family. It
is scored separately (eval/holdout.py, after the loop's models are freed) and
picked up here from results/holdout_nli.json when present. Higher is better
for it, unlike delta_strength.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

from ..config import paths
from ..experiment.run import load_results
from ..graph.state import CONDITIONS
from .holdout import load_holdout
from .metrics import SummaryMetrics, score_summary

__all__ = ["compare", "build_metric_table"]

PRIMARY = "delta_strength"
HOLDOUT = "holdout_entailment"
GUARDRAILS = ("direction_match", "content_f1", "n_words", "hedge_density", "abs_delta")
CONTRASTS = (("grounded", "plain"), ("grounded", "selfcritique"), ("selfcritique", "plain"))
TESTED = (PRIMARY, "abs_delta", "overclaims", "direction_match", "content_f1")


def build_metric_table(records: list[dict] | None = None) -> dict[str, dict[str, SummaryMetrics]]:
    """{review_id: {condition: metrics}} for reviews complete in every condition."""
    records = records if records is not None else load_results()
    table: dict[str, dict[str, SummaryMetrics]] = {}
    for r in records:
        if r.get("error") or r.get("condition") not in CONDITIONS:
            continue
        if not r.get("summary") or not r.get("target"):
            continue
        table.setdefault(r["review_id"], {})[r["condition"]] = score_summary(
            r["summary"], r["target"], r["review_id"], r["condition"]
        )
    # Pairing requires every condition present; a half-finished review would
    # otherwise bias whichever condition ran first.
    return {k: v for k, v in table.items() if all(c in v for c in CONDITIONS)}


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


@dataclass
class Contrast:
    a: str
    b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    statistic: float
    p_raw: float
    p_holm: float
    effect_size: float  # matched-pairs rank-biserial correlation


def _wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    d = x - y
    nz = d[d != 0]
    if len(nz) < 5:
        return float("nan"), 1.0, 0.0
    stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
    # Rank-biserial: proportion of signed rank favouring a, on [-1, 1].
    ranks = stats.rankdata(np.abs(nz))
    pos = ranks[nz > 0].sum()
    total = ranks.sum()
    rbc = 2 * (pos / total) - 1 if total else 0.0
    return float(stat), float(p), float(rbc)


def _contrasts_for(metric: str, n: int, column: Callable[[str], np.ndarray]) -> list[Contrast]:
    """The three paired contrasts for one metric, Holm-corrected together."""
    raw = []
    for a, b in CONTRASTS:
        xa, xb = column(a), column(b)
        stat, p, rbc = _wilcoxon(xa, xb)
        lo, hi = _bootstrap_ci(xa - xb)
        raw.append((a, b, float(xa.mean()), float(xb.mean()),
                    float((xa - xb).mean()), lo, hi, stat, p, rbc))
    adj = _holm([r[8] for r in raw])
    return [
        Contrast(
            a=r[0], b=r[1], metric=metric, n=n,
            mean_a=r[2], mean_b=r[3], mean_diff=r[4],
            ci_low=r[5], ci_high=r[6], statistic=r[7],
            p_raw=r[8], p_holm=p_adj, effect_size=r[9],
        )
        for r, p_adj in zip(raw, adj)
    ]


def compare(
    records: list[dict] | None = None,
    verbose: bool = True,
    results_dir: Path | None = None,
) -> dict:
    out_dir = Path(results_dir or paths().results)
    if records is None:
        records = load_results(out_dir)
    table = build_metric_table(records)
    reviews = sorted(table)
    if len(reviews) < 5:
        raise RuntimeError(
            f"only {len(reviews)} reviews complete in all three conditions -- "
            "run the experiment before comparing"
        )

    def col(condition: str, metric: str) -> np.ndarray:
        return np.array([float(getattr(table[r][condition], metric)) for r in reviews])

    metrics = (PRIMARY, *GUARDRAILS, "overclaims")
    descriptives = {
        c: {m: float(np.mean(col(c, m))) for m in metrics} for c in CONDITIONS
    }

    contrasts: list[Contrast] = []
    for metric in TESTED:
        contrasts += _contrasts_for(metric, len(reviews), lambda c, m=metric: col(c, m))

    # Secondary: the held-out judge, if it has been run. Only reviews scored in
    # all three conditions enter, so the test stays paired.
    holdout = load_holdout(out_dir)
    h_reviews = [r for r in reviews if all((r, c) in holdout for c in CONDITIONS)]
    holdout_info = {"available": False, "n_reviews": len(h_reviews)}
    if len(h_reviews) >= 5:
        def hcol(condition: str) -> np.ndarray:
            return np.array([holdout[(r, condition)] for r in h_reviews])

        for c in CONDITIONS:
            descriptives[c][HOLDOUT] = float(hcol(c).mean())
        contrasts += _contrasts_for(HOLDOUT, len(h_reviews), hcol)
        holdout_info["available"] = True
    elif holdout:
        print(f"[warn] held-out scores cover only {len(h_reviews)} complete reviews; skipped")

    out = {
        "n_reviews": len(reviews),
        "aggregation": "mean",
        "descriptives": descriptives,
        "contrasts": [c.__dict__ for c in contrasts],
        "holdout": holdout_info,
        "verdict": _verdict(descriptives, contrasts),
    }
    if verbose:
        print(_report(out))
    dest = out_dir / "comparison.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[write] {dest}")
    return out


def _verdict(descriptives: dict, contrasts: list[Contrast]) -> dict:
    def get(metric, a, b):
        for c in contrasts:
            if c.metric == metric and c.a == a and c.b == b:
                return c
        return None

    gp = get(PRIMARY, "grounded", "plain")
    gs = get(PRIMARY, "grounded", "selfcritique")
    gh = get(HOLDOUT, "grounded", "plain")
    guard_ok = (
        descriptives["grounded"]["direction_match"] >= descriptives["plain"]["direction_match"] - 0.05
        and descriptives["grounded"]["content_f1"] >= descriptives["plain"]["content_f1"] - 0.05
    )
    return {
        "grounded_beats_plain": bool(gp and gp.p_holm < 0.05 and gp.mean_diff < 0),
        "grounded_beats_selfcritique": bool(gs and gs.p_holm < 0.05 and gs.mean_diff < 0),
        "guardrails_hold": bool(guard_ok),
        # Higher entailment is better, so a grounded win is a positive diff.
        "holdout_grounded_vs_plain": (
            None if gh is None else {"mean_diff": gh.mean_diff, "p_holm": gh.p_holm}
        ),
        "note": (
            "A reduction in delta_strength counts only if guardrails_hold; "
            "otherwise the reviser bought calibration with content."
        ),
    }


def _report(out: dict) -> str:
    L = ["=" * 78,
         f"THREE-CONDITION COMPARISON   n={out['n_reviews']} reviews, paired",
         "=" * 78, "",
         f"  {'metric':<20}" + "".join(f"{c:>16}" for c in CONDITIONS)]
    for m in (PRIMARY, "abs_delta", "overclaims", "direction_match", "content_f1",
              "hedge_density", "n_words", HOLDOUT):
        if m not in out["descriptives"][CONDITIONS[0]]:
            continue
        L.append(f"  {m:<20}" + "".join(f"{out['descriptives'][c][m]:>16.3f}" for c in CONDITIONS))
    L += ["", "  paired contrasts (Wilcoxon signed-rank, Holm-corrected)", ""]
    for c in out["contrasts"]:
        star = "*" if c["p_holm"] < 0.05 else " "
        n_note = f" (n={c['n']})" if c["n"] != out["n_reviews"] else ""
        L.append(
            f"  {star} {c['metric']:<18} {c['a']:>13} vs {c['b']:<13} "
            f"diff={c['mean_diff']:+.3f} [{c['ci_low']:+.3f},{c['ci_high']:+.3f}] "
            f"p={c['p_holm']:.4f} rbc={c['effect_size']:+.2f}{n_note}"
        )
    v = out["verdict"]
    h = v["holdout_grounded_vs_plain"]
    holdout_line = (
        "not run (notebook Cell 11)" if h is None
        else f"diff={h['mean_diff']:+.3f}  p={h['p_holm']:.4f}  (higher = better)"
    )
    L += ["", "  verdict:",
          f"    grounded beats plain ......... {v['grounded_beats_plain']}",
          f"    grounded beats self-critique . {v['grounded_beats_selfcritique']}",
          f"    guardrails hold .............. {v['guardrails_hold']}",
          f"    held-out judge, grounded-plain {holdout_line}",
          f"    {v['note']}", "=" * 78]
    return "\n".join(L)


if __name__ == "__main__":
    compare()
