"""Secondary metric: the held-out NLI judge over every experiment summary.

Run on the GPU after the loop's models are freed (notebook Cell 12). The
scores are cached to results/holdout_nli.json, so the comparison can be re-run
anywhere -- including on a laptop with no GPU -- and picks them up on its own.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import HOLDOUT_NLI_MODEL, MAX_ABSTRACTS_PER_REVIEW, paths
from ..data.cochrane import load_review_index
from ..experiment.run import load_results
from ..graph.state import CONDITIONS

__all__ = ["score_holdout", "load_holdout", "HOLDOUT_NAME"]

HOLDOUT_NAME = "holdout_nli.json"


def _key(review_id: str, condition: str) -> str:
    return f"{review_id}::{condition}"


def score_holdout(
    records: list[dict] | None = None,
    split: str = "dev",
    results_dir: Path | None = None,
    judge=None,
    max_abstracts: int = MAX_ABSTRACTS_PER_REVIEW,
) -> Path:
    """Score every finished (review, condition) summary and cache the result."""
    from ..models.nli_holdout import entailment_scores

    out_dir = Path(results_dir or paths().results)
    records = records if records is not None else load_results(out_dir)
    reviews = load_review_index(split)
    todo = [
        r for r in records
        if not r.get("error") and r.get("summary")
        and r.get("condition") in CONDITIONS and r.get("review_id") in reviews
    ]
    if not todo:
        raise RuntimeError("no experiment results to score -- run the experiment first")

    scores = entailment_scores(
        [r["summary"] for r in todo],
        [reviews[r["review_id"]].documents(max_abstracts) for r in todo],
        judge=judge,
    )
    data = {
        _key(r["review_id"], r["condition"]): (None if math.isnan(s) else round(float(s), 4))
        for r, s in zip(todo, scores)
    }
    model = getattr(judge, "model_name", type(judge).__name__) if judge else HOLDOUT_NLI_MODEL
    dest = out_dir / HOLDOUT_NAME
    dest.write_text(json.dumps({"model": model, "scores": data}, indent=2), encoding="utf-8")
    n_nan = sum(1 for v in data.values() if v is None)
    print(f"[write] {dest}  ({len(data)} summaries scored, {n_nan} without claims)")
    return dest


def load_holdout(results_dir: Path | None = None) -> dict[tuple[str, str], float]:
    """{(review_id, condition): score}; empty when the judge has not been run."""
    path = Path(results_dir or paths().results) / HOLDOUT_NAME
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8")).get("scores", {})
    out: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        if value is None:
            continue
        review_id, condition = key.split("::", 1)
        out[(review_id, condition)] = float(value)
    return out
