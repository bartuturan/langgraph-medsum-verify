"""Run the three conditions over a set of Cochrane reviews.

Two properties matter more than speed here.

*Pairing.* The Round-0 draft is generated once per review and injected into all
three conditions, so any difference between them is attributable to what
happens after the draft. Drafts are cached to disk, which also means a resumed
run reuses the identical draft rather than regenerating a slightly different
one.

*Crash safety.* Results are appended to JSONL after every single
(review, condition), and a restart skips whatever is already on disk. A dropped
Kaggle session costs minutes, not the day.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..config import MAX_ROUNDS, N_EXPERIMENT_DOCS, SEED, paths
from ..data.cochrane import Review, load_reviews
from ..graph.build import run_condition
from ..graph.state import CONDITIONS
from ..models.writer import Writer
from ..verify.verifier import Verifier

__all__ = ["ExperimentRunner", "select_reviews", "load_results", "RESULTS_NAME"]

RESULTS_NAME = "experiment.jsonl"
DRAFTS_NAME = "drafts.jsonl"


def select_reviews(
    split: str = "dev",
    n: int = N_EXPERIMENT_DOCS,
    seed: int = SEED,
    min_studies: int = 2,
    max_studies: int = 25,
) -> list[Review]:
    """A deterministic sample of genuinely multi-document reviews.

    Single-study reviews are excluded because the task is multi-document
    summarization; very large ones are excluded because they blow the writer's
    context and slow the run without changing what is being measured.
    """
    import random

    pool = [
        r for r in load_reviews(split)
        if min_studies <= r.n_studies <= max_studies and r.target
    ]
    pool.sort(key=lambda r: r.review_id)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A half-written final line from a killed session; ignore it.
                continue
    return out


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def load_results(results_dir: Path | None = None) -> list[dict]:
    return _read_jsonl((results_dir or paths().results) / RESULTS_NAME)


@dataclass
class ExperimentRunner:
    writer: Writer
    verifier: Verifier
    results_dir: Path | None = None
    max_rounds: int = MAX_ROUNDS

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir or paths().results)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.results_dir / RESULTS_NAME
        self.drafts_path = self.results_dir / DRAFTS_NAME

    # -- drafts ------------------------------------------------------------

    def _draft_cache(self) -> dict[str, str]:
        return {r["review_id"]: r["draft"] for r in _read_jsonl(self.drafts_path)}

    def get_draft(self, review: Review, cache: dict[str, str]) -> str:
        """One draft per review, reused by all three conditions and across restarts."""
        if review.review_id in cache:
            return cache[review.review_id]
        state = run_condition(
            "plain", review.review_id, review.documents(), self.writer,
            None, "", self.max_rounds,
        )
        draft = state["summary"]
        cache[review.review_id] = draft
        _append_jsonl(self.drafts_path, {"review_id": review.review_id, "draft": draft})
        return draft

    # -- main loop ---------------------------------------------------------

    def completed(self) -> set[tuple[str, str]]:
        return {(r["review_id"], r["condition"]) for r in load_results(self.results_dir)}

    def run(
        self,
        reviews: Sequence[Review],
        conditions: Iterable[str] = CONDITIONS,
        verbose: bool = True,
    ) -> list[dict]:
        conditions = list(conditions)
        done = self.completed()
        cache = self._draft_cache()
        written: list[dict] = []

        for n, review in enumerate(reviews, 1):
            pending = [c for c in conditions if (review.review_id, c) not in done]
            if not pending:
                if verbose:
                    print(f"[{n}/{len(reviews)}] {review.review_id} already done")
                continue
            try:
                draft = self.get_draft(review, cache)
            except Exception:
                _append_jsonl(
                    self.results_path,
                    {"review_id": review.review_id, "condition": "draft",
                     "error": traceback.format_exc()[-1500:]},
                )
                if verbose:
                    print(f"[{n}/{len(reviews)}] {review.review_id} DRAFT FAILED")
                continue

            for condition in pending:
                t0 = time.time()
                try:
                    state = run_condition(
                        condition, review.review_id, review.documents(), self.writer,
                        self.verifier if condition == "grounded" else None,
                        draft, self.max_rounds,
                    )
                    record = {
                        "review_id": review.review_id,
                        "condition": condition,
                        "n_studies": review.n_studies,
                        "draft": draft,
                        "summary": state.get("summary", ""),
                        "rounds": state.get("round", 0),
                        "n_llm_calls": state.get("n_llm_calls", 0),
                        "history": state.get("history", []),
                        "reports": state.get("reports", []),
                        "critiques": state.get("critiques", []),
                        "seconds": round(time.time() - t0, 2),
                        # target is stored for scoring only; the loop never saw it
                        "target": review.target,
                    }
                except Exception:
                    record = {
                        "review_id": review.review_id, "condition": condition,
                        "error": traceback.format_exc()[-1500:],
                        "seconds": round(time.time() - t0, 2),
                    }
                _append_jsonl(self.results_path, record)
                written.append(record)
                if verbose:
                    flag = "ERR" if "error" in record else f"r{record['rounds']}"
                    print(
                        f"[{n}/{len(reviews)}] {review.review_id} {condition:13s} "
                        f"{flag} {record['seconds']:6.1f}s"
                    )
        return written
