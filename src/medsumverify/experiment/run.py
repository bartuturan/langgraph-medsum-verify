"""Run the three conditions over a set of Cochrane reviews.

Two properties matter more than speed here.

*Pairing.* The Round-0 draft is generated once per review and injected into all
three conditions, so any difference between them is attributable to what
happens after the draft. Drafts are cached to disk, which also means a resumed
run reuses the identical draft rather than regenerating a slightly different
one.

*Crash safety.* Results are appended to JSONL after every single
(review, condition). A restart skips whatever finished successfully and
retries whatever failed, so an interruption within a session costs minutes,
not the day. A new Kaggle session starts empty unless Persistence is switched
on; see the restore lines in notebook Cell 3.
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

    # -- bookkeeping -------------------------------------------------------

    def _outcomes(self) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        """(succeeded, failed-and-never-succeeded) keys across all attempts on disk.

        A key that failed once and later succeeded counts as succeeded. Draft
        failures are recorded under the pseudo-condition "draft".
        """
        ok: set[tuple[str, str]] = set()
        bad: set[tuple[str, str]] = set()
        for r in load_results(self.results_dir):
            key = (r.get("review_id"), r.get("condition"))
            (bad if r.get("error") else ok).add(key)
        return ok, bad - ok

    def completed(self) -> set[tuple[str, str]]:
        """(review, condition) pairs that finished *successfully*.

        Failed attempts are deliberately excluded so that a resume retries
        them. They used to count as finished: one transient failure -- a CUDA
        out-of-memory, a download hiccup -- then skipped that step for good, and
        the review silently fell out of the paired analysis, which needs all
        three conditions.
        """
        return self._outcomes()[0]

    def failed(self) -> set[tuple[str, str]]:
        """Conditions whose attempts so far have all failed; the next run retries them."""
        return {k for k in self._outcomes()[1] if k[1] in CONDITIONS}

    def discard(self, *conditions: str) -> int:
        """Drop every record of the given conditions so the next run redoes them.

        For when something a condition depends on changes after it ran -- the
        verifier and its scorer, which only the grounded condition uses. The
        cached drafts are kept, so the redo starts from the identical Round-0
        text and the pairing holds. The file is copied aside first, so nothing
        is lost. Returns the number of records dropped.
        """
        unknown = set(conditions) - set(CONDITIONS)
        if not conditions or unknown:
            raise ValueError(f"expected one or more of {CONDITIONS}, got {conditions!r}")
        if not self.results_path.exists():
            return 0

        import shutil

        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.results_path.with_name(f"{self.results_path.name}.{stamp}.bak")
        shutil.copy2(self.results_path, backup)

        kept, dropped = [], 0
        for r in _read_jsonl(self.results_path):
            if r.get("condition") in conditions:
                dropped += 1
            else:
                kept.append(r)
        with open(self.results_path, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[discard] removed {dropped} record(s) for {', '.join(conditions)}; "
              f"backup: {backup.name}")
        return dropped

    # -- main loop ---------------------------------------------------------

    def run(
        self,
        reviews: Sequence[Review],
        conditions: Iterable[str] = CONDITIONS,
        verbose: bool = True,
    ) -> list[dict]:
        conditions = list(conditions)
        done, failed_before = self._outcomes()
        cache = self._draft_cache()
        written: list[dict] = []
        n_failed_drafts = 0

        if verbose:
            wanted = {r.review_id for r in reviews}
            retrying = sorted(
                (rid, c) for rid, c in failed_before
                if rid in wanted and (c in conditions or (c == "draft" and rid not in cache))
            )
            if retrying:
                shown = ", ".join(f"{rid}/{c}" for rid, c in retrying[:8])
                more = f" and {len(retrying) - 8} more" if len(retrying) > 8 else ""
                print(f"[resume] retrying {len(retrying)} step(s) that failed before: {shown}{more}")

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
                n_failed_drafts += 1
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

        if verbose:
            n_err = sum(1 for r in written if "error" in r) + n_failed_drafts
            msg = f"[done] {len(written) - (n_err - n_failed_drafts)} step(s) finished"
            if n_err:
                msg += f", {n_err} failed -- re-run to retry them"
            print(msg)
        return written
