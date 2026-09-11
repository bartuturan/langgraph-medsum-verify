"""Experiment-runner guarantees: sampling stability, pairing, crash safety."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from medsumverify.experiment.run import ExperimentRunner, load_results, select_reviews
from medsumverify.graph.state import CONDITIONS
from medsumverify.models.factcheck import FakeFactChecker
from medsumverify.models.retriever import FakeRetriever
from medsumverify.models.writer import FakeWriter
from medsumverify.verify.verifier import Verifier


@pytest.fixture(scope="module")
def reviews():
    return select_reviews("dev", n=6)


@pytest.fixture
def runner(tmp_path):
    return ExperimentRunner(
        FakeWriter(), Verifier(FakeFactChecker(), FakeRetriever()), results_dir=tmp_path
    )


def test_sampling_is_deterministic():
    assert [r.review_id for r in select_reviews("dev", n=10)] == [
        r.review_id for r in select_reviews("dev", n=10)
    ]


def test_smaller_run_is_a_prefix_of_the_larger_one():
    """Shrinking the run must not resample.

    "If you run out of time, shrink the number of documents" only works if the
    30 already on disk are the first 30 of the 50 -- otherwise cutting the run
    invalidates everything computed so far.
    """
    big = [r.review_id for r in select_reviews("dev", n=50)]
    assert [r.review_id for r in select_reviews("dev", n=30)] == big[:30]


def test_selection_excludes_single_study_reviews(reviews):
    assert all(r.n_studies >= 2 for r in reviews), "task is multi-document"
    assert all(r.target for r in reviews), "scoring needs the reference"


def test_all_conditions_share_one_draft(runner, reviews):
    runner.run(reviews[:2], verbose=False)
    recs = load_results(runner.results_dir)
    by_review: dict[str, set[str]] = {}
    for r in recs:
        by_review.setdefault(r["review_id"], set()).add(r["draft"])
    assert all(len(v) == 1 for v in by_review.values()), "conditions diverged before round 0"


def test_results_are_written_per_record_and_resume(runner, reviews):
    runner.run(reviews[:2], verbose=False)
    first = len(load_results(runner.results_dir))
    assert first == 6, f"expected 2 reviews x 3 conditions, got {first}"

    again = runner.run(reviews[:2], verbose=False)
    assert again == [], "resume re-ran completed work"
    assert len(load_results(runner.results_dir)) == first


def test_partial_results_extend_rather_than_restart(runner, reviews):
    runner.run(reviews[:2], conditions=["plain"], verbose=False)
    assert len(load_results(runner.results_dir)) == 2
    runner.run(reviews[:2], verbose=False)
    assert len(load_results(runner.results_dir)) == 6


def test_truncated_final_line_is_survivable(runner, reviews):
    """A session killed mid-write leaves half a line; it must not break resume."""
    runner.run(reviews[:1], verbose=False)
    with open(runner.results_path, "a", encoding="utf-8") as fh:
        fh.write('{"review_id": "CD9999", "condition": "plai')
    assert len(load_results(runner.results_dir)) == 3


def test_records_carry_everything_the_analysis_needs(runner, reviews):
    runner.run(reviews[:1], verbose=False)
    rec = load_results(runner.results_dir)[0]
    for key in ("review_id", "condition", "draft", "summary", "target", "rounds", "history"):
        assert key in rec, f"missing {key}"
    json.dumps(rec)


def test_draft_cache_survives_a_new_runner(tmp_path, reviews):
    a = ExperimentRunner(FakeWriter(), Verifier(FakeFactChecker(), FakeRetriever()), results_dir=tmp_path)
    a.run(reviews[:1], conditions=["plain"], verbose=False)
    drafted = load_results(tmp_path)[0]["draft"]

    b = ExperimentRunner(FakeWriter(), Verifier(FakeFactChecker(), FakeRetriever()), results_dir=tmp_path)
    b.run(reviews[:1], conditions=["grounded"], verbose=False)
    assert load_results(tmp_path)[-1]["draft"] == drafted, "restart produced a different draft"


# --------------------------------------------------------------------------
# Failures must be retried on resume, not treated as finished
# --------------------------------------------------------------------------


class FlakyVerifier:
    """Raises on its first `fails` calls -- a stand-in for a transient CUDA OOM."""

    def __init__(self, inner, fails: int = 1):
        self.inner, self.fails = inner, fails

    def verify(self, summary, documents):
        if self.fails > 0:
            self.fails -= 1
            raise RuntimeError("simulated CUDA out of memory")
        return self.inner.verify(summary, documents)


@dataclass
class FlakyDraftWriter(FakeWriter):
    """Fails its first draft request, then behaves."""

    fail_drafts: int = 1

    def chat(self, system, user, max_new_tokens=320, role="draft"):
        if role == "draft" and self.fail_drafts > 0:
            self.fail_drafts -= 1
            raise RuntimeError("simulated download hiccup")
        return super().chat(system, user, max_new_tokens, role)


def test_failed_condition_is_retried_on_resume(tmp_path, reviews):
    """A transient failure must not permanently drop a review from the comparison.

    Failed records used to count as finished, so the step was never retried
    and the review silently fell out of the paired analysis, which needs all
    three conditions.
    """
    flaky = FlakyVerifier(Verifier(FakeFactChecker(), FakeRetriever()), fails=1)
    runner = ExperimentRunner(FakeWriter(), flaky, results_dir=tmp_path)
    rid = reviews[0].review_id

    runner.run(reviews[:1], verbose=False)
    assert (rid, "grounded") in runner.failed()
    assert (rid, "grounded") not in runner.completed()

    retried = runner.run(reviews[:1], verbose=False)
    assert [(r["condition"], "error" in r) for r in retried] == [("grounded", False)]
    assert not runner.failed()

    from medsumverify.eval.compare import build_metric_table

    assert rid in build_metric_table(load_results(tmp_path)), \
        "the successful retry must complete the review for the paired analysis"


def test_failed_draft_is_retried_on_resume(tmp_path, reviews):
    runner = ExperimentRunner(
        FlakyDraftWriter(), Verifier(FakeFactChecker(), FakeRetriever()), results_dir=tmp_path
    )
    assert runner.run(reviews[:1], verbose=False) == [], "nothing can run without a draft"
    [rec] = load_results(tmp_path)
    assert rec["condition"] == "draft" and "error" in rec

    written = runner.run(reviews[:1], verbose=False)
    assert sorted(r["condition"] for r in written) == sorted(CONDITIONS)


def test_success_is_never_rerun_even_after_an_earlier_failure(tmp_path, reviews):
    """An error record followed by a success means done -- don't redo the success."""
    flaky = FlakyVerifier(Verifier(FakeFactChecker(), FakeRetriever()), fails=1)
    runner = ExperimentRunner(FakeWriter(), flaky, results_dir=tmp_path)
    runner.run(reviews[:1], verbose=False)   # grounded fails
    runner.run(reviews[:1], verbose=False)   # grounded retried, succeeds
    assert runner.run(reviews[:1], verbose=False) == []


# --------------------------------------------------------------------------
# Redoing one condition after something it depends on changed
# --------------------------------------------------------------------------


def test_discard_redoes_only_that_condition_from_the_same_drafts(runner, reviews):
    runner.run(reviews[:2], verbose=False)
    drafts = runner._draft_cache()

    assert runner.discard("grounded") == 2
    assert {r["condition"] for r in load_results(runner.results_dir)} == {"plain", "selfcritique"}
    assert runner._draft_cache() == drafts, "discard must not touch the cached drafts"
    assert list(runner.results_dir.glob("experiment.jsonl.*.bak")), "no backup was written"

    redo = runner.run(reviews[:2], verbose=False)
    assert sorted(r["condition"] for r in redo) == ["grounded", "grounded"]
    assert all(r["draft"] == drafts[r["review_id"]] for r in redo), "redo broke the pairing"


def test_discard_rejects_unknown_or_missing_conditions(runner):
    with pytest.raises(ValueError):
        runner.discard("groundd")
    with pytest.raises(ValueError):
        runner.discard()


def test_discard_on_an_empty_run_is_a_no_op(runner):
    assert runner.discard("grounded") == 0


def test_run_survives_its_results_folder_disappearing(tmp_path, reviews):
    """Regression: deleting results/ after the runner was built crashed the run.

    The write that records a result raised FileNotFoundError -- including the
    one inside the handler meant to record a failure -- so the whole notebook
    cell died instead of logging and moving on.
    """
    import shutil

    d = tmp_path / "results"
    runner = ExperimentRunner(
        FakeWriter(), Verifier(FakeFactChecker(), FakeRetriever()), results_dir=d
    )
    shutil.rmtree(d)
    written = runner.run(reviews[:1], verbose=False)
    assert len(written) == 3 and not any("error" in r for r in written)
    assert len(load_results(d)) == 3, "results must land in the recreated folder"
