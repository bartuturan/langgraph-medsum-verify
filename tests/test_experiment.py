"""Experiment-runner guarantees: sampling stability, pairing, crash safety."""

from __future__ import annotations

import json

import pytest

from medsumverify.experiment.run import ExperimentRunner, load_results, select_reviews
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
