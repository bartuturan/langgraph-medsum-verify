"""The held-out judge's scoring logic, and its wiring into the comparison."""

from __future__ import annotations

import math
import re

import pytest

from medsumverify.models.nli_holdout import entailment_scores


class WordJudge:
    """Stand-in judge: entails when every content word of the claim is in the premise."""

    model_name = "word-judge"

    def score(self, premises, hypotheses):
        out = []
        for p, h in zip(premises, hypotheses):
            hw = set(re.findall(r"[a-z]{4,}", h.lower()))
            pw = set(re.findall(r"[a-z]{4,}", p.lower()))
            out.append(1.0 if hw and hw <= pw else 0.0)
        return out


def test_claim_supported_only_by_a_later_study_is_found():
    """The reason for scoring study by study.

    Concatenating the source would truncate to BERT's 512 tokens, roughly the
    first abstract, and this claim would look unsupported.
    """
    docs = [
        "Study one examined something unrelated entirely.",
        "Aspirin reduced mortality in older patients.",
    ]
    [s] = entailment_scores(["Aspirin reduced mortality."], [docs], judge=WordJudge())
    assert s == 1.0


def test_score_is_the_mean_over_claims():
    docs = ["Aspirin reduced mortality in older patients."]
    [s] = entailment_scores(
        ["Aspirin reduced mortality. Statins prevent strokes entirely."], [docs], judge=WordJudge()
    )
    assert s == pytest.approx(0.5)


def test_empty_summary_or_source_is_nan_not_zero():
    a, b = entailment_scores(
        ["", "Aspirin reduced mortality."], [["x" * 30], []], judge=WordJudge()
    )
    assert math.isnan(a) and math.isnan(b)


@pytest.fixture
def finished_run(tmp_path):
    from medsumverify.experiment.run import ExperimentRunner, select_reviews
    from medsumverify.models.factcheck import FakeFactChecker
    from medsumverify.models.retriever import FakeRetriever
    from medsumverify.models.writer import FakeWriter
    from medsumverify.verify.verifier import Verifier

    runner = ExperimentRunner(
        FakeWriter(), Verifier(FakeFactChecker(), FakeRetriever()), results_dir=tmp_path
    )
    runner.run(select_reviews("dev", n=6), verbose=False)
    return tmp_path


def test_scores_are_cached_and_reloaded(finished_run):
    from medsumverify.eval.holdout import load_holdout, score_holdout

    score_holdout(results_dir=finished_run, judge=WordJudge())
    scores = load_holdout(finished_run)
    assert len(scores) == 18, "6 reviews x 3 conditions"
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_comparison_picks_up_the_holdout_metric(finished_run):
    from medsumverify.eval.compare import HOLDOUT, compare
    from medsumverify.eval.holdout import score_holdout

    score_holdout(results_dir=finished_run, judge=WordJudge())
    out = compare(results_dir=finished_run, verbose=False)
    assert out["holdout"]["available"]
    assert {c["metric"] for c in out["contrasts"]} >= {HOLDOUT}
    assert HOLDOUT in out["descriptives"]["grounded"]
    assert out["verdict"]["holdout_grounded_vs_plain"] is not None
    assert (finished_run / "comparison.json").exists()


def test_comparison_still_runs_without_the_holdout(finished_run):
    from medsumverify.eval.compare import compare

    out = compare(results_dir=finished_run, verbose=False)
    assert not out["holdout"]["available"]
    assert out["verdict"]["holdout_grounded_vs_plain"] is None
