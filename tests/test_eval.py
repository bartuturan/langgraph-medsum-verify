"""Evaluation-side guarantees, including the ones that keep the result honest."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from medsumverify.eval.metrics import content_f1, score_summary

SRC = Path(__file__).resolve().parents[1] / "src" / "medsumverify"


def test_overclaim_is_detected_against_the_reference():
    m = score_summary("The drug reduces mortality.", "The drug may reduce mortality.")
    assert m.delta_strength > 0 and m.overclaims


def test_underclaim_is_detected_too():
    """Over-hedging must be visible, not silently scored as an improvement."""
    m = score_summary("The drug may possibly reduce mortality.", "The drug reduces mortality.")
    assert m.delta_strength < 0 and not m.overclaims
    assert m.abs_delta > 0, "abs_delta must catch miscalibration in either direction"


def test_matched_strength_scores_near_zero():
    text = "The drug may reduce mortality."
    assert abs(score_summary(text, text).delta_strength) < 1e-9


def test_content_f1_notices_vacuous_summaries():
    target = "Antibiotics reduce postoperative wound infection in abdominal surgery."
    rich = "Antibiotics may reduce postoperative wound infection in abdominal surgery."
    vacuous = "There is insufficient evidence to draw conclusions."
    assert content_f1(rich, target) > content_f1(vacuous, target)


def test_hedge_density_rises_with_hedging():
    target = "The drug reduces mortality."
    plain = score_summary("The drug reduces mortality.", target)
    hedged = score_summary("The drug may possibly reduce mortality.", target)
    assert hedged.hedge_density > plain.hedge_density


def test_direction_match_is_reported():
    m = score_summary("The drug reduces mortality.", "The drug increases mortality.")
    assert not m.direction_match


def test_metrics_are_json_serialisable():
    import json

    json.dumps(score_summary("The drug reduces mortality.", "The drug may reduce mortality.").to_dict())


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_the_loop_cannot_reach_the_holdout_judge():
    """Circularity guard.

    The grounded condition optimizes against MiniCheck, so the held-out NLI
    model is what makes the final score independent. If anything under graph/
    or verify/ ever imports it, that independence is gone -- and it would be a
    quiet, plausible-looking change. Fail loudly instead.
    """
    offenders = []
    for sub in ("graph", "verify"):
        for path in (SRC / sub).rglob("*.py"):
            if any("nli_holdout" in imp for imp in _imports_of(path)):
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"held-out judge reachable from the loop: {offenders}"


def test_the_loop_cannot_reach_the_reference_target():
    """The Cochrane conclusion is the yardstick; no condition may read it.

    Checked as an import invariant rather than a text search: the data modules
    are the only things that carry `target`, so loop code importing either of
    them is the failure mode worth catching. The reference reaches results only
    via the experiment runner, which stores it for scoring after the fact.
    """
    forbidden = ("data.cochrane", "data.annotations", "medsumverify.data")
    offenders = []
    for sub in ("graph", "verify"):
        for path in (SRC / sub).rglob("*.py"):
            for imp in _imports_of(path):
                if any(f in imp for f in forbidden):
                    offenders.append(f"{path.relative_to(SRC)} -> {imp}")
    assert not offenders, f"loop code can read the reference conclusion: {offenders}"


def test_holm_correction_is_monotone():
    from medsumverify.eval.compare import _holm

    adj = _holm([0.01, 0.02, 0.03])
    assert all(a >= b for a, b in zip(adj, [0.01, 0.02, 0.03]))
    assert all(0 <= a <= 1 for a in adj)


def test_compare_refuses_to_report_on_too_little_data():
    from medsumverify.eval.compare import compare

    with pytest.raises(RuntimeError):
        compare(records=[], verbose=False)


def _records(target, plain, selfcritique, grounded, n=8):
    out = []
    for i in range(n):
        for condition, summary in (("plain", plain), ("selfcritique", selfcritique),
                                   ("grounded", grounded)):
            out.append({"review_id": f"CD{i:04d}", "condition": condition,
                        "summary": summary, "target": target})
    return out


def test_verdict_rewards_fixing_an_overclaim(tmp_path):
    from medsumverify.eval.compare import compare

    recs = _records(
        target="Aspirin may reduce mortality.",
        plain="Aspirin significantly reduces mortality.",
        selfcritique="Aspirin significantly reduces mortality.",
        grounded="Aspirin may reduce mortality.",
    )
    out = compare(recs, verbose=False, results_dir=tmp_path)
    assert out["primary_metric"] == "abs_delta"
    assert out["verdict"]["grounded_beats_plain"]


def test_verdict_does_not_reward_pushing_a_weak_claim_weaker(tmp_path):
    """The reason the primary metric is two-sided.

    Plain already underclaims. Grounded hedges further, which *lowers* the
    signed delta -- the old verdict counted that as a win. It moves the summary
    further from the reviewer's conclusion, so abs_delta must call it a loss.
    """
    from medsumverify.eval.compare import compare

    recs = _records(
        target="Aspirin significantly reduces mortality.",
        plain="Aspirin may reduce mortality.",
        selfcritique="Aspirin may reduce mortality.",
        grounded="Aspirin may possibly reduce mortality.",
    )
    out = compare(recs, verbose=False, results_dir=tmp_path)
    assert out["verdict"]["signed_shift_grounded_vs_plain"]["mean_diff"] < 0, "setup: signed delta fell"
    assert not out["verdict"]["grounded_beats_plain"]
