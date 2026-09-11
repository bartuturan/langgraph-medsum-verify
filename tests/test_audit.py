"""Scoring the blind audit sheet, including sheets the way Excel re-saves them."""

from __future__ import annotations

import csv
import json

import pytest

from medsumverify.eval.audit_sheet import (
    KEY_NAME,
    SCORED_NAME,
    SHEET_NAME,
    build_audit_sheet,
    read_sheet,
    score_audit,
)

CONDITIONS = ("plain", "selfcritique", "grounded")
CLAIM = "The drug reduces mortality."
EVIDENCE = "Pooled analysis suggests the drug may be associated with reduced mortality."


def _write(tmp_path, verdicts, delimiter=",", encoding="utf-8", notes=""):
    """Sheet + key where row i belongs to condition i % 3."""
    key, rows = {}, []
    for i, verdict in enumerate(verdicts):
        item = f"A{i + 1:03d}"
        key[item] = {"review_id": f"CD{i:04d}", "condition": CONDITIONS[i % 3],
                     "claim_index": 0, "claim": CLAIM}
        rows.append({"id": item, "claim": CLAIM, "evidence": EVIDENCE,
                     "verdict": verdict, "notes": notes})
    (tmp_path / KEY_NAME).write_text(json.dumps(key), encoding="utf-8")
    with open(tmp_path / SHEET_NAME, "w", encoding=encoding, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "claim", "evidence", "verdict", "notes"],
                           delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)


def test_counts_and_rates_per_condition(tmp_path):
    _write(tmp_path, ["overclaim", "faithful", "faithful"] * 4)
    out = score_audit(tmp_path, verbose=False)
    assert out["n_labelled"] == 12
    assert out["per_condition"]["plain"]["counts"]["overclaim"] == 4
    assert out["per_condition"]["plain"]["overclaim_rate"] == 1.0
    assert out["per_condition"]["grounded"]["overclaim_rate"] == 0.0
    lo, hi = out["per_condition"]["grounded"]["overclaim_ci95"]
    assert lo == 0.0 and 0 < hi < 1, "Wilson interval must stay informative at k=0"


def test_three_contrasts_are_reported(tmp_path):
    _write(tmp_path, ["overclaim", "faithful", "faithful"] * 4)
    out = score_audit(tmp_path, verbose=False)
    assert {(t["a"], t["b"]) for t in out["tests"]} == {
        ("grounded", "plain"), ("grounded", "selfcritique"), ("selfcritique", "plain")
    }
    assert all(0 <= t["p_holm"] <= 1 for t in out["tests"])


@pytest.mark.parametrize(
    "delimiter,encoding",
    [(",", "utf-8"), (";", "utf-8-sig"), (";", "cp1252")],
    ids=["plain-csv", "excel-semicolon-bom", "excel-windows-codepage"],
)
def test_reads_sheets_as_excel_saves_them(tmp_path, delimiter, encoding):
    _write(tmp_path, ["overclaim", "faithful", "faithful"], delimiter, encoding, notes="revisé")
    rows = read_sheet(tmp_path / SHEET_NAME)
    assert [r["verdict"] for r in rows] == ["overclaim", "faithful", "faithful"]
    assert score_audit(tmp_path, verbose=False)["n_labelled"] == 3


def test_aliases_accepted_invalid_reported_blank_skipped(tmp_path):
    _write(tmp_path, ["OVER", " ok ", "maybe", ""])
    out = score_audit(tmp_path, verbose=False)
    assert out["n_labelled"] == 2
    assert out["invalid"] == [{"id": "A003", "verdict": "maybe"}]
    assert out["blank"] == ["A004"]


def test_nothing_labelled_refuses_to_score(tmp_path):
    _write(tmp_path, ["", "", ""])
    with pytest.raises(RuntimeError):
        score_audit(tmp_path, verbose=False)


def test_missing_files_are_explained(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_audit(tmp_path, verbose=False)


def test_agreement_with_the_automatic_check(tmp_path):
    """A hedged source and a bare assertion: the instrument calls it an overclaim."""
    _write(tmp_path, ["overclaim"] * 6)
    ag = score_audit(tmp_path, verbose=False)["agreement_with_automatic"]
    assert ag["n"] == 6
    assert ag["exact"] == 1.0
    assert ag["overclaim_recall"] == 1.0


def test_scored_result_is_written(tmp_path):
    _write(tmp_path, ["overclaim", "faithful", "underclaim"] * 2)
    score_audit(tmp_path, verbose=False)
    saved = json.loads((tmp_path / SCORED_NAME).read_text(encoding="utf-8"))
    assert saved["n_labelled"] == 6


def test_build_refuses_to_overwrite_a_filled_sheet(tmp_path):
    """Re-running the notebook cell must never wipe someone's labels."""
    _write(tmp_path, ["overclaim", "", ""])
    with pytest.raises(FileExistsError):
        build_audit_sheet(results_dir=tmp_path)
    assert read_sheet(tmp_path / SHEET_NAME)[0]["verdict"] == "overclaim"
