"""The MiniCheck label-token guard, including the case that broke a Kaggle run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from medsumverify.models.factcheck import check_label_token_ids

# Measured on lytang/MiniCheck-Flan-T5-Large's own tokenizer.
REAL_ENCODING = {"0": [3, 632, 1], "1": [209, 1]}


class FakeTok:
    def __init__(self, table):
        self.table = table

    def __call__(self, text):
        return SimpleNamespace(input_ids=self.table[text])


def test_the_real_checkpoint_passes():
    """Regression: id 3 is '▁' and id 209 is '▁1', and that is correct.

    The first guard demanded that the ids *spell* "0" and "1" and so rejected
    this exact checkpoint on Kaggle. What MiniCheck needs is that they are the
    first tokens of the two labels.
    """
    check_label_token_ids(FakeTok(REAL_ENCODING))


@pytest.mark.parametrize(
    "table",
    [
        {"0": [632, 1], "1": [209, 1]},
        {"0": [3, 632, 1], "1": [3, 209, 1]},
        {"0": [], "1": [209, 1]},
    ],
    ids=["zero-moved", "one-moved", "empty-encoding"],
)
def test_moved_label_tokens_are_refused(table):
    with pytest.raises(RuntimeError, match="label tokens moved"):
        check_label_token_ids(FakeTok(table))


def test_against_the_real_tokenizer_when_cached():
    """Runs only where the tokenizer is already downloaded; never hits the network."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "lytang/MiniCheck-Flan-T5-Large", local_files_only=True
        )
    except Exception as exc:  # not cached, or offline
        pytest.skip(f"MiniCheck tokenizer not cached locally: {exc}")
    check_label_token_ids(tok)
    # The doc/claim separator must be the real end-of-sequence token, not the
    # literal characters "</s>" -- every score depends on it.
    ids = tok(f"predict: A drug may help.{tok.eos_token}A drug helps.").input_ids
    assert ids.count(tok.eos_token_id) == 2
