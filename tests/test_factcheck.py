"""The MiniCheck label-token guard, including the case that broke a Kaggle run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from medsumverify.models.factcheck import (
    REFUTED_PROBE,
    SUPPORTED_PROBE,
    check_label_separation,
    check_label_token_ids,
)

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


class TestLabelSeparation:
    """The guard that would have caught the d_model**-0.5 rescale on Kaggle."""

    def test_a_discriminating_model_passes(self):
        check_label_separation(lambda docs, claims: [0.969, 0.009])

    def test_the_squashed_kaggle_output_is_refused(self):
        """The exact numbers the rescale produced for these two probes.

        Both sit near 0.5, so the flags were still right -- softmax over two
        logits keeps the sign -- but no threshold other than 0.5 could be
        fitted and the probability shown to the Reviser was meaningless.
        """
        with pytest.raises(RuntimeError, match="not discriminating"):
            check_label_separation(lambda docs, claims: [0.5316, 0.4624])

    def test_the_error_names_the_rescale(self):
        with pytest.raises(RuntimeError, match=r"d_model\*\*-0\.5"):
            check_label_separation(lambda docs, claims: [0.51, 0.49])

    @pytest.mark.parametrize(
        "scores",
        [[0.009, 0.969], [0.969, 0.969], [0.009, 0.009]],
        ids=["inverted", "both-supported", "both-refuted"],
    )
    def test_degenerate_orderings_are_refused(self, scores):
        with pytest.raises(RuntimeError, match="not discriminating"):
            check_label_separation(lambda docs, claims: scores)

    def test_the_probes_are_passed_through_as_given(self):
        seen = {}

        def spy(docs, claims):
            seen["pairs"] = list(zip(docs, claims))
            return [0.969, 0.009]

        check_label_separation(spy)
        assert seen["pairs"] == [SUPPORTED_PROBE, REFUTED_PROBE]


def test_self_test_does_not_recurse_into_ensure():
    """`self_test` must call `_score_loaded`, not `score`.

    `score` calls `_ensure`, which calls `self_test` -- so wiring the guard to
    the public method is an infinite recursion that only shows up with real
    weights loaded.
    """
    from medsumverify.models.factcheck import MiniCheckFactChecker

    fc = MiniCheckFactChecker()
    fc._tok = FakeTok(REAL_ENCODING)
    fc._score_loaded = lambda docs, claims: [0.969, 0.009]
    fc.self_test()  # must not recurse or touch the network


class StubRegistry:
    """Hands back a pre-made (tokenizer, model) so `_ensure` never loads weights."""

    def __init__(self, payload):
        self.payload = payload
        self.loads = 0

    def get(self, key, loader):
        self.loads += 1
        return self.payload


def test_a_failed_self_test_raises_every_time(monkeypatch):
    """Not just on the first call.

    The registry caches the model, so an `_ensure` guarded only by
    `self._model is not None` would skip the self-test on the second call and
    let a known-broken checkpoint through. Guarding on `_verified` too is what
    makes the refusal stick.
    """
    from medsumverify.models import factcheck as fcmod

    reg = StubRegistry((FakeTok(REAL_ENCODING), object()))
    monkeypatch.setattr(fcmod, "get_registry", lambda: reg)

    fc = fcmod.MiniCheckFactChecker()
    # A model that has stopped discriminating -- the squashed Kaggle output.
    fc._score_loaded = lambda docs, claims: [0.5316, 0.4624]

    for _ in range(2):
        with pytest.raises(RuntimeError, match="not discriminating"):
            fc._ensure()
    assert not fc._verified


def test_a_passing_self_test_runs_once_and_is_remembered(monkeypatch):
    from medsumverify.models import factcheck as fcmod

    reg = StubRegistry((FakeTok(REAL_ENCODING), object()))
    monkeypatch.setattr(fcmod, "get_registry", lambda: reg)

    fc = fcmod.MiniCheckFactChecker()
    calls = []
    fc._score_loaded = lambda docs, claims: (calls.append(1), [0.969, 0.009])[1]

    fc._ensure()
    after_first = len(calls)
    fc._ensure()
    assert fc._verified
    # A healthy load scores the probes twice: once for the rescale check, once
    # for the separation guard. What matters is that a second _ensure adds none.
    assert after_first == 2
    assert len(calls) == after_first, "self-test should not re-run once it has passed"
    assert reg.loads == 1


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


# --------------------------------------------------------------------------
# Cancelling a d_model**-0.5 rescale the loader could not prevent. Pinning the
# config was not enough on Kaggle: the probes still came back 0.5269 / 0.4636.
# --------------------------------------------------------------------------


class FakeHead:
    def __init__(self, torch, scale=1.0):
        self.weight = torch.tensor([scale], dtype=torch.float32)


class FakeModel:
    """A stand-in whose probe output tracks its lm_head scale, as the real one does."""

    def __init__(self, torch, logit_delta=4.0, applied_scale=1.0, d_model=1024):
        self.config = SimpleNamespace(d_model=d_model, tie_word_embeddings=True)
        self.lm_head = FakeHead(torch, applied_scale)
        self.logit_delta = logit_delta

    def probs(self):
        import math

        d = self.logit_delta * float(self.lm_head.weight[0])
        hi = 1 / (1 + math.exp(-d))
        return [hi, 1 - hi]


def _checker_over(model):
    from medsumverify.models.factcheck import MiniCheckFactChecker

    fc = MiniCheckFactChecker()
    fc._model = model
    fc._tok = FakeTok(REAL_ENCODING)
    fc._score_loaded = lambda docs, claims: model.probs()
    return fc


def test_a_healthy_model_is_left_alone():
    torch = pytest.importorskip("torch")
    m = FakeModel(torch, applied_scale=1.0)          # probes ~0.982 / ~0.018
    fc = _checker_over(m)
    assert fc._undo_rescale_if_present() is False
    assert float(m.lm_head.weight[0]) == 1.0
    fc.self_test()


def test_the_rescale_is_detected_and_cancelled():
    torch = pytest.importorskip("torch")
    m = FakeModel(torch, applied_scale=1 / 32)       # exactly the Kaggle failure
    fc = _checker_over(m)
    hi, lo = fc._probe()
    assert 0.40 < lo < 0.50 < hi < 0.60, "precondition: the squashed band"

    assert fc._undo_rescale_if_present() is True
    assert float(m.lm_head.weight[0]) == pytest.approx(1.0)
    fc.self_test()                                    # now passes


def test_the_repair_is_idempotent():
    torch = pytest.importorskip("torch")
    m = FakeModel(torch, applied_scale=1 / 32)
    fc = _checker_over(m)
    assert fc._undo_rescale_if_present() is True
    assert fc._undo_rescale_if_present() is False     # second call is a no-op
    assert float(m.lm_head.weight[0]) == pytest.approx(1.0)


def test_a_differently_broken_model_is_not_silently_rescaled():
    """The repair must not paper over a checkpoint that is wrong another way.

    Inverted probes are not the rescale's signature -- dividing logits by 32
    cannot reorder them -- so the weights are left untouched and self_test
    rejects the load.
    """
    torch = pytest.importorskip("torch")
    m = FakeModel(torch, logit_delta=-0.2, applied_scale=1.0)   # hi < lo
    fc = _checker_over(m)
    assert fc._undo_rescale_if_present() is False
    assert float(m.lm_head.weight[0]) == 1.0
    with pytest.raises(RuntimeError, match="not discriminating"):
        fc.self_test()


def test_a_failed_repair_still_fails_the_load():
    """Repair is attempted, but self_test is the authority either way."""
    torch = pytest.importorskip("torch")
    m = FakeModel(torch, logit_delta=0.02, applied_scale=1 / 32)  # too weak to rescue
    fc = _checker_over(m)
    fc._undo_rescale_if_present()
    with pytest.raises(RuntimeError, match="not discriminating"):
        fc.self_test()
