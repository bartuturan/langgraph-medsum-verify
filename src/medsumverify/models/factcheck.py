"""MiniCheck: does this document support this claim?

Implemented against `transformers` directly rather than the `minicheck` pip
package, which pulls its own pinned dependency tree and tends to fight the
Kaggle image. The format below is the one MiniCheck's own inference code uses
for the Flan-T5 checkpoint:

    input           "predict: " + document + <eos> + claim
    decoder input   a single zero token (T5's decoder-start / pad id)
    score           softmax over the first-step logits at token ids [3, 209];
                    index 1 is P(supported)

Why 3 and 209: the model was trained to emit the label "0" or "1", and those
ids are the *first tokens* of each label. "1" is the single token '▁1' (209),
but "0" splits into '▁' (3) + '0' (632), so the first decoding step for
"unsupported" is the bare word-boundary piece. Measured on the real tokenizer:

    tok("0").input_ids == [3, 632, 1]
    tok("1").input_ids == [209, 1]

`check_label_token_ids()` verifies exactly that at load time, and the loader
refuses to run if it no longer holds -- a silent mismatch would poison every
number downstream. (An earlier version instead checked that the ids *spell*
"0"/"1", which is false for id 3, and so rejected a correct checkpoint.)

The d_model**-0.5 trap
----------------------
`T5ForConditionalGeneration` multiplies the decoder output by
`d_model ** -0.5` before the lm_head **when `config.tie_word_embeddings` is
true**. MiniCheck's config says true, but the checkpoint carries its own
fine-tuned `lm_head.weight`, so the two disagree and the installed
transformers version decides what happens. transformers 5.17 notices the
conflict, refuses to tie, and skips the rescale; older versions applied it.

For flan-t5-large, d_model is 1024, so the rescale divides every logit by
exactly 32. Softmax over two logits then returns `sigmoid(delta / 32)`, which
squeezes the whole output into a narrow band around 0.5 -- measured on a
Kaggle run, [0.4624, 0.5316] across 259 claims, against [0.0079, 0.9827] for
the same pairs loaded correctly (Spearman +0.99997 between them).

What that did and did not break is worth keeping straight, because it is
counter-intuitive: `sigmoid` is monotone and does not move the sign of
`delta`, so `p < 0.5` is invariant to the rescale and every `unsupported`
flag fired exactly where it should have (0 disagreements in 259). What died
was the *magnitude*: `tau_support` could only ever be 0.5 (0.4 flags nothing,
0.6 flags everything), and the probability quoted to the Reviser in
`ClaimReport.reason()` was meaningless.

So the config is pinned at load, and `check_label_separation()` verifies the
outcome rather than the mechanism -- the mechanism is the part that changed
under us once already.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import FACTCHECK_MODEL
from .registry import device_of, dtype_kwargs, get_registry, preferred_dtype

__all__ = [
    "FactChecker",
    "MiniCheckFactChecker",
    "FakeFactChecker",
    "load_factchecker",
    "check_label_token_ids",
    "check_label_separation",
]

# MiniCheck reads its verdict off these two vocabulary positions.
_UNSUPPORTED_TOKEN_ID = 3
_SUPPORTED_TOKEN_ID = 209


def check_label_token_ids(tokenizer) -> None:
    """Fail loudly unless ids 3 and 209 are the first tokens of "0" and "1".

    That is the property MiniCheck's scoring depends on: at the first decoding
    step the model starts either the label "0" or the label "1", and those are
    the two logits compared.
    """
    first = {}
    for label in ("0", "1"):
        ids = list(tokenizer(label).input_ids)
        first[label] = ids[0] if ids else None
    expected = {"0": _UNSUPPORTED_TOKEN_ID, "1": _SUPPORTED_TOKEN_ID}
    if first != expected:
        raise RuntimeError(
            f"MiniCheck label tokens moved: '0' now starts with id {first['0']} and "
            f"'1' with id {first['1']} (expected {_UNSUPPORTED_TOKEN_ID} and "
            f"{_SUPPORTED_TOKEN_ID}). Scores would be meaningless -- check the "
            "checkpoint and tokenizer."
        )


# Two pairs whose answer is not in doubt, used to check at load time that the
# model still discriminates at all. Measured with a correctly loaded checkpoint:
# 0.969 and 0.009. Under the tie_word_embeddings rescale described above they
# come back as 0.532 and 0.462, which is what these bounds are set to catch.
SUPPORTED_PROBE = ("Aspirin reduced mortality in older patients.", "Aspirin reduces mortality.")
REFUTED_PROBE = ("Aspirin reduced mortality in older patients.", "Aspirin increases mortality.")
_PROBE_HIGH = 0.70
_PROBE_LOW = 0.30


def check_label_separation(score_fn) -> None:
    """Fail loudly unless the two probe pairs land on opposite sides.

    `check_label_token_ids` proves the verdict is being read off the right two
    vocabulary positions. It cannot tell whether the numbers coming out mean
    anything, and on a Kaggle run they did not: every score landed in
    [0.462, 0.532] and the 0.5 threshold sliced the distribution at its own
    mean. The flags survived that by luck -- softmax over two logits keeps the
    sign of their difference, so p < 0.5 was still correct -- but the
    probabilities were unusable and no other threshold could ever have been
    fitted.

    `score_fn` takes (docs, claims) and returns P(supported) per pair.
    """
    docs = [SUPPORTED_PROBE[0], REFUTED_PROBE[0]]
    claims = [SUPPORTED_PROBE[1], REFUTED_PROBE[1]]
    hi, lo = score_fn(docs, claims)
    if not (hi >= _PROBE_HIGH and lo <= _PROBE_LOW):
        raise RuntimeError(
            f"MiniCheck is not discriminating: the supported probe scored {hi:.4f} "
            f"(expected >= {_PROBE_HIGH}) and the refuted one {lo:.4f} "
            f"(expected <= {_PROBE_LOW}). If both sit near 0.5 the decoder output "
            "is being rescaled by d_model**-0.5 -- see this module's docstring. "
            "Scores would be unusable; check the transformers version and that "
            "tie_word_embeddings is False."
        )


class FactChecker(Protocol):
    def score(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        """P(claim is supported by doc) for each pair."""


@dataclass
class MiniCheckFactChecker:
    model_name: str = FACTCHECK_MODEL
    batch_size: int = 16
    max_input_tokens: int = 1024

    def __post_init__(self) -> None:
        self._tok = None
        self._model = None
        self._verified = False

    def _ensure(self):
        # `_verified` is separate from `_model` so a failed self-test raises on
        # every later call rather than only the first: the registry caches the
        # model, so a second `_ensure` would otherwise sail past a broken load.
        if self._model is not None and self._verified:
            return
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

        reg = get_registry()

        def _load():
            tok = AutoTokenizer.from_pretrained(self.model_name)
            cfg = AutoConfig.from_pretrained(self.model_name)
            # MiniCheck ships a separately fine-tuned lm_head but its config
            # still says tie_word_embeddings=True, and T5 reads that flag to
            # decide whether to rescale the decoder output by d_model**-0.5.
            # Pinning it False is the fix; `check_label_separation` below is
            # the net, because which of the two a given transformers version
            # does is exactly what changed under us. See the module docstring.
            cfg.tie_word_embeddings = False
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, config=cfg, **dtype_kwargs(preferred_dtype())
            )
            model.to(device_of()).eval()
            return tok, model

        self._tok, self._model = reg.get(f"factcheck:{self.model_name}", _load)
        self.self_test()
        self._verified = True

    def self_test(self) -> None:
        check_label_token_ids(self._tok)
        # `_score_loaded`, not `score`: the latter calls `_ensure` and would
        # recurse back into here.
        check_label_separation(self._score_loaded)

    def score(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must be the same length")
        if not docs:
            return []
        self._ensure()
        return self._score_loaded(docs, claims)

    def _score_loaded(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        """The forward pass itself. Assumes the model is already loaded."""
        import torch

        eos = self._tok.eos_token or "</s>"
        texts = [f"predict: {d}{eos}{c}" for d, c in zip(docs, claims)]
        out: list[float] = []
        dev = device_of()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self._tok(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_input_tokens,
            ).to(dev)
            dec = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long, device=dev)
            with torch.no_grad():
                logits = self._model(**enc, decoder_input_ids=dec).logits[:, 0, :]
            pair = logits[:, torch.tensor([_UNSUPPORTED_TOKEN_ID, _SUPPORTED_TOKEN_ID], device=dev)]
            probs = torch.nn.functional.softmax(pair.float(), dim=-1)[:, 1]
            out.extend(probs.detach().cpu().tolist())
        return out


@dataclass
class FakeFactChecker:
    """Lexical stand-in so the loop runs on CPU in seconds.

    Content-word overlap between claim and document. Crude, but it moves in the
    right direction on the fixtures, which is all the plumbing tests need.
    """

    threshold: float = 0.5

    _STOP = frozenset(
        "the a an of and or in on for to is are was were be been being with by "
        "that this these those it its as at from not no there their they which "
        "may might could can will would should than then also more less".split()
    )

    def _bag(self, text: str) -> set[str]:
        import re

        return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in self._STOP}

    def score(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        out = []
        for doc, claim in zip(docs, claims):
            c, d = self._bag(claim), self._bag(doc)
            out.append(0.0 if not c else round(len(c & d) / len(c), 4))
        return out


def load_factchecker(fake: bool = False) -> FactChecker:
    return FakeFactChecker() if fake else MiniCheckFactChecker()
