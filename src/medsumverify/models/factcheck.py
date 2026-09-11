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

    def _ensure(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        reg = get_registry()

        def _load():
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **dtype_kwargs(preferred_dtype())
            )
            model.to(device_of()).eval()
            return tok, model

        self._tok, self._model = reg.get(f"factcheck:{self.model_name}", _load)
        self.self_test()

    def self_test(self) -> None:
        check_label_token_ids(self._tok)

    def score(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must be the same length")
        if not docs:
            return []
        self._ensure()
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
