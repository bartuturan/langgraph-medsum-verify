"""MiniCheck: does this document support this claim?

Implemented against `transformers` directly rather than the `minicheck` pip
package, which pulls its own pinned dependency tree and tends to fight the
Kaggle image. The format below is the one MiniCheck's own inference code uses
for the Flan-T5 checkpoint:

    input           "predict: " + document + <eos> + claim
    decoder input   a single zero token
    score           softmax over logits at token ids [3, 209]; index 1 is
                    P(supported)

Those two token ids are load-bearing and undocumented in the model card, so
`self_test()` decodes them at load time and the loader refuses to run if they
are not the expected "0"/"1" pair. A silent mismatch here would poison every
number downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import FACTCHECK_MODEL
from .registry import device_of, get_registry, preferred_dtype

__all__ = ["FactChecker", "MiniCheckFactChecker", "FakeFactChecker", "load_factchecker"]

# MiniCheck reads its verdict off these two vocabulary positions.
_UNSUPPORTED_TOKEN_ID = 3
_SUPPORTED_TOKEN_ID = 209


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
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        reg = get_registry()

        def _load():
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, torch_dtype=preferred_dtype()
            )
            model.to(device_of()).eval()
            return tok, model

        self._tok, self._model = reg.get(f"factcheck:{self.model_name}", _load)
        self.self_test()

    def self_test(self) -> None:
        """Fail loudly if the label token ids are not what MiniCheck expects."""
        neg = self._tok.convert_ids_to_tokens(_UNSUPPORTED_TOKEN_ID)
        pos = self._tok.convert_ids_to_tokens(_SUPPORTED_TOKEN_ID)
        cleaned = {t.replace("▁", "") for t in (neg, pos)}
        if cleaned != {"0", "1"}:
            raise RuntimeError(
                f"MiniCheck label tokens moved: id {_UNSUPPORTED_TOKEN_ID}={neg!r}, "
                f"id {_SUPPORTED_TOKEN_ID}={pos!r} (expected '0'/'1'). "
                "Scores would be meaningless -- check the checkpoint."
            )

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
