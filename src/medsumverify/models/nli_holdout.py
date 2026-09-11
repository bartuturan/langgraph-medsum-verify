"""Held-out entailment judge. Evaluation only -- never reachable from the loop.

The grounded condition optimizes against MiniCheck. Scoring it with MiniCheck
would measure how well it hit its own target. This model is a different
architecture trained on different data (MedNLI/MNLI), is loaded only after the
loop's models are freed, and is not importable from anything under graph/ or
verify/ -- a test enforces that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import HOLDOUT_NLI_MODEL
from .registry import device_of, get_registry, preferred_dtype

__all__ = ["Judge", "HoldoutNLI", "entailment_scores"]


class Judge(Protocol):
    def score(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        """P(premise entails hypothesis) per pair."""


@dataclass
class HoldoutNLI:
    model_name: str = HOLDOUT_NLI_MODEL
    batch_size: int = 32
    max_length: int = 512

    def __post_init__(self) -> None:
        self._tok = None
        self._model = None
        self._entail_idx: int | None = None

    def _ensure(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def _load():
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, torch_dtype=preferred_dtype()
            )
            model.to(device_of()).eval()
            return tok, model

        self._tok, self._model = get_registry().get(f"nli:{self.model_name}", _load)

        # Label order varies between NLI checkpoints; read it off the config
        # rather than assuming index 0 or 2 means entailment. (For the default
        # checkpoint it is {0: contradiction, 1: entailment, 2: neutral}.)
        labels = {
            i: str(v).lower() for i, v in (self._model.config.id2label or {}).items()
        }
        for i, name in labels.items():
            if "entail" in name:
                self._entail_idx = int(i)
                break
        if self._entail_idx is None:
            raise RuntimeError(
                f"no entailment label found in {labels!r} for {self.model_name}"
            )

    def score(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        """P(premise entails hypothesis) per pair."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must be the same length")
        if not premises:
            return []
        self._ensure()
        import torch

        out: list[float] = []
        for i in range(0, len(premises), self.batch_size):
            p = list(premises[i: i + self.batch_size])
            h = list(hypotheses[i: i + self.batch_size])
            enc = self._tok(
                p, h, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(device_of())
            with torch.no_grad():
                logits = self._model(**enc).logits.float()
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, self._entail_idx]
            out.extend(probs.detach().cpu().tolist())
        return out


def entailment_scores(
    summaries: Sequence[str],
    source_documents: Sequence[Sequence[str]],
    judge: Judge | None = None,
) -> list[float]:
    """Per summary: mean over claims of the best entailment across its studies.

    Each claim is scored against every included study *separately* and the
    best-supporting study counts. Feeding the concatenated source instead would
    be truncated to BERT's 512 tokens -- roughly the first abstract -- so any
    claim about a later study would look unsupported.

    Deliberately no retrieval step: the loop's retriever would then be a shared
    component between the thing being measured and the measurement, and scoring
    every study makes selection unnecessary. Summaries with no claims, or
    reviews with no usable source, score NaN rather than 0.
    """
    from ..verify.segment import segment

    judge = judge or HoldoutNLI()
    out: list[float] = []
    for summary, docs in zip(summaries, source_documents):
        claims = segment(summary)
        docs = [d for d in docs if d and d.strip()]
        if not claims or not docs:
            out.append(float("nan"))
            continue
        premises = [d for _ in claims for d in docs]
        hypotheses = [c for c in claims for _ in docs]
        scores = judge.score(premises, hypotheses)
        n = len(docs)
        per_claim = [max(scores[i * n: (i + 1) * n]) for i in range(len(claims))]
        out.append(sum(per_claim) / len(per_claim))
    return out


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)
