"""Held-out entailment judge. Evaluation only -- never reachable from the loop.

The grounded condition optimizes against MiniCheck. Scoring it with MiniCheck
would measure how well it hit its own target. This model is a different
architecture trained on different data (MedNLI/MNLI), is loaded only after the
loop's models are freed, and is not importable from anything under graph/ or
verify/ -- a test enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import HOLDOUT_NLI_MODEL
from .registry import device_of, get_registry, preferred_dtype

__all__ = ["HoldoutNLI", "entailment_scores"]


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
        # rather than assuming index 0 or 2 means entailment.
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


def entailment_scores(summaries: Sequence[str], sources: Sequence[str]) -> list[float]:
    """Mean per-claim entailment of each summary against its source."""
    from ..verify.segment import segment

    judge = HoldoutNLI()
    out = []
    for summary, source in zip(summaries, sources):
        claims = segment(summary)
        if not claims:
            out.append(float("nan"))
            continue
        scores = judge.score([source[:4000]] * len(claims), claims)
        out.append(sum(scores) / len(scores))
    return out
