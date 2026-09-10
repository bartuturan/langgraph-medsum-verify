"""Pull the evidence sentences that bear on a claim.

MiniCheck sees at most ~1k tokens, and a Cochrane review can carry 134
abstracts, so the whole source cannot be handed over wholesale. Each claim gets
its own top-k evidence passage, retrieved from the abstract sentences with a
biomedical sentence encoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import EVIDENCE_CHAR_CAP, RETRIEVE_TOP_K, RETRIEVER_MODEL
from .registry import device_of, get_registry

__all__ = ["Retriever", "DenseRetriever", "FakeRetriever", "load_retriever", "split_sentences"]

_ABBREV = re.compile(r"\b(?:e\.g|i\.e|vs|no|fig|approx|et al|cf|Dr|Mr|Ms)\.$", re.I)


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    """Cheap sentence split for source abstracts.

    Deliberately not pysbd: this runs over every sentence of up to 25 abstracts
    per claim-set and the extra precision buys nothing for retrieval.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts, buf = [], ""
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        buf = f"{buf} {chunk}".strip() if buf else chunk
        if _ABBREV.search(buf):
            continue
        if len(buf) >= min_chars:
            parts.append(buf)
            buf = ""
    if buf and len(buf) >= min_chars:
        parts.append(buf)
    return parts


class Retriever(Protocol):
    def top_k(self, claim: str, candidates: Sequence[str], k: int) -> list[tuple[str, float]]:
        ...


@dataclass
class DenseRetriever:
    model_name: str = RETRIEVER_MODEL
    _model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            reg = get_registry()
            self._model = reg.get(
                f"retriever:{self.model_name}",
                lambda: SentenceTransformer(self.model_name, device=device_of()),
            )
        return self._model

    def top_k(self, claim: str, candidates: Sequence[str], k: int = RETRIEVE_TOP_K):
        if not candidates:
            return []
        import numpy as np

        model = self._ensure()
        emb = model.encode(
            [claim, *candidates], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        sims = emb[1:] @ emb[0]
        order = np.argsort(-sims)[:k]
        return [(candidates[i], float(sims[i])) for i in order]


@dataclass
class FakeRetriever:
    """TF-IDF cosine. Runs on CPU, no downloads, and is a sane fallback."""

    def top_k(self, claim: str, candidates: Sequence[str], k: int = RETRIEVE_TOP_K):
        if not candidates:
            return []
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        try:
            tfidf = TfidfVectorizer(stop_words="english").fit_transform([claim, *candidates])
        except ValueError:  # empty vocabulary
            return [(c, 0.0) for c in candidates[:k]]
        sims = (tfidf[1:] @ tfidf[0].T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        return [(candidates[i], float(sims[i])) for i in order]


def build_evidence(
    claim: str,
    source_sentences: Sequence[str],
    retriever: Retriever,
    k: int = RETRIEVE_TOP_K,
    char_cap: int = EVIDENCE_CHAR_CAP,
) -> tuple[str, list[tuple[str, float]]]:
    """Return (evidence passage, ranked sentences) for one claim."""
    ranked = retriever.top_k(claim, list(source_sentences), k)
    passage, used = "", []
    for sent, score in ranked:
        if len(passage) + len(sent) + 1 > char_cap:
            break
        passage = f"{passage} {sent}".strip()
        used.append((sent, score))
    if not used and ranked:  # a single sentence longer than the cap
        sent, score = ranked[0]
        passage, used = sent[:char_cap], [(sent, score)]
    return passage, used


def load_retriever(fake: bool = False) -> Retriever:
    return FakeRetriever() if fake else DenseRetriever()
