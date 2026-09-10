"""Cochrane review loading: many trial abstracts in, one conclusion out."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from ..config import MAX_ABSTRACTS_PER_REVIEW, paths
from .download import ensure_cochrane

__all__ = ["Review", "load_reviews", "load_review_index"]


@dataclass
class Review:
    """One systematic review: the trials that went in, the conclusion that came out."""

    review_id: str
    pmids: list[str]
    titles: list[str]
    abstracts: list[str]
    target: str | None = None  # the human reviewer's conclusion; never shown to the loop
    split: str = ""

    @property
    def n_studies(self) -> int:
        return len(self.abstracts)

    def documents(self, max_abstracts: int = MAX_ABSTRACTS_PER_REVIEW) -> list[str]:
        """Title + abstract per study, truncated to keep the writer's context sane.

        Reviews range from 1 to 134 included trials; the long tail would blow the
        context window for no measurable gain, so we take the first N.
        """
        docs = []
        for title, abstract in list(zip(self.titles, self.abstracts))[:max_abstracts]:
            title = (title or "").strip()
            abstract = (abstract or "").strip()
            docs.append(f"{title}\n{abstract}".strip() if title else abstract)
        return [d for d in docs if d]

    def source_text(self, max_abstracts: int = MAX_ABSTRACTS_PER_REVIEW) -> str:
        return "\n\n".join(
            f"[Study {i}] {d}" for i, d in enumerate(self.documents(max_abstracts), 1)
        )


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(ensure_cochrane() / f"{name}.parquet")


@lru_cache(maxsize=4)
def load_reviews(split: str = "dev") -> tuple[Review, ...]:
    """Load a split. Targets are attached for train/dev.

    The test split has no public targets -- MSLR2022 withheld them -- so test
    Reviews come back with target=None. Test-split targets live in the
    annotation file instead; see data/annotations.py.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError(f"unknown split {split!r}")

    inputs = _read(f"{split}_inputs")
    targets: dict[str, str] = {}
    if split != "test":
        tdf = _read(f"{split}_targets")
        targets = dict(zip(tdf.ReviewID.astype(str), tdf.Target.astype(str)))

    out: list[Review] = []
    for review_id, grp in inputs.groupby("ReviewID", sort=True):
        rid = str(review_id)
        out.append(
            Review(
                review_id=rid,
                pmids=[str(x) for x in grp.PMID.tolist()],
                titles=["" if pd.isna(t) else str(t) for t in grp.Title.tolist()],
                abstracts=["" if pd.isna(a) else str(a) for a in grp.Abstract.tolist()],
                target=targets.get(rid),
                split=split,
            )
        )
    return tuple(out)


def load_review_index(split: str = "dev") -> dict[str, Review]:
    return {r.review_id: r for r in load_reviews(split)}
