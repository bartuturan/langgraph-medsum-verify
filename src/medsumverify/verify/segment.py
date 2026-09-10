"""Split a summary into individual claims.

Sentence-level, deterministic, via pysbd. Cochrane conclusions run 2-5
sentences, so a sentence is close to the natural claim unit and a deterministic
split keeps the verifier reproducible -- an LLM decomposer would put a second
uncontrolled generator inside the measuring instrument.
"""

from __future__ import annotations

import re
from functools import lru_cache

__all__ = ["segment", "clean_summary"]

_BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+", re.MULTILINE)
_HEADER = re.compile(r"^\s*(?:#+\s*|\*\*)?(?:summary|conclusions?|abstract|"
                     r"authors'? conclusions?|background|results?)\s*:?\s*(?:\*\*)?\s*$",
                     re.IGNORECASE | re.MULTILINE)
_WS = re.compile(r"[ \t]+")


@lru_cache(maxsize=1)
def _segmenter():
    import pysbd

    return pysbd.Segmenter(language="en", clean=False)


def clean_summary(text: str) -> str:
    """Strip the scaffolding a chat model wraps around its answer."""
    text = (text or "").strip()
    text = _HEADER.sub("", text)
    text = _BULLET.sub("", text)
    text = text.replace("**", "")
    text = _WS.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def segment(text: str, min_chars: int = 15) -> list[str]:
    """Return claim-sized sentences. Fragments below min_chars are dropped."""
    cleaned = clean_summary(text)
    if not cleaned:
        return []
    out: list[str] = []
    for line in cleaned.splitlines():
        for sent in _segmenter().segment(line):
            sent = sent.strip()
            if len(sent) >= min_chars:
                out.append(sent)
    return out
