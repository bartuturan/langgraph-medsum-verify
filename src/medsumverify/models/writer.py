"""The 3B writer that drafts, self-critiques, and revises.

A small model is a deliberate choice: the distortions have to actually occur
for there to be anything to measure. Greedy decoding by default so the shared
Round-0 draft is reproducible -- the paired design depends on all three
conditions starting from a byte-identical draft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ..config import SEED, WRITER_MODEL
from .registry import device_of, get_registry, preferred_dtype

__all__ = ["Writer", "QwenWriter", "FakeWriter", "load_writer"]


class Writer(Protocol):
    def chat(
        self, system: str, user: str, max_new_tokens: int = 320, role: str = "draft"
    ) -> str:
        """`role` is one of draft | critique | revise.

        Passed explicitly rather than sniffed from the prompt text: FakeWriter
        needs to know which role it is playing, and inferring it from
        substrings silently mislabelled the self-critique call, which made the
        control condition do nothing at all while still looking like it ran.
        It also gives the real writer a per-role call count for the fairness
        comparison between conditions.
        """


@dataclass
class QwenWriter:
    model_name: str = WRITER_MODEL
    temperature: float = 0.0        # greedy: the shared draft must be reproducible
    seed: int = SEED
    max_input_tokens: int = 8192

    def __post_init__(self) -> None:
        self._tok = None
        self._model = None
        self.n_calls = 0
        self.calls_by_role: dict[str, int] = {}

    def _ensure(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        reg = get_registry()

        def _load():
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=preferred_dtype()
            )
            model.to(device_of()).eval()
            return tok, model

        self._tok, self._model = reg.get(f"writer:{self.model_name}", _load)

    def chat(
        self, system: str, user: str, max_new_tokens: int = 320, role: str = "draft"
    ) -> str:
        self._ensure()
        import torch

        torch.manual_seed(self.seed)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._tok(
            text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        ).to(device_of())
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self._tok.eos_token_id)
        if self.temperature and self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self._model.generate(**enc, **kwargs)
        self.n_calls += 1
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        return self._tok.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# --------------------------------------------------------------------------
# CPU stand-in
# --------------------------------------------------------------------------

# A draft that mixes planted distortions with faithful sentences, so the
# Reviser has to touch some claims and leave the others alone.
DEFAULT_FAKE_DRAFT = (
    "Twelve trials involving 3400 participants were included. "
    "The intervention reduces mortality. "
    "It is associated with fewer hospital admissions. "
    "There is insufficient evidence regarding long-term harms."
)

# Applied by FakeWriter when asked to revise: the scripted equivalent of
# "soften this claim to match its evidence".
_HEDGES = [
    (re.compile(r"\breduces\b", re.I), "may be associated with reduced"),
    (re.compile(r"\bprevents\b", re.I), "may help prevent"),
    (re.compile(r"\bshortens\b", re.I), "may shorten"),
    (re.compile(r"\bimproves\b", re.I), "may improve"),
    (re.compile(r"\bis effective\b", re.I), "may be effective"),
    (re.compile(r"\bsignificantly\b", re.I), "possibly"),
    (re.compile(r"\bclearly\b", re.I), "possibly"),
]


@dataclass
class FakeWriter:
    """Deterministic scripted writer -- no torch, no GPU, milliseconds per call.

    Lets the whole three-role graph be exercised on CPU: routing, the round
    cap, and the rule that unflagged claims pass through untouched. It hedges
    whatever it is told to revise, which is also the degenerate behaviour the
    anti-degeneracy guardrails exist to catch.
    """

    draft: str = ""
    n_calls: int = 0
    log: list[str] = field(default_factory=list)
    calls_by_role: dict[str, int] = field(default_factory=dict)

    def chat(
        self, system: str, user: str, max_new_tokens: int = 320, role: str = "draft"
    ) -> str:
        self.n_calls += 1
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        self.log.append(role)

        if role == "draft":
            return self.draft or DEFAULT_FAKE_DRAFT

        if role == "critique":
            # Must match the format the critic prompt asks for, or the control
            # condition parses to zero flags and quietly becomes a no-op.
            return "CLAIM 2: causal language where the source supports only an association"

        # Revise: hedge the one claim we were handed, leave everything else alone.
        claim = _extract_block(user, "CLAIM")
        for pattern, replacement in _HEDGES:
            claim = pattern.sub(replacement, claim)
        return claim.strip()


def _extract_block(text: str, header: str) -> str:
    """Pull one labelled block out of a prompt (up to the next ALL-CAPS label)."""
    m = re.search(
        rf"^{re.escape(header)}:\s*\n(.*?)(?=\n[A-Z][A-Z ]{{2,}}:|\Z)", text, re.S | re.M
    )
    return (m.group(1) if m else text).strip()


def load_writer(fake: bool = False, **kwargs) -> Writer:
    return FakeWriter(**kwargs) if fake else QwenWriter(**kwargs)
