"""Which dtype keyword `from_pretrained` gets, on each side of the 4.56 rename."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from medsumverify.models.registry import dtype_kwargs

SRC = Path(__file__).resolve().parents[1] / "src" / "medsumverify"
TINY_T5 = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"


@pytest.mark.parametrize(
    "version,key",
    [
        ("4.40.0", "torch_dtype"),   # oldest the project allows
        ("4.55.0", "torch_dtype"),   # measured: rejects dtype= with a TypeError
        ("4.55.4", "torch_dtype"),
        ("4.56.0", "dtype"),         # measured: first release that accepts dtype=
        ("4.57.1", "dtype"),
        ("5.17.0", "dtype"),         # measured: warns on torch_dtype=
    ],
)
def test_keyword_follows_the_rename(version, key):
    assert dtype_kwargs("fp16", version=version) == {key: "fp16"}


def test_no_loader_names_the_argument_itself():
    """Passing torch_dtype= directly is the deprecated path this helper replaces."""
    offenders = [
        str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
        if "torch_dtype=" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"loaders bypassing dtype_kwargs: {offenders}"


def test_real_load_is_fp16_and_silent():
    """End to end on a tiny model: the dtype lands and nothing is deprecated.

    Runs only where torch, transformers and the tiny checkpoint are available
    locally; never hits the network.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForSeq2SeqLM

    # transformers logs through its own non-propagating logger, so attach to it
    # directly rather than relying on pytest's caplog.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger("transformers")
    logger.addHandler(handler)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            TINY_T5, local_files_only=True, **dtype_kwargs(torch.float16)
        )
    except OSError as exc:
        pytest.skip(f"tiny test model not cached locally: {exc}")
    finally:
        logger.removeHandler(handler)

    assert model.get_input_embeddings().weight.dtype == torch.float16
    assert "deprecated" not in buf.getvalue().lower(), buf.getvalue()
