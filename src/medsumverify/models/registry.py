"""Load-once model registry.

Everything the pipeline needs sits on one 16GB GPU at the same time:

    Qwen2.5-3B-Instruct  fp16   ~6.2 GB   drafter / reviser / self-critic
    MiniCheck-Flan-T5-L  fp16   ~1.6 GB   fact-checker
    S-PubMedBert         fp16   ~0.25GB   retriever
                                 -------
                                 ~8.1 GB  + KV cache and activations

The held-out NLI judge is deliberately *not* part of this set. It is loaded by
the evaluation code after `free()`, because scoring the loop with a model the
loop had access to would be circular.

Kaggle T4s are sm75: fp16, never bf16.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Registry", "get_registry", "device_of", "gpu_report", "preferred_dtype", "dtype_kwargs"]

_REGISTRY: "Registry | None" = None


def _torch():
    import torch

    return torch


def device_of() -> str:
    try:
        torch = _torch()
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def preferred_dtype():
    torch = _torch()
    if not torch.cuda.is_available():
        return torch.float32
    # T4 (sm75) has no bf16 support; fp16 everywhere keeps Kaggle runs uniform.
    return torch.float16


# `from_pretrained` renamed `torch_dtype` to `dtype` in transformers 4.56.0.
# Measured on a tiny T5: 4.55.0 rejects `dtype` with a TypeError, while 4.56.0
# and 5.17 accept both but warn that `torch_dtype` is deprecated -- so a later
# release may drop it, and every model load here would then fail.
_DTYPE_KWARG_SINCE = "4.56.0"


def dtype_kwargs(dtype, version: str | None = None) -> dict:
    """The keyword argument `from_pretrained` expects for the weight dtype.

    Every model loader goes through this rather than naming the argument
    itself. `version` overrides the installed transformers version, for tests.
    """
    from packaging.version import Version

    if version is None:
        import transformers

        version = transformers.__version__
    key = "dtype" if Version(version) >= Version(_DTYPE_KWARG_SINCE) else "torch_dtype"
    return {key: dtype}


def gpu_report(tag: str = "") -> str:
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return f"[gpu] {tag} cpu-only"
        alloc = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        name = torch.cuda.get_device_name(0)
        return f"[gpu] {tag} {name}: {alloc:.2f} GB now / {peak:.2f} GB peak / {total:.1f} GB total"
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"[gpu] {tag} unavailable ({exc})"


@dataclass
class Registry:
    """Holds the loaded models so nothing gets loaded twice."""

    _cache: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def free(self, *keys: str) -> None:
        """Drop models and release VRAM. Call before loading the eval-only judge."""
        targets = keys or tuple(self._cache)
        for k in targets:
            self._cache.pop(k, None)
        try:
            import gc

            gc.collect()
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover
            pass

    @property
    def loaded(self) -> tuple[str, ...]:
        return tuple(self._cache)


def get_registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY
