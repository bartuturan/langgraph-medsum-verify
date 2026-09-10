"""Central configuration: paths, model ids, seeds, thresholds.

Paths resolve differently on Kaggle vs. a local checkout, so everything goes
through ``paths()`` rather than being hardcoded at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Data sources (verified 2026-09-10)
# --------------------------------------------------------------------------

# The MSLR2022 HF repo ships only a loading script + dummy zips, and script
# based loading was removed in `datasets` 3.0. We pull the tarball directly.
MSLR_TARBALL_URL = "https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz"

# Human annotations of claim strength / effect direction, 100 Cochrane test
# reviews x 6 systems.
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/allenai/mslr-annotated-dataset/"
    "main/data/data_with_overlap_scores.json"
)

# Cochrane ships five CSVs. There is deliberately no test-targets.csv: MSLR2022
# was a shared task and the test targets were withheld. Test-split targets come
# from the annotation JSON instead.
COCHRANE_FILES = {
    "train_inputs": "train-inputs.csv",
    "train_targets": "train-targets.csv",
    "dev_inputs": "dev-inputs.csv",
    "dev_targets": "dev-targets.csv",
    "test_inputs": "test-inputs.csv",
}

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

WRITER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FACTCHECK_MODEL = "lytang/MiniCheck-Flan-T5-Large"
RETRIEVER_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
# Eval only. Never used inside the loop -- that is the point of it.
HOLDOUT_NLI_MODEL = "pritamdeka/PubMedBERT-MNLI-MedNLI"

# --------------------------------------------------------------------------
# Experiment knobs
# --------------------------------------------------------------------------

SEED = 20260910
MAX_ROUNDS = 2
N_EXPERIMENT_DOCS = 50  # floor 30 if GPU hours run short
RETRIEVE_TOP_K = 5
EVIDENCE_CHAR_CAP = 1500
MAX_ABSTRACTS_PER_REVIEW = 25  # long reviews blow up retrieval cost for no gain


@dataclass(frozen=True)
class Thresholds:
    """Verifier decision thresholds.

    Defaults are a priori. `scripts/calibrate.py` re-fits them on the 50
    calibration review_ids and writes the result to results/thresholds.json;
    reported numbers always come from the disjoint 50 validation review_ids.
    """

    # Flag when the claim sits this many *integer* strength levels above its
    # evidence. Integer levels, not the continuous score: the within-band nudges
    # that make the score rankable would otherwise push a genuine 3-vs-2 gap
    # (0.85) under a 1.0 cutoff, and the human construct is itself defined on
    # the integer scale (strength_generated > strength_target).
    delta_level: int = 1
    # Flag when MiniCheck P(supported) falls below this.
    tau_support: float = 0.5
    # Evidence sentences below this fraction of the top retrieval score are
    # treated as off-topic and excluded from the evidence-strength estimate.
    relevance_floor: float = 0.6
    # ...and never read strength off more than this many sentences.
    max_evidence_sentences: int = 3


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Paths:
    root: Path
    cache: Path
    results: Path

    def ensure(self) -> "Paths":
        for p in (self.cache, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def cochrane_parquet(self) -> Path:
        return self.cache / "cochrane"

    @property
    def annotations_json(self) -> Path:
        return self.cache / "data_with_overlap_scores.json"

    @property
    def tarball(self) -> Path:
        return self.cache / "mslr_data.tar.gz"


def on_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def paths() -> Paths:
    """Resolve I/O locations. Override the root with MEDSUM_ROOT."""
    env = os.environ.get("MEDSUM_ROOT")
    if env:
        root = Path(env)
    elif on_kaggle():
        root = Path("/kaggle/working")
    else:
        root = Path(__file__).resolve().parents[2]
    return Paths(root=root, cache=root / "data_cache", results=root / "results").ensure()
