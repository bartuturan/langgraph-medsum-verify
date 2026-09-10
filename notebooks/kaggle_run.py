"""Kaggle driver. Paste each CELL block into its own notebook cell.

Kept as a .py file so it is diffable in git; notebooks/build_notebook.py turns
it into kaggle_run.ipynb. Order matters -- the held-out judge is loaded only
after the loop's models are freed, so the two never share the GPU.

Accelerator: GPU T4 x2 or P100. Internet: ON (models + data are downloaded).
Expect roughly 2 hours end to end for 50 reviews.
"""

# ---------------------------------------------------------------- CELL 1 ----
# Install. Pin nothing that Kaggle already ships; pysbd and langgraph are the
# only genuinely missing pieces.
"""
!pip install -q pysbd langgraph
!pip install -q git+https://github.com/USERNAME/medsum-verify.git
"""

# ---------------------------------------------------------------- CELL 2 ----
# Environment check. Stop here if the GPU is missing -- everything below
# assumes one.
"""
import torch, medsumverify
from medsumverify.models.registry import gpu_report
print(gpu_report("start"))
assert torch.cuda.is_available(), "enable the GPU accelerator"
print("torch", torch.__version__)
"""

# ---------------------------------------------------------------- CELL 3 ----
# Data. Downloads the 264 MB MSLR tarball once and converts to parquet, then
# the human annotation file. Both are cached under /kaggle/working.
"""
from medsumverify.data.download import ensure_all
from medsumverify.data.cochrane import load_reviews
from medsumverify.data.annotations import load_annotated

ensure_all()
dev = load_reviews("dev")
ann = load_annotated()
print(f"dev reviews: {len(dev)}   annotated pairs: {len(ann)}")
"""

# ---------------------------------------------------------------- CELL 4 ----
# RESULT 1a: the strength scorer against human labels. Pure CPU, no models.
# Runs in seconds and is worth having on the record before any GPU time.
"""
from medsumverify.eval.validate_strength import validate_strength
validate_strength()
"""

# ---------------------------------------------------------------- CELL 5 ----
# Load the three loop models onto one GPU and confirm the memory budget.
"""
from medsumverify.models.factcheck import load_factchecker
from medsumverify.models.retriever import load_retriever
from medsumverify.models.writer import load_writer
from medsumverify.models.registry import gpu_report

factchecker = load_factchecker()
retriever   = load_retriever()
writer      = load_writer()

# Touch each one so the weights actually land on the GPU before we measure.
print(factchecker.score(["Aspirin may reduce mortality."], ["Aspirin reduces mortality."]))
print(retriever.top_k("mortality", ["Aspirin reduced deaths.", "Unrelated sentence."], 1))
print(writer.chat("You are terse.", "Say OK.", max_new_tokens=5))
print(gpu_report("all three loaded"))
"""

# ---------------------------------------------------------------- CELL 6 ----
# RESULT 1: the full verifier against human labels, with the real retriever and
# fact-checker. This is the gate -- if AUROC < 0.65, fix the verifier before
# spending GPU hours on the experiment.
"""
from medsumverify.eval.validate_verifier import validate_verifier
v = validate_verifier(fake=False)
assert v.gate_passed, "verifier does not track human judgements -- stop and fix it"
"""

# ---------------------------------------------------------------- CELL 7 ----
# Smoke run: 3 reviews, all three conditions. Read the traces before committing
# to the full run.
"""
from medsumverify.experiment.run import ExperimentRunner, select_reviews
from medsumverify.verify.verifier import Verifier

verifier = Verifier(factchecker, retriever)
runner = ExperimentRunner(writer, verifier)
smoke = runner.run(select_reviews("dev", n=3))
for r in smoke:
    print(f"\n--- {r['review_id']} / {r['condition']} rounds={r['rounds']}")
    print("   ", r['summary'][:300])
"""

# ---------------------------------------------------------------- CELL 8 ----
# Full run. Appends to results/experiment.jsonl after every single
# (review, condition), and skips anything already on disk, so re-running this
# cell after a dropped session resumes rather than restarts.
"""
from medsumverify.experiment.run import ExperimentRunner, select_reviews

reviews = select_reviews("dev", n=50)
runner.run(reviews)
"""

# ---------------------------------------------------------------- CELL 9 ----
# Free the loop's models before loading the held-out judge, so the two never
# coexist and the independence of the final score is structural, not just
# claimed.
"""
from medsumverify.models.registry import get_registry, gpu_report
get_registry().free()
print(gpu_report("freed"))
"""

# --------------------------------------------------------------- CELL 10 ----
# The comparison, plus the blinded audit sheet to fill in by hand.
"""
from medsumverify.eval.compare import compare
from medsumverify.eval.audit_sheet import build_audit_sheet

compare()
build_audit_sheet(n_claims=60)
"""

# --------------------------------------------------------------- CELL 11 ----
# Persist. Kaggle keeps /kaggle/working across a "Save Version", but mirroring
# to a dataset is what survives a session that dies mid-run.
"""
!ls -la /kaggle/working/results/
!tar czf /kaggle/working/results.tar.gz -C /kaggle/working results
print("download results.tar.gz from the output pane")
"""
