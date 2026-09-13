"""Kaggle driver. Paste each CELL block into its own notebook cell.

Kept as a .py file so it is diffable in git; notebooks/build_notebook.py turns
it into kaggle_run.ipynb. Order matters -- the held-out judge is loaded only
after the loop's models are freed, so the two never share the GPU.

Accelerator: GPU T4 x2. Not P100: Kaggle's current PyTorch build needs CUDA
compute capability 7.0+, the P100 is 6.0, and it fails with "no kernel image
is available". Internet: ON (models + data are downloaded).
Expect roughly 3 hours end to end for 50 reviews.
"""

# -------------------------------------------------------------- CELL 1 ----
# Install. Pin nothing that Kaggle already ships; pysbd and langgraph are the
# only genuinely missing pieces.
#
# --force-reinstall: pip otherwise sees the same version number already
# installed and silently keeps the old code, so a fix pushed to GitHub would
# never arrive. --no-deps: everything else the package needs ships with Kaggle
# or comes from the line above. If the package was already imported in this
# session, restart the kernel after this cell (Run -> Restart) and continue
# from Cell 2, or the old code stays loaded in memory.
"""
!pip install -q pysbd langgraph
!pip install -q --force-reinstall --no-deps git+https://github.com/bartuturan/langgraph-medsum-verify.git
"""

# -------------------------------------------------------------- CELL 2 ----
# Environment check. Stop here if the GPU is missing -- everything below
# assumes one.
"""
import torch, medsumverify
from medsumverify.models.registry import gpu_report
print(gpu_report("start"))
assert torch.cuda.is_available(), "enable the GPU accelerator"
print("torch", torch.__version__)
"""

# -------------------------------------------------------------- CELL 3 ----
# Data. Downloads the 264 MB MSLR tarball once and converts to parquet, then
# the human annotation file. Both are cached under /kaggle/working.
#
# Resuming in a NEW session after a crash? Easiest: have "Persistence: Files
# only" switched on in the session options -- it has to be on *before* the
# session that dies. /kaggle/working then carries over and nothing needs
# restoring. Otherwise a new session starts empty: attach the previous saved
# version's output (Add Input -> Your Work -> this notebook) and uncomment the
# two lines below. The cp restores the results, the cached drafts (so every
# condition still starts from the same draft) and the gate result,
# validation_verifier.json. The grep must print "gate_passed": true before you
# skip Cell 6 on a resume. Steps that failed last time are retried, not skipped.
"""
from medsumverify.data.download import ensure_all
from medsumverify.data.cochrane import load_reviews
from medsumverify.data.annotations import load_annotated

# !mkdir -p /kaggle/working/results && cp /kaggle/input/*/results/*.jsonl /kaggle/input/*/results/*.json /kaggle/working/results/
# !grep gate_passed /kaggle/working/results/validation_verifier.json

ensure_all()
dev = load_reviews("dev")
ann = load_annotated()
print(f"dev reviews: {len(dev)}   annotated pairs: {len(ann)}")
"""

# -------------------------------------------------------------- CELL 4 ----
# RESULT 1a: the strength scorer against human labels. Pure CPU, no models.
# Runs in seconds and is worth having on the record before any GPU time.
"""
from medsumverify.eval.validate_strength import validate_strength
validate_strength()
"""

# -------------------------------------------------------------- CELL 5 ----
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

# -------------------------------------------------------------- CELL 6 ----
# RESULT 1: the full verifier against human labels, with the real retriever and
# fact-checker. This is the gate -- if AUROC < 0.65, fix the verifier before
# spending GPU hours on the experiment.
"""
from medsumverify.eval.validate_verifier import validate_verifier
v = validate_verifier(fake=False)
assert v.gate_passed, "verifier does not track human judgements -- stop and fix it"
"""

# -------------------------------------------------------------- CELL 7 ----
# Smoke run: 3 reviews, all three conditions. Read the traces before committing
# to the full run.
"""
from medsumverify.experiment.run import ExperimentRunner, select_reviews
from medsumverify.verify.verifier import Verifier

verifier = Verifier(factchecker, retriever)
# gate=False is the pre-registered loop: the Reviser's output goes into the
# summary whatever it says. The gated variant is Cell 10.
runner = ExperimentRunner(writer, verifier, gate=False)
smoke = runner.run(select_reviews("dev", n=3))
for r in smoke:
    print(f"\n--- {r['review_id']} / {r['condition']} rounds={r['rounds']}")
    print("   ", r['summary'][:300])
"""

# -------------------------------------------------------------- CELL 8 ----
# GO / NO-GO. In the annotated corpus only 9% of summaries overclaim and 57%
# *under*claim -- those 2022-era seq2seq systems hedge into vagueness. If Qwen
# does the same, there is no distortion to measure and the experiment answers
# nothing. Check the drafts before spending the GPU hours.
#
# What matters is whether overclaiming happens often enough to measure, not
# the sign of the mean: a few large underclaims can drag the mean below zero
# while overclaiming is still common. (The first version of this cell judged
# by the mean and misreported exactly that: 42% of drafts overclaimed.) Qwen
# misjudges strength in both directions, which is why the comparison's primary
# metric is abs_delta, miscalibration either way.
"""
from medsumverify.eval.metrics import score_summary
from medsumverify.experiment.run import ExperimentRunner, select_reviews
import numpy as np

probe = select_reviews("dev", n=12)
cache = runner._draft_cache()
deltas = []
for rv in probe:
    draft = runner.get_draft(rv, cache)
    m = score_summary(draft, rv.target, rv.review_id, "plain")
    deltas.append(m.delta_strength)
    print(f"{rv.review_id}  delta={m.delta_strength:+.2f}  {draft[:90]}")

d = np.array(deltas)
over = float(np.mean(d > 0))
print(f"\nmean delta {d.mean():+.3f} | mean |delta| {np.abs(d).mean():.3f} | "
      f"overclaim {over:.0%} | underclaim {np.mean(d < 0):.0%}   (annotated corpus: 9% overclaim)")
print("verdict:", f"overclaims in {over:.0%} of drafts -- common enough to measure, proceed" if over >= 0.2
      else f"overclaims in only {over:.0%} of drafts -- too rare to measure reliably")
"""

# -------------------------------------------------------------- CELL 9 ----
# Full run. Appends to results/experiment.jsonl after every single
# (review, condition), and skips anything already on disk, so re-running this
# cell after an interruption in the same session resumes rather than restarts.
# The drafts from the previous cells are cached and reused. For a crash that
# ends the session, see the restore line in Cell 3.
#
# If the verifier or its scorer changed after grounded already ran, call
# runner.discard("grounded") first. Plain and self-critique never use the
# verifier, so they and the cached drafts stay, and only grounded is redone.
"""
from medsumverify.experiment.run import ExperimentRunner, select_reviews

reviews = select_reviews("dev", n=50)
runner.run(reviews)   # ungated -- `runner` was built with gate=False in Cell 7
"""

# ------------------------------------------------------------- CELL 10 ----
# The gated re-run. Everything is held constant except one thing: a rewrite
# that comes back *stronger* than the claim it was meant to fix is retried once
# with a note saying so, and dropped for the original if the retry strengthens
# too. The first attempt's prompt is byte-identical to the ungated run, so any
# difference is the gate and nothing else.
#
# Drafts are copied over first, so Round-0 text is identical and the comparison
# stays paired. All three conditions run: plain costs nothing (the draft is
# cached) and self-critique has to be redone because it shares the Reviser --
# gating only the grounded arm would confound the gate with the grounding.
#
# Roughly 20 minutes at the timings from the ungated run.
"""
import shutil
from pathlib import Path
from medsumverify.experiment.run import ExperimentRunner, select_reviews

gated_dir = Path("results_gated")
gated_dir.mkdir(exist_ok=True)
shutil.copy("results/drafts.jsonl", gated_dir / "drafts.jsonl")

gated_runner = ExperimentRunner(writer, verifier, results_dir=gated_dir, gate=True)
gated_runner.run(select_reviews("dev", n=50))

from medsumverify.eval.gate_report import replay, report
replay("results")        # what the gate would have done to the ungated run
report(gated_dir)        # what it actually did
"""

# ------------------------------------------------------------- CELL 11 ----
# Free the loop's models before loading the held-out judge, so the two never
# coexist and the independence of the final score is structural, not just
# claimed. The notebook's own variables have to go first: the registry drops
# its handles, but `writer`, `factchecker`, `retriever`, `verifier` and
# `runner` still point at the models, and nothing is released while they do.
"""
import gc
for name in ("runner", "gated_runner", "verifier", "writer", "factchecker", "retriever"):
    globals().pop(name, None)
from medsumverify.models.registry import get_registry, gpu_report
get_registry().free()
gc.collect()
print(gpu_report("freed"))
"""

# ------------------------------------------------------------- CELL 12 ----
# Secondary metric: a MedNLI-trained judge from a different model family that
# the loop can never reach (a test enforces that). Each claim is scored against
# every included study separately and the best one counts, so no abstract is
# cut off by BERT's 512-token window. Writes results/holdout_nli.json, which
# the comparison below picks up on its own. A minute or two.
"""
from pathlib import Path
from medsumverify.eval.holdout import score_holdout
from medsumverify.models.registry import gpu_report

for d in ("results", "results_gated"):
    if Path(d, "experiment.jsonl").exists():
        score_holdout(results_dir=Path(d))
print(gpu_report("held-out judge"))
"""

# ------------------------------------------------------------- CELL 13 ----
# The comparison, plus the blinded audit sheet to fill in by hand. Scoring the
# filled-in sheet happens afterwards, on your own machine:
#     python -m medsumverify.eval.audit_sheet score
"""
from pathlib import Path
from medsumverify.eval.compare import compare
from medsumverify.eval.audit_sheet import build_audit_sheet

compare()
build_audit_sheet(n_claims=60)

# The gated run is a separate three-condition comparison on the same drafts.
if Path("results_gated", "experiment.jsonl").exists():
    compare(results_dir=Path("results_gated"))
"""

# ------------------------------------------------------------- CELL 14 ----
# Persist. Download results.tar.gz from the output pane; "Save Version" also
# keeps /kaggle/working as this version's output, which is what Cell 3's
# restore line reads from.
"""
!ls -la /kaggle/working/results/ /kaggle/working/results_gated/
!tar czf /kaggle/working/results.tar.gz -C /kaggle/working results results_gated
print("download results.tar.gz from the output pane")
"""
