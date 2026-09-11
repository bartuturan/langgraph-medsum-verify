# medsum-verify

A three-role summarization loop for medical evidence, where the critic runs
**real checks** instead of asking a language model whether the summary looks
right — and an experiment designed so the result can actually be interpreted.

## The distortion being hunted

The source says *"may be associated with reduced mortality."*
The summary says *"reduces mortality."*

That quiet strengthening is the characteristic failure in medical text, and it
is invisible to similarity metrics: to an embedding model the two sentences are
nearly identical. Detecting it needs something that measures **how strongly a
claim is stated** against **how strongly the evidence supports it**.

## The loop

```
Drafter  →  Verifier  →  Reviser  →  Verifier  →  …  (2 rounds max)
```

Built with LangGraph, as `StateGraph` nodes over pure functions. The Verifier
does five things per claim:

| step | what it does |
|---|---|
| segment | pysbd splits the summary; each sentence is a claim |
| retrieve | top-k abstract sentences relevant to that claim (S-PubMedBERT) |
| support | MiniCheck-Flan-T5-Large → P(supported) |
| strength | how strongly the claim is *stated*, on a 0–3 scale |
| evidence | how strongly the retrieved evidence *supports* it, **same scorer** |

Using the identical scorer on both sides is what makes the delta mean anything.
A claim is flagged when it outruns its evidence, when the fact-checker finds no
support, or when it asserts the opposite direction. The Reviser rewrites **only
flagged claims**; everything else passes through byte-identical.

## The experiment

Three conditions, all starting from a **byte-identical Round-0 draft**, so
every comparison is paired:

| | critique source | rounds |
|---|---|---|
| **plain** | none | 0 |
| **selfcritique** | the model's own opinion, no tools | 2 |
| **grounded** | the verifier's tool output | 2 |

The middle condition is the whole point. Without it, a win by `grounded` cannot
be attributed — you can't tell grounding from simply going around the loop
twice. **`selfcritique` and `grounded` are the same graph with one node
swapped**, and they share the same Reviser prompt, so the only difference is
where the flags come from.

## Two commitments that keep it honest

**Validate the instrument before measuring with it.** The verifier is checked
against human annotations of claim strength and effect direction before it is
used to judge anything.

**Don't grade the loop on its own signals.** `grounded` optimizes MiniCheck
support and its own strength delta against the abstracts. Scoring it on those
would be circular. The primary metric is instead the distance, in either
direction, between the summary's claim strength and that of the **Cochrane
reviewer's own conclusion** — which no condition ever sees. Two tests enforce
this structurally: loop code may not import the held-out
NLI judge, and may not import the data modules that carry reference conclusions.

**Anti-degeneracy guardrails.** A reviser that prepends "may" to every sentence
would erase every overclaim while destroying the summary. Direction accuracy,
content overlap, hedge density and length are reported unconditionally, and a
strength win with a guardrail drop is labelled a hedging artifact, not a win.

## Results so far

### Result 1a — the strength scorer tracks human judgement (CPU, no models)

Out-of-fold over 518 annotated summaries / 274 Cochrane test reviews, grouped by
review and stratified by outcome. Aggregation chosen on training folds only.

| | value |
|---|---|
| Spearman ρ vs human claim strength (generated summaries) | **+0.707** |
| …vs human claim strength (reference conclusions) | +0.330 |
| human–human ceiling (36 doubly-annotated pairs) | +0.914 |
| overclaim detection AUROC | **0.788** |
| overclaim detection AUPRC | 0.415 (baseline 0.093, **4.5× lift**) |

An unsupervised lexicon reaches ~77% of the achievable rank correlation. These
are the numbers for the current scorer, which softens a clause-framing hedge
("the evidence suggests that …") by one step instead of flattening the clause
to "weak". That change was prompted by Qwen's drafts, not by this data, and it
nudged every number here slightly up (ρ +0.706 → +0.707, AUROC 0.783 → 0.788):
it does no harm on the annotated corpus, which rarely uses the frame. Whether it
helps on Qwen's own text is for the blind audit to say.

### Result 1 — the full verifier, on a Kaggle T4

Real retrieval and MiniCheck over the same 518 summaries, with the first scorer
(a re-run with the current scorer is pending):

| | CPU stand-ins | **real models** | strength vs reference (1a) |
|---|---|---|---|
| overclaim AUROC | 0.651 | **0.708** | 0.788 |
| overclaim AUPRC (baseline 0.093) | 0.137 | **0.160** | 0.415 |

It passes the pre-registered gate (AUROC ≥ 0.65), but that number flatters it.
AUROC measures how well the strength gap *ranks* summaries; the loop acts on
*flags*, and at the fitted threshold the overclaim flag fires on 1.5% of
summaries — 2 of the 48 human-judged overclaims caught (recall 4%, precision
25%). 93% of all flags are MiniCheck's "unsupported" (685 of 733), and
effect-direction flags sit at chance (AUROC 0.539 against the 55 true flips).

**The gap to 1a localises the weak point**: 1a compares the summary with the
*reference conclusion*, which is how the human label is defined, while the
verifier compares it with the *source abstracts*. Individual trial abstracts
state their own results confidently, so evidence strength reads high and masks
overclaims — a single confident trial does not license a confident review-level
conclusion. See "Known limitations".

### Base rates worth knowing before reading any of this

Measured on the annotation file, not assumed:

- Overclaiming occurs in **9.1%** of annotated summaries. **Under**claiming
  occurs in **56.6%**. The 2022-era fine-tuned seq2seq systems hedge into
  vagueness rather than overstate — so detection is an imbalanced problem and
  AUPRC matters more than AUROC.
- Per-system overclaim rates range from 2.1% to 18.1%, and the two worst
  offenders are the least seq2seq-like systems. That is the main reason to
  expect an instruction-tuned 3B model to overclaim more than this corpus does.
- 45% of effect-direction mismatches are **omissions** (the summary states no
  direction at all), not flips. Only ~9% are genuine flips. Validating a flip
  detector against raw mismatch would measure the wrong thing, so the two are
  scored separately.

### Qwen's drafts — the go/no-go

On a 12-review probe Qwen overclaims in **5 of 12 drafts (42%, Wilson 95% CI
19–68%)** — far above the corpus's 9%, so the distortion does occur — and
underclaims in the other 7, for a mean shift of −0.29. It misjudges strength in
both directions. That is why the primary comparison metric is `abs_delta`,
miscalibration either way: under the signed delta, a reviser that pushed an
already-weak claim weaker would score as a win. The switch was made after this
probe and **before any comparison between conditions was run**.

On all 50 experiment drafts, the first scorer's verifier raised 78 flags across
94 claims: 57 unsupported, 14 direction, 7 overclaim. Most drafts open with "The
evidence suggests that …", which that scorer read as "weak" whatever followed —
the blind spot the current scorer fixes.

## Data

| | |
|---|---|
| Source | `mslr_data.tar.gz` from the MSLR2022 shared task (AI2) |
| Reviews | Cochrane: 3752 train / 470 dev / 470 test, one row per included study |
| Annotations | `allenai/mslr-annotated-dataset`, 636 rows over 597 (review, system) pairs, 274 reviews, 6 systems |

Two traps, both handled:

- **`load_dataset("allenai/mslr2022")` does not work.** That repo ships only a
  loading script and dummy zips, and script-based loading was removed in
  `datasets` 3.0. The tarball is downloaded and parsed directly.
- **`test-targets.csv` does not exist** — test targets were withheld for the
  shared task. They survive in the annotation file, which carries `target` for
  all 470 test reviews. This is why validation runs on **test** and the
  experiment on **dev**: the two sets are disjoint by construction.

## Models — 8.1 GB, all on one 16 GB GPU

| role | model | fp16 |
|---|---|---|
| drafter / reviser / self-critic | `Qwen/Qwen2.5-3B-Instruct` | 6.2 GB |
| fact-checker | `lytang/MiniCheck-Flan-T5-Large` | 1.6 GB |
| retriever | `pritamdeka/S-PubMedBert-MS-MARCO` | 0.25 GB |
| held-out judge (eval only) | `pritamdeka/PubMedBERT-MNLI-MedNLI` | 0.25 GB |

A small writer is deliberate: distortions have to actually occur for there to be
anything to measure. The held-out judge loads only after the loop's models are
freed. Kaggle T4s are sm75 — fp16, never bf16.

MiniCheck is called through `transformers` directly rather than the `minicheck`
package, using the format from MiniCheck's own inference code
(`"predict: " + doc + <eos> + claim`, one zero decoder token, softmax over token
ids `[3, 209]`). Those ids are the *first tokens* of the two labels: `"1"` is
the single token `▁1` (209), while `"0"` splits into `▁` (3) + `0`. They are
undocumented in the model card and load-bearing, so at startup the loader checks
that `"0"` and `"1"` still begin with 3 and 209, and refuses to run if not.

## Running it

Local, CPU, no GPU and no model downloads — the whole loop runs against fakes:

```bash
pip install -e ".[dev,eval]"
pytest -q                                            # 158 tests
python -m medsumverify.eval.validate_strength        # Result 1a
python -m medsumverify.eval.validate_verifier --fake # Result 1, CPU lower bound
```

On Kaggle (**GPU T4 x2**, internet on), open `notebooks/kaggle_run.ipynb`. Not
P100: Kaggle's current PyTorch build needs CUDA compute capability 7.0+, and the
P100 is 6.0. Regenerate the notebook from its diffable source with
`python notebooks/build_notebook.py`.

Results are appended to `results/experiment.jsonl` after every single
(review, condition). Re-running the experiment cell skips what already
finished and retries what failed, so an interruption *within* a session costs
minutes. Across sessions, switch on **Persistence: Files only** in the Kaggle
session options before you start: `/kaggle/working` then carries over and the
next session resumes by itself. Without it, a new session starts empty; attach
the previous saved version's output (Add Input → Your Work) and uncomment the
two restore lines in Cell 3.

After the run, fill in `results/audit_sheet.csv` (see
`results/audit_instructions.txt`) and score it:

```bash
python -m medsumverify.eval.audit_sheet score
```

## Known limitations

- **Evidence strength is read from individual trial abstracts.** A Cochrane
  reviewer's caution comes from synthesising *across* trials — inconsistency,
  small samples, risk of bias — which a per-claim max over retrieved sentences
  cannot see. This is the diagnosed cause of the gap between Result 1a and
  Result 1, and the obvious next mechanism is a cross-trial consistency penalty
  that lowers warranted strength when retrieved evidence disagrees.
- **The human construct and ours differ slightly.** `strength_target` is
  annotated on the reference conclusion; the verifier compares against the
  abstracts. Related, since the conclusion is a human's calibrated reading of
  those same abstracts — but not identical.
- **48 overclaim positives** is thin. Hence out-of-fold CV rather than a single
  split, and AUPRC with a prevalence baseline rather than AUROC alone.
- The strength scorer is English- and Cochrane-shaped, and lexicon-based. It
  will not transfer unchanged to other domains.

## Layout

```
src/medsumverify/
  data/       download (tarball, not `datasets`), cochrane loader, annotations
  models/     writer, fact-checker, retriever, held-out judge, GPU registry
  verify/     segment, strength (0-3), direction, verifier
  graph/      state, prompts, nodes, three LangGraph builds
  experiment/ runner with per-document JSONL checkpointing and resume
  eval/       validate_strength, validate_verifier, metrics, compare, holdout, audit_sheet
tests/        158 tests, all CPU
notebooks/    kaggle_run.py (diffable) -> kaggle_run.ipynb
```
