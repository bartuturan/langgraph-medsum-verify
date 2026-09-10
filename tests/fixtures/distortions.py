"""Hand-built source/claim pairs with known ground truth.

Two jobs:

1. Unit-test the verifier. Each case states whether the claim overstates its
   source, so a regression in the lexicon shows up immediately.
2. Feed FakeWriter. The canned "drafts" contain planted distortions, which
   makes the whole three-role loop runnable on CPU in seconds without a GPU.

Cases are written to be unambiguous. Anything genuinely borderline belongs in
the validation against human annotations, not in a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    source: str          # what the trial evidence actually says
    claim: str           # what the summary says
    overclaim: bool      # is the claim stated more strongly than the source?
    kind: str            # strengthened | faithful | direction_flip | weakened


CASES: tuple[Case, ...] = (
    # ---- the characteristic failure: quiet strengthening --------------------
    Case(
        "hedge_to_assertion",
        "Pooled analysis suggests that the intervention may be associated with "
        "reduced mortality, although confidence intervals were wide.",
        "The intervention reduces mortality.",
        True, "strengthened",
    ),
    Case(
        "association_to_causation",
        "Vitamin D supplementation was associated with fewer respiratory "
        "infections in observational data.",
        "Vitamin D supplementation prevents respiratory infections.",
        True, "strengthened",
    ),
    Case(
        "insufficient_to_effective",
        "There is insufficient evidence to determine whether acupuncture "
        "relieves chronic pain.",
        "Acupuncture is effective for chronic pain.",
        True, "strengthened",
    ),
    Case(
        "trend_to_significant",
        "There was a trend towards improved survival that did not reach "
        "statistical significance.",
        "The treatment significantly improves survival.",
        True, "strengthened",
    ),
    Case(
        "low_certainty_to_strong",
        "Low certainty evidence suggests a possible reduction in hospital "
        "admissions.",
        "The programme clearly reduces hospital admissions.",
        True, "strengthened",
    ),
    Case(
        "single_trial_to_general",
        "One small trial reported that the drug might shorten symptom duration.",
        "The drug shortens symptom duration.",
        True, "strengthened",
    ),
    # ---- direction problems -------------------------------------------------
    Case(
        "null_to_benefit",
        "There was no significant difference in relapse rates between the "
        "intervention and control groups.",
        "The intervention reduces relapse rates.",
        True, "direction_flip",
    ),
    Case(
        "harm_to_benefit",
        "The higher dose increased the rate of serious adverse events.",
        "The higher dose improves safety outcomes.",
        True, "direction_flip",
    ),
    # ---- faithful controls: must NOT fire ----------------------------------
    Case(
        "faithful_hedged",
        "Pooled analysis suggests the intervention may be associated with "
        "reduced mortality.",
        "The intervention may be associated with reduced mortality.",
        False, "faithful",
    ),
    Case(
        "faithful_assertion",
        "The trial demonstrated that the vaccine significantly reduced "
        "infection rates compared with placebo.",
        "The vaccine significantly reduces infection rates.",
        False, "faithful",
    ),
    Case(
        "faithful_insufficient",
        "There is insufficient evidence to determine whether acupuncture "
        "relieves chronic pain.",
        "There is insufficient evidence to determine whether acupuncture "
        "relieves chronic pain.",
        False, "faithful",
    ),
    Case(
        "faithful_null",
        "There was no significant difference in relapse rates between groups.",
        "No significant difference in relapse rates was found between groups.",
        False, "faithful",
    ),
    Case(
        "faithful_association",
        "Supplementation was associated with fewer respiratory infections.",
        "Supplementation is associated with fewer respiratory infections.",
        False, "faithful",
    ),
    # ---- the opposite error: excessive hedging ------------------------------
    # The dominant failure in the annotated MSLR data is *under*claiming, and a
    # reviser that hedges everything must be caught too.
    Case(
        "weakened_from_strong",
        "High quality evidence shows the vaccine significantly reduces "
        "infection rates.",
        "The vaccine may possibly be associated with reduced infection rates.",
        False, "weakened",
    ),
    Case(
        "weakened_to_vacuous",
        "The trial demonstrated that early mobilisation shortens hospital stay.",
        "There is insufficient evidence to determine the effect of early "
        "mobilisation.",
        False, "weakened",
    ),
)

STRENGTHENED = tuple(c for c in CASES if c.kind == "strengthened")
FAITHFUL = tuple(c for c in CASES if c.kind == "faithful")
WEAKENED = tuple(c for c in CASES if c.kind == "weakened")
DIRECTION_FLIPS = tuple(c for c in CASES if c.kind == "direction_flip")


# The canned draft ships with FakeWriter in the package (src must not depend on
# tests); re-exported here so fixtures and tests refer to a single name.
from medsumverify.models.writer import DEFAULT_FAKE_DRAFT as FAKE_DRAFT

FAKE_SOURCE = (
    "[Study 1] A randomised trial of the intervention. Pooled analysis suggests "
    "the intervention may be associated with reduced mortality, although "
    "confidence intervals were wide.\n\n"
    "[Study 2] Cohort follow-up. The intervention was associated with fewer "
    "hospital admissions over 12 months.\n\n"
    "[Study 3] Safety report. Long-term harms were not assessed in any "
    "included trial."
)
