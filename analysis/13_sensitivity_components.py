"""
13_sensitivity_components.py
============================

v3 pre-decision pelvic arrival tool — SENSITIVITY ANALYSES + COMPONENT
DECOMPOSITION (CONTRACTS §7 row 13, §Cross-cutting rigor; SAP §3/§6/§8).

This script is a SELF-CONTAINED fork of ``11_pelvic_thresholds.py`` (house
style: copy-adapt, NO cross-script imports). It INHERITS script 11's
memory-bounded, OOM-safe, convergence-guarded change-point/spline engine
VERBATIM — every critical fix is reproduced so 13 cannot reintroduce the
raw-shock_index SI-tail non-convergence or the MI OOM:

  * ``SI_WINSOR_CAP = 2.0`` winsorization of ``shock_index`` at load time.
  * ``_assert_converged`` guard keyed on finite ``llf`` / ``cov_params``.
  * HC1 robust SE reserved for FINAL point-estimate models only.
  * deviance-only fast inner fitter (sklearn newton-cholesky) in the
    grid / bootstrap loop (deviance is covariance-type invariant).
  * joblib process-parallel row-level bootstrap.
  * SEQUENTIAL one-at-a-time MI (m=20) — never holds all imputations in RAM.
  * interior 5th–95th percentile restriction + >=100 events/side guard.
  * secondary bootstraps capped (250-style computational deviation, SAP §11).

Deliverables OWNED by this script (and ONLY these — never touches script 11's
``table7_threshold_or.csv`` / ``table7_absrisk.csv`` / ``spline_*.csv`` /
``changepoint_*.json`` / ``component_splines.csv`` / ``tipping_point_bound.json``
nor any 04–10 / 12 / 14 / 15 / 17 file):

  derived/component_decomp_data.csv   F8 fig-data: adjusted component-decomposition
                                      splines (exposure, component, grid, pred_risk,
                                      lci, uci) for the composite's 3 components.
  derived/component_venn.json         (1) Venn set counts among composite-positive
                                      patients; (2) component change-points (SI+SBP)
                                      with the death-only-vs-composite CO-LOCATION
                                      proof; (3) script-11 composite reference.
  tables/table7_sensitivity.csv       The full v3 named sensitivity menu (below).
  logs/sensitivity_components_diagnostics.txt   (13-namespaced, non-colliding)
  logs/sensitivity_components_manifest.json     (13-namespaced, non-colliding)

v3 NAMED SENSITIVITY MENU (CONTRACTS §7, `13` row). Every arm exists so a reviewer
can audit ONE definition change at a time. Each arm names the EXACT outcome column
it swaps in; `variant` values are the CONTRACTS strings, character-for-character.

  variant                    death arm            transfusion arm            cohort
  -------------------------  -------------------  -------------------------  --------------
  ring_only_reference        (script 11 headline reference row-group)
  sens_component_hc          --                   --   (received_hc alone)   ring-only
  sens_component_death       death_hemorrhagic    --                          ring-only
  sens_component_tx          --                   substantial_transfusion     ring-only
  sens_v2def_outcome         death_inhosp         substantial_tx_ml_only      ring-only
  sens_tx_ml_only            death_hemorrhagic    substantial_tx_ml_only      ring-only
  sens_death_allcause        death_inhosp         substantial_transfusion     ring-only
  sens_death_headais         death_hem_headais    substantial_transfusion     ring-only
  sens_death_le48h           death_le48h          substantial_transfusion     ring-only
  sens_death_missing_incl    death_hem_missing_i  substantial_transfusion     ring-only
  sens_death_dayfallback     death_hem_dayfallbk  substantial_transfusion     ring-only
  sens_death_edincl          death_hem_edincl     substantial_transfusion     ring-only
  sens_death_le24h           <=24 h + headAIS<4   substantial_transfusion     ring-only
  sens_death_ed              death_ed             substantial_transfusion     ring-only
  sens_death_dayrule         death_hem_dayrule    substantial_transfusion     ring-only
  sens_death_le48h_dayrule   death_le48h_dayrule  substantial_transfusion     ring-only
  sens_any_tx                death_hemorrhagic    blood_4h > 0                ring-only
  sens_tx_2000               death_hemorrhagic    >=2000 mL OR >=8 U          ring-only
  sens_acetabulum            (primary composite)                              ring∪acetab
  sens_mi                    (primary composite, blood_4h imputed m=20)       ring-only
  sens_exclude_nslo          (primary composite)               ring-only minus sbp_unrecordable==1
  sens_with_ring             (primary composite, unstable_ring restored to the score)

TWO LENSES PER ARM. Wherever both are meaningful an arm is reported twice:
  * ``analysis="changepoint"`` / ``"prespecified_threshold"`` — the Analysis-B
    (association) lens, i.e. does the SI/SBP dose–response inflection move when
    the outcome definition moves. This is the lens the v2 table already used.
  * ``analysis="score_auroc"`` / ``"score_points"`` — the Analysis-A (prediction)
    lens: the ring-free v3 score REFIT on TRAIN against the arm's outcome and
    evaluated on the held-out TEST slice. Selection is NOT re-run (no LASSO here);
    the variable set is held at refit.json["variables_kept"] so the arm isolates
    the OUTCOME change from selection noise. Deviations from that rule are named
    per-arm in the `notes` column (only `sens_with_ring` adds a variable).

THE FOUR-RUNG RECOVERY LADDER (CONTRACTS §3/§7, SAP amendment D-8) — ALL FOUR
RUNGS ARE REFIT HERE, not just the two edges:
  429 deaths carry no `finaldischarge_hrs`, so the primary rule (which requires a
  timing value) cannot see them. The pre-specified answer is a ladder of
  progressively more inclusive recovery rules, each an arm of its own:

    rung 1  death_hemorrhagic               4,191  primary — timed, <=48 h
    rung 2  death_hemorrhagic_dayfallback   4,209  + untimed death, calendar-day index <=2
    rung 3  death_hemorrhagic_edincl        4,253  + untimed ED death (necessarily <48 h)
    rung 4  death_hemorrhagic_missing_incl  4,478  + ALL untimed deaths (upper bound)

  Bracket width 287. Reporting only rungs 1 and 4 brackets the composite but leaves
  the two INTERMEDIATE, clinically-motivated rules unevaluated — and those are the
  rules a reviewer will actually ask about, because they recover deaths on a stated
  clinical justification rather than by assumption. Rungs 2 and 3 were named in
  CONTRACTS §7 from the start and were buildable from the cohort the whole time;
  they were absent from this script until the §7-coverage gate was rebuilt to be
  falsifiable (see CONTRACTS_S7_VARIANTS below).

READ THIS BEFORE INTERPRETING ``sens_v2def_outcome``:
  The documented no-refit pre-check "AUROC 0.876 -> 0.871" for dropping
  `unstable_ring` was computed under the **v2 outcome definition**. 0.871 is
  therefore the comparator for THIS ARM ONLY — never for the v3 headline, whose
  outcome is a different event set. A `sens_v2def_outcome` AUROC landing
  materially below ~0.86 signals a FORK DEFECT, not a scientific finding
  (CONTRACTS §13 G3e). The script prints this verdict explicitly.

v3 CHANGES vs the v2 fork (CONTRACTS §0, §4, §9-J outcome-name sweep):
  * Fix 2/2b — the composite's death arm and the non-discretionary component
    anchor are `death_hemorrhagic` (= death_inhosp & ais_head<4 &
    finaldischarge_days<=2). `death_inhosp` survives ONLY as the explicitly named
    `sens_death_allcause` / `sens_v2def_outcome` arms.
  * Fix 1 — the transfusion arm is the unit-aware `substantial_transfusion`. The
    v2 mL-only rule is the named `sens_tx_ml_only` arm, and the variant-d MI arm
    derives the composite with the unit-aware rule too (it previously hard-coded
    `blood_4h >= 1000`).
  * Fix 3 — `unstable_ring` is removed from the SCORE but **DELIBERATELY RETAINED**
    in `CORE_ADJUSTMENT` and `VIF_AUDIT_VARS` here (CONTRACTS §6 asymmetry, §13
    G3f). Do NOT "propagate" Fix 3 into this file. `sens_with_ring` quantifies
    what the concession cost.
  * Fix 4 — `sbp_unrecordable` carries NO integer points in any score arm
    (CONTRACTS §5.1); `sens_exclude_nslo` drops those rows entirely.
  * v3-K — `ais_head` is stripped from the adjustment set on MORTALITY-ONLY runs
    (it is part of the outcome definition there; adjusting for it is circular).
    It is retained on composite runs, where the outcome is not a deterministic
    function of it. `finaldischarge_days` is never an adjustment covariate.
  * No hard-coded v2 result constants. The reference row-group's composite rate
    is computed from the cohort, not the superseded 12.95 % (CONTRACTS §13 G5b).
  * NO HARD-CODED COHORT SCHEMA. `01`'s death component moved from a calendar-day
    rule (`finaldischarge_days <= 2`) to true elapsed hours
    (`finaldischarge_hrs <= 48`) — the day field is a calendar INDEX whose day-3
    bucket spans 24.25-71.17 h. This script reads the column list from the cohort
    at run time, prefers the hour field wherever a timing cut is needed, and
    resolves the menu against what actually exists. When `01` also emits the
    retained day-rule columns, the hour-vs-day CROSSWALK becomes its own pair of
    arms (`sens_death_dayrule`, `sens_death_le48h_dayrule`) — the crosswalk is a
    reportable robustness check in its own right.
  * DESCRIPTIVE-ONLY change-points are represented EXPLICITLY. Script 11 demotes
    an under-powered stratum and writes `breakpoint: NaN` with the key present,
    so there is no exception to catch. Every change-point row carries a
    `descriptive_only` flag and a `guard_reason`, and a co-location verdict
    against a demoted reference is reported as UNDETERMINED — never as "does not
    co-locate". With the v3 death arm this thin, expect this to fire.

Component-decomposition science (composite validity, CONTRACTS §13 G5e gate):
  The composite = received_hc | death_hemorrhagic | substantial_transfusion. We
  refit the adjusted spline + change-point for each of the 3 components SEPARATELY
  on both exposures. The punchline: the NON-DISCRETIONARY death-only anchor
  inflects where the composite inflects (co-location) — the composite is not an
  artifact of the discretionary HC-receipt component. (death_hemorrhagic
  reproduces script 11's si_mortality/sbp_mortality curves — an internal-
  consistency check.) NOTE the v3 event set is ~3,874 vs v2's 13,426, so
  stratified mortality-only change-points may fall below the >=100-events/side
  guard and are then reported descriptive-only, per the locked SAP §6.
"""

from __future__ import annotations

# Pin BLAS/OpenMP to 1 thread/process BEFORE numpy/sklearn import so the joblib
# process-parallel bootstrap does not oversubscribe cores (inherited from 11).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Silence the sklearn 1.8 penalty= deprecation FutureWarning IN THE JOBLIB WORKERS
# too (loky child processes re-import sklearn and do not inherit the in-process
# warnings filter) — otherwise the ~4,400 bootstrap fits flood the run log. The
# engine keeps penalty=None to match script 11 EXACTLY; this only hides the log
# noise. _assert_converged still RAISES (not warns) on any non-finite fit, so real
# problems are unaffected.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import matplotlib

matplotlib.use("Agg")  # defensive: before any pyplot import (CONTRACTS §1)

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
import patsy
import scipy
import scipy.stats as st
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression  # fast inner-loop fitter (no HC1)
from sklearn.metrics import roc_auc_score  # score-arm discrimination (Analysis-A lens)
from joblib import Parallel, delayed  # parallel bootstrap

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
if hasattr(st, "ConstantInputWarning"):
    warnings.filterwarnings("ignore", category=st.ConstantInputWarning)

# ---------------------------------------------------------------------------
# §1 Canonical paths (PAT_TBI house pattern — absolute root, derived sub-dirs)
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
CODE = config.ANALYSIS_DIR
DERIVED = config.DERIVED
TABLES = config.TABLES
FIGURES = config.FIGURES
LOGS = config.LOGS

COHORT_CSV = os.path.join(DERIVED, "pelvic_cohort_predecision.csv")

# Score-arm inputs (READ-ONLY artifacts of 02 / 03 / 05). Needed for the
# Analysis-A lens (`sens_v2def_outcome`, `sens_exclude_nslo`, `sens_with_ring`,
# and the score AUROC reported alongside every outcome-swap arm). Re-engineering
# the X matrix from the cohort CSV here would risk silently diverging from 02, so
# the engineered matrix is consumed as-is.
CANDIDATES_PARQUET = os.path.join(DERIVED, "candidates.parquet")
TRAIN_KEYS_CSV = os.path.join(DERIVED, "train_inc_keys.csv")
TEST_KEYS_CSV = os.path.join(DERIVED, "test_inc_keys.csv")
REFIT_JSON = os.path.join(DERIVED, "refit.json")

# OWNED outputs
DECOMP_CSV = os.path.join(DERIVED, "component_decomp_data.csv")
VENN_JSON = os.path.join(DERIVED, "component_venn.json")
TABLE7_SENS = os.path.join(TABLES, "table7_sensitivity.csv")
DIAG_TXT = os.path.join(LOGS, "sensitivity_components_diagnostics.txt")
MANIFEST_JSON = os.path.join(LOGS, "sensitivity_components_manifest.json")

# script 11 reference artifacts (READ-ONLY, for the co-location comparison; the
# whole script degrades gracefully to hard-coded CONTRACTS constants if absent).
REF_CHANGEPOINT_JSONS = {
    "si_composite": os.path.join(DERIVED, "changepoint_si_composite.json"),
    "si_mortality": os.path.join(DERIVED, "changepoint_si_mortality.json"),
    "sbp_composite": os.path.join(DERIVED, "changepoint_sbp_composite.json"),
    "sbp_mortality": os.path.join(DERIVED, "changepoint_sbp_mortality.json"),
}
REF_THRESHOLD_OR_CSV = os.path.join(TABLES, "table7_threshold_or.csv")
# LAST-RESORT fallback only. These are the **v2** change-point locations, computed
# under the **v2 outcome definition** (all-cause death + mL-only transfusion). Under
# the v3 outcome they are SUPERSEDED and must never reach a table, figure, or
# sentence. Script 11 runs in Batch 3, before this script in Batch 4, so the real
# JSONs should always be present; if one is not, the run is flagged stale and the
# co-location verdict is reported as UNRELIABLE rather than silently computed
# against a v2 number (CONTRACTS §13 G5b).
REF_FALLBACK_V2_STALE = {"si_composite": 0.60, "si_mortality": 1.18,
                         "sbp_composite": 122.0, "sbp_mortality": 120.0}

# ---------------------------------------------------------------------------
# Reproducibility constants (CONTRACTS §1)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
ALPHA = 0.05
Z = st.norm.ppf(1.0 - ALPHA / 2.0)

# shock_index winsorization cap (inherited from 11; the SI-tail convergence fix).
# The raw SI has a pathological upper tail (max≈53) from near-zero SBP
# denominators; fed raw into cr(shock_index,df=4)+HC1 it makes the spline
# NON-CONVERGENT and poisons every HC1 SE to NaN. Cap at 2.0 (applied ONCE at
# load so the winsorized SI flows everywhere; clip(upper=) preserves NaN so the
# missing-exposure count is unchanged). sbp2 is physiologically bounded — NOT touched.
SI_WINSOR_CAP = 2.0

# ---------------------------------------------------------------------------
# Adjustment set (CONTRACTS §6). HC1 robust SE, no clustering.
# ---------------------------------------------------------------------------
CORE_ADJUSTMENT: list[str] = [
    "age", "sex_male", "mechanism_cat", "iss", "ais_head",
    "unstable_ring", "cc_anticoag", "charlson", "acs_level", "puf_year",
]
CATEGORICAL_ADJ: frozenset[str] = frozenset({"mechanism_cat", "acs_level", "puf_year"})

# Per-run mediator/exposure strip (CONTRACTS §6 rule): never adjust for the
# exposure or a mediator.
EXPOSURE_STRIP: dict[str, list[str]] = {
    "shock_index": ["sbp2", "pulse2", "shock_index", "sbp_lt90", "sbp_unrecordable"],
    "sbp2": ["sbp2", "shock_index", "pulse2", "sbp_lt90", "sbp_unrecordable"],
}

# CONTRACTS §6 / §13 G3f — `unstable_ring` STAYS here. Fix 3 removes it from the
# SCORE, not from the association model's collinearity audit. Asserted at run time.
VIF_AUDIT_VARS: list[str] = ["iss", "ais_head", "unstable_ring"]
VIF_THRESHOLD = 5.0

# CONTRACTS §6 v3 note / §9 v3-K — outcome columns that are DETERMINISTIC functions
# of an adjustment covariate. On those runs the covariate is stripped from the
# adjustment set: adjusting for a variable that appears in the outcome definition
# is circular. Composite runs keep `ais_head` (the composite is not a deterministic
# function of it). `finaldischarge_days` is never an adjustment covariate anywhere.
OUTCOME_DEFINING_STRIP: dict[str, list[str]] = {
    "death_hemorrhagic": ["ais_head"],
    "death_hemorrhagic_headais": ["ais_head"],
    "death_hemorrhagic_missing_incl": ["ais_head"],
    "death_le48h": [],          # timing-only restriction; ais_head not involved
    "death_inhosp": [],
    "death_ed": [],
}

# Change-point grids + interior restriction + event guard (CONTRACTS §9-H2)
GRID_SI = np.round(np.arange(0.60, 1.60 + 1e-9, 0.02), 4)
GRID_SBP = np.arange(70.0, 130.0 + 1e-9, 2.0)
INTERIOR_LO_PCT, INTERIOR_HI_PCT = 5.0, 95.0
MIN_EVENTS_PER_SIDE = 100

KNOT_PCTS = (5.0, 35.0, 65.0, 95.0)
SI_UNIT = 0.1
SBP_UNIT = 10.0

# Pre-specified clinical thresholds (SAP §6; Bonferroni over m=4).
PRESPEC_THRESHOLDS: dict[str, list[float]] = {"shock_index": [1.0, 1.3], "sbp2": [90.0, 70.0]}
BONFERRONI_M = 4

# ---------------------------------------------------------------------------
# Outcome columns (CONTRACTS §4). v3 Fix 2/2b: the death arm is death_hemorrhagic.
# ---------------------------------------------------------------------------
COMPOSITE_COL = "major_hemorrhage_event"          # received_hc | death_hemorrhagic | substantial_transfusion
DEATH_COL = "death_hemorrhagic"                   # PRIMARY death component + mortality-only anchor
TX_COL = "substantial_transfusion"                # unit-aware (Fix 1)
HC_COL = "received_hc"

# The composite's 3 components (fig-data F8) — variant names per CONTRACTS §7.
COMPONENT_OUTCOMES = [HC_COL, DEATH_COL, TX_COL]
COMPONENT_LABEL = {
    HC_COL: "HC-only (DISCRETIONARY)",
    DEATH_COL: "death-only (NON-DISCRETIONARY; head AIS<4 & <=48h)",
    TX_COL: "transfusion-only (unit-aware)",
}
COMPONENT_VARIANT = {
    HC_COL: "sens_component_hc",
    DEATH_COL: "sens_component_death",
    TX_COL: "sens_component_tx",
}

# ---------------------------------------------------------------------------
# CONTRACTS §7 — THE MENU, HAND-TRANSCRIBED. THE COVERAGE GATE'S ONLY EXPECTATION.
# ---------------------------------------------------------------------------
# Transcribed row-for-row from CONTRACTS.md §7, table "`13` — v3 named sensitivity
# menu", character-for-character. It is the CONTRACT, not a summary of this file.
#
# *** IT MUST NEVER BE DERIVED FROM ANYTHING IN THIS SCRIPT. ***
# Not from OUTCOME_SWAP_SPEC, not from COMPONENT_VARIANT, not from the emitted
# table. The whole point is that the two sides of the comparison have independent
# provenance: this side is the document, the other side is the code.
#
# WHY THIS EXISTS (CONTRACTS §13 preamble — "a gate that cannot fail is not
# verification"). The previous coverage check built its expectation as
# `[v for v, _ in swap_variants]` — i.e. from the very menu it was auditing. It
# could detect an arm that CRASHED at run time and nothing else; an arm the script
# had simply never defined was, by construction, never expected and therefore never
# missing. It duly printed
#     "[OK] every CONTRACTS §7 variant is present in table7_sensitivity.csv"
# and wrote `variants_missing = []` into the manifest while `sens_death_dayfallback`
# and `sens_death_edincl` — rungs 2 and 3 of the pre-specified four-rung recovery
# ladder — had zero occurrences anywhere in this file. Third instance of this exact
# failure mode in the project (G2b-a's join tautology; G4c's swallowed comparison).
#
# EDITING RULE, in this order and no other:
#   1. CONTRACTS §7 changes  ->  2. edit THIS list  ->  3. the gate fails loudly
#   ->  4. implement the arm  ->  5. the gate passes.
# Editing the code first and this list second re-creates the tautology by hand.
CONTRACTS_S7_VARIANTS: tuple[str, ...] = (
    "sens_v2def_outcome",         # primary refit on major_hemorrhage_event_v2def
    "sens_tx_ml_only",            # composite using substantial_transfusion_ml_only
    "sens_death_allcause",        # composite using death_inhosp (rung 0)
    "sens_death_headais",         # composite using death_hemorrhagic_headais (rung 1a)
    "sens_death_le48h",           # composite using death_le48h (rung 1b)
    "sens_death_missing_incl",    # composite using death_hemorrhagic_missing_incl (rung 4)
    "sens_death_dayfallback",     # composite using death_hemorrhagic_dayfallback (rung 2)
    "sens_death_edincl",          # composite using death_hemorrhagic_edincl (rung 3)
    "sens_death_dayrule",         # composite using death_hemorrhagic_dayrule (crosswalk)
    "sens_death_le24h",           # death_inhosp & ais_head<4 & finaldischarge_hrs<=24
    "sens_death_ed",              # composite using death_ed
    "sens_exclude_nslo",          # refit + rescore excluding sbp_unrecordable==1
    "sens_with_ring",             # score refit WITH unstable_ring restored
    "sens_component_hc",          # v2 component-only arms (carried)
    "sens_component_death",
    "sens_component_tx",
    "sens_any_tx",                # v2 transfusion sensitivities (carried)
    "sens_tx_2000",
    "sens_acetabulum",            # acetabulum-inclusive cohort (carried)
    "sens_mi",                    # MI m=20 (carried)
)

# Of those, the arms that exist ONLY in, or MUST also carry, the Analysis-A
# (score-refit) lens. Also hand-transcribed against §7 + the two-lens rule in this
# module's docstring — NOT derived from OUTCOME_SWAP_SPEC. A contract arm that is
# emitted with a change-point row but silently loses its score row is a coverage
# hole of the same shape as a missing arm, and the flat name check cannot see it.
CONTRACTS_S7_SCORE_LENS_VARIANTS: tuple[str, ...] = (
    "sens_v2def_outcome",
    "sens_tx_ml_only",
    "sens_death_allcause",
    "sens_death_headais",
    "sens_death_le48h",
    "sens_death_missing_incl",
    "sens_death_dayfallback",
    "sens_death_edincl",
    "sens_death_dayrule",
    "sens_death_le24h",
    "sens_death_ed",
    "sens_any_tx",
    "sens_tx_2000",
    "sens_exclude_nslo",
    "sens_with_ring",
)

# Variants this script emits that CONTRACTS §7 does NOT name. Declared here so
# "extra" means "undeclared" rather than "unexplained", and so a typo'd variant
# string shows up as an undeclared extra instead of blending into the table.
EXTRA_DECLARED_VARIANTS: dict[str, str] = {
    "ring_only_reference":
        "script-11 composite reference row-group; the comparator every arm is read "
        "against (§8 T7 ownership is shared 11+13)",
    "primary_score_reference":
        "internal Analysis-A contrast row so each score arm is a like-for-like delta; "
        "the AUROC of record is 07's table4_discrimination.csv",
    "sens_death_le48h_dayrule":
        "hour-vs-day crosswalk twin of sens_death_le48h under the SUPERSEDED "
        "calendar-day rule; supplies the §13 G2b-e day-vs-hour evidence",
}

# Fix-1 transfusion decoder constants. Mirrors CONTRACTS §3 exactly; `01` owns the
# canonical derivation and this script only RE-DERIVES the sensitivity variants
# (>=2000 mL / >=8 U, and the MI arm's imputed composite) from raw blood_4h.
BLOOD_UNITS_MAX = 100          # 0 < v <= 100  -> reads as UNITS
BLOOD_ML_MIN_SCALE = 200       # v >= 200      -> reads as mL
BLOOD_UNITS_MIN_SUBSTANTIAL = 4      # >=4 U / 4 h
BLOOD_ML_MIN_SUBSTANTIAL = 1000      # >=1000 mL / 4 h (~4 U; the v2 rule)
BLOOD_UNITS_MIN_MASSIVE = 8          # >=8 U  / 4 h  (the unit twin of >=2000 mL)
BLOOD_ML_MIN_MASSIVE = 2000          # >=2000 mL / 4 h

# Exposure vitals imputed by MI.
EXPOSURE_VITALS = ["sbp2", "pulse2", "shock_index"]

# sens_mi: impute blood_4h (+ vitals) congenially; derive the composite.
# v3: the observed death arm carried into the imputation model is death_hemorrhagic.
MI_D_NUMERIC = ["sbp2", "pulse2", "shock_index", "blood_4h", "age", "iss",
                "ais_head", "sex_male", "unstable_ring", "cc_anticoag",
                "charlson", HC_COL, DEATH_COL]
MI_D_BINARY = {"sex_male", "unstable_ring", "cc_anticoag", HC_COL, DEATH_COL}

EXPOSURES = ["shock_index", "sbp2"]

# ---------------------------------------------------------------------------
# Score-arm (Analysis-A lens) constants — mirror 05_refit_integer.py exactly.
# ---------------------------------------------------------------------------
# BETA_DROP is the FALLBACK only. The live value is READ from
# refit.json["beta_cutoff"] in load_score_inputs() so 05 and 13 cannot drift: a
# re-derivation that agrees today can diverge tomorrow, a read cannot.
BETA_DROP = 0.15
SCORE_VIF_MAX = 5.0
SCORE_MIN_KEPT = 5
# v3 = 2 anchors, not 3 (Fix 3 removed `unstable_ring`); age>=65 == both bands.
FORCE_IN_ANCHORS = ["age_65_74", "age_ge75", "cc_anticoag"]

# ---- Fix 9 / D-11: the integer-scale anchor rule (CONTRACTS §5.3) -----------
# PARITY TOKEN. 05_refit_integer.py writes this exact string into refit.json and
# 12_temporal_external_validation.py carries the same literal. It must stay
# BYTE-IDENTICAL in all three scripts and is asserted against refit.json in
# load_score_inputs() (§13 G8e). It says "BETA_DROP" rather than "0.15" so moving
# the cutoff cannot make the recorded rule lie.
ANCHOR_SELECTION_RULE = (
    "anchor = min{ beta_v : v in point_vars, beta_v > 0, |beta_v| >= BETA_DROP }; "
    "variables retained solely by force-in and sitting below the BETA_DROP screen "
    "are excluded from the anchor search (CONTRACTS §5.3 / SAP D-11)"
)

# §13 G8c — card-scale tripwire. Reported loudly, never raised: every arm here is
# a named SENSITIVITY, and aborting the whole sensitivity menu over one arm's
# scale artefact would destroy more information than it protects. 05 raises.
CARD_SCALE_TRIPWIRE = 20
# Fix 4 — carries no integer points in any score arm (CONTRACTS §5.1).
DEFAULT_OVERRIDE_VARS = ("sbp_unrecordable",)
# Fix 3 — must never appear in a score variable list except in `sens_with_ring`.
RING_VAR = "unstable_ring"
# CONTRACTS §13 G3e — the apples-to-apples comparator for `sens_v2def_outcome`.
V2DEF_AUROC_EXPECTED = 0.871
V2DEF_AUROC_FORK_DEFECT_FLOOR = 0.86

# Arms with fewer events than this are flagged `low_event_flag=1` in the table and
# called out in the diagnostics. v3's death arm is 3,874 events (14.7 % of the
# composite; it was 43 % in v2), so several death-ladder arms are materially
# thinner than their v2 counterparts and a reader must be able to see which.
LOW_EVENT_FLAG_THRESHOLD = 1000

# ---------------------------------------------------------------------------
# Cohort schema — DISCOVERED AT RUN TIME, never assumed.
# ---------------------------------------------------------------------------
# `01` is under active revision: the death component moved from a calendar-day
# rule (`finaldischarge_days <= 2`) to true elapsed hours (`finaldischarge_hrs
# <= 48`), because the day field is a calendar INDEX whose day-3 bucket spans
# 24.25-71.17 h. `01` additionally emits `death_hemorrhagic_dayrule` /
# `death_le48h_dayrule` preserving the superseded day-rule variants. Nothing here
# hard-codes which timing field exists: the menu is filtered against the columns
# actually present, and the hour-vs-day crosswalk becomes its own reportable arm
# whenever both are available.
TIMING_HRS_COL = "finaldischarge_hrs"
TIMING_DAYS_COL = "finaldischarge_days"

# Must exist or the menu is meaningless -> hard fail.
REQUIRED_OUTCOME_COLS = [
    COMPOSITE_COL, HC_COL, DEATH_COL, TX_COL,
    "death_inhosp", "death_hemorrhagic_headais", "death_le48h",
    "death_hemorrhagic_missing_incl", "death_ed",
    # Recovery-ladder rungs 2 and 3 (CONTRACTS §3 table / §7 menu). REQUIRED, not
    # optional: they are the same class of column as `death_hemorrhagic_missing_incl`
    # (unconditionally emitted by `01`) and the two arms that use them are named in
    # the contract. Their absence means `01` is broken, not that the menu shrank.
    "death_hemorrhagic_dayfallback", "death_hemorrhagic_edincl",
    "substantial_transfusion_ml_only", "major_hemorrhage_event_headais",
    "major_hemorrhage_event_v2def", "blood_4h", "ais_head", "sbp_unrecordable",
]
# Used when present; their absence removes an arm rather than crashing the run.
OPTIONAL_OUTCOME_COLS = [
    TIMING_HRS_COL, TIMING_DAYS_COL,
    "death_hemorrhagic_dayrule", "death_le48h_dayrule",
    "patient_state_composite",
]

NUMERIC_COHORT_COLS = [
    "shock_index", "sbp2", "pulse2", "iss", "ais_head", "age", "sex_male",
    "charlson", "blood_4h",
    "sbp_unrecordable", "unstable_ring", "cc_anticoag", "age_ge65",
] + REQUIRED_OUTCOME_COLS + OPTIONAL_OUTCOME_COLS


# ---------------------------------------------------------------------------
# Logging / hashing / versions
# ---------------------------------------------------------------------------
def _logger() -> logging.Logger:
    lg = logging.getLogger("pelvic_sens13")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg


LOGGER = _logger()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__, "patsy": patsy.__version__,
    }


def _logistic(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _percentile_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(arr, ALPHA / 2.0)), float(np.quantile(arr, 1.0 - ALPHA / 2.0)))


# ---------------------------------------------------------------------------
# Data load + winsorization (CONTRACTS §3 ANALYTIC-COHORT RULE)
# ---------------------------------------------------------------------------
def load_cohort(sample: int | None, acetabulum_inclusive: bool = False) -> pd.DataFrame:
    """Load the cohort CSV, filter to the analytic set, winsorize SI at load.

    acetabulum_inclusive=False -> cohort_primary==1 (ring-only, N≈239,736).
    acetabulum_inclusive=True  -> cohort_primary==1 | has_acetabulum==1
                                   (the free §13 acetabulum sensitivity)."""
    if not os.path.exists(COHORT_CSV):
        raise SystemExit(f"Cohort CSV not found: {COHORT_CSV} (run 01_build_cohort.py).")
    LOGGER.info("Loading cohort %s (acetabulum_inclusive=%s) ...", COHORT_CSV, acetabulum_inclusive)
    df = pd.read_csv(COHORT_CSV, keep_default_na=True, low_memory=False)
    if "cohort_primary" not in df.columns:
        raise SystemExit("cohort CSV missing cohort_primary (CONTRACTS §3).")
    if acetabulum_inclusive:
        if "has_acetabulum" not in df.columns:
            raise SystemExit("cohort CSV missing has_acetabulum (needed for acetabulum sensitivity).")
        mask = (df["cohort_primary"] == 1) | (df["has_acetabulum"] == 1)
        df = df.loc[mask].copy()
        LOGGER.info("Acetabulum-inclusive cohort (primary|has_acetabulum): N=%d", len(df))
    else:
        df = df.loc[df["cohort_primary"] == 1].copy()
        LOGGER.info("Analytic cohort (cohort_primary==1): N=%d", len(df))
    if sample is not None and sample < len(df):
        df = df.sample(n=int(sample), random_state=RANDOM_SEED).reset_index(drop=True)
        LOGGER.info("--sample: downsampled to N=%d (seed=%d)", len(df), RANDOM_SEED)
    for c in NUMERIC_COHORT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Every outcome column the sensitivity menu names must actually exist. A silent
    # KeyError deep inside a variant builder would be attributed to the wrong arm.
    absent = [c for c in REQUIRED_OUTCOME_COLS if c not in df.columns]
    if absent:
        raise SystemExit(
            f"cohort CSV is missing outcome column(s) the v3 sensitivity menu "
            f"requires: {absent}. Re-run 01_build_cohort.py (CONTRACTS §3).")
    # Report the discovered timing schema rather than assuming one. `01` may emit
    # the hour field, the day field, or (during the transition) both.
    have_opt = [c for c in OPTIONAL_OUTCOME_COLS if c in df.columns]
    LOGGER.info("cohort schema: %d columns; optional/timing columns present: %s",
                df.shape[1], have_opt or "none")
    if TIMING_HRS_COL not in df.columns and TIMING_DAYS_COL not in df.columns:
        LOGGER.warning("neither '%s' nor '%s' is present — the <24 h death arm cannot be "
                       "built and will be dropped from the menu.",
                       TIMING_HRS_COL, TIMING_DAYS_COL)
    for c in ["mechanism_cat", "acs_level", "puf_year"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    # ---- Winsorize shock_index at SI_WINSOR_CAP (once, at load) ----
    if "shock_index" in df.columns:
        raw_max = float(df["shock_index"].max())
        n_over = int((df["shock_index"] > SI_WINSOR_CAP).sum())
        df["shock_index"] = df["shock_index"].clip(upper=SI_WINSOR_CAP)
        LOGGER.info("Winsorized shock_index at %.1f: %d rows capped (raw max %.2f -> %.2f).",
                    SI_WINSOR_CAP, n_over, raw_max, float(df["shock_index"].max()))
    return df


def adjustment_for(exposure: str, outcome: str | None = None) -> list[str]:
    """Adjustment set for one (exposure, outcome) run.

    Two strips, both mandated by CONTRACTS:
      * §6 rule — never adjust for the exposure or a mediator (EXPOSURE_STRIP).
      * §6 v3 note / §9 v3-K — never adjust for a covariate that appears in the
        OUTCOME's own definition. `ais_head` is dropped on the mortality-only
        runs (`death_hemorrhagic` and its head-AIS variants) and retained on the
        composite runs. `finaldischarge_days` is outcome-defining everywhere and
        is never in CORE_ADJUSTMENT to begin with.
    """
    strip = set(EXPOSURE_STRIP.get(exposure, [exposure]))
    if outcome is not None:
        strip |= set(OUTCOME_DEFINING_STRIP.get(outcome, []))
    return [c for c in CORE_ADJUSTMENT if c not in strip]


def _rhs(adjustment: Sequence[str]) -> str:
    parts = [f"C({c})" if c in CATEGORICAL_ADJ else c for c in adjustment]
    return " + ".join(parts)


def complete_case(df: pd.DataFrame, exposure: str, outcome: str,
                  adjustment: Sequence[str]) -> pd.DataFrame:
    cols = [outcome, exposure] + list(adjustment)
    cols = [c for c in cols if c in df.columns]
    return df.dropna(subset=cols).copy()


# ---------------------------------------------------------------------------
# HC1 robust logistic + convergence guard (inherited fix, CONTRACTS §9-H1)
# ---------------------------------------------------------------------------
def _assert_converged(res, ctx: str, check_cov: bool = False) -> None:
    """Loud guard: a fit whose REPORTED quantities would be NaN MUST raise. Always
    requires finite llf (non-finite llf -> NaN LR chi2). When check_cov is set (the
    HC1 point models), ALSO requires a fully finite parameter covariance (exactly
    what CIs / spline bands are built from). Keyed on finite llf/cov_params — NOT on
    the statsmodels ``converged`` flag (large interaction models legitimately hit
    maxiter with converged=False yet a finite at-optimum llf)."""
    conv = getattr(res, "converged", None)
    llf = float(getattr(res, "llf", np.nan))
    if not np.isfinite(llf):
        raise RuntimeError(
            f"non-finite log-likelihood GLM fit [{ctx}]: converged={conv}, llf={llf} "
            "(→ NaN LR chi2). Refusing to write NaN (winsorization / convergence guard).")
    if check_cov:
        cov = np.asarray(res.cov_params())
        n_bad = int(np.sum(~np.isfinite(cov)))
        if n_bad > 0:
            raise RuntimeError(
                f"non-finite parameter covariance ({n_bad}/{cov.size} cells) [{ctx}]: the "
                "reported CIs / spline bands would be NaN. Refusing to write NaN.")


def fit_hc1(formula: str, data: pd.DataFrame):
    """smf GLM Binomial with HC1 robust SE (no clustering). Guarded: raises on
    non-finite llf or non-finite parameter covariance — no silent NaN outputs."""
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
    res = model.fit(cov_type="HC1", maxiter=300)
    _assert_converged(res, f"fit_hc1: {formula[:80]}", check_cov=True)
    return res


def lr_test(data: pd.DataFrame, full_formula: str, reduced_formula: str) -> dict[str, float]:
    full = smf.glm(full_formula, data, family=sm.families.Binomial()).fit(maxiter=300)
    _assert_converged(full, f"lr_test full: {full_formula[:80]}")
    red = smf.glm(reduced_formula, data, family=sm.families.Binomial()).fit(maxiter=300)
    _assert_converged(red, f"lr_test reduced: {reduced_formula[:80]}")
    chi2 = 2.0 * (full.llf - red.llf)
    dfree = int(round(full.df_model - red.df_model))
    p = float(1.0 - st.chi2.cdf(chi2, df=max(dfree, 1)))
    return {"chi2": float(chi2), "df": int(dfree), "p": p}


# ---------------------------------------------------------------------------
# Adjusted 4-knot RCS spline + prediction (HC1)
# ---------------------------------------------------------------------------
def spline_formula(exposure: str, outcome: str, adjustment: Sequence[str]) -> str:
    return f"{outcome} ~ cr({exposure}, df=4) + {_rhs(adjustment)}"


def linear_formula(exposure: str, outcome: str, adjustment: Sequence[str]) -> str:
    return f"{outcome} ~ {exposure} + {_rhs(adjustment)}"


def adjonly_formula(outcome: str, adjustment: Sequence[str]) -> str:
    return f"{outcome} ~ {_rhs(adjustment)}"


def _reference_row(data: pd.DataFrame, exposure: str, adjustment: Sequence[str]) -> dict[str, Any]:
    ref: dict[str, Any] = {}
    for c in adjustment:
        col = data[c]
        if c in CATEGORICAL_ADJ:
            m = col.mode(dropna=True)
            ref[c] = m.iloc[0] if len(m) else col.dropna().iloc[0]
        else:
            ref[c] = float(col.mean(skipna=True))
    return ref


def predict_spline(fit, data: pd.DataFrame, exposure: str, adjustment: Sequence[str],
                   grid: np.ndarray) -> pd.DataFrame:
    """Predicted risk + 95% CI along ``grid`` (covariates at reference). Propagates
    the HC1 robust covariance to the response-scale CI via get_prediction."""
    ref = _reference_row(data, exposure, adjustment)
    newdata = pd.DataFrame({exposure: grid})
    for k, v in ref.items():
        newdata[k] = v
    pr = fit.get_prediction(newdata)
    sf = pr.summary_frame(alpha=ALPHA)
    mean = sf["mean"].to_numpy()
    lo = sf["mean_ci_lower"].to_numpy()
    hi = sf["mean_ci_upper"].to_numpy()
    return pd.DataFrame({"exposure_grid": grid, "pred_risk": mean,
                         "lci": np.minimum(lo, hi), "uci": np.maximum(lo, hi)})


def spline_grid_for(exposure: str, data: pd.DataFrame) -> np.ndarray:
    s = data[exposure].dropna()
    lo, hi = np.percentile(s, 1.0), np.percentile(s, 99.0)
    step = 0.02 if exposure == "shock_index" else 2.0
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    return np.round(np.arange(lo, hi + step / 2, step), 4)


# ---------------------------------------------------------------------------
# Segmented-logistic change-point (FULL adjusted refit; fast deviance fitter)
# ---------------------------------------------------------------------------
def build_adj_design_ni(data: pd.DataFrame, adjustment: Sequence[str]) -> np.ndarray:
    """patsy adjustment design WITHOUT the intercept (sklearn adds its own)."""
    X = patsy.dmatrix("~ " + _rhs(adjustment), data, return_type="dataframe")
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    return X.to_numpy().astype(float)


def _seg_deviance(Xadj_ni: np.ndarray, x: np.ndarray, y: np.ndarray,
                  bp: float) -> tuple[float, float, float]:
    """Full adjusted segmented logistic at breakpoint ``bp`` via sklearn
    newton-cholesky (no HC1 — deviance is cov-type invariant). Exposure enters as
    b_pre*x + b_delta*max(0,x-bp); beta_post = b_pre + b_delta."""
    X = np.column_stack([Xadj_ni, x, np.maximum(x - bp, 0.0)])
    m = LogisticRegression(penalty=None, solver="newton-cholesky", max_iter=100).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    eps = 1e-12
    dev = -2.0 * float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    coef = m.coef_[0]
    return dev, float(coef[-2]), float(coef[-2] + coef[-1])


def interior_grid(x: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo = float(np.percentile(x, INTERIOR_LO_PCT))
    hi = float(np.percentile(x, INTERIOR_HI_PCT))
    g = grid[(grid >= lo) & (grid <= hi)]
    return g, lo, hi


def changepoint_search(Xadj_ni: np.ndarray, x: np.ndarray, y: np.ndarray,
                       grid: np.ndarray) -> dict[str, Any]:
    """Deviance-minimizing breakpoint over ``grid`` (interior-restricted) with a
    >=MIN_EVENTS_PER_SIDE guard. Full adjusted refit at each grid point."""
    best = {"breakpoint": float("nan"), "deviance": float("inf"),
            "beta_pre": float("nan"), "beta_post": float("nan"), "n_valid_grid": 0}
    n_valid = 0
    for bp in grid:
        ev_lo = int(y[x < bp].sum())
        ev_hi = int(y[x >= bp].sum())
        if min(ev_lo, ev_hi) < MIN_EVENTS_PER_SIDE:
            continue
        n_valid += 1
        try:
            dev, b_pre, b_post = _seg_deviance(Xadj_ni, x, y, float(bp))
        except Exception:  # noqa: BLE001
            continue
        if dev < best["deviance"]:
            best = {"breakpoint": float(bp), "deviance": float(dev),
                    "beta_pre": b_pre, "beta_post": b_post, "n_valid_grid": 0}
    best["n_valid_grid"] = n_valid
    return best


def _boot_one_changepoint(Xadj_ni: np.ndarray, x: np.ndarray, y: np.ndarray,
                          grid: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    take = rng.choice(x.shape[0], size=x.shape[0], replace=True)
    return changepoint_search(Xadj_ni[take], x[take], y[take], grid)["breakpoint"]


def bootstrap_changepoint(Xadj_ni: np.ndarray, x: np.ndarray, y: np.ndarray,
                          grid: np.ndarray, n_boot: int, seed: int = RANDOM_SEED,
                          n_jobs: int = -1) -> tuple[float, float, list[float]]:
    """Row-level nonparametric bootstrap of the change-point, FULL adjusted refit
    per resample, parallelized across cores (joblib; loky memmaps read-only arrays)."""
    samples = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
        delayed(_boot_one_changepoint)(Xadj_ni, x, y, grid, seed + i)
        for i in range(int(n_boot))
    )
    lo, hi = _percentile_ci(samples)
    return lo, hi, list(samples)


def _or_per_unit_from_slope(beta: float, unit: float) -> float:
    return float(np.exp(beta * unit))


def compute_changepoint(dcc: pd.DataFrame, exposure: str, outcome: str,
                        adjustment: Sequence[str], grid_full: np.ndarray,
                        n_boot: int, seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Point change-point (+ optional bootstrap CI) with all diagnostics. Reused by
    the component decomposition AND the sensitivity variants."""
    x = dcc[exposure].to_numpy(float)
    y = dcc[outcome].to_numpy(float)
    g, ilo, ihi = interior_grid(x, grid_full)
    Xadj = build_adj_design_ni(dcc, adjustment)
    cp = changepoint_search(Xadj, x, y, g)
    unit = SI_UNIT if exposure == "shock_index" else SBP_UNIT
    res: dict[str, Any] = {
        "exposure": exposure, "outcome": outcome, "breakpoint": cp["breakpoint"],
        "beta_pre": cp["beta_pre"], "beta_post": cp["beta_post"],
        "or_per_unit_pre": (_or_per_unit_from_slope(cp["beta_pre"], unit)
                            if np.isfinite(cp["beta_pre"]) else float("nan")),
        "or_per_unit_post": (_or_per_unit_from_slope(cp["beta_post"], unit)
                             if np.isfinite(cp["beta_post"]) else float("nan")),
        "n": int(len(dcc)), "events": int(y.sum()),
        "interior_lo": round(ilo, 4), "interior_hi": round(ihi, 4),
        "grid_min": float(g.min()) if g.size else float("nan"),
        "grid_max": float(g.max()) if g.size else float("nan"),
        "grid_step": (0.02 if exposure == "shock_index" else 2.0),
        "n_valid_grid": cp["n_valid_grid"], "n_boot": int(n_boot),
        "lci": float("nan"), "uci": float("nan"),
    }
    if n_boot and n_boot > 0 and np.isfinite(cp["breakpoint"]) and g.size:
        lo, hi, samples = bootstrap_changepoint(Xadj, x, y, g, int(n_boot), seed=seed)
        res["lci"], res["uci"] = lo, hi
        _bs = np.array([s for s in samples if s is not None and np.isfinite(s)], dtype=float)
        res["boot_n_ok"] = int(_bs.size)
        res["boot_n_unique"] = int(np.unique(_bs).size) if _bs.size else 0
    res["at_grid_boundary"] = bool(
        np.isfinite(cp["breakpoint"]) and g.size
        and (abs(cp["breakpoint"] - g.min()) < 1e-9 or abs(cp["breakpoint"] - g.max()) < 1e-9))
    return res


# ---------------------------------------------------------------------------
# Pre-specified thresholds: adjusted OR + g-computed absolute risk + RD (HC1)
# ---------------------------------------------------------------------------
def _indicator(data: pd.DataFrame, exposure: str, t: float) -> pd.Series:
    """Danger-side indicator: SI -> I(SI>=t); SBP -> I(SBP<t)."""
    if exposure == "shock_index":
        return (data[exposure] >= t).astype(float)
    return (data[exposure] < t).astype(float)


def _danger_side_label(exposure: str, t: float) -> str:
    return f"SI>={t}" if exposure == "shock_index" else f"SBP<{t}"


def threshold_gcomp(data: pd.DataFrame, exposure: str, outcome: str, t: float,
                    adjustment: Sequence[str]) -> dict[str, Any]:
    """Fit outcome ~ ind + adjustment (HC1); g-compute adjusted risk on the danger
    side vs safe side + adjusted OR; delta-method RD 95% CI on the HC1 covariance."""
    d = data.copy()
    d["_ind"] = _indicator(d, exposure, t)
    formula = f"{outcome} ~ _ind + {_rhs(adjustment)}"
    fit = fit_hc1(formula, d)
    di = fit.model.data.design_info

    def _std(indval: float) -> tuple[float, np.ndarray, np.ndarray]:
        dd = d.copy()
        dd["_ind"] = float(indval)
        X = patsy.build_design_matrices([di], dd, return_type="dataframe")[0]
        X = X.reindex(columns=fit.params.index, fill_value=0.0).to_numpy()
        p = _logistic(X @ fit.params.to_numpy())
        return float(np.mean(p)), X, p

    r_safe, X0, p0 = _std(0.0)
    r_danger, X1, p1 = _std(1.0)
    n = X0.shape[0]
    grad = (X1.T @ (p1 * (1 - p1))) / n - (X0.T @ (p0 * (1 - p0))) / n
    cov = np.asarray(fit.cov_params())
    se_rd = float(np.sqrt(max(float(grad @ cov @ grad), 0.0)))
    rd = r_danger - r_safe
    beta = float(fit.params["_ind"])
    se_b = float(fit.bse["_ind"])
    p_raw = float(fit.pvalues["_ind"])
    return {
        "adj_risk_danger": r_danger, "adj_risk_safe": r_safe, "adj_risk_diff": rd,
        "rd_lci": rd - Z * se_rd, "rd_uci": rd + Z * se_rd,
        "adj_or": float(np.exp(beta)), "or_lci": float(np.exp(beta - Z * se_b)),
        "or_uci": float(np.exp(beta + Z * se_b)), "p_raw": p_raw,
        "n_danger": int((d["_ind"] == 1).sum()), "n_safe": int((d["_ind"] == 0).sum()),
    }


# ---------------------------------------------------------------------------
# VIF audit (CONTRACTS §6) — replicate 11 so 13's adjustment set matches.
# ---------------------------------------------------------------------------
def vif_audit(df: pd.DataFrame, diag: list[str]) -> list[str]:
    # CONTRACTS §13 G3f (asymmetry guard). Fix 3 removed `unstable_ring` from the
    # SCORE; removing it here would degrade the association model while answering
    # an objection that was never about the association model (§6).
    assert RING_VAR in VIF_AUDIT_VARS, (
        f"CONTRACT VIOLATION (§13 G3f): '{RING_VAR}' must remain in "
        f"13.VIF_AUDIT_VARS. Fix 3 removes it from the SCORE, not from adjustment.")
    assert RING_VAR in CORE_ADJUSTMENT, (
        f"CONTRACT VIOLATION (§6/§13 G3f): '{RING_VAR}' must remain in "
        f"13.CORE_ADJUSTMENT.")
    diag.append(f"  [G3f] '{RING_VAR}' retained in CORE_ADJUSTMENT and VIF_AUDIT_VARS "
                f"(deliberate §6 asymmetry — out of the score, in the adjustment set).")
    sub = df[VIF_AUDIT_VARS].dropna().astype(float)
    X = sm.add_constant(sub.values)
    names = ["const"] + VIF_AUDIT_VARS
    vifs = {}
    diag.append("\n=== VIF audit {iss, ais_head, unstable_ring} ===")
    for i, nm in enumerate(names):
        # CONTRACTS §13 preamble rule 1/2. This previously read
        #   `except Exception: v = float("nan")`
        # and the decision below is `vifs.get("ais_head", 0.0) > VIF_THRESHOLD`.
        # `nan > 5.0` is False, so an *uncomputable* VIF silently produced the
        # benign answer — "no collinearity, ais_head RETAINED" — from a swallowed
        # exception. Same shape as G4c. The failure now surfaces.
        v = float(variance_inflation_factor(X, i))
        if not np.isfinite(v):
            diag.append(f"  {nm:16s} VIF=NON-FINITE")
            LOGGER.error("GATE VIF-AUDIT: NOT EVALUATED — FAIL. VIF for %r is %r.", nm, v)
            raise SystemExit(
                f"GATE VIF-AUDIT: NOT EVALUATED — FAIL. The VIF for {nm!r} is not a "
                f"finite number, so the collinearity decision that governs "
                f"CORE_ADJUSTMENT cannot be made. Silence is not a pass: a non-finite "
                f"VIF must not be allowed to read as 'below threshold' (CONTRACTS §13 "
                f"preamble).")
        vifs[nm] = v
        diag.append(f"  {nm:16s} VIF={v:.3f}")
    dropped = []
    if vifs["ais_head"] > VIF_THRESHOLD:
        dropped.append("ais_head")
        diag.append(f"  -> ais_head VIF>{VIF_THRESHOLD}: DROPPED (prefer unstable_ring+iss).")
    else:
        diag.append(f"  -> all VIF<={VIF_THRESHOLD}: ais_head RETAINED.")
    return dropped


# ---------------------------------------------------------------------------
# DELIVERABLE 1 — component decomposition (splines + change-points; validity)
# ---------------------------------------------------------------------------
def build_component_decomposition(df: pd.DataFrame, n_boot_death: int, n_boot_other: int,
                                  diag: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """For each exposure x component: adjusted 4-knot RCS spline (HC1, fig-data F8)
    + deviance-min change-point. death_hemorrhagic (headline non-discretionary
    anchor) gets the full bootstrap CI; HC/tx get a reduced-boot CI (secondary
    deviation). The adjustment set is stripped PER OUTCOME (§9 v3-K): `ais_head`
    leaves on the mortality-only run, because it is part of that outcome's own
    definition."""
    spline_rows: list[pd.DataFrame] = []
    cps: dict[str, Any] = {ex: {} for ex in EXPOSURES}
    diag.append("\n=== DELIVERABLE 1: COMPONENT DECOMPOSITION (composite validity) ===")
    for exposure in EXPOSURES:
        grid_full = GRID_SI if exposure == "shock_index" else GRID_SBP
        for comp in COMPONENT_OUTCOMES:
            adj = adjustment_for(exposure, outcome=comp)
            stripped = [c for c in adjustment_for(exposure) if c not in adj]
            if stripped:
                diag.append(f"  [v3-K] {exposure}/{comp}: adjustment strip {stripped} "
                            f"(outcome-defining; adjusting for it would be circular)")
            dcc = complete_case(df, exposure, comp, adj)
            n, ev = len(dcc), int(dcc[comp].sum())
            if ev < 2 * MIN_EVENTS_PER_SIDE:
                diag.append(f"  [POWER] {exposure}/{comp}: only {ev} events — the "
                            f">={MIN_EVENTS_PER_SIDE}-events/side guard may bind; a "
                            f"non-finite change-point here is descriptive-only, "
                            f"not a failure (locked SAP §6).")
            # ---- adjusted spline (fig-data F8) ----
            try:
                fit = fit_hc1(spline_formula(exposure, comp, adj), dcc)
                grid = spline_grid_for(exposure, dcc)
                pred = predict_spline(fit, dcc, exposure, adj, grid)
                pred.insert(0, "component", comp)
                pred.insert(0, "exposure", exposure)
                spline_rows.append(pred)
                n_nan_band = int((~np.isfinite(pred["lci"])).sum() + (~np.isfinite(pred["uci"])).sum())
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("component spline %s/%s FAILED: %s", exposure, comp, exc)
                n_nan_band = -1
            # ---- change-point ----
            nb = n_boot_death if comp == DEATH_COL else n_boot_other
            cp = compute_changepoint(dcc, exposure, comp, adj, grid_full, nb)
            cp["component"] = comp
            cp["label"] = COMPONENT_LABEL[comp]
            cps[exposure][comp] = cp
            LOGGER.info("[decomp %s/%s] N=%d ev=%d bp=%s CI[%s,%s] band_nan=%d",
                        exposure, comp, n, ev,
                        round(cp["breakpoint"], 3) if np.isfinite(cp["breakpoint"]) else "nan",
                        round(cp["lci"], 3) if np.isfinite(cp["lci"]) else "nan",
                        round(cp["uci"], 3) if np.isfinite(cp["uci"]) else "nan", n_nan_band)
            diag.append(
                f"  {exposure:11s} {comp:24s} N={n} ev={ev} bp={cp['breakpoint']} "
                f"CI[{cp['lci']},{cp['uci']}] boundary={cp['at_grid_boundary']} "
                f"band_nonfinite={n_nan_band}")
    decomp_df = pd.concat(spline_rows, ignore_index=True) if spline_rows else pd.DataFrame()
    return decomp_df, cps


# ---------------------------------------------------------------------------
# DELIVERABLE 2 — Venn set counts among composite-positive patients
# ---------------------------------------------------------------------------
def build_venn(df: pd.DataFrame, diag: list[str]) -> dict[str, Any]:
    """Set arithmetic on the analytic cohort among composite-positive patients.

    v3: the death set is `death_hemorrhagic` (Fix 2/2b) and the transfusion set is
    the unit-aware `substantial_transfusion` (Fix 1) — i.e. exactly the three sets
    that define `major_hemorrhage_event` in v3. Using v2's `death_inhosp` here
    would produce orphan composite-positives and a region sum that does not close.
    """
    h = (df[HC_COL] == 1)
    d = (df[DEATH_COL] == 1)
    t = (df[TX_COL] == 1)
    comp = (df[COMPOSITE_COL] == 1)
    regions = {
        "hc_only": int((h & ~d & ~t).sum()),
        "death_only": int((~h & d & ~t).sum()),
        "transfusion_only": int((~h & ~d & t).sum()),
        "hc_and_death": int((h & d & ~t).sum()),
        "hc_and_transfusion": int((h & ~d & t).sum()),
        "death_and_transfusion": int((~h & d & t).sum()),
        "all_three": int((h & d & t).sum()),
    }
    region_sum = sum(regions.values())
    n_comp = int(comp.sum())
    # a composite-positive with h=d=t=0 would be a definition error
    orphan = int((comp & ~h & ~d & ~t).sum())
    venn = {
        "cohort": "cohort_primary==1",
        "n_analytic": int(len(df)),
        "composite_positive": n_comp,
        "composite_definition": f"{HC_COL} | {DEATH_COL} | {TX_COL}",
        "marginals": {HC_COL: int(h.sum()), DEATH_COL: int(d.sum()),
                      TX_COL: int(t.sum())},
        "regions": regions,
        "region_sum": region_sum,
        "check_sum_equals_composite": bool(region_sum == n_comp),
        "orphan_composite_positive": orphan,
        "pct_of_composite": {k: round(100.0 * v / n_comp, 2) for k, v in regions.items()} if n_comp else {},
    }
    diag.append("\n=== DELIVERABLE 2: COMPONENT VENN (among composite-positive) ===")
    diag.append(f"  composite-positive N={n_comp}; region sum={region_sum}; "
                f"sum==composite: {venn['check_sum_equals_composite']}; orphans={orphan}")
    for k, v in regions.items():
        diag.append(f"    {k:24s} {v:7d}  ({round(100.0*v/n_comp,2) if n_comp else 0}% of composite)")
    return venn


# ---------------------------------------------------------------------------
# Composite reference (script 11) + co-location determination
# ---------------------------------------------------------------------------
def load_composite_reference() -> dict[str, Any]:
    """Read script 11's 4 change-point JSONs for the co-location comparison; fall
    back to hard-coded CONTRACTS §13.4 constants if any are absent (self-contained)."""
    ref: dict[str, Any] = {"source": "script_11_changepoint_jsons", "values": {},
                           "detail": {}, "stale_tags": [], "stale": False,
                           "descriptive_only_tags": []}
    for tag, path in REF_CHANGEPOINT_JSONS.items():
        try:
            with open(path) as f:
                j = json.load(f)
            # Script 11 now DEMOTES an under-powered stratum to descriptive-only and
            # writes `breakpoint: NaN` with the key still PRESENT — so there is no
            # KeyError to catch and a NaN would otherwise propagate silently into
            # this table as if it were a failed estimate. Branch on the explicit
            # `descriptive_only` flag (and on non-finiteness as a backstop) and
            # represent the stratum as "descriptive-only, no breakpoint estimated",
            # which is a different claim from "missing" or "does not co-locate".
            bp_raw = j.get("breakpoint", float("nan"))
            try:
                bp = float(bp_raw) if bp_raw is not None else float("nan")
            except (TypeError, ValueError):
                bp = float("nan")
            desc_only = bool(j.get("descriptive_only", False)) or (not np.isfinite(bp))
            ref["values"][tag] = bp
            ref["detail"][tag] = {
                "breakpoint": bp, "lci": j.get("lci"), "uci": j.get("uci"),
                "mi_location": j.get("mi_pooled_location", {}).get("location"),
                "descriptive_only": desc_only,
                "guard_reason": j.get("guard_reason"),
                "adjustment": j.get("adjustment"),
                "adjustment_stripped": j.get("adjustment_stripped"),
                "mortality_only_run": j.get("mortality_only_run"),
            }
            if desc_only:
                ref["descriptive_only_tags"].append(tag)
                LOGGER.warning(
                    "script-11 reference '%s' is DESCRIPTIVE-ONLY (no breakpoint "
                    "estimated; guard_reason=%r). The co-location verdict against it is "
                    "UNDETERMINED and is reported as such — not as 'does not co-locate'.",
                    tag, j.get("guard_reason"))
        except Exception:  # noqa: BLE001
            ref["values"][tag] = REF_FALLBACK_V2_STALE[tag]
            ref["detail"][tag] = {"breakpoint": REF_FALLBACK_V2_STALE[tag],
                                  "source": "V2_FALLBACK_CONSTANT_STALE",
                                  "descriptive_only": False, "guard_reason": None}
            ref["stale_tags"].append(tag)
            ref["stale"] = True
            ref["source"] = "mixed_(some_V2_STALE_fallback)"
            LOGGER.warning(
                "script-11 change-point JSON missing for '%s' (%s). Falling back to the "
                "**v2** constant %.4f, which was computed under the v2 outcome definition "
                "and is SUPERSEDED. The co-location verdict for this exposure is "
                "UNRELIABLE and must not be reported. Run 11_pelvic_thresholds.py first.",
                tag, path, REF_FALLBACK_V2_STALE[tag])
    return ref


def colocation_report(component_cps: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Death-only (non-discretionary) component change-point vs the COMPOSITE
    change-point (script 11), per exposure. Co-location = within 1 grid step OR the
    death-only bootstrap CI covers the composite breakpoint. Also cross-checks the
    death-only component bp against script 11's mortality-only curve (should match)."""
    out: dict[str, Any] = {}
    for exposure in EXPOSURES:
        tag_comp = "si_composite" if exposure == "shock_index" else "sbp_composite"
        tag_mort = "si_mortality" if exposure == "shock_index" else "sbp_mortality"
        step = 0.02 if exposure == "shock_index" else 2.0
        d = component_cps[exposure][DEATH_COL]
        death_bp = float(d["breakpoint"])
        comp_bp = float(ref["values"][tag_comp])
        mort_bp = float(ref["values"][tag_mort])
        delta = death_bp - comp_bp if (np.isfinite(death_bp) and np.isfinite(comp_bp)) else float("nan")
        ci_covers = bool(np.isfinite(d["lci"]) and np.isfinite(d["uci"])
                         and d["lci"] - 1e-9 <= comp_bp <= d["uci"] + 1e-9)
        colocates = bool((np.isfinite(delta) and abs(delta) <= step + 1e-9) or ci_covers)
        stale = bool(tag_comp in ref.get("stale_tags", []) or tag_mort in ref.get("stale_tags", []))

        # A demoted (descriptive-only) reference or an unestimable death-only
        # breakpoint means the verdict is UNDETERMINED. That is a different claim
        # from "does not co-locate" and must never be collapsed into False.
        ref_desc = bool(ref["detail"].get(tag_comp, {}).get("descriptive_only", False))
        death_desc = not np.isfinite(death_bp)
        undetermined = bool(stale or ref_desc or death_desc)
        if undetermined:
            why = ("script-11 reference unavailable (v2 stale fallback)" if stale else
                   "script-11 composite change-point is DESCRIPTIVE-ONLY "
                   f"(guard_reason={ref['detail'].get(tag_comp, {}).get('guard_reason')!r})"
                   if ref_desc else
                   "death-only component change-point not estimated "
                   f"(>= {MIN_EVENTS_PER_SIDE} events/side guard; the v3 death arm is a much "
                   "smaller event set than v2's)")
            verdict = "UNDETERMINED — descriptive-only, no breakpoint estimated"
        else:
            why = ""
            verdict = "co-locates" if colocates else "does NOT co-locate"

        out[exposure] = {
            "death_component_column": DEATH_COL,
            "reference_is_v2_stale_fallback": stale,
            "reference_descriptive_only": ref_desc,
            "reference_guard_reason": ref["detail"].get(tag_comp, {}).get("guard_reason"),
            "death_only_descriptive_only": death_desc,
            "verdict_reliable": (not undetermined),
            "verdict": verdict,
            "verdict_undetermined_reason": why,
            "death_only_breakpoint": (death_bp if np.isfinite(death_bp) else None),
            "death_only_ci": [d["lci"], d["uci"]],
            "composite_breakpoint_script11": (comp_bp if np.isfinite(comp_bp) else None),
            "mortality_only_breakpoint_script11": (mort_bp if np.isfinite(mort_bp) else None),
            "delta_death_minus_composite": (round(delta, 4) if np.isfinite(delta) else None),
            "grid_step": step,
            "ci_covers_composite": ci_covers,
            # None (not False) whenever the verdict is undetermined.
            "colocates": (None if undetermined else colocates),
            "composite_at_grid_floor": bool(exposure == "shock_index"
                                            and np.isfinite(comp_bp)
                                            and abs(comp_bp - float(GRID_SI.min())) < 1e-9),
            "death_component_matches_script11_mortality": (
                None if (death_desc or not np.isfinite(mort_bp)) else
                bool(abs(death_bp - mort_bp) <= step + 1e-9)),
        }
    return out


# ---------------------------------------------------------------------------
# DELIVERABLE 3 — sensitivity variants -> table7_sensitivity.csv
# ---------------------------------------------------------------------------
SENS_COLS = ["variant", "arm", "cohort_def", "n", "n_events", "composite_rate_pct",
             "low_event_flag",
             "exposure", "outcome_def", "analysis", "term",
             "adj_or", "or_lci", "or_uci", "p_raw", "p_bonferroni",
             "changepoint", "cp_lci", "cp_uci", "at_grid_boundary",
             # A change-point row is DESCRIPTIVE-ONLY when the >=100-events/side guard
             # (or the interior restriction) prevented an estimate. That is a
             # different fact from a missing value, and with the v3 death arm this
             # thin it is expected on some strata — so it gets its own column
             # instead of being inferred from a NaN.
             "descriptive_only", "guard_reason",
             # Analysis-A (prediction) lens — populated by the score arms only.
             "auroc", "auroc_lci", "auroc_uci", "n_score_vars", "score_vars",
             "notes"]

# Every row carries the arm's event count so a reader can see WHICH arms are thin
# rather than inferring it from a wide CI. v3's death arm is 3,874 events (14.7 %
# of the composite; 43 % in v2), so the death-ladder arms are materially thinner
# than their v2 counterparts.
def _low_event_flag(ev) -> int:
    # CONTRACTS §13 preamble rule 1 — the coercion is EXPLICIT and its success is
    # ASSERTED. This previously read `except (TypeError, ValueError): return 0`,
    # i.e. an unreadable event count silently produced "not thin" — the benign
    # answer, in the same shape as G4c's swallowed comparison. Every call site
    # passes an `int(...)` sum, so a failure here means the caller is broken and
    # must surface, not be absorbed.
    val = pd.to_numeric(ev, errors="coerce")
    val = float(val) if val is not None else float("nan")
    if not np.isfinite(val):
        raise ValueError(
            f"low_event_flag: event count {ev!r} is not a finite number. The flag "
            f"marks which sensitivity arms are too thin to read as inferential; "
            f"defaulting it to 0 ('not thin') on an unreadable count would mislabel "
            f"an arm as adequately powered (CONTRACTS §13 preamble).")
    return int(val < LOW_EVENT_FLAG_THRESHOLD)


def _blank_row() -> dict[str, Any]:
    return {c: np.nan for c in SENS_COLS}


def _cp_descriptive_only(cp: dict[str, Any]) -> tuple[int, str]:
    """Was this change-point demoted to descriptive-only? Prefer an explicit flag,
    fall back to non-finiteness. Returns (flag, human-readable reason)."""
    bp = cp.get("breakpoint", float("nan"))
    try:
        finite = np.isfinite(float(bp))
    except (TypeError, ValueError):
        finite = False
    flag = bool(cp.get("descriptive_only", False)) or (not finite)
    if not flag:
        return 0, ""
    reason = cp.get("guard_reason")
    if not reason:
        n_ev = cp.get("events")
        reason = (f"no breakpoint estimated: the >= {MIN_EVENTS_PER_SIDE}-events/side guard "
                  f"or the interior 5th-95th percentile restriction left no admissible grid "
                  f"point" + (f" (arm events = {n_ev})" if n_ev is not None else ""))
    return 1, str(reason)


def _cp_row(variant, arm, cohort_def, n, ev, rate, exposure, outcome_def, cp, notes=""):
    row = _blank_row()
    desc, reason = _cp_descriptive_only(cp)
    if desc and notes:
        notes = f"{notes} | DESCRIPTIVE-ONLY: {reason}"
    elif desc:
        notes = f"DESCRIPTIVE-ONLY: {reason}"
    row.update({
        "variant": variant, "arm": arm, "cohort_def": cohort_def, "n": n, "n_events": ev,
        "composite_rate_pct": round(rate, 4), "low_event_flag": _low_event_flag(ev),
        "exposure": exposure, "outcome_def": outcome_def,
        "analysis": "changepoint",
        "term": ("SI inflection (rising)" if exposure == "shock_index" else "SBP inflection (falling)"),
        "changepoint": (round(cp["breakpoint"], 4) if np.isfinite(cp["breakpoint"]) else np.nan),
        "cp_lci": (round(cp["lci"], 4) if np.isfinite(cp["lci"]) else np.nan),
        "cp_uci": (round(cp["uci"], 4) if np.isfinite(cp["uci"]) else np.nan),
        "at_grid_boundary": cp.get("at_grid_boundary", np.nan),
        "descriptive_only": desc, "guard_reason": reason, "notes": notes,
    })
    return row


def _thr_row(variant, arm, cohort_def, n, ev, rate, exposure, outcome_def, t, g, notes=""):
    p_bonf = float(min(1.0, g["p_raw"] * BONFERRONI_M))
    row = _blank_row()
    row.update({
        "variant": variant, "arm": arm, "cohort_def": cohort_def, "n": n, "n_events": ev,
        "composite_rate_pct": round(rate, 4), "low_event_flag": _low_event_flag(ev),
        "descriptive_only": 0, "guard_reason": "",
        "exposure": exposure, "outcome_def": outcome_def,
        "analysis": "prespecified_threshold", "term": _danger_side_label(exposure, t),
        "adj_or": round(g["adj_or"], 4), "or_lci": round(g["or_lci"], 4), "or_uci": round(g["or_uci"], 4),
        "p_raw": g["p_raw"], "p_bonferroni": p_bonf,
        "notes": notes,
    })
    return row


def run_composite_variant(df: pd.DataFrame, variant: str, arm: str, cohort_def: str,
                          composite: pd.Series, outcome_def: str, adj_map: dict[str, list[str]],
                          n_boot_sbp: int, boot_si: int, diag: list[str],
                          notes: str = "") -> list[dict[str, Any]]:
    """Attach a redefined composite to df, then per exposure recompute the change-point
    (SBP bootstrapped = headline; SI point/reduced-boot) + the 2 pre-specified threshold
    ORs. One row-group. df is the (possibly acetabulum-inclusive) cohort."""
    work = df.copy()
    work["_compvar"] = composite.astype(float).to_numpy()
    n = int(len(work))
    ev = int(work["_compvar"].sum())
    rate = 100.0 * work["_compvar"].mean()
    rows: list[dict[str, Any]] = []
    diag.append(f"\n--- VARIANT {variant} [{arm}] : {outcome_def} | cohort={cohort_def} "
                f"| N={n} events={ev} ({rate:.2f}%) ---")
    if ev < LOW_EVENT_FLAG_THRESHOLD:
        diag.append(f"    [POWER] {ev} events (< {LOW_EVENT_FLAG_THRESHOLD}) — this arm is "
                    f"thin; low_event_flag=1 on every row.")
    for exposure in EXPOSURES:
        adj = adj_map[exposure]
        grid_full = GRID_SI if exposure == "shock_index" else GRID_SBP
        dcc = complete_case(work, exposure, "_compvar", adj)
        nb = n_boot_sbp if exposure == "sbp2" else boot_si
        cp = compute_changepoint(dcc, exposure, "_compvar", adj, grid_full, nb)
        rows.append(_cp_row(variant, arm, cohort_def, n, ev, rate, exposure, outcome_def, cp,
                            notes=notes))
        diag.append(f"    {exposure:11s} change-point bp={cp['breakpoint']} "
                    f"CI[{cp['lci']},{cp['uci']}] boundary={cp['at_grid_boundary']}")
        for t in PRESPEC_THRESHOLDS[exposure]:
            g = threshold_gcomp(dcc, exposure, "_compvar", t, adj)
            rows.append(_thr_row(variant, arm, cohort_def, n, ev, rate, exposure, outcome_def, t, g,
                                notes=notes))
            diag.append(f"      {_danger_side_label(exposure,t):9s} OR={g['adj_or']:.3f} "
                        f"[{g['or_lci']:.3f},{g['or_uci']:.3f}]")
    return rows


# ---------------------------------------------------------------------------
# THE NAMED VARIANT REGISTRY — one definition source for BOTH lenses
# ---------------------------------------------------------------------------
# Every arm's outcome is built HERE and nowhere else, so the change-point lens and
# the score-AUROC lens can never drift apart on what "sens_death_le48h" means.
# `variant` strings are the CONTRACTS §7 names, character-for-character.
#
# NOTE on missing `finaldischarge_days`: `NaN <= k` is False, so a death with no
# discharge-day offset is NOT counted as within the window. That is the
# pre-specified conservative rule (CONTRACTS §3); `sens_death_missing_incl` is the
# bounding arm that counts all of them IN, and the pair brackets the truth.


def _tx_units_or_ml(blood: pd.Series, ml_min: float, units_min: float) -> pd.Series:
    """Unit-aware transfusion rule (Fix 1 / CONTRACTS §3), parameterised by cut.

    mL branch  : v >= ml_min
    units branch: units_min <= v <= BLOOD_UNITS_MAX
    Nothing may be classified through the empirical 101-199 gap: the units branch
    is capped at BLOOD_UNITS_MAX (100) and the mL branch starts at ml_min (>=1000),
    so the gap is unreachable from either side.
    """
    b = pd.to_numeric(blood, errors="coerce")
    return ((b >= ml_min) | ((b >= units_min) & (b <= BLOOD_UNITS_MAX))).astype(float)


def build_variant_outcome(df: pd.DataFrame, variant: str) -> tuple[pd.Series, str]:
    """Return (outcome Series aligned to df.index, human-readable definition)."""
    hc = (df[HC_COL] == 1)
    blood = df["blood_4h"] if "blood_4h" in df.columns else pd.Series(np.nan, index=df.index)

    def _comp(death_mask: pd.Series, tx_mask: pd.Series) -> pd.Series:
        return (hc | death_mask | (tx_mask == 1)).astype(float)

    tx_primary = (df[TX_COL] == 1)
    death_primary = (df[DEATH_COL] == 1)

    if variant in ("primary_score_reference", "sens_acetabulum", "sens_exclude_nslo",
                   "sens_with_ring"):
        return (df[COMPOSITE_COL].astype(float),
                f"{COMPOSITE_COL} = {HC_COL} | {DEATH_COL} | {TX_COL}")

    if variant == "sens_v2def_outcome":
        return (df["major_hemorrhage_event_v2def"].astype(float),
                "major_hemorrhage_event_v2def = received_hc | death_inhosp | "
                "substantial_transfusion_ml_only  (the EXACT v2 primary)")

    if variant == "sens_tx_ml_only":
        return (_comp(death_primary, (df["substantial_transfusion_ml_only"] == 1)),
                f"{HC_COL} | {DEATH_COL} | substantial_transfusion_ml_only  "
                "(v2 mL-only transfusion rule, blood_4h >= 1000)")

    if variant == "sens_death_allcause":
        return (_comp((df["death_inhosp"] == 1), tx_primary),
                f"{HC_COL} | death_inhosp | {TX_COL}  (rung 0: all-cause in-hospital death)")

    if variant == "sens_death_headais":
        return (_comp((df["death_hemorrhagic_headais"] == 1), tx_primary),
                f"{HC_COL} | death_hemorrhagic_headais | {TX_COL}  "
                "(rung 1a: head AIS<4 only, NO timing restriction) "
                "== major_hemorrhage_event_headais")

    if variant == "sens_death_le48h":
        return (_comp((df["death_le48h"] == 1), tx_primary),
                f"{HC_COL} | death_le48h | {TX_COL}  "
                "(rung 1b: finaldischarge_days<=2 only, NO head-AIS restriction)")

    if variant == "sens_death_missing_incl":
        return (_comp((df["death_hemorrhagic_missing_incl"] == 1), tx_primary),
                f"{HC_COL} | death_hemorrhagic_missing_incl | {TX_COL}  "
                "(rung 4: BOUNDING arm — the deaths with no finaldischarge_hrs "
                "counted as within 48 h instead of outside it)")

    if variant == "sens_death_dayfallback":
        # RECOVERY LADDER RUNG 2 (CONTRACTS §3/§7; SAP amendment D-8). An untimed
        # death — one with no `finaldischarge_hrs` — is recovered when the CALENDAR-
        # DAY index says day <= 2. The day field is an INDEX, not elapsed time (its
        # day-3 bucket spans 24.25-71.17 h), so this recovers deaths the hour rule
        # cannot see WITHOUT claiming hour-level precision for them. It is the first
        # interior rung between the primary (4,191) and the upper bound (4,478).
        return (_comp((df["death_hemorrhagic_dayfallback"] == 1), tx_primary),
                f"{HC_COL} | death_hemorrhagic_dayfallback | {TX_COL}  "
                "(rung 2: PRIMARY plus an UNTIMED death recovered when "
                "finaldischarge_days<=2 — calendar-day fallback for the 429 deaths "
                "with no elapsed-hours value)")

    if variant == "sens_death_edincl":
        # RECOVERY LADDER RUNG 3 (CONTRACTS §3/§7; SAP amendment D-8). An untimed
        # death that occurred in the ED is recovered on a clinical, not statistical,
        # justification: an ED death is NECESSARILY within 48 h of arrival, so no
        # timing assumption is being made — the timing is implied by the location.
        return (_comp((df["death_hemorrhagic_edincl"] == 1), tx_primary),
                f"{HC_COL} | death_hemorrhagic_edincl | {TX_COL}  "
                "(rung 3: PRIMARY plus an UNTIMED ED death — an ED death is "
                "necessarily within 48 h, so the recovery makes no timing assumption)")

    if variant == "sens_death_le24h":
        # Prefer TRUE ELAPSED HOURS when `01` provides them. The day field is a
        # calendar INDEX (its day-3 bucket spans 24.25-71.17 h), so `days <= 1` is
        # only an approximation of "within 24 h" and is used solely as a fallback.
        if TIMING_HRS_COL in df.columns:
            tcut = pd.to_numeric(df[TIMING_HRS_COL], errors="coerce") <= 24
            tdesc = f"{TIMING_HRS_COL}<=24 (true elapsed hours)"
        elif TIMING_DAYS_COL in df.columns:
            tcut = pd.to_numeric(df[TIMING_DAYS_COL], errors="coerce") <= 1
            tdesc = (f"{TIMING_DAYS_COL}<=1 (calendar-day FALLBACK — the day field is an "
                     f"index, not elapsed time)")
        else:
            raise KeyError("sens_death_le24h needs finaldischarge_hrs or finaldischarge_days")
        d24 = ((df["death_inhosp"] == 1)
               & (pd.to_numeric(df["ais_head"], errors="coerce") < 4) & tcut)
        return (_comp(d24, tx_primary),
                f"{HC_COL} | (death_inhosp & ais_head<4 & {tdesc}) | {TX_COL}  "
                "(stricter <24 h death window)")

    if variant == "sens_death_dayrule":
        # Hour-rule vs day-rule crosswalk on the PRIMARY death component. The
        # crosswalk is itself a reportable robustness check: it isolates what the
        # calendar-index-to-elapsed-hours correction changed, and nothing else.
        return (_comp((df["death_hemorrhagic_dayrule"] == 1), tx_primary),
                f"{HC_COL} | death_hemorrhagic_dayrule | {TX_COL}  "
                "(SUPERSEDED calendar-day rule: death_inhosp & ais_head<4 & "
                "finaldischarge_days<=2, retained as the hour-vs-day crosswalk)")

    if variant == "sens_death_le48h_dayrule":
        return (_comp((df["death_le48h_dayrule"] == 1), tx_primary),
                f"{HC_COL} | death_le48h_dayrule | {TX_COL}  "
                "(rung 1b under the SUPERSEDED calendar-day rule; pairs with "
                "sens_death_le48h for the hour-vs-day crosswalk)")

    if variant == "sens_death_ed":
        return (_comp((df["death_ed"] == 1), tx_primary),
                f"{HC_COL} | death_ed | {TX_COL}  "
                "(ED-death surrogate, ED_DISPOSITION_CODE=='MOR' — earliest death available)")

    if variant == "sens_any_tx":
        return (_comp(death_primary, (pd.to_numeric(blood, errors="coerce") > 0).astype(float)),
                f"{HC_COL} | {DEATH_COL} | blood_4h > 0  (ANY documented transfusion)")

    if variant == "sens_tx_2000":
        return (_comp(death_primary,
                      _tx_units_or_ml(blood, BLOOD_ML_MIN_MASSIVE, BLOOD_UNITS_MIN_MASSIVE)),
                f"{HC_COL} | {DEATH_COL} | (blood_4h >= {BLOOD_ML_MIN_MASSIVE} mL OR "
                f"{BLOOD_UNITS_MIN_MASSIVE} <= blood_4h <= {BLOOD_UNITS_MAX} U)  "
                "(unit-aware twin of the v2 >=2000 mL arm — registered deviation D-2)")

    raise KeyError(f"unknown sensitivity variant '{variant}'")


# Arms that swap the OUTCOME on the ring-only cohort. Ordered as they appear in
# CONTRACTS §7, with the hour-vs-day crosswalk arms appended. (cohort/score-
# structure arms are handled separately in main().)
#
# `requires` names the cohort columns an arm needs BEYOND the always-present core.
# The menu is filtered against the columns `01` actually emitted, so a schema
# change upstream removes an arm with a logged note instead of crashing the run.
OUTCOME_SWAP_SPEC: list[tuple[str, str, tuple[str, ...]]] = [
    ("sens_v2def_outcome", "main", ("major_hemorrhage_event_v2def",)),
    ("sens_tx_ml_only", "main", ("substantial_transfusion_ml_only",)),
    ("sens_death_allcause", "main", ("death_inhosp",)),
    ("sens_death_headais", "main", ("death_hemorrhagic_headais",)),
    ("sens_death_le48h", "main", ("death_le48h",)),
    ("sens_death_missing_incl", "main", ("death_hemorrhagic_missing_incl",)),
    # Recovery-ladder rungs 2 and 3 — the two INTERIOR rungs (CONTRACTS §7, SAP D-8).
    ("sens_death_dayfallback", "main", ("death_hemorrhagic_dayfallback",)),
    ("sens_death_edincl", "main", ("death_hemorrhagic_edincl",)),
    ("sens_death_le24h", "main", ()),          # needs a timing col; checked below
    ("sens_death_ed", "main", ("death_ed",)),
    ("sens_any_tx", "main", ("blood_4h",)),
    ("sens_tx_2000", "main", ("blood_4h",)),
    # hour-rule vs day-rule crosswalk (only if `01` kept the day-rule variants)
    ("sens_death_dayrule", "main", ("death_hemorrhagic_dayrule",)),
    ("sens_death_le48h_dayrule", "main", ("death_le48h_dayrule",)),
]


def outcome_swap_variants(cols) -> list[tuple[str, str]]:
    """Filter the outcome-swap menu to the arms buildable from `cols`."""
    have = set(cols)
    out: list[tuple[str, str]] = []
    for variant, arm, needs in OUTCOME_SWAP_SPEC:
        if variant == "sens_death_le24h":
            ok = bool({TIMING_HRS_COL, TIMING_DAYS_COL} & have)
            missing: list[str] = [] if ok else [f"{TIMING_HRS_COL} or {TIMING_DAYS_COL}"]
        else:
            missing = [c for c in needs if c not in have]
        if missing:
            LOGGER.warning("sensitivity arm '%s' DROPPED — cohort lacks %s", variant, missing)
        else:
            out.append((variant, arm))
    return out


# ---------------------------------------------------------------------------
# §7-COVERAGE GATE, PART 1 OF 2 — STATIC: does the CODE implement the CONTRACT?
# ---------------------------------------------------------------------------
# Arms with a bespoke code path in main() rather than an OUTCOME_SWAP_SPEC row.
# Hand-maintained against main() — deliberately NOT derived from CONTRACTS_S7_
# VARIANTS, which would make the comparison below circular again.
STRUCTURAL_VARIANTS: tuple[str, ...] = (
    "sens_acetabulum",          # main(): run_composite_variant on the acetab cohort
    "sens_mi",                  # main(): run_variant_d_mi
    "sens_exclude_nslo",        # main(): run_score_variant(row_filter="exclude_nslo")
    "sens_with_ring",           # main(): run_score_variant(extra_vars=[RING_VAR])
    "ring_only_reference",      # main(): reference_rowgroup
    "primary_score_reference",  # main(): run_score_variant reference row
)


def implemented_variant_names() -> set[str]:
    """Every variant name this SCRIPT is capable of emitting, assembled from the
    code's own structures. This is the CODE side of the coverage comparison; the
    CONTRACT side is CONTRACTS_S7_VARIANTS. The two must have independent
    provenance or the gate is a tautology (CONTRACTS §13 preamble, rule 3)."""
    return ({v for v, _, _ in OUTCOME_SWAP_SPEC}
            | set(COMPONENT_VARIANT.values())
            | set(STRUCTURAL_VARIANTS))


def gate_s7_static_coverage(diag: list[str]) -> dict[str, Any]:
    """GATE S7-STATIC. Fail BEFORE any compute if a CONTRACTS §7 arm has no
    implementation at all. Runs in the first second of main() rather than after
    the ~66-minute analysis, because "the contract names an arm this file has never
    heard of" is knowable without touching data.

    Falsifiable by construction: delete an arm from OUTCOME_SWAP_SPEC (or from
    STRUCTURAL_VARIANTS) and this raises. That is the property the old check
    lacked — its expectation moved with the code, so removing an arm removed the
    expectation along with it."""
    impl = implemented_variant_names()
    not_implemented = [v for v in CONTRACTS_S7_VARIANTS if v not in impl]
    # A name in the score-lens list that is not in the flat contract list is a
    # transcription slip in THIS file, and would otherwise pass silently.
    score_orphans = [v for v in CONTRACTS_S7_SCORE_LENS_VARIANTS
                     if v not in CONTRACTS_S7_VARIANTS]
    undeclared = sorted(impl - set(CONTRACTS_S7_VARIANTS) - set(EXTRA_DECLARED_VARIANTS))
    diag.append("\n=== GATE S7-STATIC — CONTRACTS §7 menu vs this script's implementation ===")
    diag.append(f"  contract names {len(CONTRACTS_S7_VARIANTS)} variants; this script "
                f"implements {len(impl)} ({len(CONTRACTS_S7_VARIANTS) - len(not_implemented)} "
                f"of the contract's + {len(EXTRA_DECLARED_VARIANTS)} declared extras)")
    if score_orphans:
        raise SystemExit(
            f"GATE S7-STATIC: FAIL — CONTRACTS_S7_SCORE_LENS_VARIANTS names "
            f"{score_orphans}, which are absent from CONTRACTS_S7_VARIANTS. One of the "
            f"two hand-transcribed lists is wrong; fix the transcription against "
            f"CONTRACTS.md §7 before running.")
    if not_implemented:
        diag.append(f"  [FAIL] contract arms with NO implementation: {not_implemented}")
        LOGGER.error("GATE S7-STATIC FAIL — CONTRACTS §7 names %s, which this script "
                     "cannot build.", not_implemented)
        raise SystemExit(
            f"GATE S7-STATIC: FAIL — CONTRACTS §7 names sensitivity arm(s) this script "
            f"does not implement: {not_implemented}.\n"
            f"  Nothing was computed. Either implement the arm(s) (OUTCOME_SWAP_SPEC + "
            f"build_variant_outcome, or a bespoke path in main() plus "
            f"STRUCTURAL_VARIANTS), or — if an arm genuinely cannot be built — register "
            f"it as an explicit DEVIATION in the manuscript's Methods and remove it "
            f"from CONTRACTS §7 by amendment. Silently dropping it is the failure mode "
            f"this gate exists to prevent (CONTRACTS §13 preamble).")
    diag.append("  [PASS] every CONTRACTS §7 variant has an implementation in this file.")
    if undeclared:
        diag.append(f"  [NOTE] implemented but not in §7 and not declared in "
                    f"EXTRA_DECLARED_VARIANTS: {undeclared}")
        LOGGER.warning("variants implemented but undeclared (neither §7 nor "
                       "EXTRA_DECLARED_VARIANTS): %s", undeclared)
    for v, why in EXTRA_DECLARED_VARIANTS.items():
        diag.append(f"  [extra, declared] {v}: {why}")
    return {"evaluated": True, "contract_n": len(CONTRACTS_S7_VARIANTS),
            "implemented_n": len(impl), "not_implemented": not_implemented,
            "undeclared_extras": undeclared, "verdict": "PASS"}


# ---------------------------------------------------------------------------
# ANALYSIS-A LENS — score refit + integer conversion + held-out AUROC per arm
# ---------------------------------------------------------------------------
# Mirrors 05_refit_integer.py's rule EXACTLY (|beta|<BETA_DROP drop with force-in
# anchors immune -> anchor-protected VIF loop -> round(beta/|beta_min_pos|) with a
# +/-1 floor over POINT-ROLE vars only) and 06's scoring. Selection is NOT re-run:
# LASSO + stability selection belong to 04 and re-running them per arm would
# confound the outcome change with selection noise. The one arm that changes the
# variable list is `sens_with_ring`, and it says so in its `notes`.
#
# Fix 9 / D-11 (CONTRACTS §5.3, §13 G8) — THE ANCHOR RULE IS AMENDED, and these
# arms mirror the amendment. This is a DEVIATION from the locked SAP's literal
# integer-conversion rule, not a bugfix.
#
#   LOCKED SAP: round(beta/|beta_min_pos|) with a +/-1 floor, force-in anchors
#   immune to the |beta| >= 0.15 screen. Read literally, |beta_min_pos| ranges
#   over EVERY retained point-role variable, including an anchor retained only
#   because it was pre-registered.
#
#   WHAT IT PRODUCED: under the v3 outcome the force-in anchor `cc_anticoag`
#   collapsed to beta = 0.062 (OR 1.06, 95% CI 0.976-1.159, non-significant) —
#   restricting the death arm to early hemorrhagic death removed the late deaths
#   in which anticoagulation mattered. As the denominator it divided every other
#   coefficient by 0.062: mech_fall -27, sbp_lt90 +18, gcs_motor_le5 +15,
#   observed range -35..+96 (v2 was -7..+32). That defeats Fix 4 / D-4, whose
#   entire purpose was the objection that +16 is implausible at bedside.
#
#   AMENDED RULE (identical in 05, 12 and 13):
#       anchor = min{ beta_v : v in point_vars, beta_v > 0, |beta_v| >= BETA_DROP }
#
#   JUSTIFICATION: a coefficient retained by pre-registration rather than by
#   evidence should not set the unit of the scale. The screen exists precisely to
#   identify coefficients large enough to be meaningful; the denominator should
#   come from that set. The force-in anchor is still retained and reported — it
#   simply no longer defines the unit, and may legitimately score 0 points.
#
# EVERY ARM HERE RE-DERIVES ITS OWN ANCHOR, because every arm refits on a
# different outcome and therefore has different coefficients. It cannot read 05's
# anchor VALUE. What it does read from refit.json is the RULE (asserted equal to
# this script's literal) and the BETA_DROP cutoff. Each arm's anchor variable and
# beta are RECORDED in `notes` and in the diagnostic log, and compared against
# refit.json's anchor, so a divergence is visible rather than silent (§13 G8e).

def select_integer_anchor(beta_by_var, point_vars, beta_drop, force_in_anchors=(),
                          emit=None):
    """|beta_min_positive| under CONTRACTS §5.3 (Fix 9 / D-11). AMENDED RULE.

        anchor = min{ beta_v : v in point_vars, beta_v > 0, |beta_v| >= beta_drop }

    Copy-adapted from 05_refit_integer.py (house style: no cross-script imports);
    12_temporal_external_validation.py carries the same function. All three must
    stay behaviourally identical — the rule string is asserted against
    refit.json's in load_score_inputs() (§13 G8e).

    The divisor is the smallest positive coefficient among POINT-ROLE variables
    that passed the pre-specified |beta| >= beta_drop screen ON THEIR OWN
    EVIDENCE. A variable retained *solely* by force-in and sitting below the
    screen is excluded FROM THE ANCHOR SEARCH ONLY — it stays in the model, stays
    on the card, and may legitimately score 0 points.

    DEGENERATE FALLBACK: if no point-role variable clears the screen, fall back to
    the locked rule (min positive beta over all point-role vars) with a loud
    warning rather than crashing. That matters more here than anywhere else: a
    sensitivity arm on a rare outcome is exactly where every coefficient can end
    up below the screen, and an exception would take out the arm.

    Returns (anchor_beta, anchor_var, meta).
    """
    beta_drop = float(beta_drop)
    force_in_anchors = set(force_in_anchors or ())
    say = emit if emit is not None else (lambda m: LOGGER.warning("%s", m))
    # Insertion order follows `point_vars` so ties break identically in 05/12/13.
    positive = {v: float(beta_by_var[v]) for v in point_vars
                if float(beta_by_var[v]) > 0}
    if not positive:
        raise RuntimeError("no positive-beta point-role variable")

    eligible = {v: b for v, b in positive.items() if abs(b) >= beta_drop}
    excluded = []
    for v in point_vars:
        b = float(beta_by_var[v])
        if abs(b) >= beta_drop:
            continue
        is_anchor = v in force_in_anchors
        excluded.append({
            "variable": v,
            "beta": round(b, 6),
            "abs_beta": round(abs(b), 6),
            "positive": bool(b > 0),
            "force_in_anchor": bool(is_anchor),
            "reason": (
                f"|beta| = {abs(b):.4f} < screen {beta_drop} — "
                + ("retained SOLELY by force-in pre-registration, so it is not "
                   "eligible to set the integer scale"
                   if is_anchor else
                   "below the screen, not eligible to set the integer scale")
                + " (CONTRACTS §5.3 / D-11). Still retained, still reported; may "
                  "score 0 points."),
        })

    fallback = False
    if eligible:
        anchor_var = min(eligible, key=eligible.get)
        anchor = float(eligible[anchor_var])
    else:
        fallback = True
        anchor_var = min(positive, key=positive.get)
        anchor = float(positive[anchor_var])
        say(f"[D-11 DEGENERATE FALLBACK] no point-role variable passed the "
            f"|beta| >= {beta_drop} screen on this arm; the amended anchor rule "
            f"(CONTRACTS §5.3) has an EMPTY eligible set, so this card falls back "
            f"to the LOCKED SAP rule. Fallback anchor = {anchor:.4f} "
            f"({anchor_var}) — every point value on this arm's card is scaled by a "
            f"coefficient that did NOT clear the screen.")

    meta = {
        "anchor_selection_rule": ANCHOR_SELECTION_RULE,
        "anchor_min_positive_var": anchor_var,
        "anchor_min_positive_beta": anchor,
        "anchor_eligible_vars": [v for v in point_vars if v in eligible],
        "anchor_excluded_forced_in": excluded,
        "anchor_screen_cutoff": beta_drop,
        "anchor_n_positive_point_vars": len(positive),
        "anchor_n_eligible": len(eligible),
        "anchor_degenerate_fallback": bool(fallback),
        "anchor_locked_sap_would_have_chosen": {
            "variable": min(positive, key=positive.get),
            "beta": round(float(min(positive.values())), 6),
        },
    }

    # ---- DEFINITIONAL INVARIANTS — NOT verification (CONTRACTS §13 rule 3) ----
    # `anchor_var` is `min()` over `eligible`/`positive`, which were built by
    # filtering `point_vars` on `b > 0` and `abs(b) >= beta_drop` fifteen lines
    # above. So all three of these are TRUE BY CONSTRUCTION and cannot fail as
    # written. They are retained as cheap tripwires against a future refactor of
    # the filtering block, and are labelled here so nobody counts them as evidence
    # that §13 G8a passed — G8a's falsifiable form lives in `05`, where the anchor
    # in `refit.json` is checked against the frozen cohort's expected value.
    assert anchor_var in point_vars, \
        f"anchor {anchor_var!r} is not a point-role variable (CONTRACTS §5.1a)"
    assert anchor > 0, f"anchor beta must be positive; got {anchor}"
    if not fallback:
        assert abs(anchor) >= beta_drop, (
            f"§13 G8a VIOLATION: anchor {anchor_var!r} has |beta| = {abs(anchor):.4f} "
            f"< screen {beta_drop}. A variable retained solely by force-in must "
            f"not set the integer scale (CONTRACTS §5.3 / D-11).")

    # ---- FALSIFIABLE CROSS-CHECK — recomputed from the RAW INPUTS -------------
    # Independent one-expression re-derivation of the amended rule straight from
    # (beta_by_var, point_vars, beta_drop), bypassing `positive`/`eligible`/
    # `excluded` entirely. It disagrees with the block above if that block's
    # filtering is ever edited wrongly — which the invariants above, being
    # downstream of the same filtering, cannot detect. This one CAN fail.
    _cand = {v: float(beta_by_var[v]) for v in point_vars
             if float(beta_by_var[v]) > 0
             and (fallback or abs(float(beta_by_var[v])) >= beta_drop)}
    _expect_var = min(_cand, key=_cand.get)
    if _expect_var != anchor_var or float(_cand[_expect_var]) != anchor:
        raise AssertionError(
            f"§13 G8a/G8e VIOLATION: the anchor-selection block returned "
            f"{anchor_var!r} (beta={anchor:.6f}), but re-deriving the amended rule "
            f"directly from the inputs gives {_expect_var!r} "
            f"(beta={_cand[_expect_var]:.6f}). The rule is duplicated in 05/12/13 and "
            f"must be behaviourally identical in all three; a divergence here means "
            f"every sensitivity arm's card is on a different scale from the headline.")
    return anchor, anchor_var, meta


def load_score_inputs(sample: int | None) -> dict[str, Any] | None:
    """Read the 02/03/05 artifacts the score arms need. Returns None if absent."""
    missing = [p for p in (CANDIDATES_PARQUET, TRAIN_KEYS_CSV, TEST_KEYS_CSV, REFIT_JSON)
               if not os.path.exists(p)]
    if missing:
        LOGGER.error("Score-arm inputs missing: %s. The Analysis-A lens (score AUROC, "
                     "incl. sens_v2def_outcome / sens_exclude_nslo / sens_with_ring) "
                     "CANNOT be evaluated and is therefore a FAIL, not a skip "
                     "(CONTRACTS §13). Run 02/03/05 first.", missing)
        return None

    with open(REFIT_JSON) as f:
        refit = json.load(f)
    vars_kept = list(refit["variables_kept"])
    if RING_VAR in vars_kept:
        raise SystemExit(
            f"CONTRACT VIOLATION (Fix 3 / §13 G3c): '{RING_VAR}' is present in "
            f"refit.json['variables_kept']. It is removed from the SCORE in v3. "
            f"Re-run 04/05 on the 17-variable pool.")

    if "point_vars" in refit and "override_vars" in refit:
        point_vars, override_vars = list(refit["point_vars"]), list(refit["override_vars"])
        role_src = "refit.json"
    else:
        override_vars = [v for v in DEFAULT_OVERRIDE_VARS if v in vars_kept]
        point_vars = [v for v in vars_kept if v not in override_vars]
        role_src = "DERIVED (refit.json predates Fix 4)"
        LOGGER.warning("refit.json lacks 'point_vars'/'override_vars' (CONTRACTS §5.1(d)); "
                       "deriving override=%s. Fallback, not the contract.", override_vars)
    assert not (set(point_vars) & set(override_vars)), \
        "CONTRACT VIOLATION (§5.1(e)): point_vars and override_vars overlap."

    df = pd.read_parquet(CANDIDATES_PARQUET)
    df = df[df["cohort_primary"] == 1].copy()
    for k in ("puf_year", "inc_key"):
        df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
    tr = pd.read_csv(TRAIN_KEYS_CSV)
    te = pd.read_csv(TEST_KEYS_CSV)
    for frame in (tr, te):
        for k in ("puf_year", "inc_key"):
            frame[k] = pd.to_numeric(frame[k], errors="coerce").astype("Int64")
    train = df.merge(tr[["puf_year", "inc_key"]], on=["puf_year", "inc_key"], how="inner")
    test = df.merge(te[["puf_year", "inc_key"]], on=["puf_year", "inc_key"], how="inner")
    if len(train) != len(tr) or len(test) != len(te):
        raise SystemExit(f"train/test key join drift: train {len(train)}/{len(tr)}, "
                         f"test {len(test)}/{len(te)}")
    if sample:
        # Downsample BOTH partitions so the smoke run exercises the same code path.
        if sample < len(train):
            train = train.sample(n=int(sample), random_state=RANDOM_SEED).reset_index(drop=True)
        if sample < len(test):
            test = test.sample(n=int(sample), random_state=RANDOM_SEED).reset_index(drop=True)

    lasso_pool = list(refit.get("lasso_x_cols", []))
    force_spec = list(refit.get("force_in_spec", FORCE_IN_ANCHORS))

    # ---- Fix 9 / D-11 — READ the cutoff and the anchor RULE from 05 ---------
    # Read, do not re-declare: a re-derivation that agrees today can diverge
    # tomorrow, a read cannot. The BETA_DROP module constant is now a fallback for
    # a pre-Fix-4 artifact only, never the source of truth (§13 G8e).
    raw_cut = refit.get("beta_cutoff")
    try:
        beta_drop = float(raw_cut)
        if not np.isfinite(beta_drop) or beta_drop < 0:
            raise ValueError
        cut_src = "refit.json['beta_cutoff']"
    except (TypeError, ValueError):
        beta_drop = float(BETA_DROP)
        cut_src = f"FALLBACK literal {BETA_DROP}"
        LOGGER.warning("refit.json carries no usable 'beta_cutoff' (got %r); using %s "
                       "so the beta screen and the ±1 integer floor still match "
                       "05_refit_integer.py. Fix 05 if this persists.", raw_cut, beta_drop)

    recorded_rule = refit.get("anchor_selection_rule")
    if recorded_rule is None:
        LOGGER.warning("refit.json carries no 'anchor_selection_rule' (CONTRACTS §5.3 / "
                       "§13 G8b) — that artifact predates D-11. These arms apply the "
                       "AMENDED rule regardless; re-run 05_refit_integer.py so the two "
                       "are provably the same rule.")
    elif str(recorded_rule) != ANCHOR_SELECTION_RULE:
        raise SystemExit(
            "CONTRACT VIOLATION (§13 G8e): the integer-anchor rule recorded by "
            "05_refit_integer.py differs from this script's implementation. The rule "
            "is duplicated in 05/12/13 and MUST be identical, or every sensitivity "
            "arm reports a differently-scaled card while the headline table looks "
            f"correct.\n  refit.json: {recorded_rule!r}\n  13 (this):  "
            f"{ANCHOR_SELECTION_RULE!r}")
    refit_anchor = {"var": refit.get("anchor_min_positive_var"),
                    "beta": refit.get("anchor_min_positive_beta"),
                    "rule": recorded_rule}

    LOGGER.info("score arms: train N=%d, test N=%d, spec=%d vars (point=%d, override=%d; "
                "roles from %s)", len(train), len(test), len(vars_kept),
                len(point_vars), len(override_vars), role_src)
    LOGGER.info("D-11 anchor rule READ from refit.json (%s); beta cutoff %.4f [%s]; "
                "05's anchor = %s (beta=%s)",
                "present" if recorded_rule else "ABSENT — stale artifact",
                beta_drop, cut_src, refit_anchor["var"], refit_anchor["beta"])
    return {"train": train, "test": test, "vars_kept": vars_kept,
            "point_vars": point_vars, "override_vars": override_vars,
            "lasso_pool": lasso_pool, "force_spec": force_spec, "refit": refit,
            "role_src": role_src, "beta_drop": beta_drop, "beta_drop_src": cut_src,
            "refit_anchor": refit_anchor}


def _score_bootstrap_auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int,
                            seed: int = RANDOM_SEED) -> tuple[float, float]:
    if n_boot <= 0 or y.sum() == 0 or y.sum() == len(y):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        out[b] = np.nan if (yb.sum() == 0 or yb.sum() == n) else roc_auc_score(yb, s[idx])
    out = out[np.isfinite(out)]
    if out.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def _fit_logit_score(X: pd.DataFrame, y: np.ndarray):
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    return sm.Logit(y, Xc).fit(disp=False, maxiter=200, method="newton")


def _score_vifs(X: pd.DataFrame) -> dict[str, float]:
    Xc = sm.add_constant(X, has_constant="add").astype(float).values
    cols = ["const"] + list(X.columns)
    out = {}
    for i, nm in enumerate(cols):
        if nm == "const":
            continue
        try:
            out[nm] = float(variance_inflation_factor(Xc, i))
        except Exception:  # noqa: BLE001
            out[nm] = float("nan")
    return out


def run_score_variant(inp: dict[str, Any], variant: str, arm: str, cohort_def: str,
                      n_boot: int, diag: list[str], notes: str = "",
                      extra_vars: Sequence[str] = (), extra_anchors: Sequence[str] = (),
                      row_filter: str | None = None) -> list[dict[str, Any]]:
    """Refit the ring-free v3 score on TRAIN against `variant`'s outcome, convert to
    integer points, and score the held-out TEST slice. Returns 2 rows: the integer
    card's AUROC and the continuous linear predictor's AUROC."""
    train, test = inp["train"], inp["test"]
    if row_filter == "exclude_nslo":
        # CONTRACTS §5.1 / §7 `sens_exclude_nslo`: drop the rows entirely.
        for nm, fr in (("train", train), ("test", test)):
            if "sbp_unrecordable" not in fr.columns:
                raise SystemExit(f"sens_exclude_nslo needs 'sbp_unrecordable' in the "
                                 f"{nm} frame; candidates.parquet does not carry it.")
        n_drop_tr = int((train["sbp_unrecordable"] == 1).sum())
        n_drop_te = int((test["sbp_unrecordable"] == 1).sum())
        train = train[train["sbp_unrecordable"] != 1].copy()
        test = test[test["sbp_unrecordable"] != 1].copy()
        diag.append(f"    [{variant}] dropped no-signs-of-life rows: train {n_drop_tr}, "
                    f"test {n_drop_te} (total {n_drop_tr + n_drop_te}; cohort-wide n=608)")

    y_tr, def_tr = build_variant_outcome(train, variant)
    y_te, def_te = build_variant_outcome(test, variant)
    y_tr = y_tr.astype(int).to_numpy()
    y_te = y_te.astype(int).to_numpy()

    variables = list(dict.fromkeys(list(inp["vars_kept"]) + list(extra_vars)))
    anchors = list(dict.fromkeys(list(inp["force_spec"]) + list(extra_anchors)))
    override_vars = [v for v in inp["override_vars"] if v in variables]

    absent = [v for v in variables if v not in train.columns]
    if absent:
        raise SystemExit(f"[{variant}] score variables absent from candidates.parquet: {absent}")

    # ---- SINGULARITY GUARD ------------------------------------------------
    # Filtering rows can make a column constant (the canonical case: once
    # `sbp_unrecordable == 1` rows are dropped, that column is all-zero and the
    # unpenalized refit goes singular). Dropping the ROWS is not enough — the
    # COLUMN has to leave the design matrix too. Handle it explicitly and say so.
    degenerate = [v for v in variables if train[v].astype(float).nunique(dropna=False) < 2]
    if degenerate:
        variables = [v for v in variables if v not in degenerate]
        anchors = [a for a in anchors if a not in degenerate]
        override_vars = [v for v in override_vars if v not in degenerate]
        diag.append(f"    [{variant}] SINGULARITY GUARD: dropped zero-variance column(s) "
                    f"{degenerate} from the design matrix (constant after the row filter); "
                    f"the unpenalized refit would otherwise be singular.")
        LOGGER.info("[%s] singularity guard dropped %s", variant, degenerate)

    # BETA_DROP is READ from refit.json (05 owns it) so the screen and the ±1
    # floor cannot drift from 05's; the module constant is a fallback only.
    beta_drop = float(inp.get("beta_drop", BETA_DROP))

    dropped_beta: list[str] = []
    dropped_vif: list[str] = []
    anchor_meta: dict[str, Any] = {}
    anchor_beta = float("nan")
    anchor_var = ""
    try:
        res = _fit_logit_score(train[variables], y_tr)
        beta = res.params.drop("const")
        weak = [v for v in beta[beta.abs() < beta_drop].index.tolist() if v not in anchors]
        dropped_beta = list(weak)
        variables = [v for v in variables if v not in weak]
        for a in anchors:                      # force-in re-insert (belt & suspenders)
            if a not in variables and a in train.columns \
                    and train[a].astype(float).nunique(dropna=False) >= 2:
                variables.append(a)
        res = _fit_logit_score(train[variables], y_tr)

        for _ in range(20):                    # anchor-protected VIF loop
            vifs = _score_vifs(train[variables])
            droppable = {v: q for v, q in vifs.items() if v not in anchors}
            if not droppable:
                break
            mx = max(droppable, key=lambda k: droppable[k])
            if droppable[mx] <= SCORE_VIF_MAX:
                break
            dropped_vif.append(mx)
            variables = [v for v in variables if v != mx]
            if len(variables) < SCORE_MIN_KEPT:
                raise RuntimeError(f"[{variant}] VIF loop fell below MIN_KEPT")
            res = _fit_logit_score(train[variables], y_tr)

        params = res.params
        override_vars = [v for v in override_vars if v in variables]
        point_vars = [v for v in variables if v not in override_vars]
        # ---- Fix 9 / D-11 anchor (CONTRACTS §5.3) ----
        # The divisor is the smallest positive POINT-ROLE beta that PASSED the
        # |beta| >= beta_drop screen. A force-in anchor sitting below the screen is
        # excluded from the SEARCH only — it keeps its coefficient, stays on this
        # arm's card, and may score 0. Registered deviation, not a bugfix.
        beta_by_var = {v: float(params.loc[v]) for v in variables}
        anchor_beta, anchor_var, anchor_meta = select_integer_anchor(
            beta_by_var, point_vars, beta_drop, force_in_anchors=anchors,
            emit=lambda m, _v=variant: LOGGER.warning("[%s] %s", _v, m))
        points: dict[str, int] = {}
        for v in point_vars:
            b = float(params.loc[v])
            p = int(round(b / anchor_beta))
            # ±1 floor — UNCHANGED by D-11, same exact condition as 05. A variable
            # BELOW the screen (only a force-in anchor can be here) correctly does
            # NOT get floored: it keeps 0. Do not special-case it up to ±1.
            if p == 0 and abs(b) >= beta_drop:
                p = 1 if b > 0 else -1
            points[v] = p
        for v in override_vars:
            points[v] = 0              # Fix 4: in the LP, never on the card

        # §13 G8c — card-scale tripwire (reported, never raised: see the constant).
        _oversized = {v: p for v, p in points.items()
                      if v in point_vars and abs(int(p)) > CARD_SCALE_TRIPWIRE}
        if _oversized:
            diag.append(f"    [{variant}] [G8c][WARN] point value(s) beyond "
                        f"±{CARD_SCALE_TRIPWIRE}: {_oversized} — fingerprint of a "
                        f"COLLAPSED ANCHOR; inspect anchor={anchor_var} "
                        f"(beta={anchor_beta:.4f}) before reading this arm's card.")
        _zero = sorted(v for v in point_vars if int(points[v]) == 0)
        if _zero:
            _fi = sorted(set(_zero) & set(anchors))
            diag.append(f"    [{variant}] point-role variable(s) at 0 points: {_zero}"
                        + (f"  *** incl. pre-registered force-in anchor(s) {_fi} — "
                           f"retained and reported at 0, never dropped "
                           f"(CONTRACTS §5.3) ***" if _fi else ""))
        # §13 G8e — record this arm's anchor and compare against 05's. Every arm
        # refits on a different outcome, so a DIFFERENT anchor variable does not by
        # itself mean the rule diverged; the rule string is the hard gate (asserted
        # at load). The discrepancy is made visible, never silent.
        _ra = (inp.get("refit_anchor") or {})
        if _ra.get("var") and _ra["var"] != anchor_var:
            diag.append(f"    [{variant}] [G8e][NOTE] anchor differs from the primary "
                        f"card: 05 used {_ra['var']} (beta={_ra['beta']}), this arm "
                        f"uses {anchor_var} (beta={anchor_beta:.4f}). Same RULE, "
                        f"different outcome/rows — do not compare point values "
                        f"across arms without saying so.")
        if anchor_meta.get("anchor_excluded_forced_in"):
            _ex = ", ".join(f"{e['variable']}(beta={e['beta']:+.4f}"
                            + (", force-in" if e["force_in_anchor"] else "") + ")"
                            for e in anchor_meta["anchor_excluded_forced_in"])
            diag.append(f"    [{variant}] D-11 excluded from the anchor search "
                        f"(|beta| < {beta_drop}): {_ex}")

        score_te = np.zeros(len(test), dtype=np.float64)
        for v, p in points.items():
            if p:
                score_te += p * test[v].astype(np.float64).to_numpy()
        lp_te = np.full(len(test), float(params.loc["const"]), dtype=np.float64)
        for v in variables:
            lp_te += float(params.loc[v]) * test[v].astype(np.float64).to_numpy()

        auc_int = float(roc_auc_score(y_te, score_te))
        lo_i, hi_i = _score_bootstrap_auc_ci(y_te, score_te, n_boot)
        auc_lp = float(roc_auc_score(y_te, lp_te))
        lo_l, hi_l = _score_bootstrap_auc_ci(y_te, lp_te, n_boot)
        ok = True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[%s] score arm FAILED: %s", variant, exc)
        diag.append(f"    [{variant}] SCORE ARM FAILED: {exc}")
        auc_int = auc_lp = lo_i = hi_i = lo_l = hi_l = float("nan")
        points, ok = {}, False

    n_te, ev_te = int(len(test)), int(y_te.sum())
    rate = 100.0 * float(y_te.mean()) if n_te else float("nan")
    base = {
        "variant": variant, "arm": arm, "cohort_def": cohort_def,
        "n": n_te, "n_events": ev_te, "composite_rate_pct": round(rate, 4),
        "low_event_flag": _low_event_flag(ev_te),
        "descriptive_only": 0, "guard_reason": "",
        "exposure": "pelvic_score", "outcome_def": def_te,
        "n_score_vars": (len(points) if ok else np.nan),
        "score_vars": ("|".join(f"{v}:{p:+d}" for v, p in points.items()) if ok else ""),
    }
    # §13 G8e — the anchor this arm actually used, carried into the TABLE (not
    # only the log), so a reader can tell at a glance whether two arms' point
    # values are on the same scale.
    if ok and anchor_var:
        _excl = [e["variable"] for e in anchor_meta.get("anchor_excluded_forced_in", [])]
        anchor_note = (f"D-11 anchor={anchor_var} (beta={anchor_beta:.4f}); "
                       f"screen |beta|>={beta_drop}"
                       + (f"; excluded from the anchor search: {_excl}" if _excl else "")
                       + ("; DEGENERATE FALLBACK to the locked rule"
                          if anchor_meta.get("anchor_degenerate_fallback") else ""))
    else:
        anchor_note = "D-11 anchor=n/a (arm failed)"
    note_full = (f"{notes} | refit on TRAIN (N={len(train):,}, events={int(y_tr.sum()):,}); "
                 f"selection NOT re-run (variable set held at refit.json); "
                 f"beta-drop={dropped_beta or 'none'}; vif-drop={dropped_vif or 'none'}; "
                 f"override(0 pts)={override_vars or 'none'}; {anchor_note}; "
                 f"boot={n_boot}").strip(" |")

    rows: list[dict[str, Any]] = []
    r1 = _blank_row(); r1.update(base)
    r1.update({"analysis": "score_auroc", "term": "integer card (pelvic_score)",
               "auroc": round(auc_int, 4) if np.isfinite(auc_int) else np.nan,
               "auroc_lci": round(lo_i, 4) if np.isfinite(lo_i) else np.nan,
               "auroc_uci": round(hi_i, 4) if np.isfinite(hi_i) else np.nan,
               "notes": note_full})
    rows.append(r1)
    r2 = _blank_row(); r2.update(base)
    r2.update({"analysis": "score_auroc", "term": "continuous linear predictor",
               "auroc": round(auc_lp, 4) if np.isfinite(auc_lp) else np.nan,
               "auroc_lci": round(lo_l, 4) if np.isfinite(lo_l) else np.nan,
               "auroc_uci": round(hi_l, 4) if np.isfinite(hi_l) else np.nan,
               "notes": note_full})
    rows.append(r2)
    if ok:
        r3 = _blank_row(); r3.update(base)
        r3.update({"analysis": "score_points", "term": "integer points (refit on this arm)",
                   "notes": note_full})
        rows.append(r3)

    diag.append(f"    [{variant}] score AUROC integer={auc_int:.4f} [{lo_i:.4f},{hi_i:.4f}] "
                f"lp={auc_lp:.4f}  test N={n_te:,} events={ev_te:,} ({rate:.2f}%)  "
                f"vars={len(points)}")
    return rows


# ---------------------------------------------------------------------------
# Variant (d) — residual blood-missingness, MI arm (impute blood_4h congenially)
# ---------------------------------------------------------------------------
def _mi_d_prepare(df: pd.DataFrame) -> dict[str, Any]:
    """Prep the sens_mi MI ONCE (single trimmed copy). Congenial: blood_4h is
    imputed jointly with the exposure vitals + adjustment + the observed composite
    components (received_hc, death_hemorrhagic). Frames are generated one-at-a-time
    and discarded by the caller — the m imputations never coexist in RAM (OOM-safe)."""
    base = df.copy()
    for c in [x for x in CORE_ADJUSTMENT if x in CATEGORICAL_ADJ]:
        if base[c].isna().any():
            mm = base[c].mode(dropna=True)
            base[c] = base[c].fillna(mm.iloc[0] if len(mm) else "Unknown")
    num_cols = [c for c in MI_D_NUMERIC if c in base.columns]
    keep_extra = [c for c in CATEGORICAL_ADJ if c in base.columns]
    prep = {"num_matrix": base[num_cols].astype(float), "num_cols": num_cols,
            "cat_frame": base[keep_extra].copy(), "index": base.index}
    del base
    return prep


def _mi_d_impute_one(prep: dict[str, Any], i: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate ONE imputed frame (blood_4h + vitals imputed), derive the composite."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    imp = IterativeImputer(sample_posterior=True, random_state=seed + i, max_iter=5)
    filled = imp.fit_transform(prep["num_matrix"])
    fdf = pd.DataFrame(filled, columns=prep["num_cols"], index=prep["index"])
    for c in prep["num_cols"]:
        if c in MI_D_BINARY:
            fdf[c] = fdf[c].round().clip(0, 1)
        elif c in ("sbp2", "pulse2"):
            fdf[c] = fdf[c].clip(lower=1.0)
        elif c == "shock_index":
            fdf[c] = fdf[c].clip(lower=0.01, upper=SI_WINSOR_CAP)
        elif c == "blood_4h":
            fdf[c] = fdf[c].clip(lower=0.0)
    out = pd.concat([fdf, prep["cat_frame"]], axis=1)
    # Fix 1 — the imputed composite must use the UNIT-AWARE rule, not v2's mL-only
    # `blood_4h >= 1000`. Using the old rule here would make the MI arm answer a
    # question about a transfusion definition the paper no longer uses.
    subst = _tx_units_or_ml(out["blood_4h"], BLOOD_ML_MIN_SUBSTANTIAL,
                            BLOOD_UNITS_MIN_SUBSTANTIAL)
    # Fix 2/2b — death arm is death_hemorrhagic, carried through the imputation
    # model as an observed binary.
    out["_compvar"] = ((out[HC_COL] == 1) | (out[DEATH_COL] == 1) | (subst == 1)).astype(float)
    return out


def run_variant_d_mi(df: pd.DataFrame, adj_map: dict[str, list[str]], m: int,
                     within_var_sbp: float, within_var_si: float, seed: int,
                     diag: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Variant-d MI arm: impute blood_4h (m=20, one-at-a-time), derive the composite,
    Rubin-pool the change-point LOCATION (both exposures) + the 4 threshold ORs."""
    prep = _mi_d_prepare(df)
    cp_locs = {ex: [] for ex in EXPOSURES}
    comp_rates: list[float] = []
    thr_acc = {(ex, t): {"lor": [], "lor_var": [], "rd": [], "rd_var": []}
               for ex in EXPOSURES for t in PRESPEC_THRESHOLDS[ex]}
    for i in range(int(m)):
        fi = _mi_d_impute_one(prep, i, seed)
        comp_rates.append(100.0 * float(fi["_compvar"].mean()))
        for exposure in EXPOSURES:
            adj = adj_map[exposure]
            grid_full = GRID_SI if exposure == "shock_index" else GRID_SBP
            cc = fi.dropna(subset=["_compvar", exposure] + adj)
            x = cc[exposure].to_numpy(float)
            y = cc["_compvar"].to_numpy(float)
            g, _, _ = interior_grid(x, grid_full)
            if g.size:
                try:
                    Xadj = build_adj_design_ni(cc, adj)
                    r = changepoint_search(Xadj, x, y, g)
                    if np.isfinite(r["breakpoint"]):
                        cp_locs[exposure].append(r["breakpoint"])
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("variant-d MI cp %s i=%d failed: %s", exposure, i, exc)
            for t in PRESPEC_THRESHOLDS[exposure]:
                try:
                    gg = threshold_gcomp(cc, exposure, "_compvar", float(t), adj)
                    a = thr_acc[(exposure, t)]
                    a["lor"].append(np.log(gg["adj_or"]))
                    a["lor_var"].append(((np.log(gg["or_uci"]) - np.log(gg["or_lci"])) / (2 * Z)) ** 2)
                    a["rd"].append(gg["adj_risk_diff"])
                    a["rd_var"].append(((gg["rd_uci"] - gg["rd_lci"]) / (2 * Z)) ** 2)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("variant-d MI thr %s t=%s i=%d failed: %s", exposure, t, i, exc)
        del fi, cc

    def _pool_loc(locs, within):
        if not locs:
            return {"location": float("nan"), "m_ok": 0}
        mk = len(locs)
        theta = float(np.mean(locs))
        between = float(np.var(locs, ddof=1)) if mk > 1 else 0.0
        w = float(within) if np.isfinite(within) else 0.0
        se = float(np.sqrt(max(w + (1.0 + 1.0 / mk) * between, 0.0)))
        return {"location": round(theta, 4), "lci": round(theta - Z * se, 4),
                "uci": round(theta + Z * se, 4), "m_ok": mk,
                "between_var": round(between, 6), "within_var": round(w, 6),
                "per_imputation": [round(v, 4) for v in locs]}

    loc_pool = {"sbp2": _pool_loc(cp_locs["sbp2"], within_var_sbp),
                "shock_index": _pool_loc(cp_locs["shock_index"], within_var_si)}
    mean_rate = float(np.mean(comp_rates)) if comp_rates else float("nan")

    rows: list[dict[str, Any]] = []
    cohort_def = "cohort_primary==1; blood_4h MI m=%d (Rubin)" % m
    mi_outcome_def = (f"{HC_COL} | {DEATH_COL} | unit-aware substantial transfusion "
                      f"(blood_4h imputed; >= {BLOOD_ML_MIN_SUBSTANTIAL} mL OR "
                      f"{BLOOD_UNITS_MIN_SUBSTANTIAL}-{BLOOD_UNITS_MAX} U)")
    mean_events = (mean_rate / 100.0) * len(df) if np.isfinite(mean_rate) else np.nan
    for exposure in EXPOSURES:
        lp = loc_pool[exposure]
        cp = {"breakpoint": lp.get("location", float("nan")),
              "lci": lp.get("lci", float("nan")), "uci": lp.get("uci", float("nan")),
              "at_grid_boundary": np.nan}
        rows.append(_cp_row("sens_mi", "MI", cohort_def, int(len(df)),
                            (round(mean_events) if np.isfinite(mean_events) else np.nan),
                            mean_rate, exposure, mi_outcome_def, cp,
                            notes=f"Rubin-pooled location over m={lp.get('m_ok',0)}; "
                                  f"n_events is the mean across imputations"))
        for t in PRESPEC_THRESHOLDS[exposure]:
            a = thr_acc[(exposure, t)]
            if not a["lor"]:
                continue
            mk = len(a["lor"])
            lor_bar = float(np.mean(a["lor"]))
            w = float(np.mean(a["lor_var"]))
            b = float(np.var(a["lor"], ddof=1)) if mk > 1 else 0.0
            se = float(np.sqrt(max(w + (1.0 + 1.0 / mk) * b, 0.0)))
            g = {"adj_or": float(np.exp(lor_bar)), "or_lci": float(np.exp(lor_bar - Z * se)),
                 "or_uci": float(np.exp(lor_bar + Z * se)), "p_raw": np.nan}
            row = _thr_row("sens_mi", "MI", cohort_def, int(len(df)),
                           (round(mean_events) if np.isfinite(mean_events) else np.nan),
                           mean_rate, exposure, mi_outcome_def, t, g,
                           notes=f"Rubin-pooled OR over m={mk}")
            row["p_bonferroni"] = np.nan  # p not defined under Rubin here
            rows.append(row)
    diag.append(f"\n--- VARIANT sens_mi [MI m={m}] mean composite rate={mean_rate:.3f}% "
                f"SBP loc={loc_pool['sbp2'].get('location')} SI loc={loc_pool['shock_index'].get('location')} ---")
    return rows, {"loc_pool": loc_pool, "mean_composite_rate_pct": round(mean_rate, 4),
                  "per_imputation_rates_pct": [round(r, 4) for r in comp_rates]}


# ---------------------------------------------------------------------------
# Reference row-group (ring-only main composite, from script 11) for the table
# ---------------------------------------------------------------------------
def reference_rowgroup(ref: dict[str, Any], df: pd.DataFrame,
                       diag: list[str]) -> list[dict[str, Any]]:
    """Pull script 11's ring-only main-composite pre-specified threshold ORs +
    change-points into the sensitivity table as the head-to-head reference.

    The composite N / event count / rate are COMPUTED FROM THE COHORT, never
    hard-coded. v2's 12.95 % is superseded by the v3 outcome redefinition and
    stamping it into a v3 table would launder a v2 number (CONTRACTS §13 G5b).
    """
    rows: list[dict[str, Any]] = []
    cohort_def = "cohort_primary==1 (ring-only, script 11)"
    n_ref = int(len(df))
    ev_ref = int(pd.to_numeric(df[COMPOSITE_COL], errors="coerce").fillna(0).sum())
    rate_ref = 100.0 * ev_ref / n_ref if n_ref else float("nan")
    stale_note = " [REFERENCE IS A v2 STALE FALLBACK — DO NOT REPORT]" if ref.get("stale") else ""
    # change-points from the reference detail
    cp_map = {"shock_index": ref["detail"].get("si_composite", {}),
              "sbp2": ref["detail"].get("sbp_composite", {})}
    try:
        tor = pd.read_csv(REF_THRESHOLD_OR_CSV)
        tor = tor[(tor["outcome"] == COMPOSITE_COL) & (tor["analysis"] == "prespecified_threshold")]
    except Exception:  # noqa: BLE001
        tor = pd.DataFrame()
    for exposure in EXPOSURES:
        d = cp_map[exposure]
        cp = {"breakpoint": float(d.get("breakpoint", np.nan)),
              "lci": float(d.get("lci")) if d.get("lci") is not None else np.nan,
              "uci": float(d.get("uci")) if d.get("uci") is not None else np.nan,
              "at_grid_boundary": np.nan}
        rows.append(_cp_row("ring_only_reference", "reference", cohort_def, n_ref, ev_ref,
                            rate_ref, exposure,
                            f"{COMPOSITE_COL} = {HC_COL} | {DEATH_COL} | {TX_COL}", cp,
                            notes="script 11 headline composite change-point" + stale_note))
        if not tor.empty:
            sub = tor[tor["exposure"] == exposure]
            for _, r in sub.iterrows():
                row = _blank_row()
                row.update({
                    "variant": "ring_only_reference", "arm": "reference", "cohort_def": cohort_def,
                    "n": int(r["n"]), "n_events": ev_ref, "composite_rate_pct": round(rate_ref, 4),
                    "low_event_flag": _low_event_flag(ev_ref),
                    "exposure": exposure,
                    "outcome_def": f"{COMPOSITE_COL} = {HC_COL} | {DEATH_COL} | {TX_COL}",
                    "analysis": "prespecified_threshold", "term": r["term"],
                    "adj_or": round(float(r["adj_or"]), 4), "or_lci": round(float(r["lci"]), 4),
                    "or_uci": round(float(r["uci"]), 4), "p_raw": float(r["p_raw"]),
                    "p_bonferroni": float(r["p_bonferroni"]),
                    "notes": "script 11 table7_threshold_or (reference)",
                })
                rows.append(row)
    diag.append("\n=== DELIVERABLE 3: reference row-group (ring-only, script 11) loaded ===")
    diag.append(f"  reference composite: N={n_ref:,} events={ev_ref:,} ({rate_ref:.3f}%) "
                f"— computed from the cohort, not hard-coded{stale_note}")
    return rows


def component_rowgroup(component_cps: dict[str, Any], df: pd.DataFrame,
                       diag: list[str]) -> list[dict[str, Any]]:
    """Emit the component-only arms (`sens_component_hc` / `_death` / `_tx`) into the
    sensitivity table. The change-points are already computed in Deliverable 1; this
    surfaces them as named arms so the menu in table7_sensitivity.csv is complete."""
    rows: list[dict[str, Any]] = []
    n = int(len(df))
    diag.append("\n=== component-only arms -> table7_sensitivity ===")
    for comp in COMPONENT_OUTCOMES:
        variant = COMPONENT_VARIANT[comp]
        ev = int(pd.to_numeric(df[comp], errors="coerce").fillna(0).sum())
        rate = 100.0 * ev / n if n else float("nan")
        strip = OUTCOME_DEFINING_STRIP.get(comp, [])
        note = (f"component-only arm ({COMPONENT_LABEL[comp]}); "
                f"change-point from the Deliverable-1 decomposition"
                + (f"; adjustment strip {strip} (§9 v3-K)" if strip else ""))
        for exposure in EXPOSURES:
            cp = component_cps[exposure][comp]
            rows.append(_cp_row(variant, "component", "cohort_primary==1", n, ev, rate,
                                exposure, comp, cp, notes=note))
        diag.append(f"  {variant:22s} outcome={comp:26s} events={ev:,} ({rate:.3f}%)"
                    + ("  [LOW-EVENT]" if ev < LOW_EVENT_FLAG_THRESHOLD else ""))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 script 13 — sensitivity analyses + component decomposition.")
    p.add_argument("--sample", type=int, default=None, help="Row-downsample N (seed 42) smoke test.")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iters for the headline death-only component change-point CI (default 1000).")
    p.add_argument("--n-boot-comp-other", type=int, default=400,
                   help="Bootstrap iters for the HC/transfusion component change-point CIs (secondary; default 400).")
    p.add_argument("--n-boot-sens", type=int, default=400,
                   help="Bootstrap iters for the SBP composite change-point CI per sensitivity variant (default 400).")
    p.add_argument("--n-boot-sens-si", type=int, default=0,
                   help="Bootstrap iters for the SI composite change-point per variant (default 0 = point-only).")
    p.add_argument("--mi-m", type=int, default=20, help="sens_mi blood_4h MI m (default 20).")
    p.add_argument("--n-boot-sens-new", type=int, default=200,
                   help="Bootstrap iters for the SBP change-point on the v3 outcome-swap "
                        "arms (default 200; the carried v2 arms use --n-boot-sens).")
    p.add_argument("--n-boot-auroc", type=int, default=1000,
                   help="Bootstrap iters for the score-arm AUROC CIs (default 1000).")
    p.add_argument("--no-newarm-changepoints", action="store_true",
                   help="Emit only the score-AUROC lens for the v3 outcome-swap arms "
                        "(skips their change-point/threshold-OR row-groups). Escape hatch "
                        "for runtime; the arms and their event counts are still reported.")
    p.add_argument("--no-score-arms", action="store_true",
                   help="Skip the Analysis-A (score refit) lens entirely. NOT a valid "
                        "production setting: sens_v2def_outcome / sens_exclude_nslo / "
                        "sens_with_ring exist only in that lens (CONTRACTS §13 G3e).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    for d in (DERIVED, TABLES, LOGS):
        os.makedirs(d, exist_ok=True)

    # ---- GATE S7-STATIC, before any compute -------------------------------
    # "CONTRACTS §7 names an arm this file cannot build" is knowable without data.
    # Failing here costs a second; failing at the end costs the ~66-minute run.
    _pregate: list[str] = []
    s7_static = gate_s7_static_coverage(_pregate)

    df = load_cohort(args.sample, acetabulum_inclusive=False)
    diag: list[str] = [
        "v3 script 13 — SENSITIVITY ANALYSES + COMPONENT DECOMPOSITION (CONTRACTS §7 row 13)",
        f"analytic N (cohort_primary==1{', sampled' if args.sample else ''}) = {len(df)}",
        f"seed={RANDOM_SEED}; n_boot(death)={args.n_boot}, n_boot(HC/tx)={args.n_boot_comp_other}, "
        f"n_boot_sens(SBP)={args.n_boot_sens}, n_boot_sens(SI)={args.n_boot_sens_si}, mi_m={args.mi_m}",
        "engine inherited from 11: SI winsor=2.0; convergence guard; HC1 for point models; "
        "deviance-only newton-cholesky inner fit; joblib bootstrap; one-at-a-time MI; interior 5-95pct; >=100 ev/side.",
    ]
    diag += _pregate          # the static §7 gate ran first; keep its lines in order

    # VIF audit (match script 11; may drop ais_head from CORE_ADJUSTMENT)
    dropped = vif_audit(df, diag)
    if dropped:
        for d_ in dropped:
            if d_ in CORE_ADJUSTMENT:
                CORE_ADJUSTMENT.remove(d_)
        diag.append(f"CORE_ADJUSTMENT after VIF drop: {CORE_ADJUSTMENT}")
    # Composite-run adjustment (ais_head retained: the composite is not a
    # deterministic function of it). Mortality-only runs get their own strip via
    # adjustment_for(exposure, outcome=...).
    adj_map = {ex: adjustment_for(ex) for ex in EXPOSURES}

    # ---- composite reference (script 11) ----
    ref = load_composite_reference()
    diag.append(f"\ncomposite reference (script 11): {ref['values']} (source={ref['source']})")

    # =====================================================================
    # DELIVERABLE 1 — component decomposition
    # =====================================================================
    decomp_df, component_cps = build_component_decomposition(
        df, args.n_boot, args.n_boot_comp_other, diag)
    decomp_df.to_csv(DECOMP_CSV, index=False)
    LOGGER.info("wrote %s (%d rows)", DECOMP_CSV, len(decomp_df))
    coloc = colocation_report(component_cps, ref)
    diag.append("\n=== CO-LOCATION (death-only vs composite; script 11) ===")
    for ex, c in coloc.items():
        diag.append(f"  {ex}: death-only bp={c['death_only_breakpoint']} "
                    f"CI{c['death_only_ci']} vs composite bp={c['composite_breakpoint_script11']} "
                    f"-> delta={c['delta_death_minus_composite']}  VERDICT: {c['verdict']}"
                    f" (matches_11_mortality={c['death_component_matches_script11_mortality']}; "
                    f"composite_at_floor={c['composite_at_grid_floor']})")
        if not c["verdict_reliable"]:
            diag.append(f"      -> {c['verdict_undetermined_reason']}. "
                        f"'UNDETERMINED' is NOT 'does not co-locate'; do not report it as one.")
            LOGGER.warning("co-location for %s is UNDETERMINED: %s", ex,
                           c["verdict_undetermined_reason"])

    # =====================================================================
    # DELIVERABLE 2 — Venn + write component_venn.json (Venn + CPs + reference)
    # =====================================================================
    venn = build_venn(df, diag)

    def _cp_export(cp: dict[str, Any]) -> dict[str, Any]:
        keep = ["exposure", "outcome", "component", "label", "breakpoint", "lci", "uci",
                "beta_pre", "beta_post", "or_per_unit_pre", "or_per_unit_post", "n", "events",
                "interior_lo", "interior_hi", "grid_min", "grid_max", "grid_step",
                "n_valid_grid", "n_boot", "at_grid_boundary", "boot_n_ok", "boot_n_unique"]
        return {k: cp.get(k) for k in keep if k in cp}

    venn_out = {
        "venn": venn,
        "component_changepoints": {ex: {comp: _cp_export(component_cps[ex][comp])
                                        for comp in COMPONENT_OUTCOMES} for ex in EXPOSURES},
        "composite_reference_script11": ref,
        "colocation": coloc,
        "meta": {"script": "13_sensitivity_components.py", "seed": RANDOM_SEED,
                 "analytic_n": int(len(df)), "generated_epoch": time.time()},
    }
    with open(VENN_JSON, "w") as f:
        json.dump(venn_out, f, indent=2, default=str)
    LOGGER.info("wrote %s", VENN_JSON)

    # =====================================================================
    # DELIVERABLE 3 — sensitivity variants -> table7_sensitivity.csv
    # =====================================================================
    # Discover which arms this cohort schema can actually support (the hour-rule /
    # day-rule transition in `01` changes the available timing columns).
    swap_variants = outcome_swap_variants(df.columns)
    diag.append(f"\n=== SENSITIVITY MENU RESOLVED AGAINST THE COHORT SCHEMA ===")
    diag.append(f"  timing columns present: "
                f"{[c for c in (TIMING_HRS_COL, TIMING_DAYS_COL) if c in df.columns] or 'none'}")
    diag.append(f"  outcome-swap arms buildable ({len(swap_variants)}): "
                f"{[v for v, _ in swap_variants]}")
    dropped_arms = [v for v, _, _ in OUTCOME_SWAP_SPEC if v not in {x for x, _ in swap_variants}]
    if dropped_arms:
        diag.append(f"  arms DROPPED (columns absent from this cohort build): {dropped_arms}")

    all_rows: list[dict[str, Any]] = []
    all_rows += reference_rowgroup(ref, df, diag)

    # ---- component-only arms (change-points reused from Deliverable 1) ----
    all_rows += component_rowgroup(component_cps, df, diag)

    # ---- the v3 outcome-swap arms (Analysis-B lens) ----------------------
    # Each arm's outcome comes from build_variant_outcome() — the SAME function the
    # score arms use — so the two lenses can never disagree about what an arm means.
    if args.no_newarm_changepoints:
        diag.append("\n[ARGS] --no-newarm-changepoints: the v3 outcome-swap arms report the "
                    "score-AUROC lens only; their change-point/threshold-OR row-groups are "
                    "skipped for runtime.")
        LOGGER.warning("--no-newarm-changepoints set: skipping change-points for the v3 "
                       "outcome-swap arms.")
    else:
        for variant, arm in swap_variants:
            y_v, def_v = build_variant_outcome(df, variant)
            all_rows += run_composite_variant(
                df, variant, arm, "cohort_primary==1", y_v, def_v, adj_map,
                args.n_boot_sens_new, args.n_boot_sens_si, diag,
                notes="v3 named sensitivity (CONTRACTS §7 row 13)")

    # ---- acetabulum-inclusive cohort: primary composite on primary|has_acetabulum ----
    df_acet = load_cohort(args.sample, acetabulum_inclusive=True)
    all_rows += run_composite_variant(
        df_acet, "sens_acetabulum", "acetabulum_inclusive",
        "cohort_primary==1 | has_acetabulum==1", df_acet[COMPOSITE_COL].astype(float),
        f"{COMPOSITE_COL} = {HC_COL} | {DEATH_COL} | {TX_COL}",
        adj_map, args.n_boot_sens, args.n_boot_sens_si, diag,
        notes=f"ring∪acetabulum N={len(df_acet)} vs ring-only N={len(df)}; "
              f"compare to ring_only_reference")
    del df_acet

    # ---- residual blood-missingness: exclude arm (the sens_mi comparator) ----
    obs = df["blood_4h"].notna()
    df_excl = df.loc[obs].copy()
    excl_rows = run_composite_variant(
        df_excl, "sens_mi", "exclude", "cohort_primary==1 & blood_4h observed",
        df_excl[COMPOSITE_COL].astype(float),
        f"{COMPOSITE_COL} (complete-case: blood_4h observed only)", adj_map,
        args.n_boot_sens, args.n_boot_sens_si, diag,
        notes=f"exclude {int((~obs).sum())} blood-missing rows ({100*(~obs).mean():.2f}%); "
              f"complete-case comparator for the MI arm")
    all_rows += excl_rows
    # within-variance for the MI Rubin within-term = exclude-arm SBP/SI bootstrap variance
    def _within_from_rows(rows, exposure):
        for r in rows:
            if r["analysis"] == "changepoint" and r["exposure"] == exposure \
               and np.isfinite(r.get("cp_lci", np.nan)) and np.isfinite(r.get("cp_uci", np.nan)):
                return ((r["cp_uci"] - r["cp_lci"]) / (2 * Z)) ** 2
        return float("nan")
    wv_sbp = _within_from_rows(excl_rows, "sbp2")
    wv_si = _within_from_rows(excl_rows, "shock_index")
    del df_excl

    # ---- residual blood-missingness: MI arm (impute blood_4h, m=20, one-at-a-time) ----
    LOGGER.info("sens_mi (m=%d, one-at-a-time; imputing blood_4h) ...", args.mi_m)
    mi_rows, mi_detail = run_variant_d_mi(df, adj_map, args.mi_m, wv_sbp, wv_si, RANDOM_SEED, diag)
    all_rows += mi_rows

    # =====================================================================
    # DELIVERABLE 3b — ANALYSIS-A LENS: score refit + held-out AUROC per arm
    # =====================================================================
    score_summary: dict[str, Any] = {"ran": False}
    if args.no_score_arms:
        diag.append("\n[ARGS] --no-score-arms: the Analysis-A lens was SKIPPED. "
                    "sens_v2def_outcome / sens_exclude_nslo / sens_with_ring are NOT in "
                    "this table. Not a valid production setting (CONTRACTS §13 G3e).")
        LOGGER.error("--no-score-arms set: sens_v2def_outcome / sens_exclude_nslo / "
                     "sens_with_ring were NOT produced.")
    else:
        inp = load_score_inputs(args.sample)
        if inp is None:
            diag.append("\n[FAIL] Analysis-A lens could not be evaluated: 02/03/05 artifacts "
                        "absent. Per CONTRACTS §13 a gate that cannot be evaluated is a FAIL, "
                        "not a skip. sens_v2def_outcome / sens_exclude_nslo / sens_with_ring "
                        "are MISSING from table7_sensitivity.csv.")
        else:
            nb_auc = min(args.n_boot_auroc, 200) if args.sample else args.n_boot_auroc
            diag.append(f"\n=== DELIVERABLE 3b: SCORE ARMS (Analysis-A lens; boot={nb_auc}) ===")
            diag.append(f"  frozen spec: {len(inp['vars_kept'])} vars "
                        f"(point={len(inp['point_vars'])}, override={len(inp['override_vars'])}); "
                        f"anchors={inp['force_spec']}; roles from {inp['role_src']}")

            # (i) internal reference: the ring-free v3 score refit on the v3 PRIMARY
            #     outcome via this exact code path, so every arm below is a like-for-
            #     like delta against it. The AUROC of record for the paper is 07's
            #     table4_discrimination.csv — this row exists for CONTRAST only.
            all_rows += run_score_variant(
                inp, "primary_score_reference", "reference", "cohort_primary==1 (TEST slice)",
                nb_auc, diag,
                notes="INTERNAL CONTRAST ONLY — the headline AUROC of record is 07's "
                      "table4_discrimination.csv; this row is recomputed by 13's refit path "
                      "so each sensitivity arm below is a like-for-like delta")

            # (ii) every outcome-swap arm, same score, different outcome
            # The score arms read candidates.parquet, whose column set can differ
            # from the cohort CSV's; filter the menu against IT, not against the
            # cohort frame.
            score_swaps = outcome_swap_variants(inp["train"].columns)
            for variant, arm in score_swaps:
                all_rows += run_score_variant(
                    inp, variant, arm, "cohort_primary==1 (TEST slice)", nb_auc, diag,
                    notes="outcome swapped; score variable set unchanged")

            # (iii) sens_exclude_nslo — Fix 4 / §5.1: drop the no-signs-of-life rows
            #       ENTIRELY (rows AND the then-constant column; see the singularity
            #       guard inside run_score_variant).
            all_rows += run_score_variant(
                inp, "sens_exclude_nslo", "exclude_nslo",
                "cohort_primary==1 minus sbp_unrecordable==1 (TEST slice)", nb_auc, diag,
                notes="Fix 4 — no-signs-of-life-on-arrival rows dropped entirely "
                      "(cohort-wide n=608); the sbp_unrecordable column becomes constant "
                      "and is removed from the design matrix by the singularity guard",
                row_filter="exclude_nslo")

            # (iv) sens_with_ring — what the D2 concession cost. `unstable_ring` is
            #      restored to the score variable list AND to the force-in anchors,
            #      per refit.json's persisted lasso_x_cols / force_in_spec.
            if RING_VAR in inp["train"].columns:
                all_rows += run_score_variant(
                    inp, "sens_with_ring", "with_ring", "cohort_primary==1 (TEST slice)",
                    nb_auc, diag,
                    notes=(f"'{RING_VAR}' restored to the score variable list and to the "
                           f"force-in anchors (refit.json lasso_x_cols={len(inp['lasso_pool'])} "
                           f"+ ring; force_in_spec={inp['force_spec']} + ring). LASSO "
                           f"re-selection over the 18-variable pool is NOT re-run here — "
                           f"that is 04's machinery — so this arm isolates the ring's "
                           f"marginal contribution at a FIXED variable set. Delta vs "
                           f"primary_score_reference = the cost of the D2 concession."),
                    extra_vars=[RING_VAR], extra_anchors=[RING_VAR])
            else:
                LOGGER.warning("'%s' absent from candidates.parquet — sens_with_ring skipped.",
                               RING_VAR)
                diag.append(f"  [WARN] sens_with_ring SKIPPED: '{RING_VAR}' not in "
                            f"candidates.parquet.")
            score_summary = {"ran": True, "n_boot_auroc": nb_auc,
                             "vars_kept": inp["vars_kept"], "point_vars": inp["point_vars"],
                             "override_vars": inp["override_vars"],
                             "force_spec": inp["force_spec"], "role_src": inp["role_src"],
                             "n_train": int(len(inp["train"])), "n_test": int(len(inp["test"]))}

    sens_df = pd.DataFrame(all_rows, columns=SENS_COLS)
    sens_df.to_csv(TABLE7_SENS, index=False)
    LOGGER.info("wrote %s (%d rows, %d variants)", TABLE7_SENS, len(sens_df),
                sens_df["variant"].nunique())

    # ---- G3e verdict: sens_v2def_outcome is the ONLY apples-to-apples comparator ----
    v2def_verdict: dict[str, Any] = {"evaluated": False}
    v2row = sens_df[(sens_df["variant"] == "sens_v2def_outcome")
                    & (sens_df["analysis"] == "score_auroc")
                    & (sens_df["term"] == "integer card (pelvic_score)")]
    diag.append("\n=== §13 G3e — sens_v2def_outcome vs the 0.871 pre-check ===")
    if len(v2row) and np.isfinite(float(v2row.iloc[0]["auroc"])):
        a = float(v2row.iloc[0]["auroc"])
        defect = bool(a < V2DEF_AUROC_FORK_DEFECT_FLOOR)
        v2def_verdict = {"evaluated": True, "auroc": a,
                         "expected": V2DEF_AUROC_EXPECTED,
                         "fork_defect_floor": V2DEF_AUROC_FORK_DEFECT_FLOOR,
                         "fork_defect_suspected": defect}
        msg = (f"  sens_v2def_outcome AUROC = {a:.4f} (integer card) vs the documented "
               f"no-refit pre-check {V2DEF_AUROC_EXPECTED:.3f}.\n"
               f"  This comparison is valid for THIS ARM ONLY — it is the ring-free score "
               f"evaluated on the reconstructed v2 outcome. It is NOT comparable to the v3 "
               f"headline, whose outcome is a different event set.\n"
               f"  VERDICT: " + ("**FORK DEFECT SUSPECTED** — landing below "
                                 f"~{V2DEF_AUROC_FORK_DEFECT_FLOOR:.2f} here indicates the "
                                 f"fork changed something it did not claim to, NOT a "
                                 f"scientific finding. Investigate before reporting."
                                 if defect else
                                 f"consistent with the pre-check (>= "
                                 f"{V2DEF_AUROC_FORK_DEFECT_FLOOR:.2f}); the fork is faithful."))
        diag.append(msg)
        (LOGGER.error if defect else LOGGER.info)(msg.replace("\n", " "))
    else:
        diag.append("  NOT EVALUATED — the score arms did not run. Per CONTRACTS §13 a gate "
                    "that cannot be evaluated is a FAIL, not a skip.")
        LOGGER.error("G3e NOT EVALUATED: sens_v2def_outcome score arm absent.")

    # ---- per-arm event ledger (which arms are thin?) ----
    diag.append("\n=== PER-ARM EVENT LEDGER (v3's death arm is ~14.7% of the composite; "
                "43% in v2 — several death-ladder arms are materially thinner) ===")
    led = (sens_df.groupby("variant", dropna=False)
                  .agg(n_events=("n_events", "max"), rate=("composite_rate_pct", "max"),
                       low=("low_event_flag", "max"), rows=("variant", "size"))
                  .reset_index())
    for _, r in led.iterrows():
        diag.append(f"  {str(r['variant']):26s} events={r['n_events']}  "
                    f"rate={r['rate']}%  rows={int(r['rows'])}"
                    + ("   [LOW-EVENT — treat as descriptive]" if r["low"] == 1 else ""))

    # ---- robustness summary (headline SBP composite change-point across variants) ----
    diag.append("\n=== HEADLINE ROBUSTNESS: SBP composite change-point across variants ===")
    sbp_cps = sens_df[(sens_df["exposure"] == "sbp2") & (sens_df["analysis"] == "changepoint")]
    for _, r in sbp_cps.iterrows():
        diag.append(f"  {r['variant']:26s} [{str(r['arm']):20s}] SBP bp={r['changepoint']} "
                    f"CI[{r['cp_lci']},{r['cp_uci']}]")

    # ---- robustness summary (score AUROC across arms; Analysis-A lens) ----
    diag.append("\n=== HEADLINE ROBUSTNESS: score AUROC (integer card) across arms ===")
    auc_rows = sens_df[(sens_df["analysis"] == "score_auroc")
                       & (sens_df["term"] == "integer card (pelvic_score)")]
    if len(auc_rows):
        for _, r in auc_rows.iterrows():
            diag.append(f"  {r['variant']:26s} AUROC={r['auroc']} "
                        f"[{r['auroc_lci']},{r['auroc_uci']}]  "
                        f"events={r['n_events']}  vars={r['n_score_vars']}")
    else:
        diag.append("  (none — the Analysis-A lens did not run)")

    # =====================================================================
    # §7-COVERAGE GATE, PART 2 OF 2 — RUNTIME: did the CONTRACT's arms LAND?
    # =====================================================================
    # The expectation is CONTRACTS_S7_VARIANTS — a hand transcription of the
    # contract — and NOTHING derived from this run. The superseded version built
    # `expected_variants` from `[v for v, _ in swap_variants]`, i.e. from the same
    # OUTCOME_SWAP_SPEC it was auditing, so it could not detect an arm the script
    # had never defined: it printed "[OK] every CONTRACTS §7 variant is present"
    # over two missing recovery-ladder rungs. See CONTRACTS_S7_VARIANTS above and
    # the §13 preamble. Part 1 (gate_s7_static_coverage) already proved every
    # contract arm is IMPLEMENTED; this part proves each one actually EMITTED.
    present = set(sens_df["variant"].dropna().unique().tolist())
    lenses = (sens_df.dropna(subset=["variant"])
                     .groupby("variant")["analysis"]
                     .agg(lambda s: set(s.dropna().tolist())).to_dict())

    # Skip-flags are NOT exemptions. Per §13 preamble rule 2 an unevaluable gate is
    # a FAIL, and both flags are documented as invalid production settings — each
    # one removes whole arms or a whole lens from the table by construction, so the
    # coverage question genuinely cannot be answered under them.
    gate_blockers: list[str] = []
    if args.no_score_arms:
        gate_blockers.append("--no-score-arms (the Analysis-A lens did not run, so "
                             "sens_exclude_nslo / sens_with_ring cannot be present)")
    if args.no_newarm_changepoints:
        gate_blockers.append("--no-newarm-changepoints (the outcome-swap arms have no "
                             "change-point lens)")
    # `--sample` is deliberately NOT a blocker: down-sampling invalidates the arms'
    # NUMBERS, not the menu's COVERAGE, and this gate asks only whether every
    # contract arm landed. Calling a fully-answerable question "NOT EVALUATED"
    # would be its own kind of dishonest gate. Recorded as provenance instead.
    gate_provenance = (f"--sample {args.sample}: coverage evaluated normally; the arms' "
                       f"numbers are smoke-grade, not production"
                       if args.sample else "full cohort (production)")

    missing_arms = [v for v in CONTRACTS_S7_VARIANTS if v not in present]
    missing_score_lens = [v for v in CONTRACTS_S7_SCORE_LENS_VARIANTS
                          if v in present and "score_auroc" not in lenses.get(v, set())]
    # An arm whose name is in the table but whose AUROC is NaN is NOT covered — the
    # `except Exception` in run_score_variant emits a row on failure so the arm is
    # visible, which means a name-only check would count a silently failed arm as
    # present. Coverage is about RESULTS landing, not row labels.
    _auroc_rows = sens_df[(sens_df["analysis"] == "score_auroc")
                          & (sens_df["term"] == "integer card (pelvic_score)")]
    empty_score_arms = sorted(
        {str(r["variant"]) for _, r in _auroc_rows.iterrows()
         if str(r["variant"]) in set(CONTRACTS_S7_SCORE_LENS_VARIANTS)
         and not np.isfinite(pd.to_numeric(r["auroc"], errors="coerce"))})
    undeclared_extras = sorted(present - set(CONTRACTS_S7_VARIANTS)
                               - set(EXTRA_DECLARED_VARIANTS))

    diag.append("\n=== GATE S7-COVERAGE — CONTRACTS §7 menu vs table7_sensitivity.csv ===")
    diag.append(f"  expectation source: CONTRACTS_S7_VARIANTS (hand-transcribed from "
                f"CONTRACTS.md §7), {len(CONTRACTS_S7_VARIANTS)} arms — NOT derived from "
                f"OUTCOME_SWAP_SPEC or from this run")
    diag.append(f"  variants emitted: {len(present)} "
                f"({len(CONTRACTS_S7_VARIANTS) - len(missing_arms)} of the contract's + "
                f"{len(present) - (len(CONTRACTS_S7_VARIANTS) - len(missing_arms))} extras)")
    diag.append(f"  run provenance: {gate_provenance}")
    gate_ok = True
    if gate_blockers:
        gate_ok = False
        diag.append(f"  GATE S7-COVERAGE: NOT EVALUATED — FAIL. Run flags prevent a "
                    f"meaningful evaluation: {gate_blockers}. Silence is not a pass "
                    f"(CONTRACTS §13 preamble, rule 2).")
        LOGGER.error("GATE S7-COVERAGE: NOT EVALUATED — FAIL (%s)", "; ".join(gate_blockers))
    if missing_arms:
        gate_ok = False
        diag.append(f"  [FAIL] CONTRACTS §7 arms ABSENT from table7_sensitivity.csv: "
                    f"{missing_arms}")
        LOGGER.error("GATE S7-COVERAGE FAIL — contract arms absent from the table: %s. "
                     "Report them as an explicit DEVIATION; do not ship a table that "
                     "claims coverage it does not have.", missing_arms)
    if missing_score_lens:
        gate_ok = False
        diag.append(f"  [FAIL] arms present but with NO score_auroc row (Analysis-A lens "
                    f"lost silently): {missing_score_lens}")
        LOGGER.error("GATE S7-COVERAGE FAIL — no score_auroc row for: %s", missing_score_lens)
    if empty_score_arms:
        gate_ok = False
        diag.append(f"  [FAIL] arms with a score_auroc ROW but a NON-FINITE AUROC (the arm "
                    f"failed and the row is a placeholder): {empty_score_arms}")
        LOGGER.error("GATE S7-COVERAGE FAIL — non-finite AUROC for: %s", empty_score_arms)
    if gate_ok:
        diag.append(f"  [PASS] all {len(CONTRACTS_S7_VARIANTS)} CONTRACTS §7 arms are "
                    f"present; all {len(CONTRACTS_S7_SCORE_LENS_VARIANTS)} score-lens arms "
                    f"carry a score_auroc row; every one of those AUROCs is finite.")
    if undeclared_extras:
        diag.append(f"  [NOTE] emitted but neither in §7 nor declared in "
                    f"EXTRA_DECLARED_VARIANTS: {undeclared_extras}")
        LOGGER.warning("undeclared extra variants in the table: %s", undeclared_extras)
    # Per-arm lens ledger — makes a half-populated arm visible at a glance.
    diag.append("  lens ledger (variant -> analyses emitted):")
    for v in list(CONTRACTS_S7_VARIANTS) + sorted(set(present) - set(CONTRACTS_S7_VARIANTS)):
        diag.append(f"    {v:26s} {sorted(lenses.get(v, set())) or 'ABSENT'}")
    # Names kept for backward compatibility with the manifest schema below.
    expected_variants = list(CONTRACTS_S7_VARIANTS)
    absent_variants = missing_arms

    # ---- diagnostics + manifest (13-namespaced, non-colliding) ----
    with open(DIAG_TXT, "w") as f:
        f.write("\n".join(diag) + "\n")
    manifest = {
        "script": "13_sensitivity_components.py", "random_seed": RANDOM_SEED, "alpha": ALPHA,
        "n_boot_death": args.n_boot, "n_boot_comp_other": args.n_boot_comp_other,
        "n_boot_sens_sbp": args.n_boot_sens, "n_boot_sens_si": args.n_boot_sens_si,
        "n_boot_sens_new": args.n_boot_sens_new, "n_boot_auroc": args.n_boot_auroc,
        "newarm_changepoints": (not args.no_newarm_changepoints),
        "score_arms": score_summary,
        "mi_m": args.mi_m, "sample": args.sample,
        "input_cohort": os.path.relpath(COHORT_CSV, config.REPO_ROOT), "input_sha256": _sha256(COHORT_CSV),
        "analytic_n": int(len(df)),
        "outcome_columns": {"composite": COMPOSITE_COL, "death": DEATH_COL,
                            "transfusion": TX_COL, "hc": HC_COL},
        "core_adjustment": CORE_ADJUSTMENT, "vif_dropped": dropped,
        "vif_audit_vars": VIF_AUDIT_VARS,
        "g3f_ring_retained_in_adjustment": bool(RING_VAR in CORE_ADJUSTMENT
                                                and RING_VAR in VIF_AUDIT_VARS),
        "outcome_defining_strip": OUTCOME_DEFINING_STRIP,
        "si_winsor_cap": SI_WINSOR_CAP, "min_events_per_side": MIN_EVENTS_PER_SIDE,
        "low_event_flag_threshold": LOW_EVENT_FLAG_THRESHOLD,
        "cohort_schema": {
            "n_columns": int(df.shape[1]),
            "timing_columns_present": [c for c in (TIMING_HRS_COL, TIMING_DAYS_COL)
                                       if c in df.columns],
            "optional_columns_present": [c for c in OPTIONAL_OUTCOME_COLS if c in df.columns],
            "outcome_swap_arms_built": [v for v, _ in swap_variants],
            "outcome_swap_arms_dropped": dropped_arms,
        },
        "composite_reference_script11": ref["values"],
        "composite_reference_is_v2_stale_fallback": bool(ref.get("stale")),
        "composite_reference_stale_tags": ref.get("stale_tags", []),
        "composite_reference_descriptive_only_tags": ref.get("descriptive_only_tags", []),
        "colocation": {ex: {"death_only_bp": coloc[ex]["death_only_breakpoint"],
                            "composite_bp": coloc[ex]["composite_breakpoint_script11"],
                            "colocates": coloc[ex]["colocates"],
                            "verdict": coloc[ex]["verdict"],
                            "verdict_reliable": coloc[ex]["verdict_reliable"],
                            "verdict_undetermined_reason":
                                coloc[ex]["verdict_undetermined_reason"]}
                       for ex in EXPOSURES},
        "g3e_v2def_verdict": v2def_verdict,
        "variants_present": sorted(present),
        "variants_missing": absent_variants,
        # ---- §7-coverage gate, both parts (expectation = the hand-transcribed
        # contract list, never this run's own output) ----
        "gate_s7_static": s7_static,
        "gate_s7_coverage": {
            "expectation_source": "CONTRACTS_S7_VARIANTS (hand-transcribed from "
                                  "CONTRACTS.md §7; not derived from OUTCOME_SWAP_SPEC "
                                  "or from the emitted table)",
            "contract_variants": list(CONTRACTS_S7_VARIANTS),
            "contract_n": len(CONTRACTS_S7_VARIANTS),
            "missing_arms": missing_arms,
            "missing_score_lens": missing_score_lens,
            "score_arms_with_nonfinite_auroc": empty_score_arms,
            "undeclared_extras": undeclared_extras,
            "declared_extras": sorted(EXTRA_DECLARED_VARIANTS),
            "not_evaluated_blockers": gate_blockers,
            "run_provenance": gate_provenance,
            "verdict": "PASS" if gate_ok else "FAIL",
            "lens_ledger": {v: sorted(lenses.get(v, set()))
                            for v in sorted(set(list(CONTRACTS_S7_VARIANTS) + list(present)))},
        },
        "sens_mi": mi_detail,
        "outputs": {"component_decomp_data": os.path.relpath(DECOMP_CSV, config.REPO_ROOT),
                    "component_venn": os.path.relpath(VENN_JSON, config.REPO_ROOT),
                    "table7_sensitivity": os.path.relpath(TABLE7_SENS, config.REPO_ROOT)},
        "packages": _package_versions(), "runtime_sec": round(time.time() - t0, 2),
    }
    with open(MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    LOGGER.info("DONE in %.1f min. death-only co-location: %s",
                (time.time() - t0) / 60.0,
                {ex: (round(component_cps[ex][DEATH_COL]["breakpoint"], 3)
                      if np.isfinite(component_cps[ex][DEATH_COL]["breakpoint"]) else None,
                      coloc[ex]["colocates"]) for ex in EXPOSURES})
    # Artifacts are written FIRST (partial results and the FAIL verdict are more
    # useful than nothing), then the exit code carries the gate's verdict. A run
    # that shipped an incomplete table must not report success.
    if not gate_ok:
        LOGGER.error("EXIT NON-ZERO: GATE S7-COVERAGE verdict = FAIL. table7_sensitivity.csv "
                     "and the manifest were written, and both record what is missing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
