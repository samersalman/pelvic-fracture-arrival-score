"""
11_pelvic_thresholds.py
=======================

Analysis B — physiologic THRESHOLD analysis for the v3 pre-decision pelvic
arrival tool (SAP §6; CONTRACTS §7 row 11, §9-H).

Forked from an earlier time-to-intervention threshold script (house style: copy-adapt, no
cross-script imports). This is a self-contained script; it reads only the
Batch-1 cohort CSV and writes only the artifacts it owns (§8, §13.4).

Run matrix (SAP §6 / CONTRACTS): 2 exposures (shock_index, sbp2) x 2 outcomes
(``major_hemorrhage_event`` composite; ``death_hemorrhagic`` mortality-only
anchor) = 4 headline curves.

v3 OUTCOME SWEEP (CONTRACTS §9-J) — read before touching any outcome name
-------------------------------------------------------------------------
``major_hemorrhage_event`` KEEPS its name but CHANGES MEANING; the mortality-only
anchor's COLUMN NAME changed from ``death_inhosp`` to ``death_hemorrhagic``:

    major_hemorrhage_event = received_hc | death_hemorrhagic | substantial_transfusion
    death_hemorrhagic      = death_inhosp & (ais_head < 4) & (finaldischarge_hrs <= 48)

The death rule is an **HOURS** rule on ``finaldischarge_hrs``. An earlier
calendar-day operationalisation (``finaldischarge_days <= 2``) is SUPERSEDED and
survives only as the named crosswalk ``death_hemorrhagic_dayrule``. The
authoritative string is ``logs/qc_manifest.json →
death_variants.death_hemorrhagic_definition``; this script emits no definition
string of its own, so the line above is documentation, not a source of truth.

``death_inhosp`` survives in v3 ONLY as a named sensitivity arm (CONTRACTS §4
rung 0, owned by `13`) and is not run here.

v3-K PER-OUTCOME ADJUSTMENT STRIP (CONTRACTS §6 note, SAP D-1)
--------------------------------------------------------------
``ais_head`` is an INGREDIENT of ``death_hemorrhagic``. Adjusting a mortality-only
model for a variable that appears in its own outcome definition is conditioning
on the outcome, so ``ais_head`` is stripped from ``CORE_ADJUSTMENT`` on the
mortality-only runs and RETAINED on the composite runs (where the outcome is not
a deterministic function of it). **Both** timing fields are likewise
outcome-defining and are never adjustment covariates in any run:
``finaldischarge_hrs`` (the field that actually defines the outcome — the more
dangerous of the two) and ``finaldischarge_days`` (the superseded proxy).
Implemented as ``OUTCOME_DEFINING_STRIP`` + ``NEVER_ADJUST`` and applied by
``adjustment_for(exposure, outcome)``, i.e. exactly the mechanism this script
already uses to strip the active exposure and its mediators. The stripped set is
logged per run and written into every change-point JSON as ``adjustment`` /
``adjustment_stripped``.

v3 FIX-3 BOUNDARY — ``unstable_ring`` STAYS HERE (CONTRACTS §6, §13 G3f)
-----------------------------------------------------------------------
Fix 3 (D2) removes ``unstable_ring`` from the SCORE's candidate pool and
force-in anchors — files 02/04/05. It is deliberately RETAINED in this script's
``CORE_ADJUSTMENT``, in ``VIF_AUDIT_VARS``, and as the ring-stability stratified
change-point stratum. Analysis B is an ASSOCIATION model, not a bedside
instrument: adjusting it for a confounder ascertained later in the admission is
not a claim that the confounder was available on arrival, and dropping it would
leave the SI/SBP dose-response curves confounded by injury pattern. The
asymmetry is registered in SAP_amendment_v3.md D-3 and asserted below. An agent
that "consistently" propagates Fix 3 into this file silently degrades Analysis B
and fails gate G3f.

v3 POWER GUARD (CONTRACTS §13 G5e, locked SAP §6)
-------------------------------------------------
The mortality-only anchor lost roughly two thirds of v2's all-cause deaths (exact
cohort-wide counts: ``logs/qc_manifest.json → death_variants``; per CONTRACTS
§3.1 they are read, never hard-coded).
Stratified change-points — elderly∩anticoagulated above all — may now fall below
the pre-specified >=100-events-per-side guard. The locked SAP already prescribes
the response: report those strata DESCRIPTIVE-ONLY. That branch is implemented
explicitly (``descriptive_only`` + ``guard_reason`` on every stratum entry, a
``_guard_summary`` block in ``changepoint_stratified.json``, a loud WARNING per
demotion, and a ``descriptive_only`` list in the manifest). ``MIN_EVENTS_PER_SIDE``
is NOT relaxed — a demoted stratum is a reportable finding, not a threshold to
lower.

The FOUR required changes vs the source script (CONTRACTS §9-H):
  1. HC1 robust SE (no clustering — there is NO facility id in this data;
     mirrors ``pelvic_analysis.fit_logistic_robust``). The source used nonrobust.
  2. Change-point grids (SI 0.60-1.60 by 0.02; SBP 70-130 by 2) RESTRICTED to
     the interior 5th-95th percentile with a >=100-events-per-side guard
     (the source had neither); ROW-level bootstrap CI (1000x) — the source used a
     pseudo-site cluster bootstrap that has no analogue here.
  3. MI (m=20): imputation model INCLUDES the outcome (congeniality) AND
     imputes the exposure vitals (sbp2/pulse2/shock_index — the exposure IS
     the missing variable). The change-point LOCATION is pooled across the 20
     imputations (Rubin) — the source pooled only threshold RD/OR.
  4. Tipping-point bound: refit the change-point assigning all missing-exposure
     rows to the worst-tail vs the best-tail; report the range.

Also (SAP §6): pre-specified thresholds Bonferroni m=4 (SI 1.0/1.3; SBP 90/70)
with g-computed adjusted absolute risk each side + risk difference; categorical
band ORs (reference SI<0.8 / SBP>=110); per-0.1-SI and per-10-mmHg adjusted
ORs; stratified change-points (ring stability, age>=65, cc_anticoag,
elderly∩anticoag) with a spline x stratum interaction-LR test (pre-registered
directional hypothesis: elderly/anticoag inflect EARLIER).

Outputs owned by this script (CONTRACTS §7/§8):
  tables/table7_threshold_or.csv         pre-specified thresholds + bands + per-unit ORs
  tables/table7_absrisk.csv              g-computed adjusted absolute risk + RD (CC + MI)
  derived/spline_si_composite.csv        4 headline spline curves (fig-data F7)
  derived/spline_si_mortality.csv
  derived/spline_sbp_composite.csv
  derived/spline_sbp_mortality.csv
  derived/changepoint_si_composite.json  4 change-point results (+ MI + tipping)
  derived/changepoint_si_mortality.json
  derived/changepoint_sbp_composite.json
  derived/changepoint_sbp_mortality.json
  derived/changepoint_stratified.json    strata change-points + interaction p
  derived/component_splines.csv          component-decomposition splines (fig-data F8)
  derived/tipping_point_bound.json       MNAR tipping-point bounds
  logs/thresholds_diagnostics.txt        VIF audit, LR tests, knots, gate summary
  logs/thresholds_manifest.json          reproducibility manifest

Performance note (coordinator-directed optimization, 2026-07-02): the
deviance-minimizing segmented-logistic change-point uses a FULL adjusted refit
at every grid point and every bootstrap resample, but the science is preserved
exactly while the computation is made tractable by two changes: (1) the inner
grid/bootstrap fits DROP ``cov_type="HC1"`` (the breakpoint is chosen by minimum
DEVIANCE, which is covariance-type invariant; HC1 is reserved for the FINAL
reported point models — the 4 splines, per-unit ORs, threshold ORs, g-computed
RDs); (2) the inner fits use sklearn ``LogisticRegression(penalty=None,
solver="newton-cholesky")`` — validated to return the IDENTICAL coefficients and
breakpoint as the statsmodels GLM at ~6-8x lower cost — and the 1000 row-level
bootstrap iterations are parallelized across CPU cores with joblib. Full grids
(SI 0.60-1.60 by 0.02; SBP 70-130 by 2), interior 5th-95th restriction, >=100
events/side, full N, and MI m=20 are all unchanged.

MEMORY (the OOM fix): MI processes imputations ONE AT A TIME (``run_mi`` /
``_mi_impute_one``) — generate imputation i, accumulate its change-point +
threshold Rubin components, discard the frame before i+1. The m=20 imputations
never coexist in RAM.

DOCUMENTED COMPUTATIONAL DEVIATIONS (SAP §11):
  * SECONDARY bootstraps capped at 250 iters (``--n-boot-strat``, default 250:
    the headline-stratum change-point CIs). The PRIMARY complete-case main
    change-point CI stays at the full 1000 (``--n-boot``).
  * MI within-imputation variance defaults to the complete-case row-bootstrap
    variance (Rubin within-term; ``--n-boot-mi 0``) — a per-imputation bootstrap
    x m=20 x 4 curves is not wall-clock tractable. The change-point LOCATION is
    still pooled across all 20 imputations (Rubin between-term). Set
    ``--n-boot-mi > 0`` to enable a per-imputation within-variance bootstrap.
"""

from __future__ import annotations

# Pin BLAS/OpenMP to 1 thread per process BEFORE numpy/sklearn import so the
# joblib process-parallel bootstrap does not oversubscribe cores (nested
# process x BLAS-thread parallelism is a known slowdown).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")  # defensive: set before any pyplot import (CONTRACTS §1)

import argparse
import hashlib
import json
import logging
import os
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
from joblib import Parallel, delayed  # parallel bootstrap

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn penalty= deprecation
warnings.filterwarnings("ignore", category=st.ConstantInputWarning) if hasattr(
    st, "ConstantInputWarning"
) else None

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

# ---------------------------------------------------------------------------
# Reproducibility constants (CONTRACTS §1, §Reproducibility)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
ALPHA = 0.05
N_BOOT_DEFAULT = 1000
MI_M_DEFAULT = 20
Z = st.norm.ppf(1.0 - ALPHA / 2.0)

# shock_index winsorization cap (computational deviation, 2026-07-03; SAP §11).
# The raw SI has a pathological upper tail (max≈53; p95≈1.26; ~1,300 rows >2.0,
# 19 rows >5.0) from near-zero SBP denominators. Fed raw into cr(shock_index,
# df=4)+HC1 it makes the SI×composite spline NON-CONVERGENT (converged=False,
# llf=NaN) and poisons EVERY HC1 SE to NaN. SI>2 is essentially always a data
# artifact; capping at 2.0 restores convergence (verified converged=True, finite
# llf + SEs) with negligible clinical loss. Applied ONCE in load_cohort so the
# winsorized SI flows to every downstream model. sbp2 is already bounded — NOT
# touched.
SI_WINSOR_CAP = 2.0

# ---------------------------------------------------------------------------
# Adjustment set (CONTRACTS §6). HC1 robust SE, no clustering.
#   CORE_ADJUSTMENT strips the source script's time_to_hc / sbp / shock_index from the
#   constant — never adjust for the exposure or a mediator.
# ---------------------------------------------------------------------------
CORE_ADJUSTMENT: list[str] = [
    "age",
    "sex_male",
    "mechanism_cat",
    "iss",
    "ais_head",
    "unstable_ring",
    "cc_anticoag",
    "charlson",
    "acs_level",
    "puf_year",
]
# Only these adjustment covariates enter patsy as C(...); charlson (0-5),
# ais_head (0-6 AIS), sex_male/unstable_ring/cc_anticoag (0/1) and age/iss are
# entered linearly (parsimony + robustness in stratum/bootstrap subsets).
CATEGORICAL_ADJ: frozenset[str] = frozenset({"mechanism_cat", "acs_level", "puf_year"})

# Per-run mediator/exposure strip (CONTRACTS §6 rule). None of these currently
# sit inside CORE_ADJUSTMENT, so the strip is defensive/documented (a no-op on
# the locked set), but it enforces the rule if the set ever changes.
EXPOSURE_STRIP: dict[str, list[str]] = {
    # SI spline -> drop SBP + HR components/markers.
    "shock_index": ["sbp2", "pulse2", "shock_index", "sbp_lt90", "sbp_unrecordable"],
    # SBP spline -> drop SBP + ALL shock markers.
    "sbp2": ["sbp2", "shock_index", "pulse2", "sbp_lt90", "sbp_unrecordable"],
}

# ---------------------------------------------------------------------------
# v3-K  PER-OUTCOME strip: never adjust for a variable that DEFINES the outcome
#       (CONTRACTS §6 v3 note; SAP_amendment_v3.md D-1).
#
#   death_hemorrhagic = death_inhosp & (ais_head < 4) & (finaldischarge_hrs <= 48)
#
# so on ANY mortality-only run, `ais_head` is part of the outcome's own
# definition and adjusting for it is conditioning on the outcome. It STAYS on
# the composite runs, where the composite is not a deterministic function of it
# (an event can arrive via received_hc or substantial_transfusion at any head
# AIS).
#
# TIMING FIELDS — BOTH, always. `finaldischarge_hrs` is THE outcome-defining
# timing field (the ≤48 h conjunct is computed from it) and is therefore the
# more dangerous of the two; `finaldischarge_days` is the SUPERSEDED calendar-day
# proxy, retained in the cohort only as the `*_dayrule` crosswalk. Neither is
# arrival-knowable and neither may ever be an adjustment covariate. Both are in
# CONTRACTS §5's FORBIDDEN_X and neither is in CORE_ADJUSTMENT to begin with —
# the entries below and NEVER_ADJUST are the belt-and-braces enforcement.
#
# Same mechanism as EXPOSURE_STRIP above, applied in the same place, so there is
# exactly one code path that decides an adjustment set.
# ---------------------------------------------------------------------------
# Every outcome-defining timing column, named once. Anything that restricts an
# outcome on time-to-final-discharge is defined by BOTH of these, whichever the
# active rule happens to use.
TIMING_OUTCOME_FIELDS: list[str] = ["finaldischarge_hrs", "finaldischarge_days"]

OUTCOME_DEFINING_STRIP: dict[str, list[str]] = {
    "death_hemorrhagic": ["ais_head", *TIMING_OUTCOME_FIELDS],
    # sensitivity death rungs, in case a future arm is run through this script
    "death_hemorrhagic_headais": ["ais_head"],
    "death_hemorrhagic_missing_incl": ["ais_head", *TIMING_OUTCOME_FIELDS],
    "death_hemorrhagic_dayrule": ["ais_head", *TIMING_OUTCOME_FIELDS],
    "death_hemorrhagic_dayfallback": ["ais_head", *TIMING_OUTCOME_FIELDS],
    "death_hemorrhagic_edincl": ["ais_head", *TIMING_OUTCOME_FIELDS],
    "death_le48h": list(TIMING_OUTCOME_FIELDS),
    "death_le48h_dayrule": list(TIMING_OUTCOME_FIELDS),
    # patient_state_composite = death_hemorrhagic | substantial_transfusion — NOT a
    # deterministic function of ais_head, so ais_head is retained (as for the
    # primary composite). Listed explicitly so the decision is on the record.
    "patient_state_composite": [],
    "major_hemorrhage_event": [],
}

# Outcomes for which the mortality-only adjustment rule applies. Used by asserts
# and by the diagnostics so the strip cannot silently stop happening. Membership
# is UNCHANGED: the additional recovery rungs above get their `ais_head` /
# timing strips from OUTCOME_DEFINING_STRIP (which `adjustment_for` applies
# irrespective of membership here), so widening this set would add no guard —
# only new assert surface on arms this script does not run.
MORTALITY_ONLY_OUTCOMES: frozenset[str] = frozenset(
    {"death_hemorrhagic", "death_hemorrhagic_headais",
     "death_hemorrhagic_missing_incl", "death_le48h", "death_inhosp"}
)
# Columns that may NEVER appear in an adjustment set in ANY run (CONTRACTS §6).
# BOTH timing fields — `finaldischarge_hrs` is the one the v3 outcome is actually
# built from; omitting it while listing only the superseded day field was the
# live hole here.
NEVER_ADJUST: frozenset[str] = frozenset(TIMING_OUTCOME_FIELDS)

# VIF audit set (CONTRACTS §6): prefer unstable_ring + iss; drop ais_head if flagged.
VIF_AUDIT_VARS: list[str] = ["iss", "ais_head", "unstable_ring"]
VIF_THRESHOLD = 5.0

# --- CONTRACTS §13 G3f — the Fix-3 asymmetry guard, at import time -----------
# Fix 3 removes `unstable_ring` from the SCORE (02/04/05). It MUST survive here.
assert "unstable_ring" in CORE_ADJUSTMENT, (
    "CONTRACTS §6 / §13 G3f: `unstable_ring` must remain in 11.CORE_ADJUSTMENT. "
    "Fix 3 (D2) removes it from the SCORE's candidate pool and force-in anchors "
    "(files 02/04/05) ONLY. Analysis B is an association model; dropping the ring "
    "here would leave the SI/SBP dose-response confounded by injury pattern.")
assert "unstable_ring" in VIF_AUDIT_VARS, (
    "CONTRACTS §6 / §13 G3f: `unstable_ring` must remain in 11.VIF_AUDIT_VARS "
    "(the audit prefers unstable_ring + iss and drops ais_head if flagged).")
assert not (set(CORE_ADJUSTMENT) & NEVER_ADJUST), (
    f"CONTRACTS §6: {sorted(set(CORE_ADJUSTMENT) & NEVER_ADJUST)} is outcome-defining "
    "and must never be an adjustment covariate.")

# ---------------------------------------------------------------------------
# Change-point grids + interior restriction + event guard (CONTRACTS §9-H2)
# ---------------------------------------------------------------------------
GRID_SI = np.round(np.arange(0.60, 1.60 + 1e-9, 0.02), 4)
GRID_SBP = np.arange(70.0, 130.0 + 1e-9, 2.0)
INTERIOR_LO_PCT, INTERIOR_HI_PCT = 5.0, 95.0

# PRE-SPECIFIED POWER GUARD — locked SAP §6, CONTRACTS §9-H2 / §13 G5e.
# DO NOT RELAX. v3's mortality-only anchor (`death_hemorrhagic`) carries roughly
# a third of v2's all-cause death events (exact counts: `logs/qc_manifest.json →
# death_variants`), so stratified mortality change-points
# can now fall under this guard. The prescribed response is to report those
# strata DESCRIPTIVE-ONLY (see `stratified_changepoints` / `_demote`), never to
# lower the threshold to manufacture an estimate.
MIN_EVENTS_PER_SIDE = 100
# A stratum needs at least one grid point with >=MIN_EVENTS_PER_SIDE on BOTH
# sides, so it needs >=2x that many events before a search is even attempted.
MIN_EVENTS_STRATUM = 2 * MIN_EVENTS_PER_SIDE
MIN_N_STRATUM = 50

# Spline knot percentiles (Harrell 5/35/65/95, documented; CONTRACTS §7).
KNOT_PCTS = (5.0, 35.0, 65.0, 95.0)

# Per-unit OR increments.
SI_UNIT = 0.1       # per 0.1 SI (higher = worse)
SBP_UNIT = 10.0     # per 10 mmHg (lower = worse -> reported as per-10-lower)

# Pre-specified clinical thresholds (SAP §6; Bonferroni over m=4).
#   SI danger = high -> indicator I(SI >= t).  SBP danger = low -> I(SBP < t).
PRESPEC_THRESHOLDS: dict[str, list[float]] = {"shock_index": [1.0, 1.3], "sbp2": [90.0, 70.0]}
BONFERRONI_M = 4

# Categorical bands (SAP §6): reference SI<0.8 / SBP>=110.
SI_BANDS = [("[0.8,1.0)", 0.8, 1.0), ("[1.0,1.3)", 1.0, 1.3), (">=1.3", 1.3, np.inf)]
SI_BAND_REF = ("<0.8", -np.inf, 0.8)
SBP_BANDS = [("[90,110)", 90.0, 110.0), ("[70,90)", 70.0, 90.0), ("<70", -np.inf, 70.0)]
SBP_BAND_REF = (">=110", 110.0, np.inf)

# ---------------------------------------------------------------------------
# Outcomes (CONTRACTS §4 / §9-J). Composite name unchanged, DEFINITION changed;
# the mortality-only anchor's COLUMN NAME changed from death_inhosp.
# ---------------------------------------------------------------------------
COMPOSITE_OUTCOME = "major_hemorrhage_event"      # received_hc | death_hemorrhagic | subst_tx
MORTALITY_OUTCOME = "death_hemorrhagic"           # death_inhosp & ais_head<4 & fdd<=2
# v2's all-cause arm. CONTRACTS §4 rung 0 — a NAMED SENSITIVITY owned by `13`.
# Present here only so the name is documented, never run.
DEATH_SENSITIVITY_ALLCAUSE = "death_inhosp"
ANALYSIS_OUTCOMES = (COMPOSITE_OUTCOME, MORTALITY_OUTCOME)

# The 4 headline curves. Tags are UNCHANGED (`*_mortality`) so the fig-data
# filenames and render_figures' F7 lookups keep working; only the underlying
# outcome column moved.
RUN_MATRIX = [
    ("shock_index", COMPOSITE_OUTCOME, "si_composite"),
    ("shock_index", MORTALITY_OUTCOME, "si_mortality"),
    ("sbp2", COMPOSITE_OUTCOME, "sbp_composite"),
    ("sbp2", MORTALITY_OUTCOME, "sbp_mortality"),
]

# Component outcomes for the decomposition splines (fig-data F8). The death arm
# is the v3 hemorrhagic component of the composite it decomposes — decomposing
# `major_hemorrhage_event` into an arm that is not one of its own ingredients
# would not be a decomposition.
COMPONENT_OUTCOMES = ["received_hc", MORTALITY_OUTCOME, "substantial_transfusion"]

assert DEATH_SENSITIVITY_ALLCAUSE not in [o for _, o, _ in RUN_MATRIX], (
    "CONTRACTS §4/§9-J: `death_inhosp` is v3's all-cause SENSITIVITY arm (rung 0), "
    "not the mortality-only anchor. The anchor is `death_hemorrhagic`.")
assert DEATH_SENSITIVITY_ALLCAUSE not in COMPONENT_OUTCOMES, (
    "CONTRACTS §4/§9-J: component splines decompose the v3 composite, whose death "
    "ingredient is `death_hemorrhagic`.")

# Exposure vitals imputed by MI (§9-H3).
EXPOSURE_VITALS = ["sbp2", "pulse2", "shock_index"]


# ---------------------------------------------------------------------------
# Logging / hashing / versions
# ---------------------------------------------------------------------------
def _logger() -> logging.Logger:
    lg = logging.getLogger("pelvic_thresholds")
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
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "patsy": patsy.__version__,
    }


def _logistic(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _percentile_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(arr, ALPHA / 2.0)),
        float(np.quantile(arr, 1.0 - ALPHA / 2.0)),
    )


# ---------------------------------------------------------------------------
# Data load + complete-case (CONTRACTS §3 ANALYTIC-COHORT RULE)
# ---------------------------------------------------------------------------
def load_cohort(sample: int | None) -> pd.DataFrame:
    if not os.path.exists(COHORT_CSV):
        raise SystemExit(f"Cohort CSV not found: {COHORT_CSV} (run 01_build_cohort.py).")
    LOGGER.info("Loading cohort %s ...", COHORT_CSV)
    df = pd.read_csv(COHORT_CSV, keep_default_na=True, low_memory=False)
    if "cohort_primary" not in df.columns:
        raise SystemExit("cohort CSV missing cohort_primary (CONTRACTS §3).")
    df = df.loc[df["cohort_primary"] == 1].copy()
    LOGGER.info("Analytic cohort (cohort_primary==1): N=%d", len(df))
    if sample is not None and sample < len(df):
        df = df.sample(n=int(sample), random_state=RANDOM_SEED).reset_index(drop=True)
        LOGGER.info("--sample: downsampled to N=%d (seed=%d)", len(df), RANDOM_SEED)
    # coerce dtypes we rely on
    for c in ["shock_index", "sbp2", "pulse2", "iss", "ais_head", "age", "sex_male", "charlson"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["mechanism_cat", "acs_level", "puf_year"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    # ---- Winsorize shock_index at SI_WINSOR_CAP (2.0), ONCE, at load time so
    #      the capped SI is used EVERYWHERE downstream (4-knot spline, per-0.1-SI
    #      ORs, threshold models, component splines, change-point grid+bootstrap,
    #      MI input, tipping-point). clip(upper=) PRESERVES NaN, so the
    #      missing-exposure count (n_missing_exposure) is unchanged. sbp2 is
    #      already physiologically bounded and is deliberately NOT touched. See
    #      the SI_WINSOR_CAP constant for the full rationale. ----
    if "shock_index" in df.columns:
        raw_max = float(df["shock_index"].max())
        n_over = int((df["shock_index"] > SI_WINSOR_CAP).sum())
        df["shock_index"] = df["shock_index"].clip(upper=SI_WINSOR_CAP)
        LOGGER.info("Winsorized shock_index at %.1f: %d rows capped (raw max %.2f -> %.2f).",
                    SI_WINSOR_CAP, n_over, raw_max, float(df["shock_index"].max()))
    return df


def adjustment_for(exposure: str, outcome: str | None = None) -> list[str]:
    """CORE_ADJUSTMENT with the active exposure + its mediators stripped (§6),
    AND — new in v3 — with any covariate that DEFINES the outcome stripped
    (§6 v3 note / §9-K / SAP D-1).

    Two independent strips, one code path:
      * EXPOSURE_STRIP        — never adjust for the exposure or its mediators.
      * OUTCOME_DEFINING_STRIP — never adjust for a variable that appears in the
        outcome's own definition. On `death_hemorrhagic` this removes `ais_head`
        (it is a conjunct of the outcome). On the composite it removes nothing.

    ``outcome=None`` keeps the pre-v3 behaviour (exposure strip only) so any
    caller that legitimately has no single outcome is unchanged; every analysis
    call site in this script passes the outcome.
    """
    strip = set(EXPOSURE_STRIP.get(exposure, [exposure]))
    if outcome is not None:
        strip |= set(OUTCOME_DEFINING_STRIP.get(outcome, []))
    strip |= NEVER_ADJUST
    adj = [c for c in CORE_ADJUSTMENT if c not in strip]
    if outcome in MORTALITY_ONLY_OUTCOMES:
        # Hard guard: a mortality-only model may not carry ais_head. This is the
        # single most likely silent defect in the v3 repoint — an outcome name
        # gets swapped but the adjustment set does not follow.
        assert "ais_head" not in adj, (
            f"v3-K VIOLATION: `ais_head` survived in the adjustment set for "
            f"mortality-only outcome '{outcome}'. It is a conjunct of that "
            f"outcome's definition; adjusting for it conditions on the outcome "
            f"(CONTRACTS §6 v3 note, SAP D-1).")
    assert not (set(adj) & NEVER_ADJUST), (
        f"CONTRACTS §6: outcome-defining column(s) {sorted(set(adj) & NEVER_ADJUST)} "
        f"reached the adjustment set for '{outcome}'.")
    return adj


def adjustment_stripped(exposure: str, outcome: str | None = None) -> dict[str, list[str]]:
    """What ``adjustment_for`` removed, and why — for the diagnostics log and the
    change-point JSON (CONTRACTS §9-K: 'Log the stripped set per run')."""
    exp_strip = sorted(set(EXPOSURE_STRIP.get(exposure, [exposure])) & set(CORE_ADJUSTMENT))
    out_strip = sorted(set(OUTCOME_DEFINING_STRIP.get(outcome, [])) & set(CORE_ADJUSTMENT))
    return {"exposure_or_mediator": exp_strip,
            "outcome_defining": out_strip,
            "never_adjust": sorted(NEVER_ADJUST)}


def _rhs(adjustment: Sequence[str]) -> str:
    parts = []
    for c in adjustment:
        parts.append(f"C({c})" if c in CATEGORICAL_ADJ else c)
    return " + ".join(parts)


def complete_case(df: pd.DataFrame, exposure: str, outcome: str,
                  adjustment: Sequence[str]) -> pd.DataFrame:
    """Drop rows NaN in any model column (exposure/outcome/adjustment).

    This must match the exact patsy column set or the numpy arrays and the
    design matrix will misalign (learned the hard way).
    """
    cols = [outcome, exposure] + list(adjustment)
    cols = [c for c in cols if c in df.columns]
    return df.dropna(subset=cols).copy()


# ---------------------------------------------------------------------------
# §9-H1  HC1 robust logistic (mirrors pelvic_analysis.fit_logistic_robust)
# ---------------------------------------------------------------------------
def _assert_converged(res, ctx: str, check_cov: bool = False) -> None:
    """Loud guard (fix 2026-07-03, SAP §11): a fit whose REPORTED quantities would
    be NaN MUST raise — never silently propagate NaN into the outputs (the prior
    run wrote chi2=NaN + blank CIs from the raw-shock_index tail). Always requires
    a finite log-likelihood (a non-finite llf → NaN LR chi2). When ``check_cov``
    is set (the HC1 point models — the 4 splines, the threshold / per-unit / band
    ORs), ALSO requires a fully finite parameter covariance, which is exactly what
    the reported CIs and spline bands are built from.

    Why ``cov_params`` and NOT ``bse``: HC1's sandwich can carry a slightly
    NEGATIVE variance on a diagonal cell (a benign finite-sample quirk with
    collinear spline bases) → sqrt → NaN in ``bse`` — YET the full covariance is
    finite and the reference-covariate prediction variance x'Vx stays positive &
    finite. (Verified: the CLEAN SBP splines carry NaN ``bse`` on the sbp2 terms
    but 0 non-finite cov cells and finite bands.) The genuine raw-SI poisoning
    instead makes EVERY cov cell non-finite (576/576) → NaN bands. cov_params
    finiteness is therefore the exact discriminator — 0 non-finite cells for every
    clean / winsorized curve, all cells non-finite for the raw-SI defect.

    Why the statsmodels ``converged`` flag is NOT treated as fatal: the large
    spline×stratum INTERACTION models legitimately hit ``maxiter`` with
    ``converged=False`` yet a FINITE, at-optimum llf (IRLS is monotone in llf), and
    the prior clean run computed valid finite LR chi2 from exactly these fits (e.g.
    the SBP interaction LRs). Gating on ``converged`` alone would nuke those valid
    LRs (a regression) and would discard the now-finite winsorized-SI interaction
    LR. Finite llf (+ finite cov for the HC1 point models) is the true signal that
    the reported output is well-defined; ``converged`` is kept only for the error
    message."""
    conv = getattr(res, "converged", None)
    llf = float(getattr(res, "llf", np.nan))
    if not np.isfinite(llf):
        raise RuntimeError(
            f"non-finite log-likelihood GLM fit [{ctx}]: converged={conv}, "
            f"llf={llf} (→ NaN LR chi2). Refusing to write NaN (winsorization / "
            "convergence guard)."
        )
    if check_cov:
        cov = np.asarray(res.cov_params())
        n_bad = int(np.sum(~np.isfinite(cov)))
        if n_bad > 0:
            raise RuntimeError(
                f"non-finite parameter covariance ({n_bad}/{cov.size} cells) [{ctx}]: "
                "the reported CIs / spline bands would be NaN. Refusing to write NaN "
                "(winsorization / convergence guard)."
            )


def fit_hc1(formula: str, data: pd.DataFrame):
    """smf GLM Binomial with HC1 heteroskedasticity-robust SE (no clustering).
    Guarded: raises on non-convergence, non-finite llf, or a non-finite parameter
    covariance (→ NaN CIs / spline bands); no silent NaN outputs."""
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
    res = model.fit(cov_type="HC1", maxiter=300)
    _assert_converged(res, f"fit_hc1: {formula[:80]}", check_cov=True)
    return res


def lr_test(data: pd.DataFrame, full_formula: str, reduced_formula: str) -> dict[str, float]:
    """Nested LR test on the working-independence likelihood (robust cov does not
    change the LR statistic). statsmodels (reliable df_model rank + convergence);
    used for spline overall + non-linearity + spline x stratum interaction."""
    full = smf.glm(full_formula, data, family=sm.families.Binomial()).fit(maxiter=300)
    _assert_converged(full, f"lr_test full: {full_formula[:80]}")
    red = smf.glm(reduced_formula, data, family=sm.families.Binomial()).fit(maxiter=300)
    _assert_converged(red, f"lr_test reduced: {reduced_formula[:80]}")
    chi2 = 2.0 * (full.llf - red.llf)
    dfree = int(round(full.df_model - red.df_model))
    p = float(1.0 - st.chi2.cdf(chi2, df=max(dfree, 1)))
    return {"chi2": float(chi2), "df": int(dfree), "p": p}


# ---------------------------------------------------------------------------
# §6.1 Adjusted 4-knot restricted cubic spline + LR tests + prediction (HC1)
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
    """Predicted risk + 95% CI along ``grid`` (covariates at reference).

    Uses statsmodels ``get_prediction`` so the formula (incl. the cr() spline
    basis) is re-applied to new data exactly and the HC1 robust covariance is
    propagated to the CI on the response scale.
    """
    ref = _reference_row(data, exposure, adjustment)
    newdata = pd.DataFrame({exposure: grid})
    for k, v in ref.items():
        newdata[k] = v
    pr = fit.get_prediction(newdata)
    sf = pr.summary_frame(alpha=ALPHA)
    mean = sf["mean"].to_numpy()
    lo = sf["mean_ci_lower"].to_numpy()
    hi = sf["mean_ci_upper"].to_numpy()
    return pd.DataFrame({
        "exposure_grid": grid,
        "pred_risk": mean,
        "lci": np.minimum(lo, hi),
        "uci": np.maximum(lo, hi),
    })


def spline_grid_for(exposure: str, data: pd.DataFrame) -> np.ndarray:
    s = data[exposure].dropna()
    lo, hi = np.percentile(s, 1.0), np.percentile(s, 99.0)
    step = 0.02 if exposure == "shock_index" else 2.0
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    return np.round(np.arange(lo, hi + step / 2, step), 4)


# ---------------------------------------------------------------------------
# §9-H2  Segmented-logistic change-point (FULL adjusted refit; fast fitter)
#
# PERFORMANCE (coordinator-directed optimization, 2026-07-02):
#   The breakpoint is selected by MINIMUM DEVIANCE, which is identical
#   regardless of covariance type -> the inner grid/bootstrap fits drop
#   ``cov_type="HC1"`` entirely (HC1 only affects SEs, which are NOT used to
#   pick the breakpoint). HC1 is reserved for the FINAL reported point models
#   (the 4 splines, per-unit ORs, threshold ORs, g-computed RDs). The inner
#   fits use sklearn ``LogisticRegression(penalty=None, solver="newton-cholesky")``
#   (validated to return the IDENTICAL coefficients & breakpoint as the
#   statsmodels GLM at ~6-8x lower cost), and the 1000 row-level bootstrap
#   iterations are parallelized across CPU cores with joblib. Each grid point
#   is a FULL adjusted refit (adjustment coefficients re-estimated per
#   breakpoint and per bootstrap resample) — the science is preserved exactly;
#   only the computation is faster.
# ---------------------------------------------------------------------------
def build_adj_design_ni(data: pd.DataFrame, adjustment: Sequence[str]) -> np.ndarray:
    """patsy adjustment design WITHOUT the intercept (sklearn adds its own),
    rows aligned to ``data`` (assumed already complete-case on these columns)."""
    X = patsy.dmatrix("~ " + _rhs(adjustment), data, return_type="dataframe")
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    return X.to_numpy().astype(float)


def _seg_deviance(Xadj_ni: np.ndarray, x: np.ndarray, y: np.ndarray,
                  bp: float) -> tuple[float, float, float]:
    """Full adjusted segmented logistic at breakpoint ``bp`` via sklearn
    newton-cholesky (no HC1 — deviance is cov-type invariant). Returns
    (deviance, beta_pre, beta_post) where the exposure enters as
    b_pre*x + b_delta*max(0,x-bp); beta_post = b_pre + b_delta."""
    X = np.column_stack([Xadj_ni, x, np.maximum(x - bp, 0.0)])
    m = LogisticRegression(penalty=None, solver="newton-cholesky",
                           max_iter=100).fit(X, y)
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
    """Deviance-minimizing breakpoint over ``grid`` (already interior-restricted)
    with a >=MIN_EVENTS_PER_SIDE guard. Full adjusted refit at each grid point.

    v3: also reports ``max_min_events_per_side`` — the best both-sides event count
    any grid point achieved — so a search that returns NaN can be attributed
    (guard-blocked vs every fit failing) instead of silently vanishing. The guard
    itself is unchanged and is NOT relaxed (locked SAP §6).
    """
    best = {"breakpoint": float("nan"), "deviance": float("inf"),
            "beta_pre": float("nan"), "beta_post": float("nan"), "n_valid_grid": 0}
    n_valid = 0
    n_fit_ok = 0
    best_min_side = 0
    for bp in grid:
        ev_lo = int(y[x < bp].sum())
        ev_hi = int(y[x >= bp].sum())
        best_min_side = max(best_min_side, min(ev_lo, ev_hi))
        if min(ev_lo, ev_hi) < MIN_EVENTS_PER_SIDE:
            continue
        n_valid += 1
        try:
            dev, b_pre, b_post = _seg_deviance(Xadj_ni, x, y, float(bp))
        except Exception:  # noqa: BLE001
            continue
        n_fit_ok += 1
        if dev < best["deviance"]:
            best = {"breakpoint": float(bp), "deviance": float(dev),
                    "beta_pre": b_pre, "beta_post": b_post, "n_valid_grid": 0}
    best["n_valid_grid"] = n_valid
    best["n_grid_fits_ok"] = n_fit_ok
    best["n_grid_points"] = int(grid.size)
    best["max_min_events_per_side"] = int(best_min_side)
    best["guard_blocked"] = bool(n_valid == 0)
    return best


def guard_verdict(res: dict[str, Any], n: int, events: int,
                  context: str) -> dict[str, Any] | None:
    """Decide whether a change-point must be reported DESCRIPTIVE-ONLY.

    Locked SAP §6 / CONTRACTS §13 G5e. Returns ``None`` when the change-point is
    estimable, otherwise a structured demotion record. THREE failure modes are
    distinguished, because they mean different things in the manuscript:

      pre_screen  — the stratum has < MIN_EVENTS_STRATUM events (or < MIN_N_STRATUM
                    rows) in total, so no split can put >=100 events on both sides.
      guard       — events exist but NO interior grid point clears
                    >=MIN_EVENTS_PER_SIDE on both sides (mass piled on one side).
      fit_failure — grid points cleared the guard but every adjusted refit failed.

    In all three cases the correct report is a description of the stratum, not an
    estimated inflection point. The guard is never relaxed to avoid this branch.
    """
    if res.get("guard_blocked"):
        return {"descriptive_only": True,
                "guard_reason": "guard_no_grid_point_meets_min_events_per_side",
                "min_events_per_side_required": MIN_EVENTS_PER_SIDE,
                "max_min_events_per_side_achieved": res.get("max_min_events_per_side", 0),
                "n": int(n), "events": int(events),
                "n_grid_points": res.get("n_grid_points", 0),
                "context": context,
                "reporting_rule": ("locked SAP §6 / CONTRACTS §13 G5e — report this "
                                   "stratum DESCRIPTIVE-ONLY; do not quote an "
                                   "inflection point.")}
    if not np.isfinite(res.get("breakpoint", float("nan"))):
        return {"descriptive_only": True,
                "guard_reason": "fit_failure_at_every_valid_grid_point",
                "min_events_per_side_required": MIN_EVENTS_PER_SIDE,
                "n_valid_grid": res.get("n_valid_grid", 0),
                "n_grid_fits_ok": res.get("n_grid_fits_ok", 0),
                "n": int(n), "events": int(events),
                "context": context,
                "reporting_rule": ("locked SAP §6 / CONTRACTS §13 G5e — report this "
                                   "stratum DESCRIPTIVE-ONLY.")}
    return None


def _boot_one_changepoint(Xadj_ni: np.ndarray, x: np.ndarray, y: np.ndarray,
                          grid: np.ndarray, seed: int) -> float:
    """One row-level bootstrap resample -> breakpoint (module-level for joblib)."""
    rng = np.random.default_rng(seed)
    take = rng.choice(x.shape[0], size=x.shape[0], replace=True)
    return changepoint_search(Xadj_ni[take], x[take], y[take], grid)["breakpoint"]


def bootstrap_changepoint(Xadj_ni: np.ndarray, x: np.ndarray, y: np.ndarray,
                          grid: np.ndarray, n_boot: int,
                          seed: int = RANDOM_SEED,
                          n_jobs: int = -1) -> tuple[float, float, list[float]]:
    """Row-level nonparametric bootstrap of the change-point, FULL adjusted
    refit per resample, parallelized across cores (joblib). Per-iter seed
    ``seed + i`` keeps it reproducible (loky memmaps the read-only arrays)."""
    samples = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
        delayed(_boot_one_changepoint)(Xadj_ni, x, y, grid, seed + i)
        for i in range(int(n_boot))
    )
    lo, hi = _percentile_ci(samples)
    return lo, hi, list(samples)


def _or_per_unit_from_slope(beta: float, unit: float) -> float:
    return float(np.exp(beta * unit))


# ---------------------------------------------------------------------------
# §9-H3  Multiple imputation (outcome-in-model + exposure imputed) + Rubin pool
# ---------------------------------------------------------------------------
def _mi_prepare(df_full: pd.DataFrame, exposure: str, outcome: str,
                adjustment: Sequence[str]) -> dict[str, Any]:
    """Shared MI prep done ONCE (holds a single trimmed cohort copy). Congenial
    imputation: the outcome is IN the model and the exposure vitals ARE imputed.
    Per-imputation frames are generated one-at-a-time by ``_mi_impute_one`` and
    discarded by the caller -> the m=20 imputations never coexist in RAM (the OOM
    fix). Categorical adjustment cols are mode-filled (≈100% complete)."""
    base = df_full.copy()
    for c in [x for x in adjustment if x in CATEGORICAL_ADJ]:
        if base[c].isna().any():
            mm = base[c].mode(dropna=True)
            base[c] = base[c].fillna(mm.iloc[0] if len(mm) else "Unknown")
    numeric_adj = [c for c in adjustment if c not in CATEGORICAL_ADJ and c in base.columns]
    vitals = [v for v in EXPOSURE_VITALS if v in base.columns]
    block_cols = list(dict.fromkeys(vitals + numeric_adj + [outcome]))
    binary_cols = {"sex_male", "unstable_ring", "cc_anticoag", outcome}
    keep_cols = [c for c in dict.fromkeys(vitals + [outcome] + list(adjustment))
                 if c in base.columns]
    prep = {"base_keep": base[keep_cols].copy(),
            "num_matrix": base[block_cols].astype(float),
            "block_cols": block_cols, "binary_cols": binary_cols,
            "keep_cols": keep_cols, "outcome": outcome}
    del base
    return prep


def _mi_impute_one(prep: dict[str, Any], i: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate ONE imputed frame (iteration i); caller discards it before i+1."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    imp = IterativeImputer(sample_posterior=True, random_state=seed + i, max_iter=5)
    filled = imp.fit_transform(prep["num_matrix"])
    filled_df = pd.DataFrame(filled, columns=prep["block_cols"],
                             index=prep["base_keep"].index)
    fi = prep["base_keep"].copy()
    for c in prep["block_cols"]:
        if c == prep["outcome"] or c not in prep["keep_cols"]:
            continue  # outcome fully observed here; never overwrite it
        col = filled_df[c]
        if c in prep["binary_cols"]:
            col = col.round().clip(0, 1)
        elif c in ("sbp2", "pulse2"):
            col = col.clip(lower=1.0)
        elif c == "shock_index":
            # cap imputed SI at SI_WINSOR_CAP too (consistency with the observed
            # winsorization; keeps a posterior draw from re-introducing the
            # extreme-leverage tail into the MI change-point fits).
            col = col.clip(lower=0.01, upper=SI_WINSOR_CAP)
        fi[c] = col
    return fi


def _rubin_pool(estimates: list[float], variances: list[float],
                m: int) -> tuple[float, float]:
    """Rubin's rules -> (pooled_estimate, pooled_SE)."""
    bar = float(np.mean(estimates))
    w = float(np.mean(variances)) if variances else 0.0
    b = float(np.var(estimates, ddof=1)) if m > 1 else 0.0
    se = float(np.sqrt(max(w + (1.0 + 1.0 / m) * b, 0.0)))
    return bar, se


def run_mi(df_full: pd.DataFrame, exposure: str, outcome: str,
           adjustment: Sequence[str], grid_full: np.ndarray,
           thresholds: Sequence[float], m: int, n_boot_mi: int,
           within_var_cc: float, seed: int = RANDOM_SEED) -> tuple[dict[str, Any], dict[float, Any]]:
    """SINGLE-PASS, memory-safe MI (one imputed frame at a time; the OOM fix).

    Per imputation i: generate frame -> change-point LOCATION (Rubin-pooled;
    §9-H3) + optional per-imputation within-boot + threshold g-computation for
    each pre-specified threshold (Rubin-pooled RD & OR) -> DISCARD the frame.
    Never holds all m frames in RAM. Congenial: outcome-in-model, exposure
    vitals imputed. Returns (cp_location_pool, {threshold: rd/or_pool})."""
    prep = _mi_prepare(df_full, exposure, outcome, adjustment)
    cp_locs: list[float] = []
    cp_within: list[float] = []
    thr_acc = {t: {"rd": [], "rd_var": [], "rd_d": [], "rd_s": [], "lor": [], "lor_var": []}
               for t in thresholds}
    for i in range(int(m)):
        fi = _mi_impute_one(prep, i, seed)
        cc = fi.dropna(subset=[outcome, exposure] + list(adjustment))
        x = cc[exposure].to_numpy(float)
        y = cc[outcome].to_numpy(float)
        g, _, _ = interior_grid(x, grid_full)
        if g.size:
            try:
                Xadj_ni = build_adj_design_ni(cc, adjustment)
                res = changepoint_search(Xadj_ni, x, y, g)
                if np.isfinite(res["breakpoint"]):
                    cp_locs.append(res["breakpoint"])
                    if n_boot_mi > 0:
                        _, _, samp = bootstrap_changepoint(Xadj_ni, x, y, g, n_boot_mi,
                                                           seed=seed + 100 * (i + 1))
                        arr = np.array([s for s in samp if s is not None and np.isfinite(s)])
                        if arr.size > 1:
                            cp_within.append(float(np.var(arr, ddof=1)))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("MI CP i=%d failed: %s", i, exc)
        for t in thresholds:
            try:
                gg = threshold_gcomp(cc, exposure, outcome, float(t), adjustment)
                a = thr_acc[t]
                a["rd"].append(gg["adj_risk_diff"])
                a["rd_var"].append(((gg["rd_uci"] - gg["rd_lci"]) / (2 * Z)) ** 2)
                a["rd_d"].append(gg["adj_risk_danger"])
                a["rd_s"].append(gg["adj_risk_safe"])
                a["lor"].append(np.log(gg["adj_or"]))
                a["lor_var"].append(((np.log(gg["or_uci"]) - np.log(gg["or_lci"])) / (2 * Z)) ** 2)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("MI thr t=%s i=%d failed: %s", t, i, exc)
        del fi, cc

    # ---- pool the change-point location (Rubin) ----
    if cp_locs:
        m_ok = len(cp_locs)
        theta = float(np.mean(cp_locs))
        between = float(np.var(cp_locs, ddof=1)) if m_ok > 1 else 0.0
        if n_boot_mi > 0 and cp_within:
            within = float(np.mean(cp_within))
            wm = f"per-imputation row bootstrap (B={n_boot_mi}), averaged over m"
        else:
            within = float(within_var_cc) if np.isfinite(within_var_cc) else 0.0
            wm = "complete-case row-bootstrap variance (low-missingness approx)"
        se = float(np.sqrt(max(within + (1.0 + 1.0 / m_ok) * between, 0.0)))
        cp_pool = {"location": round(theta, 4), "lci": round(theta - Z * se, 4),
                   "uci": round(theta + Z * se, 4), "m_ok": m_ok,
                   "n_boot_mi": int(n_boot_mi), "within_var": round(within, 6),
                   "within_method": wm, "between_var": round(between, 6),
                   "per_imputation_locations": [round(v, 4) for v in cp_locs]}
    else:
        cp_pool = {"location": float("nan"), "lci": float("nan"), "uci": float("nan"),
                   "m_ok": 0, "within_method": "no valid imputation change-point"}

    # ---- pool each threshold (Rubin RD & OR) ----
    thr_pool: dict[float, Any] = {}
    for t in thresholds:
        a = thr_acc[t]
        if not a["rd"]:
            thr_pool[t] = {}
            continue
        mm = len(a["rd"])
        rd_bar, rd_se = _rubin_pool(a["rd"], a["rd_var"], mm)
        lor_bar, lor_se = _rubin_pool(a["lor"], a["lor_var"], mm)
        thr_pool[t] = {
            "adj_risk_danger": float(np.mean(a["rd_d"])),
            "adj_risk_safe": float(np.mean(a["rd_s"])),
            "adj_risk_diff": rd_bar, "rd_lci": rd_bar - Z * rd_se, "rd_uci": rd_bar + Z * rd_se,
            "adj_or": float(np.exp(lor_bar)), "or_lci": float(np.exp(lor_bar - Z * lor_se)),
            "or_uci": float(np.exp(lor_bar + Z * lor_se)), "m_ok": mm,
        }
    return cp_pool, thr_pool


# ---------------------------------------------------------------------------
# §9-H2 part 4  Tipping-point bound (MNAR)
# ---------------------------------------------------------------------------
def tipping_point_bound(df_full: pd.DataFrame, exposure: str, outcome: str,
                        adjustment: Sequence[str], grid_full: np.ndarray) -> dict[str, Any]:
    """Refit the change-point assigning ALL missing-exposure rows to the worst-tail
    vs the best-tail; report the two breakpoints (the range). Worst-tail: SI->
    high (grid max), SBP-> low (grid min). Best-tail: the opposite. Missing rows
    also need a complete adjustment row; drop rows missing adjustment covariates."""
    worst = np.nanmax(grid_full) if exposure == "shock_index" else np.nanmin(grid_full)
    best = np.nanmin(grid_full) if exposure == "shock_index" else np.nanmax(grid_full)
    out: dict[str, Any] = {"exposure": exposure, "outcome": outcome,
                           "n_missing_exposure": int(df_full[exposure].isna().sum())}
    for tag, fillval in (("worst_tail", worst), ("best_tail", best)):
        d = df_full.copy()
        d[exposure] = d[exposure].fillna(float(fillval))
        d = d.dropna(subset=[outcome, exposure] + list(adjustment)).copy()
        x = d[exposure].to_numpy(float)
        y = d[outcome].to_numpy(float)
        g, ilo, ihi = interior_grid(x, grid_full)
        try:
            Xadj_ni = build_adj_design_ni(d, adjustment)
            res = changepoint_search(Xadj_ni, x, y, g)
            out[f"{tag}_bp"] = res["breakpoint"]
            out[f"{tag}_fill"] = float(fillval)
        except Exception as exc:  # noqa: BLE001
            out[f"{tag}_bp"] = float("nan")
            out[f"{tag}_fill"] = float(fillval)
            LOGGER.debug("tipping %s fit failed: %s", tag, exc)
    bps = [out.get("worst_tail_bp"), out.get("best_tail_bp")]
    bps = [b for b in bps if b is not None and np.isfinite(b)]
    out["bound_lo"] = float(min(bps)) if bps else float("nan")
    out["bound_hi"] = float(max(bps)) if bps else float("nan")
    out["bound_range"] = (out["bound_hi"] - out["bound_lo"]) if bps else float("nan")
    return out


# ---------------------------------------------------------------------------
# §6.2  Pre-specified thresholds: adjusted OR + g-computed absolute risk + RD
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
    side (ind=1) vs safe side (ind=0) + adjusted OR. Delta-method RD 95% CI on the
    HC1 covariance (marginal standardization over the observed covariate dist.)."""
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
    # delta-method RD variance
    n = X0.shape[0]
    g = (X1.T @ (p1 * (1 - p1))) / n - (X0.T @ (p0 * (1 - p0))) / n
    cov = np.asarray(fit.cov_params())
    se_rd = float(np.sqrt(max(float(g @ cov @ g), 0.0)))
    rd = r_danger - r_safe
    # adjusted OR for the indicator
    beta = float(fit.params["_ind"])
    se_b = float(fit.bse["_ind"])
    p_raw = float(fit.pvalues["_ind"])
    return {
        "adj_risk_danger": r_danger,
        "adj_risk_safe": r_safe,
        "adj_risk_diff": rd,
        "rd_lci": rd - Z * se_rd,
        "rd_uci": rd + Z * se_rd,
        "adj_or": float(np.exp(beta)),
        "or_lci": float(np.exp(beta - Z * se_b)),
        "or_uci": float(np.exp(beta + Z * se_b)),
        "p_raw": p_raw,
        "n_danger": int((d["_ind"] == 1).sum()),
        "n_safe": int((d["_ind"] == 0).sum()),
    }


def per_unit_or(data: pd.DataFrame, exposure: str, outcome: str,
                adjustment: Sequence[str]) -> dict[str, Any]:
    """Per-0.1-SI (higher) / per-10-mmHg (lower) adjusted OR (linear exposure, HC1)."""
    fit = fit_hc1(linear_formula(exposure, outcome, adjustment), data)
    beta = float(fit.params[exposure])
    se = float(fit.bse[exposure])
    if exposure == "shock_index":
        unit, label = SI_UNIT, "per +0.1 SI"
    else:
        unit, label = -SBP_UNIT, "per -10 mmHg (lower)"
    b1 = float(np.exp((beta - Z * se) * unit))
    b2 = float(np.exp((beta + Z * se) * unit))
    return {
        "term": label,
        "adj_or": float(np.exp(beta * unit)),
        "lci": min(b1, b2),  # unit<0 for SBP flips the bound order
        "uci": max(b1, b2),
        "p_raw": float(fit.pvalues[exposure]),
    }


def band_ors(data: pd.DataFrame, exposure: str, outcome: str,
             adjustment: Sequence[str]) -> list[dict[str, Any]]:
    """Categorical band ORs vs reference (SI<0.8 / SBP>=110)."""
    d = data.copy()
    if exposure == "shock_index":
        ref = SI_BAND_REF
        bands = SI_BANDS
    else:
        ref = SBP_BAND_REF
        bands = SBP_BANDS
    order = [ref] + bands
    labels = [b[0] for b in order]

    def _assign(v):
        for lbl, lo, hi in order:
            if (v >= lo) and (v < hi):
                return lbl
        return labels[0]

    d["_band"] = d[exposure].apply(_assign)
    d["_band"] = pd.Categorical(d["_band"], categories=labels, ordered=True)
    formula = f"{outcome} ~ C(_band, Treatment(reference='{labels[0]}')) + {_rhs(adjustment)}"
    fit = fit_hc1(formula, d)
    rows = []
    for lbl in labels[1:]:
        key = [k for k in fit.params.index if k.endswith(f"[T.{lbl}]") and "_band" in k]
        if not key:
            continue
        k = key[0]
        beta, se = float(fit.params[k]), float(fit.bse[k])
        rows.append({
            "term": f"{exposure} band {lbl} vs {labels[0]}",
            "adj_or": float(np.exp(beta)),
            "lci": float(np.exp(beta - Z * se)),
            "uci": float(np.exp(beta + Z * se)),
            "p_raw": float(fit.pvalues[k]),
            "n": int((d["_band"] == lbl).sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# §6.3  Stratified change-points + spline x stratum interaction LR
# ---------------------------------------------------------------------------
STRATUM_DEFS = [
    ("unstable_ring", "unstable_ring", 1, 0),   # (col, name, level_hi, level_lo)
    ("age_ge65", "age_ge65", 1, 0),
    ("cc_anticoag", "cc_anticoag", 1, 0),
]
# labels for the two levels of each stratum var
STRATUM_LEVEL_LABELS = {
    "unstable_ring": {1: "unstable_ring", 0: "stable_ring"},
    "age_ge65": {1: "elderly", 0: "young"},
    "cc_anticoag": {1: "anticoag", 0: "no_anticoag"},
}


def stratified_changepoints(df_cc: pd.DataFrame, exposure: str, outcome: str,
                            adjustment: Sequence[str], grid_full: np.ndarray,
                            n_boot: int, tag: str = "") -> dict[str, Any]:
    """Change-point within each stratum + spline x stratum interaction-LR test.
    Bootstrap CIs for the headline strata (elderly/young/elderly∩anticoag);
    point + descriptive for the rest.

    v3 POWER GUARD (locked SAP §6, CONTRACTS §13 G5e): a stratum that cannot
    support the pre-specified >=100-events-per-side change-point guard is
    DEMOTED to descriptive-only — it keeps its n, its event count, and an
    explicit ``descriptive_only``/``guard_reason`` pair, and it carries NO
    breakpoint. With `death_hemorrhagic` carrying roughly a third of v2's
    all-cause death events (exact counts: `logs/qc_manifest.json →
    death_variants`) this is expected to bite on the mortality curves,
    elderly∩anticoagulated first. Demotions are collected in
    ``result['descriptive_only']`` and logged as WARNINGs so they cannot pass
    unnoticed into the manuscript as if they were estimates. The guard is NOT
    relaxed.
    """
    result: dict[str, Any] = {}
    demoted: list[dict[str, Any]] = []

    # --- stratum change-points (single-var strata) ---
    strata: list[tuple[str, pd.Series]] = []
    for col, name, hi, lo in STRATUM_DEFS:
        strata.append((STRATUM_LEVEL_LABELS[name][hi], df_cc[col] == hi))
        strata.append((STRATUM_LEVEL_LABELS[name][lo], df_cc[col] == lo))
    # elderly∩anticoag (pre-registered headline)
    strata.append(("elderly_anticoag", (df_cc["age_ge65"] == 1) & (df_cc["cc_anticoag"] == 1)))
    strata.append(("young_or_noac", ~((df_cc["age_ge65"] == 1) & (df_cc["cc_anticoag"] == 1))))

    # Bootstrap CIs only for the headline strata (age modification is the
    # pre-registered mechanism + the "non-overlapping CI" gate; the interaction
    # LR is the formal test for the null strata, so ring/anticoag get point +
    # interaction-LR only, keeping the 4-curve x stratified x 1000-boot tractable).
    boot_strata = {"elderly", "young", "elderly_anticoag"}
    cp_out: dict[str, Any] = {}
    for label, mask in strata:
        sub = df_cc.loc[mask]
        # drop the stratum's own covariate from adjustment to avoid collinearity
        adj = list(adjustment)
        if label in ("unstable_ring", "stable_ring"):
            adj = [c for c in adj if c != "unstable_ring"]
        if label in ("anticoag", "no_anticoag") or label in ("elderly_anticoag", "young_or_noac"):
            adj = [c for c in adj if c != "cc_anticoag"]
        sub = sub.dropna(subset=[outcome, exposure] + adj).copy()
        x = sub[exposure].to_numpy(float)
        y = sub[outcome].to_numpy(float)
        n = int(len(sub))
        ev = int(y.sum())
        ctx = f"{tag or exposure}|{outcome}|{label}"

        # --- power guard, pre-screen: too few events for ANY qualifying split ---
        if n < MIN_N_STRATUM or ev < MIN_EVENTS_STRATUM:
            dem = {"descriptive_only": True,
                   "guard_reason": ("pre_screen_insufficient_events"
                                    if ev < MIN_EVENTS_STRATUM else "pre_screen_insufficient_n"),
                   "min_events_per_side_required": MIN_EVENTS_PER_SIDE,
                   "min_events_stratum_required": MIN_EVENTS_STRATUM,
                   "n": n, "events": ev, "context": ctx,
                   "reporting_rule": ("locked SAP §6 / CONTRACTS §13 G5e — report this "
                                      "stratum DESCRIPTIVE-ONLY; do not quote an "
                                      "inflection point.")}
            cp_out[label] = {"n": n, "events": ev, "breakpoint": float("nan"),
                             "note": f"descriptive_only_<{MIN_EVENTS_STRATUM}_events", **dem}
            demoted.append(dem)
            LOGGER.warning("POWER GUARD: %s DEMOTED to descriptive-only "
                           "(n=%d, events=%d < %d required). SAP §6 / §13 G5e.",
                           ctx, n, ev, MIN_EVENTS_STRATUM)
            continue
        g, ilo, ihi = interior_grid(x, grid_full)
        try:
            Xadj_ni = build_adj_design_ni(sub, adj)
            res = changepoint_search(Xadj_ni, x, y, g)
        except Exception as exc:  # noqa: BLE001
            dem = {"descriptive_only": True, "guard_reason": f"fit_failed: {exc}",
                   "n": n, "events": ev, "context": ctx,
                   "reporting_rule": "locked SAP §6 / CONTRACTS §13 G5e — descriptive-only."}
            cp_out[label] = {"n": n, "events": ev, "breakpoint": float("nan"),
                             "note": f"fit_failed: {exc}", **dem}
            demoted.append(dem)
            LOGGER.warning("POWER GUARD: %s DEMOTED to descriptive-only (fit failed: %s).",
                           ctx, exc)
            continue

        # --- power guard, post-search: no grid point cleared >=100 events/side ---
        verdict = guard_verdict(res, n, ev, ctx)
        if verdict is not None:
            cp_out[label] = {"n": n, "events": ev, "breakpoint": float("nan"),
                             "n_valid_grid": res["n_valid_grid"],
                             "interior_lo": round(ilo, 4), "interior_hi": round(ihi, 4),
                             "note": verdict["guard_reason"], **verdict}
            demoted.append(verdict)
            LOGGER.warning("POWER GUARD: %s DEMOTED to descriptive-only (%s; best both-sides "
                           "event count %s < %d). SAP §6 / §13 G5e — NOT relaxed.",
                           ctx, verdict["guard_reason"],
                           verdict.get("max_min_events_per_side_achieved", "n/a"),
                           MIN_EVENTS_PER_SIDE)
            continue

        entry = {"n": n, "events": ev, "breakpoint": res["breakpoint"],
                 "n_valid_grid": res["n_valid_grid"],
                 "max_min_events_per_side": res.get("max_min_events_per_side"),
                 "descriptive_only": False, "guard_reason": None,
                 "interior_lo": round(ilo, 4), "interior_hi": round(ihi, 4)}
        if label in boot_strata and n_boot > 0:
            # deterministic per-label seed offset (Python's hash() is per-process
            # randomized -> would break seed-42 reproducibility)
            seed_off = int(hashlib.md5(label.encode()).hexdigest(), 16) % 9973
            lo, hi, _ = bootstrap_changepoint(Xadj_ni, x, y, g, n_boot,
                                              seed=RANDOM_SEED + seed_off)
            entry["lci"] = lo
            entry["uci"] = hi
        cp_out[label] = entry
    result["strata"] = cp_out

    # --- spline x stratum interaction LR (age_ge65, cc_anticoag, unstable_ring) ---
    inter: dict[str, Any] = {}
    for col, name, hi, lo in STRATUM_DEFS:
        adj = [c for c in adjustment if c != col]
        try:
            full = (f"{outcome} ~ cr({exposure}, df=4) * C({col}) + {_rhs(adj)}")
            red = (f"{outcome} ~ cr({exposure}, df=4) + C({col}) + {_rhs(adj)}")
            lr = lr_test(df_cc, full, red)
            inter[name] = lr
        except Exception as exc:  # noqa: BLE001
            inter[name] = {"chi2": float("nan"), "df": 0, "p": float("nan"),
                           "note": f"failed: {exc}"}
    result["interaction_lr"] = inter

    # --- directional-hypothesis check (elderly earlier?) ---
    eld = cp_out.get("elderly", {})
    yng = cp_out.get("young", {})
    hyp: dict[str, Any] = {}
    if eld.get("descriptive_only") or yng.get("descriptive_only"):
        # The pre-registered "elderly inflect earlier" test is not evaluable when
        # either side was demoted. Say so explicitly rather than emitting an
        # empty dict that reads as "not tested".
        hyp["evaluable"] = False
        hyp["reason"] = ("one or both age strata are descriptive-only under the "
                         ">=100-events-per-side guard (SAP §6 / §13 G5e): "
                         f"elderly={eld.get('guard_reason')}, young={yng.get('guard_reason')}")
        hyp["elderly_events"] = eld.get("events")
        hyp["young_events"] = yng.get("events")
    elif np.isfinite(eld.get("breakpoint", np.nan)) and np.isfinite(yng.get("breakpoint", np.nan)):
        hyp["evaluable"] = True
        if exposure == "shock_index":
            # earlier = LOWER SI for elderly
            hyp["direction_supported"] = bool(eld["breakpoint"] < yng["breakpoint"])
            hyp["hypothesis"] = "elderly inflect at LOWER SI"
        else:
            # earlier = HIGHER SBP for elderly
            hyp["direction_supported"] = bool(eld["breakpoint"] > yng["breakpoint"])
            hyp["hypothesis"] = "elderly inflect at HIGHER SBP"
        hyp["elderly_bp"] = eld["breakpoint"]
        hyp["young_bp"] = yng["breakpoint"]
        if "lci" in eld and "uci" in yng:
            # non-overlap of the two CIs
            hyp["ci_nonoverlap"] = bool(
                (eld.get("lci", np.nan) > yng.get("uci", np.nan))
                or (eld.get("uci", np.nan) < yng.get("lci", np.nan))
            )
    else:
        hyp["evaluable"] = False
        hyp["reason"] = "no finite breakpoint in one or both age strata"
    result["directional_hypothesis"] = hyp

    # --- power-guard ledger for this curve (SAP §6 / CONTRACTS §13 G5e) ---
    result["descriptive_only"] = demoted
    result["descriptive_only_labels"] = sorted(
        {lbl for lbl, e in cp_out.items() if isinstance(e, dict) and e.get("descriptive_only")})
    result["min_events_per_side"] = MIN_EVENTS_PER_SIDE
    result["min_events_stratum"] = MIN_EVENTS_STRATUM
    if result["descriptive_only_labels"]:
        LOGGER.warning("POWER GUARD SUMMARY [%s|%s]: %d/%d strata are DESCRIPTIVE-ONLY -> %s",
                       tag or exposure, outcome, len(result["descriptive_only_labels"]),
                       len(cp_out), result["descriptive_only_labels"])
    return result


# ---------------------------------------------------------------------------
# §6.1 helper: complete-case bootstrap variance of the change-point (for MI within)
# ---------------------------------------------------------------------------
def _var_from_ci(lci: float, uci: float) -> float:
    if not (np.isfinite(lci) and np.isfinite(uci)):
        return float("nan")
    return ((uci - lci) / (2 * Z)) ** 2


# ---------------------------------------------------------------------------
# Per-curve orchestration
# ---------------------------------------------------------------------------
def run_curve(df: pd.DataFrame, exposure: str, outcome: str, tag: str,
              n_boot: int, mi_m: int, diag: list[str],
              n_boot_strat: int = 250, n_boot_mi: int = 0) -> dict[str, Any]:
    """Run one headline curve: spline (HC1) + change-point (+ boot CI, MI pool,
    tipping) + stratified change-points. Writes spline CSV + changepoint JSON.
    Returns the changepoint dict (for the stratified file + gate summary)."""
    # v3-K: the adjustment set is now a function of the OUTCOME as well as the
    # exposure — `ais_head` is dropped on the mortality-only runs because it is a
    # conjunct of `death_hemorrhagic` (CONTRACTS §6 v3 note / SAP D-1).
    adj = adjustment_for(exposure, outcome)
    stripped = adjustment_stripped(exposure, outcome)
    grid_full = GRID_SI if exposure == "shock_index" else GRID_SBP
    df_cc = complete_case(df, exposure, outcome, adj)
    n_cc = len(df_cc)
    n_ev = int(df_cc[outcome].sum())
    LOGGER.info("[%s] complete-case N=%d events=%d", tag, n_cc, n_ev)
    diag.append(f"\n=== CURVE {tag}: {outcome} ~ {exposure} (adj={adj}) ===")
    diag.append(f"complete-case N={n_cc}, events={n_ev}")
    diag.append(f"adjustment stripped — exposure/mediator: {stripped['exposure_or_mediator']}; "
                f"outcome-defining: {stripped['outcome_defining']}; "
                f"never-adjust: {stripped['never_adjust']}")
    if outcome in MORTALITY_ONLY_OUTCOMES:
        diag.append(f"v3-K: mortality-only run -> `ais_head` REMOVED from CORE_ADJUSTMENT "
                    f"(it is a conjunct of {outcome}); ais_head in adj = "
                    f"{'ais_head' in adj} (must be False)")
        LOGGER.info("[%s] v3-K mortality-only run: ais_head stripped from adjustment "
                    "(outcome-defining). adj=%s", tag, adj)
    diag.append(f"unstable_ring retained in adjustment (§13 G3f): {'unstable_ring' in adj}")

    # ---- Spline (HC1) + LR tests + prediction ----
    knots = tuple(round(float(np.percentile(df_cc[exposure].dropna(), p)), 3) for p in KNOT_PCTS)
    diag.append(f"RCS 4-knot cr({exposure},df=4); Harrell 5/35/65/95 percentiles = {knots}")
    fit_sp = fit_hc1(spline_formula(exposure, outcome, adj), df_cc)
    lr_overall = lr_test(df_cc, spline_formula(exposure, outcome, adj),
                         adjonly_formula(outcome, adj))
    lr_nonlin = lr_test(df_cc, spline_formula(exposure, outcome, adj),
                        linear_formula(exposure, outcome, adj))
    diag.append(f"LR overall: chi2={lr_overall['chi2']:.2f} df={lr_overall['df']} p={lr_overall['p']:.3g}")
    diag.append(f"LR non-linearity: chi2={lr_nonlin['chi2']:.2f} df={lr_nonlin['df']} p={lr_nonlin['p']:.3g}")

    pgrid = spline_grid_for(exposure, df_cc)
    spline_df = predict_spline(fit_sp, df_cc, exposure, adj, pgrid)
    # elderly vs young overlays (fit within-stratum splines; fig-data F7)
    for strat_lbl, mask in (("elderly", df_cc["age_ge65"] == 1), ("young", df_cc["age_ge65"] == 0)):
        sub = df_cc.loc[mask]
        adj_s = [c for c in adj if c != "age_ge65"]
        try:
            fit_s = fit_hc1(spline_formula(exposure, outcome, adj_s), sub)
            pred_s = predict_spline(fit_s, sub, exposure, adj_s, pgrid)
            spline_df[f"pred_risk_{strat_lbl}"] = pred_s["pred_risk"].to_numpy()
            spline_df[f"lci_{strat_lbl}"] = pred_s["lci"].to_numpy()
            spline_df[f"uci_{strat_lbl}"] = pred_s["uci"].to_numpy()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[%s] %s overlay spline failed: %s", tag, strat_lbl, exc)
    spline_df.insert(0, "outcome", outcome)
    spline_df.insert(0, "exposure", exposure)
    spline_path = os.path.join(DERIVED, f"spline_{tag}.csv")
    spline_df.to_csv(spline_path, index=False)
    LOGGER.info("[%s] wrote %s", tag, spline_path)

    # ---- Change-point: point estimate + interior restriction + event guard ----
    x = df_cc[exposure].to_numpy(float)
    y = df_cc[outcome].to_numpy(float)
    g, ilo, ihi = interior_grid(x, grid_full)
    Xadj_ni = build_adj_design_ni(df_cc, adj)
    cp = changepoint_search(Xadj_ni, x, y, g)
    unit = SI_UNIT if exposure == "shock_index" else SBP_UNIT
    # g can be empty if the interior 5-95 pct window admits no grid point; guard the
    # format call so the POWER-GUARD branch below (not a ValueError here) is what
    # reports it.
    g_lo = float(g.min()) if g.size else float("nan")
    g_hi = float(g.max()) if g.size else float("nan")
    diag.append(f"change-point grid interior=[{ilo:.3f},{ihi:.3f}] "
                f"({g_lo:.3f}..{g_hi:.3f}), valid grid pts={cp['n_valid_grid']} "
                f"of {cp['n_grid_points']} (>= {MIN_EVENTS_PER_SIDE} events/side required; "
                f"best achieved={cp['max_min_events_per_side']})")
    diag.append(f"change-point breakpoint={cp['breakpoint']}")

    # --- POWER GUARD on the MAIN curve (locked SAP §6 / CONTRACTS §13 G5e) ---
    # v3's mortality-only anchor is ~29 % of v2's event count. If NO interior grid
    # point clears >=100 events on both sides, the change-point is not estimable
    # and this curve is reported DESCRIPTIVE-ONLY. Short-circuit the 1,000-iteration
    # bootstrap in that case: it would burn a full parallel sweep only to return
    # NaN, and a NaN CI printed beside a NaN point estimate reads like a bug rather
    # than a pre-specified refusal. The guard is NOT relaxed.
    main_verdict = guard_verdict(cp, n_cc, n_ev, context=f"{tag}|MAIN")
    if main_verdict is not None:
        LOGGER.warning("POWER GUARD: MAIN curve %s (%s ~ %s) DEMOTED to descriptive-only "
                       "(%s; N=%d events=%d). Bootstrap skipped. SAP §6 / §13 G5e.",
                       tag, outcome, exposure, main_verdict["guard_reason"], n_cc, n_ev)
        diag.append(f"POWER GUARD: MAIN curve DESCRIPTIVE-ONLY — {main_verdict['guard_reason']} "
                    f"(required >={MIN_EVENTS_PER_SIDE} events/side; best achieved "
                    f"{cp['max_min_events_per_side']}). Change-point NOT reported; the spline, "
                    f"per-unit ORs and threshold ORs for this curve are unaffected.")
        lci = uci = float("nan")
        boot_samples = []
        within_var_cc = float("nan")
    else:
        # bootstrap CI (joblib-parallel row bootstrap; full adjusted refit per resample)
        LOGGER.info("[%s] bootstrapping change-point (%d iters) ...", tag, n_boot)
        lci, uci, boot_samples = bootstrap_changepoint(Xadj_ni, x, y, g, n_boot)
        within_var_cc = _var_from_ci(lci, uci)
    at_boundary = bool(np.isfinite(cp["breakpoint"]) and g.size and
                       (abs(cp["breakpoint"] - g_lo) < 1e-9 or abs(cp["breakpoint"] - g_hi) < 1e-9))
    ci_excludes_endpoints = bool(np.isfinite(lci) and np.isfinite(uci) and g.size
                                 and lci > g_lo + 1e-9 and uci < g_hi - 1e-9)

    # bootstrap breakpoint distribution summary (degenerate-CI diagnostic)
    _bs = np.array([s for s in boot_samples if s is not None and np.isfinite(s)], dtype=float)
    boot_summary = {
        "n_boot_ok": int(_bs.size),
        "boot_bp_min": float(_bs.min()) if _bs.size else float("nan"),
        "boot_bp_max": float(_bs.max()) if _bs.size else float("nan"),
        "boot_bp_p2_5": float(np.percentile(_bs, 2.5)) if _bs.size else float("nan"),
        "boot_bp_p97_5": float(np.percentile(_bs, 97.5)) if _bs.size else float("nan"),
        "boot_bp_n_unique": int(np.unique(_bs).size) if _bs.size else 0,
        "ci_width_lt_grid_resolution": bool(_bs.size and (float(_bs.max()) - float(_bs.min())) < (
            0.02 if exposure == "shock_index" else 2.0) + 1e-9),
    }

    # MI (single-pass, memory-safe): pooled change-point location + threshold RD/OR
    LOGGER.info("[%s] MI (m=%d, one-at-a-time) location + threshold pooling ...", tag, mi_m)
    try:
        mi_loc, mi_thresh = run_mi(df, exposure, outcome, adj, grid_full,
                                   list(PRESPEC_THRESHOLDS[exposure]), mi_m,
                                   n_boot_mi, within_var_cc)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[%s] MI pooling failed: %s", tag, exc)
        mi_loc = {"location": float("nan"), "note": f"MI failed: {exc}"}
        mi_thresh = {}

    # tipping-point bound
    tip = tipping_point_bound(df, exposure, outcome, adj, grid_full)

    cp_json = {
        "exposure": exposure,
        "outcome": outcome,
        "tag": tag,
        # v3-K provenance: the exact adjustment set this run used, and what was
        # taken out of CORE_ADJUSTMENT to get there (CONTRACTS §9-K).
        "adjustment": list(adj),
        "adjustment_stripped": stripped,
        "mortality_only_run": bool(outcome in MORTALITY_ONLY_OUTCOMES),
        "ais_head_in_adjustment": bool("ais_head" in adj),
        "unstable_ring_in_adjustment": bool("unstable_ring" in adj),
        # Power guard (SAP §6 / §13 G5e): None when the change-point is estimable.
        "descriptive_only": bool(main_verdict is not None),
        "guard": main_verdict,
        "n": n_cc,
        "n_events": n_ev,
        "breakpoint": cp["breakpoint"],
        "lci": lci,
        "uci": uci,
        "beta_pre": cp["beta_pre"],
        "beta_post": cp["beta_post"],
        "or_per_unit_pre": (_or_per_unit_from_slope(cp["beta_pre"], unit)
                            if np.isfinite(cp["beta_pre"]) else float("nan")),
        "or_per_unit_post": (_or_per_unit_from_slope(cp["beta_post"], unit)
                             if np.isfinite(cp["beta_post"]) else float("nan")),
        "unit": unit,
        "deviance_at_bp": cp["deviance"],
        "grid_min": float(g.min()) if g.size else float("nan"),
        "grid_max": float(g.max()) if g.size else float("nan"),
        "grid_step": (0.02 if exposure == "shock_index" else 2.0),
        "interior_lo": round(ilo, 4),
        "interior_hi": round(ihi, 4),
        "n_valid_grid": cp["n_valid_grid"],
        "min_events_per_side": MIN_EVENTS_PER_SIDE,
        "n_boot": int(n_boot),
        "at_grid_boundary": at_boundary,
        "ci_excludes_endpoints": ci_excludes_endpoints,
        "bootstrap_distribution": boot_summary,
        "lr_overall": lr_overall,
        "lr_nonlinearity": lr_nonlin,
        "mi_pooled_location": mi_loc,
        "tipping_point": tip,
        "method": ("deviance-minimizing segmented logistic (FULL adjusted refit "
                   "per grid point via sklearn newton-cholesky, no HC1 in the "
                   "inner loop — deviance is cov-type invariant); HC1 robust SE "
                   "reserved for the spline/threshold/per-unit point models; "
                   "joblib-parallel row-level bootstrap CI (1000 iters, full refit "
                   "per resample); interior 5-95pct restriction + >=100 events/side."),
    }
    cp_path = os.path.join(DERIVED, f"changepoint_{tag}.json")
    with open(cp_path, "w") as f:
        json.dump(cp_json, f, indent=2, default=str)
    LOGGER.info("[%s] wrote %s (bp=%s CI[%s,%s])", tag, cp_path,
                round(cp["breakpoint"], 3) if np.isfinite(cp["breakpoint"]) else "nan",
                round(lci, 3) if np.isfinite(lci) else "nan",
                round(uci, 3) if np.isfinite(uci) else "nan")

    # ---- Stratified change-points (attach to the stratified file later) ----
    LOGGER.info("[%s] stratified change-points ...", tag)
    strat = stratified_changepoints(df_cc, exposure, outcome, adj, grid_full,
                                    n_boot_strat, tag=tag)
    strat["adjustment"] = list(adj)
    strat["adjustment_stripped"] = stripped
    strat["outcome"] = outcome
    strat["exposure"] = exposure
    strat["main_curve_descriptive_only"] = bool(main_verdict is not None)
    if strat.get("descriptive_only_labels"):
        diag.append(f"POWER GUARD [{tag}]: descriptive-only strata = "
                    f"{strat['descriptive_only_labels']} "
                    f"(>= {MIN_EVENTS_PER_SIDE} events/side required, "
                    f">= {MIN_EVENTS_STRATUM} events/stratum to attempt). "
                    "Report these descriptively; do not quote an inflection point.")

    return {"cp_json": cp_json, "stratified": strat, "adj": adj, "mi_thresh": mi_thresh}


# ---------------------------------------------------------------------------
# Component-decomposition splines (fig-data F8)
# ---------------------------------------------------------------------------
def build_component_splines(df: pd.DataFrame, diag: list[str] | None = None) -> pd.DataFrame:
    """Component-decomposition splines (fig-data F8).

    v3-K: the adjustment set is resolved PER COMPONENT, not per exposure — the
    `death_hemorrhagic` component is a mortality-only outcome and must drop
    `ais_head` from CORE_ADJUSTMENT, while `received_hc` and
    `substantial_transfusion` keep it (CONTRACTS §6 v3 note / SAP D-1).
    """
    rows = []
    for exposure in ("shock_index", "sbp2"):
        for comp in COMPONENT_OUTCOMES:
            adj = adjustment_for(exposure, comp)
            if diag is not None and comp in MORTALITY_ONLY_OUTCOMES:
                diag.append(f"component spline {exposure}/{comp}: v3-K mortality-only -> "
                            f"ais_head in adj = {'ais_head' in adj} (must be False)")
            dcc = complete_case(df, exposure, comp, adj)
            n_comp_ev = int(dcc[comp].sum())
            if n_comp_ev < 50:
                LOGGER.warning("component spline %s/%s skipped: only %d events (<50).",
                               exposure, comp, n_comp_ev)
                if diag is not None:
                    diag.append(f"component spline {exposure}/{comp}: SKIPPED "
                                f"({n_comp_ev} events < 50)")
                continue
            try:
                fit = fit_hc1(spline_formula(exposure, comp, adj), dcc)
                grid = spline_grid_for(exposure, dcc)
                pred = predict_spline(fit, dcc, exposure, adj, grid)
                pred.insert(0, "component", comp)
                pred.insert(0, "exposure", exposure)
                rows.append(pred)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("component spline %s/%s failed: %s", exposure, comp, exc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# VIF audit (CONTRACTS §6)
# ---------------------------------------------------------------------------
def vif_audit(df: pd.DataFrame, diag: list[str]) -> list[str]:
    """Audit VIF among {iss, ais_head, unstable_ring}; drop ais_head if flagged."""
    sub = df[VIF_AUDIT_VARS].dropna().astype(float)
    X = sm.add_constant(sub.values)
    names = ["const"] + VIF_AUDIT_VARS
    vifs = {}
    diag.append("\n=== VIF audit {iss, ais_head, unstable_ring} ===")
    for i, nm in enumerate(names):
        try:
            v = float(variance_inflation_factor(X, i))
        except Exception:  # noqa: BLE001
            v = float("nan")
        vifs[nm] = v
        diag.append(f"  {nm:16s} VIF={v:.3f}")
    dropped = []
    if vifs.get("ais_head", 0.0) > VIF_THRESHOLD:
        dropped.append("ais_head")
        diag.append(f"  -> ais_head VIF>{VIF_THRESHOLD}: DROPPED (prefer unstable_ring+iss).")
    else:
        diag.append(f"  -> all VIF<={VIF_THRESHOLD}: ais_head RETAINED.")
    return dropped


# ---------------------------------------------------------------------------
# Threshold tables (T7)
# ---------------------------------------------------------------------------
def build_threshold_tables(df: pd.DataFrame,
                           mi_thresh_all: dict[str, Any],
                           diag: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """table7_threshold_or.csv (prespec thresholds + bands + per-unit) and
    table7_absrisk.csv (g-computed adjusted absolute risk + RD; CC + MI).

    MI rows are precomputed inside ``run_curve`` (which owns the imputed frames)
    and passed in via ``mi_thresh_all[f'{exposure}|{outcome}'][t]`` — avoids
    re-imputing and avoids holding all four curves' frames in memory at once."""
    or_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    for exposure in ("shock_index", "sbp2"):
        # v3-K: adjustment resolved PER OUTCOME (was per exposure). The
        # mortality-only rows must not carry `ais_head`.
        for outcome in ANALYSIS_OUTCOMES:
            adj = adjustment_for(exposure, outcome)
            if diag is not None:
                diag.append(f"T7 {exposure}|{outcome}: adj={adj} "
                            f"(ais_head={'ais_head' in adj}, "
                            f"unstable_ring={'unstable_ring' in adj})")
            dcc = complete_case(df, exposure, outcome, adj)
            mi_thresh = mi_thresh_all.get(f"{exposure}|{outcome}", {})
            # ---- pre-specified thresholds (Bonferroni m=4) ----
            for t in PRESPEC_THRESHOLDS[exposure]:
                g = threshold_gcomp(dcc, exposure, outcome, t, adj)
                p_bonf = float(min(1.0, g["p_raw"] * BONFERRONI_M))
                or_rows.append({
                    "exposure": exposure, "outcome": outcome, "analysis": "prespecified_threshold",
                    "term": _danger_side_label(exposure, t),
                    "adj_or": round(g["adj_or"], 4), "lci": round(g["or_lci"], 4),
                    "uci": round(g["or_uci"], 4), "p_raw": g["p_raw"],
                    "p_bonferroni": p_bonf, "n": g["n_danger"] + g["n_safe"],
                    "notes": f"danger side={_danger_side_label(exposure, t)}; Bonferroni m={BONFERRONI_M}",
                })
                ar_rows.append({
                    "exposure": exposure, "outcome": outcome, "threshold": t,
                    "denom_set": "complete_case", "danger_side": _danger_side_label(exposure, t),
                    "adj_risk_danger_pct": round(g["adj_risk_danger"] * 100, 4),
                    "adj_risk_safe_pct": round(g["adj_risk_safe"] * 100, 4),
                    "adj_risk_diff_pp": round(g["adj_risk_diff"] * 100, 4),
                    "rd_lci_pp": round(g["rd_lci"] * 100, 4), "rd_uci_pp": round(g["rd_uci"] * 100, 4),
                    "adj_or": round(g["adj_or"], 4), "or_lci": round(g["or_lci"], 4),
                    "or_uci": round(g["or_uci"], 4),
                    "n_danger": g["n_danger"], "n_safe": g["n_safe"],
                    "ci_method": "delta_method_HC1", "notes": "g-computation over observed covariates",
                })
                mi = mi_thresh.get(t)
                if mi:
                    ar_rows.append({
                            "exposure": exposure, "outcome": outcome, "threshold": t,
                            "denom_set": "MI_m20", "danger_side": _danger_side_label(exposure, t),
                            "adj_risk_danger_pct": round(mi["adj_risk_danger"] * 100, 4),
                            "adj_risk_safe_pct": round(mi["adj_risk_safe"] * 100, 4),
                            "adj_risk_diff_pp": round(mi["adj_risk_diff"] * 100, 4),
                            "rd_lci_pp": round(mi["rd_lci"] * 100, 4), "rd_uci_pp": round(mi["rd_uci"] * 100, 4),
                            "adj_or": round(mi["adj_or"], 4), "or_lci": round(mi["or_lci"], 4),
                            "or_uci": round(mi["or_uci"], 4),
                            "n_danger": np.nan, "n_safe": np.nan,
                            "ci_method": "MI_rubin", "notes": f"Rubin-pooled over m={mi['m_ok']}",
                        })
            # ---- categorical band ORs ----
            for r in band_ors(dcc, exposure, outcome, adj):
                or_rows.append({
                    "exposure": exposure, "outcome": outcome, "analysis": "categorical_band",
                    "term": r["term"], "adj_or": round(r["adj_or"], 4),
                    "lci": round(r["lci"], 4), "uci": round(r["uci"], 4),
                    "p_raw": r["p_raw"], "p_bonferroni": np.nan, "n": r["n"],
                    "notes": "reference = SI<0.8 / SBP>=110",
                })
            # ---- per-unit OR ----
            pu = per_unit_or(dcc, exposure, outcome, adj)
            or_rows.append({
                "exposure": exposure, "outcome": outcome, "analysis": "per_unit",
                "term": pu["term"], "adj_or": round(pu["adj_or"], 4),
                "lci": round(pu["lci"], 4), "uci": round(pu["uci"], 4),
                "p_raw": pu["p_raw"], "p_bonferroni": np.nan, "n": len(dcc),
                "notes": "linear-exposure adjusted OR",
            })
    return pd.DataFrame(or_rows), pd.DataFrame(ar_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 Analysis B — physiologic threshold analysis (SAP §6).")
    p.add_argument("--sample", type=int, default=None,
                   help="Row-downsample N (seed 42) for smoke tests.")
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT,
                   help="Change-point row-bootstrap iterations (default 1000).")
    p.add_argument("--mi-m", type=int, default=MI_M_DEFAULT,
                   help="Multiple-imputation m (default 20).")
    p.add_argument("--n-boot-strat", type=int, default=250,
                   help="Bootstrap iters for headline-stratum change-point CIs "
                        "(default 250 cap; complete-case main CI stays at --n-boot=1000).")
    p.add_argument("--n-boot-mi", type=int, default=0,
                   help="Per-imputation bootstrap iters for the MI within-variance "
                        "(default 0 = complete-case bootstrap variance as the Rubin "
                        "within-term; a per-imputation boot x m=20 x 4 curves is not "
                        "wall-clock tractable. Set >0 to enable per-imputation boots).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    for d in (DERIVED, TABLES, LOGS):
        os.makedirs(d, exist_ok=True)

    df = load_cohort(args.sample)
    diag: list[str] = [
        "v3 Analysis B — physiologic THRESHOLD analysis (SAP §6; CONTRACTS §9-H)",
        f"analytic N (cohort_primary==1{', sampled' if args.sample else ''}) = {len(df)}",
        f"seed={RANDOM_SEED}, n_boot(CC)={args.n_boot}, n_boot_strat={args.n_boot_strat}, "
        f"mi_m={args.mi_m}, n_boot_mi={args.n_boot_mi}",
        "change-point inner fit = sklearn newton-cholesky FULL refit (no HC1); "
        "joblib-parallel bootstrap; HC1 reserved for spline/threshold/per-unit models.",
    ]

    # ---- v3 outcome sweep + Fix-3 asymmetry, restated at run time (§9-J, §13 G3f) ----
    for _o in ANALYSIS_OUTCOMES:
        if _o not in df.columns:
            raise SystemExit(
                f"cohort CSV has no '{_o}' column. v3 repointed the mortality-only "
                f"anchor from '{DEATH_SENSITIVITY_ALLCAUSE}' to '{MORTALITY_OUTCOME}' "
                "(CONTRACTS §4/§9-J); 01_build_cohort.py owns the column.")
    diag.append(f"outcomes (CONTRACTS §4/§9-J): composite={COMPOSITE_OUTCOME}, "
                f"mortality-only anchor={MORTALITY_OUTCOME} "
                f"(v2's {DEATH_SENSITIVITY_ALLCAUSE} is a named sensitivity, owned by 13)")
    for _o in ANALYSIS_OUTCOMES:
        diag.append(f"  {_o}: n={int(pd.to_numeric(df[_o], errors='coerce').fillna(0).sum())} "
                    f"of {len(df)}")

    # VIF audit (may drop ais_head)
    dropped = vif_audit(df, diag)
    if dropped:
        for d_ in dropped:
            if d_ in CORE_ADJUSTMENT:
                CORE_ADJUSTMENT.remove(d_)
        diag.append(f"CORE_ADJUSTMENT after VIF drop: {CORE_ADJUSTMENT}")

    # §13 G3f — re-assert AFTER the VIF audit mutates CORE_ADJUSTMENT. The audit
    # may only ever drop `ais_head`; if `unstable_ring` disappeared, Analysis B is
    # silently degraded and the gate must fail here, not in review.
    if "unstable_ring" not in CORE_ADJUSTMENT:
        raise SystemExit(
            "§13 G3f FAIL: `unstable_ring` left CORE_ADJUSTMENT at run time. Fix 3 "
            "removes it from the SCORE (02/04/05) only — Analysis B is an "
            "association model and needs it to keep the SI/SBP dose-response "
            "unconfounded by injury pattern (CONTRACTS §6, SAP D-3).")
    if "unstable_ring" not in VIF_AUDIT_VARS:
        raise SystemExit("§13 G3f FAIL: `unstable_ring` left VIF_AUDIT_VARS.")
    diag.append("§13 G3f OK: unstable_ring retained in CORE_ADJUSTMENT and VIF_AUDIT_VARS "
                "(Fix 3 applies to the SCORE only — deliberate asymmetry, SAP D-3).")
    if any(s[0] == "unstable_ring" for s in STRATUM_DEFS):
        diag.append("§13 G3f OK: ring-stability stratified change-point still runs.")
    else:
        raise SystemExit("§13 G3f FAIL: ring-stability stratum removed from STRATUM_DEFS.")

    # ---- 4 headline curves ----
    curve_results: dict[str, Any] = {}
    mi_thresh_all: dict[str, Any] = {}
    stratified_all: dict[str, Any] = {}
    for exposure, outcome, tag in RUN_MATRIX:
        res = run_curve(df, exposure, outcome, tag, args.n_boot, args.mi_m, diag,
                        n_boot_strat=args.n_boot_strat, n_boot_mi=args.n_boot_mi)
        curve_results[tag] = res["cp_json"]
        mi_thresh_all[f"{exposure}|{outcome}"] = res["mi_thresh"]
        stratified_all[tag] = res["stratified"]

    # ---- POWER-GUARD LEDGER (locked SAP §6 / CONTRACTS §13 G5e) ----
    # v3's mortality-only anchor is ~29 % of v2's event count, so demotions are
    # EXPECTED here and must be impossible to miss. Print loudly, log to the
    # diagnostics, and write a machine-readable summary into the stratified file
    # (which `10` reads for the elderly change-point tie-in).
    guard_summary: dict[str, Any] = {
        "min_events_per_side": MIN_EVENTS_PER_SIDE,
        "min_events_stratum": MIN_EVENTS_STRATUM,
        "policy": ("locked SAP §6 / CONTRACTS §13 G5e: a change-point that cannot "
                   "meet the pre-specified >=100-events-per-side guard is reported "
                   "DESCRIPTIVE-ONLY. The guard is not relaxed to obtain an estimate."),
        "main_curves_descriptive_only": [t for t in curve_results
                                         if curve_results[t].get("descriptive_only")],
        "strata_descriptive_only": {},
    }
    for tag_, sblock in stratified_all.items():
        labels = sblock.get("descriptive_only_labels") or []
        if labels:
            guard_summary["strata_descriptive_only"][tag_] = labels
    n_dem_strata = sum(len(v) for v in guard_summary["strata_descriptive_only"].values())
    banner_line = "!" * 78
    diag.append("\n=== POWER GUARD (SAP §6 / §13 G5e) ===")
    diag.append(f"MIN_EVENTS_PER_SIDE={MIN_EVENTS_PER_SIDE} (NOT relaxed); "
                f"stratum attempt threshold={MIN_EVENTS_STRATUM} events")
    if guard_summary["main_curves_descriptive_only"] or n_dem_strata:
        print(banner_line, file=sys.stderr)
        print("POWER GUARD — DESCRIPTIVE-ONLY DEMOTIONS (locked SAP §6 / CONTRACTS §13 G5e)",
              file=sys.stderr)
        if guard_summary["main_curves_descriptive_only"]:
            msg = ("MAIN curves demoted: "
                   + ", ".join(guard_summary["main_curves_descriptive_only"]))
            print("  " + msg, file=sys.stderr)
            diag.append("MAIN curves DESCRIPTIVE-ONLY: "
                        + ", ".join(guard_summary["main_curves_descriptive_only"]))
        for tag_, labels in guard_summary["strata_descriptive_only"].items():
            print(f"  {tag_}: {', '.join(labels)}", file=sys.stderr)
            diag.append(f"strata DESCRIPTIVE-ONLY [{tag_}]: {', '.join(labels)}")
        print("  These MUST be reported descriptively. Do not quote an inflection point,",
              file=sys.stderr)
        print("  do not widen the grid, do not lower MIN_EVENTS_PER_SIDE.", file=sys.stderr)
        print(banner_line, file=sys.stderr)
    else:
        print("POWER GUARD: no change-point demoted — every curve and stratum met "
              f">= {MIN_EVENTS_PER_SIDE} events/side.", file=sys.stderr)
        diag.append("no demotions: every curve and stratum met the guard.")
    stratified_all["_guard_summary"] = guard_summary

    # ---- stratified change-points file ----
    strat_path = os.path.join(DERIVED, "changepoint_stratified.json")
    with open(strat_path, "w") as f:
        json.dump(stratified_all, f, indent=2, default=str)
    LOGGER.info("wrote %s", strat_path)

    # ---- tipping-point bound file (aggregate the 4 curves' tipping) ----
    tip_all = {tag: curve_results[tag]["tipping_point"] for tag in curve_results}
    tip_path = os.path.join(DERIVED, "tipping_point_bound.json")
    with open(tip_path, "w") as f:
        json.dump(tip_all, f, indent=2, default=str)
    LOGGER.info("wrote %s", tip_path)

    # ---- threshold tables (T7) ----
    LOGGER.info("building threshold tables (T7) ...")
    t7_or, t7_ar = build_threshold_tables(df, mi_thresh_all, diag)
    t7_or.to_csv(os.path.join(TABLES, "table7_threshold_or.csv"), index=False)
    t7_ar.to_csv(os.path.join(TABLES, "table7_absrisk.csv"), index=False)
    LOGGER.info("wrote table7_threshold_or.csv (%d rows), table7_absrisk.csv (%d rows)",
                len(t7_or), len(t7_ar))

    # ---- component-decomposition splines (F8) ----
    LOGGER.info("building component-decomposition splines ...")
    comp = build_component_splines(df, diag)
    comp.to_csv(os.path.join(DERIVED, "component_splines.csv"), index=False)
    LOGGER.info("wrote component_splines.csv (%d rows)", len(comp))

    # ---- gate summary (§13.4) ----
    diag.append("\n=== GATE SUMMARY (CONTRACTS §13.4) ===")
    for tag in curve_results:
        cj = curve_results[tag]
        flag = " [DESCRIPTIVE-ONLY]" if cj.get("descriptive_only") else ""
        diag.append(f"  {tag}: bp={cj['breakpoint']} CI[{round(cj['lci'],3) if np.isfinite(cj['lci']) else 'nan'},"
                    f"{round(cj['uci'],3) if np.isfinite(cj['uci']) else 'nan'}] "
                    f"boundary={cj['at_grid_boundary']} ci_excl_endpoints={cj['ci_excludes_endpoints']}"
                    f"{flag}")
        diag.append(f"      outcome={cj['outcome']} n={cj['n']} events={cj['n_events']} "
                    f"ais_head_in_adj={cj['ais_head_in_adjustment']} "
                    f"unstable_ring_in_adj={cj['unstable_ring_in_adjustment']}")
    # §13 G5e cross-checks that a human must read: does the mortality-only
    # inflection co-locate with the composite one, on each exposure?
    for expo, t_comp, t_mort in (("shock_index", "si_composite", "si_mortality"),
                                 ("sbp2", "sbp_composite", "sbp_mortality")):
        bc = curve_results.get(t_comp, {}).get("breakpoint", float("nan"))
        bm = curve_results.get(t_mort, {}).get("breakpoint", float("nan"))
        if np.isfinite(bc) and np.isfinite(bm):
            diag.append(f"  §13 G5e co-location [{expo}]: composite bp={bc} vs "
                        f"{MORTALITY_OUTCOME} bp={bm} (|diff|={abs(bc - bm):.4f})")
        else:
            diag.append(f"  §13 G5e co-location [{expo}]: NOT EVALUABLE — "
                        f"composite bp={bc}, {MORTALITY_OUTCOME} bp={bm} "
                        "(a descriptive-only curve cannot be co-located)")

    with open(os.path.join(LOGS, "thresholds_diagnostics.txt"), "w") as f:
        f.write("\n".join(diag) + "\n")

    manifest = {
        "script": "11_pelvic_thresholds.py",
        "random_seed": RANDOM_SEED, "alpha": ALPHA,
        "n_boot": args.n_boot, "n_boot_strat": args.n_boot_strat,
        "mi_m": args.mi_m, "n_boot_mi": args.n_boot_mi, "sample": args.sample,
        "changepoint_inner_fitter": "sklearn_LogisticRegression_newton-cholesky_penalty-none",
        "changepoint_cov_type": "none_in_inner_loop_HC1_for_final_point_models",
        "bootstrap_parallel": "joblib_n_jobs_-1_full_refit_per_resample",
        "input_cohort": COHORT_CSV, "input_sha256": _sha256(COHORT_CSV),
        "analytic_n": int(len(df)),
        "core_adjustment": CORE_ADJUSTMENT,
        "vif_dropped": dropped,
        # v3-K: the adjustment set is outcome-dependent. Record the per-run set so
        # the manuscript can state exactly what each model adjusted for.
        "adjustment_per_run": {tag: {"outcome": curve_results[tag]["outcome"],
                                     "exposure": curve_results[tag]["exposure"],
                                     "adjustment": curve_results[tag]["adjustment"],
                                     "stripped": curve_results[tag]["adjustment_stripped"],
                                     "mortality_only_run": curve_results[tag]["mortality_only_run"],
                                     "ais_head_in_adjustment":
                                         curve_results[tag]["ais_head_in_adjustment"]}
                               for tag in curve_results},
        "outcomes": {"composite": COMPOSITE_OUTCOME,
                     "mortality_only_anchor": MORTALITY_OUTCOME,
                     "death_sensitivity_allcause_not_run_here": DEATH_SENSITIVITY_ALLCAUSE,
                     "component_outcomes": COMPONENT_OUTCOMES},
        "fix3_asymmetry": {"unstable_ring_in_core_adjustment":
                               "unstable_ring" in CORE_ADJUSTMENT,
                           "unstable_ring_in_vif_audit_vars":
                               "unstable_ring" in VIF_AUDIT_VARS,
                           "ring_stratum_runs":
                               any(s[0] == "unstable_ring" for s in STRATUM_DEFS),
                           "note": ("CONTRACTS §6 / §13 G3f — Fix 3 removes the ring "
                                    "from the SCORE (02/04/05) only.")},
        "run_matrix": [{"exposure": e, "outcome": o, "tag": t} for e, o, t in RUN_MATRIX],
        "grids": {"si": [float(GRID_SI.min()), float(GRID_SI.max()), 0.02],
                  "sbp": [float(GRID_SBP.min()), float(GRID_SBP.max()), 2.0]},
        "min_events_per_side": MIN_EVENTS_PER_SIDE,
        "power_guard": guard_summary,
        "prespec_thresholds": PRESPEC_THRESHOLDS, "bonferroni_m": BONFERRONI_M,
        "changepoints": {tag: {"breakpoint": curve_results[tag]["breakpoint"],
                               "lci": curve_results[tag]["lci"],
                               "uci": curve_results[tag]["uci"],
                               "at_grid_boundary": curve_results[tag]["at_grid_boundary"],
                               "ci_excludes_endpoints": curve_results[tag]["ci_excludes_endpoints"],
                               "descriptive_only": curve_results[tag].get("descriptive_only", False),
                               "guard_reason": ((curve_results[tag].get("guard") or {})
                                                .get("guard_reason")),
                               "n_events": curve_results[tag]["n_events"],
                               "mi_location": curve_results[tag]["mi_pooled_location"].get("location")}
                         for tag in curve_results},
        "packages": _package_versions(),
        "runtime_sec": round(time.time() - t0, 2),
    }
    with open(os.path.join(LOGS, "thresholds_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    LOGGER.info("DONE in %.1f min. Change-points: %s",
                (time.time() - t0) / 60.0,
                {tag: (round(curve_results[tag]["breakpoint"], 3)
                       if np.isfinite(curve_results[tag]["breakpoint"])
                       else ("DESCRIPTIVE-ONLY" if curve_results[tag].get("descriptive_only")
                             else None))
                 for tag in curve_results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
