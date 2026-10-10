"""
Pelvic Arrival Decision-Support Tool (v3) — Step 02: Engineer arrival predictor pool
=====================================================================================

Forks PAT_TBI/v4/code/02_engineer_features.py. Builds the 17-variable
arrival/bedside-extractable LASSO candidate matrix (LASSO_X_COLS) from the raw
pre-decision cohort columns, per the engineering recipes in
`extractability_audit/candidate_audit_table.csv` (Batch 1, `source_columns`) and
CONTRACTS §5. Carries every cohort column through so downstream scripts have
outcomes, comparator inputs, subgroup vars, the full-information comparator
inputs (iss/ais_head), and `cohort_primary` (the analytic-cohort flag).

v3 CHANGES vs the v2 fork
-------------------------
  * **Fix 3 (D2, reviewer request).** `unstable_ring` is REMOVED from LASSO_X_COLS
    (18 -> 17). It remains a pass-through cohort column in candidates.parquet
    and in pelvic_cohort_predecision.csv — Fix 3 removes it from the *score*,
    not from the *data*. `11` still adjusts for it (CORE_ADJUSTMENT),
    `13` still VIF-audits it, and `15` still reports it in Table 1. That
    asymmetry is deliberate and defended in CONTRACTS §6 / SAP D-3.
  * **Fix 2 / 2b + Fix 1 + Fix 7 outcome plumbing.** Step 01's cohort CSV grew
    well past v2's 50 columns: the death ladder and its untimed-death recovery
    rungs (`death_hemorrhagic`, `death_hemorrhagic_headais`,
    `death_hemorrhagic_missing_incl`, `death_hemorrhagic_dayrule`,
    `death_hemorrhagic_dayfallback`, `death_hemorrhagic_edincl`, `death_le48h`,
    `death_le48h_dayrule`, `death_ed`), the two timing fields
    (`finaldischarge_hrs` primary, `finaldischarge_days` crosswalk), the
    transfusion decoder outputs (`substantial_transfusion_ml_only`,
    `blood_4h_scale`, `mtp_equivalent`), the composite sensitivities
    (`major_hemorrhage_event_headais`, `major_hemorrhage_event_v2def`) and the
    balanced-resuscitation fields (`plasma_4h`, `platelet_4h`). All are carried
    into candidates.parquet by the blanket pass-through and are asserted
    present (REQUIRED_CARRY_COLS).
  * **NO HARD-CODED COHORT COLUMN COUNT.** 02 does not carry a
    `N_COHORT_COLS_EXPECTED` literal any more. Such a constant went stale twice
    (50 -> 63 -> 68) and each time halted the pipeline at step 02 over a
    schema change that harmed nothing. The expected schema is now DERIVED at
    runtime from an artifact 01 itself emits — `v3/logs/qc_manifest.json`
    ["cohort_columns"], with `01_build_cohort.COHORT_COLUMNS` (read by `ast`,
    never imported) as the fallback/cross-check. The only HARD requirement is
    REQUIRED_CARRY_COLS: the columns downstream scripts read BY NAME. See
    `check_cohort_schema()`.
  * **OUTCOME_COL is unchanged in NAME (`major_hemorrhage_event`) but changed
    in MEANING** — its death arm is now `death_hemorrhagic`
    (= death_inhosp & ais_head < 4 & **finaldischarge_hrs <= 48**). The old
    `finaldischarge_days <= 2` calendar-day rule is SUPERSEDED; it survives
    only as the explicitly-labelled `*_dayrule` sensitivity/crosswalk columns
    (CONTRACTS §4, §9 v3-J).
  * **BOTH timing fields are FORBIDDEN as predictors.** `finaldischarge_hrs` is
    outcome-defining (it builds `death_hemorrhagic`) and is emphatically NOT
    arrival-available — it is the arrival-to-final-discharge elapsed-hours
    offset, knowable only at the end of the admission. `finaldischarge_days` is
    the same quantity at calendar-day resolution. Neither may reach
    LASSO_X_COLS or any adjustment set. Same for `ais_head`: forbidden as an X,
    legitimate as an outcome ingredient (CONTRACTS §5 note — that is outcome
    construction, not leakage).

Inputs
------
  v3/derived/pelvic_cohort_predecision.csv            (Step 01 output, 291,766 rows;
                                                       width = len(01.COHORT_COLUMNS), 68 as of this writing)
  v3/logs/qc_manifest.json                            (Step 01 output; ["cohort_columns"] = the emitted schema)

Outputs
-------
  v3/derived/candidates.parquet                       (every Step-01 cohort column, passed through unchanged,
                                                       + the engineered X names not already among them:
                                                       68 + 13 = 81 under 01's current schema. Both terms are
                                                       computed at runtime — do NOT hard-code either.)
  v3/tables/table2_candidate_coverage.csv             (per-X N / N_positive / Mean / Min / Max, analytic cohort)

Key = (puf_year, inc_key). Row count = 291,766 (acetabulum-inclusive). The
analytic cohort is `cohort_primary == 1` (N = 239,736); `03`/`04`/…/`12` filter
to it. Acetabulum rows are retained for the free §13 sensitivity ONLY.

Engineering recipes (CONTRACTS §5 / audit table)
------------------------------------------------
  age_65_74        = (age >= 65) & (age < 75)                 [audit source: age]
  age_ge75         = (age >= 75)                              [audit source: age]
  sex_female       = (sex_male == 0)   (null-biased -> 0)     [audit source: sex_male]
  sbp_lt90         = cohort sbp_lt90, NaN -> 0 (null-biased)  [already 0/1 in CSV; source sbp2]
  sbp_lt70         = (sbp2 < 70)       (null-biased -> 0)     [audit source: sbp2]
  sbp_unrecordable = cohort sbp_unrecordable                  [already 0/1 in CSV]
  hr_ge120         = (pulse2 >= 120)   (null-biased -> 0)     [audit source: pulse2]
  shock_index      = cohort shock_index, NaN -> 0.7           [engineered pass-through, NOT a LASSO X; §5 CORRECTION]
  si_ge0_9         = (shock_index >= 0.9)                     [audit source: pulse2|sbp2]
  si_ge1_0         = (shock_index >= 1.0)                     [audit source: pulse2|sbp2]
  gcs_motor_le5    = (motor2 <= 5)     (null-biased -> 0)     [audit source: motor2]
  gcs_le8          = (gcs2 <= 8)       (null-biased -> 0)     [audit source: gcs2]
  mech_high_energy = mechanism_cat in {MVC-occupant, Motorcycle, Pedestrian}
  mech_mvc         = mechanism_cat == "MVC-occupant"
  mech_fall        = mechanism_cat == "Fall"
  air_transport    = transport_cat == "Air"
  cc_anticoag      = cohort cc_anticoag (preexisting code 31) [already 0/1; FORCE-IN anchor]
  cc_bleeding      = cohort cc_bleeding (preexisting code 4)  [already 0/1]
  (unstable_ring   = REMOVED from the pool in v3 — Fix 3 / D-2. Still a
                     pass-through column; see the v3 CHANGES block above.)

Arrival-availability of the retained non-physiologic variables (Fix 3 defence)
-----------------------------------------------------------------------------
The D-2 objection is that a score advertised as running on data known
*immediately upon arrival* must not contain a variable that is not. The wording was tightened
from "at arrival" to "immediately upon arrival" in review, so the standard
is strict. `unstable_ring` fails it — ring stability is a radiographic read that
is not available at the door — and is removed. The four remaining non-laboratory,
non-vital-sign predictors each pass it, and the Methods section states, for each,
*who* reports it and *when*:

  mech_high_energy  WHO: the transporting EMS crew (or, for walk-ins, the
  mech_mvc               patient/family). WHEN: verbally at the bedside handoff,
  mech_fall              before any imaging or registrar coding. Mechanism of
                         injury is the opening clause of every prehospital
                         report ("restrained driver, 55 mph, rollover"), is
                         required on the run sheet, and is the single most
                         reliably transmitted element of the handoff. The
                         registry field `mechanism_cat` is the registrar's later
                         transcription of that same handoff statement, not an
                         independent post-hoc determination — the information
                         exists at t=0 even though the coded field is entered
                         later.
  air_transport     WHO: self-evident to the receiving team. WHEN: on arrival,
                         by direct observation — the patient either comes off a
                         helicopter or does not. This requires no report, no
                         imaging, and no chart. It is the most unambiguously
                         arrival-available variable in the pool.

The same test applied to the two comorbidity flags: `cc_anticoag` and
`cc_bleeding` come from the medication/history portion of the EMS or family
handoff ("he's on Eliquis"), which is routinely available at the bedside and is
the reason anticoagulation is a pre-registered force-in anchor. The physiologic
variables (SBP, HR, shock index, GCS) are measured by the trauma team in the bay
within the first minute. Age and sex are stated at registration.

The distinction that matters is between information *available at the door* and
a *field keyed into the registry later*. Every retained predictor is the former.
`unstable_ring` was the latter, which is why it is the only one dropped.

Missing-data handling for the X pool (CONTRACTS §5, §10)
--------------------------------------------------------
  * Binaries are NULL-BIASED -> 0 when the underlying vital is NaN. A missing SBP
    does NOT become sbp_lt90=1; it becomes 0. This is enforced transparently by
    deriving threshold binaries via comparison operators (NaN comparisons yield
    False -> 0).
  * The engineered continuous `shock_index` column (carried in the parquet for
    06's SI comparator / 11's exposure, but NOT a LASSO candidate — §5 CORRECTION)
    is filled with the constant 0.7
    (PAT_TBI precedent, `si_raw.fillna(0.7)`). Rationale: (a) 02 runs BEFORE the
    train/test split (03), so a strict train-fold ("training-eligible") median is
    not computable here without a circular dependency on 03; (b) 0.7 is
    leakage-free — no test-row / outcome information enters the design matrix,
    which is the TRIPOD-correct posture for a prediction model; (c) 0.7 is the
    canonical "normal" shock index, and the imputed subset is only ~2.7% of the
    analytic cohort (04 refits on the train fold regardless). The analytic-cohort
    median (0.658) is nearly identical; if a strict train-only median-impute is
    ever preferred, 04 can re-impute on the train fold (it holds train_inc_keys).
  * The full MNAR rigor (MI m=20 + tipping-point) lives in Analysis B (script 11)
    and in the deployable `sbp_unrecordable` sentinel (CONTRACTS §10) — NOT here.

Leakage audit (assertions, CONTRACTS §5 / §13 G3a)
--------------------------------------------------
  assert set(LASSO_X_COLS) & FORBIDDEN_X == set()
  assert 'major_hemorrhage_event' not in LASSO_X_COLS
  assert 'unstable_ring' not in LASSO_X_COLS            (Fix 3 / D-2)
  assert 'finaldischarge_hrs' not in LASSO_X_COLS       (Fix 2b — outcome-defining, primary)
  assert 'finaldischarge_days' not in LASSO_X_COLS      (Fix 2b — outcome-defining, superseded day rule)
  assert len(LASSO_X_COLS) == 17
  every REQUIRED_CARRY_COLS outcome column present in the cohort CSV and the parquet
  the cohort schema matches what 01 declared (derived, not hard-coded — see
    check_cohort_schema())
  every X col numeric and NaN-free.

Name-collision note (shock_index, sbp_lt90)
-------------------------------------------
  `shock_index` and `sbp_lt90` exist BOTH as raw cohort columns (with NaN) and as
  engineered NaN-free columns (sbp_lt90 is a LASSO X; shock_index is engineered
  pass-through only per §5 CORRECTION). In candidates.parquet they hold the
  ENGINEERED (NaN-free) values — the raw-NaN versions remain in
  pelvic_cohort_predecision.csv, which scripts 11/13/15 read directly. The other
  three overlapping X names (sbp_unrecordable, cc_anticoag, cc_bleeding) are
  identical in both roles. `unstable_ring` is likewise normalized in place, but
  as a PASS-THROUGH column only — it is no longer a LASSO candidate (Fix 3).
"""
import argparse
import ast
import json
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# ---- Canonical paths (CONTRACTS §1) ---------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
CODE = config.ANALYSIS_DIR
DERIVED = config.DERIVED
TABLES = config.TABLES
LOGS = config.LOGS
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

IN_COHORT = os.path.join(DERIVED, "pelvic_cohort_predecision.csv")
OUT_CANDIDATES = os.path.join(DERIVED, "candidates.parquet")
OUT_COVERAGE = os.path.join(TABLES, "table2_candidate_coverage.csv")

# Schema provenance for the cohort CSV (see check_cohort_schema). Both are
# READ-ONLY here and both are owned by Step 01.
QC_MANIFEST = os.path.join(LOGS, "qc_manifest.json")
BUILD_COHORT_PY = os.path.join(CODE, "01_build_cohort.py")

# Continuous-X missing-fill constant (PAT_TBI precedent; see module docstring).
SI_FILL = 0.7

# ---- LASSO candidate X columns (17 arrival/bedside-extractable predictors) --
# Order is fixed and meaningful for downstream consumers. ALL binary/indicator.
#
# v3 = 17 (was 18): `unstable_ring` REMOVED — Fix 3 / D-2 (CONTRACTS §5, §13 G3a).
# Ring stability is a radiographic determination, not information available
# immediately upon arrival; a score that advertises arrival-only inputs may not
# contain it. It stays in the cohort CSV, in `11.CORE_ADJUSTMENT`, in
# `11`/`13`.VIF_AUDIT_VARS, as a change-point stratum, and in Table 1 — see the
# deliberate-asymmetry note in CONTRACTS §6. Do NOT "propagate" the removal.
#
# `sbp_unrecordable` STAYS here and stays in 05's unpenalized refit. Fix 4 / D-6
# + W2 removes it from the INTEGER POINT SCALE only, turning it into a binary
# automatic-escalate override (CONTRACTS §5.1). Its display label everywhere
# downstream is "no signs of life on arrival (unrecordable systolic pressure)" —
# The clinical reading: an SBP of 0 is what the registry records for a
# patient with no signs of life on first assessment. It is a physiologic state,
# NOT a missingness or data-quality artifact, and must never be labelled as one.
#
# The continuous `shock_index` is engineered below and carried in
# candidates.parquet (06's SI comparator / 11's exposure) but is NOT a LASSO
# candidate — CONTRACTS §5 CORRECTION: a raw 0–53 term cannot be an integer
# bedside-card point value (it made the score −5…227 and triple-counted SI);
# SI enters the score via si_ge0_9 + si_ge1_0.
# ⚠️ This EXACT list literal must be byte-identical in 04_lasso_stability.py
#    and 05_refit_integer.py (CONTRACTS §5: "keep LASSO_X_COLS synced").
LASSO_X_COLS = [
    "age_65_74", "age_ge75", "sex_female",
    "sbp_lt90", "sbp_lt70", "sbp_unrecordable",
    "hr_ge120",
    "si_ge0_9", "si_ge1_0",
    "gcs_motor_le5", "gcs_le8",
    "mech_high_energy", "mech_mvc", "mech_fall",
    "air_transport",
    "cc_anticoag", "cc_bleeding",
]

# ---- FORBIDDEN columns (leakage audit, CONTRACTS §5) ----------------------
# Every treatment/outcome var + registrar-coded severity. ISS/AIS are offered
# only as a separate "full-information" comparator model in 07 — never in the X.
#
# v3 additions: the whole new outcome family (Fix 1 / 2 / 2b / 7) plus BOTH
# discharge-timing fields. Per the CONTRACTS handoff, `finaldischarge_hrs` is the
# single most dangerous column in the cohort CSV: it is outcome-DEFINING (it
# builds `death_hemorrhagic`, hence `major_hemorrhage_event`) AND it is the
# arrival-to-final-discharge elapsed-hours offset, i.e. knowable only when the
# admission ends. It is not arrival-available under any reading and must never
# reach LASSO_X_COLS or any adjustment set in any script. `finaldischarge_days`
# is the same quantity at calendar-day resolution and is equally forbidden.
#
# MAINTENANCE RULE: this set must list EVERY outcome-defining or post-arrival
# column 01 emits, not merely the ones that happen to collide with today's
# LASSO_X_COLS. It is the defence, so it leads the predictor pool rather than
# trailing it. When 01's schema grows, add the new columns HERE first —
# `check_cohort_schema()` will tell you the schema moved, but it cannot tell you
# which of the new columns are post-arrival. The v3 list is transcribed from
# CONTRACTS §5.
#
# NOTE (CONTRACTS §5, do not "fix"): `ais_head`, `finaldischarge_hrs` and
# `finaldischarge_days` are forbidden as PREDICTORS while simultaneously being
# ingredients of the primary outcome. That is outcome construction, not leakage.
# The assert below covers predictors only; no such column may be deleted from the
# cohort CSV to "resolve" the apparent conflict.
FORBIDDEN_X = {
    # Primary + co-primary outcomes and their composite sensitivities
    "major_hemorrhage_event", "major_hemorrhage_event_headais",
    "major_hemorrhage_event_v2def", "patient_state_composite",
    # Hemorrhage-control receipt and every route/type field behind it
    "received_hc", "surg_hc", "angio_hc", "hc_route", "surg_class",
    "embo_any", "embo_pelvis", "pelvic_specific_surg",
    "surg_type_raw", "angio_raw", "embo_site_raw",
    # Death ladder (CONTRACTS §4) — every rung, incl. the untimed-death recovery
    # ladder (`*_dayrule` / `*_dayfallback` / `*_edincl`) and the ED surrogate
    "death_inhosp", "death_hemorrhagic", "death_hemorrhagic_headais",
    "death_hemorrhagic_missing_incl", "death_hemorrhagic_dayrule",
    "death_hemorrhagic_dayfallback", "death_hemorrhagic_edincl",
    "death_le48h", "death_le48h_dayrule", "death_ed",
    # Timing fields behind the <=48h restriction — outcome-defining, end-of-stay.
    # `finaldischarge_hrs` is the PRIMARY rule (<=48 h); `finaldischarge_days`
    # (<=2 d) is the superseded crosswalk. Both post-arrival, both forbidden.
    "finaldischarge_hrs", "finaldischarge_days",
    # Transfusion family (Fix 1 decoder + Fix 7 secondary + exploratory arms)
    "substantial_transfusion", "substantial_transfusion_ml_only",
    "mtp_equivalent", "blood_4h", "blood_4h_scale", "blood_24h",
    "plasma_4h", "platelet_4h",
    # Downstream resource-use outcomes
    "icu_los", "vent_days",
    # Registrar-coded severity (post-hoc, not knowable at arrival)
    "iss", "ais_head", "ais_head_ge3", "ais_chest", "ais_abd",
}

# ---- Outcome / sensitivity columns that MUST reach candidates.parquet -------
# Downstream consumers read every one of these BY NAME — most out of the parquet
# (06/07/08/09/12 outcomes; 17 the MTP-equivalent secondary), the death-ladder
# rungs out of the cohort CSV (13's sensitivity arms). They arrive in the parquet
# via the blanket cohort pass-through below; this list turns that implicit
# dependency into an explicit, failing assertion so an upstream 01 schema change
# cannot silently starve a downstream script — and it fails HERE, at step 02,
# with provenance, rather than as a bare KeyError eleven scripts later.
#
# THIS list — not a column count — is the safety property. Add to it whenever a
# downstream script starts reading a new 01 column by name.
REQUIRED_CARRY_COLS = [
    # keys / cohort flags
    "inc_key", "puf_year", "cohort_primary", "has_acetabulum",
    # PRIMARY + co-primaries (CONTRACTS §4)
    "major_hemorrhage_event", "patient_state_composite", "death_hemorrhagic",
    # composite sensitivities
    "major_hemorrhage_event_headais", "major_hemorrhage_event_v2def",
    # death ladder rungs + ED surrogate + the untimed-death recovery ladder
    "death_inhosp", "death_hemorrhagic_headais", "death_le48h",
    "death_hemorrhagic_missing_incl", "death_ed",
    "death_hemorrhagic_dayrule", "death_hemorrhagic_dayfallback",
    "death_hemorrhagic_edincl", "death_le48h_dayrule",
    # timing fields: hrs = PRIMARY (<=48 h) rule, days = superseded crosswalk
    "finaldischarge_hrs", "finaldischarge_days",
    # transfusion family (Fix 1) + MTP-equivalent secondary (Fix 7)
    "substantial_transfusion", "substantial_transfusion_ml_only",
    "mtp_equivalent", "blood_4h", "blood_4h_scale",
    "plasma_4h", "platelet_4h",
    # HC arm of the composite
    "received_hc",
    # comparator inputs / full-information benchmark / subgroup vars
    "shock_index", "sbp2", "pulse2", "gcs2", "motor2",
    "iss", "ais_head", "age", "age_ge65", "sex_male",
    "acs_level", "charlson", "mechanism_cat", "transport_cat",
    "unstable_ring",
]

# There is deliberately NO `N_COHORT_COLS_EXPECTED` here. See the module
# docstring and `check_cohort_schema()`: the expected width of the Step-01 cohort
# CSV is DERIVED from 01's own artifacts, never asserted against a literal.
# N_LASSO_X_EXPECTED is different in kind: 17 is a pre-registered design decision
# (CONTRACTS §5, §13 G3a) that this script OWNS, not an upstream fact it observes,
# so it is correctly a hard-coded constant.
N_LASSO_X_EXPECTED = 17


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# Cohort-schema provenance (replaces the old N_COHORT_COLS_EXPECTED tripwire)
# ---------------------------------------------------------------------------
def _schema_from_manifest():
    """Cohort schema as recorded by the Step-01 run that wrote the CSV.

    `01_build_cohort.py` writes `logs/qc_manifest.json` with
    `["cohort_columns"] = list(COHORT_COLUMNS)` in the same run that writes
    `pelvic_cohort_predecision.csv` with `columns=COHORT_COLUMNS`. The two
    therefore agree by construction, and any disagreement means one of them is
    stale (a partial or superseded 01 run). Returns None if unavailable.
    """
    try:
        with open(QC_MANIFEST, "r", encoding="utf-8") as fh:
            cols = json.load(fh).get("cohort_columns")
    except (OSError, ValueError):
        return None
    return list(cols) if cols else None


def _schema_from_source():
    """01's `COHORT_COLUMNS` literal, read with `ast` — no import, no execution.

    01_build_cohort.py is the source of truth for the cohort schema, but it is a
    ~170k-line-equivalent script with real work at import time, so it must not be
    imported. Parsing the module and `literal_eval`-ing the single list literal
    gives the declared schema for free. Returns None if it cannot be recovered.
    """
    try:
        with open(BUILD_COHORT_PY, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "COHORT_COLUMNS"
                   for t in node.targets):
            continue
        try:
            return list(ast.literal_eval(node.value))
        except (ValueError, SyntaxError):
            return None
    return None


def check_cohort_schema(cohort_cols):
    """Verify the cohort CSV's schema WITHOUT hard-coding its width.

    The old check was `assert len(cohort_cols) == 63`. That constant went stale
    twice (50 -> 63 -> 68, most recently when the death rule moved from calendar
    days to true elapsed hours and 01 gained five columns) and each time it
    halted the entire pipeline at step 02 over a change that broke nothing. A
    count assert that cannot self-update is a tripwire on our own pipeline, not
    a safety check.

    What actually matters is checked elsewhere and loudly: every column in
    REQUIRED_CARRY_COLS must be present (the caller asserts that). What is
    checked HERE is that the CSV carries exactly the schema 01 declared, using
    01's own artifacts as the expectation, so the check updates itself the
    moment 01 changes:

      * HARD  — CSV schema == `logs/qc_manifest.json["cohort_columns"]`, when the
                manifest is readable. Same run, same list; a set difference means
                the CSV and its manifest come from different 01 runs.
      * WARN  — CSV schema vs `01_build_cohort.COHORT_COLUMNS` (ast). A
                difference means 01's source has changed since the CSV was
                built. That is worth shouting about, but it is not this script's
                call to make, so it does not halt the run.
      * SKIP  — if neither artifact is recoverable, log it and rely on
                REQUIRED_CARRY_COLS alone. Never invent a number.
    """
    n = len(cohort_cols)
    observed = set(cohort_cols)

    manifest_cols = _schema_from_manifest()
    if manifest_cols is None:
        print(f"  WARN  {QC_MANIFEST} unreadable or missing 'cohort_columns' — "
              f"cannot cross-check the {n}-column cohort schema against the "
              f"Step-01 run that produced it. Falling back to "
              f"REQUIRED_CARRY_COLS only.", flush=True)
    else:
        dropped = [c for c in manifest_cols if c not in observed]
        added = [c for c in cohort_cols if c not in set(manifest_cols)]
        assert not dropped and not added, (
            f"COHORT SCHEMA MISMATCH: {IN_COHORT} ({n} columns) does not match "
            f"the schema recorded in {QC_MANIFEST} ({len(manifest_cols)} "
            f"columns). 01 writes both in the same run, so this means one of "
            f"them is stale — re-run 01_build_cohort.py.\n"
            f"  declared by 01 but absent from the CSV: {dropped}\n"
            f"  present in the CSV but not declared:    {added}")
        print(f"  OK  cohort schema = {n} columns, matches "
              f"qc_manifest.json['cohort_columns'] from the Step-01 run that "
              f"wrote this CSV (no hard-coded count)", flush=True)

    source_cols = _schema_from_source()
    if source_cols is None:
        print(f"  note  could not recover COHORT_COLUMNS from "
              f"{os.path.basename(BUILD_COHORT_PY)} — source cross-check "
              f"skipped (non-fatal).", flush=True)
    elif set(source_cols) != observed:
        stale_missing = [c for c in source_cols if c not in observed]
        stale_extra = [c for c in cohort_cols if c not in set(source_cols)]
        print(f"  WARN  cohort CSV ({n} cols) differs from "
              f"01_build_cohort.COHORT_COLUMNS ({len(source_cols)} cols) — 01's "
              f"source has changed since this CSV was built. Re-run 01 before "
              f"trusting downstream results.\n"
              f"        in 01's source, not in the CSV: {stale_missing}\n"
              f"        in the CSV, not in 01's source: {stale_extra}", flush=True)
    else:
        print(f"  OK  cohort schema also matches "
              f"01_build_cohort.COHORT_COLUMNS ({len(source_cols)} columns)",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Smoke test: real row-downsample of the cohort "
                             "(df.sample(n=N, random_state=42)) after loading.")
    args = parser.parse_args()

    banner("Pelvic v3 — Step 02: engineer arrival predictor pool + carry pass-through")

    print(f"Reading {IN_COHORT}", flush=True)
    df = pd.read_csv(IN_COHORT, low_memory=False)
    print(f"  cohort shape: {df.shape}", flush=True)

    # The pass-through cohort columns (exact order preserved). The WIDTH is
    # whatever 01 emitted — verified against 01's own artifacts below, never
    # against a literal in this file.
    cohort_cols = df.columns.tolist()
    check_cohort_schema(cohort_cols)

    # Outcome plumbing (CONTRACTS §3/§4): every column a downstream script needs
    # as an outcome or a sensitivity arm must be present BEFORE we build the
    # pass-through, otherwise the failure surfaces several scripts later as a
    # KeyError with no provenance. THIS is the hard requirement — a genuinely
    # missing required column halts the run here, named, with the remedy.
    missing_carry = [c for c in REQUIRED_CARRY_COLS if c not in cohort_cols]
    assert not missing_carry, (
        f"Step 01 cohort CSV is missing {len(missing_carry)} column(s) that "
        f"downstream scripts read BY NAME: {missing_carry}\n"
        f"  cohort CSV : {IN_COHORT} ({len(cohort_cols)} columns)\n"
        f"  These are outcome / sensitivity / comparator columns, so the run "
        f"cannot continue: 02 would write a candidates.parquet that starves a "
        f"downstream script.\n"
        f"  Fix upstream, not here — add the column(s) to "
        f"01_build_cohort.COHORT_COLUMNS and re-run 01. Only remove a name from "
        f"REQUIRED_CARRY_COLS when NO downstream script reads it any more.")
    print(f"  OK  all {len(REQUIRED_CARRY_COLS)} required outcome/sensitivity/"
          f"comparator columns present in the Step 01 cohort", flush=True)

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE TEST] downsampled to {len(df):,}", flush=True)

    # Keys — keep as plain integers (no nulls verified in Step 01). Downstream
    # PAT-forked scripts coerce via to_numeric().astype("Int64"); int64 upcasts
    # cleanly and merges on (puf_year, inc_key).
    df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("int64")
    df["puf_year"] = pd.to_numeric(df["puf_year"], errors="coerce").astype("int64")

    # --------------------------------------------------------------
    # Numeric coercions of raw engineering inputs (keep NaN for now;
    # comparisons below null-bias to 0).
    # --------------------------------------------------------------
    age = pd.to_numeric(df["age"], errors="coerce")
    sbp = pd.to_numeric(df["sbp2"], errors="coerce")
    pulse = pd.to_numeric(df["pulse2"], errors="coerce")
    si = pd.to_numeric(df["shock_index"], errors="coerce")
    motor = pd.to_numeric(df["motor2"], errors="coerce")
    gcs = pd.to_numeric(df["gcs2"], errors="coerce")
    sex_male = pd.to_numeric(df["sex_male"], errors="coerce")
    mech = df["mechanism_cat"].astype("string").str.strip()
    trans = df["transport_cat"].astype("string").str.strip()

    banner(f"Engineering the {len(LASSO_X_COLS)} LASSO_X_COLS "
           f"(binaries null-biased -> 0)")

    # ---- Demographics (age split >=65; sex) ----
    df["age_65_74"] = ((age >= 65) & (age < 75)).fillna(False).astype(int)
    df["age_ge75"] = (age >= 75).fillna(False).astype(int)
    df["sex_female"] = (sex_male == 0).fillna(False).astype(int)   # NaN sex -> 0

    # ---- Vitals: BP (null-biased -> 0; a missing SBP is NOT hypotension) ----
    # sbp_lt90 exists 0/1 (with NaN) in the cohort CSV; null-bias -> 0.
    df["sbp_lt90"] = pd.to_numeric(df["sbp_lt90"], errors="coerce").fillna(0).astype(int)
    df["sbp_lt70"] = (sbp < 70).fillna(False).astype(int)
    # sbp_unrecordable ("no signs of life on arrival"): raw SBP2 == 0, captured
    # pre-mask in Step 01; already 0/1. Stays a LASSO candidate and stays in 05's
    # unpenalized refit; Fix 4 removes it from the INTEGER POINT SCALE only and
    # makes it a binary automatic-escalate override (CONTRACTS §5.1). Nothing to
    # special-case here — 02 and 04 treat it as any other candidate.
    df["sbp_unrecordable"] = pd.to_numeric(df["sbp_unrecordable"], errors="coerce").fillna(0).astype(int)

    # ---- Vitals: HR ----
    df["hr_ge120"] = (pulse >= 120).fillna(False).astype(int)

    # ---- Shock index: continuous X (NaN -> 0.7) + 2 thresholds ----
    df["shock_index"] = si.fillna(SI_FILL)
    df["si_ge0_9"] = (si >= 0.9).fillna(False).astype(int)
    df["si_ge1_0"] = (si >= 1.0).fillna(False).astype(int)

    # ---- Neuro / GCS (null-biased -> 0) ----
    df["gcs_motor_le5"] = (motor <= 5).fillna(False).astype(int)
    df["gcs_le8"] = (gcs <= 8).fillna(False).astype(int)

    # ---- Mechanism (mechanism_cat 100% defined) ----
    df["mech_high_energy"] = mech.isin(["MVC-occupant", "Motorcycle", "Pedestrian"]).fillna(False).astype(int)
    df["mech_mvc"] = (mech == "MVC-occupant").fillna(False).astype(int)
    df["mech_fall"] = (mech == "Fall").fillna(False).astype(int)

    # ---- Transport (transport_cat 100% defined) ----
    df["air_transport"] = (trans == "Air").fillna(False).astype(int)

    # ---- Comorbidity (coag) — LASSO candidates, already 0/1, no NaN ----
    df["cc_anticoag"] = pd.to_numeric(df["cc_anticoag"], errors="coerce").fillna(0).astype(int)
    df["cc_bleeding"] = pd.to_numeric(df["cc_bleeding"], errors="coerce").fillna(0).astype(int)

    # ---- ICD-anchored ring: PASS-THROUGH ONLY in v3, NOT a LASSO candidate ----
    # Fix 3 / D-2. Normalized in place so the parquet dtype is stable for the
    # consumers that legitimately still need it (09 subgroups; 11's
    # CORE_ADJUSTMENT + stratified change-point; 13's VIF audit; 15's Table 1).
    # Removing it from the DATA would break those; only the SCORE loses it.
    df["unstable_ring"] = pd.to_numeric(df["unstable_ring"], errors="coerce").fillna(0).astype(int)

    # --------------------------------------------------------------
    # Leakage audit + NaN-free / numeric asserts (CONTRACTS §5)
    # --------------------------------------------------------------
    banner("Leakage audit + NaN-free assertions")
    x_set = set(LASSO_X_COLS)
    intersect = x_set & FORBIDDEN_X
    assert not intersect, f"LEAKAGE: forbidden columns in LASSO_X_COLS: {intersect}"
    assert "major_hemorrhage_event" not in LASSO_X_COLS, \
        "major_hemorrhage_event (the primary outcome) CANNOT be a predictor"

    # --- Fix 3 / D-2 guard (CONTRACTS §13 G3a) -----------------------------
    # `unstable_ring` is a radiographic read, not information available
    # immediately upon arrival. It must be absent from the SCORE pool while
    # remaining present in the DATA (asserted separately below).
    assert "unstable_ring" not in LASSO_X_COLS, (
        "Fix 3 / D-2 VIOLATION: `unstable_ring` is back in LASSO_X_COLS. Ring "
        "stability is a radiographic determination and cannot enter a score "
        "that claims arrival-only inputs. It belongs in 11.CORE_ADJUSTMENT and "
        "13.VIF_AUDIT_VARS — not here (CONTRACTS §5, §6, §13 G3a).")

    # --- Fix 2b guard: neither outcome-defining timing field is ever a predictor
    # `finaldischarge_hrs` is the PRIMARY rule (death_hemorrhagic requires
    # <= 48 h); `finaldischarge_days` (<= 2 d) is the superseded calendar-day
    # crosswalk retained for the `*_dayrule` sensitivity columns only.
    for _timing_col in ("finaldischarge_hrs", "finaldischarge_days"):
        assert _timing_col not in LASSO_X_COLS, (
            f"LEAKAGE: `{_timing_col}` is the arrival-to-final-discharge offset "
            f"— outcome-defining (it builds death_hemorrhagic) and knowable only "
            f"at the end of the admission. It can never be a predictor.")
        assert _timing_col in FORBIDDEN_X, (
            f"`{_timing_col}` must be listed in FORBIDDEN_X (CONTRACTS §5): the "
            f"leakage assert is only as good as the forbidden set behind it.")

    assert len(LASSO_X_COLS) == N_LASSO_X_EXPECTED, (
        f"LASSO_X_COLS must be exactly {N_LASSO_X_EXPECTED} arrival predictors "
        f"(v3 = 18 - unstable_ring); got {len(LASSO_X_COLS)}")
    assert len(set(LASSO_X_COLS)) == N_LASSO_X_EXPECTED, \
        "LASSO_X_COLS has duplicate names"

    missing_x = [c for c in LASSO_X_COLS if c not in df.columns]
    assert not missing_x, f"X columns missing from df: {missing_x}"
    for c in LASSO_X_COLS:
        n_nan = int(df[c].isna().sum())
        assert n_nan == 0, f"{c} has {n_nan} NaN — forbidden in the X matrix"
        assert np.issubdtype(df[c].dtype, np.number), f"{c} is non-numeric ({df[c].dtype})"
    print(f"  OK  {len(LASSO_X_COLS)} X columns: numeric, NaN-free, no leakage", flush=True)
    print(f"  OK  set(LASSO_X_COLS) & FORBIDDEN_X == set()", flush=True)
    print(f"  OK  'unstable_ring' not in LASSO_X_COLS (Fix 3 / D-2)", flush=True)
    print(f"  OK  'finaldischarge_hrs' / 'finaldischarge_days' not in "
          f"LASSO_X_COLS and both in FORBIDDEN_X (Fix 2b)", flush=True)
    print(f"  OK  'sbp_unrecordable' IS in LASSO_X_COLS "
          f"(Fix 4 changes 05's point scale, not this pool)", flush=True)

    # --------------------------------------------------------------
    # Select final columns (every pass-through cohort col + the engineered X
    # names that are not already cohort columns) and write. Both counts are
    # computed here, not asserted: the 4 overlapping X names (sbp_lt90,
    # sbp_unrecordable, cc_anticoag, cc_bleeding) plus the engineered-but-not-
    # candidate `shock_index` and the retained-but-not-candidate `unstable_ring`
    # are already in cohort_cols and now hold engineered values.
    # --------------------------------------------------------------
    new_x = [c for c in LASSO_X_COLS if c not in cohort_cols]
    keep = cohort_cols + new_x
    # Sanity: keep has no duplicates and every X is present.
    assert len(keep) == len(set(keep)), "duplicate column in keep list"
    assert all(c in keep for c in LASSO_X_COLS), "an X column dropped from keep"

    out = df[keep].copy()

    # Post-selection contract check: every outcome / sensitivity / comparator
    # column a downstream script reads by name actually made it into the
    # parquet, and the deliberately-retained `unstable_ring` survived as a
    # pass-through even though it is no longer a candidate (CONTRACTS §6).
    missing_out = [c for c in REQUIRED_CARRY_COLS if c not in out.columns]
    assert not missing_out, f"columns dropped before write: {missing_out}"
    assert "unstable_ring" in out.columns, (
        "`unstable_ring` must remain a PASS-THROUGH column in "
        "candidates.parquet — 09/11/13/15 read it. Fix 3 removes it from the "
        "score, not from the data (CONTRACTS §6).")

    banner("Writing candidates.parquet")
    print(f"  candidate matrix shape: {out.shape}  "
          f"(= {len(cohort_cols)} pass-through + {len(new_x)} new X)", flush=True)
    print(f"  cohort_primary carried: {'cohort_primary' in out.columns}", flush=True)
    print(f"  outcome plumbing OK: primary='{'major_hemorrhage_event' in out.columns}' "
          f"co-primary(state)='{'patient_state_composite' in out.columns}' "
          f"co-primary(mortality)='{'death_hemorrhagic' in out.columns}' "
          f"secondary(MTP)='{'mtp_equivalent' in out.columns}'", flush=True)
    out.to_parquet(OUT_CANDIDATES, index=False)
    print(f"  wrote {OUT_CANDIDATES}", flush=True)
    print(f"  columns ({len(keep)}): {keep}", flush=True)

    # --------------------------------------------------------------
    # Coverage table — on the ANALYTIC cohort (cohort_primary == 1)
    # --------------------------------------------------------------
    banner("Candidate coverage table (analytic cohort, cohort_primary == 1)")
    ac = out[out["cohort_primary"] == 1]
    print(f"  analytic N = {len(ac):,}", flush=True)
    rows = []
    for c in LASSO_X_COLS:
        s = ac[c]
        n_pos = int(s.sum()) if s.dtype.kind in "biu" else int((s != 0).sum())
        rows.append({
            "Variable": c,
            "N": len(ac),
            "N_positive": n_pos,
            "Mean": round(float(s.mean()), 4),
            "Min": round(float(s.min()), 4),
            "Max": round(float(s.max()), 4),
        })
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT_COVERAGE, index=False)
    print(f"  wrote {OUT_COVERAGE}", flush=True)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(cov.to_string(index=False))

    print("\nPelvic v3 Step 02 complete.", flush=True)


if __name__ == "__main__":
    main()
