#!/usr/bin/env python3
"""
01_build_cohort.py — PRE-DECISION cohort build for the Pelvic Hemorrhage
Arrival Decision-Support Tool (v3), NTDB/TQP 2019-2024.

Fork of v2/code/01_build_cohort.py.  The COHORT FUNNEL IS UNCHANGED from v2
(same 7 pre-decision STROBE steps, same analytic N).  What changes in v3 is the
OUTCOME DEFINITION, via two reviewer/analyst-driven fixes:

  ---------------------------------------------------------------------------
  FIX 1 (S1, analyst-found) — unit-aware transfusion decoder
  ---------------------------------------------------------------------------
  `TQIP_HC_BLOOD_4HR` is documented as millilitres, but its values are bimodal
  and the decoder column `TQIP_HC_BLOOD_UNIT` is 100% blank.  Non-zero values
  cluster at 1-50 (modes 1,2,3,4,6 -> plainly UNITS) and at >=250 (modes 300,
  350, 700 -> plainly mL), with an almost-empty 101-199 gap between them.
  Under the v2 rule (`blood_4h >= 1000`) a patient given 36 UNITS in 4 h scored
  as "no substantial transfusion".  v3 splits on the empirical gap:

        substantial_transfusion = (blood_4h >= 1000)            # mL branch
                                | (4 <= blood_4h <= 100)        # units branch

  New/retained columns: `substantial_transfusion` (v3 primary component),
  `substantial_transfusion_ml_only` (the exact v2 rule, kept as a named
  sensitivity so the change is auditable), `blood_4h_scale` (units / ml /
  ambiguous / zero / missing), `mtp_equivalent` (Fix 7 secondary outcome —
  definitionally IDENTICAL to the new `substantial_transfusion`).
  Audit -> logs/fix1_transfusion_audit.md.

  ---------------------------------------------------------------------------
  FIX 2 (W1, reviewer request) — hemorrhagic-death component, COMPLIED WITH IN FULL
  ---------------------------------------------------------------------------
  All-cause in-hospital death is too heterogeneous to sit in a HEMORRHAGE
  composite (a third of these deaths carry head AIS >= 4, and half occur days
  to weeks after arrival).  The request was for BOTH halves — death <= 48 h
  AND head AIS < 4 — and BOTH are now built:

    death_hemorrhagic = death_inhosp & (ais_head < 4)
                                     & (finaldischarge_hrs <= 48)   <- PRIMARY
    death_hemorrhagic_headais = death_inhosp & (ais_head < 4)       <- sens.
    death_le48h       = death_inhosp & (finaldischarge_hrs <= 48)   <- sens.
    death_hemorrhagic_dayrule / death_le48h_dayrule
                      = the SUPERSEDED calendar-day (<=2) versions  <- sens.
    death_hemorrhagic_dayfallback  = PRIMARY, but an untimed death
                                     with day index <= 2 is kept    <- sens.
    death_hemorrhagic_edincl       = PRIMARY, but an untimed ED
                                     death is kept (an ED death is
                                     necessarily within 48 h)       <- sens.
    death_hemorrhagic_missing_incl = as PRIMARY but ANY missing hours
                                     value counts as <= 48 h        <- bound
    death_inhosp      = all-cause in-hospital death                 <- sens.
    death_ed          = ED_DISPOSITION_CODE == 'MOR'                <- sens.

  The <= 48 h half was initially believed infeasible because all 33 *_DATE /
  *_TIME columns of the COMPILED export are 0.0000% filled.  That premise was
  false twice over.  First: the compiled export is a lossy projection, and the
  RAW TQP PUF files carry `FINALDISCHARGEDAYS` at 98.8-99.8% fill.  Second, and
  corrected here: they ALSO carry **`FINALDISCHARGEHRS`** — elapsed hours to two
  decimals, ~98.8% filled — which an earlier build missed because the header
  search had been widened from `*_DATE`/`*_TIME` to `*DAYS` but never to `*HRS`.
  The claim that "AY 2019-2024 has no minute- or hour-level fields" was FALSE
  and is withdrawn.

  Both fields are ALREADY ARRIVAL-ANCHORED and are used DIRECTLY — they must NOT
  be differenced against `HOSPITALARRIVAL*`, which is INCIDENT-anchored (that
  subtraction mixes origins and manufactures impossible negative values).

  The timing cut is `finaldischarge_hrs <= 48` — W1's request computed EXACTLY.
  The old `finaldischarge_days <= 2` rule is retained only as a sensitivity: the
  day field is a calendar-day INDEX, not elapsed time, and day 3 straddles 48 h
  (24.13-71.63 h), so the day rule was 100% specific but materially
  under-sensitive.  Full evidence: logs/timing_evidence_v3.md + the crosswalk in
  logs/fix2_death_audit.md.

  MISSING-VALUE RULE: a missing `finaldischarge_hrs` is treated as NOT within
  48 h, so such a death does not enter the primary death component.  The rule
  shrinks rather than inflates the outcome, and its cost is BOUNDED (not
  assumed) by the retained `death_hemorrhagic_missing_incl` column.

  ---------------------------------------------------------------------------
  DEFECT FIX carried in the same pass — AIS chapter-0 mis-parse (v2-inherited)
  ---------------------------------------------------------------------------
  The compiled AIS export stores `AISPreDot` as a FLOAT, so a chapter-0 code
  such as '010008' round-trips as 10008.0 -> '10008' and the body-region digit
  read from position 0 is actually the code's SECOND digit.  External/chapter-0
  codes were therefore being scored as HEAD (7,026 rows nationally) or CHEST
  (192 rows) injuries.  Fixed with `.str.zfill(6)` before the region digit is
  taken; direction is inflation-only.  This matters far more in v3 than it did
  in v2, because Fix 2 promotes `ais_head` from an adjustment covariate to a
  constituent of the PRIMARY OUTCOME.  Registered as a deviation from the v2 SAP.
  The AIS-6 exclusion is region-independent, so the cohort funnel is unaffected.

  The composites are recomputed on the new components:

        major_hemorrhage_event  = received_hc | death_hemorrhagic
                                              | substantial_transfusion
        patient_state_composite = death_hemorrhagic | substantial_transfusion
        major_hemorrhage_event_headais = received_hc | death_hemorrhagic_headais
                                              | substantial_transfusion
                                  (head-AIS-half-only variant, sensitivity)
        major_hemorrhage_event_v2def = received_hc | death_inhosp
                                              | substantial_transfusion_ml_only
                                  (the exact v2 primary, auditable sensitivity
                                   AND the fork-fidelity regression gate)

  Audit -> logs/fix2_death_audit.md.

Carried over from v2 (unchanged):
  * NO HC-inclusion filter.  Hemorrhage control (HC) is an OUTCOME component,
    not a cohort filter.  `f_keep` does not conjoin `f_hc`.
  * 7-step pre-decision STROBE funnel:
        adult -> ring -> blunt -> direct-arrival -> not-DOA/AIS6 -> valid-outcome
    Emits the ordered exclusion counts to derived/cohort_exclusions.json.
  * Predictor/flag columns: unstable_ring (ICD S32.81*), cc_anticoag
    (preexisting code 31), cc_bleeding (code 4), sbp_unrecordable
    (SBP2==0 sentinel captured BEFORE the 0->NaN mask).
  * Shared MNAR table derived/informative_missingness.csv (composite &
    mortality rate in vitals-MISSING vs vitals-OBSERVED strata, by year).
  * Real --sample N CLI arg caps the MAIN scan (nrows) + co-caps the ascending
    inc_key side tables by ceiling for a fast end-to-end smoke test.

Outputs (CONTRACTS.md §7, row 01 — all under v3/):
    derived/pelvic_cohort_predecision.csv   (cohort schema, CONTRACTS §3 + v3)
    derived/cohort_exclusions.json          (ordered {step_label: N_remaining})
    derived/informative_missingness.csv     (shared MNAR table, §10)
    logs/cohort_flow.txt                    (human-readable STROBE flow)
    logs/missingness.csv                    (per-variable missingness)
    logs/qc_manifest.json                   (machine-readable manifest)
    logs/unknown_hc_codes.csv               (HC-type adjudication worksheet)
    logs/fix1_transfusion_audit.md          (v3 NEW — Fix 1 verification)
    logs/fix2_death_audit.md                (v3 NEW — Fix 2 verification)

Reads (chunked, filtered — never whole-table):
    _DATA/.../compiled_PUF_AY_2019_2024_with_VL.csv (MAIN, 7,327,714 rows)
    _DATA/.../ntdb_2019_2024_icd_diagnoses.csv      (S32 cohort + S32.81*)
    _DATA/.../ntdb_2019_2024_ais_diagnoses.csv      (AIS regions, AIS-6 excl.)
    _DATA/.../ntdb_2019_2024_preexisting.csv        (Charlson + codes 31/4)
    _DATA/NTDB/PUF/PUF AY {2019..2024}/CSV/PUF_TRAUMA.csv
                                                   (RAW: FINALDISCHARGEDAYS)

GATE (full run, no --sample): cohort_primary==1 -> N = 239,736; CSV rows =
291,766.  The funnel is untouched by v3, so any deviation is a regression.

Interpreter: python3.  Seed: 42.  Idempotent (overwrites its outputs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# §1. Paths (absolute module root — PAT_TBI house pattern, CONTRACTS §1).
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

# NTDB_DATA is read lazily from the environment (see config.py); importing this
# module for inspection does not require it, running the build does.
DATA_DIR = config.ntdb_compiled_dir()

MAIN_CSV = os.path.join(DATA_DIR, "compiled_PUF_AY_2019_2024_with_VL.csv")
ICD_CSV = os.path.join(DATA_DIR, "ntdb_2019_2024_icd_diagnoses.csv")
AIS_CSV = os.path.join(DATA_DIR, "ntdb_2019_2024_ais_diagnoses.csv")
PRE_CSV = os.path.join(DATA_DIR, "ntdb_2019_2024_preexisting.csv")

# --- v3 W1 full-comply: RAW TQP PUF day-offset timing ----------------------
# The compiled export above is a LOSSY PROJECTION of the raw PUF: all 33 of its
# *_DATE / *_TIME columns are 0.0000% filled across all 7,327,714 rows and all
# six admission years.  The RAW PUF files carry day-resolution offsets under a
# *DAYS naming scheme, which is exactly why a name-based search of the export
# never surfaced them.  Full audit: v3/logs/timing_evidence_v3.md.
PUF_DIR = config.ntdb_puf_dir()
PUF_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)
PUF_TRAUMA_TMPL = os.path.join(PUF_DIR, "PUF AY {year}", "CSV", "PUF_TRAUMA.csv")
# Dictionary definition: "Time from ED/Hospital Arrival to Final Discharge
# (days)".  Verified present with key `inc_key` in all six raw years.
PUF_FD_COL = "FINALDISCHARGEDAYS"
# ---- v3 DEFECT FIX: the raw PUF ALSO carries an HOURS field ----------------
# `FINALDISCHARGEHRS` is present in all six raw years at ~98.8% fill, with
# TWO-DECIMAL precision (min 0.0).  The earlier evidence pass searched raw
# headers for `MIN` and `HOUR` and concluded "AY 2019-2024 has no minute- or
# hour-level fields" — but the fields are named `*HRS`, so the search missed
# them.  That is the SAME failure mode as the original `*_DATE`-only search
# which missed `*DAYS`.  Corrected here.
#
# Why this matters: `FINALDISCHARGEDAYS` is a CALENDAR-DAY INDEX, not elapsed
# time.  Verified crosswalk (AY 2021): day 1 -> 0.00-23.92 h, day 2 ->
# 0.02-47.90 h, day 3 -> 24.13-71.63 h, day 4 -> 48.52-95.05 h.  Day 3
# STRADDLES 48 h.  So `days <= 2` is 100% specific (0 rows with hrs > 48) but
# badly UNDER-sensitive: in AY 2021 alone, 109,008 records at hrs <= 48 fall on
# day >= 3.  The hours field computes the requested rule EXACTLY.
PUF_FD_HRS_COL = "FINALDISCHARGEHRS"
# NOT joined: HOSPITALARRIVALHRS (46.8% filled, and INCIDENT-anchored — the
# same anchor mismatch that makes the DAYS subtraction wrong).
PUF_KEY_COL = "inc_key"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Domain constants (CONTRACTS §3, §4).
# ---------------------------------------------------------------------------
# S32 pelvic-ring inclusion prefixes (longest-prefix match on decimal codes).
RING_PREFIXES = ("S32.1", "S32.3", "S32.5", "S32.6", "S32.81", "S32.82")
ACET_PREFIX = "S32.4"  # isolated acetabulum -> sensitivity flag only
UNSTABLE_RING_PREFIX = "S32.81"  # disrupted/unstable ring (S32.82 = undisrupted)

# Preexisting NTDS numeric codes NOT in the Charlson buckets (CONTRACTS §3):
CODE_ANTICOAG = "31"   # anticoagulant therapy -> cc_anticoag
CODE_BLEEDING = "4"    # bleeding disorder     -> cc_bleeding

# --- v3 FIX 1 (S1): unit-aware transfusion decoder --------------------------
# TQIP_HC_BLOOD_4HR is bimodal and TQIP_HC_BLOOD_UNIT is 100% blank, so the
# scale must be inferred from magnitude.  The empirical gap between the two
# modes is 101-199 (see logs/fix1_transfusion_audit.md), so the split is 100/200.
BLOOD_UNITS_MAX = 100            # 0 < v <= 100  -> reads as UNITS
BLOOD_ML_MIN_SCALE = 200         # v >= 200      -> reads as mL
BLOOD_UNITS_MIN_SUBSTANTIAL = 4  # >=4 U/4h  = critical administration threshold
BLOOD_ML_MIN_SUBSTANTIAL = 1000  # >=1000 mL/4h = the v2 rule (~4 U), kept as
                                 # substantial_transfusion_ml_only sensitivity

# --- v3 FIX 2 (W1): hemorrhagic-death component -----------------------------
# A death with a severe head injury (head AIS >= 4) is not attributable to
# pelvic hemorrhage; it is excluded from the primary composite's death arm.
AIS_HEAD_SEVERE = 4              # death_hemorrhagic requires ais_head < 4
ED_DEATH_CODE = "MOR"            # ED_DISPOSITION_CODE value for died-in-ED

# ---- the <=48 h half of W1, computed EXACTLY -------------------------------
# PRIMARY RULE: `finaldischarge_hrs <= 48`.  `FINALDISCHARGEHRS` is a genuine
# elapsed-time measure in hours to two decimals, arrival-anchored, ~98.8%
# filled.  This is the requested "<= 48 h" rule computed exactly, not approximated.
DEATH_LE48H_MAX_HOURS = 48.0
# SENSITIVITY RULE (retained, and superseded as primary): the calendar-day
# index `finaldischarge_days <= 2`.  It is 100% SPECIFIC (no record with
# hrs > 48 lands on day <= 2) but badly UNDER-SENSITIVE, because the day field
# is a calendar index rather than elapsed time and day 3 straddles 48 h
# (24.13-71.63 h in AY 2021).  Kept so the day-vs-hour crosswalk is reportable
# and the earlier day-based result stays auditable.
DEATH_LE48H_MAX_DAYS = 2

# Blunt-mechanism penetrating exclusions (first-3 of dot-stripped e-code).
PENETRATING_ECODES = frozenset({
    "W32", "W33", "W34", "X72", "X73", "X74", "X93", "X94", "X95",
    "Y22", "Y23", "Y24",                       # firearm
    "W26", "X78", "X99", "Y28",                # stab / sharp
})

# HC-qualifying surgical types (all 8 are HC-qualifying surgical).
HC_SURG_SET = frozenset({
    "LAP", "PELVIC_PACK", "THOR", "STERN", "NECK", "EXTREM", "MANGLED", "SKIN_ST",
})
SURG_CLASS = {
    "LAP": "definitive", "PELVIC_PACK": "temporizing", "THOR": "definitive",
    "STERN": "definitive", "NECK": "definitive", "EXTREM": "definitive",
    "MANGLED": "temporizing", "SKIN_ST": "other",
}
PELVIC_SPECIFIC_SURG = frozenset({"LAP", "PELVIC_PACK"})
ANGIO_THERAPEUTIC = frozenset({"ANGIO_EMBO", "ANGIO_STENT"})  # 'ANGIO' = diagnostic

# Charlson: NTDB PREEXISTINGCONDITION numeric code -> Charlson bucket
# (charlson = unweighted count of the 10 derivable buckets; house convention).
CHARLSON_CODE_BUCKET = {
    "7": "chf",           # Congestive Heart Failure
    "9": "renal",         # Chronic Renal Failure
    "10": "cva",          # Cerebrovascular Accident
    "11": "dm",           # Diabetes Mellitus
    "5": "cancer",        # Currently Receiving Chemotherapy for Cancer
    "12": "cancer",       # Disseminated Cancer
    "17": "mi",           # History of Myocardial Infarction
    "34": "mi",           # Myocardial Infarction
    "18": "pvd",          # History of Peripheral Vascular Disease
    "35": "pvd",          # Peripheral Arterial Disease
    "23": "copd",         # Chronic Obstructive Pulmonary Disease
    "25": "liver",        # Cirrhosis
    "26": "dementia",     # Dementia
}

# MAIN columns to read (explicit names; avoids the duplicated
# HOSPITAL_ARRIVAL_DATE / 'Unnamed: 82' / raw ICD10 — none is selected).
MAIN_COLS = [
    "inc_key", "PUF_YEAR",
    "AGE_NUMBER", "AGE_UNITS", "SEX", "RACE", "ETHNICITY", "PAYMENT_SOURCE",
    "SBP2", "P2", "GCS2", "MOTOR2", "ISS",
    "CAUSE_E_CODES10", "HOSPITAL_TRANSFER", "PREHOSPITAL_ARREST",
    "TRANS", "VERIFICATION_LEVEL",
    "TQIP_HC_SURG_TYPE", "TQIP_HC_ANGIO", "TQIP_HC_EMBO_SITE",
    "TQIP_HC_BLOOD_4HR", "TQIP_HC_BLOOD_24HR", "TQIP_HC_BLOOD_UNIT",
    # v3 Fix 7 (F1) — exploratory balanced-resuscitation subgroup (CONTRACTS §2).
    # Verified present in the MAIN header at 0-indexed positions 126 / 130
    # (1-indexed 127 / 131, as the contract states).  NOT decoded: the subgroup
    # is exploratory, so these are emitted raw-numeric and only documented.
    "TQIP_HC_PLASMA_4HR", "TQIP_HC_PLATE_4HR",
    "TOTAL_DAYS_ICU", "VENTDAYS",
    "DC_DISPOSITION_CODE", "ED_DISPOSITION_CODE",
    # timing-gate verification columns (expected 0% filled):
    "ARRIVAL_DATE", "DISCHARGE_DATE", "ED_DC_DATE",
    "TQIP_HC_SURG_DATE", "TQIP_HC_ANGIO_DATE",
]

CHUNK_MAIN = 200_000
CHUNK_SIDE = 1_000_000

# Adjustment-set covariates (CONTRACTS §6) — used only for the >5%-missing note.
ADJUSTMENT_COVARS = [
    "age", "sex_male", "race_eth_cat", "payment_cat", "sbp2", "pulse2",
    "motor2", "gcs2", "iss", "ais_head", "mechanism_cat", "transport_cat",
    "acs_level", "charlson",
]

# Final CSV column order — the contract (CONTRACTS §3 + the v3 additions).
# DO NOT reorder the v2 columns; v3 columns are inserted beside their siblings
# (every downstream reader indexes by NAME, never by position).
COHORT_COLUMNS = [
    # identifiers / exposure-context
    "inc_key", "puf_year", "cohort_primary", "has_acetabulum", "hc_route",
    "surg_hc", "angio_hc", "embo_any", "embo_pelvis",
    "surg_type_raw", "angio_raw", "embo_site_raw", "surg_class",
    "pelvic_specific_surg",
    # outcomes & components
    #   v3 NEW: the whole death family (death_hemorrhagic is now head-AIS AND
    #   <=48 h), finaldischarge_days, blood_4h_scale,
    #   substantial_transfusion_ml_only, mtp_equivalent, plasma/platelet,
    #   major_hemorrhage_event_headais, major_hemorrhage_event_v2def
    "death_inhosp", "death_hemorrhagic", "death_hemorrhagic_headais",
    "death_hemorrhagic_missing_incl", "death_hemorrhagic_dayrule",
    "death_hemorrhagic_dayfallback", "death_hemorrhagic_edincl",
    "death_le48h", "death_le48h_dayrule", "death_ed",
    "finaldischarge_hrs", "finaldischarge_days",
    "blood_4h", "blood_4h_scale", "plasma_4h", "platelet_4h",
    "icu_los", "vent_days",
    "received_hc", "substantial_transfusion",
    "substantial_transfusion_ml_only", "mtp_equivalent",
    "major_hemorrhage_event", "major_hemorrhage_event_headais",
    "major_hemorrhage_event_v2def",
    "patient_state_composite",
    # physiology / covariates
    "sbp2", "pulse2", "shock_index", "sbp_lt90", "sbp_unrecordable",
    "motor2", "gcs2", "iss", "ais_head", "ais_head_ge3", "ais_chest", "ais_abd",
    "age", "age_ge65", "sex_male", "race_eth_cat", "ethnicity_hispanic",
    "payment_cat", "mechanism_cat", "transport_cat", "acs_level", "charlson",
    # flags
    "unstable_flag", "tbi_severe", "moribund",
    "cc_anticoag", "cc_bleeding", "unstable_ring",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _clean_str(series: pd.Series) -> pd.Series:
    """NaN -> '', strip. (dtype=str reads leave real NaN as float nan.)"""
    return series.fillna("").astype(str).str.strip()


def _mechanism_from_ecode(ec3: str) -> str:
    """Map first-3 of a dot-stripped uppercased ICD-10 e-code to subtype."""
    if not ec3 or len(ec3) < 3:
        return "Other blunt"
    letter = ec3[0]
    try:
        num = int(ec3[1:3])
    except ValueError:
        return "Other blunt"
    if letter == "V":
        if 30 <= num <= 79:
            return "MVC-occupant"
        if 20 <= num <= 29:
            return "Motorcycle"
        if 1 <= num <= 9:
            return "Pedestrian"
        return "Other blunt"            # V10-V19 etc.
    if letter == "W":
        if 0 <= num <= 19:
            return "Fall"
        return "Other blunt"
    return "Other blunt"                # X/Y non-penetrating, animal, struck...


def _chunk_first_key(chunk: pd.DataFrame) -> str:
    """First (== minimum, ascending files) inc_key of a chunk, cleaned."""
    return str(chunk["inc_key"].iloc[0]).strip()


def _past_ceiling(first_key: str, ceiling) -> bool:
    """True once an ascending-sorted chunk's minimum inc_key exceeds the sample
    ceiling (12-digit fixed-width numeric keys -> int compare, str fallback)."""
    if ceiling is None:
        return False
    try:
        return int(first_key) > int(ceiling)
    except (ValueError, TypeError):
        return str(first_key) > str(ceiling)


def get_sample_ceiling(sample_n: int) -> str:
    """Smoke mode: read the first `sample_n` MAIN rows, return their max inc_key.
    Because ICD/AIS/PRE are sorted ascending by inc_key, every candidate among
    the first `sample_n` MAIN encounters has a key <= this ceiling, so capping
    the side scans at the ceiling keeps linkage exact while staying fast."""
    df = pd.read_csv(MAIN_CSV, usecols=["inc_key"], dtype=str, nrows=sample_n)
    keys = df["inc_key"].fillna("").astype(str).str.strip()
    keys = keys[keys != ""]
    return keys.max() if len(keys) else None


# ---------------------------------------------------------------------------
# Pass 1 — ICD table -> ring / acetabulum inc_key sets (+ S32 freqs + S32.81*).
# ---------------------------------------------------------------------------
def build_s32_sets(ceiling=None):
    print("[1/5] Scanning ICD diagnoses for S32 pelvic codes ...", flush=True)
    t0 = time.time()
    ring_set: set[str] = set()
    acet_set: set[str] = set()
    prefix_patients: dict[str, set] = {p: set() for p in RING_PREFIXES}
    prefix_patients[ACET_PREFIX] = set()

    n_rows = 0
    for chunk in pd.read_csv(
        ICD_CSV, usecols=["inc_key", "ICDDIAGNOSISCODE"],
        dtype=str, chunksize=CHUNK_SIDE,
    ):
        if len(chunk) and _past_ceiling(_chunk_first_key(chunk), ceiling):
            break
        n_rows += len(chunk)
        code = chunk["ICDDIAGNOSISCODE"].fillna("").astype(str).str.strip().str.upper()
        m = code.str.startswith("S32")
        if not m.any():
            continue
        ik = chunk.loc[m, "inc_key"].astype(str).str.strip()
        sub = code[m]
        for p in RING_PREFIXES:
            sel = sub.str.startswith(p)
            if sel.any():
                keys = ik[sel]
                ring_set.update(keys)
                prefix_patients[p].update(keys)
        sel_ac = sub.str.startswith(ACET_PREFIX)
        if sel_ac.any():
            keys = ik[sel_ac]
            acet_set.update(keys)
            prefix_patients[ACET_PREFIX].update(keys)

    s32_freq = {p: len(prefix_patients[p]) for p in list(RING_PREFIXES) + [ACET_PREFIX]}
    # NEW: S32.81* disrupted/unstable-ring set (built internally in the source and
    # discarded; here it is returned and merged as the 0/1 `unstable_ring` column).
    unstable_ring_set = set(prefix_patients[UNSTABLE_RING_PREFIX])
    print(f"      ICD rows={n_rows:,}  ring_patients={len(ring_set):,}  "
          f"acetabulum_patients={len(acet_set):,}  "
          f"unstable_ring(S32.81*)={len(unstable_ring_set):,}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ring_set, acet_set, s32_freq, unstable_ring_set


# ---------------------------------------------------------------------------
# Pass 2 — AIS table -> region maxima (head/chest/abd) + any-AIS-6 set.
# ---------------------------------------------------------------------------
def build_ais(candidate: set, ceiling=None):
    print("[2/5] Scanning AIS diagnoses (regions + AIS-6) ...", flush=True)
    t0 = time.time()
    region_map = {"1": "ais_head", "4": "ais_chest", "5": "ais_abd"}
    parts = []
    ais6: set[str] = set()
    for chunk in pd.read_csv(
        AIS_CSV, usecols=["inc_key", "AISPreDot", "AISSeverity"],
        dtype=str, chunksize=CHUNK_SIDE,
    ):
        if len(chunk) and _past_ceiling(_chunk_first_key(chunk), ceiling):
            break
        chunk["inc_key"] = chunk["inc_key"].astype(str).str.strip()
        chunk = chunk[chunk["inc_key"].isin(candidate)]
        if chunk.empty:
            continue
        sev = pd.to_numeric(chunk["AISSeverity"], errors="coerce")
        predot = (chunk["AISPreDot"].fillna("").astype(str)
                  .str.replace(r"\.0$", "", regex=True).str.strip())
        # ---- v3 DEFECT FIX (inherited from v2): restore the leading zero ----
        # The compiled AIS export writes AISPreDot as a FLOAT, so a chapter-0
        # code such as '010008' round-trips as 10008.0 -> '10008' — five digits,
        # leading zero destroyed.  Taking [0] of that string then reads the
        # SECOND digit as the body-region chapter, so '010008' (hypothermia, an
        # External/chapter-0 code) was being scored as a HEAD injury.
        # Nationally 12,898 of 25,483,291 AIS rows are 5-digit; every one is a
        # true chapter-0 code, and the mis-parse maps 7,026 of them to region 1
        # (HEAD) and 192 to region 4 (CHEST).  zfill(6) restores the chapter
        # digit before it is read.  Direction is INFLATION-ONLY: the fix can
        # only move a code OUT of a mapped region into unmapped chapter 0, never
        # the reverse.  Canonical AIS predot is 6 digits; nothing longer exists
        # in this export, so zfill is a no-op for the other 25,470,393 rows.
        predot = predot.str.zfill(6)
        region = predot.str[0]
        # AIS-6 exclusion = ANY body region severity == 6.  Region-independent,
        # so the cohort funnel is untouched by the zfill fix.
        ais6.update(chunk.loc[sev == 6, "inc_key"])
        # Region maxima use valid severities 1..6 only (drop 9=unknown / NaN).
        valid = sev.isin([1, 2, 3, 4, 5, 6]) & region.isin(region_map.keys())
        if valid.any():
            parts.append(pd.DataFrame({
                "inc_key": chunk.loc[valid, "inc_key"].values,
                "region_col": region[valid].map(region_map).values,
                "sev": sev[valid].values,
            }))
    if parts:
        long = pd.concat(parts, ignore_index=True)
        wide = (long.groupby(["inc_key", "region_col"])["sev"].max()
                .unstack("region_col"))
        for c in ("ais_head", "ais_chest", "ais_abd"):
            if c not in wide.columns:
                wide[c] = np.nan
        ais_regions = wide[["ais_head", "ais_chest", "ais_abd"]].reset_index()
    else:
        ais_regions = pd.DataFrame(columns=["inc_key", "ais_head", "ais_chest", "ais_abd"])
    print(f"      AIS region rows kept for {len(ais_regions):,} patients; "
          f"AIS-6 patients={len(ais6):,}  ({time.time()-t0:.0f}s)", flush=True)
    return ais_regions, ais6


# ---------------------------------------------------------------------------
# Pass 3 — Preexisting table -> Charlson index + cc_anticoag(31) + cc_bleeding(4).
# ---------------------------------------------------------------------------
def build_charlson(candidate: set, ceiling=None):
    print("[3/5] Scanning preexisting conditions (Charlson + codes 31/4) ...",
          flush=True)
    t0 = time.time()
    parts = []
    anticoag_set: set[str] = set()   # NEW: cc_anticoag (code 31)
    bleeding_set: set[str] = set()   # NEW: cc_bleeding (code 4)
    for chunk in pd.read_csv(
        PRE_CSV,
        usecols=["inc_key", "PREEXISTINGCONDITION", "PREEXISTINGCONDITIONANSWER"],
        dtype=str, chunksize=CHUNK_SIDE,
    ):
        if len(chunk) and _past_ceiling(_chunk_first_key(chunk), ceiling):
            break
        chunk["inc_key"] = chunk["inc_key"].astype(str).str.strip()
        chunk = chunk[chunk["inc_key"].isin(candidate)]
        if chunk.empty:
            continue
        ans = pd.to_numeric(chunk["PREEXISTINGCONDITIONANSWER"], errors="coerce")
        chunk = chunk[ans == 1]  # 1 = Yes ; 2 = No ; NaN = unknown
        if chunk.empty:
            continue
        cond = (chunk["PREEXISTINGCONDITION"].fillna("").astype(str)
                .str.replace(r"\.0$", "", regex=True).str.strip())
        # NEW: codes 31/4 are NOT Charlson components — collect their inc_key sets
        # here (after the answer==1 filter, on normalized `cond`); do NOT add them
        # to CHARLSON_CODE_BUCKET (that would corrupt `charlson`).
        anticoag_set.update(chunk.loc[cond == CODE_ANTICOAG, "inc_key"].values)
        bleeding_set.update(chunk.loc[cond == CODE_BLEEDING, "inc_key"].values)
        bucket = cond.map(CHARLSON_CODE_BUCKET)
        keep = bucket.notna()
        if keep.any():
            parts.append(pd.DataFrame({
                "inc_key": chunk.loc[keep, "inc_key"].values,
                "bucket": bucket[keep].values,
            }))
    if parts:
        long = pd.concat(parts, ignore_index=True).drop_duplicates()
        charlson = (long.groupby("inc_key")["bucket"].nunique()
                    .rename("charlson").reset_index())
    else:
        charlson = pd.DataFrame(columns=["inc_key", "charlson"])
    print(f"      Charlson computed for {len(charlson):,} patients "
          f"(>=1 comorbidity); cc_anticoag(31)={len(anticoag_set):,}  "
          f"cc_bleeding(4)={len(bleeding_set):,}  ({time.time()-t0:.0f}s)",
          flush=True)
    return charlson, anticoag_set, bleeding_set


# ---------------------------------------------------------------------------
# Pass 4 — RAW TQP PUF -> FINALDISCHARGEDAYS (v3 W1 full-comply).
# ---------------------------------------------------------------------------
def build_finaldischarge_days(candidate: set):
    """Join `FINALDISCHARGEHRS` + `FINALDISCHARGEDAYS` from the RAW PUF files.

    IMPLEMENTATION GOTCHA (evidence pack §8) — do NOT compute
    `FINALDISCHARGE* - HOSPITALARRIVAL*`.  The two families have DIFFERENT
    ANCHORS: `FINALDISCHARGE*` is measured from ED/hospital ARRIVAL, while
    `HOSPITALARRIVAL*` is measured from the INCIDENT.  Subtracting them mixes
    origins and produces impossible negative "LOS" values.  Both
    `FINALDISCHARGEHRS` and `FINALDISCHARGEDAYS` are already arrival-anchored —
    they are used DIRECTLY.

    No inc_key-ceiling short-circuit is applied here: unlike the ICD/AIS/PRE
    side tables, the raw PUF files are NOT assumed to be inc_key-sorted, so the
    scan is a full read filtered to the candidate set (correctness over speed).
    """
    print("[4/5] Scanning RAW PUF_TRAUMA for FINALDISCHARGEHRS/DAYS ...",
          flush=True)
    t0 = time.time()
    cols = ["inc_key", "finaldischarge_hrs", "finaldischarge_days"]
    if not candidate:
        print("      candidate set empty (smoke mode) -> skipping raw PUF scan",
              flush=True)
        return pd.DataFrame(columns=cols), OrderedDict()

    parts = []
    per_year = OrderedDict()
    for yr in PUF_YEARS:
        path = PUF_TRAUMA_TMPL.format(year=yr)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        n_rows = 0
        n_filled_d = 0
        n_filled_h = 0
        got = []
        for chunk in pd.read_csv(
            path, usecols=[PUF_KEY_COL, PUF_FD_COL, PUF_FD_HRS_COL],
            dtype=str, chunksize=CHUNK_SIDE, low_memory=False,
        ):
            n_rows += len(chunk)
            fd = _clean_str(chunk[PUF_FD_COL])
            fh = _clean_str(chunk[PUF_FD_HRS_COL])
            n_filled_d += int((fd != "").sum())
            n_filled_h += int((fh != "").sum())
            ik = chunk[PUF_KEY_COL].astype(str).str.strip()
            keep = ik.isin(candidate)
            if keep.any():
                got.append(pd.DataFrame({
                    "inc_key": ik[keep].values,
                    "finaldischarge_hrs": fh[keep].values,
                    "finaldischarge_days": fd[keep].values,
                }))
        sub = (pd.concat(got, ignore_index=True) if got
               else pd.DataFrame(columns=cols))
        per_year[str(yr)] = {
            "raw_rows": int(n_rows),
            "raw_fill_pct_days": round(100.0 * n_filled_d / n_rows, 4) if n_rows else 0.0,
            "raw_fill_pct_hrs": round(100.0 * n_filled_h / n_rows, 4) if n_rows else 0.0,
            "cohort_keys_matched": int(len(sub)),
        }
        parts.append(sub)
        print(f"      AY {yr}: raw rows={n_rows:,}  HRS fill="
              f"{per_year[str(yr)]['raw_fill_pct_hrs']}%  DAYS fill="
              f"{per_year[str(yr)]['raw_fill_pct_days']}%  cohort matches="
              f"{len(sub):,}", flush=True)

    fd_days = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=cols)
    # ---- v3 DEFECT FIX: real duplicate-key gate (CONTRACTS join contract) ----
    # This MUST fail on a duplicated inc_key rather than silently repairing it.
    # The previous implementation called drop_duplicates() and THEN asserted the
    # row count was unchanged — a tautology that could never fire.  A duplicate
    # key here would silently multiply cohort rows in the downstream left merge.
    n_keys = int(len(fd_days))
    n_unique = int(fd_days["inc_key"].nunique())
    n_dupes = n_keys - n_unique
    assert n_dupes == 0, (
        f"raw-PUF join: {n_dupes:,} duplicate inc_key row(s) across the six "
        f"PUF_TRAUMA files ({n_keys:,} rows, {n_unique:,} unique keys). The "
        "join must be 1:1; a duplicate would multiply cohort rows.")
    for c, src in (("finaldischarge_hrs", PUF_FD_HRS_COL),
                   ("finaldischarge_days", PUF_FD_COL)):
        fd_days[c] = pd.to_numeric(fd_days[c], errors="coerce")
        neg = int((fd_days[c] < 0).sum())
        assert neg == 0, f"{src}: {neg} negative value(s) — anchor is wrong"
    print(f"      matched {len(fd_days):,} of {len(candidate):,} candidate "
          f"inc_keys ({round(100.0*len(fd_days)/len(candidate), 3)}%); "
          f"duplicate-key check: {n_keys:,} rows / {n_unique:,} unique -> "
          f"{n_dupes} dupes; non-null hrs="
          f"{int(fd_days['finaldischarge_hrs'].notna().sum()):,} days="
          f"{int(fd_days['finaldischarge_days'].notna().sum()):,}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    per_year["_join_check"] = {
        "rows": n_keys, "unique_inc_keys": n_unique, "duplicate_rows": n_dupes,
        "assert": "hard failure on any duplicate inc_key (not a dedup)",
    }
    return fd_days, per_year


# ---------------------------------------------------------------------------
# Pass 5 — MAIN table: pre-decision STROBE funnel + collect candidate rows.
# ---------------------------------------------------------------------------
def scan_main(ring_set: set, candidate: set, ais6: set, sample_n=None):
    print("[5/5] Scanning MAIN table (pre-decision funnel + collect) ...",
          flush=True)
    t0 = time.time()
    # Cumulative counts c[k] = rows passing steps 1..k (7 pre-decision steps).
    c = {k: 0 for k in range(1, 8)}
    collected = []
    reader = pd.read_csv(
        MAIN_CSV, usecols=MAIN_COLS, dtype=str, chunksize=CHUNK_MAIN,
        nrows=sample_n,  # smoke mode: cap the MAIN scan at N rows
    )
    for chunk in reader:
        n = len(chunk)
        ik = chunk["inc_key"].astype(str).str.strip()

        # step 2 — adult
        age = pd.to_numeric(chunk["AGE_NUMBER"], errors="coerce")
        units = _clean_str(chunk["AGE_UNITS"]).str.upper()
        f_adult = (age >= 18) & (units == "Y")

        # step 3 — pelvic ring (S32) ; candidate = ring OR acetabulum
        in_ring = ik.isin(ring_set)
        in_cand = ik.isin(candidate)

        # step 4 — blunt mechanism (drop blank e-code; exclude penetrating)
        ec_raw = _clean_str(chunk["CAUSE_E_CODES10"]).str.upper()
        ec3 = ec_raw.str.replace(".", "", regex=False).str[:3]
        blank_ecode = ec_raw == ""
        f_blunt = (~blank_ecode) & (~ec3.isin(PENETRATING_ECODES))

        # step 5 — direct arrival
        f_direct = _clean_str(chunk["HOSPITAL_TRANSFER"]).str.upper() == "N"

        # step 6 — exclude DOA / unsurvivable (was step 7; HC gate DROPPED)
        arrest = _clean_str(chunk["PREHOSPITAL_ARREST"]).str.upper() == "Y"
        is_ais6 = ik.isin(ais6)
        f_notdoa = (~arrest) & (~is_ais6)

        # step 7 — valid outcome (disposition known or ED death)
        dc = _clean_str(chunk["DC_DISPOSITION_CODE"]).str.upper()
        ed = _clean_str(chunk["ED_DISPOSITION_CODE"]).str.upper()
        f_valid = (dc != "") | (ed == "MOR")

        # cumulative masks (each step conditioned on all prior; NO HC gate)
        m2 = f_adult
        m3 = m2 & in_ring
        m4 = m3 & f_blunt
        m5 = m4 & f_direct
        m6 = m5 & f_notdoa
        m7 = m6 & f_valid

        c[1] += n
        c[2] += int(m2.sum())
        c[3] += int(m3.sum())
        c[4] += int(m4.sum())
        c[5] += int(m5.sum())
        c[6] += int(m6.sum())
        c[7] += int(m7.sum())

        # Collect candidate (ring OR acetabulum) rows passing steps 2,4,5,6,7.
        # Ring-only step 3 is NOT applied here so the acetabulum-inclusive
        # sensitivity cohort is retained with cohort_primary=0.  f_hc is GONE.
        f_keep = f_adult & in_cand & f_blunt & f_direct & f_notdoa & f_valid
        if f_keep.any():
            collected.append(chunk[f_keep].copy())

    cohort = (pd.concat(collected, ignore_index=True)
              if collected else pd.DataFrame(columns=MAIN_COLS))
    print(f"      MAIN rows={c[1]:,}  pre-decision(step7)={c[7]:,}  "
          f"candidate-collected={len(cohort):,}  ({time.time()-t0:.0f}s)",
          flush=True)
    return cohort, c


# ---------------------------------------------------------------------------
# Derive typed cohort columns (CONTRACTS §3).
# ---------------------------------------------------------------------------
def derive_cohort(cohort, ring_set, acet_set, ais_regions, charlson,
                  unstable_ring_set, anticoag_set, bleeding_set, fd_days):
    df = cohort.copy()
    df["inc_key"] = df["inc_key"].astype(str).str.strip()
    df["puf_year"] = pd.to_numeric(df["PUF_YEAR"], errors="coerce").astype("Int64")

    # --- cohort membership ---
    df["cohort_primary"] = df["inc_key"].isin(ring_set).astype(int)
    df["has_acetabulum"] = df["inc_key"].isin(acet_set).astype(int)

    # --- HC route / exposure-context (text-decoded columns) ---
    surg_t = _clean_str(df["TQIP_HC_SURG_TYPE"]).str.upper()
    df["surg_type_raw"] = _clean_str(df["TQIP_HC_SURG_TYPE"])
    df["surg_hc"] = surg_t.isin(HC_SURG_SET).astype(int)

    angio_v = _clean_str(df["TQIP_HC_ANGIO"]).str.upper()
    df["angio_raw"] = _clean_str(df["TQIP_HC_ANGIO"])
    embo_raw = _clean_str(df["TQIP_HC_EMBO_SITE"])
    df["embo_site_raw"] = embo_raw
    df["embo_any"] = (embo_raw != "").astype(int)
    df["embo_pelvis"] = (embo_raw.str.upper() == "PELVIS").astype(int)
    df["angio_hc"] = (angio_v.isin(ANGIO_THERAPEUTIC) | (embo_raw != "")).astype(int)

    s = df["surg_hc"].astype(bool)
    a = df["angio_hc"].astype(bool)
    df["hc_route"] = np.select(
        [s & a, s & ~a, ~s & a],
        ["both", "surgical_only", "angio_only"],
        default="none",
    )
    df["surg_class"] = surg_t.map(SURG_CLASS)
    df["surg_class"] = df["surg_class"].where(df["surg_hc"] == 1, "none").fillna("none")
    df["pelvic_specific_surg"] = (
        surg_t.isin(PELVIC_SPECIFIC_SURG) & (df["surg_hc"] == 1)
    ).astype(int)

    # --- outcomes & components (CONTRACTS §3, §4; v3 Fixes 1 & 2) ---
    dc = _clean_str(df["DC_DISPOSITION_CODE"]).str.upper()
    ed = _clean_str(df["ED_DISPOSITION_CODE"]).str.upper()
    df["death_inhosp"] = ((dc == "D") | (ed == "MOR")).astype(int)
    # v3 FIX 2 — earliest-death sensitivity.  ED_DISPOSITION_CODE=='MOR' is the
    # only time-anchored death signal that survives de-identification (every
    # date/time column on this export is 0% filled), so it stands in for the
    # infeasible "death <= 48 h" restriction that was requested.
    df["death_ed"] = (ed == ED_DEATH_CODE).astype(int)
    df["blood_4h"] = pd.to_numeric(df["TQIP_HC_BLOOD_4HR"], errors="coerce")
    # v3 FIX 7 (F1) — plasma/platelet 4 h volumes for the EXPLORATORY
    # balanced-resuscitation subgroup.  Emitted raw-numeric and NaN-preserving:
    # deliberately NO decoder (the subgroup is exploratory, so the honest move
    # is to document the scale ambiguity, not to invent a second split rule).
    # Their band distributions are reported in logs/fix1_transfusion_audit.md.
    df["plasma_4h"] = pd.to_numeric(df["TQIP_HC_PLASMA_4HR"], errors="coerce")
    df["platelet_4h"] = pd.to_numeric(df["TQIP_HC_PLATE_4HR"], errors="coerce")
    df["icu_los"] = pd.to_numeric(df["TOTAL_DAYS_ICU"], errors="coerce")
    df["vent_days"] = pd.to_numeric(df["VENTDAYS"], errors="coerce")

    df["received_hc"] = ((df["surg_hc"] == 1) | (df["angio_hc"] == 1)).astype(int)

    # ---- v3 FIX 1 (S1) — unit-aware transfusion decoder --------------------
    # TQIP_HC_BLOOD_UNIT is 100% blank, so the scale is inferred from magnitude
    # and the split is placed in the empirical 101-199 gap between the two modes.
    b = df["blood_4h"]
    df["blood_4h_scale"] = np.select(
        [b.isna(),
         b == 0,
         (b > 0) & (b <= BLOOD_UNITS_MAX),
         b >= BLOOD_ML_MIN_SCALE],
        ["missing", "zero", "units", "ml"],
        default="ambiguous",          # 100 < v < 200 (and any stray negative)
    )
    # The plan states the rule as
    #     (value >= 1000 AND value >= 200) OR (value <= 100 AND value >= 4)
    # The first conjunct is REDUNDANT — `>= 1000` already implies `>= 200` — so
    # the semantics implemented here are exactly the two branches:
    #     mL branch    : blood_4h >= 1000            (>=1000 mL / 4 h)
    #     units branch : 4 <= blood_4h <= 100        (>=4 U / 4 h)
    # blood_4h NaN -> both comparisons False -> 0 (residual ~8% missing is
    # handled as a downstream sensitivity, exactly as in v2).
    _ml_branch = b >= BLOOD_ML_MIN_SUBSTANTIAL
    _units_branch = (b >= BLOOD_UNITS_MIN_SUBSTANTIAL) & (b <= BLOOD_UNITS_MAX)
    df["substantial_transfusion"] = (_ml_branch | _units_branch).astype(int)
    # The exact v2 rule, retained as a NAMED SENSITIVITY so the change is
    # auditable end-to-end.  By construction it is a subset of the v3 rule.
    df["substantial_transfusion_ml_only"] = _ml_branch.astype(int)
    # ---- v3 FIX 7 (F1) — MTP-equivalent secondary outcome ------------------
    # IDENTITY, STATED EXPLICITLY: `mtp_equivalent` is DEFINITIONALLY IDENTICAL
    # to the v3 `substantial_transfusion` above (>=4 U or >=1000 mL within 4 h —
    # the critical-administration-threshold concept, PMID 25440492).  It is
    # emitted as its own column ONLY because script 17 analyses it as a
    # standalone secondary outcome.  It is NOT an independent endpoint and must
    # never be reported as if it added information beyond the transfusion
    # component of the primary composite.
    df["mtp_equivalent"] = df["substantial_transfusion"].astype(int)

    # --- demographics ---
    df["age"] = pd.to_numeric(df["AGE_NUMBER"], errors="coerce")
    df["age_ge65"] = (df["age"] >= 65).astype(int)
    sex = _clean_str(df["SEX"]).str.upper()
    df["sex_male"] = np.select([sex == "M", sex == "F"], [1.0, 0.0], default=np.nan)

    eth = _clean_str(df["ETHNICITY"]).str.upper()
    race = _clean_str(df["RACE"]).str.upper()
    hisp = eth == "H"
    df["ethnicity_hispanic"] = hisp.astype(int)
    df["race_eth_cat"] = np.select(
        [hisp, (~hisp) & (race == "W"), (~hisp) & (race == "B"),
         (~hisp) & (race == "A")],
        ["Hispanic", "Non-Hispanic White", "NH Black", "Asian"],
        default="Other/Unknown",
    )

    pay = _clean_str(df["PAYMENT_SOURCE"]).str.upper()
    df["payment_cat"] = np.select(
        [pay == "COMM-C", pay == "MCARE-M", pay == "AGENCY-A", pay == "SELF"],
        ["Commercial", "Medicare", "Medicaid/Agency", "Self-pay"],
        default="Other/Unknown",   # PENDING-P, CVP, CHARITY, blank
    )

    # --- physiology (capture SBP2==0 sentinel BEFORE the 0->NaN mask, §3) ---
    sbp = pd.to_numeric(df["SBP2"], errors="coerce")
    pulse = pd.to_numeric(df["P2"], errors="coerce")
    df["sbp_unrecordable"] = (sbp == 0).astype(int)  # NEW: MNAR-as-signal sentinel
    df["sbp2"] = sbp.mask(sbp == 0)
    df["pulse2"] = pulse.mask(pulse == 0)
    df["shock_index"] = df["pulse2"] / df["sbp2"]
    df["sbp_lt90"] = (df["sbp2"] < 90).astype(float).mask(df["sbp2"].isna())
    df["motor2"] = pd.to_numeric(df["MOTOR2"], errors="coerce")
    df["gcs2"] = pd.to_numeric(df["GCS2"], errors="coerce")
    df["iss"] = pd.to_numeric(df["ISS"], errors="coerce")

    # --- AIS regions (absence of a region injury -> 0) ---
    df = df.merge(ais_regions, on="inc_key", how="left")
    for col in ("ais_head", "ais_chest", "ais_abd"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["ais_head_ge3"] = (df["ais_head"] >= 3).astype(int)

    # --- v3 W1: FINALDISCHARGEDAYS from the RAW PUF (arrival-anchored) ------
    # Merged HERE so the death block below can use it.  Left merge on a
    # de-duplicated right frame => row count and row order are preserved.
    _n_before_fd = len(df)
    df = df.merge(fd_days, on="inc_key", how="left")
    assert len(df) == _n_before_fd, (
        f"finaldischarge_days merge changed row count "
        f"{_n_before_fd} -> {len(df)}; the raw-PUF frame is not unique on inc_key")
    for _c in ("finaldischarge_hrs", "finaldischarge_days"):
        if _c not in df.columns:
            df[_c] = np.nan
        df[_c] = pd.to_numeric(df[_c], errors="coerce")

    # ---- v3 FIX 2 (W1, FULL COMPLIANCE) — death family + composites --------
    # Placed HERE, after the AIS and FINALDISCHARGEDAYS merges, because it needs
    # both `ais_head` (filled to 0 above => 100% defined) and the day counter.
    #
    # The request was for BOTH halves: death <= 48 h AND head AIS < 4.  Both
    # are now built.  The <=48 h half was initially believed infeasible because
    # every *_DATE/*_TIME column in the COMPILED export is 0% filled; the RAW
    # PUF's day-resolution `FINALDISCHARGEDAYS` closes that gap.  See
    # v3/logs/timing_evidence_v3.md and the DEATH_LE48H_MAX_DAYS comment above
    # for why `<= 2` days is the provably-<48 h operationalization.
    #
    # TIMING RULE: the PRIMARY cut is on `finaldischarge_hrs <= 48` — a genuine
    # elapsed-hours measure, so this is W1's "<= 48 h" computed EXACTLY.  The
    # calendar-day cut (`finaldischarge_days <= 2`) is retained only as a named
    # sensitivity: it is 100% specific but badly under-sensitive, because the
    # day field is a calendar INDEX and day 3 straddles 48 h.
    #
    # MISSING-VALUE RULE (explicit, conservative): `finaldischarge_hrs` is NaN
    # for a minority of deaths.  A NaN comparison is False in pandas, so such a
    # death does NOT enter `death_le48h` / `death_hemorrhagic` — i.e.
    # MISSING -> NOT within 48 h.  This shrinks the primary outcome rather than
    # inflating it.  The effect of the rule is BOUNDED, not assumed, by
    # `death_hemorrhagic_missing_incl` below, which flips missing to included.
    _le48 = df["finaldischarge_hrs"] <= DEATH_LE48H_MAX_HOURS      # NaN -> False
    _fd_missing = df["finaldischarge_hrs"].isna()
    _le48_day = df["finaldischarge_days"] <= DEATH_LE48H_MAX_DAYS  # NaN -> False
    _died = df["death_inhosp"] == 1
    _head_ok = df["ais_head"] < AIS_HEAD_SEVERE

    # timing half alone (sensitivity) — on hours, and on the day rule
    df["death_le48h"] = (_died & _le48).astype(int)
    df["death_le48h_dayrule"] = (_died & _le48_day).astype(int)
    # head-AIS half alone — this was the v3 primary before W1 was fully
    # complied with; now DEMOTED to a named sensitivity.
    df["death_hemorrhagic_headais"] = (_died & _head_ok).astype(int)
    # >>> NEW PRIMARY composite death component: BOTH halves of W1, exactly <<<
    df["death_hemorrhagic"] = (_died & _head_ok & _le48).astype(int)
    # Same construction on the calendar-day rule, retained so the day-vs-hour
    # crosswalk stays auditable and the earlier day-based result reproducible.
    df["death_hemorrhagic_dayrule"] = (_died & _head_ok & _le48_day).astype(int)
    # Bounding sensitivity for the missing-value rule: identical to
    # death_hemorrhagic except that a missing HOURS value is treated as
    # <=48 h rather than as >48 h.  The gap between the two columns is exactly
    # the number of deaths the missingness rule costs.  This is the UPPER bound.
    df["death_hemorrhagic_missing_incl"] = (
        _died & _head_ok & (_le48 | _fd_missing)
    ).astype(int)
    # ---- two PRE-SPECIFIED partial-recovery sensitivities ------------------
    # A minority of deaths have no `finaldischarge_hrs` but DO carry usable
    # weaker evidence of early death.  The primary column deliberately ignores
    # both (one operationalisation per column, conservative), and each recovery
    # route is emitted as its own DECLARED column so the choice is visible and
    # auditable rather than buried in a fallback branch.
    #
    # (a) DAY-FALLBACK: hours missing but the calendar-day index says <= 2.
    #     Defensible because the day rule is 100% specific among timed records
    #     (zero days<=2 & hrs>48), but it MIXES two operationalisations inside
    #     one column, which is why it is not the primary.
    df["death_hemorrhagic_dayfallback"] = (
        _died & _head_ok & (_le48 | (_fd_missing & _le48_day))
    ).astype(int)
    # (b) ED-INCLUSIVE: hours missing but the patient died in the ED.  An ED
    #     death is necessarily within 48 h, so this is a logical implication
    #     rather than an imputation.  Pre-specified HERE, before the numbers
    #     were used, precisely so it cannot read as a post-hoc rescue.
    df["death_hemorrhagic_edincl"] = (
        _died & _head_ok & (_le48 | (_fd_missing & (df["death_ed"] == 1)))
    ).astype(int)

    # PRIMARY composite, rebuilt on the v3 components (Fix 1 + Fix 2-full).
    df["major_hemorrhage_event"] = (
        (df["received_hc"] == 1)
        | (df["death_hemorrhagic"] == 1)
        | (df["substantial_transfusion"] == 1)
    ).astype(int)
    # Co-primary anti-circularity composite (HC dropped), also on v3 components.
    df["patient_state_composite"] = (
        (df["death_hemorrhagic"] == 1) | (df["substantial_transfusion"] == 1)
    ).astype(int)
    # Named sensitivity: the head-AIS-only variant of the composite (i.e. the
    # W1-partial-compliance build), so the marginal effect of adding the <=48 h
    # half is visible as a single column difference.
    df["major_hemorrhage_event_headais"] = (
        (df["received_hc"] == 1)
        | (df["death_hemorrhagic_headais"] == 1)
        | (df["substantial_transfusion"] == 1)
    ).astype(int)
    # The EXACT v2 primary, retained UNCHANGED as the fork-fidelity gate and an
    # auditable sensitivity, so v2 -> v3 movement stays decomposable.
    df["major_hemorrhage_event_v2def"] = (
        (df["received_hc"] == 1)
        | (df["death_inhosp"] == 1)
        | (df["substantial_transfusion_ml_only"] == 1)
    ).astype(int)

    # --- mechanism subtype ---
    ec3 = (_clean_str(df["CAUSE_E_CODES10"]).str.upper()
           .str.replace(".", "", regex=False).str[:3])
    mech_lookup = {v: _mechanism_from_ecode(v) for v in ec3.unique()}
    df["mechanism_cat"] = ec3.map(mech_lookup)

    # --- transport ---
    tr = _clean_str(df["TRANS"]).str.upper()
    df["transport_cat"] = np.select(
        [tr.isin(["AMB", "AUTO", "POLICE"]), tr.isin(["AIR-H", "AIR-FW"])],
        ["Ground", "Air"], default="Other",
    )

    # --- ACS verification level (keep blank as its own 'Unknown' level) ---
    vl = _clean_str(df["VERIFICATION_LEVEL"])
    df["acs_level"] = np.select([vl == "1", vl == "2", vl == "3"],
                                ["1", "2", "3"], default="Unknown")

    # --- Charlson (no preexisting record -> 0) ---
    df = df.merge(charlson, on="inc_key", how="left")
    df["charlson"] = pd.to_numeric(df["charlson"], errors="coerce").fillna(0).astype(int)

    # --- NEW flags: unstable_ring (S32.81*), cc_anticoag(31), cc_bleeding(4) ---
    df["unstable_ring"] = df["inc_key"].isin(unstable_ring_set).astype(int)
    df["cc_anticoag"] = df["inc_key"].isin(anticoag_set).astype(int)
    df["cc_bleeding"] = df["inc_key"].isin(bleeding_set).astype(int)

    # --- existing subgroup flags ; NaN comparisons -> False (not-unstable) ---
    df["unstable_flag"] = (
        (df["sbp2"] < 90) | (df["shock_index"] > 1) | (df["blood_4h"] > 0)
    ).astype(int)
    df["tbi_severe"] = (df["ais_head"] >= 4).astype(int)
    df["moribund"] = (df["sbp2"] < 70).astype(int)

    return df[COHORT_COLUMNS].copy()


# ---------------------------------------------------------------------------
# QC gates (data-driven; timing 0% filled on this export -> NOT modeled).
# ---------------------------------------------------------------------------
def compute_gates(cohort_raw, cohort):
    """cohort_raw carries the MAIN datetime columns; cohort has typed outcomes."""
    raw = cohort_raw.reset_index(drop=True)
    typ = cohort.reset_index(drop=True)

    arr = _clean_str(raw["ARRIVAL_DATE"]) != ""
    surg_dt = _clean_str(raw["TQIP_HC_SURG_DATE"]) != ""
    angio_dt = _clean_str(raw["TQIP_HC_ANGIO_DATE"]) != ""
    hc_dt = surg_dt | angio_dt
    timing_valid = arr & hc_dt                      # need arrival AND an HC datetime
    timing_valid_pct = round(100.0 * float(timing_valid.mean()), 4) if len(raw) else 0.0
    timing_gate_pass = bool(timing_valid_pct >= 70.0)   # <70% => not modeled

    dc = _clean_str(raw["DC_DISPOSITION_CODE"]).str.upper()
    ed = _clean_str(raw["ED_DISPOSITION_CODE"]).str.upper()
    disch_dt = _clean_str(raw["DISCHARGE_DATE"]) != ""
    eddc_dt = _clean_str(raw["ED_DC_DATE"]) != ""
    death_dt_derivable = ((dc == "D") & disch_dt) | ((ed == "MOR") & eddc_dt)
    deaths = typ["death_inhosp"] == 1
    n_deaths = int(deaths.sum())
    frac_death_dt = (round(100.0 * float(death_dt_derivable[deaths].mean()), 4)
                     if n_deaths else 0.0)
    mortality_24h_gate_pass = bool(frac_death_dt >= 70.0)
    # NOTE: this gate describes the COMPILED export only (all its date/time
    # columns are empty).  It does NOT name the study's outcome — the v3
    # primary outcome is set from the actual v3 definitions in main().
    primary_outcome = ("24h_in_hospital_mortality" if mortality_24h_gate_pass
                       else "in_hospital_mortality")

    return {
        "timing_gate_pass": timing_gate_pass,
        "timing_valid_pct": timing_valid_pct,
        "mortality_24h_gate_pass": mortality_24h_gate_pass,
        "primary_outcome": primary_outcome,
        "_n_deaths": n_deaths,
        "_frac_death_dt_pct": frac_death_dt,
    }


# ---------------------------------------------------------------------------
# STROBE flow (7 pre-decision steps — HC gate removed, CONTRACTS §9-A).
# ---------------------------------------------------------------------------
STEP_LABELS = {
    1: "All NTDB/TQP 2019-2024 encounters",
    2: "Adult (AGE_NUMBER >= 18 & AGE_UNITS == 'Y')",
    3: "Pelvic-ring fracture (>=1 ICD-10 S32.1/.3/.5/.6/.81/.82)",
    4: "Blunt mechanism (non-blank e-code; penetrating excluded)",
    5: "Direct arrival (HOSPITAL_TRANSFER == 'N')",
    6: "Exclude DOA / unsurvivable (PREHOSPITAL_ARREST == 'Y' or any AIS severity 6)",
    7: "Valid outcome (disposition known or ED death)",
}
N_STEPS = 7


def build_cohort_flow(c):
    """Return list of {step,label,n_in,n_out,n_dropped} (each n_in = prior n_out)."""
    flow = []
    prev = c[1]
    for k in range(1, N_STEPS + 1):
        n_out = c[k]
        n_in = c[1] if k == 1 else prev
        flow.append({"step": k, "label": STEP_LABELS[k],
                     "n_in": int(n_in), "n_out": int(n_out),
                     "n_dropped": int(n_in - n_out)})
        prev = n_out
    return flow


def write_cohort_flow_txt(path, flow, gates, n_primary, n_acet, route_counts,
                          composite_n, composite_rate, fix1, fix2):
    lines = [
        "=" * 74,
        "Pelvic Hemorrhage Arrival Decision-Support Tool (NTDB/TQP 2019-2024) v3",
        "PRE-DECISION STROBE cohort flow  |  01_build_cohort.py  |  seed=42",
        "=" * 74,
        "",
        "QC GATES (data-driven):",
        f"  timing_gate_pass          = {gates['timing_gate_pass']} "
        f"(valid interval pct = {gates['timing_valid_pct']}%)  -> timing NOT modeled",
        f"  mortality_24h_gate_pass   = {gates['mortality_24h_gate_pass']} "
        f"(deaths with derivable death datetime = {gates['_frac_death_dt_pct']}% "
        f"of {gates['_n_deaths']})",
        f"  compiled-export death anchor = {gates['primary_outcome']}",
        "     ^ describes the COMPILED EXPORT ONLY (its date/time columns are",
        "       empty). It is NOT this study's outcome. The v3 primary outcome",
        "       and mortality anchor are stated in the block below; day- and",
        "       hour-resolution timing comes from the RAW PUF.",
        "",
        "v3 PRIMARY OUTCOME        = major_hemorrhage_event",
        "     = received_hc | death_hemorrhagic | substantial_transfusion",
        f"v3 MORTALITY-ONLY ANCHOR  = death_hemorrhagic = death_inhosp & "
        f"(ais_head < {AIS_HEAD_SEVERE}) & "
        f"(finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g})",
        "     timing cut is on HOURS (exact); the calendar-day rule is a",
        "     retained SENSITIVITY and crosswalk, not the primary.",
        "",
        "PRE-DECISION STROBE FLOW (HC-inclusion filter DROPPED; HC is an outcome).",
        "The FUNNEL IS IDENTICAL TO v2 -- v3 changes only the OUTCOME definitions",
        "(Fix 1 transfusion decoder, Fix 2 death component); no exclusion step moved.",
    ]
    for st in flow:
        lines.append(f"  Step {st['step']}. {st['label']}")
        lines.append(
            f"           n_in={st['n_in']:,}  n_out={st['n_out']:,}  "
            f"dropped={st['n_dropped']:,}"
        )
    lines += [
        "",
        f"PRIMARY (ring-only) analytic cohort N  = {n_primary:,}   (v2 gate: 239,736)",
        f"ACETABULUM-INCLUSIVE cohort N          = {n_acet:,}   (v2 gate: 291,766)",
        "",
        "-" * 74,
        "v3 OUTCOME DEFINITIONS (both fixes applied; primary cohort)",
        "-" * 74,
        "PRIMARY composite  major_hemorrhage_event",
        "    = received_hc | death_hemorrhagic | substantial_transfusion",
        f"    = {composite_n:,} ({composite_rate}%)",
        f"      received_hc             = {fix2['received_hc_n']:,}",
        f"      death_hemorrhagic       = {fix2['death_hemorrhagic_n']:,}"
        f"   (death_inhosp & ais_head < {AIS_HEAD_SEVERE})",
        f"      substantial_transfusion = {fix1['n_v3_unit_aware']:,}"
        f"   (>=1000 mL/4h OR 4-100 U/4h)",
        "",
        "CO-PRIMARY  patient_state_composite (HC dropped, anti-circularity)",
        "    = death_hemorrhagic | substantial_transfusion",
        f"    = {fix2['patient_state_composite_n']:,} "
        f"({fix2['patient_state_composite_rate_pct']}%)",
        "",
        "SENSITIVITY composites (retained for audit)",
        "    major_hemorrhage_event_headais",
        "    = received_hc | death_hemorrhagic_headais | substantial_transfusion",
        f"    = {fix2['composite_headais_n']:,}",
        "    major_hemorrhage_event_v2def  (exact v2 primary)",
        "    = received_hc | death_inhosp | substantial_transfusion_ml_only",
        f"    = {fix2['composite_v2def_n']:,}",
        "",
        "FIX 1 (S1) unit-aware transfusion decoder:",
        f"    v2 mL-only (>=1000)   = {fix1['n_v2_ml_only']:,}",
        f"    v3 unit-aware         = {fix1['n_v3_unit_aware']:,}"
        f"   (delta {fix1['delta']:+,}; reclassified {fix1['n_reclassified']:,})",
        f"    ambiguous 101-199 gap = {fix1['n_ambiguous_gap']:,} "
        f"(flagged by a branch: {fix1['n_ambiguous_gap_flagged']})",
        "    -> logs/fix1_transfusion_audit.md",
        "",
        "FIX 2 (W1, FULL COMPLIANCE) hemorrhagic-death component:",
        f"    death_hemorrhagic (NEW primary)  = {fix2['death_hemorrhagic_n']:,}"
        f"   = death_inhosp & ais_head<{AIS_HEAD_SEVERE} & "
        f"finaldischarge_hrs<={DEATH_LE48H_MAX_HOURS:g}",
        "    ladder A (head-AIS first):",
        f"        {fix2['death_inhosp_n']:,} -> "
        f"{fix2['death_hemorrhagic_headais_n']:,} -> "
        f"{fix2['death_hemorrhagic_n']:,}",
        "    ladder B (timing first):",
        f"        {fix2['death_inhosp_n']:,} -> {fix2['death_le48h_n']:,} -> "
        f"{fix2['death_hemorrhagic_n']:,}",
        f"    superseded calendar-day rule -> "
        f"{fix2['death_hemorrhagic_dayrule_n']:,}  (day rule missed "
        f"{fix2['dayrule_missed_true_le48h_deaths']:,} true <=48h deaths)",
        "    sensitivities:",
        f"        death_hemorrhagic_headais      = {fix2['death_hemorrhagic_headais_n']:,}",
        f"        death_le48h                    = {fix2['death_le48h_n']:,}",
        f"        death_hemorrhagic_dayrule      = {fix2['death_hemorrhagic_dayrule_n']:,}",
        f"        death_le48h_dayrule            = {fix2['death_le48h_dayrule_n']:,}",
        f"        death_hemorrhagic_missing_incl = {fix2['death_hemorrhagic_missing_incl_n']:,}"
        f"  (missingness bound; rule costs {fix2['deaths_lost_to_missingness_rule']:,})",
        f"        death_inhosp (all-cause)       = {fix2['death_inhosp_n']:,}",
        f"        death_ed (earliest-death)      = {fix2['death_ed_n']:,}",
        f"    finaldischarge_hrs fill: cohort "
        f"{fix2['finaldischarge_hrs_fill_pct']}%, deaths "
        f"{fix2['finaldischarge_hrs_fill_pct_deaths']}%  "
        f"(RAW PUF_TRAUMA.csv, arrival-anchored, used directly)",
        f"    ascertainment: deaths missing timing derivation="
        f"{fix2['ascertainment_by_split']['derivation_2019_2022']['deaths_missing_pct']}%"
        f" vs validation="
        f"{fix2['ascertainment_by_split']['validation_2023_2024']['deaths_missing_pct']}%"
        f"  [{fix2['ascertainment_by_split']['differential_pp']:+.3f} pp -> 12]",
        "    -> logs/fix2_death_audit.md  |  evidence: logs/timing_evidence_v3.md",
        "",
        "HC-route counts (primary cohort; now an OUTCOME):",
        f"  surgical_only = {route_counts.get('surgical_only', 0):,}",
        f"  angio_only    = {route_counts.get('angio_only', 0):,}",
        f"  both          = {route_counts.get('both', 0):,}",
        f"  none          = {route_counts.get('none', 0):,}",
        "=" * 74,
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_unknown_hc_codes(path, primary_df):
    """HC-type adjudication worksheet: every observed surgical/angio/embo value."""
    rows = []
    surg_class_for = dict(SURG_CLASS)
    for val, cnt in primary_df["surg_type_raw"].value_counts().items():
        if val == "":
            continue
        rows.append(("surgical", val, int(cnt),
                     surg_class_for.get(str(val).upper(), "unrecognized")))
    for val, cnt in primary_df["angio_raw"].value_counts().items():
        if val == "":
            continue
        v = str(val).upper()
        cls = ("therapeutic" if v in ANGIO_THERAPEUTIC
               else "diagnostic_only" if v == "ANGIO" else "unrecognized")
        rows.append(("angio", val, int(cnt), cls))
    for val, cnt in primary_df["embo_site_raw"].value_counts().items():
        if val == "":
            continue
        rows.append(("embo", val, int(cnt), "therapeutic"))
    out = pd.DataFrame(rows, columns=["kind", "value", "count", "assigned_class"])
    out.to_csv(path, index=False)


def write_missingness(path, primary_df):
    core = ["age", "sex_male", "race_eth_cat", "payment_cat", "sbp2", "pulse2",
            "shock_index", "sbp_lt90", "sbp_unrecordable", "motor2", "gcs2",
            "iss", "ais_head", "mechanism_cat", "transport_cat", "acs_level",
            "charlson", "death_inhosp", "death_hemorrhagic",
            "death_hemorrhagic_headais", "death_hemorrhagic_missing_incl",
            "death_le48h", "death_le48h_dayrule", "death_hemorrhagic_dayrule",
            "death_hemorrhagic_dayfallback", "death_hemorrhagic_edincl",
            "death_ed", "finaldischarge_hrs", "finaldischarge_days",
            "blood_4h", "plasma_4h", "platelet_4h", "icu_los", "vent_days",
            "received_hc", "substantial_transfusion",
            "substantial_transfusion_ml_only", "mtp_equivalent",
            "major_hemorrhage_event", "major_hemorrhage_event_headais",
            "major_hemorrhage_event_v2def",
            "patient_state_composite", "cc_anticoag", "cc_bleeding",
            "unstable_ring"]
    n = len(primary_df)
    rows = []
    for v in core:
        s = primary_df[v]
        n_missing = int(s.isna().sum())
        rows.append((v, n_missing, round(100.0 * n_missing / n, 3) if n else 0.0))
    pd.DataFrame(rows, columns=["variable", "n_missing", "pct_missing"]).to_csv(
        path, index=False)
    return rows


def write_informative_missingness(path, primary_df):
    """Shared MNAR table (CONTRACTS §10 / SAP §8): composite rate, mortality
    rate, and N in vitals-MISSING vs vitals-OBSERVED strata, by puf_year and
    overall.  Missing = sbp2 is NaN OR sbp_unrecordable==1 (the latter is a
    subset of the former, written as the explicit union for fidelity)."""
    df = primary_df.copy()
    missing = df["sbp2"].isna() | (df["sbp_unrecordable"] == 1)
    df = df.assign(vitals_stratum=np.where(missing, "MISSING", "OBSERVED"))

    def _agg(sub):
        n = len(sub)
        return {
            "n": int(n),
            "composite_rate_pct": round(100.0 * float(sub["major_hemorrhage_event"].mean()), 3) if n else np.nan,
            "mortality_rate_pct": round(100.0 * float(sub["death_inhosp"].mean()), 3) if n else np.nan,
        }

    rows = []
    years = sorted([int(y) for y in df["puf_year"].dropna().unique()])
    for yr in years:
        sub_y = df[df["puf_year"] == yr]
        for stratum in ("MISSING", "OBSERVED"):
            a = _agg(sub_y[sub_y["vitals_stratum"] == stratum])
            rows.append({"puf_year": str(yr), "vitals_stratum": stratum, **a})
    for stratum in ("MISSING", "OBSERVED"):
        a = _agg(df[df["vitals_stratum"] == stratum])
        rows.append({"puf_year": "Overall", "vitals_stratum": stratum, **a})

    out = pd.DataFrame(rows, columns=["puf_year", "vitals_stratum", "n",
                                      "composite_rate_pct", "mortality_rate_pct"])
    out.to_csv(path, index=False)
    # return the overall MISSING-stratum mortality for the verification print
    ov_missing = out[(out["puf_year"] == "Overall") & (out["vitals_stratum"] == "MISSING")]
    miss_mort = float(ov_missing["mortality_rate_pct"].iloc[0]) if len(ov_missing) else np.nan
    return out, miss_mort


# ---------------------------------------------------------------------------
# v3 FIX 1 verification — logs/fix1_transfusion_audit.md
# ---------------------------------------------------------------------------
# Bands used throughout the audit.  `hi=None` means open-ended.
BLOOD_BANDS = [
    ("1-50", 1, 50), ("51-100", 51, 100), ("101-199", 101, 199),
    ("200-249", 200, 249), (">=250", 250, None),
]
# Finer split inside 1-50, because 1-3 U sits BELOW the >=4 U substantial
# threshold and therefore does NOT reclassify (this is the whole reason the
# reclassified count is smaller than the plan's headline 6,592).
BLOOD_BANDS_FINE = [
    ("1-3", 1, 3), ("4-50", 4, 50), ("51-100", 51, 100), ("101-199", 101, 199),
    ("200-249", 200, 249), ("250-999", 250, 999), (">=1000", 1000, None),
]


def _band_counts(nz: pd.Series, bands) -> "OrderedDict[str, int]":
    out = OrderedDict()
    for label, lo, hi in bands:
        m = (nz >= lo) if hi is None else ((nz >= lo) & (nz <= hi))
        out[label] = int(m.sum())
    return out


def write_fix1_audit(path, primary_df, full_df):
    """Verification artifact for Fix 1 (unit-aware transfusion decoder).

    Hard-asserts the four properties the plan requires:
      1. reclassified count == count of records in the 4-100 units band
      2. no record in the 101-199 gap is flagged by either branch
      3. substantial_transfusion_ml_only <= substantial_transfusion elementwise
      4. blood_4h_scale partitions the cohort exactly (no leftovers)
    """
    n = len(primary_df)
    b = primary_df["blood_4h"]
    new = primary_df["substantial_transfusion"].astype(bool)
    old = primary_df["substantial_transfusion_ml_only"].astype(bool)

    n_missing = int(b.isna().sum())
    n_zero = int((b == 0).sum())
    nz = b[b > 0]
    n_nonzero = int(len(nz))
    blood_min = float(b.min()) if b.notna().any() else np.nan
    blood_max = float(b.max()) if b.notna().any() else np.nan

    bands = _band_counts(nz, BLOOD_BANDS)
    bands_fine = _band_counts(nz, BLOOD_BANDS_FINE)
    bands_full = _band_counts(full_df["blood_4h"][full_df["blood_4h"] > 0], BLOOD_BANDS)

    n_v2 = int(old.sum())
    n_v3 = int(new.sum())
    reclassified = int((new & ~old).sum())
    n_units_band_substantial = int(((b >= BLOOD_UNITS_MIN_SUBSTANTIAL)
                                    & (b <= BLOOD_UNITS_MAX)).sum())
    n_units_band_below = int(((b > 0) & (b < BLOOD_UNITS_MIN_SUBSTANTIAL)).sum())
    n_gap = int(((b > BLOOD_UNITS_MAX) & (b < BLOOD_ML_MIN_SCALE)).sum())
    n_gap_flagged = int((((b > BLOOD_UNITS_MAX) & (b < BLOOD_ML_MIN_SCALE)) & new).sum())

    # ---- hard assertions (CONTRACTS §13 G1a-G1e) -------------------------
    assert reclassified == n_units_band_substantial, (
        f"Fix 1: reclassified={reclassified} != 4-100 units band="
        f"{n_units_band_substantial}")
    # G1b — nothing may be classified substantial through the empirical gap
    assert n_gap_flagged == 0, (
        f"Fix 1: {n_gap_flagged} record(s) in the {BLOOD_UNITS_MAX+1}-"
        f"{BLOOD_ML_MIN_SCALE-1} ambiguous gap were flagged by a branch")
    # G1c — strict superset: zero newly NEGATIVE
    newly_negative = int((old & ~new).sum())
    assert newly_negative == 0, (
        f"Fix 1: {newly_negative} record(s) flagged by the v2 mL-only rule are "
        "NOT flagged by the v3 rule; the new rule must be a strict superset")
    assert bool((old <= new).all()), (
        "Fix 1: substantial_transfusion_ml_only is NOT a subset of "
        "substantial_transfusion")
    # G1d — blood_4h_scale partitions the cohort with zero NaN
    assert int(primary_df["blood_4h_scale"].isna().sum()) == 0, (
        "Fix 1: blood_4h_scale contains NaN")
    scale_counts = primary_df["blood_4h_scale"].value_counts().to_dict()
    assert int(sum(scale_counts.values())) == n, "Fix 1: blood_4h_scale lost rows"
    # G1e — mtp_equivalent is element-wise identical to substantial_transfusion
    n_mtp_diff = int((primary_df["mtp_equivalent"]
                      != primary_df["substantial_transfusion"]).sum())
    assert n_mtp_diff == 0, (
        f"Fix 1: mtp_equivalent differs from substantial_transfusion on "
        f"{n_mtp_diff} row(s); they are definitionally identical")

    # ---- Fix 7 companion fields: plasma / platelet 4 h volumes -----------
    # Same unit-ambiguity caution as blood_4h, but DELIBERATELY NO DECODER:
    # the balanced-resuscitation subgroup is exploratory, so these fields are
    # documented, not recoded.
    comp_stats = {}
    for col, src in (("plasma_4h", "TQIP_HC_PLASMA_4HR"),
                     ("platelet_4h", "TQIP_HC_PLATE_4HR")):
        s = primary_df[col]
        s_nz = s[s > 0]
        cb = _band_counts(s_nz, BLOOD_BANDS_FINE)
        n_lo = int((s_nz <= BLOOD_UNITS_MAX).sum())          # units-like cluster
        n_hi = int((s_nz >= BLOOD_ML_MIN_SCALE).sum())       # mL-like cluster
        n_mid = int(((s_nz > BLOOD_UNITS_MAX)
                     & (s_nz < BLOOD_ML_MIN_SCALE)).sum())   # gap
        nnz = int(len(s_nz))
        # Data-driven verdict: call it bimodal only when BOTH clusters hold a
        # non-trivial share (>=5% of non-zero) and the gap between them is thin.
        share_lo = (n_lo / nnz) if nnz else 0.0
        share_hi = (n_hi / nnz) if nnz else 0.0
        bimodal = bool(nnz and share_lo >= 0.05 and share_hi >= 0.05)
        comp_stats[col] = {
            "source_column": src,
            "fill_pct": round(100.0 * float(s.notna().mean()), 3) if n else 0.0,
            "n_missing": int(s.isna().sum()),
            "n_zero": int((s == 0).sum()),
            "n_nonzero": nnz,
            "min": (float(s.min()) if s.notna().any() else None),
            "max": (float(s.max()) if s.notna().any() else None),
            "bands": dict(cb),
            "n_units_like_le100": n_lo, "n_ml_like_ge200": n_hi,
            "n_gap_101_199": n_mid,
            "pct_units_like_of_nonzero": round(100.0 * share_lo, 2),
            "pct_ml_like_of_nonzero": round(100.0 * share_hi, 2),
            "bimodal_units_vs_ml": bimodal,
            "decoder_applied": False,
            "top_values": {f"{v:g}": int(c)
                           for v, c in s_nz.value_counts().head(10).items()},
        }

    top_values = nz.value_counts().head(20)
    mode_units = nz[nz <= BLOOD_UNITS_MAX].value_counts().head(6)
    mode_ml = nz[nz >= BLOOD_ML_MIN_SCALE].value_counts().head(6)

    def _pct(x):
        return round(100.0 * x / n, 3) if n else 0.0

    L = []
    A = L.append
    A("# Fix 1 verification — unit-aware transfusion decoder (S1)")
    A("")
    A("Generated by `v3/code/01_build_cohort.py`. Analytic cohort = "
      f"`cohort_primary == 1`, **N = {n:,}** "
      f"(full CSV incl. acetabulum rows: {len(full_df):,}).")
    A("")
    A("## 1. Why the decoder is needed")
    A("")
    A("`TQIP_HC_BLOOD_4HR` is documented as millilitres, but the decoder column "
      "`TQIP_HC_BLOOD_UNIT` is **100% blank** and the values are bimodal: a "
      "small-integer cluster that can only be UNITS and a large cluster that can "
      "only be mL. Under the v2 rule (`blood_4h >= 1000`) a patient given 36 "
      "units in 4 h scored as *no substantial transfusion*.")
    A("")
    A(f"- `blood_4h` fill: **{round(100.0 * b.notna().mean(), 3)}%** "
      f"(missing n = {n_missing:,})")
    A(f"- exact zeros: **{n_zero:,}** ({_pct(n_zero)}%)")
    A(f"- non-zero: **{n_nonzero:,}** ({_pct(n_nonzero)}%)")
    A(f"- observed range: {blood_min:g} to {blood_max:g} (no negative values "
      "-> nothing falls through the scale classifier)")
    A("")
    A("## 2. Non-zero `blood_4h` by band")
    A("")
    A("| Band | n (analytic, N=%s) | n (full CSV, N=%s) | Reading |"
      % (f"{n:,}", f"{len(full_df):,}"))
    A("|---|---|---|---|")
    reading = {"1-50": "units", "51-100": "units (sparse tail)",
               "101-199": "**ambiguous gap**", "200-249": "mL (sparse head)",
               ">=250": "mL"}
    for label, _lo, _hi in BLOOD_BANDS:
        A(f"| {label} | {bands[label]:,} | {bands_full[label]:,} | "
          f"{reading[label]} |")
    A("")
    A("**Reconciliation with the plan.** The plan's headline distribution "
      "(1-50 -> 6,592; 51-200 -> 69; >=250 -> 32,136) was computed on the "
      "**full CSV**, i.e. *before* the `cohort_primary == 1` ring-only filter. "
      "On this run the full-CSV figures are: 1-50 = "
      f"**{bands_full['1-50']:,}** (plan: 6,592), 51-199 = "
      f"**{bands_full['51-100'] + bands_full['101-199']:,}** (plan quotes "
      "51-200 -> 69), >=250 = "
      f"**{bands_full['>=250']:,}** (plan: 32,136 — the plan's figure is a "
      ">=200 count, which also picks up the sparse 200-249 head). The "
      "analytic-cohort column is what the analysis actually uses; every "
      "downstream number in this audit is analytic-cohort.")
    A("")
    A("### Fine bands (analytic cohort) — why 1-3 matters")
    A("")
    A("| Band | n | Flagged by v3 rule? |")
    A("|---|---|---|")
    flagged_by = {"1-3": "no (below the >=4 U threshold)", "4-50": "**yes** (units branch)",
                  "51-100": "**yes** (units branch)", "101-199": "no (ambiguous gap)",
                  "200-249": "no (mL, below 1000)", "250-999": "no (mL, below 1000)",
                  ">=1000": "**yes** (mL branch)"}
    for label, _lo, _hi in BLOOD_BANDS_FINE:
        A(f"| {label} | {bands_fine[label]:,} | {flagged_by[label]} |")
    A("")
    A("Modal non-zero values (top 20): " +
      ", ".join(f"{v:g}({c:,})" for v, c in top_values.items()))
    A("")
    A(f"- units-cluster modes (<= {BLOOD_UNITS_MAX}): " +
      ", ".join(f"{v:g} (n={c:,})" for v, c in mode_units.items()))
    A(f"- mL-cluster modes (>= {BLOOD_ML_MIN_SCALE}): " +
      ", ".join(f"{v:g} (n={c:,})" for v, c in mode_ml.items()))
    A("")
    A("## 3. The implemented rule")
    A("")
    A("```")
    A("substantial_transfusion = (blood_4h >= 1000)                 # mL branch")
    A("                        | ((blood_4h >= 4) & (blood_4h <= 100))  # units branch")
    A("```")
    A("")
    A("The plan writes the mL branch as `(value >= 1000 AND value >= 200)`. The "
      "second conjunct is **redundant** — `>= 1000` already implies `>= 200` — "
      "so only the semantics are implemented, not the literal expression. This "
      "is stated in a code comment at the definition site.")
    A("")
    A("`blood_4h_scale` (retained column) partitions the cohort:")
    A("")
    A("| scale | n | % of cohort |")
    A("|---|---|---|")
    for k in ("units", "ml", "ambiguous", "zero", "missing"):
        v = int(scale_counts.get(k, 0))
        A(f"| {k} | {v:,} | {_pct(v)}% |")
    A("")
    A("## 4. Assertions (CONTRACTS §13 G1 — all PASS; the script raises otherwise)")
    A("")
    A(f"- **G1a — band table.** 1-50 band = **{bands['1-50']:,}** (analytic) / "
      f"**{bands_full['1-50']:,}** (full CSV). The contract's carried-forward "
      f"gate value is **6,592**, which reproduces exactly on the **full CSV**; "
      f"see the reconciliation note in §2 above. Reported, not silently passed.")
    A(f"- **G1b — no flag through the gap.** Records with "
      f"{BLOOD_UNITS_MAX} < `blood_4h` < {BLOOD_ML_MIN_SCALE} = "
      f"**{n_gap:,}**; of those flagged by either branch = "
      f"**{n_gap_flagged}**. PASS")
    A(f"- **G1c — strict superset.** Newly positive "
      f"(`substantial_transfusion==1 & substantial_transfusion_ml_only==0`) = "
      f"**{reclassified:,}**, which equals the 4-100 band "
      f"(**{n_units_band_substantial:,}**). Newly negative = "
      f"**{newly_negative}**. PASS")
    A(f"- **G1d — scale partition.** `blood_4h_scale` has zero NaN and its "
      f"categories sum to {int(sum(scale_counts.values())):,} = N. PASS")
    A(f"- **G1e — MTP identity.** `mtp_equivalent != substantial_transfusion` "
      f"on **{n_mtp_diff}** rows. PASS")
    A("")
    A("### Expected-vs-observed on the plan's 6,592")
    A("")
    A(f"The plan expects the reclassified count to match the **6,592** records "
      f"in the 1-50 band. It does **not**, and should not, for two compounding "
      f"reasons, both quantified here:")
    A("")
    A(f"- **Values 1-3 are below the >= 4 U substantial threshold.** In the "
      f"analytic cohort the 1-50 band splits "
      f"**{bands_fine['1-3']:,} (1-3 U, NOT flagged)** / "
      f"**{bands_fine['4-50']:,} (4-50 U, flagged)**. So the reclassified "
      f"count is necessarily SMALLER than the 1-50 band count.")
    A(f"- **The 6,592 is a full-CSV figure.** Restricting to `cohort_primary "
      f"== 1` drops the acetabulum-only rows, taking the 1-50 band from "
      f"{bands_full['1-50']:,} to {bands['1-50']:,}.")
    A("")
    A(f"Reclassified = 4-50 band ({bands_fine['4-50']:,}) + 51-100 band "
      f"({bands_fine['51-100']:,}) = **{reclassified:,}**.")
    A("")
    A("## 5. Old vs new — explicit delta")
    A("")
    A("| Rule | Column | n | % of cohort |")
    A("|---|---|---|---|")
    A(f"| v2 (mL only, `>= 1000`) | `substantial_transfusion_ml_only` | "
      f"{n_v2:,} | {_pct(n_v2)}% |")
    A(f"| v3 (unit-aware) | `substantial_transfusion` | {n_v3:,} | "
      f"{_pct(n_v3)}% |")
    A(f"| **delta** | — | **+{n_v3 - n_v2:,}** | "
      f"**+{round(_pct(n_v3) - _pct(n_v2), 3)} pp** |")
    A("")
    A(f"`mtp_equivalent` = **{int(primary_df['mtp_equivalent'].sum()):,}** — "
      "definitionally identical to the v3 `substantial_transfusion`, emitted "
      "as its own column for script 17 only. It is **not** an independent "
      "endpoint.")
    A("")
    A("## 6. Fix 7 companion fields — `plasma_4h` / `platelet_4h`")
    A("")
    A("`TQIP_HC_PLASMA_4HR` and `TQIP_HC_PLATE_4HR` are pulled into the cohort "
      "CSV as `plasma_4h` / `platelet_4h` for the **exploratory** "
      "balanced-resuscitation subgroup (Fix 7). **No decoder is applied to "
      "them** — they are emitted raw-numeric and NaN-preserving. The same "
      "unit-ambiguity caution that motivated Fix 1 applies, so the "
      "distributions are documented here rather than recoded.")
    A("")
    A("| Field | source column | fill % | n zero | n non-zero | min | max |")
    A("|---|---|---|---|---|---|---|")
    for col in ("plasma_4h", "platelet_4h"):
        cs = comp_stats[col]
        A(f"| `{col}` | `{cs['source_column']}` | **{cs['fill_pct']}%** | "
          f"{cs['n_zero']:,} | {cs['n_nonzero']:,} | "
          f"{cs['min'] if cs['min'] is None else format(cs['min'], 'g')} | "
          f"{cs['max'] if cs['max'] is None else format(cs['max'], 'g')} |")
    A("")
    A("### Non-zero values by band")
    A("")
    A("| Band | `plasma_4h` | `platelet_4h` |")
    A("|---|---|---|")
    for label, _lo, _hi in BLOOD_BANDS_FINE:
        A(f"| {label} | {comp_stats['plasma_4h']['bands'][label]:,} | "
          f"{comp_stats['platelet_4h']['bands'][label]:,} |")
    A("")
    A("### Is a units-vs-mL bimodality present in these two fields?")
    A("")
    A("Verdict is data-driven: a field is called bimodal only when BOTH the "
      "units-like cluster (<= 100) and the mL-like cluster (>= 200) hold at "
      "least 5% of that field's non-zero values.")
    A("")
    A("| Field | units-like (<=100) | mL-like (>=200) | 101-199 gap | verdict |")
    A("|---|---|---|---|---|")
    for col in ("plasma_4h", "platelet_4h"):
        cs = comp_stats[col]
        A(f"| `{col}` | {cs['n_units_like_le100']:,} "
          f"({cs['pct_units_like_of_nonzero']}%) | "
          f"{cs['n_ml_like_ge200']:,} ({cs['pct_ml_like_of_nonzero']}%) | "
          f"{cs['n_gap_101_199']:,} | "
          f"**{'BIMODAL — same units/mL ambiguity as blood_4h' if cs['bimodal_units_vs_ml'] else 'NOT bimodal on this criterion'}** |")
    A("")
    for col in ("plasma_4h", "platelet_4h"):
        cs = comp_stats[col]
        A(f"- `{col}` top non-zero values: " +
          ", ".join(f"{v}({c:,})" for v, c in cs["top_values"].items()))
    A("")
    A("**The plan's \"27.3% filled\" figure is SUPERSEDED and must not be "
      "quoted.** It is a sampling artifact: it was read off the first 1.5M rows "
      "of a **year-ordered** file, which is ~73% admission-year 2019 — the one "
      "year in which these fields were essentially not collected (3.5%). "
      "Pool-wide fill is **77.55%**, and ~90% in 2020-2024 "
      "(`v3/logs/timing_evidence_v3.md` §6). In this pelvic cohort the observed "
      f"fill is **{comp_stats['plasma_4h']['fill_pct']}%** / "
      f"**{comp_stats['platelet_4h']['fill_pct']}%**, consistent with that "
      "correction.")
    A("")
    A("**Consequence for Fix 7.** The balanced-resuscitation subgroup is far "
      "better supported than the plan assumed — restricted to 2020-2024 it is "
      "not thin at all. What remains genuinely exploratory is the **scale**: "
      "any ratio computed from these fields inherits an undetermined scale on "
      "both numerator and denominator, so no ratio may be interpreted as a "
      "literal unit-for-unit 1:1:1 comparison unless the scales are first shown "
      "to agree. Quote 77.5% (pool-wide) or ~90% (2020-2024) — never 27.3%.")
    A("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    return {
        "blood_4h_fill_pct": round(100.0 * float(b.notna().mean()), 3) if n else 0.0,
        "n_zero": n_zero, "n_missing": n_missing, "n_nonzero": n_nonzero,
        "bands_analytic": dict(bands), "bands_fine_analytic": dict(bands_fine),
        "bands_full_csv": dict(bands_full),
        "scale_counts": {k: int(scale_counts.get(k, 0))
                         for k in ("units", "ml", "ambiguous", "zero", "missing")},
        "n_v2_ml_only": n_v2, "n_v3_unit_aware": n_v3,
        "delta": n_v3 - n_v2, "n_reclassified": reclassified,
        "n_newly_negative": newly_negative,
        "n_units_1_3_not_flagged": n_units_band_below,
        "n_ambiguous_gap": n_gap, "n_ambiguous_gap_flagged": n_gap_flagged,
        "n_mtp_vs_substantial_mismatches": n_mtp_diff,
        "fix7_plasma_platelet": comp_stats,
        "assertions_passed": True,
    }


# ---------------------------------------------------------------------------
# v3 FIX 2 verification — logs/fix2_death_audit.md
# ---------------------------------------------------------------------------
def write_fix2_audit(path, primary_df, ed_raw_primary, fix1, fd_per_year):
    """Verification artifact for Fix 2 (hemorrhagic-death component, W1 FULL).

    Hard-asserts the nesting of the whole death family and reports: the full
    restriction ladder in BOTH orderings, the head-AIS distribution among the
    deaths each restriction removes, the `finaldischarge_days` distribution and
    per-year fill, the exact cost of the missing-value rule, and the death_ed
    cross-tab.
    """
    n = len(primary_df)
    d_all = primary_df["death_inhosp"].astype(bool)
    d_hem = primary_df["death_hemorrhagic"].astype(bool)
    d_head = primary_df["death_hemorrhagic_headais"].astype(bool)
    d_le48 = primary_df["death_le48h"].astype(bool)
    d_incl = primary_df["death_hemorrhagic_missing_incl"].astype(bool)
    d_hem_day = primary_df["death_hemorrhagic_dayrule"].astype(bool)
    d_le48_day = primary_df["death_le48h_dayrule"].astype(bool)
    d_ed = primary_df["death_ed"].astype(bool)
    ais = primary_df["ais_head"]
    fd = primary_df["finaldischarge_hrs"]        # PRIMARY timing field (hours)
    fdd = primary_df["finaldischarge_days"]      # calendar-day index (sens.)

    # ---- hard assertions (CONTRACTS §13 G2a/G2c + the v3 death family) ----
    # G2a — every restricted death variant is a subset of all-cause death
    n_hem_not_all = int((d_hem & ~d_all).sum())
    assert n_hem_not_all == 0, (
        f"Fix 2: {n_hem_not_all} row(s) have death_hemorrhagic without "
        "death_inhosp; the former must be a subset of the latter")
    for name, col in (("death_hemorrhagic_headais", d_head),
                      ("death_le48h", d_le48),
                      ("death_le48h_dayrule", d_le48_day),
                      ("death_hemorrhagic_dayrule", d_hem_day),
                      ("death_hemorrhagic_missing_incl", d_incl),
                      ("death_ed", d_ed)):
        bad = int((col & ~d_all).sum())
        assert bad == 0, f"Fix 2: {bad} row(s) have {name} without death_inhosp"
    # the primary is the INTERSECTION of the two halves -> subset of each
    assert int((d_hem & ~d_head).sum()) == 0, (
        "Fix 2: death_hemorrhagic must be a subset of death_hemorrhagic_headais")
    assert int((d_hem & ~d_le48).sum()) == 0, (
        "Fix 2: death_hemorrhagic must be a subset of death_le48h")
    # the missing-inclusive bound must be a superset of the primary
    assert int((d_hem & ~d_incl).sum()) == 0, (
        "Fix 2: death_hemorrhagic must be a subset of "
        "death_hemorrhagic_missing_incl")
    # G2c — patient_state_composite is a subset of major_hemorrhage_event
    n_state_not_comp = int(((primary_df["patient_state_composite"] == 1)
                            & (primary_df["major_hemorrhage_event"] == 0)).sum())
    assert n_state_not_comp == 0, (
        f"Fix 2: {n_state_not_comp} row(s) have patient_state_composite "
        "without major_hemorrhage_event")

    # The day rule is 100% SPECIFIC **among records with an observed hours
    # value**: no death with days<=2 has hrs>48.  It is NOT a subset overall,
    # because a death can satisfy days<=2 while its hours value is missing —
    # those are the day-fallback cases, counted separately below.  Asserting the
    # unconditional subset was wrong and is corrected here.
    n_day_hr_conflict = int((d_le48_day & fd.notna() & ~d_le48).sum())
    assert n_day_hr_conflict == 0, (
        f"Fix 2: {n_day_hr_conflict} death(s) have finaldischarge_days <= "
        f"{DEATH_LE48H_MAX_DAYS} but finaldischarge_hrs > "
        f"{DEATH_LE48H_MAX_HOURS:g}; the day rule must be 100% specific among "
        "timed records")
    n_day_only_untimed = int((d_le48_day & fd.isna()).sum())

    n_all = int(d_all.sum())
    n_hem = int(d_hem.sum())
    n_head = int(d_head.sum())
    n_le48 = int(d_le48.sum())
    n_le48_day = int(d_le48_day.sum())
    n_hem_day = int(d_hem_day.sum())
    n_incl = int(d_incl.sum())
    n_ed = int(d_ed.sum())
    d_hem_dayfb = primary_df["death_hemorrhagic_dayfallback"].astype(bool)
    d_hem_edincl = primary_df["death_hemorrhagic_edincl"].astype(bool)
    n_hem_dayfb = int(d_hem_dayfb.sum())
    n_hem_edincl = int(d_hem_edincl.sum())
    # every recovery variant must be a superset of the primary and a subset of
    # the all-missing-included upper bound
    for nm, col in (("death_hemorrhagic_dayfallback", d_hem_dayfb),
                    ("death_hemorrhagic_edincl", d_hem_edincl)):
        assert int((d_hem & ~col).sum()) == 0, (
            f"Fix 2: {nm} must be a superset of death_hemorrhagic")
        assert int((col & ~d_incl).sum()) == 0, (
            f"Fix 2: {nm} must be a subset of death_hemorrhagic_missing_incl")
    # day-rule sensitivity vs the exact hours rule, among timed records
    n_dayrule_true_pos = int((d_le48_day & d_le48).sum())
    n_day3_within_48h = int((d_all & (fdd == 3) & (fd <= DEATH_LE48H_MAX_HOURS)).sum())
    dayrule_sens_pct = (round(100.0 * n_dayrule_true_pos / n_le48, 1)
                        if n_le48 else 0.0)
    n_deaths_untimed_ed = int((d_all & fd.isna() & d_ed).sum())
    n_deaths_missing_both = int((d_all & fd.isna() & fdd.isna()).sum())

    removed = d_all & ~d_hem
    n_removed = int(removed.sum())
    assert n_removed == n_all - n_hem

    # deaths removed by EACH restriction, isolated
    removed_head = d_all & ~d_head                       # head AIS >= 4
    removed_time = d_all & ~d_le48                       # >2 days or missing
    # among head-AIS-eligible deaths, those the <=2-day rule removes
    removed_time_within_head = d_head & ~d_le48
    n_removed_time_within_head = int(removed_time_within_head.sum())

    # missing-value rule cost: deaths that pass head-AIS but have NO day value
    fd_missing = fd.isna()
    n_deaths_fd_missing = int((d_all & fd_missing).sum())
    n_lost_to_missingness = int(n_incl - n_hem)
    assert n_lost_to_missingness == int((d_all & d_head & fd_missing).sum()), (
        "Fix 2: missing-inclusive bound does not equal the head-AIS-eligible "
        "deaths with a missing day counter")

    ais_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def _ais_dist(mask):
        return {int(k): int((ais[mask] == k).sum()) for k in ais_levels}

    dist_removed = _ais_dist(removed)
    dist_retained = _ais_dist(d_hem)
    dist_alldeath = _ais_dist(d_all)
    dist_removed_time = _ais_dist(removed_time_within_head)

    ed_fill_pct = (round(100.0 * float((ed_raw_primary != "").mean()), 3)
                   if len(ed_raw_primary) else 0.0)
    ais_fill_pct = round(100.0 * float(ais.notna().mean()), 3) if n else 0.0
    fd_fill_pct = round(100.0 * float(fd.notna().mean()), 3) if n else 0.0
    fd_fill_deaths_pct = (round(100.0 * float(fd[d_all].notna().mean()), 3)
                          if n_all else 0.0)

    # finaldischarge_hrs distribution among deaths (PRIMARY timing field)
    fd_deaths = fd[d_all].dropna()
    hr_bins = [(0, 24), (24, 48), (48, 72), (72, 168), (168, None)]
    hr_hist = OrderedDict()
    for lo, hi in hr_bins:
        lab = f"{lo}-{hi}h" if hi is not None else f">{lo}h"
        m = (fd_deaths > lo) | (fd_deaths == 0) if lo == 0 else (fd_deaths > lo)
        m = m & ((fd_deaths <= hi) if hi is not None else True)
        hr_hist[lab] = int(m.sum())
    fd_stats = {
        "n_with_value": int(len(fd_deaths)),
        "min": (float(fd_deaths.min()) if len(fd_deaths) else None),
        "max": (float(fd_deaths.max()) if len(fd_deaths) else None),
        "median": (float(fd_deaths.median()) if len(fd_deaths) else None),
        "q1": (float(fd_deaths.quantile(0.25)) if len(fd_deaths) else None),
        "q3": (float(fd_deaths.quantile(0.75)) if len(fd_deaths) else None),
        "n_negative": int((fd_deaths < 0).sum()),
    }
    fd_ed = fd[d_ed].dropna()
    fd_ed_median = float(fd_ed.median()) if len(fd_ed) else None
    fd_surv = fd[~d_all].dropna()
    fd_surv_median = float(fd_surv.median()) if len(fd_surv) else None
    # correlation between the two timing fields among deaths (coherence check)
    _both = d_all & fd.notna() & fdd.notna()
    fd_corr = (round(float(np.corrcoef(fd[_both], fdd[_both])[0, 1]), 6)
               if int(_both.sum()) > 2 else None)

    # cumulative HOUR cuts as a fraction of ALL deaths
    cuts = OrderedDict()
    for c in (24, 48, 72, 168):
        nc = int((d_all & (fd <= c)).sum())
        cuts[f"<={c}h"] = {
            "n": nc,
            "pct_of_all_deaths": round(100.0 * nc / n_all, 2) if n_all else 0.0,
        }

    # ---- day-index vs elapsed-hours crosswalk (why the day rule was wrong) --
    crosswalk = OrderedDict()
    for dv in range(1, 6):
        m = (fdd == dv) & fd.notna()
        if int(m.sum()):
            crosswalk[str(dv)] = {
                "n": int(m.sum()),
                "hrs_min": round(float(fd[m].min()), 2),
                "hrs_max": round(float(fd[m].max()), 2),
                "straddles_48h": bool(fd[m].min() <= 48 <= fd[m].max()),
            }
    xw_missed = int((d_all & (fd <= DEATH_LE48H_MAX_HOURS)
                     & ~(fdd <= DEATH_LE48H_MAX_DAYS)).sum())
    xw_false_pos = int((d_all & (fdd <= DEATH_LE48H_MAX_DAYS)
                        & (fd > DEATH_LE48H_MAX_HOURS)).sum())

    # ---- per-year / per-split ASCERTAINMENT of the timing field among deaths -
    fd_year = OrderedDict()
    for yr in sorted([int(y) for y in primary_df["puf_year"].dropna().unique()]):
        sub = primary_df[primary_df["puf_year"] == yr]
        sub_d = sub[sub["death_inhosp"] == 1]
        fd_year[str(yr)] = {
            "n": int(len(sub)),
            "fill_pct": (round(100.0 * float(sub["finaldischarge_hrs"].notna().mean()), 3)
                         if len(sub) else 0.0),
            "fill_pct_days": (round(100.0 * float(sub["finaldischarge_days"].notna().mean()), 3)
                              if len(sub) else 0.0),
            "n_deaths": int(len(sub_d)),
            "deaths_fill_pct": (round(100.0 * float(sub_d["finaldischarge_hrs"].notna().mean()), 3)
                                if len(sub_d) else 0.0),
            "deaths_missing_pct": (round(100.0 * float(sub_d["finaldischarge_hrs"].isna().mean()), 3)
                                   if len(sub_d) else 0.0),
            "n_deaths_missing": int(sub_d["finaldischarge_hrs"].isna().sum()),
        }
    fd_split = OrderedDict()
    for lab, yrs in (("derivation_2019_2022", [2019, 2020, 2021, 2022]),
                     ("validation_2023_2024", [2023, 2024])):
        sub = primary_df[primary_df["puf_year"].isin(yrs)]
        sub_d = sub[sub["death_inhosp"] == 1]
        fd_split[lab] = {
            "years": yrs,
            "n": int(len(sub)),
            "n_deaths": int(len(sub_d)),
            "deaths_missing_n": int(sub_d["finaldischarge_hrs"].isna().sum()),
            "deaths_missing_pct": (round(100.0 * float(sub_d["finaldischarge_hrs"].isna().mean()), 3)
                                   if len(sub_d) else 0.0),
            "death_hemorrhagic_n": int(sub["death_hemorrhagic"].sum()),
            "composite_rate_pct": (round(100.0 * float(sub["major_hemorrhage_event"].mean()), 3)
                                   if len(sub) else 0.0),
        }
    fd_split["differential_pp"] = round(
        fd_split["derivation_2019_2022"]["deaths_missing_pct"]
        - fd_split["validation_2023_2024"]["deaths_missing_pct"], 3)

    ed_and_all = int((d_ed & d_all).sum())
    ed_and_hem = int((d_ed & d_hem).sum())
    ed_not_hem = int((d_ed & ~d_hem).sum())
    ed_not_all = int((d_ed & ~d_all).sum())
    not_ed_all = int((~d_ed & d_all).sum())
    neither = int((~d_ed & ~d_all).sum())

    n_comp_v3 = int(primary_df["major_hemorrhage_event"].sum())
    n_comp_head = int(primary_df["major_hemorrhage_event_headais"].sum())
    n_comp_v2 = int(primary_df["major_hemorrhage_event_v2def"].sum())
    n_state_v3 = int(primary_df["patient_state_composite"].sum())
    n_hc = int(primary_df["received_hc"].sum())
    n_tx_v3 = int(primary_df["substantial_transfusion"].sum())
    n_tx_v2 = int(primary_df["substantial_transfusion_ml_only"].sum())

    # decomposition: hold one fix at a time against the v2 definition
    comp_fix1_only = int(((primary_df["received_hc"] == 1)
                          | (primary_df["death_inhosp"] == 1)
                          | (primary_df["substantial_transfusion"] == 1)).sum())
    comp_fix2_only = int(((primary_df["received_hc"] == 1)
                          | (primary_df["death_hemorrhagic"] == 1)
                          | (primary_df["substantial_transfusion_ml_only"] == 1)).sum())
    # composite under the missing-inclusive death bound
    comp_missing_incl = int(((primary_df["received_hc"] == 1)
                             | (primary_df["death_hemorrhagic_missing_incl"] == 1)
                             | (primary_df["substantial_transfusion"] == 1)).sum())

    def _pct(x):
        return round(100.0 * x / n, 3) if n else 0.0

    def _g(v):
        """Format a possibly-None number (empty-cohort smoke mode)."""
        return "n/a" if v is None else format(v, "g")

    L = []
    A = L.append
    A("# Fix 2 verification — hemorrhagic-death component (W1, FULL COMPLIANCE)")
    A("")
    A("Generated by `v3/code/01_build_cohort.py`. Analytic cohort = "
      f"`cohort_primary == 1`, **N = {n:,}**.")
    A("")
    A("> **Posture change.** The request was for death **<= 48 h AND head "
      "AIS < 4**. The v3 plan intended to comply on the head-AIS half and "
      "*rebut* the <= 48 h half as infeasible. That rebuttal was wrong and has "
      "been withdrawn: the compiled export is a lossy projection whose "
      "date/time columns are all empty, but the **raw** TQP PUF carries "
      "`FINALDISCHARGEDAYS` (\"time from ED/hospital arrival to final "
      "discharge, in days\") at 98.8-99.8% fill per year. Both halves are now "
      "built. Full evidence: `v3/logs/timing_evidence_v3.md`.")
    A("")
    A("## 1. The death family (all variants retained in the cohort CSV)")
    A("")
    A("| Column | Definition | Role | n | % of cohort |")
    A("|---|---|---|---|---|")
    A(f"| `death_hemorrhagic` | `death_inhosp & (ais_head < {AIS_HEAD_SEVERE}) "
      f"& (finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g})` | **PRIMARY "
      f"composite death component / mortality-only anchor** | {n_hem:,} | "
      f"{_pct(n_hem)}% |")
    A(f"| `death_hemorrhagic_headais` | `death_inhosp & (ais_head < "
      f"{AIS_HEAD_SEVERE})` | named sensitivity (head-AIS half only) | "
      f"{n_head:,} | {_pct(n_head)}% |")
    A(f"| `death_le48h` | `death_inhosp & (finaldischarge_hrs <= "
      f"{DEATH_LE48H_MAX_HOURS:g})` | named sensitivity (timing half only) | "
      f"{n_le48:,} | {_pct(n_le48)}% |")
    A(f"| `death_hemorrhagic_dayrule` | `death_inhosp & (ais_head < "
      f"{AIS_HEAD_SEVERE}) & (finaldischarge_days <= {DEATH_LE48H_MAX_DAYS})` "
      f"| named sensitivity — **SUPERSEDED** calendar-day proxy | "
      f"{n_hem_day:,} | {_pct(n_hem_day)}% |")
    A(f"| `death_le48h_dayrule` | `death_inhosp & (finaldischarge_days <= "
      f"{DEATH_LE48H_MAX_DAYS})` | named sensitivity — **SUPERSEDED** | "
      f"{n_le48_day:,} | {_pct(n_le48_day)}% |")
    A(f"| `death_hemorrhagic_dayfallback` | as PRIMARY, but an **untimed** "
      f"death with day index <= {DEATH_LE48H_MAX_DAYS} is kept | pre-specified "
      f"partial-recovery sensitivity | {n_hem_dayfb:,} | {_pct(n_hem_dayfb)}% |")
    A(f"| `death_hemorrhagic_edincl` | as PRIMARY, but an **untimed ED death** "
      f"is kept (an ED death is necessarily within 48 h) | pre-specified "
      f"partial-recovery sensitivity | {n_hem_edincl:,} | "
      f"{_pct(n_hem_edincl)}% |")
    A(f"| `death_hemorrhagic_missing_incl` | as PRIMARY, but **any** missing "
      f"hours value counts as <= 48 h | bounding sensitivity (upper bound) | "
      f"{n_incl:,} | {_pct(n_incl)}% |")
    A(f"| `death_inhosp` | all-cause in-hospital death (v2, unchanged) | named "
      f"sensitivity | {n_all:,} | {_pct(n_all)}% |")
    A(f"| `death_ed` | `ED_DISPOSITION_CODE == '{ED_DEATH_CODE}'` | named "
      f"sensitivity (earliest-death surrogate) | {n_ed:,} | {_pct(n_ed)}% |")
    A("")
    A("### CORRECTION — the timing cut is now on HOURS, not calendar days")
    A("")
    A("> **A previous version of this build used `finaldischarge_days <= 2` and "
      "asserted that AY 2019-2024 \"has no minute- or hour-level fields at "
      "all\". That assertion was FALSE.** The raw PUF carries "
      "**`FINALDISCHARGEHRS`** — elapsed hours to two decimals, "
      "arrival-anchored — in all six years. It was missed because the header "
      "search was widened from `*_DATE`/`*_TIME` to `*DAYS` but never to "
      "`*HRS`: the same failure mode, one rung further out. The raw files "
      "likewise carry `HOSPITALARRIVALHRS`, `EDDISCHARGEHRS`, `INPATIENTHRS`, "
      "`ANGIOGRAPHYHRS`, `HMRRHGCTRLSURGHRS`, `TRAUMASURGEONARRIVALHRS`.")
    A("")
    A(f"`finaldischarge_hrs` among deaths: min = {_g(fd_stats['min'])}, max = "
      f"{_g(fd_stats['max'])}, **{fd_stats['n_negative']}** negative values. "
      "`<= 48` on this field is the requested rule computed **exactly**, "
      "with no approximation.")
    A("")
    A("#### Why the calendar-day rule was wrong")
    A("")
    A("`finaldischarge_days` is a calendar-day **index**, not elapsed time. The "
      "crosswalk in this cohort makes the problem visible — day 3 straddles "
      "48 h:")
    A("")
    A("| `finaldischarge_days` | n | `finaldischarge_hrs` range | straddles 48 h? |")
    A("|---|---|---|---|")
    for dv, cw in crosswalk.items():
        A(f"| {dv} | {cw['n']:,} | {cw['hrs_min']:.2f} - {cw['hrs_max']:.2f} h | "
          f"{'**YES**' if cw['straddles_48h'] else 'no'} |")
    A("")
    A(f"Consequence among in-hospital deaths: the `<= {DEATH_LE48H_MAX_DAYS}`-"
      f"day rule produced **{xw_false_pos}** false positives (it is 100% "
      f"specific) but **missed {xw_missed:,} deaths that truly occurred within "
      f"48 hours** — every one a day-3-or-later record whose elapsed time was "
      f"under 48 h, i.e. a "
      f"{round(100.0 * xw_missed / n_le48, 1) if n_le48 else 0}% under-count of "
      "the timing half. The day-rule columns are retained as named "
      "sensitivities so the earlier result stays auditable.")
    A("")
    A(f"Quantified against the exact rule: of the **{n_le48:,}** deaths with "
      f"`finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g}`, the day rule "
      f"captures **{n_dayrule_true_pos:,}** — sensitivity "
      f"**{dayrule_sens_pct}%** — while remaining 100% specific. "
      f"**{n_day3_within_48h:,}** deaths sit on calendar day 3 yet occurred "
      "within 48 real hours.")
    A("")
    A("> **Recorded, not used:** AY2020 uniquely ships literal `YYYY-MM-DD` "
      "dates (`HospitalArrivalDate`, 100% filled) — an apparent "
      "de-identification inconsistency in that single PUF release. It is "
      "**deliberately not used**: one year of six would make the outcome "
      "definition year-inconsistent, and the literal dates are more "
      "identifiable than the interval fields this build relies on.")
    A("")
    A("## 2. Assertions (CONTRACTS §13 G2 + the v3 death family) — all PASS")
    A("")
    A(f"- **G2a — nesting.** `death_hemorrhagic & ~death_inhosp` = "
      f"**{n_hem_not_all}**. Same check passes for "
      f"`death_hemorrhagic_headais`, `death_le48h`, "
      f"`death_hemorrhagic_missing_incl`, `death_ed`. PASS")
    A("- **Intersection nesting.** `death_hemorrhagic` is a subset of both "
      "`death_hemorrhagic_headais` and `death_le48h` (it is their "
      "intersection), and a subset of `death_hemorrhagic_missing_incl`. PASS")
    A(f"- **G2c — composite nesting.** `patient_state_composite & "
      f"~major_hemorrhage_event` = **{n_state_not_comp}**. PASS")
    A("")
    A("## 3. The restriction ladder — both orderings")
    A("")
    A("The two restrictions commute (both are conjunctions), so the endpoint is "
      "identical; showing both orderings makes each one's **marginal** "
      "contribution visible.")
    A("")
    A("**Ordering A — head-AIS first, then timing:**")
    A("")
    A("| Step | Definition | n | removed at this step |")
    A("|---|---|---|---|")
    A(f"| 1 | `death_inhosp` (all-cause) | **{n_all:,}** | — |")
    A(f"| 2 | `& ais_head < {AIS_HEAD_SEVERE}` | **{n_head:,}** | "
      f"−{n_all - n_head:,} |")
    A(f"| 3 | `& finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g}` | "
      f"**{n_hem:,}** | −{n_head - n_hem:,} |")
    A("")
    A("**Ordering B — timing first, then head-AIS:**")
    A("")
    A("| Step | Definition | n | removed at this step |")
    A("|---|---|---|---|")
    A(f"| 1 | `death_inhosp` (all-cause) | **{n_all:,}** | — |")
    A(f"| 2 | `& finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g}` | "
      f"**{n_le48:,}** | −{n_all - n_le48:,} |")
    A(f"| 3 | `& ais_head < {AIS_HEAD_SEVERE}` | **{n_hem:,}** | "
      f"−{n_le48 - n_hem:,} |")
    A("")
    A(f"Both paths land on **{n_hem:,}**. Marginal reading: the head-AIS "
      f"restriction alone costs {n_all - n_head:,} deaths; the timing "
      f"restriction alone costs {n_all - n_le48:,}; applied together they cost "
      f"{n_all - n_hem:,} of {n_all:,} ("
      f"{round(100.0 * (n_all - n_hem) / n_all, 2) if n_all else 0}%), so the "
      f"two restrictions overlap on "
      f"{(n_all - n_head) + (n_all - n_le48) - (n_all - n_hem):,} deaths.")
    A("")
    A("**Superseded day-rule ladder, for comparison:** "
      f"`death_inhosp` {n_all:,} -> `& ais_head < {AIS_HEAD_SEVERE}` "
      f"{n_head:,} -> `& finaldischarge_days <= {DEATH_LE48H_MAX_DAYS}` "
      f"**{n_hem_day:,}**. The hours rule recovers "
      f"**{n_hem - n_hem_day:+,}** deaths "
      f"({round(100.0 * (n_hem - n_hem_day) / n_hem_day, 1) if n_hem_day else 0}% "
      "more) that the calendar-day index wrongly pushed past 48 h.")
    A("")
    A("## 4. `finaldischarge_hrs` — provenance, ascertainment, distribution")
    A("")
    A("Source: **raw** `_DATA/NTDB/PUF/PUF AY <year>/CSV/PUF_TRAUMA.csv`, "
      "columns `FINALDISCHARGEHRS` (primary) and `FINALDISCHARGEDAYS` "
      "(sensitivity), joined on `inc_key`. Both are **already "
      "arrival-anchored** and used directly. Neither is differenced against "
      "`HOSPITALARRIVALHRS`/`HOSPITALARRIVALDAYS` — those are "
      "*incident*-anchored, and the subtraction mixes origins and manufactures "
      "impossible negative values. (`HOSPITALARRIVALHRS` is also only ~47% "
      "filled.) The join is gated by a hard duplicate-`inc_key` assert.")
    A("")
    A(f"- `finaldischarge_hrs` fill in the analytic cohort: **{fd_fill_pct}%**")
    A(f"- fill among the {n_all:,} in-hospital deaths: **{fd_fill_deaths_pct}%** "
      f"({fd_stats['n_with_value']:,} deaths with a value; "
      f"**{n_deaths_fd_missing:,} without**)")
    A("")
    A("### Ascertainment by year and by temporal split (FINDING 3)")
    A("")
    A("This is a **data-collection artifact, not model behaviour**. It is "
      "reported here so `12` can register it explicitly rather than reading it "
      "as calibration drift.")
    A("")
    A("| PUF year | cohort n | cohort fill % | deaths n | deaths with value % | "
      "deaths MISSING % | raw file HRS fill % |")
    A("|---|---|---|---|---|---|---|")
    for yr, st in fd_year.items():
        raw = fd_per_year.get(yr, {})
        raw_s = (f"{raw.get('raw_fill_pct_hrs')}" if raw else "n/a")
        A(f"| {yr} | {st['n']:,} | {st['fill_pct']}% | {st['n_deaths']:,} | "
          f"{st['deaths_fill_pct']}% | **{st['deaths_missing_pct']}%** | "
          f"{raw_s}% |")
    A("")
    A("| Temporal split | years | N | deaths | deaths missing timing | "
      "missing % | `death_hemorrhagic` | composite rate |")
    A("|---|---|---|---|---|---|---|---|")
    for lab in ("derivation_2019_2022", "validation_2023_2024"):
        st = fd_split[lab]
        A(f"| {lab.replace('_', ' ')} | {st['years'][0]}-{st['years'][-1]} | "
          f"{st['n']:,} | {st['n_deaths']:,} | {st['deaths_missing_n']:,} | "
          f"**{st['deaths_missing_pct']}%** | {st['death_hemorrhagic_n']:,} | "
          f"{st['composite_rate_pct']}% |")
    A("")
    A(f"**Differential = {fd_split['differential_pp']:+.3f} percentage points** "
      "(derivation minus validation). Timing ascertainment is materially worse "
      "in the early years, and the gap sits **exactly on the 2019-2022 / "
      "2023-2024 boundary that `12` uses for temporal validation**. Because a "
      "missing timing value is scored as NOT within 48 h, `death_hemorrhagic` "
      "is under-counted more in the derivation period than in the validation "
      "period. `12` must report this as differential ascertainment; any "
      "apparent calibration drift across the split is confounded with it. The "
      "`death_hemorrhagic_missing_incl` column bounds the effect.")
    A("")
    A("### Distribution among deaths")
    A("")
    A(f"n with a value = {fd_stats['n_with_value']:,}; min "
      f"{_g(fd_stats['min'])} h, median **{_g(fd_stats['median'])} h**, IQR "
      f"{_g(fd_stats['q1'])}-{_g(fd_stats['q3'])} h, max {_g(fd_stats['max'])} h; "
      f"**{fd_stats['n_negative']}** negative values.")
    A("")
    A("| window | " + " | ".join(hr_hist.keys()) + " |")
    A("|---" * (len(hr_hist) + 1) + "|")
    A("| n deaths | " + " | ".join(f"{v:,}" for v in hr_hist.values()) + " |")
    A("")
    A(f"Cumulative, as a fraction of all {n_all:,} in-hospital deaths:")
    A("")
    A("| Cut | n | % of all deaths | meaning |")
    A("|---|---|---|---|")
    meaning = {"<=24h": "died within 24 h of arrival",
               "<=48h": "**within 48 h — the rule used (exact)**",
               "<=72h": "within 72 h",
               "<=168h": "within 7 days"}
    for k, v in cuts.items():
        A(f"| `finaldischarge_hrs {k}` | {v['n']:,} | "
          f"{v['pct_of_all_deaths']}% | {meaning[k]} |")
    A("")
    A("### Coherence checks")
    A("")
    A(f"- **ED deaths arrive at ~0 h.** Median `finaldischarge_hrs` among the "
      f"{n_ed:,} `death_ed` (ED-disposition deceased) patients = "
      f"**{_g(fd_ed_median)} h** — a patient who died in the ED must have a "
      "very short arrival-to-discharge interval.")
    A(f"- **Survivors separate cleanly.** Median among survivors = "
      f"{_g(fd_surv_median)} h.")
    A(f"- **Agreement with the independent day field.** "
      f"corr(`finaldischarge_hrs`, `finaldischarge_days`) among deaths = "
      f"**{fd_corr}**.")
    A(f"- **No negative values** ({fd_stats['n_negative']}), confirming the "
      "arrival anchor.")
    A("")
    A("## 5. The missing-value rule, and what it costs")
    A("")
    A("**Rule (explicit): a missing `finaldischarge_hrs` is treated as NOT "
      "within 48 h.** A NaN comparison is False in pandas, so such a death does "
      "not enter `death_le48h` or `death_hemorrhagic`. The rule is "
      "conservative — it shrinks the primary outcome rather than inflating it. "
      "Note the rule is now keyed on the HOURS field, which is slightly less "
      f"complete than the day field ({fd_fill_pct}% vs "
      f"{round(100.0 * float(fdd.notna().mean()), 3)}% cohort-wide).")
    A("")
    A(f"- in-hospital deaths with no `finaldischarge_hrs` value: "
      f"**{n_deaths_fd_missing:,}** of {n_all:,} "
      f"({round(100.0 * n_deaths_fd_missing / n_all, 2) if n_all else 0}%)")
    A(f"- of those, deaths that would otherwise qualify (head AIS < "
      f"{AIS_HEAD_SEVERE}): **{n_lost_to_missingness:,}** — this is the exact "
      "number of deaths excluded **by missingness alone**")
    A("")
    A("| Death component | n | composite n |")
    A("|---|---|---|")
    A(f"| `death_hemorrhagic` (missing -> excluded; **the rule**) | "
      f"{n_hem:,} | {n_comp_v3:,} |")

    A(f"| `death_hemorrhagic_missing_incl` (missing -> included; bound) | "
      f"{n_incl:,} | {comp_missing_incl:,} |")
    A(f"| **spread attributable to the rule** | **{n_lost_to_missingness:,}** | "
      f"**{comp_missing_incl - n_comp_v3:,}** |")
    A("")
    A("The bound is emitted as a retained column so the rule's effect is "
      "**quantified rather than assumed**; downstream sensitivity analyses can "
      "swap it in without re-deriving anything.")
    A("")
    A("### The untimed-death recovery ladder (all rungs PRE-SPECIFIED)")
    A("")
    A(f"Of the {n_deaths_fd_missing:,} deaths with no `finaldischarge_hrs`, "
      f"**{n_deaths_missing_both:,}** have no day index either, "
      f"**{n_day_only_untimed:,}** do have a day index of <= "
      f"{DEATH_LE48H_MAX_DAYS}, and **{n_deaths_untimed_ed:,}** died in the ED. "
      "Two of those facts are usable evidence of early death. The primary "
      "column deliberately ignores both — one operationalisation per column — "
      "and each recovery route is emitted as its own declared column, so the "
      "choice is visible and auditable rather than buried in a fallback "
      "branch. All rungs are declared here, **before** the numbers are used "
      "anywhere, so none of them can read as a post-hoc rescue.")
    A("")
    A("| Rung | Column | Rule for an untimed death | n | vs primary |")
    A("|---|---|---|---|---|")
    A(f"| **PRIMARY** | `death_hemorrhagic` | excluded (conservative) | "
      f"**{n_hem:,}** | — |")
    A(f"| day-fallback | `death_hemorrhagic_dayfallback` | kept if the calendar "
      f"day index <= {DEATH_LE48H_MAX_DAYS} | {n_hem_dayfb:,} | "
      f"{n_hem_dayfb - n_hem:+,} |")
    A(f"| ED-inclusive | `death_hemorrhagic_edincl` | kept if the patient died "
      f"in the ED | {n_hem_edincl:,} | {n_hem_edincl - n_hem:+,} |")
    A(f"| upper bound | `death_hemorrhagic_missing_incl` | always kept | "
      f"{n_incl:,} | {n_incl - n_hem:+,} |")
    A("")
    A("Rationale for each, for the SAP record:")
    A("")
    A("- **day-fallback** is defensible because the calendar-day rule is 100% "
      "specific among timed records (zero deaths with `days <= 2` and "
      "`hrs > 48`), so an untimed death on day 1-2 is very likely within 48 h. "
      "It is *not* the primary because it mixes two operationalisations inside "
      "a single column.")
    A("- **ED-inclusive** is a logical implication rather than an imputation: a "
      "patient whose ED disposition is 'deceased' necessarily died within 48 h "
      "of arrival. It is the strongest of the recovery routes on evidence, and "
      "the smallest in effect.")
    A("- **upper bound** assumes every untimed death was early. It is not "
      "credible as an estimate; it exists to bound the rule.")
    A("")
    A("## 6. Head-AIS distributions")
    A("")
    A("### Among all deaths, and what the combined restriction removes")
    A("")
    A("| head AIS | all deaths | removed (any reason) | retained "
      "(`death_hemorrhagic`) |")
    A("|---|---|---|---|")
    for k in [0, 1, 2, 3, 4, 5, 6]:
        A(f"| {k} | {dist_alldeath.get(k, 0):,} | {dist_removed.get(k, 0):,} | "
          f"{dist_retained.get(k, 0):,} |")
    A(f"| **total** | **{n_all:,}** | **{n_removed:,}** | **{n_hem:,}** |")
    A("")
    A(f"### Among deaths removed by the <= {DEATH_LE48H_MAX_HOURS:g}-hour "
      "restriction specifically")
    A("")
    A("These are deaths that **passed** the head-AIS filter and were then "
      "removed for dying later than 48 h (or for a missing hours value), so "
      f"every one has head AIS < {AIS_HEAD_SEVERE} by construction — "
      f"n = **{n_removed_time_within_head:,}**.")
    A("")
    A("| head AIS | 0 | 1 | 2 | 3 | 4 | 5 | 6 |")
    A("|---|---|---|---|---|---|---|---|")
    A("| n removed by timing | " +
      " | ".join(f"{dist_removed_time.get(k, 0):,}" for k in [0, 1, 2, 3, 4, 5, 6]) +
      " |")
    A("")
    A("`ais_head` is set to 0 when the patient has no head-region AIS record "
      "(i.e. no coded head injury), which is why AIS 0 is the modal stratum "
      "among retained deaths — those are deaths with no head injury at all.")
    A("")
    A("## 7. `death_ed` and its overlaps (G2d)")
    A("")
    A("| Set | n |")
    A("|---|---|")
    A(f"| `death_ed` | {n_ed:,} |")
    A(f"| `death_ed & death_inhosp` | {ed_and_all:,} |")
    A(f"| `death_ed & death_hemorrhagic` | {ed_and_hem:,} |")
    A(f"| `death_ed & ~death_hemorrhagic` | {ed_not_hem:,} |")
    A("")
    A("Full cross-tab, `death_ed` x `death_inhosp`:")
    A("")
    A("| | `death_inhosp` = 1 | `death_inhosp` = 0 |")
    A("|---|---|---|")
    A(f"| **`death_ed` = 1** | {ed_and_all:,} | {ed_not_all:,} |")
    A(f"| **`death_ed` = 0** | {not_ed_all:,} | {neither:,} |")
    A("")
    A(f"`death_ed` is a strict subset of `death_inhosp` by construction "
      f"(`death_inhosp = (DC_DISPOSITION_CODE == 'D') | (ED_DISPOSITION_CODE "
      f"== '{ED_DEATH_CODE}')`), and the cross-tab confirms it: the "
      f"`death_ed=1 & death_inhosp=0` cell is **{ed_not_all:,}**. Per G2d this "
      "cell is a data note, not a hard fail.")
    A("")
    A("## 8. Fill rates of the fields this fix depends on")
    A("")
    A("| Field | fill in analytic cohort | plan claim |")
    A("|---|---|---|")
    A(f"| `ED_DISPOSITION_CODE` (non-blank) | **{ed_fill_pct}%** | 97.2% |")
    A(f"| `ais_head` (non-NaN) | **{ais_fill_pct}%** | 100% |")
    A(f"| `finaldischarge_days` (non-NaN) | **{fd_fill_pct}%** | n/a (field was "
      "believed absent) |")
    A("")
    A("`ais_head` is 100% by construction (absence of a head-region AIS record "
      "is filled to 0), so the fill rate is a statement about the column, not "
      "about how many patients had a head AIS coded. The plan's 97.2% for "
      "`ED_DISPOSITION_CODE` is the **raw-file** figure; inside the analytic "
      "cohort it is higher because STROBE step 7 already removed most "
      "blank-disposition records.")
    A("")
    A("## 9. Composite recomputation and decomposition")
    A("")
    A("| Composite | Definition | n | % |")
    A("|---|---|---|---|")
    A(f"| `major_hemorrhage_event` (**v3 PRIMARY**) | `received_hc \\| "
      f"death_hemorrhagic \\| substantial_transfusion` | {n_comp_v3:,} | "
      f"{_pct(n_comp_v3)}% |")
    A(f"| `major_hemorrhage_event_headais` (sensitivity) | `received_hc \\| "
      f"death_hemorrhagic_headais \\| substantial_transfusion` | "
      f"{n_comp_head:,} | {_pct(n_comp_head)}% |")
    A(f"| `major_hemorrhage_event_v2def` (fidelity gate + sensitivity) | "
      f"`received_hc \\| death_inhosp \\| substantial_transfusion_ml_only` | "
      f"{n_comp_v2:,} | {_pct(n_comp_v2)}% |")
    A(f"| `patient_state_composite` (v3 co-primary) | `death_hemorrhagic \\| "
      f"substantial_transfusion` | {n_state_v3:,} | {_pct(n_state_v3)}% |")
    A("")
    A("Components of the v3 primary:")
    A("")
    A(f"- `received_hc` = {n_hc:,} ({_pct(n_hc)}%) — unchanged from v2")
    A(f"- `death_hemorrhagic` = {n_hem:,} ({_pct(n_hem)}%) — was "
      f"`death_inhosp` = {n_all:,} in v2, and {n_head:,} under the head-AIS "
      f"half alone")
    A(f"- `substantial_transfusion` = {n_tx_v3:,} ({_pct(n_tx_v3)}%) — was "
      f"{n_tx_v2:,} under the mL-only rule")
    A("")
    A("### Decomposition (each fix applied alone, on top of the v2 definition)")
    A("")
    A(f"Fix 1 alone moves the transfusion component by "
      f"**{fix1['delta']:+,}** ({fix1['n_v2_ml_only']:,} -> "
      f"{fix1['n_v3_unit_aware']:,}); Fix 2 alone moves the death component by "
      f"**{n_hem - n_all:+,}** ({n_all:,} -> {n_hem:,}). Their effect on the "
      "composite is smaller than either, because both components overlap "
      "heavily with `received_hc`.")
    A("")
    A("| Composite variant | n | delta vs v2def |")
    A("|---|---|---|")
    A(f"| v2 definition | {n_comp_v2:,} | — |")
    A(f"| + Fix 1 only (unit-aware tx, all-cause death) | {comp_fix1_only:,} | "
      f"{comp_fix1_only - n_comp_v2:+,} |")
    A(f"| + Fix 2 only (mL-only tx, W1-full hemorrhagic death) | "
      f"{comp_fix2_only:,} | {comp_fix2_only - n_comp_v2:+,} |")
    A(f"| + Fix 1 & Fix 2 head-AIS half only | {n_comp_head:,} | "
      f"{n_comp_head - n_comp_v2:+,} |")
    A(f"| **both, W1 in full (v3 PRIMARY)** | **{n_comp_v3:,}** | "
      f"**{n_comp_v3 - n_comp_v2:+,}** |")
    A("")
    A("The two fixes move the composite in opposite directions — Fix 1 adds "
      "transfused patients, Fix 2 removes head-injury and late deaths — and "
      "they overlap (a reclassified transfusion patient may also be a removed "
      "death), so the deltas do not sum exactly to the net.")
    A("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    return {
        "death_inhosp_n": n_all,
        "death_hemorrhagic_n": n_hem,
        "death_hemorrhagic_headais_n": n_head,
        "death_le48h_n": n_le48,
        "death_le48h_dayrule_n": n_le48_day,
        "death_hemorrhagic_dayrule_n": n_hem_day,
        "death_hemorrhagic_dayfallback_n": n_hem_dayfb,
        "death_hemorrhagic_edincl_n": n_hem_edincl,
        "death_hemorrhagic_missing_incl_n": n_incl,
        "dayrule_sensitivity_pct": dayrule_sens_pct,
        "dayrule_true_positives": n_dayrule_true_pos,
        "day3_deaths_within_48h": n_day3_within_48h,
        "dayrule_untimed_deaths": n_day_only_untimed,
        "deaths_untimed_that_are_ed_deaths": n_deaths_untimed_ed,
        "deaths_missing_both_timing_fields": n_deaths_missing_both,
        "death_ed_n": n_ed,
        "received_hc_n": n_hc,
        "patient_state_composite_n": n_state_v3,
        "patient_state_composite_rate_pct": _pct(n_state_v3),
        "deaths_removed_total": n_removed,
        "deaths_removed_by_head_ais_ge4": n_all - n_head,
        "deaths_removed_by_gt48h_or_missing": n_all - n_le48,
        "dayrule_missed_true_le48h_deaths": xw_missed,
        "dayrule_false_positive_deaths": xw_false_pos,
        "day_hour_crosswalk": {k: v for k, v in crosswalk.items()},
        "deaths_removed_by_timing_within_head_ais_ok": n_removed_time_within_head,
        "deaths_with_missing_finaldischarge_days": n_deaths_fd_missing,
        "deaths_lost_to_missingness_rule": n_lost_to_missingness,
        "head_ais_dist_all_deaths": dist_alldeath,
        "head_ais_dist_removed_deaths": dist_removed,
        "head_ais_dist_retained_deaths": dist_retained,
        "head_ais_dist_removed_by_timing": dist_removed_time,
        "finaldischarge_hrs_stats_deaths": fd_stats,
        "finaldischarge_hrs_hist_deaths": dict(hr_hist),
        "finaldischarge_hrs_cuts": {k: v for k, v in cuts.items()},
        "finaldischarge_hrs_median_ed_deaths": fd_ed_median,
        "finaldischarge_hrs_median_survivors": fd_surv_median,
        "finaldischarge_hrs_fill_pct": fd_fill_pct,
        "finaldischarge_hrs_fill_pct_deaths": fd_fill_deaths_pct,
        "finaldischarge_hrs_days_corr_deaths": fd_corr,
        "ascertainment_by_year": fd_year,
        "ascertainment_by_split": fd_split,
        "death_ed_and_death_inhosp": ed_and_all,
        "death_ed_and_death_hemorrhagic": ed_and_hem,
        "death_ed_not_death_inhosp": ed_not_all,   # G2d data note (expect 0)
        "death_hemorrhagic_not_death_inhosp": n_hem_not_all,
        "patient_state_not_in_composite": n_state_not_comp,
        "ed_disposition_fill_pct": ed_fill_pct,
        "ais_head_fill_pct": ais_fill_pct,
        "composite_v3_n": n_comp_v3,
        "composite_headais_n": n_comp_head,
        "composite_v2def_n": n_comp_v2,
        "composite_fix1_only_n": comp_fix1_only,
        "composite_fix2_only_n": comp_fix2_only,
        "composite_missing_incl_n": comp_missing_incl,
        "assertions_passed": True,
    }


# ---------------------------------------------------------------------------
# FORK-FIDELITY GATE (CONTRACTS §3.1, §13 G2e/G2f)
# ---------------------------------------------------------------------------
# v3 changes OUTCOME DEFINITIONS ONLY.  Every quantity below is derived from
# the cohort definition (which v3 does not touch) or from the retained v2
# outcome columns, so every one of them MUST reproduce the v2 realized figure
# exactly.  A mismatch means the FORK is defective — it is never a result.
FORK_FIDELITY_EXPECTED = OrderedDict([
    ("CSV rows (acetabulum-inclusive)",              291766),
    ("cohort_primary == 1 (analytic N)",             239736),
    ("cc_anticoag (preexisting code 31)",             28966),
    ("cc_bleeding (preexisting code 4)",               2261),
    ("unstable_ring (ICD S32.81*)",                   42798),
    ("sbp_unrecordable (raw SBP2 == 0)",                608),
    ("received_hc",                                   19583),
    ("death_inhosp (v2 all-cause)",                   13426),
    ("substantial_transfusion_ml_only (v2 rule)",     14238),
    ("major_hemorrhage_event_v2def",                  31054),
    ("patient_state_composite (v2 definition)",       22907),
    ("blood_4h 1-50 band (FULL CSV)",                  6592),
])
FORK_FIDELITY_EXPECTED_RATE_PCT = 12.95   # major_hemorrhage_event_v2def rate


def check_fork_fidelity(primary_df, full_df):
    """Recompute every carried-forward v2 figure and PASS/FAIL each one.

    Returns (all_pass, rows) where rows = [(label, observed, expected, ok)].
    """
    b = primary_df["blood_4h"]
    state_v2def = ((primary_df["death_inhosp"] == 1)
                   | (primary_df["substantial_transfusion_ml_only"] == 1))
    observed = OrderedDict([
        ("CSV rows (acetabulum-inclusive)",           int(len(full_df))),
        ("cohort_primary == 1 (analytic N)",          int(len(primary_df))),
        ("cc_anticoag (preexisting code 31)",         int(primary_df["cc_anticoag"].sum())),
        ("cc_bleeding (preexisting code 4)",          int(primary_df["cc_bleeding"].sum())),
        ("unstable_ring (ICD S32.81*)",               int(primary_df["unstable_ring"].sum())),
        ("sbp_unrecordable (raw SBP2 == 0)",          int(primary_df["sbp_unrecordable"].sum())),
        ("received_hc",                               int(primary_df["received_hc"].sum())),
        ("death_inhosp (v2 all-cause)",               int(primary_df["death_inhosp"].sum())),
        ("substantial_transfusion_ml_only (v2 rule)", int(primary_df["substantial_transfusion_ml_only"].sum())),
        ("major_hemorrhage_event_v2def",              int(primary_df["major_hemorrhage_event_v2def"].sum())),
        ("patient_state_composite (v2 definition)",   int(state_v2def.sum())),
        ("blood_4h 1-50 band (FULL CSV)",             int(((full_df["blood_4h"] >= 1) & (full_df["blood_4h"] <= 50)).sum())),
    ])
    rows = []
    for label, exp in FORK_FIDELITY_EXPECTED.items():
        obs = observed[label]
        rows.append((label, obs, exp, obs == exp))

    # the v2def RATE, to 2 dp (CONTRACTS §13 G2e)
    n_primary = len(primary_df)
    rate = (round(100.0 * observed["major_hemorrhage_event_v2def"] / n_primary, 2)
            if n_primary else 0.0)
    rows.append(("major_hemorrhage_event_v2def rate %", rate,
                 FORK_FIDELITY_EXPECTED_RATE_PCT,
                 abs(rate - FORK_FIDELITY_EXPECTED_RATE_PCT) < 1e-9))

    all_pass = all(ok for _l, _o, _e, ok in rows)
    return all_pass, rows


def print_fork_fidelity(rows, all_pass, skipped=False):
    print("-" * 70)
    print("FORK-FIDELITY GATE (CONTRACTS §3.1 / §13 G2e-G2f) — v2 figures that")
    print("MUST reproduce exactly, because v3 changed OUTCOMES ONLY:")
    if skipped:
        print("  *** SKIPPED (--sample mode; gate is only meaningful on the "
              "full run) ***")
        return
    for label, obs, exp, ok in rows:
        obs_s = f"{obs:,}" if isinstance(obs, int) else f"{obs}"
        exp_s = f"{exp:,}" if isinstance(exp, int) else f"{exp}"
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label:<44} "
              f"observed {obs_s:>9}  expected {exp_s:>9}")
    print(f"  ==> FORK-FIDELITY: {'PASS' if all_pass else '*** FAIL ***'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Build the v3 pre-decision pelvic-fracture cohort."
    )
    p.add_argument("--sample", type=int, default=None,
                   help="Smoke test: cap the MAIN scan at the first N rows and "
                        "co-cap the ascending inc_key side tables by ceiling. "
                        "Full run: omit this flag.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    sample_n = args.sample
    t_start = time.time()

    for d in (DERIVED, TABLES, FIGURES, LOGS):
        os.makedirs(d, exist_ok=True)
    for f in (MAIN_CSV, ICD_CSV, AIS_CSV, PRE_CSV):
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    for yr in PUF_YEARS:   # raw PUF (FINALDISCHARGEDAYS) — v3 W1 full-comply
        f = PUF_TRAUMA_TMPL.format(year=yr)
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    ceiling = None
    if sample_n is not None:
        ceiling = get_sample_ceiling(sample_n)
        print(f"*** SMOKE MODE --sample {sample_n:,}: MAIN capped to first "
              f"{sample_n:,} rows; side-table inc_key ceiling = {ceiling} ***",
              flush=True)

    ring_set, acet_set, s32_freq, unstable_ring_set = build_s32_sets(ceiling)
    candidate = ring_set | acet_set
    ais_regions, ais6 = build_ais(candidate, ceiling)
    charlson, anticoag_set, bleeding_set = build_charlson(candidate, ceiling)
    fd_days, fd_per_year = build_finaldischarge_days(candidate)
    cohort_raw, c = scan_main(ring_set, candidate, ais6, sample_n)

    cohort = derive_cohort(cohort_raw, ring_set, acet_set, ais_regions, charlson,
                           unstable_ring_set, anticoag_set, bleeding_set,
                           fd_days)
    cohort_raw = cohort_raw.reset_index(drop=True)

    primary = cohort[cohort["cohort_primary"] == 1].copy()
    n_primary = len(primary)
    n_acet = len(cohort)

    # composite headline
    composite_n = int(primary["major_hemorrhage_event"].sum())
    composite_rate = round(100.0 * float(primary["major_hemorrhage_event"].mean()), 3) if n_primary else 0.0

    # HC-route counts (primary cohort) + by year
    route_counts = primary["hc_route"].value_counts().to_dict()
    route_counts = {k: int(route_counts.get(k, 0))
                    for k in ("surgical_only", "angio_only", "both", "none")}
    rby = (primary.groupby([primary["puf_year"].astype("Int64").astype(str),
                            "hc_route"]).size().unstack("hc_route", fill_value=0))
    route_counts_by_year = {
        yr: {k: int(rby.loc[yr].get(k, 0)) for k in ("surgical_only", "angio_only", "both", "none")}
        for yr in rby.index
    }

    # Gate diagnostics on the PRIMARY analytic cohort.
    pmask = (cohort["cohort_primary"].values == 1)
    gates = compute_gates(cohort_raw[pmask].reset_index(drop=True),
                          primary.reset_index(drop=True))

    blood_fill = round(100.0 * float(primary["blood_4h"].notna().mean()), 3) if n_primary else 0.0
    plasma_fill = round(100.0 * float(primary["plasma_4h"].notna().mean()), 3) if n_primary else 0.0
    platelet_fill = round(100.0 * float(primary["platelet_4h"].notna().mean()), 3) if n_primary else 0.0
    missing_rows = write_missingness(os.path.join(LOGS, "missingness.csv"), primary)
    high_missing = [(v, p) for (v, _n, p) in missing_rows
                    if v in ADJUSTMENT_COVARS and p > 5.0]

    # --- shared MNAR informative-missingness table (§10) ---
    imiss_df, miss_mort = write_informative_missingness(
        os.path.join(DERIVED, "informative_missingness.csv"), primary)

    # --- NEW predictor/flag counts (verification gates) ---
    n_cc_anticoag = int(primary["cc_anticoag"].sum())
    n_cc_bleeding = int(primary["cc_bleeding"].sum())
    n_unstable_ring = int(primary["unstable_ring"].sum())
    n_received_hc = int(primary["received_hc"].sum())
    n_subst_tx = int(primary["substantial_transfusion"].sum())
    n_subst_tx_ml = int(primary["substantial_transfusion_ml_only"].sum())
    n_mtp = int(primary["mtp_equivalent"].sum())
    n_death = int(primary["death_inhosp"].sum())
    n_death_hem = int(primary["death_hemorrhagic"].sum())
    n_death_headais = int(primary["death_hemorrhagic_headais"].sum())
    n_death_le48h = int(primary["death_le48h"].sum())
    n_death_le48h_day = int(primary["death_le48h_dayrule"].sum())
    n_death_hem_day = int(primary["death_hemorrhagic_dayrule"].sum())
    n_death_hem_dayfb = int(primary["death_hemorrhagic_dayfallback"].sum())
    n_death_hem_edincl = int(primary["death_hemorrhagic_edincl"].sum())
    fd_hrs_fill = round(100.0 * float(primary["finaldischarge_hrs"].notna().mean()), 3) if n_primary else 0.0
    n_death_hem_incl = int(primary["death_hemorrhagic_missing_incl"].sum())
    n_death_ed = int(primary["death_ed"].sum())
    n_state_comp = int(primary["patient_state_composite"].sum())
    n_comp_v2def = int(primary["major_hemorrhage_event_v2def"].sum())
    n_comp_headais = int(primary["major_hemorrhage_event_headais"].sum())
    fd_fill = round(100.0 * float(primary["finaldischarge_days"].notna().mean()), 3) if n_primary else 0.0
    n_sbp_unrec = int(primary["sbp_unrecordable"].sum())
    state_comp_rate = round(100.0 * float(primary["patient_state_composite"].mean()), 3) if n_primary else 0.0

    # --- v3 Fix 1 / Fix 2 verification artifacts (hard-asserting) -----------
    print("\n[v3] Writing Fix 1 / Fix 2 verification audits ...", flush=True)
    fix1 = write_fix1_audit(os.path.join(LOGS, "fix1_transfusion_audit.md"),
                            primary, cohort)
    ed_raw_primary = _clean_str(
        cohort_raw.loc[pmask, "ED_DISPOSITION_CODE"]).reset_index(drop=True)
    fix2 = write_fix2_audit(os.path.join(LOGS, "fix2_death_audit.md"),
                            primary.reset_index(drop=True), ed_raw_primary,
                            fix1, fd_per_year)
    print(f"      Fix 1: substantial_transfusion {fix1['n_v2_ml_only']:,} (v2) "
          f"-> {fix1['n_v3_unit_aware']:,} (v3)  [{fix1['delta']:+,}]", flush=True)
    print(f"      Fix 2: death_inhosp {fix2['death_inhosp_n']:,} -> "
          f"death_hemorrhagic {fix2['death_hemorrhagic_n']:,}  "
          f"[-{fix2['deaths_removed_total']:,}: head AIS>={AIS_HEAD_SEVERE} "
          f"and/or >{DEATH_LE48H_MAX_HOURS:g}h]", flush=True)

    # --- FORK-FIDELITY GATE (CONTRACTS §3.1, §13 G2e/G2f) ------------------
    ff_pass, ff_rows = check_fork_fidelity(primary, cohort)
    print()
    print_fork_fidelity(ff_rows, ff_pass, skipped=(sample_n is not None))
    if sample_n is not None:
        ff_pass = True   # not meaningful on a truncated scan

    # --- write cohort CSV (COHORT_COLUMNS order is the contract) -----------
    cohort_path = os.path.join(DERIVED, "pelvic_cohort_predecision.csv")
    cohort.to_csv(cohort_path, index=False, columns=COHORT_COLUMNS)

    # --- STROBE flow + ordered exclusions JSON (14_consort consumes it) ---
    flow = build_cohort_flow(c)
    write_cohort_flow_txt(os.path.join(LOGS, "cohort_flow.txt"),
                          flow, gates, n_primary, n_acet, route_counts,
                          composite_n, composite_rate, fix1, fix2)
    exclusions = OrderedDict((STEP_LABELS[k], int(c[k])) for k in range(1, N_STEPS + 1))
    with open(os.path.join(DERIVED, "cohort_exclusions.json"), "w", encoding="utf-8") as fh:
        json.dump(exclusions, fh, indent=2)

    write_unknown_hc_codes(os.path.join(LOGS, "unknown_hc_codes.csv"), primary)

    # --- funnel gate (v3 changes outcomes ONLY; the funnel must not move) ---
    gate_expected_primary, gate_expected_rows = 239736, 291766
    gate_pass = (sample_n is not None) or (
        n_primary == gate_expected_primary and n_acet == gate_expected_rows
        and ff_pass)

    notes = [
        "v3 = v2 pipeline with TWO OUTCOME-DEFINITION FIXES. The cohort FUNNEL "
        "is UNCHANGED from v2 (same 7 pre-decision STROBE steps, same "
        "exclusions): analytic N (cohort_primary==1) = 239,736; CSV rows = "
        "291,766. No exclusion step was touched.",
        "v3 PRE-DECISION cohort: HC-inclusion filter DROPPED (CONTRACTS §9-A); "
        "hemorrhage control is an OUTCOME component of major_hemorrhage_event, "
        "not a cohort filter. STROBE funnel = 7 pre-decision steps.",
        "Both QC gates FAIL on the COMPILED export: all 33 of its date/time "
        "columns are 0% filled, so no death DATETIME is derivable there. That "
        "does NOT mean time-to-death is unavailable -- see the "
        "finaldischarge_days block: the RAW PUF carries day-resolution "
        "arrival-anchored offsets, and the <=48h restriction IS built. The "
        "gates are retained unchanged as an honest statement about the "
        "compiled export only.",
        "v3 W1 FULL COMPLIANCE (supersedes the plan's 'rebut the <=48h half' "
        "posture, which was based on a false premise): FINALDISCHARGEDAYS "
        "('Time from ED/Hospital Arrival to Final Discharge (days)') is joined "
        "from the RAW PUF_TRAUMA.csv files on inc_key. It is ALREADY "
        "arrival-anchored and is used DIRECTLY -- it must NOT be differenced "
        "against HOSPITALARRIVALDAYS, which is incident-anchored (that "
        "subtraction manufactures impossible negative values). Evidence: "
        "logs/timing_evidence_v3.md.",
        "TIMING CUT IS ON HOURS. FINALDISCHARGEHRS (elapsed hours, two "
        "decimals, arrival-anchored, ~98.8% raw fill) exists in all six raw "
        "years and gives W1's '<=48h' EXACTLY. An earlier build used "
        f"'finaldischarge_days <= {DEATH_LE48H_MAX_DAYS}' and asserted that AY "
        "2019-2024 has no minute/hour fields -- that assertion was FALSE and is "
        "withdrawn. The header search had been widened from *_DATE/*_TIME to "
        "*DAYS but never to *HRS. The raw files also carry HOSPITALARRIVALHRS, "
        "EDDISCHARGEHRS, INPATIENTHRS, ANGIOGRAPHYHRS, HMRRHGCTRLSURGHRS, "
        "TRAUMASURGEONARRIVALHRS.",
        "The calendar-day rule is SUPERSEDED but retained as "
        "death_hemorrhagic_dayrule / death_le48h_dayrule. finaldischarge_days "
        "is a calendar-day INDEX, not elapsed time: day 3 straddles 48h. The "
        f"day rule had {fix2['dayrule_false_positive_deaths']} false positives "
        f"(100% specific) but MISSED {fix2['dayrule_missed_true_le48h_deaths']:,} "
        "deaths that truly occurred within 48h.",
        "MISSING-VALUE RULE (explicit): a missing finaldischarge_hrs is "
        "treated as NOT within 48h, so such a death does NOT enter death_le48h "
        "or death_hemorrhagic. Conservative -- it shrinks the primary outcome. "
        f"Deaths excluded by missingness ALONE = "
        f"{fix2['deaths_lost_to_missingness_rule']:,} (of "
        f"{fix2['deaths_with_missing_finaldischarge_days']:,} deaths with no "
        "hours value). The rule is BOUNDED, not assumed, by the retained "
        "column death_hemorrhagic_missing_incl.",
        "DIFFERENTIAL ASCERTAINMENT (for script 12): timing-field missingness "
        "among deaths is "
        f"{fix2['ascertainment_by_split']['derivation_2019_2022']['deaths_missing_pct']}% "
        "in the derivation window (2019-2022) vs "
        f"{fix2['ascertainment_by_split']['validation_2023_2024']['deaths_missing_pct']}% "
        f"in validation (2023-2024) -- a "
        f"{fix2['ascertainment_by_split']['differential_pp']:+.3f} pp gap "
        "sitting exactly on the temporal split. Because missing timing is "
        "scored as NOT within 48h, death_hemorrhagic is under-counted more in "
        "derivation. 12 must report this as a DATA-COLLECTION ARTIFACT, not as "
        "model miscalibration.",
        "AIS PARSER DEFECT FIX (inherited from v2, corrected in v3): the "
        "compiled AIS export stores AISPreDot as a FLOAT, so chapter-0 codes "
        "like '010008' round-trip to '10008' and the region digit read from "
        "position 0 was the SECOND digit -- scoring External/chapter-0 codes as "
        "HEAD (or CHEST) injuries. Fixed with .str.zfill(6) before the region "
        "digit is taken. Nationally 12,898 of 25,483,291 AIS rows are 5-digit; "
        "7,026 were mis-mapped to region 1 (head) and 192 to region 4 (chest). "
        "Direction is INFLATION-ONLY. ais_head is outcome-defining in v3 "
        "(death_hemorrhagic), so this is a corrected OUTCOME misclassification "
        "and must be registered as a deviation from the v2 SAP. The AIS-6 "
        "exclusion is region-independent, so the cohort funnel is unaffected.",
        "FIX 1 (S1) unit-aware transfusion decoder: TQIP_HC_BLOOD_UNIT is 100% "
        "blank and TQIP_HC_BLOOD_4HR is bimodal (units cluster 1-100, mL "
        "cluster >=200, near-empty 101-199 gap). substantial_transfusion = "
        "(blood_4h >= 1000) | (4 <= blood_4h <= 100). The v2 mL-only rule is "
        "retained as substantial_transfusion_ml_only (named sensitivity, and a "
        "strict subset of the new rule). blood_4h_scale records the inferred "
        "scale per row. blood_4h NaN -> not flagged (downstream sensitivity).",
        f"FIX 1 delta: substantial_transfusion {fix1['n_v2_ml_only']:,} (v2 "
        f"mL-only) -> {fix1['n_v3_unit_aware']:,} (v3 unit-aware); "
        f"{fix1['n_reclassified']:,} newly flagged, all from the 4-100 units "
        f"band. {fix1['n_units_1_3_not_flagged']:,} records at 1-3 U are below "
        f"the >=4 U threshold and are NOT flagged. "
        f"{fix1['n_ambiguous_gap']:,} records sit in the 101-199 ambiguous gap "
        f"and {fix1['n_ambiguous_gap_flagged']} of them are flagged. "
        "See logs/fix1_transfusion_audit.md.",
        "mtp_equivalent is DEFINITIONALLY IDENTICAL to the v3 "
        "substantial_transfusion (>=4 U or >=1000 mL / 4 h; PMID 25440492). It "
        "is a separate column only because script 17 analyses it as a "
        "standalone secondary outcome -- it is NOT an independent endpoint.",
        "FIX 2 (W1, FULL) hemorrhagic-death component: death_hemorrhagic = "
        f"death_inhosp & (ais_head < {AIS_HEAD_SEVERE}) & "
        f"(finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g}) is the PRIMARY "
        "composite death component AND the mortality-only anchor. The timing "
        "cut is on HOURS (exact); the calendar-day rule is SUPERSEDED and kept "
        "only as a sensitivity. Retained named sensitivities: "
        "death_hemorrhagic_headais (head-AIS half only), death_le48h (timing "
        "half only), death_hemorrhagic_dayrule / death_le48h_dayrule "
        "(superseded day proxy), death_hemorrhagic_dayfallback and "
        "death_hemorrhagic_edincl (pre-specified partial recovery of untimed "
        "deaths), death_hemorrhagic_missing_incl (upper bound), death_inhosp "
        "(all-cause), death_ed (ED_DISPOSITION_CODE=='MOR').",
        f"FIX 2 ladder: death_inhosp {fix2['death_inhosp_n']:,} -> "
        f"& ais_head<{AIS_HEAD_SEVERE} "
        f"{fix2['death_hemorrhagic_headais_n']:,} -> "
        f"& <={DEATH_LE48H_MAX_HOURS:g}h "
        f"{fix2['death_hemorrhagic_n']:,}. Other ordering: "
        f"{fix2['death_inhosp_n']:,} -> & <={DEATH_LE48H_MAX_HOURS:g}h "
        f"{fix2['death_le48h_n']:,} -> & ais_head<{AIS_HEAD_SEVERE} "
        f"{fix2['death_hemorrhagic_n']:,}. Untimed-death recovery rungs: "
        f"primary {fix2['death_hemorrhagic_n']:,} / day-fallback "
        f"{fix2['death_hemorrhagic_dayfallback_n']:,} / ED-inclusive "
        f"{fix2['death_hemorrhagic_edincl_n']:,} / all-missing-included "
        f"{fix2['death_hemorrhagic_missing_incl_n']:,}. Superseded day-rule "
        f"proxy {fix2['death_hemorrhagic_dayrule_n']:,}. death_ed = "
        f"{fix2['death_ed_n']:,}. See logs/fix2_death_audit.md.",
        "v3 PRIMARY composite = received_hc | death_hemorrhagic | "
        "substantial_transfusion. Co-primary anti-circularity = "
        "patient_state_composite (death_hemorrhagic | substantial_transfusion; "
        "HC dropped). Mortality-only anchor is now death_hemorrhagic (head AIS "
        "< 4 AND <=48h). Retained composites: "
        f"major_hemorrhage_event_headais (n = {n_comp_headais:,}, the "
        "head-AIS-half-only variant) and the exact v2 primary "
        f"major_hemorrhage_event_v2def (n = {n_comp_v2def:,}, fork-fidelity "
        "gate).",
        "cc_anticoag (preexisting code 31) and cc_bleeding (code 4) are NOT "
        "Charlson components -- collected separately in build_charlson; charlson "
        "unaffected. unstable_ring = ICD S32.81* (disrupted ring).",
        "sbp_unrecordable = raw SBP2==0 sentinel captured BEFORE the 0->NaN mask "
        "(MNAR-as-signal); it is a subset of sbp2-missing.",
        f"blood_4h fill (primary pre-decision cohort) = {blood_fill}%.",
        "FIX 7 companion fields plasma_4h (TQIP_HC_PLASMA_4HR) and "
        "platelet_4h (TQIP_HC_PLATE_4HR) are now pulled into the cohort CSV "
        "for the EXPLORATORY balanced-resuscitation subgroup. NO decoder is "
        "applied to them -- they are raw-numeric and NaN-preserving. Fill: "
        f"plasma_4h = {plasma_fill}%, platelet_4h = {platelet_fill}% "
        "(plan expected ~27.3%). Their unit-vs-mL band distributions and a "
        "data-driven bimodality verdict are in "
        "logs/fix1_transfusion_audit.md §6; any ratio built from them "
        "inherits an undetermined scale and must be labelled exploratory.",
        "FORK-FIDELITY GATE (CONTRACTS §3.1 / §13 G2e-G2f): every "
        "carried-forward v2 figure (CSV rows, analytic N, cc_anticoag, "
        "cc_bleeding, unstable_ring, sbp_unrecordable, received_hc, "
        "death_inhosp, substantial_transfusion_ml_only, "
        "major_hemorrhage_event_v2def + its rate, v2-definition "
        "patient_state_composite, full-CSV blood_4h 1-50 band) is recomputed "
        f"and checked -- result: {'PASS' if ff_pass else 'FAIL'}. See the "
        "'fork_fidelity' block above for the per-quantity ledger.",
        "NOTE on the CONTRACTS §13 G1a wording: the 6,592 figure for the "
        "blood_4h 1-50 band is a FULL-CSV (291,766-row) count and reproduces "
        "there exactly. The analytic-cohort (cohort_primary==1) 1-50 band is "
        f"{fix1['bands_analytic']['1-50']:,}. Both are reported in "
        "logs/fix1_transfusion_audit.md; the contract's 'in the analytic "
        "cohort' phrasing names the wrong denominator.",
        f"Informative-missingness: MISSING-vitals stratum mortality = "
        f"{miss_mort}% (v2 realized ~16.26%).",
    ]
    if high_missing:
        notes.append("Adjustment-set covariates with >5% missingness (MI m=20 "
                     "sensitivity advised): " +
                     ", ".join(f"{v}={p}%" for v, p in high_missing))
    if not gate_pass:
        notes.append(f"*** FUNNEL GATE FAILED: n_primary={n_primary:,} "
                     f"(expected {gate_expected_primary:,}), CSV rows="
                     f"{n_acet:,} (expected {gate_expected_rows:,}). ***")

    manifest = {
        "run_seed": RANDOM_SEED,
        "sample_mode": (None if sample_n is None else int(sample_n)),
        "version": "v3",
        "gates": {
            "timing_gate_pass": gates["timing_gate_pass"],
            "timing_valid_pct": gates["timing_valid_pct"],
            "mortality_24h_gate_pass": gates["mortality_24h_gate_pass"],
            # v3 PRIMARY OUTCOME (was stale: it still advertised the superseded
            # v2 anchor "in_hospital_mortality").
            "primary_outcome": "major_hemorrhage_event",
            "primary_outcome_definition": (
                "received_hc | death_hemorrhagic | substantial_transfusion"),
            "mortality_only_anchor": "death_hemorrhagic",
            "mortality_only_anchor_definition": (
                f"death_inhosp & (ais_head < {AIS_HEAD_SEVERE}) & "
                f"(finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g})"),
            "compiled_export_death_datetime_derivable":
                gates["primary_outcome"],
            "funnel_gate_pass": bool(gate_pass),
            "funnel_gate_expected_n_primary": gate_expected_primary,
            "funnel_gate_expected_csv_rows": gate_expected_rows,
            "fork_fidelity_pass": bool(ff_pass),
            "fork_fidelity_skipped_sample_mode": bool(sample_n is not None),
        },
        # CONTRACTS §3.1 / §13 G2e-G2f: every carried-forward v2 figure, with
        # its expected value, so the gate is auditable from the manifest alone.
        "fork_fidelity": [
            {"quantity": label, "observed": obs, "expected": exp, "pass": bool(ok)}
            for label, obs, exp, ok in ff_rows
        ],
        "cohort_flow": flow,
        "n_primary": int(n_primary),
        "n_acetabulum_inclusive": int(n_acet),
        # ---- v3 outcome definitions (PRIMARY) ----
        "primary_composite": {
            "definition": ("received_hc | death_hemorrhagic | "
                           "substantial_transfusion"),
            "major_hemorrhage_event_n": composite_n,
            "major_hemorrhage_event_rate_pct": composite_rate,
            "received_hc_n": n_received_hc,
            "death_hemorrhagic_n": n_death_hem,
            "substantial_transfusion_n": n_subst_tx,
            "patient_state_composite_n": n_state_comp,
            "patient_state_composite_rate_pct": state_comp_rate,
        },
        # ---- v2 definitions, retained side-by-side for audit ----
        "v2def_composite": {
            "definition": ("received_hc | death_inhosp | "
                           "substantial_transfusion_ml_only"),
            "major_hemorrhage_event_v2def_n": n_comp_v2def,
            "major_hemorrhage_event_v2def_rate_pct": (
                round(100.0 * n_comp_v2def / n_primary, 3) if n_primary else 0.0),
            "received_hc_n": n_received_hc,
            "death_inhosp_n": n_death,
            "substantial_transfusion_ml_only_n": n_subst_tx_ml,
            "v2_realized_reference": {
                "major_hemorrhage_event_n": 31054,
                "major_hemorrhage_event_rate_pct": 12.95,
                "received_hc_n": 19583,
                "death_inhosp_n": 13426,
                "substantial_transfusion_n": 14238,
                "patient_state_composite_n": 22907,
            },
        },
        "death_variants": {
            # NEW primary composite component AND the mortality-only anchor:
            # in-hospital death with head AIS < 4 occurring within 2 hospital
            # days of arrival (provably < 48 h).  W1 complied with IN FULL.
            "death_hemorrhagic_n": n_death_hem,
            "death_hemorrhagic_definition": (
                f"death_inhosp & (ais_head < {AIS_HEAD_SEVERE}) & "
                f"(finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g})"),
            "death_hemorrhagic_headais_n": n_death_headais,  # head-half only
            "death_le48h_n": n_death_le48h,                  # timing-half only
            "death_hemorrhagic_dayrule_n": n_death_hem_day,  # superseded rule
            "death_le48h_dayrule_n": n_death_le48h_day,      # superseded rule
            # pre-specified partial-recovery sensitivities for untimed deaths
            "death_hemorrhagic_dayfallback_n": n_death_hem_dayfb,
            "death_hemorrhagic_edincl_n": n_death_hem_edincl,
            "untimed_death_recovery_ladder": {
                "primary_no_recovery": n_death_hem,
                "day_fallback": n_death_hem_dayfb,
                "ed_inclusive": n_death_hem_edincl,
                "all_missing_included_upper_bound": n_death_hem_incl,
                "superseded_dayrule_proxy": n_death_hem_day,
            },
            "death_hemorrhagic_missing_incl_n": n_death_hem_incl,  # bound
            "death_inhosp_n": n_death,            # named sensitivity
            "death_ed_n": n_death_ed,             # earliest-death sensitivity
            "missing_value_rule": (
                "finaldischarge_hrs missing -> NOT within 48h (conservative); "
                "bounded by death_hemorrhagic_missing_incl"),
            "deaths_lost_to_missingness_rule":
                fix2["deaths_lost_to_missingness_rule"],
        },
        # v3 W1 full-comply: raw-PUF day-resolution timing (see
        # logs/timing_evidence_v3.md).  Supersedes the v2/v3-plan claim that no
        # time-to-death field exists.
        "finaldischarge_timing": {
            "source": "RAW _DATA/NTDB/PUF/PUF AY <year>/CSV/PUF_TRAUMA.csv "
                      f"columns {PUF_FD_HRS_COL} (primary) and {PUF_FD_COL} "
                      f"(sensitivity), joined on {PUF_KEY_COL}",
            "anchor": "ED/hospital ARRIVAL (used directly; NOT differenced "
                      "against HOSPITALARRIVAL*, which is incident-anchored)",
            "primary_field": "finaldischarge_hrs",
            "primary_rule": f"finaldischarge_hrs <= {DEATH_LE48H_MAX_HOURS:g}",
            "superseded_rule": f"finaldischarge_days <= {DEATH_LE48H_MAX_DAYS}",
            "dayrule_missed_true_le48h_deaths":
                fix2["dayrule_missed_true_le48h_deaths"],
            "dayrule_false_positive_deaths":
                fix2["dayrule_false_positive_deaths"],
            "day_hour_crosswalk": fix2["day_hour_crosswalk"],
            "hrs_days_corr_among_deaths":
                fix2["finaldischarge_hrs_days_corr_deaths"],
            "ascertainment_by_split": fix2["ascertainment_by_split"],
            "join_duplicate_key_check": fd_per_year.get("_join_check", {}),
            "cohort_fill_pct": fd_fill,
            "deaths_fill_pct": fix2["finaldischarge_hrs_fill_pct_deaths"],
            "raw_file_fill_by_year": {k: v for k, v in fd_per_year.items()},
            "cohort_fill_by_year": fix2["ascertainment_by_year"],
            "stats_among_deaths": fix2["finaldischarge_hrs_stats_deaths"],
            "hist_among_deaths": fix2["finaldischarge_hrs_hist_deaths"],
            "cuts": fix2["finaldischarge_hrs_cuts"],
            "median_ed_deaths": fix2["finaldischarge_hrs_median_ed_deaths"],
            "median_survivors": fix2["finaldischarge_hrs_median_survivors"],
        },
        "headais_only_composite": {
            "definition": ("received_hc | death_hemorrhagic_headais | "
                           "substantial_transfusion"),
            "major_hemorrhage_event_headais_n": n_comp_headais,
            "major_hemorrhage_event_headais_rate_pct": (
                round(100.0 * n_comp_headais / n_primary, 3) if n_primary else 0.0),
        },
        "secondary_outcomes": {
            "mtp_equivalent_n": n_mtp,
            "mtp_equivalent_note": ("definitionally identical to "
                                    "substantial_transfusion; not an "
                                    "independent endpoint"),
        },
        # Fix 7 exploratory balanced-resuscitation inputs (no decoder applied).
        "fix7_balanced_resuscitation_fields": {
            "plasma_4h_fill_pct": plasma_fill,
            "platelet_4h_fill_pct": platelet_fill,
            "plan_expected_fill_pct_SUPERSEDED": 27.3,
            "plan_figure_note": (
                "The plan's 27.3% is a SAMPLING ARTIFACT (first 1.5M rows of a "
                "year-ordered file, ~73% AY2019, the one year these fields were "
                "not collected). True pool-wide fill 77.55%, ~90% in 2020-2024. "
                "See logs/timing_evidence_v3.md section 6. Quote 77.5% or ~90%, "
                "never 27.3%."),
            "detail": fix1["fix7_plasma_platelet"],
        },
        "fix1_transfusion": fix1,
        "fix2_death": fix2,
        # v3 DEFECT FIX (inherited from v2): AIS chapter-0 mis-parse.
        "ais_predot_zfill_fix": {
            "defect": ("compiled AIS export stores AISPreDot as a float, so "
                       "chapter-0 codes ('010008') lose the leading zero and "
                       "the region digit read at position 0 is the code's "
                       "SECOND digit"),
            "fix": "AISPreDot .str.zfill(6) before taking the region digit",
            "national_5digit_ais_rows": 12898,
            "national_total_ais_rows": 25483291,
            "mismapped_to_head_region1": 7026,
            "mismapped_to_chest_region4": 192,
            "mismapped_to_abd_region5": 0,
            "direction": "inflation-only (fix can only move a code out of a "
                         "mapped region into unmapped chapter 0)",
            "funnel_impact": "none (AIS-6 exclusion is region-independent)",
            "outcome_impact": ("ais_head is outcome-defining in v3 via "
                               "death_hemorrhagic, so this corrects an OUTCOME "
                               "misclassification; register as a v2-SAP "
                               "deviation"),
            "ais_head_ge3_n": int(primary["ais_head_ge3"].sum()),
            "tbi_severe_n": int(primary["tbi_severe"].sum()),
        },
        "new_predictors": {
            "cc_anticoag_n": n_cc_anticoag,
            "cc_bleeding_n": n_cc_bleeding,
            "unstable_ring_n": n_unstable_ring,
            "sbp_unrecordable_n": n_sbp_unrec,
        },
        "hc_route_counts": route_counts,
        "hc_route_counts_by_year": route_counts_by_year,
        "s32_frequencies": {k: int(v) for k, v in s32_freq.items()},
        "blood_4h_fill_pct": blood_fill,
        "informative_missingness_missing_stratum_mortality_pct": miss_mort,
        "cohort_columns": list(COHORT_COLUMNS),
        "notes": notes,
    }
    with open(os.path.join(LOGS, "qc_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # --- summary ---
    def _p(x):
        return round(100.0 * x / n_primary, 2) if n_primary else 0.0

    print("\n" + "=" * 70)
    print("01_build_cohort.py COMPLETE (v3 pre-decision; Fix 1 + Fix 2)")
    print("=" * 70)
    print(f"  Primary (ring-only) analytic N : {n_primary:,}   "
          f"[gate: {gate_expected_primary:,}]")
    print(f"  Acetabulum-inclusive CSV rows  : {n_acet:,}   "
          f"[gate: {gate_expected_rows:,}]")
    print(f"  FUNNEL GATE                    : "
          f"{'PASS' if gate_pass else '*** FAIL ***'}"
          f"{'  (skipped in --sample mode)' if sample_n is not None else ''}")
    print("-" * 70)
    print("  v3 PRIMARY  major_hemorrhage_event")
    print("      = received_hc | death_hemorrhagic | substantial_transfusion")
    print(f"      {composite_n:,} ({composite_rate}%)        "
          f"[v2 realized: 31,054 (12.95%)]")
    print(f"    received_hc                  : {n_received_hc:,} "
          f"({_p(n_received_hc)}%)   [v2: 19,583]")
    print(f"    death_hemorrhagic  (NEW)     : {n_death_hem:,} "
          f"({_p(n_death_hem)}%)   [head AIS<4 AND <=48h; v2 used "
          f"death_inhosp: 13,426]")
    print(f"    substantial_transfusion      : {n_subst_tx:,} "
          f"({_p(n_subst_tx)}%)   [v2: 14,238]")
    print("  v3 CO-PRIMARY  patient_state_composite")
    print("      = death_hemorrhagic | substantial_transfusion")
    print(f"      {n_state_comp:,} ({state_comp_rate}%)         "
          f"[v2 realized: 22,907 (9.56%)]")
    print("-" * 70)
    print("  FIX 1 (S1) unit-aware transfusion decoder")
    print(f"    substantial_transfusion_ml_only (v2 rule) : "
          f"{n_subst_tx_ml:,}")
    print(f"    substantial_transfusion         (v3 rule) : "
          f"{n_subst_tx:,}   delta {fix1['delta']:+,}")
    print(f"    reclassified (4-100 U band)               : "
          f"{fix1['n_reclassified']:,}")
    print(f"    1-3 U, below the >=4 U threshold          : "
          f"{fix1['n_units_1_3_not_flagged']:,}  (NOT flagged)")
    print(f"    101-199 ambiguous gap                     : "
          f"{fix1['n_ambiguous_gap']:,}  "
          f"(flagged: {fix1['n_ambiguous_gap_flagged']})")
    print(f"    mtp_equivalent (== substantial_transfusion): {n_mtp:,}")
    print("  FIX 2 (W1, FULL COMPLIANCE) hemorrhagic-death component")
    print(f"    death_hemorrhagic (NEW primary component) : {n_death_hem:,}"
          f"   = death_inhosp & ais_head<{AIS_HEAD_SEVERE} & "
          f"finaldischarge_hrs<={DEATH_LE48H_MAX_HOURS:g}")
    print(f"      ladder A  {n_death:,} -> (head AIS<{AIS_HEAD_SEVERE}) "
          f"{n_death_headais:,} -> (<={DEATH_LE48H_MAX_HOURS:g}h) {n_death_hem:,}")
    print(f"      ladder B  {n_death:,} -> (<={DEATH_LE48H_MAX_HOURS:g}h) "
          f"{n_death_le48h:,} -> (head AIS<{AIS_HEAD_SEVERE}) {n_death_hem:,}")
    print(f"      superseded day rule -> {n_death_hem_day:,}  "
          f"(hours rule recovers {n_death_hem - n_death_hem_day:+,}; day rule "
          f"missed {fix2['dayrule_missed_true_le48h_deaths']:,} true <=48h "
          f"deaths, {fix2['dayrule_false_positive_deaths']} false positives)")
    print(f"    death_hemorrhagic_headais (sensitivity)   : {n_death_headais:,}")
    print(f"    death_le48h               (sensitivity)   : {n_death_le48h:,}")
    print(f"    death_hemorrhagic_dayrule (sensitivity)   : {n_death_hem_day:,}"
          f"   [SUPERSEDED calendar-day proxy]")
    print(f"    death_le48h_dayrule       (sensitivity)   : {n_death_le48h_day:,}")
    print("    untimed-death recovery ladder (all PRE-SPECIFIED):")
    print(f"        primary (untimed excluded)     = {n_death_hem:,}")
    print(f"        + day-fallback                 = {n_death_hem_dayfb:,}"
          f"  ({n_death_hem_dayfb - n_death_hem:+,})")
    print(f"        + ED-inclusive                 = {n_death_hem_edincl:,}"
          f"  ({n_death_hem_edincl - n_death_hem:+,})")
    print(f"        + all-missing (upper bound)    = {n_death_hem_incl:,}"
          f"  ({n_death_hem_incl - n_death_hem:+,})")
    print(f"    day rule vs exact: sensitivity "
          f"{fix2['dayrule_sensitivity_pct']}% "
          f"({fix2['dayrule_true_positives']:,}/{n_death_le48h:,}), "
          f"specificity 100% ({fix2['dayrule_false_positive_deaths']} FP); "
          f"{fix2['day3_deaths_within_48h']:,} day-3 deaths were within 48h")
    print(f"    death_hemorrhagic_missing_incl (bound)    : {n_death_hem_incl:,}"
          f"   [missingness rule costs "
          f"{fix2['deaths_lost_to_missingness_rule']:,} deaths]")
    print(f"    death_inhosp      (sensitivity)           : {n_death:,}")
    print(f"    death_ed          (sensitivity)           : {n_death_ed:,}")
    print(f"    deaths w/ missing finaldischarge_hrs      : "
          f"{fix2['deaths_with_missing_finaldischarge_days']:,}")
    print(f"    finaldischarge_hrs fill cohort / deaths   : {fd_hrs_fill}% / "
          f"{fix2['finaldischarge_hrs_fill_pct_deaths']}%   "
          f"(days field: {fd_fill}%)")
    _asc = fix2['ascertainment_by_split']
    print(f"    ASCERTAINMENT (FINDING 3): deaths missing timing "
          f"derivation={_asc['derivation_2019_2022']['deaths_missing_pct']}% vs "
          f"validation={_asc['validation_2023_2024']['deaths_missing_pct']}%  "
          f"[differential {_asc['differential_pp']:+.3f} pp -> script 12]")
    _fdm = fix2['finaldischarge_hrs_stats_deaths']['median']
    _fde = fix2['finaldischarge_hrs_median_ed_deaths']
    _fds = fix2['finaldischarge_hrs_median_survivors']
    _fmt = lambda v: "n/a" if v is None else format(v, "g")
    print(f"    median fd_hrs: deaths={_fmt(_fdm)}h  "
          f"ED-deaths={_fmt(_fde)}h  survivors={_fmt(_fds)}h  "
          f"corr(hrs,days)={fix2['finaldischarge_hrs_days_corr_deaths']}")
    print(f"    ED_DISPOSITION_CODE fill / ais_head fill  : "
          f"{fix2['ed_disposition_fill_pct']}% / "
          f"{fix2['ais_head_fill_pct']}%")
    print("  SENSITIVITY composites")
    print(f"    major_hemorrhage_event_headais            : {n_comp_headais:,} "
          f"({_p(n_comp_headais)}%)   [head-AIS half only]")
    print(f"    major_hemorrhage_event_v2def              : {n_comp_v2def:,} "
          f"({_p(n_comp_v2def)}%)   [v2 realized: 31,054]")
    print("-" * 70)
    print(f"  cc_anticoag (code 31)          : {n_cc_anticoag:,}")
    print(f"  cc_bleeding (code 4)           : {n_cc_bleeding:,}")
    print(f"  unstable_ring (S32.81*)        : {n_unstable_ring:,}")
    print(f"  sbp_unrecordable (SBP2==0)     : {n_sbp_unrec:,}")
    print(f"  MISSING-vitals mortality       : {miss_mort}%  "
          f"(v2 realized 16.26%)")
    print(f"  blood_4h fill (primary)        : {blood_fill}%")
    print(f"  plasma_4h / platelet_4h fill   : {plasma_fill}% / "
          f"{platelet_fill}%   (Fix 7; plan's 27.3% is a SUPERSEDED sampling "
          f"artifact -- true pool fill 77.55%)")
    for _c in ("plasma_4h", "platelet_4h"):
        _cs = fix1["fix7_plasma_platelet"][_c]
        print(f"    {_c:<12} units-like<=100={_cs['n_units_like_le100']:,} "
              f"({_cs['pct_units_like_of_nonzero']}%)  "
              f"ml-like>=200={_cs['n_ml_like_ge200']:,} "
              f"({_cs['pct_ml_like_of_nonzero']}%)  "
              f"gap={_cs['n_gap_101_199']:,}  bimodal="
              f"{_cs['bimodal_units_vs_ml']}")
    print(f"  HC-route (primary)             : "
          f"surgical_only={route_counts['surgical_only']:,}  "
          f"angio_only={route_counts['angio_only']:,}  "
          f"both={route_counts['both']:,}  none={route_counts['none']:,}")
    print(f"  primary_outcome                : {gates['primary_outcome']}")
    print(f"  cohort CSV columns             : {len(COHORT_COLUMNS)} "
          f"(v2 had 50; +{len(COHORT_COLUMNS) - 50} v3 columns)")
    print()
    print_fork_fidelity(ff_rows, ff_pass, skipped=(sample_n is not None))
    print("  Artifacts written:")
    for p in (cohort_path,
              os.path.join(DERIVED, "cohort_exclusions.json"),
              os.path.join(DERIVED, "informative_missingness.csv"),
              os.path.join(LOGS, "cohort_flow.txt"),
              os.path.join(LOGS, "missingness.csv"),
              os.path.join(LOGS, "qc_manifest.json"),
              os.path.join(LOGS, "unknown_hc_codes.csv"),
              os.path.join(LOGS, "fix1_transfusion_audit.md"),
              os.path.join(LOGS, "fix2_death_audit.md")):
        print(f"    - {p}")
    print(f"  Total runtime: {time.time() - t_start:.0f}s")
    print("=" * 70)
    if not gate_pass:
        print("\n*** GATE FAILED — a carried-forward v2 figure did not "
              "reproduce. v3 changed OUTCOME DEFINITIONS ONLY, so this means "
              "the FORK is defective, not that a result moved. STOP and "
              "investigate; do NOT proceed to 02. ***")
        sys.exit(2)


if __name__ == "__main__":
    main()
