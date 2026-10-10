"""
15_table1.py — Pelvic Hemorrhage Arrival Decision-Support Tool (v3): Table 1
baseline characteristics on the analytic cohort (`cohort_primary == 1`,
N = 239,736; NTDB/TQP 2019-2024).

Fork of PAT_TBI/v4/code/14_table1.py, re-specified for the pelvic study. Two
things change materially vs the source:
  1. TWO stratification panels are emitted side-by-side in one CSV:
       Panel A — by the PRIMARY outcome `major_hemorrhage_event` (no-event vs
                 event), with SMD + p-value.
       Panel B — by the pre-registered elderly x anticoagulation strata
                 {Young (<65), Elderly / no-AC, Elderly / AC}, with the omnibus
                 p-value and the max pairwise SMD (imbalance across the 3 strata).
  2. The variable BLOCKS are respecified for pelvic (demographics / physiology /
     injury / comorbid); all TBI-specific rows are dropped, and multi-level
     categoricals (race, payer, mechanism, transport, ACS level, age band) are
     expanded into one N(%) row per level under a header row carrying the
     omnibus chi-square p-value.

v3 changes (CONTRACTS §3/§4/§5.1/§6; SAP amendment D-1, D-2, D-3, D-4, D-5)
--------------------------------------------------------------------------
  * NEW "Outcomes" block. `major_hemorrhage_event` keeps its name but changes
    meaning, so Table 1 now prints the whole outcome family of CONTRACTS §4 —
    the primary composite, its two composite-level sensitivities
    (`_headais`, `_v2def`), the co-primary patient-state composite, the three
    composite components, the COMPLETE death ladder, the v2 mL-only transfusion
    rule, and the pre-specified secondary `mtp_equivalent`. A reader can
    therefore read every restriction's cost straight off Table 1.
    The death ladder is printed in full because each rung isolates one decision:
      restriction ladder   all-cause (rung 0) -> head-AIS-only (1a) ->
                           timing-only (1b) -> BOTH = `death_hemorrhagic`,
                           the PRIMARY (rung 2)
      recovery ladder      what happens to the in-hospital deaths carrying no
                           `finaldischarge_hrs`: day-fallback -> ED-inclusive ->
                           all-untimed-kept, the last being the UPPER BOUND of
                           the ascertainment bracket. Omitting the bounding arm
                           hides the cost of the missing-data rule, which is the
                           one thing the ladder exists to expose. The bracket's
                           two ends are COMPUTED and printed at run time (no
                           script may hard-code the realized gates, CONTRACTS
                           §3.1).
      day-rule crosswalk   `*_dayrule` — what the superseded calendar-day proxy
                           would have produced. Audit trail only, never primary.
    NOTE: for the three composite components, the Panel-A split is DEFINITIONAL
    (a component cannot be positive in the no-event stratum), so the Panel-A
    SMD/p for those rows describe the composite's construction, not an
    association. The Panel-B (elderly x anticoagulation) columns are the
    informative ones for those rows.
  * `sbp_unrecordable` is RE-LABELLED to "No signs of life on arrival
    (unrecordable systolic pressure)" (Fix 4 / D-4, CONTRACTS §5.1). It is a
    clinical finding, not a data-quality artifact, and the v2 wording is retired
    everywhere (verification gate G4f greps for it).
  * `unstable_ring` STAYS in Table 1. Fix 3 (D-2/D-3) removed it from the SCORE,
    not from the cohort description; it also remains in the Analysis-B
    adjustment set (CONTRACTS §6, deliberate asymmetry). Gate G3f asserts the
    retention — do NOT "propagate" Fix 3 into this file. A post-write assert
    below fails loud if the row ever disappears.

Input : v3/derived/pelvic_cohort_predecision.csv   (filtered to cohort_primary==1)
Output: v3/tables/table1_baseline.csv

CSV columns:
    Block, Variable, Overall,
    NoEvent, Event, SMD_event, P_event,                 # Panel A (outcome)
    Young, Elderly_noAC, Elderly_AC, SMD_strata, P_strata   # Panel B (elderly x AC)

Missing data: complete-case per variable (the Overall/group denominators are the
per-variable non-missing counts — the transparency denominator). Underscore
identifiers never leak into the Variable column (LABEL_MAP / level maps enforce
clinical prose; a post-write assert re-checks). Seed 42.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# §1 Canonical paths — v3 only
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = Path(config.DERIVED)
TABLES = Path(config.TABLES)

INPUT_CSV = DERIVED / "pelvic_cohort_predecision.csv"
OUTPUT_CSV = TABLES / "table1_baseline.csv"

OUTCOME_COL = "major_hemorrhage_event"
SEED = 42

# ---------------------------------------------------------------------------
# Clinical-prose label map (top-level variable names). Every identifier used in
# BLOCKS must appear here — render fails loud otherwise, so no underscore id
# leaks into the CSV.
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    # Demographics
    "age":                "Age, years",
    "age_band":           "Age category",
    "female":             "Female sex",
    "race_eth_cat":       "Race / ethnicity",
    "ethnicity_hispanic": "Hispanic ethnicity",
    "payment_cat":        "Primary payer",
    # Physiology at arrival
    "sbp2":               "Systolic blood pressure, mmHg",
    "pulse2":             "Heart rate, beats/min",
    "shock_index":        "Shock index (HR / SBP)",
    "gcs2":               "Glasgow Coma Scale, total",
    "motor2":             "Glasgow Coma Scale, motor component",
    "sbp_lt90":           "Systolic blood pressure < 90 mmHg",
    # Fix 4 / D-4 (CONTRACTS §5.1): clinical reading, NOT a data-quality label.
    # The v2 wording ("unrecordable (= 0)") is retired and gate G4f greps for it.
    "sbp_unrecordable":   "No signs of life on arrival (unrecordable systolic pressure)",
    # Injury characteristics
    "mechanism_cat":      "Mechanism of injury",
    "transport_cat":      "Mode of transport",
    "iss":                "Injury Severity Score",
    "ais_head":           "Head AIS severity",
    # RETAINED in v3 by design (CONTRACTS §6 / gate G3f): dropped from the score,
    # kept in the cohort description and in the Analysis-B adjustment set.
    "unstable_ring":      "Unstable pelvic ring (ICD-10 S32.81*)",
    "acs_level":          "ACS trauma-center verification level",
    "charlson":           "Charlson Comorbidity Index",
    # Comorbidities
    "cc_anticoag":        "Anticoagulant therapy (preexisting)",
    "cc_bleeding":        "Bleeding disorder (preexisting)",
    # ---- v3 outcome family (CONTRACTS §4; SAP amendment D-1, D-2, D-5) -------
    # Primary composite + composite-level sensitivities + co-primary anchor
    "major_hemorrhage_event":
        "Major hemorrhage event — primary composite (Panel A stratifier)",
    "major_hemorrhage_event_headais":
        "Major hemorrhage event under the head-AIS-only death arm "
        "(composite sensitivity)",
    "major_hemorrhage_event_v2def":
        "Major hemorrhage event, v2 definition — all-cause death, volume-only "
        "transfusion (fork-fidelity comparator)",
    "patient_state_composite":
        "Patient-state composite (hemorrhagic death or substantial transfusion)",
    # Composite components
    "received_hc":
        "Hemorrhage-control procedure, surgical or angiographic (composite component)",
    "death_hemorrhagic":
        "Hemorrhagic death — in-hospital, head AIS < 4, within 48 h (composite component)",
    "substantial_transfusion":
        "Substantial transfusion, ≥ 1000 mL or ≥ 4 U PRBC within 4 h (composite component)",
    # Death RESTRICTION ladder (each restriction separable; CONTRACTS §4)
    "death_hemorrhagic_headais":
        "Death with head AIS < 4 (head-AIS restriction only)",
    "death_le48h":
        "Death within 48 h of arrival (timing restriction only)",
    "death_inhosp":
        "In-hospital death, all cause (v2 definition)",
    # Death RECOVERY ladder — the in-hospital deaths with no elapsed-hours
    # value. The primary rule EXCLUDES them (conservative); these three rungs
    # bracket what that rule costs, the last being the upper bound. Reporting
    # the bounding arm is the whole point of the ladder (CONTRACTS §3, §4).
    "death_hemorrhagic_dayfallback":
        "Hemorrhagic death, untimed deaths recovered by calendar day ≤ 2 "
        "(recovery rung 2)",
    "death_hemorrhagic_edincl":
        "Hemorrhagic death, untimed deaths recovered if the death occurred in the "
        "emergency department (recovery rung 3)",
    "death_hemorrhagic_missing_incl":
        "Hemorrhagic death, every untimed death retained — UPPER BOUND of the "
        "ascertainment bracket (recovery rung 4)",
    # Superseded calendar-day proxy, retained as an auditable crosswalk only.
    # The primary timing rule is elapsed hours; these rows exist so the
    # day-versus-hour decision can be checked rather than taken on trust.
    "death_hemorrhagic_dayrule":
        "Hemorrhagic death by the superseded calendar-day proxy (head AIS < 4, "
        "calendar day ≤ 2) — crosswalk only",
    "death_le48h_dayrule":
        "Early death by the superseded calendar-day proxy (calendar day ≤ 2, no "
        "head-AIS restriction) — crosswalk only",
    "death_ed":
        "Death in the emergency department",
    # Transfusion sensitivity + pre-specified secondary outcome
    "substantial_transfusion_ml_only":
        "Substantial transfusion, volume rule only (≥ 1000 mL within 4 h)",
    "mtp_equivalent":
        "MTP-equivalent activation, ≥ 4 U or ≥ 1000 mL PRBC within 4 h "
        "(identical by definition to substantial transfusion)",
}

# Outcome rows whose Panel-A split is definitional (a component of the primary
# composite cannot be positive in the no-event stratum; a strict SUPERSET of the
# composite is necessarily 100 % positive in the event stratum). Reported for
# transparency; the Panel-B (elderly x anticoagulation) columns carry the
# informative contrast for these rows.
DEFINITIONAL_IN_PANEL_A = {
    "major_hemorrhage_event", "received_hc", "death_hemorrhagic",
    "substantial_transfusion", "mtp_equivalent", "patient_state_composite",
    # superset of the primary composite (its death arm is the rung-1a superset
    # of death_hemorrhagic), so its Event column is 100 % by construction.
    "major_hemorrhage_event_headais",
}

# Ordered (raw_value, display_label) levels for each multi-level categorical.
# The raw_value must match the cohort CSV verbatim; display_label is prose.
CATEGORICALS: dict[str, list[tuple[str, str]]] = {
    "age_band": [
        ("18-44", "18-44 years"),
        ("45-64", "45-64 years"),
        ("65-74", "65-74 years"),
        (">=75",  "≥ 75 years"),
    ],
    "race_eth_cat": [
        ("Non-Hispanic White", "Non-Hispanic White"),
        ("NH Black",           "Non-Hispanic Black"),
        ("Hispanic",           "Hispanic"),
        ("Asian",              "Asian"),
        ("Other/Unknown",      "Other / unknown"),
    ],
    "payment_cat": [
        ("Medicare",        "Medicare"),
        ("Commercial",      "Commercial / private"),
        ("Medicaid/Agency", "Medicaid / government"),
        ("Self-pay",        "Self-pay / uninsured"),
        ("Other/Unknown",   "Other / unknown"),
    ],
    "mechanism_cat": [
        ("Fall",         "Fall"),
        ("MVC-occupant", "Motor vehicle crash (occupant)"),
        ("Motorcycle",   "Motorcycle"),
        ("Pedestrian",   "Pedestrian struck"),
        ("Other blunt",  "Other blunt"),
    ],
    "transport_cat": [
        ("Ground", "Ground EMS"),
        ("Air",    "Air transport"),
        ("Other",  "Other / unknown"),
    ],
    "acs_level": [
        ("1",       "ACS Level I"),
        ("2",       "ACS Level II"),
        ("3",       "ACS Level III"),
        ("Unknown", "Verification level not recorded"),
    ],
}

# Decimal places for continuous summaries (median/IQR default 0; mean/SD default 1).
MEDIAN_DECIMALS: dict[str, int] = {"shock_index": 2}
MEAN_DECIMALS: dict[str, int] = {"age": 1}

# ---------------------------------------------------------------------------
# Variable layout — pelvic BLOCKS (TBI-specific rows removed).
#   "binary"        — 0/1 indicator, report N (%)
#   "cont_mean_sd"  — continuous, mean (SD) + t-test / ANOVA
#   "cont_med_iqr"  — continuous, median (IQR) + Mann-Whitney / Kruskal-Wallis
#   "categorical"   — multi-level; header row (omnibus chi-square) + one N(%) row
#                     per level (from CATEGORICALS)
# ---------------------------------------------------------------------------
BLOCKS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Demographics", [
        ("age",                "cont_mean_sd"),
        ("age_band",           "categorical"),
        ("female",             "binary"),
        ("race_eth_cat",       "categorical"),
        ("ethnicity_hispanic", "binary"),
        ("payment_cat",        "categorical"),
    ]),
    ("Physiology at arrival", [
        ("sbp2",             "cont_med_iqr"),
        ("pulse2",           "cont_med_iqr"),
        ("shock_index",      "cont_med_iqr"),
        ("gcs2",             "cont_med_iqr"),
        ("motor2",           "cont_med_iqr"),
        ("sbp_lt90",         "binary"),
        ("sbp_unrecordable", "binary"),
    ]),
    ("Injury characteristics", [
        ("mechanism_cat", "categorical"),
        ("transport_cat", "categorical"),
        ("iss",           "cont_med_iqr"),
        ("ais_head",      "cont_med_iqr"),
        # G3f: unstable_ring is RETAINED here. Fix 3 removed it from the score's
        # candidate pool, not from the cohort description (CONTRACTS §6).
        ("unstable_ring", "binary"),
        ("acs_level",     "categorical"),
        ("charlson",      "cont_med_iqr"),
    ]),
    ("Comorbidities", [
        ("cc_anticoag", "binary"),
        ("cc_bleeding", "binary"),
    ]),
    # v3 outcome family — primary composite, its two composite-level
    # sensitivities, the co-primary anchor, the three components, and the
    # COMPLETE death / transfusion ladder of CONTRACTS §4, in the order a reader
    # needs to follow each restriction and each missing-data decision.
    ("Outcomes", [
        ("major_hemorrhage_event",           "binary"),  # PRIMARY composite
        ("major_hemorrhage_event_headais",   "binary"),  # composite sensitivity
        ("major_hemorrhage_event_v2def",     "binary"),  # v2 primary (fork fidelity)
        ("patient_state_composite",          "binary"),  # co-primary (HC dropped)
        ("received_hc",                      "binary"),  # component 1
        ("death_hemorrhagic",                "binary"),  # component 2 / anchor, rung 2
        ("substantial_transfusion",          "binary"),  # component 3
        # --- death RESTRICTION ladder (isolates each of the two restrictions) --
        ("death_hemorrhagic_headais",        "binary"),  # rung 1a (head AIS only)
        ("death_le48h",                      "binary"),  # rung 1b (timing only)
        ("death_inhosp",                     "binary"),  # rung 0 (v2, all cause)
        # --- death RECOVERY ladder (what the missing-timing rule costs) --------
        ("death_hemorrhagic_dayfallback",    "binary"),  # recovery rung 2
        ("death_hemorrhagic_edincl",         "binary"),  # recovery rung 3
        ("death_hemorrhagic_missing_incl",   "binary"),  # recovery rung 4 = UPPER BOUND
        # --- superseded calendar-day crosswalk (audit trail, never primary) ----
        ("death_hemorrhagic_dayrule",        "binary"),
        ("death_le48h_dayrule",              "binary"),
        ("death_ed",                         "binary"),  # ED-death surrogate
        # --- transfusion sensitivity + pre-specified secondary -----------------
        ("substantial_transfusion_ml_only",  "binary"),  # v2 transfusion rule
        ("mtp_equivalent",                   "binary"),  # secondary (D-5)
    ]),
]

# The ascertainment bracket the recovery ladder exists to quantify: the primary
# death count and its upper bound. Reported as a derived line in the run log so
# the bracket is visible without opening logs/fix2_death_audit.md. Both counts
# are COMPUTED from the cohort — nothing here is a literal (CONTRACTS §3.1: no
# script may hard-code the realized outcome gates).
BRACKET_LOWER_COL = "death_hemorrhagic"                  # rung 2, PRIMARY
BRACKET_UPPER_COL = "death_hemorrhagic_missing_incl"     # rung 4, upper bound

# CONTRACTS §4 death family, verbatim. A pre-flight check asserts every member
# reaches the Outcomes block, so a rung cannot be dropped from the ladder by a
# future edit without the script failing loudly. The ladder is only informative
# if it is COMPLETE: each row isolates exactly one restriction or one
# missing-data decision, and the bounding arm quantifies what the rule costs.
CONTRACT_DEATH_FAMILY = (
    "death_hemorrhagic",                 # rung 2 — PRIMARY
    "death_hemorrhagic_headais",         # rung 1a
    "death_le48h",                       # rung 1b
    "death_inhosp",                      # rung 0
    "death_hemorrhagic_dayfallback",     # recovery rung 2
    "death_hemorrhagic_edincl",          # recovery rung 3
    "death_hemorrhagic_missing_incl",    # recovery rung 4 — upper bound
    "death_hemorrhagic_dayrule",         # superseded day-rule crosswalk
    "death_le48h_dayrule",               # superseded day-rule crosswalk
    "death_ed",                          # ED-death surrogate
)

# Output column order (Panel A = outcome; Panel B = elderly x AC strata).
OUT_COLUMNS = [
    "Block", "Variable", "Overall",
    "NoEvent", "Event", "SMD_event", "P_event",
    "Young", "Elderly_noAC", "Elderly_AC", "SMD_strata", "P_strata",
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_count_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return "NA"
    return f"{int(n):,} ({100.0 * n / denom:.1f}%)"


def fmt_mean_sd(mean: float, sd: float, dec: int) -> str:
    if np.isnan(mean):
        return "NA"
    return f"{mean:.{dec}f} ({sd:.{dec}f})"


def fmt_med_iqr(med: float, q1: float, q3: float, dec: int) -> str:
    if np.isnan(med):
        return "NA"
    return f"{med:.{dec}f} ({q1:.{dec}f}-{q3:.{dec}f})"


# ---------------------------------------------------------------------------
# SMD
# ---------------------------------------------------------------------------
def smd_binary(p1: float, p0: float) -> float:
    """Austin/Flury-Riedwyl SMD for two proportions."""
    if np.isnan(p1) or np.isnan(p0):
        return np.nan
    denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2.0)
    return 0.0 if denom == 0 else (p1 - p0) / denom


def smd_continuous(m1: float, m0: float, s1: float, s0: float) -> float:
    if np.isnan(m1) or np.isnan(m0):
        return np.nan
    denom = np.sqrt((s1 ** 2 + s0 ** 2) / 2.0)
    return 0.0 if denom == 0 else (m1 - m0) / denom


def max_pairwise(values: list[float], smd_fn) -> float:
    """Max absolute pairwise SMD across a list of (stat-tuples); smd_fn(a,b)->smd."""
    out = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            s = smd_fn(values[i], values[j])
            if not np.isnan(s):
                out = max(out, abs(s))
    return out


# ---------------------------------------------------------------------------
# Panel analyzers — each returns per-group formatted cells + SMD + p-value for
# a single stratification (a dict {group_label: boolean mask over df}).
# ---------------------------------------------------------------------------
def _binary_panel(full_s: pd.Series, groups: dict[str, pd.Series]) -> dict:
    """Binary var across k groups → per-group N(%), omnibus chi-square, SMD."""
    per = {}
    props = []
    for lab, gm in groups.items():
        sub = full_s[gm]
        sub = sub[sub.notna()].astype(int)
        n1, d = int(sub.sum()), int(len(sub))
        per[lab] = fmt_count_pct(n1, d)
        props.append(n1 / d if d else np.nan)
    # omnibus chi-square (2 x k)
    counts = []
    for lab, gm in groups.items():
        sub = full_s[gm]
        sub = sub[sub.notna()].astype(int)
        counts.append((int(sub.sum()), int(len(sub))))
    table = np.array([[c[0] for c in counts], [c[1] - c[0] for c in counts]])
    p = _chi2_p(table)
    # SMD: 2 groups → signed (2nd vs 1st); >2 → max pairwise
    if len(props) == 2:
        smd = smd_binary(props[1], props[0])
    else:
        smd = max_pairwise(props, smd_binary)
    return {"cells": per, "smd": smd, "p": p}


def _cont_panel(full_s: pd.Series, groups: dict[str, pd.Series], kind: str,
                dec_med: int, dec_mean: int) -> dict:
    """Continuous var across k groups → per-group summary, test, SMD."""
    data = {lab: full_s[gm].dropna().astype(float) for lab, gm in groups.items()}
    per = {}
    stat_tuples = []  # (mean, sd) per group for SMD
    for lab, d in data.items():
        if len(d) == 0:
            per[lab] = "NA"
            stat_tuples.append((np.nan, np.nan))
            continue
        m, sd = float(d.mean()), float(d.std(ddof=1))
        stat_tuples.append((m, sd))
        if kind == "cont_mean_sd":
            per[lab] = fmt_mean_sd(m, sd, dec_mean)
        else:
            q1, med, q3 = np.quantile(d, [0.25, 0.5, 0.75])
            per[lab] = fmt_med_iqr(med, q1, q3, dec_med)
    arrs = [d.values for d in data.values() if len(d) > 0]
    p = _cont_p(arrs, kind)
    # SMD from mean/SD (standard Table 1 convention, incl. median-reported vars)
    if len(stat_tuples) == 2:
        (m1, s1), (m0, s0) = stat_tuples[1], stat_tuples[0]
        smd = smd_continuous(m1, m0, s1, s0)
    else:
        smd = max_pairwise(
            stat_tuples,
            lambda a, b: smd_continuous(a[0], b[0], a[1], b[1]),
        )
    return {"cells": per, "smd": smd, "p": p}


def _cat_panel(full_s: pd.Series, groups: dict[str, pd.Series],
               levels: list[tuple[str, str]]) -> dict:
    """Categorical var across k groups.

    Returns the omnibus chi-square p and, per level, the per-group N(%) plus a
    per-level SMD (2 groups → signed; >2 → max pairwise). Header carries p only.
    """
    lab_list = list(groups)
    # non-missing denominator per group
    denom = {lab: int(full_s[gm].notna().sum()) for lab, gm in groups.items()}
    # contingency (levels x groups) for omnibus chi-square
    table_rows = []
    level_cells: dict[str, dict[str, str]] = {}
    level_smd: dict[str, float] = {}
    for raw, _disp in levels:
        counts, props = [], []
        cells = {}
        for lab in lab_list:
            gm = groups[lab]
            sub = full_s[gm]
            n = int((sub == raw).sum())
            counts.append(n)
            cells[lab] = fmt_count_pct(n, denom[lab])
            props.append(n / denom[lab] if denom[lab] else np.nan)
        table_rows.append(counts)
        level_cells[raw] = cells
        level_smd[raw] = (smd_binary(props[1], props[0]) if len(props) == 2
                          else max_pairwise(props, smd_binary))
    table = np.array(table_rows)
    p = _chi2_p(table)
    return {"p": p, "level_cells": level_cells, "level_smd": level_smd,
            "denom": denom}


def _chi2_p(table: np.ndarray) -> float:
    try:
        table = table[~np.all(table == 0, axis=1)]  # drop all-zero rows
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan
        _, p, _, _ = stats.chi2_contingency(table, correction=False)
        return float(p)
    except (ValueError, ZeroDivisionError):
        return np.nan


def _cont_p(arrs: list[np.ndarray], kind: str) -> float:
    arrs = [a for a in arrs if len(a) > 0]
    if len(arrs) < 2:
        return np.nan
    try:
        if kind == "cont_mean_sd":
            if len(arrs) == 2:
                _, p = stats.ttest_ind(arrs[0], arrs[1], equal_var=False)
            else:
                _, p = stats.f_oneway(*arrs)
        else:  # cont_med_iqr
            if len(arrs) == 2:
                _, p = stats.mannwhitneyu(arrs[0], arrs[1], alternative="two-sided")
            else:
                _, p = stats.kruskal(*arrs)
        return float(p)
    except (ValueError, ZeroDivisionError):
        return np.nan


# ---------------------------------------------------------------------------
# Row builders — combine Panel A (outcome) + Panel B (strata) into output rows.
# ---------------------------------------------------------------------------
def _overall_cell(full_s: pd.Series, kind: str, var: str) -> str:
    s = full_s.dropna()
    if kind == "binary":
        return fmt_count_pct(int(s.astype(int).sum()), int(len(s)))
    s = s.astype(float)
    if len(s) == 0:
        return "NA"
    if kind == "cont_mean_sd":
        return fmt_mean_sd(float(s.mean()), float(s.std(ddof=1)),
                           MEAN_DECIMALS.get(var, 1))
    q1, med, q3 = np.quantile(s, [0.25, 0.5, 0.75])
    return fmt_med_iqr(med, q1, q3, MEDIAN_DECIMALS.get(var, 0))


def build_rows(df: pd.DataFrame,
               panelA: dict[str, pd.Series],
               panelB: dict[str, pd.Series]) -> tuple[list[dict], set[str]]:
    """Emit output rows for all BLOCKS; return (rows, set-of-top-level-labels)."""
    rows: list[dict] = []
    top_labels: set[str] = set()
    a_labels = list(panelA)  # ["NoEvent", "Event"]
    b_labels = list(panelB)  # ["Young", "Elderly_noAC", "Elderly_AC"]

    def blank_panel_cells() -> dict:
        d = {"NoEvent": "", "Event": "", "SMD_event": "", "P_event": "",
             "Young": "", "Elderly_noAC": "", "Elderly_AC": "",
             "SMD_strata": "", "P_strata": ""}
        return d

    for block_name, var_list in BLOCKS:
        for var, kind in var_list:
            pretty = label_for(var)
            top_labels.add(pretty)
            if var not in df.columns:
                print(f"[15_table1] WARNING: {var} missing; filling NA row.",
                      file=sys.stderr)
                row = {"Block": block_name, "Variable": pretty, "Overall": "NA"}
                row.update(blank_panel_cells())
                for c in ("NoEvent", "Event", "SMD_event", "P_event",
                          "Young", "Elderly_noAC", "Elderly_AC",
                          "SMD_strata", "P_strata"):
                    row[c] = "NA"
                rows.append(row)
                continue

            full_s = df[var]

            if kind == "categorical":
                levels = CATEGORICALS[var]
                _check_level_coverage(var, full_s, levels)
                a = _cat_panel(full_s, panelA, levels)
                b = _cat_panel(full_s, panelB, levels)
                # header row: omnibus p only
                hdr = {"Block": block_name, "Variable": pretty, "Overall": ""}
                hdr.update(blank_panel_cells())
                hdr["P_event"] = fmt_p(a["p"])
                hdr["P_strata"] = fmt_p(b["p"])
                rows.append(hdr)
                # level rows
                for raw, disp in levels:
                    denomA = int(full_s.notna().sum())
                    n_overall = int((full_s == raw).sum())
                    lr = {"Block": block_name, "Variable": f"  {disp}",
                          "Overall": fmt_count_pct(n_overall, denomA)}
                    lr.update(blank_panel_cells())
                    for lab in a_labels:
                        lr[lab] = a["level_cells"][raw][lab]
                    lr["SMD_event"] = f"{a['level_smd'][raw]:.3f}"
                    for lab in b_labels:
                        lr[lab] = b["level_cells"][raw][lab]
                    lr["SMD_strata"] = f"{b['level_smd'][raw]:.3f}"
                    rows.append(lr)
                continue

            # binary / continuous single-row variables
            if kind == "binary":
                a = _binary_panel(full_s, panelA)
                b = _binary_panel(full_s, panelB)
            else:
                a = _cont_panel(full_s, panelA, kind,
                                MEDIAN_DECIMALS.get(var, 0),
                                MEAN_DECIMALS.get(var, 1))
                b = _cont_panel(full_s, panelB, kind,
                                MEDIAN_DECIMALS.get(var, 0),
                                MEAN_DECIMALS.get(var, 1))
            row = {"Block": block_name, "Variable": pretty,
                   "Overall": _overall_cell(full_s, kind, var)}
            row.update(blank_panel_cells())
            for lab in a_labels:
                row[lab] = a["cells"][lab]
            row["SMD_event"] = f"{a['smd']:.3f}" if not np.isnan(a["smd"]) else "NA"
            row["P_event"] = fmt_p(a["p"])
            for lab in b_labels:
                row[lab] = b["cells"][lab]
            row["SMD_strata"] = f"{b['smd']:.3f}" if not np.isnan(b["smd"]) else "NA"
            row["P_strata"] = fmt_p(b["p"])
            rows.append(row)

    return rows, top_labels


def _check_level_coverage(var: str, full_s: pd.Series,
                          levels: list[tuple[str, str]]) -> None:
    """Warn if any observed level of a categorical is not mapped (would drop it)."""
    observed = set(full_s.dropna().astype(str).unique())
    mapped = {raw for raw, _ in levels}
    unmapped = observed - mapped
    if unmapped:
        print(f"[15_table1] WARNING: {var} has unmapped level(s) {sorted(unmapped)} "
              f"— these rows will be omitted. Extend CATEGORICALS['{var}'].",
              file=sys.stderr)


def label_for(var: str) -> str:
    if var not in LABEL_MAP:
        raise KeyError(f"LABEL_MAP missing entry for '{var}'. "
                       f"Add a clinical-prose label before running.")
    return LABEL_MAP[var]


# ---------------------------------------------------------------------------
# Derived columns (age band, female) built on the analytic cohort.
# ---------------------------------------------------------------------------
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Female sex (preserve NaN from sex_male)
    df["female"] = np.where(df["sex_male"].isna(), np.nan,
                            1 - pd.to_numeric(df["sex_male"], errors="coerce"))
    # Age band (age has no missing in the analytic cohort; NTDB caps at 89)
    age = pd.to_numeric(df["age"], errors="coerce")
    band = pd.cut(age, bins=[18, 45, 65, 75, 200], right=False,
                  labels=["18-44", "45-64", "65-74", ">=75"])
    df["age_band"] = band.astype("object")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pelvic Table 1 (two panels).")
    parser.add_argument("--sample", type=int, default=None,
                        help="Row-downsample the analytic cohort to N rows "
                             "(seed 42) for a fast smoke test.")
    args = parser.parse_args()

    print(f"[15_table1] Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"[15_table1] CSV rows = {len(df):,}")

    # ANALYTIC-COHORT RULE (CONTRACTS §3): analytic = cohort_primary == 1.
    if "cohort_primary" not in df.columns:
        sys.exit("FATAL: cohort_primary column missing; cannot form analytic cohort.")
    df = df[df["cohort_primary"] == 1].copy()
    print(f"[15_table1] Analytic cohort (cohort_primary==1) = {len(df):,}")

    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).copy()
        print(f"[15_table1] --sample: downsampled to {len(df):,} rows (seed {SEED}).")

    if OUTCOME_COL not in df.columns:
        sys.exit(f"FATAL: outcome '{OUTCOME_COL}' not in cohort CSV.")

    df = add_derived(df)

    # ---- Pre-flight: every BLOCKS var must have a clinical-prose label ----
    missing_labels = [v for _, vl in BLOCKS for v, _ in vl if v not in LABEL_MAP]
    if missing_labels:
        sys.exit(f"FATAL: LABEL_MAP missing entries for: {missing_labels}.")
    missing_cat = [v for _, vl in BLOCKS for v, k in vl
                   if k == "categorical" and v not in CATEGORICALS]
    if missing_cat:
        sys.exit(f"FATAL: CATEGORICALS missing level maps for: {missing_cat}.")

    # ---- Pre-flight: the CONTRACTS §4 death ladder must be COMPLETE ----------
    # A rung silently dropped from BLOCKS would leave the reader unable to price
    # one of the restrictions (or, for the bounding arm, the missing-data rule).
    outcome_block_vars = [v for b, vl in BLOCKS if b == "Outcomes" for v, _ in vl]
    ladder_gap = [v for v in CONTRACT_DEATH_FAMILY if v not in outcome_block_vars]
    if ladder_gap:
        sys.exit(f"FATAL: the Outcomes block is missing CONTRACTS §4 death-family "
                 f"rung(s): {ladder_gap}. Every rung is reported so each "
                 f"restriction — and the cost of the missing-timing rule — is "
                 f"separable.")

    # ---- Stratification masks (aligned to df index) ----
    outcome = pd.to_numeric(df[OUTCOME_COL], errors="coerce").fillna(0).astype(int)
    n_events = int((outcome == 1).sum())
    print(f"[15_table1] Primary events (major_hemorrhage_event) = {n_events:,} "
          f"({100 * n_events / len(df):.2f}%)")

    panelA = {
        "NoEvent": (outcome == 0),
        "Event":   (outcome == 1),
    }
    age_ge65 = pd.to_numeric(df["age_ge65"], errors="coerce")
    ac = pd.to_numeric(df["cc_anticoag"], errors="coerce")
    panelB = {
        "Young":        (age_ge65 == 0),
        "Elderly_noAC": (age_ge65 == 1) & (ac == 0),
        "Elderly_AC":   (age_ge65 == 1) & (ac == 1),
    }
    print(f"[15_table1] Panel A (outcome): "
          f"NoEvent={int(panelA['NoEvent'].sum()):,}, Event={int(panelA['Event'].sum()):,}")
    print(f"[15_table1] Panel B (elderly x AC): "
          f"Young={int(panelB['Young'].sum()):,}, "
          f"Elderly/noAC={int(panelB['Elderly_noAC'].sum()):,}, "
          f"Elderly/AC={int(panelB['Elderly_AC'].sum()):,}")

    # ---- v3 outcome family: echo every rung so the run log carries the ladder
    #      and can be diffed against logs/fix2_death_audit.md + qc_manifest.json.
    print("[15_table1] v3 outcome family (analytic cohort):")
    for var in [v for b, vl in BLOCKS if b == "Outcomes" for v, _ in vl]:
        if var not in df.columns:
            print(f"           {var:<32} MISSING from cohort CSV")
            continue
        s = pd.to_numeric(df[var], errors="coerce")
        n1 = int((s == 1).sum())
        flag = " (Panel A split is definitional)" if var in DEFINITIONAL_IN_PANEL_A else ""
        print(f"           {var:<32} n = {n1:>7,}  "
              f"({100 * n1 / len(df):.3f}%){flag}")

    # Ascertainment bracket on the primary death component: rung 2 (PRIMARY,
    # untimed deaths excluded) to rung 4 (every untimed death retained). Both
    # ends are read off the cohort, never asserted against a literal.
    if BRACKET_LOWER_COL in df.columns and BRACKET_UPPER_COL in df.columns:
        lo = int((pd.to_numeric(df[BRACKET_LOWER_COL], errors="coerce") == 1).sum())
        hi = int((pd.to_numeric(df[BRACKET_UPPER_COL], errors="coerce") == 1).sum())
        print(f"[15_table1] Death-component ascertainment bracket: "
              f"{lo:,} (primary, untimed deaths excluded) -> {hi:,} (upper bound, "
              f"every untimed death retained); width = {hi - lo:,} deaths.")
        if hi < lo:
            print("[15_table1] WARNING: the upper-bound rung is SMALLER than the "
                  "primary rung — the recovery ladder must be monotone "
                  "(CONTRACTS §13 G2b-d). Check 01_build_cohort.py.",
                  file=sys.stderr)
    else:
        print("[15_table1] WARNING: cannot report the ascertainment bracket — "
              f"{BRACKET_LOWER_COL} and/or {BRACKET_UPPER_COL} absent from the "
              "cohort CSV.", file=sys.stderr)

    # ---- Build rows ----
    rows, top_labels = build_rows(df, panelA, panelB)
    out = pd.DataFrame(rows, columns=OUT_COLUMNS)

    # Panel A for the stratifier row itself is a tautology (0% vs 100%): both
    # group variances are zero, so the SMD and the chi-square are undefined
    # rather than "no imbalance". Write NA instead of a misleading 0.000.
    strat_label = label_for(OUTCOME_COL)
    is_strat = out["Variable"] == strat_label
    out.loc[is_strat, ["SMD_event", "P_event"]] = "NA"

    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[15_table1] Wrote {OUTPUT_CSV} ({len(out)} rows)")

    # ---- Post-write verification ----
    verify = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
    n_vars_defined = sum(len(v) for _, v in BLOCKS)

    # (1) every defined top-level variable label appears (header or single row)
    var_labels = {label_for(v) for _, vl in BLOCKS for v, _ in vl}
    assert len(var_labels) == n_vars_defined, (
        f"duplicate LABEL_MAP prose across {n_vars_defined} defined vars")
    emitted = set(verify["Variable"])
    missing_top = sorted(l for l in var_labels if l not in emitted)
    assert not missing_top, f"defined variables missing from output: {missing_top}"

    # (2) no raw underscore identifier leaked into the Variable column
    raw_ids = {v for _, vl in BLOCKS for v, _ in vl}
    leaked = sorted(set(verify["Variable"]) & raw_ids)
    assert not leaked, f"raw identifiers leaked into Variable column: {leaked}"

    # (3) row count = (#binary + #cont) + sum over categoricals of (1 header + #levels)
    n_expected = 0
    for _, vl in BLOCKS:
        for v, k in vl:
            n_expected += (1 + len(CATEGORICALS[v])) if k == "categorical" else 1
    assert len(verify) == n_expected, (
        f"row count {len(verify)} != expected {n_expected}")

    # (4) strata construction validity: for the anticoagulant row, the
    #     elderly/AC stratum must be 100% anticoagulated and elderly/no-AC 0%
    #     (they are defined by cc_anticoag), confirming the Panel B masks.
    ac_row = verify[verify["Variable"] == LABEL_MAP["cc_anticoag"]]
    if not ac_row.empty:
        r = ac_row.iloc[0]
        assert r["Elderly_AC"].endswith("(100.0%)"), (
            f"Elderly/AC stratum should be 100% anticoagulated; got {r['Elderly_AC']}")
        assert r["Elderly_noAC"].startswith("0 "), (
            f"Elderly/no-AC stratum should be 0% anticoagulated; got {r['Elderly_noAC']}")

    # (5) GATE G3f — unstable_ring is RETAINED in Table 1 (CONTRACTS §6).
    #     Fix 3 removed it from the score, not from the cohort description. This
    #     assert exists so a well-meaning "propagate Fix 3 everywhere" edit fails
    #     loudly instead of silently deleting a baseline row.
    assert LABEL_MAP["unstable_ring"] in emitted, (
        "G3f VIOLATION: the unstable-pelvic-ring row is missing from Table 1. "
        "Fix 3 removes unstable_ring from the SCORE only; it stays in Table 1 "
        "and in the Analysis-B adjustment set (CONTRACTS §6).")

    # (6) GATE G4f — the retired SBP-as-artifact wording must not appear.
    #     The pattern is built with \s* rather than literal spaces so that the
    #     gate's own grep over this source file does not flag this check.
    banned_re = re.compile(r"sbp\s*unrecordable|sbp\s*=\s*0|unrecordable\s*\(\s*=\s*0",
                           re.IGNORECASE)
    for lab in emitted:
        assert not banned_re.search(str(lab)), (
            f"G4f VIOLATION: Table 1 row label {lab!r} carries the retired "
            f"SBP-as-artifact wording. The approved display string is "
            f"{LABEL_MAP['sbp_unrecordable']!r}.")
    assert LABEL_MAP["sbp_unrecordable"] in emitted, (
        "G4f: the no-signs-of-life row is missing from Table 1.")

    # (7) The v3 outcome family is present in full (D-1/D-2/D-5 ladder).
    outcome_vars = [v for b, vl in BLOCKS if b == "Outcomes" for v, _ in vl]
    missing_out = [v for v in outcome_vars if label_for(v) not in emitted]
    assert not missing_out, f"outcome-family rows missing from Table 1: {missing_out}"

    # (7b) …and the CONTRACTS §4 death ladder specifically, including the
    #      bounding arm. The ladder's purpose is to let a reader see the effect
    #      of each restriction AND of the missing-timing rule; the upper-bound
    #      rung is the only row that prices the rule, so its absence is a FAIL.
    missing_ladder = [v for v in CONTRACT_DEATH_FAMILY
                      if label_for(v) not in emitted]
    assert not missing_ladder, (
        f"CONTRACTS §4 death-ladder rung(s) missing from Table 1: "
        f"{missing_ladder}")

    print(f"[15_table1] Post-write verification OK "
          f"({len(verify)} rows; {n_vars_defined} defined variables, "
          f"categoricals expanded).")
    print(f"[15_table1] G3f OK — unstable pelvic ring retained in Table 1.")
    print(f"[15_table1] G4f OK — SBP row rendered as "
          f"'{LABEL_MAP['sbp_unrecordable']}'.")
    print(f"[15_table1] Outcome family OK — {len(outcome_vars)} rows "
          f"(primary composite + 2 composite sensitivities, co-primary, "
          f"3 components, the complete {len(CONTRACT_DEATH_FAMILY)}-rung death "
          f"ladder incl. the recovery/bounding arms and the superseded day-rule "
          f"crosswalk, transfusion sensitivity, MTP-equivalent).")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[15_table1] ASSERTION FAILED: {e}", file=sys.stderr)
        raise
