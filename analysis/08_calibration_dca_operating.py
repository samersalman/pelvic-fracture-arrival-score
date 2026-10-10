"""
Pelvic Arrival Tool v3 — Step 08: Calibration / DCA / operating characteristics
================================================================================

Forked from PAT_TBI/v4/code/08_calibration_dca_operating.py. Per CONTRACTS §8 the
shared render_figures.py (Batch 5c) owns publication figures F4/F5/F9, so the
INLINE figure rendering is DISABLED here — this script emits only DATA artifacts.

Scores (risk-oriented, higher = worse): pelvic_score, SI-alone (shock_index),
rSIG (rsig, negated). The 4 TBI comparators are dropped.

v3 CHANGES
----------
**Fix 5 (D3) — the centrepiece of this revision.** Reviewer comment:
  "Going from 0.849 to 0.876 is statistically significant in a cohort this large,
   but our dataset has 240,000 patients!! What matters is whether the score
   meaningfully changes what we do. Can we report sensitivity, specificity,
   NPV/PPV at each threshold?"
Two new tables answer him:
  * `tables/table5c_operating_full.csv` — sens / spec / PPV / NPV (all with 95%
    Wilson CIs) / LR± / number-needed-to-escalate at EVERY integer score cutoff,
    for the score, the shock index and rSIG side by side, and for BOTH the
    score-only rule and the deployed score-OR-override rule.
  * `tables/table5d_matched_workload.csv` — the clinical argument. Hold the
    fraction of the cohort flagged (the WORKLOAD) approximately constant and
    compare how many events each instrument catches, with explicit
    `Delta_events_caught` / `Delta_false_positives` columns against SI-alone.
    Reported overall AND within the elderly (age >= 65) stratum.
`tables/table5_operating.csv` is UNCHANGED in schema (sens-90 rule-out /
spec-90 rule-in / Youden) — 05's downstream consumers and the v2 comparison
depend on it. These are additions, not replacements.

**Fix 4 (D6 + W2) — `sbp_unrecordable` is an automatic-escalate override**
(CONTRACTS §5.1). The deployed decision rule is
    escalate = (pelvic_score >= c_rule_in) OR (override_nslo == 1)
`score_card.json` carries the override in its own top-level `"override"` block —
NOT in `variables[]`, which holds point rows only. `risk_tiers.csv` gains a
separate `override` row. `score_range` is the OBSERVED min/max of `pelvic_score`
in the v3 test set; the theoretical column sums are reported separately and
explicitly labelled (CONTRACTS §13 G4c draws exactly this distinction).

**DCA prevalence anchor** is read at run time from `logs/qc_manifest.json`
(the realised v3 composite rate). The v2 anchor 0.1295 is superseded.

v3 BATCH-7 DEFECT FIXES (spec-compliance review)
------------------------------------------------
1. **`refit.json["theoretical_point_sum_unconstrained"]` is a two-element
   `[min, max]` list** (written by `05_refit_integer.py:509`), not a scalar.
   `int()` on it raised `TypeError` and killed the script *after* the tables
   were written but *before* `derived/score_card.json`, cascading into `10`
   (Table 6/6B, elderly DCA, `reclass.json`), F9's override banner and the WTA
   figure. Parsing now happens on the READING side, in
   `parse_theoretical_point_sum()`, matching the handling already in
   `render_figures.py:351`. `05` is NOT modified — another contract depends on
   the two-element form and this script does not own that file.
2. **Gate G4c actually evaluates.** The same type mismatch meant the
   "unconstrained point sum agrees with 05" cross-check sat inside
   `except (TypeError, ValueError): pass` and never ran — neither its `[WARN]`
   nor its `ASSERT pass` line was reachable. CONTRACTS §13: *"A gate that
   cannot be evaluated is a FAIL, not a skip."* The comparison now runs, an
   unexpected type is a hard `SystemExit` at read time (before the long DCA),
   a missing key is reported as a gate FAILURE, and the verdict is recorded in
   `score_card.json["score_range_theoretical"]["g4c_unconstrained_sum_check"]`.
3. **Table 5D is genuinely workload-matched.** Previously each instrument
   independently chased the NOMINAL workload level: the shock index (0.01 grid)
   landed within ~0.1 pp while the integer-valued `pelvic_score` could land 3+
   pp away — and the deltas were still computed against the shock-index row.
   The error ran in the score's favour (extra events bought with extra
   escalations were booked as an equal-workload gain) and produced the visible
   absurdity of `score_or_override` appearing to *reduce* benefit. T5D now
   uses the algorithm T5C already used: each block has ONE anchor instrument,
   the anchor's ACHIEVED flagged fraction is the operative workload target, and
   every other instrument is matched to that achieved fraction. `fixed_*pct`
   and `deployed_rule_in` are score-anchored; `si_clinical_*` are shock-index
   anchored (the matching direction reverses and the anchor is labelled). A
   delta is published ONLY when the two compared rows' achieved flagged
   fractions agree within `workload_tolerance_pct()`; otherwise the delta
   columns are NA and `Delta_flag` says why. Every row carries both its own and
   the reference row's achieved fraction so the match is reader-auditable.
4. **Finding 15.** `LR_pos` returned `inf` when specificity was NaN in a
   degenerate stratum; `inf` reads as "infinitely informative". It is now NaN,
   and `inf` is reserved for its one legitimate meaning (specificity == 1).

Outputs
-------
  v3/tables/table5_operating.csv          (schema UNCHANGED)
  v3/tables/table5c_operating_full.csv    (NEW — Fix 5)
  v3/tables/table5d_matched_workload.csv  (NEW — Fix 5)
  v3/derived/dca_data.csv                 (fig-data F5, OVERALL only)
  v3/derived/calibration_data.csv         (fig-data F4, OVERALL only)
  v3/derived/score_card.json              (fig-data F9, + "override" block)
  v3/derived/risk_tiers.csv               (fig-data F9, + "override" row)

Run
---
    python 08_calibration_dca_operating.py
    python 08_calibration_dca_operating.py --sample 50000   # smoke

Seed: 42
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
EPS = 1e-6

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES
LOGS = config.LOGS
os.makedirs(TABLES, exist_ok=True)

IN_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")
POINTS_CSV = os.path.join(TABLES, "table3_points.csv")
REFIT_JSON = os.path.join(DERIVED, "refit.json")
QC_MANIFEST = os.path.join(LOGS, "qc_manifest.json")
OUT_OPCHAR = os.path.join(TABLES, "table5_operating.csv")
OUT_T5C = os.path.join(TABLES, "table5c_operating_full.csv")
OUT_T5D = os.path.join(TABLES, "table5d_matched_workload.csv")
OUT_DCA = os.path.join(DERIVED, "dca_data.csv")
OUT_CALIB = os.path.join(DERIVED, "calibration_data.csv")
OUT_CARD = os.path.join(DERIVED, "score_card.json")
OUT_TIERS = os.path.join(DERIVED, "risk_tiers.csv")

OUTCOME_COL = "major_hemorrhage_event"
# Risk-tier sub-rates. v3-J: the death component of the primary composite is
# `death_hemorrhagic`, NOT `death_inhosp`. `death_inhosp` is reported alongside
# as the named v2 sensitivity.
DEATH_COMPONENT = "death_hemorrhagic"
DEATH_SENSITIVITY = "death_inhosp"
TX_COMPONENT = "substantial_transfusion"
HC_COMPONENT = "received_hc"

OVERRIDE_COL = "override_nslo"
OVERRIDE_VAR = "sbp_unrecordable"
# Fallbacks only. The canonical display strings are read at run time from
# refit.json (`override_labels` / `override_rule_text`) so that this script never
# invents a label — CONTRACTS §13 G4f greps tables and figure label sources for
# stray "SBP = 0" / "SBP unrecordable" forms.
OVERRIDE_LABEL_FALLBACK = "No signs of life on arrival (unrecordable systolic pressure)"
OVERRIDE_RULE_TEXT_FALLBACK = ("Automatic escalation regardless of score: no signs of "
                               "life on arrival (unrecordable systolic pressure)")

# DCA grid widened to p_t 0.02–0.40 (CONTRACTS §7 row 08).
THRESHOLDS = np.round(np.arange(0.02, 0.4001, 0.01), 4)
PREV_ANCHOR_V2_SUPERSEDED = 0.1295          # v2 value — must NOT be used in v3
V2_OBSERVED_SCORE_RANGE = (-7, 32)          # v2 comparison only (CONTRACTS §13 G4c)

# (label, score_col, sign) — higher(sign*score) == worse.
SCORE_SPECS = [
    ("pelvic_score", "pelvic_score", +1),
    ("SI-alone",     "shock_index",  +1),
    ("rSIG",         "rsig",         -1),
]

# ---- Fix 5 grids ----------------------------------------------------------
# Comparable-cutoff rule (stated in the manuscript Methods and in the T5C
# `Cutoff_Source` column):
#   `workload_matched` — for every integer pelvic_score cutoff we record the
#       fraction of the analysis set it flags, then pick the SI / rSIG cutoff
#       from a fine legible grid (SI: 0.01 steps; rSIG: 0.1 steps) whose flagged
#       fraction is closest to that same fraction. This makes the three
#       instruments span the SAME flagged-fraction range, which is what makes
#       the table readable side by side.
#   `clinical_anchor` — the canonical published cutoffs, reported regardless of
#       what fraction they happen to flag, so the familiar operating points are
#       still visible.
SI_GRID = np.round(np.arange(0.20, 4.0001, 0.01), 2)
RSIG_GRID = np.round(np.arange(0.0, 40.0001, 0.1), 1)
SI_CLINICAL_ANCHORS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
RSIG_CLINICAL_ANCHORS = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

# Matched-workload levels (fraction of the analysis set flagged), Fix 5 / T5D.
WORKLOAD_TARGETS_PCT = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
# Shock-index values that anchor a workload block in their own right (the
# familiar clinical alarms). For these blocks the matching direction REVERSES:
# the shock index sets the workload and the SCORE is matched to it.
SI_CLINICAL_WORKLOAD_ANCHORS = [0.9, 1.0]

# ---- T5D workload-match tolerance (Batch-7 fix 3) -------------------------
# A `Delta_events_caught` is an EQUAL-WORKLOAD statement only if the two rows
# being differenced really flag the same fraction of the stratum. Published
# deltas are gated at
#     tol_pct = min(WORKLOAD_MATCH_TOL_PCT, WORKLOAD_MATCH_TOL_REL * workload_pct)
# i.e. a quarter of a percentage point of the stratum, tightened to 10 % of the
# workload itself once the workload drops below 2.5 % (a 0.25 pp gap at a 1 %
# workload is a 25 % relative difference in escalations — not "equal workload").
#
# Why 0.25 pp:
#   * it ADMITS the archetype the SAP itself cites as the matched-workload claim
#     (SAP_amendment_v3.md D-6: score >=6 flagged 8.3 % vs shock index >=0.9 at
#     8.1 % — a 0.20 pp gap), so the tolerance does not outlaw the analysis it
#     is meant to police;
#   * it REJECTS the reproduced defect (23.23 % vs 19.69 % = 3.54 pp) by more
#     than fourteen-fold;
#   * on a test set of this size 0.25 pp is a couple of hundred patients, and
#     the exact residual is printed on every row (`Delta_workload_gap_pct` and
#     `Delta_workload_gap_n`) so a reader can bound its event contribution
#     directly rather than trusting the threshold.
WORKLOAD_MATCH_TOL_PCT = 0.25
WORKLOAD_MATCH_TOL_REL = 0.10

# T5D matches the comparators on the union of the legible grid above and the
# comparator's own observed distinct values. The legible 0.01 shock-index grid
# is coarse relative to the density of a trauma cohort near its mode — one step
# can move the flagged fraction by more than a percentage point, which is larger
# than the tolerance itself, so a legible-grid-only search would suppress
# essentially every delta for a reason that is an artifact of the grid rather
# than of the data. The union spans every ACHIEVABLE flagged fraction; legible
# values still win any tie, so cutoff labels stay round wherever roundness is
# free. (T5C is unaffected: it keeps the legible grids its documented
# `Cutoff_Source` rule describes.)
T5D_MATCH_ON_EMPIRICAL_VALUES = True

# Mutually-exclusive point groups, used ONLY for the theoretical column-sum
# report in score_card.json (CONTRACTS §13 G4c). NOTE the SBP ladder
# (sbp_lt90/sbp_lt70) and the SI ladder (si_ge0_9/si_ge1_0) are NESTED, not
# mutually exclusive — a patient with SBP 60 satisfies both — so both count
# toward the theoretical maximum. `mech_high_energy` co-occurs with `mech_mvc`
# and is likewise not in a mutex group.
MUTEX_GROUPS = {
    "age_band": ["age_65_74", "age_ge75"],
    "mechanism_exclusive": ["mech_fall", "mech_mvc"],
}


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def _flex_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def logistic_recalibrate(y, score):
    X = sm.add_constant(np.asarray(score, dtype=float), has_constant="add")
    m = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="lbfgs", maxiter=200)
    return np.clip(m.predict(X), EPS, 1 - EPS)


def net_benefit(y, prob, p_t):
    n = len(y)
    if n == 0 or p_t >= 1.0:
        return np.nan
    pred_pos = prob >= p_t
    tp = float((pred_pos & (y == 1)).sum()) / n
    fp = float((pred_pos & (y == 0)).sum()) / n
    return tp - fp * (p_t / (1.0 - p_t))


def nb_treat_all(prev, p_t):
    return np.nan if p_t >= 1.0 else prev - (1.0 - prev) * (p_t / (1.0 - p_t))


def wilson(count, nobs):
    if nobs == 0:
        return (np.nan, np.nan)
    return proportion_confint(count=count, nobs=nobs, alpha=0.05, method="wilson")


# ---------------------------------------------------------------------------
# Core 2x2 metrics
# ---------------------------------------------------------------------------
def metrics_from_flag(y, flag):
    """2x2 operating characteristics from a boolean flag vector.

    Number_needed_to_escalate = 1 / PPV  — the number of patients escalated per
    true event caught (Fix 5; NaN when nothing is flagged, inf when PPV == 0).
    """
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(bool)
    TP = int((flag & (y == 1)).sum())
    FP = int((flag & (y == 0)).sum())
    TN = int((~flag & (y == 0)).sum())
    FN = int((~flag & (y == 1)).sum())
    sens = TP / (TP + FN) if (TP + FN) else np.nan
    spec = TN / (TN + FP) if (TN + FP) else np.nan
    ppv = TP / (TP + FP) if (TP + FP) else np.nan
    npv = TN / (TN + FN) if (TN + FN) else np.nan
    # Finding 15: an UNDEFINED likelihood ratio is NaN, not inf. `inf` in a
    # published table reads as "infinitely informative"; it is reserved here for
    # its one legitimate meaning — a finite sensitivity against a specificity of
    # exactly 1. A degenerate stratum (no true negatives and no false positives,
    # so specificity is NaN) yields NaN.
    if not (np.isfinite(sens) and np.isfinite(spec)):
        lr_pos = np.nan
    elif (1.0 - spec) > 0:
        lr_pos = sens / (1.0 - spec)
    else:
        lr_pos = np.inf
    if not (np.isfinite(sens) and np.isfinite(spec)) or spec <= 0:
        lr_neg = np.nan
    else:
        lr_neg = (1.0 - sens) / spec
    n_ruledout = TN + FN
    missed_event_rate = FN / n_ruledout if n_ruledout else np.nan  # 1 - NPV
    if not np.isfinite(ppv):
        nne = np.nan
    elif ppv <= 0:
        nne = np.inf
    else:
        nne = 1.0 / ppv
    slo, shi = wilson(TP, TP + FN)
    splo, sphi = wilson(TN, TN + FP)
    plo, phi = wilson(TP, TP + FP)
    nlo, nhi = wilson(TN, TN + FN)
    return {
        "N": TP + FP + TN + FN, "N_flagged": TP + FP,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Sensitivity": sens, "Sens_CI_Low": slo, "Sens_CI_High": shi,
        "Specificity": spec, "Spec_CI_Low": splo, "Spec_CI_High": sphi,
        "PPV": ppv, "PPV_CI_Low": plo, "PPV_CI_High": phi,
        "NPV": npv, "NPV_CI_Low": nlo, "NPV_CI_High": nhi,
        "LR_pos": lr_pos, "LR_neg": lr_neg,
        "Number_needed_to_escalate": nne,
        "Missed_Event_Rate": missed_event_rate,
        "Youden_J": (sens + spec - 1.0) if (np.isfinite(sens) and np.isfinite(spec)) else np.nan,
    }


def metrics_at(y, score, cutoff):
    """Backward-compatible wrapper: flag = score >= cutoff.

    Returns the same key set the v2 `table5_operating.csv` was built from (plus
    `Number_needed_to_escalate`, which the unchanged `front` column list drops).
    """
    m = metrics_from_flag(y, np.asarray(score, dtype=float) >= cutoff)
    m["Cutoff"] = cutoff
    return m


def youden_cut(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def sens90_cut(y, s):
    """Rule-OUT cutoff: HIGHEST cutoff still achieving sensitivity >= 0.90 (i.e.
    max specificity / rules out the most patients while missing <=10% of events).
    roc thresholds are DESCENDING and tpr>=0.90 holds from its first qualifying
    index onward, so the highest qualifying threshold is ok.min(). (PAT's ok.max()
    returned the trivial sens=1.0/spec=0 extreme.)"""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(tpr >= 0.90)[0]
    if len(ok) == 0:
        return np.nan
    cut = float(thr[int(ok.min())])
    return cut if np.isfinite(cut) else np.nan


def spec90_cut(y, s):
    """Rule-IN cutoff: LOWEST cutoff still achieving specificity >= 0.90 (i.e. max
    sensitivity / catches the most events while keeping false alarms <=10%).
    spec>=0.90 holds up to its last qualifying index, so the lowest qualifying
    threshold is ok.max(). (PAT's ok.min() returned the trivial spec=1.0/sens=0
    extreme.)"""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where((1 - fpr) >= 0.90)[0]
    if len(ok) == 0:
        return np.nan
    cut = float(thr[int(ok.max())])
    return cut if np.isfinite(cut) else np.nan


def load_oriented(df, spec):
    label, col, sign = spec
    raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    s = sign * raw
    use = ~np.isnan(s)
    return {"label": label, "col": col, "sign": sign,
            "y": df[OUTCOME_COL].to_numpy()[use].astype(int),
            "s": s[use], "s_native": raw[use], "use": use, "n": int(use.sum())}


def read_prevalence_anchor():
    """DCA prevalence anchor = the realised v3 composite rate (logs/qc_manifest.json).

    The v2 anchor (0.1295) is superseded — the v3 composite has a different death
    component (Fix 2/2b) and a different transfusion rule (Fix 1).
    """
    try:
        with open(QC_MANIFEST) as f:
            man = json.load(f)
    except Exception as exc:
        print(f"  [WARN] could not read {QC_MANIFEST} ({exc})")
        return np.nan, "unavailable"
    pc = man.get("primary_composite", {}) or {}
    pct = pc.get("major_hemorrhage_event_rate_pct")
    if pct is not None:
        return float(pct) / 100.0, "qc_manifest.primary_composite.major_hemorrhage_event_rate_pct"
    n_ev = pc.get("major_hemorrhage_event_n")
    n_tot = man.get("n_primary")
    if n_ev is not None and n_tot:
        return float(n_ev) / float(n_tot), "qc_manifest.primary_composite n/n_primary"
    print(f"  [WARN] qc_manifest.json has no primary-composite rate")
    return np.nan, "unavailable"


def best_cutoff_for_target(values, direction, grid, target_frac,
                           extra_or=None, prefer=None):
    """Pick the grid cutoff whose flagged fraction is closest to `target_frac`.

    direction '>=' : flag = values >= c   (shock index, pelvic score)
    direction '<=' : flag = values <= c   (rSIG — LOWER is worse)
    `extra_or` optionally ORs in the automatic-escalate override.
    `prefer` is an optional set of cutoff values (rounded to 6 dp) that win ties
    at the SAME absolute difference — used so a legible round cutoff is chosen
    over an equally-good empirical one.

    Returns (cutoff, flag, achieved_fraction, abs_diff) or None.

    NOTE (Batch-7): the flagged fraction is now evaluated by `searchsorted` over
    the sorted values instead of one full boolean pass per grid point. This is
    an exact re-expression — NaNs never satisfy either comparison and stay in
    the denominator, and the first grid minimum still wins — but it turns an
    O(len(grid) x n) scan into O(n log n + len(grid) log n), which is what makes
    matching on the empirical value grid affordable.
    """
    values = np.asarray(values, dtype=np.float64)
    n_total = int(values.size)
    grid = np.asarray(grid, dtype=np.float64)
    if n_total == 0 or grid.size == 0:
        return None

    if extra_or is None:
        base = values
        n_forced = 0
    else:
        extra_or = np.asarray(extra_or, dtype=bool)
        base = values[~extra_or]                 # the override flags these anyway
        n_forced = int(extra_or.sum())
    obs = np.sort(base[np.isfinite(base)])

    if direction == ">=":
        n_hit = obs.size - np.searchsorted(obs, grid, side="left")
    elif direction == "<=":
        n_hit = np.searchsorted(obs, grid, side="right")
    else:
        raise ValueError(f"best_cutoff_for_target: unknown direction {direction!r}")

    fracs = (n_hit + n_forced) / float(n_total)
    diffs = np.abs(fracs - float(target_frac))
    d_min = float(diffs.min())

    idx = int(np.argmin(diffs))
    if prefer:
        tied = np.flatnonzero(diffs <= d_min + 1e-12)
        for j in tied:
            if round(float(grid[j]), 6) in prefer:
                idx = int(j)
                break

    cut = float(grid[idx])
    flag = (values >= cut) if direction == ">=" else (values <= cut)
    if extra_or is not None:
        flag = flag | extra_or
    return (cut, flag, float(fracs[idx]), float(diffs[idx]))


def workload_tolerance_pct(workload_pct):
    """Tolerance (in percentage points of the stratum) within which two rows
    count as the SAME workload. See WORKLOAD_MATCH_TOL_* for the rationale."""
    if not np.isfinite(workload_pct) or workload_pct <= 0:
        return WORKLOAD_MATCH_TOL_PCT
    return float(min(WORKLOAD_MATCH_TOL_PCT,
                     WORKLOAD_MATCH_TOL_REL * float(workload_pct)))


def build_match_grid(values, legible_grid):
    """Candidate cutoffs for T5D workload matching -> (grid, legible_value_set).

    Union of the legible grid and the comparator's own observed distinct values.
    For a threshold rule the observed values ARE the complete set of achievable
    flagged fractions, so this makes the match as good as the data allow; the
    legible set is handed to `best_cutoff_for_target(prefer=...)` so a round
    cutoff still wins whenever it is exactly as good.
    """
    legible = np.asarray(legible_grid, dtype=np.float64)
    legible_set = set(np.round(legible, 6).tolist())
    if not T5D_MATCH_ON_EMPIRICAL_VALUES:
        return legible, legible_set
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return legible, legible_set
    grid = np.unique(np.concatenate([legible, np.round(v, 6)]))
    return grid, legible_set


def fmt_cutoff(x, dp):
    """Round cutoffs print at their natural precision; matched empirical
    cutoffs print at 4 dp. The full-precision value is always in `Cutoff`."""
    x = float(x)
    return f"{x:.{dp}f}" if abs(x - round(x, dp)) < 1e-9 else f"{x:.4f}"


def parse_theoretical_point_sum(raw):
    """Parse `refit.json['theoretical_point_sum_unconstrained']` -> (min, max).

    `05_refit_integer.py:509` writes a TWO-ELEMENT LIST
    `[theo_min_unconstrained, theo_max_unconstrained]`. Reading it as a scalar
    is what killed this script before `score_card.json` was written; the same
    handling already lives in `render_figures.py:351`. A bare scalar is accepted
    and read as the maximum (older producers). Anything else raises — CONTRACTS
    §13 treats a gate that cannot be evaluated as a FAIL, so a shape this
    function does not understand must be loud, never swallowed.
    """
    if raw is None:
        return None, None

    def _as_int(v):
        if v is None:
            return None
        if isinstance(v, bool):
            raise TypeError(f"boolean {v!r} is not a point sum")
        if isinstance(v, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(v)):
                return None
            return int(round(float(v)))
        raise TypeError(f"{type(v).__name__} {v!r} is not a point sum")

    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            raise TypeError(
                f"expected a 2-element [min, max] list, got {len(raw)} elements")
        return _as_int(raw[0]), _as_int(raw[1])
    if isinstance(raw, bool):
        raise TypeError("boolean is not a point sum")
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return None, _as_int(raw)
    raise TypeError(f"unsupported type {type(raw).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Downsample TEST rows (smoke).")
    args = parser.parse_args()

    t0 = time.time()
    banner("Step 08 — Calibration / DCA / operating characteristics (DATA only)")
    df = pd.read_parquet(IN_PARQUET)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE] N={len(df):,}")
    y_all = df[OUTCOME_COL].astype(int).to_numpy()
    overall_prev = float(y_all.mean())

    PREV_ANCHOR, anchor_src = read_prevalence_anchor()
    if not np.isfinite(PREV_ANCHOR):
        PREV_ANCHOR = overall_prev
        anchor_src = "fallback: observed TEST-set prevalence"
        print("  [WARN] DCA prevalence anchor fell back to the observed test prevalence.")
    if abs(PREV_ANCHOR - PREV_ANCHOR_V2_SUPERSEDED) < 1e-6:
        print(f"  [WARN] DCA prevalence anchor equals the SUPERSEDED v2 value "
              f"{PREV_ANCHOR_V2_SUPERSEDED} — check that 01 re-ran with Fix 1/2/2b.")
    print(f"  N={len(df):,}  observed TEST prev={overall_prev:.4f}")
    print(f"  DCA prevalence anchor={PREV_ANCHOR:.4f}  (source: {anchor_src}; "
          f"v2 value {PREV_ANCHOR_V2_SUPERSEDED} is superseded)")

    # ---------------- Frozen refit artifacts (05) ----------------
    try:
        with open(REFIT_JSON) as f:
            refit = json.load(f)
    except Exception as exc:
        print(f"  [WARN] could not read {REFIT_JSON} ({exc}) — using fallback labels.")
        refit = {}
    _labels = refit.get("override_labels") or {}
    if isinstance(_labels, dict):
        OVERRIDE_LABEL = str(_labels.get(OVERRIDE_VAR) or
                             (list(_labels.values())[0] if _labels else "")) \
            or OVERRIDE_LABEL_FALLBACK
    elif isinstance(_labels, (list, tuple)) and _labels:
        OVERRIDE_LABEL = str(_labels[0])
    else:
        OVERRIDE_LABEL = OVERRIDE_LABEL_FALLBACK
    OVERRIDE_RULE_TEXT = str(refit.get("override_rule_text")
                             or OVERRIDE_RULE_TEXT_FALLBACK)
    print(f"  override label     : {OVERRIDE_LABEL}")
    print(f"  override rule text : {OVERRIDE_RULE_TEXT}")

    # ---- G4c input: 05's theoretical unconstrained point sum ----------------
    # 05 writes this as a two-element [min, max] list. Parse it HERE, at the
    # top of the run, so a shape this script does not understand fails before
    # the long DCA rather than after the tables are on disk and score_card.json
    # is not (the exact failure mode this fix exists to remove).
    theo_unc_raw = refit.get("theoretical_point_sum_unconstrained")
    try:
        theo_unc_min, theo_unc_max = parse_theoretical_point_sum(theo_unc_raw)
    except TypeError as exc:
        raise SystemExit(
            f"FATAL (G4c) — refit.json['theoretical_point_sum_unconstrained'] has an\n"
            f"  unrecognised shape: {theo_unc_raw!r} ({exc}).\n"
            f"  05_refit_integer.py:509 writes [theo_min_unconstrained,\n"
            f"  theo_max_unconstrained]; a bare scalar is also accepted and read as\n"
            f"  the maximum. CONTRACTS §13: a gate that cannot be evaluated is a\n"
            f"  FAIL, not a skip — so this halts rather than skipping the check.")
    if theo_unc_raw is None:
        print("  [WARN] G4c — refit.json carries no "
              "'theoretical_point_sum_unconstrained'; the cross-check against 05 "
              "CANNOT BE EVALUATED (CONTRACTS §13: that is a gate FAILURE, not a "
              "skip). Re-run 05_refit_integer.py.")
    else:
        print(f"  05 theoretical unconstrained point sum: "
              f"min={theo_unc_min} max={theo_unc_max}  (raw {theo_unc_raw!r})")

    # ---------------- Fix 4: the override column ----------------
    if OVERRIDE_COL not in df.columns:
        raise SystemExit(
            f"FATAL — '{OVERRIDE_COL}' missing from {os.path.basename(IN_PARQUET)}.\n"
            f"  CONTRACTS §5.1 requires 06_score_test.py to emit the binary\n"
            f"  automatic-escalate override column. Re-run 06.")
    override = pd.to_numeric(df[OVERRIDE_COL], errors="coerce").fillna(0).to_numpy() > 0
    n_override = int(override.sum())
    print(f"  {OVERRIDE_COL}: n={n_override:,} ({100*override.mean():.3f}% of test set) "
          f"— {OVERRIDE_LABEL}")

    data = {}
    for spec in SCORE_SPECS:
        d = load_oriented(df, spec)
        d["prob"] = logistic_recalibrate(d["y"], d["s"]) if d["n"] else np.array([])
        data[d["label"]] = d
        print(f"  {d['label']:<14} N={d['n']:,} prev={d['y'].mean():.4f}")

    # ---------------- DCA (OVERALL) ----------------
    banner("DCA  p_t 0.02–0.40  (treat-all / treat-none / SI-alone references)")
    # A 4th curve for the deployed score-OR-override rule (F5 selects only
    # pelvic_score / SI-alone / rSIG, so this is additive and invisible to F5).
    score_all = pd.to_numeric(df["pelvic_score"], errors="coerce").to_numpy(dtype=np.float64)
    promoted = np.where(override, np.nanmax(score_all) + 1.0, score_all)
    prob_or = logistic_recalibrate(y_all, promoted)
    dca_rows = []
    for p_t in THRESHOLDS:
        ta = nb_treat_all(PREV_ANCHOR, p_t)
        ta_obs = nb_treat_all(overall_prev, p_t)
        for label, _, _ in SCORE_SPECS:
            d = data[label]
            nb = net_benefit(d["y"], d["prob"], p_t) if d["n"] else np.nan
            dca_rows.append({"threshold": float(p_t), "score": label, "net_benefit": nb,
                             "n_used": d["n"], "treat_all_nb": ta, "treat_none_nb": 0.0,
                             "treat_all_nb_observed_prev": ta_obs})
        dca_rows.append({"threshold": float(p_t), "score": "pelvic_score_or_override",
                         "net_benefit": net_benefit(y_all, prob_or, p_t),
                         "n_used": int(len(df)), "treat_all_nb": ta, "treat_none_nb": 0.0,
                         "treat_all_nb_observed_prev": ta_obs})
    dca_df = pd.DataFrame(dca_rows)
    dca_df.to_csv(OUT_DCA, index=False)
    print(f"  wrote {OUT_DCA} ({len(dca_df)} rows; prevalence anchor {PREV_ANCHOR:.4f})")
    piv = dca_df.pivot_table(index="threshold", columns="score", values="net_benefit")
    win = piv[(piv.index >= 0.05) & (piv.index <= 0.20)]
    gap = win["pelvic_score"] - win["SI-alone"]
    print(f"  NB gap pelvic-SI over p_t 0.05–0.20: min={gap.min():+.5f} "
          f"mean={gap.mean():+.5f} (all positive: {bool((gap > 0).all())})")

    # ---------------- Calibration (OVERALL) reliability deciles ----------------
    banner("Calibration reliability deciles (fig-data F4)")
    calib_rows = []
    for label, _, _ in SCORE_SPECS:
        d = data[label]
        if d["n"] == 0:
            continue
        try:
            frac, mean_pred = calibration_curve(d["y"], d["prob"], n_bins=10, strategy="quantile")
        except Exception as exc:
            print(f"  [calib warn {label}] {exc}")
            continue
        p = np.clip(d["prob"], EPS, 1 - EPS)
        citl = float(p.mean() - d["y"].mean())
        for i in range(len(frac)):
            calib_rows.append({"score": label, "bin": i + 1,
                               "mean_pred": float(mean_pred[i]), "obs_frac": float(frac[i]),
                               "citl": citl, "n": d["n"], "prev": float(d["y"].mean())})
        print(f"  {label:<14} deciles={len(frac)} CITL={citl:+.4f}")
    pd.DataFrame(calib_rows).to_csv(OUT_CALIB, index=False)
    print(f"  wrote {OUT_CALIB}")

    # ---------------- Operating points (UNCHANGED SCHEMA) ----------------
    banner("Operating points — sens-90 rule-OUT / spec-90 rule-IN / Youden "
           "(table5_operating.csv, schema unchanged)")
    op_rows = []
    cutpoints = {}
    for label in ["pelvic_score", "SI-alone"]:
        d = data[label]
        y = d["y"]
        s = d["s_native"] if d["sign"] == +1 else -d["s_native"]
        cuts = {"youden": youden_cut(y, s), "sens_90": sens90_cut(y, s),
                "spec_90": spec90_cut(y, s)}
        cutpoints[label] = {}
        for rule, cut in cuts.items():
            if np.isnan(cut):
                print(f"  {label} {rule}: unreachable")
                continue
            m = metrics_at(y, s, cut)
            m.update({"Score": label, "Cutoff_Rule": rule})
            op_rows.append(m)
            cutpoints[label][rule] = {"cutoff": float(cut), "sens": m["Sensitivity"],
                                      "spec": m["Specificity"], "ppv": m["PPV"],
                                      "npv": m["NPV"], "lr_pos": m["LR_pos"],
                                      "missed_event_rate": m["Missed_Event_Rate"]}
            print(f"  {label:<14} {rule:<8} cut={cut:>7.3f} sens={m['Sensitivity']:.3f} "
                  f"spec={m['Specificity']:.3f} PPV={m['PPV']:.3f} NPV={m['NPV']:.3f} "
                  f"LR+={m['LR_pos']:.2f}")
    op_df = pd.DataFrame(op_rows)
    front = ["Score", "Cutoff_Rule", "Cutoff", "N", "N_flagged",
             "Sensitivity", "Sens_CI_Low", "Sens_CI_High",
             "Specificity", "Spec_CI_Low", "Spec_CI_High",
             "PPV", "PPV_CI_Low", "PPV_CI_High", "NPV", "NPV_CI_Low", "NPV_CI_High",
             "LR_pos", "LR_neg", "Missed_Event_Rate", "Youden_J", "TP", "FP", "TN", "FN"]
    op_df = op_df[[c for c in front if c in op_df.columns]]
    op_df.to_csv(OUT_OPCHAR, index=False)
    print(f"  wrote {OUT_OPCHAR} ({len(op_df)} rows) — schema unchanged from v2")

    ruleout_cut = cutpoints["pelvic_score"].get("sens_90", {}).get("cutoff", np.nan)
    escalate_cut = cutpoints["pelvic_score"].get("spec_90", {}).get("cutoff", np.nan)

    # =====================================================================
    # FIX 5 — Table 5C: full operating characteristics at every cutoff
    # =====================================================================
    banner("Fix 5 — table5c_operating_full.csv (every integer cutoff; "
           "score vs shock index vs rSIG; score-only AND score-OR-override)")

    si_all = pd.to_numeric(df["shock_index"], errors="coerce").to_numpy(dtype=np.float64)
    rsig_all = pd.to_numeric(df["rsig"], errors="coerce").to_numpy(dtype=np.float64)
    common = ~np.isnan(si_all) & ~np.isnan(rsig_all)      # score is never NaN
    print(f"  common complete-case set (score & SI & rSIG all observed): "
          f"{int(common.sum()):,} / {len(df):,} ({100*common.mean():.2f}%)")
    print("  NOTE: the score is defined for 100% of rows; SI/rSIG are not. "
          "Restricting to the")
    print("        common set removes patients with unobtainable vitals — the "
          "sickest — which is")
    print("        CONSERVATIVE for the score. T5C reports both denominators; "
          "T5D uses the common set only.")

    score_min = int(np.nanmin(score_all))
    score_max = int(np.nanmax(score_all))
    integer_grid = list(range(score_min, score_max + 1))
    print(f"  observed pelvic_score range in the v3 test set: "
          f"{score_min:+d} .. {score_max:+d}  ({len(integer_grid)} integer cutoffs)")

    def cutoff_role(c):
        roles = []
        if np.isfinite(ruleout_cut) and abs(c - ruleout_cut) < 1e-9:
            roles.append("rule_out")
        if np.isfinite(escalate_cut) and abs(c - escalate_cut) < 1e-9:
            roles.append("rule_in")
        yc = cutpoints["pelvic_score"].get("youden", {}).get("cutoff", np.nan)
        if np.isfinite(yc) and abs(c - yc) < 1e-9:
            roles.append("youden")
        return "|".join(roles)

    def in_window(c):
        if not (np.isfinite(ruleout_cut) and np.isfinite(escalate_cut)):
            return 1
        return int(min(ruleout_cut, escalate_cut) <= c <= max(ruleout_cut, escalate_cut))

    t5c_rows = []

    def _t5c_row(denom_label, y, instrument, rule, cutoff, direction, label,
                 source, flag, role="", window=np.nan,
                 marg_flagged=np.nan, marg_events=np.nan):
        m = metrics_from_flag(y, flag)
        row = {
            "Denominator": denom_label,
            "Instrument": instrument,
            "Rule": rule,
            "Cutoff": cutoff,
            "Cutoff_Direction": direction,
            "Cutoff_Label": label,
            "Cutoff_Source": source,
            "Cutoff_Role": role,
            "Within_RuleOut_RuleIn_Window": window,
            "N_used": m["N"],
            "N_events": int(np.asarray(y).sum()),
            "N_flagged": m["N_flagged"],
            "Pct_cohort_flagged": 100.0 * m["N_flagged"] / m["N"] if m["N"] else np.nan,
            "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"],
            "Sensitivity": m["Sensitivity"], "Sens_CI_Low": m["Sens_CI_Low"],
            "Sens_CI_High": m["Sens_CI_High"],
            "Specificity": m["Specificity"], "Spec_CI_Low": m["Spec_CI_Low"],
            "Spec_CI_High": m["Spec_CI_High"],
            "PPV": m["PPV"], "PPV_CI_Low": m["PPV_CI_Low"], "PPV_CI_High": m["PPV_CI_High"],
            "NPV": m["NPV"], "NPV_CI_Low": m["NPV_CI_Low"], "NPV_CI_High": m["NPV_CI_High"],
            "LR_pos": m["LR_pos"], "LR_neg": m["LR_neg"],
            "Number_needed_to_escalate": m["Number_needed_to_escalate"],
            "Missed_Event_Rate": m["Missed_Event_Rate"], "Youden_J": m["Youden_J"],
            "Override_Marginal_Flagged": marg_flagged,
            "Override_Marginal_Events": marg_events,
        }
        t5c_rows.append(row)
        return row

    max_marginal_flagged = 0
    for denom_label, mask in [("all_test_rows", np.ones(len(df), dtype=bool)),
                              ("common_complete_case", common)]:
        ys = y_all[mask]
        sc = score_all[mask]
        ov = override[mask]
        si = si_all[mask]
        rs = rsig_all[mask]

        # ---- pelvic_score, both rules, every integer cutoff ----
        for c in integer_grid:
            f_score = sc >= c
            _t5c_row(denom_label, ys, "pelvic_score", "score_only", float(c), ">=",
                     f"pelvic score >= {c:+d}", "integer_grid", f_score,
                     role=cutoff_role(c), window=in_window(c),
                     marg_flagged=0, marg_events=0)
            f_or = f_score | ov
            marg = int((~f_score & ov).sum())
            marg_ev = int((~f_score & ov & (ys == 1)).sum())
            max_marginal_flagged = max(max_marginal_flagged, marg)
            _t5c_row(denom_label, ys, "pelvic_score", "score_or_override", float(c), ">=",
                     f"pelvic score >= {c:+d} OR {OVERRIDE_LABEL.lower()}",
                     "integer_grid", f_or, role=cutoff_role(c), window=in_window(c),
                     marg_flagged=marg, marg_events=marg_ev)

        # ---- the override alone (the CSV-native "T5C footer", CONTRACTS G4e) ----
        _t5c_row(denom_label, ys, OVERRIDE_COL, "override_only", np.nan, "==",
                 OVERRIDE_LABEL, "not_applicable", ov, role="automatic_escalation")

        # ---- shock index / rSIG: workload-matched + clinical anchors ----
        for inst, vals, direction, grid, anchors, fmt in [
            ("shock_index", si, ">=", SI_GRID, SI_CLINICAL_ANCHORS, "{:.2f}"),
            ("rSIG", rs, "<=", RSIG_GRID, RSIG_CLINICAL_ANCHORS, "{:.1f}"),
        ]:
            ok = ~np.isnan(vals)
            y_i, v_i = ys[ok], vals[ok]
            if v_i.size == 0:
                continue
            # 1) workload-matched cutoffs, one per integer score cutoff, deduped
            matched = {}
            for c in integer_grid:
                target = float((sc >= c).mean())
                best = best_cutoff_for_target(v_i, direction, grid, target)
                if best is None:
                    continue
                cut, flag, _frac, _d = best
                matched.setdefault(round(cut, 4), (cut, flag))
            workload_keys = set(matched)
            # 2) clinical anchors; if an anchor coincides with a workload-matched
            #    cutoff the row is emitted ONCE with a combined Cutoff_Source, so
            #    no cutoff ever appears twice for the same instrument.
            anchor_keys = set()
            for c in anchors:
                k = round(float(c), 4)
                anchor_keys.add(k)
                if k not in matched:
                    flag = (v_i >= c) if direction == ">=" else (v_i <= c)
                    matched[k] = (float(c), flag)
            for key in sorted(matched):
                cut, flag = matched[key]
                in_w, in_a = key in workload_keys, key in anchor_keys
                src = ("workload_matched|clinical_anchor" if (in_w and in_a)
                       else ("clinical_anchor" if in_a else "workload_matched"))
                _t5c_row(denom_label, y_i, inst, "instrument_only", cut, direction,
                         f"{inst} {direction} {fmt.format(cut)}", src, flag)

    t5c = pd.DataFrame(t5c_rows)
    t5c.to_csv(OUT_T5C, index=False)
    print(f"  wrote {OUT_T5C} ({len(t5c)} rows, {t5c.shape[1]} cols)")
    print(f"  G4d — override marginal flags: max {max_marginal_flagged:,} "
          f"(must be <= n_override = {n_override:,})")
    assert max_marginal_flagged <= n_override, (
        f"FATAL — override marginal flags ({max_marginal_flagged}) exceed the "
        f"override group size ({n_override}); the OR-rule is mis-assembled.")
    _mx = t5c[(t5c["Rule"] == "score_or_override")]
    if len(_mx):
        print(f"  G4d — override marginal EVENTS across cutoffs: "
              f"max {int(_mx['Override_Marginal_Events'].max()):,}, "
              f"min {int(_mx['Override_Marginal_Events'].min()):,}")

    # =====================================================================
    # FIX 5 — Table 5D: matched-workload comparison (THE clinical argument)
    # =====================================================================
    banner("Fix 5 — table5d_matched_workload.csv (equal workload, events caught; "
           "overall + elderly >=65)")
    print("  MATCHING RULE (Batch-7 correction). Every block has exactly ONE anchor")
    print("  instrument. The anchor's ACHIEVED flagged fraction — not the nominal")
    print("  level — is the operative workload target, and every other instrument is")
    print("  matched to that achieved fraction on its own cutoff grid. This is the")
    print("  algorithm T5C already used. Previously each instrument chased the")
    print("  NOMINAL level independently: the shock index (fine grid) landed on it")
    print("  and the integer-valued score did not, yet the deltas were taken against")
    print("  the shock-index row anyway — booking extra escalations as an")
    print("  equal-workload gain.")
    print(f"    * fixed_*pct / deployed_rule_in  -> ANCHOR = pelvic_score (score_only)")
    print(f"    * si_clinical_*                  -> ANCHOR = shock_index; the matching")
    print(f"                                        direction REVERSES and the score is")
    print(f"                                        matched to the shock index.")
    print(f"  A delta is published only when the two rows being differenced agree to")
    print(f"  within min({WORKLOAD_MATCH_TOL_PCT} pp, "
          f"{100*WORKLOAD_MATCH_TOL_REL:g}% of the workload); otherwise the delta")
    print(f"  columns are NA and Delta_flag carries the reason.")

    strata = [("overall", np.ones(len(df), dtype=bool))]
    if "age_ge65" in df.columns:
        eld = pd.to_numeric(df["age_ge65"], errors="coerce").fillna(0).to_numpy() > 0
        strata.append(("elderly_ge65", eld))
        print(f"  elderly (age>=65) stratum: n={int(eld.sum()):,}")
    else:
        print("  [WARN] 'age_ge65' absent from the scored parquet — elderly "
              "stratum SKIPPED. 06 must carry it.")

    t5d_rows = []
    n_delta_suppressed = 0
    suppressed_rows = []
    DENOM = "common_complete_case_score_SI_rSIG"
    for stratum, smask in strata:
        m = smask & common
        n_an = int(m.sum())
        if n_an == 0:
            print(f"  [WARN] stratum '{stratum}' empty on the common set — skipped")
            continue
        ys = y_all[m]
        sc = score_all[m]
        ov = override[m]
        si = si_all[m]
        rs = rsig_all[m]
        n_ev = int(ys.sum())
        ev_rate = float(ys.mean())
        pct_retained = 100.0 * n_an / max(int(smask.sum()), 1)
        print(f"\n  --- stratum {stratum}: N_analysis={n_an:,} ({pct_retained:.2f}% of "
              f"the stratum), events={n_ev:,} ({100*ev_rate:.2f}%) ---")

        # ---- cutoff grids for this stratum -------------------------------
        # The score keeps its own integer resolution (it IS the instrument); the
        # comparators are matched on the union of the legible grid and their own
        # observed values, so the match is as good as the data allow.
        score_grid = np.asarray(integer_grid, dtype=float)
        si_grid_m, si_legible = build_match_grid(si, SI_GRID)
        rsig_grid_m, rsig_legible = build_match_grid(rs, RSIG_GRID)

        # ---- assemble the workload targets, each with an explicit ANCHOR ----
        # anchor_cutoff is None  -> the anchor cutoff is searched for the nominal
        #                           level, and its ACHIEVED fraction becomes the
        #                           operative target.
        # anchor_cutoff is set   -> the target IS that cutoff's achieved fraction.
        targets = [{"label": f"fixed_{t:g}pct", "anchor_instrument": "pelvic_score",
                    "anchor_rule": "score_only", "anchor_cutoff": None,
                    "nominal_pct": float(t)}
                   for t in WORKLOAD_TARGETS_PCT]
        if np.isfinite(escalate_cut):
            targets.append({"label": "deployed_rule_in",
                            "anchor_instrument": "pelvic_score",
                            "anchor_rule": "score_only",
                            "anchor_cutoff": float(escalate_cut),
                            "nominal_pct": None})
        if int(np.isfinite(si).sum()):
            for a in SI_CLINICAL_WORKLOAD_ANCHORS:
                targets.append({"label": f"si_clinical_{a:g}",
                                "anchor_instrument": "shock_index",
                                "anchor_rule": "instrument_only",
                                "anchor_cutoff": float(a),
                                "nominal_pct": None})

        for tspec in targets:
            tlabel = tspec["label"]
            a_inst, a_rule = tspec["anchor_instrument"], tspec["anchor_rule"]
            if a_inst == "pelvic_score":
                a_vals, a_dir, a_grid, a_prefer, a_dp = sc, ">=", score_grid, None, 0
            else:
                a_vals, a_dir, a_grid, a_prefer, a_dp = si, ">=", si_grid_m, si_legible, 2

            # ---- 1) resolve the anchor row; its achieved fraction is THE target
            if tspec["anchor_cutoff"] is None:
                nominal = tspec["nominal_pct"] / 100.0
                b = best_cutoff_for_target(a_vals, a_dir, a_grid, nominal, prefer=a_prefer)
                if b is None:
                    print(f"    [WARN] {tlabel}: no anchor cutoff reachable on "
                          f"{a_inst} — block skipped")
                    continue
                a_cut, a_flag, a_frac = b[0], b[1], b[2]
            else:
                a_cut = float(tspec["anchor_cutoff"])
                a_flag = (a_vals >= a_cut) if a_dir == ">=" else (a_vals <= a_cut)
                a_frac = float(a_flag.mean()) if a_flag.size else np.nan
            if not np.isfinite(a_frac) or a_frac <= 0:
                print(f"    [WARN] {tlabel}: anchor ({a_inst} {a_dir} {a_cut}) flags "
                      f"nothing — block skipped")
                continue

            operative_pct = 100.0 * a_frac          # the workload actually held constant
            tol_pct = workload_tolerance_pct(operative_pct)
            a_label = (f"pelvic score >= {int(a_cut):+d}" if a_inst == "pelvic_score"
                       else f"shock index >= {fmt_cutoff(a_cut, a_dp)}")
            print(f"\n    [{tlabel}] ANCHOR = {a_inst} ({a_rule}) at {a_label} "
                  f"-> operative workload {operative_pct:.3f}% "
                  f"(nominal {tspec['nominal_pct'] if tspec['nominal_pct'] is not None else operative_pct:.3f}%"
                  f"); tolerance +/-{tol_pct:.3f} pp")

            # ---- 2) match every instrument to the anchor's ACHIEVED fraction --
            def _entry(inst, rule, cut, direction, label, flag):
                pct = 100.0 * float(flag.mean()) if flag.size else np.nan
                return {"inst": inst, "rule": rule, "cut": float(cut), "dir": direction,
                        "label": label, "flag": flag, "pct": pct,
                        "is_anchor": (inst == a_inst and rule == a_rule)}

            entries = []
            # (a) score alone — the anchor itself for score-anchored blocks
            if a_inst == "pelvic_score" and a_rule == "score_only":
                entries.append(_entry("pelvic_score", "score_only", a_cut, ">=",
                                      f"pelvic score >= {int(a_cut):+d}", a_flag))
            else:
                b = best_cutoff_for_target(sc, ">=", score_grid, a_frac)
                if b:
                    entries.append(_entry("pelvic_score", "score_only", b[0], ">=",
                                          f"pelvic score >= {int(b[0]):+d}", b[1]))
            # (b) the DEPLOYED rule: score OR automatic escalation
            b = best_cutoff_for_target(sc, ">=", score_grid, a_frac, extra_or=ov)
            if b:
                entries.append(_entry("pelvic_score", "score_or_override", b[0], ">=",
                                      f"pelvic score >= {int(b[0]):+d} OR "
                                      f"{OVERRIDE_LABEL.lower()}", b[1]))
            # (c) shock index — the anchor itself for si_clinical_* blocks
            if a_inst == "shock_index":
                entries.append(_entry("shock_index", "instrument_only", a_cut, ">=",
                                      f"shock index >= {fmt_cutoff(a_cut, 2)}", a_flag))
            else:
                b = best_cutoff_for_target(si, ">=", si_grid_m, a_frac, prefer=si_legible)
                if b:
                    entries.append(_entry("shock_index", "instrument_only", b[0], ">=",
                                          f"shock index >= {fmt_cutoff(b[0], 2)}", b[1]))
            # (d) rSIG — always matched
            b = best_cutoff_for_target(rs, "<=", rsig_grid_m, a_frac, prefer=rsig_legible)
            if b:
                entries.append(_entry("rSIG", "instrument_only", b[0], "<=",
                                      f"rSIG <= {fmt_cutoff(b[0], 1)}", b[1]))

            # ---- 3) the delta reference stays the shock-index row -------------
            ref = next((e for e in entries if e["inst"] == "shock_index"
                        and e["rule"] == "instrument_only"), None)
            ref_m = metrics_from_flag(ys, ref["flag"]) if ref else None

            for e in entries:
                mm = metrics_from_flag(ys, e["flag"])
                gap_pct = (e["pct"] - ref["pct"]) if ref else np.nan
                gap_n = (mm["N_flagged"] - ref_m["N_flagged"]) if ref_m else np.nan
                if ref is None or ref_m is None:
                    comparable = 0
                    flag_txt = ("NO_REFERENCE: no shock-index row could be built for "
                                "this block — no delta is defined")
                elif e is ref:
                    comparable = 1
                    flag_txt = "reference_row (delta is 0 against itself by definition)"
                elif abs(gap_pct) <= tol_pct:
                    comparable = 1
                    flag_txt = (f"matched (workload gap {gap_pct:+.3f} pp, within the "
                                f"{tol_pct:.3f} pp tolerance)")
                else:
                    comparable = 0
                    flag_txt = (f"NOT_COMPARABLE: this row flags {e['pct']:.3f}% of the "
                                f"stratum against the shock-index reference's "
                                f"{ref['pct']:.3f}% (gap {gap_pct:+.3f} pp > "
                                f"{tol_pct:.3f} pp tolerance) — the delta would compare "
                                f"UNEQUAL workloads and is suppressed")
                d_ev = (mm["TP"] - ref_m["TP"]) if (comparable and ref_m) else np.nan
                d_fp = (mm["FP"] - ref_m["FP"]) if (comparable and ref_m) else np.nan
                if not comparable:
                    n_delta_suppressed += 1
                    suppressed_rows.append((stratum, tlabel, e["inst"], e["rule"],
                                            e["pct"], (ref["pct"] if ref else np.nan),
                                            tol_pct))
                t5d_rows.append({
                    "Stratum": stratum,
                    "Denominator": DENOM,
                    "N_analysis": n_an,
                    "Pct_of_stratum_retained": round(pct_retained, 3),
                    "N_events": n_ev,
                    "Event_rate": ev_rate,
                    "Workload_target_label": tlabel,
                    # The workload the block actually holds constant = the ANCHOR's
                    # achieved flagged fraction. The nominal request is kept beside
                    # it so the two are never confused.
                    "Workload_target_pct": round(operative_pct, 4),
                    "Workload_target_basis": "anchor_instrument_achieved_fraction",
                    "Workload_nominal_target_pct": (round(tspec["nominal_pct"], 4)
                                                    if tspec["nominal_pct"] is not None
                                                    else np.nan),
                    "Workload_anchor_instrument": a_inst,
                    "Workload_anchor_rule": a_rule,
                    "Workload_anchor_cutoff": a_cut,
                    "Workload_anchor_cutoff_label": a_label,
                    "Workload_anchor_pct_flagged": round(operative_pct, 4),
                    "Instrument": e["inst"],
                    "Rule": e["rule"],
                    "Is_workload_anchor": int(bool(e["is_anchor"])),
                    "Cutoff": e["cut"],
                    "Cutoff_Direction": e["dir"],
                    "Cutoff_Label": e["label"],
                    "N_flagged": mm["N_flagged"],
                    "Pct_flagged_actual": 100.0 * mm["N_flagged"] / n_an,
                    "Workload_match_abs_diff_pct": round(abs(e["pct"] - operative_pct), 4),
                    "Workload_match_tolerance_pct": round(tol_pct, 4),
                    "Workload_match_within_tolerance": int(
                        abs(e["pct"] - operative_pct) <= tol_pct),
                    "Events_caught_TP": mm["TP"],
                    "False_positives_FP": mm["FP"],
                    "Events_missed_FN": mm["FN"],
                    "True_negatives_TN": mm["TN"],
                    "Sensitivity": mm["Sensitivity"],
                    "PPV": mm["PPV"],
                    "Number_needed_to_escalate": mm["Number_needed_to_escalate"],
                    "Reference_instrument": "shock_index (instrument_only), same "
                                            "Stratum + Workload_target_label",
                    # the compared rows' achieved workloads, side by side, so the
                    # match can be audited from the CSV without re-running anything
                    "Reference_cutoff_label": (ref["label"] if ref else ""),
                    "Reference_pct_flagged": (round(ref["pct"], 4) if ref else np.nan),
                    "Reference_N_flagged": (ref_m["N_flagged"] if ref_m else np.nan),
                    "Reference_events_caught_TP": (ref_m["TP"] if ref_m else np.nan),
                    "Reference_false_positives_FP": (ref_m["FP"] if ref_m else np.nan),
                    "Delta_workload_gap_pct": (round(gap_pct, 4)
                                               if np.isfinite(gap_pct) else np.nan),
                    "Delta_workload_gap_n": gap_n,
                    "Delta_comparable": comparable,
                    "Delta_flag": flag_txt,
                    "Delta_events_caught": d_ev,
                    "Delta_false_positives": d_fp,
                })
                _d = ("     NA" if not np.isfinite(d_ev) else f"{int(d_ev):+7d}")
                _f = ("     NA" if not np.isfinite(d_fp) else f"{int(d_fp):+7d}")
                print(f"      {e['inst']:<13} {e['rule']:<18} {e['label']:<52} "
                      f"flag={mm['N_flagged']:>7,} ({e['pct']:6.3f}%) "
                      f"TP={mm['TP']:>6,} FP={mm['FP']:>7,} "
                      f"PPV={mm['PPV']:.3f} NNE={mm['Number_needed_to_escalate']:.2f} "
                      f"dTP={_d} dFP={_f} "
                      f"{'ANCHOR' if e['is_anchor'] else ('match' if comparable else 'SUPPRESSED')}")

    t5d = pd.DataFrame(t5d_rows)
    t5d.to_csv(OUT_T5D, index=False)
    print(f"\n  wrote {OUT_T5D} ({len(t5d)} rows, {t5d.shape[1]} cols)")

    # ---- workload-match verification (the gate that makes T5D quotable) ----
    if len(t5d):
        pub = t5d[(t5d["Delta_comparable"] == 1)]
        bad = pub[pub["Delta_workload_gap_pct"].abs()
                  > pub["Workload_match_tolerance_pct"] + 1e-9]
        assert len(bad) == 0, (
            "FATAL — a T5D row was published with a delta across mismatched "
            f"workloads:\n{bad[['Stratum', 'Workload_target_label', 'Instrument', 'Rule', 'Pct_flagged_actual', 'Reference_pct_flagged', 'Delta_workload_gap_pct']]}")
        assert bool(t5d.loc[t5d["Delta_comparable"] == 0,
                            ["Delta_events_caught", "Delta_false_positives"]]
                    .isna().all().all()), (
            "FATAL — a non-comparable T5D row still carries a delta.")
        print(f"  ASSERT pass: all {len(pub)} published T5D delta(s) compare rows "
              f"whose achieved flagged fractions agree within tolerance.")
    if n_delta_suppressed:
        print(f"  [WARN] {n_delta_suppressed} T5D row(s) had their delta SUPPRESSED "
              f"(NA) because the best achievable workload match still exceeded "
              f"tolerance. These rows are still published — with their achieved "
              f"fractions — but their Delta_* columns must NOT be quoted as an "
              f"equal-workload claim:")
        for st, tl, inst, rule, pct, rpct, tol in suppressed_rows:
            print(f"      {st:<13} {tl:<18} {inst:<13} {rule:<18} "
                  f"flagged {pct:6.3f}% vs shock-index {rpct:6.3f}% "
                  f"(tolerance {tol:.3f} pp)")
    else:
        print("  no T5D delta was suppressed — every comparison is workload-matched.")

    print("  HEADLINE ROWS for the WTA abstract: Stratum='elderly_ge65', "
          "Instrument='pelvic_score',")
    print("  Rule='score_only', matched against the Instrument='shock_index' row at the "
          "SAME Workload_target_label")
    print("  — and ONLY where Delta_comparable == 1. Where it is 0 the two rows sit at "
          "different")
    print("  workloads and the difference in events caught is not an equal-workload "
          "claim.")

    # ---------------- Risk tiers (pelvic_score) + override row ----------------
    banner("Risk tiers — rule-out / observe / escalate (pelvic_score) + override row")
    print(f"  rule-out cut (sens-90) = {ruleout_cut}   escalate cut (spec-90) = {escalate_cut}")
    dps = data["pelvic_score"]
    score_native = dps["s_native"]
    yb = dps["y"]
    dfull = df.loc[dps["use"]].reset_index(drop=True)
    ovr_tier = override[dps["use"]]

    if not (np.isfinite(ruleout_cut) and np.isfinite(escalate_cut) and ruleout_cut < escalate_cut):
        q1, q2 = np.percentile(score_native, [33, 66])
        ruleout_cut, escalate_cut = float(q1), float(q2)
        print(f"  [fallback] tier cuts set to score tertiles: {ruleout_cut}, {escalate_cut}")

    def _rate(mask, col):
        if col not in dfull.columns or not mask.sum():
            return np.nan
        return float(pd.to_numeric(dfull.loc[mask, col], errors="coerce").fillna(0).mean())

    tier = np.where(score_native < ruleout_cut, "rule_out",
                    np.where(score_native >= escalate_cut, "escalate", "observe"))
    tier_rows = []
    for t in ["rule_out", "observe", "escalate"]:
        m = (tier == t)
        n = int(m.sum())
        ev = int(yb[m].sum())
        rate = ev / n if n else np.nan
        lo, hi = wilson(ev, n)
        tier_rows.append({"tier": t, "row_type": "score_tier", "n": n,
                          "composite_events": ev, "composite_rate": rate,
                          "composite_ci_low": lo, "composite_ci_high": hi,
                          "death_rate": _rate(m, DEATH_COMPONENT),
                          "death_inhosp_rate": _rate(m, DEATH_SENSITIVITY),
                          "transfusion_rate": _rate(m, TX_COMPONENT),
                          "hc_rate": _rate(m, HC_COMPONENT),
                          "score_min": int(score_native[m].min()) if n else np.nan,
                          "score_max": int(score_native[m].max()) if n else np.nan,
                          "n_override_in_tier": int(ovr_tier[m].sum())})
        print(f"  {t:<9} n={n:>7,} composite={rate:.4f} [{lo:.4f},{hi:.4f}] "
              f"death({DEATH_COMPONENT})={_rate(m, DEATH_COMPONENT):.4f} "
              f"tx={_rate(m, TX_COMPONENT):.4f} override_in_tier={int(ovr_tier[m].sum()):,}")

    # Override row — reported SEPARATELY (CONTRACTS §5.1 row 08). NOTE these
    # patients ALSO appear in one of the three score tiers above: the override is
    # a parallel escalation path, not a fourth mutually-exclusive tier.
    m = ovr_tier.astype(bool)
    n = int(m.sum())
    ev = int(yb[m].sum())
    lo, hi = wilson(ev, n)
    tier_rows.append({"tier": "override", "row_type": "override_group", "n": n,
                      "composite_events": ev,
                      "composite_rate": (ev / n if n else np.nan),
                      "composite_ci_low": lo, "composite_ci_high": hi,
                      "death_rate": _rate(m, DEATH_COMPONENT),
                      "death_inhosp_rate": _rate(m, DEATH_SENSITIVITY),
                      "transfusion_rate": _rate(m, TX_COMPONENT),
                      "hc_rate": _rate(m, HC_COMPONENT),
                      "score_min": int(score_native[m].min()) if n else np.nan,
                      "score_max": int(score_native[m].max()) if n else np.nan,
                      "n_override_in_tier": n})
    print(f"  override  n={n:>7,} composite={(ev/n if n else float('nan')):.4f} "
          f"[{lo:.4f},{hi:.4f}]  (OVERLAPS the three score tiers — parallel path)")
    tiers_df = pd.DataFrame(tier_rows)
    tiers_df.to_csv(OUT_TIERS, index=False)
    print(f"  wrote {OUT_TIERS}")

    # ---------------- Score card JSON (fig-data F9) ----------------
    banner("Score card (fig-data F9) — point rows in variables[], override in its own block")
    pts = pd.read_csv(POINTS_CSV)
    if "Role" not in pts.columns:
        raise SystemExit(
            f"FATAL — {os.path.basename(POINTS_CSV)} has no 'Role' column. "
            f"CONTRACTS §5.1 (Fix 4) requires it. Re-run 05_refit_integer.py.")
    pts["Role"] = pts["Role"].astype(str).str.strip().str.lower()
    bcol = _flex_col(pts, ["β", "beta", "Beta"])
    secol = _flex_col(pts, ["SE", "se", "Std_Err"])
    orcol = _flex_col(pts, ["OR", "or", "Odds_Ratio"])
    locol = _flex_col(pts, ["95% CI lower", "CI_Low", "ci_low"])
    hicol = _flex_col(pts, ["95% CI upper", "CI_High", "ci_high"])
    labcol = _flex_col(pts, ["Label", "label", "Display", "display"])

    def _num(r, col):
        if col is None or col not in r:
            return None
        v = pd.to_numeric(pd.Series([r[col]]), errors="coerce").iloc[0]
        return None if pd.isna(v) else float(v)

    point_rows = pts[pts["Role"] == "point"]
    override_rows = pts[pts["Role"] == "override"]
    assert "unstable_ring" not in set(pts["Variable"].astype(str)), \
        "FATAL (G3c) — unstable_ring must not appear in table3_points.csv"

    variables = []
    for _, r in point_rows.iterrows():
        variables.append({
            "variable": str(r["Variable"]).strip(),
            "label": (str(r[labcol]) if labcol and pd.notna(r.get(labcol))
                      else str(r["Variable"]).strip()),
            "points": int(round(float(r["Points"]))),
            "beta": _num(r, bcol), "or": _num(r, orcol),
            "role": "point",
        })
    assert all(v["variable"] != OVERRIDE_VAR for v in variables), \
        f"FATAL (CONTRACTS §5.1) — '{OVERRIDE_VAR}' must not be in score_card['variables']"

    # ---- theoretical point-column sums (labelled; NOT the observed range) ----
    pmap = {v["variable"]: v["points"] for v in variables}
    max_unc = sum(p for p in pmap.values() if p > 0)
    min_unc = sum(p for p in pmap.values() if p < 0)
    max_mx, min_mx = max_unc, min_unc
    groups_used = {}
    for gname, gvars in MUTEX_GROUPS.items():
        present = {v: pmap[v] for v in gvars if v in pmap}
        if len(present) < 2:
            continue
        groups_used[gname] = present
        pos = [p for p in present.values() if p > 0]
        neg = [p for p in present.values() if p < 0]
        if len(pos) > 1:
            max_mx -= (sum(pos) - max(pos))
        if len(neg) > 1:
            min_mx -= (sum(neg) - min(neg))

    # ---- GATE G4c — the unconstrained point sum agrees with 05 -------------
    # This comparison previously sat inside `except (TypeError, ValueError):
    # pass`, so the two-element list from 05 meant it NEVER evaluated: neither
    # its [WARN] nor its ASSERT-pass line could print. CONTRACTS §13 — "a gate
    # that cannot be evaluated is a FAIL, not a skip" — so it now always
    # reaches a verdict, and that verdict is carried into score_card.json.
    if theo_unc_raw is None:
        g4c_status = "FAIL_not_evaluated_missing_key"
        print("  [GATE G4c] FAIL — refit.json carries no "
              "'theoretical_point_sum_unconstrained'; the cross-check against 05 "
              "cannot be evaluated. Re-run 05_refit_integer.py.")
    else:
        checks = []
        if theo_unc_max is not None:
            checks.append(("max", theo_unc_max, int(max_unc)))
        if theo_unc_min is not None:
            checks.append(("min", theo_unc_min, int(min_unc)))
        if not checks:
            g4c_status = "FAIL_not_evaluated_non_numeric"
            print(f"  [GATE G4c] FAIL — refit.json's theoretical sum "
                  f"{theo_unc_raw!r} carries no finite value to compare against "
                  f"the recomputed sums ({min_unc:+d}..{max_unc:+d}).")
        else:
            bad = [(w, a, b) for w, a, b in checks if a != b]
            if bad:
                g4c_status = "mismatch"
                for w, a, b in bad:
                    print(f"  [GATE G4c][WARN] refit.json"
                          f"['theoretical_point_sum_unconstrained'] {w}={a} != "
                          f"recomputed {w} of the printed Points column ({b}). 05 "
                          f"may define it over a different row set; BOTH are "
                          f"reported in score_card.json so neither is hidden.")
            else:
                g4c_status = "pass"
                _shown = ", ".join(f"{w}={b:+d}" for w, _a, b in checks)
                print(f"  [GATE G4c] ASSERT pass: unconstrained point sum agrees "
                      f"with 05 ({_shown})")

    # Report (do not silently drop) any point-role variable at 0 points — in
    # particular a pre-registered force-in anchor.
    zero_pts = sorted(v for v, p in pmap.items() if p == 0)
    force_in = set(refit.get("force_in_spec", []) or refit.get("forced_in_anchors", []))
    if zero_pts:
        print(f"  [REPORT] point-role variable(s) at Points == 0: {zero_pts}")
        hit = sorted(set(zero_pts) & force_in)
        if hit:
            print(f"  [REPORT] *** PRE-REGISTERED FORCE-IN ANCHOR(S) AT 0 POINTS: "
                  f"{hit} *** — report prominently; the force-in is pre-registered "
                  f"in the locked SAP and cannot be quietly changed.")

    obs_min, obs_max = int(score_min), int(score_max)
    print(f"  OBSERVED pelvic_score range in the v3 test set : {obs_min:+d} .. {obs_max:+d}")
    print(f"  v2 OBSERVED range (comparison only, G4c)       : "
          f"{V2_OBSERVED_SCORE_RANGE[0]:+d} .. {V2_OBSERVED_SCORE_RANGE[1]:+d}  "
          f"-> observed max moved {obs_max - V2_OBSERVED_SCORE_RANGE[1]:+d}")
    print(f"  THEORETICAL points-column sums (NOT observed)  : "
          f"unconstrained {min_unc:+d}..{max_unc:+d}; "
          f"mutually-exclusive-respecting {min_mx:+d}..{max_mx:+d}")
    print(f"  mutex groups applied: {groups_used}")

    override_block = None
    if len(override_rows):
        r = override_rows.iloc[0]
        ovar = str(r["Variable"]).strip()
        col_in_df = ovar if ovar in df.columns else None
        gmask = override
        n_o = int(gmask.sum())
        ev_o = int(y_all[gmask].sum())
        lo_o, hi_o = wilson(ev_o, n_o)
        # marginal contribution of the override at the DEPLOYED rule-in cutoff
        if np.isfinite(escalate_cut):
            base_flag = score_all >= escalate_cut
            marg_f = int((~base_flag & gmask).sum())
            marg_e = int((~base_flag & gmask & (y_all == 1)).sum())
        else:
            marg_f = marg_e = None

        def _grate(col):
            if col not in df.columns or not n_o:
                return None
            return float(pd.to_numeric(df.loc[gmask, col], errors="coerce").fillna(0).mean())

        override_block = {
            "variable": ovar,
            "label": OVERRIDE_LABEL,
            "column_in_scored_parquet": OVERRIDE_COL,
            "points": None,
            "in_point_sum": False,
            "role": "override",
            "beta": _num(r, bcol), "se": _num(r, secol), "or": _num(r, orcol),
            "ci_low": _num(r, locol), "ci_high": _num(r, hicol),
            "n": n_o,
            "pct_of_test": (100.0 * n_o / len(df)) if len(df) else None,
            "events": ev_o,
            "event_rate": (ev_o / n_o) if n_o else None,
            "event_rate_ci_low": (None if not np.isfinite(lo_o) else float(lo_o)),
            "event_rate_ci_high": (None if not np.isfinite(hi_o) else float(hi_o)),
            "outcome": OUTCOME_COL,
            f"{DEATH_COMPONENT}_rate": _grate(DEATH_COMPONENT),
            f"{DEATH_SENSITIVITY}_rate": _grate(DEATH_SENSITIVITY),
            f"{TX_COMPONENT}_rate": _grate(TX_COMPONENT),
            f"{HC_COMPONENT}_rate": _grate(HC_COMPONENT),
            "pelvic_score_in_group": {
                "min": int(score_all[gmask].min()) if n_o else None,
                "max": int(score_all[gmask].max()) if n_o else None,
                "median": float(np.median(score_all[gmask])) if n_o else None,
                "mean": float(score_all[gmask].mean()) if n_o else None,
            },
            "marginal_flagged_at_rule_in": marg_f,
            "marginal_events_at_rule_in": marg_e,
            "rule_text": OVERRIDE_RULE_TEXT,
            "deployed_rule": (f"escalate = (pelvic_score >= {escalate_cut}) OR "
                              f"({OVERRIDE_COL} == 1)"),
            "render_note": ("F9 must render this as a full-width banner, NOT as a "
                            "point row; it carries no point value."),
        }
        print(f"  override block: n={n_o:,} ({100*n_o/len(df):.3f}%), "
              f"composite rate={100*ev_o/n_o if n_o else float('nan'):.2f}% "
              f"[{100*lo_o:.2f}, {100*hi_o:.2f}], "
              f"marginal flags at rule-in={marg_f}, marginal events={marg_e}")
    else:
        print("  [WARN] no Role=='override' row in table3_points.csv — "
              "score_card['override'] will be null.")

    card = {
        "outcome": OUTCOME_COL,
        "outcome_definition": ("received_hc | death_hemorrhagic | "
                               "substantial_transfusion  (CONTRACTS §4, v3)"),
        "prevalence": overall_prev,
        "prevalence_source": "observed in the v3 TEST set",
        "dca_prevalence_anchor": PREV_ANCHOR,
        "dca_prevalence_anchor_source": anchor_src,
        "n_test": int(len(df)),
        "variables": variables,
        "n_point_variables": len(variables),
        "override": override_block,
        "score_range": {
            "min": obs_min,
            "max": obs_max,
            "basis": ("OBSERVED minimum and maximum of pelvic_score in the v3 TEST "
                      "set (point-role variables only). This is the number CONTRACTS "
                      "§13 G4c gates and the number the manuscript should quote as "
                      "'the card's realistic range'. It is NOT a column sum."),
            "n_test": int(len(df)),
            "v2_observed_reference": {
                "min": V2_OBSERVED_SCORE_RANGE[0],
                "max": V2_OBSERVED_SCORE_RANGE[1],
                "note": ("v2 OBSERVED range, carried as a hard-coded constant for the "
                         "G4c comparison ONLY — label it as a v2 comparison wherever "
                         "it is quoted (CONTRACTS §13 G5b)."),
            },
            "observed_max_delta_vs_v2": obs_max - V2_OBSERVED_SCORE_RANGE[1],
        },
        "score_range_theoretical": {
            "basis": ("Sum of the printed Points column over Role=='point' rows. A "
                      "THEORETICAL bound, never observed in any patient. Reported "
                      "separately from score_range per CONTRACTS §13 G4c; do not "
                      "compare it against the '+16' expectation in the plan."),
            "max_unconstrained": int(max_unc),
            "min_unconstrained": int(min_unc),
            "max_mutually_exclusive": int(max_mx),
            "min_mutually_exclusive": int(min_mx),
            # 05 reports [min, max]; the historical scalar key keeps its meaning
            # (the MAXIMUM) so downstream readers do not have to change.
            "unconstrained_sum_reported_by_05": theo_unc_max,
            "unconstrained_min_reported_by_05": theo_unc_min,
            "unconstrained_sum_reported_by_05_raw": theo_unc_raw,
            "g4c_unconstrained_sum_check": g4c_status,
            "zero_point_variables": zero_pts,
            "zero_point_force_in_anchors": sorted(set(zero_pts) & force_in),
            "mutually_exclusive_groups_applied": groups_used,
            "nesting_note": ("sbp_lt90/sbp_lt70 and si_ge0_9/si_ge1_0 are NESTED, not "
                             "mutually exclusive — a patient can satisfy both, so both "
                             "count toward the theoretical maximum."),
        },
        "operating_points": cutpoints,
        "rule_out_cutpoint": ruleout_cut,
        "rule_in_cutpoint": escalate_cut,
        "deployed_rule": (f"escalate = (pelvic_score >= {escalate_cut}) OR "
                          f"({OVERRIDE_COL} == 1)"),
        "risk_tiers": tier_rows,
        "risk_tier_definitions": {
            "rule_out": f"pelvic_score < {ruleout_cut}",
            "observe": f"{ruleout_cut} <= pelvic_score < {escalate_cut}",
            "escalate": f"pelvic_score >= {escalate_cut}",
            "override": (f"{OVERRIDE_COL} == 1 — reported separately; these patients "
                         f"ALSO appear in one of the three score tiers (parallel "
                         f"escalation path, not a fourth mutually-exclusive tier)"),
            "death_rate_component": DEATH_COMPONENT,
            "death_inhosp_rate_component": (f"{DEATH_SENSITIVITY} (v2 all-cause "
                                            f"mortality, named sensitivity)"),
        },
        "operating_tables": {
            "table5_operating.csv": "sens-90 rule-out / spec-90 rule-in / Youden (v2 schema)",
            "table5c_operating_full.csv": ("Fix 5 — every integer cutoff, score vs SI vs "
                                           "rSIG, score-only AND score-OR-override"),
            "table5d_matched_workload.csv": ("Fix 5 — equal-workload comparison; the "
                                             "WTA abstract's headline claim"),
        },
        "common_complete_case": {
            "n": int(common.sum()),
            "n_test": int(len(df)),
            "pct": 100.0 * float(common.mean()),
            "note": ("Rows where the score, the shock index AND rSIG are all "
                     "observed. table5d uses this set so that events caught are "
                     "comparable across instruments. The score alone is defined for "
                     "100% of rows, so this restriction is conservative for the "
                     "score: the excluded patients have unobtainable vitals."),
        },
    }
    with open(OUT_CARD, "w") as f:
        json.dump(card, f, indent=2)
    print(f"  wrote {OUT_CARD}")
    print(f"  RULE-IN cutpoint  = pelvic_score >= {escalate_cut}")
    print(f"  RULE-OUT cutpoint = pelvic_score <  {ruleout_cut}")
    print(f"  DEPLOYED rule     = {card['deployed_rule']}")

    banner(f"Step 08 complete — runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
