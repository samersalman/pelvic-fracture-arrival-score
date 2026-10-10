"""
Pelvic Arrival Tool v3 — Step 12: Temporal external validation (TRIPOD-3)
================================================================================

TRIPOD Type-3 temporal external validation of the pelvic-hemorrhage arrival
score. This is the *drift detector*: it re-ESTIMATES the frozen model on the
early admission years (2019-2022) and validates on the later years (2023-2024).

Design (CONTRACTS §7 row 12, §12; refit.json is the SPECIFICATION source):
  * Variable SELECTION is FIXED — the predictors in refit.json["variables_kept"]
    (chosen by LASSO on the random split). Temporal validation does NOT re-run
    LASSO; it only re-estimates the coefficients on the temporal-derivation years.
  * Derive (2019-2022): unpenalized logistic of the outcome on those predictors.
    Freeze the resulting intercept + slopes. Also convert to an integer card
    via round(beta / |min positive beta|), WITH 05_refit_integer.py's ±1 floor
    (a variable whose |beta| clears the selection cutoff but whose ratio rounds
    to 0 gets ±1, sign preserved), for true parity with the random-split card.
  * Validate (2023-2024): apply the frozen temporal model (no recalibration) and
    report AUROC (2000x bootstrap CI) plus the drift metrics that actually tell
    the TRIPOD-3 story: CITL (calibration-in-the-large), calibration slope,
    decile reliability, Brier.
  * Three outcomes, same derive -> validate cycle:
      1. COMPOSITE (headline)        major_hemorrhage_event
      2. MORTALITY ANCHOR            death_hemorrhagic          <-- v3 Fix 2/2b
      3. ANY-TRANSFUSION (COVID)     blood_4h > 0
    Because COVID (2020-2021) sits in the derivation years, the any-transfusion
    co-run is retained as the COVID-sensitive check. The primary composite stays
    the headline.

v3 CHANGES vs the v2 fork (CONTRACTS §0, §4, §5, §9-J):
  * Fix 2 + 2b — the mortality-only anchor is `death_hemorrhagic`, NOT
    `death_inhosp`. `death_inhosp` appears nowhere in this script; the all-cause
    arm is a named sensitivity owned by 13, not a temporal-validation outcome.
    The column is CONSUMED BY NAME and never recomputed here, so the exact
    restriction (all-cause death, restricted to head AIS < 4 and to an early-death
    window) is whatever `01` last emitted — `01` owns that definition and it has
    moved once already, from a calendar-day cut to true elapsed hours.
  * Fix 3 — `unstable_ring` is out of the SCORE. The frozen weight vector is
    therefore SHORTER than v2's. The load is NAME-BASED (never positional) and
    asserts, loudly: every variables_kept name exists in both temporal frames,
    the predictor block is NaN-free, and `unstable_ring` is absent. A shorter
    vector silently misaligned against a wider design matrix is the one failure
    mode this script must not have.
  * Fix 4 — `sbp_unrecordable` is an OVERRIDE, not a point row (CONTRACTS §5.1).
    It STAYS in the continuous frozen linear predictor (so every other beta is
    adjusted for it) but is EXCLUDED from the integer card: excluded from the
    |beta_min_positive| anchor search and assigned no points. Discrimination is
    reported over the FULL validation set with override patients scored on the
    point scale like everyone else (CONTRACTS §5.1, row `07/09/12`).
  * No hard-coded v2 AUROC anchor. The internal random-split reference is read
    from tables/table4_discrimination.csv at run time (read-only) or reported as
    NaN — never the superseded 0.876 constant (CONTRACTS §13 G5b).
  * D-10 / G5h — DIFFERENTIAL TIMING ASCERTAINMENT ACROSS THIS EXACT SPLIT.
    Timing missingness among deaths is year-skewed and the skew lands on the
    2019-2022 / 2023-2024 boundary used here. Because an untimed death is
    conservatively counted as NOT within 48 h, `death_hemorrhagic` is
    under-ascertained MORE in derivation than in validation, so the mortality
    arm's CITL/slope is partly DATA COLLECTION, NOT MODEL DECAY. This script
    therefore:
      (a) reads the per-split ascertainment figures from logs/qc_manifest.json
          (01 owns them; nothing is retyped) and prints the attribution;
      (b) runs the pre-specified check — re-derive/re-validate the mortality arm
          on the UPPER-BOUND rung `death_hemorrhagic_missing_incl`, which keeps
          every untimed death; a material shrink in |CITL| attributes the offset
          to ascertainment;
      (c) writes both to derived/citl_slope.json under the top-level key
          `differential_ascertainment_d10`, which 16 must quote;
      (d) REFUSES to print an unqualified temporal-drift reading for the
          mortality arm when the attribution is unavailable, and exits non-zero.
    The check is a DIAGNOSTIC: it never becomes a fourth row of
    table4b_temporal.csv, whose 3-row + `role` schema is fixed by CONTRACTS
    §7.1 and consumed unchanged by 16.
  * Fix 9 / D-11 — the INTEGER-SCALE ANCHOR is amended, and this script MIRRORS
    the amended rule (CONTRACTS §5.3, §13 G8e). This is a DEVIATION from the
    locked SAP's literal integer-conversion rule, not a bugfix.

      LOCKED SAP: integer points round(beta/|beta_min_pos|) with a ±1 floor, and
      force-in anchors immune to the |beta| >= 0.15 screen. Read literally,
      |beta_min_pos| ranges over EVERY retained point-role variable — including a
      force-in anchor retained only because it was pre-registered.

      WHAT IT PRODUCED: under the v3 outcome the force-in anchor `cc_anticoag`
      collapsed to beta = 0.062 (OR 1.06, 95% CI 0.976-1.159, non-significant),
      because restricting the death arm to early hemorrhagic death removed the
      late deaths in which anticoagulation mattered. It became the denominator
      and divided every other coefficient by 0.062: mech_fall -27, sbp_lt90 +18,
      gcs_motor_le5 +15, observed range -35..+96 (v2 was -7..+32). That defeats
      Fix 4 / D-4, whose whole purpose was the objection that +16 is
      implausible at the bedside.

      AMENDED RULE (identical in 05, 12 and 13):
          anchor = min{ beta_v : v in point_vars, beta_v > 0, |beta_v| >= BETA_DROP }

      JUSTIFICATION: a coefficient retained by pre-registration rather than by
      evidence should not set the unit of the scale. The screen exists precisely
      to identify coefficients large enough to be meaningful; the denominator
      should come from that set. The force-in anchor is still retained and
      reported — it simply no longer defines the unit. A force-in anchor may
      therefore legitimately score 0 points; it is NOT special-cased up to ±1 and
      it is NOT dropped from the card.

    PARITY IS ASSERTED HERE. This script's own literal rule string is compared
    against `refit.json["anchor_selection_rule"]` (which 05 writes) and a
    mismatch is a hard failure — the three implementations cannot be allowed to
    drift. The anchor variable itself is also compared and reported, with the
    caveat that 12 re-estimates on 2019-2022 rather than the random 2/3 TRAIN
    split, so a *different* anchor variable is possible without the rule
    differing; that comparison is recorded, never silently swallowed.

Why refit rather than deploy the published card here: the random-split integer
card (refit.json["points"]) was trained on a split that mixes 2023-2024 rows into
its TRAIN partition, so applying it to the 2023-2024 valid set would leak. The
only clean temporal test re-estimates on 2019-2022 and evaluates on 2023-2024.

Inputs
------
  v3/derived/candidates.parquet            (filter cohort_primary == 1)
  v3/derived/refit.json                    (variables_kept/point_vars -> SPEC)
  v3/derived/temporal_derive_inc_keys.csv  (2019-2022, N=151,266)
  v3/derived/temporal_valid_inc_keys.csv   (2023-2024, N=88,470)
  v3/tables/table4_discrimination.csv      (READ-ONLY internal reference; optional)
  v3/logs/qc_manifest.json                 (READ-ONLY; 01 owns it) — the D-10
      per-split ascertainment figures at
      finaldischarge_timing.ascertainment_by_split

Outputs (owned)
---------------
  v3/tables/table4b_temporal.csv           (EXACTLY 3 rows + `role`: composite +
                                            mortality + any-tx. §7.1 fixes this
                                            schema; the D-10 diagnostic is NOT a
                                            fourth row.)
  v3/derived/temporal_roc_data.pkl         (fig-data for render_figures F3 temporal overlay;
                                            `_meta.d10_attribution` carries a D-10 summary)
  v3/derived/citl_slope.json               (full drift-metric record, incl. the
                                            top-level `differential_ascertainment_d10`
                                            block that 16 must quote — G5h)

NEVER writes table4_discrimination.csv / table4_delong.csv (owned by 07).

Exit status: non-zero when G5h cannot be evaluated, after the artifacts are
written — so an unattributed mortality-arm calibration cannot pass silently.

Run
---
    python 12_temporal_external_validation.py
    python 12_temporal_external_validation.py --sample 50000     # smoke

Seed: 42
"""
from __future__ import annotations

import os

# Defensive: CONTRACTS §13 G7a requires the Agg backend before ANY pyplot import,
# in every script. This one emits fig-DATA only and never draws, but a future edit
# that adds a diagnostic plot must not be able to break headless reproducibility.
import matplotlib

matplotlib.use("Agg")

import argparse
import json
import pickle
import textwrap
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SEED = 42
N_BOOT_DEFAULT = 2000
EPS = 1e-6

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES
LOGS = config.LOGS

IN_PARQUET = os.path.join(DERIVED, "candidates.parquet")
IN_REFIT = os.path.join(DERIVED, "refit.json")
IN_DERIVE_KEYS = os.path.join(DERIVED, "temporal_derive_inc_keys.csv")
IN_VALID_KEYS = os.path.join(DERIVED, "temporal_valid_inc_keys.csv")
IN_TABLE4 = os.path.join(TABLES, "table4_discrimination.csv")   # READ-ONLY reference
IN_QC_MANIFEST = os.path.join(LOGS, "qc_manifest.json")         # READ-ONLY (01 owns it)

OUT_TABLE = os.path.join(TABLES, "table4b_temporal.csv")
OUT_PICKLE = os.path.join(DERIVED, "temporal_roc_data.pkl")
OUT_JSON = os.path.join(DERIVED, "citl_slope.json")

COMPOSITE_OUTCOME = "major_hemorrhage_event"
# v3 Fix 2/2b — the mortality anchor. NOT death_inhosp (CONTRACTS §4, §9-J).
MORTALITY_OUTCOME = "death_hemorrhagic"
ANYTX_OUTCOME = "any_transfusion"

# ---------------------------------------------------------------------------
# D-10 / CONTRACTS §13 G5h — differential timing ascertainment across THIS split
# ---------------------------------------------------------------------------
# Timing missingness among deaths is year-skewed and the skew sits exactly on the
# 2019-2022 / 2023-2024 boundary this script uses to measure drift. Because a
# death with no `finaldischarge_hrs` is conservatively counted as NOT within 48 h,
# `death_hemorrhagic` is UNDER-ASCERTAINED more in derivation than in validation.
# A CITL offset produced by that is data collection, not model decay — so the
# mortality arm's calibration cannot be read as drift without the attribution.
#
# The pre-specified test (CONTRACTS §3.4): re-run the mortality arm on the
# UPPER-BOUND rung, which keeps every untimed death. If |CITL| shrinks materially
# under it, the offset was ascertainment. Reported in the log and in
# citl_slope.json ONLY — never as a fourth row of table4b_temporal.csv, whose
# 3-row + `role` schema is fixed by CONTRACTS §7.1 and consumed by 16.
BOUNDING_OUTCOME = "death_hemorrhagic_missing_incl"   # rung 4, upper bound

# Materiality rule for "shrinks materially". CONTRACTS §3.4 does not fix a
# number, so this threshold is a REPORTING CONVENTION declared here and labelled
# as such in the JSON — both raw |CITL| values are emitted so a reader can apply
# their own. Relative reduction in the ABSOLUTE offset.
D10_MATERIAL_REL_SHRINK = 0.25

# The manifest block 01 writes. Read, never retyped (CONTRACTS §3.1: no script
# may hard-code the realized gates).
D10_MANIFEST_PATH = ("finaldischarge_timing", "ascertainment_by_split")
D10_DERIVE_KEYS = ("derivation_2019_2022", "derivation", "derive")
D10_VALID_KEYS = ("validation_2023_2024", "validation", "valid")

DERIVE_YEARS = "2019-2022"
VALID_YEARS = "2023-2024"
# Cohort funnel is untouched by v3 (CONTRACTS §3.1): 151,266 + 88,470 = 239,736.
N_DERIVE_EXPECTED = 151266
N_VALID_EXPECTED = 88470

# Fix 3 — variables that must NOT appear in the score's frozen weight vector.
FORBIDDEN_SCORE_VARS = ("unstable_ring",)
# Fix 4 — variables that stay in the linear predictor but carry NO integer points.
DEFAULT_OVERRIDE_VARS = ("sbp_unrecordable",)

# Integer-conversion |beta| floor, mirroring 05_refit_integer.py's BETA_DROP.
# 05 applies a +/-1 FLOOR after rounding: a retained variable whose |beta| clears
# the selection cutoff but whose beta/anchor ratio rounds to 0 is given +/-1
# rather than 0 (05_refit_integer.py, "if pts == 0 and abs(b) >= BETA_DROP").
# This script claims PARITY with the random-split card, so it must apply the
# identical rule — otherwise the same variable can score 0 here and +/-1 there
# and the two cards are not comparable. Read from refit.json["beta_cutoff"]
# (05 writes it) so the two scripts cannot drift apart; the literal below is a
# fallback for a pre-Fix-4 artifact only, never the source of truth.
BETA_DROP_FALLBACK = 0.15

# ---- Fix 9 / D-11: the integer-scale anchor rule (CONTRACTS §5.3) -----------
# PARITY TOKEN. 05_refit_integer.py writes this exact string into refit.json;
# 13_sensitivity_components.py carries the same literal. It must stay
# BYTE-IDENTICAL in all three scripts and is asserted against refit.json below
# (§13 G8e). It says "BETA_DROP" rather than "0.15" so moving the cutoff cannot
# make the recorded rule lie.
ANCHOR_SELECTION_RULE = (
    "anchor = min{ beta_v : v in point_vars, beta_v > 0, |beta_v| >= BETA_DROP }; "
    "variables retained solely by force-in and sitting below the BETA_DROP screen "
    "are excluded from the anchor search (CONTRACTS §5.3 / SAP D-11)"
)

# §13 G8c — card-scale tripwire. A |points| value this large means the DIVISOR
# collapsed. Reported loudly here (not raised): 12 owns a secondary, re-estimated
# card, and aborting temporal validation over a scale artefact would hide the
# drift metrics that are this script's actual product. 05 raises.
CARD_SCALE_TRIPWIRE = 20

# Force-in anchors, mirrored from 05 so a stale refit.json without
# `force_in_spec` still labels the anchor exclusions correctly (CONTRACTS §5).
FORCE_IN_ANCHORS_FALLBACK = ("age_65_74", "age_ge75", "cc_anticoag")

# Internal random-split reference — READ from table4_discrimination.csv at run
# time. NEVER a hard-coded constant: v2's 0.876 is superseded (CONTRACTS §13 G5b).
INTERNAL_AUROC_INTEGER = float("nan")
INTERNAL_AUROC_LOGIT = float("nan")


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _safe_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def bootstrap_auc_ci(y, s, n_boot, rng):
    """Percentile 95% CI for AUROC via row-level resampling."""
    n = len(y)
    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(yb, s[idx])
    aucs = aucs[~np.isnan(aucs)]
    if aucs.size == 0:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def calibration_intercept(y, lp):
    """CITL = intercept of logit(P) = a + offset(LP); slope held at 1.
    a < 0 -> frozen model over-predicts on average; a > 0 -> under-predicts."""
    try:
        X = np.ones((len(y), 1), dtype=float)
        res = sm.GLM(np.asarray(y, dtype=float), X,
                     family=sm.families.Binomial(),
                     offset=np.asarray(lp, dtype=float)).fit()
        return float(res.params[0])
    except Exception as exc:
        print(f"  [CITL warn] {exc}")
        return np.nan


def calibration_slope(y, lp):
    """Calibration slope = beta of logit(P) = a + beta*LP.  ~1.0 = well-spread."""
    try:
        X = sm.add_constant(np.asarray(lp, dtype=float), has_constant="add")
        res = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False,
                                                         method="lbfgs", maxiter=200)
        return float(res.params[1])
    except Exception as exc:
        print(f"  [slope warn] {exc}")
        return np.nan


def hosmer_lemeshow(y_true, y_pred, n_groups=10):
    try:
        groups = pd.qcut(y_pred, n_groups, duplicates="drop")
        t = pd.DataFrame({"o": y_true, "p": y_pred, "g": groups})
        agg = t.groupby("g", observed=True).agg(obs=("o", "sum"), exp=("p", "sum"),
                                                 n=("o", "count")).reset_index()
        agg["obs_ne"] = agg["n"] - agg["obs"]
        agg["exp_ne"] = agg["n"] - agg["exp"]
        agg = agg[(agg["exp"] > 0) & (agg["exp_ne"] > 0)]
        hl = (((agg["obs"] - agg["exp"]) ** 2 / agg["exp"]).sum()
              + ((agg["obs_ne"] - agg["exp_ne"]) ** 2 / agg["exp_ne"]).sum())
        dfree = max(len(agg) - 2, 1)
        return float(hl), int(dfree), float(1 - sstats.chi2.cdf(hl, df=dfree))
    except Exception as exc:
        print(f"  [HL warn] {exc}")
        return np.nan, np.nan, np.nan


def decile_reliability(y, prob, n_groups=10):
    """Decile reliability table: mean predicted vs observed per predicted-risk decile."""
    try:
        g = pd.qcut(prob, n_groups, duplicates="drop", labels=False)
        t = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(prob, dtype=float),
                          "g": g})
        agg = (t.groupby("g", observed=True)
                 .agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
                 .reset_index())
        agg["decile"] = np.arange(1, len(agg) + 1)
        return agg[["decile", "n", "mean_pred", "obs_rate"]]
    except Exception as exc:
        print(f"  [reliability warn] {exc}")
        return pd.DataFrame(columns=["decile", "n", "mean_pred", "obs_rate"])


def roc_curve_thin(y, s, max_pts=1000):
    fpr, tpr, _ = roc_curve(y, s)
    if len(fpr) > max_pts:
        idx = np.linspace(0, len(fpr) - 1, max_pts).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return np.asarray(fpr), np.asarray(tpr)


# ---------------------------------------------------------------------------
# Frozen-model SPECIFICATION load — NAME-BASED, fails loudly (Fix 3 / Fix 4)
# ---------------------------------------------------------------------------
def load_frozen_spec():
    """Read refit.json and return (vars_kept, point_vars, override_vars, beta_drop).

    NAME-BASED by construction: every downstream matrix is built as
    ``frame[vars_kept]``, so a shorter/reordered v3 weight vector can never be
    silently misaligned against a wider design matrix. Every deviation from the
    v3 contract raises here rather than propagating into an AUROC.
    """
    if not os.path.exists(IN_REFIT):
        raise SystemExit(f"refit.json not found: {IN_REFIT} (run 05_refit_integer.py).")
    with open(IN_REFIT) as f:
        refit = json.load(f)

    vars_kept = list(refit["variables_kept"])
    if not vars_kept:
        raise SystemExit("refit.json['variables_kept'] is empty — no model to validate.")
    if len(set(vars_kept)) != len(vars_kept):
        dupes = sorted({v for v in vars_kept if vars_kept.count(v) > 1})
        raise SystemExit(f"refit.json['variables_kept'] contains duplicates: {dupes}")

    # ---- Fix 3 (CONTRACTS §0 row 3, §5, §13 G3): the ring is out of the SCORE ----
    banned = [v for v in FORBIDDEN_SCORE_VARS if v in vars_kept]
    if banned:
        raise SystemExit(
            f"CONTRACT VIOLATION (Fix 3 / §13 G3c): {banned} present in "
            f"refit.json['variables_kept']. `unstable_ring` is removed from the "
            f"SCORE in v3 (it is retained in the Analysis-B adjustment set only, "
            f"§6). Re-run 04/05 with the 17-variable pool before running 12.")

    # ---- Fix 4 (CONTRACTS §5.1): point-role vs override-role split ----
    if "point_vars" in refit and "override_vars" in refit:
        point_vars = list(refit["point_vars"])
        override_vars = list(refit["override_vars"])
        src = "refit.json"
    else:
        override_vars = [v for v in DEFAULT_OVERRIDE_VARS if v in vars_kept]
        point_vars = [v for v in vars_kept if v not in override_vars]
        src = "DERIVED (refit.json predates Fix 4)"
        print("  [CONTRACT WARN] refit.json is missing 'point_vars'/'override_vars' "
              f"(CONTRACTS §5.1(d)). Deriving them: override={override_vars}. "
              "This is a fallback, not the contract — fix 05_refit_integer.py.")

    unknown_pt = [v for v in point_vars if v not in vars_kept]
    unknown_ov = [v for v in override_vars if v not in vars_kept]
    if unknown_pt or unknown_ov:
        raise SystemExit(f"refit.json point_vars/override_vars name variables that are "
                         f"not in variables_kept: point={unknown_pt} override={unknown_ov}")
    overlap = sorted(set(point_vars) & set(override_vars))
    if overlap:
        raise SystemExit(f"CONTRACT VIOLATION (§5.1(e)): point_vars and override_vars "
                         f"overlap on {overlap} — they must be disjoint.")
    uncovered = [v for v in vars_kept if v not in set(point_vars) | set(override_vars)]
    if uncovered:
        raise SystemExit(f"variables_kept not covered by point_vars ∪ override_vars: {uncovered}")
    if not point_vars:
        raise SystemExit("No point-role variables — refuse to build an integer card.")

    # ---- Integer-conversion floor, taken from 05 rather than re-declared ----
    # The temporal card claims parity with the random-split card, so the rounding
    # rule must be 05's rule, not a look-alike. 05 writes its cutoff into
    # refit.json; reading it here is what keeps the two in lockstep if it moves.
    raw_cut = refit.get("beta_cutoff")
    try:
        beta_drop = float(raw_cut)
        if not np.isfinite(beta_drop) or beta_drop < 0:
            raise ValueError
        cut_src = "refit.json['beta_cutoff']"
    except (TypeError, ValueError):
        beta_drop = float(BETA_DROP_FALLBACK)
        cut_src = f"FALLBACK literal {BETA_DROP_FALLBACK}"
        print(f"  [CONTRACT WARN] refit.json carries no usable 'beta_cutoff' "
              f"(got {raw_cut!r}); using {beta_drop} so the ±1 integer floor still "
              f"matches 05_refit_integer.py. Fix 05 if this persists.")

    print(f"  model SPECIFICATION (name-based, refit.json): {len(vars_kept)} predictors")
    print(f"    {vars_kept}")
    print(f"    point-role   ({len(point_vars)}): {point_vars}")
    print(f"    override-role({len(override_vars)}): {override_vars}   [source: {src}]")
    print(f"    Fix 3 OK: none of {list(FORBIDDEN_SCORE_VARS)} in the frozen weight vector.")
    print(f"    integer ±1 floor at |β| >= {beta_drop}   [source: {cut_src}]")

    # ---- Fix 9 / D-11 — READ 05's anchor rule and anchor; assert parity -----
    # The RULE is read, never re-invented: a re-derivation that agrees today can
    # diverge tomorrow, a read cannot. §13 G8e makes a rule mismatch a hard fail.
    force_in = list(refit.get("force_in_spec")
                    or refit.get("forced_in_anchors")
                    or FORCE_IN_ANCHORS_FALLBACK)
    recorded_rule = refit.get("anchor_selection_rule")
    if recorded_rule is None:
        print("  [G8e][CONTRACT WARN] refit.json carries no 'anchor_selection_rule' "
              "(CONTRACTS §5.3 / §13 G8b). That artifact predates D-11. This script "
              "applies the AMENDED rule regardless; re-run 05_refit_integer.py so "
              "the two are provably the same rule.")
    elif str(recorded_rule) != ANCHOR_SELECTION_RULE:
        raise SystemExit(
            "CONTRACT VIOLATION (§13 G8e): the integer-anchor rule recorded by "
            "05_refit_integer.py differs from this script's implementation. The "
            "rule is duplicated in 05/12/13 and MUST be identical, or the temporal "
            "card is scaled differently from the headline card while both look "
            f"correct.\n  refit.json: {recorded_rule!r}\n  12 (this):  "
            f"{ANCHOR_SELECTION_RULE!r}")
    refit_anchor = {"var": refit.get("anchor_min_positive_var"),
                    "beta": refit.get("anchor_min_positive_beta"),
                    "eligible_vars": refit.get("anchor_eligible_vars"),
                    "excluded": refit.get("anchor_excluded_forced_in"),
                    "rule": recorded_rule}
    print(f"    D-11 anchor rule    : READ from refit.json and matched "
          f"({'present' if recorded_rule else 'ABSENT — stale artifact'})")
    print(f"    05's anchor         : {refit_anchor['var']} "
          f"(β={refit_anchor['beta']})   [the random-split card's divisor]")
    print(f"    force-in anchors    : {force_in}")
    return vars_kept, point_vars, override_vars, beta_drop, force_in, refit_anchor


# ---------------------------------------------------------------------------
# D-10 / G5h — read the split ascertainment figures from 01's manifest
# ---------------------------------------------------------------------------
def load_ascertainment():
    """Split-level timing-ascertainment figures, or (None, reason).

    Every number is READ from logs/qc_manifest.json, which 01 owns — nothing is
    retyped here (CONTRACTS §3.1). Returns (block_or_None, reason_or_None); the
    caller turns a None into a G5h FAIL rather than a silent skip, because a
    gate that cannot be evaluated is a FAIL (CONTRACTS §13 preamble).
    """
    if not os.path.exists(IN_QC_MANIFEST):
        return None, (f"{IN_QC_MANIFEST} not found — 01_build_cohort.py owns it and "
                      f"must run before 12.")
    try:
        with open(IN_QC_MANIFEST) as f:
            man = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return None, f"qc_manifest.json unreadable: {exc}"

    node = man
    for key in D10_MANIFEST_PATH:
        if not isinstance(node, dict) or key not in node:
            return None, (f"qc_manifest.json has no "
                          f"`{'.'.join(D10_MANIFEST_PATH)}` block (stopped at "
                          f"`{key}`). 01 must emit the per-split timing "
                          f"ascertainment (CONTRACTS §3.4).")
        node = node[key]
    if not isinstance(node, dict):
        return None, f"`{'.'.join(D10_MANIFEST_PATH)}` is not an object."

    def _pick(cands):
        for c in cands:
            if isinstance(node.get(c), dict):
                return node[c]
        return None

    der, val = _pick(D10_DERIVE_KEYS), _pick(D10_VALID_KEYS)
    if der is None or val is None:
        return None, (f"ascertainment_by_split lacks a derivation and/or validation "
                      f"sub-block (looked for {D10_DERIVE_KEYS} / {D10_VALID_KEYS}; "
                      f"found {sorted(node)}).")

    def _num(d, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    return fv
        return None

    out = {
        "source": "logs/qc_manifest.json -> " + ".".join(D10_MANIFEST_PATH),
        "derivation": {
            "years": der.get("years"), "n": _num(der, "n"),
            "n_deaths": _num(der, "n_deaths"),
            "deaths_missing_timing_n": _num(der, "deaths_missing_n"),
            "deaths_missing_timing_pct": _num(der, "deaths_missing_pct"),
            "death_hemorrhagic_n": _num(der, "death_hemorrhagic_n"),
            "composite_rate_pct": _num(der, "composite_rate_pct"),
        },
        "validation": {
            "years": val.get("years"), "n": _num(val, "n"),
            "n_deaths": _num(val, "n_deaths"),
            "deaths_missing_timing_n": _num(val, "deaths_missing_n"),
            "deaths_missing_timing_pct": _num(val, "deaths_missing_pct"),
            "death_hemorrhagic_n": _num(val, "death_hemorrhagic_n"),
            "composite_rate_pct": _num(val, "composite_rate_pct"),
        },
    }
    diff = _num(node, "differential_pp")
    dp, vp = (out["derivation"]["deaths_missing_timing_pct"],
              out["validation"]["deaths_missing_timing_pct"])
    if diff is None and dp is not None and vp is not None:
        diff = round(dp - vp, 3)
    out["differential_pp"] = diff

    if dp is None or vp is None or diff is None:
        return None, ("the ascertainment block is present but missing the "
                      "per-split `deaths_missing_pct` needed for the attribution.")
    return out, None


def _direction_sentence(asc):
    """The attribution sentence G5h requires, built from the manifest numbers."""
    d, v = asc["derivation"], asc["validation"]
    return (
        f"Timing missingness among in-hospital deaths is differential across the "
        f"temporal split: {d['deaths_missing_timing_pct']:.3f} % of derivation "
        f"({DERIVE_YEARS}) deaths carry no elapsed-hours value versus "
        f"{v['deaths_missing_timing_pct']:.3f} % in validation ({VALID_YEARS}) — a "
        f"differential of {asc['differential_pp']:+.3f} percentage points. Because "
        f"an untimed death is conservatively counted as NOT within 48 h, "
        f"`{MORTALITY_OUTCOME}` is UNDER-ASCERTAINED MORE IN DERIVATION "
        f"({int(d['death_hemorrhagic_n']):,} events) than in validation "
        f"({int(v['death_hemorrhagic_n']):,} events). Any calibration-in-the-large "
        f"offset or slope deviation reported for the mortality arm is therefore "
        f"PARTLY ATTRIBUTABLE TO DATA COLLECTION, NOT MODEL DECAY, and must be "
        f"attributed as such rather than read as temporal drift.")


def run_d10_attribution(asc, dm, vm, vars_kept, point_vars, override_vars,
                        beta_drop, res_mort, n_boot, rng,
                        force_in_anchors=(), refit_anchor=None):
    """Pre-specified D-10 check: re-run the mortality arm on the UPPER-BOUND rung.

    `death_hemorrhagic_missing_incl` keeps every untimed death, so it removes the
    differential the conservative missing rule creates. If |CITL| shrinks
    materially under it, the offset was ascertainment rather than drift
    (CONTRACTS §3.4 / §13 G5h).

    Diagnostic ONLY: the result is written to the run log, citl_slope.json and
    the pickle `_meta`. It never becomes a row of table4b_temporal.csv, whose
    3-row + `role` schema is fixed by §7.1 and consumed unchanged by 16.
    """
    rec = {
        "test": ("re-run the mortality arm on the upper-bound rung "
                 f"`{BOUNDING_OUTCOME}` (every untimed death retained) and compare "
                 f"|CITL| with the primary arm `{MORTALITY_OUTCOME}`"),
        "bounding_outcome": BOUNDING_OUTCOME,
        "primary_outcome": MORTALITY_OUTCOME,
        "materiality_rule": (f"|CITL| shrinks by >= {D10_MATERIAL_REL_SHRINK:.0%} "
                             f"relative => the offset is attributed to ascertainment. "
                             f"This threshold is a REPORTING CONVENTION declared in 12, "
                             f"not a SAP-locked value; both raw |CITL| values are "
                             f"emitted so a reader may apply their own."),
        "reported_in": "citl_slope.json + run log only (NOT table4b_temporal.csv)",
    }

    missing = [nm for nm, fr in (("derive", dm), ("valid", vm))
               if BOUNDING_OUTCOME not in fr.columns]
    if missing:
        rec["status"] = "NOT_EVALUABLE"
        rec["reason"] = (f"`{BOUNDING_OUTCOME}` absent from the {missing} frame(s). "
                         f"02_engineer_features.py carries it in REQUIRED_CARRY_COLS "
                         f"(CONTRACTS §7 row 02); re-run 02 before 12.")
        return rec

    for fr in (dm, vm):
        col = pd.to_numeric(fr[BOUNDING_OUTCOME], errors="coerce")
        if col.isna().any():
            rec["status"] = "NOT_EVALUABLE"
            rec["reason"] = (f"`{BOUNDING_OUTCOME}` has {int(col.isna().sum())} NaN — "
                             f"outcomes must be int 0/1.")
            return rec
        fr[BOUNDING_OUTCOME] = col.astype(int)

    # Nesting invariant: the bounding arm is a SUPERSET of the primary arm.
    # If it is not, the two are not the same ladder and the comparison is void.
    for nm, fr in (("derive", dm), ("valid", vm)):
        viol = int(((fr[MORTALITY_OUTCOME] == 1) & (fr[BOUNDING_OUTCOME] == 0)).sum())
        if viol:
            rec["status"] = "NOT_EVALUABLE"
            rec["reason"] = (f"{viol:,} rows in the {nm} frame have "
                             f"{MORTALITY_OUTCOME}=1 but {BOUNDING_OUTCOME}=0. The "
                             f"recovery ladder must be nested (CONTRACTS §13 G2d); "
                             f"01 emitted them from divergent code.")
            return rec

    boot = max(1, min(int(n_boot), 500))     # a CI is not what this test turns on
    res_bound = run_temporal(
        f"D-10 ATTRIBUTION (upper-bound arm: {BOUNDING_OUTCOME})", BOUNDING_OUTCOME,
        dm, vm, vars_kept, point_vars, override_vars, beta_drop, boot, rng,
        force_in_anchors=force_in_anchors, refit_anchor=refit_anchor)

    c_pri = float(res_mort["citl_offset_intercept"])
    c_bnd = float(res_bound["citl_offset_intercept"])
    ok = bool(np.isfinite(c_pri) and np.isfinite(c_bnd))
    shrink_abs = (abs(c_pri) - abs(c_bnd)) if ok else float("nan")
    shrink_rel = (shrink_abs / abs(c_pri)) if (ok and abs(c_pri) > 1e-12) else float("nan")

    rec.update({
        "status": "EVALUATED" if ok else "NOT_EVALUABLE",
        "n_boot_diagnostic": boot,
        "primary": {
            "outcome": MORTALITY_OUTCOME,
            "events_derive": res_mort["events_derive"],
            "events_valid": res_mort["events_valid"],
            "citl_offset_intercept": c_pri,
            "abs_citl": abs(c_pri) if ok else float("nan"),
            "calibration_slope": res_mort["calibration_slope"],
            "auroc_logit": res_mort["auroc_logit"],
        },
        "bounding": {
            "outcome": BOUNDING_OUTCOME,
            "events_derive": res_bound["events_derive"],
            "events_valid": res_bound["events_valid"],
            "citl_offset_intercept": c_bnd,
            "abs_citl": abs(c_bnd) if ok else float("nan"),
            "calibration_slope": res_bound["calibration_slope"],
            "auroc_logit": res_bound["auroc_logit"],
        },
        "abs_citl_shrinkage": shrink_abs,
        "abs_citl_shrinkage_relative": shrink_rel,
    })
    if not ok:
        rec["reason"] = "a CITL offset was non-finite; the comparison is undefined."
        rec["verdict"] = "undetermined"
        return rec

    if shrink_rel >= D10_MATERIAL_REL_SHRINK:
        rec["verdict"] = "ascertainment_attributable"
        rec["verdict_text"] = (
            f"|CITL| falls from {abs(c_pri):.4f} to {abs(c_bnd):.4f} "
            f"({shrink_rel:.1%}) when every untimed death is retained. Under the "
            f"pre-specified D-10 check the mortality arm's offset is attributed to "
            f"DIFFERENTIAL TIMING ASCERTAINMENT, not to model decay.")
    else:
        rec["verdict"] = "not_explained_by_ascertainment"
        rec["verdict_text"] = (
            f"|CITL| moves only from {abs(c_pri):.4f} to {abs(c_bnd):.4f} "
            f"({shrink_rel:+.1%}) when every untimed death is retained, below the "
            f"{D10_MATERIAL_REL_SHRINK:.0%} materiality rule. The offset is NOT "
            f"explained by the ascertainment differential alone — but the "
            f"differential is real and still qualifies the estimate.")
    return rec


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_analytic(vars_kept, sample_n, rng_seed=SEED):
    """Load analytic cohort, split into temporal derive/valid frames."""
    df = pd.read_parquet(IN_PARQUET)
    prim = df[df["cohort_primary"] == 1].copy()
    prim["inc_key"] = prim["inc_key"].astype(str).str.strip()
    prim["puf_year"] = prim["puf_year"].astype("Int64")

    der_keys = pd.read_csv(IN_DERIVE_KEYS)
    val_keys = pd.read_csv(IN_VALID_KEYS)
    for k in (der_keys, val_keys):
        k["inc_key"] = k["inc_key"].astype(str).str.strip()
        k["puf_year"] = k["puf_year"].astype("Int64")

    dm = prim.merge(der_keys, on=["puf_year", "inc_key"], how="inner")
    vm = prim.merge(val_keys, on=["puf_year", "inc_key"], how="inner")
    print(f"  analytic cohort (cohort_primary==1): {len(prim):,}")
    print(f"  temporal DERIVE {DERIVE_YEARS}: N={len(dm):,} (expect {N_DERIVE_EXPECTED:,})"
          f"  years={sorted(dm['puf_year'].dropna().unique().tolist())}")
    print(f"  temporal VALID  {VALID_YEARS}: N={len(vm):,} (expect {N_VALID_EXPECTED:,})"
          f"  years={sorted(vm['puf_year'].dropna().unique().tolist())}")

    # Derive the any-transfusion outcome (blood_4h > 0; NaN treated as no documented
    # transfusion, consistent with the composite's substantial_transfusion component).
    for frame in (dm, vm):
        b4 = pd.to_numeric(frame["blood_4h"], errors="coerce")
        frame[ANYTX_OUTCOME] = (b4 > 0).astype(int)
    miss_d = pd.to_numeric(dm["blood_4h"], errors="coerce").isna().sum()
    miss_v = pd.to_numeric(vm["blood_4h"], errors="coerce").isna().sum()
    print(f"  any-transfusion: blood_4h missing -> 0  (derive {miss_d:,}={100*miss_d/len(dm):.1f}%, "
          f"valid {miss_v:,}={100*miss_v/len(vm):.1f}%)")

    # ---- Outcome columns must be present BY NAME and 0/1 (Fix 2/2b, §9-J) ----
    for oc in (COMPOSITE_OUTCOME, MORTALITY_OUTCOME):
        if oc not in dm.columns or oc not in vm.columns:
            raise SystemExit(
                f"outcome column '{oc}' missing from candidates.parquet. v3 requires "
                f"it (CONTRACTS §4 / §7 row 02 REQUIRED_CARRY_COLS). If you are "
                f"looking for `death_inhosp`: it is a NAMED SENSITIVITY owned by 13, "
                f"not the v3 mortality anchor.")
        for nm, frame in (("derive", dm), ("valid", vm)):
            col = pd.to_numeric(frame[oc], errors="coerce")
            if col.isna().any():
                raise SystemExit(f"{oc} has {int(col.isna().sum())} NaN in the {nm} frame "
                                 f"— outcomes must be int 0/1.")
            frame[oc] = col.astype(int)

    if sample_n and sample_n < min(len(dm), len(vm)):
        dm = dm.sample(n=sample_n, random_state=rng_seed).reset_index(drop=True)
        vm = vm.sample(n=sample_n, random_state=rng_seed).reset_index(drop=True)
        print(f"  [SMOKE] downsampled derive={len(dm):,} valid={len(vm):,}")

    # ---- HARD gate: the predictor block must exist BY NAME and be NaN-free ----
    # 02_engineer_features.py asserts the X matrix is numeric and NaN-free; if that
    # ever regresses, a NaN here would silently poison `X @ betas` and produce a
    # NaN AUROC rather than an error. Fail loudly instead.
    for nm, frame in (("derive", dm), ("valid", vm)):
        absent = [v for v in vars_kept if v not in frame.columns]
        if absent:
            raise SystemExit(f"frozen-weight variables absent from the {nm} frame "
                             f"(name-based load, no positional fallback): {absent}")
        miss = frame[vars_kept].isna().sum()
        if miss.sum():
            raise SystemExit(f"{nm} frame has NaN among the frozen predictors — refusing "
                             f"to build a misaligned design matrix:\n{miss[miss > 0]}")
    print(f"  predictor block OK: {len(vars_kept)} columns present by name, NaN-free, "
          f"in both frames.")

    prev = {oc: (float(dm[oc].mean()), float(vm[oc].mean()))
            for oc in (COMPOSITE_OUTCOME, MORTALITY_OUTCOME, ANYTX_OUTCOME)}
    for oc, (pd_, pv_) in prev.items():
        print(f"  prevalence {oc:<28} derive={pd_:.4f}  valid={pv_:.4f}")

    return dm, vm, {"blood_4h_missing_derive": int(miss_d), "blood_4h_missing_valid": int(miss_v)}


# ---------------------------------------------------------------------------
# Core: derive -> validate for one outcome
# ---------------------------------------------------------------------------
def select_integer_anchor(beta_by_var, point_vars, beta_drop, force_in_anchors=(),
                          emit=print):
    """|beta_min_positive| under CONTRACTS §5.3 (Fix 9 / D-11). AMENDED RULE.

        anchor = min{ beta_v : v in point_vars, beta_v > 0, |beta_v| >= beta_drop }

    Copy-adapted from 05_refit_integer.py (house style: no cross-script imports).
    The two implementations must stay behaviourally identical; the rule string is
    asserted against refit.json's at load time (§13 G8e).

    The divisor is the smallest positive coefficient among POINT-ROLE variables
    that passed the pre-specified |beta| >= beta_drop screen ON THEIR OWN
    EVIDENCE. A variable retained *solely* by force-in and sitting below the
    screen is excluded FROM THE ANCHOR SEARCH ONLY — it stays in the model, stays
    in the linear predictor, stays on the card, and may legitimately score 0.

    This is a DEVIATION from the locked SAP's literal rule (D-11), not a bugfix.
    Rationale: a coefficient retained by pre-registration rather than by evidence
    should not set the unit of the scale; the screen exists precisely to identify
    coefficients large enough to be meaningful, and the denominator should come
    from that set.

    DEGENERATE FALLBACK: if no point-role variable clears the screen, fall back to
    the locked rule (min positive beta over all point-role vars) with a loud
    warning rather than crashing.

    Returns (anchor_beta, anchor_var, meta).
    """
    beta_drop = float(beta_drop)
    force_in_anchors = set(force_in_anchors or ())
    # Insertion order follows `point_vars` so ties break identically in 05/12/13.
    positive = {v: float(beta_by_var[v]) for v in point_vars
                if float(beta_by_var[v]) > 0}
    if not positive:
        raise SystemExit(
            "No positive-beta POINT-role variable — refuse to build an integer card.")

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
        emit("  " + "!" * 74)
        emit("  [D-11 DEGENERATE FALLBACK] NO point-role variable passed the "
             f"|beta| >= {beta_drop} screen on these rows.")
        emit("  The amended anchor rule (CONTRACTS §5.3) has an EMPTY eligible "
             "set, so this card falls back to the LOCKED SAP rule (min positive "
             "beta over all point-role variables).")
        emit(f"  Fallback anchor = {anchor:.4f} ({anchor_var}). Every point value "
             "on this card is scaled by a coefficient that did NOT clear the "
             "screen — inspect before quoting.")
        emit("  " + "!" * 74)

    meta = {
        "anchor_selection_rule": ANCHOR_SELECTION_RULE,
        "anchor_min_positive_var": anchor_var,
        "anchor_min_positive_beta": anchor,
        "anchor_eligible_vars": [v for v in point_vars if v in eligible],
        "anchor_eligible_betas": {v: round(float(eligible[v]), 6)
                                  for v in point_vars if v in eligible},
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

    # §13 G8a — the chosen anchor must be a point-role var that passed the screen.
    assert anchor_var in point_vars, \
        f"anchor {anchor_var!r} is not a point-role variable (CONTRACTS §5.1a)"
    assert anchor > 0, f"anchor beta must be positive; got {anchor}"
    if not fallback:
        assert abs(anchor) >= beta_drop, (
            f"§13 G8a VIOLATION: anchor {anchor_var!r} has |beta| = {abs(anchor):.4f} "
            f"< screen {beta_drop}. A variable retained solely by force-in must "
            f"not set the integer scale (CONTRACTS §5.3 / D-11).")
    return anchor, anchor_var, meta


def integer_points(beta, anchor, beta_drop):
    """05_refit_integer.py's integer rule, mirrored EXACTLY (CONTRACTS §7 row 05).

        pts = round(beta / |beta_min_positive|)
        if pts == 0 and |beta| >= beta_drop:  pts = +1 if beta > 0 else -1

    The anchor itself comes from select_integer_anchor() above (CONTRACTS §5.3 /
    D-11): it is the smallest positive point-role beta THAT PASSED THE SCREEN.
    A point-role variable sitting BELOW the screen correctly keeps 0 points here —
    the floor's condition (`|beta| >= beta_drop`) does not fire for it, and that
    is the intended behaviour, not an omission to patch.

    The floor is the part that is easy to lose: without it a variable that
    survived selection (|beta| >= the cutoff) but rounds to 0 against the anchor
    silently leaves the card here while carrying +/-1 on the random-split card,
    and the two integer cards stop being comparable — which is the only reason
    this script builds an integer card at all. Sign is preserved: a protective
    beta floors to -1, never +1.
    """
    pts = int(round(float(beta) / float(anchor)))
    if pts == 0 and abs(float(beta)) >= float(beta_drop):
        pts = 1 if float(beta) > 0 else -1
    # Same sign discipline 05 asserts: rounding must never flip a coefficient.
    if float(beta) > 0:
        assert pts >= 0, f"positive β {beta} produced negative points {pts}"
    else:
        assert pts <= 0, f"non-positive β {beta} produced positive points {pts}"
    return pts


def run_temporal(label, outcome_col, dm, vm, vars_kept, point_vars, override_vars,
                 beta_drop, n_boot, rng, force_in_anchors=(), refit_anchor=None):
    banner(f"[{label}] derive {DERIVE_YEARS} -> validate {VALID_YEARS}  "
           f"(outcome = {outcome_col})")

    yd = dm[outcome_col].to_numpy(dtype=int)
    # NAME-BASED design matrix: column order is dictated by vars_kept, never by
    # the frame's own column order.
    Xd = dm[vars_kept].to_numpy(dtype=float)
    Xd_c = sm.add_constant(Xd, has_constant="add")

    # Unpenalized logistic on the derivation years (variable set fixed).
    try:
        fit = sm.Logit(yd, Xd_c).fit(disp=False, maxiter=200)   # Newton default
        method = "newton"
    except Exception:
        fit = sm.Logit(yd, Xd_c).fit(disp=False, method="bfgs", maxiter=1000)
        method = "bfgs"
    intercept = float(fit.params[0])
    betas = np.asarray(fit.params[1:], dtype=float)
    print(f"  derive fit converged ({method}); N={len(yd):,} events={int(yd.sum()):,} "
          f"({100*yd.mean():.2f}%); intercept={intercept:+.4f}")

    # ---- Integer card from the temporal betas (parity with the random-split card) ----
    # Fix 4 / CONTRACTS §5.1: override-role variables STAY in the continuous linear
    # predictor but are EXCLUDED from the |beta_min_positive| anchor search and carry
    # NO points. Any other rule would smuggle a deterministic escalation trigger into
    # the rank-based AUROC as an implicit +inf point value.
    # Fix 9 / D-11 (CONTRACTS §5.3): the anchor is the smallest positive
    # point-role beta THAT PASSED the |beta| >= beta_drop screen. A variable
    # retained solely by force-in and sitting below the screen is excluded from
    # the search — it still carries its coefficient in the linear predictor and
    # still appears on the card, possibly at 0 points.
    coef_map = {v: float(b) for v, b in zip(vars_kept, betas)}
    min_pos, min_pos_var, anchor_meta = select_integer_anchor(
        coef_map, point_vars, beta_drop, force_in_anchors=force_in_anchors)
    # PARITY WITH 05: round(beta / |beta_min_positive|) WITH the ±1 floor. See
    # integer_points() — dropping the floor makes this card disagree with the
    # random-split card on exactly the variables the floor was written for.
    point_map = {v: integer_points(coef_map[v], min_pos, beta_drop)
                 for v in point_vars}
    floored = {v: round(coef_map[v], 4) for v in point_vars
               if point_map[v] != int(round(coef_map[v] / min_pos))}
    for v in override_vars:                       # explicit, auditable zero
        point_map[v] = 0
    points = np.array([point_map[v] for v in vars_kept], dtype=float)
    print(f"  min positive beta = {min_pos:.4f} ({min_pos_var}); integer card built over "
          f"{len(point_vars)} POINT-role vars")
    print(f"  D-11 anchor search: {anchor_meta['anchor_n_eligible']} eligible of "
          f"{anchor_meta['anchor_n_positive_point_vars']} positive-beta point-role "
          f"vars -> {anchor_meta['anchor_eligible_vars']}")
    if anchor_meta["anchor_excluded_forced_in"]:
        for e in anchor_meta["anchor_excluded_forced_in"]:
            tag = "FORCE-IN ANCHOR" if e["force_in_anchor"] else "point-role"
            print(f"    excluded from the anchor search: {e['variable']} "
                  f"(beta={e['beta']:+.4f}) [{tag}] — {e['reason']}")
    _locked = anchor_meta["anchor_locked_sap_would_have_chosen"]
    if _locked["variable"] != min_pos_var:
        print(f"  D-11 ACTIVE on this card: the locked SAP rule would have divided "
              f"by {_locked['beta']:.4f} ({_locked['variable']}); the amended rule "
              f"divides by {min_pos:.4f} ({min_pos_var}).")

    # ---- §13 G8e — three-script parity against 05's recorded anchor ---------
    # 12 must APPLY the same RULE as 05. It cannot always land on the same
    # anchor VARIABLE: it re-estimates on 2019-2022 while 05 fits the random 2/3
    # TRAIN split, so the coefficients differ by construction. The rule-string
    # equality is the hard gate (asserted at load); the variable comparison is
    # recorded and printed so a divergence is visible rather than silent.
    anchor_parity = {"refit_anchor_var": None, "refit_anchor_beta": None,
                     "this_anchor_var": min_pos_var, "this_anchor_beta": min_pos,
                     "same_anchor_var": None,
                     "note": ("12 re-estimates on the temporal derivation years, so a "
                              "DIFFERENT anchor variable does not by itself indicate a "
                              "rule divergence; the rule string is the hard parity gate "
                              "(CONTRACTS §13 G8e).")}
    if refit_anchor:
        anchor_parity["refit_anchor_var"] = refit_anchor.get("var")
        anchor_parity["refit_anchor_beta"] = refit_anchor.get("beta")
        anchor_parity["same_anchor_var"] = bool(refit_anchor.get("var") == min_pos_var)
        _rb = refit_anchor.get("beta")
        _rb_txt = f"{float(_rb):.4f}" if _rb is not None else "n/a"
        if anchor_parity["same_anchor_var"]:
            print(f"  [G8e] anchor parity vs refit.json: SAME variable "
                  f"({min_pos_var}); 05 beta={_rb_txt}, temporal beta={min_pos:.4f}")
        else:
            print(f"  [G8e][NOTE] anchor variable DIFFERS from the random-split card: "
                  f"05 used {refit_anchor.get('var')} (beta={_rb_txt}), this temporal "
                  f"card uses {min_pos_var} (beta={min_pos:.4f}). Same RULE, different "
                  f"rows — recorded in citl_slope.json; do not compare the two cards' "
                  f"point values without saying so.")
    else:
        print("  [G8e][CONTRACT WARN] refit.json recorded no anchor variable — the "
              "cross-script anchor comparison cannot be evaluated. Re-run 05.")

    # §13 G8c — card-scale tripwire (reported, not raised: see the constant).
    _oversized = {v: p for v, p in point_map.items()
                  if v in point_vars and abs(int(p)) > CARD_SCALE_TRIPWIRE}
    if _oversized:
        print(f"  [G8c][WARN] temporal card carries point value(s) beyond "
              f"±{CARD_SCALE_TRIPWIRE}: {_oversized}. That is the fingerprint of a "
              f"COLLAPSED ANCHOR — inspect anchor={min_pos_var} "
              f"(beta={min_pos:.4f}) before reading anything else on this card.")
    _zero_pts = sorted(v for v in point_vars if int(point_map[v]) == 0)
    if _zero_pts:
        _fi = sorted(set(_zero_pts) & set(force_in_anchors or ()))
        print(f"  point-role variable(s) at 0 points on the temporal card: {_zero_pts}"
              + (f"  *** incl. pre-registered force-in anchor(s) {_fi} — retained and "
                 f"reported at 0, never dropped (CONTRACTS §5.3) ***" if _fi else ""))
    if floored:
        print(f"  ±1 floor applied (|β| >= {beta_drop}, ratio rounded to 0) to "
              f"{len(floored)} var(s): "
              f"{ {v: point_map[v] for v in floored} }  [β = {floored}]  "
              f"— same rule as 05_refit_integer.py")
    else:
        print(f"  ±1 floor (|β| >= {beta_drop}): no point-role variable needed it")
    if override_vars:
        print(f"  override-role (0 points, in the linear predictor only): "
              f"{ {v: round(coef_map[v], 4) for v in override_vars} }")
        # Double-counting guard, mirroring 06 (§5.1): zeroing the override columns
        # must leave the integer score unchanged.
        Xv_chk = vm[vars_kept].copy()
        Xv_chk[override_vars] = 0.0
        s_full = vm[vars_kept].to_numpy(dtype=float) @ points
        s_zero = Xv_chk.to_numpy(dtype=float) @ points
        if not np.array_equal(s_full, s_zero):
            raise SystemExit("[override guard] zeroing override columns changed "
                             "pelvic_score — the override leaked into the point sum.")
        print("  override double-counting guard: PASS (zeroed-recompute identical)")

    # Apply frozen temporal model to the validation years (no recalibration).
    yv = vm[outcome_col].to_numpy(dtype=int)
    Xv = vm[vars_kept].to_numpy(dtype=float)
    lp_valid = intercept + Xv @ betas
    prob_valid = _sigmoid(lp_valid)
    score_valid = (Xv @ points).astype(float)

    # Discrimination + bootstrap CI (continuous frozen logit + integer card).
    auc_logit = float(roc_auc_score(yv, prob_valid))
    lo_l, hi_l = bootstrap_auc_ci(yv, prob_valid, n_boot, rng)
    auc_int = float(roc_auc_score(yv, score_valid))
    lo_i, hi_i = bootstrap_auc_ci(yv, score_valid, n_boot, rng)

    # Drift / calibration metrics on the frozen probabilities.
    citl = calibration_intercept(yv, lp_valid)               # offset-intercept (canonical)
    citl_meandiff = float(prob_valid.mean() - yv.mean())     # probability-scale (script-07 style)
    slope = calibration_slope(yv, lp_valid)
    brier = float(brier_score_loss(yv, prob_valid))
    hl, hl_df, hl_p = hosmer_lemeshow(yv, prob_valid)
    reliab = decile_reliability(yv, prob_valid)

    print(f"  VALID N={len(yv):,} events={int(yv.sum()):,} ({100*yv.mean():.2f}%)")
    print(f"  AUROC  logit  = {auc_logit:.4f} [{lo_l:.4f}, {hi_l:.4f}]")
    print(f"  AUROC  integer= {auc_int:.4f} [{lo_i:.4f}, {hi_i:.4f}]")
    print(f"  CITL(offset)  = {citl:+.4f}   CITL(mean-diff) = {citl_meandiff:+.5f}")
    print(f"  slope         = {slope:.4f}   Brier = {brier:.5f}   HL p = {hl_p:.3g}")
    print(f"  prevalence derive={yd.mean():.4f}  valid={yv.mean():.4f}  "
          f"(drop = {yd.mean()-yv.mean():+.4f})")

    # ROC curves for the F3 temporal overlay.
    fpr_l, tpr_l = roc_curve_thin(yv, prob_valid)
    fpr_i, tpr_i = roc_curve_thin(yv, score_valid)

    return {
        "label": label,
        "outcome": outcome_col,
        "method": method,
        "n_derive": int(len(yd)), "events_derive": int(yd.sum()),
        "prevalence_derive": float(yd.mean()),
        "n_valid": int(len(yv)), "events_valid": int(yv.sum()),
        "prevalence_valid": float(yv.mean()),
        "auroc_logit": auc_logit, "ci_logit": [lo_l, hi_l],
        "auroc_integer": auc_int, "ci_integer": [lo_i, hi_i],
        "citl_offset_intercept": citl, "citl_meandiff": citl_meandiff,
        "calibration_slope": slope, "brier": brier,
        "hl_chisq": hl, "hl_df": hl_df, "hl_p": hl_p,
        "reliability_deciles": reliab.to_dict(orient="records"),
        "temporal_intercept": intercept,
        "temporal_coefficients": coef_map,
        "temporal_points": point_map,
        "point_vars": list(point_vars), "override_vars": list(override_vars),
        "min_positive_beta": min_pos, "min_positive_var": min_pos_var,
        # Integer rule recorded with the card so parity with 05 is auditable
        # from the artifact, not only from the run log.
        "integer_rule": ("round(beta / |beta_min_positive|) with a ±1 floor at "
                         f"|beta| >= {beta_drop} — identical to 05_refit_integer.py"),
        "integer_beta_floor": float(beta_drop),
        "integer_floor_applied_to": {v: point_map[v] for v in floored},
        # ---- Fix 9 / D-11 provenance + §13 G8e parity record ----
        # Canonical field names, matching refit.json, so the gate can find them
        # without knowing which script wrote the block.
        **anchor_meta,
        "anchor_parity_vs_refit": anchor_parity,
        "zero_point_point_vars": sorted(v for v in point_vars
                                        if int(point_map[v]) == 0),
        "zero_point_force_in_anchors": sorted(
            {v for v in point_vars if int(point_map[v]) == 0}
            & set(force_in_anchors or ())),
        "_roc_logit": {"fpr": fpr_l, "tpr": tpr_l, "auc": auc_logit, "lo": lo_l, "hi": hi_l},
        "_roc_integer": {"fpr": fpr_i, "tpr": tpr_i, "auc": auc_int, "lo": lo_i, "hi": hi_i},
    }


def _load_internal_reference():
    """Read the internal random-split AUROC anchors from table4 (READ-ONLY).

    These describe the PRIMARY COMPOSITE on the random split and are therefore
    attached ONLY to the composite row. There is no hard-coded fallback: v2's
    0.876 is superseded by the v3 outcome redefinition (CONTRACTS §13 G5b), and
    stamping it into table4b would launder a v2 number into a v3 result. If
    table4 is unreadable the reference is reported as NaN.
    """
    global INTERNAL_AUROC_INTEGER, INTERNAL_AUROC_LOGIT
    try:
        t4 = pd.read_csv(IN_TABLE4)
        row_i = t4[t4["Score"] == "pelvic_score"]
        row_l = t4[t4["Score"] == "pelvic_logit"]
        if len(row_i):
            INTERNAL_AUROC_INTEGER = float(row_i.iloc[0]["AUROC"])
        if len(row_l):
            INTERNAL_AUROC_LOGIT = float(row_l.iloc[0]["AUROC"])
        print(f"  internal random-split ref (table4, v3 primary composite): "
              f"integer={INTERNAL_AUROC_INTEGER:.4f} logit={INTERNAL_AUROC_LOGIT:.4f}")
    except Exception as exc:
        print(f"  [WARN] table4_discrimination.csv not readable ({exc}); internal "
              f"random-split reference reported as NaN. No v2 constant is substituted.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstraps", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--sample", type=int, default=0,
                    help="Downsample derive AND valid to N rows each (smoke); shrinks bootstraps.")
    args = ap.parse_args()
    n_boot = int(args.bootstraps)

    t0 = time.time()
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(DERIVED, exist_ok=True)
    banner("Step 12 — Temporal external validation (TRIPOD-3 drift detector, pelvic v3)")

    (vars_kept, point_vars, override_vars, beta_drop,
     force_in_anchors, refit_anchor) = load_frozen_spec()
    _load_internal_reference()

    if args.sample:
        n_boot = min(n_boot, 200)
        print(f"  [SMOKE] --sample {args.sample}; n_boot -> {n_boot}")

    dm, vm, aux = load_analytic(vars_kept, args.sample)
    rng = np.random.default_rng(SEED)

    # ---- Composite (headline) + mortality anchor + any-transfusion (COVID co-run) ----
    res_comp = run_temporal("COMPOSITE (headline)", COMPOSITE_OUTCOME, dm, vm,
                            vars_kept, point_vars, override_vars, beta_drop,
                            n_boot, rng, force_in_anchors=force_in_anchors,
                            refit_anchor=refit_anchor)
    res_mort = run_temporal("MORTALITY ANCHOR (death_hemorrhagic; Fix 2/2b)",
                            MORTALITY_OUTCOME, dm, vm,
                            vars_kept, point_vars, override_vars, beta_drop,
                            n_boot, rng, force_in_anchors=force_in_anchors,
                            refit_anchor=refit_anchor)
    res_tx = run_temporal("ANY-TRANSFUSION (COVID co-run)", ANYTX_OUTCOME, dm, vm,
                          vars_kept, point_vars, override_vars, beta_drop,
                          n_boot, rng, force_in_anchors=force_in_anchors,
                          refit_anchor=refit_anchor)

    # ---- D-10 / G5h: differential timing ascertainment across THIS split ----
    # Must run BEFORE the summary, because the summary refuses to state a
    # temporal-drift conclusion for the mortality arm without it.
    banner("D-10 / CONTRACTS §13 G5h — differential timing ascertainment across "
           "the temporal split")
    asc, asc_reason = load_ascertainment()
    if asc is None:
        print(f"  [G5h] ascertainment figures UNAVAILABLE: {asc_reason}")
        attribution = {
            "status": "NOT_EVALUABLE", "reason": asc_reason,
            "gate": "CONTRACTS §13 G5h / SAP deviation D-10",
        }
    else:
        d, v = asc["derivation"], asc["validation"]
        print(f"  source: {asc['source']}  (read, never retyped)")
        print(f"    derivation {DERIVE_YEARS}: N={int(d['n']):,}  deaths={int(d['n_deaths']):,}  "
              f"untimed={int(d['deaths_missing_timing_n']):,} "
              f"({d['deaths_missing_timing_pct']:.3f} %)  "
              f"{MORTALITY_OUTCOME}={int(d['death_hemorrhagic_n']):,}")
        print(f"    validation {VALID_YEARS}: N={int(v['n']):,}  deaths={int(v['n_deaths']):,}  "
              f"untimed={int(v['deaths_missing_timing_n']):,} "
              f"({v['deaths_missing_timing_pct']:.3f} %)  "
              f"{MORTALITY_OUTCOME}={int(v['death_hemorrhagic_n']):,}")
        print(f"    DIFFERENTIAL: {asc['differential_pp']:+.3f} pp "
              f"(derivation minus validation)")
        print()
        for line in textwrap.wrap(_direction_sentence(asc), width=94):
            print(f"  {line}")
        print()
        print("  Pre-specified attribution check (CONTRACTS §3.4): re-run the mortality "
              "arm on the")
        print(f"  upper-bound rung `{BOUNDING_OUTCOME}`, which retains every untimed "
              f"death.")
        attribution = run_d10_attribution(asc, dm, vm, vars_kept, point_vars,
                                          override_vars, beta_drop, res_mort,
                                          n_boot, rng,
                                          force_in_anchors=force_in_anchors,
                                          refit_anchor=refit_anchor)
        attribution["gate"] = "CONTRACTS §13 G5h / SAP deviation D-10"
        attribution["split_ascertainment"] = asc
        attribution["attribution_statement"] = _direction_sentence(asc)

    banner("D-10 attribution result")
    print(f"  status : {attribution['status']}")
    if attribution["status"] == "EVALUATED":
        p, b = attribution["primary"], attribution["bounding"]
        print(f"  primary  arm {p['outcome']:<32} events valid={p['events_valid']:,}  "
              f"CITL={p['citl_offset_intercept']:+.4f}  slope={p['calibration_slope']:.4f}")
        print(f"  bounding arm {b['outcome']:<32} events valid={b['events_valid']:,}  "
              f"CITL={b['citl_offset_intercept']:+.4f}  slope={b['calibration_slope']:.4f}")
        print(f"  |CITL| shrinkage: {attribution['abs_citl_shrinkage']:+.4f} "
              f"({attribution['abs_citl_shrinkage_relative']:+.1%}); materiality rule "
              f">= {D10_MATERIAL_REL_SHRINK:.0%}")
        print(f"  VERDICT: {attribution['verdict']}")
        for line in textwrap.wrap(attribution["verdict_text"], width=94):
            print(f"    {line}")
    else:
        print(f"  reason : {attribution.get('reason')}")

    # ---- table4b_temporal.csv (rows: composite + mortality + any-transfusion) ----
    # The internal random-split reference in table4 is the PRIMARY-COMPOSITE AUROC;
    # attaching it to the mortality / any-transfusion rows would compare different
    # outcomes. Those rows carry NaN.
    banner("Writing table4b_temporal.csv")
    rows = []
    ref_by_outcome = {COMPOSITE_OUTCOME: INTERNAL_AUROC_LOGIT,
                      MORTALITY_OUTCOME: float("nan"),
                      ANYTX_OUTCOME: float("nan")}
    role_by_outcome = {COMPOSITE_OUTCOME: "PRIMARY composite (headline)",
                       MORTALITY_OUTCOME: "co-primary mortality-only anchor",
                       ANYTX_OUTCOME: "COVID co-run (any documented transfusion)"}
    for res in (res_comp, res_mort, res_tx):
        ref = ref_by_outcome[res["outcome"]]
        rows.append({
            "outcome": res["outcome"],
            "role": role_by_outcome[res["outcome"]],
            "derivation_years": DERIVE_YEARS,
            "validation_years": VALID_YEARS,
            "N_valid": res["n_valid"],
            "events": res["events_valid"],
            "AUROC": round(res["auroc_logit"], 4),
            "CI_low": round(res["ci_logit"][0], 4),
            "CI_high": round(res["ci_logit"][1], 4),
            "CITL": round(res["citl_offset_intercept"], 4),
            "slope": round(res["calibration_slope"], 4),
            "Brier": round(res["brier"], 6),
            # ---- extras (self-documenting; beyond the §7 minimum schema) ----
            "AUROC_integer": round(res["auroc_integer"], 4),
            "CI_low_integer": round(res["ci_integer"][0], 4),
            "CI_high_integer": round(res["ci_integer"][1], 4),
            "N_derive": res["n_derive"],
            "events_derive": res["events_derive"],
            "prevalence_derive": round(res["prevalence_derive"], 4),
            "prevalence_valid": round(res["prevalence_valid"], 4),
            "CITL_meandiff": round(res["citl_meandiff"], 5),
            "HL_P": (round(res["hl_p"], 4) if not np.isnan(res["hl_p"]) else np.nan),
            "internal_random_AUROC": round(ref, 4),
            "AUROC_delta_vs_internal": round(res["auroc_logit"] - ref, 4),
            # ---- Fix 9 / D-11, §13 G8e: which coefficient set THIS card's unit.
            # Carried in the table (not only citl_slope.json) so the three-script
            # anchor cross-check needs no JSON archaeology. Row count is untouched.
            "integer_anchor_var": res.get("anchor_min_positive_var"),
            "integer_anchor_beta": (round(float(res["anchor_min_positive_beta"]), 4)
                                    if res.get("anchor_min_positive_beta") is not None
                                    else np.nan),
            "integer_anchor_same_as_refit": (
                res.get("anchor_parity_vs_refit", {}).get("same_anchor_var")),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_TABLE, index=False)
    print(f"  wrote {OUT_TABLE}")
    print(tbl[["outcome", "N_valid", "events", "AUROC", "CI_low", "CI_high",
               "CITL", "slope", "Brier", "AUROC_delta_vs_internal"]].to_string(index=False))

    # ---- temporal_roc_data.pkl (F3 temporal overlay) ----
    banner("Writing temporal_roc_data.pkl (fig-data F3)")
    roc_pkl = {
        "temporal_composite_logit": {**res_comp["_roc_logit"],
                                     "outcome": COMPOSITE_OUTCOME, "model": "frozen_logit",
                                     "valid_years": VALID_YEARS, "n": res_comp["n_valid"],
                                     "events": res_comp["events_valid"]},
        "temporal_composite_integer": {**res_comp["_roc_integer"],
                                       "outcome": COMPOSITE_OUTCOME, "model": "integer_card",
                                       "valid_years": VALID_YEARS, "n": res_comp["n_valid"],
                                       "events": res_comp["events_valid"]},
        "temporal_mortality_logit": {**res_mort["_roc_logit"],
                                     "outcome": MORTALITY_OUTCOME, "model": "frozen_logit",
                                     "valid_years": VALID_YEARS, "n": res_mort["n_valid"],
                                     "events": res_mort["events_valid"]},
        "temporal_mortality_integer": {**res_mort["_roc_integer"],
                                       "outcome": MORTALITY_OUTCOME, "model": "integer_card",
                                       "valid_years": VALID_YEARS, "n": res_mort["n_valid"],
                                       "events": res_mort["events_valid"]},
        "temporal_anytx_logit": {**res_tx["_roc_logit"],
                                 "outcome": ANYTX_OUTCOME, "model": "frozen_logit",
                                 "valid_years": VALID_YEARS, "n": res_tx["n_valid"],
                                 "events": res_tx["events_valid"]},
        "temporal_anytx_integer": {**res_tx["_roc_integer"],
                                   "outcome": ANYTX_OUTCOME, "model": "integer_card",
                                   "valid_years": VALID_YEARS, "n": res_tx["n_valid"],
                                   "events": res_tx["events_valid"]},
        "_meta": {"derive_years": DERIVE_YEARS, "valid_years": VALID_YEARS, "seed": SEED,
                  "internal_random_auroc_integer": INTERNAL_AUROC_INTEGER,
                  "internal_random_auroc_logit": INTERNAL_AUROC_LOGIT,
                  "mortality_outcome": MORTALITY_OUTCOME,
                  "point_vars": list(point_vars), "override_vars": list(override_vars),
                  "integer_beta_floor": float(beta_drop),
                  # D-10 pointer for any figure captioning the temporal overlay.
                  # Summary only — the full block lives in citl_slope.json under
                  # `differential_ascertainment_d10`. No ROC curve is added here:
                  # the upper-bound arm is a diagnostic, not a reportable curve.
                  "d10_attribution": {
                      "status": attribution.get("status"),
                      "verdict": attribution.get("verdict"),
                      "differential_pp": (asc or {}).get("differential_pp"),
                      "full_record": "derived/citl_slope.json -> "
                                     "differential_ascertainment_d10",
                  },
                  "note": ("mortality anchor is death_hemorrhagic (Fix 2/2b), NOT "
                           "death_inhosp; internal reference is the v3 primary "
                           "composite only and is NaN when table4 is absent")},
    }
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(roc_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {OUT_PICKLE}  ({[k for k in roc_pkl if k != '_meta']})")

    # ---- citl_slope.json (full drift record) ----
    banner("Writing citl_slope.json")

    def _pack(res, ref_logit, ref_integer):
        d = {k: v for k, v in res.items() if not k.startswith("_roc")}
        d["internal_random_auroc_logit"] = ref_logit
        d["internal_random_auroc_integer"] = ref_integer
        d["auroc_drop_logit_vs_internal"] = res["auroc_logit"] - ref_logit
        d["auroc_drop_integer_vs_internal"] = res["auroc_integer"] - ref_integer
        return d

    nan = float("nan")
    payload = {
        "script": "12_temporal_external_validation.py",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "n_boot": n_boot,
        "derivation_years": DERIVE_YEARS,
        "validation_years": VALID_YEARS,
        "model_specification": {
            "variables_kept": list(vars_kept), "n_predictors": len(vars_kept),
            "point_vars": list(point_vars), "override_vars": list(override_vars),
            "load": "name-based from refit.json; positional alignment never used",
            "integer_rule": ("round(beta / |beta_min_positive|) over point-role "
                             "variables, with a ±1 floor at |beta| >= "
                             f"{beta_drop} (sign preserved) — the SAME rule "
                             "05_refit_integer.py applies, which is what makes "
                             "the temporal card comparable to the random-split "
                             "card"),
            "integer_beta_floor": float(beta_drop),
            # ---- Fix 9 / D-11 (CONTRACTS §5.3, §13 G8) ----
            "anchor_selection_rule": ANCHOR_SELECTION_RULE,
            "anchor_selection_rule_source": (
                "READ from refit.json['anchor_selection_rule'] and asserted equal "
                "to this script's literal at load time — a mismatch is a hard "
                "failure (§13 G8e), so the three implementations cannot drift"),
            "anchor_rule_recorded_by_05": refit_anchor.get("rule"),
            "refit_anchor_min_positive_var": refit_anchor.get("var"),
            "refit_anchor_min_positive_beta": refit_anchor.get("beta"),
            "force_in_anchors": list(force_in_anchors),
            "anchor_note": (
                "D-11: the divisor is the smallest POSITIVE point-role coefficient "
                "that passed the |beta| >= BETA_DROP screen. A force-in anchor "
                "sitting below the screen is excluded from the ANCHOR SEARCH only "
                "— it stays in the linear predictor, stays on the card, and may "
                "score 0 points. This is a registered DEVIATION from the locked "
                "SAP's literal rule, not a bugfix. Each outcome block below "
                "carries its own anchor_min_positive_var / _beta and an "
                "anchor_parity_vs_refit record: 12 re-estimates on 2019-2022, so "
                "its anchor VARIABLE may differ from 05's while the RULE is "
                "identical."),
        },
        "design_note": (f"Variable selection fixed ({len(vars_kept)} LASSO predictors from "
                        "refit.json); coefficients re-estimated on 2019-2022 and applied "
                        "unchanged to 2023-2024. Frozen published card NOT applied to valid "
                        "(its random split leaks 2023-2024 rows). v3: `unstable_ring` is out "
                        "of the score (Fix 3) so the weight vector is shorter than v2's; "
                        "`sbp_unrecordable` is an override and carries no points (Fix 4)."),
        "outcome_note": (f"Mortality anchor is `{MORTALITY_OUTCOME}` (Fix 2 + 2b): "
                         "in-hospital death restricted to head AIS < 4 and to an "
                         "early-death window. The column is consumed BY NAME from "
                         "candidates.parquet and never recomputed here — `01` owns the "
                         "exact restriction. `death_inhosp` is a NAMED SENSITIVITY owned "
                         "by 13 and is not validated here."),
        "internal_reference_note": ("internal_random_auroc_* is the v3 PRIMARY-COMPOSITE "
                                    "random-split AUROC read from table4_discrimination.csv; "
                                    "it is NaN for the mortality and any-transfusion rows "
                                    "(different outcomes) and NaN if table4 is absent. No v2 "
                                    "constant is substituted (CONTRACTS §13 G5b)."),
        "citl_definitions": {
            "citl_offset_intercept": "intercept of logit(P)=a+offset(LP); <0 => over-prediction",
            "citl_meandiff": "mean(predicted prob) - mean(observed); >0 => over-prediction",
            "calibration_slope": "beta of logit(P)=a+beta*LP; 1.0 = ideal spread",
        },
        "any_transfusion_note": ("blood_4h > 0; missing blood_4h treated as no documented "
                                 f"transfusion. Missing derive={aux['blood_4h_missing_derive']:,}, "
                                 f"valid={aux['blood_4h_missing_valid']:,}."),
        "composite": _pack(res_comp, INTERNAL_AUROC_LOGIT, INTERNAL_AUROC_INTEGER),
        "mortality": _pack(res_mort, nan, nan),
        "any_transfusion": _pack(res_tx, nan, nan),
        # ---- D-10 / G5h (CONTRACTS §3.4, §13 G5h) --------------------------
        # The block 16 must quote when it reports the mortality arm's
        # calibration. It carries the per-split ascertainment rates, the event
        # counts, the differential, the attribution sentence, and the
        # pre-specified upper-bound re-run. It is NOT a row of
        # table4b_temporal.csv — that table keeps its 3 rows + `role` (§7.1).
        "differential_ascertainment_d10": attribution,
        "differential_ascertainment_note": (
            "READ `differential_ascertainment_d10` BEFORE quoting `mortality`. "
            "Timing missingness among deaths is differential across the temporal "
            "split, so the mortality arm's CITL/slope is partly a data-collection "
            "artifact. A temporal-drift claim made without this attribution is a "
            "CONTRACTS §13 G5h FAIL."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"  wrote {OUT_JSON}")

    # ---- Gate summary / handoff ----
    banner("TEMPORAL VALIDATION SUMMARY (CONTRACTS §13 G5d — temporal AUROC + CITL/slope)")
    g5h_ok = attribution.get("status") == "EVALUATED"
    for res in (res_comp, res_mort, res_tx):
        ref = ref_by_outcome[res["outcome"]]
        print(f"  [{res['outcome']}]  ({role_by_outcome[res['outcome']]})")
        print(f"    AUROC logit   {res['auroc_logit']:.4f} [{res['ci_logit'][0]:.4f}, "
              f"{res['ci_logit'][1]:.4f}]   vs internal {ref:.4f} "
              f"(delta {res['auroc_logit'] - ref:+.4f})")
        print(f"    AUROC integer {res['auroc_integer']:.4f} [{res['ci_integer'][0]:.4f}, "
              f"{res['ci_integer'][1]:.4f}]")
        print(f"    CITL {res['citl_offset_intercept']:+.4f}  slope {res['calibration_slope']:.4f}  "
              f"Brier {res['brier']:.5f}  "
              f"[well-calibrated if CITL~0 & slope~1]")
        if res["events_valid"] < 100:
            print(f"    [NOTE] only {res['events_valid']:,} validation events — treat the "
                  f"CITL/slope as descriptive.")
        # ---- G5h GUARD: the mortality arm never gets an unqualified reading ----
        # A drift conclusion for `death_hemorrhagic` is only meaningful once the
        # ascertainment differential has been attributed. Without the block the
        # script says so instead of saying nothing — a gate that cannot be
        # evaluated is a FAIL, not a skip (CONTRACTS §13 preamble).
        if res["outcome"] == MORTALITY_OUTCOME:
            if g5h_ok:
                asc_b = attribution["split_ascertainment"]
                print(f"    [G5h ATTRIBUTION — REQUIRED READING WITH THE LINE ABOVE]")
                print(f"      untimed deaths: derivation "
                      f"{asc_b['derivation']['deaths_missing_timing_pct']:.3f} % vs "
                      f"validation "
                      f"{asc_b['validation']['deaths_missing_timing_pct']:.3f} % "
                      f"({asc_b['differential_pp']:+.3f} pp); "
                      f"{MORTALITY_OUTCOME} "
                      f"{int(asc_b['derivation']['death_hemorrhagic_n']):,} vs "
                      f"{int(asc_b['validation']['death_hemorrhagic_n']):,} events.")
                print(f"      The outcome is under-ascertained MORE in derivation, so "
                      f"this CITL/slope is partly")
                print(f"      DATA COLLECTION, NOT MODEL DECAY. Upper-bound re-run "
                      f"({BOUNDING_OUTCOME}):")
                print(f"      CITL {attribution['bounding']['citl_offset_intercept']:+.4f} "
                      f"vs {attribution['primary']['citl_offset_intercept']:+.4f} "
                      f"(|CITL| "
                      f"{attribution['abs_citl_shrinkage_relative']:+.1%}) -> "
                      f"verdict: {attribution['verdict']}.")
            else:
                print(f"    *** GATE G5h / D-10: FAIL — NO TEMPORAL-DRIFT CONCLUSION "
                      f"MAY BE DRAWN FOR THIS ARM ***")
                print(f"      The differential-ascertainment attribution could not be "
                      f"evaluated:")
                print(f"        {attribution.get('reason')}")
                print(f"      Timing missingness among deaths is differential across "
                      f"this exact split, so the")
                print(f"      CITL/slope above may be data collection rather than "
                      f"model decay. CONTRACTS §13 G5h:")
                print(f"      \"A temporal-drift claim made without this attribution "
                      f"is a FAIL.\" Do not quote the")
                print(f"      mortality arm's calibration in 16 until this is "
                      f"resolved.")
    if res_comp["n_valid"] != N_VALID_EXPECTED and not args.sample:
        print(f"  [WARN] valid N={res_comp['n_valid']:,} != expected {N_VALID_EXPECTED:,}")
    if res_comp["n_derive"] != N_DERIVE_EXPECTED and not args.sample:
        print(f"  [WARN] derive N={res_comp['n_derive']:,} != expected {N_DERIVE_EXPECTED:,}")

    banner("GATE STATUS")
    print(f"  G5d (temporal AUROC + CITL/slope reported) : PASS")
    print(f"  G5h (D-10 differential-ascertainment attribution) : "
          f"{'PASS' if g5h_ok else 'FAIL'}")
    print(f"  16 must read: derived/citl_slope.json -> "
          f"differential_ascertainment_d10")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s\nStep 12 complete.")

    if not g5h_ok:
        # Artifacts are already written (so the failure is debuggable), but the
        # exit status is honest: this run did NOT satisfy G5h.
        raise SystemExit(
            "12 exiting non-zero: CONTRACTS §13 G5h could not be evaluated, so the "
            "mortality arm's temporal calibration is unattributed. All artifacts "
            "were still written; fix the cause and re-run before 16.")


if __name__ == "__main__":
    main()
