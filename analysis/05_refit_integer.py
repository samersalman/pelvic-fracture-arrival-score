"""
Pelvic Arrival Tool v3 — Step 05: Unpenalized refit + integer conversion
================================================================================

Forked from PAT_TBI/v4/code/05_refit_unpenalized_integer.py.

Takes the LASSO-stability-selected feature set (Step 04), refits an UNPENALIZED
logistic regression on the TRAIN rows, drops |β| < 0.15, **FORCE-INS the
pre-registered anchors**, runs an anchor-protected VIF loop, and converts to
integer points via round(β / |β_min_positive|) with a ±1 floor.

ADDED vs PAT source (CONTRACTS §9-C, §5, SAP §5):
  * FORCE-IN anchors {age_65_74, age_ge75, cc_anticoag}. The two age bands
    jointly are the "age>=65" anchor. Any anchor dropped by LASSO or by the
    |β|<0.15 cutoff is re-inserted before VIF/integer conversion.
  * The VIF loop NEVER drops an anchor (anchors excluded from the drop-candidate set).
  * cohort_primary==1 filter (CONTRACTS §3); lowercase join keys; outcome =
    major_hemorrhage_event.

v3 CHANGE — Fix 3 / D-2: `unstable_ring` is gone from the pool AND the anchors
------------------------------------------------------------------------------
Reviewer comment: "We assign +3 points for an 'unstable pelvic ring' but the ENTIRE
abstract revolves around a score that uses IMMEDIATE arrival data." Ring
stability is a radiographic determination, so it cannot sit on a card that
claims arrival-only inputs. LASSO_X_COLS drops 18 -> 17 and FORCE_IN_ANCHORS
drops 4 -> 3 entries (= 2 conceptual anchors: age>=65, implemented as the two
mutually-exclusive age bands, and cc_anticoag).

The removal is deliberately NOT propagated to the data or to Analysis B:
`unstable_ring` stays in the cohort CSV, in `11.CORE_ADJUSTMENT` and
`VIF_AUDIT_VARS`, as a change-point stratum, and in Table 1. Analysis B is an
association model, not a bedside instrument; adjusting it for a confounder
ascertained later in the admission is not a claim that the confounder was known
on arrival, and dropping it would leave the SI/SBP curves confounded by injury
pattern. See CONTRACTS §6 and SAP_amendment_v3.md D-3.

v3 CHANGE — Fix 4 / D-6 + W2: `sbp_unrecordable` becomes an OVERRIDE, not a row
------------------------------------------------------------------------------
Reviewer comment: "An initial SBP of 0 is recorded for anyone who has no signs of life
on their first assessment." This is a physiologic state — the registry's way of
recording a patient who arrives with no signs of life — and NOT a missingness or
data-quality artifact. It is labelled everywhere as
**"no signs of life on arrival (unrecordable systolic pressure)"**.

In v2 it carried β = 4.0065 (OR 54.96) -> **+16 points**, against an observed
score range of -7..+32. A reviewer flagged that as implausible on a bedside card:
one row worth half the whole scale. So in v3 it is taken off the integer scale
and made a binary automatic-escalate override:

    escalate = (pelvic_score >= c_rule_in) OR (no signs of life on arrival)

Mechanically, in THIS script (CONTRACTS §5.1 row `05`):
  (a) it STAYS a LASSO candidate and STAYS in the unpenalized refit, so every
      other β remains adjusted for it, and its β/SE/OR/CI are reported;
  (b) it is EXCLUDED from the |β_min_positive| anchor search — otherwise the
      integer divisor is chosen while an off-scale variable is in the running,
      and a single override can distort every other point value on the card;
  (c) its `Points` cell is written EMPTY (NA) — never a number, never 0;
  (d) `table3_points.csv` gains a `Role` column in {point, override};
  (e) `refit.json` gains `point_vars` / `override_vars`, asserted disjoint;
  (f) the training score distribution printed below sums POINT-ROLE vars only.
Downstream: `06` skips override rows when summing and emits `override_nslo`;
`08` reports both the score-only and score-OR-override rules; the card renders
it as a banner, not a point row; `13` runs `sens_exclude_nslo`.

v3 CHANGE — Fix 9 / D-11: the integer-scale ANCHOR is amended (POST-RUN)
------------------------------------------------------------------------------
**This is a deviation from the locked SAP's literal integer-conversion rule. It
is registered as D-11 in SAP_amendment_v3.md and specified in CONTRACTS.md §5.3.
It is NOT a bugfix — the locked rule was implemented correctly and produced the
card below.**

WHAT THE LOCKED SAP SAID. `v2/SAP_lock.md` §5 specifies integer points as
`round(β / |β_min_pos|)` with a ±1 floor, and §4 pre-registers force-in anchors
that are immune to the `|β| >= 0.15` screen. Read literally, `|β_min_pos|` is the
smallest positive β among the retained point-role variables — including a force-in
anchor that survived only because it was pre-registered.

WHAT IT PRODUCED. Under the v3 outcome (`major_hemorrhage_event`, whose death arm
is restricted to early hemorrhagic death — Fix 2/2b/2c) the pre-registered
force-in anchor `cc_anticoag` collapsed to **β = 0.062 (OR 1.06, 95% CI
0.976–1.159, non-significant)**. Restricting the death arm to `ais_head < 4` and
`finaldischarge_hrs <= 48` removed the late deaths in which anticoagulation
mattered. Retained by force-in, `cc_anticoag` was then the smallest positive β,
became the `|β_min_positive|` denominator, and divided every other coefficient by
0.062:

    mech_fall      β -1.6723  ->  -27 points
    sbp_lt90       β  1.1173  ->  +18 points
    gcs_motor_le5  β  0.9160  ->  +15 points
    observed score range: -35 .. +96      (v2 was -7 .. +32)

WHY THAT IS NOT SHIPPABLE. It defeats Fix 4 (D-4) entirely. The original
objection (comment D6) was that a **+16** point weight is implausible on a bedside
card; a **-27** row is worse than the thing objected to. The denominator was
degenerate: the scale's unit was set by a coefficient that is statistically
indistinguishable from zero, so every point value on the card became an artefact
of a null term.

THE AMENDED RULE (CONTRACTS §5.3):

    |β_min_positive| = min{ β_v : v ∈ point_vars, β_v > 0, |β_v| >= BETA_DROP }

i.e. the divisor is the smallest positive coefficient among point-role variables
**that passed the pre-specified |β| >= 0.15 screen on their own evidence**. A
variable retained *solely* by force-in and sitting below the screen is excluded
from the ANCHOR SEARCH ONLY. Everything else is unchanged: same
`round(β / anchor)`, same ±1 floor with its exact condition
(`pts == 0 and abs(β) >= BETA_DROP`), same anchor immunity from the β-drop and
from the VIF loop, same override exclusion (§5.1).

SCIENTIFIC JUSTIFICATION (quotable verbatim). *A coefficient retained by
pre-registration rather than by evidence should not set the unit of the scale.
The |β| >= 0.15 screen exists precisely to identify coefficients large enough to
be meaningful; the denominator should come from that set. The force-in anchor is
still retained in the model, still refit, still reported with its β, OR and CI —
it simply no longer defines the unit.*

CONSEQUENCES.
  * The anchor becomes `age_65_74` (β = 0.3407). The card lands at
    mech_fall -5, sbp_lt90 +3, gcs_motor_le5 +3, si_ge0_9 +2, hr_ge120 +2,
    si_ge1_0 +2, mech_high_energy +2, gcs_le8 +1, sbp_lt70 +1, sex_female -1,
    mech_mvc -1, age_65_74 +1, age_ge75 +1, cc_anticoag 0.
  * `cc_anticoag` STAYS ON THE CARD AT 0 POINTS. It is pre-registered; it is not
    hidden, not dropped, not "tidied" off F9 (CONTRACTS §5.3, §13 G8d). The ±1
    floor correctly does NOT fire for it (|β| = 0.062 < 0.15) and it is NOT
    special-cased up to ±1 — 0 is the honest value.
  * THE MODEL IS UNCHANGED; THE CARD IS A COARSER RENDERING OF IT. No
    coefficient, no selected variable, no role assignment and no intercept
    changes. `05` records `d11_rescaling_only_check` (β vector, intercept,
    point_vars/override_vars, anchor selection, and the prior run's values) so
    §13 G8f is EVIDENCED rather than asserted in prose.

    **Do NOT claim the integer AUROC is unchanged.** Rank-invariance holds for the
    continuous logit, not for the integer card: `round(β/0.062)` ->
    `round(β/0.3407)` is a coarser QUANTISATION, not a strictly monotone map.
    Patient pairs with distinct totals under the old scale can TIE under the new
    one, and ties contribute 0.5 to AUROC. The integer AUROC can therefore move,
    and will most likely move slightly DOWN, because resolution is deliberately
    being discarded. The continuous-logit AUROC must be identical to the digit —
    nothing about the fit changes. The defensible claim, and the only one to
    make: *the model is unchanged; the card is a coarser rendering of it; here is
    the cost in discrimination.* This script REPORTS both deltas and asserts
    NEITHER — an assertion of integer-AUROC invariance would be a gate designed
    to fail.

SUBSTANTIVE FINDING, not merely a scaling nuisance. In v2's all-cause-death
composite `cc_anticoag` looked predictive (β 0.253, OR 1.29, 95% CI 1.20–1.38,
+1 point). Under the hemorrhage-specific v3 outcome it is null (OR 1.06, CI
crosses 1). The reading — that preexisting anticoagulation adds essentially
nothing beyond arrival physiology once the death arm is restricted to *early
hemorrhagic* death — belongs in Results and Discussion, and it bears directly on
the elderly/anticoagulated headline. `refit.json` therefore carries
`zero_point_force_in_anchors` with β, OR, CI and the v2 comparison, so `08`, `16`
and `render_figures` can STATE it as a result instead of rediscovering it.

Outputs
-------
  v3/tables/table3_points.csv
      Variable, β, SE, OR, 95% CI lower, 95% CI upper, Points, Role
      `Points` is an INTEGER for Role=="point" rows and EMPTY for
      Role=="override" rows. Consumers must read it with pd.to_numeric(...,
      errors="coerce") and must NOT .fillna(0) — an override row is not a
      zero-point row; it must be filtered on Role instead.
  v3/derived/refit.json
      Frozen-weights structure consumed by 06 (linear predictor + points) and by
      12 (temporal external validation freezes intercept + coefficients).
      v3 adds `point_vars`, `override_vars`, `override_labels`.

Run
---
    python 05_refit_integer.py
    python 05_refit_integer.py --sample 50000        # smoke (row downsample)

Seed: 42
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES
LOGS = config.LOGS

# §13 G8f baseline — the PRE-D-11 artifacts, snapshotted with a SHA manifest by
# the SAP owner before the amended rule was implemented. READ-ONLY, and preferred
# over the in-place artifacts: after the first post-D-11 run the in-place files
# are themselves post-D-11, so comparing against them would answer "nothing
# changed" and quietly destroy the evidence the gate exists to produce. `05` never
# writes into logs/.
PRE_D11_DIR = os.path.join(LOGS, "pre_d11_snapshot")

CANDIDATES = os.path.join(DERIVED, "candidates.parquet")
TRAIN_KEYS = os.path.join(DERIVED, "train_inc_keys.csv")
LASSO_SEL = os.path.join(TABLES, "table2_lasso_selected.csv")
OUT_WEIGHTS = os.path.join(TABLES, "table3_points.csv")
OUT_JSON = os.path.join(DERIVED, "refit.json")

OUTCOME_COL = "major_hemorrhage_event"
JOIN_KEYS = ["puf_year", "inc_key"]
COHORT_FILTER_COL = "cohort_primary"

# The 17 candidate columns (CONTRACTS §5) — recorded into refit.json for 12.
# Continuous `shock_index` is engineered in candidates.parquet but is NOT a
# LASSO candidate (CONTRACTS §5 CORRECTION); SI enters via si_ge0_9 + si_ge1_0.
# v3 = 17 (was 18): `unstable_ring` removed (Fix 3 / D-2). Kept byte-identical
# with 02_engineer_features.py and 04_lasso_stability.py.
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

# Force-in anchors (CONTRACTS §5 / §9-C / SAP §4). age>=65 == both age bands, so
# three list entries express TWO conceptual anchors. v3 drops `unstable_ring`
# (Fix 3 / D-2); CONTRACTS §13 G3b asserts this exact list.
FORCE_IN_ANCHORS = ["age_65_74", "age_ge75", "cc_anticoag"]

# ---- Fix 4 / D-6 + W2: variables that leave the INTEGER POINT SCALE ---------
# `sbp_unrecordable` stays in the refit (so the other βs stay adjusted for it)
# but becomes a binary automatic-escalate override rather than a point row.
# CONTRACTS §5.1.
OVERRIDE_VARS = ["sbp_unrecordable"]

ROLE_POINT = "point"
ROLE_OVERRIDE = "override"

# Canonical display label. The clinical reading — an SBP of 0 is what
# the registry records for a patient with no signs of life on first assessment.
# This is a PHYSIOLOGIC STATE, not a missingness / data-quality artifact, and
# must never be rendered as "SBP = 0" or "SBP unrecordable" anywhere a reader
# can see it (CONTRACTS §5.1, §13 G4f). Carried into refit.json so 06/08/18/
# render_figures consume one string instead of each inventing their own.
DISPLAY_LABELS = {
    "sbp_unrecordable": "No signs of life on arrival (unrecordable systolic pressure)",
}

# Bedside rule text for the override, carried to 08's score_card.json.
OVERRIDE_RULE_TEXT = (
    "Automatic escalation regardless of score: "
    "no signs of life on arrival (unrecordable systolic pressure)"
)

BETA_DROP = 0.15
VIF_MAX = 5.0
MIN_KEPT = 5

# ---- Fix 9 / D-11: the integer-scale anchor rule (CONTRACTS §5.3) -----------
# This string is the PARITY TOKEN. `12_temporal_external_validation.py` and
# `13_sensitivity_components.py` re-implement the same rule and compare their own
# literal against the one 05 writes into refit.json. It must therefore stay
# BYTE-IDENTICAL in all three scripts. It deliberately says "BETA_DROP" rather
# than "0.15" so that moving the cutoff cannot make the recorded rule lie.
ANCHOR_SELECTION_RULE = (
    "anchor = min{ beta_v : v in point_vars, beta_v > 0, |beta_v| >= BETA_DROP }; "
    "variables retained solely by force-in and sitting below the BETA_DROP screen "
    "are excluded from the anchor search (CONTRACTS §5.3 / SAP D-11)"
)

# §13 G8c — card-scale tripwire. Not a scientific target: a |Points| value this
# large means the DENOMINATOR collapsed, so inspect the anchor before touching
# anything else. The pre-D-11 run tripped it at -35..+96.
CARD_SCALE_TRIPWIRE = 20

# ---- v2 comparison constants (hard-coded, LABELLED as v2 — §13 G5b) --------
# Read once from ../v2/tables/table3_points.csv (frozen fork, read-only) and
# frozen here so this script has no runtime dependency on v2. Every consumer must
# label these as a v2 comparison; they are NOT v3 results. Carried so that the
# `cc_anticoag` collapse (D-11) can be reported as the FINDING it is: predictive
# under v2's all-cause-death composite, null under the v3 hemorrhage-specific
# outcome. v2 outcome = `major_hemorrhage_event_v2def`
# (received_hc | death_inhosp | substantial_transfusion_ml_only).
V2_OUTCOME_LABEL = ("major_hemorrhage_event_v2def "
                    "(received_hc | death_inhosp | substantial_transfusion_ml_only)")
V2_REFERENCE_POINTS = {
    "cc_anticoag": {"beta": 0.2529, "or": 1.288, "ci_low": 1.202, "ci_high": 1.380,
                    "points": 1},
    "age_65_74":   {"beta": 0.5532, "or": 1.739, "ci_low": 1.644, "ci_high": 1.839,
                    "points": 2},
    "age_ge75":    {"beta": 0.8549, "or": 2.351, "ci_low": 2.217, "ci_high": 2.494,
                    "points": 3},
}
V2_REFERENCE_NOTE = (
    "v2 COMPARISON ONLY — hard-coded from the frozen ../v2/tables/table3_points.csv "
    "under the v2 outcome. Label as a v2 comparison wherever quoted (CONTRACTS "
    "§13 G5b); never present as a v3 result."
)


def select_integer_anchor(beta_by_var, point_vars, beta_drop, force_in_anchors=(),
                          emit=print):
    """|β_min_positive| under CONTRACTS §5.3 (Fix 9 / D-11). AMENDED RULE.

        anchor = min{ β_v : v ∈ point_vars, β_v > 0, |β_v| >= beta_drop }

    The divisor is the smallest positive coefficient among POINT-ROLE variables
    **that passed the pre-specified |β| >= beta_drop screen on their own
    evidence**. A variable retained *solely* by force-in and sitting below the
    screen is excluded FROM THE ANCHOR SEARCH ONLY — it stays in the model, stays
    in the refit, stays on the card, and is reported with its β/OR/CI. It may
    legitimately score 0 points.

    Deviation from the locked SAP, registered as D-11. Rationale: a coefficient
    retained by pre-registration rather than by evidence should not set the unit
    of the scale. The screen exists precisely to identify coefficients large
    enough to be meaningful; the denominator should come from that set. Under the
    v3 outcome the force-in anchor `cc_anticoag` collapsed to β = 0.062 and, as
    the literal `|β_min_positive|`, produced a -27-point row on a bedside card.

    DEGENERATE FALLBACK: if NO point-role variable passes the screen, fall back to
    the locked rule (min positive β over all point-role variables) with a loud
    warning rather than crashing — a degenerate model is still reportable, and an
    exception here would take the whole pipeline down over a scale choice.

    Returns (anchor_beta, anchor_var, meta) — `meta` is JSON-serialisable and is
    what `refit.json` / the parity checks in 12 and 13 record.
    """
    beta_drop = float(beta_drop)
    force_in_anchors = set(force_in_anchors or ())
    # Insertion order follows `point_vars`, so ties break deterministically and
    # identically in 05 / 12 / 13.
    positive = {v: float(beta_by_var[v]) for v in point_vars
                if float(beta_by_var[v]) > 0}
    if not positive:
        raise RuntimeError(
            "No positive-β POINT-role risk factors retained — refuse to build a "
            "score (an override-only card is not a score).")

    eligible = {v: b for v, b in positive.items() if abs(b) >= beta_drop}
    # Everything point-role that sits below the screen — any sign — is ineligible
    # to set the unit. Recorded with the reason so the run is auditable (§13 G8b).
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
                f"|β| = {abs(b):.4f} < screen {beta_drop} — "
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
        # Degenerate: no point-role coefficient cleared the screen.
        fallback = True
        anchor_var = min(positive, key=positive.get)
        anchor = float(positive[anchor_var])
        emit("  " + "!" * 74)
        emit("  [D-11 DEGENERATE FALLBACK] NO point-role variable passed the "
             f"|β| >= {beta_drop} screen.")
        emit("  The amended anchor rule (CONTRACTS §5.3) has an EMPTY eligible "
             "set, so this run falls back to the LOCKED SAP rule (min positive β "
             "over all point-role variables).")
        emit(f"  Fallback anchor = {anchor:.4f} ({anchor_var}). Every point value "
             "on this card is therefore scaled by a coefficient that did NOT "
             "clear the screen — inspect the model before quoting the card.")
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

    # ---- §13 G8a: the chosen anchor must have passed the screen -------------
    # (Suppressed only in the declared degenerate fallback, which is loud above.)
    assert anchor_var in point_vars, \
        f"anchor {anchor_var!r} is not a point-role variable (CONTRACTS §5.1a)"
    assert anchor > 0, f"anchor β must be positive; got {anchor}"
    if not fallback:
        assert abs(anchor) >= beta_drop, (
            f"§13 G8a VIOLATION: anchor {anchor_var!r} has |β| = {abs(anchor):.4f} "
            f"< screen {beta_drop}. A variable retained solely by force-in must "
            f"not set the integer scale (CONTRACTS §5.3 / D-11).")
    return anchor, anchor_var, meta

# ---- Contract guards on the constants themselves (CONTRACTS §13 G3a/G3b) ----
assert FORCE_IN_ANCHORS == ["age_65_74", "age_ge75", "cc_anticoag"], (
    f"CONTRACTS §13 G3b: FORCE_IN_ANCHORS must be exactly "
    f"['age_65_74', 'age_ge75', 'cc_anticoag']; got {FORCE_IN_ANCHORS}")
assert "unstable_ring" not in LASSO_X_COLS and "unstable_ring" not in FORCE_IN_ANCHORS, (
    "Fix 3 / D-2 VIOLATION: `unstable_ring` is back in 05's candidate pool or "
    "anchors (CONTRACTS §13 G3a/G3b).")
assert len(LASSO_X_COLS) == 17 == len(set(LASSO_X_COLS)), (
    f"LASSO_X_COLS must be 17 unique predictors; got {len(LASSO_X_COLS)}")
assert set(OVERRIDE_VARS).issubset(set(LASSO_X_COLS)), (
    "Every override variable must still be a LASSO candidate — Fix 4 removes it "
    "from the point scale, not from the model (CONTRACTS §5.1).")
assert set(FORCE_IN_ANCHORS).isdisjoint(set(OVERRIDE_VARS)), (
    "A force-in anchor cannot also be an override variable.")


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def fit_logit(X_df, y):
    Xc = sm.add_constant(X_df, has_constant="add")
    model = sm.Logit(y, Xc.astype(float))
    return model.fit(disp=False, maxiter=200, method="newton")


def compute_vif(X_df):
    Xc = sm.add_constant(X_df, has_constant="add").astype(float).values
    cols = ["const"] + list(X_df.columns)
    vifs = {}
    for i, name in enumerate(cols):
        if name == "const":
            continue
        try:
            vifs[name] = float(variance_inflation_factor(Xc, i))
        except Exception:
            vifs[name] = float("nan")
    return vifs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Row-downsample the TRAIN join to N for a smoke test.")
    args = parser.parse_args()

    banner("Pelvic v3 — Step 05: Unpenalized refit + integer conversion "
           "(|β|>=0.15, force-in anchors, anchor-protected VIF)")

    # ---------------- Load ----------------
    print(f"Loading candidates: {CANDIDATES}", flush=True)
    df = pd.read_parquet(CANDIDATES)
    if COHORT_FILTER_COL not in df.columns:
        raise KeyError(f"candidates parquet missing {COHORT_FILTER_COL}")
    df = df[df[COHORT_FILTER_COL] == 1].copy()
    print(f"  candidates (cohort_primary==1): {df.shape}", flush=True)
    for k in JOIN_KEYS:
        df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")

    train_keys = pd.read_csv(TRAIN_KEYS)
    for k in JOIN_KEYS:
        train_keys[k] = pd.to_numeric(train_keys[k], errors="coerce").astype("Int64")
    df = df.merge(train_keys, on=JOIN_KEYS, how="inner")
    print(f"  after inner-join to train: {df.shape}", flush=True)
    assert len(df) == len(train_keys), "train join row count drift"

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE] downsampled train to {len(df):,}", flush=True)

    sel = pd.read_csv(LASSO_SEL)
    freq_lookup = dict(zip(sel["Variable"].astype(str), sel["Selection_Frequency"].astype(float)))
    variables = list(sel["Variable"].astype(str))
    print(f"  {len(variables)} LASSO-selected variables: {variables}", flush=True)

    # Force-in anchors up front so the initial fit contains them even if LASSO
    # dropped them (freq < 0.85).
    forced_added_pre = [a for a in FORCE_IN_ANCHORS if a not in variables]
    for a in FORCE_IN_ANCHORS:
        if a not in variables:
            variables.append(a)
    if forced_added_pre:
        print(f"  Force-in (not LASSO-selected): {forced_added_pre}", flush=True)

    missing = [v for v in variables if v not in df.columns]
    assert not missing, f"Missing selected/anchor vars in candidates: {missing}"

    y = df[OUTCOME_COL].astype(int).values
    print(f"  n_train={len(df):,}  events={int(y.sum()):,} ({100*y.mean():.2f}%)", flush=True)

    # ---------------- Initial unpenalized fit ----------------
    banner("Initial unpenalized Logit fit")
    X = df[variables].astype(float)
    res = fit_logit(X, y)
    n_before_cutoff = len(variables)
    print(f"  converged={res.mle_retvals.get('converged', True)}  "
          f"n_features={n_before_cutoff}  intercept β={float(res.params.iloc[0]):.4f}", flush=True)

    # ---------------- Drop |β| < 0.15, then FORCE-IN anchors ----------------
    banner(f"Dropping features with |β| < {BETA_DROP} (OR < {np.exp(BETA_DROP):.2f}) "
           f"— anchors immune")
    beta = res.params.drop("const")
    weak = beta[beta.abs() < BETA_DROP].index.tolist()
    # Anchors are immune to the β-cutoff.
    weak_dropped = [v for v in weak if v not in FORCE_IN_ANCHORS]
    weak_kept_anchor = [v for v in weak if v in FORCE_IN_ANCHORS]
    print(f"  weak (|β|<{BETA_DROP}) dropped : {weak_dropped}", flush=True)
    if weak_kept_anchor:
        print(f"  weak but FORCE-IN anchor kept: {weak_kept_anchor}", flush=True)
    dropped_beta = list(weak_dropped)
    variables = [v for v in variables if v not in weak_dropped]

    # Re-insert any anchor that may have gone missing (belt & suspenders).
    forced_reinserted = []
    for a in FORCE_IN_ANCHORS:
        if a not in variables:
            variables.append(a)
            forced_reinserted.append(a)
    if forced_reinserted:
        print(f"  Force-in re-inserted anchors: {forced_reinserted}", flush=True)
    print(f"  variables after cutoff+force-in: {len(variables)} -> {variables}", flush=True)
    assert len(variables) >= MIN_KEPT, f"Only {len(variables)} vars left after β drop"

    X = df[variables].astype(float)
    res = fit_logit(X, y)

    # ---------------- VIF loop (anchor-protected) ----------------
    banner(f"VIF loop — dropping non-anchor features with VIF > {VIF_MAX} (anchors protected)")
    dropped_vif = []
    for iteration in range(20):
        vifs = compute_vif(X)
        # Consider only NON-anchor variables as drop candidates.
        droppable = {v: q for v, q in vifs.items() if v not in FORCE_IN_ANCHORS}
        if not droppable:
            print("  only anchors remain — VIF loop complete", flush=True)
            break
        max_var = max(droppable, key=lambda k: droppable[k])
        max_vif = droppable[max_var]
        # Report the true global max VIF too (may be an anchor, which we keep).
        glob_var = max(vifs, key=lambda k: vifs[k])
        print(f"  iter {iteration:2d}: max droppable VIF={max_vif:.2f} ({max_var}) | "
              f"global max VIF={vifs[glob_var]:.2f} ({glob_var})", flush=True)
        if max_vif <= VIF_MAX:
            break
        near_max = [v for v, q in droppable.items() if q >= max_vif - 0.5]
        drop_var = min(near_max, key=lambda v: freq_lookup.get(v, 0.0)) if len(near_max) > 1 else max_var
        print(f"    -> dropping {drop_var} (freq={freq_lookup.get(drop_var, 0):.2f})", flush=True)
        dropped_vif.append(drop_var)
        variables = [v for v in variables if v != drop_var]
        if len(variables) < MIN_KEPT:
            raise RuntimeError(f"VIF loop reduced features below MIN_KEPT={MIN_KEPT}")
        X = df[variables].astype(float)
        res = fit_logit(X, y)
    else:
        raise RuntimeError("VIF loop did not converge in 20 iterations")

    # ---------------- Final coefficients ----------------
    banner("Final coefficients")
    params = res.params
    bse = res.bse
    conf = res.conf_int()
    conf.columns = ["CI_low", "CI_high"]
    intercept = float(params.loc["const"])
    print(f"  intercept = {intercept:.4f}  |  {len(variables)} predictors", flush=True)

    # ---- Split the surviving variables into point-role and override-role ----
    # (Fix 4 / CONTRACTS §5.1.) Both roles are in the fitted model — only the
    # point-role ones reach the integer card.
    point_vars = [v for v in variables if v not in OVERRIDE_VARS]
    override_vars = [v for v in variables if v in OVERRIDE_VARS]
    assert set(point_vars).isdisjoint(set(override_vars)), \
        "point_vars and override_vars must be disjoint (CONTRACTS §5.1e)"
    assert sorted(point_vars + override_vars) == sorted(variables), \
        "role split lost or duplicated a variable"
    if override_vars:
        print(f"  OVERRIDE role (off the point scale, still in the refit): "
              f"{override_vars}", flush=True)
        for v in override_vars:
            print(f"    {v}  ->  \"{DISPLAY_LABELS.get(v, v)}\"", flush=True)
    else:
        print("  NOTE: no override variable survived to the final model "
              "(sbp_unrecordable was dropped by the β cutoff or the VIF loop). "
              "08 must then report the override group descriptively only.",
              flush=True)

    rows = []
    for v in variables:
        b = float(params.loc[v])
        rows.append({
            "Variable": v, "β": round(b, 4), "SE": round(float(bse.loc[v]), 4),
            "OR": round(float(np.exp(b)), 3),
            "95% CI lower": round(float(np.exp(conf.loc[v, "CI_low"])), 3),
            "95% CI upper": round(float(np.exp(conf.loc[v, "CI_high"])), 3),
            "Role": ROLE_OVERRIDE if v in override_vars else ROLE_POINT,
            "_beta_raw": b,
        })

    # ---------------- Integer conversion (POINT-ROLE VARS ONLY) --------------
    banner("Integer conversion  round(β / |β_min_positive|)  ±1 floor "
           "— point-role variables only, AMENDED anchor (Fix 9 / D-11)")
    # Two exclusions apply to the ANCHOR SEARCH, and they are different things:
    #
    #   (1) CONTRACTS §5.1(a) — OVERRIDE-role variables (`sbp_unrecordable`) are
    #       excluded because they are off the scale entirely; letting an off-scale
    #       β compete for the divisor would rescale every printed point value.
    #   (2) CONTRACTS §5.3 / D-11 — POINT-role variables that did NOT clear the
    #       |β| >= BETA_DROP screen are excluded because a coefficient retained by
    #       pre-registration rather than by evidence must not set the unit of the
    #       scale. This is a DEVIATION from the locked SAP's literal rule and is
    #       registered as D-11; the full statement of what the locked rule said,
    #       what it produced (-27 points for `mech_fall`), why the denominator was
    #       degenerate (`cc_anticoag` β = 0.062, OR 1.06, CI 0.98-1.16) and what
    #       the amended rule is, is in this module's docstring.
    #
    # Neither exclusion removes a variable from the model, from the card, or from
    # the report. Both only decide which coefficient sets the unit.
    beta_by_var = {r["Variable"]: r["_beta_raw"] for r in rows}
    print("  LOCKED SAP RULE   : anchor = min positive β over ALL point-role vars",
          flush=True)
    print("  AMENDED RULE (D-11): anchor = min positive β over point-role vars "
          f"with |β| >= {BETA_DROP}", flush=True)
    anchor, anchor_var, anchor_meta = select_integer_anchor(
        beta_by_var, point_vars, BETA_DROP, force_in_anchors=FORCE_IN_ANCHORS)
    print(f"  anchor |β_min_positive| = {anchor:.4f} ({anchor_var})", flush=True)
    print(f"  anchor search space: {anchor_meta['anchor_n_eligible']} eligible of "
          f"{anchor_meta['anchor_n_positive_point_vars']} positive-β point-role "
          f"vars -> {anchor_meta['anchor_eligible_vars']}", flush=True)
    print(f"  excluded (override role, §5.1a)      : {override_vars or 'none'}",
          flush=True)
    if anchor_meta["anchor_excluded_forced_in"]:
        print(f"  excluded (below the |β| >= {BETA_DROP} screen, §5.3 / D-11):",
              flush=True)
        for e in anchor_meta["anchor_excluded_forced_in"]:
            tag = "FORCE-IN ANCHOR" if e["force_in_anchor"] else "point-role"
            print(f"    {e['variable']:<18} β = {e['beta']:+.4f}  [{tag}]  "
                  f"{e['reason']}", flush=True)
    else:
        print(f"  excluded (below the |β| >= {BETA_DROP} screen, §5.3 / D-11): none",
              flush=True)
    locked = anchor_meta["anchor_locked_sap_would_have_chosen"]
    if locked["variable"] != anchor_var:
        print(f"  DEVIATION IS ACTIVE: the locked SAP rule would have divided by "
              f"{locked['beta']:.4f} ({locked['variable']}); the amended rule "
              f"divides by {anchor:.4f} ({anchor_var}). Ratio = "
              f"{anchor / float(locked['beta']):.2f}x coarser. Registered as D-11 "
              f"— report as a deviation, NOT as a bugfix.", flush=True)
    else:
        print("  Amended and locked rules coincide on this run (the smallest "
              "positive point-role β also cleared the screen).", flush=True)

    for r in rows:
        if r["Role"] == ROLE_OVERRIDE:
            # Empty/NA — never a number, never 0. A 0 would read as "this
            # variable was considered and found worthless", which is the exact
            # opposite of what an automatic-escalate override means.
            r["Points"] = None
            continue
        b = r["_beta_raw"]
        pts = int(round(b / anchor))
        # ±1 floor — UNCHANGED by D-11, including its exact condition. A variable
        # that cleared the screen but rounds to 0 against the anchor gets ±1 so it
        # does not silently leave the card. A variable BELOW the screen (only a
        # force-in anchor can be here) correctly does NOT get floored: it keeps 0.
        # That 0 is the intended, honest value — do NOT special-case it up to ±1.
        if pts == 0 and abs(b) >= BETA_DROP:
            pts = 1 if b > 0 else -1
        if b > 0:
            assert pts >= 0
        else:
            assert pts <= 0
        r["Points"] = pts

    # ---- §13 G8c — card-scale tripwire (a collapsed anchor's fingerprint) ----
    # Checked BEFORE anything is written, so a degenerate scale never reaches an
    # artifact. Not a scientific target: if this fires, inspect the ANCHOR first.
    _oversized = {r["Variable"]: r["Points"] for r in rows
                  if r["Role"] == ROLE_POINT and abs(int(r["Points"])) > CARD_SCALE_TRIPWIRE}
    assert not _oversized, (
        f"§13 G8c TRIPWIRE: point value(s) exceed ±{CARD_SCALE_TRIPWIRE} on the "
        f"bedside card: {_oversized}. This is the fingerprint of a COLLAPSED "
        f"INTEGER ANCHOR — inspect anchor={anchor_var} (β={anchor:.4f}) before "
        f"changing anything else. The pre-D-11 run tripped this at -35..+96 "
        f"because `cc_anticoag` (β 0.062) was the divisor (CONTRACTS §5.3).")

    # ---- Zero-point point-role variables — REPORTED, never hidden (§5.3) ----
    # A pre-registered force-in anchor that earns 0 points is a FINDING, not a
    # rendering defect. Emitted as STRUCTURED DATA (not merely a printed row) so
    # 08's banner, 16's narrative and render_figures' F9 can STATE it rather than
    # rediscover it. `08` already reads `zero_point_force_in_anchors`.
    zero_point_rows = []
    for r in rows:
        if r["Role"] != ROLE_POINT or int(r["Points"]) != 0:
            continue
        v = r["Variable"]
        blk = {
            "variable": v,
            "label": DISPLAY_LABELS.get(v, v),
            "beta": float(r["_beta_raw"]),
            "se": float(r["SE"]),
            "or": float(r["OR"]),
            "ci_low": float(r["95% CI lower"]),
            "ci_high": float(r["95% CI upper"]),
            "points": 0,
            "force_in_anchor": bool(v in FORCE_IN_ANCHORS),
            "passed_beta_screen": bool(abs(float(r["_beta_raw"])) >= BETA_DROP),
            "significant_at_0_05": bool(float(r["95% CI lower"]) > 1.0
                                        or float(r["95% CI upper"]) < 1.0),
            "beta_screen_cutoff": float(BETA_DROP),
            "retention_reason": ("pre-registered force-in anchor (locked SAP §4) — "
                                 "retained regardless of the |β| screen"
                                 if v in FORCE_IN_ANCHORS else
                                 "cleared the |β| screen but rounds to 0 against "
                                 "the anchor"),
            "disposition": ("STAYS ON THE CARD AT 0 POINTS. Pre-registered; not "
                            "dropped, not hidden, not tidied off F9 (CONTRACTS "
                            "§5.3, §13 G8d). The ±1 floor correctly does not fire "
                            "(|β| < BETA_DROP) and it is NOT special-cased up "
                            "to ±1." if v in FORCE_IN_ANCHORS else
                            "reported at 0 points."),
        }
        if v in V2_REFERENCE_POINTS:
            blk["v2_comparison"] = dict(V2_REFERENCE_POINTS[v])
            blk["v2_comparison"]["outcome"] = V2_OUTCOME_LABEL
            blk["v2_comparison"]["note"] = V2_REFERENCE_NOTE
        zero_point_rows.append(blk)

    if zero_point_rows:
        banner("Zero-point POINT-ROLE variables — reported, never dropped (§5.3)")
        for b in zero_point_rows:
            tag = "PRE-REGISTERED FORCE-IN ANCHOR" if b["force_in_anchor"] else "point-role"
            print(f"  *** {b['variable']} = 0 POINTS  [{tag}] ***", flush=True)
            print(f"      β = {b['beta']:+.4f}   OR = {b['or']:.3f} "
                  f"(95% CI {b['ci_low']:.3f}-{b['ci_high']:.3f})   "
                  f"{'SIGNIFICANT' if b['significant_at_0_05'] else 'NON-SIGNIFICANT'}",
                  flush=True)
            print(f"      {b['disposition']}", flush=True)
            v2 = b.get("v2_comparison")
            if v2:
                print(f"      v2 COMPARISON ONLY (outcome {v2['outcome']}): "
                      f"β = {v2['beta']:+.4f}, OR = {v2['or']:.3f} "
                      f"(95% CI {v2['ci_low']:.3f}-{v2['ci_high']:.3f}), "
                      f"{v2['points']:+d} points.", flush=True)
                print(f"      FINDING: the term moved from OR {v2['or']:.2f} under "
                      f"v2's all-cause-death composite to OR {b['or']:.2f} "
                      f"(CI {b['ci_low']:.2f}-{b['ci_high']:.2f}) under the v3 "
                      f"hemorrhage-specific outcome. Report it as a result "
                      f"(CONTRACTS §5.3), not as a scaling nuisance.", flush=True)

    # ---- §13 G8f — model-identity evidence, gathered BEFORE we overwrite -----
    # D-11's defence — "this changed the card's legibility, not the result" — is
    # the most scrutinised sentence in the amendment, so it is EVIDENCED here, not
    # asserted. What must be provable: the MODEL is unchanged (β vector,
    # intercept, point_vars/override_vars all element-wise identical to the prior
    # run). What must NOT be asserted: that the integer AUROC is unchanged.
    #
    # WHY NOT. Rank-invariance holds for the CONTINUOUS LOGIT — nothing about the
    # fit moves, so `training_auroc_logit` must reproduce to the digit. The
    # INTEGER card is a coarser QUANTISATION, not a strictly monotone relabelling:
    # `round(β/0.062)` -> `round(β/0.3407)` lets patient pairs that had distinct
    # totals TIE, and a tie contributes 0.5 to AUROC. So the integer AUROC can
    # move — most likely slightly DOWN, because resolution is being discarded on
    # purpose. Both deltas are REPORTED below and NEITHER is asserted; an
    # integer-AUROC-invariance assert would be a gate designed to fail.
    d11_check = {"prior_artifacts_found": False,
                 "note": ("§13 G8f — the claim under test is MODEL IDENTITY (β "
                          "vector, intercept, role split), not AUROC invariance."),
                 "auroc_interpretation": (
                     "training_auroc_logit MUST reproduce to the digit — the fit "
                     "is untouched. training_auroc_integer MAY move, most likely "
                     "slightly down: the new anchor is a coarser quantisation, "
                     "distinct totals can collapse to ties, and ties score 0.5. "
                     "Report the delta as the cost of legibility; do not assert "
                     "it is zero.")}
    # Baseline preference: the SHA-manifested pre-D-11 snapshot, which stays
    # pre-D-11 forever, over the in-place artifacts, which stop being pre-D-11 the
    # moment this script runs once.
    _snap_w = os.path.join(PRE_D11_DIR, "table3_points.csv")
    _snap_j = os.path.join(PRE_D11_DIR, "refit.json")
    if os.path.exists(_snap_w):
        _base_w, _base_j, _base_src = _snap_w, _snap_j, "logs/pre_d11_snapshot (SHA-manifested)"
    else:
        _base_w, _base_j, _base_src = OUT_WEIGHTS, OUT_JSON, "in-place artifacts (not yet overwritten)"
    d11_check["baseline_source"] = _base_src
    d11_check["baseline_table3"] = os.path.relpath(_base_w, config.REPO_ROOT)
    try:
        if os.path.exists(_base_w):
            prior = pd.read_csv(_base_w)
            prior_beta = {str(k): float(v) for k, v in
                          zip(prior["Variable"], prior["β"])}
            prior_pts = {str(k): (None if pd.isna(v) else int(v)) for k, v in
                         zip(prior["Variable"], pd.to_numeric(prior["Points"],
                                                              errors="coerce"))}
            new_beta = {r["Variable"]: float(r["β"]) for r in rows}
            new_pts = {r["Variable"]: (None if r["Points"] is None else int(r["Points"]))
                       for r in rows}
            shared = sorted(set(prior_beta) & set(new_beta))
            max_diff = max((abs(prior_beta[v] - new_beta[v]) for v in shared),
                           default=0.0)
            d11_check.update({
                "prior_artifacts_found": True,
                "variables_identical": sorted(prior_beta) == sorted(new_beta),
                "beta_identical_to_4dp": bool(max_diff <= 1e-4),
                "beta_max_abs_diff": float(max_diff),
                "prior_points": prior_pts,
                "new_points": new_pts,
                "points_changed": {v: [prior_pts.get(v), new_pts.get(v)]
                                   for v in shared if prior_pts.get(v) != new_pts.get(v)},
            })
            try:
                with open(_base_j) as _f:
                    _prior_json = json.load(_f)
                d11_check["prior_anchor_var"] = _prior_json.get("anchor_min_positive_var")
                d11_check["prior_anchor_beta"] = _prior_json.get("anchor_min_positive_beta")
                d11_check["prior_anchor_selection_rule"] = _prior_json.get(
                    "anchor_selection_rule", "LOCKED SAP literal rule (pre-D-11: "
                    "no anchor_selection_rule field was written)")
                # ---- MODEL IDENTITY: what D-11 must leave untouched ----
                _pc = _prior_json.get("coefficients") or {}
                _pi = _prior_json.get("intercept")
                _coef_diff = {v: abs(float(_pc[v]) - float(params.loc[v]))
                              for v in variables if v in _pc}
                d11_check["model_identity"] = {
                    "prior_intercept": _pi,
                    "new_intercept": float(intercept),
                    "intercept_abs_diff": (abs(float(_pi) - float(intercept))
                                           if _pi is not None else None),
                    "coefficient_max_abs_diff": (max(_coef_diff.values())
                                                 if _coef_diff else None),
                    "coefficient_vars_compared": sorted(_coef_diff),
                    "prior_point_vars": _prior_json.get("point_vars"),
                    "new_point_vars": list(point_vars),
                    "point_vars_identical": bool(
                        list(_prior_json.get("point_vars") or []) == list(point_vars)),
                    "prior_override_vars": _prior_json.get("override_vars"),
                    "new_override_vars": list(override_vars),
                    "override_vars_identical": bool(
                        list(_prior_json.get("override_vars") or []) == list(override_vars)),
                    "note": ("This is the block that substantiates 'the model is "
                             "unchanged'. It says nothing about the integer "
                             "card's discrimination — see auroc_deltas."),
                }
                # AUROC deltas: REPORTED, never asserted. Filled in below once the
                # current run's training AUROCs exist.
                d11_check["auroc_deltas"] = {
                    "prior_training_auroc_logit": _prior_json.get("training_auroc_logit"),
                    "prior_training_auroc_integer": _prior_json.get("training_auroc_integer"),
                }
            except Exception:                                   # noqa: BLE001
                pass
            banner("§13 G8f — model-identity check vs the PRE-D-11 baseline")
            print(f"  baseline          : {_base_src}", flush=True)
            print(f"  same variable set : {d11_check['variables_identical']}", flush=True)
            print(f"  β identical (4dp) : {d11_check['beta_identical_to_4dp']}  "
                  f"(max |Δβ| = {max_diff:.6f})", flush=True)
            if d11_check.get("prior_anchor_var"):
                print(f"  prior anchor      : {d11_check['prior_anchor_var']} "
                      f"(β={float(d11_check['prior_anchor_beta']):.4f})", flush=True)
            print(f"  this anchor       : {anchor_var} (β={anchor:.4f})", flush=True)
            if d11_check["points_changed"]:
                print("  Points moved (rescaling — this is the whole point of D-11):",
                      flush=True)
                for v, (a_, b_) in d11_check["points_changed"].items():
                    print(f"    {v:<18} {a_} -> {b_}", flush=True)
            if not d11_check["beta_identical_to_4dp"]:
                print("  [G8f][WARN] the β column MOVED versus the prior run. D-11 "
                      "is a rescaling only — a β change means something upstream "
                      "(01/02/03/04) also moved. Do NOT attribute it to D-11.",
                      flush=True)
        else:
            print(f"  [G8f] no pre-D-11 baseline on disk ({_base_w}) — first run; "
                  f"the model-identity comparison is not available.", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        d11_check["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [G8f][WARN] could not read the prior artifacts: {exc}", flush=True)

    weights_df = pd.DataFrame([{
        "Variable": r["Variable"], "β": r["β"], "SE": r["SE"], "OR": r["OR"],
        "95% CI lower": r["95% CI lower"], "95% CI upper": r["95% CI upper"],
        "Points": r["Points"], "Role": r["Role"],
    } for r in rows])[[
        "Variable", "β", "SE", "OR", "95% CI lower", "95% CI upper",
        "Points", "Role"
    ]]
    # object dtype so integer points render as "3" (not "3.0") and the override
    # row renders as an EMPTY cell rather than a spurious 0 / NaN literal.
    weights_df["Points"] = pd.Series(
        [r["Points"] for r in rows], index=weights_df.index, dtype=object)
    print(weights_df.to_string(index=False), flush=True)
    weights_df.to_csv(OUT_WEIGHTS, index=False)
    print(f"\n  wrote {OUT_WEIGHTS}", flush=True)

    # ---------------- Training score distribution + AUROC ----------------
    # POINT-ROLE VARS ONLY. The override contributes exactly zero to
    # pelvic_score; it acts outside the sum, at the decision rule (06/08).
    banner("Training score distribution + AUROC (point-role variables only)")
    points_map = {r["Variable"]: int(r["Points"])
                  for r in rows if r["Role"] == ROLE_POINT}
    assert set(points_map) == set(point_vars), "points_map / point_vars mismatch"
    score = np.zeros(len(df), dtype=np.int64)
    for v, pts in points_map.items():
        score += pts * df[v].astype(int).values

    # Double-counting guard, training-side mirror of 06's test-side assert
    # (CONTRACTS §5.1 "Double-counting guard"): recomputing the score with every
    # override column forced to 0 must be element-wise identical.
    score_zeroed = np.zeros(len(df), dtype=np.int64)
    for v, pts in points_map.items():
        col = df[v].astype(int).values
        if v in OVERRIDE_VARS:
            col = np.zeros_like(col)
        score_zeroed += pts * col
    assert np.array_equal(score, score_zeroed), (
        "OVERRIDE LEAKED INTO THE POINT SUM: zeroing the override columns "
        "changed pelvic_score. An override must contribute 0 points "
        "(CONTRACTS §5.1).")
    print(f"  [OK] double-counting guard: override contributes 0 to the point sum",
          flush=True)

    # Theoretical column sums, reported separately from the observed range so
    # nobody confuses "sum of the printed points" with "highest score a real
    # patient got" (CONTRACTS §13 G4c — the v2 +32 was the OBSERVED max, while
    # the printed column summed to +47 unconstrained / +45 once mutually
    # exclusive levels were respected). 08 owns the authoritative observed
    # min/max on the TEST set; this is the training-side preview.
    theo_max_unconstrained = int(sum(p for p in points_map.values() if p > 0))
    theo_min_unconstrained = int(sum(p for p in points_map.values() if p < 0))
    print(f"  theoretical point-column sum (unconstrained): "
          f"{theo_min_unconstrained}..{theo_max_unconstrained}", flush=True)
    print(f"  observed TRAIN integer-score range: "
          f"{int(np.min(score))}..{int(np.max(score))}  "
          f"(08 reports the authoritative TEST-set range)", flush=True)

    train_auroc = float(roc_auc_score(y, score))
    dist = {
        "min": int(np.min(score)), "max": int(np.max(score)),
        "mean": float(np.mean(score)), "median": float(np.median(score)),
        "p25": float(np.percentile(score, 25)), "p75": float(np.percentile(score, 75)),
        "p90": float(np.percentile(score, 90)), "p95": float(np.percentile(score, 95)),
    }
    logit_pred = res.predict(sm.add_constant(df[variables].astype(float), has_constant="add"))
    logit_auroc = float(roc_auc_score(y, logit_pred))
    print(f"  integer score range {dist['min']}..{dist['max']}  mean={dist['mean']:.2f}", flush=True)
    print(f"  training AUROC integer={train_auroc:.4f}  continuous-logit={logit_auroc:.4f}", flush=True)

    # ---- §13 G8f — the cost of D-11's legibility, REPORTED not asserted ------
    _ad = d11_check.setdefault("auroc_deltas", {})
    _ad["training_auroc_logit"] = float(logit_auroc)
    _ad["training_auroc_integer"] = float(train_auroc)
    _pl, _pi_auc = _ad.get("prior_training_auroc_logit"), _ad.get("prior_training_auroc_integer")
    if _pl is not None:
        _ad["delta_logit"] = float(logit_auroc) - float(_pl)
        _ad["logit_identical_to_4dp"] = bool(abs(_ad["delta_logit"]) <= 1e-4)
    if _pi_auc is not None:
        _ad["delta_integer"] = float(train_auroc) - float(_pi_auc)
    _ad["interpretation"] = (
        "delta_logit MUST be ~0 — the fit is untouched, so a non-zero value means "
        "something upstream moved and is NOT attributable to D-11. delta_integer "
        "MAY be non-zero and is expected to be slightly negative: the amended "
        "anchor is a coarser quantisation, so patient pairs that had distinct "
        "totals can tie, and ties contribute 0.5 to AUROC. That delta is the cost "
        "of legibility and is reported as such — it is never asserted to be zero.")
    if _pl is not None or _pi_auc is not None:
        print("  [G8f] training-AUROC deltas vs the prior run (REPORTED, not asserted):",
              flush=True)
        if _pl is not None:
            print(f"      continuous logit : {float(_pl):.4f} -> {logit_auroc:.4f}  "
                  f"(Δ {_ad['delta_logit']:+.6f})  "
                  f"{'identical to 4dp — as required' if _ad['logit_identical_to_4dp'] else 'MOVED — investigate upstream, NOT D-11'}",
                  flush=True)
        if _pi_auc is not None:
            print(f"      integer card     : {float(_pi_auc):.4f} -> {train_auroc:.4f}  "
                  f"(Δ {_ad['delta_integer']:+.6f})  — coarser quantisation; a "
                  f"small negative delta is expected and is the cost of a legible "
                  f"card, not a defect", flush=True)

    # ---------------- refit.json (frozen weights for 06 + 12) ----------------
    # Structure (documented for downstream): a self-contained frozen model.
    #   intercept                : float   (continuous-logit intercept on TRAIN)
    #   coefficients             : {var: β} (continuous-logit β for EVERY var in
    #                              the fitted model, INCLUDING override vars —
    #                              12 uses these + intercept to form the frozen
    #                              linear predictor, and the model genuinely was
    #                              fitted with the override in it. `pelvic_lp` is
    #                              the continuous logit; `pelvic_score` is the
    #                              integer card. They are different objects and
    #                              the override belongs in exactly one of them.)
    #   points                   : {var: int} POINT-ROLE VARS ONLY. An override
    #                              var never appears here, so a naive
    #                              `sum(points[v]*X[v])` can never double-count.
    #   point_vars / override_vars : the Fix-4 role split, asserted disjoint
    #                              (CONTRACTS §5.1d/e)
    #   override_labels          : canonical display strings — consumers must
    #                              use these, not invent their own (§5.1 label)
    #   override_rule_text       : the deployed bedside sentence for 08's card
    #   variables_kept           : final predictor list (order stable; BOTH roles)
    #   forced_in_anchors        : anchors present in the final model
    #   variables_dropped_beta / _vif : audit trail
    #   anchor_min_positive_beta : the integer-conversion divisor. Searched over
    #                              point-role vars that PASSED the |β| screen
    #                              (Fix 9 / D-11, CONTRACTS §5.3) — NOT over every
    #                              point-role var, which is the locked SAP's
    #                              literal rule and is a registered deviation.
    #   anchor_min_positive_var  : which variable set the unit
    #   anchor_selection_rule    : the PARITY TOKEN — 12 and 13 compare their own
    #                              literal against this string (§13 G8e)
    #   anchor_eligible_vars     : the screen-passing positive point-role set
    #                              actually searched (§13 G8b)
    #   anchor_excluded_forced_in: point-role vars excluded from the search, each
    #                              with its β and the reason (§13 G8b). A run that
    #                              does not record this cannot be audited.
    #   zero_point_force_in_anchors / zero_point_point_vars : structured record of
    #                              any point-role variable at 0 points, with β, OR,
    #                              CI and the v2 comparison, so 08/16/F9 can STATE
    #                              it as a result (CONTRACTS §5.3 note, §13 G8d)
    #   d11_rescaling_only_check : §13 G8f evidence
    #   training_* / n_train*    : provenance
    out = {
        "outcome": OUTCOME_COL,
        "intercept": intercept,
        "coefficients": {v: float(params.loc[v]) for v in variables},
        "points": {v: int(points_map[v]) for v in point_vars},
        "point_vars": list(point_vars),
        "override_vars": list(override_vars),
        "override_labels": {v: DISPLAY_LABELS.get(v, v) for v in override_vars},
        "override_rule_text": OVERRIDE_RULE_TEXT if override_vars else None,
        "variables_kept": variables,
        "forced_in_anchors": [a for a in FORCE_IN_ANCHORS if a in variables],
        "variables_dropped_beta": dropped_beta,
        "variables_dropped_vif": dropped_vif,
        "anchor_min_positive_beta": float(anchor),
        "anchor_min_positive_var": anchor_var,
        # ---- Fix 9 / D-11 provenance (CONTRACTS §5.3, §13 G8b) ----
        **anchor_meta,
        "anchor_deviation": {
            "id": "D-11",
            "kind": ("DEVIATION from the locked SAP's literal integer-conversion "
                     "rule — not a bugfix"),
            "locked_sap_rule": ("integer points round(β/|β_min_pos|) with a ±1 "
                                "floor, with force-in anchors immune to the "
                                "|β| >= 0.15 screen (v2/SAP_lock.md §4-§5); read "
                                "literally, |β_min_pos| ranges over EVERY retained "
                                "point-role variable, including a force-in anchor "
                                "retained only because it was pre-registered"),
            "what_it_produced": ("under the v3 outcome the force-in anchor "
                                 "cc_anticoag collapsed to β = 0.062 (OR 1.06, "
                                 "95% CI 0.976-1.159, non-significant) because "
                                 "restricting the death arm to early hemorrhagic "
                                 "death removed the late deaths in which "
                                 "anticoagulation mattered; it became the "
                                 "|β_min_positive| denominator and divided every "
                                 "other coefficient by 0.062, giving mech_fall "
                                 "-27, sbp_lt90 +18, gcs_motor_le5 +15 and an "
                                 "observed score range of -35..+96 (v2: -7..+32)"),
            "why_degenerate": ("the unit of the scale was set by a coefficient "
                               "statistically indistinguishable from zero, so "
                               "every point value became an artefact of a null "
                               "term; it also defeated Fix 4 / D-4, whose entire "
                               "purpose was the objection that a +16 "
                               "weight is implausible on a bedside card — a -27 "
                               "row is worse than the thing objected to"),
            "amended_rule": ANCHOR_SELECTION_RULE,
            "unchanged": ("same round(β/anchor); same ±1 floor and its exact "
                          "condition (pts == 0 and abs(β) >= BETA_DROP); same "
                          "anchor immunity from the β-drop and the VIF loop; same "
                          "override exclusion (§5.1)"),
            "justification": ("A coefficient retained by pre-registration rather "
                              "than by evidence should not set the unit of the "
                              "scale. The |β| >= 0.15 screen exists precisely to "
                              "identify coefficients large enough to be "
                              "meaningful; the denominator should come from that "
                              "set. The force-in anchor is still retained and "
                              "reported — it simply no longer defines the unit."),
            "scope": ("The MODEL is unchanged — no coefficient, no intercept, no "
                      "selected variable and no role assignment moves. Only the "
                      "integer CARD is rescaled."),
            "auroc_claim": (
                "Do NOT claim the integer AUROC is unchanged. Rank-invariance "
                "holds for the CONTINUOUS LOGIT, which must be identical to the "
                "digit because nothing about the fit changes. The INTEGER card is "
                "a coarser QUANTISATION, not a strictly monotone relabelling: "
                "pairs with distinct totals under the old anchor can tie under "
                "the new one and ties score 0.5, so the integer AUROC can move "
                "and will most likely move slightly DOWN. The defensible claim is "
                "'the model is unchanged; the card is a coarser rendering of it; "
                "here is the cost in discrimination'. Both deltas are REPORTED in "
                "d11_rescaling_only_check and neither is asserted."),
            "duplicated_in": ["05_refit_integer.py",
                              "12_temporal_external_validation.py",
                              "13_sensitivity_components.py"],
            "authority": ["CONTRACTS.md §5.3", "CONTRACTS.md §13 G8",
                          "SAP_amendment_v3.md D-11"],
        },
        "zero_point_force_in_anchors": [b for b in zero_point_rows
                                        if b["force_in_anchor"]],
        "zero_point_point_vars": zero_point_rows,
        # v2 side-by-side for the force-in anchors, so 16 can make the
        # "predictive under v2's all-cause composite, null under the v3
        # hemorrhage-specific outcome" point without reaching into the frozen
        # v2 fork. LABEL AS A v2 COMPARISON WHEREVER QUOTED (§13 G5b).
        "v2_reference_points": {
            "note": V2_REFERENCE_NOTE,
            "v2_outcome": V2_OUTCOME_LABEL,
            "v3_outcome": OUTCOME_COL,
            "variables": {
                v: {**V2_REFERENCE_POINTS[v],
                    "v3_beta": float(params.loc[v]),
                    "v3_or": float(np.exp(float(params.loc[v]))),
                    "v3_ci_low": float(np.exp(conf.loc[v, "CI_low"])),
                    "v3_ci_high": float(np.exp(conf.loc[v, "CI_high"])),
                    "v3_points": (int(points_map[v]) if v in points_map else None)}
                for v in V2_REFERENCE_POINTS if v in variables},
        },
        "d11_rescaling_only_check": d11_check,
        "theoretical_point_sum_unconstrained": [theo_min_unconstrained,
                                                theo_max_unconstrained],
        "training_score_distribution": dist,
        "training_auroc_integer": train_auroc,
        "training_auroc_logit": logit_auroc,
        "n_train": int(len(df)),
        "n_train_events": int(y.sum()),
        "beta_cutoff": float(BETA_DROP),
        "n_selected_before_cutoff": int(n_before_cutoff),
        "lasso_x_cols": LASSO_X_COLS,
        "force_in_spec": FORCE_IN_ANCHORS,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {OUT_JSON}", flush=True)

    # ---------------- Sanity assertions ----------------
    banner("Contract assertions (CONTRACTS §13 G3a-G3c, G4a, G8a-G8d)")
    assert len(variables) >= MIN_KEPT, f"Only {len(variables)} vars survived"
    for a in FORCE_IN_ANCHORS:
        assert a in variables, f"ANCHOR {a} missing from final model — force-in failed"

    # ---- G8a — the chosen anchor passed the screen and is a point-role var ----
    # (select_integer_anchor() asserts this at source; re-asserted here against
    # the WRITTEN artifact so a future edit to the emission cannot bypass it.)
    assert out["anchor_min_positive_var"] in out["point_vars"], (
        f"§13 G8a: anchor {out['anchor_min_positive_var']!r} is not a point-role "
        f"variable.")
    assert float(out["anchor_min_positive_beta"]) > 0, "§13 G8a: anchor β must be > 0"
    if not out["anchor_degenerate_fallback"]:
        assert abs(float(out["anchor_min_positive_beta"])) >= BETA_DROP, (
            f"§13 G8a: anchor {out['anchor_min_positive_var']!r} has "
            f"|β| = {abs(float(out['anchor_min_positive_beta'])):.4f} < screen "
            f"{BETA_DROP}. A variable retained solely by force-in must not set the "
            f"integer scale (CONTRACTS §5.3 / D-11).")
        assert out["anchor_min_positive_var"] in out["anchor_eligible_vars"], (
            "§13 G8a: the chosen anchor is not in the screen-passing eligible set.")
    # G8b — provenance recorded. A run that cannot be audited FAILS.
    for _k in ("anchor_selection_rule", "anchor_eligible_vars",
               "anchor_excluded_forced_in", "anchor_min_positive_var",
               "anchor_min_positive_beta"):
        assert _k in out, f"§13 G8b: refit.json is missing {_k!r} — unauditable run."
    assert out["anchor_selection_rule"] == ANCHOR_SELECTION_RULE, \
        "§13 G8e: the recorded anchor rule is not this script's literal."
    print(f"  [OK] G8a/G8b anchor = {out['anchor_min_positive_var']} "
          f"(β={float(out['anchor_min_positive_beta']):.4f}); eligible set "
          f"{out['anchor_eligible_vars']}; excluded "
          f"{[e['variable'] for e in out['anchor_excluded_forced_in']] or 'none'}",
          flush=True)

    # ---- G8d — a zero-point pre-registered anchor is RETAINED, not dropped ----
    # "It scored zero so I removed it" is precisely the tidy-up this gate exists
    # to prevent. Every force-in anchor must be on the card with β/OR/CI.
    _card_vars = set(weights_df["Variable"])
    for a in FORCE_IN_ANCHORS:
        assert a in _card_vars, (
            f"§13 G8d: force-in anchor {a!r} is missing from table3_points.csv. A "
            f"pre-registered anchor is reported even at 0 points — never dropped.")
        _row = weights_df[weights_df["Variable"] == a].iloc[0]
        assert pd.notna(_row["β"]) and pd.notna(_row["OR"]) \
            and pd.notna(_row["95% CI lower"]) and pd.notna(_row["95% CI upper"]), (
            f"§13 G8d: force-in anchor {a!r} is on the card without a populated "
            f"β / OR / CI.")
        if _row["Role"] == ROLE_POINT:
            assert pd.notna(_row["Points"]), (
                f"§13 G8d: force-in anchor {a!r} has an empty Points cell; a "
                f"zero-point anchor must carry an explicit 0, not a blank.")
    if out["zero_point_force_in_anchors"]:
        _z = [b["variable"] for b in out["zero_point_force_in_anchors"]]
        print(f"  [OK] G8d zero-point force-in anchor(s) retained on the card "
              f"at 0 points: {_z} — report prominently (CONTRACTS §5.3).",
              flush=True)
    assert list(weights_df.columns) == [
        "Variable", "β", "SE", "OR", "95% CI lower", "95% CI upper",
        "Points", "Role"], "table3_points.csv schema mismatch"

    # G3c — Fix 3: the ring must not appear on the card.
    assert "unstable_ring" not in set(weights_df["Variable"]), (
        "CONTRACTS §13 G3c: `unstable_ring` reached table3_points.csv. It is not "
        "an arrival-available variable and cannot be on the bedside card.")

    # G4a — Fix 4: Role column well-formed; override rows carry an EMPTY Points.
    assert set(weights_df["Role"]).issubset({ROLE_POINT, ROLE_OVERRIDE}), (
        f"Role must be in {{{ROLE_POINT!r}, {ROLE_OVERRIDE!r}}}; got "
        f"{sorted(set(weights_df['Role']))}")
    ov_rows = weights_df[weights_df["Role"] == ROLE_OVERRIDE]
    assert sorted(ov_rows["Variable"]) == sorted(override_vars), \
        "Role=='override' rows disagree with override_vars"
    assert ov_rows["Points"].isna().all(), (
        "CONTRACTS §5.1c: every Role=='override' row must have an EMPTY Points "
        "cell (never a number, never 0).")
    pt_rows = weights_df[weights_df["Role"] == ROLE_POINT]
    assert pt_rows["Points"].notna().all(), \
        "a Role=='point' row has an empty Points cell"
    assert set(out["point_vars"]).isdisjoint(set(out["override_vars"])), \
        "CONTRACTS §5.1e: refit.json point_vars / override_vars must be disjoint"
    assert set(out["points"]) == set(out["point_vars"]), \
        "refit.json 'points' must contain exactly the point_vars"

    # Round-trip the CSV exactly as a downstream consumer would read it, so a
    # formatting regression fails HERE and not four scripts later.
    rt = pd.read_csv(OUT_WEIGHTS)
    rt_pts = pd.to_numeric(rt["Points"], errors="coerce")
    assert rt_pts[rt["Role"] == ROLE_OVERRIDE].isna().all(), \
        "override Points did not round-trip as empty through the CSV"
    assert rt_pts[rt["Role"] == ROLE_POINT].notna().all(), \
        "a point row's Points did not round-trip as a number through the CSV"
    print(f"  [OK] table3_points.csv round-trip: "
          f"{len(pt_rows)} point rows, {len(ov_rows)} override rows", flush=True)

    print(f"\n  final model      = {len(variables)} predictors "
          f"({len(point_vars)} point-role + {len(override_vars)} override-role)")
    print(f"  final CARD size  = {len(point_vars)} point rows "
          f"(target 10-15; CONTRACTS §13 G5d)")
    if override_vars:
        print(f"  override rule    = {OVERRIDE_RULE_TEXT}")
    print("Step 05 complete.")


if __name__ == "__main__":
    main()
