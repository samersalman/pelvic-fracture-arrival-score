"""
Pelvic Arrival Tool v3 — Step 07: Discrimination + DeLong + calibration
================================================================================

Forked from PAT_TBI/v4/code/07_validation_metrics.py. The 4 TBI comparators are
dropped; the pelvic comparators are SI-alone + rSIG.

Scores (all handled HIGHER = worse via a per-score orientation sign):
  * pelvic_score              (integer bedside card, POINT ROWS ONLY)   sign +1
      ^^^ THE HEADLINE INSTRUMENT. Every "the score's AUROC" sentence is this row.
  * pelvic_score_or_override  (v3, Fix 4 — DIAGNOSTIC ONLY, see below)  sign +1
  * pelvic_logit              (continuous frozen linear predictor)      sign +1
  * SI-alone                  (shock_index = HR/SBP)                    sign +1
  * rSIG                      (rsig = SBP/HR·GCS, canonical)            sign -1
  * full_information_ISS_AIS  (logistic y~iss+ais_head fit+eval on TEST) — a
    "cost of a bedside tool" ceiling reference.

v3 CHANGE 1 — mortality anchor repointed (D-1 / CONTRACTS §4, §9-J)
--------------------------------------------------------------------
The mortality-only anchor is **`death_hemorrhagic`**, not `death_inhosp`. Its
authoritative definition is written by `01_build_cohort.py` into
`logs/qc_manifest.json` (`death_variants.death_hemorrhagic_definition`); as of
the frozen cohort it is `death_inhosp & (ais_head < 4) & (finaldischarge_hrs
<= 48)` — an HOURS rule. The earlier calendar-day operationalisation
(`finaldischarge_days <= 2`) is SUPERSEDED and survives only as the named
crosswalk `death_hemorrhagic_dayrule`. This script emits no definition string,
so it quotes none; scripts that DO emit one (`09`, `10`, `14`, `render_figures`)
read it from the manifest rather than re-typing it.
`death_inhosp` is retained here only as a NAMED SENSITIVITY (rung 0 of the
death ladder). Every table this script writes now carries an `Outcome` column
and an `Outcome_Role` column:

  | Outcome                  | Outcome_Role                    |
  |--------------------------|---------------------------------|
  | major_hemorrhage_event   | primary                         |
  | patient_state_composite  | co_primary_anticircularity      |
  | death_hemorrhagic        | co_primary_mortality_anchor     |
  | death_inhosp             | sensitivity_v2_mortality_allcause |

**PRIMARY-OUTCOME ROWS ARE WRITTEN FIRST** in all three tables, so any consumer
that selects with a first-match lookup (16_generate_manuscript.py uses
`first_where(t4_disc, Score=...)`) still resolves to the primary analysis.
`derived/roc_data.pkl` holds PRIMARY-outcome curves only — F3's contract is
unchanged.

v3 CHANGE 2 — score-OR-override AUROC (Fix 4 / CONTRACTS §5.1) — REGISTERED DEVIATION
-------------------------------------------------------------------------------------
`sbp_unrecordable` left the integer point scale and became a binary
automatic-escalate override (`override_nslo`). CONTRACTS §5.1 row `07` requires
that `pelvic_score` be evaluated on the point scale over the FULL test set with
override patients scored like everyone else — because "a deterministic override
cannot enter a rank-based AUROC without being an implicit +∞ point value — so it
never does." SAP D-4 authorises both-rules reporting only for `08`'s operating
table.

**The PRIMARY row is, and stays, `pelvic_score` on the point scale.** It is
computed exactly as §5.1 requires and is written FIRST in every table.

In ADDITION — and as an explicitly registered deviation from the §5.1 sentence
above, pending the contract owner's amendment — the discrimination of the rule
that is actually DEPLOYED is also reported:

    pelvic_score_or_override = pelvic_score, except override rows are promoted
                               to max(pelvic_score) + 1  (the implicit +∞)

Reporting the discrimination of the deployed rule is legitimate — arguably
necessary — but it is NOT the score's own discrimination, so it is fenced off:

  * `AUROC_type`     = `integer_bedside_override_promoted`
  * `Instrument_Role`= `deployed_rule_diagnostic_NOT_the_score_AUROC`
  * `Note`           = a one-line self-description containing the literal words
                       "deployed rule (score OR override)" and "do NOT quote as
                       'the score's AUROC'", so a reader of the RAW CSV cannot
                       mistake the row for the headline.

DeLong: the override-promoted rule is **not** a pre-registered comparator. A
comparison between an override-promoted decision rule and a continuous
physiologic index is not the pre-registered comparison, so those pairs are
retained only as EXPLORATORY and carry `Comparison_Role =
exploratory_not_preregistered` plus a Note saying so. The pre-registered gate
remains `pelvic_score` vs `SI-alone`.

The full sensitivity/specificity consequence of the override at every cutoff
lives in `08`'s `table5c_operating_full.csv` (that is the SAP D-4 artifact).
`derived/roc_data.pkl` also carries the override-promoted curve, but F3's draw
list in `render_figures.figure3_roc` is an explicit 4-curve whitelist that does
not include it, so it cannot reach the headline ROC figure by accident.

Retained from v2 (CONTRACTS §7 row 07, §9-D):
  * bootstrap AUROC 95% CI, 2000× (primary; `--anchor-bootstraps` for anchors)
  * DeLong pelvic_score vs SI-alone AND vs rSIG (the pre-registered pair is
    pelvic_score vs SI-alone), plus pelvic_logit and — exploratory only — the
    override-promoted rule
  * full-information (ISS/AIS) logistic benchmark AUROC
  * Harrell bootstrap-optimism-corrected AUROC (primary outcome only)

Outputs
-------
  v3/tables/table4_discrimination.csv    (+ Outcome, Outcome_Role, Instrument_Role, Note)
  v3/tables/table4_delong.csv            (+ Outcome, Outcome_Role, Comparison_Role, Note)
  v3/tables/table5_calibration.csv       (+ Outcome, Outcome_Role, Instrument_Role, Note)
  v3/derived/roc_data.pkl                (fig-data for render_figures F3 —
                                          PRIMARY outcome curves only)

Run
---
    python 07_validation.py
    python 07_validation.py --sample 50000     # smoke

Seed: 42
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SEED = 42
N_BOOT_DEFAULT = 2000
ANCHOR_BOOT_DEFAULT = 500
OPT_BOOT_DEFAULT = 200
EPS = 1e-6

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES

IN_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")
OUT_DISC = os.path.join(TABLES, "table4_discrimination.csv")
OUT_DELONG = os.path.join(TABLES, "table4_delong.csv")
OUT_CALIB = os.path.join(TABLES, "table5_calibration.csv")
OUT_PICKLE = os.path.join(DERIVED, "roc_data.pkl")

OUTCOME_COL = "major_hemorrhage_event"          # PRIMARY (CONTRACTS §4)
OVERRIDE_COL = "override_nslo"                  # Fix 4 (CONTRACTS §5.1)
FULLINFO_VARS = ["iss", "ais_head"]

# (outcome column, Outcome_Role). PRIMARY MUST BE FIRST — see module docstring.
ANCHOR_OUTCOMES = [
    ("major_hemorrhage_event",  "primary"),
    ("patient_state_composite", "co_primary_anticircularity"),
    ("death_hemorrhagic",       "co_primary_mortality_anchor"),
    ("death_inhosp",            "sensitivity_v2_mortality_allcause"),
]

# (display, score column, orientation sign, kind)  — higher(sign*score) == worse
# ORDER IS LOAD-BEARING: `pelvic_score` (the point scale) is the PRIMARY
# instrument and MUST be emitted first within every outcome block so that a
# first-match lookup in `16` resolves to it (see `_ordered`).
SCORES = [
    ("pelvic_score",             "pelvic_score",             +1, "integer_bedside"),
    ("pelvic_score_or_override", "pelvic_score_or_override", +1, "integer_bedside_override_promoted"),
    ("pelvic_logit",             "pelvic_lp",                +1, "continuous_frozen_logit"),
    ("SI-alone",                 "shock_index",              +1, "comparator"),
    ("rSIG",                     "rsig",                     -1, "comparator"),
]
SCORE_ORDER = ["pelvic_score", "pelvic_score_or_override", "pelvic_logit",
               "SI-alone", "rSIG", "full_information_ISS_AIS"]

# ---------------------------------------------------------------------------
# Self-describing row labelling (CONTRACTS §5.1 deviation fencing).
#
# `Instrument_Role` + `Note` travel with EVERY discrimination / calibration row
# so the raw CSV is unambiguous on its own — a reader who never opens the
# manuscript, the contract, or this docstring still cannot mistake the
# override-promoted diagnostic for the headline score AUROC.
# ---------------------------------------------------------------------------
HEADLINE_SCORE = "pelvic_score"
INSTRUMENT_ROLE = {
    "pelvic_score": "PRIMARY_headline_integer_score_point_scale",
    "pelvic_score_or_override": "deployed_rule_diagnostic_NOT_the_score_AUROC",
    "pelvic_logit": "continuous_frozen_model",
    "SI-alone": "prespecified_comparator",
    "rSIG": "prespecified_comparator",
    "full_information_ISS_AIS": "ceiling_benchmark_not_bedside_available",
    "pelvic_logit_optimism_corrected": "internal_validity_diagnostic",
}
INSTRUMENT_NOTE = {
    "pelvic_score": (
        "PRIMARY. This is the score's own discrimination on the integer point "
        "scale over the full test set, override patients scored like everyone "
        "else (CONTRACTS §5.1 row 07). Quote THIS row as 'the score's AUROC'."),
    "pelvic_score_or_override": (
        "DIAGNOSTIC for the deployed rule (score OR override): override rows "
        "promoted to max(pelvic_score)+1, the implicit +infinity point value. "
        "Do NOT quote as 'the score's AUROC' — the headline is the "
        "'pelvic_score' row above. Registered deviation from CONTRACTS §5.1; "
        "the pre-registered both-rules artifact is 08's "
        "table5c_operating_full.csv (SAP D-4)."),
    "pelvic_logit": (
        "Continuous frozen linear predictor; not the bedside instrument."),
    "SI-alone": "Pre-specified physiologic comparator (shock index).",
    "rSIG": "Pre-specified physiologic comparator (reverse shock index x GCS).",
    "full_information_ISS_AIS": (
        "Ceiling reference using registrar-coded ISS/AIS, which are NOT "
        "available on arrival; not a deployable alternative."),
    "pelvic_logit_optimism_corrected": (
        "Harrell bootstrap-optimism-corrected continuous refit; an internal-"
        "validity diagnostic, not a reportable instrument AUROC."),
}

# Pairwise DeLong targets: (score_1, score_2, Comparison_Role, Note).
# The PRE-REGISTERED gate is pelvic_score vs SI-alone. The override-promoted
# rule is NOT a pre-registered comparator — a deterministic decision rule versus
# a continuous physiologic index is not the registered comparison — so those
# pairs are retained EXPLORATORY-ONLY and are labelled as such in the CSV.
_EXPLORATORY_OVERRIDE_NOTE = (
    "EXPLORATORY. Score_1 is the deployed rule (score OR override) with "
    "override rows promoted to the implicit +infinity point value; comparing it "
    "to a continuous physiologic index is NOT the pre-registered comparison. "
    "Do not quote as the headline discrimination claim.")
DELONG_PAIRS = [
    ("pelvic_score", "SI-alone", "PRE_REGISTERED_PRIMARY",
     "Pre-registered discrimination gate (CONTRACTS §7 row 07)."),
    ("pelvic_score", "rSIG", "pre_specified_secondary_comparator",
     "Pre-specified secondary comparator."),
    ("pelvic_logit", "SI-alone", "exploratory_continuous_model",
     "EXPLORATORY. Score_1 is the continuous frozen logit, not the bedside card."),
    ("pelvic_logit", "rSIG", "exploratory_continuous_model",
     "EXPLORATORY. Score_1 is the continuous frozen logit, not the bedside card."),
    ("pelvic_score_or_override", "SI-alone", "exploratory_not_preregistered",
     _EXPLORATORY_OVERRIDE_NOTE),
    ("pelvic_score_or_override", "rSIG", "exploratory_not_preregistered",
     _EXPLORATORY_OVERRIDE_NOTE),
]
CALIB_SCORES = ["pelvic_logit", "pelvic_score", "pelvic_score_or_override",
                "SI-alone", "rSIG"]

# Sort keys so the PRIMARY point-scale row lands first inside every outcome
# block regardless of append order (belt-and-braces on top of the stable sort).
SCORE_RANK = {n: i for i, n in enumerate(SCORE_ORDER)}
DELONG_RANK = {(a, b): i for i, (a, b, _r, _n) in enumerate(DELONG_PAIRS)}


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# Vectorized DeLong (Sun-Xu fast form) — reused from PAT 07.
# ---------------------------------------------------------------------------
def _midrank(x):
    return rankdata(x, method="average")


def _delong_v(y, pred):
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.float64)
    pos, neg = pred[y == 1], pred[y == 0]
    n1, n0 = len(pos), len(neg)
    tz = _midrank(pred)
    V10 = (tz[y == 1] - _midrank(pos)) / n0
    V01 = 1.0 - (tz[y == 0] - _midrank(neg)) / n1
    return V10, V01


def delong_test(y, p1, p2):
    """Return (z, p, delta, se, auc1, auc2). delta>0 => p1 better."""
    V10_1, V01_1 = _delong_v(y, p1)
    V10_2, V01_2 = _delong_v(y, p2)
    n1, n0 = V10_1.shape[0], V01_1.shape[0]
    auc1, auc2 = float(V10_1.mean()), float(V10_2.mean())
    S10 = np.cov(np.vstack([V10_1, V10_2]))
    S01 = np.cov(np.vstack([V01_1, V01_2]))
    S = S10 / n1 + S01 / n0
    delta = auc1 - auc2
    var = float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])
    if var <= 0.0:
        return 0.0, 1.0, float(delta), 0.0, auc1, auc2
    se = float(np.sqrt(var))
    z = delta / se
    p = 2.0 * (1.0 - sstats.norm.cdf(abs(z)))
    return float(z), float(p), float(delta), se, auc1, auc2


def bootstrap_auc_ci(y, s, n_boot, rng):
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


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def _safe_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_recal(y, x):
    X = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    res = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="lbfgs", maxiter=200)
    return res.predict(X)


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


def calib_metrics(y, prob):
    prob = np.clip(prob, EPS, 1 - EPS)
    citl = float(prob.mean() - y.mean())
    try:
        X = sm.add_constant(_safe_logit(prob), has_constant="add")
        slope = float(sm.Logit(y.astype(int), X).fit(disp=False, method="lbfgs",
                                                      maxiter=200).params[1])
    except Exception:
        slope = np.nan
    brier = float(brier_score_loss(y, prob))
    hl, hl_df, hl_p = hosmer_lemeshow(y, prob)
    return citl, slope, brier, hl, hl_df, hl_p


# ---------------------------------------------------------------------------
# Per-outcome analysis block
# ---------------------------------------------------------------------------
def analyse_outcome(df, y_all, oriented, outcome_name, outcome_role,
                    n_boot, rng, collect_roc):
    """Run AUROC(+CI) / DeLong / calibration for one outcome.

    Returns (disc_rows, delong_rows, calib_rows, roc_data, vecs).
    `oriented` maps display-name -> (oriented score array, usable mask, kind);
    the full-information benchmark is refit per outcome inside this function.
    """
    disc_rows, delong_rows, calib_rows, roc_data, vecs = [], [], [], {}, {}
    tag = {"Outcome": outcome_name, "Outcome_Role": outcome_role}

    # ---- full-information ISS/AIS benchmark, refit for THIS outcome ----
    fi_prob = np.full(len(df), np.nan)
    fi_mask = np.zeros(len(df), dtype=bool)
    fi_cols = [c for c in FULLINFO_VARS if c in df.columns]
    if fi_cols:
        Xfi = df[fi_cols].apply(pd.to_numeric, errors="coerce")
        fi_mask = Xfi.notna().all(axis=1).to_numpy()
        Xf = Xfi[fi_mask].to_numpy(dtype=np.float64)
        yf = y_all[fi_mask]
        if fi_mask.sum() and yf.sum() not in (0, len(yf)):
            clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(Xf, yf)
            fi_prob[fi_mask] = clf.predict_proba(Xf)[:, 1]
            print(f"  full-info benchmark ({'+'.join(fi_cols)}): "
                  f"AUROC={roc_auc_score(yf, fi_prob[fi_mask]):.4f} "
                  f"(N={int(fi_mask.sum()):,})")
    local = dict(oriented)
    local["full_information_ISS_AIS"] = (fi_prob, fi_mask, "full_information_benchmark")

    # ---- Part A: AUROC + bootstrap CI ----
    for name in SCORE_ORDER:
        if name not in local:
            continue
        s, mask, kind = local[name]
        use = mask & ~np.isnan(s)
        y_sub, s_sub = y_all[use], s[use]
        vecs[name] = (use, s)
        base = dict(tag)
        base.update({"Score": name,
                     "Instrument_Role": INSTRUMENT_ROLE.get(name, "unlabelled"),
                     "N_used": int(use.sum()),
                     "N_events": int(y_sub.sum()), "AUROC_type": kind,
                     "Note": INSTRUMENT_NOTE.get(name, "")})
        if use.sum() == 0 or y_sub.sum() in (0, len(y_sub)):
            base.update({"AUROC": np.nan, "CI_Low": np.nan, "CI_High": np.nan,
                         "Optimism": np.nan, "Optimism_Corrected_AUROC": np.nan})
            disc_rows.append(base)
            continue
        auc = float(roc_auc_score(y_sub, s_sub))
        lo, hi = bootstrap_auc_ci(y_sub, s_sub, n_boot, rng)
        base.update({"AUROC": round(auc, 4),
                     "CI_Low": round(lo, 4) if np.isfinite(lo) else np.nan,
                     "CI_High": round(hi, 4) if np.isfinite(hi) else np.nan,
                     "Optimism": np.nan, "Optimism_Corrected_AUROC": np.nan})
        disc_rows.append(base)
        if collect_roc:
            fpr, tpr, _ = roc_curve(y_sub, s_sub)
            if len(fpr) > 1000:
                idx = np.linspace(0, len(fpr) - 1, 1000).astype(int)
                fpr, tpr = fpr[idx], tpr[idx]
            roc_data[name] = {"fpr": np.asarray(fpr), "tpr": np.asarray(tpr),
                              "auc": auc, "lo": lo, "hi": hi}
        print(f"  {name:<28} N={use.sum():>8,} ev={int(y_sub.sum()):>7,} "
              f"AUROC={auc:.4f} [{lo:.4f},{hi:.4f}]")

    # ---- Part B: DeLong ----
    for n1, n2, crole, cnote in DELONG_PAIRS:
        if n1 not in vecs or n2 not in vecs:
            continue
        u1, s1 = vecs[n1]
        u2, s2 = vecs[n2]
        pm = u1 & u2 & ~np.isnan(s1) & ~np.isnan(s2)
        yp = y_all[pm]
        if pm.sum() == 0 or yp.sum() in (0, len(yp)):
            print(f"  {n1} vs {n2}: DEGENERATE")
            continue
        z, pval, delta, se, a1, a2 = delong_test(yp, s1[pm], s2[pm])
        row = dict(tag)
        row.update({"Score_1": n1, "Score_2": n2, "Comparison_Role": crole,
                    "N_paired": int(pm.sum()),
                    "AUROC_1": round(a1, 4), "AUROC_2": round(a2, 4),
                    "Delta_AUC": round(delta, 4), "SE_Delta": round(se, 6),
                    "CI_Low_Delta": round(delta - 1.96 * se, 4),
                    "CI_High_Delta": round(delta + 1.96 * se, 4),
                    "Z": round(z, 4), "DeLong_P": pval,
                    "Significant_0.05": bool(pval < 0.05),
                    "Note": cnote})
        delong_rows.append(row)
        flag = "" if crole.startswith("PRE_REGISTERED") or crole.startswith("pre_") \
            else "  [EXPLORATORY]"
        print(f"  {n1:<26} vs {n2:<10}  dAUC={delta:+.4f}  Z={z:+.2f}  "
              f"p={pval:.2e}  {'SIG' if pval < 0.05 else 'ns'}{flag}")

    # ---- Part C: Calibration ----
    for name in CALIB_SCORES:
        if name not in local:
            continue
        s, mask, kind = local[name]
        use = mask & ~np.isnan(s)
        y_sub, s_sub = y_all[use].astype(int), s[use]
        if use.sum() == 0 or y_sub.sum() in (0, len(y_sub)):
            continue
        if name == "pelvic_logit" and outcome_name == OUTCOME_COL:
            # Frozen-model calibration: sigmoid of the linear predictor (NOT
            # refit). Only meaningful against the outcome the model was fit on.
            prob = _sigmoid(s_sub)
            note = "frozen_sigmoid"
        else:
            prob = np.asarray(fit_recal(y_sub, s_sub), dtype=np.float64)
            note = "logistic_recalibrated"
        citl, slope, brier, hl, hl_df, hl_p = calib_metrics(y_sub, prob)
        row = dict(tag)
        row.update({"Score": name,
                    "Instrument_Role": INSTRUMENT_ROLE.get(name, "unlabelled"),
                    "Method": note, "N": int(use.sum()),
                    "CITL": round(citl, 6), "Calibration_Slope": round(slope, 4),
                    "Brier": round(brier, 6),
                    "HL_ChiSq": round(hl, 4) if not np.isnan(hl) else np.nan,
                    "HL_df": hl_df, "HL_P": hl_p,
                    "Note": INSTRUMENT_NOTE.get(name, "")})
        calib_rows.append(row)
        print(f"  {name:<26} CITL={citl:+.4f} slope={slope:.3f} "
              f"Brier={brier:.5f} HL p={hl_p:.3g} ({note})")

    return disc_rows, delong_rows, calib_rows, roc_data, vecs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument("--anchor-bootstraps", type=int, default=ANCHOR_BOOT_DEFAULT,
                        dest="anchor_bootstraps",
                        help="Bootstrap iterations for the non-primary anchor outcomes.")
    parser.add_argument("--opt-bootstraps", type=int, default=OPT_BOOT_DEFAULT,
                        dest="opt_bootstraps")
    parser.add_argument("--sample", type=int, default=0,
                        help="Downsample TEST rows to N (smoke); also shrinks bootstraps.")
    args = parser.parse_args()
    n_boot = int(args.bootstraps)
    anchor_boot = int(args.anchor_bootstraps)
    opt_boot = int(args.opt_bootstraps)

    t0 = time.time()
    os.makedirs(TABLES, exist_ok=True)
    banner("Step 07 — Discrimination + DeLong + calibration (pelvic v3)")
    df = pd.read_parquet(IN_PARQUET)
    print(f"  rows={len(df):,}  events={int(df[OUTCOME_COL].sum()):,} "
          f"({100*df[OUTCOME_COL].mean():.2f}%)  [PRIMARY = {OUTCOME_COL}]")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        n_boot = min(n_boot, 200)
        anchor_boot = min(anchor_boot, 100)
        opt_boot = min(opt_boot, 25)
        print(f"  [SMOKE] N={len(df):,}, n_boot={n_boot}, anchor_boot={anchor_boot}, "
              f"opt_boot={opt_boot}")

    rng = np.random.default_rng(SEED)

    # ---------------- Fix 4: build the override-promoted score ----------------
    banner("Fix 4 — score-OR-override construction (CONTRACTS §5.1)")
    if OVERRIDE_COL not in df.columns:
        raise SystemExit(
            f"FATAL — '{OVERRIDE_COL}' missing from {os.path.basename(IN_PARQUET)}.\n"
            f"  CONTRACTS §5.1 row `06` requires 06_score_test.py to emit the "
            f"binary automatic-escalate override column. Re-run 06.")
    ovr = pd.to_numeric(df[OVERRIDE_COL], errors="coerce").fillna(0).to_numpy()
    ovr = (ovr > 0)
    ps = pd.to_numeric(df["pelvic_score"], errors="coerce").to_numpy(dtype=np.float64)
    promote_to = float(np.nanmax(ps)) + 1.0
    df["pelvic_score_or_override"] = np.where(ovr, promote_to, ps)
    print(f"  override_nslo n={int(ovr.sum()):,} ({100*ovr.mean():.3f}% of test set)")
    print(f"  pelvic_score observed range {np.nanmin(ps):.0f}..{np.nanmax(ps):.0f}; "
          f"override rows promoted to {promote_to:.0f} (the implicit +inf)")
    print("  pelvic_score             = point scale ONLY  -> PRIMARY / headline AUROC")
    print("  pelvic_score_or_override = deployed rule (score OR override); a "
          "DIAGNOSTIC only.")
    print("      * registered deviation from CONTRACTS §5.1 (which forbids an "
          "override-promoted")
    print("        score entering a rank-based AUROC); SAP D-4 covers only 08's "
          "operating table.")
    print("      * every emitted row carries Instrument_Role="
          "'deployed_rule_diagnostic_NOT_the_score_AUROC' + a Note column.")
    print("      * do NOT quote as 'the score's AUROC'; do NOT use its DeLong "
          "rows as the headline claim.")

    # Build oriented score vectors (higher = worse) + masks — shared across outcomes.
    oriented = {}
    for name, col, sign, kind in SCORES:
        if col not in df.columns:
            print(f"  [warn] score column '{col}' absent — '{name}' skipped")
            continue
        raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        oriented[name] = (sign * raw, ~np.isnan(raw), kind)

    # ---------------- Run every anchor outcome ----------------
    all_disc, all_delong, all_calib = [], [], []
    roc_data = {}
    for i, (ocol, orole) in enumerate(ANCHOR_OUTCOMES):
        is_primary = (i == 0)
        if ocol not in df.columns:
            print(f"\n[warn] outcome column '{ocol}' ({orole}) absent from "
                  f"{os.path.basename(IN_PARQUET)} — skipped. 06 must carry it.")
            continue
        y = pd.to_numeric(df[ocol], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        b = n_boot if is_primary else anchor_boot
        banner(f"Outcome: {ocol}  [{orole}]   events={int(y.sum()):,} "
               f"({100*y.mean():.2f}%)   n_boot={b}")
        if y.sum() in (0, len(y)):
            print("  DEGENERATE outcome (no variation) — skipped")
            continue
        d, dl, cb, rd, _vecs = analyse_outcome(
            df, y, oriented, ocol, orole, b, rng, collect_roc=is_primary)
        all_disc += d
        all_delong += dl
        all_calib += cb
        if is_primary:
            roc_data = rd

    if not all_disc:
        raise SystemExit("FATAL — no outcome could be analysed.")

    # ---------------- Harrell optimism (PRIMARY outcome only) ----------------
    banner(f"Harrell bootstrap-optimism correction (B={opt_boot}, PRIMARY outcome, "
           f"continuous-logit refit on kept predictors)")
    try:
        with open(os.path.join(DERIVED, "refit.json")) as f:
            refit = json.load(f)
        kept = refit.get("variables_kept", [])
    except Exception as exc:
        print(f"  [warn] refit.json unreadable ({exc}) — skipping optimism correction")
        kept = []
    kept = [v for v in kept if v in df.columns]
    if kept:
        y_all = pd.to_numeric(df[OUTCOME_COL], errors="coerce").to_numpy(dtype=np.int8)
        Xk = df[kept].apply(pd.to_numeric, errors="coerce")
        km = Xk.notna().all(axis=1).to_numpy()
        Xk = Xk[km].to_numpy(dtype=np.float64)
        yk = y_all[km]
        n = len(yk)
        base = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(Xk, yk)
        auc_app = float(roc_auc_score(yk, base.predict_proba(Xk)[:, 1]))
        opt_vals = []
        for b in range(opt_boot):
            idx = rng.integers(0, n, size=n)
            yb = yk[idx]
            if yb.sum() in (0, n):
                continue
            try:
                mb = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(Xk[idx], yb)
                auc_boot = roc_auc_score(yb, mb.predict_proba(Xk[idx])[:, 1])
                auc_orig = roc_auc_score(yk, mb.predict_proba(Xk)[:, 1])
                opt_vals.append(auc_boot - auc_orig)
            except Exception:
                continue
        optimism = float(np.mean(opt_vals)) if opt_vals else np.nan
        auc_corr = auc_app - optimism if opt_vals else np.nan
        print(f"  kept={kept}")
        print(f"  apparent AUROC={auc_app:.4f}  optimism={optimism:.5f}  "
              f"corrected={auc_corr:.4f}  (n_ok={len(opt_vals)})")
        all_disc.append({
            "Outcome": OUTCOME_COL, "Outcome_Role": "primary",
            "Score": "pelvic_logit_optimism_corrected",
            "Instrument_Role": INSTRUMENT_ROLE["pelvic_logit_optimism_corrected"],
            "N_used": int(km.sum()),
            "N_events": int(yk.sum()), "AUROC": round(auc_corr, 4),
            "CI_Low": np.nan, "CI_High": np.nan,
            "AUROC_type": "harrell_optimism_corrected",
            "Optimism": round(optimism, 5),
            "Optimism_Corrected_AUROC": round(auc_corr, 4),
            "Note": INSTRUMENT_NOTE["pelvic_logit_optimism_corrected"]})
    else:
        print("  [warn] no kept predictors available — skipping optimism correction")

    # ---------------- Write tables (PRIMARY rows first) ----------------
    banner("Writing tables (Outcome / Outcome_Role first columns; PRIMARY rows first)")
    role_rank = {r: i for i, (_, r) in enumerate(ANCHOR_OUTCOMES)}

    def _ordered(rows, front, rank_fn=None):
        """Sort PRIMARY-outcome rows first, then (optionally) by instrument rank.

        The secondary key guarantees that the point-scale `pelvic_score` row is
        the FIRST row of the primary block, so a first_where(Score=...)-style
        lookup in `16` can never resolve to the override-promoted diagnostic.
        """
        d = pd.DataFrame(rows)
        if d.empty:
            return d
        d["_ord"] = d["Outcome_Role"].map(role_rank).fillna(99)
        d["_ord2"] = ([rank_fn(r) for _, r in d.iterrows()]
                      if rank_fn is not None else 0)
        d = d.sort_values(["_ord", "_ord2"], kind="stable").drop(
            columns=["_ord", "_ord2"])
        cols = [c for c in front if c in d.columns] + \
               [c for c in d.columns if c not in front]
        return d[cols]

    disc_df = _ordered(all_disc, ["Outcome", "Outcome_Role", "Score",
                                  "Instrument_Role", "N_used",
                                  "N_events", "AUROC", "CI_Low", "CI_High",
                                  "AUROC_type", "Optimism",
                                  "Optimism_Corrected_AUROC", "Note"],
                       rank_fn=lambda r: SCORE_RANK.get(str(r.get("Score")), 99))
    disc_df.to_csv(OUT_DISC, index=False)
    print(f"  wrote {OUT_DISC} ({len(disc_df)} rows)")

    del_df = _ordered(all_delong, ["Outcome", "Outcome_Role", "Score_1", "Score_2",
                                   "Comparison_Role",
                                   "N_paired", "AUROC_1", "AUROC_2", "Delta_AUC",
                                   "SE_Delta", "CI_Low_Delta", "CI_High_Delta",
                                   "Z", "DeLong_P", "Significant_0.05", "Note"],
                      rank_fn=lambda r: DELONG_RANK.get(
                          (str(r.get("Score_1")), str(r.get("Score_2"))), 99))
    del_df.to_csv(OUT_DELONG, index=False)
    print(f"  wrote {OUT_DELONG} ({len(del_df)} rows)")

    cal_df = _ordered(all_calib, ["Outcome", "Outcome_Role", "Score",
                                  "Instrument_Role", "Method", "N",
                                  "CITL", "Calibration_Slope", "Brier",
                                  "HL_ChiSq", "HL_df", "HL_P", "Note"],
                      rank_fn=lambda r: SCORE_RANK.get(str(r.get("Score")), 99))
    cal_df.to_csv(OUT_CALIB, index=False)
    print(f"  wrote {OUT_CALIB} ({len(cal_df)} rows)")

    # ---- Row-ordering + labelling guard (CONTRACTS §5.1 fencing) ----
    # 1. The primary block must OPEN with the point-scale `pelvic_score` row.
    # 2. The override-promoted row must never be unlabelled.
    prim_block = disc_df[disc_df["Outcome"] == OUTCOME_COL]
    if len(prim_block):
        first_score = str(prim_block.iloc[0]["Score"])
        if first_score != HEADLINE_SCORE:
            raise SystemExit(
                f"FATAL — the first PRIMARY-outcome discrimination row is "
                f"'{first_score}', not '{HEADLINE_SCORE}'. A first-match lookup "
                f"in 16 would resolve to the wrong instrument (CONTRACTS §5.1).")
        print(f"  ORDER GUARD: primary block opens with '{first_score}' "
              f"(point scale) — first_where() in 16 resolves to the PRIMARY.")
    ovr_rows = disc_df[disc_df["Score"] == "pelvic_score_or_override"]
    if len(ovr_rows):
        bad = ovr_rows[
            (ovr_rows["AUROC_type"] != "integer_bedside_override_promoted")
            | (ovr_rows["Instrument_Role"]
               != "deployed_rule_diagnostic_NOT_the_score_AUROC")
            | ~ovr_rows["Note"].astype(str).str.contains(
                "deployed rule (score OR override)", regex=False)]
        if len(bad):
            raise SystemExit(
                "FATAL — override-promoted rows are not fully self-labelled "
                "(AUROC_type / Instrument_Role / Note). CONTRACTS §5.1 fencing "
                "requires a raw-CSV reader to be unable to mistake them for the "
                "headline score AUROC.")
        print(f"  LABEL GUARD: {len(ovr_rows)} override-promoted row(s) carry "
              f"AUROC_type + Instrument_Role + Note (all say 'deployed rule "
              f"(score OR override)').")

    # ---------------- Part D: pickle ROC data (PRIMARY only) ----------------
    banner("Part D — pickle ROC data (fig-data F3; PRIMARY outcome curves only)")
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(roc_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {OUT_PICKLE}  ({len(roc_data)} curves: {list(roc_data)})")
    if "pelvic_score_or_override" in roc_data:
        print("  NOTE: the 'pelvic_score_or_override' curve is a deployed-rule "
              "diagnostic. render_figures.figure3_roc draws from an explicit "
              "4-curve whitelist that excludes it — do not add it to F3.")

    # ---------------- Gate summary ----------------
    banner("SCORE GATE SUMMARY (CONTRACTS §13 G5d)")
    prim = disc_df[disc_df["Outcome"] == OUTCOME_COL]
    _hdr = {"pelvic_score": "PRIMARY / headline",
            "pelvic_score_or_override": "DIAGNOSTIC — deployed rule (score OR "
                                        "override); NOT the score's AUROC",
            "pelvic_logit": "continuous frozen model"}
    for nm in ["pelvic_score", "pelvic_score_or_override", "pelvic_logit"]:
        r = prim[prim["Score"] == nm]
        if len(r):
            r = r.iloc[0]
            print(f"  {nm:<26} AUROC = {r['AUROC']}  CI=[{r['CI_Low']}, "
                  f"{r['CI_High']}]   <- {_hdr[nm]}")
    ps = prim[prim["Score"] == "pelvic_score"]
    if len(ps):
        ps = ps.iloc[0]
        try:
            print(f"    gate AUROC>=0.80: {'PASS' if float(ps['AUROC']) >= 0.80 else 'REPORT (see G5d)'} | "
                  f"lower CI>=0.77: {'PASS' if float(ps['CI_Low']) >= 0.77 else 'REPORT (see G5d)'}")
        except (TypeError, ValueError):
            print("    gate: AUROC unavailable")
    print("    NOTE (G5d): 'If discrimination falls materially versus v2, that is "
          "a result to report, not a gate to fail.'")
    if len(del_df):
        sig = del_df[(del_df["Outcome"] == OUTCOME_COL)
                     & (del_df["Score_1"] == "pelvic_score")
                     & (del_df["Score_2"] == "SI-alone")]
        if len(sig):
            s0 = sig.iloc[0]
            print(f"  DeLong pelvic_score vs SI-alone (PRE-REGISTERED gate): "
                  f"p={s0['DeLong_P']:.2e} "
                  f"{'SIG' if s0['Significant_0.05'] else 'ns'}")
        _expl = del_df[del_df["Comparison_Role"].astype(str)
                       == "exploratory_not_preregistered"]
        if len(_expl):
            print(f"  {len(_expl)} DeLong row(s) involving the override-promoted "
                  f"rule are EXPLORATORY (Comparison_Role="
                  f"'exploratory_not_preregistered') — not a headline claim.")
    anch = disc_df[(disc_df["Outcome_Role"] == "co_primary_mortality_anchor")
                   & (disc_df["Score"] == "pelvic_score")]
    if len(anch):
        a0 = anch.iloc[0]
        print(f"  MORTALITY ANCHOR ({a0['Outcome']}): pelvic_score AUROC={a0['AUROC']} "
              f"[{a0['CI_Low']}, {a0['CI_High']}]  events={a0['N_events']:,}")
    sens = disc_df[(disc_df["Outcome"] == "death_inhosp")
                   & (disc_df["Score"] == "pelvic_score")]
    if len(sens):
        s0 = sens.iloc[0]
        print(f"  SENSITIVITY (death_inhosp, v2 all-cause): pelvic_score "
              f"AUROC={s0['AUROC']}  events={s0['N_events']:,}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s\nStep 07 complete.")


if __name__ == "__main__":
    main()
