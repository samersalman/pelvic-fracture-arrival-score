"""
Pelvic Arrival Tool v3 — Step 17: MTP-equivalent secondary outcome  (Fix 7 / D-5)
================================================================================

Pre-specified SECONDARY outcome requested in review (comment F1):

    mtp_equivalent = >= 4 U  OR  >= 1000 mL  PRBC within 4 h

i.e. the **critical-administration-threshold** concept, computed with the Fix-1
unit-aware decoder (CONTRACTS §3). NTDB carries no MTP-activation flag, so the
critical-administration threshold is the closest available equivalent; that
limitation is stated in the manuscript.

Framing reference — the source of the MTP-in-pelvic-fracture question, and the
reference paper:

    PMID 25440492.  (Massive transfusion protocol in pelvic fracture; cited in
    the Introduction and Discussion per SAP amendment D-5.)


################################################################################
#  READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRODUCES                    #
################################################################################

`mtp_equivalent` is **DEFINITIONALLY IDENTICAL** to the v3 `substantial_transfusion`.
It is the *transfusion component of the primary composite*, analysed standalone
under the critical-administration-threshold framing. It is **NOT an independent
endpoint**, and **NOTHING here may be presented as independent corroboration of
the primary result**. Two column names exist only so this secondary analysis reads
clinically ("MTP-equivalent activation") without anyone having to re-derive the
rule (CONTRACTS §4 warning, §13 G1e / G7c; SAP amendment D-5).

The identity is asserted element-wise at run time, printed in plain words, and
written into `table8_mtp.csv` as a `header_note` row and an `outcome_identity`
section, so a later writer cannot miss it.

Discrimination against `mtp_equivalent` is EXPECTED to be at least as good as
against the composite — transfusion is the most physiologically proximate
component — which is precisely why it must not be read as confirmation.
################################################################################


Balanced-resuscitation subgroup — EXPLORATORY, and for a specific reason
-----------------------------------------------------------------------
The 4-hour plasma and platelet fields (`TQIP_HC_PLASMA_4HR` / `TQIP_HC_PLATE_4HR`)
are **80.68 % filled in the analytic cohort** (`plasma_4h` 80.678 %, `platelet_4h`
80.682 %); 77.6 % pool-wide; ~90 % for admission years 2020-2024 against 3.53 % in
2019, when the fields were essentially not collected.

  ** The plan's "~27.3 % filled" is WRONG and must never be quoted. ** It came
  from reading the first 1.5M rows of a YEAR-ORDERED file — a window that is
  ~73 % admission-year 2019 (CONTRACTS §3.2; SAP amendment D-5 correction of
  record). This script computes the fill rate FROM THE COHORT at run time,
  asserts it was not hard-coded, and asserts it is not the 27.3 % artifact.

The subgroup nevertheless stays **EXPLORATORY**, and the fill rate is not the
reason. Both fields carry the SAME units/mL bimodality that motivated Fix 1, and
neither has a decoder (`TQIP_HC_PLASMA_UNIT` / `TQIP_HC_PLATE_UNIT` are 0.0000 %
filled): among non-zero values ~16.4 % of plasma and ~18.8 % of platelet values
sit in the units-like band (<=100) against ~83.0 % / ~78.7 % in the mL-like band
(>=200). **Any plasma:PRBC or platelet:PRBC ratio therefore has an undetermined
scale in BOTH numerator and denominator.**

Consequence, enforced in code: **this script computes NO ratio.** It reports only
**receipt flags** (`plasma_4h > 0`, `platelet_4h > 0`), which are *scale-invariant*
— a strictly positive value means the product was given under either unit reading.
Everything else about balance is reported as a distribution, openly, with the
ambiguity stated in the table itself rather than buried.

Inputs (all artifacts of earlier steps — nothing is recomputed here)
-------------------------------------------------------------------
  v3/derived/cohort_test_scored.parquet   (06) pelvic_score / pelvic_lp / comparators
  v3/derived/candidates.parquet           (02) mtp_equivalent, substantial_transfusion,
                                               plasma_4h, platelet_4h, blood_4h
  v3/derived/test_inc_keys.csv            (03) the held-out slice
  v3/derived/refit.json                   (05) frozen weights / point-vs-override roles
  v3/tables/table3_points.csv             (05) the integer card (provenance only)

Outputs (OWNED — this script writes these and nothing else)
-----------------------------------------------------------
  v3/tables/table8_mtp.csv                sectioned long table (see SECTIONS below)
  v3/derived/mtp_dca_data.csv             fig-data F10 (schema mirrors dca_data.csv)
  v3/derived/mtp_roc_data.pkl             fig-data F10 (schema mirrors roc_data.pkl)
  v3/logs/mtp_equivalent_manifest.json    17-namespaced run record (non-colliding)

Never touches table4/table5/table7 or any other script's artifact (CONTRACTS §11).

Run
---
    python 17_mtp_equivalent.py
    python 17_mtp_equivalent.py --sample 50000      # smoke

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
import platform
import sys
import time
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import statsmodels.api as sm

warnings.filterwarnings("ignore")

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

IN_SCORED = os.path.join(DERIVED, "cohort_test_scored.parquet")
IN_CANDIDATES = os.path.join(DERIVED, "candidates.parquet")
IN_TEST_KEYS = os.path.join(DERIVED, "test_inc_keys.csv")
IN_REFIT = os.path.join(DERIVED, "refit.json")
IN_POINTS = os.path.join(TABLES, "table3_points.csv")

OUT_TABLE = os.path.join(TABLES, "table8_mtp.csv")
OUT_DCA = os.path.join(DERIVED, "mtp_dca_data.csv")
OUT_ROC = os.path.join(DERIVED, "mtp_roc_data.pkl")
OUT_MANIFEST = os.path.join(LOGS, "mtp_equivalent_manifest.json")

# ---------------------------------------------------------------------------
# Reproducibility constants
# ---------------------------------------------------------------------------
SEED = 42
N_BOOT_DEFAULT = 2000
EPS = 1e-6
JOIN_KEYS = ["puf_year", "inc_key"]

MTP_COL = "mtp_equivalent"
TX_COL = "substantial_transfusion"          # the identical column (CONTRACTS §4)
PRIMARY_COL = "major_hemorrhage_event"      # context only — never a "confirmation"
PMID_FRAMING = "25440492"

# DCA grid — same window as 08 so F5 and F10 are read on the same axis.
DCA_THRESHOLDS = np.round(np.arange(0.02, 0.4001, 0.01), 4)

# (display label, column, sign) — higher(sign*value) == worse. Labels are the ones
# render_figures._dca_panel / figure3_roc already key on, so F10 can reuse them.
SCORE_SPECS: list[tuple[str, str, int]] = [
    ("pelvic_score", "pelvic_score", +1),
    ("pelvic_logit", "pelvic_lp", +1),
    ("SI-alone", "shock_index", +1),
    ("rSIG", "rsig", -1),
]
DCA_SCORES = ["pelvic_score", "SI-alone", "rSIG"]

# Fix-1 decoder bands, reused ONLY to describe the plasma/platelet scale ambiguity.
# No decoder is APPLIED to plasma_4h / platelet_4h (CONTRACTS §3.2) — these bands
# are descriptive evidence that a decoder would be needed, not a decoding.
UNITS_LIKE_MAX = 100
ML_LIKE_MIN = 200

# The wrong number, kept ONLY so the guard below can refuse to emit it.
FORBIDDEN_FILL_ARTIFACT = 27.3
FORBIDDEN_FILL_TOL = 0.5


def banner(msg: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# Metric helpers (self-contained; copy-adapted from 07/08 per house style)
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _safe_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


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
    """Vectorized DeLong (Sun-Xu fast form). Returns (z, p, delta, se, auc1, auc2);
    delta > 0 means p1 discriminates better."""
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
    if n_boot <= 0 or n == 0 or y.sum() in (0, n):
        return float("nan"), float("nan")
    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        aucs[b] = np.nan if yb.sum() in (0, len(yb)) else roc_auc_score(yb, s[idx])
    aucs = aucs[np.isfinite(aucs)]
    if aucs.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


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
    except Exception as exc:  # noqa: BLE001
        print(f"  [HL warn] {exc}")
        return float("nan"), float("nan"), float("nan")


def calibration_intercept_offset(y, lp):
    """CITL as the intercept of logit(P) = a + offset(LP), slope held at 1.
    a < 0 => the frozen model OVER-predicts on this outcome."""
    try:
        X = np.ones((len(y), 1), dtype=float)
        res = sm.GLM(np.asarray(y, dtype=float), X, family=sm.families.Binomial(),
                     offset=np.asarray(lp, dtype=float)).fit()
        return float(res.params[0])
    except Exception as exc:  # noqa: BLE001
        print(f"  [CITL warn] {exc}")
        return float("nan")


def calibration_slope(y, lp):
    try:
        X = sm.add_constant(np.asarray(lp, dtype=float), has_constant="add")
        res = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="lbfgs",
                                                        maxiter=200)
        return float(res.params[1])
    except Exception as exc:  # noqa: BLE001
        print(f"  [slope warn] {exc}")
        return float("nan")


def logistic_recalibrate(y, score):
    X = sm.add_constant(np.asarray(score, dtype=float), has_constant="add")
    m = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="lbfgs", maxiter=200)
    return np.clip(m.predict(X), EPS, 1 - EPS)


def net_benefit(y, prob, p_t):
    n = len(y)
    if n == 0 or p_t >= 1.0:
        return float("nan")
    pred_pos = prob >= p_t
    tp = float((pred_pos & (y == 1)).sum()) / n
    fp = float((pred_pos & (y == 0)).sum()) / n
    return tp - fp * (p_t / (1.0 - p_t))


def nb_treat_all(prev, p_t):
    return float("nan") if p_t >= 1.0 else prev - (1.0 - prev) * (p_t / (1.0 - p_t))


def roc_curve_thin(y, s, max_pts=1000):
    fpr, tpr, _ = roc_curve(y, s)
    if len(fpr) > max_pts:
        idx = np.linspace(0, len(fpr) - 1, max_pts).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return np.asarray(fpr), np.asarray(tpr)


# ---------------------------------------------------------------------------
# table8_mtp.csv row builder (sectioned long format)
# ---------------------------------------------------------------------------
T8_COLS = ["section", "metric", "group", "score", "n", "n_events", "event_rate_pct",
           "value", "ci_low", "ci_high", "p_value", "note"]


def t8(section, metric, note="", group="", score="", n=np.nan, n_events=np.nan,
       event_rate_pct=np.nan, value=np.nan, ci_low=np.nan, ci_high=np.nan,
       p_value=np.nan) -> dict[str, Any]:
    def _r(v, k=4):
        return round(float(v), k) if (v is not None and np.isfinite(np.float64(v))) else np.nan
    return {"section": section, "metric": metric, "group": group, "score": score,
            "n": n, "n_events": n_events, "event_rate_pct": _r(event_rate_pct, 4),
            "value": _r(value, 6), "ci_low": _r(ci_low, 6), "ci_high": _r(ci_high, 6),
            "p_value": (float(p_value) if p_value is not None
                        and np.isfinite(np.float64(p_value)) else np.nan),
            "note": note}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _column_disagreement(a: pd.Series, b: pd.Series) -> tuple[int, str]:
    """Element-wise disagreement between two copies of the same column.

    NaN==NaN counts as agreement (a column missing on both sides is not a merge
    failure). Numeric columns are compared with a float tolerance so an Int64 /
    float64 round-trip through parquet cannot masquerade as a bad join; anything
    non-numeric falls back to a string comparison. Returns (n_mismatch, method).
    """
    na, nb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    numeric = bool((na.notna() | a.isna()).all() and (nb.notna() | b.isna()).all())
    if numeric:
        fa, fb = na.astype(float), nb.astype(float)
        agree = ((fa.isna() & fb.isna()) |
                 (fa.notna() & fb.notna() & ((fa - fb).abs() <= 1e-9)))
        return int((~agree).sum()), "numeric compare, tol 1e-9, NaN==NaN"
    sa, sb = a.astype(str), b.astype(str)
    agree = (a.isna() & b.isna()) | (a.notna() & b.notna() & (sa == sb))
    return int((~agree).sum()), "string compare, NaN==NaN"


def load_inputs(sample_n: int) -> dict[str, Any]:
    """Load the scored TEST frame and attach the outcome / product columns from
    candidates.parquet. Nothing is recomputed: the cohort, the outcomes and the
    score all arrive as artifacts of 01/02/05/06."""
    for p in (IN_SCORED, IN_CANDIDATES):
        if not os.path.exists(p):
            raise SystemExit(f"required input missing: {p} (run 02 and 06 first).")

    scored = pd.read_parquet(IN_SCORED)
    cand = pd.read_parquet(IN_CANDIDATES)
    if "cohort_primary" not in cand.columns:
        raise SystemExit("candidates.parquet missing cohort_primary (CONTRACTS §3).")
    analytic = cand[cand["cohort_primary"] == 1].copy()

    need = [MTP_COL, TX_COL, "plasma_4h", "platelet_4h", "blood_4h"]
    absent = [c for c in need if c not in analytic.columns]
    if absent:
        raise SystemExit(
            f"candidates.parquet is missing {absent}. 02_engineer_features.py must carry "
            f"them (CONTRACTS §7 row 02 REQUIRED_CARRY_COLS); 17 never re-derives the "
            f"cohort.")

    for frame in (scored, analytic):
        for k in JOIN_KEYS:
            frame[k] = pd.to_numeric(frame[k], errors="coerce").astype("Int64")

    carry = [c for c in need + [PRIMARY_COL] if c in analytic.columns]
    right = analytic[JOIN_KEYS + carry].copy()

    # ---- Join integrity: merge PROVENANCE, not a NaN sniff ------------------
    # The obvious check — `merged[MTP_COL].isna().sum()` — CANNOT fail here and
    # therefore proves nothing. `suffixes=("", "_cand")` deliberately keeps the
    # LEFT (scored) frame's copy of every shared column, and 01 emits the
    # transfusion flags already filled (NaN blood_4h -> flag 0), so that column
    # is non-null whether the join matched every row or none of them. Four
    # checks that CAN fail replace it:
    #   (1) right-side keys unique      — a many-to-one join silently duplicates
    #                                     scored rows and inflates every count;
    #   (2) left-side keys unique       — 06 must emit one row per patient;
    #   (3) row-count invariant + merge INDICATOR — every scored row matched;
    #   (4) value cross-check on the columns that collided — catches a join that
    #       matched the WRONG rows (keys present, alignment wrong), the one
    #       failure mode (1)-(3) all pass through.
    dup_right = int(right.duplicated(subset=JOIN_KEYS).sum())
    if dup_right:
        raise SystemExit(
            f"candidates.parquet has {dup_right:,} duplicate {JOIN_KEYS} rows in the "
            f"analytic cohort — a many-to-one join would silently duplicate scored "
            f"TEST rows. Fix 01/02 before running 17.")
    dup_left = int(scored.duplicated(subset=JOIN_KEYS).sum())
    if dup_left:
        raise SystemExit(
            f"cohort_test_scored.parquet has {dup_left:,} duplicate {JOIN_KEYS} rows — "
            f"06 must emit exactly one scored row per patient.")

    n_scored = len(scored)
    merged = scored.merge(right, on=JOIN_KEYS, how="left", suffixes=("", "_cand"),
                          indicator="_merge_src")
    if len(merged) != n_scored:
        raise SystemExit(f"test-slice join drift: {len(merged)} != {n_scored} — a left "
                         f"join must preserve the left row count exactly.")
    n_unmatched = int((merged["_merge_src"].astype(str) != "both").sum())
    if n_unmatched:
        missing_keys = (merged.loc[merged["_merge_src"].astype(str) != "both", JOIN_KEYS]
                              .head(5).to_dict("records"))
        raise SystemExit(
            f"{n_unmatched:,} of {n_scored:,} scored TEST rows found NO match in the "
            f"analytic candidates frame — the test slice must be a subset of "
            f"cohort_primary==1. First unmatched keys: {missing_keys}")

    # (4) The collided columns must AGREE. 06's copy of an outcome and 02's copy
    # are the same column travelling by two routes; a disagreement means either
    # the join aligned the wrong rows or 06 recomputed something it should have
    # carried. Either way the MTP analysis would be built on the wrong denominator.
    collided = [c for c in carry if f"{c}_cand" in merged.columns]
    for c in collided:
        n_mismatch, how = _column_disagreement(merged[c], merged[f"{c}_cand"])
        if n_mismatch:
            raise SystemExit(
                f"join alignment FAILED: `{c}` differs on {n_mismatch:,} of "
                f"{len(merged):,} rows between cohort_test_scored.parquet and "
                f"candidates.parquet ({how}). The merge matched rows, but not the "
                f"right ones — or 06 recomputed a column it must carry unchanged.")
    merged = merged.drop(columns=["_merge_src"] +
                                 [f"{c}_cand" for c in collided])

    print(f"  analytic cohort (cohort_primary==1) : {len(analytic):,}")
    print(f"  scored TEST slice                   : {len(merged):,}")
    print(f"  join integrity                      : {n_scored:,}/{n_scored:,} matched "
          f"(indicator=='both'), right keys unique, "
          f"{len(collided)} shared column(s) cross-checked "
          f"{('[' + ', '.join(collided) + ']') if collided else ''}")

    for c in (MTP_COL, TX_COL):
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

    if sample_n and sample_n < len(merged):
        merged = merged.sample(n=int(sample_n), random_state=SEED).reset_index(drop=True)
        analytic = (analytic.sample(n=min(int(sample_n), len(analytic)), random_state=SEED)
                            .reset_index(drop=True))
        print(f"  [SMOKE] --sample {sample_n}: TEST -> {len(merged):,}, "
              f"analytic -> {len(analytic):,}. Fill rates below are SAMPLE estimates "
              f"and are NOT quotable.")
    join_integrity = {
        "keys": list(JOIN_KEYS),
        "n_scored_rows": int(n_scored),
        "n_matched_both": int(n_scored - n_unmatched),
        "n_unmatched": int(n_unmatched),
        "duplicate_keys_left": int(dup_left),
        "duplicate_keys_right": int(dup_right),
        "columns_cross_checked": list(collided),
        "method": ("pandas merge indicator ('both' required on every left row) + "
                   "row-count invariant + uniqueness of the join key on both "
                   "sides + element-wise value agreement on the shared columns. "
                   "A NaN test on a carried outcome column is NOT used: "
                   "suffixes=('', '_cand') keeps the left copy and 01 emits the "
                   "transfusion flags already filled, so such a test cannot fail."),
    }
    return {"test": merged, "analytic": analytic, "join_integrity": join_integrity}


# ---------------------------------------------------------------------------
# §13 G1e / G7c — the identity assertion, in code and in plain words
# ---------------------------------------------------------------------------
IDENTITY_SENTENCE = (
    "mtp_equivalent is DEFINITIONALLY IDENTICAL to substantial_transfusion: it is "
    "the transfusion component of the primary composite, analysed standalone under "
    "the critical-administration-threshold framing. It is NOT an independent "
    "endpoint and must NEVER be presented as independent corroboration of the "
    "primary result."
)


def assert_identity(test: pd.DataFrame, analytic: pd.DataFrame) -> dict[str, Any]:
    banner("Outcome identity assertion (CONTRACTS §4 warning; §13 G1e / G7c)")
    out: dict[str, Any] = {}
    for label, frame in (("analytic cohort", analytic), ("TEST slice", test)):
        a = pd.to_numeric(frame[MTP_COL], errors="coerce").fillna(0).astype(int).to_numpy()
        b = pd.to_numeric(frame[TX_COL], errors="coerce").fillna(0).astype(int).to_numpy()
        n_disc = int((a != b).sum())
        print(f"  {label:16s}: n={len(a):,}  mtp_equivalent={int(a.sum()):,}  "
              f"substantial_transfusion={int(b.sum()):,}  discordant={n_disc}")
        if n_disc != 0:
            raise SystemExit(
                f"CONTRACT VIOLATION (§13 G1e): mtp_equivalent and "
                f"substantial_transfusion differ on {n_disc} rows in the {label}. They "
                f"are the SAME rule by construction (CONTRACTS §3/§4); a difference "
                f"means 01_build_cohort.py emitted them from divergent code.")
        out[label] = {"n": int(len(a)), "mtp_equivalent": int(a.sum()),
                      "substantial_transfusion": int(b.sum()), "discordant": n_disc}
    print("\n  ASSERTION PASSED — 0 discordant rows at both denominators.")
    print("  " + IDENTITY_SENTENCE.replace(". ", ".\n  "))
    return out


# ---------------------------------------------------------------------------
# Balanced resuscitation — EXPLORATORY. Fill rate + scale ambiguity, stated openly.
# ---------------------------------------------------------------------------
def product_field_profile(analytic: pd.DataFrame, col: str) -> dict[str, Any]:
    """Fill rate and units/mL bimodality for one product field. COMPUTED, never
    hard-coded (CONTRACTS §13 G7c). No decoder is applied."""
    v = pd.to_numeric(analytic[col], errors="coerce")
    n = int(len(v))
    n_obs = int(v.notna().sum())
    fill = 100.0 * n_obs / n if n else float("nan")
    nz = v[(v.notna()) & (v > 0)]
    n_nz = int(len(nz))
    units_like = int(((nz > 0) & (nz <= UNITS_LIKE_MAX)).sum())
    ml_like = int((nz >= ML_LIKE_MIN).sum())
    gap = int(((nz > UNITS_LIKE_MAX) & (nz < ML_LIKE_MIN)).sum())
    prof = {
        "column": col, "n_denominator": n, "n_observed": n_obs,
        "fill_pct": round(fill, 3),
        "n_zero": int(((v.notna()) & (v == 0)).sum()), "n_nonzero": n_nz,
        "n_units_like_le100": units_like, "n_ml_like_ge200": ml_like,
        "n_gap_101_199": gap,
        "pct_units_like_of_nonzero": round(100.0 * units_like / n_nz, 2) if n_nz else float("nan"),
        "pct_ml_like_of_nonzero": round(100.0 * ml_like / n_nz, 2) if n_nz else float("nan"),
        "bimodal_units_vs_ml": bool(n_nz and units_like > 0 and ml_like > 0),
        "decoder_applied": False,
        "decoder_column_available": False,   # TQIP_HC_*_UNIT are 0.0000 % filled
    }
    # G7c guard: the emitted fill must be the computed one, and must not be the
    # 27.3 % year-ordering artifact the plan quoted.
    if np.isfinite(fill) and abs(fill - FORBIDDEN_FILL_ARTIFACT) < FORBIDDEN_FILL_TOL:
        raise SystemExit(
            f"REFUSING TO EMIT: computed {col} fill = {fill:.3f} %, which matches the "
            f"known-bad {FORBIDDEN_FILL_ARTIFACT} % sampling artifact (CONTRACTS §3.2). "
            f"That figure came from a year-ordered head sample (~73 % AY2019, where the "
            f"field was not collected). Either the frame handed to this function is a "
            f"head-slice rather than the analytic cohort, or 02 carried the wrong column.")
    return prof


def fill_by_year(analytic: pd.DataFrame, col: str) -> dict[str, Any]:
    if "puf_year" not in analytic.columns:
        return {}
    v = pd.to_numeric(analytic[col], errors="coerce")
    yr = pd.to_numeric(analytic["puf_year"], errors="coerce")
    out = {}
    for y in sorted(yr.dropna().unique().tolist()):
        m = yr == y
        out[str(int(y))] = {"n": int(m.sum()),
                            "fill_pct": round(100.0 * float(v[m].notna().mean()), 3)}
    return out


def balanced_resuscitation(test: pd.DataFrame, analytic: pd.DataFrame, n_boot: int,
                           rng, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """EXPLORATORY subgroup. Reports receipt FLAGS only — `plasma_4h > 0` and
    `platelet_4h > 0` are scale-invariant (a strictly positive value means the
    product was given under either the units or the mL reading). **No ratio is
    computed**, because a plasma:PRBC or platelet:PRBC ratio would have an
    undetermined scale in BOTH numerator and denominator (CONTRACTS §3.2)."""
    banner("EXPLORATORY balanced-resuscitation subgroup "
           "(receipt flags only — NO ratio is computed)")

    prof_pl = product_field_profile(analytic, "plasma_4h")
    prof_pt = product_field_profile(analytic, "platelet_4h")
    for p in (prof_pl, prof_pt):
        print(f"  {p['column']:12s} fill={p['fill_pct']:.3f} %  "
              f"(n_obs {p['n_observed']:,}/{p['n_denominator']:,})  nonzero={p['n_nonzero']:,}  "
              f"units-like<=100 {p['n_units_like_le100']:,} ({p['pct_units_like_of_nonzero']} %)  "
              f"mL-like>=200 {p['n_ml_like_ge200']:,} ({p['pct_ml_like_of_nonzero']} %)  "
              f"gap101-199 {p['n_gap_101_199']:,}")
    print("  -> BOTH fields are units/mL bimodal with NO decoder "
          "(TQIP_HC_PLASMA_UNIT / TQIP_HC_PLATE_UNIT are 0.0000 % filled).")
    print("  -> Therefore: receipt flags only. No ratio. No 1:1:1 claim.")

    SEC = "balanced_resuscitation_exploratory"
    for p in (prof_pl, prof_pt):
        rows.append(t8(SEC, "fill_rate_pct", score=p["column"], group="analytic cohort",
                       n=p["n_denominator"], value=p["fill_pct"],
                       note=("COMPUTED from the analytic cohort at run time, not hard-coded. "
                             "Quote this figure (~80.68 %); the plan's 27.3 % is a "
                             "year-ordering sampling artifact and is wrong (CONTRACTS §3.2).")))
        rows.append(t8(SEC, "units_like_pct_of_nonzero", score=p["column"],
                       group="non-zero values, <=100", n=p["n_nonzero"],
                       value=p["pct_units_like_of_nonzero"],
                       note="units-like band"))
        rows.append(t8(SEC, "ml_like_pct_of_nonzero", score=p["column"],
                       group="non-zero values, >=200", n=p["n_nonzero"],
                       value=p["pct_ml_like_of_nonzero"],
                       note="mL-like band"))
        rows.append(t8(SEC, "gap_101_199_n", score=p["column"], group="non-zero values",
                       n=p["n_nonzero"], value=p["n_gap_101_199"],
                       note="empirical gap between the two modes"))
        rows.append(t8(SEC, "scale_decoder_available", score=p["column"], value=0,
                       note=("NO decoder exists (TQIP_HC_PLASMA_UNIT / TQIP_HC_PLATE_UNIT "
                             "are 0.0000 % filled), so the scale of every value is "
                             "undetermined. THIS — not the fill rate — is why the subgroup "
                             "is labelled EXPLORATORY.")))

    # ---- subgroup construction: BOTH fields observed ----
    pl = pd.to_numeric(test["plasma_4h"], errors="coerce")
    pt = pd.to_numeric(test["platelet_4h"], errors="coerce")
    both_obs = pl.notna() & pt.notna()
    sub = test.loc[both_obs].copy()
    n_sub = int(len(sub))
    print(f"\n  TEST rows with BOTH product fields observed: {n_sub:,} / {len(test):,} "
          f"({100.0*n_sub/len(test):.2f} %)")
    rows.append(t8(SEC, "subgroup_n_both_fields_observed", group="TEST slice",
                   n=n_sub, value=100.0 * n_sub / len(test) if len(test) else np.nan,
                   note=("value = % of the TEST slice. Restricting to observed rows "
                         "preferentially removes admission year 2019, when the fields were "
                         "essentially not collected (~3.5 % filled) — a selection effect "
                         "that must be stated alongside any subgroup estimate.")))

    detail: dict[str, Any] = {"plasma_4h": prof_pl, "platelet_4h": prof_pt,
                              "plasma_fill_by_year": fill_by_year(analytic, "plasma_4h"),
                              "platelet_fill_by_year": fill_by_year(analytic, "platelet_4h"),
                              "n_test": int(len(test)), "n_both_observed": n_sub,
                              "ratio_computed": False,
                              "reason_exploratory": ("units/mL scale ambiguity in BOTH "
                                                     "numerator and denominator; not fill rate")}
    if n_sub == 0:
        rows.append(t8(SEC, "subgroup_skipped", note="no TEST row has both fields observed"))
        return detail

    plg = (pd.to_numeric(sub["plasma_4h"], errors="coerce") > 0).astype(int)
    ptg = (pd.to_numeric(sub["platelet_4h"], errors="coerce") > 0).astype(int)
    balanced = ((plg == 1) & (ptg == 1)).astype(int)
    y_mtp = sub[MTP_COL].astype(int).to_numpy()

    for nm, flag in (("plasma_given", plg), ("platelet_given", ptg),
                     ("balanced_both_products_given", balanced)):
        n_ev = int(flag.sum())
        rows.append(t8(SEC, f"{nm}_rate_pct", group="both fields observed", n=n_sub,
                       n_events=n_ev, value=100.0 * n_ev / n_sub,
                       note="receipt flag (>0) — SCALE-INVARIANT under either unit reading"))
        m = y_mtp == 1
        n_m = int(m.sum())
        if n_m:
            rows.append(t8(SEC, f"{nm}_rate_pct", group="MTP-equivalent positive", n=n_m,
                           n_events=int(flag.to_numpy()[m].sum()),
                           value=100.0 * float(flag.to_numpy()[m].mean()),
                           note="receipt flag among patients meeting the critical-"
                                "administration threshold"))
        print(f"  {nm:30s} overall {100.0*n_ev/n_sub:5.2f} %"
              + (f"   among MTP-equivalent {100.0*float(flag.to_numpy()[m].mean()):5.2f} %"
                 if n_m else ""))

    # ---- exploratory discrimination inside the subgroup ----
    score = pd.to_numeric(sub["pelvic_score"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(score)
    if ok.sum() and 0 < int(y_mtp[ok].sum()) < int(ok.sum()):
        a = float(roc_auc_score(y_mtp[ok], score[ok]))
        lo, hi = bootstrap_auc_ci(y_mtp[ok], score[ok], n_boot, rng)
        rows.append(t8(SEC, "AUROC", score="pelvic_score", group="MTP-equivalent | both fields observed",
                       n=int(ok.sum()), n_events=int(y_mtp[ok].sum()),
                       event_rate_pct=100.0 * float(y_mtp[ok].mean()),
                       value=a, ci_low=lo, ci_high=hi,
                       note="EXPLORATORY — subgroup is not year-representative (see selection note)"))
        detail["auroc_mtp_in_subgroup"] = a
        print(f"  AUROC pelvic_score -> mtp_equivalent, within subgroup: {a:.4f} [{lo:.4f},{hi:.4f}]")

    m = y_mtp == 1
    if int(m.sum()) > 0:
        yb = balanced.to_numpy()[m]
        sb = score[m]
        okb = np.isfinite(sb)
        if okb.sum() and 0 < int(yb[okb].sum()) < int(okb.sum()):
            a = float(roc_auc_score(yb[okb], sb[okb]))
            lo, hi = bootstrap_auc_ci(yb[okb], sb[okb], n_boot, rng)
            rows.append(t8(SEC, "AUROC", score="pelvic_score",
                           group="balanced resuscitation | MTP-equivalent positive",
                           n=int(okb.sum()), n_events=int(yb[okb].sum()),
                           event_rate_pct=100.0 * float(yb[okb].mean()),
                           value=a, ci_low=lo, ci_high=hi,
                           note=("EXPLORATORY. Endpoint is the RECEIPT of both plasma and "
                                 "platelets (>0), which is scale-invariant. This is NOT a "
                                 "1:1:1 ratio analysis and must not be described as one.")))
            detail["auroc_balanced_among_mtp"] = a
            print(f"  AUROC pelvic_score -> balanced receipt, among MTP-equivalent: "
                  f"{a:.4f} [{lo:.4f},{hi:.4f}]")

    rows.append(t8(SEC, "ratio_computed", value=0,
                   note=("ZERO ratios are computed anywhere in this analysis. A plasma:PRBC "
                         "or platelet:PRBC ratio would have an undetermined scale in BOTH "
                         "numerator and denominator (both fields are units/mL bimodal with "
                         "no decoder), so no ratio may be read as a literal 1:1:1 comparison "
                         "unless the scales are first shown to agree. CONTRACTS §3.2 / §13 G7c.")))
    return detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="v3 script 17 — MTP-equivalent secondary outcome (Fix 7 / D-5).")
    ap.add_argument("--sample", type=int, default=0,
                    help="Row-downsample the TEST and analytic frames to N (seed 42) smoke test.")
    ap.add_argument("--bootstraps", type=int, default=N_BOOT_DEFAULT,
                    help=f"AUROC bootstrap iterations (default {N_BOOT_DEFAULT}).")
    args = ap.parse_args(argv)

    t0 = time.time()
    for d in (DERIVED, TABLES, LOGS):
        os.makedirs(d, exist_ok=True)
    n_boot = min(int(args.bootstraps), 200) if args.sample else int(args.bootstraps)

    banner("Step 17 — MTP-equivalent secondary outcome (Fix 7 / F1 / PMID "
           f"{PMID_FRAMING})")
    print(f"  seed={SEED}  bootstraps={n_boot}  sample={args.sample or 'None'}")

    data = load_inputs(args.sample)
    test, analytic = data["test"], data["analytic"]
    rng = np.random.default_rng(SEED)

    # ---------------- Identity (the thing that must not be missed) ----------------
    identity = assert_identity(test, analytic)

    rows: list[dict[str, Any]] = []
    rows.append(t8("header_note", "outcome_identity", note=IDENTITY_SENTENCE))
    rows.append(t8("header_note", "outcome_definition",
                   note=(f"{MTP_COL} = >=4 U OR >=1000 mL PRBC within 4 h "
                         f"(critical-administration threshold, Fix-1 unit-aware decoder, "
                         f"CONTRACTS §3). NTDB carries no MTP-activation flag; this is the "
                         f"closest available equivalent and that limitation is stated in "
                         f"the manuscript.")))
    rows.append(t8("header_note", "framing_reference",
                   note=(f"PMID {PMID_FRAMING} — the source of the MTP-in-pelvic-fracture "
                         f"framing (review comment F1); cited in Introduction and "
                         f"Discussion per SAP amendment D-5.")))
    rows.append(t8("header_note", "interpretation_guard",
                   note=("Discrimination against this outcome is EXPECTED to be at least as "
                         "good as against the primary composite, because transfusion is the "
                         "most physiologically proximate component. That is a restatement of "
                         "the identity above, NOT independent evidence.")))
    rows.append(t8("header_note", "sample_mode", value=(args.sample or 0),
                   note=("0 = full run. Non-zero means every N, rate and fill figure in this "
                         "table is a SMOKE-TEST estimate and is not quotable.")))
    for label, d in identity.items():
        rows.append(t8("outcome_identity", "discordant_rows", group=label, n=d["n"],
                       n_events=d[MTP_COL], value=d["discordant"],
                       note=(f"{MTP_COL}={d[MTP_COL]:,} vs {TX_COL}="
                             f"{d[TX_COL]:,}; element-wise discordance must be 0 (§13 G1e)")))

    # ---------------- Discrimination ----------------
    banner("Discrimination against mtp_equivalent (held-out TEST slice)")
    y = test[MTP_COL].astype(int).to_numpy()
    n_all, n_ev = int(len(y)), int(y.sum())
    prev = float(y.mean()) if n_all else float("nan")
    print(f"  TEST N={n_all:,}  MTP-equivalent events={n_ev:,} ({100*prev:.3f} %)")
    if PRIMARY_COL in test.columns:
        yp = pd.to_numeric(test[PRIMARY_COL], errors="coerce").fillna(0).astype(int)
        print(f"  (context: primary composite events={int(yp.sum()):,} "
              f"({100*float(yp.mean()):.3f} %) — the MTP-equivalent set is a SUBSET of it, "
              f"not a second observation of it)")

    oriented: dict[str, dict[str, Any]] = {}
    roc_pkl: dict[str, Any] = {}
    for label, col, sign in SCORE_SPECS:
        if col not in test.columns:
            print(f"  [warn] '{col}' absent from cohort_test_scored.parquet — {label} skipped.")
            continue
        s_raw = pd.to_numeric(test[col], errors="coerce").to_numpy(dtype=float)
        use = np.isfinite(s_raw)
        s = sign * s_raw[use]
        ys = y[use]
        if ys.sum() in (0, len(ys)) or len(ys) == 0:
            print(f"  [warn] {label}: degenerate outcome on the usable rows — skipped.")
            continue
        auc = float(roc_auc_score(ys, s))
        lo, hi = bootstrap_auc_ci(ys, s, n_boot, rng)
        oriented[label] = {"y": ys, "s": s, "n": int(use.sum()), "sign": sign, "col": col}
        fpr, tpr = roc_curve_thin(ys, s)
        roc_pkl[label] = {"fpr": fpr, "tpr": tpr, "auc": auc, "lo": lo, "hi": hi,
                          "n": int(use.sum()), "events": int(ys.sum()),
                          "outcome": MTP_COL, "orientation_sign": sign}
        rows.append(t8("discrimination", "AUROC", score=label, group="TEST slice",
                       n=int(use.sum()), n_events=int(ys.sum()),
                       event_rate_pct=100.0 * float(ys.mean()),
                       value=auc, ci_low=lo, ci_high=hi,
                       note=(f"column '{col}', orientation sign {sign:+d} (higher = worse); "
                             f"{n_boot}x percentile bootstrap CI")))
        print(f"  {label:<14} N={use.sum():>8,} ev={int(ys.sum()):>7,} "
              f"AUROC={auc:.4f} [{lo:.4f}, {hi:.4f}]")

    # ---------------- DeLong vs SI-alone and rSIG ----------------
    banner("DeLong: pelvic_score vs SI-alone and vs rSIG (paired, complete rows)")
    delong_out: dict[str, Any] = {}
    if "pelvic_score" in oriented:
        for comp in ("SI-alone", "rSIG"):
            if comp not in oriented:
                print(f"  [warn] {comp} unavailable — DeLong skipped.")
                continue
            c1, c2 = oriented["pelvic_score"]["col"], oriented[comp]["col"]
            sub = test[[c1, c2]].apply(pd.to_numeric, errors="coerce")
            m = sub.notna().all(axis=1).to_numpy()
            yy = y[m]
            if yy.sum() in (0, len(yy)) or len(yy) == 0:
                print(f"  [warn] {comp}: degenerate paired subset — DeLong skipped.")
                continue
            p1 = oriented["pelvic_score"]["sign"] * sub[c1].to_numpy(float)[m]
            p2 = oriented[comp]["sign"] * sub[c2].to_numpy(float)[m]
            z, p, delta, se, a1, a2 = delong_test(yy, p1, p2)
            delong_out[comp] = {"n_paired": int(m.sum()), "events": int(yy.sum()),
                                "auroc_pelvic": a1, "auroc_comparator": a2,
                                "delta": delta, "se": se, "z": z, "p": p}
            rows.append(t8("delong", "delta_AUROC", score=f"pelvic_score vs {comp}",
                           group="paired complete rows", n=int(m.sum()), n_events=int(yy.sum()),
                           value=delta, ci_low=delta - 1.959964 * se,
                           ci_high=delta + 1.959964 * se, p_value=p,
                           note=(f"DeLong (Sun-Xu). pelvic_score AUROC {a1:.4f} vs {comp} "
                                 f"{a2:.4f}; z={z:.3f}. delta>0 favours the score. Paired on "
                                 f"rows where BOTH scores are observable.")))
            print(f"  pelvic_score vs {comp:<10} N={m.sum():>8,} "
                  f"delta={delta:+.4f} (SE {se:.4f}) z={z:+.3f} p={p:.3g}")
    else:
        print("  [warn] pelvic_score unavailable — no DeLong comparisons produced.")

    # ---------------- Calibration ----------------
    banner("Calibration against mtp_equivalent (CITL / slope / Brier)")
    calib: dict[str, Any] = {}
    if "pelvic_lp" in test.columns:
        lp = pd.to_numeric(test["pelvic_lp"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(lp)
        yv, lpv = y[ok], lp[ok]
        prob = _sigmoid(lpv)
        citl = calibration_intercept_offset(yv, lpv)
        citl_md = float(prob.mean() - yv.mean())
        slope = calibration_slope(yv, lpv)
        brier = float(brier_score_loss(yv, prob))
        hl, hl_df, hl_p = hosmer_lemeshow(yv, prob)
        calib["frozen"] = {"n": int(ok.sum()), "citl_offset": citl, "citl_meandiff": citl_md,
                           "slope": slope, "brier": brier, "hl_p": hl_p}
        note = ("FROZEN model as fitted to the PRIMARY composite, applied unchanged to the "
                "MTP-equivalent outcome. Miscalibration here is EXPECTED and is the honest "
                "result: the frozen model targets a higher-prevalence outcome, so it "
                "over-predicts this one. CITL<0 / mean-diff>0 = over-prediction.")
        for metric, val in (("CITL_offset_intercept", citl), ("CITL_mean_difference", citl_md),
                            ("calibration_slope", slope), ("Brier", brier),
                            ("Hosmer_Lemeshow_p", hl_p)):
            rows.append(t8("calibration", metric, score="pelvic_lp (frozen)",
                           group="TEST slice", n=int(ok.sum()), n_events=int(yv.sum()),
                           event_rate_pct=100.0 * float(yv.mean()), value=val, note=note))
        print(f"  frozen  CITL(offset)={citl:+.4f}  CITL(mean-diff)={citl_md:+.5f}  "
              f"slope={slope:.4f}  Brier={brier:.5f}  HL p={hl_p:.3g}")
    else:
        print("  [warn] 'pelvic_lp' absent — frozen-model calibration skipped.")

    if "pelvic_score" in oriented:
        d = oriented["pelvic_score"]
        try:
            prob_r = logistic_recalibrate(d["y"], d["s"])
            citl_r = calibration_intercept_offset(d["y"], _safe_logit(prob_r))
            citl_r_md = float(prob_r.mean() - d["y"].mean())
            slope_r = calibration_slope(d["y"], _safe_logit(prob_r))
            brier_r = float(brier_score_loss(d["y"], prob_r))
            hl_r, _, hl_pr = hosmer_lemeshow(d["y"], prob_r)
            calib["recalibrated"] = {"n": d["n"], "citl_offset": citl_r,
                                     "citl_meandiff": citl_r_md, "slope": slope_r,
                                     "brier": brier_r, "hl_p": hl_pr}
            note_r = ("Integer card RECALIBRATED to this outcome (intercept and slope "
                      "re-estimated on the same TEST rows). CITL~0 and slope~1 are "
                      "guaranteed by construction and are therefore NOT evidence of "
                      "calibration — the informative quantity here is the Brier score.")
            for metric, val in (("CITL_offset_intercept", citl_r),
                                ("CITL_mean_difference", citl_r_md),
                                ("calibration_slope", slope_r), ("Brier", brier_r),
                                ("Hosmer_Lemeshow_p", hl_pr)):
                rows.append(t8("calibration", metric, score="pelvic_score (recalibrated)",
                               group="TEST slice", n=d["n"], n_events=int(d["y"].sum()),
                               event_rate_pct=100.0 * float(d["y"].mean()),
                               value=val, note=note_r))
            print(f"  recal   CITL(offset)={citl_r:+.4f}  slope={slope_r:.4f}  "
                  f"Brier={brier_r:.5f}  HL p={hl_pr:.3g}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] recalibration failed: {exc}")

    # ---------------- Decision curve ----------------
    banner(f"Decision curve  p_t {DCA_THRESHOLDS.min():.2f}-{DCA_THRESHOLDS.max():.2f}  "
           f"(prevalence anchor computed from this outcome)")
    print(f"  DCA prevalence anchor = {prev:.5f} (observed MTP-equivalent rate in the TEST "
          f"slice; NOT the composite's anchor and NOT a v2 constant)")
    dca_probs: dict[str, np.ndarray] = {}
    for label in DCA_SCORES:
        if label in oriented:
            d = oriented[label]
            try:
                dca_probs[label] = logistic_recalibrate(d["y"], d["s"])
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] DCA recalibration failed for {label}: {exc}")
    dca_rows = []
    for p_t in DCA_THRESHOLDS:
        ta = nb_treat_all(prev, float(p_t))
        for label in DCA_SCORES:
            if label not in dca_probs:
                continue
            d = oriented[label]
            dca_rows.append({"threshold": float(p_t), "score": label,
                             "net_benefit": net_benefit(d["y"], dca_probs[label], float(p_t)),
                             "n_used": d["n"], "treat_all_nb": ta, "treat_none_nb": 0.0,
                             "outcome": MTP_COL, "prevalence_anchor": prev})
    dca_df = pd.DataFrame(dca_rows)
    dca_df.to_csv(OUT_DCA, index=False)
    print(f"  wrote {OUT_DCA} ({len(dca_df)} rows)")

    nb_summary: dict[str, Any] = {}
    if len(dca_df):
        piv = dca_df.pivot_table(index="threshold", columns="score", values="net_benefit")
        win = piv[(piv.index >= 0.05) & (piv.index <= 0.20)]
        for label in DCA_SCORES:
            if label in piv.columns:
                for pt in (0.05, 0.10, 0.20):
                    if pt in piv.index:
                        rows.append(t8("decision_curve", "net_benefit", score=label,
                                       group=f"p_t={pt:.2f}", n=oriented[label]["n"],
                                       value=float(piv.loc[pt, label]),
                                       note="full curve in derived/mtp_dca_data.csv (fig-data F10)"))
        if {"pelvic_score", "SI-alone"}.issubset(win.columns) and len(win):
            gap = win["pelvic_score"] - win["SI-alone"]
            nb_summary = {"gap_min": float(gap.min()), "gap_mean": float(gap.mean()),
                          "all_positive": bool((gap > 0).all())}
            rows.append(t8("decision_curve", "net_benefit_gap_vs_SI_min",
                           score="pelvic_score - SI-alone", group="p_t 0.05-0.20",
                           value=nb_summary["gap_min"],
                           note="minimum net-benefit advantage over shock-index-alone "
                                "across the pre-specified window"))
            rows.append(t8("decision_curve", "net_benefit_gap_vs_SI_mean",
                           score="pelvic_score - SI-alone", group="p_t 0.05-0.20",
                           value=nb_summary["gap_mean"],
                           note=f"all thresholds positive: {nb_summary['all_positive']}"))
            print(f"  NB gap pelvic-SI over p_t 0.05-0.20: min={nb_summary['gap_min']:+.5f} "
                  f"mean={nb_summary['gap_mean']:+.5f} "
                  f"(all positive: {nb_summary['all_positive']})")
    rows.append(t8("decision_curve", "prevalence_anchor", group="TEST slice", n=n_all,
                   n_events=n_ev, value=prev,
                   note="computed from the observed MTP-equivalent rate, not hard-coded"))

    # ---------------- Override group (Fix 4) — descriptive only ----------------
    if "override_nslo" in test.columns:
        ov = pd.to_numeric(test["override_nslo"], errors="coerce").fillna(0).astype(int)
        n_ov = int(ov.sum())
        if n_ov:
            r_ov = 100.0 * float(y[ov.to_numpy() == 1].mean())
            rows.append(t8("override_group", "MTP_equivalent_rate_pct",
                           group="no signs of life on arrival (unrecordable systolic pressure)",
                           n=n_ov, n_events=int(y[ov.to_numpy() == 1].sum()), value=r_ov,
                           note=("Descriptive only. These patients are scored on the point "
                                 "scale like everyone else in the AUROC above; the automatic-"
                                 "escalate override is a DEPLOYED decision rule (CONTRACTS "
                                 "§5.1), not a score term.")))
            print(f"\n  override group (no signs of life on arrival): n={n_ov:,} in TEST, "
                  f"MTP-equivalent rate {r_ov:.2f} %")
    else:
        print("\n  [note] 'override_nslo' not present in cohort_test_scored.parquet — "
              "override descriptives skipped (06 emits it per CONTRACTS §5.1).")

    # ---------------- Balanced resuscitation (EXPLORATORY) ----------------
    br = balanced_resuscitation(test, analytic, n_boot, rng, rows)

    # ---------------- Write table8_mtp.csv ----------------
    banner("Writing table8_mtp.csv")
    t8_df = pd.DataFrame(rows, columns=T8_COLS)
    t8_df.to_csv(OUT_TABLE, index=False)
    print(f"  wrote {OUT_TABLE} ({len(t8_df)} rows, "
          f"{t8_df['section'].nunique()} sections: "
          f"{t8_df['section'].drop_duplicates().tolist()})")

    # G7c self-check: the identity statement and the computed fill rate must both be
    # present in the emitted table, and 27.3 must not appear anywhere in it.
    has_identity = t8_df["note"].astype(str).str.contains(
        "DEFINITIONALLY IDENTICAL", case=False, na=False).any()
    fill_rows = t8_df[(t8_df["section"] == "balanced_resuscitation_exploratory")
                      & (t8_df["metric"] == "fill_rate_pct")]
    assert has_identity, "table8_mtp.csv lost the identity statement (§13 G7c)."
    assert len(fill_rows) >= 2, "table8_mtp.csv is missing the plasma/platelet fill rows."
    bad = [float(v) for v in fill_rows["value"].tolist()
           if np.isfinite(v) and abs(float(v) - FORBIDDEN_FILL_ARTIFACT) < FORBIDDEN_FILL_TOL]
    assert not bad, f"table8_mtp.csv emitted the forbidden 27.3 % fill artifact: {bad}"
    print(f"  G7c self-check PASS: identity statement present; fill rates "
          f"{[float(v) for v in fill_rows['value'].tolist()]} (computed, not 27.3).")

    # ---------------- Write mtp_roc_data.pkl ----------------
    banner("Writing mtp_roc_data.pkl (fig-data F10)")
    roc_pkl["_meta"] = {
        "outcome": MTP_COL, "outcome_definition": ">=4 U OR >=1000 mL PRBC within 4 h",
        "identity_note": IDENTITY_SENTENCE,
        "identical_to": TX_COL,
        "framing_pmid": PMID_FRAMING,
        "n_test": n_all, "events": n_ev, "prevalence": prev,
        "seed": SEED, "n_boot": n_boot, "sample_mode": int(args.sample or 0),
        "score_keys": [k for k in roc_pkl if k != "_meta"],
        "delong": delong_out,
    }
    with open(OUT_ROC, "wb") as f:
        pickle.dump(roc_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {OUT_ROC}  keys={[k for k in roc_pkl if k != '_meta']}")

    # ---------------- Manifest ----------------
    points_provenance: dict[str, Any] = {}
    try:
        with open(IN_REFIT) as f:
            refit = json.load(f)
        points_provenance = {
            "variables_kept": refit.get("variables_kept"),
            "point_vars": refit.get("point_vars"),
            "override_vars": refit.get("override_vars"),
            "n_points_rows": (int(len(pd.read_csv(IN_POINTS)))
                              if os.path.exists(IN_POINTS) else None),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  [note] refit.json/table3 provenance unavailable: {exc}")

    manifest = {
        "script": "17_mtp_equivalent.py",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED, "n_boot": n_boot, "sample": int(args.sample or 0),
        "outcome": MTP_COL,
        "outcome_identity": {"identical_to": TX_COL, "statement": IDENTITY_SENTENCE,
                             "assertion": identity},
        "join_integrity": data.get("join_integrity"),
        "framing_pmid": PMID_FRAMING,
        "n_test": n_all, "events": n_ev, "prevalence": prev,
        "discrimination": {k: {"auc": v["auc"], "lo": v["lo"], "hi": v["hi"],
                               "n": v["n"], "events": v["events"]}
                           for k, v in roc_pkl.items() if k != "_meta"},
        "delong": delong_out,
        "calibration": calib,
        "decision_curve": {"prevalence_anchor": prev,
                           "grid": [float(DCA_THRESHOLDS.min()), float(DCA_THRESHOLDS.max())],
                           "summary": nb_summary},
        "balanced_resuscitation_exploratory": br,
        "score_provenance": points_provenance,
        "outputs": {"table8_mtp": OUT_TABLE, "mtp_dca_data": OUT_DCA,
                    "mtp_roc_data": OUT_ROC},
        "packages": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "statsmodels": sm.__version__ if hasattr(sm, "__version__") else None},
        "runtime_sec": round(time.time() - t0, 2),
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  wrote {OUT_MANIFEST}")

    # ---------------- Closing statement (so it lands in the run log too) ----------------
    banner("STEP 17 SUMMARY — read the identity note before quoting anything")
    print(f"  {IDENTITY_SENTENCE}")
    print(f"\n  Balanced-resuscitation subgroup: EXPLORATORY. plasma_4h fill "
          f"{br['plasma_4h']['fill_pct']:.3f} %, platelet_4h fill "
          f"{br['platelet_4h']['fill_pct']:.3f} % in the analytic cohort "
          f"(the plan's 27.3 % is a year-ordering artifact and is wrong). It is "
          f"EXPLORATORY because both fields are units/mL bimodal with no decoder, so any "
          f"ratio would have an undetermined scale in both numerator and denominator — "
          f"which is why NO ratio is computed here.")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s\nStep 17 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
