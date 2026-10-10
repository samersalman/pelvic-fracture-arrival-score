"""
Pelvic Arrival Tool v3 — Step 04: LASSO stability selection (lambda.1se)
================================================================================

Forked from PAT_TBI/v4/code/04_lasso_stability_selection.py. Mechanism is
preserved verbatim: per-bootstrap lambda.1se LASSO stability selection with a
0.85 selection-frequency threshold and NO automatic fallback.

Adaptations for the pelvic v3 pipeline (see v3/CONTRACTS.md §7, §5, §3):
  * Root -> v3; outputs -> v3/derived + v3/tables.
  * LASSO_X_COLS = the 17 arrival/bedside-extractable predictors (CONTRACTS §5;
    kept VERBATIM in sync with 02_engineer_features.py; continuous shock_index
    dropped from the candidate pool per §5 CORRECTION).
  * **v3 Fix 3 / D-2:** `unstable_ring` removed from the pool (18 -> 17). Ring
    stability is a radiographic read, not information available immediately upon
    arrival. It remains in the DATA and in `11.CORE_ADJUSTMENT` / `13`'s VIF
    audit (CONTRACTS §6 asymmetry) — the removal is not propagated there.
  * **v3 Fix 4 / D-6 + W2:** `sbp_unrecordable` ("no signs of life on arrival
    (unrecordable systolic pressure)") is treated here as an ordinary candidate,
    with NO special casing — it participates in λ.1se and in stability selection
    exactly like every other variable. Fix 4 acts only in `05`, which keeps it in
    the unpenalized refit but takes it off the integer point scale and makes it a
    binary automatic-escalate override (CONTRACTS §5.1).
  * Outcome = major_hemorrhage_event (primary composite — same NAME as v2, new
    MEANING: its death arm is now `death_hemorrhagic`, CONTRACTS §4).
  * Join keys are lowercase ("puf_year", "inc_key") in this dataset.
  * ANALYTIC-COHORT RULE (CONTRACTS §3): filter cohort_primary == 1 (ring-only,
    N = 239,736) BEFORE the train inner-join.
  * Full-run bootstrap subsample defaults to the full training-set size (drawn
    with replacement — a standard n-out-of-n bootstrap). Smoke mode overrides
    with --sample / --bootstraps / --subsample.

`--sample` vs `--subsample` — they are NOT interchangeable
----------------------------------------------------------
  --sample N     REAL ROW-DOWNSAMPLE of the INPUT (CONTRACTS §9-E): after the
                 train inner-join, `df.sample(n=N, random_state=42)`. Shrinks
                 the training set itself, so every downstream cost (the global
                 LogisticRegressionCV *and* all 100 bootstraps) falls with it.
                 This is the flag §13's "smoke-test every script with
                 --sample 50000 before the full LASSO" means, and 04 is the long
                 pole (~19 min) that the smoke test exists to protect.
  --subsample N  size of each BOOTSTRAP RESAMPLE drawn (with replacement) FROM
                 the training set. It does not reduce the input, does not touch
                 the global CV fit, and at the default equals the full train N.
                 It is a property of the stability procedure, not a smoke knob.

⚠️ Any of the three flags marks the run as SMOKE. Smoke artifacts
(`lasso_cv_fit.pkl`, `table2_lasso_selected.csv`) overwrite the real ones in
`derived/` and `tables/`, and `05` consumes them without knowing. Re-run 04
clean before `05`.

Outputs
-------
  v3/derived/lasso_cv_fit.pkl
      {best_C, scaler_mean, scaler_scale, feature_names, cv_beta}
  v3/tables/table2_lasso_stability.csv
      Variable, Selection_Frequency, CV_Beta, CV_Abs_Beta_Rank
  v3/tables/table2_lasso_selected.csv
      Variable, Selection_Frequency, CV_Beta   (rows with Selection_Frequency>=0.85)

Run
---
    python 04_lasso_stability.py                                    # full run
    python 04_lasso_stability.py --sample 50000                     # §13 smoke
    python 04_lasso_stability.py --sample 50000 --bootstraps 5      # fastest smoke
    python 04_lasso_stability.py --bootstraps 5 --subsample 20000   # smoke

Seed: 42
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# sklearn 1.8 emits transitional FutureWarnings for penalty='l1'/n_jobs on
# LogisticRegressionCV. The explicit LogisticRegression(penalty='l1', saga)
# refits still produce genuinely L1-sparse solutions (verified: nonzero < p),
# so selection is real L1. Silence the transitional noise for readable logs.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Canonical paths (CONTRACTS §1)
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

CAND_PARQUET = os.path.join(DERIVED, "candidates.parquet")
TRAIN_KEYS_CSV = os.path.join(DERIVED, "train_inc_keys.csv")

PKL_OUT = os.path.join(DERIVED, "lasso_cv_fit.pkl")
TABLE_STABILITY = os.path.join(TABLES, "table2_lasso_stability.csv")
TABLE_SELECTED = os.path.join(TABLES, "table2_lasso_selected.csv")

# ---------------------------------------------------------------------------
# The 17 LASSO candidate columns (CONTRACTS §5). MUST match 02 exactly.
# Continuous `shock_index` is engineered in candidates.parquet but is NOT a
# LASSO candidate (CONTRACTS §5 CORRECTION); SI enters via si_ge0_9 + si_ge1_0.
# v3 = 17 (was 18): `unstable_ring` removed — Fix 3 / D-2. Do NOT re-add it here
# without also re-adding it in 02 and in 05's FORCE_IN_ANCHORS; the three lists
# are a single contract (CONTRACTS §5, §13 G3a/G3b).
# `sbp_unrecordable` stays — Fix 4 acts in 05 only (CONTRACTS §5.1).
# ---------------------------------------------------------------------------
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

N_LASSO_X_EXPECTED = 17
assert "unstable_ring" not in LASSO_X_COLS, (
    "Fix 3 / D-2 VIOLATION: `unstable_ring` is back in 04's LASSO_X_COLS "
    "(CONTRACTS §13 G3a). 02 and 04 must stay byte-identical on this list.")
assert len(LASSO_X_COLS) == N_LASSO_X_EXPECTED == len(set(LASSO_X_COLS)), (
    f"LASSO_X_COLS must be exactly {N_LASSO_X_EXPECTED} unique arrival "
    f"predictors; got {len(LASSO_X_COLS)} ({len(set(LASSO_X_COLS))} unique)")

OUTCOME_COL = "major_hemorrhage_event"
JOIN_KEYS = ["puf_year", "inc_key"]
COHORT_FILTER_COL = "cohort_primary"

# Stability threshold (parsimonious refit; no fallback — CONTRACTS §7 row 04)
STABILITY_THRESHOLD = 0.85
MIN_SELECTED_WARN = 5

# Defaults (full run); overridable with --bootstraps / --subsample for smoke test.
# DEFAULT_SUBSAMPLE = None -> use the full training-set size (n-out-of-n bootstrap).
DEFAULT_N_BOOTSTRAPS = 100
DEFAULT_SUBSAMPLE = None
DEFAULT_MIN_EVENTS = 500
SAMPLE_MIN_EVENTS = 20

CV_CS = 50
CV_FOLDS = 5


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def log(msg):
    print(msg, flush=True)


def load_data(sample=0):
    """Load the candidate matrix, filter to the analytic cohort, restrict to
    train. `sample` (CONTRACTS §9-E) is a real row-downsample of the training
    rows — see the `--sample` vs `--subsample` note in the module docstring."""
    banner("STEP 1 — Load candidates, filter cohort_primary==1, restrict to train")
    if not os.path.exists(CAND_PARQUET):
        raise FileNotFoundError(CAND_PARQUET)
    if not os.path.exists(TRAIN_KEYS_CSV):
        raise FileNotFoundError(TRAIN_KEYS_CSV)

    log(f"Reading candidates : {CAND_PARQUET}")
    df = pd.read_parquet(CAND_PARQUET)
    log(f"  candidates shape : {df.shape}")

    # ANALYTIC-COHORT RULE (CONTRACTS §3): ring-only primary cohort.
    if COHORT_FILTER_COL not in df.columns:
        raise KeyError(f"candidates parquet missing {COHORT_FILTER_COL}")
    df = df[df[COHORT_FILTER_COL] == 1].copy()
    log(f"  after cohort_primary==1 : {df.shape}")

    log(f"Reading train keys : {TRAIN_KEYS_CSV}")
    keys = pd.read_csv(TRAIN_KEYS_CSV)
    log(f"  train keys shape : {keys.shape}")

    for k in JOIN_KEYS:
        if k not in df.columns:
            raise KeyError(f"Candidates parquet missing join key: {k}")
        if k not in keys.columns:
            raise KeyError(f"Train keys CSV missing join key: {k}")
        df[k] = pd.to_numeric(df[k], errors="coerce").astype("int64")
        keys[k] = pd.to_numeric(keys[k], errors="coerce").astype("int64")

    keys = keys[JOIN_KEYS].drop_duplicates()
    df_train = df.merge(keys, on=JOIN_KEYS, how="inner")
    log(f"  after inner-join : {df_train.shape}")

    # --- §9-E smoke arg: REAL row-downsample of the training set -------------
    # Same form as every other data-touching script in the pipeline
    # (df.sample(n=N, random_state=42)); applied after the train join so the
    # analytic-cohort and split contracts are honoured first.
    if sample and sample < len(df_train):
        df_train = df_train.sample(n=sample, random_state=SEED).reset_index(drop=True)
        log(f"  [SMOKE] --sample: downsampled train to {len(df_train):,} rows")
    elif sample:
        log(f"  [SMOKE] --sample={sample:,} >= train N ({len(df_train):,}) — "
            f"no downsample applied")

    missing = [c for c in LASSO_X_COLS if c not in df_train.columns]
    if missing:
        raise KeyError(f"Candidates parquet missing {len(missing)} LASSO_X_COLS: {missing}")
    if OUTCOME_COL not in df_train.columns:
        raise KeyError(f"Candidates parquet missing outcome column: {OUTCOME_COL}")

    na_counts = df_train[LASSO_X_COLS + [OUTCOME_COL]].isna().sum()
    bad = na_counts[na_counts > 0]
    if len(bad) > 0:
        raise ValueError(f"Unexpected NaNs in LASSO inputs (02 should have cleaned):\n{bad}")

    X = df_train[LASSO_X_COLS].to_numpy(dtype=np.float64)
    y = df_train[OUTCOME_COL].to_numpy(dtype=np.int8)

    log(f"  X shape          : {X.shape}")
    log(f"  y positives      : {int(y.sum())} ({y.mean() * 100:.2f}%)")
    return X, y


def pick_lambda_1se_C(clf):
    """glmnet lambda.1se rule in sklearn parametrization: the SMALLEST C
    (= strongest L1 penalty / sparsest model) whose mean CV AUROC is within
    1 SE of the best mean CV AUROC."""
    keys = list(clf.scores_.keys())
    if len(keys) != 1:
        scores = np.mean(np.stack([clf.scores_[k] for k in keys], axis=0), axis=0)
    else:
        scores = clf.scores_[keys[0]]
    n_folds = scores.shape[0]
    mean_score = scores.mean(axis=0)
    se_score = scores.std(axis=0, ddof=1) / np.sqrt(n_folds)

    best_idx = int(np.argmax(mean_score))
    best_mean = float(mean_score[best_idx])
    best_se = float(se_score[best_idx])
    threshold = best_mean - best_se

    Cs = np.asarray(clf.Cs_, dtype=np.float64)
    candidate_mask = mean_score >= threshold
    if not candidate_mask.any():
        C_1se = float(Cs[best_idx])
        idx_1se = best_idx
    else:
        candidate_Cs = Cs.copy()
        candidate_Cs[~candidate_mask] = np.inf
        idx_1se = int(np.argmin(candidate_Cs))
        C_1se = float(Cs[idx_1se])

    diagnostics = {
        "best_idx": best_idx, "best_C": float(Cs[best_idx]),
        "best_mean": best_mean, "best_se": best_se,
        "threshold": float(threshold), "idx_1se": idx_1se, "C_1se": C_1se,
        "candidate_count": int(candidate_mask.sum()), "n_Cs": int(len(Cs)),
    }
    return C_1se, diagnostics


def fit_lasso_cv_global(X_scaled, y):
    banner("STEP 2 — Global LogisticRegressionCV (lambda.1se) on full train")
    log(f"  penalty='l1', solver='saga', Cs={CV_CS}, cv={CV_FOLDS}, scoring='roc_auc'")
    t0 = time.time()
    clf = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=CV_CS, cv=CV_FOLDS, scoring="roc_auc",
        max_iter=10000, n_jobs=-1, random_state=SEED, refit=True,
    )
    clf.fit(X_scaled, y)
    dt = time.time() - t0
    log(f"  LogisticRegressionCV fit time: {dt:.1f}s ({dt / 60:.1f} min)")

    C_1se, diag = pick_lambda_1se_C(clf)
    log(f"  best_C (lambda.min equiv) = {diag['best_C']:.6g}")
    log(f"  best mean CV AUROC        = {diag['best_mean']:.4f}")
    log(f"  C_1se (lambda.1se equiv)  = {C_1se:.6g}")
    log(f"  candidates within 1 SE    = {diag['candidate_count']} / {diag['n_Cs']}")

    lr_1se = LogisticRegression(
        penalty="l1", solver="saga", C=C_1se, max_iter=10000,
        random_state=SEED, n_jobs=-1,
    )
    lr_1se.fit(X_scaled, y)
    cv_beta = lr_1se.coef_[0].astype(np.float64)
    log(f"  cv_beta nonzero count = {int(np.sum(cv_beta != 0))} / {len(cv_beta)}")
    return C_1se, cv_beta


def fit_lasso_cv(X, y):
    log("Fitting StandardScaler on X_train ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    C_1se, cv_beta = fit_lasso_cv_global(X_scaled, y)
    return X_scaled, scaler, C_1se, cv_beta


def run_bootstraps(X_scaled, y, n_bootstraps, subsample, min_events):
    banner(f"STEP 3 — {n_bootstraps} per-bootstrap lambda.1se LASSO iterations "
           f"(n_sub={subsample:,}, min_events={min_events})")
    n_features = X_scaled.shape[1]
    n_rows = X_scaled.shape[0]
    selection_counts = np.zeros(n_features, dtype=np.int64)
    boot_C_1se_values = np.zeros(n_bootstraps, dtype=np.float64)
    t_start = time.time()

    for i in range(n_bootstraps):
        t_iter = time.time()
        rng = np.random.RandomState(SEED + i)
        idx = rng.choice(n_rows, size=subsample, replace=True)
        yb = y[idx]

        redraws = 0
        while yb.sum() < min_events:
            redraws += 1
            rng2 = np.random.RandomState(SEED + i + 10000 * redraws)
            idx = rng2.choice(n_rows, size=subsample, replace=True)
            yb = y[idx]
            if redraws > 25:
                raise RuntimeError(f"Bootstrap {i}: failed to draw >= {min_events} events")

        Xb = X_scaled[idx]
        inner_cv = LogisticRegressionCV(
            penalty="l1", solver="saga", Cs=CV_CS, cv=CV_FOLDS, scoring="roc_auc",
            max_iter=5000, n_jobs=-1, random_state=SEED + i, refit=False,
        )
        inner_cv.fit(Xb, yb)
        C_1se, _diag = pick_lambda_1se_C(inner_cv)
        boot_C_1se_values[i] = C_1se

        lr = LogisticRegression(
            penalty="l1", solver="saga", C=C_1se, max_iter=5000,
            random_state=SEED + i, n_jobs=-1,
        )
        lr.fit(Xb, yb)
        nonzero_mask = (lr.coef_[0] != 0)
        selection_counts += nonzero_mask.astype(np.int64)

        dt_iter = time.time() - t_iter
        dt_total = time.time() - t_start
        eta = (dt_total / (i + 1)) * (n_bootstraps - (i + 1))
        log(f"  boot {i + 1:3d}/{n_bootstraps} | events={int(yb.sum()):6d} | "
            f"C_1se={C_1se:.4g} | nonzero={int(nonzero_mask.sum()):2d}/{n_features} | "
            f"iter={dt_iter:5.1f}s | elapsed={dt_total / 60:5.1f}min | eta={eta / 60:5.1f}min")

    selection_freq = selection_counts / float(n_bootstraps)
    log(f"  per-bootstrap C_1se: median={np.median(boot_C_1se_values):.4g}, "
        f"min={boot_C_1se_values.min():.4g}, max={boot_C_1se_values.max():.4g}")
    return selection_freq


def build_tables(feature_names, cv_beta, selection_freq):
    banner("STEP 4 — Build stability + selected tables (threshold=0.85, no fallback)")
    abs_beta = np.abs(cv_beta)
    order = np.argsort(-abs_beta, kind="stable")
    rank = np.empty(len(feature_names), dtype=np.int64)
    rank[order] = np.arange(1, len(feature_names) + 1)

    stability = pd.DataFrame({
        "Variable": feature_names,
        "Selection_Frequency": np.round(selection_freq, 4),
        "CV_Beta": cv_beta,
        "CV_Abs_Beta_Rank": rank,
    }).sort_values(["Selection_Frequency", "CV_Abs_Beta_Rank"],
                   ascending=[False, True]).reset_index(drop=True)

    stability.to_csv(TABLE_STABILITY, index=False)
    log(f"Wrote stability table: {TABLE_STABILITY}  ({len(stability)} rows)")

    selected = stability[stability["Selection_Frequency"] >= STABILITY_THRESHOLD].copy()
    if len(selected) < MIN_SELECTED_WARN:
        log(f"  WARNING: only {len(selected)} feature(s) hit "
            f"Selection_Frequency >= {STABILITY_THRESHOLD:.2f} — passing small set to 05 "
            f"(no auto-relax; 05 force-in anchors will backstop).")

    selected = selected[["Variable", "Selection_Frequency", "CV_Beta"]].reset_index(drop=True)
    selected.to_csv(TABLE_SELECTED, index=False)
    log(f"Wrote selected table:  {TABLE_SELECTED}  ({len(selected)} rows, threshold={STABILITY_THRESHOLD:.2f})")
    return stability, selected, STABILITY_THRESHOLD


def write_pickle(scaler, best_C, cv_beta, feature_names):
    banner("STEP 5 — Write lasso_cv_fit.pkl")
    payload = {
        "best_C": float(best_C),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "feature_names": list(feature_names),
        "cv_beta": np.asarray(cv_beta, dtype=np.float64),
    }
    with open(PKL_OUT, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Wrote pickle: {PKL_OUT}  (best_C={payload['best_C']:.6g}, "
        f"n_features={len(payload['feature_names'])})")


def main():
    parser = argparse.ArgumentParser(
        description="Pelvic v3 Step 04: LASSO stability selection (lambda.1se, threshold=0.85)")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="Smoke test (CONTRACTS §9-E): real row-downsample of "
                             "the TRAIN join to N rows (df.sample(n=N, "
                             "random_state=42)) before any fitting. This is the "
                             "flag §13's '--sample 50000' smoke test uses. NOT "
                             "the same as --subsample, which sizes each bootstrap "
                             "resample and leaves the input untouched.")
    parser.add_argument("--bootstraps", type=int, default=None, metavar="N",
                        help="Number of bootstrap iterations. Default 100 (full run).")
    parser.add_argument("--subsample", type=int, default=None, metavar="N",
                        help="Bootstrap RESAMPLE size (drawn with replacement from "
                             "the training set). Default = full train N "
                             "(n-out-of-n bootstrap). Does not reduce the input.")
    args = parser.parse_args()

    banner("Pelvic v3 — Step 04: LASSO stability selection (lambda.1se, threshold=0.85)")
    log(f"SEED={SEED}  outcome={OUTCOME_COL}  #LASSO_X_COLS={len(LASSO_X_COLS)}")

    smoke_mode = (args.sample > 0) or (args.bootstraps is not None) \
        or (args.subsample is not None)
    n_bootstraps = args.bootstraps if args.bootstraps is not None else DEFAULT_N_BOOTSTRAPS
    min_events = SAMPLE_MIN_EVENTS if smoke_mode else DEFAULT_MIN_EVENTS

    X, y = load_data(sample=args.sample)

    # Full-run subsample defaults to the full training-set size.
    if args.subsample is not None:
        subsample = args.subsample
    else:
        subsample = len(X)

    log(f"{'SMOKE TEST' if smoke_mode else 'FULL RUN'}: {n_bootstraps} bootstraps, "
        f"train_n={len(X):,}, subsample={subsample:,}, min_events={min_events}")
    if smoke_mode:
        log("  *** SMOKE MODE — the pkl and tables written by this run are NOT "
            "publishable. They overwrite the real artifacts that 05 reads; "
            "re-run 04 with no flags before running 05. ***")

    X_scaled, scaler, best_C, cv_beta = fit_lasso_cv(X, y)
    selection_freq = run_bootstraps(X_scaled, y, n_bootstraps=n_bootstraps,
                                    subsample=subsample, min_events=min_events)
    stability, selected, threshold = build_tables(LASSO_X_COLS, cv_beta, selection_freq)
    write_pickle(scaler, best_C, cv_beta, LASSO_X_COLS)

    banner("DONE")
    log(f"global lambda.1se C = {best_C:.6g}  |  n selected (freq>={threshold:.2f}) = {len(selected)}")
    log("selected: " + ", ".join(selected["Variable"].tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
