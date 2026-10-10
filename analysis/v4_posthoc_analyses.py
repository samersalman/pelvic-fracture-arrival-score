"""
Pelvic Arrival Tool v4 -- reviewer-requested post-hoc analyses (SUPPLEMENTARY)
================================================================================

Reads the locked v3 analysis outputs READ-ONLY and writes aggregate CSV/JSON only to
derived/. Nothing here changes the locked score; every output is labelled
post-hoc supplementary (the corresponding SAP deviation is documented separately).

  A1 (D1)  Domain-collapsed sensitivity refit. Candidate pool = the 17 v3
           candidates minus the duplicate threshold tiers sbp_lt70, si_ge0_9 and
           gcs_le8 (14 candidates). Same stability LASSO (100 bootstraps,
           lambda.1se, 0.85 threshold, seeds 42+i), same unpenalized refit, same
           |beta|>=0.15 screen, same force-in anchors (age_65_74, age_ge75,
           cc_anticoag), same anchor-protected VIF loop, same D-11 integer rule,
           same sbp_unrecordable override. Also: correlation matrix and VIF for the
           published criteria; a replication check of the v3 05/12 logic.
  A2 (D4)  Hypotension-by-mechanism table on the v3 test set (n = 79,912) and the
           worked example from the real card.
  A3 (D5)  Anticoagulant-removal invariance check (hard asserts; exit 2 on any
           difference from v3).

Stages (run in this order)
--------------------------
  python3 v4_posthoc_analyses.py stability --tag a1        # ~30 min, background
  python3 v4_posthoc_analyses.py stability --tag parity17  # replication of v3 04
  python3 v4_posthoc_analyses.py finish                    # everything else

`stability` is the only slow step. `finish` refuses to run unless both
stability outputs exist. Seed 42.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
import scipy.stats as sstats
import sklearn
import statsmodels
import statsmodels.api as sm
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SEED = 42
np.random.seed(SEED)

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()

# Flat repository layout: v3 outputs and the outputs of this script share
# derived/, tables/ and logs/ (file names do not collide).
ROOT = config.REPO_ROOT
V3D = config.DERIVED
V3T = config.TABLES
OUT = os.environ.get("V4_OUT_OVERRIDE") or config.DERIVED   # override is for dry runs only
LOGDIR = os.environ.get("V4_LOG_OVERRIDE") or config.LOGS
LOGFILE = os.path.join(LOGDIR, "v4_posthoc_analyses.log")
SCRIPT = os.path.abspath(__file__)

LABEL = "POST-HOC SUPPLEMENTARY (reviewer-requested analysis); does not replace the locked v3 score"

# ----- v3 contract (copied from v3/code 04, 05, 12; v3 is never edited) --------
LASSO_X_COLS_V3 = [
    "age_65_74", "age_ge75", "sex_female",
    "sbp_lt90", "sbp_lt70", "sbp_unrecordable",
    "hr_ge120",
    "si_ge0_9", "si_ge1_0",
    "gcs_motor_le5", "gcs_le8",
    "mech_high_energy", "mech_mvc", "mech_fall",
    "air_transport",
    "cc_anticoag", "cc_bleeding",
]
A1_DROPPED_TIERS = ["sbp_lt70", "si_ge0_9", "gcs_le8"]
A1_COLS = [c for c in LASSO_X_COLS_V3 if c not in A1_DROPPED_TIERS]
assert len(LASSO_X_COLS_V3) == 17 and len(A1_COLS) == 14

FORCE_IN_ANCHORS = ["age_65_74", "age_ge75", "cc_anticoag"]
OVERRIDE_VARS = ["sbp_unrecordable"]
OUTCOME = "major_hemorrhage_event"
KEYS = ["puf_year", "inc_key"]
STABILITY_THRESHOLD = 0.85
N_BOOT_STAB = 100
CV_CS, CV_FOLDS = 50, 5
MIN_EVENTS = 500
BETA_DROP = 0.15
VIF_MAX = 5.0
MIN_KEPT = 5
CARD_SCALE_TRIPWIRE = 20
N_BOOT_AUC = 2000

# published-card reference values (v3) asserted in A3
V3_AUROC = 0.8823
V3_STRATA = {"rule_out": 45127, "observe": 22661, "escalate": 12124}
V3_MW = {"score_tp": 5442, "score_fp": 6287, "si_tp": 4764, "si_fp": 6971,
         "delta_events": 678, "delta_fp": -684, "n_common": 75072}

_LOG = None


def log(msg=""):
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(str(msg) + "\n")
        _LOG.flush()


def banner(msg):
    line = "=" * 78
    log(f"\n{line}\n{msg}\n{line}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
    log(f"  wrote {path}")


# =============================================================================
# Data loading (read-only)
# =============================================================================
def _norm_keys(df):
    for k in KEYS:
        df[k] = pd.to_numeric(df[k], errors="coerce").astype("int64")
    return df


_CAND = None


def load_candidates():
    global _CAND
    if _CAND is None:
        df = pd.read_parquet(os.path.join(V3D, "candidates.parquet"))
        df = df[df["cohort_primary"] == 1].copy()
        _CAND = _norm_keys(df)
    return _CAND


def load_split(name):
    """name in {train, test, temporal_derive, temporal_valid}. Inner join of the
    primary cohort to the v3 key file; asserts no key is lost or duplicated."""
    fn = {"train": "train_inc_keys.csv", "test": "test_inc_keys.csv",
          "temporal_derive": "temporal_derive_inc_keys.csv",
          "temporal_valid": "temporal_valid_inc_keys.csv"}[name]
    keys = _norm_keys(pd.read_csv(os.path.join(V3D, fn)))[KEYS].drop_duplicates()
    cand = load_candidates()
    df = cand.merge(keys, on=KEYS, how="inner")
    assert len(df) == len(keys), f"{name}: join lost rows {len(df)} vs {len(keys)}"
    return df.reset_index(drop=True)


def load_scored_test():
    sc = pd.read_parquet(os.path.join(V3D, "cohort_test_scored.parquet"))
    return _norm_keys(sc)


# =============================================================================
# Stage: stability selection (replicates v3/code/04 exactly, variable subset)
# =============================================================================
def pick_lambda_1se_C(clf):
    keys = list(clf.scores_.keys())
    if len(keys) != 1:
        scores = np.mean(np.stack([clf.scores_[k] for k in keys], axis=0), axis=0)
    else:
        scores = clf.scores_[keys[0]]
    n_folds = scores.shape[0]
    mean_score = scores.mean(axis=0)
    se_score = scores.std(axis=0, ddof=1) / np.sqrt(n_folds)
    best_idx = int(np.argmax(mean_score))
    threshold = float(mean_score[best_idx]) - float(se_score[best_idx])
    Cs = np.asarray(clf.Cs_, dtype=np.float64)
    mask = mean_score >= threshold
    if not mask.any():
        return float(Cs[best_idx])
    cand = Cs.copy()
    cand[~mask] = np.inf
    return float(Cs[int(np.argmin(cand))])


def stage_stability(tag, n_boot, sample):
    cols = {"a1": A1_COLS, "parity17": LASSO_X_COLS_V3}[tag]
    smoke = bool(sample) or n_boot != N_BOOT_STAB
    banner(f"STABILITY [{tag}] {len(cols)} candidates, {n_boot} bootstraps"
           f"{'  *** SMOKE: output goes to scratch only ***' if smoke else ''}")
    log(f"  sklearn {sklearn.__version__}  numpy {np.__version__}  seed {SEED}")
    tr = load_split("train")
    if sample and sample < len(tr):
        tr = tr.sample(n=sample, random_state=SEED).reset_index(drop=True)
    assert tr[cols + [OUTCOME]].isna().sum().sum() == 0
    X = tr[cols].to_numpy(dtype=np.float64)
    y = tr[OUTCOME].to_numpy(dtype=np.int8)
    log(f"  train N={len(y):,} events={int(y.sum()):,} ({100*y.mean():.2f}%)  cols={cols}")
    Xs = StandardScaler().fit_transform(X)
    n_rows = Xs.shape[0]
    min_events = 20 if smoke else MIN_EVENTS
    counts = np.zeros(len(cols), dtype=np.int64)
    meta = []
    t0 = time.time()
    for i in range(n_boot):
        ti = time.time()
        rng = np.random.RandomState(SEED + i)
        idx = rng.choice(n_rows, size=n_rows, replace=True)
        yb = y[idx]
        redraws = 0
        while yb.sum() < min_events:
            redraws += 1
            rng2 = np.random.RandomState(SEED + i + 10000 * redraws)
            idx = rng2.choice(n_rows, size=n_rows, replace=True)
            yb = y[idx]
            if redraws > 25:
                raise RuntimeError("bootstrap failed to draw enough events")
        Xb = Xs[idx]
        cv = LogisticRegressionCV(penalty="l1", solver="saga", Cs=CV_CS, cv=CV_FOLDS,
                                  scoring="roc_auc", max_iter=5000, n_jobs=-1,
                                  random_state=SEED + i, refit=False)
        cv.fit(Xb, yb)
        C1 = pick_lambda_1se_C(cv)
        lr = LogisticRegression(penalty="l1", solver="saga", C=C1, max_iter=5000,
                                random_state=SEED + i, n_jobs=-1)
        lr.fit(Xb, yb)
        nz = lr.coef_[0] != 0
        counts += nz.astype(np.int64)
        meta.append({"boot": i + 1, "C_1se": C1, "nonzero": int(nz.sum()),
                     "events": int(yb.sum()), "sec": round(time.time() - ti, 1)})
        el = time.time() - t0
        log(f"  boot {i+1:3d}/{n_boot} | events={int(yb.sum()):6d} | C_1se={C1:.4g} | "
            f"nonzero={int(nz.sum()):2d}/{len(cols)} | iter={time.time()-ti:5.1f}s | "
            f"elapsed={el/60:5.1f}min | eta={(el/(i+1))*(n_boot-i-1)/60:5.1f}min")
    freq = counts / float(n_boot)
    st = pd.DataFrame({"Variable": cols, "Selection_Frequency": np.round(freq, 4)})
    st = st.sort_values("Selection_Frequency", ascending=False, kind="stable").reset_index(drop=True)
    outdir = OUT if not smoke else os.path.join(os.environ.get("V4_SCRATCH", "/tmp"), "v4_smoke")
    os.makedirs(outdir, exist_ok=True)
    p_csv = os.path.join(outdir, f"a1_stability_{tag}.csv")
    st.to_csv(p_csv, index=False)
    jdump({"label": LABEL, "tag": tag, "candidates": cols, "n_bootstraps": n_boot,
           "threshold": STABILITY_THRESHOLD, "seed": SEED, "n_train": int(len(y)),
           "n_train_events": int(y.sum()), "sklearn": sklearn.__version__,
           "per_bootstrap": meta, "runtime_min": round((time.time() - t0) / 60, 1)},
          os.path.join(outdir, f"a1_stability_{tag}_meta.json"))
    log(st.to_string(index=False))
    log(f"  selected (freq>={STABILITY_THRESHOLD}): "
        f"{st[st.Selection_Frequency >= STABILITY_THRESHOLD].Variable.tolist()}")
    return 0


# =============================================================================
# 05 replication: unpenalized refit + D-11 integer card
# =============================================================================
def fit_logit(X_df, y):
    Xc = sm.add_constant(X_df, has_constant="add")
    return sm.Logit(y, Xc.astype(float)).fit(disp=False, maxiter=200, method="newton")


def compute_vif(X_df):
    Xc = sm.add_constant(X_df, has_constant="add").astype(float).values
    cols = ["const"] + list(X_df.columns)
    return {n: float(variance_inflation_factor(Xc, i)) for i, n in enumerate(cols) if n != "const"}


def select_integer_anchor(beta_by_var, point_vars):
    positive = {v: float(beta_by_var[v]) for v in point_vars if float(beta_by_var[v]) > 0}
    eligible = {v: b for v, b in positive.items() if abs(b) >= BETA_DROP}
    if not eligible:
        raise RuntimeError("no point-role variable passed the screen (degenerate)")
    av = min(eligible, key=eligible.get)
    return float(eligible[av]), av, sorted(set(point_vars) - set(eligible))


def integer_points(beta, anchor):
    pts = int(round(float(beta) / float(anchor)))
    if pts == 0 and abs(float(beta)) >= BETA_DROP:
        pts = 1 if beta > 0 else -1
    assert (pts >= 0) if beta > 0 else (pts <= 0)
    return pts


def refit_integer(df, selected, freq_lookup, outcome=OUTCOME, verbose=True):
    """Mirror of v3/code/05. `selected` is the stability-selected list in order."""
    p = log if verbose else (lambda *_: None)
    y = df[outcome].astype(int).values
    variables = list(selected)
    for a in FORCE_IN_ANCHORS:
        if a not in variables:
            variables.append(a)
    res = fit_logit(df[variables].astype(float), y)
    beta = res.params.drop("const")
    weak = [v for v in beta[beta.abs() < BETA_DROP].index if v not in FORCE_IN_ANCHORS]
    dropped_beta = list(weak)
    variables = [v for v in variables if v not in weak]
    for a in FORCE_IN_ANCHORS:
        if a not in variables:
            variables.append(a)
    assert len(variables) >= MIN_KEPT
    X = df[variables].astype(float)
    res = fit_logit(X, y)
    dropped_vif, vif_trace = [], []
    for it in range(20):
        vifs = compute_vif(X)
        droppable = {v: q for v, q in vifs.items() if v not in FORCE_IN_ANCHORS}
        if not droppable:
            break
        mv = max(droppable, key=lambda k: droppable[k])
        gv = max(vifs, key=lambda k: vifs[k])
        vif_trace.append({"iter": it, "max_droppable": mv, "max_droppable_vif": droppable[mv],
                          "global_max": gv, "global_max_vif": vifs[gv]})
        if droppable[mv] <= VIF_MAX:
            break
        near = [v for v, q in droppable.items() if q >= droppable[mv] - 0.5]
        dv = min(near, key=lambda v: freq_lookup.get(v, 0.0)) if len(near) > 1 else mv
        dropped_vif.append(dv)
        variables = [v for v in variables if v != dv]
        X = df[variables].astype(float)
        res = fit_logit(X, y)
    else:
        raise RuntimeError("VIF loop did not converge")
    params, bse, conf = res.params, res.bse, res.conf_int()
    point_vars = [v for v in variables if v not in OVERRIDE_VARS]
    override_vars = [v for v in variables if v in OVERRIDE_VARS]
    beta_by_var = {v: float(params.loc[v]) for v in variables}
    anchor, anchor_var, below = select_integer_anchor(beta_by_var, point_vars)
    points = {v: integer_points(beta_by_var[v], anchor) for v in point_vars}
    over = {v: p_ for v, p_ in points.items() if abs(p_) > CARD_SCALE_TRIPWIRE}
    assert not over, f"card-scale tripwire: {over}"
    rows = []
    for v in variables:
        rows.append({"Variable": v, "beta": float(params.loc[v]), "SE": float(bse.loc[v]),
                     "OR": float(np.exp(params.loc[v])),
                     "OR_CI_low": float(np.exp(conf.loc[v, 0])),
                     "OR_CI_high": float(np.exp(conf.loc[v, 1])),
                     "Role": "override" if v in override_vars else "point",
                     "Points": (points[v] if v in points else None)})
    lp_train = float(params.loc["const"]) + df[variables].to_numpy(float) @ np.array(
        [beta_by_var[v] for v in variables])
    score_train = df[point_vars].to_numpy(float) @ np.array([points[v] for v in point_vars], float)
    out = {"variables": variables, "point_vars": point_vars, "override_vars": override_vars,
           "intercept": float(params.loc["const"]), "beta": beta_by_var, "points": points,
           "anchor": anchor, "anchor_var": anchor_var, "below_screen": below,
           "dropped_beta": dropped_beta, "dropped_vif": dropped_vif, "vif_trace": vif_trace,
           "rows": rows, "n_train": int(len(y)), "events_train": int(y.sum()),
           "train_auroc_integer": float(roc_auc_score(y, score_train)),
           "train_auroc_logit": float(roc_auc_score(y, lp_train))}
    p(f"  variables: {variables}")
    p(f"  dropped by beta: {dropped_beta}  dropped by VIF: {dropped_vif}")
    p(f"  anchor: {anchor_var} beta={anchor:.4f}; below screen: {below}")
    p(f"  points: {points}")
    p(f"  train AUROC integer={out['train_auroc_integer']:.4f} logit={out['train_auroc_logit']:.4f}")
    return out


def score_with(df, fit):
    pts = np.array([fit["points"][v] for v in fit["point_vars"]], float)
    return df[fit["point_vars"]].to_numpy(float) @ pts


# =============================================================================
# AUROC helpers
# =============================================================================
def boot_auc_paired(y, scores, n_boot, seed=SEED):
    """Shared resample indices across all score vectors in `scores` (dict)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(scores)
    res = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        for k in names:
            res[k][b] = roc_auc_score(yb, scores[k][idx])
    return res


def pct_ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def _delong_v(y, pred):
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.float64)
    pos, neg = pred[y == 1], pred[y == 0]
    n1, n0 = len(pos), len(neg)
    tz = rankdata(pred, method="average")
    V10 = (tz[y == 1] - rankdata(pos, method="average")) / n0
    V01 = 1.0 - (tz[y == 0] - rankdata(neg, method="average")) / n1
    return V10, V01


def delong_test(y, p1, p2):
    V10_1, V01_1 = _delong_v(y, p1)
    V10_2, V01_2 = _delong_v(y, p2)
    n1, n0 = V10_1.shape[0], V01_1.shape[0]
    a1, a2 = float(V10_1.mean()), float(V10_2.mean())
    S = np.cov(np.vstack([V10_1, V10_2])) / n1 + np.cov(np.vstack([V01_1, V01_2])) / n0
    d = a1 - a2
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    if var <= 0:
        return {"auc1": a1, "auc2": a2, "delta": d, "se": 0.0, "z": 0.0, "p": 1.0}
    se = float(np.sqrt(var))
    z = d / se
    return {"auc1": a1, "auc2": a2, "delta": d, "se": se, "z": z,
            "p": float(2 * (1 - sstats.norm.cdf(abs(z))))}


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    lo, hi = proportion_confint(count=int(k), nobs=int(n), alpha=0.05, method="wilson")
    return float(lo), float(hi)


# =============================================================================
# Matched-workload (copy-adapt of v3/code/08 best_cutoff_for_target + metrics)
# =============================================================================
SI_GRID = np.round(np.arange(0.20, 4.0001, 0.01), 2)


def best_cutoff_for_target(values, grid, target_frac, prefer=None):
    values = np.asarray(values, dtype=np.float64)
    n_total = int(values.size)
    grid = np.asarray(grid, dtype=np.float64)
    obs = np.sort(values[np.isfinite(values)])
    n_hit = obs.size - np.searchsorted(obs, grid, side="left")
    fracs = n_hit / float(n_total)
    diffs = np.abs(fracs - float(target_frac))
    d_min = float(diffs.min())
    idx = int(np.argmin(diffs))
    if prefer:
        for j in np.flatnonzero(diffs <= d_min + 1e-12):
            if round(float(grid[j]), 6) in prefer:
                idx = int(j)
                break
    cut = float(grid[idx])
    return cut, values >= cut, float(fracs[idx]), float(diffs[idx])


def build_si_grid(si):
    legible = set(np.round(SI_GRID, 6).tolist())
    v = si[np.isfinite(si)]
    grid = np.unique(np.concatenate([SI_GRID, np.round(v, 6)]))
    return grid, legible


def flag_counts(y, flag):
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(bool)
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    return {"n_flagged": tp + fp, "events_caught_TP": tp, "false_positives_FP": fp,
            "events_missed_FN": int((~flag & (y == 1)).sum()),
            "sensitivity": tp / max(int(y.sum()), 1), "ppv": tp / max(tp + fp, 1)}


def matched_workload(score, si, rsig, y, anchor_cut=None, anchor_frac=None):
    """Replicates 08's T5D overall block: restrict to the common complete-case set
    (SI and rSIG observed), anchor on `score >= anchor_cut` (or a target fraction),
    match the shock index to the anchor's achieved fraction."""
    common = ~np.isnan(si) & ~np.isnan(rsig)
    s, sii, yy = score[common], si[common], y[common]
    if anchor_cut is not None:
        a_flag = s >= anchor_cut
        a_frac = float(a_flag.mean())
        a_cut = float(anchor_cut)
    else:
        grid = np.arange(np.floor(s.min()), np.ceil(s.max()) + 1)
        fr = np.array([(s >= g).mean() for g in grid])
        j = int(np.argmin(np.abs(fr - anchor_frac)))
        a_cut, a_frac = float(grid[j]), float(fr[j])
        a_flag = s >= a_cut
    grid, legible = build_si_grid(sii)
    cut, sflag, sfrac, _ = best_cutoff_for_target(sii, grid, a_frac, prefer=legible)
    a = flag_counts(yy, a_flag)
    b = flag_counts(yy, sflag)
    tol = min(0.25, 0.10 * 100 * a_frac)
    gap = 100 * (sfrac - a_frac)
    return {"n_common": int(common.sum()), "events_common": int(yy.sum()),
            "score_cutoff": a_cut, "score_pct_flagged": 100 * a_frac,
            "score": a, "si_cutoff": cut, "si_pct_flagged": 100 * sfrac,
            "si": b, "workload_gap_pp": gap, "tolerance_pp": tol,
            "within_tolerance": bool(abs(gap) <= tol + 1e-9),
            "delta_events_caught": a["events_caught_TP"] - b["events_caught_TP"],
            "delta_false_positives": a["false_positives_FP"] - b["false_positives_FP"]}


def strata_table(score, y):
    bins = {"rule_out (score <= 0)": score <= 0,
            "observe (score 1 to 4)": (score >= 1) & (score <= 4),
            "escalate (score >= 5)": score >= 5}
    out = []
    for k, m in bins.items():
        n, e = int(m.sum()), int(y[m].sum())
        lo, hi = wilson(e, n)
        out.append({"stratum": k, "n": n, "events": e, "rate": e / n if n else float("nan"),
                    "ci_low": lo, "ci_high": hi})
    return out


# =============================================================================
# Temporal (replicates v3/code/12 for the primary outcome; names-based)
# =============================================================================
def run_temporal(dm, vm, vars_kept, override_vars, n_boot, label):
    point_vars = [v for v in vars_kept if v not in override_vars]
    yd = dm[OUTCOME].to_numpy(int)
    Xd = sm.add_constant(dm[vars_kept].to_numpy(float), has_constant="add")
    fit = sm.Logit(yd, Xd).fit(disp=False, maxiter=200)
    intercept, betas = float(fit.params[0]), np.asarray(fit.params[1:], float)
    cmap = {v: float(b) for v, b in zip(vars_kept, betas)}
    anchor, avar, _ = select_integer_anchor(cmap, point_vars)
    pmap = {v: integer_points(cmap[v], anchor) for v in point_vars}
    for v in override_vars:
        pmap[v] = 0
    pts = np.array([pmap[v] for v in vars_kept], float)
    yv = vm[OUTCOME].to_numpy(int)
    Xv = vm[vars_kept].to_numpy(float)
    prob = 1 / (1 + np.exp(-(intercept + Xv @ betas)))
    sc = Xv @ pts
    rng = np.random.default_rng(SEED)
    n = len(yv)
    aucs_l, aucs_i = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aucs_l[b] = roc_auc_score(yv[idx], prob[idx])
        aucs_i[b] = roc_auc_score(yv[idx], sc[idx])
    r = {"label": label, "n_derive": int(len(yd)), "events_derive": int(yd.sum()),
         "n_valid": int(n), "events_valid": int(yv.sum()),
         "auroc_logit": float(roc_auc_score(yv, prob)), "ci_logit": pct_ci(aucs_l),
         "auroc_integer": float(roc_auc_score(yv, sc)), "ci_integer": pct_ci(aucs_i),
         "anchor_var": avar, "anchor_beta": anchor, "points": pmap}
    log(f"  [{label}] temporal valid N={n:,} ev={int(yv.sum()):,}  AUROC logit="
        f"{r['auroc_logit']:.4f} {r['ci_logit']}  integer={r['auroc_integer']:.4f} {r['ci_integer']}")
    return r


# =============================================================================
# Stage: finish
# =============================================================================
def stage_finish(args):
    global _LOG
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    _LOG = open(LOGFILE, "w")
    t_start = time.time()
    banner("v4_posthoc_analyses.py finish  |  " + LABEL)
    log(f"  python {platform.python_version()}  numpy {np.__version__}  pandas {pd.__version__}  "
        f"sklearn {sklearn.__version__}  statsmodels {statsmodels.__version__}  scipy {scipy.__version__}")
    log(f"  seed {SEED}   run {time.strftime('%Y-%m-%d %H:%M:%S')}")

    refit = json.load(open(os.path.join(V3D, "refit.json")))
    card = json.load(open(os.path.join(V3D, "score_card.json")))
    inputs = {}

    def track(path):
        inputs[os.path.relpath(path, ROOT)] = sha256_file(path)

    for fn in ["candidates.parquet", "cohort_test_scored.parquet", "train_inc_keys.csv",
               "test_inc_keys.csv", "temporal_derive_inc_keys.csv", "temporal_valid_inc_keys.csv",
               "refit.json", "score_card.json", "risk_tiers.csv"]:
        track(os.path.join(V3D, fn))
    for fn in ["table2_lasso_stability.csv", "table2_lasso_selected.csv", "table3_points.csv",
               "table4_discrimination.csv", "table4b_temporal.csv", "table5d_matched_workload.csv"]:
        track(os.path.join(V3T, fn))
    track(os.path.join(config.DERIVED, "results_bundle.json"))
    track(os.path.join(config.TABLES, "Table 1.csv"))

    # ------------------------------------------------------------------ data
    banner("Load train / test / temporal frames (read-only) and cross-check against v3 scored test")
    tr = load_split("train")
    te = load_split("test")
    td = load_split("temporal_derive")
    tv = load_split("temporal_valid")
    sc = load_scored_test()
    log(f"  train {len(tr):,}  test {len(te):,}  temporal derive {len(td):,}  temporal valid {len(tv):,}")
    assert len(tr) == 159824 and len(te) == 79912
    # NOTE: candidates.parquet carries a raw `shock_index` (NaN-bearing; v3 name-collision note in
    # 02_engineer_features). The comparator used in v3 step 06/08 is the one recomputed in the
    # scored parquet (pulse2/sbp2), so the candidate copy is dropped before the merge.
    te = te.drop(columns=["shock_index"]).merge(
        sc[KEYS + ["pelvic_score", "pelvic_lp", "override_nslo", "shock_index", "rsig"]],
        on=KEYS, how="inner")
    assert len(te) == 79912
    # features in candidates equal features in the scored parquet
    chk = te.merge(sc[KEYS + ["mech_fall", "sbp_lt90", "si_ge1_0", "age_ge75", "sex_female"]],
                   on=KEYS, suffixes=("", "_s"))
    for c in ["mech_fall", "sbp_lt90", "si_ge1_0", "age_ge75", "sex_female"]:
        assert (chk[c].values == chk[c + "_s"].values).all(), c
    log("  [OK] candidate features identical to scored-test features (5 spot columns)")
    pub_points = {k: int(v) for k, v in refit["points"].items()}
    pub_point_vars = list(refit["point_vars"])
    assert pub_point_vars == list(pub_points)
    y_te = te[OUTCOME].to_numpy(int)
    score_pub = te["pelvic_score"].to_numpy(float)
    recomputed = te[pub_point_vars].to_numpy(float) @ np.array([pub_points[v] for v in pub_point_vars], float)
    assert np.array_equal(recomputed, score_pub), "published score not reproduced from points"
    log(f"  [OK] published pelvic_score reproduced from refit.json points for all {len(te):,} test rows")
    ov = te["override_nslo"].to_numpy(int) == 1
    assert (ov == (te["sbp_unrecordable"].to_numpy(int) == 1)).all()

    # ======================================================================
    # A3: anticoagulant-removal invariance (hard asserts, BLOCKED on mismatch)
    # ======================================================================
    banner("A3  anticoagulant-removal invariance check")
    pts13 = {k: v for k, v in pub_points.items() if k != "cc_anticoag"}
    assert pub_points["cc_anticoag"] == 0 and len(pts13) == 13
    s13 = te[list(pts13)].to_numpy(float) @ np.array(list(pts13.values()), float)
    a3 = {"label": LABEL, "n_test": int(len(te)), "criteria_published_card": 14,
          "criteria_after_removal": 13, "removed": "cc_anticoag", "removed_points": 0,
          "checks": []}

    def a3_assert(name, ok, got, exp):
        a3["checks"].append({"check": name, "pass": bool(ok), "got": got, "expected": exp})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got} expected {exp}")
        if not ok:
            a3["status"] = "BLOCKED"
            jdump(a3, os.path.join(OUT, "a3_invariance_check.json"))
            log("A3 FAILED -> STOP (BLOCKED).")
            _LOG.close()
            sys.exit(2)

    a3_assert("score vector identical (13 vs 14 criteria), all test rows",
              np.array_equal(s13, score_pub), int((s13 != score_pub).sum()), 0)
    auc13 = float(roc_auc_score(y_te, s13))
    auc14 = float(roc_auc_score(y_te, score_pub))
    a3_assert("AUROC exactly equal (full float)", auc13 == auc14, repr(auc13), repr(auc14))
    a3_assert("AUROC rounds to 0.8823", round(auc13, 4) == V3_AUROC, round(auc13, 4), V3_AUROC)
    vtab = pd.read_csv(os.path.join(V3T, "table4_discrimination.csv"))
    v3_auc_tab = float(vtab.loc[vtab.Score == "pelvic_score", "AUROC"].iloc[0])
    a3_assert("AUROC equals table4_discrimination.csv pelvic_score", round(auc13, 4) == v3_auc_tab,
              round(auc13, 4), v3_auc_tab)
    st13 = strata_table(s13, y_te)
    st14 = strata_table(score_pub, y_te)
    n13 = [r["n"] for r in st13]
    a3_assert("strata n (<=0, 1-4, >=5)", n13 == list(V3_STRATA.values()), n13, list(V3_STRATA.values()))
    a3_assert("strata n and events identical to 14-criteria version",
              [(r["n"], r["events"]) for r in st13] == [(r["n"], r["events"]) for r in st14], "equal", "equal")
    rt = pd.read_csv(os.path.join(V3D, "risk_tiers.csv")).set_index("tier")
    a3_assert("strata events equal risk_tiers.csv",
              [r["events"] for r in st13] == [int(rt.loc[t, "composite_events"]) for t in ["rule_out", "observe", "escalate"]],
              [r["events"] for r in st13], [int(rt.loc[t, "composite_events"]) for t in ["rule_out", "observe", "escalate"]])
    si_all = te["shock_index"].to_numpy(float)
    rs_all = te["rsig"].to_numpy(float)
    mw14 = matched_workload(score_pub, si_all, rs_all, y_te, anchor_cut=5.0)
    mw13 = matched_workload(s13, si_all, rs_all, y_te, anchor_cut=5.0)
    a3_assert("matched-workload common n", mw13["n_common"] == V3_MW["n_common"], mw13["n_common"], V3_MW["n_common"])
    a3_assert("score >= +5 events caught", mw13["score"]["events_caught_TP"] == V3_MW["score_tp"],
              mw13["score"]["events_caught_TP"], V3_MW["score_tp"])
    a3_assert("score >= +5 false positives", mw13["score"]["false_positives_FP"] == V3_MW["score_fp"],
              mw13["score"]["false_positives_FP"], V3_MW["score_fp"])
    a3_assert("shock index events caught", mw13["si"]["events_caught_TP"] == V3_MW["si_tp"],
              mw13["si"]["events_caught_TP"], V3_MW["si_tp"])
    a3_assert("shock index false positives", mw13["si"]["false_positives_FP"] == V3_MW["si_fp"],
              mw13["si"]["false_positives_FP"], V3_MW["si_fp"])
    a3_assert("delta events caught (+678)", mw13["delta_events_caught"] == V3_MW["delta_events"],
              mw13["delta_events_caught"], V3_MW["delta_events"])
    a3_assert("delta false positives (-684)", mw13["delta_false_positives"] == V3_MW["delta_fp"],
              mw13["delta_false_positives"], V3_MW["delta_fp"])
    t5d = pd.read_csv(os.path.join(V3T, "table5d_matched_workload.csv"))
    r0 = t5d[(t5d.Stratum == "overall") & (t5d.Workload_target_label == "deployed_rule_in")
             & (t5d.Instrument == "shock_index")].iloc[0]
    a3_assert("SI cutoff equals table5d (0.946309)", abs(mw13["si_cutoff"] - float(r0["Cutoff"])) < 1e-9,
              mw13["si_cutoff"], float(r0["Cutoff"]))
    a3.update({"status": "PASS", "auroc_14_criteria": auc14, "auroc_13_criteria": auc13,
               "strata_13": st13, "strata_14": st14, "matched_workload_13": mw13,
               "matched_workload_14": mw14,
               "note": "cc_anticoag carries 0 points, so removing it from the card cannot change any "
                       "patient's score; asserted element-wise on all 79,912 test rows."})
    jdump(a3, os.path.join(OUT, "a3_invariance_check.json"))

    # ======================================================================
    # A1: correlation + VIF for the published criteria (derivation set)
    # ======================================================================
    banner("A1  correlation matrix and VIF, published criteria, derivation (train) set")
    all15 = list(refit["variables_kept"])
    pt14 = list(pub_point_vars)
    pt13 = [v for v in pt14 if v != "cc_anticoag"]
    rows = []
    vif_sets = {"fit15_as_in_v3_step05 (14 point + override)": all15,
                "point14 (card incl. anticoagulant, no override)": pt14,
                "point13 (published card criteria, no anticoagulant)": pt13,
                "fit14 (13 point + override)": pt13 + ["sbp_unrecordable"]}
    vif_summary = {}
    for name, cols in vif_sets.items():
        v = compute_vif(tr[cols].astype(float))
        mx = max(v, key=v.get)
        vif_summary[name] = {"max_vif_var": mx, "max_vif": v[mx], "vif": v}
        for k, q in v.items():
            rows.append({"section": "vif", "set": name, "var1": k, "var2": "", "value": q})
        log(f"  {name}: max VIF {v[mx]:.4f} ({mx})")
    assert abs(vif_summary[list(vif_sets)[0]]["max_vif"] - 3.59) < 0.005, "VIF 3.59 not reproduced"
    corr_cols = pt14 + ["sbp_unrecordable"]
    cm = tr[corr_cols].astype(float).corr()
    for i, a in enumerate(corr_cols):
        for j, b in enumerate(corr_cols):
            rows.append({"section": "corr_pearson_phi", "set": "train_point14_plus_override",
                         "var1": a, "var2": b, "value": float(cm.loc[a, b])})
    extra = {}
    for a, b in [("si_ge0_9", "si_ge1_0"), ("sbp_lt90", "sbp_lt70"), ("gcs_motor_le5", "gcs_le8"),
                 ("sbp_lt90", "si_ge1_0"), ("sbp_lt90", "si_ge0_9"), ("hr_ge120", "si_ge0_9")]:
        extra[f"{a}|{b}"] = float(tr[a].astype(float).corr(tr[b].astype(float)))
        log(f"  corr({a}, {b}) = {extra[f'{a}|{b}']:.4f}")
    corr_where = {}
    for nm, frm in [("train", tr), ("test", te), ("whole_primary_cohort", load_candidates())]:
        corr_where[nm] = {"si_ge0_9|si_ge1_0": float(frm["si_ge0_9"].astype(float).corr(frm["si_ge1_0"].astype(float))),
                          "sbp_lt90|sbp_lt70": float(frm["sbp_lt90"].astype(float).corr(frm["sbp_lt70"].astype(float)))}
        for k, q in corr_where[nm].items():
            rows.append({"section": "corr_key_pairs_by_dataset", "set": nm, "var1": k.split("|")[0],
                         "var2": k.split("|")[1], "value": q})
        log(f"  [{nm}] corr(si_ge0_9, si_ge1_0)={corr_where[nm]['si_ge0_9|si_ge1_0']:.4f}  "
            f"corr(sbp_lt90, sbp_lt70)={corr_where[nm]['sbp_lt90|sbp_lt70']:.4f}")
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(os.path.join(OUT, "a1_correlation_vif.csv"), index=False)
    log(f"  wrote {os.path.join(OUT, 'a1_correlation_vif.csv')}")

    # ======================================================================
    # A1: 05 replication check (published selected set -> published betas/points)
    # ======================================================================
    banner("A1  replication check: v3 step-05 logic on the v3 stability-selected set")
    v3_stab = pd.read_csv(os.path.join(V3T, "table2_lasso_stability.csv"))
    v3_freq = dict(zip(v3_stab.Variable, v3_stab.Selection_Frequency))
    v3_sel = pd.read_csv(os.path.join(V3T, "table2_lasso_selected.csv")).Variable.tolist()
    rep = refit_integer(tr, v3_sel, v3_freq, verbose=True)
    max_db = max(abs(rep["beta"][v] - refit["coefficients"][v]) for v in refit["coefficients"])
    rep_ok = (rep["points"] == pub_points and max_db < 1e-6
              and sorted(rep["variables"]) == sorted(refit["variables_kept"]))
    log(f"  max |beta - refit.json| = {max_db:.2e}; points identical = {rep['points'] == pub_points}")
    assert rep_ok, "replication of v3 step 05 failed"
    log("  [OK] step-05 logic reproduces refit.json betas and the published integer card")

    # ======================================================================
    # A1: stability replication check and the sensitivity refit
    # ======================================================================
    banner("A1  stability-selection inputs")
    sa1 = pd.read_csv(os.path.join(OUT, "a1_stability_a1.csv"))
    sp17 = pd.read_csv(os.path.join(OUT, "a1_stability_parity17.csv"))
    track(os.path.join(OUT, "a1_stability_a1.csv"))
    track(os.path.join(OUT, "a1_stability_parity17.csv"))
    cmp = sp17.merge(v3_stab[["Variable", "Selection_Frequency"]], on="Variable", suffixes=("_v4rerun", "_v3"))
    cmp["abs_diff"] = (cmp.Selection_Frequency_v4rerun - cmp.Selection_Frequency_v3).abs()
    same_sel = (set(sp17[sp17.Selection_Frequency >= STABILITY_THRESHOLD].Variable)
                == set(v3_stab[v3_stab.Selection_Frequency >= STABILITY_THRESHOLD].Variable))
    parity_info = {"max_abs_freq_diff": float(cmp.abs_diff.max()),
                   "same_selected_set": bool(same_sel),
                   "table": cmp.to_dict(orient="records")}
    log(f"  17-candidate rerun vs v3 table2: max |freq diff| = {cmp.abs_diff.max():.4f}; "
        f"same >=0.85 set = {same_sel}")
    log(cmp.to_string(index=False))

    sel_a1 = sa1[sa1.Selection_Frequency >= STABILITY_THRESHOLD].Variable.tolist()
    log(f"  A1 stability-selected (>= {STABILITY_THRESHOLD}): {sel_a1}")
    log(sa1.to_string(index=False))
    freq_a1 = dict(zip(sa1.Variable, sa1.Selection_Frequency))
    banner("A1  unpenalized refit + integer card (domain-collapsed candidate pool)")
    fa1 = refit_integer(tr, sel_a1, freq_a1, verbose=True)

    # --- scoring on the held-out test set
    s_a1 = score_with(te, fa1)
    x_te = te[fa1["variables"]].to_numpy(float)
    lp_a1 = fa1["intercept"] + x_te @ np.array([fa1["beta"][v] for v in fa1["variables"]])
    auc_a1 = float(roc_auc_score(y_te, s_a1))
    auc_a1_lp = float(roc_auc_score(y_te, lp_a1))
    auc_pub_lp = float(roc_auc_score(y_te, te["pelvic_lp"].to_numpy(float)))
    log(f"  TEST AUROC  A1 integer={auc_a1:.4f}  A1 logit={auc_a1_lp:.4f} | "
        f"published integer={auc13:.4f}  published logit={auc_pub_lp:.4f}")

    log(f"  paired bootstrap ({N_BOOT_AUC}x, seed {SEED}) ...")
    bres = boot_auc_paired(y_te, {"a1": s_a1, "pub": score_pub, "a1_lp": lp_a1}, N_BOOT_AUC)
    d_int = bres["a1"] - bres["pub"]
    dl = delong_test(y_te, s_a1, score_pub)
    delta = auc_a1 - auc13
    log(f"  delta integer AUROC (A1 - published) = {delta:+.4f}  paired-boot 95% CI "
        f"{pct_ci(d_int)}  DeLong p={dl['p']:.3g}")
    material = bool((-delta) > 0.005)
    verdict = ("MATERIAL LOSS: A1 integer AUROC is lower than the published card by more than 0.005"
               if material else
               "REDUNDANCY NOT DRIVING PERFORMANCE: A1 integer AUROC differs from the published "
               "card by no more than 0.005 (loss threshold)")
    log(f"  VERDICT: {verdict}")
    if material:
        log("  !!!!!!!! A1 SHOWS A MATERIAL AUROC LOSS (> 0.005) -- FLAG LOUDLY IN THE REPORT !!!!!!!!")

    # --- strata
    st_a1 = strata_table(s_a1, y_te)
    # A1 own score distribution
    dist = (pd.DataFrame({"score": s_a1, "y": y_te}).groupby("score")["y"]
            .agg(n="size", events="sum").reset_index())
    dist["rate"] = dist.events / dist.n
    # --- matched workload at the published workload (15.6237 percent) and A1's own best integer cutoff
    mw_a1 = matched_workload(s_a1, si_all, rs_all, y_te, anchor_frac=mw14["score_pct_flagged"] / 100.0)
    log(f"  A1 matched workload: cutoff >= {mw_a1['score_cutoff']:+.0f} flags "
        f"{mw_a1['score_pct_flagged']:.4f}% (published {mw14['score_pct_flagged']:.4f}%): "
        f"TP {mw_a1['score']['events_caught_TP']} FP {mw_a1['score']['false_positives_FP']} | "
        f"SI >= {mw_a1['si_cutoff']:.4f} flags {mw_a1['si_pct_flagged']:.4f}%: "
        f"TP {mw_a1['si']['events_caught_TP']} FP {mw_a1['si']['false_positives_FP']} | "
        f"delta events {mw_a1['delta_events_caught']:+d} FP {mw_a1['delta_false_positives']:+d} | "
        f"within tolerance {mw_a1['within_tolerance']}")
    # A1 with the same numeric cutoff (>= +5) as the published escalate threshold, for reference
    mw_a1_c5 = matched_workload(s_a1, si_all, rs_all, y_te, anchor_cut=5.0)

    # --- A1b (secondary, like-for-like scale): the same A1 model, points scaled with the
    # PUBLISHED divisor (0.3407, age_65_74) instead of the divisor A1 selects for itself.
    pub_anchor = float(refit["anchor_min_positive_beta"])
    pts_b = {v: integer_points(fa1["beta"][v], pub_anchor) for v in fa1["point_vars"]}
    s_b = te[fa1["point_vars"]].to_numpy(float) @ np.array([pts_b[v] for v in fa1["point_vars"]], float)
    auc_b = float(roc_auc_score(y_te, s_b))
    bres_b = boot_auc_paired(y_te, {"b": s_b, "pub": score_pub}, N_BOOT_AUC)
    d_b = bres_b["b"] - bres_b["pub"]
    mw_b = matched_workload(s_b, si_all, rs_all, y_te, anchor_frac=mw14["score_pct_flagged"] / 100.0)
    a1b = {"note": "SECONDARY: A1 model re-scaled with the published divisor so the card scale is "
                   "like-for-like with the published card; not the rule-faithful A1 card.",
           "divisor_beta": pub_anchor, "points": pts_b, "auroc_integer": auc_b,
           "auroc_integer_ci": pct_ci(bres_b["b"]), "delta_vs_published": auc_b - auc13,
           "delta_paired_bootstrap_ci": pct_ci(d_b), "delta_delong": delong_test(y_te, s_b, score_pub),
           "strata_same_cutoffs": strata_table(s_b, y_te), "matched_workload": mw_b,
           "score_min": float(s_b.min()), "score_max": float(s_b.max())}
    log(f"  A1b (published divisor {pub_anchor:.4f}): points {pts_b}")
    log(f"  A1b TEST AUROC integer={auc_b:.4f}  delta vs published {auc_b-auc13:+.4f} "
        f"{pct_ci(d_b)}  matched workload: score >= {mw_b['score_cutoff']:+.0f} flags "
        f"{mw_b['score_pct_flagged']:.4f}% TP {mw_b['score']['events_caught_TP']} FP "
        f"{mw_b['score']['false_positives_FP']} vs SI TP {mw_b['si']['events_caught_TP']} FP "
        f"{mw_b['si']['false_positives_FP']}")

    # --- temporal (re-estimation, as v3/code/12)
    banner("A1  temporal check (re-estimate on 2019-2022, validate on 2023-2024; mirrors v3 step 12)")
    t_pub = run_temporal(td, tv, list(refit["variables_kept"]), ["sbp_unrecordable"], N_BOOT_AUC,
                         "published variable set (parity check)")
    tt = pd.read_csv(os.path.join(V3T, "table4b_temporal.csv")).iloc[0]
    par_ok = (round(t_pub["auroc_logit"], 4) == round(float(tt["AUROC"]), 4)
              and round(t_pub["auroc_integer"], 4) == round(float(tt["AUROC_integer"]), 4))
    log(f"  parity vs table4b_temporal.csv (logit {tt['AUROC']}, integer {tt['AUROC_integer']}): "
        f"{'PASS' if par_ok else 'DIFFERS'}")
    t_a1 = run_temporal(td, tv, list(fa1["variables"]), list(fa1["override_vars"]), N_BOOT_AUC,
                        "A1 variable set")

    # --- A1 card csv
    pub_by_var = {r: v for r, v in pub_points.items()}
    card_rows = []
    for r in fa1["rows"]:
        card_rows.append({
            "variable": r["Variable"], "role": r["Role"], "beta": round(r["beta"], 4),
            "SE": round(r["SE"], 4), "OR": round(r["OR"], 3),
            "OR_CI_low": round(r["OR_CI_low"], 3), "OR_CI_high": round(r["OR_CI_high"], 3),
            "a1_points": r["Points"],
            "a1b_points_published_divisor": pts_b.get(r["Variable"], ""),
            "published_points": pub_by_var.get(r["Variable"], ""),
            "label": LABEL})
    for v in LASSO_X_COLS_V3:
        if v not in fa1["variables"]:
            why = ("dropped by design (duplicate threshold tier)" if v in A1_DROPPED_TIERS
                   else ("dropped at |beta| screen" if v in fa1["dropped_beta"]
                         else ("not stability-selected" if v not in sel_a1 else "dropped by VIF")))
            card_rows.append({"variable": v, "role": "not in model", "beta": "", "SE": "", "OR": "",
                              "OR_CI_low": "", "OR_CI_high": "", "a1_points": "",
                              "a1b_points_published_divisor": "",
                              "published_points": pub_by_var.get(v, ""), "label": why})
    pd.DataFrame(card_rows).to_csv(os.path.join(OUT, "a1_sensitivity_card.csv"), index=False)
    log(f"  wrote {os.path.join(OUT, 'a1_sensitivity_card.csv')}")
    dist.to_csv(os.path.join(OUT, "a1_score_distribution_test.csv"), index=False)

    a1_metrics = {
        "label": LABEL,
        "design": {
            "candidate_pool_v3": LASSO_X_COLS_V3, "candidate_pool_a1": A1_COLS,
            "dropped_duplicate_tiers": A1_DROPPED_TIERS,
            "kept_one_term_per_domain": ["sbp_lt90", "si_ge1_0", "hr_ge120", "gcs_motor_le5",
                                         "age_65_74", "age_ge75", "sex_female", "mech_high_energy",
                                         "mech_mvc", "mech_fall"],
            "other_v3_candidates_kept": ["sbp_unrecordable", "air_transport", "cc_anticoag", "cc_bleeding"],
            "stability": {"bootstraps": N_BOOT_STAB, "threshold": STABILITY_THRESHOLD,
                          "lambda_rule": "1se", "seed_rule": "RandomState(42+i)", "cv": "5-fold, 50 Cs, saga, roc_auc"},
            "force_in_anchors": FORCE_IN_ANCHORS,
            "cc_anticoag_forced_in": True,
            "override": OVERRIDE_VARS,
            "integer_rule": "round(beta/min positive beta passing |beta|>=0.15) with +/-1 floor (D-11)",
            "train_n": fa1["n_train"], "train_events": fa1["events_train"],
            "test_n": int(len(te)), "test_events": int(y_te.sum()),
        },
        "stability_a1": sa1.to_dict(orient="records"),
        "stability_selected": sel_a1,
        "stability_parity_check_17_candidates": parity_info,
        "step05_replication_check": {"pass": bool(rep_ok), "max_abs_beta_diff": float(max_db)},
        "fit": {k: fa1[k] for k in ["variables", "point_vars", "override_vars", "intercept", "beta",
                                    "points", "anchor", "anchor_var", "below_screen", "dropped_beta",
                                    "dropped_vif", "vif_trace", "train_auroc_integer", "train_auroc_logit"]},
        "n_point_criteria_a1": len(fa1["point_vars"]),
        "n_nonzero_point_criteria_a1": int(sum(1 for v in fa1["points"].values() if v != 0)),
        "test": {
            "auroc_integer_a1": auc_a1, "auroc_integer_a1_ci": pct_ci(bres["a1"]),
            "auroc_logit_a1": auc_a1_lp, "auroc_logit_a1_ci": pct_ci(bres["a1_lp"]),
            "auroc_integer_published": auc13, "auroc_integer_published_ci": pct_ci(bres["pub"]),
            "auroc_logit_published": auc_pub_lp,
            "delta_integer_a1_minus_published": delta,
            "delta_paired_bootstrap_ci": pct_ci(d_int), "n_bootstrap": N_BOOT_AUC, "bootstrap_seed": SEED,
            "delta_delong": dl, "delta_logit_a1_minus_published": auc_a1_lp - auc_pub_lp,
            "loss_threshold": 0.005, "material_loss": material, "verdict": verdict,
            "strata_a1_same_cutoffs": st_a1, "strata_published": st14,
            "a1_score_min": float(s_a1.min()), "a1_score_max": float(s_a1.max()),
        },
        "matched_workload": {
            "a1_matched_to_published_workload": mw_a1,
            "a1_at_same_numeric_cutoff_ge5": mw_a1_c5,
            "published_vs_shock_index": mw14,
            "note": "Shock index matched to the achieved flagged fraction of the score (v3 step 08 rule, "
                    "common complete-case set where SI and rSIG are observed). A1 uses the integer cutoff "
                    "whose flagged fraction is closest to the published 15.6237 percent.",
        },
        "a1b_secondary_published_divisor": a1b,
        "temporal": {"published_variable_set": t_pub, "a1_variable_set": t_a1,
                     "parity_with_table4b": bool(par_ok),
                     "delta_integer": t_a1["auroc_integer"] - t_pub["auroc_integer"],
                     "delta_logit": t_a1["auroc_logit"] - t_pub["auroc_logit"],
                     "note": "Temporal validation in v3 step 12 RE-ESTIMATES coefficients on 2019-2022; "
                             "it does not apply the fixed random-split card."},
        "collinearity": {"vif": vif_summary, "pairwise_corr_train": extra, "key_pairs_by_dataset": corr_where},
    }
    jdump(a1_metrics, os.path.join(OUT, "a1_sensitivity_metrics.json"))

    # ======================================================================
    # A2: hypotension by mechanism (test set)
    # ======================================================================
    banner("A2  hypotension by mechanism (test set, n = 79,912)")
    d = pd.DataFrame({
        "score": score_pub, "y": y_te, "nslo": ov.astype(int),
        "sbp_lt90": te["sbp_lt90"].to_numpy(int), "si_ge1_0": te["si_ge1_0"].to_numpy(int),
        "fall": te["mech_fall"].to_numpy(int), "female": te["sex_female"].to_numpy(int),
        "age_ge75": te["age_ge75"].to_numpy(int), "age": te["age"].to_numpy(float)})
    log(f"  overlap: sbp_lt90==1 & nslo==1 : {int(((d.sbp_lt90==1)&(d.nslo==1)).sum())}; "
        f"nslo n={int(d.nslo.sum())}")
    hypo = (d.sbp_lt90 == 1) & (d.nslo == 0)
    rows = []

    def add(group, label, mask, note=""):
        m = np.asarray(mask, bool)
        n, e = int(m.sum()), int(d.y[m].sum())
        lo, hi = wilson(e, n)
        rows.append({"group": group, "subgroup": label, "n": n, "events": e,
                     "rate_pct": 100 * e / n if n else float("nan"),
                     "ci_low_pct": 100 * lo, "ci_high_pct": 100 * hi, "note": note, "label": LABEL})
        log(f"  {group:<34} {label:<52} n={n:>6,} ev={e:>5,} rate={100*e/n if n else float('nan'):6.2f}% "
            f"[{100*lo:.1f}, {100*hi:.1f}]")

    ref_profile = (d.female == 1) & (d.age_ge75 == 1) & (d.fall == 1) & (d.sbp_lt90 == 1) & (d.si_ge1_0 == 1)
    add("i_reference_profile", "woman >=75, fall, SBP<90, SI>=1.0 (all, incl. no-signs-of-life)", ref_profile)
    add("i_reference_profile", "same, excluding no-signs-of-life", ref_profile & (d.nslo == 0))
    add("i_reference_profile", "same, excluding NSLO, by score <=4 (Observe or lower)", ref_profile & (d.nslo == 0) & (d.score <= 4))
    add("ii_fall_hypotension", "fall with SBP<90 (excl. no-signs-of-life)", hypo & (d.fall == 1))
    add("ii_fall_hypotension", "fall with SBP<90, age >=75 (excl. NSLO)", hypo & (d.fall == 1) & (d.age_ge75 == 1))
    add("ii_fall_hypotension", "fall with SBP<90, woman (excl. NSLO)", hypo & (d.fall == 1) & (d.female == 1))
    add("ii_fall_hypotension", "fall with SBP<90, score <=4 (excl. NSLO)", hypo & (d.fall == 1) & (d.score <= 4))
    add("ii_fall_hypotension", "non-fall with SBP<90 (excl. NSLO)", hypo & (d.fall == 0))
    add("iii_sbp_lt90_by_stratum", "all SBP<90 (excl. NSLO)", hypo)
    add("iii_sbp_lt90_by_stratum", "SBP<90, score <=0 (rule-out)", hypo & (d.score <= 0))
    add("iii_sbp_lt90_by_stratum", "SBP<90, score 1 to 4 (observe)", hypo & (d.score >= 1) & (d.score <= 4))
    add("iii_sbp_lt90_by_stratum", "SBP<90, score >=5 (escalate)", hypo & (d.score >= 5))
    add("iii_sbp_lt90_by_stratum", "SBP<90, score <=4 (residual: below escalate)", hypo & (d.score <= 4),
        "plan figure: 774 patients, 147 events")
    add("iv_fall_overall_test", "fall stratum, whole test set", d.fall == 1)
    add("iv_fall_overall_test", "all test patients (reference)", np.ones(len(d), bool))
    add("iv_fall_overall_test", "Observe stratum (1 to 4), all patients (reference)", (d.score >= 1) & (d.score <= 4))
    a2 = pd.DataFrame(rows)

    # whole-cohort fall figure from Table 1 / results bundle (no patient rows)
    bundle = json.load(open(os.path.join(config.DERIVED, "results_bundle.json")))["quantities"]
    f_all = int(bundle["cohort.table1.mech_fall.overall"]["value"].split(" ")[0].replace(",", ""))
    f_ev = int(bundle["cohort.table1.mech_fall.event"]["value"].split(" ")[0].replace(",", ""))
    f_no = int(bundle["cohort.table1.mech_fall.noevent"]["value"].split(" ")[0].replace(",", ""))
    assert f_ev + f_no == f_all
    lo, hi = wilson(f_ev, f_all)
    t1 = pd.read_csv(os.path.join(config.TABLES, "Table 1.csv"), dtype=str)
    t1_fall = t1[t1.apply(lambda r: r.astype(str).str.contains("1,870 \\(7.1%\\)").any(), axis=1)]
    assert len(t1_fall) == 1, "Table 1 fall row not found"
    a2 = pd.concat([a2, pd.DataFrame([{
        "group": "iv_fall_whole_cohort_table1", "subgroup": "fall, whole analytic cohort (Table 1 / results bundle)",
        "n": f_all, "events": f_ev, "rate_pct": 100 * f_ev / f_all, "ci_low_pct": 100 * lo,
        "ci_high_pct": 100 * hi,
        "note": "from bundle cohort.table1.mech_fall.overall and .event; N=239,736 cohort", "label": LABEL}])],
        ignore_index=True)
    log(f"  whole cohort falls (Table 1): {f_ev:,} events of {f_all:,} = {100*f_ev/f_all:.2f}% "
        f"[{100*lo:.2f}, {100*hi:.2f}]")
    a2.to_csv(os.path.join(OUT, "a2_hypotension_mechanism.csv"), index=False)
    log(f"  wrote {os.path.join(OUT, 'a2_hypotension_mechanism.csv')}")

    # plan-figure reconciliation
    g = a2.set_index("subgroup")
    ref_row = g.loc["woman >=75, fall, SBP<90, SI>=1.0 (all, incl. no-signs-of-life)"]
    resid_row = g.loc["SBP<90, score <=4 (residual: below escalate)"]
    recon = {"reference_profile_n": int(ref_row.n), "reference_profile_events": int(ref_row.events),
             "plan_reference": [94, 3],
             "reference_excl_nslo": [int(g.loc["same, excluding no-signs-of-life"].n),
                                 int(g.loc["same, excluding no-signs-of-life"].events)],
             "residual_n": int(resid_row.n), "residual_events": int(resid_row.events),
             "residual_rate_pct": float(resid_row.rate_pct), "plan_residual": [774, 147],
             "reference_matches_plan": bool(int(ref_row.n) == 94 and int(ref_row.events) == 3),
             "residual_matches_plan": bool(int(resid_row.n) == 774 and int(resid_row.events) == 147)}
    log(f"  plan reconciliation: {recon}")

    # worked example from the real card
    banner("A2  worked example (78-year-old woman, fall, SBP 80, HR 95)")
    age, female, mech, sbp, hr, gcs_motor, gcs_tot = 78, 1, "Fall", 80, 95, 6, 15
    si = hr / sbp
    feat = {"age_65_74": int(65 <= age < 75), "age_ge75": int(age >= 75), "sex_female": female,
            "sbp_lt90": int(sbp < 90), "sbp_lt70": int(sbp < 70), "hr_ge120": int(hr >= 120),
            "si_ge0_9": int(si >= 0.9), "si_ge1_0": int(si >= 1.0), "gcs_motor_le5": int(gcs_motor <= 5),
            "gcs_le8": int(gcs_tot <= 8), "mech_high_energy": int(mech in ("MVC-occupant", "Motorcycle", "Pedestrian")),
            "mech_mvc": int(mech == "MVC-occupant"), "mech_fall": int(mech == "Fall"), "cc_anticoag": 0}
    contrib = {v: pub_points[v] * feat[v] for v in pub_point_vars if pub_points[v] * feat[v] != 0}
    total = sum(pub_points[v] * feat[v] for v in pub_point_vars)
    stratum = ("Rule out" if total <= 0 else ("Observe" if total <= 4 else "Escalate"))
    log(f"  SI = {hr}/{sbp} = {si:.4f}; contributions {contrib}; total {total:+d}; stratum {stratum}")
    assert total == 2 and stratum == "Observe"
    assert contrib == {"age_ge75": 1, "sex_female": -1, "mech_fall": -5, "sbp_lt90": 3, "si_ge0_9": 2, "si_ge1_0": 2}
    assert card["risk_tier_definitions"]["observe"] == "1.0 <= pelvic_score < 5.0"
    assert feat["sbp_lt70"] == 0
    # same sum without the anticoagulant term
    assert sum(pub_points[v] * feat[v] for v in pt13) == total
    jdump({"label": LABEL,
           "patient": {"age": age, "sex": "female", "mechanism": "fall", "sbp": sbp, "hr": hr,
                       "shock_index": round(si, 4), "gcs_motor": gcs_motor, "gcs_total": gcs_tot},
           "contributions": contrib,
           "contribution_order_as_in_manuscript": ["age >=75: +1", "female: -1", "fall: -5",
                                                  "SBP<90: +3", "SI>=0.9: +2", "SI>=1.0: +2"],
           "total_points": total, "stratum": stratum,
           "stratum_definition": card["risk_tier_definitions"]["observe"],
           "sbp80_triggers_sbp_lt70_tier": bool(feat["sbp_lt70"]),
           "total_without_anticoagulant_row": int(sum(pub_points[v] * feat[v] for v in pt13)),
           "asserts_passed": True,
           "plan_reconciliation": recon,
           "note": "Hand computation confirmed against the real card in refit.json; "
                   "age 78 enters the >=75 band (+1), not the 65-74 band."},
          os.path.join(OUT, "a2_worked_example.json"))

    # ======================================================================
    # manifest
    # ======================================================================
    banner("Manifest")
    outputs = {}
    for fn in sorted(os.listdir(OUT)):
        pth = os.path.join(OUT, fn)
        if (os.path.isfile(pth) and fn != "v4_analyses_manifest.json"
                and fn.startswith(("a1_", "a2_", "a3_"))):
            outputs[fn] = sha256_file(pth)
    jdump({"label": LABEL, "script": os.path.relpath(SCRIPT, ROOT), "script_sha256": sha256_file(SCRIPT),
           "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "runtime_min_finish_stage": round((time.time() - t_start) / 60, 1),
           "seed": SEED, "bootstraps_stability": N_BOOT_STAB, "bootstraps_auc": N_BOOT_AUC,
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__},
           "inputs_sha256": inputs, "outputs_sha256": outputs,
           "a3_status": a3["status"], "a1_verdict": verdict,
           "notes": ["v3 tree read-only; only aggregates written; no patient-level rows in any output."]},
          os.path.join(OUT, "v4_analyses_manifest.json"))
    log("\nDONE.")
    _LOG.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("stability")
    s1.add_argument("--tag", choices=["a1", "parity17"], required=True)
    s1.add_argument("--bootstraps", type=int, default=N_BOOT_STAB)
    s1.add_argument("--sample", type=int, default=0)
    sub.add_parser("finish")
    args = ap.parse_args()
    if args.stage == "stability":
        return stage_stability(args.tag, args.bootstraps, args.sample)
    return stage_finish(args)


if __name__ == "__main__":
    sys.exit(main())
