"""
Pelvic Arrival Tool v3 — Step 09: Pre-specified subgroup AUROCs (DATA only)
================================================================================

Forked from PAT_TBI/v4/code/09_subgroups.py. Per CONTRACTS §8 the shared
render_figures.py owns figure F6 (subgroup forest), so the INLINE forest
rendering is DISABLED — this script emits only DATA artifacts.

Pelvic strata (RESPECIFIED — CONTRACTS §9-F, SAP §5/§6):
  1. Age            <65 / >=65                 (age_ge65)
  2. Sex            Male / Female              (sex_male)
  3. Ring stability Stable / Unstable          (unstable_ring)
  4. Anticoagulation No / Yes                  (cc_anticoag)
  5. Mechanism      by mechanism_cat levels    (mechanism_cat)
  6. ACS level      by acs_level levels        (acs_level)

Scores (risk-oriented, higher = worse): pelvic_score (+1), SI-alone (+1),
rSIG (-1, negated). Per cell: AUROC + bootstrap CI for each score, and the
DeLong delta of pelvic_score vs each comparator.

v3 OUTCOME SWEEP (CONTRACTS §9-J) — read before touching ``OUTCOME_COL``
------------------------------------------------------------------------
``major_hemorrhage_event`` KEEPS its name but CHANGES MEANING in v3: it is now
a composite of hemorrhage control, HEMORRHAGIC death, and substantial
transfusion.

**The definition string is NOT typed into this file.** It has been wrong twice
— first as ``death_inhosp``, then as a calendar-day death rule
(``finaldischarge_days <= 2``) that was itself superseded by an hours rule.
``01_build_cohort.py`` writes the authoritative definitions into
``logs/qc_manifest.json`` precisely so consumers can read them; this script
does exactly that (``outcome_definition()`` below) and HARD-FAILS if the
manifest or the keys are absent, rather than printing a literal that may have
gone stale. ``14_consort.py`` and ``render_figures.py`` already read the same
keys. Realized counts and rates likewise live in that manifest and are not
duplicated here (CONTRACTS §3.1: "No script may hard-code them").

``death_inhosp`` is v3's all-cause SENSITIVITY arm ONLY (CONTRACTS §4 rung 0)
and must never be scored as an outcome here.

This script discriminates ONE outcome — the primary composite — because
CONTRACTS §7/§8 give it exactly two artifacts (``table_subgroups.csv`` and the
F6 forest fig-data), whose schema the renderer depends on. The mortality-only
anchor (``death_hemorrhagic``) is exercised by `11` (threshold splines) and
`13` (sensitivity arms), NOT here. Do not add a second outcome to this script
without a CONTRACTS §8 amendment — it would change the F6 fig-data schema.

v3 FIX-3 BOUNDARY (CONTRACTS §6, §13 G3f)
-----------------------------------------
``unstable_ring`` is removed from the SCORE (files 02/04/05). It REMAINS a
pre-specified SUBGROUP stratum here — a subgroup is a description of where the
score works, not a claim that the variable is available at the door. If the
scored parquet fails to carry it, this script back-fills it from
``candidates.parquet`` (CONTRACTS §7 declares both as this script's inputs)
and hard-fails if it is unavailable from either.

Inputs
------
  v3/derived/cohort_test_scored.parquet   (from 06)
  v3/derived/candidates.parquet           (back-fill source for subgroup cols)
  v3/logs/qc_manifest.json                (from 01 — REQUIRED, outcome definition)

Outputs
-------
  v3/tables/table_subgroups.csv           (+ Outcome, Outcome_Definition columns
                                           so the table footer never retypes the
                                           definition)
  v3/derived/subgroup_forest_data.csv     (fig-data F6 — schema UNCHANGED)

Run
---
    python 09_subgroups.py
    python 09_subgroups.py --sample 50000    # smoke

Seed: 42
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
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
os.makedirs(TABLES, exist_ok=True)

SCORED_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")
CANDIDATES = os.path.join(DERIVED, "candidates.parquet")   # back-fill source (CONTRACTS §7)
QC_MANIFEST = os.path.join(LOGS, "qc_manifest.json")       # from 01 — REQUIRED
OUT_TABLE = os.path.join(TABLES, "table_subgroups.csv")
OUT_FOREST = os.path.join(DERIVED, "subgroup_forest_data.csv")

MIN_EVENTS = 50
N_BOOT = 500
MAX_BOOT_N = 50_000

# ---- v3 outcome (CONTRACTS §4 / §9-J). Same NAME as v2, new DEFINITION. -----
OUTCOME_COL = "major_hemorrhage_event"
# v2's all-cause death arm. In v3 this is a NAMED SENSITIVITY (CONTRACTS §4
# rung 0) owned by `13`. It is NOT an outcome in this script; the guard below
# fails loudly if anyone repoints OUTCOME_COL at it.
DEATH_SENSITIVITY_ONLY = "death_inhosp"

# ---------------------------------------------------------------------------
# Authoritative outcome definitions — READ, never typed (CONTRACTS §3.1, §4).
#
# `01_build_cohort.py` writes these strings into logs/qc_manifest.json. They
# have changed twice already (death_inhosp -> calendar-day rule -> hours rule),
# and every literal copy went stale. Each entry lists the manifest key paths in
# preference order; the first path that resolves to a non-empty string wins.
# If NONE resolves, this script HARD-FAILS — a missing definition is never
# silently replaced by a default.
# ---------------------------------------------------------------------------
MANIFEST_DEF_PATHS = {
    "composite": (("primary_composite", "definition"),
                  ("gates", "primary_outcome_definition")),
    "death_hemorrhagic": (("death_variants", "death_hemorrhagic_definition"),
                          ("gates", "mortality_only_anchor_definition")),
}
_DEF_CACHE: dict[str, str] = {}


def _dig(obj, path):
    """Follow a key path through nested dicts; return a non-empty str or None."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, str) and cur.strip():
        return cur.strip()
    return None


def _load_manifest():
    """Load logs/qc_manifest.json or die. No fallback, no default."""
    if not os.path.exists(QC_MANIFEST):
        raise SystemExit(
            f"FATAL — {QC_MANIFEST} not found.\n"
            f"  01_build_cohort.py owns it (CONTRACTS §7 row 01) and it is the "
            f"ONLY authoritative source of the v3 outcome definitions.\n"
            f"  This script refuses to print or emit a hard-coded definition "
            f"string: the literal has gone stale twice (death_inhosp -> "
            f"calendar-day rule -> hours rule). Re-run 01_build_cohort.py.")
    try:
        with open(QC_MANIFEST, encoding="utf-8") as fh:
            man = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"FATAL — {QC_MANIFEST} could not be read/parsed: {exc}\n"
            f"  Re-run 01_build_cohort.py.") from exc
    if not isinstance(man, dict):
        raise SystemExit(f"FATAL — {QC_MANIFEST} is not a JSON object.")
    return man


def outcome_definition(which="composite"):
    """The AUTHORITATIVE definition string for `which`, read from the manifest.

    Raises SystemExit (loudly, naming every key path tried) if the manifest is
    absent, unreadable, or carries none of the expected keys.
    """
    if which in _DEF_CACHE:
        return _DEF_CACHE[which]
    paths = MANIFEST_DEF_PATHS[which]
    man = _load_manifest()
    for path in paths:
        got = _dig(man, path)
        if got is not None:
            _DEF_CACHE[which] = got
            print(f"  [manifest] {which} definition <- qc_manifest.json"
                  f"{''.join('.' + p for p in path)}: {got}")
            return got
    tried = "; ".join(".".join(p) for p in paths)
    raise SystemExit(
        f"FATAL — no '{which}' outcome definition in {QC_MANIFEST}.\n"
        f"  Key paths tried: {tried}\n"
        f"  01_build_cohort.py must write it (CONTRACTS §4). This script will "
        f"NOT substitute a hard-coded literal — that is exactly the defect this "
        f"read replaces.")


def outcome_definition_full():
    """Composite definition, with the death component expanded — one string."""
    return (f"{outcome_definition('composite')} "
            f"(death_hemorrhagic = {outcome_definition('death_hemorrhagic')})")

JOIN_KEYS = ["puf_year", "inc_key"]
# Strata this script MUST be able to build (CONTRACTS §9-F). `unstable_ring` is
# on this list deliberately — Fix 3 removes it from the SCORE, not from the
# subgroup grid (CONTRACTS §6 asymmetry, §13 G3f).
REQUIRED_SUBGROUP_COLS = ["age_ge65", "sex_male", "unstable_ring", "cc_anticoag",
                          "mechanism_cat", "acs_level"]

# (short, pretty, col, sign)  — higher(sign*score) == worse
SCORES = [
    ("PELVIC", "pelvic_score", "pelvic_score", +1),
    ("SI",     "SI-alone",     "shock_index",  +1),
    ("RSIG",   "rSIG",         "rsig",         -1),
]
COMPARATORS = [s for s in SCORES if s[0] != "PELVIC"]
PELVIC = SCORES[0]


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ---------------- Vectorized DeLong (Sun-Xu midrank) — from PAT 09 ----------------
def _midrank(x):
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _delong_cov(y_true, preds):
    y_true = np.asarray(y_true, dtype=np.int8)
    order = np.argsort(-y_true, kind="mergesort")
    m = int(np.sum(y_true == 1))
    n = len(y_true)
    nn = n - m
    ps = preds[:, order]
    k = ps.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, nn)); tz = np.empty((k, n))
    for r in range(k):
        tx[r] = _midrank(ps[r, :m]); ty[r] = _midrank(ps[r, m:]); tz[r] = _midrank(ps[r, :])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / nn
    v01 = (tz[:, :m] - tx) / nn
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]]); sy = np.array([[sy]])
    return aucs, sx / m + sy / nn


def delong_delta(y, p1, p2):
    """Return (auc1, auc2, delta, se, z, p). delta = auc1 - auc2."""
    y = np.asarray(y)
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return (np.nan,) * 6
    aucs, cov = _delong_cov(y, np.vstack([p1, p2]))
    delta = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float(delta), 0.0, 0.0, 1.0
    se = float(np.sqrt(var))
    z = delta / se
    return float(aucs[0]), float(aucs[1]), float(delta), se, float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def boot_auroc_ci(y, p, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y); p = np.asarray(p)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    take = min(len(y), MAX_BOOT_N)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), size=take)
        if len(np.unique(y[idx])) < 2:
            boots[i] = np.nan
            continue
        boots[i] = roc_auc_score(y[idx], p[idx])
    boots = boots[~np.isnan(boots)]
    if len(boots) < 10:
        return float(auc), np.nan, np.nan
    return float(auc), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------- Input guards + subgroup-column back-fill ----------------
def assert_v3_outcome(df):
    """Fail loudly if the v3 outcome sweep (CONTRACTS §9-J) was not honoured."""
    if OUTCOME_COL == DEATH_SENSITIVITY_ONLY:
        raise SystemExit(
            f"OUTCOME_COL is '{DEATH_SENSITIVITY_ONLY}', which is v3's all-cause "
            "death SENSITIVITY arm (CONTRACTS §4 rung 0), not an outcome for "
            "subgroup discrimination. The v3 mortality-only anchor is "
            "'death_hemorrhagic' and it is exercised by 11/13, not 09.")
    if OUTCOME_COL not in df.columns:
        raise SystemExit(
            f"scored parquet has no '{OUTCOME_COL}' column. 06_score_test.py "
            "owns it (CONTRACTS §7 row 06).")
    y = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
    if y.isna().any():
        raise SystemExit(f"{OUTCOME_COL} has NaN — must be int 0/1")
    if not set(np.unique(y.to_numpy())) <= {0, 1}:
        raise SystemExit(f"{OUTCOME_COL} is not 0/1 — got {sorted(set(y.unique()))}")
    # Read (never retype) the definition; hard-fails if the manifest is absent.
    print(f"  outcome  : {OUTCOME_COL}  ({outcome_definition_full()})")


def backfill_subgroup_cols(df):
    """Back-fill any REQUIRED_SUBGROUP_COLS missing from the scored parquet by
    joining candidates.parquet on (puf_year, inc_key).

    06 carries the subgroup vars through, but it *warns-and-skips* a column it
    cannot find, and this script's `if col in df.columns` guards would then drop
    that whole stratum SILENTLY. That failure mode is live for `unstable_ring`
    specifically: Fix 3 strips it from the score, and an agent propagating that
    change too far could remove it from 06's carry list. CONTRACTS §13 G3f
    requires the ring stratification to survive, so recover it here rather than
    emit a quietly-shorter forest.
    """
    missing = [c for c in REQUIRED_SUBGROUP_COLS if c not in df.columns]
    if not missing:
        print(f"  subgroup vars: all {len(REQUIRED_SUBGROUP_COLS)} present in the scored parquet")
        return df
    print(f"  [backfill] missing from scored parquet: {missing}")
    if not os.path.exists(CANDIDATES):
        raise SystemExit(
            f"subgroup columns {missing} absent from {SCORED_PARQUET} and "
            f"{CANDIDATES} does not exist — cannot build the pre-specified "
            "subgroup grid (CONTRACTS §9-F).")
    import pyarrow.parquet as pq
    cand_cols = set(pq.ParquetFile(CANDIDATES).schema.names)
    unavailable = [c for c in missing if c not in cand_cols]
    if unavailable:
        raise SystemExit(
            f"subgroup columns {unavailable} are absent from BOTH {SCORED_PARQUET} "
            f"and {CANDIDATES}. If 'unstable_ring' is among them, Fix 3 has been "
            "over-propagated: it is removed from the SCORE (02/04/05) only and must "
            "survive in the cohort CSV, candidates.parquet, the Analysis-B adjustment "
            "set and the subgroup grid (CONTRACTS §3 note, §6, §13 G3f).")
    cand = pd.read_parquet(CANDIDATES, columns=JOIN_KEYS + missing)
    left = df.copy()
    for frame in (left, cand):
        for k in JOIN_KEYS:
            frame[k] = pd.to_numeric(frame[k], errors="coerce").astype("Int64")
    n0 = len(left)
    left = left.merge(cand, on=JOIN_KEYS, how="left", validate="1:1")
    if len(left) != n0:
        raise SystemExit(f"back-fill merge changed row count {n0:,} -> {len(left):,}")
    for c in missing:
        n_nan = int(left[c].isna().sum())
        print(f"  [backfill] {c}: recovered from candidates.parquet ({n_nan:,} NaN)")
    return left


def assert_required_strata(df):
    """CONTRACTS §9-F / §13 G3f: every pre-specified stratifier must be usable."""
    absent = [c for c in REQUIRED_SUBGROUP_COLS if c not in df.columns]
    if absent:
        raise SystemExit(
            f"pre-specified subgroup columns unavailable: {absent}. "
            "'unstable_ring' in particular is a REQUIRED stratum in v3 — Fix 3 "
            "removes it from the score (02/04/05), NOT from the subgroup grid "
            "or the Analysis-B adjustment set (CONTRACTS §6 asymmetry, §13 G3f).")
    ur = pd.to_numeric(df["unstable_ring"], errors="coerce")
    n_hi, n_lo = int((ur == 1).sum()), int((ur == 0).sum())
    if n_hi == 0 or n_lo == 0:
        raise SystemExit(
            f"unstable_ring is degenerate in the test set (n=1:{n_hi:,} n=0:{n_lo:,}) "
            "— the ring-stability stratum cannot be formed (§13 G3f).")
    print(f"  G3f OK: ring stratum intact (unstable={n_hi:,} / stable={n_lo:,})")


# ---------------- Subgroup definitions (pelvic) ----------------
def build_subgroups(df):
    subs = []
    if "age_ge65" in df.columns:
        a = pd.to_numeric(df["age_ge65"], errors="coerce")
        subs.append(("Age", "<65", a == 0))
        subs.append(("Age", ">=65", a == 1))
    if "sex_male" in df.columns:
        s = pd.to_numeric(df["sex_male"], errors="coerce")
        subs.append(("Sex", "Male", s == 1))
        subs.append(("Sex", "Female", s == 0))
    if "unstable_ring" in df.columns:
        u = pd.to_numeric(df["unstable_ring"], errors="coerce")
        subs.append(("Ring_Stability", "Stable", u == 0))
        subs.append(("Ring_Stability", "Unstable", u == 1))
    if "cc_anticoag" in df.columns:
        c = pd.to_numeric(df["cc_anticoag"], errors="coerce")
        subs.append(("Anticoagulation", "No", c == 0))
        subs.append(("Anticoagulation", "Yes", c == 1))
    if "mechanism_cat" in df.columns:
        mc = df["mechanism_cat"].astype("string").str.strip()
        for lvl in [x for x in mc.dropna().unique().tolist()]:
            subs.append(("Mechanism", str(lvl), mc == lvl))
    if "acs_level" in df.columns:
        ac = df["acs_level"].astype("string").str.strip()
        for lvl in [x for x in ac.dropna().unique().tolist()]:
            subs.append(("ACS_Level", str(lvl), ac == lvl))
    clean = []
    for name, level, mask in subs:
        m = mask.fillna(False).astype(bool) if hasattr(mask, "fillna") else np.asarray(mask, bool)
        clean.append((name, level, m))
    return clean


def oriented(sub_df, col, sign):
    raw = pd.to_numeric(sub_df[col], errors="coerce").to_numpy(dtype=np.float64)
    return sign * raw


def score_block(sub_df, subgroup, level, y_ev):
    rows = []
    too_small = y_ev < MIN_EVENTS
    y_full = pd.to_numeric(sub_df[OUTCOME_COL], errors="coerce").to_numpy()
    pel = oriented(sub_df, PELVIC[2], PELVIC[3])

    for short, pretty, col, sign in SCORES:
        row = {"Subgroup": subgroup, "Level": level, "Score": pretty,
               "N": np.nan, "N_events": np.nan, "AUROC": np.nan, "CI_Low": np.nan,
               "CI_High": np.nan, "Delta_vs_pelvic": np.nan, "Delta_SE": np.nan,
               "Delta_CI_Low": np.nan, "Delta_CI_High": np.nan, "Delta_P": np.nan,
               "N_events_too_small": too_small}
        s = oriented(sub_df, col, sign)
        m = ~np.isnan(s) & ~np.isnan(y_full)
        y = y_full[m].astype(int)
        sm_ = s[m]
        n = int(len(y)); ne = int(y.sum())
        row["N"], row["N_events"] = n, ne
        if too_small or n < 20 or ne < 2 or (n - ne) < 2:
            rows.append(row)
            continue
        auc, lo, hi = boot_auroc_ci(y, sm_)
        row["AUROC"], row["CI_Low"], row["CI_High"] = auc, lo, hi
        if short == "PELVIC":
            row.update({"Delta_vs_pelvic": 0.0, "Delta_SE": 0.0, "Delta_CI_Low": 0.0,
                        "Delta_CI_High": 0.0, "Delta_P": 1.0})
        else:
            pm = ~np.isnan(pel) & ~np.isnan(s) & ~np.isnan(y_full)
            yp = y_full[pm].astype(int)
            if len(yp) >= 20 and len(np.unique(yp)) >= 2 and int(yp.sum()) >= MIN_EVENTS:
                a1, a2, delta, se, z, pv = delong_delta(yp, pel[pm], s[pm])
                row["Delta_vs_pelvic"] = delta
                row["Delta_SE"] = se
                if np.isfinite(delta) and np.isfinite(se):
                    row["Delta_CI_Low"] = delta - 1.96 * se
                    row["Delta_CI_High"] = delta + 1.96 * se
                row["Delta_P"] = pv
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Downsample TEST rows (smoke).")
    args = parser.parse_args()

    t0 = time.time()
    banner("Step 09 — Pelvic subgroup AUROCs + DeLong deltas (DATA only; figure disabled)")
    df = pd.read_parquet(SCORED_PARQUET)
    print(f"  scored rows: {df.shape}")
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE] N={len(df):,}")

    assert_v3_outcome(df)
    df = backfill_subgroup_cols(df)
    assert_required_strata(df)

    n_ev = int(pd.to_numeric(df[OUTCOME_COL], errors="coerce").fillna(0).sum())
    print(f"  test rows={len(df):,}  events={n_ev:,} "
          f"({100 * n_ev / max(len(df), 1):.2f}%)")

    subs = build_subgroups(df)
    print(f"  {len(subs)} subgroup levels defined")

    all_rows = []
    for sub_name, level, mask in subs:
        sub_df = df.loc[mask].copy()
        n_sub = len(sub_df)
        y_ev = int(pd.to_numeric(sub_df[OUTCOME_COL], errors="coerce").fillna(0).sum())
        print(f"\n>>> {sub_name} | {level}: N={n_sub:,} events={y_ev:,}")
        if n_sub == 0:
            for _, pretty, _, _ in SCORES:
                all_rows.append({"Subgroup": sub_name, "Level": level, "Score": pretty,
                                 "N": 0, "N_events": 0, "AUROC": np.nan, "CI_Low": np.nan,
                                 "CI_High": np.nan, "Delta_vs_pelvic": np.nan, "Delta_SE": np.nan,
                                 "Delta_CI_Low": np.nan, "Delta_CI_High": np.nan, "Delta_P": np.nan,
                                 "N_events_too_small": True})
            continue
        rows = score_block(sub_df, sub_name, level, y_ev)
        for r in rows:
            if np.isfinite(r["AUROC"]):
                print(f"    {r['Score']:<14} n={int(r['N']):>7,} AUC={r['AUROC']:.4f} "
                      f"[{r['CI_Low']:.4f},{r['CI_High']:.4f}] dAUC={r['Delta_vs_pelvic']:+.4f} "
                      f"p={r['Delta_P']:.3g}")
            else:
                print(f"    {r['Score']:<14} n={int(r['N']):>7,} AUC=NaN"
                      f"{' (events<50)' if r['N_events_too_small'] else ''}")
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)[[
        "Subgroup", "Level", "Score", "N", "N_events", "AUROC", "CI_Low", "CI_High",
        "Delta_vs_pelvic", "Delta_SE", "Delta_CI_Low", "Delta_CI_High", "Delta_P",
        "N_events_too_small"]]
    # Carry the outcome + its manifest-sourced definition on the table so the
    # manuscript footer quotes the cohort build rather than a retyped literal
    # (CONTRACTS §3.1 "No script may hard-code them"). Additive columns only —
    # every consumer indexes by name (§7.1). The F6 fig-data schema below is
    # deliberately left UNCHANGED.
    out_df["Outcome"] = OUTCOME_COL
    out_df["Outcome_Definition"] = outcome_definition_full()
    out_df.to_csv(OUT_TABLE, index=False)
    print(f"\n[write] {OUT_TABLE}  ({len(out_df)} rows)")

    # Forest fig-data (F6): comparator rows with delta vs pelvic + the pelvic AUROC per cell.
    pelvic_auroc = (out_df[out_df["Score"] == PELVIC[1]]
                    .set_index(["Subgroup", "Level"])["AUROC"].to_dict())
    forest = out_df[out_df["Score"].isin([c[1] for c in COMPARATORS])].copy()
    forest["pelvic_AUROC"] = forest.apply(
        lambda r: pelvic_auroc.get((r["Subgroup"], r["Level"]), np.nan), axis=1)
    forest = forest[["Subgroup", "Level", "Score", "N", "N_events", "AUROC", "pelvic_AUROC",
                     "Delta_vs_pelvic", "Delta_SE", "Delta_CI_Low", "Delta_CI_High", "Delta_P",
                     "N_events_too_small"]]
    forest.to_csv(OUT_FOREST, index=False)
    print(f"[write] {OUT_FOREST}  ({len(forest)} rows)")

    banner("Verification")
    pat_rows = out_df[out_df["Score"] == PELVIC[1]]
    n_valid = int(out_df["AUROC"].notna().sum())
    print(f"  total rows={len(out_df):,}  rows with AUROC={n_valid:,}")
    if pat_rows["AUROC"].notna().any():
        print(f"  pelvic_score AUROC range across cells: "
              f"{pat_rows['AUROC'].min():.4f} .. {pat_rows['AUROC'].max():.4f}")

    # CONTRACTS §13 G3f — the ring-stability stratum must be present in the
    # emitted table AND in the F6 forest fig-data. A silently-shorter forest is
    # the exact failure mode Fix 3 over-propagation produces.
    ring_tbl = out_df[out_df["Subgroup"] == "Ring_Stability"]
    ring_fst = forest[forest["Subgroup"] == "Ring_Stability"]
    if ring_tbl.empty or ring_fst.empty:
        raise SystemExit(
            "G3f FAIL: no Ring_Stability rows emitted. `unstable_ring` is a "
            "REQUIRED v3 subgroup stratum (CONTRACTS §6 asymmetry / §13 G3f); "
            "Fix 3 removes it from the score only.")
    print(f"  G3f OK: Ring_Stability rows -> table {len(ring_tbl)}, forest {len(ring_fst)} "
          f"(levels: {sorted(ring_tbl['Level'].unique().tolist())})")
    print(f"  outcome sweep OK (§9-J): scored on '{OUTCOME_COL}' = "
          f"{outcome_definition_full()}")
    print(f"  ^ definition READ from {os.path.relpath(QC_MANIFEST, config.REPO_ROOT)} "
          f"(01 is the single source of truth) and carried on "
          f"{os.path.basename(OUT_TABLE)} as `Outcome_Definition`.")
    print(f"\n[done] runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
