"""
Pelvic Arrival Tool v3 — Step 06: Score the held-out TEST set
================================================================================

Forked from PAT_TBI/v4/code/06_score_test_set.py, but the 4 TBI comparators
(GCS / GCS-P / IMPACT / CRASH) are DROPPED. Pelvic comparators (CONTRACTS §7
row 06, pre-registered §12) are SI-alone + rSIG ONLY.

v3 CHANGES (Fix 4 — CONTRACTS §5.1, SAP_amendment_v3 D-4)
---------------------------------------------------------
`sbp_unrecordable` ("no signs of life on arrival (unrecordable systolic
pressure)") is NO LONGER a point row. `05` writes `tables/table3_points.csv`
with a `Role` column ∈ {point, override}; the override row carries an EMPTY
`Points` cell and `refit.json` carries disjoint `point_vars` / `override_vars`.

This script is the CONSUMER side of that contract:
  * `pelvic_score = Σ_{v ∈ point_vars} Points_v · X_v`   (override rows skipped)
  * the point weights are taken from **`refit.json["points"]`**, which `05`
    emits over POINT-ROLE VARS ONLY, so a naive `Σ points[v]·X[v]` cannot
    double-count. `table3_points.csv` is read for the `Role` split and is
    cross-checked variable-by-variable and value-by-value against
    `refit.json`; any disagreement is a hard fail (the two artifacts must come
    from the same `05` run).
  * `Points` is parsed with `errors="coerce"` and **never** `.fillna(0)` before
    the Role split — a `.fillna(0)` here would silently turn the override into
    a 0-point *point row*: numerically harmless in the sum, but it defeats the
    "assert Points is NA" check, puts `sbp_unrecordable` back into the card's
    variable list, and hides the override entirely.
  * a separate boolean column `override_nslo` is emitted; it contributes ZERO
    to `pelvic_score`
  * MANDATORY double-counting guard: the emitted score is recomputed with every
    override column forced to 0 and asserted element-wise equal (hard assert,
    not a print). If it differs, the override leaked into the point sum.
  * the `Role` column is REQUIRED. Running against a stale pre-Fix-4
    `table3_points.csv` fails loudly rather than silently scoring the override
    as a point row.

`pelvic_lp` (the continuous frozen linear predictor) DOES include the override
variable's β — by design, and this is NOT a bug to "fix". `05` keeps
`sbp_unrecordable` in the unpenalized refit so every other β stays adjusted for
it (CONTRACTS §5.1 row `05`), and `12` freezes that linear predictor. The
asymmetry in `refit.json` is deliberate:
    refit["points"]       -> point-role vars ONLY  (feeds pelvic_score)
    refit["coefficients"] -> ALL vars incl. the override (feeds pelvic_lp)
The continuous `pelvic_lp` and the integer `pelvic_score` are different objects
and the override belongs to exactly one of them (the lp). The override is
removed from the INTEGER POINT SCALE only.

Computes on the cohort_primary==1 TEST slice:
  * pelvic_score  — integer bedside card, Σ Points·X over Role=="point" rows
  * pelvic_lp     — continuous frozen linear predictor, intercept + Σ β·X from
                    refit.json (feeds the continuous-logit AUROC in 07)
  * override_nslo — binary automatic-escalate override (Fix 4); NOT in the sum
  * shock_index   — SI-alone comparator = pulse2 / sbp2  (STANDARD shock index,
                    HR/SBP; HIGHER = worse). Recomputed from the raw pass-through
                    vitals -> NaN where sbp2 or pulse2 is missing. Chosen (over
                    the reverse-SI sbp2/pulse2 literal) so the comparator is
                    risk-oriented, matches the score's si_ge0_9/si_ge1_0 features
                    and script 11's SI thresholds, and yields an interpretable
                    AUROC on the same scale used everywhere else.
  * rsig          — rSIG comparator = (sbp2 / pulse2) · gcs2  (reverse-SI × GCS,
                    canonical rSIG; LOWER = worse). NaN where any input missing.
                    Downstream scripts (07/08/09) negate rsig so every score is
                    handled higher = worse uniformly.

Carries forward the FULL v3 outcome set (CONTRACTS §3 / §4) so that 07/08/09/10
and 17 never have to re-open the cohort CSV:
  primary        major_hemorrhage_event
  co-primary     patient_state_composite, death_hemorrhagic
  components     received_hc, substantial_transfusion
  secondary      mtp_equivalent
  sensitivities  death_inhosp, death_hemorrhagic_headais,
                 death_hemorrhagic_missing_incl, death_le48h, death_ed,
                 substantial_transfusion_ml_only,
                 major_hemorrhage_event_headais, major_hemorrhage_event_v2def
plus subgroup vars (age, age_ge65, sex_male, unstable_ring, cc_anticoag,
mechanism_cat, acs_level), full-info inputs (iss, ais_head), raw vitals
(sbp2, pulse2, gcs2), the exploratory 4h product volumes (blood_4h, plasma_4h,
platelet_4h — 17's balanced-resuscitation subgroup) and every refit variable
(point + override + refit.variables_kept) needed by 07's Harrell optimism refit.

`finaldischarge_days` is deliberately NOT carried: it is outcome-defining,
FORBIDDEN as a predictor (CONTRACTS §5), and no consumer of this artifact needs
it (13 reads the cohort CSV directly).

Output
------
  v3/derived/cohort_test_scored.parquet

Run
---
    python 06_score_test.py
    python 06_score_test.py --sample 50000     # smoke (row downsample)

Seed: 42
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES
os.makedirs(DERIVED, exist_ok=True)

CANDIDATES = os.path.join(DERIVED, "candidates.parquet")
TEST_KEYS = os.path.join(DERIVED, "test_inc_keys.csv")
POINTS_CSV = os.path.join(TABLES, "table3_points.csv")
REFIT_JSON = os.path.join(DERIVED, "refit.json")
OUT_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")

OUTCOME_COL = "major_hemorrhage_event"
JOIN_KEYS = ["puf_year", "inc_key"]
COHORT_FILTER_COL = "cohort_primary"

# ---- Fix 4 (CONTRACTS §5.1) -----------------------------------------------
ROLE_COL = "Role"
ROLE_POINT = "point"
ROLE_OVERRIDE = "override"
EXPECTED_OVERRIDE_VAR = "sbp_unrecordable"
OVERRIDE_OUT_COL = "override_nslo"          # "no signs of life on arrival"
OVERRIDE_LABEL = "No signs of life on arrival (unrecordable systolic pressure)"

# ---- Columns carried through for downstream steps (07/08/09/10/17) --------
# Binary 0/1 outcome columns. The first three must be non-null (hard assert);
# the rest are filled with 0 and the fill count is printed.
OUTCOMES_REQUIRED = [
    "major_hemorrhage_event",          # PRIMARY (CONTRACTS §4)
    "patient_state_composite",         # co-primary, anti-circularity
    "death_hemorrhagic",               # co-primary mortality-only anchor (v3)
]
OUTCOMES_BINARY_CARRY = [
    "received_hc",                     # composite component
    "substantial_transfusion",         # composite component (Fix 1)
    "mtp_equivalent",                  # pre-specified secondary (Fix 7)
    # --- death sensitivity ladder (CONTRACTS §4) ---
    "death_inhosp",                    # rung 0 — v2 all-cause, now a SENSITIVITY
    "death_hemorrhagic_headais",       # rung 1a
    "death_le48h",                     # rung 1b
    "death_hemorrhagic_missing_incl",  # rung 2' — bounding upper edge
    "death_ed",                        # ED-death surrogate
    # --- composite-level sensitivities ---
    "substantial_transfusion_ml_only",
    "major_hemorrhage_event_headais",
    "major_hemorrhage_event_v2def",
]
SUBGROUP_VARS = ["age", "age_ge65", "sex_male", "unstable_ring", "cc_anticoag",
                 "mechanism_cat", "acs_level"]
FULLINFO_VARS = ["iss", "ais_head"]
RAW_VITALS = ["sbp2", "pulse2", "gcs2"]
# NaN-preserving numeric pass-throughs (17's exploratory subgroup, §3.2).
NUMERIC_CARRY = ["blood_4h", "plasma_4h", "platelet_4h"]


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def _flex_col(df, candidates):
    """Return the first column name present in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_points_table(path):
    """Read table3_points.csv and split it into point rows and override rows.

    Fix 4 / CONTRACTS §5.1 requires a `Role` column. Its absence means the file
    was written by a pre-Fix-4 `05` — fail loudly rather than silently scoring
    the override as a point row.
    """
    w = pd.read_csv(path)
    if ROLE_COL not in w.columns:
        raise SystemExit(
            f"\nFATAL — {os.path.basename(path)} has no '{ROLE_COL}' column.\n"
            f"  Columns found: {list(w.columns)}\n"
            f"  CONTRACTS §5.1 (Fix 4) requires 05_refit_integer.py to emit\n"
            f"  Role ∈ {{'{ROLE_POINT}','{ROLE_OVERRIDE}'}}, with "
            f"'{EXPECTED_OVERRIDE_VAR}' as Role='{ROLE_OVERRIDE}' and an EMPTY\n"
            f"  Points cell. This file predates that change (a stale v2-style\n"
            f"  points table). Scoring it here would silently fold the\n"
            f"  automatic-escalate override back into the integer point sum,\n"
            f"  which is exactly the double-counting Fix 4 exists to prevent.\n"
            f"  → Re-run 05_refit_integer.py before 06.\n")

    w = w.dropna(subset=["Variable"]).copy()
    w["Variable"] = w["Variable"].astype(str).str.strip()
    w = w[w["Variable"] != ""].copy()

    w["Role"] = w[ROLE_COL].astype(str).str.strip().str.lower()
    bad_roles = sorted(set(w["Role"]) - {ROLE_POINT, ROLE_OVERRIDE})
    if bad_roles:
        raise SystemExit(
            f"FATAL — unrecognised {ROLE_COL} value(s) in {path}: {bad_roles}. "
            f"Allowed: '{ROLE_POINT}', '{ROLE_OVERRIDE}' (CONTRACTS §5.1).")

    # Parse Points BEFORE any fill so an empty override cell stays NaN.
    w["Points_raw"] = pd.to_numeric(w["Points"], errors="coerce")

    pts_rows = w[w["Role"] == ROLE_POINT].copy()
    ovr_rows = w[w["Role"] == ROLE_OVERRIDE].copy()

    # (a) every point row must carry a real integer point value
    bad_pts = pts_rows.loc[pts_rows["Points_raw"].isna(), "Variable"].tolist()
    if bad_pts:
        raise SystemExit(
            f"FATAL — Role=='{ROLE_POINT}' rows with a missing Points value: "
            f"{bad_pts}. Every point row needs an integer point weight.")
    pts_rows["Points"] = pts_rows["Points_raw"].round().astype(int)

    # (b) CONTRACTS §5.1 06-row: hard assert override Points is NA
    bad_ovr = ovr_rows.loc[ovr_rows["Points_raw"].notna(),
                           ["Variable", "Points_raw"]]
    assert bad_ovr.empty, (
        "FATAL (CONTRACTS §5.1) — Role=='override' row(s) carry a numeric "
        f"Points value:\n{bad_ovr.to_string(index=False)}\n"
        "The override must have an EMPTY Points cell; a number here means the "
        "override is still on the integer point scale.")

    return w, pts_rows, ovr_rows


def main():
    parser = argparse.ArgumentParser(description="Pelvic v3 Step 06: score the TEST set")
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample N test rows for a smoke test.")
    args = parser.parse_args()

    banner("Step 06 — Score the held-out TEST set (cohort_primary==1)")
    print(f"Candidates : {CANDIDATES}\nTest keys  : {TEST_KEYS}\n"
          f"Points     : {POINTS_CSV}\nRefit json : {REFIT_JSON}\nOutput     : {OUT_PARQUET}")

    # ---------------- Load + filter + merge ----------------
    banner("Loading candidates (cohort_primary==1) + test inc_keys (inner merge)")
    df_all = pd.read_parquet(CANDIDATES)
    if COHORT_FILTER_COL not in df_all.columns:
        raise SystemExit(f"candidates parquet missing {COHORT_FILTER_COL}")
    df_all = df_all[df_all[COHORT_FILTER_COL] == 1].copy()
    print(f"  candidates (cohort_primary==1): {df_all.shape}")

    test_keys = pd.read_csv(TEST_KEYS)
    for frame in (df_all, test_keys):
        for k in JOIN_KEYS:
            frame[k] = pd.to_numeric(frame[k], errors="coerce").astype("Int64")

    df = test_keys[JOIN_KEYS].merge(df_all, on=JOIN_KEYS, how="inner", validate="1:1")
    print(f"  test slice: {df.shape}")
    if len(df) != len(test_keys):
        raise SystemExit(f"Test-slice merge {len(df):,} != test_inc_keys {len(test_keys):,}")
    print(f"  ASSERT pass: len(merged) == len(test_inc_keys) == {len(df):,}")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE] downsampled to {len(df):,}", flush=True)

    outcome = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
    if outcome.isna().any():
        raise SystemExit(f"{OUTCOME_COL} has NaN — must be int 0/1")
    df[OUTCOME_COL] = outcome.astype(int)
    n_total, n_events = len(df), int(df[OUTCOME_COL].sum())
    print(f"  TEST: N={n_total:,}  events={n_events:,} ({100*n_events/n_total:.2f}%)")
    print(f"  NOTE: {OUTCOME_COL} is the v3 composite "
          f"(received_hc | death_hemorrhagic | substantial_transfusion)")

    # ---------------- Load frozen weights (Fix 4 aware) ----------------
    banner("Loading frozen weights (table3_points + refit.json) — Fix 4 Role split")
    w_all, pts_rows, ovr_rows = load_points_table(POINTS_CSV)
    point_vars = pts_rows["Variable"].tolist()
    override_vars = ovr_rows["Variable"].tolist()

    print(f"  {len(point_vars)} POINT vars   : {point_vars}")
    print(f"  {len(override_vars)} OVERRIDE var(s): {override_vars} "
          f"(Points cell empty by contract)")

    # Disjointness (CONTRACTS §5.1 (e)).
    assert set(point_vars).isdisjoint(set(override_vars)), (
        "FATAL — a variable appears as BOTH a point row and an override row: "
        f"{sorted(set(point_vars) & set(override_vars))}")
    assert len(set(point_vars)) == len(point_vars), "duplicate point Variable rows"
    assert len(set(override_vars)) == len(override_vars), "duplicate override Variable rows"

    # The v3 contract names exactly one override variable.
    if len(override_vars) != 1 or override_vars[0] != EXPECTED_OVERRIDE_VAR:
        raise SystemExit(
            f"FATAL — CONTRACTS §5.1 specifies exactly one override variable "
            f"('{EXPECTED_OVERRIDE_VAR}'); table3_points.csv declares "
            f"{override_vars}. Reconcile 05 and CONTRACTS before scoring.")

    for v in point_vars + override_vars:
        if v not in df.columns:
            raise SystemExit(f"score variable '{v}' missing from candidates.parquet")

    with open(REFIT_JSON) as f:
        refit = json.load(f)
    intercept = float(refit["intercept"])
    # coefficients retain ALL vars including the override — deliberate (see docstring)
    coefs = {k: float(v) for k, v in refit["coefficients"].items()}

    # ---- Authoritative point weights: refit.json["points"] (point-role only) ----
    rj_points = refit.get("points")
    table3_points = {str(r["Variable"]).strip(): int(r["Points"])
                     for _, r in pts_rows.iterrows()}
    if isinstance(rj_points, dict) and rj_points:
        rj_points = {str(k): int(round(float(v))) for k, v in rj_points.items()}
        if set(rj_points) != set(table3_points):
            raise SystemExit(
                "FATAL — refit.json['points'] and table3_points.csv Role=='point' "
                "declare different variable sets. The two artifacts must come from "
                "the same 05 run.\n"
                f"  refit.json only : {sorted(set(rj_points) - set(table3_points))}\n"
                f"  table3 only     : {sorted(set(table3_points) - set(rj_points))}")
        mism = {v: (rj_points[v], table3_points[v])
                for v in rj_points if rj_points[v] != table3_points[v]}
        if mism:
            raise SystemExit(
                "FATAL — point VALUES disagree between refit.json['points'] and "
                f"table3_points.csv (var: refit, table3): {mism}")
        for v in override_vars:
            assert v not in rj_points, (
                f"FATAL — override var '{v}' appears in refit.json['points']; "
                "05 must emit point-role vars only there.")
        print("  ASSERT pass: refit.json['points'] == table3 Role=='point' "
              "(same vars, same values, override absent)")
        points_map = rj_points
        points_source = "refit.json['points'] (authoritative)"
    else:
        print("  [warn] refit.json has no 'points' dict — falling back to "
              "table3_points.csv Role=='point' rows.")
        points_map = table3_points
        points_source = "table3_points.csv Role=='point' (fallback)"
    print(f"  point weights from: {points_source}")

    # Force-in anchors that landed at 0 points must be REPORTED, not papered over
    # (the force-in is pre-registered in the locked SAP).
    zero_pts = sorted(v for v, p in points_map.items() if p == 0)
    if zero_pts:
        anchors = set(refit.get("force_in_spec", []) or refit.get("forced_in_anchors", []))
        print(f"  [REPORT] point-role variable(s) with Points == 0: {zero_pts}")
        hit = sorted(set(zero_pts) & anchors)
        if hit:
            print(f"  [REPORT] *** PRE-REGISTERED FORCE-IN ANCHOR(S) AT 0 POINTS: "
                  f"{hit} *** — this is a reportable result, not something to "
                  f"silently drop; the force-in is pre-registered in the locked SAP.")

    # Cross-check table3 against refit.json's point_vars / override_vars if present.
    rj_point = refit.get("point_vars")
    rj_override = refit.get("override_vars")
    if rj_point is not None and rj_override is not None:
        assert set(rj_point).isdisjoint(set(rj_override)), (
            "FATAL — refit.json point_vars / override_vars are not disjoint "
            f"({sorted(set(rj_point) & set(rj_override))})")
        if set(rj_point) != set(point_vars):
            raise SystemExit(
                "FATAL — refit.json['point_vars'] disagrees with "
                "table3_points.csv Role=='point'.\n"
                f"  refit.json : {sorted(rj_point)}\n"
                f"  table3     : {sorted(point_vars)}\n"
                "  The two artifacts must be written by the same 05 run.")
        if set(rj_override) != set(override_vars):
            raise SystemExit(
                "FATAL — refit.json['override_vars'] disagrees with "
                "table3_points.csv Role=='override'.\n"
                f"  refit.json : {sorted(rj_override)}\n"
                f"  table3     : {sorted(override_vars)}")
        print("  ASSERT pass: refit.json point_vars/override_vars == table3 Role split")
    else:
        print("  [warn] refit.json carries no point_vars/override_vars keys — "
              "cross-check skipped (CONTRACTS §5.1 (d) expects them).")

    # ---------------- pelvic_score (integer, POINT ROWS ONLY) ----------------
    banner("Computing pelvic_score (integer, Role=='point' only) + pelvic_lp")

    def _score_from_points(frame, pmap, zero_cols=()):
        """Σ Points·X over the `pmap` {variable: points}, `zero_cols` forced to 0."""
        acc = np.zeros(len(frame), dtype=np.float64)
        zero = set(zero_cols)
        for v, pts in pmap.items():
            pts = int(pts)
            if pts == 0:
                continue
            x = (np.zeros(len(frame), dtype=np.float64) if v in zero
                 else frame[v].astype(np.float64).values)
            acc += pts * x
        return acc

    pelvic_score = _score_from_points(df, points_map).astype(np.int64)

    # ---- MANDATORY double-counting guard (CONTRACTS §5.1, hard assert) ----
    # Recompute with EVERY override column forced to 0. If the emitted score
    # reads an override column at all, these differ.
    pelvic_score_zeroed = _score_from_points(
        df, points_map, zero_cols=override_vars).astype(np.int64)
    n_diff = int((pelvic_score != pelvic_score_zeroed).sum())
    assert n_diff == 0, (
        "FATAL DOUBLE-COUNTING (CONTRACTS §5.1) — zeroing the override "
        f"column(s) {override_vars} changed pelvic_score for {n_diff:,} rows. "
        "The automatic-escalate override has leaked into the integer point "
        "sum; the score and the override would both be counting the same "
        "patients.")
    print(f"  ASSERT pass: double-counting guard — pelvic_score is identical "
          f"with {override_vars} forced to 0 ({n_diff} rows differ)")

    # Belt-and-braces: no override variable may appear among the point rows.
    assert not (set(override_vars) & set(point_vars)), "override var in point rows"

    # pelvic_lp — frozen continuous linear predictor (intercept + Σ β·X).
    # DELIBERATELY includes the override variable's β: 05 keeps it in the
    # unpenalized refit so every other β stays adjusted for it (§5.1 row 05).
    pelvic_lp = np.full(len(df), float(intercept), dtype=np.float64)
    for v, b in coefs.items():
        if v not in df.columns:
            raise SystemExit(f"refit coefficient var '{v}' missing from candidates")
        pelvic_lp += b * df[v].astype(np.float64).values
    if EXPECTED_OVERRIDE_VAR in coefs:
        print(f"  NOTE: pelvic_lp includes β[{EXPECTED_OVERRIDE_VAR}]="
              f"{coefs[EXPECTED_OVERRIDE_VAR]:+.4f} (continuous model, by design); "
              f"pelvic_score does NOT.")
    print(f"  pelvic_score range {pelvic_score.min()}..{pelvic_score.max()}  "
          f"mean={pelvic_score.mean():.2f}")
    print(f"  pelvic_lp range {pelvic_lp.min():.3f}..{pelvic_lp.max():.3f}")

    # ---------------- override_nslo (Fix 4) ----------------
    # Canonical display string comes from 05 (refit.json) — never invented here.
    _labels = refit.get("override_labels") or {}
    ovr_label = (str(_labels.get(EXPECTED_OVERRIDE_VAR)) if isinstance(_labels, dict)
                 and _labels.get(EXPECTED_OVERRIDE_VAR) else OVERRIDE_LABEL)
    banner(f"Computing {OVERRIDE_OUT_COL} — {ovr_label}")
    ovr = np.zeros(len(df), dtype=np.int64)
    for v in override_vars:
        col = pd.to_numeric(df[v], errors="coerce").fillna(0).to_numpy()
        ovr = np.maximum(ovr, (col > 0).astype(np.int64))
    n_ovr = int(ovr.sum())
    ovr_event_rate = float(df[OUTCOME_COL].to_numpy()[ovr == 1].mean()) if n_ovr else np.nan
    print(f"  {OVERRIDE_OUT_COL}: n={n_ovr:,} ({100*n_ovr/len(df):.3f}% of test set)")
    print(f"  composite event rate in the override group: "
          f"{100*ovr_event_rate:.2f}%" if n_ovr else "  (override group empty)")
    print(f"  {OVERRIDE_OUT_COL} contributes ZERO points to pelvic_score "
          f"(deployed rule is score>=c OR override — assembled in 08)")

    # ---------------- Comparators (SI-alone + rSIG) ----------------
    banner("Computing comparators: SI-alone (standard) + rSIG (canonical)")
    sbp2 = pd.to_numeric(df["sbp2"], errors="coerce").values
    pulse2 = pd.to_numeric(df["pulse2"], errors="coerce").values
    gcs2 = pd.to_numeric(df["gcs2"], errors="coerce").values

    with np.errstate(divide="ignore", invalid="ignore"):
        # SI-alone = standard shock index HR/SBP (higher = worse), from raw vitals.
        shock_index = np.where((sbp2 > 0) & ~np.isnan(sbp2) & ~np.isnan(pulse2),
                               pulse2 / sbp2, np.nan)
        # rSIG = (SBP/HR) * GCS (canonical; LOWER = worse).
        rsig = np.where((pulse2 > 0) & ~np.isnan(sbp2) & ~np.isnan(pulse2) & ~np.isnan(gcs2),
                        (sbp2 / pulse2) * gcs2, np.nan)
    print(f"  shock_index (SI-alone): {int((~np.isnan(shock_index)).sum()):,} non-NaN "
          f"({int(np.isnan(shock_index).sum()):,} NaN)")
    print(f"  rsig (rSIG)           : {int((~np.isnan(rsig)).sum()):,} non-NaN "
          f"({int(np.isnan(rsig).sum()):,} NaN)")

    # ---------------- Assemble output ----------------
    banner("Assembling output DataFrame")
    out = pd.DataFrame({
        "puf_year": df["puf_year"].astype("int64").values,
        "inc_key": df["inc_key"].astype("int64").values,
        OUTCOME_COL: df[OUTCOME_COL].astype(int).values,
        "pelvic_score": pelvic_score,
        "pelvic_lp": pelvic_lp,
        OVERRIDE_OUT_COL: ovr,
        "shock_index": shock_index,
        "rsig": rsig,
    })

    # --- required outcomes: hard assert non-null ---
    for c in OUTCOMES_REQUIRED:
        if c in out.columns:
            continue
        if c not in df.columns:
            raise SystemExit(
                f"FATAL — required v3 outcome '{c}' missing from "
                f"candidates.parquet (CONTRACTS §3/§4). 02 must carry it.")
        s = pd.to_numeric(df[c], errors="coerce")
        assert not s.isna().any(), f"required outcome '{c}' has NaN"
        out[c] = s.astype(int).values

    # --- binary carry: fill 0, report the fill ---
    missing_carry = []
    for c in OUTCOMES_BINARY_CARRY:
        if c in out.columns:
            continue
        if c not in df.columns:
            missing_carry.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        n_na = int(s.isna().sum())
        if n_na:
            print(f"  [note] '{c}' had {n_na:,} NaN -> filled 0")
        out[c] = s.fillna(0).astype(int).values
    if missing_carry:
        print(f"  [warn] v3 outcome columns absent from candidates.parquet "
              f"(skipped): {missing_carry}")

    # --- subgroup / full-info / vitals / numeric pass-throughs / refit vars ---
    refit_kept = [v for v in refit.get("variables_kept", []) if isinstance(v, str)]
    carry = list(dict.fromkeys(
        SUBGROUP_VARS + FULLINFO_VARS + RAW_VITALS + NUMERIC_CARRY
        + point_vars + override_vars + refit_kept))
    for c in carry:
        if c in out.columns:
            continue
        if c not in df.columns:
            print(f"  [warn] carry column '{c}' missing — skipping", flush=True)
            continue
        out[c] = df[c].values

    # ---------------- Assertions ----------------
    banner("Output contract assertions")
    if len(out) != n_total:
        raise SystemExit(f"output length {len(out)} != test slice {n_total}")
    if out[OUTCOME_COL].isna().any():
        raise SystemExit(f"{OUTCOME_COL} has NaN in output")
    if pd.isna(out["pelvic_score"].astype(float)).any():
        raise SystemExit("pelvic_score has NaN — must be integer")
    out["pelvic_score"] = out["pelvic_score"].astype(np.int64)
    out[OVERRIDE_OUT_COL] = out[OVERRIDE_OUT_COL].astype(np.int64)
    assert set(pd.unique(out[OVERRIDE_OUT_COL])) <= {0, 1}, \
        f"{OVERRIDE_OUT_COL} must be 0/1"

    # Re-assert the Fix-4 guard on the EMITTED frame (belt-and-braces): the
    # override column must not be recoverable from pelvic_score.
    emitted_zeroed = _score_from_points(df, points_map, zero_cols=override_vars)
    assert np.array_equal(out["pelvic_score"].to_numpy(),
                          emitted_zeroed.astype(np.int64)), \
        "FATAL — emitted pelvic_score differs from the override-zeroed recompute"

    print(f"  ASSERT pass: N={len(out):,}, pelvic_score int64 no-NaN, outcome int 0/1")
    print(f"  ASSERT pass: {OVERRIDE_OUT_COL} is 0/1 and excluded from pelvic_score")
    print(f"  point vars carried   : {[v for v in point_vars if v in out.columns]}")
    print(f"  override var carried : {[v for v in override_vars if v in out.columns]}")
    print(f"  v3 outcomes carried  : "
          f"{[c for c in OUTCOMES_REQUIRED + OUTCOMES_BINARY_CARRY if c in out.columns]}")

    banner(f"Writing {OUT_PARQUET}")
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote {out.shape[0]:,} rows × {out.shape[1]} cols")
    print("  columns:", list(out.columns))
    print("Step 06 complete.")


if __name__ == "__main__":
    main()
