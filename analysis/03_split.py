"""
Pelvic Arrival Decision-Support Tool (v3) — Step 03: train/test + temporal splits
=====================================================================================

Forks PAT_TBI/v4/code/03_split_train_test.py. Operates on the ANALYTIC cohort
(`cohort_primary == 1`, N = 239,736; CONTRACTS §3 ANALYTIC-COHORT RULE). Produces:

  (1) a stratified (by `major_hemorrhage_event`) 2/3-derivation / 1/3-validation
      random split  ->  train_inc_keys.csv / test_inc_keys.csv, and
  (2) a TEMPORAL split for the external-validation script (12):
        derive = puf_year in {2019, 2020, 2021, 2022}
        valid  = puf_year in {2023, 2024}
      -> temporal_derive_inc_keys.csv / temporal_valid_inc_keys.csv.

Split identity is stored as (puf_year, inc_key) PAIRS, not row indices (CONTRACTS
§7 row 03). inc_key alone is not guaranteed unique across PUF years, so both
columns together are the join key.

Inputs
------
  v3/derived/candidates.parquet                         (Step 02 output)

Outputs
-------
  v3/derived/train_inc_keys.csv                         (puf_year, inc_key)
  v3/derived/test_inc_keys.csv                          (puf_year, inc_key)
  v3/derived/temporal_derive_inc_keys.csv               (puf_year, inc_key; 2019-2022)
  v3/derived/temporal_valid_inc_keys.csv                (puf_year, inc_key; 2023-2024)
  v3/tables/table_split_balance.csv                     (split, N, N_events, event_rate)

Downstream usage pattern (Steps 04, 06, 12)
-------------------------------------------
  cands = pd.read_parquet(CAND_PARQUET)
  cands = cands[cands["cohort_primary"] == 1]            # <-- analytic filter
  train_keys = pd.read_csv(TRAIN_KEYS_CSV)
  train = cands.merge(train_keys, on=["puf_year", "inc_key"], how="inner")

Contract guarantees
-------------------
  * train ∪ test == analytic cohort (cohort_primary == 1); train ∩ test == ∅.
  * Stratified on major_hemorrhage_event so |train_rate - test_rate| < 0.001.
  * temporal_derive ∪ temporal_valid == analytic cohort; disjoint by year.
  * event_rate is a float in [0, 1] (NOT a percentage).

v3 note — the stratifier changed MEANING, not NAME
--------------------------------------------------
`major_hemorrhage_event` keeps its column name but is redefined in v3 as
`received_hc | death_hemorrhagic | substantial_transfusion`, where the death arm
is now `death_inhosp & (ais_head < 4) & (finaldischarge_hrs <= 48)` (Fix 2/2b)
and the transfusion arm uses the unit-aware decoder (Fix 1). CONTRACTS §4.
The `finaldischarge_days <= 2` calendar-day rule is SUPERSEDED — it survives only
in the explicitly-labelled `*_dayrule` sensitivity/crosswalk columns, which this
script never touches.

Consequence for THIS script: the split is stratified on the v3 composite, whose
prevalence is lower than v2's, so the realised train/test event counts will not
match v2's. That is expected and is not a regression. The script is
definition-agnostic — it selects the outcome BY NAME (`OUTCOME_COL`) out of the
parquet, never by column position — so no logic change was required here for the
v3 outcome family.

DO NOT quote prevalence figures from this comment. The authoritative realised
numbers live in artifacts and are regenerated on every run:
  * `logs/qc_manifest.json["primary_composite"]`  — written by 01, the record of
    record for the composite and each of its three arms;
  * `tables/table_split_balance.csv`              — written by THIS script, the
    realised full/train/test N, N_events and event_rate;
and both are printed to stdout below. For orientation only, as of the 01 run
that produced the current cohort CSV: v3 composite 26,479 / 11.045 % (arms:
received_hc 19,583, death_hemorrhagic 4,191, substantial_transfusion 17,156)
against the v2-definition reproduction `major_hemorrhage_event_v2def`
31,054 / 12.953 %. If those disagree with the manifest, the manifest is right
and this paragraph is stale.

Seed: 42.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

# ---- Canonical paths (CONTRACTS §1) ---------------------------------------
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
TEST_KEYS_CSV = os.path.join(DERIVED, "test_inc_keys.csv")
TEMPORAL_DERIVE_CSV = os.path.join(DERIVED, "temporal_derive_inc_keys.csv")
TEMPORAL_VALID_CSV = os.path.join(DERIVED, "temporal_valid_inc_keys.csv")
BALANCE_CSV = os.path.join(TABLES, "table_split_balance.csv")

OUTCOME_COL = "major_hemorrhage_event"
KEY_COLS = ["puf_year", "inc_key"]

TEST_FRAC = 1.0 / 3.0
EVENT_RATE_DIFF_TOL = 0.001            # CONTRACTS/task tolerance
DERIVE_YEARS = [2019, 2020, 2021, 2022]
VALID_YEARS = [2023, 2024]


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def summarize(name, df):
    n = len(df)
    n_events = int(df[OUTCOME_COL].sum())
    rate = n_events / n if n else float("nan")
    print(f"  {name:<16s}  N={n:>9,}   events={n_events:>7,}   rate={rate:7.5f}",
          flush=True)
    return n, n_events, rate


def pairs_of(df):
    """Set of (puf_year, inc_key) integer tuples."""
    return set(map(tuple, df[KEY_COLS].astype("int64").values))


def write_keys(df, path):
    out = df[KEY_COLS].sort_values(KEY_COLS).reset_index(drop=True)
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out):,} rows)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Smoke-test flag (CONTRACTS §9-E), deliberately a "
                             "NO-OP here. This script does NOT resample; it "
                             "reads whatever Step 02 wrote (which already "
                             "honored --sample), so the row-downsample happens "
                             "once, upstream. Downsampling again here would "
                             "write TRUNCATED train/test key files into "
                             "derived/, which a later full run of 04/05/06 "
                             "would silently consume. Flag exists so the "
                             "pipeline driver can forward --sample uniformly.")
    args = parser.parse_args()

    banner("Pelvic v3 — Step 03: stratified train/test + temporal splits (seed 42)")
    if args.sample:
        print(f"  [SMOKE TEST pass-through] --sample={args.sample:,} "
              f"(no downsampling here; relies on Step 02 output)", flush=True)

    if not os.path.exists(CAND_PARQUET):
        sys.exit(f"ERROR: candidates parquet not found: {CAND_PARQUET}\n"
                 f"Run Step 02 (02_engineer_features.py) first.")

    print(f"Reading: {CAND_PARQUET}", flush=True)
    df = pd.read_parquet(
        CAND_PARQUET, columns=KEY_COLS + ["cohort_primary", OUTCOME_COL])
    print(f"  candidates (all rows): {len(df):,}", flush=True)

    # Type coercion
    df["puf_year"] = pd.to_numeric(df["puf_year"], errors="coerce").astype("Int64")
    df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("Int64")
    df["cohort_primary"] = pd.to_numeric(df["cohort_primary"], errors="coerce")
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors="coerce")

    # --------------------------------------------------------------
    # Filter to the ANALYTIC cohort (cohort_primary == 1) — CONTRACTS §3.
    # Acetabulum / non-primary rows are the §13 sensitivity only; they never
    # enter the modeling splits.
    # --------------------------------------------------------------
    n_all = len(df)
    df = df[df["cohort_primary"] == 1].copy()
    print(f"  analytic cohort (cohort_primary==1): {len(df):,} "
          f"(dropped {n_all - len(df):,} non-primary rows)", flush=True)

    # Drop any rows with null key / outcome — stratified split cannot handle NaN.
    n_raw = len(df)
    df = df.dropna(subset=KEY_COLS + [OUTCOME_COL]).copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    if len(df) < n_raw:
        print(f"  dropped {n_raw - len(df):,} rows with null "
              f"puf_year/inc_key/{OUTCOME_COL}", flush=True)

    # Sanity: no duplicate (puf_year, inc_key) in analytic cohort.
    dup_mask = df.duplicated(subset=KEY_COLS, keep=False)
    if dup_mask.any():
        sys.exit(f"ERROR: {int(dup_mask.sum()):,} duplicate (puf_year, inc_key) "
                 f"pairs in analytic cohort — cannot produce a clean split.")

    df = df.reset_index(drop=True)

    # ==============================================================
    # (1) Stratified 2/3-1/3 random split
    # ==============================================================
    banner(f"Stratified split (test_size=1/3, stratify={OUTCOME_COL}, seed 42)")
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_FRAC, stratify=df[OUTCOME_COL], random_state=SEED)
    train = df.loc[train_idx].copy()
    test = df.loc[test_idx].copy()

    banner("Split balance")
    n_full, d_full, r_full = summarize("full (analytic)", df)
    n_tr, d_tr, r_tr = summarize("train", train)
    n_te, d_te, r_te = summarize("test", test)

    # ==============================================================
    # (2) Temporal split (for script 12)
    # ==============================================================
    banner("Temporal split (derive 2019-2022 -> validate 2023-2024)")
    derive = df[df["puf_year"].isin(DERIVE_YEARS)].copy()
    valid = df[df["puf_year"].isin(VALID_YEARS)].copy()
    n_dv, d_dv, r_dv = summarize("temporal_derive", derive)
    n_vl, d_vl, r_vl = summarize("temporal_valid", valid)

    # --------------------------------------------------------------
    # Assertions
    # --------------------------------------------------------------
    banner("Assertions")
    full_pairs = pairs_of(df)
    train_pairs = pairs_of(train)
    test_pairs = pairs_of(test)
    derive_pairs = pairs_of(derive)
    valid_pairs = pairs_of(valid)

    # Random split partition integrity
    assert len(train_pairs) == len(train), \
        f"duplicate (puf_year, inc_key) inside train: {len(train) - len(train_pairs):,}"
    assert len(test_pairs) == len(test), \
        f"duplicate (puf_year, inc_key) inside test: {len(test) - len(test_pairs):,}"
    assert train_pairs.isdisjoint(test_pairs), "train ∩ test non-empty — split leakage"
    assert train_pairs | test_pairs == full_pairs, "train ∪ test != analytic cohort"
    print("  [OK] random: train ∪ test == analytic, train ∩ test == ∅", flush=True)

    # Stratification
    diff = abs(r_tr - r_te)
    assert diff < EVENT_RATE_DIFF_TOL, (
        f"event-rate gap train vs test ({diff:.6f}) exceeds tol {EVENT_RATE_DIFF_TOL}")
    print(f"  [OK] |train_rate - test_rate| = {diff:.6f} < {EVENT_RATE_DIFF_TOL}",
          flush=True)

    # Temporal split partition integrity
    assert derive_pairs.isdisjoint(valid_pairs), \
        "temporal derive ∩ valid non-empty — year overlap"
    assert derive_pairs | valid_pairs == full_pairs, \
        "temporal derive ∪ valid != analytic cohort (a year is unaccounted for)"
    assert n_dv + n_vl == n_full, \
        f"temporal derive N ({n_dv}) + valid N ({n_vl}) != analytic N ({n_full})"
    print(f"  [OK] temporal: derive ∪ valid == analytic; derive({n_dv:,}) + "
          f"valid({n_vl:,}) == {n_full:,}", flush=True)

    # --------------------------------------------------------------
    # Write key files
    # --------------------------------------------------------------
    banner("Writing outputs")
    write_keys(train, TRAIN_KEYS_CSV)
    write_keys(test, TEST_KEYS_CSV)
    write_keys(derive, TEMPORAL_DERIVE_CSV)
    write_keys(valid, TEMPORAL_VALID_CSV)

    balance = pd.DataFrame([
        {"split": "full",  "N": n_full, "N_events": d_full, "event_rate": r_full},
        {"split": "train", "N": n_tr,   "N_events": d_tr,   "event_rate": r_tr},
        {"split": "test",  "N": n_te,   "N_events": d_te,   "event_rate": r_te},
    ])
    balance.to_csv(BALANCE_CSV, index=False)
    print(f"  wrote {BALANCE_CSV}", flush=True)
    with pd.option_context("display.width", 160):
        print(balance.to_string(index=False))

    print("\nPelvic v3 Step 03 complete.", flush=True)


if __name__ == "__main__":
    main()
