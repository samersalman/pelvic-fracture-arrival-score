"""Parity check: score every row of a patient-level parquet with prediction.py and compare
to the stored score, override and tier. For maintainers; NOT run by the tests, and it needs
pandas and pyarrow (not part of the calculator requirements).

The patient-level file is local only and is never part of this repository. Expected raw
input columns (names used by the derivation pipeline):

    age, sex_male, mechanism_cat, sbp2, sbp_unrecordable, pulse2, gcs2, motor2

Expected stored columns:

    pelvic_score, override_nslo, and optionally tier (otherwise the deployed rule is applied)
    major_hemorrhage_event (optional, for the observed tier event rates)

sbp2 and pulse2 hold NaN where the raw value was 0 or missing; the raw SBP is rebuilt as 0
where sbp_unrecordable == 1. The scored test parquet carries no motor2 column, so pass the
candidates parquet with --motor-from and the two files are joined on (puf_year, inc_key).
Without it, motor is rebuilt from the stored gcs_motor_le5 flag (5 if set, else 6), which
checks the item but not the motor cutoff itself.

Usage:
    python calculator/parity_check.py PATH_TO_SCORED.parquet [--motor-from PATH_TO_CANDIDATES.parquet]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import prediction  # noqa: E402

TIER_ORDER = ["rule_out", "observe", "escalate"]


def _none_if_nan(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)


def expected_tier(score: int, override: int, rule_out: float, rule_in: float) -> str:
    """Deployed rule restated from the card: escalate = score >= 5 OR override."""
    if override == 1 or score >= rule_in:
        return "escalate"
    return "observe" if score >= rule_out else "rule_out"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", help="patient-level parquet with raw inputs and stored score columns")
    ap.add_argument("--motor-from", help="parquet that carries motor2 (joined on puf_year, inc_key)")
    ap.add_argument("--sample", type=int, default=0, help="score only a random sample of N rows")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    required = ["age", "sex_male", "mechanism_cat", "sbp2", "sbp_unrecordable", "pulse2", "gcs2",
                "pelvic_score", "override_nslo"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns {missing}")
        return 2

    motor_note = ""
    if "motor2" not in df.columns:
        if args.motor_from:
            cand = pd.read_parquet(args.motor_from, columns=["puf_year", "inc_key", "motor2"])
            n0 = len(df)
            df = df.merge(cand, on=["puf_year", "inc_key"], how="left", validate="1:1")
            assert len(df) == n0
            motor_note = f"motor2 joined from {Path(args.motor_from).name}: the motor cutoff IS verified from raw values"
        elif "gcs_motor_le5" in df.columns:
            df["motor2"] = np.where(df["gcs_motor_le5"] == 1, 5.0, 6.0)
            motor_note = "motor2 rebuilt from the stored gcs_motor_le5 flag: the motor item is checked, the motor cutoff is NOT"
        else:
            print("ERROR: no motor2 column, no --motor-from and no gcs_motor_le5 flag")
            return 2
    else:
        motor_note = "motor2 present in the input file"

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    with open(config.SCORE_CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)
    rule_out, rule_in = float(card["rule_out_cutpoint"]), float(card["rule_in_cutpoint"])

    n = len(df)
    calc_score = np.zeros(n, dtype=np.int64)
    calc_override = np.zeros(n, dtype=np.int64)
    calc_tier = np.empty(n, dtype=object)
    via_validated = 0
    via_lowlevel = 0
    reasons: dict[str, int] = {}

    cols = {c: df[c].to_numpy() for c in ["age", "sex_male", "mechanism_cat", "sbp2", "sbp_unrecordable", "pulse2", "gcs2", "motor2"]}
    for i in range(n):
        sbp = 0.0 if cols["sbp_unrecordable"][i] == 1 else _none_if_nan(cols["sbp2"][i])
        hr = _none_if_nan(cols["pulse2"][i])
        gcs = _none_if_nan(cols["gcs2"][i])
        motor = _none_if_nan(cols["motor2"][i])
        age = _none_if_nan(cols["age"][i])
        sex_male = _none_if_nan(cols["sex_male"][i])
        mech = cols["mechanism_cat"][i]
        raw = {
            "age": age,
            "sex": None if sex_male is None else ("male" if sex_male == 1 else "female"),
            "mechanism": mech,
            "sbp": sbp,
            "hr": hr,
            "gcs": gcs,
            "gcs_motor": motor,
        }
        try:
            res = prediction.score_patient(raw)
            calc_score[i], calc_override[i], calc_tier[i] = res["score"], int(res["override"]), res["tier"]
            via_validated += 1
        except prediction.ValidationError as exc:
            # The derivation scored these rows (a missing value adds 0 points), so score them
            # through the unvalidated derivation to keep the comparison complete.
            for field in exc.errors:
                reasons[field] = reasons.get(field, 0) + 1
            flags, unrec, _ = prediction.derive_flags(age, sex_male == 0 if sex_male is not None else False, mech, sbp, hr, gcs, motor)
            s = prediction.points_total(flags)
            ov = int(bool(unrec))
            calc_score[i], calc_override[i] = s, ov
            calc_tier[i] = prediction.tier_for(s, bool(ov))[0]
            via_lowlevel += 1

    stored_score = df["pelvic_score"].to_numpy().astype(np.int64)
    stored_override = df["override_nslo"].to_numpy().astype(np.int64)
    if "tier" in df.columns:
        stored_tier = df["tier"].to_numpy()
        tier_src = "stored tier column"
    else:
        stored_tier = np.array([expected_tier(int(s), int(o), rule_out, rule_in) for s, o in zip(stored_score, stored_override)], dtype=object)
        tier_src = "deployed rule applied to stored score and override (no tier column in file)"

    ok_score = calc_score == stored_score
    ok_override = calc_override == stored_override
    ok_tier = calc_tier == stored_tier

    print(f"rows scored            : {n:,}")
    print(f"  via score_patient    : {via_validated:,}")
    print(f"  via derive_flags     : {via_lowlevel:,} (rejected by input validation; reasons {reasons or 'none'})")
    print(f"motor input            : {motor_note}")
    print(f"score match            : {int(ok_score.sum()):,} / {n:,} ({100 * ok_score.mean():.4f}%)")
    print(f"override match         : {int(ok_override.sum()):,} / {n:,} ({100 * ok_override.mean():.4f}%)")
    print(f"tier match             : {int(ok_tier.sum()):,} / {n:,} ({100 * ok_tier.mean():.4f}%)   [expected from {tier_src}]")

    if not ok_score.all():
        bad = np.flatnonzero(~ok_score)[:10]
        print("first score mismatches (row, calculator, stored):", [(int(i), int(calc_score[i]), int(stored_score[i])) for i in bad])
    if not ok_tier.all():
        bad = np.flatnonzero(~ok_tier)[:10]
        print("first tier mismatches (row, calculator, stored):", [(int(i), calc_tier[i], stored_tier[i]) for i in bad])

    all_ok = bool(ok_score.all() and ok_override.all() and ok_tier.all())

    if "major_hemorrhage_event" in df.columns:
        y = df["major_hemorrhage_event"].to_numpy().astype(int)
        print("\nObserved event rate by calculator output vs shipped risk_tiers.csv")
        print(f"{'stratum':<34}{'n':>9}{'events':>9}{'rate %':>9}{'csv n':>9}{'csv %':>8}   match")
        score_tier = np.array([prediction.tier_for(int(s), False)[1] for s in calc_score], dtype=object)
        rows = [(f"{t} (score only)", score_tier == t, config.RISK_TIERS[t]) for t in TIER_ORDER]
        rows.append(("override group", calc_override == 1, config.RISK_TIERS["override"]))
        for label, mask, ref in rows:
            m = int(mask.sum())
            ev = int(y[mask].sum())
            rate = 100 * ev / m if m else float("nan")
            same = (m == ref["n"]) and (ev == ref["composite_events"])
            all_ok = all_ok and same
            print(f"{label:<34}{m:>9,}{ev:>9,}{rate:>9.2f}{ref['n']:>9,}{100 * ref['composite_rate']:>8.2f}   {'yes' if same else 'NO'}")
        final = np.array(calc_tier, dtype=object)
        print("\nDeployed tier (score >= 5 or override => escalate), observed event rate:")
        for t in TIER_ORDER:
            mask = final == t
            m = int(mask.sum())
            print(f"  {t:<9} n={m:>7,}  events={int(y[mask].sum()):>6,}  rate={100 * y[mask].mean():.2f}%")

    print("\nPARITY", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
