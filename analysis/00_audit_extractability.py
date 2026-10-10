#!/usr/bin/env python3
"""
00_audit_extractability.py — Arrival/bedside-extractability + coverage audit of
the v3 candidate predictor pool (CONTRACTS.md §5), pelvic arrival tool.

Pattern-matches PAT_TBI/v4/code/00_audit_extractability.py (kept/excluded lists
with explicit rationale + CSV + markdown), but computes REAL coverage on the
built pre-decision cohort: per candidate variable N present / fill-rate /
N-positive / positive-rate, engineered from the cohort's arrival columns.

Documents which candidate columns are truly arrival/bedside-extractable (kept)
and which are excluded and why — chiefly ISS/AIS (registrar-coded injury
severity, not known in the arrival decision moment) and every treatment/outcome
variable (post-decision leakage; FORBIDDEN_X, CONTRACTS §5).

Coverage is computed on the PRIMARY (ring-only, cohort_primary==1) analytic
cohort — the population the arrival score is fit on (expected N ≈ 239,736).

Inputs
------
    derived/pelvic_cohort_predecision.csv   (built by 01_build_cohort.py)

Outputs
-------
    extractability_audit/candidate_audit_table.csv
    extractability_audit/exclusion_rationale.md

Seed: 42.  Idempotent.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# §1. Paths (absolute module root — CONTRACTS §1).
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
AUDIT = os.path.join(config.INPUTS, "extractability_audit")

COHORT_CSV = os.path.join(DERIVED, "pelvic_cohort_predecision.csv")
OUT_CSV = os.path.join(AUDIT, "candidate_audit_table.csv")
OUT_MD = os.path.join(AUDIT, "exclusion_rationale.md")

SEED = 42
np.random.seed(SEED)

# High-energy mechanism = MVC-occupant | Motorcycle | Pedestrian (vs Fall/Other).
HIGH_ENERGY_MECH = {"MVC-occupant", "Motorcycle", "Pedestrian"}

_BEDSIDE = (
    "Arrival/bedside-extractable: measured or reported in the first minutes of "
    "resuscitation (demographic / measured arrival vital / GCS / mechanism / "
    "transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored "
    "ring pattern)."
)

# ---------------------------------------------------------------------------
# Candidate pool (CONTRACTS §5 / SAP §4) — engineered from cohort columns.
# Each entry: name, category, source_cols(list), kind('bin'|'cont'),
#             engineer(df) -> (present_mask, pos_mask_or_None), rationale
# ---------------------------------------------------------------------------
def _present_notna(col):
    return lambda df: (df[col].notna(), None)


CANDIDATES = [
    # --- Demographics ---
    ("age_65_74", "Demographics", ["age"], "bin",
     lambda df: (df["age"].notna(), (df["age"] >= 65) & (df["age"] < 75)), _BEDSIDE),
    ("age_ge75", "Demographics", ["age"], "bin",
     lambda df: (df["age"].notna(), df["age"] >= 75), _BEDSIDE),
    ("sex_female", "Demographics", ["sex_male"], "bin",
     lambda df: (df["sex_male"].notna(), df["sex_male"] == 0), _BEDSIDE),
    # --- Vitals: blood pressure ---
    ("sbp_lt90", "Vitals-BP", ["sbp2"], "bin",
     lambda df: (df["sbp2"].notna(), df["sbp2"] < 90), _BEDSIDE),
    ("sbp_lt70", "Vitals-BP", ["sbp2"], "bin",
     lambda df: (df["sbp2"].notna(), df["sbp2"] < 70), _BEDSIDE),
    ("sbp_unrecordable", "Vitals-BP", ["sbp_unrecordable"], "bin",
     lambda df: (pd.Series(True, index=df.index), df["sbp_unrecordable"] == 1),
     _BEDSIDE + " Unrecordable arrival SBP (sentinel 0) is itself a deployable "
     "high-risk bedside state (MNAR-as-signal)."),
    # --- Vitals: heart rate ---
    ("hr_ge120", "Vitals-HR", ["pulse2"], "bin",
     lambda df: (df["pulse2"].notna(), df["pulse2"] >= 120), _BEDSIDE),
    # --- Shock index ---
    ("shock_index", "Shock index", ["pulse2", "sbp2"], "cont",
     lambda df: (df["shock_index"].notna(), None), _BEDSIDE + " (continuous)."),
    ("si_ge0_9", "Shock index", ["pulse2", "sbp2"], "bin",
     lambda df: (df["shock_index"].notna(), df["shock_index"] >= 0.9), _BEDSIDE),
    ("si_ge1_0", "Shock index", ["pulse2", "sbp2"], "bin",
     lambda df: (df["shock_index"].notna(), df["shock_index"] >= 1.0), _BEDSIDE),
    # --- Neuro (GCS) ---
    ("gcs_motor_le5", "Neuro-GCS", ["motor2"], "bin",
     lambda df: (df["motor2"].notna(), df["motor2"] <= 5), _BEDSIDE),
    ("gcs_le8", "Neuro-GCS", ["gcs2"], "bin",
     lambda df: (df["gcs2"].notna(), df["gcs2"] <= 8), _BEDSIDE),
    # --- Mechanism ---
    ("mech_high_energy", "Mechanism", ["mechanism_cat"], "bin",
     lambda df: (df["mechanism_cat"].notna(), df["mechanism_cat"].isin(HIGH_ENERGY_MECH)),
     _BEDSIDE),
    ("mech_mvc", "Mechanism", ["mechanism_cat"], "bin",
     lambda df: (df["mechanism_cat"].notna(), df["mechanism_cat"] == "MVC-occupant"),
     _BEDSIDE),
    ("mech_fall", "Mechanism", ["mechanism_cat"], "bin",
     lambda df: (df["mechanism_cat"].notna(), df["mechanism_cat"] == "Fall"), _BEDSIDE),
    # --- Transport ---
    ("air_transport", "Transport", ["transport_cat"], "bin",
     lambda df: (df["transport_cat"].notna(), df["transport_cat"] == "Air"), _BEDSIDE),
    # --- Coagulation-relevant comorbidities ---
    ("cc_anticoag", "Comorbidity (coag)", ["cc_anticoag"], "bin",
     lambda df: (pd.Series(True, index=df.index), df["cc_anticoag"] == 1),
     "Anticoagulant use (preexisting code 31) directly changes the "
     "major-hemorrhage-pathway decision; typically known from EMS handoff, "
     "patient report, or MedicAlert. FORCE-IN anchor."),
    ("cc_bleeding", "Comorbidity (coag)", ["cc_bleeding"], "bin",
     lambda df: (pd.Series(True, index=df.index), df["cc_bleeding"] == 1),
     "Known bleeding disorder (preexisting code 4); EMS/patient-reported; "
     "candidate (LASSO may drop if weak)."),
    # --- Anatomy (ICD-anchored ring pattern) ---
    ("unstable_ring", "Anatomy (ICD)", ["unstable_ring"], "bin",
     lambda df: (pd.Series(True, index=df.index), df["unstable_ring"] == 1),
     "Disrupted/unstable pelvic ring (ICD-10 S32.81*) — ICD-anchored rather "
     "than registrar AIS-graded, so robust to AIS miscoding. FORCE-IN anchor."),
]

# ---------------------------------------------------------------------------
# Excluded / FORBIDDEN pool (documented with rationale; coverage still shown).
# ---------------------------------------------------------------------------
_REGISTRAR = (
    "Registrar-coded injury severity — assigned post hoc from the full chart by "
    "a trauma registrar, NOT knowable at arrival; excluded from the bedside "
    "score (offered only as a separate 'full-information' comparator, CONTRACTS "
    "§5). ISS/AIS also risk circularity with the hemorrhage outcome."
)
_LEAKAGE = (
    "Post-decision treatment/outcome variable — using it to predict the "
    "arrival decision is leakage/circular (FORBIDDEN_X, CONTRACTS §5)."
)

EXCLUDED = [
    # registrar-coded severity
    ("iss", "Registrar severity", ["iss"], "cont", _present_notna("iss"), _REGISTRAR),
    ("ais_head", "Registrar severity", ["ais_head"], "cont", _present_notna("ais_head"), _REGISTRAR),
    ("ais_head_ge3", "Registrar severity", ["ais_head_ge3"], "bin",
     lambda df: (df["ais_head_ge3"].notna(), df["ais_head_ge3"] == 1), _REGISTRAR),
    ("ais_chest", "Registrar severity", ["ais_chest"], "cont", _present_notna("ais_chest"), _REGISTRAR),
    ("ais_abd", "Registrar severity", ["ais_abd"], "cont", _present_notna("ais_abd"), _REGISTRAR),
    # treatment / outcome leakage
    ("received_hc", "Treatment/outcome", ["received_hc"], "bin",
     lambda df: (df["received_hc"].notna(), df["received_hc"] == 1), _LEAKAGE),
    ("surg_hc", "Treatment/outcome", ["surg_hc"], "bin",
     lambda df: (df["surg_hc"].notna(), df["surg_hc"] == 1), _LEAKAGE),
    ("angio_hc", "Treatment/outcome", ["angio_hc"], "bin",
     lambda df: (df["angio_hc"].notna(), df["angio_hc"] == 1), _LEAKAGE),
    ("embo_any", "Treatment/outcome", ["embo_any"], "bin",
     lambda df: (df["embo_any"].notna(), df["embo_any"] == 1), _LEAKAGE),
    ("blood_4h", "Treatment/outcome", ["blood_4h"], "cont", _present_notna("blood_4h"), _LEAKAGE),
    ("substantial_transfusion", "Treatment/outcome", ["substantial_transfusion"], "bin",
     lambda df: (df["substantial_transfusion"].notna(), df["substantial_transfusion"] == 1), _LEAKAGE),
    ("death_inhosp", "Treatment/outcome", ["death_inhosp"], "bin",
     lambda df: (df["death_inhosp"].notna(), df["death_inhosp"] == 1),
     _LEAKAGE + " (it is a component of the outcome)."),
    ("major_hemorrhage_event", "Treatment/outcome", ["major_hemorrhage_event"], "bin",
     lambda df: (df["major_hemorrhage_event"].notna(), df["major_hemorrhage_event"] == 1),
     _LEAKAGE + " (this IS the primary outcome)."),
    ("icu_los", "Treatment/outcome", ["icu_los"], "cont", _present_notna("icu_los"), _LEAKAGE),
    ("vent_days", "Treatment/outcome", ["vent_days"], "cont", _present_notna("vent_days"), _LEAKAGE),
]


def _coverage_row(df, name, category, source_cols, kind, engineer, kept, rationale):
    present_mask, pos_mask = engineer(df)
    present_mask = present_mask.fillna(False).astype(bool) if present_mask.dtype != bool else present_mask
    n_total = len(df)
    n_present = int(present_mask.sum())
    fill_pct = round(100.0 * n_present / n_total, 3) if n_total else 0.0
    if kind == "bin" and pos_mask is not None:
        pm = pos_mask.fillna(False).astype(bool) if pos_mask.dtype != bool else pos_mask
        pm = pm & present_mask
        n_pos = int(pm.sum())
        pos_rate = round(100.0 * n_pos / n_present, 3) if n_present else 0.0
        n_pos_out, pos_rate_out = n_pos, pos_rate
    else:  # continuous — positive undefined; report present as the coverage
        n_pos_out, pos_rate_out = "", ""
    return {
        "variable": name,
        "category": category,
        "kept_as_candidate": "Yes" if kept else "No",
        "bedside_extractable": "Yes" if kept else "No",
        "kind": kind,
        "source_columns": "|".join(source_cols),
        "n_total": n_total,
        "n_present": n_present,
        "fill_rate_pct": fill_pct,
        "n_positive": n_pos_out,
        "positive_rate_pct": pos_rate_out,
        "rationale": rationale,
    }


def write_markdown(path, kept_rows, excl_rows, n_analytic, n_total_csv):
    lines = []
    lines.append("# Pelvic Arrival Tool v3 — Extractability & Coverage Audit\n")
    lines.append("## Overview\n")
    lines.append(
        "The v3 arrival risk score is restricted, **before any LASSO**, to a "
        "candidate pool that a clinician can plausibly know in the first minutes "
        "of a pelvic-fracture resuscitation — arrival vitals, GCS, age/sex, "
        "mechanism, transport mode, EMS/MedicAlert-conveyed coagulation "
        "comorbidities, and the ICD-anchored unstable-ring pattern. This "
        "enforces a genuinely *bedside* tool and forecloses post-decision "
        "leakage. Coverage below is computed on the primary (ring-only) "
        f"analytic cohort (`cohort_primary==1`, N = {n_analytic:,}; the full "
        f"acetabulum-inclusive CSV has {n_total_csv:,} rows).\n"
    )
    lines.append(
        "Two categories are deliberately **excluded** from the bedside score: "
        "(1) **registrar-coded injury severity** (ISS and body-region AIS) — "
        "these are abstracted post hoc from the complete chart by a trauma "
        "registrar, are unavailable in the arrival decision window, and risk "
        "circularity with the hemorrhage outcome; they appear only as a "
        "separate 'full-information' comparator model. (2) **All treatment and "
        "outcome variables** (hemorrhage-control receipt, transfusion volume, "
        "death, the composite itself) — these occur *after* the arrival "
        "decision and would leak the label. Both are listed with coverage below "
        "to show the exclusion is on *principle* (timing/leakage), not on "
        "availability.\n"
    )

    def _emit(rows, header):
        lines.append(f"## {header}\n")
        by_cat, order = {}, []
        for r in rows:
            if r["category"] not in by_cat:
                by_cat[r["category"]] = []
                order.append(r["category"])
            by_cat[r["category"]].append(r)
        for cat in order:
            lines.append(f"### {cat}\n")
            lines.append("| variable | fill % | N present | N positive | positive % |")
            lines.append("|---|---|---|---|---|")
            for r in by_cat[cat]:
                lines.append(
                    f"| `{r['variable']}` | {r['fill_rate_pct']} | "
                    f"{r['n_present']:,} | "
                    f"{(f'{r['n_positive']:,}' if r['n_positive'] != '' else '—')} | "
                    f"{(r['positive_rate_pct'] if r['positive_rate_pct'] != '' else '—')} |"
                )
            lines.append("")
            # one shared rationale per category (rationale is category-stable)
            lines.append(f"*Rationale:* {by_cat[cat][0]['rationale']}\n")

    _emit(kept_rows, f"Kept — bedside-extractable candidates ({len(kept_rows)})")
    _emit(excl_rows, f"Excluded — registrar-coded severity & treatment/outcome "
                     f"leakage ({len(excl_rows)})")

    lines.append("## Downstream implication\n")
    lines.append(
        "The kept candidate matrix is the input to `02_engineer_features.py` "
        "(`LASSO_X_COLS`) and `04_lasso_stability.py`, which prune to a "
        "parsimonious 10–15-variable integer score (λ.1se + stability ≥0.85 + "
        "|β|≥0.15 floor), with `age≥65`, `cc_anticoag`, and `unstable_ring` "
        "force-in anchors. `02` must assert "
        "`set(LASSO_X_COLS) ∩ FORBIDDEN_X == ∅` — every excluded variable above "
        "belongs to FORBIDDEN_X.\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(
        description="Audit v3 candidate pool for bedside-extractability + coverage."
    )
    ap.add_argument("--sample", type=int, default=None,
                    help="Optional row downsample (seed 42) for smoke uniformity.")
    args = ap.parse_args()

    os.makedirs(AUDIT, exist_ok=True)
    if not os.path.exists(COHORT_CSV):
        raise FileNotFoundError(
            f"{COHORT_CSV} not found — run 01_build_cohort.py first.")

    print(f"Reading cohort: {COHORT_CSV}", flush=True)
    df_all = pd.read_csv(COHORT_CSV)
    n_total_csv = len(df_all)
    # Coverage is on the primary (ring-only) analytic cohort.
    df = df_all[df_all["cohort_primary"] == 1].copy().reset_index(drop=True)
    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
    n_analytic = len(df)
    print(f"  full CSV rows        : {n_total_csv:,}", flush=True)
    print(f"  analytic (ring-only) : {n_analytic:,}", flush=True)

    kept_rows = [_coverage_row(df, *c[:5], True, c[5]) for c in CANDIDATES]
    excl_rows = [_coverage_row(df, *e[:5], False, e[5]) for e in EXCLUDED]

    audit = pd.DataFrame(kept_rows + excl_rows, columns=[
        "variable", "category", "kept_as_candidate", "bedside_extractable",
        "kind", "source_columns", "n_total", "n_present", "fill_rate_pct",
        "n_positive", "positive_rate_pct", "rationale"])
    audit.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ({len(audit)} rows: "
          f"{len(kept_rows)} kept, {len(excl_rows)} excluded)", flush=True)

    write_markdown(OUT_MD, kept_rows, excl_rows, n_analytic, n_total_csv)
    print(f"Wrote {OUT_MD}", flush=True)

    # stdout summary of the candidate pool
    print("\nCandidate pool coverage (ring-only analytic cohort):")
    for r in kept_rows:
        pos = (f"{r['n_positive']:,} ({r['positive_rate_pct']}%)"
               if r["n_positive"] != "" else "continuous")
        print(f"  {r['variable']:<18} fill={r['fill_rate_pct']:>6}%  "
              f"pos={pos}", flush=True)
    print(f"\nKept candidates : {len(kept_rows)}")
    print(f"Excluded (forbidden) : {len(excl_rows)}")
    print("00_audit_extractability.py complete.", flush=True)


if __name__ == "__main__":
    main()
