#!/usr/bin/env python3
"""
19_d11_invariance_check.py -- v3 / CONTRACTS §13 gate G8f

Demonstrates, rather than asserts, what deviation D-11 (integer-scale anchor
restricted to screen-passing coefficients) did and did not change.

D-11 is the ONLY deviation made after the models were fit, so its defence --
"this changed the bedside rendering, not the model" -- is the most scrutinised
claim in the amendment. This script produces the evidence for it.

Re-anchoring is a RE-QUANTISATION, not a monotone transformation: the new
integer total is not a function of the old one, distinct old totals can collapse
to equal new totals, and ties score 0.5 in the concordance statistic. So the
integer card's discrimination MAY move and must be reported with a signed delta.
The continuous logit, which never touches the integer scale, must not move at all.

  MUST be identical (any movement => the MODEL changed => FAIL):
      beta vector, intercept, variable roles/selection, continuous-logit AUROC
  MAY move (reported, never asserted equal):
      integer pelvic_score AUROC and everything derived from the integer scale

Baseline is the frozen pre-D-11 snapshot, NOT the in-place artifacts -- those
stopped being pre-D-11 the moment the re-run overwrote them.

Reads:  logs/pre_d11_snapshot/{refit.json,table4_discrimination.csv,table3_points.csv}
        derived/refit.json, tables/table4_discrimination.csv, tables/table3_points.csv
Writes: logs/d11_invariance_check.json, logs/d11_invariance_check.csv

Usage:  python3 19_d11_invariance_check.py
"""

import json
import os
import sys

import pandas as pd

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES
LOGS = config.LOGS
SNAP = os.path.join(LOGS, "pre_d11_snapshot")

PRIMARY_OUTCOME = "major_hemorrhage_event"
# CONTRACTS §13 G8f-3: halt if the integer card moves more than this, in EITHER
# direction -- a pure re-quantisation of an unchanged model cannot buy discrimination.
AUROC_TOLERANCE = 0.005

# G8f-1: any movement here means the model changed, not the rendering.
MUST_MATCH_LISTS = ("point_vars", "override_vars", "variables_kept")


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


def load_json(path):
    if not os.path.exists(path):
        sys.exit(f"FATAL: required artifact missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def auroc_row(path, score):
    """AUROC + CI for one instrument on the PRIMARY outcome."""
    if not os.path.exists(path):
        sys.exit(f"FATAL: required artifact missing: {path}")
    df = pd.read_csv(path)
    if "Outcome" in df.columns:
        df = df[df["Outcome"] == PRIMARY_OUTCOME]
    hit = df[df["Score"] == score]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "auroc": float(r["AUROC"]),
        "ci_low": float(r["CI_Low"]) if pd.notna(r.get("CI_Low")) else None,
        "ci_high": float(r["CI_High"]) if pd.notna(r.get("CI_High")) else None,
    }


def main():
    banner("G8f -- D-11 rescaling-only invariance check")
    if not os.path.isdir(SNAP):
        sys.exit(
            f"FATAL: pre-D-11 baseline missing: {SNAP}\n"
            "  It is the ONLY surviving copy of the G8f reference; the in-place\n"
            "  artifacts were overwritten by the re-run. Without it this gate\n"
            "  cannot be evaluated -- and per CONTRACTS §13 a gate that cannot be\n"
            "  evaluated is a FAIL, not a skip."
        )

    old = load_json(os.path.join(SNAP, "refit.json"))
    new = load_json(os.path.join(DERIVED, "refit.json"))
    failures, rows = [], []

    # ---- G8f-1: the model must be untouched --------------------------------
    ob, nb = old.get("coefficients", {}), new.get("coefficients", {})
    if set(ob) != set(nb):
        failures.append(
            f"coefficient VARIABLE SET differs: "
            f"only-pre={sorted(set(ob) - set(nb))} only-post={sorted(set(nb) - set(ob))}"
        )
        max_dbeta = float("nan")
    else:
        max_dbeta = max((abs(ob[k] - nb[k]) for k in ob), default=0.0)
        if max_dbeta > 0:
            failures.append(f"beta vector moved: max |delta beta| = {max_dbeta:.3e}")
    rows.append({
        "class": "must_be_identical", "quantity": f"beta vector ({len(ob)} vars)",
        "pre": "", "post": "", "delta": f"max |d| = {max_dbeta:.3e}",
        "verdict": "PASS" if max_dbeta == 0 else "FAIL",
    })

    o_int, n_int = old.get("intercept"), new.get("intercept")
    d_int = abs(float(o_int) - float(n_int)) if None not in (o_int, n_int) else float("nan")
    if not (d_int == 0):
        failures.append(f"intercept moved by {d_int:.3e}")
    rows.append({
        "class": "must_be_identical", "quantity": "intercept",
        "pre": o_int, "post": n_int, "delta": f"{d_int:.3e}",
        "verdict": "PASS" if d_int == 0 else "FAIL",
    })

    for key in MUST_MATCH_LISTS:
        same = old.get(key) == new.get(key)
        if not same:
            failures.append(f"{key} differs between pre- and post-D-11 refit")
        rows.append({
            "class": "must_be_identical", "quantity": key, "pre": "", "post": "",
            "delta": "identical" if same else "DIFFERS",
            "verdict": "PASS" if same else "FAIL",
        })

    logit_pre = auroc_row(os.path.join(SNAP, "table4_discrimination.csv"), "pelvic_logit")
    logit_post = auroc_row(os.path.join(TABLES, "table4_discrimination.csv"), "pelvic_logit")
    d_logit = logit_post["auroc"] - logit_pre["auroc"]
    if d_logit != 0:
        failures.append(
            f"continuous-logit AUROC moved by {d_logit:+.6f} -- the integer scale "
            "cannot affect it, so the model or the data changed"
        )
    rows.append({
        "class": "must_be_identical", "quantity": "pelvic_logit AUROC",
        "pre": f"{logit_pre['auroc']:.4f}", "post": f"{logit_post['auroc']:.4f}",
        "delta": f"{d_logit:+.5f}", "verdict": "PASS" if d_logit == 0 else "FAIL",
    })

    # ---- G8f-2: the card may move; report it, never assert equality --------
    int_pre = auroc_row(os.path.join(SNAP, "table4_discrimination.csv"), "pelvic_score")
    int_post = auroc_row(os.path.join(TABLES, "table4_discrimination.csv"), "pelvic_score")
    d_int_auroc = int_post["auroc"] - int_pre["auroc"]
    over_tol = abs(d_int_auroc) > AUROC_TOLERANCE
    if over_tol:
        failures.append(
            f"integer-card AUROC moved {d_int_auroc:+.5f}, beyond the "
            f"+/-{AUROC_TOLERANCE} tolerance -- investigate in EITHER direction"
        )
    rows.append({
        "class": "may_move_reported", "quantity": "pelvic_score AUROC (integer card)",
        "pre": f"{int_pre['auroc']:.4f}", "post": f"{int_post['auroc']:.4f}",
        "delta": f"{d_int_auroc:+.5f}",
        "verdict": "WITHIN_TOLERANCE" if not over_tol else "FAIL_OVER_TOLERANCE",
    })

    def card_range(path):
        w = pd.read_csv(path)
        pts = pd.to_numeric(w.loc[w["Role"] == "point", "Points"], errors="coerce").dropna()
        return int(pts[pts < 0].sum()), int(pts[pts > 0].sum())

    pre_lo, pre_hi = card_range(os.path.join(SNAP, "table3_points.csv"))
    post_lo, post_hi = card_range(os.path.join(TABLES, "table3_points.csv"))
    rows.append({
        "class": "may_move_reported", "quantity": "card theoretical point sum",
        "pre": f"{pre_lo}..{pre_hi}", "post": f"{post_lo}..{post_hi}",
        "delta": "rescaled", "verdict": "EXPECTED",
    })

    # Comparators never touch the integer scale -- a harness check that nothing
    # else drifted between the two runs.
    for comp in ("SI-alone", "rSIG", "full_information_ISS_AIS"):
        a = auroc_row(os.path.join(SNAP, "table4_discrimination.csv"), comp)
        b = auroc_row(os.path.join(TABLES, "table4_discrimination.csv"), comp)
        if a and b:
            d = b["auroc"] - a["auroc"]
            if d != 0:
                failures.append(f"comparator {comp} moved by {d:+.6f} -- unrelated drift")
            rows.append({
                "class": "harness_check", "quantity": f"{comp} AUROC",
                "pre": f"{a['auroc']:.4f}", "post": f"{b['auroc']:.4f}",
                "delta": f"{d:+.5f}", "verdict": "PASS" if d == 0 else "FAIL",
            })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(LOGS, "d11_invariance_check.csv"), index=False)
    print(out.to_string(index=False), flush=True)

    verdict = "PASS" if not failures else "FAIL"
    payload = {
        "gate": "G8f",
        "deviation": "D-11 (integer-scale anchor restricted to screen-passing coefficients)",
        "baseline": os.path.join("logs", "pre_d11_snapshot"),
        "auroc_tolerance": AUROC_TOLERANCE,
        "model_identity": {
            "max_abs_delta_beta": max_dbeta,
            "n_coefficients": len(ob),
            "intercept_delta": d_int,
            "pelvic_logit_auroc_pre": logit_pre["auroc"],
            "pelvic_logit_auroc_post": logit_post["auroc"],
            "pelvic_logit_auroc_delta": d_logit,
        },
        "card_rescaling": {
            "pelvic_score_auroc_pre": int_pre["auroc"],
            "pelvic_score_auroc_post": int_post["auroc"],
            "pelvic_score_auroc_delta": d_int_auroc,
            "pelvic_score_ci_post": [int_post["ci_low"], int_post["ci_high"]],
            "theoretical_point_sum_pre": [pre_lo, pre_hi],
            "theoretical_point_sum_post": [post_lo, post_hi],
        },
        "manuscript_sentence": (
            "The model is unchanged -- coefficients, intercept and variable selection are "
            f"identical (max |delta beta| = {max_dbeta:.3g}) and the continuous-logit AUROC is "
            f"identical to four decimal places ({logit_post['auroc']:.4f}). The integer card is a "
            f"coarser rendering of the same model, and its discrimination is {d_int_auroc:+.4f} "
            f"lower ({int_pre['auroc']:.4f} to {int_post['auroc']:.4f}) because the wider anchor "
            "merges score levels and tied levels contribute 0.5 to the concordance statistic."
        ),
        "rows": rows,
        "failures": failures,
        "verdict": verdict,
    }
    with open(os.path.join(LOGS, "d11_invariance_check.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    banner(f"GATE G8f: {verdict}")
    for f in failures:
        print(f"  !! {f}", flush=True)
    print("\n" + payload["manuscript_sentence"], flush=True)
    print("\nWrote logs/d11_invariance_check.{json,csv}", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
