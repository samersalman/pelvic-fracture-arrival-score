"""
Pelvic Arrival Tool v3 — Step 10: Elderly / anticoagulated reclassification (HEADLINE)
================================================================================

The paper's headline differentiator. Five analyses (SAP §7 / plan Headline; v3 adds (5)):

v3 OUTCOME SWEEP (CONTRACTS §9-J) — read before touching any outcome name
-------------------------------------------------------------------------
``major_hemorrhage_event`` KEEPS its name but CHANGES MEANING: it is now
``received_hc | death_hemorrhagic | substantial_transfusion``, and
``patient_state_composite`` is the same composite with HC dropped.

**No definition string is typed into this file.** The ``death_hemorrhagic``
rule has now been wrong twice in emitted artifacts — first as plain
``death_inhosp``, then as a calendar-day rule (``finaldischarge_days <= 2``)
that was superseded by an HOURS rule. ``01_build_cohort.py`` writes the
authoritative definitions into ``logs/qc_manifest.json``; this script READS
them (``outcome_definition()`` below) and HARD-FAILS if the manifest or the
keys are absent. The calendar-day operationalisation survives only as the named
crosswalk ``death_hemorrhagic_dayrule`` and is never the primary.

The mortality COMPONENT reported in the stratified treatment-gap rates is
therefore ``death_hemorrhagic`` (the v3 primary anchor), NOT ``death_inhosp``
(now a named SENSITIVITY — CONTRACTS §4 rung 0). Realized counts/rates for both
live in ``logs/qc_manifest.json → death_variants``; per CONTRACTS §3.1 no script
may hard-code them, so they are not restated here. Both columns are reported so
the treatment-gap narrative can be stated on the right one: ``death`` rows use
``death_hemorrhagic``; ``death_allcause`` rows carry the v2 all-cause arm,
explicitly labelled. ``reclass.json`` records which column is which under
``outcome_columns`` (including a manifest-sourced
``death_primary_definition``) — `16` must read that key rather than assume,
because the "elderly die at ~the same rate as the young" sentence was built on
the ALL-CAUSE rate and may not survive the hemorrhagic restriction.

v3 FIX-3 BOUNDARY (CONTRACTS §6, §13 G3f)
-----------------------------------------
``ADJ_ARRIVAL`` below retains ``unstable_ring``. Analysis (1) is an ASSOCIATION
model (effect modification), not the bedside prediction instrument; adjusting an
association model for a confounder ascertained later in the admission is not a
claim that it was available at the door, and dropping it would leave the
age×anticoagulation interaction confounded by injury pattern. Fix 3 removes the
ring from the SCORE (02/04/05) only. Do not "propagate" it here.

Five analyses:

  1. EFFECT MODIFICATION on the FULL analytic cohort (candidates, cohort_primary==1,
     N=239,736): logistic  major_hemorrhage_event ~ age_ge65 * cc_anticoag + parsimonious
     arrival covariates, HC1 robust SE. Reports the interaction OR AND — the actual story —
     the composite rate AND the HC-receipt rate (+ death, +transfusion) in each stratum
     {young, elderly/no-AC, elderly/AC}. The "treatment gap": elderly die at ~the same rate
     as the young but receive hemorrhage control far less often (composite is deflated by
     WITHHELD treatment, not by lower bleeding physiology). Co-reported on
     patient_state_composite (drops HC — anti-circularity).

  2. CALIBRATION WITHIN ELDERLY (test set): does the frozen score UNDER-predict the elderly?
     CITL (intercept of y ~ offset(lp)) + slope (y ~ lp) + decile observed-vs-predicted, on
     the elderly test subset, using the frozen continuous linear predictor pelvic_lp.

  3. DCA WITHIN ELDERLY (test set): net-benefit of pelvic_score vs SI-alone vs treat-none/
     treat-all across p_t 0.02-0.40 on the elderly test subset. Reports the net-benefit gain
     over SI-alone at the rule-in threshold + net true-positives per 1,000 elderly. Curve
     written to derived/elderly_dca_data.csv (fig-data for F5 elderly panel; schema mirrors
     08's dca_data.csv exactly).

  4. UNDER-TRIAGE HEADLINE (test set): count of elderly composite-event patients with
     observed SI < 0.9 (missed by a shock-index heuristic) BUT pelvic_score >= the
     rule-in cutpoint -> "occult under-triaged bleeders the score reclassifies." Plus a
     categorical NRI of pelvic_score (>= rule-in) vs SI-alone (>=0.9) in the elderly
     (1,000-boot CI). The cutpoint is READ from derived/score_card.json (see Inputs) —
     never assumed; v2's value was 6 and v3's is whatever `08` derives.

  5. RECLASSIFICATION LEDGER (Fix 6 / D4, NEW in v3): the full 2x2 behind the under-triage
     headline, with the COST column visible. Reviewer comment: "'635 additional events detected'
     sounds awesome but cannot be interpreted without knowing how many additional
     interventions the score would generate." Frame = elderly test patients with an OBSERVED
     shock index (the NRI frame). Up-arm = SI<0.9 but score>=rule-in (the score escalates
     where SI reassures): n flagged, events (TP), NON-events (FP), escalations per additional
     event detected, PPV inside the newly-flagged stratum. Down-arm = SI>=0.9 but
     score<rule-in (the score DE-escalates where SI alarms): n and the events it gives up —
     a reclassification claim that shows only the up-arm is not honest. Plus the two
     marginal strategies, the net, and the treat-all-elderly comparator. Wilson 95% CIs on
     every proportion. -> tables/table6b_elderly_ledger.csv (CONTRACTS §8 T6B, §13 G5f).

Change-points are NOT computed here. v3 reads 11's derived/changepoint_stratified.json
(OPTIONAL, soft-fail) purely to carry the elderly / elderly∩anticoag inflection into
reclass.json for the manuscript narrative — including 11's descriptive_only demotions,
which the abstract must honour. CONTRACTS §7 row 10 declares this input.

Inputs (all frozen upstream):
  derived/candidates.parquet          (filter cohort_primary==1) — effect modification + rates
  derived/cohort_test_scored.parquet  (test set, has pelvic_score/pelvic_lp/shock_index/...)
  derived/score_card.json             (REQUIRED: `rule_in_cutpoint`, `rule_out_cutpoint`
                                       — read, never defaulted. v2's values were 6 and 1;
                                       v3's are whatever `08` derives, and a missing key
                                       is a HARD FAIL, not a fallback to the v2 number.)
  logs/qc_manifest.json               (REQUIRED: authoritative outcome definitions, from 01)
  derived/changepoint_stratified.json (from 11; OPTIONAL — warn + continue if absent)

Outputs (OWNED — do not touch any other script's files):
  tables/table6_elderly_reclass.csv   (tidy: interaction OR; stratified rates; calibration;
                                        DCA gain; under-triage; NRI)
  tables/table6b_elderly_ledger.csv   (NEW v3 — the 2x2 reclassification ledger, Fix 6)
  derived/reclass.json                (machine-readable headline numbers)
  derived/elderly_dca_data.csv        (fig-data F5 elderly panel)

Run
---
    python 10_elderly_anticoag_reclassification.py
    python 10_elderly_anticoag_reclassification.py --sample 50000   # smoke

Seed: 42
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
EPS = 1e-6

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = config.DERIVED
TABLES = config.TABLES
LOGS = config.LOGS
os.makedirs(TABLES, exist_ok=True)
os.makedirs(DERIVED, exist_ok=True)

CANDIDATES = os.path.join(DERIVED, "candidates.parquet")
SCORED = os.path.join(DERIVED, "cohort_test_scored.parquet")
CARD_JSON = os.path.join(DERIVED, "score_card.json")
QC_MANIFEST = os.path.join(LOGS, "qc_manifest.json")                          # from 01 (REQUIRED)
CHANGEPOINT_STRAT_JSON = os.path.join(DERIVED, "changepoint_stratified.json")  # from 11 (optional)

OUT_TABLE = os.path.join(TABLES, "table6_elderly_reclass.csv")
OUT_LEDGER = os.path.join(TABLES, "table6b_elderly_ledger.csv")   # NEW v3 — Fix 6 / D4
OUT_JSON = os.path.join(DERIVED, "reclass.json")
OUT_DCA = os.path.join(DERIVED, "elderly_dca_data.csv")

# ---- v3 outcome columns (CONTRACTS §4 / §9-J) -------------------------------
# Composite names are UNCHANGED; their DEFINITIONS changed (see module docstring).
OUTCOME = "major_hemorrhage_event"
COMPOSITE_NOHC = "patient_state_composite"  # death_hemorrhagic | tx (drops HC) — anti-circularity
# The mortality COMPONENT of both composites. v3 primary anchor.
# Its DEFINITION is not written here — see outcome_definition() below; the
# authoritative string lives in logs/qc_manifest.json.
DEATH_COL = "death_hemorrhagic"
# v2's all-cause arm — CONTRACTS §4 rung 0, a NAMED SENSITIVITY. Reported here
# only as an explicitly-labelled comparison row so the treatment-gap sentence
# can be attributed to the right column.
DEATH_COL_ALLCAUSE = "death_inhosp"
TRANSFUSION_COL = "substantial_transfusion"
HC_COL = "received_hc"

# ---------------------------------------------------------------------------
# Authoritative outcome definitions — READ, never typed (CONTRACTS §3.1, §4).
#
# `reclass.json["outcome_columns"]["death_primary_definition"]` exists so `16`
# never has to guess which death rule produced a rate. A literal here has gone
# stale twice (death_inhosp -> calendar-day rule -> hours rule), so the string
# is read at run time from the manifest `01_build_cohort.py` writes for exactly
# this purpose. `14_consort.py` and `render_figures.py` read the same keys.
# Each entry lists key paths in preference order; if NONE resolves, this script
# HARD-FAILS rather than emitting a definition the cohort does not use.
# ---------------------------------------------------------------------------
MANIFEST_DEF_PATHS = {
    "composite": (("primary_composite", "definition"),
                  ("gates", "primary_outcome_definition")),
    "death_hemorrhagic": (("death_variants", "death_hemorrhagic_definition"),
                          ("gates", "mortality_only_anchor_definition")),
}
_DEF_CACHE: dict = {}


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
            f"  reclass.json['outcome_columns']['death_primary_definition'] is "
            f"read by 16 to write a Methods sentence; emitting a hard-coded "
            f"literal here would describe a rule the data does not use. "
            f"Re-run 01_build_cohort.py.")
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


def read_cutpoint(card: dict, key: str) -> int:
    """Read an operating cutpoint from score_card.json. NO DEFAULT, EVER.

    This used to be `card.get("rule_in_cutpoint", 6)` / `.get(..., 1)` — the
    *v2* cutpoints. The enclosing `open(CARD_JSON)` already hard-fails on a
    MISSING card, so those defaults could only ever fire against a
    PARTIALLY-WRITTEN v3 card: exactly the moment a silently-wrong answer is
    most dangerous, because RULE_IN defines every under-triage count, the NRI,
    and every arm of the Fix-6 reclassification ledger.
    """
    if key not in card or card[key] is None:
        raise SystemExit(
            f"FATAL — '{key}' missing (or null) in {CARD_JSON}.\n"
            f"  08_calibration_dca_operating.py owns score_card.json "
            f"(CONTRACTS §7 row 08). There is NO default: the v2 values "
            f"(rule_in 6 / rule_out 1) are not v3's, and silently falling back "
            f"to them would mis-define every under-triage count, the NRI, and "
            f"every ledger arm in this script. Re-run 08.")
    try:
        return int(card[key])
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"FATAL — '{key}' in {CARD_JSON} is not coercible to int "
            f"(got {card[key]!r}). Re-run 08.") from exc


def outcome_definition(which="death_hemorrhagic"):
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

# DCA grid — mirror 08 (p_t 0.02-0.40).
THRESHOLDS = np.round(np.arange(0.02, 0.4001, 0.01), 4)
# Named "rule-in / escalation" threshold for the headline DCA gain (above ~13% prevalence;
# a resource-intensive but life-saving escalation is warranted at moderate predicted risk).
RULE_IN_PT = 0.20
DCA_REPORT_PTS = [0.10, 0.15, 0.20]

# Parsimonious adjustment set for the effect-modification ASSOCIATION model (all complete /
# no NaN in cohort; excludes age & anticoag which are the interaction terms; excludes
# forbidden registrar severity iss/ais).
#
# `unstable_ring` STAYS (CONTRACTS §6 asymmetry / §13 G3f). Fix 3 removed it from the SCORE
# because a bedside prediction instrument may use only door-available information; this model
# is not a prediction instrument, and dropping the ring here would confound the
# age×anticoagulation interaction by injury pattern. Do NOT propagate Fix 3 into this list.
ADJ_ARRIVAL = ["sbp_lt90", "hr_ge120", "unstable_ring", "sex_female", "mech_high_energy", "gcs_le8"]
assert "unstable_ring" in ADJ_ARRIVAL, "CONTRACTS §6/§13 G3f: ring stays in the association model"

# Shock-index heuristic the score is reclassified against (clinical comparator).
SI_HEURISTIC = 0.9


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ------------------------------------------------------------------ helpers (copied/adapted 08)
def wilson(count, nobs):
    if nobs == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(count=int(count), nobs=int(nobs), alpha=0.05, method="wilson")
    return float(lo), float(hi)


def logistic_recalibrate(y, score):
    """Recalibrate a raw score to probability (logit link) on the SAME rows — mirrors 08."""
    X = sm.add_constant(np.asarray(score, dtype=float), has_constant="add")
    m = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="lbfgs", maxiter=200)
    return np.clip(m.predict(X), EPS, 1 - EPS)


def net_benefit(y, prob, p_t):
    """Vickers-Elkin net benefit at threshold p_t (net TP proportion). ×1000 = net TP/1,000."""
    n = len(y)
    if n == 0 or p_t >= 1.0:
        return np.nan
    pred_pos = prob >= p_t
    tp = float((pred_pos & (y == 1)).sum()) / n
    fp = float((pred_pos & (y == 0)).sum()) / n
    return tp - fp * (p_t / (1.0 - p_t))


def nb_treat_all(prev, p_t):
    return np.nan if p_t >= 1.0 else prev - (1.0 - prev) * (p_t / (1.0 - p_t))


def calib_citl_slope(y, lp):
    """CITL = intercept of y ~ offset(lp) (slope fixed at 1; alpha>0 => model UNDER-predicts).
       Slope = beta of y ~ lp (beta<1 => predictions too extreme)."""
    y = np.asarray(y, dtype=float)
    lp = np.asarray(lp, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lp)
    y, lp = y[ok], lp[ok]
    ones = np.ones_like(lp)
    m_citl = sm.GLM(y, ones, family=sm.families.Binomial(), offset=lp).fit()
    citl = float(m_citl.params[0]); cci = m_citl.conf_int()
    X = sm.add_constant(lp, has_constant="add")
    m_slope = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    slope = float(m_slope.params[1]); sci = m_slope.conf_int()
    return {"citl": citl, "citl_lo": float(cci[0][0]), "citl_hi": float(cci[0][1]),
            "slope": slope, "slope_lo": float(sci[1][0]), "slope_hi": float(sci[1][1]),
            "n": int(len(y))}


# Component columns behind each reported rate. `death` is the v3 hemorrhagic anchor
# (CONTRACTS §4); `death_allcause` is v2's arm, retained as an explicitly-labelled
# sensitivity row so the treatment-gap sentence is attributable (CONTRACTS §9-J).
RATE_COMPONENTS = [("hc_receipt", HC_COL),
                   ("death", DEATH_COL),
                   ("death_allcause", DEATH_COL_ALLCAUSE),
                   ("transfusion", TRANSFUSION_COL)]


def rate_row(section, stratum, mask, df, comp_col=OUTCOME):
    """Composite / HC / death / transfusion rates (+ Wilson CI) in one stratum."""
    sub = df.loc[mask]
    n = int(len(sub))
    out = {"Section": section, "Stratum": stratum, "N": n}
    for lbl, col in [("composite", comp_col)] + RATE_COMPONENTS:
        if col in sub.columns and n:
            ev = int(pd.to_numeric(sub[col], errors="coerce").fillna(0).sum())
            r = ev / n
            lo, hi = wilson(ev, n)
            out[f"{lbl}_rate"] = r; out[f"{lbl}_events"] = ev
            out[f"{lbl}_ci_low"] = lo; out[f"{lbl}_ci_high"] = hi
        else:
            out[f"{lbl}_rate"] = np.nan; out[f"{lbl}_events"] = np.nan
            out[f"{lbl}_ci_low"] = np.nan; out[f"{lbl}_ci_high"] = np.nan
    return out


def ledger_row(arm, definition, n_denom, flagged_mask, y, note=""):
    """One row of the Fix-6 reclassification ledger (CONTRACTS §8 T6B).

    ``flagged_mask`` = who this strategy escalates within ``n_denom``. Reports the
    2x2 cell counts, the WORKLOAD (escalations), the COST (false flags), the
    number of escalations bought per event detected, and Wilson 95% CIs on every
    proportion. The requirement (D4) is that the cost column is visible
    on the same line as the benefit — so PPV and escalations-per-event sit
    beside the event count, never on a separate table.
    """
    flagged_mask = np.asarray(flagged_mask, dtype=bool)
    y = np.asarray(y).astype(int)
    n_flag = int(flagged_mask.sum())
    tp = int((flagged_mask & (y == 1)).sum())
    fp = int((flagged_mask & (y == 0)).sum())
    n_ev = int(y.sum())
    ppv_lo, ppv_hi = wilson(tp, n_flag)
    frac_lo, frac_hi = wilson(n_flag, n_denom)
    ev_lo, ev_hi = wilson(n_ev, n_denom)
    return {
        "Arm": arm,
        "Definition": definition,
        "N_denominator": int(n_denom),
        "N_events_in_denominator": n_ev,
        "Event_rate_in_denominator_pct": 100.0 * n_ev / n_denom if n_denom else np.nan,
        "Event_rate_CI_low_pct": 100.0 * ev_lo if n_denom else np.nan,
        "Event_rate_CI_high_pct": 100.0 * ev_hi if n_denom else np.nan,
        "N_flagged": n_flag,
        "Flagged_pct_of_denominator": 100.0 * n_flag / n_denom if n_denom else np.nan,
        "Flagged_pct_CI_low": 100.0 * frac_lo if n_denom else np.nan,
        "Flagged_pct_CI_high": 100.0 * frac_hi if n_denom else np.nan,
        "Flagged_events_TP": tp,
        "Flagged_nonevents_FP": fp,
        "PPV_within_flagged_pct": 100.0 * tp / n_flag if n_flag else np.nan,
        "PPV_CI_low_pct": 100.0 * ppv_lo if n_flag else np.nan,
        "PPV_CI_high_pct": 100.0 * ppv_hi if n_flag else np.nan,
        "Escalations_per_event": n_flag / tp if tp else np.nan,
        "Note": note,
    }


LEDGER_COLS = ["Arm", "Definition", "N_denominator", "N_events_in_denominator",
               "Event_rate_in_denominator_pct", "Event_rate_CI_low_pct",
               "Event_rate_CI_high_pct", "N_flagged", "Flagged_pct_of_denominator",
               "Flagged_pct_CI_low", "Flagged_pct_CI_high", "Flagged_events_TP",
               "Flagged_nonevents_FP", "PPV_within_flagged_pct", "PPV_CI_low_pct",
               "PPV_CI_high_pct", "Escalations_per_event", "Note"]


def read_changepoint_tiein():
    """Carry 11's elderly change-points into reclass.json (CONTRACTS §7 row 10).

    OPTIONAL BY DESIGN. 11 is Batch 3 and 10 is Batch 4, so the file should exist —
    but `10`'s own four analyses do not depend on it, and hard-failing here would
    make the headline script hostage to the longest-running script in the pipeline.
    Warn and continue instead.

    What matters downstream is not just the breakpoint but 11's
    ``descriptive_only`` demotions: ``death_hemorrhagic`` carries roughly a
    third of v2's ``death_inhosp`` events (exact counts:
    ``logs/qc_manifest.json → death_variants``), so the stratified
    mortality-only change-points — elderly∩anticoagulated above all — can fall
    below the locked SAP's >=100-events-per-side guard. A demoted stratum may be
    described but must NOT be quoted as an estimated inflection point in the
    abstract or manuscript.

    KEY FILTER: `11` writes a top-level ``_guard_summary`` block into this file
    alongside the per-tag stratum blocks. Keys beginning with "_" are metadata,
    never curves — iterate past them explicitly (as ``render_figures`` does)
    rather than relying on them happening to lack a ``"strata"`` key.
    """
    if not os.path.exists(CHANGEPOINT_STRAT_JSON):
        print(f"  [warn] {CHANGEPOINT_STRAT_JSON} not found — change-point tie-in skipped "
              "(11 has not run yet). reclass.json records the absence.")
        return {"available": False,
                "source": os.path.relpath(CHANGEPOINT_STRAT_JSON, config.REPO_ROOT),
                "note": "11_pelvic_thresholds.py had not written this artifact when 10 ran."}
    try:
        with open(CHANGEPOINT_STRAT_JSON) as f:
            strat = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not parse {CHANGEPOINT_STRAT_JSON}: {exc}")
        return {"available": False, "source": os.path.relpath(CHANGEPOINT_STRAT_JSON, config.REPO_ROOT), "error": str(exc)}

    want = ("elderly", "young", "elderly_anticoag")
    out = {"available": True, "source": os.path.relpath(CHANGEPOINT_STRAT_JSON, config.REPO_ROOT), "curves": {}}
    demoted = []
    skipped_meta = []
    for tag, block in strat.items():
        # `_`-prefixed keys are METADATA blocks written by 11 (`_guard_summary`),
        # not change-point curves. Skipping them explicitly — the same filter
        # render_figures.py applies — instead of relying on their shape.
        if str(tag).startswith("_"):
            skipped_meta.append(str(tag))
            continue
        if not isinstance(block, dict):
            continue
        cells = (block.get("strata") or {})
        keep = {}
        for lbl in want:
            e = cells.get(lbl)
            if not isinstance(e, dict):
                continue
            keep[lbl] = {k: e.get(k) for k in
                         ("n", "events", "breakpoint", "lci", "uci",
                          "descriptive_only", "guard_reason", "note")}
            if e.get("descriptive_only"):
                demoted.append(f"{tag}/{lbl}")
        if keep:
            out["curves"][tag] = {"strata": keep,
                                  "directional_hypothesis": block.get("directional_hypothesis"),
                                  "interaction_lr_age": (block.get("interaction_lr") or {}).get("age_ge65")}
    out["descriptive_only_strata"] = sorted(set(demoted))
    out["metadata_keys_skipped"] = sorted(skipped_meta)
    # 11's `_guard_summary` is the authoritative demotion record; carry it verbatim
    # so a reader of reclass.json can see the guard verdict without opening 11's file.
    if isinstance(strat.get("_guard_summary"), dict):
        out["guard_summary"] = strat["_guard_summary"]
    out["note"] = ("Strata listed in descriptive_only_strata FAILED the locked SAP's "
                   ">=100-events-per-side change-point guard and must be reported "
                   "descriptive-only — never quoted as an estimated inflection point "
                   "(CONTRACTS §13 G5e).")
    print(f"  change-point tie-in: {len(out['curves'])} curve(s); "
          f"descriptive-only strata: {out['descriptive_only_strata'] or 'none'}"
          + (f"; metadata keys skipped: {out['metadata_keys_skipped']}"
             if skipped_meta else ""))
    return out


def categorical_nri(y, score_flag, si_flag):
    """2-category NRI: score classifier vs SI classifier. Returns dict of components + counts."""
    y = np.asarray(y).astype(int)
    sf = np.asarray(score_flag).astype(int)
    qf = np.asarray(si_flag).astype(int)
    ev = y == 1; nev = y == 0
    n_ev, n_nev = int(ev.sum()), int(nev.sum())
    # Among events: up = score flags but SI missed (good); down = SI flagged but score missed (bad).
    up_ev = int(((sf == 1) & (qf == 0) & ev).sum())
    down_ev = int(((sf == 0) & (qf == 1) & ev).sum())
    # Among non-events: up = score falsely flags (bad); down = score correctly de-flags (good).
    up_nev = int(((sf == 1) & (qf == 0) & nev).sum())
    down_nev = int(((sf == 0) & (qf == 1) & nev).sum())
    nri_ev = (up_ev - down_ev) / n_ev if n_ev else np.nan
    nri_nev = (down_nev - up_nev) / n_nev if n_nev else np.nan
    return {"nri": (nri_ev + nri_nev) if (n_ev and n_nev) else np.nan,
            "nri_event": nri_ev, "nri_nonevent": nri_nev,
            "n_event": n_ev, "n_nonevent": n_nev,
            "event_up": up_ev, "event_down": down_ev,
            "nonevent_up": up_nev, "nonevent_down": down_nev}


# ============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="Downsample rows (smoke).")
    args = ap.parse_args()

    t0 = time.time()
    banner("Step 10 — Elderly / anticoagulated reclassification (HEADLINE)")

    with open(CARD_JSON) as f:
        card = json.load(f)

    # ---- cutpoints: READ from the card, never defaulted (see read_cutpoint) --
    RULE_IN = read_cutpoint(card, "rule_in_cutpoint")
    RULE_OUT = read_cutpoint(card, "rule_out_cutpoint")
    print(f"  rule_in_cutpoint (score>={RULE_IN})   rule_out_cutpoint (score<{RULE_OUT})"
          f"   [read from {os.path.basename(CARD_JSON)}; no defaults]")

    # Resolve (and cache) the authoritative outcome definitions NOW, so a missing
    # manifest fails before ~20 minutes of modelling rather than at the JSON write.
    outcome_definition("composite")
    outcome_definition("death_hemorrhagic")

    table_rows = []
    js = {"seed": SEED, "rule_in_cutpoint": RULE_IN, "rule_out_cutpoint": RULE_OUT}

    # ----------------------------------------------------------------------------------------
    # (1) EFFECT MODIFICATION + STRATIFIED TREATMENT-GAP RATES  (FULL cohort)
    # ----------------------------------------------------------------------------------------
    banner("(1) Effect modification + treatment-gap rates  (candidates, cohort_primary==1)")
    full = pd.read_parquet(CANDIDATES)
    full = full[full["cohort_primary"] == 1].copy()
    if args.sample and args.sample < len(full):
        full = full.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE] full cohort -> {len(full):,}")
    for c in ["age_ge65", "cc_anticoag", OUTCOME, COMPOSITE_NOHC] + ADJ_ARRIVAL:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    print(f"  N={len(full):,}  composite={full[OUTCOME].mean():.4f}  "
          f"HC={full['received_hc'].mean():.4f}")

    # ---- interaction models (HC1 robust SE) ----
    def fit_interaction(outcome_col):
        f = f"{outcome_col} ~ age_ge65 * cc_anticoag + " + " + ".join(ADJ_ARRIVAL)
        m = smf.glm(f, data=full, family=sm.families.Binomial()).fit(cov_type="HC1")
        term = "age_ge65:cc_anticoag"
        beta = float(m.params[term]); se = float(m.bse[term]); p = float(m.pvalues[term])
        ci = m.conf_int().loc[term]
        return {"formula": f, "beta": beta, "or": float(np.exp(beta)),
                "or_ci_low": float(np.exp(ci[0])), "or_ci_high": float(np.exp(ci[1])),
                "se": se, "p": p,
                "or_age_ge65": float(np.exp(m.params["age_ge65"])),
                "or_cc_anticoag": float(np.exp(m.params["cc_anticoag"])),
                "n": int(m.nobs)}

    em_primary = fit_interaction(OUTCOME)
    em_nohc = fit_interaction(COMPOSITE_NOHC)
    js["interaction_or"] = {
        "outcome": OUTCOME, "or": em_primary["or"],
        "ci_low": em_primary["or_ci_low"], "ci_high": em_primary["or_ci_high"],
        "beta": em_primary["beta"], "se": em_primary["se"], "p": em_primary["p"],
        "or_age_ge65_main": em_primary["or_age_ge65"],
        "or_cc_anticoag_main": em_primary["or_cc_anticoag"],
        "n": em_primary["n"], "adjustment": ADJ_ARRIVAL}
    js["interaction_or_patient_state_composite"] = {
        "outcome": COMPOSITE_NOHC, "or": em_nohc["or"],
        "ci_low": em_nohc["or_ci_low"], "ci_high": em_nohc["or_ci_high"], "p": em_nohc["p"]}
    print(f"  interaction OR (age_ge65 x cc_anticoag) on {OUTCOME}: "
          f"{em_primary['or']:.3f} [{em_primary['or_ci_low']:.3f},{em_primary['or_ci_high']:.3f}] "
          f"p={em_primary['p']:.3g}  (N={em_primary['n']:,})")
    print(f"    main OR age_ge65={em_primary['or_age_ge65']:.3f}  "
          f"cc_anticoag={em_primary['or_cc_anticoag']:.3f}")
    print(f"  interaction OR on {COMPOSITE_NOHC} (anti-circularity): "
          f"{em_nohc['or']:.3f} [{em_nohc['or_ci_low']:.3f},{em_nohc['or_ci_high']:.3f}] "
          f"p={em_nohc['p']:.3g}")

    table_rows.append({"Section": "EffectModification", "Metric": "interaction_OR_composite",
                       "Stratum": "age_ge65:cc_anticoag", "Value": em_primary["or"],
                       "CI_Low": em_primary["or_ci_low"], "CI_High": em_primary["or_ci_high"],
                       "N": em_primary["n"], "Note": f"HC1; p={em_primary['p']:.3g}"})
    table_rows.append({"Section": "EffectModification", "Metric": "interaction_OR_patient_state",
                       "Stratum": "age_ge65:cc_anticoag", "Value": em_nohc["or"],
                       "CI_Low": em_nohc["or_ci_low"], "CI_High": em_nohc["or_ci_high"],
                       "N": em_nohc["n"], "Note": f"drops HC (anti-circ); p={em_nohc['p']:.3g}"})
    table_rows.append({"Section": "EffectModification", "Metric": "main_OR_age_ge65",
                       "Stratum": "age_ge65", "Value": em_primary["or_age_ge65"],
                       "CI_Low": np.nan, "CI_High": np.nan, "N": em_primary["n"], "Note": "composite"})
    table_rows.append({"Section": "EffectModification", "Metric": "main_OR_cc_anticoag",
                       "Stratum": "cc_anticoag", "Value": em_primary["or_cc_anticoag"],
                       "CI_Low": np.nan, "CI_High": np.nan, "N": em_primary["n"], "Note": "composite"})

    # ---- stratified treatment-gap rates ----
    strata = {
        "young(<65)": full["age_ge65"] == 0,
        "elderly/no-AC": (full["age_ge65"] == 1) & (full["cc_anticoag"] == 0),
        "elderly/AC": (full["age_ge65"] == 1) & (full["cc_anticoag"] == 1),
        "elderly(all)": full["age_ge65"] == 1,
    }
    stratum_rates = {}
    metric_cols = {"composite": OUTCOME, **dict(RATE_COMPONENTS)}
    print("\n  Stratified rates (treatment gap = composite high but HC-receipt low in elderly):")
    print(f"  death = {DEATH_COL} (v3 anchor); death_allcause = {DEATH_COL_ALLCAUSE} (v2 sensitivity)")
    print(f"  {'stratum':16s} {'N':>8s} {'composite':>10s} {'HC':>8s} {'death':>8s} "
          f"{'death_ac':>9s} {'tx':>8s}")
    for lbl, m in strata.items():
        rr = rate_row("StratifiedRates", lbl, m, full)
        # flat table rows (one metric per line)
        for metric, col in metric_cols.items():
            table_rows.append({"Section": "StratifiedRates", "Metric": f"{metric}_rate",
                               "Stratum": lbl, "Value": rr[f"{metric}_rate"],
                               "CI_Low": rr[f"{metric}_ci_low"], "CI_High": rr[f"{metric}_ci_high"],
                               "N": rr["N"],
                               "Note": f"events={rr[f'{metric}_events']}; col={col}"})
        stratum_rates[lbl] = {"n": rr["N"],
                              "composite": rr["composite_rate"], "hc_receipt": rr["hc_receipt_rate"],
                              "death": rr["death_rate"],
                              "death_allcause": rr["death_allcause_rate"],
                              "transfusion": rr["transfusion_rate"]}
        print(f"  {lbl:16s} {rr['N']:>8,} {rr['composite_rate']:>10.4f} "
              f"{rr['hc_receipt_rate']:>8.4f} {rr['death_rate']:>8.4f} "
              f"{rr['death_allcause_rate']:>9.4f} {rr['transfusion_rate']:>8.4f}")
    js["stratum_rates"] = stratum_rates
    # Provenance so `16` never has to guess which death column a rate came from (§9-J).
    # Both definition strings are READ from logs/qc_manifest.json (written by
    # 01), never typed here — see outcome_definition(). `16` consumes
    # `death_primary_definition` to write a Methods sentence, so a stale literal
    # here would describe a rule the cohort does not use. Hard-fails if absent.
    _composite_def = outcome_definition("composite")
    _death_def = outcome_definition("death_hemorrhagic")
    js["outcome_columns"] = {
        "composite": OUTCOME,
        "composite_definition": _composite_def,
        "composite_nohc": COMPOSITE_NOHC,
        "death_primary": DEATH_COL,
        "death_primary_definition": _death_def,
        "death_sensitivity_allcause": DEATH_COL_ALLCAUSE,
        "transfusion": TRANSFUSION_COL,
        "hc": HC_COL,
        "definition_source": (
            "logs/qc_manifest.json (01_build_cohort.py) — read at run time, "
            "never hard-coded in 10_elderly_anticoag_reclassification.py"),
        "note": ("v3 CONTRACTS §4/§9-J: composite names unchanged, definitions changed. "
                 "'death' rows are the HEMORRHAGIC anchor; 'death_allcause' rows are v2's "
                 "arm, a named sensitivity only.")}
    # Coherence guard: the manifest's composite expression must actually name the
    # death column this script attributes its `death` rows to. If 01's composite
    # ever stops containing DEATH_COL, every `death`/`composite` sentence built
    # from this file is mis-attributed.
    if DEATH_COL not in _composite_def:
        raise SystemExit(
            f"FATAL — the manifest composite definition ({_composite_def!r}) does "
            f"not name '{DEATH_COL}'. 10 attributes its 'death' rows to that "
            f"column; the cohort build and this script disagree (CONTRACTS §4).")
    print(f"  outcome_columns.composite_definition      = {_composite_def}")
    print(f"  outcome_columns.death_primary_definition  = {_death_def}")
    # explicit treatment-gap summary (HC young vs elderly; death young vs elderly)
    js["treatment_gap"] = {
        "death_column": DEATH_COL,
        "hc_young": stratum_rates["young(<65)"]["hc_receipt"],
        "hc_elderly": stratum_rates["elderly(all)"]["hc_receipt"],
        "hc_elderly_AC": stratum_rates["elderly/AC"]["hc_receipt"],
        "hc_ratio_young_over_elderly": (stratum_rates["young(<65)"]["hc_receipt"] /
                                        stratum_rates["elderly(all)"]["hc_receipt"]
                                        if stratum_rates["elderly(all)"]["hc_receipt"] else np.nan),
        "death_young": stratum_rates["young(<65)"]["death"],
        "death_elderly": stratum_rates["elderly(all)"]["death"],
        "death_allcause_young": stratum_rates["young(<65)"]["death_allcause"],
        "death_allcause_elderly": stratum_rates["elderly(all)"]["death_allcause"],
        "composite_young": stratum_rates["young(<65)"]["composite"],
        "composite_elderly": stratum_rates["elderly(all)"]["composite"]}
    print(f"\n  TREATMENT GAP: HC young={js['treatment_gap']['hc_young']:.4f} vs "
          f"elderly={js['treatment_gap']['hc_elderly']:.4f} "
          f"({js['treatment_gap']['hc_ratio_young_over_elderly']:.2f}x)")
    print(f"    {DEATH_COL:24s} young={js['treatment_gap']['death_young']:.4f} vs "
          f"elderly={js['treatment_gap']['death_elderly']:.4f}")
    print(f"    {DEATH_COL_ALLCAUSE:24s} young={js['treatment_gap']['death_allcause_young']:.4f} vs "
          f"elderly={js['treatment_gap']['death_allcause_elderly']:.4f}")
    print("    ^ the 'elderly die at ~the same rate as the young' sentence in the manuscript "
          "must state WHICH of these two it is built on (CONTRACTS §9-J).")

    # ----------------------------------------------------------------------------------------
    # Load TEST set for (2)-(4)
    # ----------------------------------------------------------------------------------------
    test = pd.read_parquet(SCORED)
    if args.sample and args.sample < len(test):
        test = test.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE] test set -> {len(test):,}")
    test["age_ge65"] = pd.to_numeric(test["age_ge65"], errors="coerce")
    el = test[test["age_ge65"] == 1].copy().reset_index(drop=True)
    y_el = el[OUTCOME].astype(int).to_numpy()
    el_prev = float(y_el.mean())
    print(f"\n  ELDERLY TEST subset: N={len(el):,}  composite={el_prev:.4f} "
          f"({int(y_el.sum()):,} events)")
    js["elderly_test_n"] = int(len(el))
    js["elderly_test_prevalence"] = el_prev
    js["elderly_test_events"] = int(y_el.sum())

    # ----------------------------------------------------------------------------------------
    # (2) CALIBRATION WITHIN ELDERLY (frozen linear predictor pelvic_lp)
    # ----------------------------------------------------------------------------------------
    banner("(2) Calibration within elderly (frozen pelvic_lp -> does the score UNDER-predict?)")
    lp_el = el["pelvic_lp"].to_numpy(dtype=float)
    p_el = 1.0 / (1.0 + np.exp(-lp_el))
    cal = calib_citl_slope(y_el, lp_el)
    mean_pred = float(np.clip(p_el, EPS, 1 - EPS).mean())
    mean_obs = float(y_el.mean())
    cal.update({"mean_predicted": mean_pred, "mean_observed": mean_obs,
                "mean_pred_minus_obs": mean_pred - mean_obs})
    verdict = ("UNDER-predicts" if cal["citl"] > 0.10 else
               ("over-predicts" if cal["citl"] < -0.10 else "well-calibrated"))
    cal["verdict"] = verdict
    js["elderly_calibration"] = cal
    print(f"  CITL (intercept y~offset(lp)) = {cal['citl']:+.4f} "
          f"[{cal['citl_lo']:+.4f},{cal['citl_hi']:+.4f}]  (alpha>0 => under-predicts)")
    print(f"  Slope (y~lp)                  = {cal['slope']:.4f} "
          f"[{cal['slope_lo']:.4f},{cal['slope_hi']:.4f}]")
    print(f"  mean predicted={mean_pred:.4f}  mean observed={mean_obs:.4f}  "
          f"(pred-obs={mean_pred-mean_obs:+.4f}) -> score {verdict} the elderly")

    # decile observed-vs-predicted (fig-data for F4 elderly slice -> lives in reclass.json)
    order = np.argsort(p_el)
    deciles = np.array_split(order, 10)
    decile_rows = []
    for i, idx in enumerate(deciles, 1):
        if len(idx) == 0:
            continue
        pr = float(p_el[idx].mean()); ob = float(y_el[idx].mean())
        lo, hi = wilson(int(y_el[idx].sum()), len(idx))
        decile_rows.append({"decile": i, "n": int(len(idx)), "mean_pred": pr, "obs_frac": ob,
                            "obs_ci_low": lo, "obs_ci_high": hi})
    js["elderly_calibration_deciles"] = decile_rows
    table_rows.append({"Section": "ElderlyCalibration", "Metric": "CITL_intercept", "Stratum": "elderly",
                       "Value": cal["citl"], "CI_Low": cal["citl_lo"], "CI_High": cal["citl_hi"],
                       "N": cal["n"], "Note": "alpha>0 => under-predicts"})
    table_rows.append({"Section": "ElderlyCalibration", "Metric": "calibration_slope", "Stratum": "elderly",
                       "Value": cal["slope"], "CI_Low": cal["slope_lo"], "CI_High": cal["slope_hi"],
                       "N": cal["n"], "Note": "1.0=ideal"})
    table_rows.append({"Section": "ElderlyCalibration", "Metric": "mean_predicted", "Stratum": "elderly",
                       "Value": mean_pred, "CI_Low": np.nan, "CI_High": np.nan, "N": cal["n"],
                       "Note": f"mean_observed={mean_obs:.4f}; {verdict}"})

    # ----------------------------------------------------------------------------------------
    # (3) DCA WITHIN ELDERLY (net benefit; score vs SI-alone vs treat-none/all)
    # ----------------------------------------------------------------------------------------
    banner("(3) DCA within elderly (p_t 0.02-0.40; gain over SI-alone at rule-in threshold)")
    # per-score non-NaN subset (mirror 08). pelvic_score always defined; SI drops ~missing.
    sc_raw = pd.to_numeric(el["pelvic_score"], errors="coerce").to_numpy(dtype=float)
    si_raw = pd.to_numeric(el["shock_index"], errors="coerce").to_numpy(dtype=float)
    sc_ok = np.isfinite(sc_raw)
    si_ok = np.isfinite(si_raw)
    prob_sc = logistic_recalibrate(y_el[sc_ok], sc_raw[sc_ok])
    prob_si = logistic_recalibrate(y_el[si_ok], si_raw[si_ok])
    print(f"  pelvic_score usable N={int(sc_ok.sum()):,}  SI-alone usable N={int(si_ok.sum()):,}")

    dca_rows = []
    nb_curve = {"pelvic_score": {}, "SI-alone": {}}
    for p_t in THRESHOLDS:
        ta = nb_treat_all(el_prev, p_t)
        nb_sc = net_benefit(y_el[sc_ok], prob_sc, p_t)
        nb_si = net_benefit(y_el[si_ok], prob_si, p_t)
        nb_curve["pelvic_score"][round(float(p_t), 4)] = nb_sc
        nb_curve["SI-alone"][round(float(p_t), 4)] = nb_si
        dca_rows.append({"threshold": float(p_t), "score": "pelvic_score", "net_benefit": nb_sc,
                         "n_used": int(sc_ok.sum()), "treat_all_nb": ta, "treat_none_nb": 0.0})
        dca_rows.append({"threshold": float(p_t), "score": "SI-alone", "net_benefit": nb_si,
                         "n_used": int(si_ok.sum()), "treat_all_nb": ta, "treat_none_nb": 0.0})
    pd.DataFrame(dca_rows).to_csv(OUT_DCA, index=False)
    print(f"  wrote {OUT_DCA} ({len(dca_rows)} rows)")

    def gain_at(p_t):
        key = round(float(p_t), 4)
        return nb_curve["pelvic_score"][key] - nb_curve["SI-alone"][key]

    gains = {f"{p:.2f}": float(gain_at(p)) for p in DCA_REPORT_PTS}
    net_tp_score = {f"{p:.2f}": float(nb_curve["pelvic_score"][round(float(p), 4)] * 1000)
                    for p in DCA_REPORT_PTS}
    net_tp_gain = {f"{p:.2f}": float(gain_at(p) * 1000) for p in DCA_REPORT_PTS}
    # window summary over clinically relevant band
    band = [p for p in THRESHOLDS if 0.05 <= p <= 0.20]
    band_gain = np.array([gain_at(p) for p in band])
    elderly_dca_gain = {
        "rule_in_threshold": RULE_IN_PT,
        "gain_over_si_at_rule_in": float(gain_at(RULE_IN_PT)),
        "net_tp_per_1000_score_at_rule_in": float(nb_curve["pelvic_score"][round(RULE_IN_PT, 4)] * 1000),
        "net_tp_per_1000_gain_over_si_at_rule_in": float(gain_at(RULE_IN_PT) * 1000),
        "gain_over_si_by_pt": gains,
        "net_tp_per_1000_score_by_pt": net_tp_score,
        "net_tp_per_1000_gain_over_si_by_pt": net_tp_gain,
        "band_0.05_0.20_gain_min": float(band_gain.min()),
        "band_0.05_0.20_gain_mean": float(band_gain.mean()),
        "band_0.05_0.20_all_positive": bool((band_gain > 0).all()),
        "n_score": int(sc_ok.sum()), "n_si": int(si_ok.sum())}
    js["elderly_dca_gain"] = elderly_dca_gain
    print(f"  rule-in p_t={RULE_IN_PT}: NB gain over SI = {elderly_dca_gain['gain_over_si_at_rule_in']:+.5f}"
          f"  (net TP/1,000 gain = {elderly_dca_gain['net_tp_per_1000_gain_over_si_at_rule_in']:+.1f};"
          f" score net TP/1,000 = {elderly_dca_gain['net_tp_per_1000_score_at_rule_in']:.1f})")
    print(f"  band p_t 0.05-0.20 gain: min={elderly_dca_gain['band_0.05_0.20_gain_min']:+.5f} "
          f"mean={elderly_dca_gain['band_0.05_0.20_gain_mean']:+.5f} "
          f"all_positive={elderly_dca_gain['band_0.05_0.20_all_positive']}")
    for p in DCA_REPORT_PTS:
        print(f"    p_t={p:.2f}: score net TP/1,000={net_tp_score[f'{p:.2f}']:6.1f}  "
              f"gain over SI/1,000={net_tp_gain[f'{p:.2f}']:+6.1f}")
    table_rows.append({"Section": "ElderlyDCA", "Metric": "net_benefit_gain_over_SI_at_rule_in",
                       "Stratum": f"p_t={RULE_IN_PT}", "Value": elderly_dca_gain["gain_over_si_at_rule_in"],
                       "CI_Low": np.nan, "CI_High": np.nan, "N": int(sc_ok.sum()),
                       "Note": f"net TP/1,000 gain={elderly_dca_gain['net_tp_per_1000_gain_over_si_at_rule_in']:+.1f}"})
    table_rows.append({"Section": "ElderlyDCA", "Metric": "net_TP_per_1000_score_at_rule_in",
                       "Stratum": f"p_t={RULE_IN_PT}",
                       "Value": elderly_dca_gain["net_tp_per_1000_score_at_rule_in"],
                       "CI_Low": np.nan, "CI_High": np.nan, "N": int(sc_ok.sum()), "Note": "elderly"})

    # ----------------------------------------------------------------------------------------
    # (4) UNDER-TRIAGE HEADLINE + categorical NRI (score>=RULE_IN vs SI>=0.9)
    # ----------------------------------------------------------------------------------------
    banner("(4) Under-triage headline + categorical NRI (elderly test subset)")
    SI_HEUR = SI_HEURISTIC  # clinical shock-index heuristic (matches SI<0.9 under-triage framing)
    el_sc = pd.to_numeric(el["pelvic_score"], errors="coerce").to_numpy(dtype=float)
    el_si = pd.to_numeric(el["shock_index"], errors="coerce").to_numpy(dtype=float)
    ev_mask = y_el == 1

    # Under-triage: elderly composite-event, observed SI<0.9, score>=RULE_IN.
    n_el_events = int(ev_mask.sum())
    si_observed = np.isfinite(el_si)
    si_lt = si_observed & (el_si < SI_HEUR)          # SI-missed (would not flag)
    si_missing_ev = int((ev_mask & ~si_observed).sum())
    n_si_missed_bleeders = int((ev_mask & si_lt).sum())               # SI<0.9 among elderly bleeders
    undertriage = int((ev_mask & si_lt & (el_sc >= RULE_IN)).sum())   # ...but score rules IN
    frac_of_si_missed = undertriage / n_si_missed_bleeders if n_si_missed_bleeders else np.nan
    frac_of_all_events = undertriage / n_el_events if n_el_events else np.nan
    per_1000_elderly = undertriage / len(el) * 1000 if len(el) else np.nan
    js["undertriage_count"] = undertriage
    js["undertriage"] = {
        "count": undertriage,
        "definition": f"elderly TEST composite-event with observed SI<{SI_HEUR} but pelvic_score>={RULE_IN}",
        "elderly_test_n": int(len(el)),
        "elderly_composite_events": n_el_events,
        "si_missed_bleeders(SI<0.9)": n_si_missed_bleeders,
        "si_missing_among_events": si_missing_ev,
        "frac_of_si_missed_bleeders_recaptured": float(frac_of_si_missed),
        "frac_of_all_elderly_events": float(frac_of_all_events),
        "per_1000_elderly_test": float(per_1000_elderly),
        "population_scale_factor_test_to_full": float(len(full) / len(test)),
        "population_projected_full_cohort": int(round(undertriage * len(full) / len(test)))}
    print(f"  elderly composite events (test) = {n_el_events:,}")
    print(f"  ...with observed SI<{SI_HEUR} (SI would miss) = {n_si_missed_bleeders:,} "
          f"(+{si_missing_ev} with SI unobtainable)")
    print(f"  ...AND pelvic_score>={RULE_IN} (score reclassifies) = {undertriage:,}  <<< UNDER-TRIAGE HEADLINE")
    print(f"     = {100*frac_of_si_missed:.1f}% of SI-missed bleeders recaptured; "
          f"{100*frac_of_all_events:.1f}% of all elderly bleeders; "
          f"{per_1000_elderly:.1f} per 1,000 elderly")
    print(f"     population-projected to full cohort (x{len(full)/len(test):.2f}) "
          f"~= {js['undertriage']['population_projected_full_cohort']:,}")
    table_rows.append({"Section": "UnderTriage", "Metric": "occult_undertriaged_bleeders_reclassified",
                       "Stratum": f"SI<{SI_HEUR} & score>={RULE_IN}", "Value": undertriage,
                       "CI_Low": np.nan, "CI_High": np.nan, "N": int(len(el)),
                       "Note": f"{100*frac_of_si_missed:.1f}% of SI-missed bleeders; "
                               f"proj full~{js['undertriage']['population_projected_full_cohort']}"})

    # Categorical NRI: score(>=RULE_IN) vs SI(>=SI_HEUR) on complete-SI elderly.
    nri_ok = si_observed
    n_excl = int((~nri_ok).sum())
    yq = y_el[nri_ok]
    score_flag = (el_sc[nri_ok] >= RULE_IN).astype(int)
    si_flag = (el_si[nri_ok] >= SI_HEUR).astype(int)
    nri = categorical_nri(yq, score_flag, si_flag)
    # bootstrap CI (1000x, seed 42)
    rng = np.random.default_rng(SEED)
    n = len(yq); boots = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, n, size=n)
        rb = categorical_nri(yq[idx], score_flag[idx], si_flag[idx])
        boots[b] = rb["nri"]
    boots = boots[np.isfinite(boots)]
    nri_lo, nri_hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if len(boots) >= 10 else (np.nan, np.nan)
    nri.update({"nri_ci_low": nri_lo, "nri_ci_high": nri_hi,
                "si_threshold": SI_HEUR, "score_threshold": RULE_IN,
                "n_excluded_missing_si": n_excl})
    js["nri"] = nri
    print(f"  categorical NRI (score>={RULE_IN} vs SI>={SI_HEUR}), complete-SI elderly N={n:,} "
          f"({n_excl} excluded missing-SI):")
    print(f"    NRI = {nri['nri']:+.4f} [{nri_lo:+.4f},{nri_hi:+.4f}]  "
          f"(event={nri['nri_event']:+.4f}, non-event={nri['nri_nonevent']:+.4f})")
    print(f"    events N={nri['n_event']:,}: up(score catches SI-miss)={nri['event_up']}, "
          f"down(score misses SI-catch)={nri['event_down']}")
    print(f"    non-events N={nri['n_nonevent']:,}: down(correct de-flag)={nri['nonevent_down']}, "
          f"up(false flag)={nri['nonevent_up']}")
    table_rows.append({"Section": "NRI", "Metric": "categorical_NRI_total",
                       "Stratum": f"score>={RULE_IN} vs SI>={SI_HEUR}", "Value": nri["nri"],
                       "CI_Low": nri_lo, "CI_High": nri_hi, "N": n, "Note": "1000-boot CI"})
    table_rows.append({"Section": "NRI", "Metric": "NRI_event", "Stratum": "events",
                       "Value": nri["nri_event"], "CI_Low": np.nan, "CI_High": np.nan,
                       "N": nri["n_event"], "Note": f"up={nri['event_up']} down={nri['event_down']}"})
    table_rows.append({"Section": "NRI", "Metric": "NRI_nonevent", "Stratum": "nonevents",
                       "Value": nri["nri_nonevent"], "CI_Low": np.nan, "CI_High": np.nan,
                       "N": nri["n_nonevent"],
                       "Note": f"down={nri['nonevent_down']} up={nri['nonevent_up']}"})

    # ----------------------------------------------------------------------------------------
    # (5) RECLASSIFICATION LEDGER — the full 2x2, with the COST column visible (Fix 6 / D4)
    # ----------------------------------------------------------------------------------------
    banner("(5) Elderly reclassification LEDGER (Fix 6 / D4) — 2x2 with the cost column")
    print("  Reviewer comment: \"'635 additional events detected' sounds awesome but cannot be")
    print("  interpreted without knowing how many additional interventions the score would")
    print("  generate.\"  ->  every benefit column below is paired with its cost column.")

    # Frame = elderly TEST patients with an OBSERVED shock index. Identical to the NRI
    # frame: a patient with no recordable SI cannot be classified by the SI heuristic, so
    # they can be neither up- nor down-reclassified relative to it. Excluded rows are
    # counted and reported, never silently dropped.
    L_n_excl = int((~si_observed).sum())
    yL = y_el[si_observed].astype(int)
    scL = el_sc[si_observed]
    siL = el_si[si_observed]
    n_frame = int(len(yL))
    n_frame_ev = int(yL.sum())

    score_pos = scL >= RULE_IN            # the arrival score escalates
    si_pos = siL >= SI_HEUR               # the shock-index heuristic escalates
    si_neg = ~si_pos

    # ---- the four 2x2 cells ----
    cell_up = score_pos & si_neg          # score-only: SI reassures, score escalates
    cell_down = (~score_pos) & si_pos      # SI-only: SI alarms, score stands down
    cell_both = score_pos & si_pos
    cell_neither = (~score_pos) & si_neg

    # ---- denominators of the two reclassification arms ----
    n_si_reassuring = int(si_neg.sum())    # SI < 0.9  (the up-arm denominator; v2 = 33,127)
    n_si_alarming = int(si_pos.sum())      # SI >= 0.9 (the down-arm denominator)

    ledger_rows = []
    ledger_rows.append(ledger_row(
        "newly_flagged_up",
        f"elderly, observed SI<{SI_HEUR} (reassuring), pelvic_score>={RULE_IN}",
        n_si_reassuring, cell_up[si_neg], yL[si_neg],
        note=("THE HEADLINE ARM. Denominator = elderly with a reassuring shock index. "
              "Escalations_per_event = additional escalations bought per additional "
              "event detected; PPV = yield inside the newly-flagged stratum.")))
    ledger_rows.append(ledger_row(
        "down_classified",
        f"elderly, observed SI>={SI_HEUR} (alarming), pelvic_score<{RULE_IN}",
        n_si_alarming, cell_down[si_pos], yL[si_pos],
        note=("THE REVERSE ARM — reported for honesty, not decoration. "
              "Flagged_events_TP here are events the score GIVES UP relative to the "
              "SI heuristic (the NRI 'event down' cell); Flagged_nonevents_FP are "
              "escalations correctly AVOIDED (the NRI 'non-event down' cell). "
              "Escalations_per_event reads as escalations avoided per event lost.")))
    ledger_rows.append(ledger_row(
        "concordant_both_flag",
        f"elderly, SI>={SI_HEUR} AND pelvic_score>={RULE_IN}",
        n_frame, cell_both, yL,
        note="2x2 cell: both strategies escalate (no reclassification)."))
    ledger_rows.append(ledger_row(
        "concordant_neither_flags",
        f"elderly, SI<{SI_HEUR} AND pelvic_score<{RULE_IN}",
        n_frame, cell_neither, yL,
        note="2x2 cell: neither strategy escalates (no reclassification)."))
    ledger_rows.append(ledger_row(
        "strategy_score_alone",
        f"elderly, pelvic_score>={RULE_IN} (whole frame)",
        n_frame, score_pos, yL,
        note="Marginal of the 2x2: the arrival score used alone."))
    ledger_rows.append(ledger_row(
        "strategy_si_alone",
        f"elderly, observed SI>={SI_HEUR} (whole frame)",
        n_frame, si_pos, yL,
        note="Marginal of the 2x2: the shock-index heuristic used alone."))
    ledger_rows.append(ledger_row(
        "comparator_treat_all_elderly",
        "elderly, escalate everyone (whole frame)",
        n_frame, np.ones(n_frame, dtype=bool), yL,
        note=("COMPARATOR: escalations per event under a treat-all-elderly policy "
              "(= 1/prevalence). The newly-flagged arm must be read against this.")))

    # ---- net reclassification: up-arm minus down-arm, on one line ----
    net_flag = int(cell_up.sum()) - int(cell_down.sum())
    net_ev = int((cell_up & (yL == 1)).sum()) - int((cell_down & (yL == 1)).sum())
    net_nonev = int((cell_up & (yL == 0)).sum()) - int((cell_down & (yL == 0)).sum())
    ledger_rows.append({
        "Arm": "net_reclassification",
        "Definition": f"up-arm minus down-arm (score>={RULE_IN} vs SI>={SI_HEUR}, elderly)",
        "N_denominator": n_frame,
        "N_events_in_denominator": n_frame_ev,
        "Event_rate_in_denominator_pct": 100.0 * n_frame_ev / n_frame if n_frame else np.nan,
        "Event_rate_CI_low_pct": np.nan, "Event_rate_CI_high_pct": np.nan,
        "N_flagged": net_flag,
        "Flagged_pct_of_denominator": 100.0 * net_flag / n_frame if n_frame else np.nan,
        "Flagged_pct_CI_low": np.nan, "Flagged_pct_CI_high": np.nan,
        "Flagged_events_TP": net_ev,
        "Flagged_nonevents_FP": net_nonev,
        "PPV_within_flagged_pct": np.nan, "PPV_CI_low_pct": np.nan, "PPV_CI_high_pct": np.nan,
        # Only defined when the switch actually detects MORE events. With net_ev <= 0
        # the quantity "escalations per additional event detected" has no referent, and
        # a ratio of two negatives would print as a flattering positive number.
        "Escalations_per_event": (net_flag / net_ev) if net_ev > 0 else np.nan,
        "Note": ("NET position of switching from the SI heuristic to the score. Signed "
                 "differences, NOT counts — Wilson CIs do not apply to a difference of "
                 "two dependent binomials and are left blank rather than faked. "
                 "Escalations_per_event = net additional escalations per net additional "
                 "event detected, and is left blank unless the net event change is "
                 "positive (a ratio of two negatives is not a cost-per-benefit)."),
    })

    ledger_df = pd.DataFrame(ledger_rows)[LEDGER_COLS]
    ledger_df.to_csv(OUT_LEDGER, index=False)
    print(f"\n  frame: elderly TEST with observed SI, N={n_frame:,} "
          f"({n_frame_ev:,} events, {L_n_excl:,} excluded for unobtainable SI)")
    print(f"  SI<{SI_HEUR} (reassuring) N={n_si_reassuring:,} | "
          f"SI>={SI_HEUR} (alarming) N={n_si_alarming:,}")
    print(f"  {'arm':30s} {'n_denom':>9s} {'flagged':>8s} {'TP':>7s} {'FP':>7s} "
          f"{'PPV%':>7s} {'esc/ev':>7s}")
    for r in ledger_rows:
        print(f"  {r['Arm']:30s} {r['N_denominator']:>9,} {r['N_flagged']:>8,} "
              f"{r['Flagged_events_TP']:>7,} {r['Flagged_nonevents_FP']:>7,} "
              f"{(r['PPV_within_flagged_pct'] if np.isfinite(r['PPV_within_flagged_pct']) else float('nan')):>7.1f} "
              f"{(r['Escalations_per_event'] if np.isfinite(r['Escalations_per_event']) else float('nan')):>7.2f}")
    print(f"  wrote {OUT_LEDGER} ({len(ledger_df)} rows)")

    up = ledger_rows[0]
    down = ledger_rows[1]
    all_el = ledger_rows[6]
    print("\n  READ THIS AS ONE SENTENCE (the D4 answer):")
    print(f"    Among {n_si_reassuring:,} elderly with a reassuring shock index, the score "
          f"newly flags {up['N_flagged']:,}")
    print(f"    ({up['Flagged_events_TP']:,} events and {up['Flagged_nonevents_FP']:,} non-events) "
          f"= {up['Escalations_per_event']:.1f} escalations per additional event")
    print(f"    detected, PPV {up['PPV_within_flagged_pct']:.1f}% "
          f"[{up['PPV_CI_low_pct']:.1f}-{up['PPV_CI_high_pct']:.1f}] in the newly-flagged group,")
    print(f"    against {all_el['Escalations_per_event']:.1f} escalations per event under "
          f"treat-all-elderly.")
    print(f"    It also stands down on {down['N_flagged']:,} SI-alarming patients, giving up "
          f"{down['Flagged_events_TP']:,} events")
    print(f"    to avoid {down['Flagged_nonevents_FP']:,} escalations. NET: "
          f"{net_flag:+,} escalations for {net_ev:+,} events.")

    js["elderly_ledger"] = {
        "frame": (f"elderly (age_ge65==1) TEST rows with an observed shock index; "
                  f"score threshold >= {RULE_IN}, SI heuristic >= {SI_HEUR}"),
        "n_frame": n_frame,
        "n_frame_events": n_frame_ev,
        "n_excluded_unobtainable_si": L_n_excl,
        "n_si_reassuring_lt_0_9": n_si_reassuring,
        "n_si_alarming_ge_0_9": n_si_alarming,
        "rule_in_cutpoint": RULE_IN,
        "si_threshold": SI_HEUR,
        "rows": {r["Arm"]: {k: (None if (isinstance(r[k], float) and not np.isfinite(r[k]))
                                else r[k])
                            for k in LEDGER_COLS if k not in ("Arm", "Definition", "Note")}
                 for r in ledger_rows},
        "headline": {
            "denominator_elderly_si_reassuring": n_si_reassuring,
            "newly_flagged": up["N_flagged"],
            "newly_flagged_events": up["Flagged_events_TP"],
            "newly_flagged_nonevents": up["Flagged_nonevents_FP"],
            "escalations_per_additional_event": up["Escalations_per_event"],
            "ppv_within_newly_flagged_pct": up["PPV_within_flagged_pct"],
            "ppv_ci_low_pct": up["PPV_CI_low_pct"], "ppv_ci_high_pct": up["PPV_CI_high_pct"],
            "treat_all_elderly_escalations_per_event": all_el["Escalations_per_event"],
            "down_classified_n": down["N_flagged"],
            "down_classified_events_given_up": down["Flagged_events_TP"],
            "down_classified_escalations_avoided": down["Flagged_nonevents_FP"],
            "net_escalations": net_flag, "net_events": net_ev, "net_nonevents": net_nonev},
        "nri_cell_crosswalk": ("newly_flagged_up == NRI event_up + nonevent_up; "
                               "down_classified == NRI event_down + nonevent_down"),
        # No v2 numerals in this string, deliberately: CONTRACTS §13 G5b greps the
        # manuscript/abstract/figure source for the superseded v2 constants, and `16`
        # reads this file. Provenance is stated without seeding the artifact tree with
        # the exact digit strings that gate looks for.
        "note": ("Fix 6 / D4. Every quantity here is recomputed on v3 from "
                 "cohort_test_scored.parquet at run time. The corresponding v2 figures "
                 "are superseded, are not carried forward, and appear nowhere in this "
                 "file; see SAP_amendment_v3.md D-6 for their provenance."),
    }

    # Cross-check: the ledger's 2x2 must reproduce the NRI's up/down cells exactly.
    # They are computed from the same frame by different code paths; a mismatch means
    # one of them is wrong, and a reclassification table that disagrees with its own
    # NRI is not publishable.
    exp_up = nri["event_up"] + nri["nonevent_up"]
    exp_down = nri["event_down"] + nri["nonevent_down"]
    if (up["N_flagged"], up["Flagged_events_TP"], up["Flagged_nonevents_FP"]) != \
            (exp_up, nri["event_up"], nri["nonevent_up"]):
        raise SystemExit(
            f"LEDGER/NRI MISMATCH (up-arm): ledger {up['N_flagged']}="
            f"{up['Flagged_events_TP']}+{up['Flagged_nonevents_FP']} vs NRI {exp_up}="
            f"{nri['event_up']}+{nri['nonevent_up']}")
    if (down["N_flagged"], down["Flagged_events_TP"], down["Flagged_nonevents_FP"]) != \
            (exp_down, nri["event_down"], nri["nonevent_down"]):
        raise SystemExit(
            f"LEDGER/NRI MISMATCH (down-arm): ledger {down['N_flagged']}="
            f"{down['Flagged_events_TP']}+{down['Flagged_nonevents_FP']} vs NRI {exp_down}="
            f"{nri['event_down']}+{nri['nonevent_down']}")
    # The four cells must partition the frame.
    if int(cell_up.sum() + cell_down.sum() + cell_both.sum() + cell_neither.sum()) != n_frame:
        raise SystemExit("LEDGER: the four 2x2 cells do not partition the frame")
    # The up-arm is exactly the under-triage headline restricted to observed-SI rows.
    if up["Flagged_events_TP"] != undertriage:
        raise SystemExit(
            f"LEDGER: up-arm events {up['Flagged_events_TP']} != under-triage headline "
            f"{undertriage} — the ledger and the headline must be the same number")
    print("\n  ASSERT pass: ledger 2x2 == NRI up/down cells == under-triage headline; "
          "cells partition the frame.")

    # ----------------------------------------------------------------------------------------
    # Elderly change-point tie-in from 11 (OPTIONAL — CONTRACTS §7 row 10)
    # ----------------------------------------------------------------------------------------
    js["elderly_changepoint_tiein"] = read_changepoint_tiein()

    # ----------------------------------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------------------------------
    banner("Writing table6 + table6b + reclass.json")
    tbl = pd.DataFrame(table_rows)
    front = ["Section", "Metric", "Stratum", "Value", "CI_Low", "CI_High", "N", "Note"]
    tbl = tbl[[c for c in front if c in tbl.columns]]
    tbl.to_csv(OUT_TABLE, index=False)
    print(f"  wrote {OUT_TABLE} ({len(tbl)} rows)")
    print(f"  wrote {OUT_LEDGER} ({len(ledger_df)} rows)  [T6B, Fix 6 / D4]")

    js["_note_changepoint"] = ("Change-points are not COMPUTED here. v3 READS script 11's "
                               "derived/changepoint_stratified.json (optional, soft-fail) into "
                               "js['elderly_changepoint_tiein'] for the manuscript narrative, "
                               "including 11's descriptive_only demotions.")
    with open(OUT_JSON, "w") as f:
        json.dump(js, f, indent=2, default=float)
    print(f"  wrote {OUT_JSON}")

    banner(f"Step 10 complete — runtime {time.time()-t0:.1f}s")
    print("  HEADLINE NUMBERS:")
    print(f"    interaction OR (composite)     = {em_primary['or']:.3f} "
          f"[{em_primary['or_ci_low']:.3f},{em_primary['or_ci_high']:.3f}]")
    print(f"    treatment gap: HC young {js['treatment_gap']['hc_young']:.3f} vs "
          f"elderly {js['treatment_gap']['hc_elderly']:.3f}; death near-flat "
          f"({js['treatment_gap']['death_young']:.3f} vs {js['treatment_gap']['death_elderly']:.3f})")
    print(f"    under-triage reclassified      = {undertriage:,} elderly bleeders "
          f"(SI<0.9 but score>={RULE_IN})")
    print(f"    categorical NRI                = {nri['nri']:+.4f} [{nri_lo:+.4f},{nri_hi:+.4f}]")
    print(f"    elderly DCA gain over SI @rule-in = "
          f"{elderly_dca_gain['net_tp_per_1000_gain_over_si_at_rule_in']:+.1f} net TP/1,000")
    print("    LEDGER (T6B, the D4 answer — quote the cost with the benefit or not at all):")
    print(f"      denominator (elderly, SI<{SI_HEUR}) = {n_si_reassuring:,}; "
          f"newly flagged = {up['N_flagged']:,}")
    print(f"      = {up['Flagged_events_TP']:,} events + {up['Flagged_nonevents_FP']:,} non-events "
          f"-> {up['Escalations_per_event']:.1f} escalations/event, "
          f"PPV {up['PPV_within_flagged_pct']:.1f}%")
    print(f"      treat-all-elderly comparator  = {all_el['Escalations_per_event']:.1f} "
          f"escalations/event")
    print(f"      down-classified               = {down['N_flagged']:,} "
          f"({down['Flagged_events_TP']:,} events given up, "
          f"{down['Flagged_nonevents_FP']:,} escalations avoided)")
    print(f"      NET                           = {net_flag:+,} escalations for "
          f"{net_ev:+,} events")
    print(f"    death component reported        = {DEATH_COL} "
          f"(all-cause {DEATH_COL_ALLCAUSE} co-reported as a labelled sensitivity)")


if __name__ == "__main__":
    main()
