#!/usr/bin/env python3
"""
render_figures.py -- Publication figure renderer for the Pelvic Hemorrhage
Arrival Decision-Support Tool (v3 / "pre-decision"), NTDB/TQP 2019-2024.

Batch 5c of the v3 pipeline (see CONTRACTS.md sec.7/sec.8/sec.11). Analysis
scripts (05-13, 17) emit fig-DATA artifacts under derived/ + tables/; THIS script
renders the publication figures F2-F10 from those artifacts. It renders NOTHING
except figures/figure2*..figure10* -- F1 (STROBE) is owned by 14_consort.py and
the blinded WTA .jpg is owned by 18_wta_figure.py; neither is touched here.

Self-contained per house style (CONTRACTS sec.11: copy-adapt, do NOT import
across pipeline scripts). The Okabe-Ito palette + matplotlib Agg/rcParams block
are copied-adapted from an earlier figure script; the _save_figure()
PNG+PDF (300 dpi) helper is copied-adapted from
an earlier pelvic-fracture figure script.

Figures (CONTRACTS sec.8):
  F2  figure2_score_distribution <- derived/cohort_test_scored.parquet + risk_tiers.csv
  F3  figure3_roc                <- derived/roc_data.pkl + temporal_roc_data.pkl
  F4  figure4_calibration        <- derived/calibration_data.csv + reclass.json (elderly)
  F5  figure5_dca (CENTERPIECE)  <- derived/dca_data.csv + elderly_dca_data.csv
  F6  figure6_subgroup_forest    <- derived/subgroup_forest_data.csv
  F7  figure7_threshold_splines  <- derived/spline_{si,sbp}_composite.csv + changepoint_*.json
  F8  figure8_component          <- derived/component_decomp_data.csv  (script 13; may be absent)
  F9  figure9_score_card         <- derived/score_card.json + tables/table3_points.csv + risk_tiers.csv
  F10 figure10_mtp_equivalent    <- derived/mtp_roc_data.pkl + derived/mtp_dca_data.csv  (script 17; NEW)

v3 changes (CONTRACTS sec.8 "F9 -- v3 required changes"; SAP amendment D-1..D-5)
------------------------------------------------------------------------------
  * F9 no longer prints a `sbp_unrecordable` POINT ROW. Fix 4 (D-4) moves it off
    the integer scale and makes it a binary automatic-escalate OVERRIDE, so the
    card renders it as a full-width banner above the point grid, visually
    distinct from the point rows and explicitly carrying NO point value. Data
    comes from score_card.json["override"] (CONTRACTS sec.5.1).
  * F9 renders no `unstable_ring` row (Fix 3 / D-3, gate G3d). The label stays
    defined because F7/F8 still use it for stratum text -- but the card filters
    it out unconditionally, so the gate holds even if an upstream artifact is
    stale.
  * VAR_LABELS covers the WHOLE 17-variable candidate pool (CONTRACTS sec.5) --
    any candidate can survive selection onto the card, and F9 is the source for
    the single blinded .jpg attached to the WTA abstract, so a raw snake_case
    identifier on a point row is a submission-visible defect. A missing label is
    now a LOUD failure (stdout + stderr + a DRAFT stamp on the figure itself),
    never a silent `str(var)` fallback; _audit_var_labels() checks the map
    against the pool at startup and again before F9 draws.
  * F9's stated maximum is the OBSERVED score maximum from
    score_card.json["score_range"], NOT the sum of the points column. The two
    numbers differ (gate G4c) and are labelled so the reader cannot confuse
    them.
  * The mortality anchor is `death_hemorrhagic`, not `death_inhosp` (D-1/D-8);
    component labels are repointed accordingly, and `death_inhosp` keeps a
    label only so a SENSITIVITY arm never renders as a raw identifier.
  * F10 is new (D-5 / Fix 7) and carries the mandatory identity statement:
    `mtp_equivalent` is definitionally identical to `substantial_transfusion`
    and is NOT independent corroboration of the primary result (CONTRACTS sec.4).

Per-figure failure-tolerant: a missing/partial input warns + skips that one
figure and the run continues (mirrors pelvic_figures.main()). F10 in particular
degrades panel-by-panel because script 17 lands in a later batch.

Usage:
  python3 render_figures.py             # render F2-F10 (missing inputs skipped)
  python3 render_figures.py --only 8    # render a single figure (2..10)
"""

import os
import re
import sys
import json
import pickle
import argparse
import textwrap
import warnings

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # MUST precede pyplot import (CONTRACTS sec.1)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CANONICAL PATHS  (CONTRACTS sec.1 -- module-level absolute root, no argparse paths)
# =============================================================================
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

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DPI = 300

# ---- Publication rcParams (Arial, 300 dpi, fonttype 42) -- adapted from an earlier script ----
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,   # editable/embedded TrueType (CONTRACTS sec.1)
    "ps.fonttype": 42,
})

# ---- Okabe-Ito color-blind-safe palette (copied from tth_figures_minutes.py) ----
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

# Named model/role colors (consistent across every figure)
C_SCORE = OKABE_ITO["blue"]        # pelvic_score (the tool)
C_SI = OKABE_ITO["vermilion"]      # shock-index comparator
C_RSIG = OKABE_ITO["green"]        # rSIG comparator
C_FULL = OKABE_ITO["orange"]       # full-information (ISS/AIS) benchmark
C_TEMPORAL = OKABE_ITO["purple"]   # temporal 2023-24 validation overlay
C_TREATALL = OKABE_ITO["grey"]     # DCA treat-all
C_TREATNONE = OKABE_ITO["black"]   # DCA treat-none
C_EVENT = OKABE_ITO["vermilion"]   # composite event
C_NOEVENT = OKABE_ITO["sky_blue"]  # no event
C_ELDERLY = OKABE_ITO["vermilion"]
C_YOUNG = OKABE_ITO["blue"]
C_OVERALL = OKABE_ITO["grey"]

# Risk-tier traffic-light colors (Okabe-Ito adjacent; CB-distinguishable)
TIER_ORDER = ["rule_out", "observe", "escalate"]
TIER_COLOR = {"rule_out": OKABE_ITO["green"], "observe": OKABE_ITO["orange"],
              "escalate": OKABE_ITO["vermilion"]}
TIER_DISPLAY = {"rule_out": "Rule-out", "observe": "Observe", "escalate": "Escalate"}

# DCA rule-in decision threshold to annotate. The tool is a rule-IN aid, so the
# high-specificity end of the pre-specified DCA window (p_t 0.05-0.20, verification
# gate CONTRACTS sec.13.3) is the rule-in decision threshold.
RULE_IN_PT = 0.20
DCA_PRESPEC = (0.05, 0.20)

# The FULL v3 LASSO candidate pool (CONTRACTS sec.5: 17 variables after Fix 3
# dropped `unstable_ring`). ANY of these can survive selection and land on the
# F9 card, so EVERY one of them must carry a clinical label -- see
# VAR_LABELS and _audit_var_labels(). Kept as an explicit list rather than
# implied by VAR_LABELS' keys so the audit compares two independent sources.
LASSO_CANDIDATE_POOL = (
    "age_65_74", "age_ge75", "sex_female",
    "sbp_lt90", "sbp_lt70", "sbp_unrecordable",
    "hr_ge120", "si_ge0_9", "si_ge1_0",
    "gcs_motor_le5", "gcs_le8",
    "mech_high_energy", "mech_mvc", "mech_fall",
    "air_transport",
    "cc_anticoag", "cc_bleeding",
)

# Human-readable variable labels (score-card predictors) -- reused by F6/F9.
#
# Wording is CROSS-WALKED to 15_table1.py's LABEL_MAP / CATEGORICALS so a reader
# can match a card row to its Table 1 row: `cc_bleeding` and `air_transport`
# carry Table 1's exact strings ("Bleeding disorder (preexisting)" and "Air
# transport", the latter from CATEGORICALS["transport_cat"]).
#
# `unstable_ring` KEEPS its label: Fix 3 removes it from the SCORE, not from the
# data, and F7/F8 still use it for stratum text (CONTRACTS sec.8 F9-1). Only the
# F9 card excludes it -- see CARD_EXCLUDED_VARS.
#
# F9 is the source for the single blinded .jpg attached to the WTA abstract
# (CONTRACTS sec.8 / sec.13 G6), so a raw snake_case identifier printed on a
# point row is a submission-visible defect, not a cosmetic one. Nothing is
# allowed to fall through silently: _var_label() warns loudly and F9 stamps the
# figure itself (see UNLABELLED_VARS).
VAR_LABELS = {
    # ---- age (force-in anchor, implemented as two bands) --------------------
    "age_65_74": "Age 65–74",
    "age_ge75": "Age ≥75",
    # alias: CONTRACTS sec.5 names the anchor `age_ge65`; the score carries the
    # two bands, but a stale artifact could still hand us the collapsed name.
    "age_ge65": "Age ≥65",
    # ---- demographics ------------------------------------------------------
    "sex_female": "Female sex",
    # ---- arrival physiology ------------------------------------------------
    "sbp_lt90": "SBP <90 mmHg",
    "sbp_lt70": "SBP <70 mmHg",
    # Fix 4 / D-4 (CONTRACTS sec.5.1): a physiologic state, NOT a missingness or
    # data-quality artifact. The v2 wording is retired and gate G4f greps for it
    # (see RETIRED_LABEL_PATTERNS). 05 carries the canonical string in
    # refit.json["override_labels"]; that file wins over this literal at runtime.
    "sbp_unrecordable": "No signs of life on arrival (unrecordable systolic pressure)",
    "hr_ge120": "Heart rate ≥120",
    "si_ge0_9": "Shock index ≥0.9",
    "si_ge1_0": "Shock index ≥1.0",
    "gcs_motor_le5": "GCS motor ≤5",
    "gcs_le8": "GCS ≤8",
    # ---- mechanism ---------------------------------------------------------
    "mech_high_energy": "High-energy mechanism",
    "mech_mvc": "MVC (occupant)",
    "mech_fall": "Fall mechanism",
    # ---- prehospital -------------------------------------------------------
    # Table 1 renders transport_cat=="Air" as "Air transport"; same string here
    # so the card row and the Table 1 row read identically.
    "air_transport": "Air transport",
    # ---- preexisting conditions -------------------------------------------
    "cc_anticoag": "Anticoagulant use",
    # Table 1: "Bleeding disorder (preexisting)" -- reused verbatim.
    "cc_bleeding": "Bleeding disorder (preexisting)",
    # ---- NOT a candidate: kept for F7/F8 stratum text only (Fix 3) ----------
    "unstable_ring": "Unstable pelvic ring (S32.81)",
}

# Variables whose label was requested but not found. Populated by _var_label();
# read by F9 so an unlabelled row is disclosed ON THE FIGURE, not only in a log
# line that nobody rereads before attaching the .jpg to a submission.
UNLABELLED_VARS: set = set()

# ---- Fix 4 (D-4) / Fix 3 (D-3): what the BEDSIDE CARD may show ---------------
# Override-role variables never carry a point value; they render as a banner.
OVERRIDE_VARS = ("sbp_unrecordable",)
# Never renderable as a card row, whatever an upstream artifact says (gate G3d).
CARD_EXCLUDED_VARS = ("unstable_ring",)
ROLE_POINT, ROLE_OVERRIDE = "point", "override"

# Fallbacks only. Preference order at runtime is
#   refit.json["override_labels"] / ["override_rule_text"]  (05 owns the string)
#   -> score_card.json["override"]["label"] / ["rule_text"] (08 relays it)
#   -> these literals.
OVERRIDE_BANNER_HEADLINE = "AUTOMATIC ESCALATION — regardless of score"
OVERRIDE_RULE_TEXT_FALLBACK = (
    "Automatic escalation regardless of score: "
    "no signs of life on arrival (unrecordable systolic pressure)"
)
C_OVERRIDE = OKABE_ITO["vermilion"]   # banner accent; matches the escalate tier

# Retired v2 wording. Any label reaching a rendered figure is rewritten through
# _sanitize_label() so a stale artifact cannot put it back on a figure (G4f).
RETIRED_LABEL_PATTERNS = (
    (r"SBP\s*unrecordable\s*\(?\s*=?\s*0?\s*\)?", "no signs of life on arrival "
                                                  "(unrecordable systolic pressure)"),
    (r"\bSBP\s*=\s*0\b", "no signs of life on arrival (unrecordable systolic pressure)"),
    (r"\bsbp_unrecordable\b", "no signs of life on arrival "
                              "(unrecordable systolic pressure)"),
)


# =============================================================================
# I/O HELPERS  (graceful skip -- never crash on a missing/partial artifact)
# =============================================================================
def _read_csv(path):
    if not os.path.exists(path):
        print(f"  SKIP: {os.path.basename(path)} not found ({path})")
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP: error reading {os.path.basename(path)}: {exc}")
        return None


def _read_json(path):
    if not os.path.exists(path):
        print(f"  SKIP: {os.path.basename(path)} not found ({path})")
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP: error reading {os.path.basename(path)}: {exc}")
        return None


def _read_pickle(path):
    if not os.path.exists(path):
        print(f"  SKIP: {os.path.basename(path)} not found ({path})")
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP: error reading {os.path.basename(path)}: {exc}")
        return None


def _read_parquet(path, columns=None):
    if not os.path.exists(path):
        print(f"  SKIP: {os.path.basename(path)} not found ({path})")
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        try:  # column projection unsupported / column missing -> full read
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP: error reading {os.path.basename(path)}: {exc}")
            return None


def _save_figure(fig, basename):
    """Save PNG (300 dpi) AND PDF, both bbox_inches='tight', then close.
    Copied-adapted from pelvic_figures._save_figure (the sec.8 PNG+PDF contract)."""
    os.makedirs(FIGURES, exist_ok=True)
    png_path = os.path.join(FIGURES, f"{basename}.png")
    pdf_path = os.path.join(FIGURES, f"{basename}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {basename}.png / {basename}.pdf  ->  {FIGURES}")


def _flex_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _finite(value):
    """float(value) if it is a real finite number, else None.

    Guards every artifact-supplied number that reaches a plot coordinate. A
    present-but-NaN JSON key (e.g. a descriptive-only change-point) survives
    `.get(k, default)` untouched, so finiteness has to be checked explicitly.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _changepoint_block(obj):
    """Normalize a changepoint_*.json payload to the single block to annotate.

    Accepts either a flat block (`{"breakpoint": ..., ...}`) or a per-stratum
    container (`{"elderly": {...}, "_guard_summary": {...}}`). Keys beginning
    with "_" are metadata, never curves — iterating them as strata is a crash.
    """
    if not isinstance(obj, dict):
        return {}
    if "breakpoint" in obj or "descriptive_only" in obj:
        return obj
    for k, v in obj.items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, dict) and ("breakpoint" in v or "descriptive_only" in v):
            return v
    return {}


def _sanitize_label(text):
    """Rewrite retired v2 wording out of any string bound for a figure (G4f).

    Labels can arrive from a JSON/CSV artifact rather than from VAR_LABELS, so
    the approved display string is enforced at the point of rendering instead of
    being trusted upstream.
    """
    s = str(text)
    for pat, repl in RETIRED_LABEL_PATTERNS:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s


def _humanize_var(var):
    """Least-bad readable string for an UNLABELLED identifier.

    Never a substitute for a real clinical label -- it exists only so that a
    labelling gap degrades to `Air Transport` rather than to `air_transport` on
    a figure a reviewer will read. Every call is accompanied by a loud warning.
    """
    s = str(var).replace("_", " ").strip()
    return (s[:1].upper() + s[1:]) if s else str(var)


def _var_label(var, overrides=None):
    """Human label for a predictor: 05's canonical string > VAR_LABELS > LOUD.

    An identifier with no clinical label used to fall through to `str(var)`
    silently, which put a raw snake_case token on the bedside card (F9) -- the
    figure that becomes the blinded WTA .jpg. A gap is now recorded in
    UNLABELLED_VARS and announced on stdout AND stderr the first time it is
    seen, and F9 stamps it on the figure. Fix the gap by adding the variable to
    VAR_LABELS, using 15_table1.py's wording so the card and Table 1 agree.
    """
    if overrides and var in overrides:
        return _sanitize_label(overrides[var])
    if var in VAR_LABELS:
        return _sanitize_label(VAR_LABELS[var])
    if var not in UNLABELLED_VARS:
        UNLABELLED_VARS.add(var)
        msg = (f"  *** UNLABELLED VARIABLE: `{var}` has no entry in "
               f"VAR_LABELS. Rendering it as \"{_humanize_var(var)}\" — this is "
               f"a FALLBACK, not a clinical label. Add `{var}` to VAR_LABELS "
               f"(reuse 15_table1.py's wording) before any figure is used in a "
               f"manuscript or a WTA attachment. ***")
        print(msg)
        print(msg, file=sys.stderr)
    return _sanitize_label(_humanize_var(var))


def _audit_var_labels(pool=LASSO_CANDIDATE_POOL, context="startup"):
    """Every LASSO candidate must carry a clinical label. Returns the gaps.

    The card can only ever show variables selected from the candidate pool, so
    checking VAR_LABELS against the pool catches a gap BEFORE a figure renders
    rather than after a reviewer reads a raw identifier off it (CONTRACTS
    sec.5 = 17 candidates; sec.8 F9 = the card + the blinded WTA .jpg).
    """
    missing = [v for v in pool if v not in VAR_LABELS]
    if missing:
        msg = (f"  *** VAR_LABELS GAP ({context}): {len(missing)} of {len(pool)} "
               f"LASSO candidates have no clinical label: {missing}. Any of them "
               f"can survive selection onto the F9 score card. Add them to "
               f"VAR_LABELS (CONTRACTS sec.5). ***")
        print(msg)
        print(msg, file=sys.stderr)
    else:
        print(f"  VAR_LABELS audit ({context}): all {len(pool)} LASSO candidates "
              f"carry a clinical label.")
    return missing


def _load_refit_labels():
    """Canonical override label/rule text from 05's refit.json (single source).

    Returns (labels_dict, rule_text_or_None, theoretical_sum_or_None). Every
    element degrades to None/{} if refit.json is absent or pre-Fix-4.
    """
    refit = _read_json(os.path.join(DERIVED, "refit.json"))
    if not isinstance(refit, dict):
        return {}, None, None
    labels = refit.get("override_labels") or {}
    if not isinstance(labels, dict):
        labels = {}
    rule = refit.get("override_rule_text") or None
    theo = refit.get("theoretical_point_sum_unconstrained")
    theo_max = None
    if isinstance(theo, (list, tuple)) and len(theo) == 2:
        theo_max = theo[1]
    elif isinstance(theo, (int, float)):
        theo_max = theo
    return labels, rule, theo_max


# =============================================================================
# v3 OUTCOME PROSE  (CONTRACTS sec.4)
# `major_hemorrhage_event` keeps its v2 NAME and changes its DEFINITION, so
# every outcome-facing figure states the definition rather than relying on the
# name (this is also the answer to reviewer comment D5, "is this a composite?").
# The component list is READ from logs/qc_manifest.json where available so the
# caption tracks the cohort build instead of a hand-maintained sentence.
# =============================================================================
OUTCOME_PROSE = {
    "received_hc": "hemorrhage-control procedure (surgical or angiographic)",
    "death_hemorrhagic": ("hemorrhagic death (in-hospital, head AIS <4, "
                          "within 48 h of arrival)"),
    "substantial_transfusion": ("substantial transfusion (≥1000 mL or ≥4 U "
                                "packed red cells within 4 h)"),
    # sensitivity arms — labelled so none renders as a raw identifier
    "death_inhosp": "in-hospital death from any cause",
    "death_hemorrhagic_headais": "in-hospital death with head AIS <4",
    "death_le48h": "in-hospital death within 48 h of arrival",
    "death_ed": "death in the emergency department",
    "substantial_transfusion_ml_only": "substantial transfusion (≥1000 mL within 4 h)",
    "mtp_equivalent": ("MTP-equivalent activation (≥4 U or ≥1000 mL packed red "
                       "cells within 4 h)"),
}
FALLBACK_COMPOSITE_COMPONENTS = ("received_hc", "death_hemorrhagic",
                                 "substantial_transfusion")

# The identity statement CONTRACTS sec.4 requires wherever mtp_equivalent is
# shown: it is the transfusion component re-framed, never independent evidence.
MTP_IDENTITY_NOTE = (
    "MTP-equivalent activation is definitionally identical to the "
    "substantial-transfusion component of the primary composite: it is that "
    "component analysed standalone under the critical-administration-threshold "
    "framing, not independent corroboration of the primary result."
)


def _composite_components():
    """Component columns of the v3 primary composite; never raises."""
    man = _read_json(os.path.join(LOGS, "qc_manifest.json"))
    if isinstance(man, dict):
        expr = (man.get("primary_composite") or {}).get("definition", "")
        toks = [t.strip() for t in str(expr).split("|") if t.strip()]
        toks = [t for t in toks if t in OUTCOME_PROSE]
        if toks:
            return toks
    print("  NOTE: composite definition not readable from qc_manifest.json; "
          "using the CONTRACTS sec.4 fallback.")
    return list(FALLBACK_COMPOSITE_COMPONENTS)


def _composite_definition_text():
    """One-sentence prose definition of the v3 primary composite."""
    parts = [OUTCOME_PROSE.get(c, c.replace("_", " ")) for c in _composite_components()]
    body = (", ".join(parts[:-1]) + ", or " + parts[-1]) if len(parts) > 1 else \
        (parts[0] if parts else "(undefined)")
    return f"Major hemorrhage event (primary composite) = {body}."


def _figure_footnote(fig, text, y=-0.015, fontsize=8.2, wrap_chars=150):
    """Stamp a wrapped caption under a figure (included by bbox_inches='tight')."""
    lines = []
    for para in ([text] if isinstance(text, str) else list(text)):
        lines.extend(textwrap.wrap(str(para), width=wrap_chars) or [""])
    fig.text(0.008, y, "\n".join(lines), ha="left", va="top",
             fontsize=fontsize, color="#333333", style="italic")


SUBGROUP_LABELS = {
    "Age": "Age", "Sex": "Sex", "Ring_Stability": "Ring stability",
    "Anticoagulation": "Anticoagulation", "Mechanism": "Mechanism",
    "ACS_Level": "ACS level",
}


def _pretty_level(subgroup, level):
    s = str(level)
    repl = {">=65": "≥65", "<65": "<65", ">=75": "≥75"}
    s = repl.get(s, s)
    if subgroup == "ACS_Level":
        return f"Level {s}"
    return s


# =============================================================================
# FIGURE 2: SCORE-BY-OUTCOME distribution + risk-tier observed rates
# =============================================================================
def figure2_score_distribution():
    print("\n" + "=" * 64)
    print("Figure 2: Score distribution by outcome + risk tiers")
    print("=" * 64)

    df = _read_parquet(os.path.join(DERIVED, "cohort_test_scored.parquet"),
                       columns=["pelvic_score", "major_hemorrhage_event"])
    tiers = _read_csv(os.path.join(DERIVED, "risk_tiers.csv"))
    if df is None or "pelvic_score" not in df.columns or "major_hemorrhage_event" not in df.columns:
        print("  SKIP: scored cohort missing pelvic_score / major_hemorrhage_event.")
        return

    score = pd.to_numeric(df["pelvic_score"], errors="coerce")
    ev = pd.to_numeric(df["major_hemorrhage_event"], errors="coerce")
    m = score.notna() & ev.notna()
    score, ev = score[m].astype(int), ev[m].astype(int)
    if len(score) == 0:
        print("  SKIP: no valid scored rows.")
        return

    smin, smax = int(score.min()), int(score.max())
    bins = np.arange(smin - 0.5, smax + 1.5, 1.0)
    prev = float(ev.mean())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ----- Panel A: conditional score distributions by outcome (density) -----
    axA.hist(score[ev == 0], bins=bins, density=True, color=C_NOEVENT, alpha=0.62,
             label=f"No event (n={int((ev == 0).sum()):,})", edgecolor="white", linewidth=0.3)
    axA.hist(score[ev == 1], bins=bins, density=True, color=C_EVENT, alpha=0.62,
             label=f"Major hemorrhage event, composite (n={int((ev == 1).sum()):,})",
             edgecolor="white", linewidth=0.3)

    # Shade tier regions + annotate observed composite rate.
    if tiers is not None and {"tier", "score_min", "score_max"}.issubset(tiers.columns):
        tiers = tiers.set_index("tier")
        ymax = axA.get_ylim()[1]
        for t in TIER_ORDER:
            if t not in tiers.index:
                continue
            lo = float(tiers.loc[t, "score_min"]) - 0.5
            hi = float(tiers.loc[t, "score_max"]) + 0.5
            axA.axvspan(lo, hi, color=TIER_COLOR[t], alpha=0.08, zorder=0)
            rate = float(tiers.loc[t, "composite_rate"]) * 100 if "composite_rate" in tiers.columns else np.nan
            axA.text((lo + hi) / 2.0, ymax * 0.97,
                     f"{TIER_DISPLAY[t]}\n{rate:.1f}%",
                     ha="center", va="top", fontsize=9, fontweight="bold",
                     color=TIER_COLOR[t])
        for t in TIER_ORDER[:-1]:
            if t in tiers.index:
                axA.axvline(float(tiers.loc[t, "score_max"]) + 0.5, color="#444444",
                            linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    axA.set_xlabel("Pelvic arrival score (integer points)")
    axA.set_ylabel("Density within outcome group")
    axA.set_title(f"Score distribution by outcome (test set, N={len(score):,}; "
                  f"prevalence {prev * 100:.1f}%)", fontsize=12)
    axA.legend(loc="lower right", framealpha=0.92, fontsize=9)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.set_xlim(smin - 0.8, smax + 0.8)
    axA.text(-0.06, 1.05, "A", transform=axA.transAxes, fontsize=16, fontweight="bold", va="top")

    # ----- Panel B: observed composite rate by tier (with 95% CI) -----
    if tiers is not None and "composite_rate" in tiers.reset_index().columns:
        tt = tiers.reset_index() if tiers.index.name == "tier" else tiers
        tt = tt.set_index("tier")
        xs, heights, elo, ehi, cols, labs = [], [], [], [], [], []
        for i, t in enumerate(TIER_ORDER):
            if t not in tt.index:
                continue
            r = float(tt.loc[t, "composite_rate"]) * 100
            lo = float(tt.loc[t, "composite_ci_low"]) * 100 if "composite_ci_low" in tt.columns else r
            hi = float(tt.loc[t, "composite_ci_high"]) * 100 if "composite_ci_high" in tt.columns else r
            xs.append(i); heights.append(r); elo.append(max(0, r - lo)); ehi.append(max(0, hi - r))
            cols.append(TIER_COLOR[t])
            n = int(tt.loc[t, "n"]) if "n" in tt.columns else None
            labs.append(f"{TIER_DISPLAY[t]}\n(score {int(tt.loc[t, 'score_min'])} to "
                        f"{int(tt.loc[t, 'score_max'])})" + (f"\nN={n:,}" if n else ""))
        axB.bar(xs, heights, width=0.62, color=cols, edgecolor="white", linewidth=0.8, zorder=2)
        axB.errorbar(xs, heights, yerr=[elo, ehi], fmt="none", ecolor="#333333",
                     elinewidth=1.5, capsize=6, capthick=1.5, zorder=3)
        for x, h in zip(xs, heights):
            axB.text(x, h + max(heights) * 0.02, f"{h:.1f}%", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
        axB.set_xticks(xs)
        axB.set_xticklabels(labs, fontsize=9)
        axB.set_ylabel("Observed major-hemorrhage rate (%)")
        axB.set_ylim(0, max(heights) * 1.20)
        axB.set_title("Observed rate by risk tier", fontsize=12)
        axB.spines["top"].set_visible(False)
        axB.spines["right"].set_visible(False)
        axB.grid(axis="y", alpha=0.25, linestyle="--")
    else:
        axB.text(0.5, 0.5, "risk_tiers.csv unavailable", ha="center", va="center",
                 transform=axB.transAxes)
        axB.axis("off")
    axB.text(-0.10, 1.05, "B", transform=axB.transAxes, fontsize=16, fontweight="bold", va="top")

    plt.tight_layout(w_pad=3.0)
    # v3: the composite kept its name and changed its meaning — state it (D5).
    _figure_footnote(fig, _composite_definition_text())
    _save_figure(fig, "figure2_score_distribution")


# =============================================================================
# FIGURE 3: ROC overlay (score vs SI vs rSIG vs full-info) + temporal dashed
# =============================================================================
def _roc_curve(entry):
    """Return (fpr, tpr, auc, lo, hi) from a roc-dict entry, or None."""
    if not isinstance(entry, dict):
        return None
    try:
        fpr = np.asarray(entry["fpr"], dtype=float)
        tpr = np.asarray(entry["tpr"], dtype=float)
    except Exception:
        return None
    return fpr, tpr, entry.get("auc"), entry.get("lo"), entry.get("hi")


def figure3_roc():
    print("\n" + "=" * 64)
    print("Figure 3: ROC overlay (+ temporal 2023-24)")
    print("=" * 64)

    roc = _read_pickle(os.path.join(DERIVED, "roc_data.pkl"))
    if not isinstance(roc, dict):
        print("  SKIP: roc_data.pkl unavailable / not a dict.")
        return
    troc = _read_pickle(os.path.join(DERIVED, "temporal_roc_data.pkl"))  # optional overlay

    # (key-in-roc, display label, color) in draw order
    plan = [
        ("pelvic_score", "Pelvic score", C_SCORE),
        ("full_information_ISS_AIS", "Full information (ISS/AIS)", C_FULL),
        ("rSIG", "rSIG", C_RSIG),
        ("SI-alone", "Shock index alone", C_SI),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], linestyle=":", linewidth=1.2, zorder=1)

    drew = 0
    handles = []
    for key, label, col in plan:
        cur = _roc_curve(roc.get(key))
        if cur is None:
            print(f"  NOTE: roc entry '{key}' missing/invalid; skipping that curve.")
            continue
        fpr, tpr, auc, lo, hi = cur
        lw = 2.6 if key == "pelvic_score" else 1.8
        ax.plot(fpr, tpr, color=col, linewidth=lw, zorder=3,
                solid_capstyle="round")
        if auc is not None and lo is not None and hi is not None:
            leg = f"{label}: AUROC {auc:.3f} ({lo:.3f}–{hi:.3f})"
        elif auc is not None:
            leg = f"{label}: AUROC {auc:.3f}"
        else:
            leg = label
        handles.append(Patch(color=col, label=leg))
        drew += 1

    # Temporal 2023-24 dashed overlay (integer card == the deployed score).
    if isinstance(troc, dict):
        tkey = None
        for cand in ("temporal_composite_integer", "temporal_composite_logit"):
            if cand in troc:
                tkey = cand
                break
        cur = _roc_curve(troc.get(tkey)) if tkey else None
        if cur is not None:
            fpr, tpr, auc, lo, hi = cur
            yrs = troc[tkey].get("valid_years", "2023-2024")
            ax.plot(fpr, tpr, color=C_TEMPORAL, linewidth=2.0, linestyle="--", zorder=2)
            leg = f"Temporal {yrs} (score)"
            if auc is not None:
                leg += f": AUROC {auc:.3f}"
                if lo is not None and hi is not None:
                    leg += f" ({lo:.3f}–{hi:.3f})"
            handles.append(Patch(facecolor="none", edgecolor=C_TEMPORAL, hatch="",
                                 label=leg))
            # dashed line proxy for the legend
            handles[-1] = plt.Line2D([0], [0], color=C_TEMPORAL, linestyle="--",
                                     linewidth=2.0, label=leg)
        else:
            print("  NOTE: temporal_roc_data.pkl present but no usable temporal curve.")

    if drew == 0:
        print("  SKIP: no ROC curves could be drawn.")
        plt.close(fig)
        return

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("1 − specificity (false-positive rate)")
    ax.set_ylabel("Sensitivity (true-positive rate)")
    ax.set_title("Discrimination for major hemorrhage: internal test + temporal validation",
                 fontsize=12)
    ax.legend(handles=handles, loc="lower right", framealpha=0.94, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    _save_figure(fig, "figure3_roc")


# =============================================================================
# FIGURE 4: Calibration (overall + elderly)
# =============================================================================
def figure4_calibration():
    print("\n" + "=" * 64)
    print("Figure 4: Calibration (overall + elderly)")
    print("=" * 64)

    cal = _read_csv(os.path.join(DERIVED, "calibration_data.csv"))
    reclass = _read_json(os.path.join(DERIVED, "reclass.json"))
    ecd = reclass.get("elderly_calibration_deciles") if isinstance(reclass, dict) else None

    have_overall = cal is not None and {"mean_pred", "obs_frac"}.issubset(cal.columns)
    have_elderly = isinstance(ecd, list) and len(ecd) > 0
    if not have_overall and not have_elderly:
        print("  SKIP: neither overall nor elderly calibration data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))

    def _panel(ax, xs, ys, lo, hi, title, n, citl, letter):
        lim = max(0.02, float(np.nanmax([np.nanmax(xs), np.nanmax(ys)])) * 1.08)
        ax.plot([0, lim], [0, lim], color=OKABE_ITO["grey"], linestyle=":", linewidth=1.3,
                label="Perfect calibration", zorder=1)
        if lo is not None and hi is not None:
            yerr = [np.clip(np.asarray(ys) - np.asarray(lo), 0, None),
                    np.clip(np.asarray(hi) - np.asarray(ys), 0, None)]
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=C_SCORE, alpha=0.55,
                        elinewidth=1.3, capsize=3, zorder=2)
        ax.plot(xs, ys, marker="o", color=C_SCORE, linewidth=1.8, markersize=7,
                markerfacecolor="white", markeredgecolor=C_SCORE, markeredgewidth=1.8,
                zorder=3, label="Observed vs predicted")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Mean predicted probability of major hemorrhage")
        ax.set_ylabel("Observed major-hemorrhage frequency")
        sub = title
        if n:
            sub += f"  (N={n:,})"
        ax.set_title(sub, fontsize=12)
        if citl is not None and np.isfinite(citl):
            ax.text(0.04, 0.96, f"CITL = {citl:+.3f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OKABE_ITO["grey"], alpha=0.9))
        ax.legend(loc="lower right", framealpha=0.92, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, linestyle="--")
        ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top")

    # ----- Panel A: overall (pelvic_score deciles) -----
    if have_overall:
        c = cal.copy()
        if "score" in c.columns and (c["score"] == "pelvic_score").any():
            c = c[c["score"] == "pelvic_score"]
        c = c.sort_values(_flex_col(c, ["bin", "decile"]) or c.columns[0])
        n = int(c["n"].iloc[0]) if "n" in c.columns and len(c) else None
        citl = float(c["citl"].iloc[0]) if "citl" in c.columns and len(c) else None
        _panel(axes[0], c["mean_pred"].to_numpy(float), c["obs_frac"].to_numpy(float),
               None, None, "Overall calibration (pelvic score)", n, citl, "A")
    else:
        axes[0].text(0.5, 0.5, "Overall calibration unavailable", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].axis("off")

    # ----- Panel B: elderly (>=65) deciles with CI -----
    if have_elderly:
        e = pd.DataFrame(ecd)
        e = e.sort_values("decile") if "decile" in e.columns else e
        xs = e["mean_pred"].to_numpy(float)
        ys = e["obs_frac"].to_numpy(float)
        lo = e["obs_ci_low"].to_numpy(float) if "obs_ci_low" in e.columns else None
        hi = e["obs_ci_high"].to_numpy(float) if "obs_ci_high" in e.columns else None
        n = int(e["n"].sum()) if "n" in e.columns else None
        ecitl = None
        if isinstance(reclass, dict):
            ec = reclass.get("elderly_calibration")
            if isinstance(ec, dict):
                ecitl = ec.get("citl")
        _panel(axes[1], xs, ys, lo, hi, "Elderly (≥65) calibration", n, ecitl, "B")
    else:
        axes[1].text(0.5, 0.5, "Elderly calibration unavailable", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].axis("off")

    plt.tight_layout(w_pad=3.0)
    _figure_footnote(fig, _composite_definition_text())
    _save_figure(fig, "figure4_calibration")


# =============================================================================
# FIGURE 5: DECISION-CURVE ANALYSIS  (CENTERPIECE; overall + elderly)
# =============================================================================
def _dca_panel(ax, dca, title, letter, include_rsig=False):
    """Draw one DCA panel from a dca_data frame. Returns True if drawn."""
    need = {"threshold", "score", "net_benefit"}
    if dca is None or not need.issubset(dca.columns):
        return False
    dca = dca.copy()
    dca["threshold"] = pd.to_numeric(dca["threshold"], errors="coerce")
    dca["net_benefit"] = pd.to_numeric(dca["net_benefit"], errors="coerce")

    # treat-all / treat-none (carried per row; identical across scores at a threshold)
    base = dca.drop_duplicates("threshold").sort_values("threshold")
    thr = base["threshold"].to_numpy(float)
    if "treat_all_nb" in dca.columns:
        ax.plot(thr, base["treat_all_nb"].to_numpy(float), color=C_TREATALL,
                linewidth=1.6, linestyle="-", label="Treat all", zorder=2)
    ax.axhline(0.0, color=C_TREATNONE, linewidth=1.4, linestyle="-", label="Treat none", zorder=2)

    # comparators / score
    series = [("pelvic_score", "Pelvic score", C_SCORE, 2.8, "-"),
              ("SI-alone", "Shock index alone", C_SI, 1.9, "-")]
    if include_rsig:
        series.append(("rSIG", "rSIG", C_RSIG, 1.6, "-"))
    for key, label, col, lw, ls in series:
        sub = dca[dca["score"] == key].sort_values("threshold")
        if len(sub) == 0:
            continue
        ax.plot(sub["threshold"].to_numpy(float), sub["net_benefit"].to_numpy(float),
                color=col, linewidth=lw, linestyle=ls, label=label, zorder=4,
                solid_capstyle="round")

    # rule-in decision threshold + pre-specified window
    ax.axvspan(DCA_PRESPEC[0], DCA_PRESPEC[1], color=OKABE_ITO["yellow"], alpha=0.10, zorder=0)
    ax.axvline(RULE_IN_PT, color="#444444", linestyle="--", linewidth=1.3, zorder=3)
    ymax = ax.get_ylim()[1]
    ax.text(RULE_IN_PT, ymax * 0.02, f"  rule-in threshold\n  (p$_t$={RULE_IN_PT:.2f})",
            rotation=0, ha="left", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xlim(thr.min(), thr.max())
    # y floor a touch below 0 so treat-none is visible
    ax.set_ylim(min(-0.01, ax.get_ylim()[0]), None)
    ax.set_xlabel("Threshold probability p$_t$")
    ax.set_ylabel("Net benefit")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", framealpha=0.94, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--")
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    return True


def figure5_dca():
    print("\n" + "=" * 64)
    print("Figure 5: Decision-curve analysis (CENTERPIECE)")
    print("=" * 64)

    dca = _read_csv(os.path.join(DERIVED, "dca_data.csv"))
    edca = _read_csv(os.path.join(DERIVED, "elderly_dca_data.csv"))
    if dca is None and edca is None:
        print("  SKIP: neither overall nor elderly DCA data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    ok_a = _dca_panel(axes[0], dca, "Decision-curve analysis: overall test set", "A",
                      include_rsig=False)
    if not ok_a:
        axes[0].text(0.5, 0.5, "Overall DCA unavailable", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].axis("off")
    ok_b = _dca_panel(axes[1], edca, "Decision-curve analysis: elderly (≥65)", "B",
                      include_rsig=False)
    if not ok_b:
        axes[1].text(0.5, 0.5, "Elderly DCA unavailable", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].axis("off")

    if not ok_a and not ok_b:
        print("  SKIP: no DCA panel could be drawn.")
        plt.close(fig)
        return

    plt.tight_layout(w_pad=3.0)
    _figure_footnote(fig, _composite_definition_text())
    _save_figure(fig, "figure5_dca")


# =============================================================================
# FIGURE 6: Subgroup forest -- DeLong dAUROC (pelvic - comparator) by stratum
# =============================================================================
def figure6_subgroup_forest():
    print("\n" + "=" * 64)
    print("Figure 6: Subgroup forest (DeLong dAUROC vs comparators)")
    print("=" * 64)

    df = _read_csv(os.path.join(DERIVED, "subgroup_forest_data.csv"))
    need = {"Subgroup", "Level", "Score", "Delta_vs_pelvic"}
    if df is None or not need.issubset(df.columns):
        print(f"  SKIP: subgroup_forest_data.csv missing required columns "
              f"(have {list(df.columns) if df is not None else None}).")
        return

    comp_color = {"SI-alone": C_SI, "rSIG": C_RSIG}
    comp_off = {"SI-alone": +0.16, "rSIG": -0.16}

    # Preserve first-seen subgroup + level ordering.
    subgroups = list(dict.fromkeys(df["Subgroup"]))
    rows = []  # (y, kind, ...) built top-down
    y = 0.0
    ytick_pos, ytick_lab = [], []
    for sg in subgroups:
        rows.append(("header", y, sg))
        y -= 1.0
        levels = list(dict.fromkeys(df[df["Subgroup"] == sg]["Level"]))
        for lv in levels:
            block = df[(df["Subgroup"] == sg) & (df["Level"] == lv)]
            pauc = block["pelvic_AUROC"].iloc[0] if "pelvic_AUROC" in block.columns else np.nan
            ytick_pos.append(y)
            lab = _pretty_level(sg, lv)
            if np.isfinite(pauc):
                lab += f"  (score AUROC {pauc:.3f})"
            ytick_lab.append(lab)
            for _, r in block.iterrows():
                rows.append(("point", y, r))
            y -= 1.0
        y -= 0.4  # gap between subgroups

    fig_h = max(8.0, (-y) * 0.40 + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.2, zorder=1)

    seen = set()
    for kind, yy, payload in rows:
        if kind == "header":
            ax.text(-0.002, yy, SUBGROUP_LABELS.get(str(payload), str(payload)),
                    fontsize=11, fontweight="bold", ha="right", va="center")
            continue
        r = payload
        sc = str(r["Score"])
        col = comp_color.get(sc, OKABE_ITO["grey"])
        off = comp_off.get(sc, 0.0)
        d = float(r["Delta_vs_pelvic"])
        lo = float(r["Delta_CI_Low"]) if "Delta_CI_Low" in r and pd.notna(r["Delta_CI_Low"]) else d
        hi = float(r["Delta_CI_High"]) if "Delta_CI_High" in r and pd.notna(r["Delta_CI_High"]) else d
        ax.errorbar(d, yy + off, xerr=[[max(0, d - lo)], [max(0, hi - d)]], fmt="o",
                    color=col, ecolor=col, elinewidth=1.6, capsize=3, markersize=6,
                    zorder=3, label=sc if sc not in seen else None)
        seen.add(sc)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=9)
    ax.set_ylim(y + 0.5, 1.0)
    ax.set_xlabel("ΔAUROC = pelvic score − comparator  (positive favors pelvic score)")
    ax.set_title("Subgroup discrimination advantage over shock-index comparators (DeLong)",
                 fontsize=12)
    # Legend from the two comparator colors.
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=comp_color[k],
                          markersize=8, label=k) for k in ("SI-alone", "rSIG") if k in df["Score"].values]
    ax.legend(handles=handles, loc="lower right", framealpha=0.94, fontsize=9,
              title="Comparator")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linestyle="--")

    plt.tight_layout()
    _save_figure(fig, "figure6_subgroup_forest")


# =============================================================================
# FIGURE 7: Headline threshold splines (SI & SBP; elderly vs young; change-points)
# =============================================================================
def _spline_panel(ax, df, exposure_label, letter, changepoint=None,
                  vlines=None, cp_annotate=True, legend_loc="upper right"):
    """Plot overall/elderly/young RCS curves with CI bands from a spline CSV."""
    xcol = _flex_col(df, ["exposure_grid", "grid", "x"])
    if xcol is None or "pred_risk" not in df.columns:
        return False
    df = df.sort_values(xcol)
    x = df[xcol].to_numpy(float)

    def _band(pred, lci, uci, color, label, lw=2.2, ls="-", band=True):
        if pred not in df.columns:
            return
        y = df[pred].to_numpy(float)
        ax.plot(x, y * 100, color=color, linewidth=lw, linestyle=ls, label=label, zorder=4)
        if band and lci in df.columns and uci in df.columns:
            ax.fill_between(x, df[lci].to_numpy(float) * 100, df[uci].to_numpy(float) * 100,
                            color=color, alpha=0.14, zorder=2, linewidth=0)

    # Elderly + young emphasized; overall as thin grey reference.
    _band("pred_risk", "lci", "uci", C_OVERALL, "Overall", lw=1.6, ls="-", band=False)
    _band("pred_risk_young", "lci_young", "uci_young", C_YOUNG, "Young (<65)")
    _band("pred_risk_elderly", "lci_elderly", "uci_elderly", C_ELDERLY, "Elderly (≥65)")

    # Change-point (SBP only) with CI shading.
    #
    # v3: `11` demotes an under-powered stratum to DESCRIPTIVE-ONLY (the locked
    # SAP's >=100-events-per-side guard) and still writes the key, with
    # `breakpoint: NaN` and `descriptive_only: true`. The mortality anchor lost
    # most of its events under the new death definition, so this path is
    # expected, not hypothetical: a `.get(key, default)` fallback never fires on
    # a present-but-NaN key, and NaN would otherwise reach axvline() and print
    # "change-point nan". Branch on the flag AND re-check finiteness.
    if changepoint is not None and cp_annotate:
        cp = _changepoint_block(changepoint)
        bp = _finite(cp.get("breakpoint"))
        if cp.get("descriptive_only") or bp is None:
            reason = cp.get("guard_reason") or (
                "below the pre-specified events-per-side guard"
                if cp.get("descriptive_only") else "not estimable")
            print(f"  NOTE: change-point reported descriptive-only ({reason}); "
                  f"the spline is drawn without an inflection annotation.")
            ax.text(0.99, 0.02, f"Change-point descriptive-only: {reason}.",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    color="#555555", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=OKABE_ITO["grey"], alpha=0.9))
        else:
            lo, hi = _finite(cp.get("lci")), _finite(cp.get("uci"))
            if lo is not None and hi is not None and hi > lo:
                ax.axvspan(lo, hi, color=OKABE_ITO["grey"], alpha=0.18, zorder=1)
            ax.axvline(bp, color=OKABE_ITO["black"], linestyle="--", linewidth=1.6,
                       zorder=5)
            ytop = ax.get_ylim()[1]
            ci_txt = (f" (95% CI {lo:.0f}–{hi:.0f})"
                      if lo is not None and hi is not None else "")
            ax.text(bp, ytop * 0.92, f"change-point\n{bp:.0f}{ci_txt}", ha="center",
                    va="top", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=OKABE_ITO["black"], alpha=0.9))

    # pre-specified vertical thresholds (SI: 1.0 / 1.3)
    if vlines:
        for xv in vlines:
            ax.axvline(xv, color=OKABE_ITO["purple"], linestyle=":", linewidth=1.5, zorder=3)
            ax.text(xv, ax.get_ylim()[1] * 0.06, f" {xv:g}", color=OKABE_ITO["purple"],
                    ha="left", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel(exposure_label)
    ax.set_ylabel("Predicted major-hemorrhage risk (%)")
    ax.legend(loc=legend_loc, framealpha=0.94, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--")
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    return True


def figure7_threshold_splines():
    print("\n" + "=" * 64)
    print("Figure 7: Headline threshold splines (SI & SBP)")
    print("=" * 64)

    si = _read_csv(os.path.join(DERIVED, "spline_si_composite.csv"))
    sbp = _read_csv(os.path.join(DERIVED, "spline_sbp_composite.csv"))
    cp_sbp = _read_json(os.path.join(DERIVED, "changepoint_sbp_composite.json"))
    # SI change-point is a grid-boundary/monotone case -> do NOT draw an inflection line.

    if si is None and sbp is None:
        print("  SKIP: no spline data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))

    ok_si = False
    if si is not None:
        ok_si = _spline_panel(axes[0], si, "Shock index on arrival", "A",
                              changepoint=None, vlines=(1.0, 1.3), cp_annotate=False,
                              legend_loc="upper left")
        if ok_si:
            axes[0].set_title("Shock index (pre-specified thresholds 1.0 / 1.3)", fontsize=12)
    if not ok_si:
        axes[0].text(0.5, 0.5, "SI spline unavailable", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].axis("off")

    ok_sbp = False
    if sbp is not None:
        ok_sbp = _spline_panel(axes[1], sbp, "Systolic blood pressure on arrival (mmHg)", "B",
                               changepoint=cp_sbp, vlines=None, cp_annotate=True)
        if ok_sbp:
            _cp = _changepoint_block(cp_sbp)
            annotated = (not _cp.get("descriptive_only")
                         and _finite(_cp.get("breakpoint")) is not None)
            axes[1].set_title(
                "Systolic blood pressure (change-point annotated)" if annotated
                else "Systolic blood pressure on arrival", fontsize=12)
    if not ok_sbp:
        axes[1].text(0.5, 0.5, "SBP spline unavailable", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].axis("off")

    if not ok_si and not ok_sbp:
        print("  SKIP: no spline panel could be drawn.")
        plt.close(fig)
        return

    plt.tight_layout(w_pad=3.0)
    _save_figure(fig, "figure7_threshold_splines")


# =============================================================================
# FIGURE 8: Component-decomposition splines (from script 13; may be absent)
# =============================================================================
# v3: the composite's death arm is `death_hemorrhagic` (D-1 + D-8), NOT
# `death_inhosp`. The v2 name keeps a label and a distinct (grey) color so a
# SENSITIVITY arm can still be plotted without being mistaken for the primary
# anchor and without ever rendering as a raw identifier.
COMPONENT_LABELS = {
    "received_hc": "Hemorrhage-control receipt",
    "death_hemorrhagic": "Hemorrhagic death (head AIS <4, ≤48 h)",
    "substantial_transfusion": "Substantial transfusion (≥1000 mL or ≥4 U)",
    "major_hemorrhage_event": "Composite (any component)",
    # sensitivity / ladder arms
    "death_inhosp": "In-hospital death, any cause (sensitivity)",
    "death_hemorrhagic_headais": "Death, head AIS <4 only (sensitivity)",
    "death_le48h": "Death ≤48 h only (sensitivity)",
    "death_ed": "Death in the emergency department (sensitivity)",
    "substantial_transfusion_ml_only": "Substantial transfusion, ≥1000 mL only (sensitivity)",
    "patient_state_composite": "Patient-state composite (hemorrhage control dropped)",
    "mtp_equivalent": "MTP-equivalent activation",
}
COMPONENT_COLOR = {
    "received_hc": OKABE_ITO["blue"],
    "death_hemorrhagic": OKABE_ITO["vermilion"],
    "substantial_transfusion": OKABE_ITO["green"],
    "major_hemorrhage_event": OKABE_ITO["black"],
    "death_inhosp": OKABE_ITO["grey"],
    "death_hemorrhagic_headais": OKABE_ITO["orange"],
    "death_le48h": OKABE_ITO["purple"],
    "death_ed": OKABE_ITO["sky_blue"],
    "substantial_transfusion_ml_only": OKABE_ITO["yellow"],
    "patient_state_composite": OKABE_ITO["purple"],
    "mtp_equivalent": OKABE_ITO["green"],
}
EXPOSURE_LABEL = {"shock_index": "Shock index on arrival",
                  "sbp2": "Systolic blood pressure on arrival (mmHg)"}


def figure8_component():
    print("\n" + "=" * 64)
    print("Figure 8: Component-decomposition splines")
    print("=" * 64)

    path = os.path.join(DERIVED, "component_decomp_data.csv")
    df = _read_csv(path)
    if df is None:
        print("  SKIP: component_decomp_data.csv not found "
              "(script 13 not finished) -- re-run `--only 8` after 13 completes.")
        return

    xcol = _flex_col(df, ["exposure_grid", "grid", "x"])
    ccol = _flex_col(df, ["component", "outcome_component", "comp"])
    ecol = _flex_col(df, ["exposure", "exposure_name"])
    if xcol is None or "pred_risk" not in df.columns or ccol is None or ecol is None:
        print(f"  SKIP: component_decomp_data.csv present but schema incomplete "
              f"(cols={list(df.columns)}); expected exposure/component/exposure_grid/pred_risk.")
        return

    exposures = list(dict.fromkeys(df[ecol]))
    if len(exposures) == 0:
        print("  SKIP: no exposures in component_decomp_data.csv.")
        return

    # v3-J outcome-name sweep: the composite's death arm is `death_hemorrhagic`.
    # Seeing `death_inhosp` here is legitimate ONLY as a sensitivity arm, so say
    # so in the log rather than silently drawing it as the primary anchor.
    seen_components = list(dict.fromkeys(df[ccol]))
    if "death_inhosp" in seen_components and "death_hemorrhagic" not in seen_components:
        print("  NOTE: component data carries `death_inhosp` and no "
              "`death_hemorrhagic` — in v3 the composite's death arm is "
              "`death_hemorrhagic` (CONTRACTS sec.4). Plotted as the labelled "
              "sensitivity arm; confirm script 13 was re-pointed.")
    unknown = [c for c in seen_components if c not in COMPONENT_LABELS]
    if unknown:
        print(f"  NOTE: unlabelled component(s) {unknown} — rendering humanized "
              f"names; add them to COMPONENT_LABELS if they are permanent.")

    fig, axes = plt.subplots(1, len(exposures), figsize=(7.6 * len(exposures), 6.2),
                             squeeze=False)
    axes = axes[0]
    drew = 0
    for ax, exp in zip(axes, exposures):
        sub = df[df[ecol] == exp].sort_values(xcol)
        x = sub[xcol].to_numpy(float)
        comps = list(dict.fromkeys(sub[ccol]))
        for comp in comps:
            b = sub[sub[ccol] == comp].sort_values(xcol)
            col = COMPONENT_COLOR.get(comp, OKABE_ITO["grey"])
            clab = COMPONENT_LABELS.get(
                comp, _sanitize_label(str(comp).replace("_", " ")))
            ax.plot(b[xcol].to_numpy(float), b["pred_risk"].to_numpy(float) * 100,
                    color=col, linewidth=2.2, label=clab, zorder=3)
            if "lci" in b.columns and "uci" in b.columns:
                ax.fill_between(b[xcol].to_numpy(float), b["lci"].to_numpy(float) * 100,
                                b["uci"].to_numpy(float) * 100, color=col, alpha=0.12,
                                zorder=1, linewidth=0)
        ax.set_xlabel(EXPOSURE_LABEL.get(exp, str(exp)))
        ax.set_ylabel("Predicted component risk (%)")
        ax.set_title(f"Component decomposition: {EXPOSURE_LABEL.get(exp, exp)}", fontsize=12)
        ax.legend(loc="upper right", framealpha=0.94, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22, linestyle="--")
        ax.text(-0.08, 1.05, chr(65 + drew), transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top")
        drew += 1

    if drew == 0:
        print("  SKIP: nothing drawn for component decomposition.")
        plt.close(fig)
        return
    plt.tight_layout(w_pad=3.0)
    _save_figure(fig, "figure8_component")


# =============================================================================
# FIGURE 9: Bedside score card + automatic-escalate override + risk tiers
#
# v3 (Fix 3 / Fix 4; CONTRACTS sec.5.1 + sec.8 "F9 -- v3 required changes"):
#   1. NO `unstable_ring` row (gate G3d) -- filtered unconditionally.
#   2. `sbp_unrecordable` is NOT a point row. It renders as an automatic-escalate
#      BANNER (gate G4a/G4e) carrying no point value.
#   3. The stated maximum is the OBSERVED test-set maximum from
#      score_card.json["score_range"], not the points-column sum (gate G4c).
#
# The three drawing helpers below are deliberately standalone (ax-in, nothing
# global) so 18_wta_figure.py can copy-adapt the card into its blinded .jpg
# without re-deriving the override contract. See the HANDOFF note in the report.
# =============================================================================
def _card_point_rows(pts, card, labels):
    """Point-role card rows only.

    Returns ([(variable, points, label)], notes, source). Enforces both card
    exclusions locally so the figure satisfies G3d/G4a even against a stale
    artifact: `Role != "point"` rows and CARD_EXCLUDED_VARS never reach the grid.
    """
    items, notes, source = [], [], None

    if pts is not None:
        vcol = _flex_col(pts, ["Variable", "variable"])
        pcol = _flex_col(pts, ["Points", "points"])
        rcol = _flex_col(pts, ["Role", "role"])
        if vcol and pcol:
            source = "table3_points.csv"
            if rcol is None:
                notes.append(
                    "table3_points.csv has no `Role` column (pre-Fix-4 artifact); "
                    f"falling back to the override name list {list(OVERRIDE_VARS)}.")
            for _, r in pts.iterrows():
                var = str(r[vcol]).strip()
                role = (str(r[rcol]).strip().lower() if rcol is not None
                        else (ROLE_OVERRIDE if var in OVERRIDE_VARS else ROLE_POINT))
                if var in CARD_EXCLUDED_VARS:
                    notes.append(
                        f"GATE G3d: `{var}` was present in table3_points.csv and has "
                        f"been EXCLUDED from the card (Fix 3 removed it from the "
                        f"score). Check script 02/05 if this persists.")
                    continue
                if role != ROLE_POINT:
                    continue                       # override rows -> banner
                try:
                    p = int(round(float(r[pcol])))
                except (TypeError, ValueError):
                    notes.append(f"`{var}` has Role='{role}' but an unreadable Points "
                                 f"cell ({r[pcol]!r}); row omitted from the card.")
                    continue
                items.append((var, p, _var_label(var, labels)))

    if not items and isinstance(card, dict):
        source = "score_card.json[variables]"
        for v in card.get("variables", []):
            var = str(v.get("variable", "")).strip()
            if var in CARD_EXCLUDED_VARS or var in OVERRIDE_VARS:
                notes.append(f"`{var}` filtered out of score_card.json[variables] "
                             f"(it must not be a point row in v3).")
                continue
            try:
                p = int(round(float(v.get("points"))))
            except (TypeError, ValueError):
                continue
            items.append((var, p, _var_label(var, labels)))

    items.sort(key=lambda t: t[1], reverse=True)
    return items, notes, source


def _card_override_block(card, tiers, labels, rule_text):
    """The automatic-escalate override descriptor, or None.

    Preference: score_card.json["override"] (CONTRACTS sec.5.1) -> the `override`
    row of risk_tiers.csv -> None (banner skipped, card still renders).
    """
    blk = None
    if isinstance(card, dict):
        raw = card.get("override")
        if isinstance(raw, dict) and raw:
            blk = dict(raw)
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            blk = dict(raw[0])

    if blk is None and tiers is not None and "tier" in tiers.columns:
        row = tiers[tiers["tier"].astype(str).str.strip().str.lower() == ROLE_OVERRIDE]
        if len(row):
            r = row.iloc[0]
            blk = {"variable": OVERRIDE_VARS[0] if OVERRIDE_VARS else "",
                   "n": r.get("n"),
                   "event_rate": r.get("composite_rate")}
            print("  NOTE: score_card.json carries no 'override' block; using the "
                  "`override` row of risk_tiers.csv instead.")

    if blk is None:
        return None

    var = str(blk.get("variable") or (OVERRIDE_VARS[0] if OVERRIDE_VARS else ""))
    blk["variable"] = var
    blk["label"] = _sanitize_label(blk.get("label") or _var_label(var, labels))
    blk["rule_text"] = _sanitize_label(
        blk.get("rule_text") or rule_text or OVERRIDE_RULE_TEXT_FALLBACK)
    return blk


def _draw_override_banner(ax, blk, y_top, height, n_test=None, rule_in=None):
    """Full-width automatic-escalate callout, visually unlike a point row.

    Point rows: no border, flat #F4F6F8 zebra fill, solid badge holding a signed
    integer. The banner: solid vermilion border + tinted fill + a solid accent
    bar (the same vermilion as the `escalate` tier), a bold uppercase headline,
    and an OUTLINED badge reading "NO POINTS" where a point value would sit --
    so the card states, rather than merely omits, that this rule bypasses the sum.
    """
    x0, x1 = 0.015, 0.985
    ax.add_patch(FancyBboxPatch((x0, y_top - height), x1 - x0, height,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=C_OVERRIDE, ec="none", alpha=0.13, zorder=1))
    ax.add_patch(FancyBboxPatch((x0, y_top - height), x1 - x0, height,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="none", ec=C_OVERRIDE, linewidth=1.6, zorder=3))
    ax.add_patch(FancyBboxPatch((x0, y_top - height), 0.020, height,
                                boxstyle="square,pad=0", fc=C_OVERRIDE, ec="none",
                                zorder=4))

    # Headline / criterion: split 05's canonical rule sentence on its colon so
    # the contract string (CONTRACTS sec.8 F9-2) is what the reader sees.
    rule = str(blk.get("rule_text") or OVERRIDE_RULE_TEXT_FALLBACK)
    if ":" in rule:
        head, crit = rule.split(":", 1)
        head, crit = head.strip().upper(), crit.strip()
        crit = crit[:1].upper() + crit[1:] if crit else str(blk.get("label", ""))
    else:
        head, crit = OVERRIDE_BANNER_HEADLINE.upper(), str(blk.get("label", ""))

    xt = 0.045
    ax.text(xt, y_top - 0.020, head, fontsize=11.0, fontweight="bold",
            color=C_OVERRIDE, va="top", zorder=5)
    ax.text(xt, y_top - 0.048, crit, fontsize=10.5, color="#111111", va="top", zorder=5)

    # Descriptives — whatever 08 supplied, never invented here. Percentages are
    # RECOMPUTED from the supplied counts wherever both are present: the override
    # group is <1% of the test set, so a "is this a fraction or a percent?"
    # heuristic on `pct_of_test` would be ambiguous exactly here.
    bits = []
    n_ov = blk.get("n")
    if n_ov is not None and pd.notna(n_ov):
        n_ov = float(n_ov)
        if n_test:
            pct = 100.0 * n_ov / float(n_test)
        elif blk.get("pct_of_test") is not None:
            pct = float(blk["pct_of_test"])        # key is named pct -> percent
        else:
            pct = None
        frac = f" ({pct:.2f}% of the test set)" if pct is not None else ""
        bits.append(f"n = {int(n_ov):,}{frac}")
    ev_ov, er = blk.get("events"), blk.get("event_rate")
    if ev_ov is not None and pd.notna(ev_ov) and n_ov:
        er = 100.0 * float(ev_ov) / float(n_ov)
    elif er is not None and pd.notna(er):
        er = float(er)
        er = er * 100.0 if er <= 1.0 else er
    else:
        er = None
    if er is not None:
        bits.append(f"observed major-hemorrhage rate {er:.1f}%")
    orv = blk.get("or", blk.get("odds_ratio"))
    if orv is not None and pd.notna(orv):
        bits.append(f"adjusted OR {float(orv):.1f}")
    if bits:
        # Keep clear of the badge at x>=0.865: shrink before overlapping it.
        txt = "   ·   ".join(bits)
        ax.text(xt, y_top - 0.074, txt, fontsize=(8.2 if len(txt) <= 92 else 7.3),
                color="#555555", va="top", zorder=5)

    # 08 supplies the deployed rule verbatim; compose one only if it did not.
    deployed = blk.get("deployed_rule")
    if deployed:
        rule_line = f"Carries no points — {_sanitize_label(deployed).rstrip('.')}."
    elif rule_in is not None and np.isfinite(rule_in):
        rule_line = (f"Carries no points — escalate if score ≥ {int(rule_in)} OR this "
                     f"criterion is present.")
    else:
        rule_line = "Carries no points — escalation is automatic."
    ax.text(xt, y_top - 0.096, rule_line,
            fontsize=(8.0 if len(rule_line) <= 96 else 7.2),
            color="#555555", style="italic", va="top", zorder=5)

    # The override's marginal cost at the rule-in cutpoint (CONTRACTS sec.13 G4d):
    # bounded and visible on the card, never hidden inside an aggregate.
    mf = blk.get("marginal_flagged_at_rule_in")
    me = blk.get("marginal_events_at_rule_in")
    if mf is not None and pd.notna(mf):
        extra = (f"Beyond the score alone this flags {int(float(mf)):,} additional "
                 f"patients")
        if me is not None and pd.notna(me):
            extra += f", of whom {int(float(me)):,} had a major hemorrhage event"
        ax.text(xt, y_top - 0.116, extra + ".", fontsize=7.6, color="#666666",
                va="top", zorder=5)

    # Outlined "NO POINTS" badge where a point value would otherwise sit.
    by = y_top - height / 2.0
    ax.add_patch(FancyBboxPatch((0.865, by - 0.026), 0.10, 0.052,
                                boxstyle="round,pad=0.006", fc="white",
                                ec=C_OVERRIDE, linewidth=1.2, linestyle="--", zorder=5))
    ax.text(0.915, by, "NO\nPOINTS", fontsize=7.6, fontweight="bold", color=C_OVERRIDE,
            ha="center", va="center", linespacing=1.05, zorder=6)


def _draw_point_grid(ax, items, top, bot):
    """Zebra-striped criterion/points grid. Rows carry no border (cf. banner)."""
    n = max(len(items), 1)
    row_h = (top - bot) / n
    ax.text(0.03, top + 0.012, "Criterion", fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.965, top + 0.012, "Points", fontsize=10, fontweight="bold",
            va="bottom", ha="right")
    for i, (_var, p, label) in enumerate(items):
        yc = top - (i + 0.5) * row_h
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch((0.015, yc - row_h / 2 + 0.004), 0.97,
                                        row_h - 0.008, boxstyle="round,pad=0.002",
                                        fc="#F4F6F8", ec="none", zorder=0))
        ax.text(0.03, yc, label, fontsize=10.5, va="center")
        badge = C_EVENT if p > 0 else (C_NOEVENT if p < 0 else OKABE_ITO["grey"])
        ax.add_patch(FancyBboxPatch((0.865, yc - row_h * 0.34), 0.10, row_h * 0.68,
                                    boxstyle="round,pad=0.006", fc=badge, ec="none",
                                    zorder=2))
        ax.text(0.915, yc, f"{p:+d}", fontsize=11, fontweight="bold", color="white",
                ha="center", va="center", zorder=3)


def _draw_risk_tiers(ax, tiers, card, has_override=False):
    """Traffic-light tier cards (score tiers only; an `override` row is skipped)."""
    ax.text(0.02, 0.985, "Risk stratification (test set)", fontsize=15,
            fontweight="bold", va="top")
    tlist = []
    if tiers is not None and {"tier", "composite_rate"}.issubset(tiers.columns):
        tmap = tiers.set_index("tier")
        for t in TIER_ORDER:
            if t in tmap.index:
                tlist.append((t, tmap.loc[t]))
    elif isinstance(card, dict):
        for row in card.get("risk_tiers", []):
            if row.get("tier") in TIER_ORDER:      # never the override row
                tlist.append((row.get("tier"), pd.Series(row)))
    if not tlist:
        return

    card_top, card_bot = 0.90, 0.14
    ch = (card_top - card_bot) / len(tlist) - 0.03
    action = {"rule_out": "Consider de-escalation / discharge pathway",
              "observe": "Serial observation; low threshold to escalate",
              "escalate": "Activate hemorrhage-control resources"}
    for i, (t, r) in enumerate(tlist):
        ytop = card_top - i * (ch + 0.03)
        col = TIER_COLOR.get(t, OKABE_ITO["grey"])
        ax.add_patch(FancyBboxPatch((0.03, ytop - ch), 0.94, ch,
                                    boxstyle="round,pad=0.006", fc=col, ec="none",
                                    alpha=0.16, zorder=1))
        ax.add_patch(FancyBboxPatch((0.03, ytop - ch), 0.028, ch,
                                    boxstyle="square,pad=0", fc=col, ec="none", zorder=2))
        rate = float(r["composite_rate"]) * 100
        smn = int(r["score_min"]) if "score_min" in r and pd.notna(r["score_min"]) else None
        smx = int(r["score_max"]) if "score_max" in r and pd.notna(r["score_max"]) else None
        nn = int(r["n"]) if "n" in r and pd.notna(r["n"]) else None
        rng = f"score {smn} to {smx}" if smn is not None else ""
        label = TIER_DISPLAY.get(t, str(t))
        if t == "escalate" and has_override:
            label += " *"
        ax.text(0.075, ytop - 0.02, label, fontsize=13, fontweight="bold",
                color=col, va="top")
        ax.text(0.965, ytop - 0.02, f"{rate:.1f}%", fontsize=17, fontweight="bold",
                color=col, va="top", ha="right")
        sub = rng + (f"   ·   N={nn:,}" if nn else "")
        # The override group OVERLAPS the score tiers -- it is a parallel
        # escalation path, not a fourth tier. 08 supplies the per-tier count so
        # the overlap is stated rather than left for the reader to assume.
        n_ov_tier = r.get("n_override_in_tier") if hasattr(r, "get") else None
        if n_ov_tier is not None and pd.notna(n_ov_tier) and float(n_ov_tier) > 0:
            sub += f"   ·   incl. {int(float(n_ov_tier)):,} auto-escalate"
        ax.text(0.075, ytop - ch * 0.45, sub, fontsize=9.5, color="#333333", va="center")
        # v3: the death component of the composite is `death_hemorrhagic` (D-1).
        drate = None
        for key in ("death_rate", "death_hemorrhagic_rate"):
            if key in r and pd.notna(r[key]):
                drate = float(r[key])
                break
        if drate is not None:
            trate = r.get("transfusion_rate", np.nan)
            ax.text(0.075, ytop - ch * 0.72,
                    f"hemorrhagic death {drate * 100:.1f}%  ·  "
                    f"substantial transfusion {float(trate) * 100:.1f}%",
                    fontsize=8.5, color="#555555", va="center")
        ax.text(0.075, ytop - ch * 0.93, action.get(t, ""), fontsize=8.5,
                color="#222222", style="italic", va="center")
    if has_override:
        ax.text(0.075, card_bot - 0.018,
                "* patients meeting the automatic-escalation criterion are escalated "
                "regardless of tier.", fontsize=7.8, color="#555555", style="italic",
                va="center")


def figure9_score_card():
    print("\n" + "=" * 64)
    print("Figure 9: Bedside score card (+ automatic-escalate override)")
    print("=" * 64)

    card = _read_json(os.path.join(DERIVED, "score_card.json"))
    pts = _read_csv(os.path.join(TABLES, "table3_points.csv"))
    tiers = _read_csv(os.path.join(DERIVED, "risk_tiers.csv"))
    if pts is None and card is None:
        print("  SKIP: neither table3_points.csv nor score_card.json available.")
        return

    # Any of the 17 LASSO candidates can survive onto this card, and this card
    # is the source for the blinded WTA .jpg -- so the label map is audited
    # before a single row is drawn (CONTRACTS sec.5 / sec.8 F9 / sec.13 G6).
    _audit_var_labels(context="F9 score card")

    ov_labels, ov_rule, theo_max_refit = _load_refit_labels()

    items, notes, source = _card_point_rows(pts, card, ov_labels)
    for msg in notes:
        print(f"  {msg}")
    if not items:
        print("  SKIP: no point-role rows could be assembled.")
        return
    print(f"  {len(items)} point rows from {source}: "
          f"{', '.join(v for v, _, _ in items)}")

    # Rows whose label had to be invented. Disclosed ON the figure, because the
    # console warning is invisible by the time this PNG becomes a submission
    # attachment. Empty in a healthy run.
    unlabelled_on_card = [v for v, _, _ in items if v in UNLABELLED_VARS]
    if unlabelled_on_card:
        print(f"  *** {len(unlabelled_on_card)} card row(s) carry a FALLBACK label, "
              f"not a clinical one: {unlabelled_on_card}. The figure is stamped "
              f"with a warning; add them to VAR_LABELS before using F9 anywhere. ***")
    print("  card row labels: " + " | ".join(lab for _, _, lab in items))

    override = _card_override_block(card, tiers, ov_labels, ov_rule)
    if override is None:
        print("  NOTE: no override descriptor found (score_card.json['override'] "
              "absent). Rendering the card WITHOUT the automatic-escalate banner "
              "— re-run --only 9 once script 08 has emitted it (CONTRACTS sec.5.1).")
    else:
        print(f"  override banner: {override['variable']} — "
              f"n={override.get('n')}, event_rate={override.get('event_rate')}")

    # ---- The three different "maxima" (gate G4c) -- never conflated ----------
    # (i)   OBSERVED test-set range  -> score_card["score_range"]  [the card footer]
    # (ii)  theoretical column sum   -> score_card["score_range_theoretical"]
    #                                   ["max_unconstrained"]
    # (iii) mutually-exclusive max   -> ...["max_mutually_exclusive"]  (the SBP and
    #                                   SI ladders are NESTED, not exclusive, so
    #                                   they both count -- 08 carries a
    #                                   `nesting_note` explaining the difference)
    smin = smax = None
    basis = None
    if isinstance(card, dict) and isinstance(card.get("score_range"), dict):
        smin = card["score_range"].get("min")
        smax = card["score_range"].get("max")
        basis = card["score_range"].get("basis")
    theo_pos = sum(p for _, p, _ in items if p > 0)     # local cross-check only
    theo_unc, theo_mutex, nesting_note = None, None, None
    theo_blk = card.get("score_range_theoretical") if isinstance(card, dict) else None
    if isinstance(theo_blk, dict):
        theo_unc = theo_blk.get("max_unconstrained")
        theo_mutex = theo_blk.get("max_mutually_exclusive")
        nesting_note = theo_blk.get("nesting_note")
    if isinstance(card, dict) and theo_mutex is None:
        for k in ("theoretical_max_mutually_exclusive", "theoretical_point_sum_mutex",
                  "theoretical_max_mutex"):
            if card.get(k) is not None:
                theo_mutex = card[k]
                break
    if theo_unc is None:
        theo_unc = theo_max_refit if theo_max_refit is not None else theo_pos
    if int(theo_unc) != int(theo_pos):
        print(f"  NOTE: upstream theoretical column sum (+{int(theo_unc)}) differs "
              f"from the sum of the rendered point rows (+{int(theo_pos)}).")
    print(f"  maxima — OBSERVED test-set max: {smax}"
          f"{f' (basis: {basis})' if basis else ''}; theoretical column sum: "
          f"+{int(theo_unc)}; mutually-exclusive max: "
          f"{('+' + str(int(theo_mutex))) if theo_mutex is not None else 'not supplied by 08'}")
    if nesting_note:
        print(f"  nesting note (08): {nesting_note}")

    n_rows = len(items)
    fig_h = max(9.2, 3.4 + 0.46 * n_rows)
    fig = plt.figure(figsize=(13.5, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.12,
                          left=0.02, right=0.99, top=0.955, bottom=0.02)
    axL = fig.add_subplot(gs[0, 0]); axL.axis("off")
    axR = fig.add_subplot(gs[0, 1]); axR.axis("off")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axR.set_xlim(0, 1); axR.set_ylim(0, 1)

    # ---------- LEFT: override banner (if any) then the point grid ----------
    axL.text(0.02, 0.985, "Pelvic Arrival Score — bedside points",
             fontsize=15, fontweight="bold", va="top")
    axL.text(0.02, 0.947, "Sum the points for all criteria present on arrival.",
             fontsize=9.5, va="top", color="#444444", style="italic")

    grid_top = 0.905
    if override is not None:
        # 5-line banner when 08 supplies the marginal flags/events, else 4.
        ban_top = 0.915
        ban_h = (0.138 if override.get("marginal_flagged_at_rule_in") is not None
                 else 0.118)
        _draw_override_banner(
            axL, override, ban_top, ban_h,
            n_test=(card.get("n_test") if isinstance(card, dict) else None),
            rule_in=(float(card.get("rule_in_cutpoint"))
                     if isinstance(card, dict) and card.get("rule_in_cutpoint") is not None
                     else None))
        grid_top = ban_top - ban_h - 0.038

    _draw_point_grid(axL, items, grid_top, 0.075)

    # ---- Footer: OBSERVED range first, theoretical sums clearly separated ----
    if smin is not None and smax is not None:
        axL.text(0.02, 0.042,
                 f"Observed score range in the held-out test set: "
                 f"{int(smin):+d} to {int(smax):+d}",
                 fontsize=9.8, va="center", color="#222222", fontweight="bold")
    theo_txt = (f"Theoretical sum of every positive point row: +{int(theo_unc)} "
                f"(never observed)")
    if theo_mutex is not None:
        theo_txt += (f"; highest attainable total once mutually exclusive criteria "
                     f"are respected: +{int(theo_mutex)}")
    axL.text(0.02, 0.014, theo_txt + ".", fontsize=7.8, va="center", color="#666666",
             style="italic")

    # ---------- RIGHT: risk tiers ----------
    _draw_risk_tiers(axR, tiers, card, has_override=(override is not None))

    # Operating-point footer (from score_card).
    if isinstance(card, dict):
        op = card.get("operating_points", {}).get("pelvic_score", {})
        ro = card.get("rule_out_cutpoint"); ri = card.get("rule_in_cutpoint")
        bits = []
        if isinstance(op.get("sens_90"), dict) and ro is not None:
            s = op["sens_90"]
            bits.append(f"Rule-out (score ≤{int(ro) - 1}): sens {s['sens'] * 100:.0f}%, "
                        f"NPV {s['npv'] * 100:.1f}%")
        if isinstance(op.get("spec_90"), dict) and ri is not None:
            s = op["spec_90"]
            tail = " or automatic escalation" if override is not None else ""
            bits.append(f"Rule-in (score ≥{int(ri)}{tail}): spec {s['spec'] * 100:.0f}%, "
                        f"PPV {s['ppv'] * 100:.0f}%")
        if bits:
            axR.text(0.03, 0.075, "   |   ".join(bits), fontsize=8.2, va="center",
                     color="#222222",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#F4F6F8", ec=OKABE_ITO["grey"]))

    fig.suptitle("Pelvic Hemorrhage Arrival Decision-Support Tool — bedside score card",
                 fontsize=15, fontweight="bold", y=0.992)
    caption = [_composite_definition_text()]
    if unlabelled_on_card:
        # Loud, on the artifact itself: a card row is showing a machine-derived
        # fallback rather than a clinical label. Deliberately impossible to miss
        # when the figure is reviewed for the WTA attachment.
        caption.append(
            "*** DRAFT — NOT FOR SUBMISSION: "
            f"{len(unlabelled_on_card)} criterion row(s) are rendered from a "
            f"variable name with no clinical label defined "
            f"({', '.join(unlabelled_on_card)}). Add them to "
            f"render_figures.VAR_LABELS and re-render before this figure is used. ***")
        axL.text(0.02, 0.999, "DRAFT — UNLABELLED CRITERION ON THIS CARD",
                 fontsize=9.5, fontweight="bold", color=C_OVERRIDE, va="bottom")
    _figure_footnote(fig, caption, y=-0.004, fontsize=8.0)
    _save_figure(fig, "figure9_score_card")


# =============================================================================
# FIGURE 10 (NEW, v3): MTP-equivalent activation -- ROC + decision curve
#
# Fix 7 / D-5 (PMID 25440492). Consumes the two fig-data artifacts
# written by 17_mtp_equivalent.py. 17 lands in a LATER batch than this renderer,
# so F10 is written to degrade rather than crash: a missing pickle skips panel A,
# a missing CSV skips panel B, both missing skips the figure entirely, and an
# unexpected key layout is normalized where it can be and reported where it
# cannot.
#
# CONTRACTS sec.4: `mtp_equivalent` is definitionally identical to
# `substantial_transfusion`. The identity statement is a required part of this
# figure, not optional caption garnish.
# =============================================================================
MTP_ROC_PLAN = [
    ("pelvic_score", "Pelvic arrival score", C_SCORE, 2.6),
    ("pelvic_logit", "Pelvic score (continuous logit)", C_FULL, 1.6),
    ("SI-alone", "Shock index alone", C_SI, 1.8),
    ("rSIG", "rSIG", C_RSIG, 1.8),
]


def _roc_xy(entry):
    """(fpr, tpr, auc, lo, hi) from a loosely-specified ROC entry, or None."""
    if not isinstance(entry, dict):
        return None
    fpr_k = next((k for k in ("fpr", "FPR", "x", "one_minus_spec") if k in entry), None)
    tpr_k = next((k for k in ("tpr", "TPR", "y", "sens") if k in entry), None)
    if fpr_k is None or tpr_k is None:
        return None
    try:
        fpr = np.asarray(entry[fpr_k], dtype=float)
        tpr = np.asarray(entry[tpr_k], dtype=float)
    except (TypeError, ValueError):
        return None
    if fpr.size == 0 or fpr.size != tpr.size:
        return None

    def _get(*keys):
        for k in keys:
            if entry.get(k) is not None:
                try:
                    return float(entry[k])
                except (TypeError, ValueError):
                    return None
        return None

    return (fpr, tpr, _get("auc", "auroc", "AUROC"),
            _get("lo", "ci_low", "lci", "auc_lo"),
            _get("hi", "ci_high", "uci", "auc_hi"))


def _coerce_roc_dict(obj, depth=0):
    """Normalize whatever 17 pickled into {curve_name: roc_entry_dict}."""
    if obj is None or depth > 2:
        return None
    if isinstance(obj, pd.DataFrame):                     # long-format fallback
        cols = {c.lower(): c for c in obj.columns}
        if "fpr" in cols and "tpr" in cols:
            key = next((cols[c] for c in ("score", "model", "curve", "name")
                        if c in cols), None)
            if key is None:
                return {"pelvic_score": {"fpr": obj[cols["fpr"]].to_numpy(float),
                                         "tpr": obj[cols["tpr"]].to_numpy(float)}}
            return {str(nm): {"fpr": sub[cols["fpr"]].to_numpy(float),
                              "tpr": sub[cols["tpr"]].to_numpy(float)}
                    for nm, sub in obj.groupby(key)}
        return None
    if not isinstance(obj, dict):
        return None
    if _roc_xy(obj) is not None:                          # a bare single curve
        return {"pelvic_score": obj}
    # Keys beginning with "_" are metadata blocks (17 writes `_meta`), never curves.
    curves = {k: v for k, v in obj.items()
              if not str(k).startswith("_") and _roc_xy(v) is not None}
    if curves:
        return curves
    for k in ("roc", "curves", "roc_data", "data", "mtp_equivalent"):
        if isinstance(obj.get(k), (dict, pd.DataFrame)):
            got = _coerce_roc_dict(obj[k], depth + 1)
            if got:
                return got
    return None


def _normalize_dca(dca):
    """Make an arbitrary MTP DCA frame satisfy _dca_panel's column contract."""
    if dca is None:
        return None
    d = dca.copy()
    ren = {}
    for want, cands in (("threshold", ("threshold", "p_t", "pt", "threshold_probability")),
                        ("net_benefit", ("net_benefit", "nb", "netbenefit")),
                        ("score", ("score", "model", "strategy", "curve")),
                        ("treat_all_nb", ("treat_all_nb", "treat_all", "nb_all"))):
        col = _flex_col(d, list(cands))
        if col is not None and col != want:
            ren[col] = want
    d = d.rename(columns=ren)
    if not {"threshold", "net_benefit"}.issubset(d.columns):
        print(f"  NOTE: mtp_dca_data.csv lacks threshold/net_benefit columns "
              f"(have {list(dca.columns)}); DCA panel skipped.")
        return None
    if "score" not in d.columns:
        print("  NOTE: mtp_dca_data.csv has no `score` column; treating the single "
              "net-benefit series as the pelvic score.")
        d["score"] = "pelvic_score"
    return d


def figure10_mtp_equivalent():
    print("\n" + "=" * 64)
    print("Figure 10: MTP-equivalent activation — ROC + decision curve")
    print("=" * 64)

    roc_raw = _read_pickle(os.path.join(DERIVED, "mtp_roc_data.pkl"))
    dca_raw = _read_csv(os.path.join(DERIVED, "mtp_dca_data.csv"))
    if roc_raw is None and dca_raw is None:
        print("  SKIP: neither mtp_roc_data.pkl nor mtp_dca_data.csv found "
              "(script 17 not run yet) — re-run `--only 10` after 17 completes.")
        return

    roc = _coerce_roc_dict(roc_raw)
    if roc_raw is not None and roc is None:
        print(f"  NOTE: mtp_roc_data.pkl present but no fpr/tpr curve could be read "
              f"(top-level type {type(roc_raw).__name__}); ROC panel skipped.")
    dca = _normalize_dca(dca_raw)

    n_panels = int(roc is not None) + int(dca is not None)
    if n_panels == 0:
        print("  SKIP: no usable MTP panel.")
        return

    fig, axarr = plt.subplots(1, n_panels, figsize=(7.7 * n_panels, 6.6), squeeze=False)
    axes = list(axarr[0])
    letters = iter(["A", "B"])

    # ----- Panel A: ROC against MTP-equivalent activation -----
    if roc is not None:
        ax = axes.pop(0)
        ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], linestyle=":", linewidth=1.2,
                zorder=1)
        planned = [k for k, _, _, _ in MTP_ROC_PLAN]
        extras = [k for k in roc if k not in planned]
        if extras:
            print(f"  NOTE: unplanned ROC curve(s) in mtp_roc_data.pkl: {extras} "
                  f"— drawn in grey.")
        plan = list(MTP_ROC_PLAN) + [(k, str(k), OKABE_ITO["grey"], 1.4) for k in extras]
        handles, drew = [], 0
        for key, label, col, lw in plan:
            cur = _roc_xy(roc.get(key))
            if cur is None:
                continue
            fpr, tpr, auc, lo, hi = cur
            ax.plot(fpr, tpr, color=col, linewidth=lw, zorder=3, solid_capstyle="round")
            leg = label
            if auc is not None:
                leg += f": AUROC {auc:.3f}"
                if lo is not None and hi is not None:
                    leg += f" ({lo:.3f}–{hi:.3f})"
            handles.append(plt.Line2D([0], [0], color=col, linewidth=lw, label=leg))
            drew += 1
        if drew == 0:
            ax.text(0.5, 0.5, "MTP ROC data unusable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
        else:
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_xlabel("1 − specificity (false-positive rate)")
            ax.set_ylabel("Sensitivity (true-positive rate)")
            ax.set_title("Discrimination for MTP-equivalent activation", fontsize=12)
            ax.legend(handles=handles, loc="lower right", framealpha=0.94, fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, linestyle="--")
            ax.text(-0.08, 1.05, next(letters), transform=ax.transAxes, fontsize=16,
                    fontweight="bold", va="top")

    # ----- Panel B: decision curve against MTP-equivalent activation -----
    if dca is not None:
        ax = axes.pop(0)
        has_rsig = "rSIG" in set(dca["score"].astype(str))
        ok = _dca_panel(ax, dca, "Decision-curve analysis: MTP-equivalent activation",
                        next(letters), include_rsig=has_rsig)
        if not ok:
            ax.text(0.5, 0.5, "MTP DCA data unusable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout(w_pad=3.0)
    mtp_def = OUTCOME_PROSE["mtp_equivalent"]
    _figure_footnote(
        fig,
        [f"Secondary outcome — {mtp_def}, using the unit-aware transfusion decoder.",
         MTP_IDENTITY_NOTE],
        y=-0.02, fontsize=8.0)
    _save_figure(fig, "figure10_mtp_equivalent")


# =============================================================================
# MAIN
# =============================================================================
FIGURE_MAP = {
    "2": ("Figure 2: Score distribution", figure2_score_distribution),
    "3": ("Figure 3: ROC overlay", figure3_roc),
    "4": ("Figure 4: Calibration", figure4_calibration),
    "5": ("Figure 5: DCA (centerpiece)", figure5_dca),
    "6": ("Figure 6: Subgroup forest", figure6_subgroup_forest),
    "7": ("Figure 7: Threshold splines", figure7_threshold_splines),
    "8": ("Figure 8: Component decomposition", figure8_component),
    "9": ("Figure 9: Score card", figure9_score_card),
    "10": ("Figure 10: MTP-equivalent ROC + DCA", figure10_mtp_equivalent),
}
ALL_KEYS = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]


def main():
    parser = argparse.ArgumentParser(
        description="Pelvic Arrival Tool v3 -- publication figure renderer (F2-F10).")
    parser.add_argument("--only", type=str, default=None, choices=ALL_KEYS,
                        help="Render only one figure (2..10).")
    args = parser.parse_args()

    print("=" * 64)
    print("Pelvic Arrival Tool v3 -- figure renderer (F2-F10)")
    print(f"  figures dir: {FIGURES}")
    print("=" * 64)

    # Label completeness is checked ONCE up front, before anything renders, so a
    # gap is reported even for a run that skips F9 for a missing artifact.
    label_gaps = _audit_var_labels(context="startup")

    keys = [args.only] if args.only else ALL_KEYS
    ran = 0
    for key in keys:
        name, func = FIGURE_MAP[key]
        try:
            func()
            ran += 1
        except Exception as exc:  # noqa: BLE001 - one bad figure never kills the rest
            print(f"  ERROR generating {name}: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 64)
    print(f"Figure rendering complete ({ran}/{len(keys)} figure functions ran).")
    if label_gaps or UNLABELLED_VARS:
        # Last line of the run log, so it is the thing still on screen. A figure
        # showing a machine-derived label must not reach a manuscript or the
        # blinded WTA attachment (CONTRACTS sec.8 / sec.13 G6).
        gaps = sorted(set(label_gaps) | set(UNLABELLED_VARS))
        print(f"*** LABEL WARNING: {len(gaps)} variable(s) rendered or renderable "
              f"without a clinical label: {gaps}. Add them to VAR_LABELS "
              f"(reuse 15_table1.py's wording) and re-render. ***")
        print(f"*** LABEL WARNING: {gaps} ***", file=sys.stderr)
    print("=" * 64)


if __name__ == "__main__":
    main()
