"""
14_consort.py — STROBE cohort-assembly flow for the Pelvic Hemorrhage Arrival
Decision-Support Tool (v3, NTDB/TQP 2019-2024).

Fork of PAT_TBI/v4/code/13_consort.py, re-specified for the pelvic pre-decision
funnel (the TBI-specific step labels / 3-box collapse are replaced with a linear
7-step STROBE cascade). Reads the ordered exclusion log written by
01_build_cohort.py and the split-balance table written by 03_split.py, then
renders:

  v3/tables/table_consort.csv            — Step / N before / N after / N excluded / %
  v3/figures/figure1_STROBE.png          — raster STROBE diagram (300 dpi)
  v3/figures/figure1_STROBE.pdf          — vector STROBE diagram (TrueType-embedded)

This script OWNS Figure 1 rendering (CONTRACTS §8: F1 is rendered by 14, not by
render_figures.py). House style follows FORD/JTACS (Arial, grayscale, thin 0.6pt
borders). All counts are DERIVED from the JSON + split table — nothing hardcoded.

v3 changes (CONTRACTS §3 note / §4; SAP amendment D-1, D-2, D-8)
---------------------------------------------------------------
  * THE FUNNEL IS UNCHANGED. v3 touches outcome definitions only; the seven
    pre-decision STROBE steps, the analytic N (239,736) and the CSV row count
    (291,766) are hard regression gates, not results (CONTRACTS §3, G2f). No
    exclusion step is added, removed, reordered or re-worded here.
  * The OUTCOME ANNOTATIONS change. `major_hemorrhage_event` keeps its name but
    changes meaning, so the analytic box now names it as the primary composite
    and the figure carries an explicit definition footnote (answers reviewer
    comment D5, "is this a composite outcome?"). The definition is READ from
    logs/qc_manifest.json where available and only falls back to a literal.
  * Derivation / validation event counts are now taken EXACTLY from
    table_split_balance.csv instead of being back-estimated from the pooled
    event rate.

Inputs (READ-ONLY)
------------------
  v3/derived/cohort_exclusions.json
    An ordered dict {step_description: N_remaining_after_that_step}. The 7-step
    pre-decision funnel:
        7,327,714 → adult → pelvic-ring S32 → blunt → direct-arrival
                  → not-DOA/AIS-6 → valid-outcome = 239,736.
    Per-step *excluded* counts are derived as N_{i-1} - N_i.
  v3/tables/table_split_balance.csv
    Rows keyed by split in {full, train, test}; supplies derivation N (= train),
    validation N (= test), and the per-split event counts / rates.
  v3/logs/qc_manifest.json   (OPTIONAL)
    `primary_composite.definition` — the composite's component expression, used
    to render the figure's outcome-definition footnote in clinical prose. If the
    manifest is absent the footnote falls back to COMPOSITE_COMPONENT_PROSE.

Seed: 42 (convention; not used here). Run from any cwd.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import (repo convention)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# §1 Canonical paths — v3 only
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()
DERIVED = Path(config.DERIVED)
TABLES = Path(config.TABLES)
FIGURES = Path(config.FIGURES)
LOGS = Path(config.LOGS)

EXCLUSIONS_JSON = DERIVED / "cohort_exclusions.json"
SPLIT_CSV = TABLES / "table_split_balance.csv"
QC_MANIFEST = LOGS / "qc_manifest.json"
OUT_CSV = TABLES / "table_consort.csv"
OUT_PNG = FIGURES / "figure1_STROBE.png"
OUT_PDF = FIGURES / "figure1_STROBE.pdf"

SEED = 42  # noqa: F841

# ---------------------------------------------------------------------------
# v3 primary outcome (CONTRACTS §4). `major_hemorrhage_event` keeps its v2 NAME
# and changes its DEFINITION, so the figure must state the definition rather
# than rely on the name. Prose is keyed on the component column names emitted by
# 01_build_cohort.py so the footnote tracks the cohort build instead of a
# hand-maintained sentence.
# ---------------------------------------------------------------------------
PRIMARY_OUTCOME_COL = "major_hemorrhage_event"
PRIMARY_OUTCOME_NAME = "major hemorrhage event"

COMPOSITE_COMPONENT_PROSE: dict[str, str] = {
    "received_hc": "a hemorrhage-control procedure (surgical or angiographic)",
    "death_hemorrhagic": (
        "hemorrhagic death (in-hospital death with head AIS < 4, within 48 h of arrival)"
    ),
    "substantial_transfusion": (
        "substantial transfusion (≥ 1000 mL or ≥ 4 units of packed red cells within 4 h)"
    ),
    # Retained so a sensitivity-arm definition never renders as a raw identifier.
    "death_inhosp": "in-hospital death from any cause",
    "death_hemorrhagic_headais": "in-hospital death with head AIS < 4",
    "death_le48h": "in-hospital death within 48 h of arrival",
    "substantial_transfusion_ml_only": "substantial transfusion (≥ 1000 mL within 4 h)",
}

# Fallback used only if logs/qc_manifest.json is unavailable (CONTRACTS §4).
FALLBACK_COMPOSITE_COMPONENTS = [
    "received_hc", "death_hemorrhagic", "substantial_transfusion",
]

# Pre-decision funnel note (CONTRACTS §9-A): hemorrhage control is an OUTCOME
# component, so it is deliberately NOT an eligibility criterion.
PREDECISION_NOTE = (
    "Hemorrhage control is a component of the outcome, not an eligibility criterion: "
    "the cascade above is the pre-decision cohort, so no patient is excluded for "
    "failing to receive an intervention."
)

# ---------------------------------------------------------------------------
# Map Step-1 JSON keys -> (clinical-English step label, exclusion-reason text).
#
# The JSON keys for this project are already human-readable prose (unlike the
# PAT_TBI snake_case keys), so this map is keyed on the exact strings emitted by
# 01_build_cohort.py. Any key NOT recognized here still renders via a humanized
# fallback so the table never silently drops a step. Order is preserved.
#
# These are the PELVIC pre-decision exclusions (respecified from the TBI source):
#   entry → adult → pelvic-ring fracture → blunt → direct arrival
#         → not DOA/AIS-6 → valid outcome.
# ---------------------------------------------------------------------------
STEP_LABEL_MAP: dict[str, tuple[str, str]] = {
    "All NTDB/TQP 2019-2024 encounters": (
        "All NTDB/TQP encounters, 2019-2024",
        "(entry)",
    ),
    "Adult (AGE_NUMBER >= 18 & AGE_UNITS == 'Y')": (
        "Adult patients (age >= 18 years)",
        "Age < 18 years, or age not recorded in years",
    ),
    "Pelvic-ring fracture (>=1 ICD-10 S32.1/.3/.5/.6/.81/.82)": (
        "Pelvic-ring fracture (ICD-10 S32.1/.3/.5/.6/.81/.82)",
        "No qualifying pelvic-ring fracture diagnosis",
    ),
    "Blunt mechanism (non-blank e-code; penetrating excluded)": (
        "Blunt mechanism of injury",
        "Penetrating or unspecified (blank) mechanism e-code",
    ),
    "Direct arrival (HOSPITAL_TRANSFER == 'N')": (
        "Direct arrival (non-transfer)",
        "Transferred in from another facility (outcome not attributable)",
    ),
    "Exclude DOA / unsurvivable (PREHOSPITAL_ARREST == 'Y' or any AIS severity 6)": (
        "Excluding dead-on-arrival / unsurvivable injury",
        "Prehospital cardiac arrest, or any AIS severity 6 injury",
    ),
    "Valid outcome (disposition known or ED death)": (
        "Analytic cohort (valid outcome ascertainable)",
        "Unknown ED/hospital disposition (outcome not ascertainable)",
    ),
}

# ---------------------------------------------------------------------------
# Figure text (short, figure-ready). Indexed by JSON step order (7 main boxes,
# 6 transitions). Counts are injected from the JSON at render time.
# ---------------------------------------------------------------------------
FIGURE_STEP_LINES: list[list[str]] = [
    ["All NTDB/TQP encounters", "2019-2024"],
    ["Adult patients", "(age ≥ 18 years)"],
    ["Pelvic-ring fracture", "(ICD-10 S32.1/.3/.5/.6/.81/.82)"],
    ["Blunt mechanism of injury"],
    ["Direct arrival (non-transfer)"],
    ["Survivable, observable injury", "(not DOA; no AIS-6 injury)"],
    ["Analytic cohort", "(valid outcome ascertainable)"],
]

FIGURE_EXCL_LABELS: list[str] = [
    "Age < 18 y or age not in years",
    "No pelvic-ring fracture diagnosis",
    "Penetrating / unspecified mechanism",
    "Transferred in from another facility",
    "Prehospital arrest or any AIS-6 injury",
    "Unknown disposition (outcome not ascertainable)",
]


def _humanize_unknown(key: str) -> tuple[str, str]:
    """Fallback label for an unrecognized JSON step key."""
    pretty = key.replace("_", " ").strip()
    return (f"After step: {pretty}", pretty)


# ---------------------------------------------------------------------------
# Load and normalize
# ---------------------------------------------------------------------------
def load_exclusions(path: Path) -> list[tuple[str, int]]:
    """Return [(step_name, N_remaining), ...] in JSON insertion order."""
    if not path.exists():
        sys.exit(f"FATAL: exclusion log not found at {path}")
    with open(path) as f:
        log = json.load(f)
    if not isinstance(log, dict) or not log:
        sys.exit(f"FATAL: exclusion log at {path} is empty or not a dict")
    items = [(k, int(v)) for k, v in log.items()]
    return items


def load_split_balance(path: Path) -> dict:
    """Read table_split_balance.csv → derivation/validation Ns + event counts.

    Rows are keyed by `split` in {full, train, test}. Derivation = train (2/3),
    validation = test (1/3). Fails loud if the table or a required row is absent.
    Per-split event counts are read EXACTLY (v3) rather than back-estimated from
    the pooled rate.
    """
    if not path.exists():
        sys.exit(f"FATAL: split-balance table not found at {path}")
    sb = pd.read_csv(path).set_index("split")
    for need in ("full", "train", "test"):
        if need not in sb.index:
            sys.exit(f"FATAL: split-balance table {path} missing row '{need}'")
    return {
        "derivation_n": int(sb.loc["train", "N"]),
        "validation_n": int(sb.loc["test", "N"]),
        "derivation_events": int(sb.loc["train", "N_events"]),
        "validation_events": int(sb.loc["test", "N_events"]),
        "derivation_rate": float(sb.loc["train", "event_rate"]),
        "validation_rate": float(sb.loc["test", "event_rate"]),
        "full_events": int(sb.loc["full", "N_events"]),
        "full_rate": float(sb.loc["full", "event_rate"]),
    }


def composite_components() -> list[str]:
    """Component column names of the v3 primary composite.

    Preference order: the `primary_composite.definition` expression written by
    01_build_cohort.py into logs/qc_manifest.json (so the figure footnote tracks
    the cohort build), then the CONTRACTS §4 fallback list. Never raises — a
    missing manifest degrades to the fallback with a warning.
    """
    if QC_MANIFEST.exists():
        try:
            with open(QC_MANIFEST) as f:
                man = json.load(f)
            expr = (man.get("primary_composite") or {}).get("definition", "")
            toks = [t.strip() for t in str(expr).split("|") if t.strip()]
            toks = [t for t in toks if t in COMPOSITE_COMPONENT_PROSE]
            if toks:
                return toks
            print(f"  WARNING: could not parse a composite definition from "
                  f"{QC_MANIFEST.name} (got {expr!r}); using the CONTRACTS §4 "
                  f"fallback.", file=sys.stderr)
        except (OSError, ValueError) as exc:
            print(f"  WARNING: {QC_MANIFEST.name} unreadable ({exc}); using the "
                  f"CONTRACTS §4 fallback composite definition.", file=sys.stderr)
    else:
        print(f"  WARNING: {QC_MANIFEST} not found; using the CONTRACTS §4 "
              f"fallback composite definition.", file=sys.stderr)
    return list(FALLBACK_COMPOSITE_COMPONENTS)


def composite_definition_text(components: list[str]) -> str:
    """Render the composite definition as one clinical-prose sentence."""
    parts = [COMPOSITE_COMPONENT_PROSE.get(c, c.replace("_", " ")) for c in components]
    if len(parts) > 1:
        body = ", ".join(parts[:-1]) + ", or " + parts[-1]
    else:
        body = parts[0] if parts else "(undefined)"
    return (f"Primary outcome — {PRIMARY_OUTCOME_NAME} (composite): "
            f"{body}. Any one component is sufficient.")


def build_figure_counts(items: list[tuple[str, int]], split: dict) -> dict:
    """Collapse the ordered JSON log + split table into the cascade figure model.

    Derives (never hardcodes) the per-transition exclusion counts as successive
    differences of the remaining-N series.
    """
    counts = [n for _, n in items]
    excl = [counts[i - 1] - counts[i] for i in range(1, len(counts))]
    return {
        "step_counts": counts,           # length n
        "excl_counts": excl,             # length n-1
        "analytic": counts[-1],
        "derivation": split["derivation_n"],
        "validation": split["validation_n"],
        "derivation_events": split["derivation_events"],
        "validation_events": split["validation_events"],
        "derivation_rate": split["derivation_rate"],
        "validation_rate": split["validation_rate"],
        "events": split["full_events"],
        "rate": split["full_rate"],
        "composite_definition": split.get("composite_definition", ""),
    }


def to_consort_rows(items: list[tuple[str, int]]) -> pd.DataFrame:
    """Translate the JSON step list into CONSORT-style rows (granular, all steps).

    Schema:
      Step             : 1-indexed step number
      Step_Label       : clinical-English description of the cohort at this step
      N_before         : N at the prior step (entry row blank)
      N_after          : N remaining after this step
      N_excluded       : N_before - N_after (entry row blank)
      Pct_excluded     : 100 * N_excluded / N_before (entry row blank)
      Exclusion_reason : prose reason for the exclusion (entry row "(entry)")
    """
    rows = []
    prev_n = None
    for i, (key, n_after) in enumerate(items, start=1):
        label, reason = STEP_LABEL_MAP.get(key, _humanize_unknown(key))
        if prev_n is None:
            n_excl: int | str = ""
            pct: str = ""
            n_before: int | str = ""
        else:
            n_excl_int = max(0, prev_n - n_after)
            n_excl = n_excl_int
            pct = f"{(100.0 * n_excl_int / prev_n):.2f}" if prev_n > 0 else ""
            n_before = prev_n
        rows.append({
            "Step": i,
            "Step_Label": label,
            "N_before": n_before,
            "N_after": n_after,
            "N_excluded": n_excl,
            "Pct_excluded": pct,
            "Exclusion_reason": reason,
        })
        prev_n = n_after
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure — linear 7-step STROBE cascade (respecified from PAT's 3-box collapse)
# ---------------------------------------------------------------------------
def draw_consort_figure(c: dict) -> None:
    """Render the pelvic STROBE cohort-flow diagram (FORD/JTACS house style).

    A vertical trunk of main cohort boxes (one per JSON step) flows top-to-bottom;
    each transition taps a dashed arrow to an italic exclusion side box; the final
    analytic box carries the primary-event annotation and a T-bar split into
    derivation / held-out validation child boxes. `c` = build_figure_counts().
    """
    step_counts = c["step_counts"]
    excl_counts = c["excl_counts"]
    n = len(step_counts)

    if len(FIGURE_STEP_LINES) != n:
        sys.exit(f"FATAL: FIGURE_STEP_LINES has {len(FIGURE_STEP_LINES)} entries; "
                 f"JSON funnel has {n} steps. Re-sync the figure labels.")
    if len(FIGURE_EXCL_LABELS) != n - 1:
        sys.exit(f"FATAL: FIGURE_EXCL_LABELS has {len(FIGURE_EXCL_LABELS)} entries; "
                 f"expected {n - 1} transitions.")

    # --- JAMA / FORD house style ------------------------------------------
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,   # TrueType (embeddable)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    FIG_W, FIG_H = 7.2, 10.4
    BOX_FC, BOX_EC, BOX_LW = "white", "black", 0.6
    TEXT_C = "black"
    ARROW_LW = 0.8
    ARROW_KW = dict(arrowstyle="-|>", mutation_scale=8, color="black",
                    linewidth=ARROW_LW, shrinkA=0, shrinkB=0)
    EXCL_KW = dict(ARROW_KW)
    EXCL_KW["linestyle"] = (0, (2.5, 2.0))  # dashed for exclusion lines

    # y floor is negative so the outcome-definition footnote has its own band
    # below the split boxes; bbox_inches="tight" trims any unused whitespace.
    Y_FLOOR = -13.0

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(Y_FLOOR, 100)
    ax.set_aspect("auto")
    ax.set_axis_off()

    # --- Geometry (0-100 x 0-100) -----------------------------------------
    MAIN_X, MAIN_W = 31.0, 50.0
    BOX_H, ANALYTIC_H = 8.0, 11.0
    EXCL_X, EXCL_W, EXCL_H = 80.0, 39.0, 8.5

    Y_TOP, Y_BOT_MAIN = 96.0, 30.0
    ys = list(np.linspace(Y_TOP, Y_BOT_MAIN, n))  # main-box centers, top→bottom

    # --- Drawing helpers ---------------------------------------------------
    def draw_box(cx, cy, w, h, lines, *, italic=False, fontsize=8.0):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.30,rounding_size=0.50",
            linewidth=BOX_LW, edgecolor=BOX_EC, facecolor=BOX_FC, clip_on=False,
        ))
        style = "italic" if italic else "normal"
        m = len(lines)
        step = (h - 1.6) / max(m, 1) if m > 1 else 0
        top_y = cy + (step * (m - 1)) / 2
        for i, line in enumerate(lines):
            weight = "bold" if (i == 0 and not italic) else "normal"
            ax.text(cx, top_y - i * step, line, ha="center", va="center",
                    fontsize=fontsize, color=TEXT_C,
                    fontstyle=style, fontweight=weight)

    def varrow(x, y_from, y_to):
        ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), **ARROW_KW))

    def excl_arrow(y_tap, x_box_left):
        x_stub = MAIN_X + MAIN_W / 2 - 0.5  # just inside the trunk's right edge
        ax.plot([MAIN_X, x_stub], [y_tap, y_tap], color="black",
                linewidth=ARROW_LW, linestyle=(0, (2.5, 2.0)),
                solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch((x_stub, y_tap), (x_box_left, y_tap), **EXCL_KW))

    # --- Main column boxes + transition arrows + exclusion taps ------------
    for i in range(n):
        cy = ys[i]
        is_last = (i == n - 1)
        h = ANALYTIC_H if is_last else BOX_H
        lines = list(FIGURE_STEP_LINES[i]) + [f"N = {step_counts[i]:,}"]
        if is_last:
            # analytic box: label lines + N + italic primary-event line
            ax.add_patch(FancyBboxPatch(
                (MAIN_X - MAIN_W / 2, cy - h / 2), MAIN_W, h,
                boxstyle="round,pad=0.30,rounding_size=0.50",
                linewidth=BOX_LW, edgecolor=BOX_EC, facecolor=BOX_FC, clip_on=False,
            ))
            ax.text(MAIN_X, cy + 3.4, FIGURE_STEP_LINES[i][0],
                    ha="center", va="center", fontsize=8.0, fontweight="bold", color=TEXT_C)
            ax.text(MAIN_X, cy + 0.9, FIGURE_STEP_LINES[i][1],
                    ha="center", va="center", fontsize=8.0, color=TEXT_C)
            ax.text(MAIN_X, cy - 1.5, f"N = {step_counts[i]:,}",
                    ha="center", va="center", fontsize=8.0, color=TEXT_C)
            # v3: name the outcome as the composite it now is (comment D5); the
            # definition itself is spelled out in the footnote below the figure.
            ax.text(MAIN_X, cy - 3.9,
                    f"Major hemorrhage event (composite): "
                    f"n = {c['events']:,} ({100 * c['rate']:.1f}%)",
                    ha="center", va="center", fontsize=6.9, color=TEXT_C,
                    fontstyle="italic")
        else:
            draw_box(MAIN_X, cy, MAIN_W, h, lines)

        # transition arrow to the next box + exclusion side box
        if i < n - 1:
            y_bottom_this = cy - h / 2
            next_h = ANALYTIC_H if (i + 1 == n - 1) else BOX_H
            y_top_next = ys[i + 1] + next_h / 2
            varrow(MAIN_X, y_bottom_this, y_top_next)
            y_tap = (y_bottom_this + y_top_next) / 2
            excl_arrow(y_tap, EXCL_X - EXCL_W / 2)
            draw_box(EXCL_X, y_tap, EXCL_W, EXCL_H, [
                f"Excluded (n = {excl_counts[i]:,}):",
                FIGURE_EXCL_LABELS[i],
            ], italic=True, fontsize=7.3)

    # --- Derivation / validation split (stub → T-bar → two arrows) --------
    y_analytic = ys[-1]
    SPLIT_W, SPLIT_H = 26.0, 9.5
    X_DERIV, X_VALID = MAIN_X - 16.0, MAIN_X + 16.0
    Y_SPLIT = 9.0
    Y_T_BAR = Y_SPLIT + SPLIT_H / 2 + 6.0
    ax.plot([MAIN_X, MAIN_X], [y_analytic - ANALYTIC_H / 2, Y_T_BAR],
            color="black", linewidth=ARROW_LW, solid_capstyle="butt")
    ax.plot([X_DERIV, X_VALID], [Y_T_BAR, Y_T_BAR],
            color="black", linewidth=ARROW_LW, solid_capstyle="butt")
    varrow(X_DERIV, Y_T_BAR, Y_SPLIT + SPLIT_H / 2)
    varrow(X_VALID, Y_T_BAR, Y_SPLIT + SPLIT_H / 2)

    # Event counts per split are read exactly from table_split_balance.csv (v3);
    # they are no longer back-estimated from the pooled event rate.
    draw_box(X_DERIV, Y_SPLIT, SPLIT_W, SPLIT_H, [
        "Derivation cohort",
        "(2/3, stratified)",
        f"N = {c['derivation']:,}",
        f"events n = {c['derivation_events']:,} ({100 * c['derivation_rate']:.1f}%)",
    ], fontsize=7.6)
    draw_box(X_VALID, Y_SPLIT, SPLIT_W, SPLIT_H, [
        "Held-out validation",
        "(1/3, stratified)",
        f"N = {c['validation']:,}",
        f"events n = {c['validation_events']:,} ({100 * c['validation_rate']:.1f}%)",
    ], fontsize=7.6)

    # --- Outcome-definition footnote (answers reviewer comment D5) ---------
    foot_y = Y_SPLIT - SPLIT_H / 2 - 3.2
    for para, style in ((c.get("composite_definition", ""), "italic"),
                        (PREDECISION_NOTE, "normal")):
        if not para:
            continue
        for line in textwrap.wrap(para, width=118):
            ax.text(2.0, foot_y, line, ha="left", va="top", fontsize=6.9,
                    color=TEXT_C, fontstyle=style)
            foot_y -= 2.1
        foot_y -= 0.9

    # --- Sanity: arithmetic must reconcile or fail loud -------------------
    for i in range(1, n):
        assert step_counts[i - 1] - excl_counts[i - 1] == step_counts[i], \
            f"cascade mismatch at step {i}"
    assert c["derivation"] + c["validation"] == c["analytic"], \
        "split does not sum to analytic N"
    assert c["derivation_events"] + c["validation_events"] == c["events"], \
        ("split event counts do not sum to the full-cohort event count "
         f"({c['derivation_events']:,} + {c['validation_events']:,} != {c['events']:,})")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pelvic STROBE consort flow (F1).")
    # --sample accepted for pipeline/smoke-harness parity (CONTRACTS §9-E). This
    # script consumes only pre-aggregated counts, so it is a documented no-op.
    parser.add_argument("--sample", type=int, default=None,
                        help="(no-op) accepted for harness parity; consort uses "
                             "pre-aggregated counts, not row-level data.")
    args = parser.parse_args()
    if args.sample is not None:
        print(f"[14_consort] --sample {args.sample} is a no-op "
              f"(consort reads pre-aggregated counts).")

    print(f"loading exclusion log: {EXCLUSIONS_JSON}")
    items = load_exclusions(EXCLUSIONS_JSON)
    print(f"  {len(items)} steps recorded")
    for k, nn in items:
        print(f"    {k}: N = {nn:,}")

    df = to_consort_rows(items)

    # Sanity: final N is the pre-registered pelvic analytic cohort (~239,736).
    final_n = int(df["N_after"].iloc[-1])
    if not (200_000 <= final_n <= 300_000):
        print(
            f"  WARNING: final cohort N = {final_n:,} is outside the expected "
            f"~239,736 pelvic-ring analytic range; verify Step 1 inputs.",
            file=sys.stderr,
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows; final N = {final_n:,})")

    print(f"loading split balance: {SPLIT_CSV}")
    split = load_split_balance(SPLIT_CSV)
    print(f"  derivation N = {split['derivation_n']:,} "
          f"(events {split['derivation_events']:,}); "
          f"validation N = {split['validation_n']:,} "
          f"(events {split['validation_events']:,}); "
          f"full events = {split['full_events']:,} "
          f"({100 * split['full_rate']:.2f}%)")

    # v3 primary composite — read the component list (not a hard-coded sentence)
    # so the F1 footnote always matches the outcome the cohort build emitted.
    comps = composite_components()
    split["composite_definition"] = composite_definition_text(comps)
    print(f"  primary outcome components: {' | '.join(comps)}")
    print(f"  F1 footnote: {split['composite_definition']}")

    counts = build_figure_counts(items, split)
    draw_consort_figure(counts)


if __name__ == "__main__":
    main()
