#!/usr/bin/env python3
"""
render_exhibits.py -- v4 exhibit renderer (plan section B4) for the Pelvic
Hemorrhage Arrival Decision-Support Tool (NTDB/TQP 2019-2024).

Emits the three main-text figures and the three main-text tables:

  figures/figure_1.png / .tif      participant flow, 7 rungs (7,327,714 -> 239,736)
  figures/figure_2.png / .tif      A: ROC (test + temporal)   B: calibration (overall + elderly)
  figures/figure_3.png / .tif      A: decision curve          B: matched-workload contrast
  figures/figure_legends.md        one publication-voice legend per figure
  tables/Table 1.csv / .md         baseline characteristics
  tables/Table 2.csv / .md         the arrival score card
  tables/Table 3.csv / .md         discrimination + matched-workload utility

House style (CONTRACTS section 1): self-contained, module-level absolute roots,
no argparse path arguments, COPY-ADAPTED helpers -- this file imports nothing
from any other pipeline script, so v3/code/render_figures.py stays byte-stable.
The Okabe-Ito palette, the publication rcParams block, the label-discipline
helpers (`_sanitize_label` / `_var_label` / VAR_LABELS / RETIRED_LABEL_PATTERNS)
and `_save_figure` are copy-adapted from v3/code/render_figures.py; the flow
ladder is copy-adapted from v3/code/14_consort.py. `_save_figure` here carries
the added TIF writer (300 dpi, LZW) that the plan describes as a three-line
patch -- applied to THIS copy, never to the v3 original.

sklearn and statsmodels are NEVER imported: every number here is read from a
finished v3 artifact, and both packages are pathologically slow in this
environment.

Verification discipline
-----------------------
Every gate prints an explicit `GATE <id>: PASS|FAIL` line. A gate that is
declared but never evaluated prints `GATE <id>: NOT EVALUATED -- FAIL` in the
summary and the process exits non-zero. There is no `except: pass` around any
assertion anywhere in this file.

Usage:
  python3 render_exhibits.py
"""

from __future__ import annotations

import os
import re
import sys
import json
import pickle
import textwrap
import traceback

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # MUST precede pyplot import (CONTRACTS section 1)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from PIL import Image  # noqa: E402



# =============================================================================
# CANONICAL PATHS (CONTRACTS section 1 -- module-level absolute roots)
# =============================================================================
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()

# Flat repository layout: inputs from the v3 stage and outputs of this stage
# share derived/, tables/, figures/ and logs/ (no filename collisions).
V3_DERIVED = config.DERIVED
V3_TABLES = config.TABLES
V3_LOGS = config.LOGS

V4_FIGURES = config.FIGURES
V4_TABLES = config.TABLES

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DPI = 300

# ---- Publication rcParams (copy-adapted from v3 render_figures.py:102) ------
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
    "pdf.fonttype": 42,   # editable/embedded TrueType (CONTRACTS section 1)
    "ps.fonttype": 42,
})

# ---- Okabe-Ito colour-blind-safe palette (copy of v3 render_figures.py:121) --
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
C_SCORE = OKABE_ITO["blue"]        # the pelvic arrival score
# Proper-noun display name of the instrument (v3 manuscript rename). Every
# figure, table and legend label below is built from this one constant.
SCORE_NAME = "Pelvic-Fracture Arrival Score"
C_SI = OKABE_ITO["vermilion"]      # shock-index comparator
C_RSIG = OKABE_ITO["green"]        # rSIG comparator
C_TEMPORAL = OKABE_ITO["purple"]   # temporal 2023-2024 validation overlay
C_ELDERLY = OKABE_ITO["orange"]    # elderly stratum
C_TREATALL = OKABE_ITO["grey"]     # DCA treat-all
C_TREATNONE = OKABE_ITO["black"]   # DCA treat-none

RULE_IN_PT = 0.20
DCA_PRESPEC = (0.05, 0.20)

# =============================================================================
# LABEL DISCIPLINE (copy-adapted from v3 render_figures.py; gates G4a/G4f/G3d)
# =============================================================================
VAR_LABELS = {
    "age_65_74": "Age 65-74",
    "age_ge75": "Age >=75",
    "age_ge65": "Age >=65",
    "sex_female": "Female sex",
    "sbp_lt90": "SBP <90 mmHg",
    "sbp_lt70": "SBP <70 mmHg",
    "sbp_unrecordable": "No signs of life on arrival (unrecordable systolic pressure)",
    "hr_ge120": "Heart rate >=120",
    "si_ge0_9": "Shock index >=0.9",
    "si_ge1_0": "Shock index >=1.0",
    "gcs_motor_le5": "GCS motor <=5",
    "gcs_le8": "GCS <=8",
    "mech_high_energy": "High-energy mechanism",
    "mech_mvc": "MVC (occupant)",
    "mech_fall": "Fall mechanism",
    "air_transport": "Air transport",
    "cc_anticoag": "Anticoagulant use",
    "cc_bleeding": "Bleeding disorder (preexisting)",
}
# Typographic upgrade applied at render time only (the ASCII keys above are what
# the gates grep; the display strings carry proper mathematical symbols).
_TYPO = ((">=", "\u2265"), ("<=", "\u2264"), ("65-74", "65\u201374"))

# Never renderable as a card row, whatever an upstream artifact says (gate G3d).
CARD_EXCLUDED_VARS = ("unstable_ring",)
# v4 (D5): a scored-zero candidate is not a card criterion. Removed from the
# Table 2 card ONLY; it stays in table3_points.csv / refit.json / the results
# bundle so the Results text can still quote its null odds ratio.
CARD_DROPPED_VARS = ("cc_anticoag",)
OVERRIDE_VARS = ("sbp_unrecordable",)
ROLE_POINT, ROLE_OVERRIDE = "point", "override"

# The one approved override sentence (CONTRACTS section 8 "F9 -- v3 required
# changes" item 2; gate G4f). refit.json owns the string; this literal is the
# equality target the gate asserts against.
OVERRIDE_RULE_TEXT = (
    "Automatic escalation regardless of score: "
    "no signs of life on arrival (unrecordable systolic pressure)"
)

# Retired v2 wording. Any label bound for an exhibit is rewritten through
# _sanitize_label() so a stale artifact cannot put it back (gate G4f).
RETIRED_LABEL_PATTERNS = (
    (r"SBP\s*unrecordable\s*\(?\s*=?\s*0?\s*\)?",
     "no signs of life on arrival (unrecordable systolic pressure)"),
    (r"\bSBP\s*=\s*0\b",
     "no signs of life on arrival (unrecordable systolic pressure)"),
    (r"\bsbp_unrecordable\b",
     "no signs of life on arrival (unrecordable systolic pressure)"),
)
# Missingness framings that must never describe the override (gate G4f).
FORBIDDEN_OVERRIDE_FRAMINGS = (
    "sbp = 0", "sbp=0", "sbp unrecordable", "missing systolic",
    "systolic missing", "not recorded", "blood pressure missing",
    "missing blood pressure", "unavailable systolic",
)

# =============================================================================
# BANNED CONSTANTS (execution brief, hard prohibition 1). Each encodes a bug
# this project has already shipped; none may appear in any emitted v4 file.
# Digit patterns are boundary-anchored so a legitimate longer number
# (e.g. 0.463978, 1,635) cannot raise a false alarm.
# =============================================================================
BANNED_PATTERNS = [
    ("0.876 (superseded v2 AUROC)", r"(?<![\d])0\.876(?![\d])"),
    ("31,054 (v2 composite count)", r"(?<![\d,.])31,?054(?![\d,.])"),
    ("0.1295 (superseded v2 rate)", r"(?<![\d])0\.1295(?![\d])"),
    ("635 (superseded v2 count)", r"(?<![\d,.])635(?![\d,.])"),
    ("+32 (v2 observed score max)", r"\+32(?![\d.,])"),
    ("-7..102 (v2 score range)", r"[-\u2212]7\s*\.\.\s*102"),
    ("27.3 (plan-era constant)", r"(?<![\d])27\.3(?![\d])"),
    ("'infeasible' (retracted comment)", r"(?i)\binfeasible\b"),
    ("'day resolution' (retired timing framing)", r"(?i)\bday resolution\b"),
    ("3,884 (superseded day-rule death count)", r"(?<![\d,.])3,?884(?![\d,.])"),
    ("4,209 (superseded death count)", r"(?<![\d,.])4,?209(?![\d,.])"),
    ("4,253 (superseded death count)", r"(?<![\d,.])4,?253(?![\d,.])"),
    ("4,478 (superseded death count)", r"(?<![\d,.])4,?478(?![\d,.])"),
]

# =============================================================================
# GATE LEDGER -- declared up front so an unevaluated gate is a FAIL, not a skip
# =============================================================================
GATES_DECLARED = [
    ("IN-1", "every required v3 input artifact exists and is readable"),
    ("F1-a", "flow ladder closes at every rung (n_before - n_excluded == n_after)"),
    ("F1-b", "flow ladder closes across consecutive rungs (n_after[k] == n_before[k+1])"),
    ("F1-c", "table_consort.csv agrees with cohort_exclusions.json; 7 rungs, 7,327,714 -> 239,736"),
    ("F1-d", "derivation + validation == analytic N, and split events sum to cohort events"),
    ("F1-e", "figure_1 rendered with no displayed number adjusted"),
    ("F2-a", "ROC AUROCs drawn are the INTEGER card (test 0.8823, temporal 0.8846)"),
    ("F2-b", "ROC comparator denominators kept separate and labelled (79,912 / 77,739 / 75,072 / 88,470)"),
    ("F2-c", "calibration panels drawn from artifacts; elderly decile n sums to reclass n"),
    ("F3-a", "decision curve drawn for score / shock index / treat-all / treat-none"),
    ("F3-b", "matched-workload deltas taken from Delta_* columns with Delta_comparable == 1"),
    ("F3-c", "matched-workload reference instrument is shock_index; no score-vs-rSIG delta drawn"),
    ("F3-d", "figure 3B labelled to n = 75,072 (matched set), not to the test cohort"),
    ("T1-a", "Table 1 carries Overall / No event / Event / SMD only (strata columns to Supplement)"),
    ("T1-b", "Table 1 outcome-definition crosswalk rows routed to the Supplement"),
    ("T1-c", "every Table 1 mean (SD) row is marked as one, as footnote 1 promises; "
             "no median (IQR) row carries the mark"),
    ("T2-1", "no unstable_ring row anywhere in Table 2 (G3c/G3d)"),
    ("T2-2", "override renders as a full-width banner with NO points value (G4a/G4f, F9 spec)"),
    ("T2-3", "v4 D5: cc_anticoag is absent from Table 2 (rows, footnotes, title); exactly 13 point "
             "rows plus the override; no remaining point value is 0; the null estimate stays in the source"),
    ("T2-4", "Table 2 carries exactly the 13 point rows (the 14 in table3_points.csv minus cc_anticoag)"),
    ("T2-5", "every remaining Table 2 row's beta / OR / CI / points equals the v3 value in "
             "table3_points.csv and score_card.json; order equals v3 order minus the dropped row"),
    ("T3-a", "only the pelvic_score row is flagged as the score's AUROC; logit / deployed rule labelled"),
    ("T3-b", "Table 3 utility block reproduces table5d values exactly"),
    ("TIF", "every emitted TIF opens with PIL and reports format == TIFF"),
    ("MIN", "every negative number in an emitted exhibit carries U+2212; range "
            "hyphens are untouched and no numeral's value moved"),
    ("BAN", "no banned constant appears in any emitted v4 exhibit file"),
    ("OUT", "every declared output file exists and is non-empty"),
]
GATE_RESULTS: dict[str, tuple[bool, str]] = {}


def gate(gid: str, ok, detail: str) -> bool:
    """Record and print one gate verdict. Never swallows anything."""
    ok = bool(ok)
    if gid in GATE_RESULTS:
        raise RuntimeError(f"gate {gid} evaluated twice")
    GATE_RESULTS[gid] = (ok, detail)
    print(f"GATE {gid}: {'PASS' if ok else 'FAIL'} \u2014 {detail}")
    return ok


# =============================================================================
# I/O + FORMAT HELPERS
# =============================================================================
def _req(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"required input artifact missing: {path}")
    return path


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(_req(path), low_memory=False)


def _read_json(path: str):
    with open(_req(path), "r") as fh:
        return json.load(fh)


def _read_pickle(path: str):
    with open(_req(path), "rb") as fh:
        return pickle.load(fh)


def _sanitize_label(text) -> str:
    """Rewrite retired v2 wording out of any string bound for an exhibit (G4f)."""
    s = str(text)
    for pat, repl in RETIRED_LABEL_PATTERNS:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s


def _typo(s: str) -> str:
    out = str(s)
    for a, b in _TYPO:
        out = out.replace(a, b)
    return out


def _score_cutoff(s) -> str:
    """Display form of the score's cut-point label from the v3 matched-workload
    table ("pelvic score >= +5"). The v3 artifact still carries the retired
    short name; the instrument is named once beside the label, so the cut-point
    reads "score >= +5"."""
    return _typo(re.sub(r"(?i)\bpelvic score\b", "score", str(s)))


def _var_label(var: str, overrides: dict | None = None) -> str:
    """Clinical label for a predictor. A gap is a LOUD failure, never a fallback."""
    if overrides and var in overrides:
        return _typo(_sanitize_label(overrides[var]))
    if var in VAR_LABELS:
        return _typo(_sanitize_label(VAR_LABELS[var]))
    raise KeyError(
        f"UNLABELLED VARIABLE `{var}`: no entry in VAR_LABELS. A raw snake_case "
        f"identifier on a manuscript exhibit is a submission-visible defect. Add "
        f"the variable to VAR_LABELS before rendering."
    )


def n_(x) -> str:
    return f"{int(round(float(x))):,}"


def pct_(x, d: int = 1) -> str:
    return f"{float(x):.{d}f}"


def auc_(a, lo, hi) -> str:
    return f"{float(a):.4f} ({float(lo):.4f}\u2013{float(hi):.4f})"


def signed_(x, d: int = 4) -> str:
    """Signed fixed-point, but a value that rounds to zero prints as plain zero.

    Prevents `-0.0000` (a CITL of -5.4e-07) from reading as a negative miscalibration.
    """
    v = float(x)
    if abs(v) < 0.5 * 10 ** (-d):
        return f"{0.0:.{d}f}"
    return f"{v:+.{d}f}"


# -----------------------------------------------------------------------------
# TYPOGRAPHIC MINUS SIGN (U+2212)
# -----------------------------------------------------------------------------
# `RESULTS.md` and the ledger renderer in build_bundle.py (`_signed`, lines
# 354-391) both print a negative number with U+2212. The exhibits printed ASCII
# hyphens for the SAME values, so one submission document carried two conventions
# for `-684` and `-269`. `_minus()` closes that split. It is purely typographic:
# it rewrites ONLY a hyphen that opens a numeric token.
#
# "Opens a numeric token" is delimiter-anchored on both sides:
#   * followed by a digit    -- so `rule-out`, `chi-squared`, `follow-up` are safe;
#   * NOT preceded by a digit -- so the IQR range `132 (114-151)` and the age band
#     `18-44 years` keep their ASCII hyphen. (That is also what keeps the
#     sign-anchored score-vs-rSIG margin sweeps behaving identically: they read a
#     digit-preceded hyphen as a range, never as a delta.)
#   * NOT preceded by a letter -- so `ICD-10`, `ICD-10-CM`, `COVID-19` are safe.
MINUS = "−"
_SIGN_HYPHEN = re.compile(r"(?<![0-9A-Za-z])-(?=[0-9])")


def _minus(s) -> str:
    """ASCII hyphen -> U+2212, but only where the hyphen is a numeric sign.

    The post-condition proves that no character other than a sign hyphen moved:
    mapping every U+2212 back to an ASCII hyphen, in both the input and the
    output, must give the same string. A digit, a decimal point or a thousands
    separator therefore cannot be altered here -- no VALUE can change, only the
    glyph that carries its sign.
    """
    src = str(s)
    out = _SIGN_HYPHEN.sub(MINUS, src)
    assert out.replace(MINUS, "-") == src.replace(MINUS, "-"), (
        f"_minus() altered something other than a sign hyphen: {src!r} -> {out!r}")
    return out


# `_minus` is load-bearing for every emitted exhibit, so its behaviour is asserted
# at import rather than assumed. Every case below is a string that really occurs
# in these tables or legends; the negative half proves the rule is not a blanket
# hyphen sweep.
for _src, _want in (
    ("-0.0567", "−0.0567"),                            # a CITL offset
    ("+678 events caught; -684 false positives",
     "+678 events caught; −684 false positives"),      # a workload delta
    ("-6 to +17", "−6 to +17"),                        # observed score range
    ("-1", "−1"),                                      # a points cell
    ("-0.577", "−0.577"),                              # a Table 1 SMD
    ("132 (114-151)", "132 (114-151)"),                     # IQR range   -- unchanged
    ("18-44 years", "18-44 years"),                         # age band    -- unchanged
    ("ICD-10-CM S32.1", "ICD-10-CM S32.1"),                 # code label  -- unchanged
    ("Non-Hispanic White", "Non-Hispanic White"),           # compound    -- unchanged
    ("rule-out (score -6 to 0)", "rule-out (score −6 to 0)"),   # both at once
    ("calibration-in-the-large offset -0.0567",
     "calibration-in-the-large offset −0.0567"),       # both at once
):
    if _minus(_src) != _want:
        raise AssertionError(
            "_minus() self-test failed: %r -> %r, want %r"
            % (_src, _minus(_src), _want))
del _src, _want


def _save_figure(fig, basename: str) -> list[str]:
    """Save PNG + TIF, both 300 dpi, bbox_inches='tight'; close; return paths.

    Copy-adapted from v3/code/render_figures.py::_save_figure (the section 8
    300-dpi contract). The TIF writer is the v4 addition: LZW-compressed TIFF is
    what JTACS production wants, and it is written HERE rather than patched into
    the v3 renderer so v3 stays byte-stable.
    """
    os.makedirs(V4_FIGURES, exist_ok=True)
    png_path = os.path.join(V4_FIGURES, f"{basename}.png")
    tif_path = os.path.join(V4_FIGURES, f"{basename}.tif")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(tif_path, format="tiff", dpi=DPI, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  saved: {basename}.png / {basename}.tif  ->  {V4_FIGURES}")
    return [png_path, tif_path]


def _figure_footnote(fig, text, y=-0.015, fontsize=8.2, wrap_chars=150):
    """Stamp a wrapped caption under a figure (included by bbox_inches='tight')."""
    lines = []
    for para in ([text] if isinstance(text, str) else list(text)):
        lines.extend(textwrap.wrap(str(para), width=wrap_chars) or [""])
    fig.text(0.008, y, "\n".join(lines), ha="left", va="top",
             fontsize=fontsize, color="#333333", style="italic")


# =============================================================================
# TABLE WRITERS -- CSV + Markdown, same rows in both
# =============================================================================
def _md_escape(cell) -> str:
    return str("" if cell is None else cell).replace("|", "\\|").replace("\n", " ")


def write_table(name: str, columns: list[str], rows: list[list],
                footnotes: list[str], title: str,
                merged_rows: set[int] | None = None) -> list[str]:
    """Emit `<name>.csv` and `<name>.md` under tables/.

    `merged_rows` holds 0-based indices into `rows` whose first cell spans the
    full table width -- the mechanism the score card's override banner uses.
    In the CSV/MD those rows carry the banner text in column 0 and EMPTY strings
    in every other column, so the banner carries no point value in any format.

    Every cell, header, footnote and the title pass through `_minus()` FIRST, so
    the two formats cannot disagree about the sign glyph and a regeneration
    cannot reintroduce the ASCII/U+2212 split.
    """
    os.makedirs(V4_TABLES, exist_ok=True)
    merged_rows = set(merged_rows or ())
    paths = []

    # ---- typographic minus, applied once, at the single choke point ---------
    raw_columns, raw_rows, raw_footnotes, raw_title = columns, rows, footnotes, title
    columns = [_minus(c) for c in raw_columns]
    rows = [[(c if c is None else _minus(c)) for c in r] for r in raw_rows]
    footnotes = [_minus(f) for f in raw_footnotes]
    title = _minus(raw_title)
    # Proof that the pass moved sign glyphs and nothing else: mapping U+2212 back
    # to an ASCII hyphen restores the input exactly, cell for cell.
    def _back(x):
        return None if x is None else str(x).replace(MINUS, "-")
    assert [_back(c) for c in columns] == [_back(c) for c in raw_columns]
    assert [[_back(c) for c in r] for r in rows] == \
           [[_back(c) for c in r] for r in raw_rows]
    assert [_back(f) for f in footnotes] == [_back(f) for f in raw_footnotes]
    assert _back(title) == _back(raw_title)

    # ---- CSV ----------------------------------------------------------------
    csv_path = os.path.join(V4_TABLES, f"{name}.csv")
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    paths.append(csv_path)

    # ---- Markdown -----------------------------------------------------------
    md_path = os.path.join(V4_TABLES, f"{name}.md")
    with open(md_path, "w") as fh:
        fh.write(f"# {name}. {title}\n\n")
        fh.write("| " + " | ".join(_md_escape(c) for c in columns) + " |\n")
        fh.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for i, r in enumerate(rows):
            if i in merged_rows:
                fh.write("| **" + _md_escape(r[0]) + "** |"
                         + "|".join([""] * (len(columns) - 1)) + "|\n")
            else:
                fh.write("| " + " | ".join(_md_escape(c) for c in r) + " |\n")
        if footnotes:
            fh.write("\n")
            for fn in footnotes:
                fh.write(f"- {fn}\n")
    paths.append(md_path)

    print(f"  wrote: {name}.csv / {name}.md  ->  {V4_TABLES}")
    return paths


# =============================================================================
# FIGURE 1 -- participant flow (copy-adapted from v3/code/14_consort.py)
# =============================================================================
FIGURE_STEP_LINES = [
    ["All NTDB/TQP encounters", "2019\u20132024"],
    ["Adult patients", "(age \u2265 18 years)"],
    ["Pelvic-ring fracture", "(ICD-10-CM diagnosis S32.1/.3/.5/.6/.81/.82)"],
    ["Blunt mechanism of injury"],
    ["Direct arrival (non-transfer)"],
    ["Survivable, observable injury", "(not dead on arrival; no AIS-6 injury)"],
    ["Analytic cohort", "(valid outcome ascertainable)"],
]
FIGURE_EXCL_LABELS = [
    "Age < 18 y, or age not recorded in years",
    "No qualifying pelvic-ring fracture diagnosis",
    "Penetrating or unspecified mechanism",
    "Transferred in from another facility",
    "Prehospital cardiac arrest, or any AIS-6 injury",
    "Unknown disposition (outcome not ascertainable)",
]


def figure_1(art: dict) -> list[str]:
    print("\n" + "=" * 72)
    print("FIGURE 1 -- participant flow")
    print("=" * 72)

    excl = art["cohort_exclusions"]
    consort = art["consort"]
    split = art["split"]
    qc = art["qc"]

    json_steps = [(k, int(v)) for k, v in excl.items()]
    n_steps = len(json_steps)

    # ---- gate F1-c: the CSV and the JSON describe the same ladder ------------
    csv_after = [int(x) for x in consort["N_after"].tolist()]
    json_after = [v for _, v in json_steps]
    same = (csv_after == json_after)
    endpoints = (json_after[0] == 7_327_714 and json_after[-1] == 239_736)
    seven = (n_steps == 7 and len(consort) == 7)
    manifest_n = int(qc["n_primary"])
    gate("F1-c",
         same and endpoints and seven and manifest_n == json_after[-1],
         f"7 rungs; cohort_exclusions.json N_after == table_consort.csv N_after "
         f"({same}); ladder {n_(json_after[0])} -> {n_(json_after[-1])}; "
         f"qc_manifest.n_primary = {n_(manifest_n)}")

    # ---- gate F1-a: within-rung closure -------------------------------------
    per_rung = []
    ok_a = True
    for _, r in consort.iterrows():
        if str(r["N_before"]).strip() in ("", "nan"):
            continue  # entry rung has no predecessor
        nb, na, nx = int(float(r["N_before"])), int(float(r["N_after"])), int(float(r["N_excluded"]))
        closed = (nb - nx == na)
        ok_a = ok_a and closed
        per_rung.append(f"rung {int(r['Step'])}: {n_(nb)} - {n_(nx)} = {n_(na)} "
                        f"[{'ok' if closed else 'BROKEN'}]")
    for line in per_rung:
        print(f"    {line}")
    gate("F1-a", ok_a and len(per_rung) == n_steps - 1,
         f"{len(per_rung)} of {n_steps - 1} transitions checked; "
         f"n_before - n_excluded == n_after at every rung")

    # ---- gate F1-b: across-rung closure -------------------------------------
    ok_b = True
    chain = []
    for k in range(len(consort) - 1):
        a_k = int(float(consort.iloc[k]["N_after"]))
        b_k1 = int(float(consort.iloc[k + 1]["N_before"]))
        linked = (a_k == b_k1)
        ok_b = ok_b and linked
        chain.append(f"rung {k + 1} N_after {n_(a_k)} == rung {k + 2} N_before "
                     f"{n_(b_k1)} [{'ok' if linked else 'BROKEN'}]")
    for line in chain:
        print(f"    {line}")
    gate("F1-b", ok_b and len(chain) == n_steps - 1,
         f"{len(chain)} consecutive-rung links checked; every rung k N_after "
         f"equals rung k+1 N_before")

    counts = json_after
    excl_counts = [counts[i - 1] - counts[i] for i in range(1, len(counts))]

    # Hard stop: nothing is drawn on a ladder that does not close.
    for i in range(1, n_steps):
        assert counts[i - 1] - excl_counts[i - 1] == counts[i], \
            f"cascade mismatch at rung {i}: no displayed number may be adjusted"

    # ---- gate F1-d: split closure -------------------------------------------
    sb = split.set_index("split")
    d_n, v_n = int(sb.loc["train", "N"]), int(sb.loc["test", "N"])
    d_e, v_e = int(sb.loc["train", "N_events"]), int(sb.loc["test", "N_events"])
    f_e = int(sb.loc["full", "N_events"])
    assert d_n + v_n == counts[-1], "split does not sum to the analytic N"
    assert d_e + v_e == f_e, "split event counts do not sum to the cohort events"
    gate("F1-d", True,
         f"{n_(d_n)} + {n_(v_n)} = {n_(counts[-1])}; events {n_(d_e)} + {n_(v_e)} "
         f"= {n_(f_e)}")

    # ---- draw ---------------------------------------------------------------
    assert len(FIGURE_STEP_LINES) == n_steps, "FIGURE_STEP_LINES out of sync"
    assert len(FIGURE_EXCL_LABELS) == n_steps - 1, "FIGURE_EXCL_LABELS out of sync"

    temporal = art["citl"]["composite"]
    with plt.rc_context({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        FIG_W, FIG_H = 7.4, 9.03
        BOX_FC, BOX_EC, BOX_LW = "white", "black", 0.6
        ARROW_LW = 0.8
        ARROW_KW = dict(arrowstyle="-|>", mutation_scale=8, color="black",
                        linewidth=ARROW_LW, shrinkA=0, shrinkB=0)
        EXCL_KW = dict(ARROW_KW)
        EXCL_KW["linestyle"] = (0, (2.5, 2.0))

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.set_xlim(0, 100)
        ax.set_ylim(2.0, 100)
        ax.set_axis_off()

        MAIN_X, MAIN_W = 31.0, 50.0
        BOX_H, ANALYTIC_H = 8.0, 11.0
        EXCL_X, EXCL_W, EXCL_H = 80.0, 39.0, 8.5
        ys = list(np.linspace(96.0, 30.0, n_steps))

        def draw_box(cx, cy, w, h, lines, *, italic=False, fontsize=8.0):
            ax.add_patch(FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="round,pad=0.30,rounding_size=0.50",
                linewidth=BOX_LW, edgecolor=BOX_EC, facecolor=BOX_FC, clip_on=False))
            style = "italic" if italic else "normal"
            m = len(lines)
            step = (h - 1.6) / max(m, 1) if m > 1 else 0
            top_y = cy + (step * (m - 1)) / 2
            for i, line in enumerate(lines):
                weight = "bold" if (i == 0 and not italic) else "normal"
                ax.text(cx, top_y - i * step, line, ha="center", va="center",
                        fontsize=fontsize, color="black", fontstyle=style,
                        fontweight=weight)

        def varrow(x, y_from, y_to):
            ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), **ARROW_KW))

        for i in range(n_steps):
            cy = ys[i]
            is_last = (i == n_steps - 1)
            h = ANALYTIC_H if is_last else BOX_H
            if is_last:
                ax.add_patch(FancyBboxPatch(
                    (MAIN_X - MAIN_W / 2, cy - h / 2), MAIN_W, h,
                    boxstyle="round,pad=0.30,rounding_size=0.50",
                    linewidth=BOX_LW, edgecolor=BOX_EC, facecolor=BOX_FC, clip_on=False))
                ax.text(MAIN_X, cy + 3.4, FIGURE_STEP_LINES[i][0], ha="center",
                        va="center", fontsize=8.0, fontweight="bold")
                ax.text(MAIN_X, cy + 0.9, FIGURE_STEP_LINES[i][1], ha="center",
                        va="center", fontsize=8.0)
                ax.text(MAIN_X, cy - 1.5, f"N = {n_(counts[i])}", ha="center",
                        va="center", fontsize=8.0)
                ax.text(MAIN_X, cy - 3.9,
                        f"Major hemorrhage event (composite): n = {n_(f_e)} "
                        f"({100 * f_e / counts[-1]:.1f}%)",
                        ha="center", va="center", fontsize=6.9, fontstyle="italic")
            else:
                draw_box(MAIN_X, cy, MAIN_W, h,
                         list(FIGURE_STEP_LINES[i]) + [f"N = {n_(counts[i])}"])

            if i < n_steps - 1:
                y_bottom = cy - h / 2
                next_h = ANALYTIC_H if (i + 1 == n_steps - 1) else BOX_H
                y_top_next = ys[i + 1] + next_h / 2
                varrow(MAIN_X, y_bottom, y_top_next)
                y_tap = (y_bottom + y_top_next) / 2
                x_stub = MAIN_X + MAIN_W / 2 - 0.5
                ax.plot([MAIN_X, x_stub], [y_tap, y_tap], color="black",
                        linewidth=ARROW_LW, linestyle=(0, (2.5, 2.0)),
                        solid_capstyle="butt")
                ax.add_patch(FancyArrowPatch((x_stub, y_tap),
                                             (EXCL_X - EXCL_W / 2, y_tap), **EXCL_KW))
                draw_box(EXCL_X, y_tap, EXCL_W, EXCL_H,
                         [f"Excluded (n = {n_(excl_counts[i])}):", FIGURE_EXCL_LABELS[i]],
                         italic=True, fontsize=7.3)

        SPLIT_W, SPLIT_H = 26.0, 9.5
        X_DERIV, X_VALID = MAIN_X - 16.0, MAIN_X + 16.0
        Y_SPLIT = 9.0
        Y_T_BAR = Y_SPLIT + SPLIT_H / 2 + 6.0
        ax.plot([MAIN_X, MAIN_X], [ys[-1] - ANALYTIC_H / 2, Y_T_BAR], color="black",
                linewidth=ARROW_LW, solid_capstyle="butt")
        ax.plot([X_DERIV, X_VALID], [Y_T_BAR, Y_T_BAR], color="black",
                linewidth=ARROW_LW, solid_capstyle="butt")
        varrow(X_DERIV, Y_T_BAR, Y_SPLIT + SPLIT_H / 2)
        varrow(X_VALID, Y_T_BAR, Y_SPLIT + SPLIT_H / 2)
        draw_box(X_DERIV, Y_SPLIT, SPLIT_W, SPLIT_H, [
            "Derivation cohort", "(2/3, stratified)", f"N = {n_(d_n)}",
            f"events n = {n_(d_e)} ({100 * d_e / d_n:.1f}%)"], fontsize=7.6)
        draw_box(X_VALID, Y_SPLIT, SPLIT_W, SPLIT_H, [
            "Internal held-out test", "(1/3, stratified)", f"N = {n_(v_n)}",
            f"events n = {n_(v_e)} ({100 * v_e / v_n:.1f}%)"], fontsize=7.6)

        paths = _save_figure(fig, "figure_1")

    gate("F1-e", os.path.exists(paths[0]) and os.path.exists(paths[1]),
         "figure_1 rendered straight from cohort_exclusions.json / "
         "table_consort.csv / table_split_balance.csv; no displayed number adjusted")
    return paths


# =============================================================================
# FIGURE 2 -- A: ROC (test + temporal)   B: calibration (overall + elderly)
# =============================================================================
def figure_2(art: dict) -> list[str]:
    print("\n" + "=" * 72)
    print("FIGURE 2 -- discrimination and calibration")
    print("=" * 72)

    roc = art["roc"]
    troc = art["temporal_roc"]
    disc = art["disc"]
    t4b = art["t4b"]
    cal = art["calibration"]
    reclass = art["reclass"]

    def disc_row(score, auroc_type):
        sub = disc[(disc["Outcome"] == "major_hemorrhage_event")
                   & (disc["Score"] == score)
                   & (disc["AUROC_type"] == auroc_type)]
        assert len(sub) == 1, f"expected 1 discrimination row for {score}/{auroc_type}"
        return sub.iloc[0]

    r_score = disc_row("pelvic_score", "integer_bedside")
    r_si = disc_row("SI-alone", "comparator")
    r_rsig = disc_row("rSIG", "comparator")
    t_row = t4b[t4b["outcome"] == "major_hemorrhage_event"]
    assert len(t_row) == 1, "expected exactly 1 PRIMARY temporal row"
    t_row = t_row.iloc[0]

    # ---- gate F2-a: the AUROCs drawn are the INTEGER card --------------------
    pk = roc["pelvic_score"]
    tpk = troc["temporal_composite_integer"]
    ok_a = (
        abs(float(pk["auc"]) - float(r_score["AUROC"])) < 5e-5
        and abs(float(tpk["auc"]) - float(t_row["AUROC_integer"])) < 5e-5
        and round(float(pk["auc"]), 4) == 0.8823
        and round(float(tpk["auc"]), 4) == 0.8846
        and tpk["model"] == "integer_card"
        and str(r_score["Instrument_Role"]).startswith("PRIMARY")
    )
    gate("F2-a", ok_a,
         f"test {round(float(pk['auc']), 4)} (table4_discrimination "
         f"AUROC_type=integer_bedside) and temporal "
         f"{round(float(tpk['auc']), 4)} (table4b AUROC_integer, model="
         f"{tpk['model']}); continuous logit 0.8861/0.8883 not drawn")

    n_test, n_si, n_rsig = int(r_score["N_used"]), int(r_si["N_used"]), int(r_rsig["N_used"])
    n_temp = int(tpk["n"])
    ok_b = (n_test == 79912 and n_si == 77739 and n_rsig == 75072 and n_temp == 88470
            and len({n_test, n_si, n_rsig, n_temp}) == 4)
    gate("F2-b", ok_b,
         f"score n = {n_(n_test)}, shock index n = {n_(n_si)}, rSIG n = "
         f"{n_(n_rsig)}, temporal n = {n_(n_temp)} \u2014 four distinct "
         f"denominators, each printed in its own legend entry")

    # ---- calibration inputs -------------------------------------------------
    co = cal[cal["score"] == "pelvic_score"].sort_values("bin")
    assert len(co) > 0, "no pelvic_score rows in calibration_data.csv"
    n_cal = int(co["n"].iloc[0])
    citl_o = float(co["citl"].iloc[0])

    ed = pd.DataFrame(reclass["elderly_calibration_deciles"]).sort_values("decile")
    ecal = reclass["elderly_calibration"]
    n_eld = int(ecal["n"])
    citl_e = float(ecal["citl"])
    sum_dec = int(ed["n"].sum())
    gate("F2-c", sum_dec == n_eld and len(ed) == 10 and len(co) >= 8,
         f"overall calibration on n = {n_(n_cal)} in {len(co)} risk groups "
         f"(integer scores tie, so 10 requested deciles collapse to {len(co)}); "
         f"elderly on n = {n_(n_eld)} in {len(ed)} deciles, decile n sums to "
         f"{n_(sum_dec)}")

    # ---- draw ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.7))

    # ---------------- Panel A: ROC -------------------------------------------
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], linestyle=":", linewidth=1.2, zorder=1)
    series = [
        (roc["pelvic_score"], C_SCORE, 2.7, "-",
         f"{SCORE_NAME}, integer card: internal test set\n"
         f"    n = {n_(n_test)}: AUROC {auc_(pk['auc'], pk['lo'], pk['hi'])}"),
        (troc["temporal_composite_integer"], C_TEMPORAL, 2.1, "--",
         f"{SCORE_NAME}, re-estimated integer card,\n    temporal validation "
         f"2023\u20132024\n    n = {n_(n_temp)}: AUROC "
         f"{auc_(tpk['auc'], tpk['lo'], tpk['hi'])}"),
        (roc["rSIG"], C_RSIG, 1.8, "-",
         f"rSIG (systolic pressure \u00f7 heart rate \u00d7 Glasgow Coma Scale)\n"
         f"    n = {n_(n_rsig)}: AUROC "
         f"{auc_(roc['rSIG']['auc'], roc['rSIG']['lo'], roc['rSIG']['hi'])}"),
        (roc["SI-alone"], C_SI, 1.8, "-",
         f"Shock index alone\n    n = {n_(n_si)}: AUROC "
         f"{auc_(roc['SI-alone']['auc'], roc['SI-alone']['lo'], roc['SI-alone']['hi'])}"),
    ]
    handles = []
    for entry, col, lw, ls, label in series:
        ax.plot(np.asarray(entry["fpr"], float), np.asarray(entry["tpr"], float),
                color=col, linewidth=lw, linestyle=ls, zorder=3,
                solid_capstyle="round")
        handles.append(plt.Line2D([0], [0], color=col, linewidth=lw, linestyle=ls,
                                  label=label))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("1 \u2212 specificity (false-positive rate)")
    ax.set_ylabel("Sensitivity (true-positive rate)")
    ax.set_title("Discrimination for major hemorrhage", fontsize=12)
    ax.legend(handles=handles, loc="lower right", framealpha=0.95, fontsize=7.4,
              handlelength=2.6, labelspacing=0.9, borderpad=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")
    ax.text(-0.10, 1.05, "A", transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top")

    # ---------------- Panel B: calibration -----------------------------------
    ax = axes[1]
    xo = co["mean_pred"].to_numpy(float)
    yo = co["obs_frac"].to_numpy(float)
    xe = ed["mean_pred"].to_numpy(float)
    ye = ed["obs_frac"].to_numpy(float)
    lo_e = ed["obs_ci_low"].to_numpy(float)
    hi_e = ed["obs_ci_high"].to_numpy(float)
    lim = float(max(xo.max(), yo.max(), xe.max(), ye.max())) * 1.08

    ax.plot([0, lim], [0, lim], color=OKABE_ITO["grey"], linestyle=":", linewidth=1.3,
            zorder=1, label="Perfect calibration")
    ax.errorbar(xe, ye, yerr=[np.clip(ye - lo_e, 0, None), np.clip(hi_e - ye, 0, None)],
                fmt="none", ecolor=C_ELDERLY, alpha=0.6, elinewidth=1.3, capsize=3, zorder=2)
    ax.plot(xo, yo, marker="o", color=C_SCORE, linewidth=1.9, markersize=7.5,
            markerfacecolor="white", markeredgecolor=C_SCORE, markeredgewidth=1.9,
            zorder=4, label=f"Overall test set, n = {n_(n_cal)} ({len(co)} risk groups)")
    ax.plot(xe, ye, marker="s", color=C_ELDERLY, linewidth=1.7, markersize=6.5,
            markerfacecolor="white", markeredgecolor=C_ELDERLY, markeredgewidth=1.7,
            linestyle="--", zorder=3,
            label=f"Age \u2265 65 y, n = {n_(n_eld)} (10 deciles)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability of major hemorrhage")
    ax.set_ylabel("Observed major-hemorrhage frequency")
    ax.set_title("Calibration", fontsize=12)
    ax.text(0.035, 0.965,
            _minus(f"Calibration-in-the-large\n  overall {signed_(citl_o)}\n"
                   f"  age \u2265 65 y {signed_(citl_e)}"),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=OKABE_ITO["grey"],
                      alpha=0.92))
    ax.legend(loc="lower right", framealpha=0.94, fontsize=8.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")
    ax.text(-0.10, 1.05, "B", transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top")

    plt.tight_layout(w_pad=3.0)
    return _save_figure(fig, "figure_2")


# =============================================================================
# FIGURE 3 -- A: decision curve   B: matched-workload contrast
# =============================================================================
def _workload_rows(w: pd.DataFrame, stratum: str):
    """The deployed-rule-in score row and its shock-index reference row.

    The score row is selected by name, NOT by position, and the score-OR-override
    variant is excluded by its Cutoff_Label so the point-scale rule is what the
    exhibit contrasts.
    """
    s = w[(w["Stratum"] == stratum)
          & (w["Workload_target_label"] == "deployed_rule_in")]
    score = s[(s["Instrument"] == "pelvic_score") & (s["Rule"] == "score_only")]
    ref = s[(s["Instrument"] == "shock_index") & (s["Rule"] == "instrument_only")]
    assert len(score) == 1, f"expected 1 score row for stratum {stratum}, got {len(score)}"
    assert len(ref) == 1, f"expected 1 shock-index row for stratum {stratum}, got {len(ref)}"
    # Belt and braces: `Rule == score_only` must also be the point-scale label, not
    # the score-OR-override variant, whose Cutoff_Label carries an " OR " clause.
    assert " OR " not in str(score.iloc[0]["Cutoff_Label"]), \
        "Rule=='score_only' row carries an override clause; artifact schema changed"
    return score.iloc[0], ref.iloc[0]


def figure_3(art: dict) -> list[str]:
    print("\n" + "=" * 72)
    print("FIGURE 3 -- clinical utility")
    print("=" * 72)

    dca = art["dca"]
    edca = art["edca"]
    w = art["t5d"]

    s_all, r_all = _workload_rows(w, "overall")
    s_eld, r_eld = _workload_rows(w, "elderly_ge65")

    # ---- gate F3-b: deltas come from the CSV's own columns -------------------
    ok_b = True
    detail_bits = []
    for tag, srow in (("overall", s_all), ("elderly_ge65", s_eld)):
        comparable = int(srow["Delta_comparable"]) == 1
        within = int(srow["Workload_match_within_tolerance"]) == 1
        d_ev = float(srow["Delta_events_caught"])
        d_fp = float(srow["Delta_false_positives"])
        finite = np.isfinite(d_ev) and np.isfinite(d_fp)
        ok_b = ok_b and comparable and within and finite
        detail_bits.append(f"{tag}: Delta_comparable={int(srow['Delta_comparable'])}, "
                           f"Delta_events_caught={d_ev:+.0f}, "
                           f"Delta_false_positives={d_fp:+.0f}")
    gate("F3-b", ok_b,
         "; ".join(detail_bits)
         + " \u2014 read verbatim from table5d_matched_workload.csv; "
           "Events_caught_TP is never differenced across rows")

    # ---- gate F3-c: the reference instrument is the shock index --------------
    refs = {str(s_all["Reference_instrument"]), str(s_eld["Reference_instrument"])}
    ok_c = all(x.startswith("shock_index") for x in refs) \
        and set(w["Reference_instrument"].astype(str).str[:11]) == {"shock_index"}
    gate("F3-c", ok_c,
         "every Reference_instrument in table5d is shock_index; no "
         "score-vs-rSIG matched-workload delta exists in any artifact, so none "
         "is drawn or computed")

    # ---- gate F3-d: panel B denominator --------------------------------------
    n_all, e_all = int(s_all["N_analysis"]), int(s_all["N_events"])
    n_eld, e_eld = int(s_eld["N_analysis"]), int(s_eld["N_events"])
    den_all = str(s_all["Denominator"])
    label_all = (f"Overall matched set: n = {n_(n_all)}; {n_(e_all)} events")
    label_eld = (f"Age \u2265 65 y, matched set: n = {n_(n_eld)}; {n_(e_eld)} events")
    gate("F3-d", n_all == 75072 and e_all == 8358
         and den_all == "common_complete_case_score_SI_rSIG"
         and n_(n_all) in label_all and "79,912" not in label_all,
         f"panel B overall label reads \"{label_all}\" "
         f"(Denominator = {den_all}); the 79,912 test cohort is not used here")

    # ---- draw ---------------------------------------------------------------
    fig = plt.figure(figsize=(14.6, 7.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], hspace=0.75, wspace=0.26)
    axA = fig.add_subplot(gs[:, 0])
    axB1 = fig.add_subplot(gs[0, 1])
    axB2 = fig.add_subplot(gs[1, 1])

    # ---------------- Panel A: decision curve --------------------------------
    base = dca.drop_duplicates("threshold").sort_values("threshold")
    thr = base["threshold"].to_numpy(float)
    ebase = edca.drop_duplicates("threshold").sort_values("threshold")
    axA.plot(thr, base["treat_all_nb"].to_numpy(float), color=C_TREATALL,
             linewidth=1.6, linestyle="-", zorder=2,
             label="Treat all: overall test set")
    axA.plot(ebase["threshold"].to_numpy(float),
             ebase["treat_all_nb"].to_numpy(float), color=C_TREATALL,
             linewidth=1.3, linestyle="--", zorder=2,
             label="Treat all: age ≥ 65 y")
    axA.axhline(0.0, color=C_TREATNONE, linewidth=1.4, linestyle="-", zorder=2,
                label="Treat none (escalate no one)")

    drew_a = 0
    for frame, stratum_label, ls, alpha in (
            (dca, "overall test set", "-", 1.0),
            (edca, "age \u2265 65 y", "--", 0.95)):
        for key, col, disp in (("pelvic_score", C_SCORE, SCORE_NAME),
                               ("SI-alone", C_SI, "Shock index alone")):
            sub = frame[frame["score"] == key].sort_values("threshold")
            if len(sub) == 0:
                continue
            nn = int(sub["n_used"].iloc[0])
            axA.plot(sub["threshold"].to_numpy(float),
                     sub["net_benefit"].to_numpy(float),
                     color=col, linewidth=2.5 if ls == "-" else 1.7, linestyle=ls,
                     alpha=alpha, zorder=4, solid_capstyle="round",
                     label=f"{disp}: {stratum_label}, n = {n_(nn)}")
            drew_a += 1

    axA.axvspan(DCA_PRESPEC[0], DCA_PRESPEC[1], color=OKABE_ITO["yellow"],
                alpha=0.12, zorder=0)
    axA.axvline(RULE_IN_PT, color="#444444", linestyle="--", linewidth=1.3, zorder=3)
    axA.set_xlim(float(thr.min()), float(thr.max()))
    axA.set_ylim(-0.02, 0.115)
    axA.text(RULE_IN_PT, -0.016, "  rule-in threshold\n  (p$_t$ = 0.20)", ha="left",
             va="bottom", fontsize=8.4, color="#333333")
    axA.set_xlabel("Threshold probability p$_t$")
    axA.set_ylabel("Net benefit")
    axA.set_title("Decision-curve analysis", fontsize=12)
    axA.legend(loc="upper right", framealpha=0.95, fontsize=7.6, labelspacing=0.55)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.grid(alpha=0.22, linestyle="--")
    axA.text(-0.11, 1.045, "A", transform=axA.transAxes, fontsize=17,
             fontweight="bold", va="top")
    gate("F3-a", drew_a == 4,
         f"{drew_a} instrument curves drawn (pelvic score and shock index, "
         f"overall and age \u2265 65 y) against treat-all and treat-none")

    # ---------------- Panel B: matched-workload contrast ---------------------
    def wl_panel(ax, srow, rrow, title, letter=None, header=False):
        """Two bar groups, instrument named INSIDE its own bar (no legend to
        collide with the bars), the delta column read straight from the CSV."""
        cats = ["Events caught", "False positives"]
        score_vals = [int(srow["Events_caught_TP"]), int(srow["False_positives_FP"])]
        ref_vals = [int(rrow["Events_caught_TP"]), int(rrow["False_positives_FP"])]
        # Deltas are READ, never differenced (this encodes a removed bug).
        deltas = [float(srow["Delta_events_caught"]), float(srow["Delta_false_positives"])]
        comparable = int(srow["Delta_comparable"]) == 1

        y = np.arange(len(cats))
        h = 0.36
        ax.barh(y - h / 2, score_vals, height=h, color=C_SCORE, zorder=3)
        ax.barh(y + h / 2, ref_vals, height=h, color=C_SI, zorder=3)
        xmax = max(score_vals + ref_vals)
        d_x = xmax * 1.10
        for i in range(len(cats)):
            ax.text(xmax * 0.018, y[i] - h / 2, SCORE_NAME, va="center",
                    ha="left", fontsize=7.8, color="white", fontweight="bold", zorder=5)
            ax.text(xmax * 0.018, y[i] + h / 2, "Shock index", va="center",
                    ha="left", fontsize=7.8, color="white", fontweight="bold", zorder=5)
            ax.text(score_vals[i] + xmax * 0.015, y[i] - h / 2, n_(score_vals[i]),
                    va="center", ha="left", fontsize=8.6, color=C_SCORE,
                    fontweight="bold")
            ax.text(ref_vals[i] + xmax * 0.015, y[i] + h / 2, n_(ref_vals[i]),
                    va="center", ha="left", fontsize=8.6, color=C_SI)
            if comparable:
                ax.text(d_x, y[i], _minus(f"{deltas[i]:+,.0f}"), va="center", ha="left",
                        fontsize=11, fontweight="bold",
                        color=(OKABE_ITO["green"] if
                               ((i == 0 and deltas[i] > 0) or (i == 1 and deltas[i] < 0))
                               else OKABE_ITO["vermilion"]))
        ax.set_yticks(y)
        ax.set_yticklabels(cats, fontsize=9.6)
        ax.set_ylim(1.62, -0.66)      # inverted; extra headroom for the \u0394 header
        ax.set_xlim(0, xmax * 1.45)
        ax.set_xlabel("Patients", fontsize=10)
        ax.set_title(title, fontsize=10.4, pad=17)
        ax.text(0.5, 1.035,
                f"{_score_cutoff(srow['Cutoff_Label'])}: {n_(srow['N_flagged'])} flagged "
                f"({float(srow['Pct_flagged_actual']):.2f}%)   \u00b7   "
                f"{_typo(str(rrow['Cutoff_Label']))}: {n_(rrow['N_flagged'])} flagged "
                f"({float(rrow['Pct_flagged_actual']):.2f}%)",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8,
                color="#444444")
        if header:
            ax.text(d_x, -0.52, "\u0394 score \u2212\nshock index", va="center",
                    ha="left", fontsize=7.8, style="italic", color="#444444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, linestyle="--", axis="x")
        if letter:
            ax.text(-0.13, 1.20, letter, transform=ax.transAxes, fontsize=17,
                    fontweight="bold", va="top")

    wl_panel(axB1, s_all, r_all, label_all, letter="B", header=True)
    wl_panel(axB2, s_eld, r_eld, label_eld, header=True)

    # No tight_layout here: the plate mixes a full-height gridspec cell with two
    # stacked cells, which tight_layout cannot solve (it warns and may misplace
    # the axes). The gridspec ratios above already set the spacing.
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.115)
    return _save_figure(fig, "figure_3")


# =============================================================================
# TABLE 1 -- baseline characteristics
# =============================================================================
# Outcome rows kept in the main text: the primary composite, its three named
# components, and the co-primary patient-state composite. Everything else in the
# Outcomes block is the outcome-definition crosswalk and goes to the Supplement
# (it is also where every superseded death count lives).
#
# Footnote 1 of Table 1 promises median (interquartile range) "unless marked as
# mean (standard deviation)".  Exactly one continuous row is a mean (SD) --
# age -- and it shipped UNMARKED, so the footnote's escape clause pointed at
# nothing and the row read as a median with an IQR of a single number.  The mark
# is applied HERE, in the generator, so a regeneration cannot lose it again, and
# gate T1-c re-derives it from the rendered cells rather than trusting a list.
#
# The two shape detectors below read the RENDERED Overall cell, which is the
# only place the dispersion convention is visible:
#   mean (SD)        "58.3 (21.8)"      -- one bare number inside the parens
#   median (IQR)     "132 (114-151)"    -- a range inside the parens
# n (%) rows are excluded by both: their percent sign and their thousands
# separators fall outside the character classes.
T1_DISPERSION_MARK = ", mean (SD)"
_T1_MEAN_SD = re.compile(r"^\s*-?[\d.]+\s*\(\s*-?[\d.]+\s*\)\s*$")
_T1_MEDIAN_IQR = re.compile(
    r"^\s*[-\u2212]?[\d,.]+\s*\(\s*[-\u2212]?[\d,.]+\s*[-\u2212]\s*[\d,.]+\s*\)\s*$")

# Import-time self-test: both detectors must separate the two conventions and
# must both reject an n (%) cell, or the mark is applied to the wrong rows.
assert _T1_MEAN_SD.match("58.3 (21.8)") and not _T1_MEDIAN_IQR.match("58.3 (21.8)")
assert _T1_MEDIAN_IQR.match("132 (114-151)") and not _T1_MEAN_SD.match("132 (114-151)")
assert _T1_MEDIAN_IQR.match("0 (0-1)") and not _T1_MEAN_SD.match("0 (0-1)")
assert not _T1_MEAN_SD.match("73,302 (30.6%)")
assert not _T1_MEDIAN_IQR.match("73,302 (30.6%)")
assert not _T1_MEAN_SD.match("608 (0.3%)") and not _T1_MEDIAN_IQR.match("608 (0.3%)")

T1_OUTCOME_KEEP = (
    "Major hemorrhage event \u2014 primary composite (Panel A stratifier)",
    "Patient-state composite (hemorrhagic death or substantial transfusion)",
    "Hemorrhage-control procedure, surgical or angiographic (composite component)",
    "Hemorrhagic death \u2014 in-hospital, head AIS < 4, within 48 h (composite component)",
    "Substantial transfusion, \u2265 1000 mL or \u2265 4 U PRBC within 4 h (composite component)",
)


def table_1(art: dict) -> list[str]:
    print("\n" + "=" * 72)
    print("TABLE 1 -- baseline characteristics")
    print("=" * 72)

    t1 = art["t1"]
    qc = art["qc"]
    n_cohort = int(qc["n_primary"])
    n_event = int(qc["primary_composite"]["major_hemorrhage_event_n"])
    n_noevent = n_cohort - n_event

    drop_cols = ["Young", "Elderly_noAC", "Elderly_AC", "SMD_strata", "P_strata"]
    keep_cols = ["Block", "Variable", "Overall", "NoEvent", "Event", "SMD_event", "P_event"]
    assert all(c in t1.columns for c in keep_cols + drop_cols), \
        "table1_baseline.csv schema changed; index by column name (CONTRACTS 7.1)"

    sub = t1[keep_cols].copy()
    n_out_before = int((t1["Block"] == "Outcomes").sum())
    mask = (sub["Block"] != "Outcomes") | (sub["Variable"].isin(T1_OUTCOME_KEEP))
    dropped = sub[~mask]["Variable"].tolist()
    sub = sub[mask].copy()
    n_out_after = int((sub["Block"] == "Outcomes").sum())

    gate("T1-a",
         not any(c in sub.columns for c in drop_cols),
         f"columns emitted = {list(sub.columns)}; the Age\u00d7anticoagulation "
         f"stratum columns {drop_cols} are routed to the Supplement")
    gate("T1-b",
         n_out_after == len(T1_OUTCOME_KEEP) and len(dropped) == n_out_before - n_out_after
         and len(dropped) > 0,
         f"{len(dropped)} outcome-definition crosswalk rows routed to the "
         f"Supplement; {n_out_after} outcome rows kept (primary composite, its "
         f"three components, and the co-primary patient-state composite)")

    columns = ["Characteristic",
               f"Overall (n = {n_(n_cohort)})",
               f"No major hemorrhage event (n = {n_(n_noevent)})",
               f"Major hemorrhage event (n = {n_(n_event)})",
               "SMD", "P"]
    rows, merged = [], set()
    for block in ["Demographics", "Physiology at arrival", "Injury characteristics",
                  "Comorbidities", "Outcomes"]:
        blk = sub[sub["Block"] == block]
        if len(blk) == 0:
            continue
        merged.add(len(rows))
        rows.append([block, "", "", "", "", ""])
        for _, r in blk.iterrows():
            def cell(v):
                return "" if (pd.isna(v) or str(v).strip().lower() == "nan") else str(v)
            label = _sanitize_label(r["Variable"]).replace(
                " \u2014 primary composite (Panel A stratifier)",
                " (primary composite)")
            overall = cell(r["Overall"])
            # A mean (SD) row is marked as one, because footnote 1 says a row
            # that is not median (IQR) will be.
            if _T1_MEAN_SD.match(overall):
                label += T1_DISPERSION_MARK
            rows.append([f"    {label}",
                         overall, cell(r["NoEvent"]), cell(r["Event"]),
                         cell(r["SMD_event"]), cell(r["P_event"])])

    footnotes = [
        "Continuous variables are median (interquartile range) unless marked as "
        "mean (standard deviation); categorical variables are n (%).",
        "SMD = standardised mean difference between patients with and without a "
        "major hemorrhage event; a value above 0.10 marks a difference large "
        "enough to matter clinically, independent of sample size.",
        "P values are chi-squared (categorical) or Kruskal-Wallis (continuous). "
        "With 239,736 patients almost any difference reaches statistical "
        "significance, so the SMD, not P, should carry the reader's judgement.",
        "Major hemorrhage event (primary composite) = a hemorrhage-control "
        "procedure (surgical or angiographic), hemorrhagic death (in-hospital, "
        "head AIS < 4, within 48 h of arrival), or substantial transfusion "
        "(\u2265 1000 mL or \u2265 4 units of packed red cells within 4 h).",
        "Pelvic-ring instability is reported here as a baseline characteristic "
        "and is retained in every adjusted analysis, but it is deliberately not "
        "a criterion on the score card: it is read off a computed-tomography "
        "scan and so is not available at the moment the score is meant to be "
        "used.",
        "The age \u00d7 anticoagulation stratum columns and the full "
        "outcome-definition crosswalk appear in the Supplement.",
        "Source: v3/tables/table1_baseline.csv; denominators from "
        "v3/logs/qc_manifest.json.",
    ]

    # T1-c: the dispersion convention is re-derived from the rendered cells, not
    # from a hard-coded row list, so it fails if the mark is dropped, if it lands
    # on a median (IQR) row, or if footnote 1 stops promising it.  Both shape
    # classes must be non-empty, so the gate cannot pass by finding nothing.
    marked = sorted(r[0].strip() for r in rows if T1_DISPERSION_MARK in r[0])
    mean_sd = sorted(r[0].strip() for r in rows if _T1_MEAN_SD.match(r[1]))
    med_iqr = sorted(r[0].strip() for r in rows if _T1_MEDIAN_IQR.match(r[1]))
    promised = any("unless marked as mean (standard deviation)" in f
                   for f in footnotes)
    med_iqr_marked = [lab for lab in med_iqr if T1_DISPERSION_MARK in lab]
    gate("T1-c",
         bool(mean_sd) and bool(med_iqr) and marked == mean_sd
         and not med_iqr_marked and promised,
         # evidence, not a restatement of the verdict: every clause below is a
         # measured count, so a FAIL line says which clause failed
         f"mean (SD) rows={len(mean_sd)} {mean_sd}; marked={len(marked)} "
         f"{marked}; median (IQR) rows={len(med_iqr)}, of which marked="
         f"{len(med_iqr_marked)}; footnote 1 promises the "
         f"'{T1_DISPERSION_MARK.strip(', ')}' mark={promised}")

    return write_table("Table 1", columns, rows, footnotes,
                       "Baseline characteristics of the analytic cohort, overall "
                       "and by major hemorrhage event", merged_rows=merged)


# =============================================================================
# TABLE 2 -- the arrival score card
# =============================================================================
def table_2(art: dict) -> list[str]:
    print("\n" + "=" * 72)
    print("TABLE 2 -- the arrival score card")
    print("=" * 72)

    pts = art["t3points"]
    card = art["card"]
    refit = art["refit"]
    tiers = art["tiers"]

    assert {"Variable", "\u03b2", "OR", "95% CI lower", "95% CI upper", "Points",
            "Role"}.issubset(set(pts.columns)), \
        "table3_points.csv schema changed; index by column name (CONTRACTS 7.1)"

    # ---- gate T2-1: no unstable_ring row, whatever the artifact says ---------
    src_has_ring = bool((pts["Variable"].astype(str).str.strip()
                         .isin(CARD_EXCLUDED_VARS)).any())
    card_has_ring = any(str(v.get("variable", "")).strip() in CARD_EXCLUDED_VARS
                        for v in card.get("variables", []))

    point_rows_v3 = pts[(pts["Role"].astype(str).str.strip().str.lower() == ROLE_POINT)
                        & (~pts["Variable"].astype(str).str.strip().isin(CARD_EXCLUDED_VARS))]
    point_rows_v3 = point_rows_v3.copy()
    point_rows_v3["_pts"] = point_rows_v3["Points"].astype(float).round().astype(int)
    point_rows_v3 = point_rows_v3.sort_values("_pts", ascending=False)
    # v3 row order (sort is stable on the file order) minus the dropped variable.
    point_rows = point_rows_v3[~point_rows_v3["Variable"].astype(str).str.strip()
                               .isin(CARD_DROPPED_VARS)]

    ov_labels = refit.get("override_labels") or {}
    ov_rule = _sanitize_label(refit.get("override_rule_text") or OVERRIDE_RULE_TEXT)

    columns = ["Criterion", "\u03b2", "Odds ratio (95% CI)", "Points"]
    rows, merged = [], set()
    for _, r in point_rows.iterrows():
        var = str(r["Variable"]).strip()
        rows.append([
            _var_label(var, ov_labels),
            f"{float(r['\u03b2']):+.4f}",
            f"{float(r['OR']):.3f} ({float(r['95% CI lower']):.3f}\u2013"
            f"{float(r['95% CI upper']):.3f})",
            f"{int(r['_pts']):+d}",
        ])
    n_point_rows = len(rows)

    banner_idx = len(rows)
    merged.add(banner_idx)
    rows.append([ov_rule, "", "", ""])

    ov = card["override"]
    ov_n, ov_ev = int(ov["n"]), int(ov["events"])
    ov_or = float(ov["or"])

    # ---- gate T2-1 (emitted) -------------------------------------------------
    emitted_text = "\n".join(" | ".join(str(c) for c in r) for r in rows)
    ring_in_output = bool(re.search(r"(?i)unstable[_ ]?ring", emitted_text))
    gate("T2-1",
         (not ring_in_output) and (not src_has_ring) and (not card_has_ring),
         f"no unstable_ring row: absent from table3_points.csv ({not src_has_ring}), "
         f"absent from score_card.json[variables] ({not card_has_ring}), and no "
         f"'unstable ring' string in any emitted Table 2 cell")

    # ---- gate T2-2: the override banner --------------------------------------
    banner = rows[banner_idx]
    exact = (banner[0] == OVERRIDE_RULE_TEXT)
    no_points = (banner[1] == "" and banner[2] == "" and banner[3] == "")
    is_merged = banner_idx in merged
    low = banner[0].lower()
    no_bad_framing = not any(bad in low for bad in FORBIDDEN_OVERRIDE_FRAMINGS)
    gate("T2-2", exact and no_points and is_merged and no_bad_framing,
         f"banner text matches the contract string verbatim ({exact}); "
         f"full-width merged row ({is_merged}); every value cell empty, so the "
         f"banner carries no point value ({no_points}); no 'SBP = 0' or "
         f"missingness framing ({no_bad_framing})")

    # ---- gate T2-3: cc_anticoag is gone from the card (v4, D5) ---------------
    ac = pts[pts["Variable"].astype(str).str.strip() == "cc_anticoag"]
    assert len(ac) == 1, ("cc_anticoag must still exist in table3_points.csv "
                          "(its null OR is quoted in the Results text)")
    ac = ac.iloc[0]
    assert np.isfinite(float(ac["OR"])) and int(round(float(ac["Points"]))) == 0
    ac_label = _var_label("cc_anticoag", ov_labels)
    ac_in_rows = [r for r in rows if r[0] == ac_label]
    point_vals = [r[3] for r in rows[:n_point_rows]]
    no_zero = all(int(str(v).replace(MINUS, "-")) != 0 for v in point_vals)
    ac_in_text = bool(re.search(r"(?i)anticoag", emitted_text))
    gate("T2-3",
         len(ac_in_rows) == 0 and not ac_in_text and n_point_rows == 13
         and banner_idx == 13 and no_zero,
         f"cc_anticoag (\"{ac_label}\") absent from the rendered rows ({len(ac_in_rows) == 0}) "
         f"and from every row cell ({not ac_in_text}); {n_point_rows} point rows + 1 override banner; "
         f"no remaining point value is 0 ({no_zero}); the null estimate "
         f"(\u03b2 {float(ac['\u03b2']):+.4f}, OR {float(ac['OR']):.3f}) is still in "
         f"table3_points.csv for the Results text")

    # ---- gate T2-4 -----------------------------------------------------------
    gate("T2-4",
         n_point_rows == 13 and int(card["n_point_variables"]) - len(CARD_DROPPED_VARS) == 13
         and len(point_rows_v3) == int(card["n_point_variables"]),
         f"{n_point_rows} point rows rendered; score_card.json n_point_variables "
         f"= {int(card['n_point_variables'])} minus {len(CARD_DROPPED_VARS)} dropped zero-point row")

    # ---- gate T2-5: surviving rows are the v3 values, in v3 order ------------
    cvars = {str(v["variable"]).strip(): v for v in card["variables"]}
    exp_order = [str(v).strip() for v in point_rows_v3["Variable"]
                 if str(v).strip() not in CARD_DROPPED_VARS]
    got_order = [str(v).strip() for v in point_rows["Variable"]]
    vals_ok = True
    for (_, r), cell in zip(point_rows.iterrows(), rows[:n_point_rows]):
        cv = cvars[str(r["Variable"]).strip()]
        vals_ok &= (int(cv["points"]) == int(r["_pts"])
                    and round(float(cv["beta"]), 4) == round(float(r["\u03b2"]), 4)
                    and round(float(cv["or"]), 3) == round(float(r["OR"]), 3)
                    and cell[1] == f"{float(cv['beta']):+.4f}"
                    and cell[3] == f"{int(cv['points']):+d}")
    gate("T2-5", exp_order == got_order and len(got_order) == 13 and vals_ok,
         f"13 rendered rows follow the v3 order minus cc_anticoag ({exp_order == got_order}); "
         f"each row's \u03b2, OR, CI and points match table3_points.csv and "
         f"score_card.json[variables] ({vals_ok})")

    rng = card["score_range"]
    theo = card["score_range_theoretical"]
    tier_txt = []
    for _, t in tiers[tiers["row_type"] == "score_tier"].iterrows():
        _b = lambda v: ("0" if int(v) == 0 else f"{int(v):+d}")
        tier_txt.append(f"{str(t['tier']).replace('_', '-')} "
                        f"(score {_b(t['score_min'])} to {_b(t['score_max'])}): "
                        f"n = {n_(t['n'])}, {n_(t['composite_events'])} events "
                        f"({100 * float(t['composite_rate']):.1f}%)")

    footnotes = [
        "The score is the sum of the points for every criterion the patient meets "
        "on arrival; there is no imaging, laboratory or operative information in it.",
        f"Points are round(\u03b2 / |\u03b2 of the anchor variable|), the anchor "
        f"being {_var_label(str(refit['anchor_min_positive_var']), ov_labels)} "
        f"(\u03b2 = {float(refit['anchor_min_positive_beta']):.4f}); a variable "
        f"whose |\u03b2| is below 0.15 is allowed to round to zero.",
        f"Observed score range in the held-out test set (n = "
        f"{n_(rng['n_test'])}): {int(rng['min']):+d} to {int(rng['max']):+d}. "
        f"The theoretical range implied by summing the points column is "
        f"{int(theo['min_unconstrained']):+d} to "
        f"{int(theo['max_unconstrained']):+d}; no patient attains it.",
        f"The automatic-escalation criterion sits outside the point scale: it is "
        f"never added to the sum. In the test set {n_(ov_n)} patients met it "
        f"({100.0 * ov_n / int(card['n_test']):.2f}% of {n_(card['n_test'])}), of "
        f"whom {n_(ov_ev)} had a major hemorrhage event "
        f"({100.0 * ov_ev / ov_n:.1f}%); adjusted odds ratio {ov_or:.1f}.",
        "Risk tiers on the point scale \u2014 " + "; ".join(tier_txt) + ".",
        "Source: v3/tables/table3_points.csv, v3/derived/score_card.json, "
        "v3/derived/refit.json, v3/derived/risk_tiers.csv.",
    ]
    return write_table("Table 2", columns, rows, footnotes,
                       f"The {SCORE_NAME} card: 13 point-scoring criteria "
                       "and one automatic-escalation criterion", merged_rows=merged)


# =============================================================================
# TABLE 3 -- discrimination AND matched-workload utility, two blocks
# =============================================================================
def table_3(art: dict) -> list[str]:
    print("\n" + "=" * 72)
    print("TABLE 3 -- discrimination and matched-workload utility")
    print("=" * 72)

    disc = art["disc"]
    t4b = art["t4b"]
    delong = art["delong"]
    w = art["t5d"]
    citl = art["citl"]["composite"]

    def drow(score, auroc_type):
        sub = disc[(disc["Outcome"] == "major_hemorrhage_event")
                   & (disc["Score"] == score)
                   & (disc["AUROC_type"] == auroc_type)]
        assert len(sub) == 1, f"expected 1 row for {score}/{auroc_type}"
        return sub.iloc[0]

    r_score = drow("pelvic_score", "integer_bedside")
    r_dep = drow("pelvic_score_or_override", "integer_bedside_override_promoted")
    r_logit = drow("pelvic_logit", "continuous_frozen_logit")
    r_si = drow("SI-alone", "comparator")
    r_rsig = drow("rSIG", "comparator")
    r_full = drow("full_information_ISS_AIS", "full_information_benchmark")
    t_row = t4b[t4b["outcome"] == "major_hemorrhage_event"].iloc[0]

    def dl(score1, score2):
        sub = delong[(delong["Outcome"] == "major_hemorrhage_event")
                     & (delong["Score_1"] == score1) & (delong["Score_2"] == score2)]
        assert len(sub) == 1, f"expected 1 DeLong row for {score1} vs {score2}"
        return sub.iloc[0]

    d_si = dl("pelvic_score", "SI-alone")
    d_rsig = dl("pelvic_score", "rSIG")

    # FIX 2 -- the DeLong P printed in Table 3 is DERIVED from the artifact, not
    # typed. table4_delong.csv stores DeLong_P == 0.0 (underflow), so a literal
    # 0.0 must never reach prose; anything >= 0.001 prints at three decimals.
    def dl_p(row, label):
        assert "DeLong_P" in row.index, (
            f"table4_delong.csv has no 'DeLong_P' column -- the {label} P value "
            f"cannot be derived and must not be typed")
        p = float(row["DeLong_P"])
        assert np.isfinite(p) and 0.0 <= p <= 1.0, (
            f"DeLong_P for {label} is not a probability: {p!r}")
        return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"

    p_si = dl_p(d_si, "score vs shock index")
    p_rsig = dl_p(d_rsig, "score vs rSIG")

    HEAD = (f"{SCORE_NAME}, integer card: internal test set "
            "(the score's AUROC)")
    columns = ["Block", "Row", "N", "Events", "Result", "Detail"]
    rows, merged = [], set()

    merged.add(len(rows))
    rows.append(["Block A. Discrimination for major hemorrhage "
                 "(area under the receiver-operating-characteristic curve)",
                 "", "", "", "", ""])
    rows.append(["A", HEAD, n_(r_score["N_used"]), n_(r_score["N_events"]),
                 auc_(r_score["AUROC"], r_score["CI_Low"], r_score["CI_High"]),
                 "Held-out internal validation set; the integer bedside card, which is the "
                 "instrument a clinician actually uses. This is the row quotable "
                 "as the score's discrimination."])
    rows.append(["A", f"{SCORE_NAME}, integer card re-estimated in "
                 "2019\u20132022: temporal validation, 2023\u20132024",
                 n_(t_row["N_valid"]), n_(t_row["events"]),
                 auc_(t_row["AUROC_integer"], t_row["CI_low_integer"],
                      t_row["CI_high_integer"]),
                 f"Re-estimated on 2019\u20132022 admissions (n = "
                 f"{n_(t_row['N_derive'])}, {n_(t_row['events_derive'])} events). "
                 f"Calibration slope {float(citl['calibration_slope']):.4f}; "
                 f"calibration-in-the-large offset "
                 f"{float(citl['citl_offset_intercept']):+.4f}."])
    rows.append(["A", "Shock index alone", n_(r_si["N_used"]), n_(r_si["N_events"]),
                 auc_(r_si["AUROC"], r_si["CI_Low"], r_si["CI_High"]),
                 "Pre-specified comparator, on its own denominator: the patients in "
                 "whom a shock index is computable."])
    rows.append(["A", "rSIG (systolic pressure \u00f7 heart rate \u00d7 Glasgow "
                 "Coma Scale)", n_(r_rsig["N_used"]), n_(r_rsig["N_events"]),
                 auc_(r_rsig["AUROC"], r_rsig["CI_Low"], r_rsig["CI_High"]),
                 "Pre-specified comparator, on its own denominator: the patients in "
                 "whom all three of pressure, heart rate and Glasgow Coma Scale are "
                 "recorded."])
    rows.append(["A", "Full-information reference (Injury Severity Score and "
                 "Abbreviated Injury Scale)", n_(r_full["N_used"]),
                 n_(r_full["N_events"]),
                 auc_(r_full["AUROC"], r_full["CI_Low"], r_full["CI_High"]),
                 "Ceiling benchmark only. These codes are assigned by a registrar "
                 "after discharge and are not available on arrival, so this is not "
                 "a deployable alternative."])
    rows.append(["A", "Continuous frozen logistic model: NOT the bedside "
                 "instrument", n_(r_logit["N_used"]), n_(r_logit["N_events"]),
                 auc_(r_logit["AUROC"], r_logit["CI_Low"], r_logit["CI_High"]),
                 "The underlying continuous model before it is rounded to whole "
                 "points. Reported for completeness; it is not the score and is "
                 "not quotable as the score's AUROC."])
    rows.append(["A", "Deployed rule (score \u2265 +5 OR automatic escalation): "
                 "diagnostic, NOT the score's AUROC",
                 n_(r_dep["N_used"]), n_(r_dep["N_events"]),
                 auc_(r_dep["AUROC"], r_dep["CI_Low"], r_dep["CI_High"]),
                 "A diagnostic for the rule as deployed, with automatic-escalation "
                 "patients promoted above the top of the point scale. Reported for "
                 "completeness; it is not quotable as the score's AUROC."])
    rows.append(["A", "Score versus shock index (DeLong paired comparison)",
                 n_(d_si["N_paired"]), n_(r_si["N_events"]),
                 f"\u0394AUROC {float(d_si['Delta_AUC']):+.4f} "
                 f"({float(d_si['CI_Low_Delta']):.4f}\u2013"
                 f"{float(d_si['CI_High_Delta']):.4f})",
                 f"{p_si}. On this paired subset the score reads "
                 f"{float(d_si['AUROC_1']):.4f} and the shock index "
                 f"{float(d_si['AUROC_2']):.4f}; the paired subset is smaller than "
                 f"the full test set, so these are not the headline values."])
    rows.append(["A", "Score versus rSIG (DeLong paired comparison)",
                 n_(d_rsig["N_paired"]), n_(r_rsig["N_events"]),
                 f"\u0394AUROC {float(d_rsig['Delta_AUC']):+.4f} "
                 f"({float(d_rsig['CI_Low_Delta']):.4f}\u2013"
                 f"{float(d_rsig['CI_High_Delta']):.4f})",
                 f"{p_rsig}. On this paired subset the score reads "
                 f"{float(d_rsig['AUROC_1']):.4f} and rSIG "
                 f"{float(d_rsig['AUROC_2']):.4f}."])

    # ---- gate T3-a -----------------------------------------------------------
    head_rows = [r for r in rows if "(the score's AUROC)" in str(r[1])]
    logit_row = [r for r in rows if "Continuous frozen logistic model" in str(r[1])]
    dep_row = [r for r in rows if "Deployed rule" in str(r[1])]
    ok_a = (len(head_rows) == 1
            and head_rows[0][4].startswith("0.8823")
            and "0.8787" in head_rows[0][4] and "0.8860" in head_rows[0][4]
            and len(logit_row) == 1 and "NOT the bedside instrument" in logit_row[0][1]
            and logit_row[0][4].startswith("0.8861")
            and len(dep_row) == 1 and "NOT the score's AUROC" in dep_row[0][1]
            and dep_row[0][4].startswith("0.8838"))
    gate("T3-a", ok_a,
         f"headline row = pelvic_score {head_rows[0][4]}; the continuous logit "
         f"({logit_row[0][4].split(' ')[0]}) and the deployed rule "
         f"({dep_row[0][4].split(' ')[0]}) each carry an explicit "
         f"not-the-score's-AUROC label")

    # ---- Block B -------------------------------------------------------------
    merged.add(len(rows))
    rows.append(["Block B. Clinical utility at matched workload "
                 "(each instrument's cut-point set so that it flags the same "
                 "share of patients)", "", "", "", "", ""])

    b_checks = []
    for stratum, disp in (("overall", "Overall matched set"),
                          ("elderly_ge65", "Age \u2265 65 y, matched set")):
        s, r = _workload_rows(w, stratum)
        n_a, e_a = n_(s["N_analysis"]), n_(s["N_events"])
        for who, row_obj, col in ((f"{disp}: {SCORE_NAME}, "
                                   f"{_score_cutoff(s['Cutoff_Label'])}", s, "score"),
                                  (f"{disp}: shock index, "
                                   f"{_typo(str(r['Cutoff_Label']))} "
                                   f"(workload-matched reference)", r, "ref")):
            rows.append(["B", who, n_a, e_a,
                         f"{n_(row_obj['N_flagged'])} flagged "
                         f"({float(row_obj['Pct_flagged_actual']):.4f}%)",
                         f"Events caught {n_(row_obj['Events_caught_TP'])}; false "
                         f"positives {n_(row_obj['False_positives_FP'])}; positive "
                         f"predictive value "
                         f"{100 * float(row_obj['PPV']):.3f}%; sensitivity "
                         f"{100 * float(row_obj['Sensitivity']):.3f}%; "
                         f"{float(row_obj['Number_needed_to_escalate']):.2f} "
                         f"escalations per event caught."])
        assert int(s["Delta_comparable"]) == 1, \
            f"Delta_comparable != 1 for {stratum}: no delta may be shown"
        d_ev = float(s["Delta_events_caught"])
        d_fp = float(s["Delta_false_positives"])
        rows.append(["B", f"{disp}: difference, score minus shock index, "
                          f"at the same workload", n_a, e_a,
                     f"{d_ev:+,.0f} events caught; {d_fp:+,.0f} false positives",
                     "Taken from the Delta_events_caught and "
                     "Delta_false_positives columns of the matched-workload "
                     "table; the two instruments' event counts are never "
                     "subtracted from one another by hand."])
        b_checks.append((stratum, int(s["N_analysis"]), int(s["N_events"]),
                         int(s["N_flagged"]), float(s["Pct_flagged_actual"]),
                         int(s["Events_caught_TP"]), int(s["False_positives_FP"]),
                         100 * float(s["PPV"]), int(r["N_flagged"]),
                         float(r["Pct_flagged_actual"]), int(r["Events_caught_TP"]),
                         int(r["False_positives_FP"]), 100 * float(r["PPV"]),
                         d_ev, d_fp))

    ov = b_checks[0]
    ok_b = (ov[1] == 75072 and ov[2] == 8358 and ov[3] == 11729
            and round(ov[4], 4) == 15.6237 and ov[5] == 5442 and ov[6] == 6287
            and round(ov[7], 3) == 46.398 and ov[8] == 11735
            and round(ov[9], 4) == 15.6317 and ov[10] == 4764 and ov[11] == 6971
            and round(ov[12], 3) == 40.597 and ov[13] == 678.0 and ov[14] == -684.0)
    gate("T3-b", ok_b,
         f"overall: n = {n_(ov[1])} with {n_(ov[2])} events; score \u2265 +5 flags "
         f"{n_(ov[3])} ({ov[4]:.4f}%), TP {n_(ov[5])}, FP {n_(ov[6])}, PPV "
         f"{ov[7]:.3f}%; shock index \u2265 0.9463 flags {n_(ov[8])} "
         f"({ov[9]:.4f}%), TP {n_(ov[10])}, FP {n_(ov[11])}, PPV {ov[12]:.3f}%; "
         f"\u0394 {ov[13]:+.0f} events / {ov[14]:+.0f} false positives; all "
         f"read verbatim from table5d_matched_workload.csv")

    eld = b_checks[1]
    print(f"    elderly: n = {n_(eld[1])} with {n_(eld[2])} events; score flags "
          f"{n_(eld[3])} ({eld[4]:.4f}%), TP {n_(eld[5])}, FP {n_(eld[6])}, PPV "
          f"{eld[7]:.3f}%; shock index flags {n_(eld[8])} ({eld[9]:.4f}%), TP "
          f"{n_(eld[10])}, FP {n_(eld[11])}, PPV {eld[12]:.3f}%; \u0394 "
          f"{eld[13]:+.0f} events / {eld[14]:+.0f} false positives")

    footnotes = [
        "AUROC (area under the receiver-operating-characteristic curve) is the "
        "probability that a randomly chosen patient who bleeds scores higher than "
        "a randomly chosen patient who does not; 0.5 is a coin toss and 1.0 is "
        "perfect separation.",
        "Every AUROC quoted for the score is the integer bedside card. The "
        "continuous model and the deployed rule are shown separately and are "
        "labelled as such.",
        "Each denominator is stated on its own row and none is pooled with "
        "another: the score is defined for every patient, whereas the shock index "
        "and rSIG require vital signs that are not always recorded.",
        "Calibration-in-the-large is how far the average predicted risk sits from "
        "the observed event rate; the calibration slope is 1.0 when predicted risks "
        "are neither too extreme nor too flat.",
        "DeLong's test compares two AUROCs measured on the same patients, so the "
        "paired denominator is the smaller of the two instruments'.",
        "In Block B the two instruments are compared at the same workload: the "
        "cut-point for each is chosen so that both flag the same share of "
        "patients, so the comparison is yield at equal cost rather than yield at "
        "any cost. Differences are reported only where the matched-workload table "
        "marks the pair comparable.",
        "Source: v3/tables/table4_discrimination.csv, table4b_temporal.csv, "
        "table4_delong.csv, table5d_matched_workload.csv; "
        "v3/derived/citl_slope.json.",
    ]
    return write_table("Table 3", columns, rows, footnotes,
                       "Discrimination and matched-workload clinical utility of "
                       f"the {SCORE_NAME}", merged_rows=merged)


# =============================================================================
# FIGURE LEGENDS
# =============================================================================
def figure_legends(art: dict) -> list[str]:
    print("\n" + "=" * 72)
    print("FIGURE LEGENDS")
    print("=" * 72)

    qc, card, citl = art["qc"], art["card"], art["citl"]["composite"]
    disc, t4b = art["disc"], art["t4b"]
    reclass = art["reclass"]
    w = art["t5d"]
    excl = art["cohort_exclusions"]
    split = art["split"].set_index("split")

    def drow(score, auroc_type):
        sub = disc[(disc["Outcome"] == "major_hemorrhage_event")
                   & (disc["Score"] == score) & (disc["AUROC_type"] == auroc_type)]
        return sub.iloc[0]

    r_score = drow("pelvic_score", "integer_bedside")
    r_si = drow("SI-alone", "comparator")
    r_rsig = drow("rSIG", "comparator")
    t_row = t4b[t4b["outcome"] == "major_hemorrhage_event"].iloc[0]
    s_all, r_all = _workload_rows(w, "overall")
    s_eld, r_eld = _workload_rows(w, "elderly_ge65")

    counts = [int(v) for v in excl.values()]
    n_start, n_end = counts[0], counts[-1]
    cal = art["calibration"]
    co = cal[cal["score"] == "pelvic_score"]
    n_cal = int(co["n"].iloc[0])
    n_eldcal = int(reclass["elderly_calibration"]["n"])

    path = os.path.join(V4_FIGURES, "figure_legends.md")
    _chunks: list[str] = []
    _raw_chunks: list[str] = []

    def fw(s: str) -> None:
        """Append one legend chunk, normalising sign hyphens to U+2212.

        Every negative number in this file therefore carries the same glyph as
        `RESULTS.md` and the ledger. `_minus` is delimiter-anchored, so the
        `ICD-10-CM` code label and the `2019` / `2024` year spans are untouched.
        """
        _raw_chunks.append(s)
        _chunks.append(_minus(s))

    if True:
        fw("# Figure legends\n\n")

        fw("## Figure 1. Assembly of the analytic cohort.\n\n")
        fw(
            f"Of {n_(n_start)} patient encounters recorded in the National Trauma "
            f"Data Bank between 2019 and 2024, {n_(n_end)} formed the analytic "
            f"cohort. Patients were excluded, in order, for age under 18 years "
            f"({n_(counts[0] - counts[1])}), absence of a pelvic-ring fracture "
            f"diagnosis ({n_(counts[1] - counts[2])}; ICD-10-CM S32.1, .3, .5, .6, "
            f".81 or .82, identified from diagnosis codes rather than procedure "
            f"codes), a penetrating or unspecified mechanism "
            f"({n_(counts[2] - counts[3])}), transfer in from another facility "
            f"({n_(counts[3] - counts[4])}), prehospital cardiac arrest or an "
            f"unsurvivable injury ({n_(counts[4] - counts[5])}), and an "
            f"unascertainable hospital disposition ({n_(counts[5] - counts[6])}). "
            f"A major hemorrhage event occurred in "
            f"{n_(int(split.loc['full', 'N_events']))} patients "
            f"({100 * float(split.loc['full', 'event_rate']):.1f}%). The cohort was "
            f"split at random, stratified on the outcome, into a derivation set of "
            f"{n_(int(split.loc['train', 'N']))} patients "
            f"({n_(int(split.loc['train', 'N_events']))} events) and an internal "
            f"held-out test set of {n_(int(split.loc['test', 'N']))} "
            f"({n_(int(split.loc['test', 'N_events']))} events). For the temporal "
            f"validation (internal) the same cohort is re-partitioned by admission year into "
            f"{n_(int(citl['n_derive']))} patients admitted 2019\u20132022 and "
            f"{n_(int(citl['n_valid']))} admitted 2023\u20132024. Hemorrhage "
            f"control is a component of the outcome, not an eligibility criterion, "
            f"so no patient was excluded for failing to receive an intervention. "
            f"Every count in the diagram is reproduced from "
            f"`v3/derived/cohort_exclusions.json` and "
            f"`v3/tables/table_consort.csv`; the ladder closes at each rung and "
            f"between consecutive rungs without adjustment.\n\n")

        fw(f"## Figure 2. Discrimination and calibration of the "
                 f"{SCORE_NAME}.\n\n")
        fw(
            f"**(A)** Receiver-operating-characteristic curves for the primary "
            f"composite outcome. Both values quoted for the score are the integer "
            f"bedside card, not the underlying continuous model. In the internal "
            f"held-out test set (n = {n_(r_score['N_used'])}, "
            f"{n_(r_score['N_events'])} events) the card achieved an area under "
            f"the curve of "
            f"{auc_(r_score['AUROC'], r_score['CI_Low'], r_score['CI_High'])}; on "
            f"temporal validation (internal) in patients admitted 2023\u20132024 "
            f"(n = {n_(t_row['N_valid'])}, {n_(t_row['events'])} events), with the model "
            f"re-estimated in 2019\u20132022, it "
            f"achieved {auc_(t_row['AUROC_integer'], t_row['CI_low_integer'], t_row['CI_high_integer'])}. "
            f"The comparators are shown on their own denominators, which are "
            f"smaller because each requires vital signs that are not always "
            f"recorded: shock index alone, n = {n_(r_si['N_used'])}, "
            f"{auc_(r_si['AUROC'], r_si['CI_Low'], r_si['CI_High'])}; rSIG "
            f"(systolic pressure \u00f7 heart rate \u00d7 Glasgow Coma Scale), "
            f"n = {n_(r_rsig['N_used'])}, "
            f"{auc_(r_rsig['AUROC'], r_rsig['CI_Low'], r_rsig['CI_High'])}. "
            f"**(B)** Calibration, plotting observed event frequency against mean "
            f"predicted probability. Points lying on the dotted diagonal indicate "
            f"predictions that match what actually happened. The overall curve "
            f"covers the whole internal held-out test set (n = {n_(n_cal)}); because the card "
            f"produces whole-number scores, patients tie and the ten requested "
            f"risk deciles collapse into {len(co)} distinct groups. The second "
            f"curve covers patients aged 65 years or older (n = {n_(n_eldcal)}) in "
            f"ten deciles with 95% confidence intervals. Calibration-in-the-large "
            f"(the gap between mean predicted and observed risk) was "
            f"{signed_(reclass['elderly_calibration']['citl'])} in the elderly, "
            f"where the frozen model is carried into the subgroup unchanged. "
            f"The panel also prints "
            f"{signed_(co['citl'].iloc[0])} for the internal held-out test set as a whole; "
            f"that value is zero by construction, because the overall curve's "
            f"probabilities are re-fitted on the same set they are then judged "
            f"against, which forces the mean predicted risk to equal the observed "
            f"rate. It is an identity, not a measure of performance, and carries "
            f"no evidential weight; the main text reports calibration only where "
            f"the model is applied to data it was not fitted on. Source artifacts: "
            f"`roc_data.pkl`, `temporal_roc_data.pkl`, `calibration_data.csv`, "
            f"`reclass.json`.\n\n")

        fw(f"## Figure 3. Clinical utility of the {SCORE_NAME}.\n\n")
        fw(
            f"**(A)** Decision-curve analysis. Net benefit expresses the value of "
            f"a rule as the number of true escalations per 100 patients after "
            f"subtracting false escalations at the weight a clinician implicitly "
            f"assigns them by choosing a threshold probability. A curve above both "
            f"the treat-all and the treat-none lines means the rule is worth using "
            f"at that threshold. Curves are shown for the whole internal held-out test set "
            f"(score n = 79,912; shock index n = 77,739) and, dashed, for "
            f"patients aged 65 years or older (score n = 37,235; shock index "
            f"n = 35,944); the shaded band marks the pre-specified threshold window "
            f"of 0.05 to 0.20 and the dashed vertical line the rule-in threshold of "
            f"0.20. **(B)** Yield at matched workload. Cut-points are chosen so "
            f"that each instrument flags the same share of patients, so the two "
            f"rules cost the same to run and only their yield differs. In the "
            f"matched set (n = {n_(s_all['N_analysis'])} patients with "
            f"{n_(s_all['N_events'])} events, being those in whom the score, the "
            f"shock index and rSIG are all observable), a threshold of "
            f"{_score_cutoff(s_all['Cutoff_Label'])} flagged {n_(s_all['N_flagged'])} "
            f"patients ({float(s_all['Pct_flagged_actual']):.2f}%) and caught "
            f"{n_(s_all['Events_caught_TP'])} events at "
            f"{n_(s_all['False_positives_FP'])} false positives, against "
            f"{n_(r_all['Events_caught_TP'])} events at "
            f"{n_(r_all['False_positives_FP'])} false positives for "
            f"{_typo(str(r_all['Cutoff_Label']))} flagging "
            f"{n_(r_all['N_flagged'])} ({float(r_all['Pct_flagged_actual']):.2f}%) "
            f"is a difference of {float(s_all['Delta_events_caught']):+,.0f} "
            f"events caught and {float(s_all['Delta_false_positives']):+,.0f} false "
            f"positives. Among patients aged 65 years or older "
            f"(n = {n_(s_eld['N_analysis'])}, {n_(s_eld['N_events'])} events) the "
            f"same contrast was "
            f"{float(s_eld['Delta_events_caught']):+,.0f} events and "
            f"{float(s_eld['Delta_false_positives']):+,.0f} false positives. "
            f"Differences are the matched-workload table's own delta columns and "
            f"are shown only where that table marks the pair comparable. Source "
            f"artifacts: `dca_data.csv`, `elderly_dca_data.csv`, "
            f"`table5d_matched_workload.csv`.\n")

    body = "".join(_chunks)
    # Proof that the U+2212 pass moved sign glyphs and nothing else.
    assert body.replace(MINUS, "-") == "".join(_raw_chunks).replace(MINUS, "-"), \
        "the legend minus-sign pass altered something other than a sign hyphen"
    with open(path, "w") as fh:
        fh.write(body)

    print(f"  wrote: figure_legends.md  ->  {V4_FIGURES}")
    return [path]


# =============================================================================
# MAIN
# =============================================================================
def load_artifacts() -> dict:
    paths = {
        "cohort_exclusions": os.path.join(V3_DERIVED, "cohort_exclusions.json"),
        "consort": os.path.join(V3_TABLES, "table_consort.csv"),
        "split": os.path.join(V3_TABLES, "table_split_balance.csv"),
        "qc": os.path.join(V3_LOGS, "qc_manifest.json"),
        "roc": os.path.join(V3_DERIVED, "roc_data.pkl"),
        "temporal_roc": os.path.join(V3_DERIVED, "temporal_roc_data.pkl"),
        "calibration": os.path.join(V3_DERIVED, "calibration_data.csv"),
        "reclass": os.path.join(V3_DERIVED, "reclass.json"),
        "dca": os.path.join(V3_DERIVED, "dca_data.csv"),
        "edca": os.path.join(V3_DERIVED, "elderly_dca_data.csv"),
        "card": os.path.join(V3_DERIVED, "score_card.json"),
        "refit": os.path.join(V3_DERIVED, "refit.json"),
        "tiers": os.path.join(V3_DERIVED, "risk_tiers.csv"),
        "citl": os.path.join(V3_DERIVED, "citl_slope.json"),
        "t1": os.path.join(V3_TABLES, "table1_baseline.csv"),
        "t3points": os.path.join(V3_TABLES, "table3_points.csv"),
        "disc": os.path.join(V3_TABLES, "table4_discrimination.csv"),
        "t4b": os.path.join(V3_TABLES, "table4b_temporal.csv"),
        "delong": os.path.join(V3_TABLES, "table4_delong.csv"),
        "t5d": os.path.join(V3_TABLES, "table5d_matched_workload.csv"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    gate("IN-1", not missing,
         f"{len(paths)} required v3 artifacts present"
         if not missing else f"MISSING: {missing}")
    assert not missing, f"required artifacts missing: {missing}"

    art = {}
    for key, p in paths.items():
        if p.endswith(".json"):
            art[key] = _read_json(p)
        elif p.endswith(".pkl"):
            art[key] = _read_pickle(p)
        else:
            art[key] = _read_csv(p)
    return art


def check_tifs(fig_paths: list[str]) -> None:
    tifs = [p for p in fig_paths if p.endswith(".tif")]
    ok = len(tifs) == 3
    detail = []
    for p in tifs:
        with Image.open(p) as im:
            fmt, size, mode = im.format, im.size, im.mode
        good = (fmt == "TIFF")
        ok = ok and good
        detail.append(f"{os.path.basename(p)}: PIL format={fmt}, {size[0]}x{size[1]}px, "
                      f"mode={mode}, {os.path.getsize(p) / 1e6:.1f} MB "
                      f"[{'ok' if good else 'NOT TIFF'}]")
    for d in detail:
        print(f"    {d}")
    gate("TIF", ok, f"{len(tifs)} TIF files opened with PIL; all report format == TIFF")


def check_minus_signs(text_paths: list[str]) -> None:
    """GATE MIN -- one minus-sign convention across every emitted exhibit.

    Three-part, and each part can fail independently:
      (a) NO ASCII hyphen is used as a numeric sign anywhere (the defect);
      (b) at least one U+2212 sign IS present, so the gate cannot pass vacuously
          on a file that simply has no negative numbers;
      (c) a range hyphen SURVIVES, proving the pass is targeted rather than a
          blanket hyphen sweep -- `132 (114-151)` must still read as a range, or
          the sign-anchored score-vs-rSIG margin sweeps change meaning.
    """
    sign_hits, n_minus, n_range = [], 0, 0
    scanned = 0
    for p in text_paths:
        if not p.endswith((".csv", ".md")):
            continue
        text = open(p, "r", encoding="utf-8").read()
        scanned += 1
        for m in _SIGN_HYPHEN.finditer(text):
            lo = max(0, m.start() - 45)
            sign_hits.append(f"{os.path.basename(p)} :: ...{text[lo:m.end() + 12]}...")
        n_minus += len(re.findall(r"(?<![0-9A-Za-z])" + MINUS + r"(?=[0-9])", text))
        n_range += len(re.findall(r"(?<=[0-9])-(?=[0-9])", text))
    for h in sign_hits[:8]:
        print(f"    HIT: {h}")
    # 3 Table .csv + 3 Table .md + figure_legends.md
    gate("MIN",
         (not sign_hits) and n_minus > 0 and n_range > 0 and scanned == 7,
         f"{len(sign_hits)} ASCII sign hyphens across {scanned} emitted text files; "
         f"{n_minus} U+2212 signs present (non-vacuous); {n_range} digit-to-digit "
         f"range hyphens correctly left as ASCII")


def check_banned(all_paths: list[str]) -> None:
    hits = []
    scanned = 0
    for p in all_paths:
        if p.endswith((".png", ".tif")):
            continue
        text = open(p, "r").read()
        scanned += 1
        for name, pat in BANNED_PATTERNS:
            m = re.search(pat, text)
            if m:
                hits.append(f"{os.path.basename(p)} :: {name} :: '{m.group(0)}'")
    for h in hits:
        print(f"    HIT: {h}")
    gate("BAN", not hits,
         f"{len(hits)} banned constants across {scanned} emitted text files "
         f"({len(BANNED_PATTERNS)} patterns each)")


def check_outputs(all_paths: list[str]) -> None:
    expected = [
        os.path.join(V4_FIGURES, f) for f in
        ("figure_1.png", "figure_1.tif", "figure_2.png", "figure_2.tif",
         "figure_3.png", "figure_3.tif", "figure_legends.md")
    ] + [
        os.path.join(V4_TABLES, f"Table {i}.{ext}")
        for i in (1, 2, 3) for ext in ("csv", "md")
    ]
    bad = [p for p in expected
           if not os.path.exists(p) or os.path.getsize(p) == 0]
    gate("OUT", not bad,
         f"{len(expected)} declared output files present and non-empty"
         if not bad else f"MISSING OR EMPTY: {bad}")


def main() -> int:
    print("=" * 72)
    print("render_exhibits.py -- v4 exhibits (plan section B4)")
    print(f"  seed {RANDOM_SEED} | {DPI} dpi | Okabe-Ito | backend "
          f"{matplotlib.get_backend()}")
    print(f"  reading  : {config.REPO_ROOT} (derived/, tables/, logs/)")
    print(f"  writing  : {V4_FIGURES}")
    print(f"             {V4_TABLES}")
    print("=" * 72)

    os.makedirs(V4_FIGURES, exist_ok=True)
    os.makedirs(V4_TABLES, exist_ok=True)

    art = load_artifacts()

    fig_paths = []
    fig_paths += figure_1(art)
    fig_paths += figure_2(art)
    fig_paths += figure_3(art)

    tbl_paths = []
    tbl_paths += table_1(art)
    tbl_paths += table_2(art)
    tbl_paths += table_3(art)

    leg_paths = figure_legends(art)

    print("\n" + "=" * 72)
    print("FILE-LEVEL GATES")
    print("=" * 72)
    check_tifs(fig_paths)
    check_minus_signs(tbl_paths + leg_paths)
    check_banned(tbl_paths + leg_paths)
    check_outputs(fig_paths + tbl_paths + leg_paths)
    return 0


def summarise_and_exit(hard_error: bool) -> None:
    print("\n" + "=" * 72)
    print("GATE SUMMARY")
    print("=" * 72)
    failed = 0
    for gid, desc in GATES_DECLARED:
        if gid not in GATE_RESULTS:
            print(f"GATE {gid}: NOT EVALUATED \u2014 FAIL  ({desc})")
            failed += 1
            continue
        ok, _ = GATE_RESULTS[gid]
        if not ok:
            print(f"GATE {gid}: FAIL  ({desc})")
            failed += 1
    extra = [g for g in GATE_RESULTS if g not in {x[0] for x in GATES_DECLARED}]
    if extra:
        print(f"UNDECLARED GATES EVALUATED: {extra}")
        failed += len(extra)
    total = len(GATES_DECLARED)
    print(f"\n{total - failed} of {total} declared gates PASS; {failed} FAIL"
          + ("; an exception was raised during the run" if hard_error else ""))
    sys.exit(1 if (failed or hard_error) else 0)


if __name__ == "__main__":
    err = False
    try:
        main()
    except BaseException:  # reported, never swallowed -- the run still exits non-zero
        traceback.print_exc()
        err = True
    summarise_and_exit(err)
