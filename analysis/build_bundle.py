#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bundle.py  --  v4 results bundle for the Pelvic-Fracture Arrival Score
==============================================================================
Task T1 of the v4 build (plan section B1 + the C3 denominator ledger).

WHAT THIS IS
------------
The SINGLE SOURCE OF TRUTH for every number that may appear in the JTACS
manuscript, the JTACS 300-word abstract, or any v4 memo.  It reads ONLY
finished v3 artifacts, re-derives every quantity from them, and emits:

    derived/results_bundle.json     machine-readable {value, printed, accessor, label}
    logs/NUMBERS-LEDGER.md          human-readable, grouped by manuscript section

ANYTHING ABSENT FROM THE BUNDLE CANNOT APPEAR IN THE MANUSCRIPT.

USAGE
-----
    python3 build_bundle.py            # build both artifacts
    python3 build_bundle.py --verify   # re-read the emitted JSON, re-derive every
                                       # value, printed string AND printed_alt from
                                       # the v3 artifacts, assert equality, print a
                                       # per-check PASS/FAIL table

Both modes exit non-zero on any failure.

DISCIPLINE ENCODED HERE (each line is a bug this project has already shipped)
----------------------------------------------------------------------------
1.  House style (CONTRACTS section 1): module-level ABSOLUTE root constants, no
    argparse path arguments, no relative resolution, and NO imports from
    v3/code/ -- helpers are copy-adapted, not imported.
2.  Dependency manifest reused VERBATIM from v3/code/16_generate_manuscript.py
    lines 112-162 (21 tables, 10 figures, 13 derived, 3 logs).  Preflight fails
    loudly, naming every missing file.
3.  Every dataframe lookup is a boolean mask on NAMED columns.  Never `iloc`
    for a column, never a fixed offset (CONTRACTS section 7.1: several tables
    gained leading Outcome / Outcome_Role columns in Batch 2).  A mask that does
    not select exactly one row raises, printing the mask and the row count.
4.  Authority precedence, hard-coded nowhere:
        logs/qc_manifest.json -> derived/refit.json -> derived/score_card.json
        -> derived/reclass.json -> the eight named tables
        -> logs/d11_invariance_check.json
5.  A GATE THAT CANNOT BE EVALUATED IS A FAIL, NEVER A SKIP (CONTRACTS
    section 13 preamble).  No assertion and no assertion operand sits inside a
    bare or broad `except`.  Freshness gates raise AssertionError directly
    (not `assert`, so that -O cannot strip them).
6.  01_build_cohort.py carries, near line 813, a RETRACTED comment about the
    <= 48 h death restriction.  The LIVE code at line 933 uses the elapsed-hours
    rule (`finaldischarge_hrs <= 48`).  Read the code, not the comment.  The
    retracted framing is in BANNED_CONSTANTS and must never be emitted.

Seed 42.  sklearn / statsmodels are NEVER imported (nothing here fits a model).
"""

import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# CANONICAL PATHS  (CONTRACTS section 1; PAT_TBI house pattern -- absolute root)
# ============================================================================
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config  # noqa: E402
config.ensure_dirs()

# Flat repository layout: v3 and v4 artifacts share derived/, tables/,
# figures/ and logs/ (no filename collisions between the two stages).
V3_CODE = config.ANALYSIS_DIR
V3_DERIVED = config.DERIVED
V3_TABLES = config.TABLES
V3_FIGURES = config.FIGURES
V3_LOGS = config.LOGS

V4_BUNDLE = config.DERIVED
OUT_JSON = os.path.join(config.DERIVED, "results_bundle.json")
# The human-readable ledger is an internal working file; it is written to logs/
# and excluded from version control.
OUT_LEDGER = os.path.join(config.LOGS, "NUMBERS-LEDGER.md")


# ============================================================================
# DEPENDENCY MANIFEST -- REUSED VERBATIM FROM 16_generate_manuscript.py:112-162
# Do not invent a new manifest; do not add or drop entries.
# ============================================================================
REQUIRED_TABLES = [
    "table1_baseline.csv",
    "table2_candidate_coverage.csv",
    "table2_lasso_selected.csv",
    "table2_lasso_stability.csv",
    "table3_points.csv",
    "table4_discrimination.csv",
    "table4_delong.csv",
    "table4b_temporal.csv",
    "table5_calibration.csv",
    "table5_operating.csv",
    "table5c_operating_full.csv",     # v3 NEW (Fix 5 / D-6) -- section 7 row 08
    "table5d_matched_workload.csv",   # v3 NEW (Fix 5 / D-6) -- section 7 row 08
    "table6_elderly_reclass.csv",
    "table6b_elderly_ledger.csv",     # v3 NEW (Fix 6 / D-6) -- section 7 row 10, G5f
    "table7_threshold_or.csv",
    "table7_absrisk.csv",
    "table7_sensitivity.csv",
    "table8_mtp.csv",                 # v3 NEW (Fix 7 / D-5) -- section 7 row 17
    "table_consort.csv",
    "table_split_balance.csv",
    "table_subgroups.csv",
]
REQUIRED_FIGURES = [
    "figure1_STROBE.png",
    "figure2_score_distribution.png",
    "figure3_roc.png",
    "figure4_calibration.png",
    "figure5_dca.png",
    "figure6_subgroup_forest.png",
    "figure7_threshold_splines.png",
    "figure8_component.png",
    "figure9_score_card.png",
    "figure10_mtp_equivalent.png",    # v3 NEW (Fix 7) -- section 8 F10
]
REQUIRED_DERIVED = [
    "score_card.json", "refit.json", "risk_tiers.csv", "reclass.json",
    "citl_slope.json",
    "component_venn.json", "changepoint_sbp_composite.json",
    "changepoint_si_composite.json", "changepoint_sbp_mortality.json",
    "changepoint_si_mortality.json", "tipping_point_bound.json",
    "cohort_exclusions.json", "informative_missingness.csv",
]
REQUIRED_LOGS = ["qc_manifest.json", "missingness.csv",
                 "d11_invariance_check.json"]

# Additional artifacts this script reads that are NOT in 16's manifest.  They are
# preflighted separately so a missing one is still a loud, named failure.
EXTRA_INPUTS = [
    (V3_LOGS, "unknown_hc_codes.csv"),
    (V3_DERIVED, "dca_data.csv"),
]

AUTHORITY_PRECEDENCE = [
    "logs/qc_manifest.json",
    "derived/refit.json",
    "derived/score_card.json",
    "derived/reclass.json",
    "the eight named tables under v3/tables/",
    "logs/d11_invariance_check.json",
]


# ============================================================================
# BANNED CONSTANTS
# Downstream verifiers (verify_manuscript.py, check_abstract.py) copy-adapt THIS
# list -- it is the shared definition, so the ban cannot drift between scripts.
#
#   0.876   v2-era AUROC                       31054  v2-definition composite N
#   0.1295  v2-era prevalence                    635  v2-era count
#   3884    superseded calendar-day death rule (the source plan mistyped it as
#           3874; a grep for 3874 never fires -- the real value is 3884)
#   4209    day_fallback death count           4253  ed_inclusive death count
#   4478    all-missing upper bound            27.3  superseded plan fill rate
#   the two banned WORDS below are the RETRACTED framing of the <= 48 h death
#           restriction (01_build_cohort.py:~813).  The live code at line 933
#           uses the elapsed-hours rule.  They appear in this file exactly
#           once each, as the list literal that enforces the ban.
#
# 4,191 is the ONLY quotable death count.
# ============================================================================
BANNED_CONSTANTS = [
    0.876,
    31054,
    0.1295,
    635,
    3884,
    4209,
    4253,
    4478,
    27.3,
    "infeasible",
    "day resolution",
]


def banned_patterns():
    """Compile one regex per banned constant.

    Numbers are matched with tight boundaries so that a banned integer embedded
    in a longer number or in a thousands-grouped number is NOT a false hit:
      3884  matches "3884" and "3,884"  but not "13884", "0.3884", "38840".
    Strings are matched case-insensitively.
    """
    pats = []
    for b in BANNED_CONSTANTS:
        if isinstance(b, str):
            pats.append((b, re.compile(re.escape(b), re.IGNORECASE)))
        elif isinstance(b, int):
            plain = str(b)
            grouped = f"{b:,}"
            alts = {re.escape(plain), re.escape(grouped)}
            pats.append((plain, re.compile(
                r"(?<![\d.,])(?:" + "|".join(sorted(alts)) + r")(?![\d,])")))
        else:
            plain = repr(b)
            pats.append((plain, re.compile(
                r"(?<![\d.,])" + re.escape(plain) + r"(?![\d])")))
    return pats


def scan_banned(text, where):
    """Return a list of (banned_token, matched_text) hits found in `text`."""
    hits = []
    for token, pat in banned_patterns():
        for m in pat.finditer(text):
            hits.append((token, m.group(0), where))
    return hits


# ============================================================================
# LOADERS  (copy-adapted, not imported -- CONTRACTS section 1)
# ============================================================================
_JSON_CACHE = {}
_CSV_CACHE = {}


def load_json(path):
    if path not in _JSON_CACHE:
        with open(path, "r", encoding="utf-8") as fh:
            _JSON_CACHE[path] = json.load(fh)
    return _JSON_CACHE[path]


def load_csv(path):
    if path not in _CSV_CACHE:
        _CSV_CACHE[path] = pd.read_csv(path)
    return _CSV_CACHE[path]


def jget(obj, dotted, origin):
    """Fetch a dotted path out of a loaded JSON.  A missing key is a LOUD
    failure, never a silent default (CONTRACTS section 9-L / section 13 G5a)."""
    cur = obj
    walked = []
    for part in dotted.split("."):
        walked.append(part)
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(
                    f"{origin}: key path {'.'.join(walked)!r} not present. "
                    f"Available at that level: {sorted(cur.keys())[:40]}")
            cur = cur[part]
        elif isinstance(cur, list):
            idx = int(part)
            if idx >= len(cur):
                raise IndexError(
                    f"{origin}: index {idx} out of range at "
                    f"{'.'.join(walked)!r} (len={len(cur)})")
            cur = cur[idx]
        else:
            raise TypeError(
                f"{origin}: cannot walk into a {type(cur).__name__} at "
                f"{'.'.join(walked)!r}")
    return cur


def pick(df, fname, **crit):
    """Boolean mask on NAMED columns.  Returns (row, accessor_string).

    CONTRACTS section 7.1: several tables gained leading columns, so positional
    indexing reads the wrong column.  Everything here is by name.  A mask that
    does not select exactly one row raises with the mask and the row count.
    """
    mask = pd.Series(True, index=df.index)
    parts = []
    for col, want in crit.items():
        if col not in df.columns:
            raise KeyError(f"{fname}: column {col!r} not present. "
                           f"Columns = {list(df.columns)}")
        mask = mask & (df[col].astype(str) == str(want))
        parts.append(f"{col}=={want}")
    sub = df[mask]
    acc = f"{fname}[{' & '.join(parts)}]"
    if len(sub) != 1:
        raise ValueError(
            "MASK DID NOT SELECT EXACTLY ONE ROW\n"
            f"  accessor      : {acc}\n"
            f"  rows selected : {len(sub)}\n"
            f"  mask criteria : {crit}\n"
            f"  matching rows : {list(sub.index)}")
    return sub.loc[sub.index[0]], acc


def pick_contains(df, fname, col, *substrings):
    """Boolean mask built from substring containment on ONE named column.

    Used for table1_baseline.csv, whose Variable labels carry typographic dashes
    and >= glyphs; matching on an ASCII substring is both exact and robust.
    """
    if col not in df.columns:
        raise KeyError(f"{fname}: column {col!r} not present. "
                       f"Columns = {list(df.columns)}")
    mask = pd.Series(True, index=df.index)
    for s in substrings:
        mask = mask & df[col].astype(str).str.contains(s, regex=False, na=False)
    sub = df[mask]
    crit = " & ".join(f"{col} contains {s!r}" for s in substrings)
    acc = f"{fname}[{crit}]"
    if len(sub) != 1:
        raise ValueError(
            "MASK DID NOT SELECT EXACTLY ONE ROW\n"
            f"  accessor      : {acc}\n"
            f"  rows selected : {len(sub)}\n"
            f"  mask criteria : {crit}\n"
            f"  matching rows : {list(sub[col].tolist())}")
    return sub.loc[sub.index[0]], acc


def cell(row, col, origin):
    """Read one named cell of a picked row.  Missing column -> loud failure."""
    if col not in row.index:
        raise KeyError(f"{origin}: column {col!r} not present. "
                       f"Columns = {list(row.index)}")
    return native(row[col])


def native(v):
    """numpy / pandas scalar -> plain Python, NaN -> None."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        if np.isnan(v):
            return None
        if float(v).is_integer() and abs(v) < 1e15:
            return v  # keep as float; callers cast where an int is meant
        return v
    if isinstance(v, (np.str_,)):
        return str(v)
    return v


# ============================================================================
# FORMATTERS -- "value as it should be printed"
# ============================================================================
EN = "\u2013"      # en dash
MINUS = "\u2212"   # true minus sign
GE = "\u2265"
LE = "\u2264"


def f_int(n):
    return f"{int(round(float(n))):,}"


def f_signed_int(n):
    n = int(round(float(n)))
    return (f"+{n:,}" if n > 0 else (f"{MINUS}{abs(n):,}" if n < 0 else "0"))


def f_dec(x, nd):
    s = f"{float(x):.{nd}f}"
    return s.replace("-", MINUS, 1) if s.startswith("-") else s


def f_pct(x, nd=1):
    """x already expressed in percent units."""
    return f"{float(x):.{nd}f}%"


def f_prop_pct(p, nd=1):
    """p is a proportion in [0,1]."""
    return f"{100.0 * float(p):.{nd}f}%"


def f_signed_prop_pct(p, nd=1):
    """p is a SIGNED proportion in [-1, 1] (a relative change, not a share).

    Same 100x rendering as f_prop_pct, but carrying the house typographic
    minus that f_dec uses, so a negative percentage reads like every other
    negative value in the ledger.
    """
    s = f_prop_pct(p, nd)
    return s.replace("-", MINUS, 1) if s.startswith("-") else s


def f_ci(est, lo, hi, nd=4):
    return f"{f_dec(est, nd)} ({f_dec(lo, nd)}{EN}{f_dec(hi, nd)})"


def f_ci_pct(est, lo, hi, nd=1):
    return (f"{float(est):.{nd}f}% ({float(lo):.{nd}f}"
            f"{EN}{float(hi):.{nd}f}%)")


def f_ci_prop_pct(est, lo, hi, nd=1):
    return (f"{100 * float(est):.{nd}f}% ({100 * float(lo):.{nd}f}"
            f"{EN}{100 * float(hi):.{nd}f}%)")


def pct_alts(x):
    """Acceptable renderings of a percent-unit value at coarser precision."""
    return [f"{float(x):.2f}%", f"{float(x):.1f}%", f"{float(x):.0f}%",
            f"{float(x):.2f}", f"{float(x):.1f}"]


def prop_pct_alts(p):
    return pct_alts(100.0 * float(p))


# ============================================================================
# printed_alt FAITHFULNESS
# ----------------------------------------------------------------------------
# `printed_alt` is not decoration.  verify_manuscript.py resolves a manuscript
# numeral against `printed` AND against every `printed_alt`, so an alt string is
# a live licence to print a number.  Until this pass, --verify compared `value`,
# `printed`, `accessor` and `label` and said nothing about `printed_alt`: a
# corrupted alt string ('45.2%' -> '46.2%') left --verify at 1,227 PASS / exit 0
# while licensing the wrong digit in the manuscript.  That is a laundering path,
# and it is closed here in two independent layers:
#
#   LAYER A (verify(), the per-quantity loop)  -- the on-disk alt LIST must equal
#     the list build_all() re-derives from the v3 artifacts, element for element.
#     Exact, no tolerance: the alts are produced by the same renderer call
#     (pct_alts / prop_pct_alts / an inline f-string) on the same re-derived
#     value, so any drift in either direction is a FAIL.
#
#   LAYER B (alt_faithful, below)  -- every alt string must also be a faithful
#     rendering of the numbers the entry itself carries.  Layer A proves disk ==
#     generator; layer B proves the generator's own output is not lying about
#     the value.  It tolerates the deliberate ABSOLUTE-VALUE renderings (value
#     -684 is also rendered '684' for "684 fewer") by admitting |c| as a
#     candidate -- but never a wrong digit, because a candidate still has to
#     round to the token at the token's OWN precision.
#
# Slack is granted in exactly one place: a candidate taken from an already
# ROUNDED `printed` token, when the alt is coarser still (p > d).  That is real
# double rounding -- 'printed' 95.25 (2 dp) comes from a full-precision CI bound
# whose 1 dp rendering is legitimately '95.3'.  A candidate that came from
# `value` is exact and gets no slack, and an integer-precision printed token
# (p == 0) gets none either, so '684' -> '685' still FAILS.
# ============================================================================
_ALT_NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?")


def _alt_tokens(s):
    """[(numeric value, decimals shown)] for every number in a rendered string."""
    out = []
    for m in _ALT_NUM.finditer(str(s)):
        tok = m.group(0).replace(",", "").replace(MINUS, "-").lstrip("+")
        out.append((float(tok), len(tok.split(".")[1]) if "." in tok else 0))
    return out


def _alt_candidates(value, printed):
    """[(number, source precision or None)] the entry itself supplies.

    None means the number is exact (it came from `value`); an integer means it
    was read off `printed` and is itself rounded to that many decimals.  Each
    number is admitted as-is, as |x|, and scaled x100 (a proportion rendered as
    a percent), the scaling shifting the source precision by two places.
    """
    base = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        base.append((float(value), None))
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                base.append((float(v), None))
    for v, p in _alt_tokens(printed):
        base.append((v, p))
    out = []
    for v, p in base:
        pc = None if p is None else max(0, p - 2)
        out.extend([(v, p), (abs(v), p), (100.0 * v, pc), (abs(100.0 * v), pc)])
    return out


def alt_faithful(alt, value, printed):
    """(ok, why) -- is `alt` a faithful rendering of this entry's own numbers?"""
    toks = _alt_tokens(alt)
    if not toks:
        return False, f"{alt!r} carries no numeric token"
    cands = _alt_candidates(value, printed)
    if not cands:
        return False, f"{alt!r}: the entry supplies no number to re-derive from"
    for t, d in toks:
        hit = False
        for c, p in cands:
            tol = 0.5 * 10.0 ** (-d)
            if p is not None and p > d:
                tol += 0.5 * 10.0 ** (-p)     # the source was rounded first
            if abs(t - c) <= tol * (1 + 1e-9):
                hit = True
                break
        if not hit:
            return False, (f"{alt!r}: {t!r} at {d} dp is not a rendering of any "
                           f"number this entry carries (value={value!r}, "
                           f"printed={printed!r})")
    return True, "ok"


# Import-time self-test.  Six positive cases covering every alt shape the bundle
# emits, and five negatives -- a wrong digit at each of three precisions, an
# alt with no number in it, and the integer case a looser tolerance would miss.
assert alt_faithful("45.2%", 45.18388791593695, "45.184%")[0]
assert alt_faithful("0.882 (0.879–0.886)", 0.8823,
                    "0.8823 (0.8787–0.8860)")[0]
assert alt_faithful("684", -684.0, "−684")[0]          # absolute value
assert alt_faithful("0.17–0.20", [0.173, 0.204], "0.173–0.204")[0]
assert alt_faithful("95.4% (95.3–95.6%)", 0.9541364253260164,
                    "95.41% (95.25–95.57%)")[0]        # double rounding
assert alt_faithful("+3.592", 3.592, "3.592 pp")[0]
assert not alt_faithful("46.2%", 45.18388791593695, "45.184%")[0]
assert not alt_faithful("0.883", 0.8823, "0.8823")[0]
assert not alt_faithful("685", -684.0, "−684")[0]
assert not alt_faithful("95.4% (95.7–95.6%)", 0.9541364253260164,
                        "95.41% (95.25–95.57%)")[0]
assert not alt_faithful("none", 1.0, "1")[0]
# KNOWN LIMIT of layer B, stated rather than hidden: it is set-scoped, not
# position-scoped, so swapping one of an entry's own numbers into another slot
# ('95.3' -> '95.4', the point estimate) reads as faithful here.  Layer A is
# exact and positional, so that corruption is still a FAIL of --verify overall;
# layer B is the second net, not the first.
assert alt_faithful("95.4% (95.4–95.6%)", 0.9541364253260164,
                    "95.41% (95.25–95.57%)")[0]

# Non-vacuity floor for the alt coverage check in verify().  The shipped bundle
# carries 1,092 alt strings over 390 quantities; a bundle that has quietly lost
# its alts must not pass by having nothing left to compare.
ALT_STRING_FLOOR = 900
ALT_ENTRY_FLOOR = 300


# ============================================================================
# BUNDLE ACCUMULATOR
# ============================================================================
SECTION_ORDER = ["Cohort", "Card", "Discrimination", "Temporal",
                 "Utility", "Elderly", "Null", "Methods-only"]


class Bundle(object):
    def __init__(self):
        self.q = OrderedDict()

    def add(self, key, value, accessor, label, section,
            printed=None, alt=None, denominator=None, source=None):
        if key in self.q:
            raise KeyError(f"duplicate bundle key {key!r}")
        if section not in SECTION_ORDER:
            raise ValueError(f"unknown section {section!r} for {key!r}")
        value = native(value)
        if printed is None:
            printed = default_printed(value)
        entry = OrderedDict()
        entry["value"] = value
        entry["printed"] = printed
        entry["accessor"] = accessor
        entry["label"] = label
        entry["section"] = section
        entry["printed_alt"] = list(alt or [])
        entry["denominator"] = denominator
        entry["source"] = source or accessor.split("[")[0].split(".")[0]
        self.q[key] = entry
        return entry

    def __len__(self):
        return len(self.q)


def default_printed(value):
    if value is None:
        return "\u2014"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f_int(value)
    if isinstance(value, float):
        if float(value).is_integer() and abs(value) < 1e9:
            return f_int(value)
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(default_printed(v) for v in value)
    return str(value)


# ============================================================================
# PREFLIGHT
# ============================================================================
def preflight():
    print("=" * 78)
    print("PREFLIGHT  --  build_bundle.py  (manifest reused verbatim from "
          "16_generate_manuscript.py:112-162)")
    print("=" * 78)
    missing = []
    groups = [
        (f"TABLES ({len(REQUIRED_TABLES)})", REQUIRED_TABLES, V3_TABLES),
        (f"FIGURES ({len(REQUIRED_FIGURES)})", REQUIRED_FIGURES, V3_FIGURES),
        (f"DERIVED ({len(REQUIRED_DERIVED)})", REQUIRED_DERIVED, V3_DERIVED),
        (f"LOGS ({len(REQUIRED_LOGS)})", REQUIRED_LOGS, V3_LOGS),
    ]
    for group, names, base in groups:
        n_ok = sum(1 for nm in names
                   if os.path.exists(os.path.join(base, nm)))
        for nm in names:
            if not os.path.exists(os.path.join(base, nm)):
                missing.append(os.path.join(base, nm))
        print(f"  [{group}] {n_ok}/{len(names)} present   ({base})")
    n_ok = 0
    for base, nm in EXTRA_INPUTS:
        if os.path.exists(os.path.join(base, nm)):
            n_ok += 1
        else:
            missing.append(os.path.join(base, nm))
    print(f"  [EXTRA ({len(EXTRA_INPUTS)})] {n_ok}/{len(EXTRA_INPUTS)} present")

    if missing:
        print("\nPREFLIGHT FAILED: "
              f"{len(missing)} required input(s) missing:")
        for m in missing:
            print("   - " + m)
        print("=" * 78)
        sys.exit(2)
    total = (len(REQUIRED_TABLES) + len(REQUIRED_FIGURES)
             + len(REQUIRED_DERIVED) + len(REQUIRED_LOGS) + len(EXTRA_INPUTS))
    print(f"\nPREFLIGHT PASSED: all {total} required inputs present.")
    print("=" * 78)


# ============================================================================
# ARTIFACT HANDLES
# ============================================================================
class Art(object):
    """One place that knows which file each accessor prefix resolves to."""

    def __init__(self):
        self.qc_path = os.path.join(V3_LOGS, "qc_manifest.json")
        self.refit_path = os.path.join(V3_DERIVED, "refit.json")
        self.card_path = os.path.join(V3_DERIVED, "score_card.json")
        self.reclass_path = os.path.join(V3_DERIVED, "reclass.json")
        self.citl_path = os.path.join(V3_DERIVED, "citl_slope.json")
        self.d11_path = os.path.join(V3_LOGS, "d11_invariance_check.json")
        self.cexcl_path = os.path.join(V3_DERIVED, "cohort_exclusions.json")

        self.qc = load_json(self.qc_path)
        self.refit = load_json(self.refit_path)
        self.card = load_json(self.card_path)
        self.reclass = load_json(self.reclass_path)
        self.citl = load_json(self.citl_path)
        self.d11 = load_json(self.d11_path)
        self.cexcl = load_json(self.cexcl_path)

        self.t1 = load_csv(os.path.join(V3_TABLES, "table1_baseline.csv"))
        self.t2sel = load_csv(os.path.join(V3_TABLES,
                                           "table2_lasso_selected.csv"))
        self.t2stab = load_csv(os.path.join(V3_TABLES,
                                            "table2_lasso_stability.csv"))
        self.t3 = load_csv(os.path.join(V3_TABLES, "table3_points.csv"))
        self.t4 = load_csv(os.path.join(V3_TABLES,
                                        "table4_discrimination.csv"))
        self.t4d = load_csv(os.path.join(V3_TABLES, "table4_delong.csv"))
        self.t4b = load_csv(os.path.join(V3_TABLES, "table4b_temporal.csv"))
        self.t5cal = load_csv(os.path.join(V3_TABLES,
                                           "table5_calibration.csv"))
        self.t5op = load_csv(os.path.join(V3_TABLES, "table5_operating.csv"))
        self.t5d = load_csv(os.path.join(V3_TABLES,
                                         "table5d_matched_workload.csv"))
        self.t6 = load_csv(os.path.join(V3_TABLES,
                                        "table6_elderly_reclass.csv"))
        self.t6b = load_csv(os.path.join(V3_TABLES,
                                         "table6b_elderly_ledger.csv"))
        self.tcon = load_csv(os.path.join(V3_TABLES, "table_consort.csv"))
        self.tsplit = load_csv(os.path.join(V3_TABLES,
                                            "table_split_balance.csv"))
        self.tiers = load_csv(os.path.join(V3_DERIVED, "risk_tiers.csv"))
        self.hc = load_csv(os.path.join(V3_LOGS, "unknown_hc_codes.csv"))
        self.miss = load_csv(os.path.join(V3_LOGS, "missingness.csv"))
        self.dca = load_csv(os.path.join(V3_DERIVED, "dca_data.csv"))

    # -- convenience getters that also produce the accessor string ---------
    def q(self, dotted):
        return jget(self.qc, dotted, "qc_manifest.json"), f"qc_manifest.{dotted}"

    def qk(self, dotted, literal_key):
        """Fetch qc_manifest.<dotted>[<literal_key>] where the key itself
        contains a dot (e.g. the S32 prefix frequencies), which the dotted-path
        walker cannot express."""
        parent = jget(self.qc, dotted, "qc_manifest.json")
        if not isinstance(parent, dict) or literal_key not in parent:
            raise KeyError(
                f"qc_manifest.json: literal key {literal_key!r} not present "
                f"under {dotted!r}. Available: "
                f"{sorted(parent.keys()) if isinstance(parent, dict) else parent}")
        return parent[literal_key], f'qc_manifest.{dotted}["{literal_key}"]'

    def r(self, dotted):
        return jget(self.refit, dotted, "refit.json"), f"refit.{dotted}"

    def c(self, dotted):
        return jget(self.card, dotted, "score_card.json"), f"score_card.{dotted}"

    def rc(self, dotted):
        return jget(self.reclass, dotted, "reclass.json"), f"reclass.{dotted}"

    def rck(self, dotted, literal_key):
        """reclass.<dotted>[<literal_key>] where the key itself contains a dot
        (e.g. 'si_missed_bleeders(SI<0.9)')."""
        parent = jget(self.reclass, dotted, "reclass.json")
        if not isinstance(parent, dict) or literal_key not in parent:
            raise KeyError(
                f"reclass.json: literal key {literal_key!r} not present under "
                f"{dotted!r}. Available: "
                f"{sorted(parent.keys()) if isinstance(parent, dict) else parent}")
        return parent[literal_key], f'reclass.{dotted}["{literal_key}"]'

    def ci(self, dotted):
        return (jget(self.citl, dotted, "citl_slope.json"),
                f"citl_slope.json.{dotted}")

    def d(self, dotted):
        return (jget(self.d11, dotted, "d11_invariance_check.json"),
                f"d11_invariance_check.json.{dotted}")


ACCESSOR_LEGEND = OrderedDict([
    ("qc_manifest.", "v3/logs/qc_manifest.json"),
    ("refit.", "v3/derived/refit.json"),
    ("score_card.", "v3/derived/score_card.json"),
    ("reclass.", "v3/derived/reclass.json"),
    ("citl_slope.json.", "v3/derived/citl_slope.json"),
    ("d11_invariance_check.json.", "v3/logs/d11_invariance_check.json"),
    ("cohort_exclusions.json", "v3/derived/cohort_exclusions.json"),
    ("risk_tiers.csv", "v3/derived/risk_tiers.csv"),
    ("dca_data.csv", "v3/derived/dca_data.csv"),
    ("unknown_hc_codes.csv", "v3/logs/unknown_hc_codes.csv"),
    ("missingness.csv", "v3/logs/missingness.csv"),
    ("table*.csv", "v3/tables/"),
])


# ============================================================================
# FRESHNESS + CLOSURE GATES
# A gate that cannot be evaluated is a FAIL, never a skip.  These are hard
# raises, not `assert` (so that python -O cannot strip them), and none of them
# sits inside a bare or broad `except`.
# ============================================================================
def _hard(gate_id, ok, msg, results):
    results.append((gate_id, bool(ok), msg))
    if not ok:
        raise AssertionError(f"GATE {gate_id}: FAIL -- {msg}")


# ============================================================================
# C7_pct_scale  --  PERCENT-SCALE CLOSURE
# ----------------------------------------------------------------------------
# THE DEFECT THIS GATE EXISTS TO CATCH: a source field's scale was ASSUMED
# rather than CHECKED.  score_card.override.pct_of_test is already a percentage
# (201 / 79,912 = 0.2515 %); routing it through the proportion formatter
# multiplied it by 100 a second time, and the ledger printed a hundredfold
# overstatement -- a quarter OF the test set instead of a quarter of one percent
# of it.  Nothing in the build noticed, because 0.2515 is a perfectly plausible
# proportion.  Only the numerator and the denominator can tell the two apart.
#
# Every bundle entry that is RENDERED as a percentage -- in `printed` or in any
# `printed_alt` -- is therefore checked four ways:
#
#   C7a  EXACT CLOSURE.  Where a numerator and a denominator are available (a
#        sibling count in the bundle, or one of the nine denominators), the
#        printed percentage MUST equal 100 * num / den to the tolerance implied
#        by its own printed precision.  Every %-bearing printed_alt too.
#   C7b  EMBEDDED-COUNT BRACKET.  For a "N (pct%)" rendering the numerator is
#        inside the string; the denominator it implies must lie between the
#        labelled denominator and its complete-case floor.
#   C7c  SCALE PLAUSIBILITY, where no numerator/denominator pair is available.
#        (i)   the rendering must express the stored value at exactly one of the
#              two scales -- 1x (the source is already a percentage) or 100x
#              (the source is a proportion) -- and never at neither;
#        (ii)  a percentage of a count-based subgroup may not exceed 100;
#        (iii) a value rendered at 100x must itself lie in [-1, 1];
#        (iv)  a *pct*-named source field (qc_manifest, score_card, or a table
#              column) is ALREADY a percentage and may never be re-multiplied.
#   C7d  MANDATORY-CLOSURE COVERAGE.  C7c is blind wherever the stored value is
#        small enough that its own 1x rendering rounds to itself, because then
#        the wrong scale is still a valid reading.  Every key listed in
#        C7A_REQUIRED must therefore actually reach C7a; if the ratio rule that
#        closes it is deleted, or the key vanishes, that is a
#        NOT EVALUATED -- FAIL rather than a silent loss of coverage.
#
# Coverage is total: a percent-rendered entry that reaches none of the first
# three is reported as NOT EVALUATED -- FAIL, never skipped.  No assertion here
# sits inside a bare or broad `except`.
# ============================================================================

# A Table 1 percentage may use a complete-case denominator rather than the full
# labelled one; the largest per-variable missingness in v3 is 4.402 %
# (missingness.csv), so 0.90 is a floor with headroom.  A 100x scale error moves
# the implied denominator by two orders of magnitude and cannot hide inside it.
COMPLETE_CASE_FLOOR = 0.90

_PCT_TOKEN_RE = re.compile(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*%")
_ANY_NUM_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_LEAD_COUNT_RE = re.compile(r"^\s*([+-]?\d[\d,]*)\s*\(")

# (printed-key suffix, numerator sibling(s), denominator sibling(s)).  A "@name"
# denominator is one of the nine denominators.  Multiple specs on one side are
# summed (sensitivity = tp / (tp + fn)).  A rule applies only when EVERY term it
# names resolves to a number, so the same suffix can carry one rule per family.
PCT_RATIO_RULES = [
    ("pct_of_test",    ("n",),               ("@test_set",)),   # card override + tiers
    ("pct_excluded",   ("dropped",),         ("dropped", "n")),  # CONSORT rung
    ("event_rate",     ("events",),          ("n",)),            # override, split
    ("rate",           ("events",),          ("n",)),            # risk tiers
    ("sensitivity",    ("tp",),              ("tp", "fn")),      # table5 operating
    ("specificity",    ("tn",),              ("tn", "fp")),
    ("ppv",            ("tp",),              ("tp", "fp")),
    ("npv",            ("tn",),              ("tn", "fn")),
    ("sensitivity",    ("events_caught",),   ("n_events",)),     # table5d matched
    ("ppv",            ("events_caught",),   ("n_flagged",)),
    ("pct_flagged",    ("n_flagged",),       ("n_analysis",)),
    ("event_rate_pct", ("n_events",),        ("n_denominator",)),  # elderly ledger
    ("flagged_pct",    ("n_flagged",),       ("n_denominator",)),
    ("ppv_pct",        ("flagged_events",),  ("n_flagged",)),
    # G5h/D-10: a relative change stored as a FRACTION of the primary arm's
    # |CITL|.  Closing it against its own numerator and denominator is what
    # distinguishes "-68.7 %" (correct, one multiplication) from "-0.69 %"
    # (the fraction mistaken for a percentage).
    ("abs_citl_shrinkage_relative",
                       ("abs_citl_shrinkage",), ("primary_abs_citl",)),
]

# C7d.  For most entries C7c can adjudicate the scale on its own: a proportion
# rendered at 100x cannot also read as a 1x percentage.  It CANNOT do so when
# the stored value is small enough that its 1x rendering rounds to the value
# itself -- "-0.69%" is a perfectly valid 1x reading of -0.6868, so C7c accepts
# the exact defect this gate exists to catch and only the C7a closure sees it.
# For such a key the ratio rule is the whole check, so its ABSENCE is a
# NOT EVALUATED -- FAIL, never a silent drop to the weaker branch.  Proven by
# falsification: with the rule deleted and the value re-rendered as if it were
# already a percentage, C7a and C7c both stay silent and only C7d fails.
C7A_REQUIRED = ("temporal.ascertainment.abs_citl_shrinkage_relative",)


def _pct_ascii(text):
    """Normalize a printed rendering for numeral scanning: the typographic
    minus becomes ASCII, and the en dash of a CI becomes a separator so that
    '1.53-1.77%' parses as two numerals rather than one subtraction."""
    return str(text).replace(MINUS, "-").replace(EN, " ")


def _pct_numerals(text, pct_only):
    """[(value, half_ulp)] for the numerals in a printed rendering.

    pct_only keeps only numerals carrying a percent sign.  The tolerance is read
    from each numeral's OWN printed precision, so '0.3%' is checked to +-0.05
    and '0.2515%' to +-0.00005.
    """
    out = []
    src = _pct_ascii(text)
    for m in (_PCT_TOKEN_RE if pct_only else _ANY_NUM_RE).finditer(src):
        tok = m.group(1) if pct_only else m.group(0)
        nd = len(tok.split(".")[1]) if "." in tok else 0
        out.append((float(tok.replace(",", "")), 0.5 * 10 ** (-nd)))
    return out


def _numeric_leaves(value):
    """Every number stored in an entry's value (scalars and CI pairs alike)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            out.extend(_numeric_leaves(x))
        return out
    return []


def _pct_named_source(key, accessor):
    """True when the SOURCE FIELD is percent-named ('*_pct', 'pct_*',
    'Pct_excluded', 'percent_*') and is therefore already in percent units.

    Only the TERMINAL segment of the bundle key and of the accessor is
    inspected: 'fixed_15pct' is a workload-target label in the middle of a key,
    not a field name, and must not be mistaken for one.
    """
    tails = (str(key).rsplit(".", 1)[-1].lower(),
             str(accessor).rsplit(".", 1)[-1].lower())
    return any(("pct" in t) or ("percent" in t) for t in tails)


def _pct_term(spec, parent, quantities, dens):
    """One side of a ratio: '@name' is one of the nine denominators; anything
    else is a sibling key under the same parent.  None when unavailable."""
    if spec.startswith("@"):
        d = dens.get(spec[1:])
        return None if d is None else float(d[0])
    e = quantities.get(parent + "." + spec)
    if e is None:
        return None
    v = e["value"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _pct_ratio(key, quantities, dens):
    """(numerator, denominator, how) for a percent entry, or None when no
    numerator/denominator pair is available for it."""
    parent, _, suffix = key.rpartition(".")
    base = suffix[:-5] if suffix.endswith("_full") else suffix
    for want, nspecs, dspecs in PCT_RATIO_RULES:
        if want != base:
            continue
        nvals = [_pct_term(s, parent, quantities, dens) for s in nspecs]
        dvals = [_pct_term(s, parent, quantities, dens) for s in dspecs]
        if any(x is None for x in nvals + dvals):
            continue
        den = sum(dvals)
        if den == 0:
            continue
        return (sum(nvals), den,
                " + ".join(nspecs) + " / " + " + ".join(dspecs))
    return None


def check_percent_scale(A, B):
    """Gate C7_pct_scale.  Returns (ok, detail).  See the block comment above."""
    quantities = B.q
    dens = nine_denominators(A)
    n_primary = jget(A.qc, "n_primary", "qc_manifest.json")
    n_composite = jget(A.qc, "primary_composite.major_hemorrhage_event_n",
                       "qc_manifest.json")
    # Table 1 prints "N (pct%)" against three labelled denominators, each read
    # from qc_manifest -- never a literal.
    t1_denoms = {
        "full analytic cohort": float(n_primary),
        "full analytic cohort, no-event stratum": float(n_primary - n_composite),
        "full analytic cohort, event stratum": float(n_composite),
    }

    fails = []
    closed_by_c7a = set()
    n_entries = n_a = n_b = n_c = 0
    for key, e in quantities.items():
        printed = e.get("printed")
        alts = [str(x) for x in (e.get("printed_alt") or [])]
        pct_alts_here = [x for x in alts if "%" in x]
        if not isinstance(printed, str):
            continue
        if "%" not in printed and not pct_alts_here:
            continue
        n_entries += 1

        head = _pct_numerals(printed, True)
        if "%" in printed and not head:
            fails.append((key, f"{printed!r} carries a percent sign but no "
                               "parsable percent numeral"))
            continue
        evaluated = False

        # ---- C7a  exact numerator / denominator closure -------------------
        ratio = _pct_ratio(key, quantities, dens)
        if ratio is not None:
            num, den, how = ratio
            expect = 100.0 * num / den
            n_a += 1
            evaluated = True
            closed_by_c7a.add(key)
            if head and abs(expect - head[0][0]) > head[0][1] + 1e-9:
                fails.append((key, f"C7a printed {printed!r} but "
                                   f"100 * {num:g} / {den:g} = {expect:.6f}% "
                                   f"({how})"))
            for alt in pct_alts_here:
                a = _pct_numerals(alt, True)
                if not a:
                    fails.append((key, f"C7a printed_alt {alt!r} carries a "
                                       "percent sign but no parsable numeral"))
                elif abs(expect - a[0][0]) > a[0][1] + 1e-9:
                    fails.append((key, f"C7a printed_alt {alt!r} is not "
                                       f"{expect:.6f}% ({how})"))

        # ---- C7b  embedded-count bracket ("N (pct%)") ----------------------
        lead = (_LEAD_COUNT_RE.match(_pct_ascii(printed))
                if isinstance(e["value"], str) else None)
        if lead is not None and e.get("denominator") in t1_denoms and head:
            count = float(lead.group(1).replace(",", ""))
            labelled = t1_denoms[e["denominator"]]
            n_b += 1
            evaluated = True
            if count == 0.0:
                if abs(head[0][0]) > head[0][1] + 1e-9:
                    fails.append((key, f"C7b {printed!r}: the numerator is 0 "
                                       "but a non-zero percentage is printed"))
            else:
                lo = 100.0 * count / labelled
                hi = 100.0 * count / (COMPLETE_CASE_FLOOR * labelled)
                if hi < head[0][0] - head[0][1] - 1e-9 \
                        or lo > head[0][0] + head[0][1] + 1e-9:
                    fails.append((key, f"C7b {printed!r}: {count:,.0f} over a "
                                       f"denominator between "
                                       f"{COMPLETE_CASE_FLOOR:g} x "
                                       f"{labelled:,.0f} and {labelled:,.0f} "
                                       f"gives [{lo:.4f}, {hi:.4f}]%, which "
                                       "does not contain the printed value"))

        # ---- C7c  scale plausibility --------------------------------------
        leaves = _numeric_leaves(e["value"])
        seen = _pct_numerals(printed, False)
        for alt in alts:
            seen.extend(_pct_numerals(alt, False))
        printed_pcts = list(head)
        for alt in pct_alts_here:
            printed_pcts.extend(_pct_numerals(alt, True))
        if leaves:
            n_c += 1
            evaluated = True
            as_pct = all(any(abs(x - p) <= t + 1e-9 for p, t in seen)
                         for x in leaves)
            as_prop = all(any(abs(100.0 * x - p) <= t + 1e-9 for p, t in seen)
                          for x in leaves)
            if not as_pct and not as_prop:
                fails.append((key, f"C7c {printed!r} renders the stored value "
                                   f"{e['value']!r} at neither 1x (already a "
                                   "percentage) nor 100x (a proportion)"))
            elif as_prop and not as_pct \
                    and any(abs(x) > 1.0 + 1e-12 for x in leaves):
                fails.append((key, f"C7c {printed!r} multiplies the stored "
                                   f"value by 100, but {e['value']!r} is "
                                   "outside [-1, 1] and so is not a "
                                   "proportion"))
            elif _pct_named_source(key, e["accessor"]) and not as_pct:
                fails.append((key, f"C7c the source field is *pct*-named "
                                   f"({e['accessor']}) and is ALREADY a "
                                   f"percentage, but {printed!r} multiplies it "
                                   "by 100 a second time"))

        # ---- range: a percentage of a count-based subgroup is <= 100 -------
        over = [p for p, t in printed_pcts if abs(p) > 100.0 + t]
        if over:
            fails.append((key, f"C7c {printed!r} prints a percentage of a "
                               f"count-based quantity outside +-100: {over}"))

        if not evaluated:
            fails.append((key, f"NOT EVALUATED — FAIL: {printed!r} is "
                               "percent-rendered but reached none of C7a / "
                               "C7b / C7c"))

    # ---- C7d  mandatory-closure coverage ----------------------------------
    n_d = 0
    for key in C7A_REQUIRED:
        n_d += 1
        if key not in quantities:
            fails.append((key, "C7d NOT EVALUATED — FAIL: C7A_REQUIRED names "
                               "a quantity that is not in the bundle, so its "
                               "scale is checked by nothing"))
        elif key not in closed_by_c7a:
            fails.append((key, "C7d NOT EVALUATED — FAIL: the stored value is "
                               "a signed fraction whose 1x rendering is "
                               "numerically indistinguishable from its own "
                               "percentage, so C7c cannot adjudicate its "
                               "scale, and PCT_RATIO_RULES carries no rule "
                               "that closes it"))

    n_checks = n_a + n_b + n_c + n_d + n_entries
    if fails:
        shown = "; ".join(f"{k}: {m}" for k, m in fails[:8])
        more = "" if len(fails) <= 8 else f" (+{len(fails) - 8} more)"
        return False, (f"{len(fails)} percent-scale violation(s) over "
                       f"{n_entries} percent-rendered entries -- {shown}{more}")
    return True, (f"{n_checks} percent-scale checks over {n_entries} "
                  f"percent-rendered bundle entries: {n_a} exact "
                  f"numerator/denominator closures, {n_b} embedded-count "
                  f"brackets, {n_c} scale-mode checks, {n_d} mandatory-closure "
                  f"coverage checks, {n_entries} range "
                  "checks -- every printed percentage equals 100 * num / den "
                  "to its own printed precision, and no *pct*-named source "
                  "field is multiplied by 100 a second time")


def run_gates(A, B=None):
    """Returns a list of (gate_id, passed, message).  Raises on the first fail.

    B is the built bundle; C7_pct_scale checks the emitted renderings against
    their own numerators and denominators and cannot run without it.
    """
    res = []

    # ---- F1  death_hemorrhagic == 4191 (the ONLY quotable death count) ----
    dh = jget(A.qc, "primary_composite.death_hemorrhagic_n", "qc_manifest.json")
    _hard("F1", dh == 4191,
          f"qc_manifest.primary_composite.death_hemorrhagic_n = {dh} "
          "(required 4191 -- the hour rule is live)", res)

    # ---- F2  analytic cohort N ------------------------------------------
    np_ = jget(A.qc, "n_primary", "qc_manifest.json")
    _hard("F2", np_ == 239736,
          f"qc_manifest.n_primary = {np_} (required 239736)", res)

    # ---- F3  test-set N --------------------------------------------------
    nt = jget(A.card, "n_test", "score_card.json")
    _hard("F3", nt == 79912,
          f"score_card.n_test = {nt} (required 79912)", res)

    # ---- F4a  cc_anticoag on the card at 0 points ------------------------
    row, _ = pick(A.t3, "table3_points.csv", Variable="cc_anticoag")
    pts = cell(row, "Points", "table3_points.csv")
    role = cell(row, "Role", "table3_points.csv")
    _hard("F4a", (pts == 0) and (role == "point"),
          f"table3_points.csv cc_anticoag Points={pts!r} Role={role!r} "
          "(required 0 / 'point'; a card missing a pre-registered anchor FAILS)",
          res)
    card_vars = [v["variable"] for v in jget(A.card, "variables",
                                             "score_card.json")]
    _hard("F4b", "cc_anticoag" in card_vars,
          "score_card.variables must carry cc_anticoag; it lists "
          f"{len(card_vars)} point variables and cc_anticoag is "
          f"{'present' if 'cc_anticoag' in card_vars else 'ABSENT'}", res)
    cc_pts_card = [v["points"] for v in jget(A.card, "variables",
                                             "score_card.json")
                   if v["variable"] == "cc_anticoag"]
    _hard("F4c", cc_pts_card == [0],
          f"score_card cc_anticoag points = {cc_pts_card} (required [0])", res)

    # ---- F4d  no unstable_ring anywhere on the card ----------------------
    t3_vars = [str(v) for v in A.t3["Variable"].tolist()]
    refit_pts = jget(A.refit, "points", "refit.json")
    refit_pv = jget(A.refit, "point_vars", "refit.json")
    ring_free = ("unstable_ring" not in t3_vars
                 and "unstable_ring" not in card_vars
                 and "unstable_ring" not in refit_pts
                 and "unstable_ring" not in refit_pv)
    _hard("F4d", ring_free,
          "unstable_ring must not appear in table3_points.csv, "
          "score_card.variables, refit.points or refit.point_vars", res)

    # ---- F5  every table5d Reference_instrument is the shock-index string -
    refs = sorted(set(str(x) for x in A.t5d["Reference_instrument"].tolist()))
    si_only = (len(refs) == 1
               and refs[0].startswith("shock_index")
               and not any("rsig" in r.lower() for r in refs))
    _hard("F5", si_only,
          "table5d_matched_workload.csv Reference_instrument values = "
          f"{refs} -- every value must be the shock-index string, so NO rSIG "
          "reference row exists and the rSIG matched-workload margin is "
          "UNQUOTABLE (no hand-differencing)", res)

    # ---- C1  hc_route_counts closure -------------------------------------
    so = jget(A.qc, "hc_route_counts.surgical_only", "qc_manifest.json")
    ao = jget(A.qc, "hc_route_counts.angio_only", "qc_manifest.json")
    bo = jget(A.qc, "hc_route_counts.both", "qc_manifest.json")
    hcn = jget(A.qc, "primary_composite.received_hc_n", "qc_manifest.json")
    _hard("C1", so + ao + bo == hcn,
          f"hc_route_counts {so} + {ao} + {bo} = {so + ao + bo} must equal "
          f"the received_hc component {hcn}", res)

    # ---- C2  flow ladder closes at every rung and across rungs -----------
    flow = jget(A.qc, "cohort_flow", "qc_manifest.json")
    cex_vals = list(A.cexcl.values())
    ok = len(flow) == 7 and len(cex_vals) == 7
    for i, step in enumerate(flow):
        ok = ok and (step["n_in"] - step["n_dropped"] == step["n_out"])
        ok = ok and (step["n_out"] == cex_vals[i])
        if i > 0:
            ok = ok and (flow[i - 1]["n_out"] == step["n_in"])
    con_after = [int(x) for x in A.tcon["N_after"].tolist()]
    ok = ok and (con_after == [s["n_out"] for s in flow])
    ok = ok and (flow[-1]["n_out"] == np_)
    _hard("C2", ok,
          "7-rung flow ladder must close at every rung, across consecutive "
          "rungs, and agree across qc_manifest.cohort_flow / "
          "cohort_exclusions.json / table_consort.csv", res)

    # ---- C3  composite components sum >= composite (patients counted once)
    comp = jget(A.qc, "primary_composite.major_hemorrhage_event_n",
                "qc_manifest.json")
    stx = jget(A.qc, "primary_composite.substantial_transfusion_n",
               "qc_manifest.json")
    _hard("C3", hcn + dh + stx >= comp and comp >= max(hcn, dh, stx),
          f"composite {comp} must be <= {hcn}+{dh}+{stx} and >= each component "
          "(a patient with two components is counted once)", res)

    # ---- C4  refit betas round to the printed card betas -----------------
    coefs = jget(A.refit, "coefficients", "refit.json")
    bad = []
    for var, beta in coefs.items():
        row_v, _ = pick(A.t3, "table3_points.csv", Variable=var)
        printed_beta = cell(row_v, "\u03b2", "table3_points.csv")
        if abs(round(float(beta), 4) - float(printed_beta)) > 1e-9:
            bad.append((var, beta, printed_beta))
    _hard("C4", not bad,
          f"refit.coefficients must round to table3_points.csv beta; "
          f"mismatches = {bad}", res)

    # ---- C5  test N agrees across score_card / split balance / table4 ----
    row_sp, _ = pick(A.tsplit, "table_split_balance.csv", split="test")
    n_sp = int(cell(row_sp, "N", "table_split_balance.csv"))
    ev_sp = int(cell(row_sp, "N_events", "table_split_balance.csv"))
    row_d, _ = pick(A.t4, "table4_discrimination.csv",
                    Outcome="major_hemorrhage_event", Score="pelvic_score",
                    AUROC_type="integer_bedside")
    n_d = int(cell(row_d, "N_used", "table4_discrimination.csv"))
    ev_d = int(cell(row_d, "N_events", "table4_discrimination.csv"))
    _hard("C5", nt == n_sp == n_d and ev_sp == ev_d,
          f"test N: score_card {nt}, split_balance {n_sp}, table4 {n_d}; "
          f"test events: split_balance {ev_sp}, table4 {ev_d}", res)

    # ---- C6  the nine denominators, each sourced, none merged ------------
    d_ok, d_detail = check_nine_denominators(A)
    _hard("C6", d_ok, f"nine-denominator ledger: {d_detail}", res)

    # ---- C7  percent scale, closed against numerators and denominators ---
    # A source field's scale is CHECKED here, never assumed.  Without the built
    # bundle the gate cannot be evaluated, and a gate that cannot be evaluated
    # is a FAIL, never a skip.
    if B is None:
        _hard("C7_pct_scale", False,
              "NOT EVALUATED — FAIL: the percent-scale gate reads the "
              "emitted renderings and run_gates() was called without the "
              "built bundle", res)
    p_ok, p_detail = check_percent_scale(A, B)
    _hard("C7_pct_scale", p_ok, p_detail, res)

    return res


def nine_denominators(A):
    """The C3 ledger.  Each denominator is read from an artifact -- never a
    literal -- and carries its own label.  They are NEVER merged."""
    out = OrderedDict()

    v, acc = A.q("n_primary")
    out["cohort_full"] = (int(v), acc, "full analytic cohort")

    v, acc = A.c("n_test")
    out["test_set"] = (int(v), acc, "test set")

    row, acc = pick(A.t5op, "table5_operating.csv",
                    Score="SI-alone", Cutoff_Rule="youden")
    out["si_alone_operating"] = (
        int(cell(row, "N", "table5_operating.csv")), acc + ".N",
        "shock-index-alone operating denominator "
        "(patients with a computable shock index)")

    row, acc = pick(A.t5d, "table5d_matched_workload.csv",
                    Stratum="overall", Workload_target_label="deployed_rule_in",
                    Instrument="pelvic_score", Rule="score_only")
    out["matched_set"] = (
        int(cell(row, "N_analysis", "table5d_matched_workload.csv")),
        acc + ".N_analysis",
        "matched set (common complete case: score + shock index + rSIG all "
        "observed)")

    v, acc = A.rc("elderly_test_n")
    out["elderly_test"] = (int(v), acc, "elderly test set (age >= 65)")

    row, acc = pick(A.t6b, "table6b_elderly_ledger.csv",
                    Arm="strategy_si_alone")
    out["elderly_observed_si"] = (
        int(cell(row, "N_denominator", "table6b_elderly_ledger.csv")),
        acc + ".N_denominator",
        "elderly with an observed shock index")

    row, acc = pick(A.t5d, "table5d_matched_workload.csv",
                    Stratum="elderly_ge65",
                    Workload_target_label="deployed_rule_in",
                    Instrument="pelvic_score", Rule="score_only")
    out["elderly_matched"] = (
        int(cell(row, "N_analysis", "table5d_matched_workload.csv")),
        acc + ".N_analysis",
        "elderly matched set (common complete case)")

    row, acc = pick(A.t6b, "table6b_elderly_ledger.csv",
                    Arm="newly_flagged_up")
    out["elderly_si_reassuring"] = (
        int(cell(row, "N_denominator", "table6b_elderly_ledger.csv")),
        acc + ".N_denominator",
        "elderly with a reassuring shock index (< 0.9)")

    row, acc = pick(A.t6b, "table6b_elderly_ledger.csv",
                    Arm="down_classified")
    out["elderly_si_alarming"] = (
        int(cell(row, "N_denominator", "table6b_elderly_ledger.csv")),
        acc + ".N_denominator",
        "elderly with an alarming shock index (>= 0.9)")

    return out


# Plan section C3 asserts these nine values.  They are checked against the
# artifacts, never substituted for them: on a disagreement the ARTIFACT wins and
# the gate fails loudly so the discrepancy is recorded, not silently absorbed.
PLAN_C3_EXPECTED = OrderedDict([
    ("cohort_full", 239736),
    ("test_set", 79912),
    ("si_alone_operating", 77739),
    ("matched_set", 75072),
    ("elderly_test", 37235),
    ("elderly_observed_si", 35944),
    ("elderly_matched", 34083),
    ("elderly_si_reassuring", 33025),
    ("elderly_si_alarming", 2919),
])


def check_nine_denominators(A):
    got = nine_denominators(A)
    if len(got) != 9:
        return False, f"expected 9 denominators, built {len(got)}"
    diffs = []
    for k, expect in PLAN_C3_EXPECTED.items():
        if k not in got:
            return False, f"denominator {k!r} not built"
        if got[k][0] != expect:
            diffs.append(f"{k}: artifact={got[k][0]} plan={expect}")
    if len(set(v[0] for v in got.values())) != 9:
        return False, "two denominators collided on the same value"
    if diffs:
        return False, ("artifact disagrees with plan section C3 -- "
                       + "; ".join(diffs))
    return True, "all nine reproduce from the artifacts"


# ============================================================================
# EXTRACTION
# ============================================================================
def build_all(A):
    B = Bundle()
    add = B.add

    # ---------------------------------------------------------------- COHORT
    flow = jget(A.qc, "cohort_flow", "qc_manifest.json")
    for i, step in enumerate(flow, start=1):
        add(f"cohort.flow.rung{i}.n", int(step["n_out"]),
            f"qc_manifest.cohort_flow.{i - 1}.n_out",
            f"Participant-flow rung {i} of 7: {step['label']} "
            "(N remaining after this rung)", "Cohort",
            printed=f_int(step["n_out"]))
        add(f"cohort.flow.rung{i}.label", str(step["label"]),
            f"qc_manifest.cohort_flow.{i - 1}.label",
            f"Participant-flow rung {i} of 7 label", "Cohort",
            printed=str(step["label"]))
        if i > 1:
            add(f"cohort.flow.rung{i}.dropped", int(step["n_dropped"]),
                f"qc_manifest.cohort_flow.{i - 1}.n_dropped",
                f"Participants excluded at rung {i}: {step['label']}",
                "Cohort", printed=f_int(step["n_dropped"]))
            row, acc = pick(A.tcon, "table_consort.csv", Step=i)
            pe = cell(row, "Pct_excluded", "table_consort.csv")
            add(f"cohort.flow.rung{i}.pct_excluded", pe,
                acc + ".Pct_excluded",
                f"Percent of the rung-{i} input excluded at rung {i}",
                "Cohort", printed=f_pct(pe, 2), alt=pct_alts(pe),
                denominator=f_int(step["n_in"]) + " entering rung " + str(i))

    v, acc = A.q("n_primary")
    add("cohort.n_primary", int(v), acc,
        "Full analytic cohort (adults, blunt pelvic-ring fracture, direct "
        "arrival, ascertainable outcome), NTDB/TQP 2019-2024", "Cohort",
        printed=f_int(v), denominator="full analytic cohort")

    v, acc = A.q("n_acetabulum_inclusive")
    add("cohort.n_acetabulum_inclusive", int(v), acc,
        "Acetabulum-inclusive cohort (pre-decision CSV rows) -- named "
        "sensitivity frame only, never the analytic denominator", "Cohort",
        printed=f_int(v))

    v, acc = A.q("primary_composite.definition")
    add("cohort.composite.definition", str(v), acc,
        "Primary composite outcome definition", "Cohort", printed=str(v))

    v, acc = A.q("primary_composite.major_hemorrhage_event_n")
    add("cohort.composite.events_n", int(v), acc,
        "Major hemorrhage events (primary composite) in the full cohort",
        "Cohort", printed=f_int(v), denominator="full analytic cohort")
    v, acc = A.q("primary_composite.major_hemorrhage_event_rate_pct")
    add("cohort.composite.rate_pct", v, acc,
        "Major hemorrhage event rate in the full cohort", "Cohort",
        printed=f_pct(v, 3), alt=pct_alts(v),
        denominator="full analytic cohort")

    for comp_key, comp_label in [
            ("received_hc",
             "Composite component: therapeutic hemorrhage control received "
             "(operative or therapeutic angiographic)"),
            ("death_hemorrhagic",
             "Composite component: hemorrhagic death (in-hospital, head "
             "AIS < 4, within 48 h of arrival by elapsed hours)"),
            ("substantial_transfusion",
             "Composite component: substantial transfusion "
             "(>= 1000 mL or >= 4 U PRBC within 4 h)")]:
        v, acc = A.q(f"primary_composite.{comp_key}_n")
        add(f"cohort.composite.{comp_key}_n", int(v), acc, comp_label,
            "Cohort", printed=f_int(v), denominator="full analytic cohort")

    v, acc = A.q("primary_composite.patient_state_composite_n")
    add("cohort.patient_state_composite.events_n", int(v), acc,
        "Treatment-free co-primary (patient-state composite: hemorrhagic "
        "death or substantial transfusion) -- the anti-circularity endpoint",
        "Cohort", printed=f_int(v), denominator="full analytic cohort")
    v, acc = A.q("primary_composite.patient_state_composite_rate_pct")
    add("cohort.patient_state_composite.rate_pct", v, acc,
        "Patient-state composite rate in the full cohort", "Cohort",
        printed=f_pct(v, 3), alt=pct_alts(v),
        denominator="full analytic cohort")

    v, acc = A.q("gates.mortality_only_anchor_definition")
    add("cohort.death_hemorrhagic.definition", str(v), acc,
        "Hemorrhagic-death anchor definition (elapsed-hours rule; the live "
        "code, not the retracted comment)", "Cohort", printed=str(v))

    # HC route closure
    route_total = 0
    for rk, rl in [("surgical_only", "surgical hemorrhage control only"),
                   ("angio_only", "therapeutic angiography only"),
                   ("both", "both surgical and therapeutic angiographic")]:
        v, acc = A.q(f"hc_route_counts.{rk}")
        route_total += int(v)
        add(f"cohort.hc_route.{rk}", int(v), acc,
            f"Hemorrhage-control route: {rl}", "Cohort", printed=f_int(v),
            denominator="patients receiving therapeutic hemorrhage control")
    add("cohort.hc_route.total", route_total,
        "qc_manifest.hc_route_counts.surgical_only + .angio_only + .both",
        "Hemorrhage-control route closure "
        f"({f_int(jget(A.qc, 'hc_route_counts.surgical_only', 'qc_manifest.json'))}"
        f" + {f_int(jget(A.qc, 'hc_route_counts.angio_only', 'qc_manifest.json'))}"
        f" + {f_int(jget(A.qc, 'hc_route_counts.both', 'qc_manifest.json'))}) "
        "= the received_hc component", "Cohort", printed=f_int(route_total))
    v, acc = A.q("hc_route_counts.none")
    add("cohort.hc_route.none", int(v), acc,
        "No hemorrhage-control procedure", "Cohort", printed=f_int(v),
        denominator="full analytic cohort")

    # HC type counts (unknown_hc_codes.csv)
    for _, r in A.hc.iterrows():
        kind = str(r["kind"])
        value = str(r["value"])
        cnt = int(r["count"])
        cls = str(r["assigned_class"])
        acc = (f"unknown_hc_codes.csv[kind=={kind} & value=={value}].count")
        excl = " -- EXCLUDED from the outcome" if cls == "diagnostic_only" else ""
        add(f"cohort.hc_type.{kind}.{value}", cnt, acc,
            f"Hemorrhage-control type count: {kind} / {value} "
            f"(class: {cls}){excl}", "Cohort", printed=f_int(cnt))
    row, acc = pick(A.hc, "unknown_hc_codes.csv", kind="angio", value="ANGIO")
    add("cohort.hc_type.diagnostic_only_excluded",
        int(cell(row, "count", "unknown_hc_codes.csv")), acc + ".count",
        "Diagnostic angiography alone -- did NOT qualify as therapeutic "
        "hemorrhage control and is excluded from the outcome", "Cohort",
        printed=f_int(cell(row, "count", "unknown_hc_codes.csv")))
    n_surg_types = int((A.hc["kind"].astype(str) == "surgical").sum())
    add("cohort.hc_type.n_surgical_categories", n_surg_types,
        "unknown_hc_codes.csv[kind==surgical] row count",
        "Number of operative hemorrhage-control categories that qualified",
        "Cohort", printed=str(n_surg_types))

    # Predictor prevalence in the full cohort
    for pk, pl in [
            ("cc_anticoag_n", "Preexisting anticoagulant therapy"),
            ("cc_bleeding_n", "Preexisting bleeding disorder"),
            ("sbp_unrecordable_n",
             "No signs of life on arrival (unrecordable systolic pressure)"),
            ("unstable_ring_n",
             "Unstable pelvic ring (ICD-10 S32.81*) -- BASELINE DESCRIPTIVE "
             "and adjustment covariate only; NOT a score-card variable")]:
        v, acc = A.q(f"new_predictors.{pk}")
        add(f"cohort.prevalence.{pk}", int(v), acc, pl, "Cohort",
            printed=f_int(v), denominator="full analytic cohort")

    # S32 prefix frequencies
    for pref in ["S32.1", "S32.3", "S32.5", "S32.6", "S32.81", "S32.82"]:
        v, acc = A.qk("s32_frequencies", pref)
        add(f"cohort.s32_freq.{pref.replace('.', '_')}", int(v), acc,
            f"Pelvic-ring inclusion diagnoses carrying ICD-10-CM {pref}*",
            "Cohort", printed=f_int(v))

    # Split balance
    for sp, sl in [("full", "full analytic cohort"),
                   ("train", "derivation (random 2/3) split"),
                   ("test", "validation (random 1/3) split")]:
        row, acc = pick(A.tsplit, "table_split_balance.csv", split=sp)
        add(f"cohort.split.{sp}.n",
            int(cell(row, "N", "table_split_balance.csv")), acc + ".N",
            f"Random split: N in the {sl}", "Cohort",
            printed=f_int(cell(row, "N", "table_split_balance.csv")))
        add(f"cohort.split.{sp}.events",
            int(cell(row, "N_events", "table_split_balance.csv")),
            acc + ".N_events",
            f"Random split: composite events in the {sl}", "Cohort",
            printed=f_int(cell(row, "N_events", "table_split_balance.csv")))
        er = cell(row, "event_rate", "table_split_balance.csv")
        add(f"cohort.split.{sp}.event_rate", er, acc + ".event_rate",
            f"Random split: composite event rate in the {sl}", "Cohort",
            printed=f_prop_pct(er, 3), alt=prop_pct_alts(er))

    # Table 1 -- curated baseline rows (Overall / NoEvent / Event / SMD)
    t1_rows = [
        ("age_years", ["Age, years"], "Age, years -- mean (SD)"),
        ("female_sex", ["Female sex"], "Female sex"),
        ("sbp", ["Systolic blood pressure, mmHg"],
         "Systolic blood pressure at arrival, mmHg -- median (IQR)"),
        ("heart_rate", ["Heart rate, beats/min"],
         "Heart rate at arrival, beats/min -- median (IQR)"),
        ("shock_index", ["Shock index (HR / SBP)"],
         "Shock index at arrival -- median (IQR)"),
        ("gcs_total", ["Glasgow Coma Scale, total"],
         "Glasgow Coma Scale total at arrival -- median (IQR)"),
        ("sbp_lt90", ["Systolic blood pressure < 90 mmHg"],
         "Systolic blood pressure below 90 mmHg at arrival"),
        ("nslo", ["No signs of life on arrival"],
         "No signs of life on arrival (unrecordable systolic pressure)"),
        ("mech_fall", ["Fall"], "Mechanism: fall"),
        ("mech_mvc", ["Motor vehicle crash (occupant)"],
         "Mechanism: motor vehicle crash (occupant)"),
        ("iss", ["Injury Severity Score"],
         "Injury Severity Score -- median (IQR)"),
        ("unstable_ring", ["Unstable pelvic ring"],
         "Unstable pelvic ring (ICD-10 S32.81*) -- baseline descriptive, NOT "
         "a score-card variable"),
        ("charlson", ["Charlson Comorbidity Index"],
         "Charlson Comorbidity Index -- median (IQR)"),
        ("cc_anticoag", ["Anticoagulant therapy (preexisting)"],
         "Preexisting anticoagulant therapy"),
        ("composite", ["primary composite (Panel A stratifier)"],
         "Major hemorrhage event (primary composite)"),
        ("received_hc", ["Hemorrhage-control procedure, surgical or "
                         "angiographic"],
         "Composite component: therapeutic hemorrhage control"),
        ("death_hemorrhagic", ["within 48 h (composite component)"],
         "Composite component: hemorrhagic death"),
        ("substantial_transfusion", ["Substantial transfusion, ",
                                     "(composite component)"],
         "Composite component: substantial transfusion"),
        ("patient_state_composite", ["Patient-state composite"],
         "Treatment-free co-primary: patient-state composite"),
    ]
    for key, subs, label in t1_rows:
        row, acc = pick_contains(A.t1, "table1_baseline.csv", "Variable", *subs)
        for col, cl, den in [
                ("Overall", "whole cohort", "full analytic cohort"),
                ("NoEvent", "no-event stratum",
                 "full analytic cohort, no-event stratum"),
                ("Event", "event stratum",
                 "full analytic cohort, event stratum")]:
            val = cell(row, col, "table1_baseline.csv")
            if val is None:
                continue
            add(f"cohort.table1.{key}.{col.lower()}", str(val),
                acc + f".{col}", f"{label} -- {cl}", "Cohort",
                printed=str(val), denominator=den)
        smd = cell(row, "SMD_event", "table1_baseline.csv")
        if smd is not None:
            add(f"cohort.table1.{key}.smd", smd, acc + ".SMD_event",
                f"{label} -- standardised mean difference, event vs no event",
                "Cohort", printed=f_dec(smd, 3))

    # Missingness of the arrival predictors
    for mv, ml in [("sbp2", "systolic blood pressure at arrival"),
                   ("pulse2", "heart rate at arrival"),
                   ("shock_index", "shock index at arrival"),
                   ("gcs2", "Glasgow Coma Scale total at arrival"),
                   ("motor2", "Glasgow Coma Scale motor at arrival"),
                   ("iss", "Injury Severity Score")]:
        row, acc = pick(A.miss, "missingness.csv", variable=mv)
        n_m = int(cell(row, "n_missing", "missingness.csv"))
        p_m = cell(row, "pct_missing", "missingness.csv")
        add(f"cohort.missing.{mv}.n", n_m, acc + ".n_missing",
            f"Missing values: {ml}", "Cohort", printed=f_int(n_m),
            denominator="full analytic cohort")
        add(f"cohort.missing.{mv}.pct", p_m, acc + ".pct_missing",
            f"Percent missing: {ml}", "Cohort", printed=f_pct(p_m, 3),
            alt=pct_alts(p_m), denominator="full analytic cohort")

    v, acc = A.q("blood_4h_fill_pct")
    add("cohort.blood4h_fill_pct", v, acc,
        "Fill rate of the 4-hour PRBC field in the analytic cohort", "Cohort",
        printed=f_pct(v, 3), alt=pct_alts(v))
    v, acc = A.q("informative_missingness_missing_stratum_mortality_pct")
    add("cohort.informative_missingness_mortality_pct", v, acc,
        "All-cause in-hospital mortality among patients with missing arrival "
        "vitals (informative-missingness probe)", "Cohort",
        printed=f_pct(v, 2), alt=pct_alts(v))

    # ------------------------------------------------------------------ CARD
    v, acc = A.c("n_point_variables")
    add("card.n_point_variables", int(v), acc,
        "Number of point-scoring variables on the arrival card", "Card",
        printed=str(int(v)))

    point_rows = A.t3[A.t3["Role"].astype(str) == "point"]
    for var in point_rows["Variable"].astype(str).tolist():
        row, acc = pick(A.t3, "table3_points.csv", Variable=var)
        beta = cell(row, "\u03b2", "table3_points.csv")
        se = cell(row, "SE", "table3_points.csv")
        orv = cell(row, "OR", "table3_points.csv")
        lo = cell(row, "95% CI lower", "table3_points.csv")
        hi = cell(row, "95% CI upper", "table3_points.csv")
        pts = int(cell(row, "Points", "table3_points.csv"))
        add(f"card.beta.{var}", beta, acc + ".\u03b2",
            f"Score card: log-odds coefficient for {var}", "Card",
            printed=f_dec(beta, 4))
        add(f"card.se.{var}", se, acc + ".SE",
            f"Score card: standard error for {var}", "Card",
            printed=f_dec(se, 4))
        add(f"card.or.{var}", orv, acc + ".OR",
            f"Score card: odds ratio for {var}", "Card",
            printed=f_dec(orv, 3), alt=[f"{orv:.2f}"])
        add(f"card.or_ci.{var}", [lo, hi],
            acc + ".[95% CI lower, 95% CI upper]",
            f"Score card: 95% confidence interval for the {var} odds ratio",
            "Card", printed=f"{f_dec(lo, 3)}{EN}{f_dec(hi, 3)}",
            alt=[f"{lo:.2f}{EN}{hi:.2f}"])
        add(f"card.or_full.{var}", orv, acc + ".OR + .[95% CI]",
            f"Score card: odds ratio with 95% CI for {var}", "Card",
            printed=f"{f_dec(orv, 3)} ({f_dec(lo, 3)}{EN}{f_dec(hi, 3)})",
            alt=[f"{orv:.2f} ({lo:.2f}{EN}{hi:.2f})"])
        add(f"card.points.{var}", pts, acc + ".Points",
            f"Score card: integer points for {var}", "Card",
            printed=f_signed_int(pts) if pts != 0 else "0")

    ov_row, ov_acc = pick(A.t3, "table3_points.csv", Role="override")
    ov_var = str(cell(ov_row, "Variable", "table3_points.csv"))
    add("card.override.variable", ov_var, ov_acc + ".Variable",
        "Automatic-escalation override variable (carries NO point value)",
        "Card", printed=ov_var)
    v, acc = A.r("override_labels." + ov_var)
    add("card.override.label", str(v), acc,
        "Override label -- the only permitted wording (never 'SBP = 0', never "
        "a missingness framing)", "Card", printed=str(v))
    v, acc = A.r("override_rule_text")
    add("card.override.rule_text", str(v), acc,
        "Override banner text for Table 2 / Figure 9", "Card", printed=str(v))
    for col, key, nd, lbl in [
            ("\u03b2", "beta", 4, "log-odds coefficient"),
            ("SE", "se", 4, "standard error"),
            ("OR", "or", 3, "odds ratio")]:
        val = cell(ov_row, col, "table3_points.csv")
        add(f"card.override.{key}", val, ov_acc + f".{col}",
            f"Override row: {lbl}", "Card", printed=f_dec(val, nd))
    ov_lo = cell(ov_row, "95% CI lower", "table3_points.csv")
    ov_hi = cell(ov_row, "95% CI upper", "table3_points.csv")
    ov_or = cell(ov_row, "OR", "table3_points.csv")
    add("card.override.or_full", ov_or, ov_acc + ".OR + .[95% CI]",
        "Override row: odds ratio with 95% CI", "Card",
        printed=f"{f_dec(ov_or, 3)} ({f_dec(ov_lo, 3)}{EN}{f_dec(ov_hi, 3)})",
        alt=[f"{ov_or:.1f} ({ov_lo:.1f}{EN}{ov_hi:.1f})"])
    for ok_, ol_, fmt in [
            ("n", "patients in the test set meeting the override", "int"),
            ("events", "composite events among override patients", "int"),
            ("event_rate", "composite event rate among override patients",
             "prop"),
            ("pct_of_test", "override patients as a percent of the test set",
             "pct")]:
        v, acc = A.c(f"override.{ok_}")
        if fmt == "int":
            add(f"card.override.{ok_}", int(v), acc,
                f"Override: {ol_}", "Card", printed=f_int(v),
                denominator="test set")
        elif fmt == "pct":
            # score_card.override.pct_of_test is ALREADY in percent units
            # (201 / 79,912 = 0.2515 %).  It must NOT go through f_prop_pct /
            # prop_pct_alts, which would multiply by 100 a second time and
            # print a hundredfold overstatement.  Gate C7_pct_scale enforces
            # this for every percent-rendered entry in the bundle.
            # pct_alts() is not used verbatim here: its 0-decimal rung renders
            # a sub-1 % share as "0%", which a downstream numeral resolver must
            # never accept for this quantity.  The alternates below are the
            # 2-dp and 4-dp renderings, with and without the percent sign.
            add(f"card.override.{ok_}", v, acc, f"Override: {ol_}", "Card",
                printed=f_pct(v, 2),
                alt=[f_pct(v, 2), f_pct(v, 4),
                     f"{float(v):.2f}", f"{float(v):.4f}"],
                denominator="test set")
        else:
            add(f"card.override.{ok_}", v, acc, f"Override: {ol_}", "Card",
                printed=f_prop_pct(v, 1), alt=prop_pct_alts(v),
                denominator="test set")
    v, acc = A.c("override.event_rate_ci_low")
    v2, acc2 = A.c("override.event_rate_ci_high")
    add("card.override.event_rate_ci", [v, v2], acc + " + " + acc2,
        "Override: 95% CI for the composite event rate among override "
        "patients", "Card",
        printed=f"{f_prop_pct(v, 1)}{EN}{f_prop_pct(v2, 1)}",
        denominator="test set")
    v, acc = A.c("deployed_rule")
    add("card.deployed_rule", str(v), acc,
        "Deployed escalation rule (score threshold OR override)", "Card",
        printed=str(v))

    for k, lbl in [("min", "observed minimum"), ("max", "observed maximum")]:
        v, acc = A.c(f"score_range.{k}")
        add(f"card.range.observed_{k}", int(v), acc,
            f"Score card: {lbl} of the point score in the test set "
            "(point-role variables only)", "Card", printed=f_signed_int(v))
    add("card.range.observed",
        [int(jget(A.card, "score_range.min", "score_card.json")),
         int(jget(A.card, "score_range.max", "score_card.json"))],
        "score_card.score_range.min + score_card.score_range.max",
        "Score card: observed range of the point score in the test set",
        "Card",
        printed=(f_signed_int(jget(A.card, 'score_range.min', 'sc')) + " to "
                 + f_signed_int(jget(A.card, 'score_range.max', 'sc'))))
    for k, lbl in [("min_mutually_exclusive",
                    "theoretical minimum, mutually exclusive groups applied"),
                   ("max_mutually_exclusive",
                    "theoretical maximum, mutually exclusive groups applied"),
                   ("min_unconstrained", "theoretical unconstrained minimum"),
                   ("max_unconstrained", "theoretical unconstrained maximum")]:
        v, acc = A.c(f"score_range_theoretical.{k}")
        add(f"card.range.theoretical_{k}", int(v), acc,
            f"Score card: {lbl} (a bound, never observed in any patient)",
            "Card", printed=f_signed_int(v))

    for tk, tl in [("rule_out", "rule-out tier (score below the rule-out cut)"),
                   ("observe", "observation tier"),
                   ("escalate", "escalation tier (score at or above the "
                                "rule-in cut)")]:
        row, acc = pick(A.tiers, "risk_tiers.csv", tier=tk, row_type="score_tier")
        n = int(cell(row, "n", "risk_tiers.csv"))
        ev = int(cell(row, "composite_events", "risk_tiers.csv"))
        rt = cell(row, "composite_rate", "risk_tiers.csv")
        lo = cell(row, "composite_ci_low", "risk_tiers.csv")
        hi = cell(row, "composite_ci_high", "risk_tiers.csv")
        smin = int(cell(row, "score_min", "risk_tiers.csv"))
        smax = int(cell(row, "score_max", "risk_tiers.csv"))
        add(f"card.tier.{tk}.n", n, acc + ".n",
            f"Risk tier: test patients in the {tl}", "Card", printed=f_int(n),
            denominator="test set")
        n_test_card = float(jget(A.card, "n_test", "score_card.json"))
        pct_tier = 100.0 * n / n_test_card
        add(f"card.tier.{tk}.pct_of_test", pct_tier,
            acc + ".n / score_card.n_test",
            f"Risk tier: percent of the test set in the {tl}", "Card",
            printed=f_pct(pct_tier, 1), alt=pct_alts(pct_tier),
            denominator="test set")
        add(f"card.tier.{tk}.events", ev, acc + ".composite_events",
            f"Risk tier: composite events in the {tl}", "Card",
            printed=f_int(ev), denominator=f"{f_int(n)} in the {tl}")
        add(f"card.tier.{tk}.rate", rt, acc + ".composite_rate",
            f"Risk tier: composite event rate in the {tl}", "Card",
            printed=f_prop_pct(rt, 2), alt=prop_pct_alts(rt),
            denominator=f"{f_int(n)} in the {tl}")
        add(f"card.tier.{tk}.rate_ci", [lo, hi],
            acc + ".[composite_ci_low, composite_ci_high]",
            f"Risk tier: 95% CI for the composite event rate in the {tl}",
            "Card", printed=f_ci_prop_pct(rt, lo, hi, 2))
        add(f"card.tier.{tk}.score_range", [smin, smax],
            acc + ".[score_min, score_max]",
            f"Risk tier: point-score range defining the {tl}", "Card",
            printed=f"{f_signed_int(smin)} to {f_signed_int(smax)}")
        add(f"card.tier.{tk}.death_rate",
            cell(row, "death_rate", "risk_tiers.csv"), acc + ".death_rate",
            f"Risk tier: hemorrhagic-death rate in the {tl}", "Card",
            printed=f_prop_pct(cell(row, "death_rate", "risk_tiers.csv"), 2),
            denominator=f"{f_int(n)} in the {tl}")

    for ck, cl in [("rule_out_cutpoint", "rule-out cutpoint"),
                   ("rule_in_cutpoint", "rule-in cutpoint")]:
        v, acc = A.c(ck)
        add(f"card.{ck}", int(v), acc,
            f"Score card: {cl}", "Card", printed=f_signed_int(v))

    v, acc = A.r("anchor_min_positive_var")
    add("card.anchor.variable", str(v), acc,
        "Integer-scale anchor variable (smallest positive screen-passing "
        "point-role coefficient)", "Card", printed=str(v))
    v, acc = A.r("anchor_min_positive_beta")
    add("card.anchor.beta", v, acc,
        "Integer-scale anchor coefficient", "Card", printed=f_dec(v, 4))

    # -------------------------------------------------------- DISCRIMINATION
    for _, r in A.t4.iterrows():
        outcome = str(r["Outcome"])
        score = str(r["Score"])
        row, acc = pick(A.t4, "table4_discrimination.csv",
                        Outcome=outcome, Score=score)
        au = cell(row, "AUROC", "table4_discrimination.csv")
        lo = cell(row, "CI_Low", "table4_discrimination.csv")
        hi = cell(row, "CI_High", "table4_discrimination.csv")
        nu = int(cell(row, "N_used", "table4_discrimination.csv"))
        ne = int(cell(row, "N_events", "table4_discrimination.csv"))
        role = str(cell(row, "Instrument_Role", "table4_discrimination.csv"))
        orole = str(cell(row, "Outcome_Role", "table4_discrimination.csv"))
        atype = str(cell(row, "AUROC_type", "table4_discrimination.csv"))
        base = f"disc.{outcome}.{score}"
        lab = (f"AUROC of {score} for {outcome} ({orole}); instrument role: "
               f"{role}; AUROC type: {atype}")
        add(base + ".auroc", au, acc + ".AUROC", lab, "Discrimination",
            printed=f_dec(au, 4), alt=[f"{au:.3f}"],
            denominator=f"N = {f_int(nu)}")
        if lo is not None and hi is not None:
            add(base + ".auroc_ci", [lo, hi], acc + ".[CI_Low, CI_High]",
                lab + " -- 95% CI", "Discrimination",
                printed=f"{f_dec(lo, 4)}{EN}{f_dec(hi, 4)}")
            add(base + ".auroc_full", au, acc + ".AUROC + .[CI_Low, CI_High]",
                lab + " -- with 95% CI", "Discrimination",
                printed=f_ci(au, lo, hi, 4),
                alt=[f"{au:.3f} ({lo:.3f}{EN}{hi:.3f})"],
                denominator=f"N = {f_int(nu)}")
        add(base + ".n_used", nu, acc + ".N_used",
            lab + " -- analysis N", "Discrimination", printed=f_int(nu))
        add(base + ".n_events", ne, acc + ".N_events",
            lab + " -- events", "Discrimination", printed=f_int(ne),
            denominator=f"N = {f_int(nu)}")

    for _, r in A.t4d.iterrows():
        outcome = str(r["Outcome"])
        s1 = str(r["Score_1"])
        s2 = str(r["Score_2"])
        row, acc = pick(A.t4d, "table4_delong.csv", Outcome=outcome,
                        Score_1=s1, Score_2=s2)
        d = cell(row, "Delta_AUC", "table4_delong.csv")
        lo = cell(row, "CI_Low_Delta", "table4_delong.csv")
        hi = cell(row, "CI_High_Delta", "table4_delong.csv")
        p = cell(row, "DeLong_P", "table4_delong.csv")
        npair = int(cell(row, "N_paired", "table4_delong.csv"))
        crole = str(cell(row, "Comparison_Role", "table4_delong.csv"))
        base = f"delong.{outcome}.{s1}_vs_{s2}"
        lab = (f"DeLong paired comparison {s1} vs {s2} for {outcome} "
               f"({crole})")
        add(base + ".delta", d, acc + ".Delta_AUC",
            lab + " -- AUROC difference", "Discrimination",
            printed=f_dec(d, 4), alt=[f"{d:.3f}"],
            denominator=f"{f_int(npair)} paired")
        add(base + ".delta_ci", [lo, hi],
            acc + ".[CI_Low_Delta, CI_High_Delta]",
            lab + " -- 95% CI for the difference", "Discrimination",
            printed=f"{f_dec(lo, 4)}{EN}{f_dec(hi, 4)}")
        add(base + ".p", p, acc + ".DeLong_P", lab + " -- P value",
            "Discrimination",
            printed=("P < 0.001" if p < 0.001 else f"P = {p:.3f}"))
        add(base + ".n_paired", npair, acc + ".N_paired",
            lab + " -- paired N", "Discrimination", printed=f_int(npair))
        add(base + ".auroc_1", cell(row, "AUROC_1", "table4_delong.csv"),
            acc + ".AUROC_1",
            lab + f" -- AUROC of {s1} on the paired denominator",
            "Discrimination",
            printed=f_dec(cell(row, "AUROC_1", "table4_delong.csv"), 4),
            denominator=f"{f_int(npair)} paired")
        add(base + ".auroc_2", cell(row, "AUROC_2", "table4_delong.csv"),
            acc + ".AUROC_2",
            lab + f" -- AUROC of {s2} on the paired denominator",
            "Discrimination",
            printed=f_dec(cell(row, "AUROC_2", "table4_delong.csv"), 4),
            denominator=f"{f_int(npair)} paired")

    # Calibration of the headline instrument on the test set
    for oc in ["major_hemorrhage_event", "patient_state_composite",
               "death_hemorrhagic"]:
        row, acc = pick(A.t5cal, "table5_calibration.csv",
                        Outcome=oc, Score="pelvic_score")
        for col, key, nd, lbl in [
                ("Brier", "brier", 4, "Brier score"),
                ("CITL", "citl", 4, "calibration-in-the-large"),
                ("Calibration_Slope", "slope", 4, "calibration slope"),
                ("HL_ChiSq", "hl_chisq", 1, "Hosmer-Lemeshow chi-square")]:
            v = cell(row, col, "table5_calibration.csv")
            add(f"disc.{oc}.pelvic_score.{key}", v, acc + f".{col}",
                f"Test-set {lbl} of the integer card for {oc}",
                "Discrimination", printed=f_dec(v, nd))

    # -------------------------------------------------------------- TEMPORAL
    for oc, lbl in [("major_hemorrhage_event", "PRIMARY composite (headline)"),
                    ("death_hemorrhagic", "co-primary mortality-only anchor"),
                    ("any_transfusion",
                     "COVID co-run (any documented transfusion)")]:
        row, acc = pick(A.t4b, "table4b_temporal.csv", outcome=oc)
        base = f"temporal.{oc}"
        for col, key, nd, desc in [
                ("AUROC_integer", "auroc_integer", 4,
                 "temporal-validation AUROC of the INTEGER CARD"),
                ("AUROC", "auroc_continuous", 4,
                 "temporal-validation AUROC of the CONTINUOUS frozen logit"),
                ("CITL", "citl", 4,
                 "temporal calibration-in-the-large offset"),
                ("slope", "slope", 4, "temporal calibration slope"),
                ("Brier", "brier", 4, "temporal Brier score")]:
            v = cell(row, col, "table4b_temporal.csv")
            add(f"{base}.{key}", v, acc + f".{col}",
                f"{desc} for {oc} ({lbl}), derived 2019-2022 and applied "
                "unchanged to 2023-2024", "Temporal",
                printed=f_dec(v, nd), alt=[f"{abs(v):.4f}", f"{v:.3f}"])
        ilo = cell(row, "CI_low_integer", "table4b_temporal.csv")
        ihi = cell(row, "CI_high_integer", "table4b_temporal.csv")
        iau = cell(row, "AUROC_integer", "table4b_temporal.csv")
        add(f"{base}.auroc_integer_ci", [ilo, ihi],
            acc + ".[CI_low_integer, CI_high_integer]",
            f"95% CI for the temporal integer-card AUROC ({oc})", "Temporal",
            printed=f"{f_dec(ilo, 4)}{EN}{f_dec(ihi, 4)}")
        add(f"{base}.auroc_integer_full", iau,
            acc + ".AUROC_integer + .[CI_low_integer, CI_high_integer]",
            f"Temporal integer-card AUROC with 95% CI ({oc}, {lbl})",
            "Temporal", printed=f_ci(iau, ilo, ihi, 4),
            alt=[f"{iau:.3f} ({ilo:.3f}{EN}{ihi:.3f})"])
        clo = cell(row, "CI_low", "table4b_temporal.csv")
        chi = cell(row, "CI_high", "table4b_temporal.csv")
        cau = cell(row, "AUROC", "table4b_temporal.csv")
        add(f"{base}.auroc_continuous_full", cau,
            acc + ".AUROC + .[CI_low, CI_high]",
            f"Temporal CONTINUOUS-LOGIT AUROC with 95% CI ({oc}) -- label as "
            "continuous or omit; never quoted as the score's AUROC",
            "Temporal", printed=f_ci(cau, clo, chi, 4))
        for col, key, desc in [
                ("N_valid", "n_valid", "temporal validation N (2023-2024)"),
                ("events", "events_valid",
                 "events in the temporal validation window"),
                ("N_derive", "n_derive", "temporal derivation N (2019-2022)"),
                ("events_derive", "events_derive",
                 "events in the temporal derivation window")]:
            v = int(cell(row, col, "table4b_temporal.csv"))
            add(f"{base}.{key}", v, acc + f".{col}",
                f"{desc} for {oc}", "Temporal", printed=f_int(v))
        for col, key, desc in [
                ("prevalence_derive", "prevalence_derive",
                 "outcome prevalence in the temporal derivation window"),
                ("prevalence_valid", "prevalence_valid",
                 "outcome prevalence in the temporal validation window")]:
            v = cell(row, col, "table4b_temporal.csv")
            add(f"{base}.{key}", v, acc + f".{col}",
                f"{desc} for {oc}", "Temporal",
                printed=f_prop_pct(v, 2), alt=prop_pct_alts(v))
        add(f"{base}.anchor_var",
            str(cell(row, "integer_anchor_var", "table4b_temporal.csv")),
            acc + ".integer_anchor_var",
            f"Integer-scale anchor variable re-estimated on 2019-2022 ({oc})",
            "Temporal",
            printed=str(cell(row, "integer_anchor_var",
                             "table4b_temporal.csv")))

    for ck, key, nd, desc in [
            ("auroc_integer", "auroc_integer", 4,
             "temporal integer-card AUROC (full precision)"),
            ("auroc_logit", "auroc_logit", 4,
             "temporal continuous-logit AUROC (full precision)"),
            ("citl_offset_intercept", "citl_offset", 4,
             "temporal calibration-in-the-large offset intercept "
             "(< 0 = over-prediction)"),
            ("calibration_slope", "calibration_slope", 4,
             "temporal calibration slope (1.0 = ideal spread)"),
            ("citl_meandiff", "citl_meandiff", 5,
             "temporal mean predicted minus mean observed"),
            ("brier", "brier", 5, "temporal Brier score"),
            ("prevalence_derive", "prevalence_derive", 5,
             "temporal derivation prevalence"),
            ("prevalence_valid", "prevalence_valid", 5,
             "temporal validation prevalence")]:
        v, acc = A.ci(f"composite.{ck}")
        add(f"temporal.citl.{key}", v, acc,
            f"Composite {desc}", "Temporal", printed=f_dec(v, nd),
            alt=[f"{abs(v):.4f}"])
    for ck, key, desc in [
            ("n_derive", "n_derive", "temporal derivation N (2019-2022)"),
            ("events_derive", "events_derive", "temporal derivation events"),
            ("n_valid", "n_valid", "temporal validation N (2023-2024)"),
            ("events_valid", "events_valid", "temporal validation events")]:
        v, acc = A.ci(f"composite.{ck}")
        add(f"temporal.citl.{key}", int(v), acc,
            f"Composite {desc}", "Temporal", printed=f_int(v))
    v, acc = A.ci("n_boot")
    add("temporal.citl.n_boot", int(v), acc,
        "Bootstrap iterations for the temporal AUROC confidence intervals",
        "Temporal", printed=f_int(v))
    for ck, key, desc in [("derivation_years", "derivation_years",
                           "temporal derivation window"),
                          ("validation_years", "validation_years",
                           "temporal validation window")]:
        v, acc = A.ci(ck)
        add(f"temporal.{key}", str(v), acc, desc, "Temporal",
            printed=str(v).replace("-", EN))

    # G5h differential ascertainment
    for split, slabel in [("derivation_2019_2022", "derivation window "
                                                   "(2019-2022)"),
                          ("validation_2023_2024", "validation window "
                                                   "(2023-2024)")]:
        for fk, key, kind, desc in [
                ("n", "n", "int", "N in the"),
                ("n_deaths", "n_deaths", "int", "in-hospital deaths in the"),
                ("deaths_missing_n", "deaths_missing_n", "int",
                 "deaths with no recorded discharge-timing value in the"),
                ("deaths_missing_pct", "deaths_missing_pct", "pct",
                 "percent of deaths with no recorded discharge-timing value "
                 "in the"),
                ("death_hemorrhagic_n", "death_hemorrhagic_n", "int",
                 "hemorrhagic deaths in the"),
                ("composite_rate_pct", "composite_rate_pct", "pct",
                 "composite event rate in the")]:
            v, acc = A.q(
                f"finaldischarge_timing.ascertainment_by_split.{split}.{fk}")
            if kind == "int":
                add(f"temporal.ascertainment.{split}.{key}", int(v), acc,
                    f"G5h differential ascertainment: {desc} {slabel}",
                    "Temporal", printed=f_int(v))
            else:
                add(f"temporal.ascertainment.{split}.{key}", v, acc,
                    f"G5h differential ascertainment: {desc} {slabel}",
                    "Temporal", printed=f_pct(v, 3), alt=pct_alts(v))
    v, acc = A.q("finaldischarge_timing.ascertainment_by_split.differential_pp")
    add("temporal.ascertainment.differential_pp", v, acc,
        "G5h differential ascertainment: derivation-minus-validation gap in "
        "the proportion of deaths lacking a discharge-timing value, in "
        "percentage points -- a calibration offset is partly ascertainment, "
        "not decay", "Temporal", printed=f"{v:.3f} pp",
        alt=[f"{v:.2f} pp", f"{v:.1f} pp", f"+{v:.3f}", f"{v:.3f}"])
    v, acc = A.q("finaldischarge_timing.cohort_fill_pct")
    add("temporal.ascertainment.cohort_fill_pct", v, acc,
        "Discharge-timing field fill rate across the analytic cohort",
        "Temporal", printed=f_pct(v, 3), alt=pct_alts(v))
    v, acc = A.q("finaldischarge_timing.deaths_fill_pct")
    add("temporal.ascertainment.deaths_fill_pct", v, acc,
        "Discharge-timing field fill rate among deaths", "Temporal",
        printed=f_pct(v, 3), alt=pct_alts(v))

    # G5h / D-10 upper-bound re-run.  BOTH arms are the MORTALITY arm, never
    # the composite: the primary arm is `death_hemorrhagic` and the bounding
    # arm `death_hemorrhagic_missing_incl` (every untimed death retained).
    # Every label names the arm and every entry carries the denominator its
    # events sit in, so a sentence quoting one of these figures cannot
    # silently attach it to the composite offset.  Read from the artifact;
    # nothing here is hard-coded.
    DA = "differential_ascertainment_d10"
    arm_den = {
        "primary": ("mortality arm death_hemorrhagic, derived on the "
                    "temporal derivation window 2019-2022 and applied "
                    "unchanged to the temporal validation window 2023-2024"),
        "bounding": ("mortality arm death_hemorrhagic_missing_incl (every "
                     "untimed death retained), derived on the temporal "
                     "derivation window 2019-2022 and applied unchanged to "
                     "the temporal validation window 2023-2024"),
    }
    for arm, alabel in [
            ("primary", "PRIMARY mortality arm death_hemorrhagic (an untimed "
                        "death counted as NOT within 48 h)"),
            ("bounding", "UPPER-BOUND mortality arm "
                         "death_hemorrhagic_missing_incl (every untimed "
                         "death retained)")]:
        oc, acc = A.ci(f"{DA}.{arm}.outcome")
        add(f"temporal.ascertainment.{arm}_outcome", str(oc), acc,
            f"G5h/D-10 upper-bound re-run, {alabel}: outcome variable",
            "Temporal", printed=str(oc), denominator=arm_den[arm])
        for fk, key, nd, desc in [
                ("citl_offset_intercept", "citl_offset", 4,
                 "temporal calibration-in-the-large offset intercept "
                 "(< 0 = over-prediction)"),
                ("abs_citl", "abs_citl", 4,
                 "absolute temporal calibration-in-the-large offset, |CITL|")]:
            v, acc = A.ci(f"{DA}.{arm}.{fk}")
            add(f"temporal.ascertainment.{arm}_{key}", v, acc,
                f"G5h/D-10 upper-bound re-run, {alabel}: {desc}", "Temporal",
                printed=f_dec(v, nd), alt=[f"{abs(v):.4f}"],
                denominator=arm_den[arm])
        for fk, key, desc in [
                ("events_derive", "events_derive",
                 "events in the temporal derivation window (2019-2022)"),
                ("events_valid", "events_valid",
                 "events in the temporal validation window (2023-2024)")]:
            v, acc = A.ci(f"{DA}.{arm}.{fk}")
            add(f"temporal.ascertainment.{arm}_{key}", int(v), acc,
                f"G5h/D-10 upper-bound re-run, {alabel}: {desc}", "Temporal",
                printed=f_int(v), denominator=arm_den[arm])
    v, acc = A.ci(f"{DA}.abs_citl_shrinkage")
    add("temporal.ascertainment.abs_citl_shrinkage", v, acc,
        "G5h/D-10 upper-bound re-run, mortality arm: absolute change in "
        "|CITL| from the primary arm death_hemorrhagic to the upper-bound "
        "arm death_hemorrhagic_missing_incl (negative = |CITL| GREW)",
        "Temporal", printed=f_dec(v, 4), alt=[f"{abs(v):.4f}"],
        denominator=arm_den["bounding"])
    # A FRACTION of the primary arm's |CITL|, NOT a percentage: the artifact
    # stores -0.6868 and the honest rendering multiplies it by 100 exactly
    # once.  C7a closes it against its own numerator and denominator siblings
    # through the abs_citl_shrinkage_relative rule in PCT_RATIO_RULES; C7c
    # independently refuses any second multiplication.
    v, acc = A.ci(f"{DA}.abs_citl_shrinkage_relative")
    add("temporal.ascertainment.abs_citl_shrinkage_relative", v, acc,
        "G5h/D-10 upper-bound re-run, mortality arm: change in |CITL| as a "
        "FRACTION of the primary arm's |CITL| -- negative means the offset "
        "GREW under the upper bound, so the 25% relative-shrinkage rule for "
        "attributing the offset to ascertainment is not met", "Temporal",
        printed=f_signed_prop_pct(v, 1),
        alt=[f_signed_prop_pct(v, 2), f_dec(v, 4)],
        denominator=arm_den["bounding"])
    v, acc = A.ci(f"{DA}.verdict")
    add("temporal.ascertainment.verdict", str(v), acc,
        "G5h/D-10 upper-bound re-run, mortality arm: the verdict recorded in "
        "the artifact on whether the ascertainment differential explains the "
        "temporal calibration offset", "Temporal", printed=str(v),
        denominator=arm_den["bounding"])

    # --------------------------------------------------------------- UTILITY
    for score in ["pelvic_score", "SI-alone"]:
        for rule in ["youden", "sens_90", "spec_90"]:
            row, acc = pick(A.t5op, "table5_operating.csv",
                            Score=score, Cutoff_Rule=rule)
            n = int(cell(row, "N", "table5_operating.csv"))
            den = ("test set" if score == "pelvic_score"
                   else "shock-index-alone operating denominator")
            base = f"util.operating.{score}.{rule}"
            cut = cell(row, "Cutoff", "table5_operating.csv")
            add(base + ".cutoff", cut, acc + ".Cutoff",
                f"Operating point ({rule}) for {score}: cutoff", "Utility",
                printed=(f_signed_int(cut) if score == "pelvic_score"
                         else f_dec(cut, 4)),
                denominator=den)
            add(base + ".n", n, acc + ".N",
                f"Operating point ({rule}) for {score}: analysis N", "Utility",
                printed=f_int(n), denominator=den)
            add(base + ".n_flagged",
                int(cell(row, "N_flagged", "table5_operating.csv")),
                acc + ".N_flagged",
                f"Operating point ({rule}) for {score}: patients flagged",
                "Utility",
                printed=f_int(cell(row, "N_flagged", "table5_operating.csv")),
                denominator=den)
            for est, lo, hi, key, lbl in [
                    ("Sensitivity", "Sens_CI_Low", "Sens_CI_High",
                     "sensitivity", "sensitivity"),
                    ("Specificity", "Spec_CI_Low", "Spec_CI_High",
                     "specificity", "specificity"),
                    ("PPV", "PPV_CI_Low", "PPV_CI_High", "ppv",
                     "positive predictive value"),
                    ("NPV", "NPV_CI_Low", "NPV_CI_High", "npv",
                     "negative predictive value")]:
                e = cell(row, est, "table5_operating.csv")
                l_ = cell(row, lo, "table5_operating.csv")
                h_ = cell(row, hi, "table5_operating.csv")
                add(base + f".{key}", e, acc + f".{est}",
                    f"Operating point ({rule}) for {score}: {lbl}", "Utility",
                    printed=f_prop_pct(e, 3), alt=prop_pct_alts(e),
                    denominator=den)
                add(base + f".{key}_ci", [l_, h_], acc + f".[{lo}, {hi}]",
                    f"Operating point ({rule}) for {score}: {lbl} 95% CI",
                    "Utility",
                    printed=f"{f_prop_pct(l_, 2)}{EN}{f_prop_pct(h_, 2)}",
                    denominator=den)
                add(base + f".{key}_full", e,
                    acc + f".{est} + .[{lo}, {hi}]",
                    f"Operating point ({rule}) for {score}: {lbl} with 95% CI",
                    "Utility", printed=f_ci_prop_pct(e, l_, h_, 2),
                    alt=[f_ci_prop_pct(e, l_, h_, 1)], denominator=den)
            for col, key, nd, lbl in [
                    ("LR_pos", "lr_pos", 3, "positive likelihood ratio"),
                    ("LR_neg", "lr_neg", 3, "negative likelihood ratio"),
                    ("Youden_J", "youden_j", 4, "Youden J"),
                    ("Missed_Event_Rate", "missed_event_rate", 4,
                     "missed-event rate below the cut")]:
                v = cell(row, col, "table5_operating.csv")
                add(base + f".{key}", v, acc + f".{col}",
                    f"Operating point ({rule}) for {score}: {lbl}", "Utility",
                    printed=f_dec(v, nd), denominator=den)
            for col in ["TP", "FP", "TN", "FN"]:
                v = int(cell(row, col, "table5_operating.csv"))
                add(base + f".{col.lower()}", v, acc + f".{col}",
                    f"Operating point ({rule}) for {score}: {col}", "Utility",
                    printed=f_int(v), denominator=den)

    # Matched workload
    for stratum, sl, den_label in [
            ("overall", "whole matched set",
             "matched set (common complete case)"),
            ("elderly_ge65", "elderly matched set",
             "elderly matched set (common complete case)")]:
        for target in ["deployed_rule_in", "fixed_15pct"]:
            for instrument, rule in [("pelvic_score", "score_only"),
                                     ("shock_index", "instrument_only"),
                                     ("rSIG", "instrument_only")]:
                row, acc = pick(A.t5d, "table5d_matched_workload.csv",
                                Stratum=stratum,
                                Workload_target_label=target,
                                Instrument=instrument, Rule=rule)
                base = f"util.matched.{stratum}.{target}.{instrument}"
                lab = (f"Matched workload ({target}, {sl}), {instrument} "
                       f"[{rule}]")
                for col, key, kind, d in [
                        ("N_analysis", "n_analysis", "int", "analysis N"),
                        ("N_events", "n_events", "int", "events"),
                        ("N_flagged", "n_flagged", "int", "patients flagged"),
                        ("Events_caught_TP", "events_caught", "int",
                         "events caught (true positives)"),
                        ("False_positives_FP", "false_positives", "int",
                         "false positives"),
                        ("Events_missed_FN", "events_missed", "int",
                         "events missed (false negatives)")]:
                    v = int(cell(row, col, "table5d_matched_workload.csv"))
                    add(f"{base}.{key}", v, acc + f".{col}",
                        f"{lab}: {d}", "Utility", printed=f_int(v),
                        denominator=den_label)
                for col, key, kind, d in [
                        ("Pct_flagged_actual", "pct_flagged", "pct",
                         "percent of the denominator flagged"),
                        ("Sensitivity", "sensitivity", "prop", "sensitivity"),
                        ("PPV", "ppv", "prop", "positive predictive value")]:
                    v = cell(row, col, "table5d_matched_workload.csv")
                    if kind == "pct":
                        add(f"{base}.{key}", v, acc + f".{col}",
                            f"{lab}: {d}", "Utility", printed=f_pct(v, 4),
                            alt=pct_alts(v), denominator=den_label)
                    else:
                        add(f"{base}.{key}", v, acc + f".{col}",
                            f"{lab}: {d}", "Utility",
                            printed=f_prop_pct(v, 3), alt=prop_pct_alts(v),
                            denominator=den_label)
                add(f"{base}.cutoff_label",
                    str(cell(row, "Cutoff_Label",
                             "table5d_matched_workload.csv")),
                    acc + ".Cutoff_Label", f"{lab}: operating rule", "Utility",
                    printed=str(cell(row, "Cutoff_Label",
                                     "table5d_matched_workload.csv")))
                add(f"{base}.nne",
                    cell(row, "Number_needed_to_escalate",
                         "table5d_matched_workload.csv"),
                    acc + ".Number_needed_to_escalate",
                    f"{lab}: number needed to escalate per event caught",
                    "Utility",
                    printed=f_dec(cell(row, "Number_needed_to_escalate",
                                       "table5d_matched_workload.csv"), 3),
                    denominator=den_label)
                if instrument != "shock_index":
                    dc = int(cell(row, "Delta_comparable",
                                  "table5d_matched_workload.csv"))
                    de = cell(row, "Delta_events_caught",
                              "table5d_matched_workload.csv")
                    df_ = cell(row, "Delta_false_positives",
                               "table5d_matched_workload.csv")
                    add(f"{base}.delta_comparable", dc,
                        acc + ".Delta_comparable",
                        f"{lab}: workload-match comparability guard "
                        "(1 = the delta may be quoted)", "Utility",
                        printed=str(dc))
                    add(f"{base}.delta_events_caught", de,
                        acc + ".Delta_events_caught",
                        f"{lab}: additional events caught versus the "
                        "SHOCK-INDEX reference at matched workload",
                        "Utility", printed=f_signed_int(de),
                        alt=[str(int(abs(de)))], denominator=den_label)
                    add(f"{base}.delta_false_positives", df_,
                        acc + ".Delta_false_positives",
                        f"{lab}: change in false positives versus the "
                        "SHOCK-INDEX reference at matched workload",
                        "Utility", printed=f_signed_int(df_),
                        alt=[str(int(abs(df_)))], denominator=den_label)
                add(f"{base}.reference_instrument",
                    str(cell(row, "Reference_instrument",
                             "table5d_matched_workload.csv")),
                    acc + ".Reference_instrument",
                    f"{lab}: the reference every delta is taken against "
                    "(shock index only -- there is no rSIG reference row, so "
                    "a score-versus-rSIG margin is not quotable)", "Utility",
                    printed=str(cell(row, "Reference_instrument",
                                     "table5d_matched_workload.csv")))

    # Decision curve, overall
    for pt in [0.05, 0.10, 0.15, 0.20]:
        for score in ["pelvic_score", "SI-alone", "rSIG"]:
            row, acc = pick(A.dca, "dca_data.csv", threshold=pt, score=score)
            nb = cell(row, "net_benefit", "dca_data.csv")
            add(f"util.dca.pt{int(round(pt * 100))}.{score}.net_benefit", nb,
                acc + ".net_benefit",
                f"Decision curve: net benefit of {score} at a threshold "
                f"probability of {pt:.2f}", "Utility", printed=f_dec(nb, 5))
        row, acc = pick(A.dca, "dca_data.csv", threshold=pt,
                        score="pelvic_score")
        ta = cell(row, "treat_all_nb", "dca_data.csv")
        add(f"util.dca.pt{int(round(pt * 100))}.treat_all.net_benefit", ta,
            acc + ".treat_all_nb",
            f"Decision curve: net benefit of escalating everyone at a "
            f"threshold probability of {pt:.2f}", "Utility",
            printed=f_dec(ta, 5))

    # ---------------------------------------------------------------- ELDERLY
    for _, r in A.t6b.iterrows():
        arm = str(r["Arm"])
        row, acc = pick(A.t6b, "table6b_elderly_ledger.csv", Arm=arm)
        defn = str(cell(row, "Definition", "table6b_elderly_ledger.csv"))
        base = f"elderly.ledger.{arm}"
        add(base + ".definition", defn, acc + ".Definition",
            f"Elderly ledger arm '{arm}': definition", "Elderly",
            printed=defn)
        nden = int(cell(row, "N_denominator", "table6b_elderly_ledger.csv"))
        add(base + ".n_denominator", nden, acc + ".N_denominator",
            f"Elderly ledger arm '{arm}': denominator ({defn})", "Elderly",
            printed=f_int(nden), denominator=defn)
        for col, key, kind, d in [
                ("N_events_in_denominator", "n_events", "int",
                 "events in the denominator"),
                ("N_flagged", "n_flagged", "int", "patients flagged"),
                ("Flagged_events_TP", "flagged_events", "int",
                 "flagged patients who had an event"),
                ("Flagged_nonevents_FP", "flagged_nonevents", "int",
                 "flagged patients who had no event")]:
            v = cell(row, col, "table6b_elderly_ledger.csv")
            if v is None:
                continue
            add(base + f".{key}", int(v), acc + f".{col}",
                f"Elderly ledger arm '{arm}': {d}", "Elderly",
                printed=(f_signed_int(v) if arm == "net_reclassification"
                         else f_int(v)),
                denominator=defn)
        for col, key, d in [
                ("Event_rate_in_denominator_pct", "event_rate_pct",
                 "event rate in the denominator"),
                ("Flagged_pct_of_denominator", "flagged_pct",
                 "percent of the denominator flagged"),
                ("PPV_within_flagged_pct", "ppv_pct",
                 "positive predictive value within the flagged group"),
                ("Escalations_per_event", "escalations_per_event",
                 "escalations per event")]:
            v = cell(row, col, "table6b_elderly_ledger.csv")
            if v is None:
                continue
            if key == "escalations_per_event":
                add(base + f".{key}", v, acc + f".{col}",
                    f"Elderly ledger arm '{arm}': {d}", "Elderly",
                    printed=f_dec(v, 3), alt=[f"{v:.2f}", f"{v:.1f}"],
                    denominator=defn)
            else:
                add(base + f".{key}", v, acc + f".{col}",
                    f"Elderly ledger arm '{arm}': {d}", "Elderly",
                    printed=f_pct(v, 3), alt=pct_alts(v), denominator=defn)
        for col, key, d in [("PPV_CI_low_pct", "ppv_ci_low",
                             "PPV 95% CI lower bound"),
                            ("PPV_CI_high_pct", "ppv_ci_high",
                             "PPV 95% CI upper bound")]:
            v = cell(row, col, "table6b_elderly_ledger.csv")
            if v is None:
                continue
            add(base + f".{key}", v, acc + f".{col}",
                f"Elderly ledger arm '{arm}': {d}", "Elderly",
                printed=f_pct(v, 3), alt=pct_alts(v), denominator=defn)

    for ek, kind, desc in [
            ("elderly_test_n", "int", "elderly patients in the test set"),
            ("elderly_test_events", "int",
             "composite events among elderly test patients"),
            ("elderly_test_prevalence", "prop",
             "composite event rate among elderly test patients")]:
        v, acc = A.rc(ek)
        if kind == "int":
            add(f"elderly.{ek}", int(v), acc,
                f"Elderly: {desc}", "Elderly", printed=f_int(v),
                denominator="elderly test set")
        else:
            add(f"elderly.{ek}", v, acc, f"Elderly: {desc}", "Elderly",
                printed=f_prop_pct(v, 2), alt=prop_pct_alts(v),
                denominator="elderly test set")

    for ek, key, nd, desc in [
            ("citl", "citl", 4, "calibration-in-the-large"),
            ("slope", "slope", 4, "calibration slope"),
            ("mean_predicted", "mean_predicted", 4, "mean predicted risk"),
            ("mean_observed", "mean_observed", 4, "mean observed risk")]:
        v, acc = A.rc(f"elderly_calibration.{ek}")
        add(f"elderly.calibration.{key}", v, acc,
            f"Elderly test-set {desc} of the score", "Elderly",
            printed=f_dec(v, nd), denominator="elderly test set")
    v, acc = A.rc("elderly_calibration.verdict")
    add("elderly.calibration.verdict", str(v), acc,
        "Elderly calibration verdict recorded by the pipeline", "Elderly",
        printed=str(v))

    for ek, key, kind, desc in [
            ("n_frame", "n_frame", "int",
             "elderly test patients with an observed shock index"),
            ("n_excluded_unobtainable_si", "n_excluded_unobtainable_si", "int",
             "elderly test patients excluded because the shock index was "
             "unobtainable"),
            ("n_si_reassuring_lt_0_9", "n_si_reassuring", "int",
             "elderly with a reassuring shock index (< 0.9)"),
            ("n_si_alarming_ge_0_9", "n_si_alarming", "int",
             "elderly with an alarming shock index (>= 0.9)"),
            ("n_frame_events", "n_frame_events", "int",
             "composite events in the elderly shock-index frame")]:
        v, acc = A.rc(f"elderly_ledger.{ek}")
        add(f"elderly.frame.{key}", int(v), acc, f"Elderly: {desc}",
            "Elderly", printed=f_int(v))

    for uk, key, kind, desc in [
            ("count", "count", "int",
             "elderly composite events with a reassuring shock index that the "
             "score flags for escalation"),
            ("si_missed_bleeders(SI<0.9)", "si_missed_bleeders", "int",
             "elderly composite events the shock-index heuristic misses"),
            ("frac_of_si_missed_bleeders_recaptured", "frac_recaptured",
             "prop",
             "fraction of shock-index-missed elderly bleeders the score "
             "recaptures"),
            ("frac_of_all_elderly_events", "frac_of_all_events", "prop",
             "fraction of all elderly events recaptured"),
            ("per_1000_elderly_test", "per_1000", "float",
             "recaptured events per 1,000 elderly test patients")]:
        if "." in uk:
            v, acc = A.rck("undertriage", uk)
        else:
            v, acc = A.rc(f"undertriage.{uk}")
        if kind == "int":
            add(f"elderly.undertriage.{key}", int(v), acc,
                f"Elderly undertriage: {desc}", "Elderly", printed=f_int(v),
                denominator="elderly test set")
        elif kind == "prop":
            add(f"elderly.undertriage.{key}", v, acc,
                f"Elderly undertriage: {desc}", "Elderly",
                printed=f_prop_pct(v, 1), alt=prop_pct_alts(v))
        else:
            add(f"elderly.undertriage.{key}", v, acc,
                f"Elderly undertriage: {desc}", "Elderly",
                printed=f_dec(v, 2))
    v, acc = A.rc("undertriage.definition")
    add("elderly.undertriage.definition", str(v), acc,
        "Elderly undertriage: definition", "Elderly", printed=str(v))

    for nk, key, kind, desc in [
            ("nri", "nri", "prop", "categorical net reclassification index"),
            ("nri_event", "nri_event", "prop", "event-arm NRI component"),
            ("nri_nonevent", "nri_nonevent", "prop",
             "non-event-arm NRI component"),
            ("nri_ci_low", "nri_ci_low", "prop", "NRI 95% CI lower bound"),
            ("nri_ci_high", "nri_ci_high", "prop", "NRI 95% CI upper bound"),
            ("n_event", "n_event", "int", "events in the NRI frame"),
            ("n_nonevent", "n_nonevent", "int", "non-events in the NRI frame"),
            ("event_up", "event_up", "int",
             "events reclassified upward by the score"),
            ("event_down", "event_down", "int",
             "events reclassified downward by the score"),
            ("nonevent_up", "nonevent_up", "int",
             "non-events reclassified upward by the score"),
            ("nonevent_down", "nonevent_down", "int",
             "non-events reclassified downward by the score"),
            ("n_excluded_missing_si", "n_excluded_missing_si", "int",
             "elderly excluded from the NRI frame for a missing shock index")]:
        v, acc = A.rc(f"nri.{nk}")
        if kind == "int":
            add(f"elderly.nri.{key}", int(v), acc,
                f"Elderly reclassification: {desc}", "Elderly",
                printed=f_int(v),
                denominator="elderly with an observed shock index")
        else:
            add(f"elderly.nri.{key}", v, acc,
                f"Elderly reclassification: {desc}", "Elderly",
                printed=f_dec(v, 4), alt=prop_pct_alts(v),
                denominator="elderly with an observed shock index")
    v, acc = A.rc("nri.nri")
    lo, _ = A.rc("nri.nri_ci_low")
    hi, _ = A.rc("nri.nri_ci_high")
    add("elderly.nri.nri_full", v, "reclass.nri.nri + .nri_ci_low + .nri_ci_high",
        "Elderly reclassification: categorical NRI with 1,000-bootstrap 95% CI",
        "Elderly", printed=f_ci(v, lo, hi, 4),
        alt=[f"{v:.3f} ({lo:.3f}{EN}{hi:.3f})"],
        denominator="elderly with an observed shock index")

    for gk, key, nd, desc in [
            ("gain_over_si_at_rule_in", "gain_over_si", 5,
             "net-benefit gain over the shock index at the rule-in threshold"),
            ("net_tp_per_1000_score_at_rule_in", "net_tp_per_1000", 3,
             "net true positives per 1,000 elderly at the rule-in threshold"),
            ("net_tp_per_1000_gain_over_si_at_rule_in",
             "net_tp_per_1000_gain", 3,
             "net true positives per 1,000 elderly gained over the shock "
             "index at the rule-in threshold")]:
        v, acc = A.rc(f"elderly_dca_gain.{gk}")
        add(f"elderly.dca.{key}", v, acc,
            f"Elderly decision curve: {desc}", "Elderly", printed=f_dec(v, nd),
            alt=[f"{v:.2f}", f"{v:.1f}"])
    for gk, key, desc in [("n_score", "n_score",
                           "elderly denominator for the score decision curve"),
                          ("n_si", "n_si",
                           "elderly denominator for the shock-index decision "
                           "curve")]:
        v, acc = A.rc(f"elderly_dca_gain.{gk}")
        add(f"elderly.dca.{key}", int(v), acc,
            f"Elderly decision curve: {desc}", "Elderly", printed=f_int(v))

    # ------------------------------------------------------------------ NULL
    for ik, key, nd, desc in [
            ("or", "or", 4, "age >= 65 x anticoagulation interaction odds "
                            "ratio for the composite"),
            ("ci_low", "ci_low", 4, "interaction odds ratio 95% CI lower"),
            ("ci_high", "ci_high", 4, "interaction odds ratio 95% CI upper"),
            ("p", "p", 3, "interaction P value"),
            ("beta", "beta", 4, "interaction log-odds coefficient"),
            ("se", "se", 4, "interaction standard error"),
            ("or_age_ge65_main", "or_age_main", 4,
             "main-effect odds ratio for age >= 65"),
            ("or_cc_anticoag_main", "or_anticoag_main", 4,
             "main-effect odds ratio for anticoagulation")]:
        v, acc = A.rc(f"interaction_or.{ik}")
        add(f"null.interaction.{key}", v, acc,
            f"Pre-registered effect-modification test: {desc}", "Null",
            printed=(f"{v:.3f}" if key == "p" else f_dec(v, nd)),
            alt=[f"{v:.2f}"])
    io, acc_io = A.rc("interaction_or.or")
    ilo, _ = A.rc("interaction_or.ci_low")
    ihi, _ = A.rc("interaction_or.ci_high")
    ip, _ = A.rc("interaction_or.p")
    add("null.interaction.or_full", io,
        "reclass.interaction_or.or + .ci_low + .ci_high + .p",
        "Pre-registered effect-modification test: interaction odds ratio with "
        "95% CI and P value -- reported at FULL precision as Results "
        "material, not softened into the Discussion", "Null",
        printed=f"{f_dec(io, 4)} ({f_dec(ilo, 4)}{EN}{f_dec(ihi, 4)}), "
                f"P = {ip:.3f}",
        alt=[f"{io:.2f} ({ilo:.2f}{EN}{ihi:.2f}), P = {ip:.3f}"],
        denominator="full analytic cohort")
    v, acc = A.rc("interaction_or.n")
    add("null.interaction.n", int(v), acc,
        "Pre-registered effect-modification test: analysis N", "Null",
        printed=f_int(v), denominator="full analytic cohort")
    v, acc = A.rc("interaction_or.adjustment")
    add("null.interaction.adjustment", list(v), acc,
        "Pre-registered effect-modification test: adjustment set", "Null",
        printed=", ".join(v))

    for ik, key, nd, desc in [
            ("or", "or", 4, "interaction odds ratio for the treatment-free "
                            "patient-state composite"),
            ("ci_low", "ci_low", 4, "95% CI lower"),
            ("ci_high", "ci_high", 4, "95% CI upper"),
            ("p", "p", 3, "P value")]:
        v, acc = A.rc(f"interaction_or_patient_state_composite.{ik}")
        add(f"null.interaction_patient_state.{key}", v, acc,
            f"Anti-circularity effect-modification test: {desc}", "Null",
            printed=(f"{v:.3f}" if key == "p" else f_dec(v, nd)),
            alt=[f"{v:.2f}"])

    for stratum, slug, sl in [
            ("young(<65)", "young", "patients under 65"),
            ("elderly/no-AC", "elderly_no_ac",
             "patients 65 or older not on anticoagulation"),
            ("elderly/AC", "elderly_ac",
             "patients 65 or older on anticoagulation"),
            ("elderly(all)", "elderly_all", "all patients 65 or older")]:
        v, acc = A.rc(f"stratum_rates.{stratum}.n")
        add(f"null.stratum.{slug}.n", int(v), acc,
            f"Stratified rates: N among {sl}", "Null", printed=f_int(v),
            denominator="full analytic cohort")
        for mk, mkey, mdesc in [
                ("composite", "composite", "major hemorrhage event rate"),
                ("hc_receipt", "hc_receipt",
                 "therapeutic hemorrhage-control receipt rate"),
                ("death", "death", "hemorrhagic-death rate"),
                ("transfusion", "transfusion", "substantial-transfusion rate"),
                ("death_allcause", "death_allcause",
                 "all-cause in-hospital death rate (named sensitivity)")]:
            v, acc = A.rc(f"stratum_rates.{stratum}.{mk}")
            add(f"null.stratum.{slug}.{mkey}", v, acc,
                f"Stratified rates: {mdesc} among {sl}", "Null",
                printed=f_prop_pct(v, 2), alt=prop_pct_alts(v),
                denominator=f"{sl}")

    v, acc = A.rc("treatment_gap.hc_ratio_young_over_elderly")
    add("null.treatment_gap.hc_ratio", v, acc,
        "Ratio of hemorrhage-control receipt, patients under 65 versus "
        "65 or older", "Null", printed=f_dec(v, 3),
        alt=[f"{v:.2f}", f"{v:.1f}"])

    row, acc = pick(A.t3, "table3_points.csv", Variable="cc_anticoag")
    cc_or = cell(row, "OR", "table3_points.csv")
    cc_lo = cell(row, "95% CI lower", "table3_points.csv")
    cc_hi = cell(row, "95% CI upper", "table3_points.csv")
    add("null.cc_anticoag.or_full", cc_or,
        acc + ".OR + .[95% CI lower, 95% CI upper]",
        "Anticoagulation on the arrival card: odds ratio with 95% CI, scoring "
        "ZERO points and retained as a pre-registered anchor -- a card missing "
        "a pre-registered anchor FAILS", "Null",
        printed=f"{f_dec(cc_or, 3)} ({f_dec(cc_lo, 3)}{EN}{f_dec(cc_hi, 3)})",
        alt=[f"{cc_or:.2f} ({cc_lo:.2f}{EN}{cc_hi:.2f})"])
    add("null.cc_anticoag.points",
        int(cell(row, "Points", "table3_points.csv")), acc + ".Points",
        "Anticoagulation on the arrival card: integer points", "Null",
        printed="0")

    # ------------------------------------------------------------ METHODS-ONLY
    v, acc = A.q("run_seed")
    add("methods.seed", int(v), acc, "Random seed used throughout the pipeline",
        "Methods-only", printed=str(int(v)))
    v, acc = A.q("version")
    add("methods.pipeline_version", str(v), acc,
        "Analysis pipeline version", "Methods-only", printed=str(v))
    add("methods.ring_prefixes",
        ["S32.1", "S32.3", "S32.5", "S32.6", "S32.81", "S32.82"],
        "01_build_cohort.py:232 RING_PREFIXES (mirrored in "
        "cohort_exclusions.json rung 3 and table_consort.csv Step 3)",
        "Pelvic-ring inclusion: ICD-10-CM DIAGNOSIS code prefixes, "
        "prefix-scanned on ICDDIAGNOSISCODE (diagnosis codes -- NOT procedure "
        "codes, not CPT, not ICD-10-PCS)", "Methods-only",
        printed="S32.1/.3/.5/.6/.81/.82")
    add("methods.rsig_formula", "rsig = (SBP / HR) * GCS",
        "06_score_test.py:460 (rsig = (sbp2 / pulse2) * gcs2)",
        "Reverse shock index multiplied by the Glasgow Coma Scale (rSIG): "
        "systolic pressure divided by heart rate, multiplied by GCS",
        "Methods-only",
        printed="rSIG = systolic pressure \u00f7 heart rate \u00d7 Glasgow "
                "Coma Scale")
    add("methods.hc_definition",
        "surgical hemorrhage control (8 operative categories) OR therapeutic "
        "angioembolization OR angiographic stenting; diagnostic angiography "
        "alone did not qualify",
        "01_build_cohort.py:276-286 (HC_SURG_SET, ANGIO_THERAPEUTIC) + "
        "unknown_hc_codes.csv",
        "Therapeutic hemorrhage control -- the outcome component definition "
        "(PI comment F3)", "Methods-only",
        printed="eight operative categories, therapeutic angioembolization or "
                "angiographic stenting; diagnostic angiography alone did not "
                "qualify")

    n_cand = len(A.t2stab)
    add("methods.lasso.n_candidates", n_cand,
        "table2_lasso_stability.csv row count",
        "Candidate arrival predictors entered into the LASSO stability "
        "selection", "Methods-only", printed=str(n_cand))
    n_sel = len(A.t2sel)
    add("methods.lasso.n_selected", n_sel,
        "table2_lasso_selected.csv row count",
        "Predictors selected by LASSO with bootstrap stability", "Methods-only",
        printed=str(n_sel))
    min_freq = float(A.t2sel["Selection_Frequency"].min())
    add("methods.lasso.min_selection_frequency", min_freq,
        "table2_lasso_selected.csv.Selection_Frequency (minimum)",
        "Lowest bootstrap selection frequency among the selected predictors",
        "Methods-only", printed=f_dec(min_freq, 2))
    v, acc = A.r("variables_kept")
    add("methods.model.n_variables_kept", len(v), acc + " (length)",
        "Predictors retained in the fitted arrival model (14 point-scoring "
        "plus 1 override)", "Methods-only", printed=str(len(v)))
    v, acc = A.r("forced_in_anchors")
    add("methods.model.forced_in_anchors", list(v), acc,
        "Pre-registered force-in anchors retained regardless of coefficient "
        "magnitude", "Methods-only", printed=", ".join(v))
    v, acc = A.r("variables_dropped_beta")
    add("methods.model.dropped_by_beta_screen", list(v), acc,
        "Predictors dropped by the coefficient-magnitude screen", "Methods-only",
        printed=", ".join(v) if v else "none")
    v, acc = A.r("anchor_screen_cutoff")
    add("methods.model.beta_screen_cutoff", v, acc,
        "Coefficient-magnitude screen applied before integer conversion",
        "Methods-only", printed=f_dec(v, 2))
    v, acc = A.r("anchor_selection_rule")
    add("methods.model.anchor_rule", str(v), acc,
        "Integer-scale anchor selection rule (SAP deviation D-11)",
        "Methods-only", printed=str(v))
    v, acc = A.r("intercept")
    add("methods.model.intercept", v, acc,
        "Intercept of the frozen arrival logistic model", "Methods-only",
        printed=f_dec(v, 4))

    # D-11 block
    for dk, key, nd, desc in [
            ("model_identity.max_abs_delta_beta", "max_abs_delta_beta", 4,
             "largest absolute change in any coefficient across the D-11 "
             "re-anchoring"),
            ("model_identity.intercept_delta", "intercept_delta", 4,
             "change in the intercept across the D-11 re-anchoring"),
            ("model_identity.pelvic_logit_auroc_pre", "logit_auroc_pre", 4,
             "continuous-logit AUROC before D-11"),
            ("model_identity.pelvic_logit_auroc_post", "logit_auroc_post", 4,
             "continuous-logit AUROC after D-11"),
            ("card_rescaling.pelvic_score_auroc_pre", "card_auroc_pre", 4,
             "integer-card AUROC before D-11"),
            ("card_rescaling.pelvic_score_auroc_post", "card_auroc_post", 4,
             "integer-card AUROC after D-11"),
            ("card_rescaling.pelvic_score_auroc_delta", "card_auroc_delta", 4,
             "change in the integer-card AUROC caused by D-11 -- the number is "
             "PRINTED; 'discrimination was unchanged' is forbidden shorthand"),
            ("auroc_tolerance", "auroc_tolerance", 3,
             "pre-specified AUROC tolerance for the D-11 invariance gate")]:
        v, acc = A.d(dk)
        add(f"methods.d11.{key}", v, acc, f"D-11 invariance check: {desc}",
            "Methods-only", printed=f_dec(v, nd),
            alt=[f"{abs(v):.4f}", f"{v:.3f}"])
    v, acc = A.d("model_identity.n_coefficients")
    add("methods.d11.n_coefficients", int(v), acc,
        "D-11 invariance check: number of coefficients compared", "Methods-only",
        printed=str(int(v)))
    v, acc = A.d("card_rescaling.theoretical_point_sum_post")
    add("methods.d11.theoretical_point_sum_post", list(v), acc,
        "D-11 invariance check: theoretical point-sum range after "
        "re-anchoring", "Methods-only",
        printed=f"{f_signed_int(v[0])} to {f_signed_int(v[1])}")
    v, acc = A.d("card_rescaling.pelvic_score_ci_post")
    add("methods.d11.card_auroc_ci_post", list(v), acc,
        "D-11 invariance check: 95% CI for the post-D-11 integer-card AUROC",
        "Methods-only", printed=f"{f_dec(v[0], 4)}{EN}{f_dec(v[1], 4)}")
    v, acc = A.d("verdict")
    add("methods.d11.verdict", str(v), acc,
        "D-11 invariance check (gate G8f): verdict", "Methods-only",
        printed=str(v))

    sent, acc = A.d("manuscript_sentence")
    add("methods.d11.manuscript_sentence_ascii", str(sent), acc,
        "D-11 manuscript sentence, VERBATIM as the artifact stores it (ASCII)",
        "Methods-only", printed=str(sent))
    norm = normalize_d11_sentence(str(sent))
    add("methods.d11.manuscript_sentence_normalized", norm,
        acc + " under exactly two recorded typographic normalizations",
        "D-11 manuscript sentence after EXACTLY TWO recorded typographic "
        "normalizations: '--' to an en dash, and '|delta beta|' to '|\u0394\u03b2|'. "
        "Never paraphrased; nothing else altered.", "Methods-only",
        printed=norm)
    add("methods.d11.normalizations",
        ["-- \u2192 \u2013", "|delta beta| \u2192 |\u0394\u03b2|"],
        acc + " (normalization register)",
        "The only two transformations permitted on the D-11 sentence, logged "
        "as normalizations in the deviation register", "Methods-only",
        printed="'--' \u2192 '\u2013'; '|delta beta|' \u2192 '|\u0394\u03b2|'")

    v, acc = A.q("gates.primary_outcome_definition")
    add("methods.outcome.primary_definition", str(v), acc,
        "Primary composite outcome definition as recorded by the cohort build",
        "Methods-only", printed=str(v))
    v, acc = A.rc("outcome_columns.composite_nohc")
    add("methods.outcome.treatment_free_co_primary", str(v), acc,
        "Treatment-free co-primary outcome column (anti-circularity design)",
        "Methods-only", printed=str(v))
    v, acc = A.c("prevalence")
    add("methods.dca_prevalence_test", v, acc,
        "Composite prevalence observed in the test set", "Methods-only",
        printed=f_prop_pct(v, 3), alt=prop_pct_alts(v))
    v, acc = A.c("dca_prevalence_anchor")
    add("methods.dca_prevalence_anchor", v, acc,
        "Prevalence anchor used for the decision curve, read from the cohort "
        "manifest rather than hard-coded", "Methods-only",
        printed=f_prop_pct(v, 3), alt=prop_pct_alts(v))

    return B


def normalize_d11_sentence(ascii_sentence):
    """Apply EXACTLY the two recorded typographic normalizations, and nothing
    else.  Both must be present exactly once, or the normalization is refused
    (a normalization that cannot be evaluated is a FAIL, never a skip)."""
    if ascii_sentence.count("--") != 1:
        raise AssertionError(
            "D-11 normalization: expected exactly one '--' in "
            f"manuscript_sentence, found {ascii_sentence.count('--')}")
    if ascii_sentence.count("|delta beta|") != 1:
        raise AssertionError(
            "D-11 normalization: expected exactly one '|delta beta|' in "
            f"manuscript_sentence, found "
            f"{ascii_sentence.count('|delta beta|')}")
    out = ascii_sentence.replace("--", "\u2013")
    out = out.replace("|delta beta|", "|\u0394\u03b2|")
    # Confirm nothing else moved: undoing the two substitutions must restore
    # the original byte-for-byte.
    back = out.replace("|\u0394\u03b2|", "|delta beta|").replace("\u2013", "--")
    if back != ascii_sentence:
        raise AssertionError(
            "D-11 normalization altered more than the two recorded "
            "transformations")
    return out


# ============================================================================
# SERIALISATION
# ============================================================================
def bundle_payload(A, B):
    gate_rows = run_gates(A, B)
    dens = nine_denominators(A)
    payload = OrderedDict()
    payload["_meta"] = OrderedDict([
        ("generated_by", "analysis/build_bundle.py"),
        ("generated_utc",
         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("seed", RANDOM_SEED),
        ("project_root", "."),
        ("rule", "ANYTHING ABSENT FROM THIS BUNDLE CANNOT APPEAR IN THE "
                 "MANUSCRIPT."),
        ("authority_precedence", AUTHORITY_PRECEDENCE),
        ("accessor_legend", ACCESSOR_LEGEND),
        ("manifest_counts", OrderedDict([
            ("tables", len(REQUIRED_TABLES)),
            ("figures", len(REQUIRED_FIGURES)),
            ("derived", len(REQUIRED_DERIVED)),
            ("logs", len(REQUIRED_LOGS)),
        ])),
        ("n_quantities", len(B.q)),
        ("sections", SECTION_ORDER),
        ("notes", [
            "Every value is re-derived from a v3 artifact; nothing is "
            "hard-coded.",
            "Every dataframe lookup is a boolean mask on named columns; a mask "
            "that does not select exactly one row raises.",
            "The score-versus-rSIG matched-workload margin does not exist in "
            "any artifact and is not quotable, computed, or lifted from "
            "correspondence.",
            "Every quoted AUROC is the integer card unless the accessor says "
            "continuous.",
            "The nine denominators are never merged; each carries its own "
            "label.",
        ]),
    ])
    payload["freshness_gates"] = [
        OrderedDict([("gate", g), ("passed", ok), ("detail", msg)])
        for (g, ok, msg) in gate_rows
    ]
    payload["denominators"] = OrderedDict(
        (k, OrderedDict([("value", v), ("printed", f_int(v)),
                         ("accessor", acc), ("label", lab)]))
        for k, (v, acc, lab) in dens.items())
    payload["quantities"] = B.q
    return payload


def write_json(payload):
    os.makedirs(V4_BUNDLE, exist_ok=True)
    txt = json.dumps(payload, indent=1, ensure_ascii=False)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    return txt


LEDGER_SECTION_BLURB = {
    "Cohort": "Participant flow, the analytic cohort, the composite outcome "
              "and its three components, hemorrhage-control routes and types, "
              "baseline characteristics, and the random split.",
    "Card": "The 14-row arrival point card, the automatic-escalation override "
            "(which carries no points), the score range, the risk tiers and "
            "the two cutpoints.",
    "Discrimination": "Test-set AUROCs for every instrument and outcome, the "
                      "DeLong paired comparisons, and test-set calibration. "
                      "Only the pelvic_score integer_bedside row is quotable "
                      "as 'the score's AUROC'.",
    "Temporal": "Derivation on 2019-2022, validation on 2023-2024, plus the "
                "mandatory G5h differential-ascertainment block that any "
                "drift claim must carry.",
    "Utility": "Operating characteristics at the three named cutpoints, the "
               "matched-workload comparison against the shock index, and the "
               "decision curve.",
    "Elderly": "The elderly ledger (both directions), reclassification, "
               "calibration and the elderly decision curve.",
    "Null": "The pre-registered effect-modification test and the stratified "
            "rates, reported at equal precision as Results material.",
    "Methods-only": "Design constants, definitions, the modelling rule, and "
                    "the D-11 invariance block.",
}


def write_ledger(A, payload, B):
    dens = payload["denominators"]
    lines = []
    lines.append("# NUMBERS LEDGER \u2014 v4 Pelvic-Fracture Arrival Score")
    lines.append("")
    lines.append("**Anything absent from this ledger cannot appear in the "
                 "manuscript.**")
    lines.append("")
    lines.append(f"Generated by `v4/code/build_bundle.py` on "
                 f"{payload['_meta']['generated_utc']} \u00b7 seed "
                 f"{payload['_meta']['seed']} \u00b7 "
                 f"{payload['_meta']['n_quantities']} quantities.")
    lines.append("")
    lines.append("Every value below is re-derived from a v3 artifact at build "
                 "time; nothing is hard-coded. The machine-readable twin is "
                 "`v4/bundle/results_bundle.json`; "
                 "`python3 v4/code/build_bundle.py --verify` re-derives every "
                 "row and fails on any drift.")
    lines.append("")
    lines.append("**Authority precedence.** "
                 + " \u2192 ".join(f"`{p}`" for p in AUTHORITY_PRECEDENCE)
                 + ".")
    lines.append("")
    lines.append("**Accessor legend.**")
    lines.append("")
    lines.append("| accessor prefix | file |")
    lines.append("|---|---|")
    for pref, path in ACCESSOR_LEGEND.items():
        lines.append(f"| `{pref}` | `{path}` |")
    lines.append("")

    # -- the nine denominators ------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## The nine denominators")
    lines.append("")
    lines.append("Plan \u00a7C3. These nine are **never merged**. Every "
                 "sentence that uses one names it. A number quoted without its "
                 "denominator label is a gate failure, not a style issue.")
    lines.append("")
    lines.append("| N | label | accessor |")
    lines.append("|---:|---|---|")
    for k, d in dens.items():
        lines.append(f"| **{d['printed']}** | {d['label']} | "
                     f"`{d['accessor']}` |")
    lines.append("")
    lines.append("Verified against the artifacts on this run. The plan asserts "
                 "the same nine values; the artifact is authoritative, and the "
                 "C6 gate fails loudly if the two ever disagree.")
    lines.append("")

    # -- freshness gates -------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## Freshness and closure gates")
    lines.append("")
    lines.append("Every gate below is a hard raise. A gate that cannot be "
                 "evaluated is a FAIL, never a skip; none sits inside a broad "
                 "`except`.")
    lines.append("")
    lines.append("| gate | verdict | what it asserts |")
    lines.append("|---|---|---|")
    for g in payload["freshness_gates"]:
        verdict = "PASS" if g["passed"] else "FAIL"
        lines.append(f"| `{g['gate']}` | {verdict} | "
                     f"{md_escape(g['detail'])} |")
    lines.append("")

    # -- quantities by section -------------------------------------------
    for section in SECTION_ORDER:
        rows = [(k, e) for k, e in payload["quantities"].items()
                if e["section"] == section]
        if not rows:
            continue
        lines.append("---")
        lines.append("")
        lines.append(f"## {section}")
        lines.append("")
        lines.append(LEDGER_SECTION_BLURB.get(section, ""))
        lines.append("")
        lines.append(f"{len(rows)} quantities.")
        lines.append("")
        lines.append("| printed value | accessor | label / denominator |")
        lines.append("|---|---|---|")
        for key, e in rows:
            den = f" \u00b7 *denominator:* {e['denominator']}" \
                if e.get("denominator") else ""
            lines.append(
                f"| **{md_escape(e['printed'])}** | `{e['accessor']}` "
                f"<br/>`{key}` | {md_escape(e['label'])}{md_escape(den)} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Standing prohibitions carried by this ledger")
    lines.append("")
    lines.append("1. Every quoted AUROC is the **integer card**. The "
                 "continuous-logit values are labelled as continuous or "
                 "omitted, and the deployed-rule row is a separate diagnostic "
                 "\u2014 never \"the score's AUROC\".")
    lines.append("2. **No hand-differencing.** Every matched-workload delta in "
                 "this ledger is taken against the shock-index reference the "
                 "artifact carries. A score-versus-rSIG margin exists in no "
                 "artifact and may not be quoted, computed, or lifted from "
                 "correspondence.")
    lines.append("3. **Denominators never merge.** Each of the nine gets its "
                 "own label in the sentence that uses it.")
    lines.append("4. The override is *\"no signs of life on arrival "
                 "(unrecordable systolic pressure)\"* \u2014 never a systolic "
                 "value of zero, never a missingness framing.")
    lines.append("5. `cc_anticoag` stays on the card at 0 points; there is no "
                 "`unstable_ring` row anywhere on the card.")
    lines.append("6. The null is reported at equal precision, in the Results.")
    lines.append("7. The D-11 sentence is quoted under exactly two recorded "
                 "typographic normalizations and never paraphrased; the "
                 "integer card cost is printed, and \"discrimination was "
                 "unchanged\" is forbidden.")
    lines.append("")
    txt = "\n".join(lines) + "\n"
    os.makedirs(V4_BUNDLE, exist_ok=True)
    with open(OUT_LEDGER, "w", encoding="utf-8") as fh:
        fh.write(txt)
    return txt, sum(1 for k, e in payload["quantities"].items()) + len(dens)


def md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


# ============================================================================
# VERIFY
# ============================================================================
def values_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        denom = max(abs(float(a)), abs(float(b)), 1.0)
        return abs(float(a) - float(b)) / denom < 1e-12
    return a == b


def verify():
    print("=" * 78)
    print("VERIFY  --  build_bundle.py --verify")
    print("=" * 78)
    if not os.path.exists(OUT_JSON):
        print(f"CHECK bundle_present : NOT EVALUATED \u2014 FAIL "
              f"({OUT_JSON} does not exist; run the build first)")
        return 1
    with open(OUT_JSON, "r", encoding="utf-8") as fh:
        on_disk = json.load(fh)

    # Re-derive everything from the artifacts.  This is deliberately OUTSIDE
    # any try/except: a failure here must crash loudly, not be swallowed.
    A = Art()
    B = build_all(A)
    gate_rows = run_gates(A, B)
    dens = nine_denominators(A)

    rows = []      # (check_id, verdict, detail)
    n_fail = 0

    def record(cid, ok, detail, evaluated=True):
        nonlocal n_fail
        if not evaluated:
            verdict = "NOT EVALUATED \u2014 FAIL"
            n_fail += 1
        elif ok:
            verdict = "PASS"
        else:
            verdict = "FAIL"
            n_fail += 1
        rows.append((cid, verdict, detail))

    # -- gates -----------------------------------------------------------
    for gid, ok, msg in gate_rows:
        record(f"gate.{gid}", ok, msg)

    # -- nine denominators ----------------------------------------------
    disk_dens = on_disk.get("denominators")
    if disk_dens is None:
        record("denominators.block", False,
               "results_bundle.json has no 'denominators' block",
               evaluated=False)
    else:
        for k, (val, acc, lab) in dens.items():
            if k not in disk_dens:
                record(f"denominator.{k}", False,
                       "absent from the emitted bundle", evaluated=False)
                continue
            d = disk_dens[k]
            ok = (d.get("value") == val and d.get("accessor") == acc
                  and d.get("label") == lab)
            record(f"denominator.{k}", ok,
                   f"{f_int(val)} \u00b7 {lab} \u00b7 {acc}"
                   + ("" if ok else f"  [on disk: {d.get('value')} / "
                                    f"{d.get('accessor')}]"))
        for k in disk_dens:
            if k not in dens:
                record(f"denominator.{k}", False,
                       "present on disk but not re-derivable from the "
                       "artifacts", evaluated=False)

    # -- every quantity ---------------------------------------------------
    disk_q = on_disk.get("quantities")
    n_alt_strings = 0        # alt strings actually compared (layer A coverage)
    n_alt_entries = 0        # quantities carrying at least one alt
    unfaithful = []          # layer B findings
    if disk_q is None:
        record("quantities.block", False,
               "results_bundle.json has no 'quantities' block", evaluated=False)
    else:
        for key, e in B.q.items():
            if key not in disk_q:
                record(f"q.{key}", False,
                       "re-derived but absent from the emitted bundle",
                       evaluated=False)
                continue
            d = disk_q[key]
            problems = []
            if not values_equal(d.get("value"), e["value"]):
                problems.append(f"value {d.get('value')!r} != {e['value']!r}")
            if d.get("printed") != e["printed"]:
                problems.append(
                    f"printed {d.get('printed')!r} != {e['printed']!r}")
            if d.get("accessor") != e["accessor"]:
                problems.append(
                    f"accessor {d.get('accessor')!r} != {e['accessor']!r}")
            if d.get("label") != e["label"]:
                problems.append("label drifted")
            # LAYER A -- printed_alt is re-derived, element for element, by the
            # same renderer that produced it.  An alt string is a licence to
            # print a number (verify_manuscript resolves manuscript numerals
            # against printed AND printed_alt), so it is checked exactly.
            disk_alt = list(d.get("printed_alt") or [])
            rederived_alt = list(e["printed_alt"])
            if disk_alt != rederived_alt:
                problems.append(
                    f"printed_alt {disk_alt!r} != {rederived_alt!r}")
            n_alt_strings += len(disk_alt)
            if disk_alt:
                n_alt_entries += 1
            # LAYER B -- and it must be a faithful rendering of the entry's own
            # numbers, so a generator bug cannot ship a licence for a digit the
            # value does not support.
            for a in disk_alt:
                ok_a, why = alt_faithful(a, d.get("value"), d.get("printed"))
                if not ok_a:
                    unfaithful.append(f"{key}: {why}")
            record(f"q.{key}", not problems,
                   e["printed"] if not problems else "; ".join(problems))
        for key in disk_q:
            if key not in B.q:
                record(f"q.{key}", False,
                       "present on disk but not re-derivable from the "
                       "artifacts", evaluated=False)

        # Non-vacuity for both alt layers.  Without this, a bundle that had lost
        # its printed_alt lists would pass both of them by having nothing left
        # to check -- the classic vacuous pass CONTRACTS section 13 forbids.
        record("alt.coverage",
               n_alt_strings >= ALT_STRING_FLOOR
               and n_alt_entries >= ALT_ENTRY_FLOOR,
               f"{n_alt_strings} printed_alt strings over {n_alt_entries} "
               f"quantities re-derived and compared element-for-element "
               f"(floors {ALT_STRING_FLOOR} / {ALT_ENTRY_FLOOR})")
        record("alt.faithful_to_value", not unfaithful,
               f"every one of the {n_alt_strings} alt strings is a faithful "
               f"rendering of its own quantity's numbers"
               if not unfaithful else "; ".join(unfaithful[:6])
               + (f" (+{len(unfaithful) - 6} more)"
                  if len(unfaithful) > 6 else ""))

    # -- meta -------------------------------------------------------------
    meta = on_disk.get("_meta", {})
    record("meta.n_quantities", meta.get("n_quantities") == len(B.q),
           f"{meta.get('n_quantities')} on disk vs {len(B.q)} re-derived")
    record("meta.seed", meta.get("seed") == RANDOM_SEED,
           f"{meta.get('seed')}")
    mc = meta.get("manifest_counts", {})
    record("meta.manifest_counts",
           mc.get("tables") == len(REQUIRED_TABLES)
           and mc.get("figures") == len(REQUIRED_FIGURES)
           and mc.get("derived") == len(REQUIRED_DERIVED)
           and mc.get("logs") == len(REQUIRED_LOGS),
           f"tables {mc.get('tables')} / figures {mc.get('figures')} / "
           f"derived {mc.get('derived')} / logs {mc.get('logs')}")

    # -- banned constants over both emitted files -------------------------
    with open(OUT_JSON, "r", encoding="utf-8") as fh:
        json_txt = fh.read()
    hits = scan_banned(json_txt, "results_bundle.json")
    record("banned.results_bundle_json", not hits,
           "clean" if not hits else f"{hits}")
    if os.path.exists(OUT_LEDGER):
        with open(OUT_LEDGER, "r", encoding="utf-8") as fh:
            led_txt = fh.read()
        hits2 = scan_banned(led_txt, "NUMBERS-LEDGER.md")
        record("banned.numbers_ledger_md", not hits2,
               "clean" if not hits2 else f"{hits2}")
        record("ledger.denominator_table",
               "## The nine denominators" in led_txt,
               "'The nine denominators' table present")
        record("ledger.rule_statement",
               "**Anything absent from this ledger cannot appear in the "
               "manuscript.**" in led_txt,
               "bold rule statement present at the top")
    else:
        record("banned.numbers_ledger_md", False,
               f"{OUT_LEDGER} does not exist", evaluated=False)

    # BANNED_CONSTANTS integrity -- downstream verifiers copy-adapt this list,
    # so its contents are themselves a checked invariant.
    expected_banned = [0.876, 31054, 0.1295, 635, 3884, 4209, 4253, 4478, 27.3]
    numeric_banned = [b for b in BANNED_CONSTANTS if not isinstance(b, str)]
    string_banned = [b for b in BANNED_CONSTANTS if isinstance(b, str)]
    record("banned.list_integrity",
           numeric_banned == expected_banned and len(string_banned) == 2
           and len(BANNED_CONSTANTS) == 11,
           f"{len(BANNED_CONSTANTS)} entries: 9 numeric + 2 retracted-framing "
           "strings")
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
        self_txt = fh.read()
    record("banned.words_confined_to_the_literal",
           all(self_txt.count(w) == 1 for w in string_banned),
           "each banned word occurs in build_bundle.py exactly once, as the "
           "BANNED_CONSTANTS literal that enforces the ban")

    # -- print ------------------------------------------------------------
    print(f"\n{'CHECK':<62} {'VERDICT':<22} DETAIL")
    print("-" * 78)
    shown_fail = 0
    for cid, verdict, detail in rows:
        if verdict == "PASS":
            continue
        shown_fail += 1
        print(f"{cid:<62} {verdict:<22} {detail}")
    if shown_fail == 0:
        print("(no failures)")
    print("-" * 78)
    n_pass = sum(1 for _, v, _ in rows if v == "PASS")
    print(f"{n_pass} PASS   {n_fail} FAIL   ({len(rows)} checks)")
    by_sec = OrderedDict()
    for key, e in B.q.items():
        by_sec[e["section"]] = by_sec.get(e["section"], 0) + 1
    print("\nQuantities re-derived by section:")
    for s in SECTION_ORDER:
        if s in by_sec:
            print(f"  {s:<16} {by_sec[s]:>5}")
    print(f"  {'TOTAL':<16} {len(B.q):>5}")
    print("\nThe nine denominators, as re-derived from the artifacts:")
    for k, (val, acc, lab) in dens.items():
        print(f"  {f_int(val):>9}  {lab:<62} {acc}")
    print("=" * 78)
    if n_fail:
        print("VERIFY FAILED")
        return 1
    print("VERIFY PASSED \u2014 every quantity in results_bundle.json "
          "re-derives from the v3 artifacts: value, printed, every "
          "printed_alt, accessor and label, plus the nine denominators and "
          "the named build gates.")
    print("=" * 78)
    return 0


# ============================================================================
# BUILD
# ============================================================================
def build():
    preflight()
    A = Art()
    B = build_all(A)
    payload = bundle_payload(A, B)
    json_txt = write_json(payload)
    ledger_txt, _ = write_ledger(A, payload, B)

    # Banned-constant scan over both emitted artifacts.  A hit is fatal.
    hits = scan_banned(json_txt, "results_bundle.json")
    hits += scan_banned(ledger_txt, "NUMBERS-LEDGER.md")
    if hits:
        print("\nBANNED CONSTANT FOUND IN AN EMITTED v4 FILE:")
        for tok, matched, where in hits:
            print(f"  {where}: banned {tok!r} matched as {matched!r}")
        sys.exit(3)

    n_rows = len(payload["quantities"]) + len(payload["denominators"])
    print()
    print("=" * 78)
    print("BUILD COMPLETE")
    print("=" * 78)
    print(f"  results_bundle.json : {OUT_JSON}")
    print(f"  NUMBERS-LEDGER.md   : {OUT_LEDGER}")
    print(f"  quantities          : {len(payload['quantities'])}")
    print(f"  denominators        : {len(payload['denominators'])}")
    print(f"  ledger rows         : {n_rows}  "
          "(quantity rows + the nine-denominator table)")
    print(f"  freshness gates     : "
          f"{sum(1 for g in payload['freshness_gates'] if g['passed'])}"
          f"/{len(payload['freshness_gates'])} PASS")
    print("  banned-constant scan: clean over both emitted files")
    print()
    by_sec = OrderedDict()
    for key, e in payload["quantities"].items():
        by_sec[e["section"]] = by_sec.get(e["section"], 0) + 1
    for s in SECTION_ORDER:
        if s in by_sec:
            print(f"    {s:<16} {by_sec[s]:>5}")
    print()
    print("  The nine denominators:")
    for k, d in payload["denominators"].items():
        print(f"    {d['printed']:>9}  {d['label']:<62} {d['accessor']}")
    print("=" * 78)
    print("Anything absent from this bundle cannot appear in the manuscript.")
    return 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    sys.exit(build())
