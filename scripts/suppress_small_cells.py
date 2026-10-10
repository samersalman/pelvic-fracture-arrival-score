#!/usr/bin/env python3
"""Build the shareable aggregate tables, figures and derived files.

Reads an allow-list of aggregate files from the local project folder (given on
the command line, never hard-coded), applies small-cell suppression, and writes
the shareable versions into this repository. Also writes docs/SUPPRESSION.md.

Usage
    python scripts/suppress_small_cells.py --proj <project folder> [--out <repo root>]

The project folder is the local pelvic-fracture working folder that holds the
v3/ and v4/ subfolders. It is not part of this repository, so this script can be
run only by the author. Its output is committed; the script documents how that
output was produced.

Rule (applied to every shipped file)
    1. Any count of 1 to 9 is replaced by the marker "<10". Zero is not
       suppressed.
    2. Complementary disclosure: a withheld cell must not be recoverable by
       subtracting the other cells from a shown total or sibling. Where it
       would be, the smallest remaining cell of the same additive relation is
       also replaced, by the marker "withheld". For nested operating-point
       tables the whole row is dropped, and categories are collapsed in the
       score-distribution table.
    3. Any percentage, rate, confidence interval, SMD or P value computed from
       a suppressed numerator is replaced by "withheld".
    4. Non-count numbers are never suppressed: coefficients, AUROCs, integer
       score points, scores, tier numbers, cutoffs, spline knots, years, ranks,
       degrees of freedom, bin indices and flags.

Deterministic: no randomness, sorted iteration, no timestamps in outputs.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys

PRIMARY = "<10"
SECONDARY = "withheld"

# ----------------------------------------------------------------------------
# Allow-list. Source path (relative to the project folder) -> shipped path.
# Anything not listed here is never read.
# ----------------------------------------------------------------------------
V3_DERIVED_JSON = [
    "changepoint_sbp_composite.json", "changepoint_sbp_mortality.json",
    "changepoint_si_composite.json", "changepoint_si_mortality.json",
    "changepoint_stratified.json", "citl_slope.json", "cohort_exclusions.json",
    "component_venn.json", "reclass.json", "refit.json", "score_card.json",
    "tipping_point_bound.json",
]
V3_DERIVED_CSV = [
    "calibration_data.csv", "component_decomp_data.csv", "component_splines.csv",
    "dca_data.csv", "elderly_dca_data.csv", "informative_missingness.csv",
    "mtp_dca_data.csv", "risk_tiers.csv", "spline_sbp_composite.csv",
    "spline_sbp_mortality.csv", "spline_si_composite.csv",
    "spline_si_mortality.csv", "subgroup_forest_data.csv",
]
V3_TABLES = [
    "table1_baseline.csv", "table2_candidate_coverage.csv",
    "table2_lasso_selected.csv", "table2_lasso_stability.csv",
    "table3_points.csv", "table4_delong.csv", "table4_discrimination.csv",
    "table4b_temporal.csv", "table5_calibration.csv", "table5_operating.csv",
    "table5c_operating_full.csv", "table5d_matched_workload.csv",
    "table6_elderly_reclass.csv", "table6b_elderly_ledger.csv",
    "table7_absrisk.csv", "table7_sensitivity.csv", "table7_threshold_or.csv",
    "table8_mtp.csv", "table_consort.csv", "table_split_balance.csv",
    "table_subgroups.csv",
]
V3_FIGURES_PNG = [
    "figure10_mtp_equivalent.png", "figure1_STROBE.png",
    "figure2_score_distribution.png", "figure3_roc.png",
    "figure4_calibration.png", "figure5_dca.png",
    "figure6_subgroup_forest.png", "figure7_threshold_splines.png",
    "figure8_component.png",
]
# PDFs are vector files: they carry exact plotted coordinates. Those of the ROC
# and histogram figures would encode tail operating points and tail bins with
# fewer than 10 patients, so only the PDFs whose plotted points are all large
# aggregates are shipped.
V3_FIGURES_PDF = [
    "figure1_STROBE.pdf", "figure4_calibration.pdf", "figure5_dca.pdf",
    "figure6_subgroup_forest.pdf", "figure7_threshold_splines.pdf",
    "figure8_component.pdf",
]
V4_DERIVED_CSV = [
    "a1_correlation_vif.csv", "a1_score_distribution_test.csv",
    "a1_sensitivity_card.csv", "a1_stability_a1.csv",
    "a1_stability_parity17.csv", "a2_hypotension_mechanism.csv",
]
V4_DERIVED_JSON = [
    "a1_sensitivity_metrics.json", "a1_stability_a1_meta.json",
    "a1_stability_parity17_meta.json", "a2_worked_example.json",
    "a3_invariance_check.json",
]
V4_TABLES = ["Table 1.csv", "Table 2.csv", "Table 3.csv",
             "Supplementary Table S1.csv"]
V4_FIGURES_PNG = ["figure_1.png", "figure_2.png", "figure_3.png"]
# logs/ files that the analysis scripts read. Each was screened for counts of
# 1 to 9, absolute paths, names and patient-level content (see docs/SUPPRESSION.md).
V3_LOGS_COPY = ["d11_invariance_check.csv", "d11_invariance_check.json"]
# Frozen baseline read by analysis/19 and analysis/05. Only the three files that
# those scripts read are shipped; the snapshot's score card (override split by
# tier) and operating table are not read by any script and are not shipped.
SNAPSHOT_FILES = ["refit.json", "table3_points.csv", "table4_discrimination.csv"]
SNAPSHOT_DIR = "v3/logs/pre_d11_snapshot"
CALC_CARD = "calculator/data/score_card.json"
CALC_TIERS = "calculator/data/risk_tiers.csv"
# Keys removed from logs/qc_manifest.json (path in the JSON -> reason).
QC_DROP = [
    (("fix1_transfusion", "bands_full_csv"),
     "band counts of the acetabulum-inclusive file; three of its bands differ from the analytic-cohort bands by 1 to 9 patients, because the analytic cohort is nested in that file"),
    (("ais_predot_zfill_fix", "verification_inc_key"),
     "a single encounter key used as a spot check; a patient-level identifier"),
]

# Deliberately not shipped (reason recorded in docs/SUPPRESSION.md).
NOT_SHIPPED = [
    ("v3/derived/pelvic_cohort_predecision files, the candidates and cohort_test_scored columnar files, and all *_inc_keys files",
     "patient-level data (one row per patient or patient keys)"),
    ("all serialized Python objects in v3/derived (lasso_cv_fit, roc_data, mtp_roc_data, temporal_roc_data)",
     "binary objects that cannot be screened; the roc_data objects hold patient-level curve points"),
    ("v3/logs/11_pelvic_thresholds.log (76 MB) and the other v3/logs files (run logs, audit notes, manifests of individual steps)",
     "unreviewed run logs, contain absolute paths and internal workflow notes; only missingness.csv, qc_manifest.json, d11_invariance_check.csv and .json and three files of pre_d11_snapshot were reviewed and shipped"),
    ("v3/logs/pre_d11_snapshot/score_card.json and table5_operating.csv",
     "no script reads them; the snapshot score card holds an override split by score tier with counts under 10"),
    ("v3/figures/figure9_score_card.png and .pdf",
     "the card image prints the override count of the lowest tier (a count between 1 and 9) baked into the picture, which cannot be suppressed in an image"),
    ("v3/figures/figure2_score_distribution.pdf, figure3_roc.pdf, figure10_mtp_equivalent.pdf",
     "vector files encode exact tail bins and tail ROC points with fewer than 10 patients; the PNG versions are shipped"),
    ("v4/figures, the TIFF versions", "duplicates of the shipped PNG files"),
    ("v4/figures/figure_legends, the Markdown copies of the v4 tables, the Word copies of Tables 1 to 3, and any manuscript Word file",
     "manuscript text and internal footnotes (the Markdown copy of Supplementary Table S1 has footnotes that name an internal script and a hash list)"),
    ("v4/derived/v4_analyses_manifest.json",
     "internal provenance: absolute paths, an internal plan folder name, hashes of patient-level inputs and of pre-suppression outputs that would no longer match"),
    ("v4/bundle/NUMBERS-LEDGER.md, PI-comments, plans, VERIFY-REPORT.txt, checklists, manuscript folders",
     "internal documents"),
]

# ----------------------------------------------------------------------------
# Text scrub applied to every shipped text file: absolute paths and
# co-author attributions. Order matters.
# ----------------------------------------------------------------------------
ABS_ROOT = None   # set from --proj at run time; never hard-coded
# (regex, replacement, description). Personal names are matched generically
# (a title followed by one word) so that no name is written in this file.
SCRUB_PATTERNS = [
    (r"\(v4 response to Dr\. \w+\)", "(reviewer-requested)", "personal attribution replaced by 'reviewer-requested'"),
    (r"\bi_[a-z]+_profile\b", "i_posthoc_profile", "group label made neutral"),
    (r"Fix 6 / D4 \(\w+\)", "Fix 6 / D4 (reviewer-requested)", "personal attribution replaced by 'reviewer-requested'"),
    (r"Dr\. \w+'s objection", "a reviewer's objection", "personal attribution replaced by 'a reviewer'"),
    (r"Hand computation by Dr\. \w+", "Hand computation requested by a reviewer", "personal attribution replaced by 'a reviewer'"),
    (r"\(Dr\. \w+, comment F1\)", "(reviewer comment F1)", "personal attribution replaced by 'reviewer comment'"),
    (r"\bPI comment F3", "reviewer comment F3", "internal role label replaced by 'reviewer comment'"),
    (r"\bv4_[a-z]+_analyses\.py", "v4_posthoc_analyses.py", "analysis script name made neutral"),
    (r"\bcandidates\.p[a]rquet", "the local candidates table", "name of a local patient-level file made generic"),
    (r"\bcohort_test_scored\.p[a]rquet", "the local scored test table", "name of a local patient-level file made generic"),
    (r"does not replace the locked v3 score", "does not replace the primary score", "internal version label replaced by 'the primary score'"),
    # JSON text only (the escaped form of the dash); a dash in the middle of a sentence becomes a semicolon
    (r" \\" + "u2014 ", "; ", "em-dash in a JSON sentence replaced by a semicolon"),
]
FORBID_NAMES = []  # extra surnames to refuse, given with --forbid-names (not stored in the repository)
NAME_RE = None
METHOD_NAME_RE = re.compile(r"Wilson (CIs?|score|interval)")


LOG = []   # (file, kind, detail)


def note(file, kind, detail):
    LOG.append((file, kind, detail))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path):
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8")


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def scrub_text(txt, shipped):
    out = txt
    changed = []
    if ABS_ROOT:
        for a, why in ((ABS_ROOT + "/", "absolute project path made relative"), (ABS_ROOT, "absolute project path made relative")):
            if a in out:
                if why not in changed:
                    changed.append(why)
                out = out.replace(a, "" if a.endswith("/") else ".")
    for pat, rep, why in SCRUB_PATTERNS:
        if re.search(pat, out):
            if why not in changed:
                changed.append(why)
            out = re.sub(pat, rep, out)
    if changed:
        note(shipped, "text scrub", "; ".join(changed))
    return out


def parse_csv(txt):
    return list(csv.reader(io.StringIO(txt, newline="")))


def dump_csv(rows):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


def is_small(x):
    return x is not None and 1 <= x <= 9


def to_int(s):
    s = s.strip().replace(",", "")
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        return int(float(s))
    return None


# ----------------------------------------------------------------------------
# Generic additive-relation secondary suppression.
# relations: list of lists of cell keys that must sum to a shown total. A
# relation with exactly one withheld cell gets one more withheld, the smallest
# remaining cell (leftmost on ties). Iterates to a fixed point.
# ----------------------------------------------------------------------------
def secondary_suppress(values, withheld, relations):
    added = []
    changed = True
    while changed:
        changed = False
        for rel in relations:
            hit = [k for k in rel if k in withheld]
            if len(hit) == 1:
                cand = [k for k in rel if k not in withheld and values.get(k) is not None]
                if not cand:
                    raise RuntimeError("no candidate for secondary suppression in %r" % (rel,))
                cand.sort(key=lambda k: (values[k], rel.index(k)))
                withheld.add(cand[0])
                added.append(cand[0])
                changed = True
    return added


# ----------------------------------------------------------------------------
# table1_baseline.csv (n (%) cells)
# ----------------------------------------------------------------------------
CELL_RE = re.compile(r"^([\d,]+) \(([\d.]+)%\)$")


def handle_table1_baseline(src_txt, shipped):
    rows = parse_csv(src_txt)
    hdr = rows[0]
    ix = {h: i for i, h in enumerate(hdr)}
    rel_cols = [["Overall", "NoEvent", "Event"], ["Overall", "Young", "Elderly_noAC", "Elderly_AC"]]
    count_cols = ["Overall", "NoEvent", "Event", "Young", "Elderly_noAC", "Elderly_AC"]
    prim = sec = der = 0
    for r in rows[1:]:
        vals = {}
        for c in count_cols:
            m = CELL_RE.match(r[ix[c]].strip())
            if m:
                vals[c] = int(m.group(1).replace(",", ""))
        withheld = {c for c, v in vals.items() if is_small(v)}
        if not withheld:
            continue
        prim_cells = set(withheld)
        relations = []
        for rel in rel_cols:
            if all(c in vals for c in rel):
                # the first element is the total; only components can be chosen
                relations.append(rel[1:] + [rel[0]])
        # candidates exclude the total (last element) by making it unselectable
        values = {c: vals[c] for c in vals if c != "Overall"}
        rels = [[c for c in rel if c != "Overall"] for rel in relations]
        sec_added = secondary_suppress(values, withheld, rels)
        for c in withheld:
            r[ix[c]] = PRIMARY if c in prim_cells else SECONDARY
        prim += len(prim_cells)
        sec += len(sec_added)
        # derived stats
        ev_touch = withheld & {"Overall", "NoEvent", "Event"}
        st_touch = withheld & {"Young", "Elderly_noAC", "Elderly_AC"}
        if ev_touch:
            for c in ("SMD_event", "P_event"):
                if r[ix[c]].strip() not in ("", "NA"):
                    r[ix[c]] = SECONDARY
                    der += 1
        if st_touch:
            for c in ("SMD_strata", "P_strata"):
                if r[ix[c]].strip() not in ("", "NA"):
                    r[ix[c]] = SECONDARY
                    der += 1
    nested_hidden = t1_nested_pass(rows, ix)
    note(shipped, "suppressed", "%d primary cells (<10), %d complementary cells, %d derived SMD/P cells" % (prim, sec, der))
    if nested_hidden:
        note(shipped, "suppressed (nested definitions)",
             "cells in the strata columns of outcome rows of the untimed-death recovery ladder withheld, because an outcome definition that adds a few patients to a narrower one would otherwise differ from it by 1 to 9 patients in a stratum, directly or as the elderly total; the complementary cells and the derived SMD and P values of those rows are withheld too")
    return dump_csv(rows)


# ----------------------------------------------------------------------------
# Nested definitions: counts of sets that contain one another.
# A difference of two shown counts is itself the size of a set (the patients
# that the larger definition adds). When that difference is 1 to 9, the cell is
# as small as any cell the first rule withholds, so the cell of the larger set is
# withheld (rule 2), with the cells that would let a reader recompute it and the
# statistics computed from them. Totals a reader can derive by adding or
# subtracting shown cells count as shown.
# ----------------------------------------------------------------------------
FILE_SECRETS_BY_FILE = {}   # shipped path -> patterns for withheld true values in that file only (never written anywhere)


def add_secret(rel, pattern):
    FILE_SECRETS_BY_FILE.setdefault(rel, []).append(re.compile(pattern))


def num_or_none(s):
    return s if isinstance(s, int) and not isinstance(s, bool) else None


# ---- tables/table1_baseline.csv: outcome rows that contain one another ----
T1_PATH = "tables/table1_baseline.csv"
T1_COLS = ["Overall", "NoEvent", "Event", "Young", "Elderly_noAC", "Elderly_AC"]
T1_EQS = (["Overall", "NoEvent", "Event"], ["Overall", "Young", "Elderly_noAC", "Elderly_AC"])


def t1_key(v):
    """Short key of an Outcomes row, from its label (labels are matched by wording, not by position)."""
    if v.startswith("Major hemorrhage event") and "primary composite" in v:
        return "comp"
    if "head-AIS-only death arm" in v:
        return "comp_headais"
    if v.startswith("Patient-state composite"):
        return "patient_state"
    if v.startswith("Hemorrhage-control procedure"):
        return "hc"
    if v.startswith("Hemorrhagic death") and "in-hospital, head AIS < 4, within 48 h" in v:
        return "death_hem"
    if v.startswith("Substantial transfusion"):
        return "tx_vol" if "volume rule only" in v else "tx"
    if v.startswith("Death with head AIS < 4"):
        return "death_headais"
    if v.startswith("Death within 48 h of arrival"):
        return "death48"
    if v.startswith("In-hospital death, all cause"):
        return "death_all"
    for n in (2, 3, 4):
        if v.startswith("Hemorrhagic death") and "recovery rung %d" % n in v:
            return "rung%d" % n
    if v.startswith("Hemorrhagic death by the superseded calendar-day proxy"):
        return "proxy_hem"
    if v.startswith("Early death by the superseded calendar-day proxy"):
        return "proxy_early"
    if v == "Death in the emergency department":
        return "ed_death"
    return None


# (contained, containing) by definition. The recovery ladder adds untimed deaths to the primary death component.
T1_NESTED = [
    ("death_hem", "rung2"), ("rung2", "rung3"), ("rung3", "rung4"), ("death_hem", "rung3"),
    ("death_hem", "rung4"), ("rung2", "rung4"),
    ("comp", "comp_headais"), ("patient_state", "comp"), ("hc", "comp"), ("death_hem", "comp"), ("tx", "comp"),
    ("tx_vol", "tx"), ("death_hem", "patient_state"), ("tx", "patient_state"),
    ("death_hem", "death_headais"), ("death_hem", "death48"), ("death_headais", "death_all"),
    ("death48", "death_all"), ("ed_death", "death_all"), ("proxy_hem", "proxy_early"),
    ("proxy_hem", "death_headais"), ("proxy_early", "death_all"),
]


def t1_shown(r, ix):
    out = {}
    for c in T1_COLS:
        m = CELL_RE.match(r[ix[c]].strip())
        if m:
            out[c] = int(m.group(1).replace(",", ""))
    return out


def t1_closure(shown):
    """Everything a reader can compute from the shown cells of one row (single unknown in a sum), plus the elderly total."""
    v = dict(shown)
    changed = True
    while changed:
        changed = False
        for cells in T1_EQS:
            unk = [c for c in cells if c not in v]
            if len(unk) == 1:
                u = unk[0]
                if u == "Overall":
                    v[u] = sum(v[c] for c in cells[1:])
                else:
                    v[u] = v["Overall"] - sum(v[c] for c in cells[1:] if c != u)
                changed = True
    if "Elderly_noAC" in v and "Elderly_AC" in v:
        v["Elderly"] = v["Elderly_noAC"] + v["Elderly_AC"]
    elif "Overall" in v and "Young" in v:
        v["Elderly"] = v["Overall"] - v["Young"]
    return v


def t1_violation(rows, ix, keymap):
    for sk, lk in T1_NESTED:
        va = t1_closure(t1_shown(rows[keymap[sk]], ix))
        vb = t1_closure(t1_shown(rows[keymap[lk]], ix))
        for c in T1_COLS + ["Elderly"]:
            if c in va and c in vb:
                if vb[c] < va[c]:
                    raise RuntimeError("Table 1 outcome rows %s and %s are not nested in column %s" % (sk, lk, c))
                if 0 < vb[c] - va[c] < 10:
                    return sk, lk, c
    return None


def t1_keymap(rows, ix):
    keymap = {}
    for i, r in enumerate(rows[1:], 1):
        if r[ix["Block"]] == "Outcomes":
            k = t1_key(r[ix["Variable"]])
            if k:
                if k in keymap:
                    raise RuntimeError("Table 1 outcome row matched twice: " + k)
                keymap[k] = i
    missing = {k for p in T1_NESTED for k in p} - set(keymap)
    if missing:
        raise RuntimeError("Table 1 outcome rows not found: %s" % sorted(missing))
    return keymap


def t1_hide(r, ix, col, hidden, rowkey):
    r[ix[col]] = SECONDARY   # no value pattern is searched for: a cell text such as "379 (1.5%)" also occurs in unrelated rows
    hidden.append((rowkey, col))


def t1_enforce(r, ix, hidden, rowkey):
    """A withheld cell must not be the single unknown of a sum over shown cells."""
    while True:
        shown = t1_shown(r, ix)
        for cells in T1_EQS:
            unk = [c for c in cells if c not in shown]
            if len(unk) == 1:
                cand = [c for c in cells if c in shown and c != "Overall"]
                if not cand:
                    raise RuntimeError("no complementary cell for Table 1 row %s" % rowkey)
                cand.sort(key=lambda c: (shown[c], cells.index(c)))
                t1_hide(r, ix, cand[0], hidden, rowkey)
                break
        else:
            return


def t1_nested_pass(rows, ix):
    keymap = t1_keymap(rows, ix)
    hidden = []
    while True:
        hit = t1_violation(rows, ix, keymap)
        if not hit:
            break
        sk, lk, c = hit
        r = rows[keymap[lk]]
        sh = t1_shown(r, ix)
        if c == "Elderly":
            tgt = next(x for x in ("Young", "Elderly_noAC", "Elderly_AC") if x in sh)
        else:
            if c not in sh:
                raise RuntimeError("derivable but withheld cell in Table 1 row %s column %s" % (lk, c))
            tgt = c
        t1_hide(r, ix, tgt, hidden, lk)
        t1_enforce(r, ix, hidden, lk)
    for rowkey in sorted({h[0] for h in hidden}):
        r = rows[keymap[rowkey]]
        cols = {h[1] for h in hidden if h[0] == rowkey}
        pairs = []
        if cols & {"Overall", "NoEvent", "Event"}:
            pairs += ["SMD_event", "P_event"]
        if cols & {"Young", "Elderly_noAC", "Elderly_AC"}:
            pairs += ["SMD_strata", "P_strata"]
        for c in pairs:
            if r[ix[c]].strip() not in ("", "NA", PRIMARY, SECONDARY):
                r[ix[c]] = SECONDARY
    return hidden


# ---- subgroup tables: a denominator that needs one more measurement is a subset ----
# (contained, containing): shock index needs pulse as well as pressure; the reverse shock index also needs the Glasgow Coma Scale.
SUBGROUP_NEST = [("rSIG", "SI-alone"), ("SI-alone", "pelvic_score"), ("rSIG", "pelvic_score")]
SUBGROUP_COLS = ("N", "N_events")


def handle_subgroups(src_txt, shipped):
    rows = parse_csv(src_txt)
    ix = {h: i for i, h in enumerate(rows[0])}
    cell = {}
    for r in rows[1:]:
        cell[(r[ix["Subgroup"]], r[ix["Level"]], r[ix["Score"]])] = r
    order = [(r[ix["Subgroup"]], r[ix["Level"]], r[ix["Score"]]) for r in rows[1:]]
    true = {k: {c: to_int(cell[k][ix[c]]) for c in SUBGROUP_COLS} for k in cell}
    hidden = set()   # (subgroup, level, score, column)
    prim = set()     # the cells that differ by 1 to 9; the rest of hidden are complementary
    changed = True
    while changed:
        changed = False
        for (sg, lv, sc) in order:
            for small, large in SUBGROUP_NEST:
                if sc != large or (sg, lv, small) not in cell:
                    continue
                for c in SUBGROUP_COLS:
                    a, b = true[(sg, lv, small)][c], true[(sg, lv, large)][c]
                    if (sg, lv, small, c) in hidden or (sg, lv, large, c) in hidden:
                        continue
                    if 0 < b - a < 10:
                        hidden.add((sg, lv, large, c))
                        prim.add((sg, lv, large, c))
                        changed = True
        # complementary: the levels of one subgroup add up to a total that is shown elsewhere
        for (sg, lv, sc, c) in sorted(hidden):
            part = [k for k in order if k[0] == sg and k[2] == sc]
            if sum(1 for k in part if (k[0], k[1], k[2], c) in hidden) == 1:
                cand = [k for k in part if (k[0], k[1], k[2], c) not in hidden]
                cand.sort(key=lambda k: (true[k][c], part.index(k)))
                hidden.add((cand[0][0], cand[0][1], cand[0][2], c))
                changed = True
    # the hidden levels together must not differ from the same levels of a smaller denominator by 1 to 9
    groups = {}
    for (sg, lv, sc, c) in hidden:
        groups.setdefault((sg, sc, c), []).append(lv)
    for (sg, sc, c), lvs in groups.items():
        for other in {k[2] for k in order if k[0] == sg} - {sc}:
            if all((sg, lv, other, c) not in hidden for lv in lvs):
                d = sum(true[(sg, lv, sc)][c] - true[(sg, lv, other)][c] for lv in lvs)
                if 0 < abs(d) < 10:
                    raise RuntimeError("withheld subgroup levels still differ by 1 to 9 from another denominator")
    for (sg, lv, sc, c) in sorted(hidden):
        r = cell[(sg, lv, sc)]
        add_secret(shipped, r"(?<![\d.])%d(?![\d.])" % true[(sg, lv, sc)][c])
        r[ix[c]] = SECONDARY
    lab = lambda ks: ", ".join(sorted("%s=%s (%s)" % (sg, lv, sc) for (sg, lv, sc, c) in ks))
    note(shipped, "suppressed (nested denominators)",
         "%d N_events cells withheld. Event counts of a comparator whose denominator needs one more measurement (shock index needs pulse, the reverse shock index also needs the Glasgow Coma Scale) differed from those of the next narrower denominator by 1 to 9 patients in: %s. Complementary cells (the other level of the subgroup, so that a withheld cell cannot be recovered from the shown total): %s" %
         (len(hidden), lab(prim), lab(hidden - prim)))
    return dump_csv(rows)


# ---- derived/changepoint_stratified.json and derived/reclass.json ----
STRATA_AC = ("anticoag", "no_anticoag", "elderly_anticoag", "young_or_noac")
HIDDEN_STRATA = set()   # (outcome, measure) hidden in the shock-index curves


def strata_small_diffs(d, outcome, meas):
    """Cells whose count under the pressure-only curve (patients with a recorded pressure) exceeds the count under
    the shock-index curve (patients with pressure and pulse) by 1 to 9, including cells a reader derives by
    adding and subtracting strata (age x anticoagulant)."""
    si, sb = d["si_" + outcome]["strata"], d["sbp_" + outcome]["strata"]

    def cells(st):
        g = lambda k: num_or_none(st[k][meas])
        v = {k: g(k) for k in st}
        e, ea, a, n, y = v["elderly"], v["elderly_anticoag"], v["anticoag"], v["no_anticoag"], v["young"]
        v["d_elderly_noac"] = None if e is None or ea is None else e - ea
        v["d_young_ac"] = None if a is None or ea is None else a - ea
        v["d_young_noac"] = None if n is None or v["d_elderly_noac"] is None else n - v["d_elderly_noac"]
        return v
    a, b = cells(si), cells(sb)
    return sorted(k for k in a if a[k] is not None and b[k] is not None and 0 < b[k] - a[k] < 10)


def handle_changepoint_stratified(src_txt, shipped):
    d = json.loads(src_txt)
    for outcome in ("composite", "mortality"):
        for meas in ("n", "events"):
            if strata_small_diffs(d, outcome, meas):
                for k in STRATA_AC:
                    v = d["si_" + outcome]["strata"][k][meas]
                    add_secret(shipped, r'"%s":\s*%d\b' % (meas, v))
                    d["si_" + outcome]["strata"][k][meas] = SECONDARY
                HIDDEN_STRATA.add((outcome, meas))
                if strata_small_diffs(d, outcome, meas):
                    raise RuntimeError("strata still differ by 1 to 9 after withholding")
    note(shipped, "suppressed (nested denominators)",
         "n and events of the strata %s of the shock-index curves (outcomes %s) withheld: the patients with a recorded pressure but no pulse (the difference between the pressure-only and shock-index curves) numbered 1 to 9 in the anticoagulant strata, directly or by subtracting the age strata" %
         (", ".join(STRATA_AC), ", ".join(sorted({o for o, _ in HIDDEN_STRATA}))))
    return json_style_dump(d, src_txt)


def handle_reclass(src_txt, shipped):
    d = json.loads(src_txt)
    cur = d["elderly_changepoint_tiein"]["curves"]
    for (outcome, meas) in sorted(HIDDEN_STRATA):
        node = cur["si_" + outcome]["strata"]["elderly_anticoag"]
        add_secret(shipped, r'"%s":\s*%d\b' % (meas, node[meas]))
        node[meas] = SECONDARY
    note(shipped, "suppressed (nested denominators)",
         "elderly_changepoint_tiein: n and events of the elderly_anticoag stratum of the shock-index curves withheld, as in derived/changepoint_stratified.json (the same cells)")
    return json_style_dump(d, src_txt)


# ---- tables/table7_sensitivity.csv: outcome variants of one cohort slice ----
T7_REFERENCE = ("ring_only_reference", "primary_score_reference", "sens_with_ring")


def handle_table7_sensitivity(src_txt, shipped):
    rows = parse_csv(src_txt)
    ix = {h: i for i, h in enumerate(rows[0])}
    ev = {}   # (slice, variant) -> events, in file order
    for r in rows[1:]:
        if r[ix["arm"]] == "MI":
            continue   # a pooled mean over imputations, not a count of patients
        sl = "test" if "TEST slice" in r[ix["cohort_def"]] else "full"
        ev.setdefault((sl, r[ix["variant"]]), to_int(r[ix["n_events"]]))
    hide = []
    for sl in ("full", "test"):
        items = [(v, e) for (s, v), e in ev.items() if s == sl and e is not None]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (va, ea), (vb, eb) = items[i], items[j]
                if 0 < abs(ea - eb) < 10:
                    if (sl, va) in hide or (sl, vb) in hide:
                        continue
                    victim = vb if vb not in T7_REFERENCE else va
                    if victim in T7_REFERENCE:
                        raise RuntimeError("two reference rows differ by 1 to 9 events")
                    hide.append((sl, victim))
    nrows = 0
    for r in rows[1:]:
        sl = "test" if "TEST slice" in r[ix["cohort_def"]] else "full"
        if (sl, r[ix["variant"]]) in hide and r[ix["arm"]] != "MI":
            add_secret(shipped, r"(?<![\d.,])%d(?![\d.,])" % to_int(r[ix["n_events"]]))
            r[ix["n_events"]] = SECONDARY
            r[ix["composite_rate_pct"]] = SECONDARY   # events / n to four decimals: the count again
            nrows += 1
    note(shipped, "suppressed (nested definitions)",
         "n_events and composite_rate_pct withheld in %d rows (variants %s; slices %s): the event total of the variant differs from the total of the reference definition of the same cohort slice by 1 to 9 patients (the untimed deaths that the variant adds), so the shown total would reveal that count" %
         (nrows, ", ".join(sorted({v for _, v in hide})), ", ".join(sorted({s for s, _ in hide}))))
    return dump_csv(rows)


# ----------------------------------------------------------------------------
# risk_tiers.csv and score_card.json: n_override_in_tier
# ----------------------------------------------------------------------------
FILE_SECRETS = []   # patterns for withheld true values in data files (compiled at run time, never written anywhere)
DOC_SECRETS = []    # patterns for withheld true values in prose (docs/SUPPRESSION.md)


def override_tier_values(tier_to_n, total):
    """tier_to_n: ordered dict tier -> n override patients in that score tier.
    Returns dict tier -> marker or None (shown)."""
    withheld = {t for t, v in tier_to_n.items() if is_small(v)}
    prim = set(withheld)
    rel = [list(tier_to_n.keys())]
    secondary_suppress(dict(tier_to_n), withheld, rel)
    for t in withheld:
        FILE_SECRETS.append(re.compile(r'"n_override_in_tier":\s*%d\b' % tier_to_n[t]))
        DOC_SECRETS.append(re.compile(r"(?i)(override (group|patients?|split)|rule-out|observe|escalate)[^\n.;|]{0,100}\b%d\b" % tier_to_n[t]))
        DOC_SECRETS.append(re.compile(r"(?i)\b%d\b[^\n.;|]{0,100}(override (group|patients?|split)|rule-out|observe|escalate)" % tier_to_n[t]))
    vals = [str(v) for v in tier_to_n.values()]
    DOC_SECRETS.append(re.compile(r"\b%s\s*(/|,|and)\s*%s\b" % (re.escape(vals[0]), re.escape(vals[1]))))
    return {t: (PRIMARY if t in prim else SECONDARY if t in withheld else None) for t in tier_to_n}


def handle_risk_tiers(src_txt, shipped):
    rows = parse_csv(src_txt)
    hdr = rows[0]
    ix = {h: i for i, h in enumerate(hdr)}
    tiers = {}
    for r in rows[1:]:
        if r[ix["row_type"]] == "score_tier":
            tiers[r[ix["tier"]]] = int(r[ix["n_override_in_tier"]])
    marks = override_tier_values(tiers, None)
    cells = 0
    for r in rows[1:]:
        m = marks.get(r[ix["tier"]]) if r[ix["row_type"]] == "score_tier" else None
        if m:
            r[ix["n_override_in_tier"]] = m
            cells += 1
    note(shipped, "suppressed",
         "n_override_in_tier: %s (<10) and %s (complementary, so the first cannot be recovered as the group total minus the other tiers)" %
         ([t for t, m in marks.items() if m == PRIMARY], [t for t, m in marks.items() if m == SECONDARY]))
    return dump_csv(rows)


def json_style_dump(obj, original):
    for kw in ({"indent": 2, "ensure_ascii": True}, {"indent": 2, "ensure_ascii": False}):
        if json.dumps(json.loads(original), **kw) == original:
            return json.dumps(obj, **kw)
    raise RuntimeError("cannot reproduce JSON layout")


# ----------------------------------------------------------------------------
# score_card.json: suppress n_override_in_tier, remove internal wording
# ----------------------------------------------------------------------------
EM_DASH = chr(0x2014)
# (parent key, key) -> new key. Nothing in the calculator or the analysis scripts reads these keys.
CARD_KEY_RENAMES = {
    ("override", "column_in_scored_" + "p" + "arquet"): "column_in_scored_table",
    ("score_range", "v2_observed_reference"): "earlier_version_observed_reference",
    ("score_range", "observed_max_delta_vs_v2"): "observed_max_delta_vs_earlier_version",
    ("score_range_theoretical", "g4c_unconstrained_sum_check"): "unconstrained_sum_check",
}
# (parent key, key) of fields removed.
CARD_DROP = [("override", "render_note")]
# Exact string value -> neutral wording. Every entry must match once or the run stops.
CARD_STRINGS = {
    "received_hc | death_hemorrhagic | substantial_transfusion  (CONTRACTS \u00a74, v3)":
        "received_hc | death_hemorrhagic | substantial_transfusion",
    "observed in the v3 TEST set": "observed in the test set",
    "OBSERVED minimum and maximum of pelvic_score in the v3 TEST set (point-role variables only). This is the number CONTRACTS \u00a713 G4c gates and the number the manuscript should quote as 'the card's realistic range'. It is NOT a column sum.":
        "Observed minimum and maximum of pelvic_score in the test set (point-role variables only). This is the card's realistic range. It is not a column sum.",
    "v2 OBSERVED range, carried as a hard-coded constant for the G4c comparison ONLY " + EM_DASH + " label it as a v2 comparison wherever it is quoted (CONTRACTS \u00a713 G5b).":
        "Observed range of an earlier version of the score, carried as a constant for comparison only.",
    "Sum of the printed Points column over Role=='point' rows. A THEORETICAL bound, never observed in any patient. Reported separately from score_range per CONTRACTS \u00a713 G4c; do not compare it against the '+16' expectation in the plan.":
        "Sum of the printed Points column over Role=='point' rows. A theoretical bound, never observed in any patient. Reported separately from score_range.",
    "sbp_lt90/sbp_lt70 and si_ge0_9/si_ge1_0 are NESTED, not mutually exclusive " + EM_DASH + " a patient can satisfy both, so both count toward the theoretical maximum.":
        "sbp_lt90/sbp_lt70 and si_ge0_9/si_ge1_0 are nested, not mutually exclusive: a patient can satisfy both, so both count toward the theoretical maximum.",
    "override_nslo == 1 " + EM_DASH + " reported separately; these patients ALSO appear in one of the three score tiers (parallel escalation path, not a fourth mutually-exclusive tier)":
        "override_nslo == 1; reported separately. These patients also appear in one of the three score tiers (parallel escalation path, not a fourth mutually exclusive tier)",
    "death_inhosp (v2 all-cause mortality, named sensitivity)":
        "death_inhosp (all-cause in-hospital mortality, named sensitivity)",
    "sens-90 rule-out / spec-90 rule-in / Youden (v2 schema)":
        "sens-90 rule-out / spec-90 rule-in / Youden",
    "Fix 5 " + EM_DASH + " every integer cutoff, score vs SI vs rSIG, score-only AND score-OR-override":
        "Every integer cutoff, score vs SI vs rSIG, score-only and score-or-override",
    "Fix 5 " + EM_DASH + " equal-workload comparison; the WTA abstract's headline claim":
        "Equal-workload comparison across instruments",
}
CARD_FORBIDDEN = ("CONTRACTS", "G4c", "G5b", "F9", "Fix ", "WTA", "render_note", "v2", "v3", "p" + "arquet", EM_DASH)
# Everything the calculator (calculator/config.py, prediction.py, app.py, parity_check.py)
# and the documentation read from the card. Checked equal between source and shipped card.
CARD_READ_PATHS = [
    ("rule_out_cutpoint",), ("rule_in_cutpoint",), ("n_test",),
    ("score_range", "min"), ("score_range", "max"), ("override", "label"),
    ("override", "rule_text"), ("override", "n"), ("override", "events"),
    ("override", "event_rate"), ("deployed_rule",), ("n_point_variables",),
]


def _card_clean(obj, parent, used):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if (parent, k) in CARD_DROP:
                used.add(("drop", parent, k))
                continue
            nk = CARD_KEY_RENAMES.get((parent, k), k)
            if nk != k:
                used.add(("rename", parent, k))
            out[nk] = _card_clean(v, k, used)
        return out
    if isinstance(obj, list):
        return [_card_clean(v, parent, used) for v in obj]
    if isinstance(obj, str) and obj in CARD_STRINGS:
        used.add(("string", obj))
        return CARD_STRINGS[obj]
    return obj


def _leaves(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, path + (i,))
    else:
        yield path, obj


def _card_path_new(path):
    """Map a source-card leaf path to its path in the cleaned card."""
    out = []
    parent = ""
    for seg in path:
        if isinstance(seg, str):
            out.append(CARD_KEY_RENAMES.get((parent, seg), seg))
            parent = seg
        else:
            out.append(seg)
    return tuple(out)


def handle_score_card(src_txt, shipped):
    d = json.loads(src_txt)
    tiers = {t["tier"]: t["n_override_in_tier"] for t in d["risk_tiers"] if t["row_type"] == "score_tier"}
    marks = override_tier_values(tiers, None)
    for t in d["risk_tiers"]:
        m = marks.get(t["tier"]) if t["row_type"] == "score_tier" else None
        if m:
            t["n_override_in_tier"] = m
    suppressed = copy_json(d)
    used = set()
    clean = _card_clean(d, "", used)
    # every declared wording change must have matched
    expected = {("string", k) for k in CARD_STRINGS} | {("drop",) + k for k in CARD_DROP} | {("rename",) + k for k in CARD_KEY_RENAMES}
    if used != expected:
        raise RuntimeError("score card cleaning map out of date; unmatched: %s" % sorted(map(str, expected - used)))
    # forbidden wording gone (keys and values)
    for path, val in _leaves(clean):
        for tok in CARD_FORBIDDEN:
            if any(isinstance(seg, str) and tok in seg for seg in path) or (isinstance(val, str) and tok in val):
                raise RuntimeError("score card still contains %r at %r" % (tok, path))
    # nothing numeric changed: compare the suppressed card with the cleaned card, leaf by leaf
    src_leaves = {_card_path_new(p): v for p, v in _leaves(suppressed)}
    new_leaves = dict(_leaves(clean))
    for p, v in src_leaves.items():
        if isinstance(v, str):
            continue
        if p[0:1] == ("override",) and p[-1] == "render_note":
            continue
        if new_leaves.get(p, "MISSING") != v:
            raise RuntimeError("score card value changed at %r" % (p,))
    for p, v in new_leaves.items():
        if not isinstance(v, str) and p not in src_leaves:
            raise RuntimeError("score card gained a value at %r" % (p,))
    # the read paths, point table and tier rates are identical to the source card
    def get(o, path):
        for seg in path:
            o = o[seg]
        return o
    src_card = json.loads(src_txt)
    for path in CARD_READ_PATHS:
        if get(src_card, path) != get(clean, path):
            raise RuntimeError("score card read path changed: %r" % (path,))
    if [(v["variable"], v["points"], v["role"]) for v in src_card["variables"]] != \
            [(v["variable"], v["points"], v["role"]) for v in clean["variables"]]:
        raise RuntimeError("score card point table changed")
    keep = ("tier", "row_type", "n", "composite_events", "composite_rate", "composite_ci_low", "composite_ci_high", "score_min", "score_max")
    if [tuple(t[k] for k in keep) for t in src_card["risk_tiers"]] != [tuple(t[k] for k in keep) for t in clean["risk_tiers"]]:
        raise RuntimeError("score card tier rows changed")
    note(shipped, "suppressed",
         "risk_tiers[].n_override_in_tier: %s (<10), %s (complementary). No point value, cutpoint, tier definition or probability was changed." %
         ([t for t, m in marks.items() if m == PRIMARY], [t for t, m in marks.items() if m == SECONDARY]))
    note(shipped, "text scrub",
         "internal wording removed (document-section references, version labels, one render hint field, em-dashes); keys renamed where they carried a version label; "
         "every number, point value, cutpoint, tier definition and rate is unchanged and was checked equal to the source")
    return json_style_dump(clean, src_txt)


def copy_json(o):
    return json.loads(json.dumps(o))


# ----------------------------------------------------------------------------
# a2_hypotension_mechanism.csv
# ----------------------------------------------------------------------------
def handle_a2(src_txt, shipped):
    rows = parse_csv(src_txt)
    hdr = rows[0]
    ix = {h: i for i, h in enumerate(hdr)}
    derived = ["rate_pct", "ci_low_pct", "ci_high_pct"]
    prim = der = sec = 0
    by_label = {r[ix["subgroup"]]: r for r in rows[1:]}
    for r in rows[1:]:
        ev = to_int(r[ix["events"]])
        if is_small(ev):
            r[ix["events"]] = PRIMARY
            prim += 1
            for c in derived:
                r[ix[c]] = SECONDARY
                der += 1
    # nested subgroup pairs: parent n minus child n is itself a count
    pairs = [("same, excluding no-signs-of-life", "same, excluding NSLO, by score <=4 (Observe or lower)"),
             ("woman >=75, fall, SBP<90, SI>=1.0 (all, incl. no-signs-of-life)",
              "same, excluding NSLO, by score <=4 (Observe or lower)")]
    done = set()
    for parent, child in pairs:
        p, c = by_label[parent], by_label[child]
        if child in done:
            continue
        pn, cn = to_int(p[ix["n"]]), to_int(c[ix["n"]])
        if 0 < pn - cn < 10:
            for col in ["n", "events"] + derived:
                if c[ix[col]] not in (PRIMARY, SECONDARY):
                    c[ix[col]] = SECONDARY
                    sec += 1
            done.add(child)
    # neutral wording for two row notes that referred to an earlier plan and to the results bundle
    resid = by_label["SBP<90, score <=4 (residual: below escalate)"]
    parts = [by_label["SBP<90, score <=0 (rule-out)"], by_label["SBP<90, score 1 to 4 (observe)"]]
    if resid[ix["note"]].startswith("plan figure"):
        if (to_int(resid[ix["n"]]), to_int(resid[ix["events"]])) != tuple(sum(to_int(p[ix[c]]) for p in parts) for c in ("n", "events")):
            raise RuntimeError("residual row is not the sum of the rule-out and observe rows")
        resid[ix["note"]] = "n and events equal the sum of the rule-out and observe rows of this group"
    whole = by_label["fall, whole analytic cohort (Table 1 / results bundle)"]
    if whole[ix["note"]].startswith("from bundle cohort"):
        whole[ix["note"]] = "counts as in Table 1 and derived/results_bundle.json; N=239,736 cohort"
    note(shipped, "suppressed",
         "%d primary event cells (<10); %d derived rate/CI cells; %d complementary cells (the subset row that excludes no-signs-of-life patients by score: its n differs from its parent row by fewer than 10 patients)" % (prim, der, sec))
    note(shipped, "text scrub", "two row notes that cited an earlier plan and the results bundle internals replaced by neutral wording")
    return dump_csv(rows)


# ----------------------------------------------------------------------------
# a1_score_distribution_test.csv: collapse adjacent scores
# ----------------------------------------------------------------------------
def handle_a1_dist(src_txt, shipped):
    rows = parse_csv(src_txt)
    body = rows[1:]
    bins = [{"lo": r[0], "hi": r[0], "n": int(float(r[1])), "ev": int(float(r[2])), "rate": r[3], "k": 1} for r in body]
    merges = []
    while True:
        bad = [i for i, b in enumerate(bins) if is_small(b["n"]) or is_small(b["ev"])]
        if not bad:
            break
        i = bad[0]
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(bins)]
        j = sorted(nbrs, key=lambda j: (bins[j]["n"], j))[0]
        a, b = (bins[min(i, j)], bins[max(i, j)])
        merged = {"lo": a["lo"], "hi": b["hi"], "n": a["n"] + b["n"], "ev": a["ev"] + b["ev"], "k": a["k"] + b["k"]}
        merged["rate"] = repr(merged["ev"] / merged["n"])
        merges.append((a["lo"], b["hi"]))
        bins[min(i, j):max(i, j) + 1] = [merged]
    out = [["score_min", "score_max", "n", "events", "rate"]]
    for b in bins:
        out.append([b["lo"], b["hi"], str(b["n"]), str(b["ev"]), b["rate"]])
    collapsed = [(b["lo"], b["hi"]) for b in bins if b["k"] > 1]
    note(shipped, "collapsed",
         "adjacent scores merged so that no n or event count is 1 to 9; merged ranges (score_min to score_max): %s. Columns changed from score to score_min and score_max." %
         ", ".join("%s to %s" % c for c in collapsed))
    return dump_csv(out)


# ----------------------------------------------------------------------------
# a2_worked_example.json: drop the internal reconciliation block
# ----------------------------------------------------------------------------
def handle_a2_worked(src_txt, shipped):
    d = json.loads(src_txt)
    blk = d.pop("plan_reconciliation")
    note(shipped, "block removed",
         "plan_reconciliation (internal check against an earlier plan; held an event count between 1 and 9 and personal names in key names). Patient example and point contributions unchanged.")
    return json_style_dump(d, src_txt)


# ----------------------------------------------------------------------------
# table5c_operating_full.csv: nested operating points
# ----------------------------------------------------------------------------
def fnum(s):
    try:
        return float(s)
    except ValueError:
        return None


def family_of(instrument):
    i = instrument.lower()
    if i.startswith("pelvic"):
        return "pelvic"
    if i in ("shock_index", "si-alone"):
        return "si"
    return i


def small_row_cells(tp, fp, tn, fn):
    return [n for n, v in (("TP", tp), ("FP", fp), ("TN", tn), ("FN", fn), ("N_flagged", tp + fp)) if is_small(v)]


# Denominator rank: larger rank contains the smaller ones.
def den_rank_5c(name):
    return 2 if name == "all_test_rows" else 1


def mk_row(fam, den, cut, direction, rule, tp, fp, tn, fn, src, pos=None, role=""):
    t = None if cut is None else (cut if direction != "<=" else -cut)
    # The score-only and score-or-override rules are NOT treated as ordered:
    # their flagged sets differ by the override patients, who are absent from the
    # complete-case denominator, so the conservative reading is that rows of the
    # two rules may be nested in either direction.
    rr = 0
    return {"fam": fam, "den": den, "t": t, "rr": rr, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "src": src, "pos": pos, "role": role}


def nested_violation(a, b):
    """Return the first count in which the two rows are nested sets whose size
    difference is 1 to 9, else None. Flagged sets nest when the first row has the
    larger denominator, the looser cutoff and the broader rule; unflagged sets
    nest in the opposite cutoff direction."""
    if a["fam"] != b["fam"] or a["t"] is None or b["t"] is None or a["den"] < b["den"]:
        return None
    if a["t"] <= b["t"] and a["rr"] >= b["rr"]:
        for k in ("TP", "FP"):
            if 0 < a[k] - b[k] < 10:
                return k
    if a["t"] >= b["t"] and a["rr"] <= b["rr"]:
        for k in ("TN", "FN"):
            if 0 < a[k] - b[k] < 10:
                return k
    return None


def any_violation(a, b):
    return nested_violation(a, b) or nested_violation(b, a)


def external_rows_5d(src_txt):
    rows = parse_csv(src_txt)
    ix = {h: i for i, h in enumerate(rows[0])}
    out = []
    for r in rows[1:]:
        try:
            cut = fnum(r[ix["Cutoff"]])
            den = 1 if r[ix["Stratum"]] == "overall" else 0
            out.append(mk_row(family_of(r[ix["Instrument"]]), den, cut, r[ix["Cutoff_Direction"]], r[ix["Rule"]],
                              int(float(r[ix["Events_caught_TP"]])), int(float(r[ix["False_positives_FP"]])),
                              int(float(r[ix["True_negatives_TN"]])), int(float(r[ix["Events_missed_FN"]])), "table5d"))
        except ValueError:
            pass
    return out


def external_rows_5(src_txt):
    rows = parse_csv(src_txt)
    ix = {h: i for i, h in enumerate(rows[0])}
    return [mk_row(family_of(r[ix["Score"]]), 2, fnum(r[ix["Cutoff"]]), ">=", "score_only",
                   int(r[ix["TP"]]), int(r[ix["FP"]]), int(r[ix["TN"]]), int(r[ix["FN"]]), "table5") for r in rows[1:]]


def handle_table5c(src_txt, shipped, ext_rows):
    rows = parse_csv(src_txt)
    hdr = rows[0]
    ix = {h: i for i, h in enumerate(hdr)}
    recs = []
    for pos, r in enumerate(rows[1:]):
        rec = mk_row(family_of(r[ix["Instrument"]]), den_rank_5c(r[ix["Denominator"]]), fnum(r[ix["Cutoff"]]),
                     r[ix["Cutoff_Direction"]], r[ix["Rule"]],
                     int(float(r[ix["TP"]])), int(float(r[ix["FP"]])), int(float(r[ix["TN"]])), int(float(r[ix["FN"]])),
                     "table5c", pos, r[ix["Cutoff_Role"]].strip())
        rec.update({"row": r, "denname": r[ix["Denominator"]], "inst": r[ix["Instrument"]],
                    "rulename": r[ix["Rule"]], "cut": fnum(r[ix["Cutoff"]])})
        recs.append(rec)
    dropped = {}

    # (a) a single cell of 1 to 9 in the row
    for rec in recs:
        bad = small_row_cells(rec["TP"], rec["FP"], rec["TN"], rec["FN"])
        if bad:
            dropped[rec["pos"]] = "cell %s in 1..9" % "/".join(bad)
    # (b) the override marginal columns (score-or-override minus score-only)
    for rec in recs:
        for col in ("Override_Marginal_Flagged", "Override_Marginal_Events"):
            v = fnum(rec["row"][ix[col]])
            if v is not None and is_small(int(v)) and rec["pos"] not in dropped:
                dropped[rec["pos"]] = "%s in 1..9" % col

    def alive():
        return [r for r in recs if r["pos"] not in dropped]

    def extremeness(r):
        n = r["TP"] + r["FP"] + r["TN"] + r["FN"]
        return (min(r["TP"] + r["FP"], r["TN"] + r["FN"]), r["pos"])

    # (c) nested differences, within the table and against table5d / table5
    while True:
        live = alive()
        hit = None
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                k = any_violation(live[i], live[j])
                if k:
                    hit = (live[i], live[j], k, "another row of this table")
                    break
            if hit:
                break
            for e in ext_rows:
                k = any_violation(live[i], e)
                if k:
                    hit = (live[i], None, k, "a %s row" % e["src"])
                    break
            if hit:
                break
        if hit:
            a, b, k, who = hit
            cand = [x for x in (a, b) if x is not None and not x["role"]]
            if not cand:
                raise RuntimeError("protected rows conflict: %r" % ((a["cut"], b and b["cut"]),))
            victim = min(cand, key=extremeness)
            dropped[victim["pos"]] = "nested difference in %s with %s is 1..9" % (k, who)
            continue
        # override marginal series per denominator: score-or-override minus
        # score-only at the same cutoff; any two cutoffs give a nested band.
        only = {(r["denname"], r["cut"]): r for r in alive() if r["rulename"] == "score_only"}
        marg = []
        for r in alive():
            if r["rulename"] == "score_or_override":
                o = only.get((r["denname"], r["cut"]))
                if o is not None:
                    marg.append({"den": r["denname"], "cut": r["cut"], "TP": r["TP"] - o["TP"],
                                 "FP": r["FP"] - o["FP"], "pos": r["pos"], "role": r["role"]})
        hit = None
        for den in sorted({m["den"] for m in marg}):
            ms = sorted([m for m in marg if m["den"] == den], key=lambda m: m["cut"])
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    for k in ("TP", "FP"):
                        if 0 < abs(ms[i][k] - ms[j][k]) < 10:
                            hit = (ms[i], ms[j], k)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            break
        a, b, k = hit
        cand = [x for x in (a, b) if not x["role"]]
        if not cand:
            raise RuntimeError("protected marginal conflict")
        victim = min(cand, key=lambda x: (abs(x[k]), x["pos"]))
        dropped[victim["pos"]] = "override marginal %s differs from another cutoff's by 1..9" % k

    kept = [rec["row"] for rec in recs if rec["pos"] not in dropped]
    detail = []
    for rec in recs:
        if rec["pos"] in dropped:
            cut = "none" if rec["cut"] is None else ("%g" % rec["cut"])
            detail.append("%s / %s / %s / cutoff %s: %s" % (rec["denname"], rec["inst"], rec["rulename"], cut, dropped[rec["pos"]]))
    note(shipped, "rows dropped", "%d of %d rows dropped (collapse by omission). %s" % (len(dropped), len(recs), "; ".join(detail)))
    return dump_csv([hdr] + kept)


# ----------------------------------------------------------------------------
# logs/qc_manifest.json: drop the two items that fail the screen
# ----------------------------------------------------------------------------
def handle_qc_manifest(src_txt, shipped):
    d = json.loads(src_txt)
    why = []
    for path, reason in QC_DROP:
        o = d
        for seg in path[:-1]:
            o = o[seg]
        if path[-1] in o:
            del o[path[-1]]
            why.append("%s (%s)" % (".".join(path), reason))
    if len(why) != len(QC_DROP):
        raise RuntimeError("qc_manifest.json no longer holds the items listed in QC_DROP")
    # no identifier-like value (a long run of digits as a string) may remain
    for path, val in _leaves(d):
        if isinstance(val, str) and re.fullmatch(r"\d{9,}", val):
            raise RuntimeError("identifier-like value in qc_manifest.json at %r" % (path,))
    note(shipped, "keys removed", "; ".join(why) + ". No script reads these keys. All other counts in the file are 0 or at least 10 and are not nested differences of 1 to 9.")
    return json_style_dump(d, src_txt)


# ----------------------------------------------------------------------------
# Self-check: nested differences in the written files
# ----------------------------------------------------------------------------
NON_PATIENT_KEY = re.compile(r"(n_boot|boot_n|n_valid_grid|n_unique|n_coef|n_predictors|n_selected|point_var|point_crit|n_quantities|n_bootstraps|n_surgical|min_events|max_min_events|directional_hypothesis|gcs_total|nonzero|n_score_vars|anchor_n_eligible|hl_df|per_bootstrap)", re.I)
NOT_COUNT_KEY = re.compile(r"(rate|pct|percent|_ci|lci|uci|prob|ppv|npv|delta_(?!events))", re.I)
COUNT_KEY = re.compile(r"(^n$|^n_|_n$|_n_|^N$|^N_|events?|^tp$|^fp$|^tn$|^fn$|flagged|missed|caught|_tp$|_fp$|_tn$|_fn$|positive|n_missing|patients|deaths|marginals|regions)", re.I)
PCT_CELL = re.compile(r"^([\d,]+) \((\d+(\.\d+)?%|<0\.1%)\)$")
# Files whose count cells are not compared by the generic scan: the operating-point tables (compared by the
# nested-row rule), the score-distribution bins (merged until none is small, adjacent cells disjoint) and bootstrap metadata.
SCAN_SKIP = {"tables/table5c_operating_full.csv", "tables/table5d_matched_workload.csv", "tables/table5_operating.csv",
             "derived/a1_stability_a1_meta.json", "derived/a1_stability_parity17_meta.json", "derived/a1_score_distribution_test.csv",
             "tables/table7_sensitivity.csv"}   # table7 is compared within each cohort slice by its own check
# Cells of one partition are disjoint; their difference is not the size of any set.
PARTITION_LOC = re.compile(r"(reliability_deciles|calibration_deciles|by_year|crosswalk|head_ais_dist|\.n_flagged$)")
# Reviewed pairs that are not nested (value pair, file) with the reason.
REVIEWED_NOT_NESTED = {
    ("derived/a2_hypotension_mechanism.csv", frozenset((147, 155))): "fall subgroup versus score subgroup: cross-cutting definitions",
    ("derived/a2_hypotension_mechanism.csv", frozenset((31, 35))): "age subgroup versus sex subgroup: cross-cutting definitions",
    ("derived/a2_hypotension_mechanism.csv", frozenset((2319, 2327))): "non-fall subgroup versus score subgroup: cross-cutting definitions",
}
# Printed in the manuscript's Table 3 next to the primary count; withholding it here would not remove it.
KNOWN_PRINTED = {
    frozenset((8825, 8826)): "event counts of the full-information (ISS and AIS) reference row and the primary row in Table 3, "
                             "both printed in the manuscript; the denominators differ by 30 patients",
}
KNOWN_PRINTED_FILES = {"tables/Table 3.csv", "tables/table4_discrimination.csv", "logs/pre_d11_snapshot/table4_discrimination.csv"}
HEADLINE_LOC = re.compile(r"^(primary_composite\.|death_variants\.|fix2_death\.(composite|death_)|design\.|venn\.)")


def count_cells(out, rels):
    cells = []   # (value, file, measure, locator, row dict or None)
    for rel in rels:
        if rel in SCAN_SKIP or not rel.endswith((".csv", ".json")):
            continue
        txt = read_text(os.path.join(out, rel))
        if rel.endswith(".csv"):
            rows = parse_csv(txt)
            h = rows[0]
            for ri, r in enumerate(rows[1:], 2):
                ctx = dict(zip(h, r))
                for ci, c in enumerate(r[:len(h)]):
                    s = c.strip()
                    m = re.fullmatch(r"-?(\d[\d,]*)(\.0+)?", s)
                    m2 = PCT_CELL.match(s)
                    if m and COUNT_KEY.search(h[ci]) and not NON_PATIENT_KEY.search(h[ci]) and not NOT_COUNT_KEY.search(h[ci]):
                        v = int(m.group(1).replace(",", ""))
                    elif m2:
                        v = int(m2.group(1).replace(",", ""))
                    else:
                        continue
                    if v >= 10:
                        cells.append((v, rel, h[ci], "r%d" % ri, ctx))
        else:
            def walk(o, path):
                if isinstance(o, dict):
                    for k, v in o.items():
                        walk(v, path + [str(k)])
                elif isinstance(o, list):
                    for i, v in enumerate(o):
                        walk(v, path + [str(i)])
                elif isinstance(o, (int, float)) and not isinstance(o, bool) and o == o and abs(o) != float("inf") and float(o) == int(o):
                    keys = [x for x in path if not x.isdigit()]
                    if keys and COUNT_KEY.search(keys[-1]) and not NOT_COUNT_KEY.search(keys[-1]) and not any(NON_PATIENT_KEY.search(x) for x in keys) and int(o) >= 10:
                        cells.append((int(o), rel, keys[-1], ".".join(path), None))
            walk(json.loads(txt), [])
    return cells


def pair_reason(a, b):
    """Reason why two count cells with a difference of 1 to 9 are not nested sets, or None if the pair is unexplained."""
    (va, fa, ma, la, ca), (vb, fb, mb, lb, cb) = a, b
    if PARTITION_LOC.search(la) and PARTITION_LOC.search(lb):
        return "cells of one partition (deciles, years, score tiers) are disjoint"
    if fa == fb == "derived/informative_missingness.csv":
        return "years are disjoint strata"
    if fa == fb == "tables/table1_baseline.csv":
        return "different variables; the known nestings between outcome rows are checked by t1_nested_pass"
    if fa == fb and fa in ("tables/table_subgroups.csv", "derived/subgroup_forest_data.csv"):
        if (ca["Subgroup"], ca["Level"]) != (cb["Subgroup"], cb["Level"]):
            return "different levels of one variable are disjoint; different variables cross-cut"
        return None
    if fa == fb == "derived/a2_hypotension_mechanism.csv":
        return REVIEWED_NOT_NESTED.get((fa, frozenset((va, vb))))
    if fa == fb and fa in KNOWN_PRINTED_FILES and frozenset((va, vb)) in KNOWN_PRINTED:
        return KNOWN_PRINTED[frozenset((va, vb))]
    return None


def check_nested_differences(out, rels):
    """Stops the run if any two counts that could be nested sets differ by 1 to 9. See docs/SUPPRESSION.md."""
    problems = []
    # 1. every pair of counts of one measure within a file
    cells = count_cells(out, rels)
    groups = {}
    for c in cells:
        groups.setdefault((c[1], c[2]), []).append(c)
    for key, cs in groups.items():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if 0 < abs(cs[i][0] - cs[j][0]) < 10 and cs[i][3] != cs[j][3] and not pair_reason(cs[i], cs[j]):
                    problems.append("%s %s: %s (%s) vs %s (%s)" % (key[0], key[1], cs[i][0], cs[i][3], cs[j][0], cs[j][3]))
    # 2. headline event and denominator totals, across files
    head = [c for c in cells if (c[1] == "tables/Table 3.csv" and c[2] in ("Events", "N")) or
            (c[1] in ("tables/table4_discrimination.csv", "logs/pre_d11_snapshot/table4_discrimination.csv") and c[2] in ("N_events", "N_used")) or
            (c[1] == "tables/table_split_balance.csv") or
            (c[1] == "tables/table4b_temporal.csv") or
            (c[1] == "logs/qc_manifest.json" and HEADLINE_LOC.search(c[3])) or
            (c[1] == "derived/a1_sensitivity_metrics.json" and c[3].startswith("design.")) or
            (c[1] == "derived/component_venn.json" and c[3].startswith("venn."))]
    t7 = parse_csv(read_text(os.path.join(out, "tables/table7_sensitivity.csv")))
    t7ix = {h: i for i, h in enumerate(t7[0])}
    for ri, r in enumerate(t7[1:], 2):
        v = to_int(r[t7ix["n_events"]]) if r[t7ix["arm"]] != "MI" else None
        if v is not None and v >= 10:
            head.append((v, "tables/table7_sensitivity.csv", "n_events", "r%d" % ri, None))
    seen = set()
    for i in range(len(head)):
        for j in range(i + 1, len(head)):
            a, b = head[i], head[j]
            if 0 < abs(a[0] - b[0]) < 10:
                if frozenset((a[0], b[0])) in KNOWN_PRINTED and min(a, b)[1] in KNOWN_PRINTED_FILES:   # the smaller count is the printed one
                    continue
                if (a[0], b[0], a[1], b[1]) in seen:
                    continue
                seen.add((a[0], b[0], a[1], b[1]))
                problems.append("headline counts: %s (%s %s) vs %s (%s %s)" % (a[0], a[1], a[3], b[0], b[1], b[3]))
    # 3. Table 1 outcome rows, from what a reader can derive
    rows = parse_csv(read_text(os.path.join(out, T1_PATH)))
    ix = {h: i for i, h in enumerate(rows[0])}
    hit = t1_violation(rows, ix, t1_keymap(rows, ix))
    if hit:
        problems.append("tables/table1_baseline.csv: outcome rows %s and %s differ by 1 to 9 in %s" % hit)
    # 4. subgroup tables: the same level under a narrower denominator
    for rel in ("tables/table_subgroups.csv", "derived/subgroup_forest_data.csv"):
        rows = parse_csv(read_text(os.path.join(out, rel)))
        ix = {h: i for i, h in enumerate(rows[0])}
        cell = {(r[ix["Subgroup"]], r[ix["Level"]], r[ix["Score"]]): r for r in rows[1:]}
        for (sg, lv, sc), r in cell.items():
            for small, large in SUBGROUP_NEST:
                if sc == large and (sg, lv, small) in cell:
                    for c in SUBGROUP_COLS:
                        a, b = to_int(cell[(sg, lv, small)][ix[c]]), to_int(r[ix[c]])
                        if a is not None and b is not None and 0 < b - a < 10:
                            problems.append("%s: %s=%s %s differs between %s and %s by 1 to 9" % (rel, sg, lv, c, small, large))
    # 5. strata of the change-point curves
    d = json.loads(read_text(os.path.join(out, "derived/changepoint_stratified.json")))
    for outcome in ("composite", "mortality"):
        for meas in ("n", "events"):
            bad = strata_small_diffs(d, outcome, meas)
            if bad:
                problems.append("derived/changepoint_stratified.json: %s %s strata differ by 1 to 9: %s" % (outcome, meas, bad))
    # 6. operating points: nested rows of tables 5, 5c and 5d
    ext = external_rows_5d(read_text(os.path.join(out, "tables/table5d_matched_workload.csv"))) + \
        external_rows_5(read_text(os.path.join(out, "tables/table5_operating.csv")))
    r5 = parse_csv(read_text(os.path.join(out, "tables/table5c_operating_full.csv")))
    i5 = {h: i for i, h in enumerate(r5[0])}
    recs = [mk_row(family_of(r[i5["Instrument"]]), den_rank_5c(r[i5["Denominator"]]), fnum(r[i5["Cutoff"]]), r[i5["Cutoff_Direction"]],
                   r[i5["Rule"]], int(float(r[i5["TP"]])), int(float(r[i5["FP"]])), int(float(r[i5["TN"]])), int(float(r[i5["FN"]])), "table5c")
            for r in r5[1:]]
    allr = recs + ext
    for i in range(len(allr)):
        for j in range(i + 1, len(allr)):
            if any_violation(allr[i], allr[j]):
                problems.append("operating-point rows %s/%s and %s/%s are nested and differ by 1 to 9" % (allr[i]["src"], allr[i]["t"], allr[j]["src"], allr[j]["t"]))
    if problems:
        sys.exit("NESTED-DIFFERENCE CHECK FAILED:\n  " + "\n  ".join(problems[:40]))
    return len(cells), len(head), len(allr)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proj", required=True, help="local project folder containing v3/ and v4/")
    ap.add_argument("--out", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="repository root to write into (default: parent of scripts/)")
    ap.add_argument("--forbid-names", default="", help="comma-separated surnames that must not appear in any shipped file")
    args = ap.parse_args()
    proj, out = os.path.abspath(args.proj), os.path.abspath(args.out)
    global ABS_ROOT, NAME_RE
    ABS_ROOT = proj
    names = [n for n in args.forbid_names.split(",") if n.strip()]
    NAME_RE = re.compile("|".join([r"/" + "Users/", r"Dr\. \w+", r"H-5\d+", r"Baylor", r"Ben Taub"] + [r"(?<![A-Za-z])" + re.escape(n.strip()) + r"(?![A-Za-z])" for n in names]), re.I)

    shipped = []   # (relative shipped path, source relative path, mode)

    def ship_bytes(src_rel, dst_rel, mode="copy"):
        data = open(os.path.join(proj, src_rel), "rb").read()
        write_bytes(os.path.join(out, dst_rel), data)
        shipped.append((dst_rel, src_rel, mode))

    def ship_text(src_rel, dst_rel, handler=None, ext=None):
        raw = read_text(os.path.join(proj, src_rel))
        mode = "copy"
        txt = raw
        if handler is not None:
            txt = handler(raw, dst_rel, *(ext or ()))
            mode = "suppressed"
        txt2 = scrub_text(txt, dst_rel)
        if txt2 != raw and mode == "copy":
            mode = "scrubbed"
        write_bytes(os.path.join(out, dst_rel), txt2.encode("utf-8"))
        shipped.append((dst_rel, src_rel, mode))

    # name-collision check across the flat layout
    targets = []
    for n in V3_DERIVED_JSON + V3_DERIVED_CSV + V4_DERIVED_CSV + V4_DERIVED_JSON + ["results_bundle.json"]:
        targets.append("derived/" + n)
    for n in V3_TABLES + V4_TABLES + ["missingness.csv"]:
        targets.append("tables/" + n)
    for n in V3_FIGURES_PNG + V3_FIGURES_PDF + V4_FIGURES_PNG:
        targets.append("figures/" + n)
    for n in V3_LOGS_COPY + ["qc_manifest.json"]:
        targets.append("logs/" + n)
    for n in SNAPSHOT_FILES + ["MANIFEST.sha256"]:
        targets.append("logs/pre_d11_snapshot/" + n)
    dup = {t for t in targets if targets.count(t) > 1}
    if dup:
        sys.exit("name collision in flat layout: %s" % sorted(dup))

    # ---- derived (v3) ----
    for n in V3_DERIVED_JSON:
        h = {"score_card.json": handle_score_card, "changepoint_stratified.json": handle_changepoint_stratified,
             "reclass.json": handle_reclass}.get(n)
        ship_text("v3/derived/" + n, "derived/" + n, h)
    for n in V3_DERIVED_CSV:
        h = {"risk_tiers.csv": handle_risk_tiers, "subgroup_forest_data.csv": handle_subgroups}.get(n)
        ship_text("v3/derived/" + n, "derived/" + n, h)
    # ---- derived (v4) ----
    for n in V4_DERIVED_CSV:
        h = {"a1_score_distribution_test.csv": handle_a1_dist, "a2_hypotension_mechanism.csv": handle_a2}.get(n)
        ship_text("v4/derived/" + n, "derived/" + n, h)
    for n in V4_DERIVED_JSON:
        h = handle_a2_worked if n == "a2_worked_example.json" else None
        ship_text("v4/derived/" + n, "derived/" + n, h)
    ship_text("v4/bundle/results_bundle.json", "derived/results_bundle.json")
    # ---- tables ----
    ext5 = external_rows_5d(read_text(os.path.join(proj, "v3/tables/table5d_matched_workload.csv"))) + \
        external_rows_5(read_text(os.path.join(proj, "v3/tables/table5_operating.csv")))
    for n in V3_TABLES:
        if n == "table1_baseline.csv":
            ship_text("v3/tables/" + n, "tables/" + n, handle_table1_baseline)
        elif n == "table5c_operating_full.csv":
            ship_text("v3/tables/" + n, "tables/" + n, handle_table5c, (ext5,))
        elif n == "table7_sensitivity.csv":
            ship_text("v3/tables/" + n, "tables/" + n, handle_table7_sensitivity)
        elif n == "table_subgroups.csv":
            ship_text("v3/tables/" + n, "tables/" + n, handle_subgroups)
        else:
            ship_text("v3/tables/" + n, "tables/" + n)
    for n in V4_TABLES:
        if n == "Supplementary Table S1.csv":
            ship_bytes("v4/tables/" + n, "tables/" + n)   # CRLF file, copied byte for byte
        else:
            ship_text("v4/tables/" + n, "tables/" + n)
    ship_text("v3/logs/missingness.csv", "tables/missingness.csv")
    # ---- figures ----
    for n in V3_FIGURES_PNG + V3_FIGURES_PDF:
        ship_bytes("v3/figures/" + n, "figures/" + n)
    for n in V4_FIGURES_PNG:
        ship_bytes("v4/figures/" + n, "figures/" + n)
    # ---- logs read by the analysis scripts ----
    ship_text("v3/logs/qc_manifest.json", "logs/qc_manifest.json", handle_qc_manifest)
    for n in V3_LOGS_COPY:
        ship_text("v3/logs/" + n, "logs/" + n)
    snap_hashes = []
    for n in SNAPSHOT_FILES:
        ship_text(SNAPSHOT_DIR + "/" + n, "logs/pre_d11_snapshot/" + n)
        snap_hashes.append((sha256_file(os.path.join(out, "logs/pre_d11_snapshot/" + n)), n))
    # MANIFEST.sha256: the original manifest lines for the shipped files, checked against the originals
    orig = {}
    for line in read_text(os.path.join(proj, SNAPSHOT_DIR, "MANIFEST.sha256")).splitlines():
        h, _, name = line.partition("  ")
        orig[name.strip()] = h.strip()
    for h, n in snap_hashes:
        if orig.get(n) != h:
            sys.exit("snapshot file %s does not match the original MANIFEST.sha256" % n)
    write_bytes(os.path.join(out, "logs/pre_d11_snapshot/MANIFEST.sha256"),
                "".join("%s  %s\n" % (h, n) for h, n in snap_hashes).encode("utf-8"))
    shipped.append(("logs/pre_d11_snapshot/MANIFEST.sha256", SNAPSHOT_DIR + "/MANIFEST.sha256", "subset"))
    note("logs/pre_d11_snapshot/MANIFEST.sha256", "subset",
         "the original manifest lists five files; the lines of the three shipped files are kept (each hash checked against the original), the lines of the two files not shipped are removed")

    # ---- calculator data: derived from the shipped files, never edited by hand ----
    calc_card = read_text(os.path.join(out, "derived/score_card.json"))
    write_bytes(os.path.join(out, CALC_CARD), calc_card.encode("utf-8"))
    tiers_rows = parse_csv(read_text(os.path.join(out, "derived/risk_tiers.csv")))
    drop_ix = tiers_rows[0].index("n_override_in_tier")
    write_bytes(os.path.join(out, CALC_TIERS),
                dump_csv([[c for i, c in enumerate(r) if i != drop_ix] for r in tiers_rows]).encode("utf-8"))
    calc_files = [(CALC_CARD, "derived/score_card.json", "byte copy"),
                  (CALC_TIERS, "derived/risk_tiers.csv", "same file without the column n_override_in_tier")]

    # ---- checks on what was written ----
    problems = []
    for rel, src, mode in shipped + [(c[0], c[1], c[2]) for c in calc_files]:
        p = os.path.join(out, rel)
        if os.path.getsize(p) > 5 * 1024 * 1024:
            problems.append("over 5 MB: " + rel)
        if rel.endswith((".csv", ".json")):
            txt = read_text(p)
            for m in NAME_RE.finditer(txt):
                if METHOD_NAME_RE.match(txt, m.start()):
                    continue   # the Wilson score interval is a method, not a person
                problems.append("identifying text %r in %s" % (m.group(0), rel))
            if rel.endswith(".csv"):
                hdr = parse_csv(txt)[0]
                if any(h.lower() in ("inc_key", "patient_id", "id") for h in hdr):
                    problems.append("patient key column in " + rel)
    if problems:
        sys.exit("CHECK FAILED:\n  " + "\n  ".join(problems))

    ncells, nhead, nrows = check_nested_differences(out, [r for r, _, _ in shipped] + [c[0] for c in calc_files])
    print("nested-difference check passed: %d count cells compared within files, %d headline counts across files, %d operating-point rows" % (ncells, nhead, nrows))
    write_suppression_md(out, shipped, calc_files)
    # the withheld true values must not appear anywhere in what was written
    for rel in [r for r, _, _ in shipped if r.endswith((".csv", ".json"))] + [c[0] for c in calc_files]:
        txt = read_text(os.path.join(out, rel))
        for pat in FILE_SECRETS:
            if pat.search(txt):
                sys.exit("CHECK FAILED: a withheld true value appears in " + rel)
    for rel, pats in FILE_SECRETS_BY_FILE.items():
        txt = read_text(os.path.join(out, rel))
        for pat in pats:
            if pat.search(txt):
                sys.exit("CHECK FAILED: a withheld true value appears in " + rel)
    doc = read_text(os.path.join(out, "docs", "SUPPRESSION.md"))
    for pat in DOC_SECRETS:
        if pat.search(doc):
            sys.exit("CHECK FAILED: a withheld true value appears in docs/SUPPRESSION.md")
    print("shipped %d files; log entries %d" % (len(shipped), len(LOG)))


# ----------------------------------------------------------------------------
# docs/SUPPRESSION.md
# ----------------------------------------------------------------------------
TENSION = """\
## Where suppression weakens "sufficient to verify the reported numbers"

The Data Sharing Statement wording that matches this repository is: aggregated, de-identified tables and figures sufficient to verify the manuscript's reported numbers, with the exception of one post hoc subgroup count under 10 (cells with fewer than 10 patients suppressed, with complementary cells withheld). Every number printed in the manuscript was compared with the suppressed files. One printed number cannot be verified from the repository.

| Manuscript item | Printed value | Repository status |
|---|---|---|
| Discussion/Results, women aged 75 years or older who fell, systolic pressure below 90 mmHg and shock index of 1.0 or higher, test set | An event count under 10 among 94 patients, with its percentage and 95% confidence interval | The event count, the percentage and the interval are withheld in `derived/a2_hypotension_mechanism.csv` (the denominator, 94, is shown). A reader can see that the profile exists and how many patients it has, but cannot reproduce the printed event count, rate or interval from the repository. |

No other count under 10 is printed in the main text, Table 1, Table 2, Table 3, the figure legends or Supplementary Table S1. Checked items: every count in Table 1 (the smallest nonzero cell is 95), Table 3 (the smallest is 742), every n printed in the figures, the override count 201 and its 164 events (both shown), and every patient count in Supplementary Table S1 (the smallest is 630). Counts that the repository withholds and the manuscript does not print: the override patients in the Rule-out tier (a count between 1 and 9), the override patients in the Observe tier (withheld as the complementary cell), the patients with no signs of life among older patients on anticoagulants in the extended Table 1 (a count between 1 and 9), two recovered-death outcome rows of the extended Table 1 (counts between 1 and 9), the A1 score-card tail scores, and the extreme cutoffs of the operating-point table.
"""


def write_suppression_md(out, shipped, calc_files):
    lines = []
    A = lines.append
    A("# Small-cell suppression")
    A("")
    A("Every aggregate file in `tables/`, `derived/` and `figures/` was produced by `scripts/suppress_small_cells.py` from the author's local project folder. The script is the exact record of what was done. It is deterministic and has no randomness.")
    A("")
    A("## Rule")
    A("")
    A("1. Counts of 1 to 9 patients are replaced by the marker `%s`. Zero is not suppressed." % PRIMARY)
    A("2. Complementary disclosure. A withheld cell must not be recoverable by subtracting the other cells from a shown total or from a sibling row or column. Where it would be, one more cell of the same additive relation, the smallest remaining one, is replaced by the marker `%s`. Where cells are nested (cumulative operating-point rows, ordered scores), rows are dropped or adjacent categories are merged so that no difference between two shown rows is a count of 1 to 9. The same holds for definitions and denominators that contain one another: an outcome variant that adds a few patients to another, a denominator that needs one more measurement (the shock index needs a pulse as well as a pressure, the reverse shock index also needs the Glasgow Coma Scale), a stratum of a curve restricted to patients with a recorded vital sign. When two such counts, or two totals that a reader can derive by adding and subtracting shown cells, differ by 1 to 9, the cell of the larger set is withheld, with its complementary cells and the statistics computed from them." % SECONDARY)
    A("3. A percentage, rate, confidence interval, standardized mean difference or P value computed from a suppressed count is replaced by `%s`." % SECONDARY)
    A("4. Numbers that are not counts of patients are never suppressed.")
    A("")
    A("## Machine-readable markers")
    A("")
    A("| Marker | Meaning |")
    A("|---|---|")
    A("| `%s` | The true count is 1 to 9. |" % PRIMARY)
    A("| `%s` | The true value is not necessarily small. It is hidden so that a `%s` cell cannot be recovered by subtraction, or it is a statistic computed from a hidden count. |" % (SECONDARY, PRIMARY))
    A("")
    A("Both markers appear as the whole cell value, in CSV cells and as JSON string values. A shown integer from 1 to 9 outside the columns listed next is therefore a defect.")
    A("")
    A("## Values from 1 to 9 that are not counts and are left as they are")
    A("")
    A("| Where | Column or key | What the value is |")
    A("|---|---|---|")
    A("| calibration_data.csv, citl_slope.json, reclass.json | bin, decile | bin or decile index |")
    A("| subgroup_forest_data.csv, table_subgroups.csv | Level | ACS verification level or similar category code |")
    A("| table3_points.csv, table5 files, score_card.json, refit.json, a1_sensitivity_card.csv, a1_sensitivity_metrics.json, a2_worked_example.json | Points, points, a1_points, published_points, score_min, score_max, Cutoff, rule_in_cutpoint, rule_out_cutpoint | integer score points, scores and cutoffs |")
    A("| table2_lasso_stability.csv | CV_Abs_Beta_Rank | rank |")
    A("| table5_calibration.csv, citl_slope.json | HL_df, df, hl_df | degrees of freedom |")
    A("| table_consort.csv | Step | step number |")
    A("| table5d_matched_workload.csv | Is_workload_anchor, Workload_match_within_tolerance, Delta_comparable | 0/1 flags |")
    A("| table5d_matched_workload.csv | Delta_workload_gap_n | difference between the numbers flagged by two different instruments, each of several thousand patients; not a cell of patients |")
    A("| table5d_matched_workload.csv, table5c_operating_full.csv | Workload_nominal_target_pct, Workload_anchor_cutoff, Cutoff | percent targets and cutoffs |")
    A("| Table 1.csv, table1_baseline.csv, results_bundle.json | median (interquartile range) cells such as `9 (5-17)`, `6 (6-6)` | a summary of a measured value, not a count |")
    A("| table5c_operating_full.csv | Within_RuleOut_RuleIn_Window | 0/1 flag |")
    A("| *.json | boot_n_unique, boot_bp_n_unique, nonzero, boot, n_point_variables, n_surgical_categories | counts of bootstrap runs, model terms or code categories, not of patients |")
    A("| any file | Selection_Frequency, Calibration_Slope, Max, value and similar columns equal to 1.0 | proportions and coefficients |")
    A("")
    A("## Files shipped")
    A("")
    A("SHA-256 is of the shipped version. Mode: `copy` is byte-identical to the project file, `scrubbed` differs only by removal of an absolute path, a co-author attribution or an internal label, `suppressed` was altered by the rule above (including removal of items that fail the screen), `subset` is the original manifest reduced to the files that are shipped.")
    A("")
    A("| Shipped file | Source (project folder) | Mode | SHA-256 |")
    A("|---|---|---|---|")
    for rel, src, mode in sorted(shipped):
        A("| `%s` | `%s` | %s | `%s` |" % (rel, src, mode, sha256_file(os.path.join(out, rel))))
    A("")
    A("Calculator data files are produced by the same script from the shipped files above, never edited by hand:")
    A("")
    A("| Calculator file | Made from | Relation | SHA-256 |")
    A("|---|---|---|---|")
    for rel, src, how in calc_files:
        A("| `%s` | `%s` | %s | `%s` |" % (rel, src, how, sha256_file(os.path.join(out, rel))))
    A("")
    A("## Files altered")
    A("")
    for f, kind, detail in sorted(LOG, key=lambda x: (x[0], x[1])):
        A("- `%s`, %s: %s" % (f, kind, detail))
    A("")
    A("## Files deliberately not shipped")
    A("")
    A("| Source | Reason |")
    A("|---|---|")
    for s, why in NOT_SHIPPED:
        A("| %s | %s |" % (s, why))
    A("")
    A("`derived/results_bundle.json` is shipped. It passes the small-cell screen: its only integers from 1 to 9 are score points, cutpoints, flags and category counts. Only its three absolute path fields were made relative.")
    A("")
    A("## Logs that the analysis scripts read")
    A("")
    A("Three groups of files from the project's `v3/logs/` are shipped because scripts read them: `logs/qc_manifest.json` (read by scripts 02, 08, 09, 10, 12, 14, `render_figures.py`, `build_bundle.py` and `render_exhibits.py`), `logs/d11_invariance_check.json` and `.csv` (`build_bundle.py` requires the JSON; script 19 writes both) and three files of `logs/pre_d11_snapshot/` (the frozen baseline that scripts 05 and 19 compare against). Each was screened for counts of 1 to 9, absolute paths, names and patient-level content. Findings:")
    A("")
    A("- `qc_manifest.json`: two items removed (listed under Files altered); every other count is 0 or at least 10. It still contains internal wording (section references and labels of project planning documents that are not distributed). Re-running `analysis/01_build_cohort.py` rewrites this file with the removed items; do not commit a regenerated copy without screening it again.")
    A("- `d11_invariance_check.json` and `.csv`: AUROC values and point-sum ranges only. Shipped unchanged.")
    A("- `pre_d11_snapshot/`: `refit.json`, `table3_points.csv` and `table4_discrimination.csv` shipped unchanged (the SHA-256 of each equals the original manifest line). The snapshot's score card and operating table are not read by any script and are not shipped.")
    A("")
    A("## Notes on specific cases")
    A("")
    A("- `table5d_matched_workload.csv` and `table6b_elderly_ledger.csv` were listed as suspected, and contain no cell under 10. They are shipped unchanged. One note in table6b names a confidence-interval method; it is a statistical term, not a person.")
    A("- The override group has 201 patients with 164 events (both shown). Its split across the three score tiers is not published as a whole: the Rule-out tier count is a number between 1 and 9 (`<10`), the Observe tier count is withheld as the complementary cell so that the Rule-out count cannot be recovered by subtraction from the group total and the Escalate tier count, and the Escalate tier count is shown in `derived/risk_tiers.csv` and `derived/score_card.json`. Documents and the README must not quote the split by tier.")
    A("- `figure9_score_card` is not shipped because its picture prints a count between 1 and 9. The calculator in `calculator/` shows the same card from `derived/score_card.json`.")
    A("- Pairwise checks on the operating-point tables compare every pair of rows of the same instrument across `table5c`, `table5d` and `table5_operating`. They do not test every possible sum of three or more rows.")
    A("- Nested definitions and denominators (rule 2, last sentences). Files altered by it: `tables/table7_sensitivity.csv` (the variant `sens_death_dayfallback`, its event total and rate in the full-cohort and test-slice rows), `tables/table1_baseline.csv` (two rows of the untimed-death recovery ladder, in the strata columns and their SMD and P value), `tables/table_subgroups.csv` and `derived/subgroup_forest_data.csv` (shock-index event counts of two subgroups and the complementary levels), `derived/changepoint_stratified.json` and `derived/reclass.json` (n and events of the anticoagulant strata of the shock-index change-point curves). Each is listed with its cells under Files altered; the withheld values are not stated. The change-point estimates, confidence intervals and all other numbers in those files are unchanged.")
    A("- Self-check for nested differences. After writing the files the generator compares every pair of counts of one measure within each shipped file, every pair of headline event and denominator totals across files, the outcome rows of Table 1, the subgroup tables, the change-point strata (including cells derived by subtraction) and the operating-point rows, and stops if any two that could be nested sets differ by 1 to 9. Pairs that are not nested (cells of one partition, different variables of one table) are exempt by a stated rule in the script.")
    A("- One nested difference is left and cannot be withheld here: the event count of the full-information (ISS and AIS) reference row of Table 3 differs from the primary event count by a number between 1 and 9 (the denominators differ by 30 patients). Both counts are printed in the manuscript's Table 3, which this repository reproduces unchanged, and `logs/pre_d11_snapshot/table4_discrimination.csv` must stay byte-identical to the frozen baseline. The generator lists it as a known exception.")
    A("- The generator checks its own output: it stops if the withheld override values, written as a field value or as a split, appear in any shipped data file or in this document.")
    A("")
    A(TENSION)
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    with open(os.path.join(out, "docs", "SUPPRESSION.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
