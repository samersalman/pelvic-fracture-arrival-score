#!/usr/bin/env python3
"""Check the numbers, links and paths quoted in the README and the docs.

Runs from a clean clone and uses only files that ship with the repository and the
Python standard library:

    python scripts/check_readme_numbers.py

What it does
  1. CLAIMS: an explicit list of every number-like claim made in README.md,
     docs/REPRODUCE.md and docs/DATA_AVAILABILITY.md. For each claim it checks that the
     quoted text is present in the document and that the value matches the shipped
     tables or derived files.
  2. COVERAGE: every number-like token left in the prose of those documents (code
     blocks, inline code, URLs, link targets, HTML comments and step or tier labels
     removed) must be covered by a claim above. A new, unchecked number fails.
  3. PATHS: every relative link, every file path in inline code and every entry of the
     repository tree in the README must exist in the repository.
  4. HYGIENE: no em-dash in the documents and metadata files.
  5. PLACEHOLDERS: reports whether the release placeholders (written between
     two at-signs) are still present, which is expected until scripts/add_doi.py runs.

Exit status 0 when everything passes, 1 otherwise. Two platform facts (the Render free
tier idle time and cold-start time) cannot be checked against any file; they are listed
as not verifiable here and do not fail the run.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "docs/REPRODUCE.md", "docs/DATA_AVAILABILITY.md"]
META = ["README.md", "CITATION.cff", ".zenodo.json", "LICENSE", "docs/REPRODUCE.md", "docs/DATA_AVAILABILITY.md"]
EN_DASH = chr(0x2013)
EM_DASH = chr(0x2014)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def rows(rel: str) -> list[dict]:
    with open(ROOT / rel, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def js(rel: str):
    return json.loads(text(rel))


def fmt(n) -> str:
    return f"{int(n):,}"


# ---------------------------------------------------------------------------
# verifiers: each returns (ok, detail)
# ---------------------------------------------------------------------------
def v_cohort():
    c = rows("tables/table_consort.csv")[-1]
    s = {r["split"]: r for r in rows("tables/table_split_balance.csv")}
    b = js("derived/results_bundle.json")["denominators"]["cohort_full"]["value"]
    t1 = rows("tables/Table 1.csv")[0]
    hdr = list(t1.keys())[1]
    vals = [int(c["N_after"]), int(s["full"]["N"]), int(b)]
    ok = all(v == 239736 for v in vals) and "n = 239,736" in hdr
    return ok, f"consort {vals[0]}, split table {vals[1]}, results bundle {vals[2]}, Table 1 header '{hdr}'"


def v_events():
    s = {r["split"]: r for r in rows("tables/table_split_balance.csv")}
    e = int(s["full"]["N_events"])
    pct = round(float(s["full"]["event_rate"]) * 100, 1)
    hdr = list(rows("tables/Table 1.csv")[0].keys())[3]
    return e == 26479 and pct == 11.0 and "n = 26,479" in hdr, f"events {e}, rate {pct}%, Table 1 header '{hdr}'"


def v_split():
    s = {r["split"]: r for r in rows("tables/table_split_balance.csv")}
    tr, te = int(s["train"]["N"]), int(s["test"]["N"])
    b = js("derived/results_bundle.json")["denominators"]["test_set"]["value"]
    card = js("derived/score_card.json")["n_test"]
    return (tr, te, b, card) == (159824, 79912, 79912, 79912), f"train {tr}, test {te}, bundle {b}, card {card}"


def v_auroc_test():
    r = next(x for x in rows("tables/table4_discrimination.csv")
             if x["Outcome"] == "major_hemorrhage_event" and x["Score"] == "pelvic_score")
    t3 = rows("tables/Table 3.csv")[1]["Result"]
    q = js("derived/results_bundle.json")["quantities"]["disc.major_hemorrhage_event.pelvic_score.auroc_full"]["printed"]
    want = f"0.8823 (0.8787{EN_DASH}0.8860)"
    ok = (float(r["AUROC"]), float(r["CI_Low"]), float(r["CI_High"])) == (0.8823, 0.8787, 0.886) \
        and t3 == want and q == want and int(r["N_used"]) == 79912
    return ok, f"table4 {r['AUROC']} ({r['CI_Low']}, {r['CI_High']}), Table 3 '{t3}', bundle '{q}'"


def v_auroc_temporal():
    r = next(x for x in rows("tables/table4b_temporal.csv") if x["outcome"] == "major_hemorrhage_event")
    t3 = next(x for x in rows("tables/Table 3.csv") if "temporal validation" in x["Row"])["Result"]
    q = js("derived/results_bundle.json")["quantities"]["temporal.major_hemorrhage_event.auroc_integer_full"]["printed"]
    want = f"0.8846 (0.8812{EN_DASH}0.8881)"
    ok = (float(r["AUROC_integer"]), float(r["CI_low_integer"]), float(r["CI_high_integer"])) == (0.8846, 0.8812, 0.8881) \
        and t3 == want and q == want
    return ok, f"table4b {r['AUROC_integer']} ({r['CI_low_integer']}, {r['CI_high_integer']}), Table 3 '{t3}', bundle '{q}'"


def v_temporal_years():
    r = next(x for x in rows("tables/table4b_temporal.csv") if x["outcome"] == "major_hemorrhage_event")
    flow = rows("tables/table_consort.csv")[0]["Step_Label"]
    ok = r["derivation_years"] == "2019-2022" and r["validation_years"] == "2023-2024" and "2019-2024" in flow
    return ok, f"derivation {r['derivation_years']}, validation {r['validation_years']}, cohort flow label '{flow}'"


def v_tiers():
    t = {r["tier"]: r for r in rows("derived/risk_tiers.csv")}
    c = {r["tier"]: r for r in rows("calculator/data/risk_tiers.csv")}
    card = {x["tier"]: x for x in js("derived/score_card.json")["risk_tiers"]}
    got = [round(float(t[k]["composite_rate"]) * 100, 1) for k in ("rule_out", "observe", "escalate")]
    got_c = [round(float(c[k]["composite_rate"]) * 100, 1) for k in ("rule_out", "observe", "escalate")]
    got_j = [round(float(card[k]["composite_rate"]) * 100, 1) for k in ("rule_out", "observe", "escalate")]
    ok = got == got_c == got_j == [1.6, 10.4, 47.2]
    return ok, f"risk_tiers.csv {got}, calculator copy {got_c}, score_card.json {got_j}"


def v_cutpoints():
    t = {r["tier"]: r for r in rows("derived/risk_tiers.csv")}
    card = js("derived/score_card.json")
    ok = (int(t["rule_out"]["score_max"]) == 0 and int(t["observe"]["score_min"]) == 1
          and int(t["observe"]["score_max"]) == 4 and int(t["escalate"]["score_min"]) == 5
          and float(card["rule_out_cutpoint"]) == 1.0 and float(card["rule_in_cutpoint"]) == 5.0)
    return ok, (f"rule_out max {t['rule_out']['score_max']}, observe {t['observe']['score_min']} to "
                f"{t['observe']['score_max']}, escalate min {t['escalate']['score_min']}, cutpoints "
                f"{card['rule_out_cutpoint']} and {card['rule_in_cutpoint']}")


def v_thirteen():
    pts = [r for r in rows("tables/table3_points.csv") if r["Role"] == "point"]
    nonzero = [r["Variable"] for r in pts if r["Points"] not in ("0", "")]
    zero = [r["Variable"] for r in pts if r["Points"] in ("0",)]
    card = js("derived/score_card.json")["variables"]
    card_nz = sum(1 for v in card if v.get("role") == "point" and int(v["points"]) != 0)
    ok = len(nonzero) == 13 and zero == ["cc_anticoag"] and card_nz == 13
    return ok, f"non-zero point rows {len(nonzero)}, zero-point rows {zero}, score_card non-zero {card_nz}"


def v_python():
    req = text("requirements.txt").splitlines()[0]
    pipe = text("analysis/README_PIPELINE.md")
    render = text("calculator/render.yaml")
    ok = "Python 3.14" in req and "3.14.0" in req and "Python 3.14" in pipe and "PYTHON_VERSION" in render and "3.12" in render
    return ok, f"requirements.txt line 1 '{req}', render.yaml pins 3.12"


def v_step04():
    pipe = text("analysis/README_PIPELINE.md")
    ok = "about 30 minutes" in pipe and (ROOT / "tables/table2_lasso_selected.csv").exists()
    return ok, "README_PIPELINE says about 30 minutes; tables/table2_lasso_selected.csv ships"


def v_seed():
    ok = "(42)" in text("analysis/README_PIPELINE.md")
    return ok, "README_PIPELINE states a fixed seed (42)"


def v_ports():
    app = text("calculator/app.py")
    procfile = text("calculator/Procfile")
    render = text("calculator/render.yaml")
    ok = "port=5000" in app and "--bind" not in procfile and "--bind" not in render
    return ok, "app.py main block uses port 5000; no --bind in Procfile or render.yaml (gunicorn default port 8000)"


def v_version():
    ok = "version: 1.0.0" in text("CITATION.cff") and js(".zenodo.json")["version"] == "1.0.0"
    return ok, "CITATION.cff and .zenodo.json both carry version 1.0.0"


def v_year():
    ok = "Copyright (c) 2026" in text("LICENSE") and "year: 2026" in text("CITATION.cff")
    return ok, "LICENSE and CITATION.cff both say 2026"


def v_suppression():
    s = text("docs/SUPPRESSION.md")
    ok = "Counts of 1 to 9 patients are replaced by the marker `<10`" in s and "Zero is not suppressed" in s
    return ok, "docs/SUPPRESSION.md states the 1 to 9 rule with the <10 marker"


def v_ci_level():
    hdr = list(rows("tables/Table 2.csv")[0].keys())
    ok = any("95% CI" in h for h in hdr)
    return ok, f"Table 2 header carries '95% CI': {hdr[2]}"


def v_platform():
    return None, "platform behaviour of the Render free tier; not checkable against any shipped file"


# claim = (document, quoted text that must appear, numeric tokens it covers, verifier)
CLAIMS = [
    ("README.md", "239,736 eligible patients", ["239,736"], v_cohort),
    ("README.md", "26,479 (11.0%)", ["26,479", "11.0%"], v_events),
    ("README.md", "0.8823 (95% CI 0.8787 to 0.8860)", ["0.8823", "0.8787", "0.8860"], v_auroc_test),
    ("README.md", "95% CI", ["95%"], v_ci_level),
    ("README.md", "0.8846 (95% CI 0.8812 to 0.8881)", ["0.8846", "0.8812", "0.8881"], v_auroc_temporal),
    ("README.md", "admission years 2019 to 2024", ["2019", "2024"], v_temporal_years),
    ("README.md", "on 2023 to 2024 admissions", ["2023"], v_temporal_years),
    ("README.md", "re-estimated on 2019 to 2022", ["2022"], v_temporal_years),
    ("README.md", "13 point-scoring criteria", ["13"], v_thirteen),
    ("README.md", "with 0 points", ["0"], v_thirteen),
    ("README.md", "Rule-out (score below 1), Observe (score 1 to 4)", ["1", "4"], v_cutpoints),
    ("README.md", "Escalate (score 5 or higher", ["5"], v_cutpoints),
    ("README.md", "1.6% (Rule-out), 10.4% (Observe) and 47.2% (Escalate)", ["1.6%", "10.4%", "47.2%"], v_tiers),
    ("README.md", "Python 3.12", ["3.12"], v_python),
    ("README.md", "about 15 minutes", ["15"], v_platform),
    ("README.md", "roughly 30 to 60 seconds", ["30", "60"], v_platform),
    ("README.md", "(port 5000)", ["5000"], v_ports),
    ("README.md", "about 30 minutes", [], v_step04),
    ("README.md", "(0.8823)", [], v_auroc_test),
    ("README.md", "Counts below 10 patients are suppressed", ["10"], v_suppression),
    ("README.md", "cells under 10 patients suppressed", [], v_suppression),
    ("README.md", "version   = {1.0.0}", ["1.0.0"], v_version),
    ("README.md", "year      = {2026}", ["2026"], v_year),
    ("README.md", "year   = {2026}", [], v_year),
    ("docs/REPRODUCE.md", "239,736 patients", ["239,736"], v_cohort),
    ("docs/REPRODUCE.md", "| 26,479 |", ["26,479"], v_events),
    ("docs/REPRODUCE.md", "159,824 and 79,912 patients", ["159,824", "79,912"], v_split),
    ("docs/REPRODUCE.md", "0.8823 (95% CI 0.8787 to 0.8860)", ["0.8823", "0.8787", "0.8860"], v_auroc_test),
    ("docs/REPRODUCE.md", "(0.8846)", ["0.8846"], v_auroc_temporal),
    ("docs/REPRODUCE.md", "95% CI", ["95%"], v_ci_level),
    ("docs/REPRODUCE.md", "admission years 2019 to 2024", ["2019", "2024"], v_temporal_years),
    ("docs/REPRODUCE.md", "Python 3.14", ["3.14", "3.14.0"], v_python),
    ("docs/REPRODUCE.md", "about 30 minutes", ["30"], v_step04),
    ("docs/REPRODUCE.md", "random seed of 42", ["42"], v_seed),
    ("docs/REPRODUCE.md", "cells under 10 patients suppressed", ["10"], v_suppression),
    ("docs/DATA_AVAILABILITY.md", "Counts of 1 to 9 patients", ["1", "9"], v_suppression),
    ("docs/DATA_AVAILABILITY.md", "fewer than 10 patients per cell", ["10"], v_suppression),
    ("docs/DATA_AVAILABILITY.md", "admission years 2019 to 2024", ["2019", "2024"], v_temporal_years),
]

# Number-like tokens that are labels rather than claims would be listed here with a
# reason. The list is empty: every number in the prose must be covered by a claim.
ALLOWED_TOKENS: dict[str, str] = {}


def strip_for_scan(src: str) -> str:
    s = re.sub(r"<!--\s*add_doi:[a-z]+:begin\s*-->.*?<!--\s*add_doi:[a-z]+:end\s*-->", " ", src, flags=re.S)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`[^`\n]*`", " ", s)
    s = re.sub(r"\]\([^)]*\)", "] ", s)
    s = re.sub(r"<https?://[^>]*>", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\bTiers? \d\b", " ", s)
    s = re.sub(r"\bTable S?\d\b", " ", s)  # exhibit labels
    s = re.sub(r"\b[Ss]teps? \d\d(?: to \d\d)?(?: and \d\d)?\b", " ", s)
    s = re.sub(r"\b[Ss]ection \d\b", " ", s)
    s = re.sub(r"^[#\s]*\d+\.\s", " ", s, flags=re.M)  # numbered list and heading markers
    s = re.sub(r"\(section \d\)", " ", s)
    s = re.sub(r"`", " ", s)
    s = re.sub(r"\b[Ss]teps? \d\d\b", " ", s)
    return s


NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)*%?(?![\w])")


def tokens(src: str) -> list[str]:
    out = []
    for m in NUM.finditer(src):
        t = m.group(0).rstrip(",")
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# path checks
# ---------------------------------------------------------------------------
EXTERNAL_NAMES = {
    "PUF_TRAUMA.csv",  # raw NTDB file, not shipped
}
# Files that a rebuild writes (step 01) and that are deliberately not shipped. The documents
# name them as outputs to look at after a rebuild.
REBUILD_OUTPUTS = {
    "logs/cohort_flow.txt",
}
FILE_EXT = (".py", ".csv", ".json", ".md", ".yaml", ".yml", ".cff", ".txt", ".ini", ".png", ".pdf", ".html", ".css")


def exists_anywhere(name: str) -> bool:
    return any(ROOT.rglob(name)) if name else False


def check_paths() -> list[str]:
    problems: list[str] = []
    for rel in DOCS:
        src = text(rel)
        base = (ROOT / rel).parent
        body = re.sub(r"<!--\s*add_doi:[a-z]+:begin\s*-->.*?<!--\s*add_doi:[a-z]+:end\s*-->", " ", src, flags=re.S)
        body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
        # relative markdown links
        for m in re.finditer(r"\]\(([^)]+)\)", body):
            tgt = m.group(1).strip()
            if tgt.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = tgt.split("#")[0]
            if path and not (base / path).exists():
                problems.append(f"{rel}: broken link target {tgt}")
        # inline code spans that look like paths
        nofence = re.sub(r"```.*?```", " ", body, flags=re.S)
        for m in re.finditer(r"`([^`\n]+)`", nofence):
            span = m.group(1).strip()
            if any(ch in span for ch in "<>*$=|()") or " " in span and not span.endswith(FILE_EXT):
                continue
            if span.startswith(("/", "compiled_NTDB/", "PUF/")) or span in EXTERNAL_NAMES or span in REBUILD_OUTPUTS or span.startswith(("python", "pip")):
                continue
            is_dir = span.endswith("/")
            is_file = span.endswith(FILE_EXT) or span in {".gitignore", ".zenodo.json"}
            if not (is_dir or is_file):
                continue
            if is_file and span.endswith(".md") and span.startswith("../"):
                if not (base / span).exists():
                    problems.append(f"{rel}: missing path `{span}`")
                continue
            if "/" in span.rstrip("/"):
                if not ((ROOT / span).exists() or (base / span).exists()):
                    problems.append(f"{rel}: missing path `{span}`")
            elif is_dir:
                if not ((ROOT / span).is_dir() or (ROOT / "analysis" / span).is_dir()):
                    problems.append(f"{rel}: missing directory `{span}`")
            else:
                if not (ROOT / span).exists() and not exists_anywhere(span):
                    problems.append(f"{rel}: missing file `{span}`")
        # repository tree in the README
        if rel == "README.md":
            m = re.search(r"```\npelvic-fracture-arrival-score/\n(.*?)```", src, flags=re.S)
            if not m:
                problems.append("README.md: repository tree block not found")
            else:
                for line in m.group(1).splitlines():
                    depth_one = line.startswith(("├── ", "└── "))
                    content = re.sub(r"^[│ ]*[├└]── ", "", line)
                    content = content.split("#")[0].strip()
                    for name in [c.strip() for c in content.split(",") if c.strip()]:
                        isdir = name.endswith("/")
                        if not (isdir or name.endswith(FILE_EXT) or name in {".gitignore", "LICENSE"}):
                            continue
                        bare = name.rstrip("/")
                        if depth_one:
                            if not (ROOT / bare).exists():
                                problems.append(f"README.md tree: missing top-level entry {name}")
                        elif not exists_anywhere(bare):
                            problems.append(f"README.md tree: missing entry {name}")
    return problems


# ---------------------------------------------------------------------------
def main() -> int:
    failures = 0
    print("Claims")
    print("------")
    covered: dict[str, set[str]] = {d: set() for d in DOCS}
    for doc, quote, covers, verifier in CLAIMS:
        present = quote in text(doc)
        ok, detail = verifier()
        if ok is None:
            status = "NOT VERIFIABLE HERE" if present else "FAIL (quote missing)"
        else:
            status = "PASS" if (ok and present) else "FAIL"
        if not present:
            detail = f"quote not found in {doc}; " + detail
        if status.startswith("FAIL"):
            failures += 1
        else:
            covered[doc].update(covers)
        print(f"[{status}] {doc}: {quote!r}\n        {detail}")

    print("\nCoverage of number-like tokens in the prose")
    print("--------------------------------------------")
    for doc in DOCS:
        toks = tokens(strip_for_scan(text(doc)))
        bad = sorted({t for t in toks if t not in covered[doc] and t not in ALLOWED_TOKENS})
        if bad:
            failures += 1
            print(f"[FAIL] {doc}: unchecked number-like tokens {bad}")
        else:
            print(f"[PASS] {doc}: all {len(toks)} number-like tokens are covered")

    print("\nPaths and links")
    print("---------------")
    problems = check_paths()
    for p in problems:
        print(f"[FAIL] {p}")
    if problems:
        failures += 1
    else:
        print("[PASS] every relative link, inline-code path and README tree entry exists")

    print("\nDashes")
    print("------")
    dash_hits = [rel for rel in META if EM_DASH in text(rel)]
    if dash_hits:
        failures += 1
        print(f"[FAIL] em-dash found in {dash_hits}")
    else:
        print("[PASS] no em-dash in the documents and metadata files")

    print("\nRelease placeholders")
    print("--------------------")
    for rel in ["README.md", "CITATION.cff", ".zenodo.json"]:
        n = len(re.findall(r"@@[A-Z_]+@@", text(rel)))
        print(f"[INFO] {rel}: {n} placeholder(s) still present" + ("" if n else " (none)"))

    print("\nResult: " + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK GROUP(S) FAILED"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
