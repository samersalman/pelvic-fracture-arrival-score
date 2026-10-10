# Data availability

## What is in this repository

Only aggregate outputs, the analysis code and the calculator. No patient-level data, no patient identifiers and no free-text fields are stored here.

- `tables/`: result tables as CSV. Counts are cohort-level, risk-stratum-level or subgroup-level.
- `derived/`: aggregate and intermediate outputs of the pipeline (JSON and CSV), including the fitted model (`refit.json`), the score card (`score_card.json`), the tier table (`risk_tiers.csv`) and a results bundle (`results_bundle.json`).
- `figures/`: manuscript figures as PNG, with PDF versions of some of them.
- `inputs/extractability_audit/`: a small reference table of the candidate predictors and the reasons some were excluded.
- `logs/`: a quality-control manifest and the record of an invariance check that the analysis scripts read.
- `analysis/`, `calculator/`, `tests/`, `scripts/`: code.

## Small-cell suppression

The rule applied to every shared aggregate file is:

- Counts of 1 to 9 patients are replaced by the marker `<10`. Zero is not suppressed.
- Where a suppressed count could be recovered by subtracting the other cells from a shown total, one further cell of the same relation is replaced by the marker `withheld`. Where rows are nested, as in cumulative operating-point tables, rows are dropped or adjacent categories are merged.
- Percentages, rates, confidence intervals and test statistics computed from a suppressed count are also marked `withheld`.
- Values that are not counts of patients (point weights, cutpoints, coefficients, AUROCs, bin numbers) are not suppressed.

This is a rule of fewer than 10 patients per cell. It is not a minimum cell size of any larger number, and cells with 10 or more patients are shown as they are. The categories of file that were affected are the baseline table, operating-point and matched-workload tables, subgroup tables, the score-distribution table, one table from a post-hoc analysis of low blood pressure, and the tier table. Some figures are not shipped, or are shipped as PNG only, because a plot can encode small counts that cannot be suppressed inside an image.

The full rule, the exact markers, the list of shipped files with checksums and the list of files that were not shared are in [`SUPPRESSION.md`](SUPPRESSION.md). It was written by `scripts/suppress_small_cells.py`, which is the exact record of what was done.

The shipped tables allow the numbers printed in the manuscript to be checked against aggregate outputs, with one exception: the event count, percentage and confidence interval for one post hoc subgroup (a count under 10) are withheld. The shipped files are for inspection. Re-running most analysis scripts needs patient-level intermediate files built from the NTDB (see [`REPRODUCE.md`](REPRODUCE.md)).

## What is not in this repository

Patient-level NTDB data are not redistributed and must be obtained directly from the American College of Surgeons (ACS). This covers the raw Public Use Files and every file derived from them at the level of the individual patient, including the cohort file, the engineered-feature table, the scored test set, the lists of patient keys that define the splits, and serialized model objects. The pipeline creates these locally when you run it (see `analysis/README_PIPELINE.md`), and `.gitignore` keeps them out of the repository.

## How to obtain the NTDB Public Use Files

The NTDB Public Use Files are distributed by the ACS under a data use agreement.

1. Go to the National Trauma Data Bank page of the ACS: <https://www.facs.org/quality-programs/trauma/quality/national-trauma-data-bank/>.
2. Submit a research data application and sign the data use agreement.
3. Request admission years 2019 to 2024 to match the cohort used here.

The pipeline was run on a compiled version of the Public Use Files (four compiled multi-year CSV files plus the yearly `PUF_TRAUMA.csv` files). The expected folder layout is documented at the top of `analysis/config.py` and in `analysis/README_PIPELINE.md`. The scripts that compiled those four files are not part of this repository; the columns that step 01 reads from them are named in `analysis/01_build_cohort.py`, which is the specification for rebuilding equivalent files from the raw files. The ACS files are de-identified at source before release.

## Using the shared files

Rebuilding the analysis writes unsuppressed outputs over the shipped files. Use a separate copy of the repository for a rebuild, and review any rebuilt table against your own data use agreement before you share it.
