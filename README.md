# Pelvic-Fracture Arrival Score

A bedside integer score for early recognition of major hemorrhage in adults with blunt pelvic-ring fracture, built only from information available on arrival (age, sex, mechanism, systolic pressure, heart rate, Glasgow Coma Scale). It was derived and internally validated on the American College of Surgeons National Trauma Data Bank (NTDB) Public Use Files, admission years 2019 to 2024. Among 239,736 eligible patients, 26,479 (11.0%) had a major hemorrhage event. In the held-out test set the integer card had an area under the receiver-operating-characteristic curve (AUROC) of 0.8823 (95% CI 0.8787 to 0.8860), and in a temporal validation on 2023 to 2024 admissions (coefficients re-estimated on 2019 to 2022) the AUROC was 0.8846 (95% CI 0.8812 to 0.8881). This repository holds the analysis code, aggregate result tables and figures, and a small web calculator. It contains no patient-level data.

<!-- add_doi:badges:begin -->
<!-- placeholder: scripts/add_doi.py replaces this block after the Zenodo release -->
[![DOI](https://zenodo.org/badge/DOI/@@DOI_CONCEPT@@.svg)](https://doi.org/@@DOI_CONCEPT@@)
[![Calculator](https://img.shields.io/badge/calculator-live-brightgreen)](@@RENDER_URL@@)
<!-- add_doi:badges:end -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research use only.** The score has internal and temporal validation only. It has not been externally validated, it is not a diagnostic device, and it does not replace clinical judgment. It does not localize bleeding to the pelvis. The calculator reports the observed event rate of a stratum in the held-out test set, which is not an individual probability.

## How the score works

- The card has 13 point-scoring criteria (see `tables/table3_points.csv` for the weights). Anticoagulant use was forced into the model as a pre-specified anchor and rounds to zero points, so it adds nothing to the score. It appears in `tables/table3_points.csv` with 0 points; it is not printed on the card in the paper.
- An automatic-escalation override applies to a patient with no signs of life on arrival, defined as an unrecordable systolic pressure. It sits outside the point scale.
- Strata: Rule-out (score below 1), Observe (score 1 to 4), Escalate (score 5 or higher, or the override).
- Observed major hemorrhage rate in the held-out test set by score stratum: 1.6% (Rule-out), 10.4% (Observe) and 47.2% (Escalate). These are score-only strata, and the full table with confidence intervals is `derived/risk_tiers.csv`.
- Unrecorded vital signs add zero points, as in the derivation.

## Calculator

<!-- add_doi:calculator:begin -->
Hosted instance: @@RENDER_URL@@
<!-- add_doi:calculator:end -->

- It runs on the free tier of Render. The service sleeps after about 15 minutes without traffic, so the first request after a pause can take roughly 30 to 60 seconds (a cold start). The `/healthz` page is the quick way to wake it.
- No data entered into the calculator are stored. There is no database and no logging of inputs by the application.
- The code is in `calculator/`. Its point weights, cutpoints and stratum rates are read from the two data files in `calculator/data/`, which are taken from `derived/score_card.json` and `derived/risk_tiers.csv`.

## Reproduction

Three tiers, depending on how deep you want to go.

### Tier 1: run the calculator (no NTDB access needed)

The hosted calculator runs on Python 3.12 (see `calculator/render.yaml`). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r calculator/requirements-dev.txt
pytest
cd calculator
gunicorn app:app
```

Then open <http://127.0.0.1:8000> and check <http://127.0.0.1:8000/healthz>. `pytest` runs the calculator tests, which use fixed synthetic inputs only. On Windows, where gunicorn is unavailable, run `python app.py` inside `calculator/` instead (port 5000).

### Tier 2: inspect the aggregate tables and figures (no NTDB access needed)

Everything reported in the manuscript that can be shared is committed as aggregate output.

- `tables/Table 1.csv`, `tables/Table 2.csv`, `tables/Table 3.csv`: the main-text tables (baseline characteristics, the score card, and performance including the matched-workload comparison).
- `tables/Supplementary Table S1.csv`: the supplementary table.
- `tables/table3_points.csv`, `derived/refit.json`, `derived/score_card.json`: the fitted model and the integer card.
- `tables/table4_discrimination.csv`, `tables/table4b_temporal.csv`: discrimination in the test set and in the temporal validation.
- `tables/table_consort.csv`, `tables/table_split_balance.csv`: cohort flow and the derivation and test split.
- `figures/`: the manuscript figures as PNG, with PDF for some of them.

Counts below 10 patients are suppressed in every shipped file, with further cells withheld where needed so that a suppressed value cannot be recovered by subtraction. The rule, the markers used and the list of files are in [`docs/SUPPRESSION.md`](docs/SUPPRESSION.md). Because of this, a few cells in the shipped tables differ from the unsuppressed originals, and the shipped files are for inspection rather than for re-running every script.

### Tier 3: rebuild from the NTDB Public Use Files

This needs your own copy of the NTDB Public Use Files for admission years 2019 to 2024 (see [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md)). The full guide is [`docs/REPRODUCE.md`](docs/REPRODUCE.md), and the script-by-script dependency table is [`analysis/README_PIPELINE.md`](analysis/README_PIPELINE.md).

```bash
pip install -r requirements.txt
export NTDB_DATA="/path/to/NTDB"

python analysis/01_build_cohort.py
python analysis/02_engineer_features.py
python analysis/03_split.py
python analysis/05_refit_integer.py
python analysis/06_score_test.py
python analysis/07_validation.py
```

What this does and does not allow:

- Steps 01 to 07 need only `NTDB_DATA`. They reproduce the cohort flow (analytic cohort of 239,736 patients, written by step 01 to `derived/cohort_exclusions.json` and `logs/cohort_flow.txt`) and the test-set AUROC of the integer score (0.8823). Only step 01 reads the raw data. The cohort-flow table `tables/table_consort.csv` is rendered from that flow by `analysis/14_consort.py`.
- Step 01 reads four multi-year CSV files that the authors compiled from the raw Public Use Files. The scripts that compiled them are not in this repository. The columns that step 01 reads from them are named in `analysis/01_build_cohort.py`, and their file names and folder layout are in `analysis/config.py`, so equivalent files can be rebuilt from the raw files.
- Step 04 (stability selection by bootstrap, about 30 minutes) can be skipped. `tables/table2_lasso_selected.csv` is shipped, and step 05 reads it, so step 05 can run directly after step 03.
- Two steps can be re-run from the shipped files alone, on a clean clone: `analysis/14_consort.py` (it rebuilds the cohort-flow table and the flow diagram from `derived/cohort_exclusions.json` and `tables/table_split_balance.csv`; the rebuilt PDF differs from the shipped one only in embedded metadata) and `analysis/19_d11_invariance_check.py`. Every other step needs patient-level intermediate files that are never shipped.
- `build_bundle.py`, `render_exhibits.py` and `v4_posthoc_analyses.py` read unsuppressed outputs of the earlier steps, including patient-level intermediate files that are never shipped. They run only after the earlier steps have been rebuilt locally.
- A rebuild writes unsuppressed outputs into `tables/`, `derived/`, `figures/` and `logs/`, overwriting the shipped copies. Work in a separate copy of the repository and do not publish rebuilt outputs without checking them against your data use agreement.

## Repository structure

```
pelvic-fracture-arrival-score/
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt          # pinned packages for analysis/
├── pytest.ini
├── .gitignore
├── analysis/                 # pipeline scripts, config.py, README_PIPELINE.md
├── calculator/               # Flask calculator (deploys to Render)
│   ├── app.py, prediction.py, config.py
│   ├── data/                 # score_card.json, risk_tiers.csv
│   ├── templates/, static/
│   ├── parity_check.py
│   └── requirements.txt, requirements-dev.txt, render.yaml, Procfile
├── derived/                  # aggregate and intermediate outputs (JSON, CSV)
├── tables/                   # result tables (CSV)
├── figures/                  # manuscript figures (PNG, some PDF)
├── inputs/                   # extractability_audit/, small reference inputs
├── logs/                     # QC manifest and the invariance check record
├── tests/                    # calculator tests (synthetic inputs only)
├── scripts/                  # suppression, README number check, release helpers
└── docs/
    ├── REPRODUCE.md
    ├── DATA_AVAILABILITY.md
    └── SUPPRESSION.md
```

## Data availability

Patient-level NTDB data are not redistributed here and must be obtained from the American College of Surgeons at <https://www.facs.org/quality-programs/trauma/quality/national-trauma-data-bank/>. This repository ships only aggregate, de-identified tables and figures, with cells under 10 patients suppressed. Details are in [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Checking the numbers in this README

`python scripts/check_readme_numbers.py` re-reads the shipped tables and confirms every figure quoted in this README and in the docs, and that every relative link and file path resolves.

## Citation

The manuscript is under review. Until it is published, please cite the software record and the manuscript as follows. A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

```bibtex
@unpublished{salman2026pelvic,
  author = {Salman, Samer G.},
  title  = {Early Recognition of Major Hemorrhage in Blunt Pelvic-Ring Fracture: Development and Temporal Validation of an Arrival-Data Bedside Score},
  year   = {2026},
  note   = {Manuscript under review}
}
```

<!-- add_doi:citation:begin -->
<!-- placeholder: scripts/add_doi.py replaces this block after the Zenodo release -->
```bibtex
@software{salman2026pelvic_software,
  author    = {Salman, Samer G.},
  title     = {Pelvic-Fracture Arrival Score: code, aggregate tables and web calculator},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {@@DOI_VERSION@@},
  url       = {https://github.com/samersalman/pelvic-fracture-arrival-score}
}
```

Concept DOI (all versions): [@@DOI_CONCEPT@@](https://doi.org/@@DOI_CONCEPT@@)
<!-- add_doi:citation:end -->

## License

MIT. See [`LICENSE`](LICENSE).
