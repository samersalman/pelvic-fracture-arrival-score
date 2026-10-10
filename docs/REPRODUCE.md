# Reproducing the analysis

This guide covers a rebuild of the cohort, the score and the validation from the NTDB Public Use Files. If you only want to look at the results, you do not need it: the aggregate tables and figures are in `tables/`, `derived/` and `figures/`, and the calculator runs without any NTDB data (see the README).

The script-by-script list of what each step reads and writes is in [`../analysis/README_PIPELINE.md`](../analysis/README_PIPELINE.md). It is not repeated here.

## What can and cannot be re-run

- With NTDB access, the full chain from step 01 can be re-run. Steps 01 to 07 reproduce the cohort flow and the test-set AUROC of the integer score.
- Without NTDB access, only `analysis/14_consort.py` and `analysis/19_d11_invariance_check.py` can be run, because they read only shipped files. Step 14 rebuilds the cohort-flow table `tables/table_consort.csv` and the flow diagram from `derived/cohort_exclusions.json` and `tables/table_split_balance.csv` (the rebuilt PDF differs from the shipped one only in its embedded creation date). Every other step reads patient-level intermediate files that are created by earlier steps and are never shipped.
- `build_bundle.py`, `render_exhibits.py` and `v4_posthoc_analyses.py` read the unsuppressed outputs of the earlier steps. They run only after those steps have been rebuilt locally.
- The shipped tables have cells under 10 patients suppressed (see [`SUPPRESSION.md`](SUPPRESSION.md)). A rebuild produces the unsuppressed versions, so a few cells will differ from the shipped files.

## 1. Prerequisites

- Python 3.14 (the packages in `requirements.txt` were run under 3.14.0).
- The NTDB Public Use Files for admission years 2019 to 2024, obtained from the ACS ([`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md)).
- Enough disk space and memory for the multi-year NTDB files.

## 2. Setup

```bash
git clone https://github.com/samersalman/pelvic-fracture-arrival-score.git
cd pelvic-fracture-arrival-score
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The rebuild writes into `tables/`, `derived/`, `figures/` and `logs/`, overwriting the shipped copies with unsuppressed outputs. Do the rebuild in a copy of the repository that you do not intend to publish.

## 3. Point the scripts at the data

```bash
export NTDB_DATA="/path/to/NTDB"
```

`NTDB_DATA` must contain the folders `compiled_NTDB/compiled_NTDB_2019_2024/` and `PUF/PUF AY <year>/CSV/` for each admission year from 2019 to 2024. The four compiled file names and the per-year `PUF_TRAUMA.csv` files are listed in the docstring of `analysis/config.py`. Only step 01 reads the raw data. If the variable is unset, step 01 stops with a message that says so.

Step 01 reads four multi-year CSV files that the authors compiled from the raw Public Use Files (the main file and the diagnosis, injury-severity and pre-existing-condition tables). The scripts that compiled those four files are not in this repository. The columns that step 01 reads from each of them are named in `analysis/01_build_cohort.py`, and the file names and folder layout are in `analysis/config.py`. A reader who rebuilds equivalent files from the raw Public Use Files can run the pipeline from step 01, but the compile step itself cannot be re-run from this repository.

## 4. Run order

Every script finds the repository root from its own location, so each can be started from any directory.

```bash
python analysis/01_build_cohort.py
python analysis/02_engineer_features.py
python analysis/03_split.py
python analysis/05_refit_integer.py
python analysis/06_score_test.py
python analysis/07_validation.py
```

Step 04 (`04_lasso_stability.py`, stability selection by bootstrap) takes about 30 minutes. It can be skipped, because the selected-variable table `tables/table2_lasso_selected.csv` is shipped and step 05 reads it. To include it, run step 04 after step 03 and before step 05.

The remaining steps, in the order given in `analysis/README_PIPELINE.md`, build the calibration and operating-point tables, the subgroup, threshold, temporal-validation and sensitivity analyses, the cohort-flow table, Table 1, the figures, and the post-hoc analyses. Most scripts accept `--sample N` for a quick smoke run.

A fixed random seed of 42 is used throughout.

## 5. What to check

After steps 01 to 07, these values should match the manuscript:

| Check | Where to look | Expected |
|---|---|---|
| Analytic cohort | last entry of `derived/cohort_exclusions.json`, or the cohort-flow text in `logs/cohort_flow.txt` (both written by step 01). Running `analysis/14_consort.py` afterwards renders the same flow as `tables/table_consort.csv` | 239,736 patients |
| Major hemorrhage events | `tables/table_split_balance.csv`, row `full` | 26,479 |
| Derivation and test split | `tables/table_split_balance.csv` | 159,824 and 79,912 patients |
| Test-set AUROC of the integer score | `tables/table4_discrimination.csv`, outcome `major_hemorrhage_event`, score `pelvic_score` | 0.8823 (95% CI 0.8787 to 0.8860) |

The temporal-validation AUROC (0.8846) comes from `analysis/12_temporal_external_validation.py`, which is part of the longer run.

`python scripts/check_readme_numbers.py` confirms that the numbers quoted in the README and the docs agree with the shipped files. After a rebuild, run it again to compare your rebuilt files with those same quoted numbers.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: The environment variable NTDB_DATA is not set` | The variable is not set in this shell | `export NTDB_DATA=...` (section 3) |
| `NTDB_DATA points to a folder that does not exist` | Wrong path | Check the path and the folder layout in `analysis/config.py` |
| A file not found in step 01 | A compiled file or a yearly `PUF_TRAUMA.csv` is missing or named differently | Compare with the layout in `analysis/config.py` |
| A later step cannot find `derived/candidates.parquet` or a keys file | An earlier step has not been run in this copy | Run steps 01 to 03 first; these files are never shipped |
| `pytest` collects the wrong files | Run from a directory other than the repository root | Run `pytest` from the repository root; `pytest.ini` limits collection to `tests/` |
