Analysis pipeline: run order and file dependencies

This folder holds every script needed to rebuild the analysis from the raw
National Trauma Data Bank (NTDB) files. The scripts were written in two stages.
Stage v3 builds the cohort, fits and validates the score. Stage v4 adds the
results bundle, the main-text exhibits and the reviewer-requested post-hoc
analyses. Both stages read and write the same flat folders (see Layout).

Environment

- Python 3.14 (developed and run under 3.14.0). Install the pinned packages
  with: pip install -r requirements.txt (repository root).
- Only 01_build_cohort.py needs the raw data. Set NTDB_DATA to the folder that
  holds the NTDB files (layout below). All other scripts read the outputs of
  earlier steps.
- Every script finds the repository root from its own location, so it can be
  started from any working directory: python3 analysis/<script>.py
- Output folders are created on demand.
- A fixed seed (42) is used throughout. The long runs (04, 11, 13, and the two
  stability runs in v4_posthoc_analyses.py) use bootstrap resampling.

NTDB_DATA layout

    $NTDB_DATA/compiled_NTDB/compiled_NTDB_2019_2024/
        compiled_PUF_AY_2019_2024_with_VL.csv
        ntdb_2019_2024_icd_diagnoses.csv
        ntdb_2019_2024_ais_diagnoses.csv
        ntdb_2019_2024_preexisting.csv
    $NTDB_DATA/PUF/PUF AY <year>/CSV/PUF_TRAUMA.csv        (year 2019 to 2024)

Layout (flat, relative to the repository root)

    analysis/   scripts and config.py
    inputs/     reference inputs (inputs/extractability_audit/)
    derived/    intermediate and aggregate outputs of both stages
    tables/     table outputs of both stages
    figures/    figure outputs of both stages
    logs/       manifests, QC outputs and run logs

File names do not collide between the v3 and v4 stages. Provenance strings
inside some outputs and docstrings say "v3/derived/..." or "v3/tables/...";
read these as derived/... and tables/... in this repository.

Patient-level files written by the pipeline (never committed; see .gitignore):
derived/pelvic_cohort_predecision.csv, derived/candidates.parquet,
derived/cohort_test_scored.parquet, derived/*_inc_keys.csv, and the pickle files
derived/lasso_cv_fit.pkl, derived/roc_data.pkl, derived/temporal_roc_data.pkl,
derived/mtp_roc_data.pkl.

Run order, stage v3

    01_build_cohort.py                  (needs NTDB_DATA; about 3 minutes)
    02_engineer_features.py
    03_split.py
    04_lasso_stability.py               (about 30 minutes, 100 bootstraps)
    05_refit_integer.py
    06_score_test.py
    07_validation.py
    08_calibration_dca_operating.py
    11_pelvic_thresholds.py             (must precede 10 and 13)
    09_subgroups.py
    10_elderly_anticoag_reclassification.py
    12_temporal_external_validation.py
    13_sensitivity_components.py
    14_consort.py
    15_table1.py
    17_mtp_equivalent.py
    19_d11_invariance_check.py          (needs logs/pre_d11_snapshot/, see below)
    render_figures.py                   (figures 2 to 10; figure 1 comes from 14)
    00_audit_extractability.py          (optional; regenerates inputs/extractability_audit/)

Run order, stage v4 (after all of stage v3)

    build_bundle.py                     (then: build_bundle.py --verify)
    render_exhibits.py
    v4_posthoc_analyses.py stability --tag a1          (about 30 minutes)
    v4_posthoc_analyses.py stability --tag parity17    (about 30 minutes)
    v4_posthoc_analyses.py finish

Smoke runs: most scripts accept --sample N (and 01 accepts --sample N for a
capped scan). 04, 07, 11, 12, 13 and 17 also accept bootstrap-count options.

Reproduction without the long bootstrap (04): tables/table2_lasso_selected.csv
is shipped, so 04 can be skipped and 05 run directly after 03.

Dependency table (script -> reads -> writes)

All names are relative to the folder shown in front of them.

00_audit_extractability.py
  reads : derived/pelvic_cohort_predecision.csv
  writes: inputs/extractability_audit/candidate_audit_table.csv,
          inputs/extractability_audit/exclusion_rationale.md

01_build_cohort.py
  reads : NTDB_DATA files (four compiled CSVs and six PUF_TRAUMA.csv files)
  writes: derived/pelvic_cohort_predecision.csv (patient-level),
          derived/cohort_exclusions.json, derived/informative_missingness.csv,
          logs/qc_manifest.json, logs/missingness.csv, logs/unknown_hc_codes.csv,
          logs/cohort_flow.txt, logs/fix1_transfusion_audit.md,
          logs/fix2_death_audit.md

02_engineer_features.py
  reads : derived/pelvic_cohort_predecision.csv, logs/qc_manifest.json,
          analysis/01_build_cohort.py (parsed as text, schema cross-check only)
  writes: derived/candidates.parquet (patient-level),
          tables/table2_candidate_coverage.csv

03_split.py
  reads : derived/candidates.parquet
  writes: derived/train_inc_keys.csv, derived/test_inc_keys.csv,
          derived/temporal_derive_inc_keys.csv, derived/temporal_valid_inc_keys.csv
          (all patient-level keys), tables/table_split_balance.csv

04_lasso_stability.py
  reads : derived/candidates.parquet, derived/train_inc_keys.csv
  writes: derived/lasso_cv_fit.pkl, tables/table2_lasso_stability.csv,
          tables/table2_lasso_selected.csv

05_refit_integer.py
  reads : derived/candidates.parquet, derived/train_inc_keys.csv,
          tables/table2_lasso_selected.csv,
          logs/pre_d11_snapshot/ (optional; used as baseline when present)
  writes: tables/table3_points.csv, derived/refit.json

06_score_test.py
  reads : derived/candidates.parquet, derived/test_inc_keys.csv,
          tables/table3_points.csv, derived/refit.json
  writes: derived/cohort_test_scored.parquet (patient-level)

07_validation.py
  reads : derived/cohort_test_scored.parquet, derived/refit.json
  writes: tables/table4_discrimination.csv, tables/table4_delong.csv,
          tables/table5_calibration.csv, derived/roc_data.pkl

08_calibration_dca_operating.py
  reads : derived/cohort_test_scored.parquet, tables/table3_points.csv,
          derived/refit.json, logs/qc_manifest.json
  writes: tables/table5_operating.csv, tables/table5c_operating_full.csv,
          tables/table5d_matched_workload.csv, derived/dca_data.csv,
          derived/calibration_data.csv, derived/score_card.json,
          derived/risk_tiers.csv

11_pelvic_thresholds.py
  reads : derived/pelvic_cohort_predecision.csv
  writes: derived/spline_*.csv, derived/changepoint_*.json,
          derived/changepoint_stratified.json, derived/tipping_point_bound.json,
          derived/component_splines.csv, tables/table7_threshold_or.csv,
          tables/table7_absrisk.csv, logs/thresholds_diagnostics.txt,
          logs/thresholds_manifest.json

09_subgroups.py
  reads : derived/cohort_test_scored.parquet, derived/candidates.parquet,
          logs/qc_manifest.json
  writes: tables/table_subgroups.csv, derived/subgroup_forest_data.csv

10_elderly_anticoag_reclassification.py
  reads : derived/candidates.parquet, derived/cohort_test_scored.parquet,
          derived/score_card.json, logs/qc_manifest.json,
          derived/changepoint_stratified.json (optional)
  writes: tables/table6_elderly_reclass.csv, tables/table6b_elderly_ledger.csv,
          derived/reclass.json, derived/elderly_dca_data.csv

12_temporal_external_validation.py
  reads : derived/candidates.parquet, derived/refit.json,
          derived/temporal_derive_inc_keys.csv, derived/temporal_valid_inc_keys.csv,
          tables/table4_discrimination.csv, logs/qc_manifest.json
  writes: tables/table4b_temporal.csv, derived/temporal_roc_data.pkl,
          derived/citl_slope.json

13_sensitivity_components.py
  reads : derived/pelvic_cohort_predecision.csv, derived/candidates.parquet,
          derived/train_inc_keys.csv, derived/test_inc_keys.csv, derived/refit.json,
          derived/changepoint_*_composite.json, derived/changepoint_*_mortality.json,
          tables/table7_threshold_or.csv, tables/table4_discrimination.csv
  writes: derived/component_decomp_data.csv, derived/component_venn.json,
          tables/table7_sensitivity.csv, logs/sensitivity_components_diagnostics.txt,
          logs/sensitivity_components_manifest.json

14_consort.py
  reads : derived/cohort_exclusions.json, tables/table_split_balance.csv,
          logs/qc_manifest.json
  writes: tables/table_consort.csv, figures/figure1_STROBE.png,
          figures/figure1_STROBE.pdf

15_table1.py
  reads : derived/pelvic_cohort_predecision.csv
  writes: tables/table1_baseline.csv

17_mtp_equivalent.py
  reads : derived/cohort_test_scored.parquet, derived/candidates.parquet,
          derived/test_inc_keys.csv, derived/refit.json, tables/table3_points.csv
  writes: tables/table8_mtp.csv, derived/mtp_dca_data.csv,
          derived/mtp_roc_data.pkl, logs/mtp_equivalent_manifest.json

19_d11_invariance_check.py
  reads : logs/pre_d11_snapshot/{refit.json, table4_discrimination.csv,
          table3_points.csv}, derived/refit.json,
          tables/table4_discrimination.csv, tables/table3_points.csv
  writes: logs/d11_invariance_check.json, logs/d11_invariance_check.csv

render_figures.py
  reads : derived/{refit.json, cohort_test_scored.parquet, risk_tiers.csv,
          roc_data.pkl, temporal_roc_data.pkl, calibration_data.csv, reclass.json,
          dca_data.csv, elderly_dca_data.csv, subgroup_forest_data.csv,
          spline_si_composite.csv, spline_sbp_composite.csv,
          changepoint_sbp_composite.json, component_decomp_data.csv,
          score_card.json, mtp_roc_data.pkl, mtp_dca_data.csv},
          tables/table3_points.csv, logs/qc_manifest.json
  writes: figures/figure2_score_distribution through figures/figure10_mtp_equivalent
          (.png and .pdf)

build_bundle.py
  reads : 21 tables/, 13 derived/ and 3 logs/ artifacts named in its dependency
          manifest, plus logs/unknown_hc_codes.csv and derived/dca_data.csv
  writes: derived/results_bundle.json, logs/NUMBERS-LEDGER.md (internal working
          file, ignored by git)

render_exhibits.py
  reads : derived/{cohort_exclusions.json, roc_data.pkl, temporal_roc_data.pkl,
          calibration_data.csv, reclass.json, dca_data.csv, elderly_dca_data.csv,
          score_card.json, refit.json, risk_tiers.csv, citl_slope.json},
          tables/{table_split_balance.csv, table1_baseline.csv, table3_points.csv,
          table4_discrimination.csv, table4b_temporal.csv, table4_delong.csv,
          table5d_matched_workload.csv, table_consort.csv}, logs/qc_manifest.json
  writes: tables/Table 1.csv, Table 2.csv, Table 3.csv (and .md versions),
          figures/figure_1, figure_2, figure_3 (.png and .tif),
          figures/figure_legends.md

v4_posthoc_analyses.py
  reads : derived/{candidates.parquet, cohort_test_scored.parquet,
          train_inc_keys.csv, test_inc_keys.csv, temporal_derive_inc_keys.csv,
          temporal_valid_inc_keys.csv, refit.json, score_card.json,
          risk_tiers.csv, results_bundle.json}, tables/{table2_lasso_stability.csv,
          table2_lasso_selected.csv, table3_points.csv, table4_discrimination.csv,
          table4b_temporal.csv, table5d_matched_workload.csv, Table 1.csv}
  writes: derived/a1_stability_a1.csv, derived/a1_stability_parity17.csv (and
          *_meta.json), derived/a1_correlation_vif.csv,
          derived/a1_sensitivity_card.csv, derived/a1_score_distribution_test.csv,
          derived/a1_sensitivity_metrics.json, derived/a2_hypotension_mechanism.csv,
          derived/a2_worked_example.json, derived/a3_invariance_check.json,
          derived/v4_analyses_manifest.json, logs/v4_posthoc_analyses.log

Notes on what a clean clone can and cannot do

- Steps 01 to 07 need only NTDB_DATA (and, for 05, tables/table2_lasso_selected.csv
  if 04 is skipped). They reproduce the cohort flow (analytic N = 239,736) and the
  test-set AUROC of the integer score (0.8823).
- 19_d11_invariance_check.py needs the frozen baseline in logs/pre_d11_snapshot/.
  It is a record of an earlier fit that cannot be regenerated from the current
  code, so it ships with the repository: refit.json, table3_points.csv and
  table4_discrimination.csv (the three files that scripts 05 and 19 read) and a
  MANIFEST.sha256 that lists exactly these three files. The snapshot originally
  held two more files (a score card and an operating table); no script reads
  them and they are not shipped. Script 19 compares the snapshot with
  derived/refit.json, tables/table3_points.csv and
  tables/table4_discrimination.csv, all of which ship, so 19 can be run on a
  clean clone and reproduces logs/d11_invariance_check.json.
- logs/qc_manifest.json ships and is read by 02, 08, 09, 10, 12, 14,
  render_figures.py, build_bundle.py and render_exhibits.py. It is the screened
  copy: two items that are not safe to share (a single encounter key and
  band counts of the acetabulum-inclusive file) were removed, and no script
  reads them. 01_build_cohort.py rewrites this file with those items; do not
  commit a regenerated copy without screening it. logs/d11_invariance_check.json
  also ships (build_bundle.py requires it). Other logs written by the scripts
  (run logs, audit notes, per-step manifests, cohort_flow.txt,
  unknown_hc_codes.csv) are not shipped; apart from build_bundle.py, no script
  reads them. build_bundle.py also names logs/missingness.csv (shipped here as
  tables/missingness.csv) and logs/unknown_hc_codes.csv (not shipped), and it
  cannot be run from this repository in any case (see the next item).
- build_bundle.py, render_exhibits.py and v4_posthoc_analyses.py read the full,
  unsuppressed outputs of stage v3 (including the patient-level parquet and
  pickle files). They can only be re-run after the complete stage v3 has been
  rebuilt locally. The aggregate tables shipped in this repository have small
  cells suppressed and are for inspection, not for re-running these three scripts.
- Internal references inside comments and docstrings to CONTRACTS sections,
  deviation labels (D-n), "Fix n" and batch numbers point to project planning
  documents that are not distributed with this repository.
