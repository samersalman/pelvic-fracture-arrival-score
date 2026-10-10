# Small-cell suppression

Every aggregate file in `tables/`, `derived/` and `figures/` was produced by `scripts/suppress_small_cells.py` from the author's local project folder. The script is the exact record of what was done. It is deterministic and has no randomness.

## Rule

1. Counts of 1 to 9 patients are replaced by the marker `<10`. Zero is not suppressed.
2. Complementary disclosure. A withheld cell must not be recoverable by subtracting the other cells from a shown total or from a sibling row or column. Where it would be, one more cell of the same additive relation, the smallest remaining one, is replaced by the marker `withheld`. Where cells are nested (cumulative operating-point rows, ordered scores), rows are dropped or adjacent categories are merged so that no difference between two shown rows is a count of 1 to 9. The same holds for definitions and denominators that contain one another: an outcome variant that adds a few patients to another, a denominator that needs one more measurement (the shock index needs a pulse as well as a pressure, the reverse shock index also needs the Glasgow Coma Scale), a stratum of a curve restricted to patients with a recorded vital sign. When two such counts, or two totals that a reader can derive by adding and subtracting shown cells, differ by 1 to 9, the cell of the larger set is withheld, with its complementary cells and the statistics computed from them.
3. A percentage, rate, confidence interval, standardized mean difference or P value computed from a suppressed count is replaced by `withheld`.
4. Numbers that are not counts of patients are never suppressed.

## Machine-readable markers

| Marker | Meaning |
|---|---|
| `<10` | The true count is 1 to 9. |
| `withheld` | The true value is not necessarily small. It is hidden so that a `<10` cell cannot be recovered by subtraction, or it is a statistic computed from a hidden count. |

Both markers appear as the whole cell value, in CSV cells and as JSON string values. A shown integer from 1 to 9 outside the columns listed next is therefore a defect.

## Values from 1 to 9 that are not counts and are left as they are

| Where | Column or key | What the value is |
|---|---|---|
| calibration_data.csv, citl_slope.json, reclass.json | bin, decile | bin or decile index |
| subgroup_forest_data.csv, table_subgroups.csv | Level | ACS verification level or similar category code |
| table3_points.csv, table5 files, score_card.json, refit.json, a1_sensitivity_card.csv, a1_sensitivity_metrics.json, a2_worked_example.json | Points, points, a1_points, published_points, score_min, score_max, Cutoff, rule_in_cutpoint, rule_out_cutpoint | integer score points, scores and cutoffs |
| table2_lasso_stability.csv | CV_Abs_Beta_Rank | rank |
| table5_calibration.csv, citl_slope.json | HL_df, df, hl_df | degrees of freedom |
| table_consort.csv | Step | step number |
| table5d_matched_workload.csv | Is_workload_anchor, Workload_match_within_tolerance, Delta_comparable | 0/1 flags |
| table5d_matched_workload.csv | Delta_workload_gap_n | difference between the numbers flagged by two different instruments, each of several thousand patients; not a cell of patients |
| table5d_matched_workload.csv, table5c_operating_full.csv | Workload_nominal_target_pct, Workload_anchor_cutoff, Cutoff | percent targets and cutoffs |
| Table 1.csv, table1_baseline.csv, results_bundle.json | median (interquartile range) cells such as `9 (5-17)`, `6 (6-6)` | a summary of a measured value, not a count |
| table5c_operating_full.csv | Within_RuleOut_RuleIn_Window | 0/1 flag |
| *.json | boot_n_unique, boot_bp_n_unique, nonzero, boot, n_point_variables, n_surgical_categories | counts of bootstrap runs, model terms or code categories, not of patients |
| any file | Selection_Frequency, Calibration_Slope, Max, value and similar columns equal to 1.0 | proportions and coefficients |

## Files shipped

SHA-256 is of the shipped version. Mode: `copy` is byte-identical to the project file, `scrubbed` differs only by removal of an absolute path, a co-author attribution or an internal label, `suppressed` was altered by the rule above (including removal of items that fail the screen), `subset` is the original manifest reduced to the files that are shipped.

| Shipped file | Source (project folder) | Mode | SHA-256 |
|---|---|---|---|
| `derived/a1_correlation_vif.csv` | `v4/derived/a1_correlation_vif.csv` | copy | `cbc2fbe526f8a3c5d8bd8ebccb87691a9e9203027d51b2f982c7acc4f2fa8b8d` |
| `derived/a1_score_distribution_test.csv` | `v4/derived/a1_score_distribution_test.csv` | suppressed | `6c35755513a5a13f29ca9f91b8df2d99d28d16deecd8be719f41ab83916eb714` |
| `derived/a1_sensitivity_card.csv` | `v4/derived/a1_sensitivity_card.csv` | scrubbed | `e719b426a066e7efbfb7701f753f2abc86a9a8371814a7abee3ea3d4b25a74a3` |
| `derived/a1_sensitivity_metrics.json` | `v4/derived/a1_sensitivity_metrics.json` | scrubbed | `8bd514ac99cb92979dbc5a149fbf7e4349957af8144f3e599e99934269779eb5` |
| `derived/a1_stability_a1.csv` | `v4/derived/a1_stability_a1.csv` | copy | `c0482d8a782f5f26754660e4ebbe5e12f59a597767505ca2377a04229943614d` |
| `derived/a1_stability_a1_meta.json` | `v4/derived/a1_stability_a1_meta.json` | scrubbed | `0b6c1110e7fbb574498241e7b3fb6fc7fd78c777997a51a4fccc6b2864b728b0` |
| `derived/a1_stability_parity17.csv` | `v4/derived/a1_stability_parity17.csv` | copy | `9b5b50c3b97da48806d347cb6ffa639b718f91180da5c7b40ef6b228c2b1c830` |
| `derived/a1_stability_parity17_meta.json` | `v4/derived/a1_stability_parity17_meta.json` | scrubbed | `69dda54acbb13571790c765e22ee1c89180741ec394a3f6331f4de25afd7a0d5` |
| `derived/a2_hypotension_mechanism.csv` | `v4/derived/a2_hypotension_mechanism.csv` | suppressed | `bbe72fc9ba61ed0b69956923962b66940981960847213f29087d621c5f1b260b` |
| `derived/a2_worked_example.json` | `v4/derived/a2_worked_example.json` | suppressed | `0c542fcdc2879ad4c1caa312a7d3873405106d2cfda4ed72f03aee0edd9b2628` |
| `derived/a3_invariance_check.json` | `v4/derived/a3_invariance_check.json` | scrubbed | `d322a367abc979851b5952dd132708f25cb21a814d3f1a1a7c5863193b50fcb3` |
| `derived/calibration_data.csv` | `v3/derived/calibration_data.csv` | copy | `a8e96570f8ed03fe232879185e008f452e75df2115bf3455e47192f93516981f` |
| `derived/changepoint_sbp_composite.json` | `v3/derived/changepoint_sbp_composite.json` | scrubbed | `407517942042b646cb88bd7f21eb7689159d00b4db836a6ba33336df221a8097` |
| `derived/changepoint_sbp_mortality.json` | `v3/derived/changepoint_sbp_mortality.json` | scrubbed | `154a9fb9a9c10e76f0167d64f7f076f074b039d54fd69dd720336137c3f1ece3` |
| `derived/changepoint_si_composite.json` | `v3/derived/changepoint_si_composite.json` | scrubbed | `05fadbad2882cbb3d3c89adb122273c130208355c7d42aaf186dfb3f207ff507` |
| `derived/changepoint_si_mortality.json` | `v3/derived/changepoint_si_mortality.json` | scrubbed | `a01db1b1c970da3ee215fbbfdfe987bbe89782ba68e328c093e43b374897139b` |
| `derived/changepoint_stratified.json` | `v3/derived/changepoint_stratified.json` | suppressed | `f96fefb6f7a98985065c20f9526c02f37ae7e87100d0d417b588b856dabe0c86` |
| `derived/citl_slope.json` | `v3/derived/citl_slope.json` | scrubbed | `e44de4bf973eb946ee55a4f11e21201006f94845ed46469d4d27074b2cdc4b6e` |
| `derived/cohort_exclusions.json` | `v3/derived/cohort_exclusions.json` | copy | `73172d7779bdb6a8e664c4f1b67315b8491ca86d6f0da21b6958ecf05e7f3ab5` |
| `derived/component_decomp_data.csv` | `v3/derived/component_decomp_data.csv` | copy | `bdde6e273942587cb09943d9dc0e06f2a573b1d3503c9ea3453f46945e78b5a8` |
| `derived/component_splines.csv` | `v3/derived/component_splines.csv` | copy | `bdde6e273942587cb09943d9dc0e06f2a573b1d3503c9ea3453f46945e78b5a8` |
| `derived/component_venn.json` | `v3/derived/component_venn.json` | copy | `e7a7c13b120e1cfa60ca023d48aab881d988e259598268700279ef3dfbbc8d73` |
| `derived/dca_data.csv` | `v3/derived/dca_data.csv` | copy | `8a22b56d8587bb36d04d8ee1aef6cb63c9b832d72cd03ff48b866e91c6ad1369` |
| `derived/elderly_dca_data.csv` | `v3/derived/elderly_dca_data.csv` | copy | `f94611b79b2df074147080e48d6eb48c81b24262cf0d424c0fa55e050fe97cf1` |
| `derived/informative_missingness.csv` | `v3/derived/informative_missingness.csv` | copy | `539da12fb9cd9b0d1a3b4628e8d0594e83e0edc724616d6e77f1475216b570b6` |
| `derived/mtp_dca_data.csv` | `v3/derived/mtp_dca_data.csv` | copy | `cf979ec2f93b027bc492899d3073eacdfa219b14d1e84de7e07f8568c2e34969` |
| `derived/reclass.json` | `v3/derived/reclass.json` | suppressed | `9b24e85eb38170f41b13931a70888d162ae94cf1a5635047ec537dee213ba749` |
| `derived/refit.json` | `v3/derived/refit.json` | scrubbed | `9e5da31fcfc2f9cbb34cd3c2214662c1a6a6f493912c54470f45eef7c7c9e49c` |
| `derived/results_bundle.json` | `v4/bundle/results_bundle.json` | scrubbed | `1fe5889681e0926d05907d4cbda4162af73572bc09aec70991cf1e96ca0557ea` |
| `derived/risk_tiers.csv` | `v3/derived/risk_tiers.csv` | suppressed | `6877eb334b4545e2465cdc129bcfbda26dc6753c41dd4a8fcea13f1b68cbdecb` |
| `derived/score_card.json` | `v3/derived/score_card.json` | suppressed | `e97ccbd0ce627676982f10e3e47b7b259ec27512a87bca26d3d7d0d1343aa552` |
| `derived/spline_sbp_composite.csv` | `v3/derived/spline_sbp_composite.csv` | copy | `5e89c5a8ae6f62523ce536f38dc20d39fbb846270a0dd33a573493d6fe6bf68e` |
| `derived/spline_sbp_mortality.csv` | `v3/derived/spline_sbp_mortality.csv` | copy | `5729177ac5870e7d9a0c497eb9a11d0d10e65410a1eca46465a8b0cc8e21d984` |
| `derived/spline_si_composite.csv` | `v3/derived/spline_si_composite.csv` | copy | `5db61eca68d7c8c1088bf952ca2d7d17ea9c5907f7744dd63e994d24eb68ead1` |
| `derived/spline_si_mortality.csv` | `v3/derived/spline_si_mortality.csv` | copy | `f49a8a296808a8e984e237541b82ee2509d9302d4c141e008b7a3dad347b00f2` |
| `derived/subgroup_forest_data.csv` | `v3/derived/subgroup_forest_data.csv` | suppressed | `9fbc8c37d8eedf570084ecc2969a1c1593638fe4bb2ddb78da9997c391df948c` |
| `derived/tipping_point_bound.json` | `v3/derived/tipping_point_bound.json` | copy | `1b4eef4cc4873e12b7981916e9c82f72755dbd380b91d27405227128398cf447` |
| `figures/figure10_mtp_equivalent.png` | `v3/figures/figure10_mtp_equivalent.png` | copy | `858fe07a78e22dc088802ae660edf521ef9146021f7e270c6971e2cdb510b273` |
| `figures/figure1_STROBE.pdf` | `v3/figures/figure1_STROBE.pdf` | copy | `f7603442021767d8421b3d67be320e28e88d68f12356a1d8cdc8d151e43bfea7` |
| `figures/figure1_STROBE.png` | `v3/figures/figure1_STROBE.png` | copy | `ece4f569355d82e0dfb5b42dc7398cdbd5bf43171bda0f5cc846fb41cc67c310` |
| `figures/figure2_score_distribution.png` | `v3/figures/figure2_score_distribution.png` | copy | `0729cdd758f95d09442a415ff36cc3a3ac9e5500e16c432492832ac64343ecb6` |
| `figures/figure3_roc.png` | `v3/figures/figure3_roc.png` | copy | `dd51660c7950d8bd7be08c92eb53c7227f0ebfd86970c88195ed6eff252ccdfb` |
| `figures/figure4_calibration.pdf` | `v3/figures/figure4_calibration.pdf` | copy | `6c45b75c104400e7c1bfc4b67b02593f7d5ff133194b077368a74c4dfda77a9a` |
| `figures/figure4_calibration.png` | `v3/figures/figure4_calibration.png` | copy | `f2536e25d51a5f19e6f7c90a5a4b76dd9ce929ba53c9cade0173dbd45baf6e42` |
| `figures/figure5_dca.pdf` | `v3/figures/figure5_dca.pdf` | copy | `f82b063b4a65463b24ad4450d15a21d12bfc6042bc4dc4baeca7ce7c8be9257e` |
| `figures/figure5_dca.png` | `v3/figures/figure5_dca.png` | copy | `9ad574c18631a17573fb9883fd3ad387462bb81dcb3fb8818dfdc1c1e67b5a66` |
| `figures/figure6_subgroup_forest.pdf` | `v3/figures/figure6_subgroup_forest.pdf` | copy | `7424db42e269ecb31510f72683d160d8f639d499db00dc746abe06bdbf5fbf1b` |
| `figures/figure6_subgroup_forest.png` | `v3/figures/figure6_subgroup_forest.png` | copy | `623d3ca940a32d48b756f7392f0465d6a6d40e8cb766bd9ff8be9e8c558bf525` |
| `figures/figure7_threshold_splines.pdf` | `v3/figures/figure7_threshold_splines.pdf` | copy | `e386ad5bc6ff1ca808a57cac9b6b787b0c0acf600d1f6450742596a040e76d1c` |
| `figures/figure7_threshold_splines.png` | `v3/figures/figure7_threshold_splines.png` | copy | `733c4c1f4ae299c673327183c9b76c44e98ed1b23d01140c55fe82a2576f55e0` |
| `figures/figure8_component.pdf` | `v3/figures/figure8_component.pdf` | copy | `4ea89e4c1591fb20807f2ed4ddda4f62c42c4050f203a64484209982ae80cdc0` |
| `figures/figure8_component.png` | `v3/figures/figure8_component.png` | copy | `b7a54693e0a605299fb4fb0c3f2d0decaab49bc5d8358baa2ca0c6e3f2968313` |
| `figures/figure_1.png` | `v4/figures/figure_1.png` | copy | `4def27b9048a97a9fdd7afa6a2b2ee50278ccaf4111006f7c90b6a56dbfc3719` |
| `figures/figure_2.png` | `v4/figures/figure_2.png` | copy | `efb215e2d84ec4ebd018b4f4cb2ca6ce35b41f8c3406007b593b8ab09cd0dabf` |
| `figures/figure_3.png` | `v4/figures/figure_3.png` | copy | `001c9cb2518836e21f6223176a7a36c931e11ce0abd30c58f0cf41ce1367377d` |
| `logs/d11_invariance_check.csv` | `v3/logs/d11_invariance_check.csv` | copy | `02ce11b3d50bf662039431f7268fd28547171096393a6d4f8542c1c7b706d19a` |
| `logs/d11_invariance_check.json` | `v3/logs/d11_invariance_check.json` | copy | `933f0fbcd922334698a8088764bc532e3aff516ca627f04ed76c34aa9d1eac35` |
| `logs/pre_d11_snapshot/MANIFEST.sha256` | `v3/logs/pre_d11_snapshot/MANIFEST.sha256` | subset | `e47266fe0e075464d3f29f02219e428e75371883233a01d90c88dcd8cd02a5ca` |
| `logs/pre_d11_snapshot/refit.json` | `v3/logs/pre_d11_snapshot/refit.json` | copy | `514802cf61bfbf7803cc7ee07714cf97b3145ecab571b226c71bacf83d4ff833` |
| `logs/pre_d11_snapshot/table3_points.csv` | `v3/logs/pre_d11_snapshot/table3_points.csv` | copy | `1b1327865f2f472793421254e05d757cf8037ced268e19b6fbec9f37022d67ae` |
| `logs/pre_d11_snapshot/table4_discrimination.csv` | `v3/logs/pre_d11_snapshot/table4_discrimination.csv` | copy | `1dddd16395003012e842c07a438882362253299541ab353537bd5a71678bb0d8` |
| `logs/qc_manifest.json` | `v3/logs/qc_manifest.json` | suppressed | `220431f5b487b4b43753906666dd0d833bc3134344097fdcb9d41dfd2317d378` |
| `tables/Supplementary Table S1.csv` | `v4/tables/Supplementary Table S1.csv` | copy | `0746b114f13ee0ebb137a9865d4b3526aa5527d34064905d66fe818e2dafe42e` |
| `tables/Table 1.csv` | `v4/tables/Table 1.csv` | copy | `a85d4b3633c54b8b58990ed47149fc71d4922110586b9fbcd2ec621a7fc746d1` |
| `tables/Table 2.csv` | `v4/tables/Table 2.csv` | copy | `9aef5cca404b603c360b5b6eab228aa7ad8176af01b29b6a35ebb9904dc80301` |
| `tables/Table 3.csv` | `v4/tables/Table 3.csv` | copy | `60ed285ff10c6b2ea956c0a63d69f688ba142f24325758684c288377d2f8e29c` |
| `tables/missingness.csv` | `v3/logs/missingness.csv` | copy | `a57e5e0d1b4830d715fd903158490aa47fe8a65510e4c56389f7b7d51848e72d` |
| `tables/table1_baseline.csv` | `v3/tables/table1_baseline.csv` | suppressed | `c16fe7edf5a53741bff5ee3b4c3789661226e443148844962bc9349b6b1ac1d1` |
| `tables/table2_candidate_coverage.csv` | `v3/tables/table2_candidate_coverage.csv` | copy | `75d8c8ad4459767e2b88ed6a99ae7bc0cdd0782c06255660808264ad9ed65534` |
| `tables/table2_lasso_selected.csv` | `v3/tables/table2_lasso_selected.csv` | copy | `391b648007a149066a00604db478ecd9ae57db505c8a66473c663a7fba4ce6f0` |
| `tables/table2_lasso_stability.csv` | `v3/tables/table2_lasso_stability.csv` | copy | `3adfbc0ab5decc086e9c2bd634c80e4d132ac163ff5b2129be64174cb81f1379` |
| `tables/table3_points.csv` | `v3/tables/table3_points.csv` | copy | `fd125dd52253f2bcf250ae16bf3edef97a5bfc84c21420f674aaab408f63c4a9` |
| `tables/table4_delong.csv` | `v3/tables/table4_delong.csv` | copy | `906d472011056c1c63a4f825dad234cd946753664f41aa1d785f66ded7b0d6da` |
| `tables/table4_discrimination.csv` | `v3/tables/table4_discrimination.csv` | copy | `ccdb6a3848a43b07818ea8ce2d19ae1c17f7f35a8af16ad14434b8bce7f14275` |
| `tables/table4b_temporal.csv` | `v3/tables/table4b_temporal.csv` | copy | `e7faf5ea4b9fe70037a283cc5368d72c81f26241a00a5efd5e79dc3145df88a9` |
| `tables/table5_calibration.csv` | `v3/tables/table5_calibration.csv` | copy | `0154096597037680462e0b86ce39e323c4ce8ab6666270bbaaebf59b106eb38e` |
| `tables/table5_operating.csv` | `v3/tables/table5_operating.csv` | copy | `6c2fa8f9fdb48e8f98738e27f9e3dc36790bebd81792b91aa67aed89db49ff20` |
| `tables/table5c_operating_full.csv` | `v3/tables/table5c_operating_full.csv` | suppressed | `620bd52ca9ab7ef24856b6992b5afa452cb7f91b73406fdcf18a6f65269823a8` |
| `tables/table5d_matched_workload.csv` | `v3/tables/table5d_matched_workload.csv` | copy | `97666abdb4b06f7dc06a4a78643aa2cbec186d8a21875acb1d20f9662fef631a` |
| `tables/table6_elderly_reclass.csv` | `v3/tables/table6_elderly_reclass.csv` | copy | `2ba8340eb38745511433bb33350e078bc935b76f4db2fc009f6deb8c042f57c0` |
| `tables/table6b_elderly_ledger.csv` | `v3/tables/table6b_elderly_ledger.csv` | copy | `3cf9d9c283288731b46f117102a75ff9eea5d0688f45cbdb4e113119c19a80c8` |
| `tables/table7_absrisk.csv` | `v3/tables/table7_absrisk.csv` | copy | `6ed90853f7e21e62b88da2472a329b1c5a022c6e7cab4597ddbe7ee526192a57` |
| `tables/table7_sensitivity.csv` | `v3/tables/table7_sensitivity.csv` | suppressed | `3c3c4a601a3d18ce3a39e2dd720aebac0a1d47a6cb35439bff38976cbfbc9c9b` |
| `tables/table7_threshold_or.csv` | `v3/tables/table7_threshold_or.csv` | copy | `6c6dd16b386a6a16f5a54d23f226eb5a85febce75b9e1c70c36d36244a767341` |
| `tables/table8_mtp.csv` | `v3/tables/table8_mtp.csv` | scrubbed | `5b287d98d59f31a4534b15ccff01f4613798069f1a55f21ca0d1e45c000fae1c` |
| `tables/table_consort.csv` | `v3/tables/table_consort.csv` | copy | `45797f6191e90afddf848f2234de8edd640b882b5016d2154e4a77d0f91bea2f` |
| `tables/table_split_balance.csv` | `v3/tables/table_split_balance.csv` | copy | `f82318d5a457149795601e6dcdace7d28544174d6b4237097375222d49ffb311` |
| `tables/table_subgroups.csv` | `v3/tables/table_subgroups.csv` | suppressed | `243319936a418cf48f5e62c42cb0265c7b0b7547286c080100c521e2e817be93` |

Calculator data files are produced by the same script from the shipped files above, never edited by hand:

| Calculator file | Made from | Relation | SHA-256 |
|---|---|---|---|
| `calculator/data/score_card.json` | `derived/score_card.json` | byte copy | `e97ccbd0ce627676982f10e3e47b7b259ec27512a87bca26d3d7d0d1343aa552` |
| `calculator/data/risk_tiers.csv` | `derived/risk_tiers.csv` | same file without the column n_override_in_tier | `437375f196bf2aab751a252c80adf140e8a0c46324bdc8dde1301a9a5e4b2aaf` |

## Files altered

- `derived/a1_score_distribution_test.csv`, collapsed: adjacent scores merged so that no n or event count is 1 to 9; merged ranges (score_min to score_max): -12.0 to -11.0, -5.0 to -4.0, 28.0 to 30.0. Columns changed from score to score_min and score_max.
- `derived/a1_sensitivity_card.csv`, text scrub: personal attribution replaced by 'reviewer-requested'; internal version label replaced by 'the primary score'
- `derived/a1_sensitivity_metrics.json`, text scrub: personal attribution replaced by 'reviewer-requested'; internal version label replaced by 'the primary score'
- `derived/a1_stability_a1_meta.json`, text scrub: personal attribution replaced by 'reviewer-requested'; internal version label replaced by 'the primary score'
- `derived/a1_stability_parity17_meta.json`, text scrub: personal attribution replaced by 'reviewer-requested'; internal version label replaced by 'the primary score'
- `derived/a2_hypotension_mechanism.csv`, suppressed: 3 primary event cells (<10); 9 derived rate/CI cells; 1 complementary cells (the subset row that excludes no-signs-of-life patients by score: its n differs from its parent row by fewer than 10 patients)
- `derived/a2_hypotension_mechanism.csv`, text scrub: two row notes that cited an earlier plan and the results bundle internals replaced by neutral wording
- `derived/a2_hypotension_mechanism.csv`, text scrub: personal attribution replaced by 'reviewer-requested'; group label made neutral; internal version label replaced by 'the primary score'
- `derived/a2_worked_example.json`, block removed: plan_reconciliation (internal check against an earlier plan; held an event count between 1 and 9 and personal names in key names). Patient example and point contributions unchanged.
- `derived/a2_worked_example.json`, text scrub: personal attribution replaced by 'reviewer-requested'; personal attribution replaced by 'a reviewer'; internal version label replaced by 'the primary score'
- `derived/a3_invariance_check.json`, text scrub: personal attribution replaced by 'reviewer-requested'; internal version label replaced by 'the primary score'
- `derived/changepoint_sbp_composite.json`, text scrub: em-dash in a JSON sentence replaced by a semicolon
- `derived/changepoint_sbp_mortality.json`, text scrub: em-dash in a JSON sentence replaced by a semicolon
- `derived/changepoint_si_composite.json`, text scrub: em-dash in a JSON sentence replaced by a semicolon
- `derived/changepoint_si_mortality.json`, text scrub: em-dash in a JSON sentence replaced by a semicolon
- `derived/changepoint_stratified.json`, suppressed (nested denominators): n and events of the strata anticoag, no_anticoag, elderly_anticoag, young_or_noac of the shock-index curves (outcomes composite, mortality) withheld: the patients with a recorded pressure but no pulse (the difference between the pressure-only and shock-index curves) numbered 1 to 9 in the anticoagulant strata, directly or by subtracting the age strata
- `derived/citl_slope.json`, text scrub: name of a local patient-level file made generic; em-dash in a JSON sentence replaced by a semicolon
- `derived/reclass.json`, suppressed (nested denominators): elderly_changepoint_tiein: n and events of the elderly_anticoag stratum of the shock-index curves withheld, as in derived/changepoint_stratified.json (the same cells)
- `derived/reclass.json`, text scrub: absolute project path made relative; personal attribution replaced by 'reviewer-requested'; name of a local patient-level file made generic; em-dash in a JSON sentence replaced by a semicolon
- `derived/refit.json`, text scrub: absolute project path made relative; personal attribution replaced by 'a reviewer'; em-dash in a JSON sentence replaced by a semicolon
- `derived/results_bundle.json`, text scrub: absolute project path made relative; internal role label replaced by 'reviewer comment'
- `derived/risk_tiers.csv`, suppressed: n_override_in_tier: ['rule_out'] (<10) and ['observe'] (complementary, so the first cannot be recovered as the group total minus the other tiers)
- `derived/score_card.json`, suppressed: risk_tiers[].n_override_in_tier: ['rule_out'] (<10), ['observe'] (complementary). No point value, cutpoint, tier definition or probability was changed.
- `derived/score_card.json`, text scrub: internal wording removed (document-section references, version labels, one render hint field, em-dashes); keys renamed where they carried a version label; every number, point value, cutpoint, tier definition and rate is unchanged and was checked equal to the source
- `derived/subgroup_forest_data.csv`, suppressed (nested denominators): 4 N_events cells withheld. Event counts of a comparator whose denominator needs one more measurement (shock index needs pulse, the reverse shock index also needs the Glasgow Coma Scale) differed from those of the next narrower denominator by 1 to 9 patients in: ACS_Level=3 (SI-alone), Anticoagulation=Yes (SI-alone). Complementary cells (the other level of the subgroup, so that a withheld cell cannot be recovered from the shown total): ACS_Level=Unknown (SI-alone), Anticoagulation=No (SI-alone)
- `logs/pre_d11_snapshot/MANIFEST.sha256`, subset: the original manifest lists five files; the lines of the three shipped files are kept (each hash checked against the original), the lines of the two files not shipped are removed
- `logs/qc_manifest.json`, keys removed: fix1_transfusion.bands_full_csv (band counts of the acetabulum-inclusive file; three of its bands differ from the analytic-cohort bands by 1 to 9 patients, because the analytic cohort is nested in that file); ais_predot_zfill_fix.verification_inc_key (a single encounter key used as a spot check; a patient-level identifier). No script reads these keys. All other counts in the file are 0 or at least 10 and are not nested differences of 1 to 9.
- `tables/table1_baseline.csv`, suppressed: 3 primary cells (<10), 3 complementary cells, 6 derived SMD/P cells
- `tables/table1_baseline.csv`, suppressed (nested definitions): cells in the strata columns of outcome rows of the untimed-death recovery ladder withheld, because an outcome definition that adds a few patients to a narrower one would otherwise differ from it by 1 to 9 patients in a stratum, directly or as the elderly total; the complementary cells and the derived SMD and P values of those rows are withheld too
- `tables/table5c_operating_full.csv`, rows dropped: 42 of 217 rows dropped (collapse by omission). all_test_rows / pelvic_score / score_or_override / cutoff -1: Override_Marginal_Flagged in 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 0: Override_Marginal_Flagged in 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 1: Override_Marginal_Flagged in 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 2: Override_Marginal_Flagged in 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 3: Override_Marginal_Events in 1..9; all_test_rows / pelvic_score / score_only / cutoff 4: nested difference in TN with another row of this table is 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 6: override marginal FP differs from another cutoff's by 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 7: override marginal TP differs from another cutoff's by 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 8: override marginal TP differs from another cutoff's by 1..9; all_test_rows / pelvic_score / score_or_override / cutoff 9: override marginal TP differs from another cutoff's by 1..9; all_test_rows / pelvic_score / score_only / cutoff 10: nested difference in TP with a table5d row is 1..9; all_test_rows / pelvic_score / score_only / cutoff 12: nested difference in TP with a table5d row is 1..9; all_test_rows / pelvic_score / score_only / cutoff 15: nested difference in FP with another row of this table is 1..9; all_test_rows / pelvic_score / score_only / cutoff 17: cell TP/FP/N_flagged in 1..9; all_test_rows / shock_index / instrument_only / cutoff 0.2: cell FN in 1..9; all_test_rows / shock_index / instrument_only / cutoff 3.96: cell TP/FP in 1..9; all_test_rows / rSIG / instrument_only / cutoff 0.8: cell TP/FP/N_flagged in 1..9; all_test_rows / rSIG / instrument_only / cutoff 1.2: nested difference in TN with another row of this table is 1..9; all_test_rows / rSIG / instrument_only / cutoff 40: nested difference in FN with another row of this table is 1..9; common_complete_case / pelvic_score / score_only / cutoff -4: nested difference in FN with another row of this table is 1..9; common_complete_case / pelvic_score / score_or_override / cutoff -4: nested difference in FN with another row of this table is 1..9; common_complete_case / pelvic_score / score_only / cutoff 10: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_or_override / cutoff 10: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_only / cutoff 11: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_or_override / cutoff 11: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_only / cutoff 12: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_or_override / cutoff 12: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_only / cutoff 13: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_or_override / cutoff 13: nested difference in TP with another row of this table is 1..9; common_complete_case / pelvic_score / score_only / cutoff 17: cell TP/FP/N_flagged in 1..9; common_complete_case / pelvic_score / score_or_override / cutoff 17: cell TP/FP/N_flagged in 1..9; common_complete_case / shock_index / instrument_only / cutoff 0.2: cell FN in 1..9; common_complete_case / shock_index / instrument_only / cutoff 0.37: nested difference in FN with another row of this table is 1..9; common_complete_case / shock_index / instrument_only / cutoff 0.57: nested difference in FN with another row of this table is 1..9; common_complete_case / shock_index / instrument_only / cutoff 1.07: nested difference in TN with a table5d row is 1..9; common_complete_case / shock_index / instrument_only / cutoff 1.3: nested difference in TN with a table5d row is 1..9; common_complete_case / shock_index / instrument_only / cutoff 1.54: nested difference in TP with a table5d row is 1..9; common_complete_case / shock_index / instrument_only / cutoff 3.96: cell TP/FP in 1..9; common_complete_case / rSIG / instrument_only / cutoff 0.8: cell TP/FP/N_flagged in 1..9; common_complete_case / rSIG / instrument_only / cutoff 2.8: nested difference in TP with a table5d row is 1..9; common_complete_case / rSIG / instrument_only / cutoff 4.2: nested difference in TP with a table5d row is 1..9; common_complete_case / rSIG / instrument_only / cutoff 40: nested difference in FN with another row of this table is 1..9
- `tables/table7_sensitivity.csv`, suppressed (nested definitions): n_events and composite_rate_pct withheld in 9 rows (variants sens_death_dayfallback; slices full, test): the event total of the variant differs from the total of the reference definition of the same cohort slice by 1 to 9 patients (the untimed deaths that the variant adds), so the shown total would reveal that count
- `tables/table8_mtp.csv`, text scrub: personal attribution replaced by 'reviewer comment'
- `tables/table_subgroups.csv`, suppressed (nested denominators): 4 N_events cells withheld. Event counts of a comparator whose denominator needs one more measurement (shock index needs pulse, the reverse shock index also needs the Glasgow Coma Scale) differed from those of the next narrower denominator by 1 to 9 patients in: ACS_Level=3 (SI-alone), Anticoagulation=Yes (SI-alone). Complementary cells (the other level of the subgroup, so that a withheld cell cannot be recovered from the shown total): ACS_Level=Unknown (SI-alone), Anticoagulation=No (SI-alone)

## Files deliberately not shipped

| Source | Reason |
|---|---|
| v3/derived/pelvic_cohort_predecision files, the candidates and cohort_test_scored columnar files, and all *_inc_keys files | patient-level data (one row per patient or patient keys) |
| all serialized Python objects in v3/derived (lasso_cv_fit, roc_data, mtp_roc_data, temporal_roc_data) | binary objects that cannot be screened; the roc_data objects hold patient-level curve points |
| v3/logs/11_pelvic_thresholds.log (76 MB) and the other v3/logs files (run logs, audit notes, manifests of individual steps) | unreviewed run logs, contain absolute paths and internal workflow notes; only missingness.csv, qc_manifest.json, d11_invariance_check.csv and .json and three files of pre_d11_snapshot were reviewed and shipped |
| v3/logs/pre_d11_snapshot/score_card.json and table5_operating.csv | no script reads them; the snapshot score card holds an override split by score tier with counts under 10 |
| v3/figures/figure9_score_card.png and .pdf | the card image prints the override count of the lowest tier (a count between 1 and 9) baked into the picture, which cannot be suppressed in an image |
| v3/figures/figure2_score_distribution.pdf, figure3_roc.pdf, figure10_mtp_equivalent.pdf | vector files encode exact tail bins and tail ROC points with fewer than 10 patients; the PNG versions are shipped |
| v4/figures, the TIFF versions | duplicates of the shipped PNG files |
| v4/figures/figure_legends, the Markdown copies of the v4 tables, the Word copies of Tables 1 to 3, and any manuscript Word file | manuscript text and internal footnotes (the Markdown copy of Supplementary Table S1 has footnotes that name an internal script and a hash list) |
| v4/derived/v4_analyses_manifest.json | internal provenance: absolute paths, an internal plan folder name, hashes of patient-level inputs and of pre-suppression outputs that would no longer match |
| v4/bundle/NUMBERS-LEDGER.md, PI-comments, plans, VERIFY-REPORT.txt, checklists, manuscript folders | internal documents |

`derived/results_bundle.json` is shipped. It passes the small-cell screen: its only integers from 1 to 9 are score points, cutpoints, flags and category counts. Only its three absolute path fields were made relative.

## Logs that the analysis scripts read

Three groups of files from the project's `v3/logs/` are shipped because scripts read them: `logs/qc_manifest.json` (read by scripts 02, 08, 09, 10, 12, 14, `render_figures.py`, `build_bundle.py` and `render_exhibits.py`), `logs/d11_invariance_check.json` and `.csv` (`build_bundle.py` requires the JSON; script 19 writes both) and three files of `logs/pre_d11_snapshot/` (the frozen baseline that scripts 05 and 19 compare against). Each was screened for counts of 1 to 9, absolute paths, names and patient-level content. Findings:

- `qc_manifest.json`: two items removed (listed under Files altered); every other count is 0 or at least 10. It still contains internal wording (section references and labels of project planning documents that are not distributed). Re-running `analysis/01_build_cohort.py` rewrites this file with the removed items; do not commit a regenerated copy without screening it again.
- `d11_invariance_check.json` and `.csv`: AUROC values and point-sum ranges only. Shipped unchanged.
- `pre_d11_snapshot/`: `refit.json`, `table3_points.csv` and `table4_discrimination.csv` shipped unchanged (the SHA-256 of each equals the original manifest line). The snapshot's score card and operating table are not read by any script and are not shipped.

## Notes on specific cases

- `table5d_matched_workload.csv` and `table6b_elderly_ledger.csv` were listed as suspected, and contain no cell under 10. They are shipped unchanged. One note in table6b names a confidence-interval method; it is a statistical term, not a person.
- The override group has 201 patients with 164 events (both shown). Its split across the three score tiers is not published as a whole: the Rule-out tier count is a number between 1 and 9 (`<10`), the Observe tier count is withheld as the complementary cell so that the Rule-out count cannot be recovered by subtraction from the group total and the Escalate tier count, and the Escalate tier count is shown in `derived/risk_tiers.csv` and `derived/score_card.json`. Documents and the README must not quote the split by tier.
- `figure9_score_card` is not shipped because its picture prints a count between 1 and 9. The calculator in `calculator/` shows the same card from `derived/score_card.json`.
- Pairwise checks on the operating-point tables compare every pair of rows of the same instrument across `table5c`, `table5d` and `table5_operating`. They do not test every possible sum of three or more rows.
- Nested definitions and denominators (rule 2, last sentences). Files altered by it: `tables/table7_sensitivity.csv` (the variant `sens_death_dayfallback`, its event total and rate in the full-cohort and test-slice rows), `tables/table1_baseline.csv` (two rows of the untimed-death recovery ladder, in the strata columns and their SMD and P value), `tables/table_subgroups.csv` and `derived/subgroup_forest_data.csv` (shock-index event counts of two subgroups and the complementary levels), `derived/changepoint_stratified.json` and `derived/reclass.json` (n and events of the anticoagulant strata of the shock-index change-point curves). Each is listed with its cells under Files altered; the withheld values are not stated. The change-point estimates, confidence intervals and all other numbers in those files are unchanged.
- Self-check for nested differences. After writing the files the generator compares every pair of counts of one measure within each shipped file, every pair of headline event and denominator totals across files, the outcome rows of Table 1, the subgroup tables, the change-point strata (including cells derived by subtraction) and the operating-point rows, and stops if any two that could be nested sets differ by 1 to 9. Pairs that are not nested (cells of one partition, different variables of one table) are exempt by a stated rule in the script.
- One nested difference is left and cannot be withheld here: the event count of the full-information (ISS and AIS) reference row of Table 3 differs from the primary event count by a number between 1 and 9 (the denominators differ by 30 patients). Both counts are printed in the manuscript's Table 3, which this repository reproduces unchanged, and `logs/pre_d11_snapshot/table4_discrimination.csv` must stay byte-identical to the frozen baseline. The generator lists it as a known exception.
- The generator checks its own output: it stops if the withheld override values, written as a field value or as a split, appear in any shipped data file or in this document.

## Where suppression weakens "sufficient to verify the reported numbers"

The Data Sharing Statement wording that matches this repository is: aggregated, de-identified tables and figures sufficient to verify the manuscript's reported numbers, with the exception of one post hoc subgroup count under 10 (cells with fewer than 10 patients suppressed, with complementary cells withheld). Every number printed in the manuscript was compared with the suppressed files. One printed number cannot be verified from the repository.

| Manuscript item | Printed value | Repository status |
|---|---|---|
| Discussion/Results, women aged 75 years or older who fell, systolic pressure below 90 mmHg and shock index of 1.0 or higher, test set | An event count under 10 among 94 patients, with its percentage and 95% confidence interval | The event count, the percentage and the interval are withheld in `derived/a2_hypotension_mechanism.csv` (the denominator, 94, is shown). A reader can see that the profile exists and how many patients it has, but cannot reproduce the printed event count, rate or interval from the repository. |

No other count under 10 is printed in the main text, Table 1, Table 2, Table 3, the figure legends or Supplementary Table S1. Checked items: every count in Table 1 (the smallest nonzero cell is 95), Table 3 (the smallest is 742), every n printed in the figures, the override count 201 and its 164 events (both shown), and every patient count in Supplementary Table S1 (the smallest is 630). Counts that the repository withholds and the manuscript does not print: the override patients in the Rule-out tier (a count between 1 and 9), the override patients in the Observe tier (withheld as the complementary cell), the patients with no signs of life among older patients on anticoagulants in the extended Table 1 (a count between 1 and 9), two recovered-death outcome rows of the extended Table 1 (counts between 1 and 9), the A1 score-card tail scores, and the extreme cutoffs of the operating-point table.

