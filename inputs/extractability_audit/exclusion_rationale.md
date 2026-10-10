# Pelvic Arrival Tool v2 — Extractability & Coverage Audit

## Overview

The v2 arrival risk score is restricted, **before any LASSO**, to a candidate pool that a clinician can plausibly know in the first minutes of a pelvic-fracture resuscitation — arrival vitals, GCS, age/sex, mechanism, transport mode, EMS/MedicAlert-conveyed coagulation comorbidities, and the ICD-anchored unstable-ring pattern. This enforces a genuinely *bedside* tool and forecloses post-decision leakage. Coverage below is computed on the primary (ring-only) analytic cohort (`cohort_primary==1`, N = 239,736; the full acetabulum-inclusive CSV has 291,766 rows).

Two categories are deliberately **excluded** from the bedside score: (1) **registrar-coded injury severity** (ISS and body-region AIS) — these are abstracted post hoc from the complete chart by a trauma registrar, are unavailable in the arrival decision window, and risk circularity with the hemorrhage outcome; they appear only as a separate 'full-information' comparator model. (2) **All treatment and outcome variables** (hemorrhage-control receipt, transfusion volume, death, the composite itself) — these occur *after* the arrival decision and would leak the label. Both are listed with coverage below to show the exclusion is on *principle* (timing/leakage), not on availability.

## Kept — bedside-extractable candidates (19)

### Demographics

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `age_65_74` | 100.0 | 239,736 | 37,582 | 15.676 |
| `age_ge75` | 100.0 | 239,736 | 74,334 | 31.007 |
| `sex_female` | 99.773 | 239,191 | 127,532 | 53.318 |

*Rationale:* Arrival/bedside-extractable: measured or reported in the first minutes of resuscitation (demographic / measured arrival vital / GCS / mechanism / transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored ring pattern).

### Vitals-BP

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `sbp_lt90` | 97.604 | 233,992 | 13,617 | 5.819 |
| `sbp_lt70` | 97.604 | 233,992 | 3,076 | 1.315 |
| `sbp_unrecordable` | 100.0 | 239,736 | 608 | 0.254 |

*Rationale:* Arrival/bedside-extractable: measured or reported in the first minutes of resuscitation (demographic / measured arrival vital / GCS / mechanism / transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored ring pattern).

### Vitals-HR

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `hr_ge120` | 97.873 | 234,637 | 23,269 | 9.917 |

*Rationale:* Arrival/bedside-extractable: measured or reported in the first minutes of resuscitation (demographic / measured arrival vital / GCS / mechanism / transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored ring pattern).

### Shock index

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `shock_index` | 97.317 | 233,305 | — | — |
| `si_ge0_9` | 97.317 | 233,305 | 43,405 | 18.604 |
| `si_ge1_0` | 97.317 | 233,305 | 29,812 | 12.778 |

*Rationale:* Arrival/bedside-extractable: measured or reported in the first minutes of resuscitation (demographic / measured arrival vital / GCS / mechanism / transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored ring pattern). (continuous).

### Neuro-GCS

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `gcs_motor_le5` | 95.598 | 229,184 | 26,621 | 11.616 |
| `gcs_le8` | 95.636 | 229,273 | 18,553 | 8.092 |

*Rationale:* Arrival/bedside-extractable: measured or reported in the first minutes of resuscitation (demographic / measured arrival vital / GCS / mechanism / transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored ring pattern).

### Mechanism

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `mech_high_energy` | 100.0 | 239,736 | 104,201 | 43.465 |
| `mech_mvc` | 100.0 | 239,736 | 55,881 | 23.309 |
| `mech_fall` | 100.0 | 239,736 | 110,024 | 45.894 |

*Rationale:* Arrival/bedside-extractable: measured or reported in the first minutes of resuscitation (demographic / measured arrival vital / GCS / mechanism / transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored ring pattern).

### Transport

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `air_transport` | 100.0 | 239,736 | 22,451 | 9.365 |

*Rationale:* Arrival/bedside-extractable: measured or reported in the first minutes of resuscitation (demographic / measured arrival vital / GCS / mechanism / transport mode / EMS- or MedicAlert-conveyed comorbidity / ICD-anchored ring pattern).

### Comorbidity (coag)

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `cc_anticoag` | 100.0 | 239,736 | 28,966 | 12.082 |
| `cc_bleeding` | 100.0 | 239,736 | 2,261 | 0.943 |

*Rationale:* Anticoagulant use (preexisting code 31) directly changes the major-hemorrhage-pathway decision; typically known from EMS handoff, patient report, or MedicAlert. FORCE-IN anchor.

### Anatomy (ICD)

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `unstable_ring` | 100.0 | 239,736 | 42,798 | 17.852 |

*Rationale:* Disrupted/unstable pelvic ring (ICD-10 S32.81*) — ICD-anchored rather than registrar AIS-graded, so robust to AIS miscoding. FORCE-IN anchor.

## Excluded — registrar-coded severity & treatment/outcome leakage (15)

### Registrar severity

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `iss` | 99.967 | 239,658 | — | — |
| `ais_head` | 100.0 | 239,736 | — | — |
| `ais_head_ge3` | 100.0 | 239,736 | 30,003 | 12.515 |
| `ais_chest` | 100.0 | 239,736 | — | — |
| `ais_abd` | 100.0 | 239,736 | — | — |

*Rationale:* Registrar-coded injury severity — assigned post hoc from the full chart by a trauma registrar, NOT knowable at arrival; excluded from the bedside score (offered only as a separate 'full-information' comparator, CONTRACTS §5). ISS/AIS also risk circularity with the hemorrhage outcome.

### Treatment/outcome

| variable | fill % | N present | N positive | positive % |
|---|---|---|---|---|
| `received_hc` | 100.0 | 239,736 | 19,583 | 8.169 |
| `surg_hc` | 100.0 | 239,736 | 13,998 | 5.839 |
| `angio_hc` | 100.0 | 239,736 | 8,598 | 3.586 |
| `embo_any` | 100.0 | 239,736 | 8,269 | 3.449 |
| `blood_4h` | 91.812 | 220,106 | — | — |
| `substantial_transfusion` | 100.0 | 239,736 | 14,238 | 5.939 |
| `death_inhosp` | 100.0 | 239,736 | 13,426 | 5.6 |
| `major_hemorrhage_event` | 100.0 | 239,736 | 31,054 | 12.953 |
| `icu_los` | 36.482 | 87,460 | — | — |
| `vent_days` | 17.605 | 42,206 | — | — |

*Rationale:* Post-decision treatment/outcome variable — using it to predict the arrival decision is leakage/circular (FORBIDDEN_X, CONTRACTS §5).

## Downstream implication

The kept candidate matrix is the input to `02_engineer_features.py` (`LASSO_X_COLS`) and `04_lasso_stability.py`, which prune to a parsimonious 10–15-variable integer score (λ.1se + stability ≥0.85 + |β|≥0.15 floor), with `age≥65`, `cc_anticoag`, and `unstable_ring` force-in anchors. `02` must assert `set(LASSO_X_COLS) ∩ FORBIDDEN_X == ∅` — every excluded variable above belongs to FORBIDDEN_X.
