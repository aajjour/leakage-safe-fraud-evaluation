# Method-to-output map

| Manuscript component | Main code | Frozen evidence |
|---|---|---|
| Sparkov temporal benchmark and delay sensitivity | `code/sparkov/sparkov_decisive_verified/run_decisive.py` | `results/sparkov/` and `tables/sparkov_primary.csv` |
| IEEE-CIS chronological benchmark | `code/ieee_temporal/ieee_cis_verified/run_ieee_cis.py` | `results/ieee_temporal/` and `tables/ieee_primary.csv` |
| Matched random comparator | `code/ieee_random/ieee_cis_random_comparator_v2/run_random_comparator.py` | `results/ieee_random/` and `tables/random_vs_temporal.csv` |
| Standard BGB/diagnostic OPBGB comparison | `code/calibration/ieee_cis_bgb_opbgb_integrated/run_integrated_calibration.py` | `results/ieee_calibration/` |
| Constrained monotone OPBGB | `code/opbgb_monotone_v2/run_monotone_opbgb_ieee.py` | `code/opbgb_monotone_v2/results_full/` |
| Exact fold-resampling sensitivity intervals | `supplementary_material/statistical_uncertainty/exact_fold_resampling.py` | same directory, five CSV outputs |
| Reporting checklist | not computational | `tables/Supplementary_Table_S1_Reporting_Checklist.csv` |
