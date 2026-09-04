# Leakage-safe temporal protocol: detailed invariants and audit evidence

This note preserves implementation-level detail condensed from the main manuscript in response to the reviewer request for greater concision.

## Six governing invariants
1. **Temporal precedence:** every fitted object predates the observations it scores.
2. **Label admissibility:** target-derived information uses only labels confirmed by the relevant cutoff.
3. **Fitting locality:** each transformation is estimated only within its designated development block.
4. **Prevalence integrity:** validation, calibration, and test blocks retain natural class prevalence.
5. **Test immutability:** no modelling decision follows final-test inspection.
6. **Auditability:** pipeline state and split provenance are recoverable from code and recorded metadata.

## Expanded verification requirements
For each outer fold, record the prediction timestamp definition, label-maturity rule, ordered block boundaries, feature-availability rules, point-in-time joins, fitted preprocessing provenance, class counts before and after any training-only sampling, validation/model-selection provenance, calibration-fit/selection windows, operating-policy rules, software versions, seeds, parameter/configuration files, and final-test access record. Report all future folds, including unfavourable folds, together with convergence and boundary diagnostics.

## Full leakage taxonomy
See `tables/Supplementary_Table_S2_Leakage_Taxonomy.csv`.
