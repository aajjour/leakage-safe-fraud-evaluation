# Evidence status at final packaging

## Frozen primary evidence
- Sparkov: six untouched future monthly folds; capped and 180-day histories; 0-, 7-, 14-, and 30-day label-delay scenarios; logistic regression, LightGBM, and XGBoost.
- IEEE-CIS temporal: four chronological folds with natural-prevalence future tests.
- IEEE-CIS matched random comparator: four repeated matched benchmark realizations. These overlap in transactions and are interpreted descriptively, not as independent experimental units.

## Calibration evidence
- Platt, isotonic, and beta calibration use later development data only.
- BGB is the stable intermediate extension.
- Constrained symmetric and two-sided monotone OPBGB are confirmatory model-dependent extensions; positivity constraints and the full audit establish zero rank violations.
- Older unconstrained transform-logistic variants are retained only as diagnostic evidence because they can alter ordering.

## Verification completed
- Manuscript primary values reconciled to frozen CSV outputs.
- Exact finite fold-resampling script made portable and rerun.
- Six monotone OPBGB unit tests passed.
- Missing OPBGB runner, verifier, and vendor files restored from the numerically stable version-2 package.
- Raw-file hashes, environment lock, execution logs, and method-to-output map added.

## Scope limitations
The archive does not contain raw third-party data, a prospective bank deployment, transaction-level prediction files for every workflow. Those limitations are disclosed in the manuscript.
