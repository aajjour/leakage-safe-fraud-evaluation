# Constrained monotone OPBGB calibration, version 2

This package provides strictly increasing symmetric and two-sided OPBGB post-hoc calibrators. It does not alter the fitted IEEE-CIS base classifiers. The debug workflow uses one chronological fold and logistic regression; the full frozen outputs use four temporal folds and logistic regression, LightGBM, and XGBoost.

## Included files
- `monotone_opbgb.py`: constrained calibrators and analytic derivatives.
- `run_monotone_opbgb_ieee.py`: debug/full IEEE-CIS workflow.
- `setup_and_run.sh`: environment, unit-test, and execution launcher.
- `verify_and_package.sh`: output and rank-preservation verifier.
- `tests/test_monotone_opbgb.py`: six numerical and nesting tests.
- `vendor/bgbcal.py`: BGB implementation used by the comparison.
- `results_full/`: frozen four-fold outputs reported in the manuscript.

## Guarantees and selection rule
All shape parameters are positive, the analytic derivative is strictly positive on `(0,1)`, beta calibration is nested at `lambda_0=lambda_1=1`, penalty selection uses only the chronological calibration-selection block, and future-test data remain untouched until final evaluation. Every output row records rank violations and Spearman agreement. The older unconstrained transform-logistic OPBGB appears only in the integrated diagnostic workflow and is not used for confirmatory monotonicity claims.

## Run
```bash
export IEEE_DATA_DIR=/path/to/ieee_cis
export THREADS=6
./setup_and_run.sh debug
./verify_and_package.sh debug
```
After the debug run passes, replace `debug` by `full`. Unit tests can also be run directly with `python3 -m pytest -q`.

## Numerical safeguard
The inner odds-power transform is evaluated in log space and is not clipped before the outer calibration map. This avoids finite-precision plateaus and preserves ordering. Parameter bounds are restricted to reduce saturation risk.
