#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-debug}
ROOT=${ROOT:-$HOME/fraud_temporal_project}
DATA_DIR=${DATA_DIR:-$ROOT/data/raw/ieee_cis}
VENV=${VENV:-$ROOT/.venv-ieee-cis}
THREADS=${THREADS:-6}
MAX_TRAIN=${MAX_TRAIN:-300000}
BGB_SHRINK=${BGB_SHRINK:-1.0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/analysis/ieee_cis_bgb_opbgb_integrated/results_${MODE}"
LOGDIR="$ROOT/logs"; mkdir -p "$OUT" "$LOGDIR"
for f in train_transaction.csv train_identity.csv; do [[ -s "$DATA_DIR/$f" ]] || { echo "Missing $DATA_DIR/$f" >&2; exit 2; }; done
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then python3 -m venv "$VENV"; PYTHON_BIN="$VENV/bin/python"; fi
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m pip install -q --upgrade pip
  "$PYTHON_BIN" -m pip install -q numpy pandas scipy scikit-learn lightgbm xgboost joblib
fi
LOG="$LOGDIR/ieee_cis_bgb_opbgb_${MODE}_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$THREADS}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-$THREADS}
"$PYTHON_BIN" "$SCRIPT_DIR/run_integrated_calibration.py" --mode "$MODE" --data-dir "$DATA_DIR" --output-dir "$OUT" --threads "$THREADS" --max-train "$MAX_TRAIN" --bgb-shrink "$BGB_SHRINK" 2>&1 | tee "$LOG"
echo "Log: $LOG"
echo "Results: $OUT"
