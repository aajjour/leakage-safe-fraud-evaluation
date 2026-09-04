#!/bin/bash
set -euo pipefail
MODE="${1:-debug}"
THREADS="${THREADS:-6}"
MAX_TRAIN="${MAX_TRAIN:-300000}"
ROOT="$HOME/fraud_temporal_project"
DATA_DIR="${DATA_DIR:-$ROOT/data/raw/ieee_cis}"
OUT_DIR="$ROOT/analysis/ieee_cis_verified/results_${MODE}"
LOG_DIR="$ROOT/logs"
VENV="$ROOT/.venv-ieee-cis"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR" "$LOG_DIR"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip wheel
  "$VENV/bin/pip" install numpy pandas scikit-learn lightgbm xgboost
fi
[[ -f "$DATA_DIR/train_transaction.csv" ]] || { echo "Missing $DATA_DIR/train_transaction.csv"; exit 1; }
[[ -f "$DATA_DIR/train_identity.csv" ]] || { echo "Missing $DATA_DIR/train_identity.csv"; exit 1; }
LOG="$LOG_DIR/ieee_cis_${MODE}_$(date +%Y%m%d_%H%M%S).log"
"$VENV/bin/python" "$SCRIPT_DIR/run_ieee_cis.py" --mode "$MODE" --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" --threads "$THREADS" --max-train "$MAX_TRAIN" 2>&1 | tee "$LOG"
echo "Log: $LOG"
echo "Results: $OUT_DIR"
