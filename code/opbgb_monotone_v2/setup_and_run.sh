#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-debug}"
THREADS="${THREADS:-6}"
MAX_TRAIN="${MAX_TRAIN:-300000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fraud_temporal_project}"
DATA_DIR="${IEEE_DATA_DIR:-$PROJECT_ROOT/data/ieee_cis}"
OUT_DIR="$SCRIPT_DIR/results_${MODE}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

if [[ ! -f "$DATA_DIR/train_transaction.csv" || ! -f "$DATA_DIR/train_identity.csv" ]]; then
  echo "Missing IEEE-CIS files in $DATA_DIR" >&2
  echo "Set IEEE_DATA_DIR to the directory containing train_transaction.csv and train_identity.csv." >&2
  exit 2
fi

"$PYTHON_BIN" -m pytest -q "$SCRIPT_DIR/tests"
rm -rf "$OUT_DIR" && mkdir -p "$OUT_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/run_monotone_opbgb_ieee.py" \
  --mode "$MODE" --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" \
  --threads "$THREADS" --max-train "$MAX_TRAIN" --bgb-shrink 1.0

echo "Results: $OUT_DIR"
