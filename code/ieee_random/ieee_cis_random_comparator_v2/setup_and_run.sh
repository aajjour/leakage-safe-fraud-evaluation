#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-debug}"
ROOT="$HOME/fraud_temporal_project"
DATA_DIR="$ROOT/data/raw/ieee_cis"
OUT_DIR="$ROOT/analysis/ieee_cis_random_comparator/results_${MODE}"
LOG_DIR="$ROOT/logs"
VENV="$ROOT/.venv-ieee-cis"
THREADS="${THREADS:-6}"
MAX_TRAIN="${MAX_TRAIN:-300000}"
TEMPORAL_METRICS="$ROOT/analysis/ieee_cis_verified/results_full/metrics.csv"
mkdir -p "$OUT_DIR" "$LOG_DIR"
if [[ ! -f "$DATA_DIR/train_transaction.csv" || ! -f "$DATA_DIR/train_identity.csv" ]]; then echo "Missing IEEE-CIS raw files in $DATA_DIR"; exit 1; fi
if [[ ! -d "$VENV" ]]; then python3 -m venv "$VENV"; fi
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q numpy pandas scikit-learn lightgbm xgboost
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
LOG="$LOG_DIR/ieee_cis_random_${MODE}_$(date +%Y%m%d_%H%M%S).log"
python "$(dirname "$0")/run_random_comparator.py" --mode "$MODE" --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" --threads "$THREADS" --max-train "$MAX_TRAIN" --temporal-metrics "$TEMPORAL_METRICS" 2>&1 | tee "$LOG"
echo "Log: $LOG"
echo "Results: $OUT_DIR"
