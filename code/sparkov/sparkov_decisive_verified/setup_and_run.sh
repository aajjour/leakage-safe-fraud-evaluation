#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${FRAUD_PROJECT_ROOT:-$HOME/fraud_temporal_project}"
DATA="${SPARKOV_DATA_DIR:-$ROOT/data/raw/sparkov}"
RUN="${FRAUD_RUN_DIR:-$ROOT/analysis/sparkov_decisive}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv-sparkov"
mkdir -p "$RUN" "$ROOT/logs"
[[ -f "$DATA/fraudTrain.csv" && -f "$DATA/fraudTest.csv" ]] || { echo "Sparkov files missing in $DATA"; exit 2; }
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q pandas numpy scipy scikit-learn lightgbm xgboost pyarrow
export FRAUD_PROJECT_ROOT="$ROOT" SPARKOV_DATA_DIR="$DATA" FRAUD_RUN_DIR="$RUN" PYTHONUNBUFFERED=1
export THREADS="${THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
MODE="${1:-debug}"; export RUN_MODE="$MODE"
if [[ "$MODE" == debug ]]; then export MAX_TRAIN="${MAX_TRAIN:-120000}" DEBUG_MAX_ROWS="${DEBUG_MAX_ROWS:-350000}"; LOG="$ROOT/logs/sparkov_decisive_debug_$(date +%Y%m%d_%H%M%S).log";
elif [[ "$MODE" == full ]]; then export MAX_TRAIN="${MAX_TRAIN:-350000}"; LOG="$ROOT/logs/sparkov_decisive_full_$(date +%Y%m%d_%H%M%S).log";
else echo "Usage: $0 [debug|full]"; exit 2; fi
python "$HERE/run_decisive.py" 2>&1 | tee "$LOG"
echo "Log: $LOG"
