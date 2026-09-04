#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
ROOT="${FRAUD_PROJECT_ROOT:-$HOME/fraud_temporal_project}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
DIR="$ROOT/analysis/ieee_cis_random_comparator/results_${MODE}"
mkdir -p "$OUTPUT_DIR"
for f in random_metrics.csv random_summary.csv manifest.json; do [[ -s "$DIR/$f" ]] || { echo "Missing $DIR/$f"; exit 1; }; done
cp -f "$DIR/random_summary.csv" "$OUTPUT_DIR/ieee_cis_random_${MODE}_summary.csv"
cd "$ROOT/analysis/ieee_cis_random_comparator"
zip -qr "$OUTPUT_DIR/ieee_cis_random_${MODE}_results.zip" "results_${MODE}"
cd "$ROOT/logs"
zip -q "$OUTPUT_DIR/ieee_cis_random_${MODE}_logs.zip" ieee_cis_random_${MODE}_*.log || true
ls -lh "$OUTPUT_DIR/ieee_cis_random_${MODE}_results.zip" "$OUTPUT_DIR/ieee_cis_random_${MODE}_logs.zip"
