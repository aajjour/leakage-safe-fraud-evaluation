#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
ROOT="${FRAUD_PROJECT_ROOT:-$HOME/fraud_temporal_project}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
DIR="$ROOT/analysis/ieee_cis_verified/results_${MODE}"
mkdir -p "$OUTPUT_DIR"
for f in metrics.csv summary.csv manifest.json; do [[ -s "$DIR/$f" ]] || { echo "Missing $DIR/$f"; exit 1; }; done
python3 - <<PY2
import pandas as pd, json
p='$DIR'
m=pd.read_csv(p+'/metrics.csv'); s=pd.read_csv(p+'/summary.csv'); j=json.load(open(p+'/manifest.json'))
print('metric rows:',len(m)); print('summary rows:',len(s)); print(j); print(s.to_string(index=False))
PY2
cd "$ROOT/analysis/ieee_cis_verified"
zip -r -q "$OUTPUT_DIR/ieee_cis_${MODE}_results.zip" "results_${MODE}"
cd "$ROOT/logs"
zip -q "$OUTPUT_DIR/ieee_cis_${MODE}_logs.zip" ieee_cis_${MODE}_*.log || true
ls -lh "$OUTPUT_DIR/ieee_cis_${MODE}_results.zip" "$OUTPUT_DIR/ieee_cis_${MODE}_logs.zip"
