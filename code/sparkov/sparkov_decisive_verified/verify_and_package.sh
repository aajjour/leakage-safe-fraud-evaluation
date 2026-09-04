#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${FRAUD_PROJECT_ROOT:-$HOME/fraud_temporal_project}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
RES="$ROOT/analysis/sparkov_decisive/results"
mkdir -p "$OUTPUT_DIR"
for f in decisive_metrics.csv decisive_summary.csv decisive_predictions.csv.gz paired_tests.csv run_manifest.json; do [[ -s "$RES/$f" ]] || { echo "Missing $RES/$f"; exit 2; }; done
python3 - <<PY2
import pandas as pd
r=pd.read_csv('$RES/decisive_metrics.csv'); print('Rows:',len(r)); print(r.groupby(['model','policy','delay_days','calibration']).size().describe()); print(pd.read_csv('$RES/decisive_summary.csv').sort_values('pr_auc',ascending=False).head(15).to_string(index=False))
PY2
cd "$ROOT/analysis"; zip -qr "$OUTPUT_DIR/sparkov_decisive_results.zip" sparkov_decisive/results
cd "$ROOT/logs"; zip -q "$OUTPUT_DIR/sparkov_decisive_logs.zip" sparkov_decisive_*.log || true
echo "Created $OUTPUT_DIR/sparkov_decisive_results.zip and $OUTPUT_DIR/sparkov_decisive_logs.zip"
