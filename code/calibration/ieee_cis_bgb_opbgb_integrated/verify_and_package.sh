#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-debug}
ROOT=${ROOT:-$HOME/fraud_temporal_project}
OUT="$ROOT/analysis/ieee_cis_bgb_opbgb_integrated/results_${MODE}"
for f in calibration_metrics.csv calibration_summary.csv calibrator_parameters.csv split_audit.csv manifest.json; do [[ -s "$OUT/$f" ]] || { echo "Missing/empty $OUT/$f" >&2; exit 3; }; done
python3 - "$OUT" <<'PY'
import json,sys,pandas as pd
from pathlib import Path
p=Path(sys.argv[1]); m=pd.read_csv(p/'calibration_metrics.csv'); s=pd.read_csv(p/'split_audit.csv'); man=json.loads((p/'manifest.json').read_text())
assert not m.empty and not s.empty
assert (s[['train','validation','cal_fit','cal_select','test']]>0).all().all()
assert set(['raw','platt','beta','isotonic','bgb','opbgb_symmetric','opbgb_two_sided','nested_selected']).issubset(set(m.calibration))
print('verified metric rows:',len(m)); print('folds:',man['folds']); print('models:',man['models'])
PY
DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME/Downloads}
mkdir -p "$DOWNLOAD_DIR"
ZIP="$DOWNLOAD_DIR/ieee_cis_bgb_opbgb_${MODE}_results.zip"
rm -f "$ZIP"; (cd "$OUT" && zip -q -r "$ZIP" .)
ls -lh "$ZIP"
