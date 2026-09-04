#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-debug}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/results_${MODE}"
python3 - <<PY
from pathlib import Path
import pandas as pd
out=Path(r'''$OUT''')
req=['calibration_metrics.csv','calibration_summary.csv','calibrator_parameters.csv','split_audit.csv','manifest.json']
missing=[x for x in req if not (out/x).exists()]
if missing: raise SystemExit(f'Missing outputs: {missing}')
m=pd.read_csv(out/'calibration_metrics.csv')
mono=m[m.calibration.str.contains('_monotone')]
assert len(mono)>0
assert int(mono.rank_violations.sum())==0, mono[['model','fold','calibration','rank_violations']]
assert (mono.spearman_raw_calibrated>0.999999).all()
print('verified metric rows:',len(m))
print('monotone rows:',len(mono))
print('rank violations:',int(mono.rank_violations.sum()))
PY
DEST="${PWD}/opbgb_monotone_${MODE}_results.zip"
mkdir -p "$(dirname "$DEST")"
rm -f "$DEST"
(cd "$SCRIPT_DIR" && zip -qr "$DEST" "results_${MODE}" docs monotone_opbgb.py tests requirements.txt README.md)
ls -lh "$DEST"
