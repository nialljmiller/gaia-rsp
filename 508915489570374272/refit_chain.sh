#!/usr/bin/env bash
set -euo pipefail
cd /home/nill/MESA/GAIA_RSP/508915489570374272

echo "=== waiting for in-progress final run to exit ==="
while pgrep -f "fit_target.py final" >/dev/null; do sleep 15; done
echo "=== final (pre-rescan) done at $(date) ==="

echo "=== shape rescan (rsp_alfa) at $(date) ==="
python fit_target.py shape --force --alfa 0.8 0.9 1.0 1.2 1.3 1.5 > fit_shape_rescan.log 2>&1

echo "=== amplitude rescan (rsp_alfam) at $(date) ==="
python fit_target.py amplitude --force > fit_amplitude_rescan.log 2>&1

echo "=== final rerun at chosen alfa/alfam at $(date) ==="
python fit_target.py final --force > fit_final_rescan.log 2>&1

echo "=== REFIT CHAIN COMPLETE at $(date) ==="
