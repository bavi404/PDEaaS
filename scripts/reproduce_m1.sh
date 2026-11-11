#!/usr/bin/env bash
# Re-run the complete M1 data pipeline from the project environment and run the full test suite in
# strict mode (missing real data FAILS instead of skipping).
#
#   bash scripts/reproduce_m1.sh            # ~30 min on 4 CPU cores (dominated by exact CG solves)
#
# In this cloud container, package-index hosts are on NO_PROXY but only reachable via the proxy, so
# the downloader (which queries the live DaRUS listing) is run unchanged; pip/uv/npm need NO_PROXY cleared.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
test -x "$PY" || { echo "missing .venv (see README)"; exit 1; }
$PY -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

$PY scripts/download_pdebench.py --beta 1.0 --out data/raw   # re-verifies size + MD5 against live DaRUS listing
$PY scripts/inspect_dataset.py
$PY scripts/preprocess.py
$PY scripts/validate_downsampling.py
$PY scripts/audit_data.py
$PY scripts/verify_reference_solver.py
(cd backend && PDEFLOW_REQUIRE_DATA=1 ../$PY -m pytest -v -rA -p no:cacheprovider | tee ../reports/m1_pytest.txt)
