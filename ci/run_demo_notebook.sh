#!/usr/bin/env bash
set -euo pipefail

result="demo-results/$1"
mkdir -p "$result"
python ci/notebook_environment.py "$result/before.json"
install_status=0
python -m pip install -e '.[notebooks]' --report "$result/install.json" \
  2>&1 | tee "$result/install.log" || install_status=$?
python ci/notebook_environment.py "$result/after.json"
compare_status=0
python ci/notebook_environment.py "$result/before.json" "$result/after.json" \
  | tee "$result/comparison.json" || compare_status=$?
if [ "$install_status" != 0 ] || [ "$compare_status" != 0 ]; then
  exit 1
fi
python ci/check_papermill.py
python ci/run_demo_notebook.py "$1"
