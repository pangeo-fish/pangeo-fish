#!/usr/bin/env bash
set -euo pipefail

result="demo-results/$1"
mkdir -p "$result"
python ci/notebook_environment.py "$result/before.json"
install_status=0
# CPU-only torch first (the constraint torch==2.15.0 also matches 2.15.0+cpu),
# so that pip does not download the CUDA build and the nvidia-* packages.
python -m pip install -c ci/requirements/demo-constraints.txt torch \
  --index-url https://download.pytorch.org/whl/cpu 2>&1 | tee "$result/install-torch.log"
python -m pip install -c ci/requirements/demo-constraints.txt -e '.[notebooks]' --report "$result/install.json" \
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
