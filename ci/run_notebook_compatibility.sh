#!/usr/bin/env bash
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export MPLBACKEND=Agg
export NUMBA_NUM_THREADS=2
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""

python ci/notebook_environment.py compatibility-results/before.json
if [ "${BASELINE_ONLY:-false}" = true ]; then
  exit 0
fi

install_status=0
python -m pip install -e . --report compatibility-results/install.json \
  2>&1 | tee compatibility-results/install.log || install_status=$?
python ci/notebook_environment.py compatibility-results/after.json
compare_status=0
python ci/notebook_environment.py compatibility-results/before.json \
  compatibility-results/after.json | tee compatibility-results/comparison.json || compare_status=$?
if [ "$install_status" != 0 ] || [ "$compare_status" != 0 ]; then
  exit 1
fi

# Test tools are separate from the installation being measured. Check that adding
# them cannot silently repair or perturb the runtime we just tested.
python -m pip install pytest pytest-cov
python ci/notebook_environment.py compatibility-results/test-environment.json
python ci/notebook_environment.py compatibility-results/after.json \
  compatibility-results/test-environment.json
python -m pytest --junitxml=compatibility-results/tests.xml \
  2>&1 | tee compatibility-results/tests.log
