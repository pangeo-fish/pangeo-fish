#!/usr/bin/env bash
set -euo pipefail

result="demo-results/$1"
mkdir -p "$result"
PYFES_VERSION=2026.10.0  # conda-forge build for numpy >= 2.5.4, as in the pinned image
python ci/notebook_environment.py "$result/before.json"
install_status=0
# CPU-only torch first (the constraint torch==2.15.0 also matches 2.15.0+cpu),
# so that pip does not download the CUDA build and the nvidia-* packages.
python -m pip install -c ci/requirements/demo-constraints.txt torch \
  --index-url https://download.pytorch.org/whl/cpu 2>&1 | tee "$result/install-torch.log"
if [ "$1" = complete ]; then
  # The Complete demo needs pyfes (pangeo_fish.tide_healpix). PyPI has no wheels, so it
  # comes from conda-forge, into the image's own environment. Every conda-forge build
  # of pyfes depends on mkl, which needs the LLVM OpenMP runtime (_openmp_mutex *_llvm)
  # while the image ships the GNU one: --freeze-installed has no solution, so the
  # solver may switch _openmp_mutex (the conda-forge mechanism for that). numpy is
  # pinned to the image's version, the comparison below fails if a protected Python
  # package changes, and conda-changes.txt lists every conda package that changed.
  conda_tool=$(command -v micromamba || command -v mamba || command -v conda) || {
    echo "no conda, mamba or micromamba in the image to install pyfes" >&2
    exit 1
  }
  prefix=$(python -c 'import sys; print(sys.prefix)')
  numpy_version=$(python -c 'import numpy; print(numpy.__version__)')
  "$conda_tool" list -p "$prefix" > "$result/conda-before.txt"
  "$conda_tool" install -y -p "$prefix" -c conda-forge \
    "pyfes==$PYFES_VERSION" "numpy==$numpy_version" 2>&1 | tee "$result/install-pyfes.log"
  "$conda_tool" list -p "$prefix" > "$result/conda-after.txt"
  diff "$result/conda-before.txt" "$result/conda-after.txt" \
    > "$result/conda-changes.txt" || true
  cat "$result/conda-changes.txt"
  python -c "import pyfes; print('pyfes', pyfes.__version__)"
fi
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
