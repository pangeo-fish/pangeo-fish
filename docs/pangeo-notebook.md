# Pangeo Notebook compatibility

In Pangeo@EOSC, choose `quay.io/pangeo/pangeo-notebook:latest`, clone this
repository, and run `python -m pip install -e .` in its root directory.
Restart the notebook kernel after installation.

The migrated runtime uses Zarr-Python 3. All Pangeo-Fish writers explicitly use
**Zarr format 2 on disk**, including intermediate estimator caches and tide
region writes. Existing format 2 datasets remain readable. This change does not
convert scientific data to format 3.

Runtime dependencies are maintained in `pyproject.toml`. Xarray is bounded to
`>=2025.7.1,<2027`, Zarr to `>=3.1.2,<4`, the fsspec clients to
`>=2025.7.0,<2027`, and kerchunk to `>=0.2.9,<0.3`. The Xarray lower bound also
satisfies current Pandas' optional Xarray requirement. The existing BinGrouper
regression tests run unchanged; an old Xarray upper bound is no longer needed.
Copernicus Marine `>=2.4,<3` allows modern Click. Its own dependencies and those
of s3fs and Zarr govern boto3/aiobotocore, aiohttp and numcodecs; Pangeo-Fish does
not independently pin their transitive dependencies. Released HEALPix packages
replace floating Git dependencies. CI tests the migrated lower bounds together
on Python 3.11 and current dependencies on Python 3.13.

## Reproducible image and baseline

`ci/notebook-image.txt` records the tested multi-platform digest:

```text
quay.io/pangeo/pangeo-notebook@sha256:b64bded4c91932815cf18dfd4d1cdc812214037684f489f3b353043924700b5e
```

The Linux amd64 manifest is
`sha256:e7d9808e3770bf5d73d5fb0c88f59a2f9d2caa8656135c8f7579426090946350`.
The pristine container baseline, including all installed distribution versions,
is committed in `ci/baselines/pangeo-notebook-2026-11-06.json`.
It was collected by actually running the image in GitHub Actions; a local Mac
environment was not used as a substitute.

| Component | Pristine image |
| --- | --- |
| Python | 3.13.16 |
| Xarray | 2026.9.0 |
| Zarr | 3.4.1 |
| Dask / distributed | 2026.8.0 |
| fsspec / s3fs | 2026.9.0 |
| gcsfs | 2026.10.1 |
| NumPy / Pandas | 2.5.4 / 2.3.3 |

The image already has two `pip check` conflicts: `python-lsp-ruff` requires
missing `ruff`, and `xmip` requires `xgcm<0.7.0` but the image has `0.10.1`.
These are recorded, not repaired by Pangeo-Fish. The compatibility test rejects
additional conflicts and any version change to an existing protected package,
including Xarray, Zarr, Dask, distributed, fsspec, s3fs, gcsfs, NumPy, Pandas,
Click, boto3, argopy, icechunk, rioxarray and virtualizarr.

## Verification

The Notebook workflow pulls both the pinned digest and `latest`, records image
inspection and the pristine environment, runs ordinary `pip install -e .`,
compares the environments, and runs the complete test suite. Artifacts retain
the install report, `pip check` output, environment comparison and test results
even when a step fails. Tests cover exact enumerated HMM marginals and score,
probability normalization, valid zeros, missing observations, land masks,
timestamps, HEALPix attributes, cache reuse, actual HEALPix convolution, and
region/append/reopen round trips. No scientific expected values were relaxed.

To reproduce on a machine with Docker, run from the repository root:

```sh
mkdir -p compatibility-results
docker run --rm --user root --entrypoint bash -e HOME=/tmp \
  -v "$PWD:/work" -w /work "$(cat ci/notebook-image.txt)" \
  ci/run_notebook_compatibility.sh
```

Both workflows run for pull requests targeting `release`. Notebook compatibility
also has a Monday 06:23 UTC schedule. GitHub only activates scheduled workflows
present on the repository's default branch; when merging only into `release`,
the workflow must also be added to the default branch to activate that schedule.
Scheduled runs check out `release`.

On the local Mac, running the full suite reproduced a native `foscat`/PyTorch
crash on both unchanged `release` and the migrated source after earlier tests;
the estimator tests pass when run separately. Linux container CI is the target
environment verification; the Mac crash remains a separate limitation.
