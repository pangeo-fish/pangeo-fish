"""Record the real running image's environment, including pre-existing conflicts."""

import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def snapshot(path):
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
    )
    if check.returncode not in (0, 1) or (
        check.returncode == 1 and not check.stdout.strip()
    ):
        raise RuntimeError(f"pip check failed to run: {check.stderr}")
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in metadata.distributions()
        },
        "pip_check_returncode": check.returncode,
        "pip_check_stderr": check.stderr,
        "pip_check": check.stdout.splitlines() if check.returncode else [],
    }
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


PROTECTED = {
    "xarray",
    "zarr",
    "numpy",
    "scipy",
    "pandas",
    "numba",
    "numcodecs",
    "dask",
    "distributed",
    "fsspec",
    "s3fs",
    "gcsfs",
    "kerchunk",
    "aiohttp",
    "aiobotocore",
    "boto3",
    "botocore",
    "click",
    "torch",
    "argopy",
    "icechunk",
    "rioxarray",
    "virtualizarr",
}


def compare(before, after):
    changes = {
        name: [before["packages"][name], after["packages"].get(name)]
        for name in sorted(PROTECTED & before["packages"].keys())
        if before["packages"][name] != after["packages"].get(name)
    }
    new_conflicts = sorted(set(after["pip_check"]) - set(before["pip_check"]))
    result = {
        "protected_package_changes": changes,
        "pre_existing_conflicts": before["pip_check"],
        "new_conflicts": new_conflicts,
    }
    print(json.dumps(result, indent=2))
    return not changes and not new_conflicts


if __name__ == "__main__":
    if len(sys.argv) == 2:
        snapshot(sys.argv[1])
    else:
        sys.exit(
            0
            if compare(*(json.loads(Path(p).read_text()) for p in sys.argv[1:]))
            else 1
        )
