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
    if check.returncode not in (0, 1) or check.stderr.strip():
        raise RuntimeError(f"pip check failed to run: {check.stderr}")
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in metadata.distributions()
        },
        "pip_check_returncode": check.returncode,
        "pip_check": check.stdout.splitlines() if check.returncode else [],
    }
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    snapshot(sys.argv[1])
