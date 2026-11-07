"""Execute one of the two user-facing demos, retaining partial failure evidence."""

import base64
import configparser
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import nbformat
import papermill as pm
from demo_configuration import demo_settings
from papermill.iorw import LocalHandler, papermill_io


def check_saved_states(work):
    import numpy as np
    import xarray as xr
    import zarr

    results = []
    for path in sorted(work.rglob("states.zarr")):
        with (
            xr.open_zarr(path) as states,
            xr.open_zarr(path.parent / "combined.zarr") as emission,
        ):
            probabilities = states["states"]
            spatial = [dim for dim in probabilities.dims if dim != "time"]
            totals = probabilities.sum(spatial).compute().values
            np.testing.assert_allclose(totals, 1, atol=1e-6, rtol=1e-6)
            assert not bool(
                (probabilities < -1e-12).any().compute()
            ), "negative probability"
            xr.testing.assert_identical(states.time, emission.time)
            np.testing.assert_array_equal(states.cell_ids, emission.cell_ids)
            if "mask" in emission:
                land = probabilities.where(~emission["mask"].astype(bool)).fillna(0)
                np.testing.assert_allclose(land.compute(), 0, atol=1e-12)
            assert zarr.open_group(path).metadata.zarr_format == 2
            results.append(
                {"store": str(path.relative_to(work)), "timesteps": len(totals)}
            )
    if not results:
        raise AssertionError("No saved state probabilities were produced")
    return results


def main(name):
    if name not in {"base", "complete"}:
        raise ValueError("Only base and complete demos are supported")
    source = Path(f"notebooks/pangeo-fish-{name}-demo.ipynb")
    output = (Path("demo-results") / name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    source_code_cells = sum(c.cell_type == "code" for c in notebook.cells)
    settings, parameters = demo_settings(
        notebook, os.environ.get("DEMO_TAG_NAME", "A19124")
    )
    if settings["tag_name"] == "A18828" and name != "base":
        raise ValueError("A18828 trials are restricted to base-demo")
    # Stored outputs are not evidence of this run.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    work = output / "work"
    work.mkdir(exist_ok=True)
    credentials = os.environ.get("COPERNICUS_CREDENTIALS_FILE_CONTENT", "")
    sensitive = []
    if credentials:
        config = configparser.RawConfigParser()
        config.read_string(base64.standard_b64decode(credentials).decode("utf8"))
        sensitive = list(dict(config.items("credentials")).values())

    def redact(text):
        for value in sensitive:
            text = text.replace(value, "[REDACTED]")
        return text

    class RedactedOutput(LocalHandler):
        def write(self, buf, path):
            super().write(redact(buf), path.removeprefix("redacted://"))

        def pretty_path(self, path):
            return path.removeprefix("redacted://")

    papermill_io.register("redacted://", RedactedOutput())

    def save_notebook():
        (output / "executed.ipynb").write_text(redact(nbformat.writes(notebook)))

    if credentials:
        folder = Path.home() / ".copernicusmarine"
        folder.mkdir(exist_ok=True)
        path = folder / ".copernicusmarine-credentials"
        path.write_text(credentials)
        path.chmod(0o600)
    summary = {
        "notebook": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "settings": settings,
        "engine": "papermill",
        "parameters": parameters,
        "source_code_cells": source_code_cells,
        "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
        "completed_cells": [],
        "status": "running",
    }
    cell_started = {}
    summary["cell_seconds"] = {}

    def checkpoint():
        summary["elapsed_seconds"] = time.monotonic() - start
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def started(cell, cell_index, **kwargs):
        if cell.cell_type != "code":
            return
        print(f"Starting cell {cell_index}", flush=True)
        summary["active_cell"] = cell_index
        cell_started[cell_index] = time.monotonic()
        checkpoint()

    def completed(cell, cell_index, **kwargs):
        reply = kwargs.get("execute_reply", {})
        if reply.get("content", {}).get("status") == "ok":
            summary["completed_cells"].append(cell_index)
        summary["cell_seconds"][str(cell_index)] = (
            time.monotonic() - cell_started[cell_index]
        )
        print(f"Completed cell {cell_index}", flush=True)
        checkpoint()

    start = time.monotonic()
    try:
        notebook = pm.execute_notebook(
            notebook,
            "redacted://" + str(output / "executed.ipynb"),
            parameters=parameters,
            kernel_name="python3",
            language="python",
            cwd=str(work),
            execution_timeout=3600,
            progress_bar=False,
            on_cell_start=started,
            on_cell_executed=completed,
        )
        summary["code_cells"] = sum(c.cell_type == "code" for c in notebook.cells)
        summary["saved_states_checks"] = check_saved_states(work)
        summary["status"] = "passed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = redact(str(error))
        raise
    finally:
        if (output / "executed.ipynb").is_file():
            notebook = nbformat.read(output / "executed.ipynb", as_version=4)
        summary["code_cells"] = sum(c.cell_type == "code" for c in notebook.cells)
        summary["elapsed_seconds"] = time.monotonic() - start
        save_notebook()
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if credentials:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1])
