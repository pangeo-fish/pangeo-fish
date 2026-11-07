"""Execute one of the two user-facing demos, retaining partial failure evidence."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main(name):
    if name not in {"base", "complete"}:
        raise ValueError("Only base and complete demos are supported")
    source = Path(f"notebooks/pangeo-fish-{name}-demo.ipynb")
    output = Path("demo-results") / name
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    # Stored outputs are not evidence of this run.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    work = output / "work"
    work.mkdir(exist_ok=True)
    credentials = os.environ.get("COPERNICUS_CREDENTIALS_JSON", "")
    sensitive = []
    if credentials:
        sensitive = [
            v for v in json.loads(credentials).values() if isinstance(v, str) and v
        ]

    def redact(text):
        for value in sensitive:
            text = text.replace(value, "[REDACTED]")
        return text

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
        "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
        "completed_cells": [],
        "status": "running",
    }

    def started(cell, cell_index, **kwargs):
        print(f"Starting cell {cell_index}", flush=True)
        summary["active_cell"] = cell_index

    def completed(cell, cell_index, **kwargs):
        reply = kwargs.get("execute_reply", {})
        if reply.get("content", {}).get("status") == "ok":
            summary["completed_cells"].append(cell_index)
        print(f"Completed cell {cell_index}", flush=True)
        save_notebook()

    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(work.resolve())}},
        on_cell_start=started,
        on_cell_executed=completed,
    )
    start = time.monotonic()
    try:
        client.execute()
        summary["status"] = "passed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = redact(str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        save_notebook()
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if credentials:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1])
