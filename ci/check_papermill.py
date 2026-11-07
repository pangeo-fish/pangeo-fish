"""Exercise parameter injection, checkpoints and redaction with a real kernel.

This checks execution plumbing; the full demos check the scientific results.
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import nbformat
import run_demo_notebook as runner
from papermill.exceptions import PapermillExecutionError


def main():
    root = Path(__file__).resolve().parents[1]
    source = nbformat.read(root / "notebooks/pangeo-fish-base-demo.ipynb", as_version=4)
    parameter_index = next(
        i
        for i, cell in enumerate(source.cells)
        if "parameters" in cell.metadata.get("tags", [])
    )
    cells = [
        nbformat.v4.new_code_cell(
            "from pint import UnitRegistry\nureg = UnitRegistry()"
        )
    ]
    cells.extend(source.cells[parameter_index : parameter_index + 3])
    cells.append(
        nbformat.v4.new_code_cell(
            'assert tag_name == "A18828"\n'
            'assert target_root == "./overridden-results/A18828"\n'
            'assert receiver_buffer.to("m").magnitude == 1500\n'
            'assert maximum_speed.to("km/day").magnitude == 33\n'
            'print("papermill-test-password")'
        )
    )
    read_settings = runner.demo_settings

    def overridden_settings(notebook, tag_name):
        settings, parameters = read_settings(notebook, tag_name)
        overrides = {
            "scratch_root": "./overridden-results",
            "receiver_buffer_m": 1500,
            "maximum_speed_km_per_day": 33,
        }
        return settings | overrides, parameters | overrides

    credentials = base64.standard_b64encode(
        b"[credentials]\nusername=papermill-test-user\npassword=papermill-test-password\n"
    ).decode()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        (work / "notebooks").mkdir()
        home = work / "home"
        home.mkdir()
        notebook_path = work / "notebooks/pangeo-fish-base-demo.ipynb"
        original_cwd = Path.cwd()
        try:
            os.chdir(work)
            with (
                patch.dict(
                    os.environ,
                    {
                        "HOME": str(home),
                        "DEMO_TAG_NAME": "A18828",
                        "COPERNICUS_CREDENTIALS_FILE_CONTENT": credentials,
                    },
                ),
                patch.object(runner, "check_saved_states", return_value=[]),
                patch.object(runner, "demo_settings", side_effect=overridden_settings),
            ):
                for fail in [False, True]:
                    notebook = nbformat.v4.new_notebook(cells=cells.copy())
                    if fail:
                        notebook.cells.append(
                            nbformat.v4.new_code_cell(
                                'raise ValueError("intentional failure")'
                            )
                        )
                    nbformat.write(notebook, notebook_path)
                    if fail:
                        try:
                            runner.main("base")
                        except PapermillExecutionError:
                            pass
                        else:
                            raise AssertionError("Expected a failed execution")
                    else:
                        runner.main("base")
                    output = work / "demo-results/base"
                    summary = json.loads((output / "summary.json").read_text())
                    executed_text = (output / "executed.ipynb").read_text()
                    executed = nbformat.reads(executed_text, as_version=4)
                    assert summary["status"] == ("failed" if fail else "passed")
                    code_cells = sum(c.cell_type == "code" for c in executed.cells)
                    assert summary["code_cells"] == code_cells
                    assert len(summary["completed_cells"]) == code_cells - int(fail)
                    assert any(
                        "injected-parameters" in c.metadata.get("tags", [])
                        for c in executed.cells
                    )
                    assert "[REDACTED]" in executed_text
                    assert "papermill-test-password" not in executed_text
                    assert not (
                        home / ".copernicusmarine/.copernicusmarine-credentials"
                    ).exists()
        finally:
            os.chdir(original_cwd)
    print(
        "Papermill injection, derived settings, failure evidence and redaction passed"
    )


if __name__ == "__main__":
    main()
