"""Keep the two maintained demos comparable and safe to parameterize."""

import json
from pathlib import Path

import pytest

from ci.demo_configuration import demo_settings

ROOT = Path(__file__).resolve().parents[2]


def load_demo(name):
    return json.loads(
        (ROOT / "notebooks" / f"pangeo-fish-{name}-demo.ipynb").read_text()
    )


def test_common_defaults_are_identical():
    base, _ = demo_settings(load_demo("base"))
    complete, _ = demo_settings(load_demo("complete"))
    assert base.pop("scratch_root") != complete.pop("scratch_root")
    assert base == complete
    assert base["bbox"] == {"latitude": [46, 51.5], "longitude": [-7, 0]}
    assert base["refinement_level"] == 10
    assert base["tolerance"] == 1e-6
    assert base["track_modes"] == ["viterbi_healpix", "mode", "mean"]
    assert base["comparison_track_modes"] == ["mode", "mean"]


@pytest.mark.parametrize("name", ["base", "complete"])
def test_public_tag_override_retains_defaults_and_source(name):
    notebook = load_demo(name)
    original = json.dumps(notebook)
    defaults, _ = demo_settings(notebook)
    settings, overrides = demo_settings(notebook, "A18828")
    assert overrides == {"tag_name": "A18828"}
    assert settings == defaults | overrides
    assert json.dumps(notebook) == original
    cells = notebook["cells"]
    index = next(
        i
        for i, c in enumerate(cells)
        if "parameters" in c.get("metadata", {}).get("tags", [])
    )
    assert "derived-settings" in cells[index + 1]["metadata"]["tags"]
    assert "target_root =" in "".join(cells[index + 2]["source"])


def test_unknown_tag_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        demo_settings({"cells": []}, "unknown")


def test_ambiguous_parameters_are_rejected():
    notebook = load_demo("base")
    cell = next(
        c
        for c in notebook["cells"]
        if "parameters" in c.get("metadata", {}).get("tags", [])
    )
    notebook["cells"].append(cell)
    with pytest.raises(ValueError, match="exactly one"):
        demo_settings(notebook)
