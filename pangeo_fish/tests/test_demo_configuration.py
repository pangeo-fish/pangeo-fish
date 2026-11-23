"""Keep the two maintained demos comparable and safe to parameterize."""

import json
from pathlib import Path

import pytest

from ci.demo_configuration import (
    DEFAULT_TAG_SOURCE,
    TAG_SOURCES,
    demo_settings,
    tag_source_parameters,
)

ROOT = Path(__file__).resolve().parents[2]


def load_demo(name):
    return json.loads(
        (ROOT / "notebooks" / f"pangeo-fish-{name}-demo.ipynb").read_text()
    )


def test_common_defaults_are_identical():
    base, _ = demo_settings(load_demo("base"))
    complete, _ = demo_settings(load_demo("complete"))
    assert base.pop("scratch_root") != complete.pop("scratch_root")
    assert complete.pop("tide_method") == "legacy"
    assert complete.pop("tide_model") == "tpxo9"
    assert complete.pop("tide_model_source") is None
    assert complete.pop("tide_constants_store") is None
    assert complete.pop("tide_s3_endpoint") is None
    assert complete.pop("tide_constituents") is None
    assert complete.pop("tide_sigma_m") == 2.0
    assert complete.pop("tide_min_samples") == 6
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


@pytest.mark.parametrize("name", ["base", "complete"])
@pytest.mark.parametrize("source", sorted(TAG_SOURCES))
def test_tag_sources_only_override_tag_parameters(name, source):
    defaults, _ = demo_settings(load_demo(name))
    overrides = tag_source_parameters(source)
    assert set(overrides) <= {"tag_root", "tag_storage_options"}
    assert set(overrides) <= set(defaults)


def test_default_tag_source_is_used_when_none_is_given():
    assert DEFAULT_TAG_SOURCE in TAG_SOURCES
    assert tag_source_parameters() == tag_source_parameters(DEFAULT_TAG_SOURCE)
    assert tag_source_parameters("") == tag_source_parameters(DEFAULT_TAG_SOURCE)


def test_unknown_tag_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown tag source"):
        tag_source_parameters("elsewhere")
