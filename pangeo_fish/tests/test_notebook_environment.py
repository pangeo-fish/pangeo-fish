from copy import deepcopy

from ci.notebook_environment import compare


def test_existing_image_conflicts_are_allowed():
    before = {"packages": {"xarray": "2026.9.0"}, "pip_check": ["existing conflict"]}
    assert compare(before, deepcopy(before))


def test_new_conflicts_fail():
    before = {"packages": {"xarray": "2026.9.0"}, "pip_check": ["existing conflict"]}
    after = deepcopy(before)
    after["pip_check"].append("new conflict")
    assert not compare(before, after)


def test_core_downgrade_and_upgrade_fail():
    before = {"packages": {"xarray": "2026.9.0"}, "pip_check": []}
    for version in ("2025.3.1", "2027.1.0", None):
        after = deepcopy(before)
        after["packages"]["xarray"] = version
        assert not compare(before, after)


def test_adding_missing_project_dependencies_is_allowed():
    before = {"packages": {"xarray": "2026.9.0"}, "pip_check": []}
    after = deepcopy(before)
    after["packages"].update({"pangeo-fish": "999", "torch": "2.13.0"})
    assert compare(before, after)
