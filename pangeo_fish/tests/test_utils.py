import numpy as np
import pytest
import xarray as xr

from pangeo_fish.utils import _drop_attr, clear_attrs


@pytest.fixture
def create_xarray_dataset():
    """
    Creates a simple xarray dataset for testing
    """
    times = np.arange(3)
    latitudes = np.array([10, 20])
    longitudes = np.array([100])

    em = xr.DataArray(
        np.full((3, len(latitudes), len(longitudes)), 2.0),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        name="em",
    )
    em.attrs = {"some_attr": "value"}

    final = xr.DataArray(
        np.full((len(latitudes), len(longitudes)), 3.0),
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
        name="final",
    )
    final.attrs = {"some_attr": "value"}

    initial = xr.DataArray(np.array([1]), dims=("dummy",), name="initial")
    initial.attrs = {"some_attr": "value"}

    mask = xr.DataArray(np.array([0]), dims=("dummy",), name="mask")
    mask.attrs = {"some_attr": "value"}

    ds = xr.Dataset({"em": em, "final": final, "initial": initial, "mask": mask})

    return ds


@pytest.mark.parametrize("variables", [["em", "final"], "all"])
def test_clear_attrs(create_xarray_dataset, variables):
    """
    Tests `clear_attrs` function.
    """
    ds = create_xarray_dataset
    new_ds = clear_attrs(ds, variables=variables)

    assert new_ds.attrs == {}

    if variables is None:
        expected_vars = list(ds.variables)  # All variables
    elif variables == "all":
        expected_vars = list(getattr(ds, "variables", ds.coords))
    else:
        expected_vars = variables

    for var in expected_vars:
        assert new_ds[var].attrs == {}, f"Attributes of {var} were not removed"


@pytest.mark.parametrize(
    "attr, expected_attrs",
    [
        ("some_attr", {}),
        (
            "another_attr",
            {"em": {"some_attr": "value"}, "final": {"some_attr": "value"}},
        ),
    ],
)
def test_drop_attr(create_xarray_dataset, attr, expected_attrs):
    """
    Tests `_drop_attr` function.
    """
    ds = create_xarray_dataset
    new_ds = _drop_attr(ds, attr=attr)

    assert new_ds.attrs.get(attr) is None

    for var in new_ds.variables.values():
        assert var.attrs.get(attr) is None
