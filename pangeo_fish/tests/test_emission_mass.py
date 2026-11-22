import numpy as np
import pytest
import xarray as xr

from pangeo_fish.helpers import _check_emission_mass


def emission(values):
    time = np.array(
        ["2025-03-31T00", "2025-03-31T01", "2025-03-31T02"], dtype="datetime64[ns]"
    )
    return xr.Dataset(
        {"pdf": (("time", "cells"), np.asarray(values, dtype=float))},
        coords={"time": time},
    )


def test_valid_emission_passes():
    _check_emission_mass(emission([[0.5, 0.5], [0.0, 1.0], [np.nan, 1.0]]))


@pytest.mark.parametrize(
    "row", [[np.nan, np.nan], [0.0, 0.0], [0.0, np.nan]], ids=["nan", "zero", "mixed"]
)
def test_empty_time_step_raises_with_its_time(row):
    with pytest.raises(ValueError, match=r"at 1 time step\(s\) \(2025-03-31T01:00"):
        _check_emission_mass(emission([[0.5, 0.5], row, [0.2, 0.8]]))


def test_dask_and_missing_variable():
    _check_emission_mass(emission([[1.0, 0.0]] * 3).chunk({"time": 1}))
    _check_emission_mass(xr.Dataset())


def test_mask_and_extra_dimensions():
    ds = emission([[0.5, 0.5], [1.0, 0.0], [0.2, 0.8]])
    ds["mask"] = ("cells", [False, True])
    with pytest.raises(ValueError, match="inside the mask at 1 time step"):
        _check_emission_mass(ds)
    ds = ds.drop_vars("mask")
    stacked = xr.concat([ds, ds], dim="member").transpose("member", "time", "cells")
    _check_emission_mass(stacked)
    stacked["pdf"][:, 1] = 0.0
    with pytest.raises(ValueError, match="2025-03-31T01:00"):
        _check_emission_mass(stacked)
