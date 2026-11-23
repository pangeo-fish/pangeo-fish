import warnings

import numpy as np
import pytest
import xarray as xr

from pangeo_fish.helpers import _check_temperature_coverage

ID = "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m"


def model(temp_cells, H0=(np.nan, 10.0, 10.0, 10.0)):
    """1 x 4 cells (land + 3 sea), temperature finite only in ``temp_cells``."""
    temp = np.full((2, 2, 1, 4), np.nan)
    temp[:, :, 0, list(temp_cells)] = 12.0
    return xr.Dataset(
        {
            "TEMP": (("time", "depth", "lat", "lon"), temp),
            "H0": (("lat", "lon"), np.array(H0)[None, :]),
        },
        coords={
            "time": np.array(["2025-04-06T00", "2025-04-06T01"], "datetime64[ns]"),
            "depth": [0.5, 2.0],
            "lat": [51.4],
            "lon": [3.0, 3.1, 3.2, 3.3],
        },
    )


def test_full_coverage_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_temperature_coverage(model([1, 2, 3]), ID)
        _check_temperature_coverage(model([0, 1, 2, 3]), ID)  # land TEMP ignored


def test_no_temperature_raises():
    with pytest.raises(ValueError, match=r"no temperature in the requested area \(3"):
        _check_temperature_coverage(model([]), ID)
    with pytest.raises(ValueError, match="all NaN"):
        _check_temperature_coverage(model([0]), ID)  # only land has TEMP


def test_partial_coverage_warns_with_extent():
    with pytest.warns(UserWarning, match=r"2 of 3 sea cells.*lon 3.10 to 3.10"):
        _check_temperature_coverage(model([1]), ID)


def test_lazy_and_without_sea_or_time():
    _check_temperature_coverage(model([1, 2, 3]).chunk({"time": 1}), ID)
    _check_temperature_coverage(model([], H0=(np.nan,) * 4), ID)
    _check_temperature_coverage(model([]).isel(time=slice(0, 0)), ID)
