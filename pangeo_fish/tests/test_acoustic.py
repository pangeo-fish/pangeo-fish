import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish import acoustic

pytestmark = pytest.mark.skip(reason="Skipping all tests in this file temporarily.")


def test_emission_probability_basic():
    """
    Test simple de la fonction emission_probability.
    """

    times = pd.date_range("2025-02-12", periods=3, freq="D")
    grid = xr.Dataset(
        {"acoustic": (["time", "x", "y"], np.random.random((3, 2, 2)))},  #  3x2x2
        coords={"time": times, "x": [0, 1], "y": [0, 1]},
    )

    acoustic_data = xr.DataArray(np.random.random((3, 2, 2)), dims=("time", "x", "y"))
    stations_data = xr.DataArray(np.ones((3, 2, 2)), dims=("time", "x", "y"))

    deployment_id = xr.DataArray(np.ones((3, 2, 2)), dims=("time", "x", "y"))

    tag = xr.Dataset(
        {
            "acoustic": acoustic_data,
            "stations": stations_data,
            "deployment_id": deployment_id,
        }
    )

    result = acoustic.emission_probability(
        tag, grid, 1000, nondetections="ignore", dims=["x", "y"]
    )

    assert isinstance(result, xr.Dataset)
    assert "acoustic" in result
    assert result["acoustic"].dims == ("time", "x", "y")

    expected_map = np.random.random((3, 2, 2))
    np.testing.assert_array_equal(result["acoustic"].values, expected_map)
