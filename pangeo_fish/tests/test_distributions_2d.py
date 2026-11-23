"""The 2D grid used for the maps and videos follows the source grid."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.helpers import _lonlat_shape, render_distributions


def test_lonlat_shape_of_a_curvilinear_grid_with_time_first():
    ny, nx = 198, 252
    lat, lon = np.meshgrid(
        np.linspace(46, 51.5, ny), np.linspace(-7, 0, nx), indexing="ij"
    )
    ds = xr.Dataset(
        {"diff": (("time", "yi", "xi"), np.zeros((11, ny, nx)))},
        coords={"latitude": (("yi", "xi"), lat), "longitude": (("yi", "xi"), lon)},
    )
    assert _lonlat_shape(ds) == [nx, ny]


def test_lonlat_shape_of_a_regular_grid():
    ds = xr.Dataset(
        {"diff": (("time", "latitude", "longitude"), np.zeros((3, 4, 5)))},
        coords={"latitude": np.arange(4.0), "longitude": np.arange(5.0)},
    )
    assert _lonlat_shape(ds) == [5, 4]


def test_render_refuses_states_without_positive_values(tmp_path):
    y, x = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    data = xr.Dataset(
        {
            "states": (("time", "y", "x"), np.full((2, 3, 4), np.nan)),
            "emission": (("time", "y", "x"), np.ones((2, 3, 4))),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=2),
            "longitude": (("y", "x"), x),
            "latitude": (("y", "x"), y),
        },
    )
    with pytest.raises(ValueError, match="no grid point with states > 0"):
        render_distributions(
            data=data,
            output_path=str(tmp_path / "states"),
            frames_dir=str(tmp_path / "frames"),
            time_step=1,
        )
