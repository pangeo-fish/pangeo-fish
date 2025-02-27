import numpy as np
import pytest
import xarray as xr

from pangeo_fish import utils


def test_normalize():
    data = np.array([[3, 1, 7, 5, 4], [4, 5, 9, 0, 2], [3, 4, 3, 5, 5]])
    expected_data = np.array(
        [
            [0.15, 0.05, 0.35, 0.25, 0.2],
            [0.2, 0.25, 0.45, 0, 0.1],
            [0.15, 0.2, 0.15, 0.25, 0.25],
        ]
    )
    arr = xr.DataArray(data, dims=("t", "c"))
    actual = utils.normalize(arr, dim="c")
    expected = xr.DataArray(expected_data, dims=("t", "c"))

    expected_sum = xr.ones_like(arr["t"], dtype="float64")

    xr.testing.assert_allclose(actual, expected)
    xr.testing.assert_allclose(actual.sum(dim="c"), expected_sum)


@pytest.mark.parametrize(
    ["time", "expected"],
    (
        (
            xr.DataArray(
                np.array(
                    [
                        "2010-01-04 22:51:33",
                        "2010-01-04 22:52:03",
                        "2010-01-04 22:52:33",
                    ],
                    dtype="datetime64[ns]",
                ),
                dims="time",
            ),
            xr.DataArray(np.array(30), attrs={"units": "s"}),
        ),
        (
            xr.DataArray(
                np.array(
                    [
                        "2010-01-04 22:51:33",
                        "2010-01-04 22:53:03",
                        "2010-01-04 22:54:33",
                    ],
                    dtype="datetime64[ns]",
                ),
                dims="time",
            ),
            xr.DataArray(np.array(90), attrs={"units": "s"}),
        ),
        (
            xr.DataArray(
                np.array(
                    [
                        "2010-01-04 22:51:03",
                        "2010-01-04 22:53:03",
                        "2010-01-04 22:55:03",
                    ],
                    dtype="datetime64[ns]",
                ),
                dims="time",
            ),
            xr.DataArray(np.array(120), attrs={"units": "s"}),
        ),
    ),
)
def test_temporal_resolution(time, expected):
    actual = utils.temporal_resolution(time)

    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize(
    ["ds", "expected"],
    (
        (
            xr.Dataset(coords={"longitude": np.arange(10), "latitude": np.arange(11)}),
            ["longitude", "latitude"],
        ),
        (
            xr.Dataset(
                coords={
                    "longitude": (["y", "x"], np.arange(12).reshape(3, 4)),
                    "latitude": (["y", "x"], np.linspace(0, 1, 12).reshape(3, 4)),
                }
            ),
            ["y", "x"],
        ),
        (
            xr.Dataset(
                coords={
                    "xi": ("xi", np.arange(10), {"axis": "X"}),
                    "yi": ("yi", np.arange(11), {"axis": "Y"}),
                }
            ),
            ["yi", "xi"],
        ),
    ),
)
def test_detect_spatial_dims(ds, expected):
    actual = utils._detect_spatial_dims(ds)

    assert sorted(actual) == sorted(expected)
