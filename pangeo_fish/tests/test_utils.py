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
            xr.DataArray(np.array(30e9), attrs={"units": "ns"}),
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
            xr.DataArray(np.array(90e9), attrs={"units": "ns"}),
        ),
    ),
)
def test_temporal_resolution(time, expected):
    actual = utils.temporal_resolution(time)

    xr.testing.assert_identical(actual, expected)
