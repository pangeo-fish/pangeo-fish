import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.tags import reshape_by_bins, to_time_slice


@pytest.mark.parametrize("end", [1, 2])
@pytest.mark.parametrize(
    ["start", "stop"],
    (
        (
            np.datetime64("2016-01-12 22:51:12"),
            np.datetime64("2016-02-24 23:59:01"),
        ),
        (
            np.datetime64("2022-10-12 07:31:27"),
            np.datetime64("2022-11-01 04:47:11"),
        ),
    ),
)
def test_to_time_slice(start, stop, end):
    if end == 1:
        times_ = np.array([start, stop, "NaT"], dtype="datetime64[ns]")
    else:
        times_ = np.array([start, "NaT", stop], dtype="datetime64[ns]")

    times = xr.DataArray(times_, dims="event_name")
    actual = to_time_slice(times)

    expected = slice(start, stop)
    assert actual == expected


@pytest.mark.parametrize(
    ["tag", "intervals", "expected"],
    (
        (
            xr.Dataset(
                {"temperature": ("time", np.arange(10))},
                coords={
                    "time": xr.date_range(
                        "2022-05-01 12:00:00", freq="60 s", periods=10
                    )
                },
            ),
            xr.DataArray(
                pd.IntervalIndex.from_arrays(
                    np.array(
                        ["2022-05-01 12:00:00", "2022-05-01 12:05:00"],
                        dtype="datetime64[ns]",
                    ),
                    np.array(
                        ["2022-05-01 12:05:00", "2022-05-01 12:10:00"],
                        dtype="datetime64[ns]",
                    ),
                    closed="left",
                ),
                dims="time",
            ),
            xr.Dataset(
                {
                    "temperature": (
                        ["time", "obs"],
                        np.arange(10, dtype=float).reshape(2, 5),
                    )
                },
                coords={"obs": np.arange(5)},
            ),
        ),
        (
            xr.Dataset(
                {"temperature": ("time", np.linspace(0, 1, 12))},
                coords={
                    "time": xr.date_range(
                        "2022-07-21 13:46:25", freq="90 s", periods=12
                    )
                },
            ),
            xr.DataArray(
                pd.IntervalIndex.from_arrays(
                    np.array(
                        [
                            "2022-07-21 13:46:20",
                            "2022-07-21 13:52:25",
                            "2022-07-21 13:58:25",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    np.array(
                        [
                            "2022-07-21 13:52:25",
                            "2022-07-21 13:58:25",
                            "2022-07-21 14:04:25",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    closed="left",
                ),
                dims="time",
            ),
            xr.Dataset(
                {"temperature": (["time", "obs"], np.linspace(0, 1, 12).reshape(3, 4))},
                coords={"obs": np.arange(4)},
            ),
        ),
    ),
)
def test_reshape_by_bins(tag, intervals, expected):
    actual = reshape_by_bins(tag, dim="time", bins=intervals, other_dim="obs")

    xr.testing.assert_allclose(actual, expected.assign_coords(time=intervals))
