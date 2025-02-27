import numpy as np
import pytest
import xarray as xr

from pangeo_fish.tags import to_time_slice


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
