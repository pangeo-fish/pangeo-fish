import numpy as np
import pandas as pd
import xarray as xr

from pangeo_fish.tide import _centered_daily_mean


def test_centered_daily_mean_labels_and_samples():
    time = pd.date_range("2023-08-01", periods=72, freq="1h")  # 3 days
    da = xr.DataArray(np.arange(72.0), coords={"time": time}, dims="time")
    out = _centered_daily_mean(da)
    assert list(out.time.values) == list(
        pd.date_range("2023-08-01", periods=4, freq="1D").values
    )
    # label 08-02 covers (08-01 12:00, 08-02 12:00] = samples 13..36
    assert out.sel(time="2023-08-02").item() == np.arange(13, 37).mean()
    # partial first and last windows
    assert out.sel(time="2023-08-01").item() == np.arange(0, 13).mean()
    assert out.sel(time="2023-08-04").item() == np.arange(61, 72).mean()


def test_noon_sample_on_the_same_day_as_the_temperature():
    """Review of #344: a sample at 12:00 went to day T for the temperature and to
    T + 1 for the tide. All tide day windows now follow the temperature bins."""
    import cf_xarray  # noqa: F401

    from pangeo_fish.cf import bounds_to_bins
    from pangeo_fish.tags import reshape_by_bins
    from pangeo_fish.tide import _window_labels

    days = pd.date_range("2023-08-01", periods=4, freq="1D")
    tag_times = pd.DatetimeIndex(
        ["2023-08-02 11:50", "2023-08-02 12:00", "2023-08-02 12:10"]
    )
    tag = xr.Dataset({"depth": ("time", [1.0, 2.0, 3.0])}, coords={"time": tag_times})
    bins = (
        xr.Dataset(coords={"time": days})
        .cf.add_bounds(["time"], output_dim="bounds")
        .pipe(bounds_to_bins, bounds_dim="bounds")
        .get("time_bins")
    )
    reshaped = reshape_by_bins(tag, dim="time", bins=bins)
    temperature_day = {
        float(v): pd.Timestamp(t)
        for t, row in zip(reshaped.time.values, reshaped.depth.values)
        for v in row
        if np.isfinite(v)
    }
    tide_day = dict(zip([1.0, 2.0, 3.0], _window_labels(tag_times, "1D")))
    assert temperature_day == tide_day
    assert tide_day[2.0] == pd.Timestamp("2023-08-02")
    assert tide_day[3.0] == pd.Timestamp("2023-08-03")
    daily = _centered_daily_mean(tag["depth"])
    assert daily.sel(time="2023-08-02").item() == 1.5
    assert daily.sel(time="2023-08-03").item() == 3.0
