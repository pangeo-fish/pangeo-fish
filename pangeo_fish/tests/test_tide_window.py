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
    # label 08-02 covers [08-01 12:00, 08-02 12:00) = samples 12..35
    assert out.sel(time="2023-08-02").item() == np.arange(12, 36).mean()
    # partial first and last windows
    assert out.sel(time="2023-08-01").item() == np.arange(0, 12).mean()
    assert out.sel(time="2023-08-04").item() == np.arange(60, 72).mean()
