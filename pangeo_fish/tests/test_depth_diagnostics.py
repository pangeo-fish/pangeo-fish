import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish import helpers
from pangeo_fish.bathy import sample_max_depth


def _tag_log():
    # 3 model days, tag max depth 10, 30 and 50 m
    time = pd.date_range("2024-01-01", periods=72, freq="1h")
    depth = np.full(time.size, 5.0)
    depth[5] = 10.0
    depth[30] = 30.0
    depth[60] = 50.0
    return xr.Dataset(
        {"pressure": ("time", depth), "temperature": ("time", np.full(72, 12.0))},
        coords={"time": time},
    )


def _model_depth():
    # 2D model grid, 4 cells (one land cell)
    return xr.DataArray(
        [[20.0, 40.0], [60.0, np.nan]],
        dims=("lat", "lon"),
        coords={"lat": [50.0, 50.1], "lon": [1.0, 1.1]},
    )


MODEL_TIME = pd.date_range("2024-01-01T12:00", periods=3, freq="1D")


def test_diagnose_depth_vs_bathymetry_shares():
    diag = helpers.diagnose_depth_vs_bathymetry(
        tag_log=_tag_log(), bathymetry=_model_depth(), time=MODEL_TIME
    )

    np.testing.assert_allclose(diag["tag_max_depth"], [10, 30, 50])
    np.testing.assert_array_equal(diag["n_valid_cells"], [3, 3, 3])
    # cells 20, 40, 60 m
    np.testing.assert_allclose(diag["model_too_shallow"], [0, 1 / 3, 2 / 3])
    np.testing.assert_allclose(diag["model_max_depth"], [60, 60, 60])
    assert "reference_too_shallow" not in diag
    assert diag.attrs["relative_depth_threshold"] == 1.0


def test_diagnose_depth_vs_bathymetry_threshold_and_mask():
    mask = _model_depth().notnull() & (_model_depth() < 50)
    diag = helpers.diagnose_depth_vs_bathymetry(
        tag_log=_tag_log(),
        bathymetry=_model_depth(),
        time=MODEL_TIME,
        mask=mask,
        relative_depth_threshold=0.5,
        warn=False,
    )

    # cells 20 and 40 m vs 5, 15, 25 m
    np.testing.assert_array_equal(diag["n_valid_cells"], [2, 2, 2])
    np.testing.assert_allclose(diag["model_too_shallow"], [0, 0, 0.5])


def test_diagnose_depth_vs_bathymetry_warns_and_uses_reference():
    shallow = _model_depth() / 4  # 5, 10, 15 m
    reference = _model_depth()

    with pytest.warns(UserWarning, match="2 of 3 time steps") as record:
        diag = helpers.diagnose_depth_vs_bathymetry(
            tag_log=_tag_log(),
            bathymetry=shallow,
            time=MODEL_TIME,
            reference_bathymetry=reference,
        )
    assert "reference bathymetry has deep enough cells at 2" in str(record[0].message)

    np.testing.assert_allclose(diag["model_too_shallow"], [1 / 3, 1, 1])
    np.testing.assert_allclose(diag["reference_too_shallow"], [0, 1 / 3, 2 / 3])
    # too shallow in the model only
    np.testing.assert_allclose(diag["model_only_too_shallow"], [1 / 3, 2 / 3, 1 / 3])
    np.testing.assert_allclose(diag["reference_max_depth"], [60, 60, 60])


def test_diagnose_depth_vs_bathymetry_time_dependent_and_healpix():
    # HEALPix-like cells, bathymetry H0 + XE that depends on time
    h0 = xr.DataArray([20.0, 40.0, np.nan], dims="cells")
    xe = xr.DataArray(
        [[0.0, 0.0, np.nan], [15.0, 0.0, np.nan], [0.0, 0.0, np.nan]],
        dims=("time", "cells"),
        coords={"time": MODEL_TIME},
    )

    # without time argument, the time of the bathymetry is used
    with pytest.warns(UserWarning, match="1 of 3 time steps"):
        diag = helpers.diagnose_depth_vs_bathymetry(
            tag_log=_tag_log(), bathymetry=h0 + xe
        )

    np.testing.assert_allclose(diag["model_too_shallow"], [0, 0, 1])
    np.testing.assert_allclose(diag["model_max_depth"], [40, 40, 40])


def test_diagnose_depth_vs_bathymetry_requires_time():
    with pytest.raises(ValueError, match="time"):
        helpers.diagnose_depth_vs_bathymetry(
            tag_log=_tag_log(), bathymetry=_model_depth()
        )


def _gebco():
    lat = np.arange(49.0, 51.0, 0.1)
    lon = np.arange(0.0, 2.0, 0.1)
    elevation = np.full((lat.size, lon.size), -10.0)
    elevation[10, 10] = -80.0  # a deep hole at (50.0, 1.0)
    elevation[0, :] = 5.0  # land row
    return xr.DataArray(elevation, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def test_sample_max_depth_nearest_and_window():
    gebco = _gebco()
    lat = xr.DataArray([50.0, 50.2, 49.0, np.nan], dims="cells")
    lon = xr.DataArray([1.0, 1.0, 0.5, 1.0], dims="cells")

    nearest = sample_max_depth(gebco, lat, lon)
    np.testing.assert_allclose(nearest, [80, 10, -5, np.nan])
    assert nearest.dims == ("cells",)
    assert nearest.attrs["units"] == "m"

    # a 0.5 degree box around (50.2, 1.0) contains the hole at (50.0, 1.0)
    window = sample_max_depth(gebco, lat, lon, window=0.5)
    np.testing.assert_allclose(window, [80, 80, 10, np.nan])


def test_sample_max_depth_descending_latitude_and_names():
    gebco = _gebco().isel(lat=slice(None, None, -1))
    gebco = gebco.rename(lat="latitude", lon="longitude")

    depth = sample_max_depth(gebco, np.array([50.0, 50.3]), np.array([1.0, 1.0]))

    assert isinstance(depth, np.ndarray)
    np.testing.assert_allclose(depth, [80, 10])
