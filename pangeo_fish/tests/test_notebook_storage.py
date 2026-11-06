"""Small scientific/storage regressions run in the actual Notebook image."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from zarr.storage import LocalStore, MemoryStore

from pangeo_fish.hmm.estimator import CachedEstimator
from pangeo_fish.storage import write_time_blocks


TRANSITION = np.array([
    [0.75, 0.25, 0, 0],
    [0.25, 0.5, 0.25, 0],
    [0, 0.25, 0.75, 0],
    [0, 0, 0, 1],
])


class MatrixPredictor:
    def __init__(self, sigma):
        self.sigma = sigma

    def predict(self, values, *, mask=None):
        result = TRANSITION @ values
        return result if mask is None else np.where(mask, result, 0)


@pytest.fixture
def emissions():
    ds = xr.Dataset(
        {
            "pdf": (("time", "cells"), [[1., 0., 0., np.nan],
                                        [.2, .6, .2, np.nan],
                                        [.1, .2, .7, np.nan]]),
            "initial": ("cells", [1., 0., 0., 0.]),
            "mask": ("cells", [True, True, True, False]),
        },
        coords={
            "time": pd.date_range("2025-01-01", periods=3, freq="h"),
            "cell_ids": ("cells", np.arange(4, dtype="uint64")),
            "latitude": ("cells", [0., 1., 2., 3.]),
        },
    )
    ds.cell_ids.attrs.update(grid_name="healpix", level=0,
                             indexing_scheme="nested", ellipsoid="sphere")
    return ds


def enumerated_posterior(ds):
    """Independent exact likelihood and marginals, summing every possible path."""
    joint = np.zeros((3, 4))
    total = 0.
    for path in product(range(3), repeat=3):
        weight = float(ds.initial[path[0]])
        for t in (1, 2):
            weight *= TRANSITION[path[t], path[t-1]] * float(ds.pdf[t, path[t]])
        total += weight
        for t, cell in enumerate(path):
            joint[t, cell] += weight
    return joint / total, -np.log(total)


@pytest.mark.parametrize("kind", ["path", "string", "local", "memory"])
def test_cached_posterior_roundtrip(tmp_path, emissions, kind):
    path = tmp_path / "cache.zarr"
    cache = {"path": lambda: path, "string": lambda: str(path),
             "local": lambda: LocalStore(path), "memory": MemoryStore}[kind]()
    estimator = CachedEstimator(MatrixPredictor, sigma=1., cache=cache)
    options = dict(spatial_dims=["cells"], temporal_dims=["time"])
    # Reuse the cache after changing the observations: stale consolidated metadata
    # or stale arrays must not influence the second fit.
    for multiplier in (1., 2.):
        ds = emissions.copy(deep=True)
        ds["pdf"][1, 0] *= multiplier
        expected, score = enumerated_posterior(ds)
        result = estimator.predict_proba(ds, **options).compute()
        np.testing.assert_allclose(result.states[:, :3], expected[:, :3], atol=1e-14)
        np.testing.assert_allclose(result.states.sum("cells"), 1.)
        assert result.states[:, 3].isnull().all()
        assert result.states[0, 1] == 0.  # a real zero must not decode as NaN
        xr.testing.assert_identical(result.time, ds.time)
        xr.testing.assert_identical(result.cell_ids, ds.cell_ids)
        xr.testing.assert_identical(result.latitude, ds.latitude)
        assert estimator.score(ds, **options) == pytest.approx(score)
        root = zarr.open_group(cache, mode="r", use_consolidated=False)
        assert root.metadata.zarr_format == 2
        stored = xr.open_zarr(cache, group="emission", consolidated=False).load()
        xr.testing.assert_allclose(stored, ds.fillna(0))


@pytest.mark.parametrize("step", [1, 3, 10, 20])
def test_regions_append_and_reopen(tmp_path, emissions, step):
    ds = emissions.reindex(time=pd.date_range("2025-01-01", periods=23, freq="h"))
    ds["pdf"][4, 0] = 0.
    path = tmp_path / "regions.zarr"
    write_time_blocks(ds, path, step=step)
    xr.testing.assert_identical(xr.open_zarr(path, consolidated=False).load(), ds)
    # Exercise true append_dim as well as the tide writer's mode='a' variables.
    extra = ds.isel(time=slice(-2, None)).assign_coords(
        time=pd.date_range("2025-01-02", periods=2, freq="h")
    )
    extra.to_zarr(path, mode="a", append_dim="time", zarr_format=2, consolidated=False)
    flags = xr.DataArray(np.arange(25), dims="time", name="tide_found")
    flags.to_zarr(path, mode="a", zarr_format=2, consolidated=False)
    zarr.consolidate_metadata(path)
    actual = xr.open_zarr(path).load()
    expected = xr.concat([ds, extra], dim="time", data_vars="minimal", coords="minimal")
    xr.testing.assert_identical(actual.drop_vars("tide_found"), expected)
    np.testing.assert_array_equal(actual.tide_found, flags)
    assert zarr.open_group(path).metadata.zarr_format == 2
