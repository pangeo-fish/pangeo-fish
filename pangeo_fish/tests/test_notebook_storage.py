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

TRANSITION = np.array(
    [
        [0.75, 0.25, 0, 0],
        [0.25, 0.5, 0.25, 0],
        [0, 0.25, 0.75, 0],
        [0, 0, 0, 1],
    ]
)


def test_hand_sigma_copy_rechunked_multisigma_data(tmp_path, emissions):
    from pangeo_fish.helpers import hand_test_parameter

    source = emissions.assign(predictor_index=("time", [0, 1, 1]))
    source.to_zarr(tmp_path / "combined.zarr", mode="w", zarr_format=2)
    params = hand_test_parameter(
        target_root=str(tmp_path),
        saving_root="/manual",
        sigma_tested=[0.03],
        default_chunk_dims={"time": 1, "cells": -1},
    )
    assert params["sigma"] == [0.03]
    assert params["sigma_indices"] == [[0, 1, 2]]
    with xr.open_zarr(tmp_path / "manual/combined.zarr") as actual:
        xr.testing.assert_identical(actual.load(), source)


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
            "pdf": (
                ("time", "cells"),
                [
                    [1.0, 0.0, 0.0, np.nan],
                    [0.2, 0.6, 0.2, np.nan],
                    [0.1, 0.2, 0.7, np.nan],
                ],
            ),
            "initial": ("cells", [1.0, 0.0, 0.0, 0.0]),
            "mask": ("cells", [True, True, True, False]),
        },
        coords={
            "time": pd.date_range("2025-01-01", periods=3, freq="h"),
            "cell_ids": ("cells", np.arange(4, dtype="uint64")),
            "latitude": ("cells", [0.0, 1.0, 2.0, 3.0]),
        },
    )
    ds.cell_ids.attrs.update(
        grid_name="healpix", level=0, indexing_scheme="nested", ellipsoid="sphere"
    )
    return ds


def enumerated_posterior(ds):
    """Independent exact likelihood and marginals, summing every possible path."""
    joint = np.zeros((3, 4))
    total = 0.0
    for path in product(range(3), repeat=3):
        weight = float(ds.initial[path[0]])
        for t in (1, 2):
            weight *= TRANSITION[path[t], path[t - 1]] * float(
                ds.pdf.fillna(0)[t, path[t]]
            )
        total += weight
        for t, cell in enumerate(path):
            joint[t, cell] += weight
    return joint / total, -np.log(total)


@pytest.mark.parametrize("kind", ["path", "string", "local", "memory"])
@pytest.mark.parametrize("missing_ocean_observation", [False, True])
def test_cached_posterior_roundtrip(
    tmp_path, emissions, kind, missing_ocean_observation
):
    path = tmp_path / "cache.zarr"
    cache = {
        "path": lambda: path,
        "string": lambda: str(path),
        "local": lambda: LocalStore(path),
        "memory": MemoryStore,
    }[kind]()
    estimator = CachedEstimator(MatrixPredictor, sigma=1.0, cache=cache)
    options = dict(spatial_dims=["cells"], temporal_dims=["time"])
    if missing_ocean_observation:
        emissions["pdf"][1, 1] = np.nan
    # Reuse the cache after changing the observations: stale consolidated metadata
    # or stale arrays must not influence the second fit.
    for multiplier in (1.0, 2.0):
        ds = emissions.copy(deep=True)
        ds["pdf"][1, 0] *= multiplier
        expected, score = enumerated_posterior(ds)
        result = estimator.predict_proba(ds, **options).compute()
        np.testing.assert_allclose(result.states[:, :3], expected[:, :3], atol=1e-14)
        np.testing.assert_allclose(result.states.sum("cells"), 1.0)
        assert result.states[:, 3].isnull().all()
        assert result.states[0, 1] == 0.0  # a real zero must not decode as NaN
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
    ds["pdf"][4, 0] = 0.0
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


def test_cache_with_dimension_coordinates_only(tmp_path, emissions):
    ds = emissions.drop_vars(["cell_ids", "latitude"]).rename(cells="x")
    ds = ds.assign_coords(x=np.arange(4))
    estimator = CachedEstimator(MatrixPredictor, sigma=1.0, cache=tmp_path / "cache")
    result = estimator.predict_proba(ds, spatial_dims=["x"], temporal_dims=["time"])
    xr.testing.assert_identical(result.time, ds.time)
    xr.testing.assert_identical(result.x, ds.x)


def test_cached_healpix_convolution(tmp_path):
    from pangeo_fish.hmm.estimator import EagerEstimator
    from pangeo_fish.tests.test_estimator import predictor_factory, sample_dataset

    ds = sample_dataset.__wrapped__()
    factory = predictor_factory.__wrapped__(ds)
    eager = EagerEstimator(factory, sigma=[0.001]).predict_proba(ds)
    cached = CachedEstimator(factory, sigma=0.001, cache=tmp_path / "healpix")
    actual = cached.predict_proba(ds).compute()
    # EagerEstimator intentionally masks zeros; the cache preserves them. Compare
    # numerical probabilities on ocean cells, without changing either convention.
    np.testing.assert_allclose(actual.states, eager.fillna(0), atol=1e-7)
    np.testing.assert_allclose(actual.states.sum("cells"), 1.0, atol=1e-7)
    np.testing.assert_array_equal(actual.cell_ids, ds.cell_ids)
    xr.testing.assert_identical(actual.time, ds.time)
    assert actual.states.dggs.decode().dggs.grid_info == ds.dggs.grid_info


@pytest.mark.parametrize("step", [1, 3])
def test_tide_writer(tmp_path, monkeypatch, step):
    from pangeo_fish import tide

    times = pd.date_range("2025-01-01", periods=72, freq="h")
    values = np.arange(72 * 4, dtype=float).reshape(72, 2, 2)
    values[:, 1, 1] = np.nan
    source = xr.Dataset(
        {
            "pdf": (("time", "latitude", "longitude"), values),
            "mask": (("latitude", "longitude"), [[True, True], [True, False]]),
            "tide_found": ("time", np.arange(72) % 2),
            "behav_found": ("time", np.arange(72) % 3),
        },
        coords={"time": times, "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )

    def regrid(*, ds, **kwargs):
        # Isolate the real tide writer from geographical interpolation; the exact
        # daily mean and the missing land cell can then be checked independently.
        out = xr.Dataset(
            {
                "pdf": (("time", "cells"), ds.pdf.values.reshape(-1, 4)),
                "ocean_mask": ("cells", ds.ocean_mask.values.reshape(4)),
            },
            coords={"time": ds.time, "cell_ids": ("cells", np.arange(4))},
        )
        out.cell_ids.attrs.update(
            grid_name="healpix", level=0, indexing_scheme="nested"
        )
        return out, None

    monkeypatch.setattr(tide, "regrid_dataset_hpresample", regrid)
    ds, found, behavior = tide.tide_to_healpix(
        tmp_path, 0, "sphere", chunk_size=20, step=step, dataset=source
    )
    actual = xr.open_zarr(tmp_path / "tide_pdf_healpix.zarr").load()
    xr.testing.assert_allclose(actual.drop_vars(["tide_found", "behav_found"]), ds)
    np.testing.assert_allclose(
        actual.pdf.values, source.pdf.resample(time="1D").mean().values.reshape(3, 4)
    )
    xr.testing.assert_allclose(actual.tide_found, found)
    xr.testing.assert_allclose(actual.behav_found, behavior)
    assert actual.cell_ids.attrs["grid_name"] == "healpix"
    assert actual.cell_ids.attrs["level"] == 0
