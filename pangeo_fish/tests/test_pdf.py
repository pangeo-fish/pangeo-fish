import numpy as np
import scipy.stats
import xarray as xr

from pangeo_fish.pdf import combine_emission_pdf, normal


def test_single_emission_with_final():
    """
    One emission variable ("em") with a "final"
    """
    times = np.arange(3)
    latitudes = np.array([10, 20])
    longitudes = np.array([100])

    em = xr.DataArray(
        np.full((3, len(latitudes), len(longitudes)), 2.0),
        dims=("time", "latitude", "longitude"),
        name="normalised_pdf",
    )

    # "Final"
    final = xr.DataArray(
        np.full((len(latitudes), len(longitudes)), 3.0),
        dims=("latitude", "longitude"),
        name="final",
    )

    initial = xr.DataArray(np.array([1]), dims=("dummy",), name="initial")
    mask = xr.DataArray(np.array([0]), dims=("dummy",), name="mask")

    # Create the dataset
    ds = xr.Dataset(
        {"em": em, "final": final, "initial": initial, "mask": mask},
        coords={"latitude": latitudes, "longitude": longitudes, "time": times},
    )

    result = combine_emission_pdf(ds)

    # Time 0 and 1: sum is 4 → pdf = 2/4 = 0.5
    # Time 2: sum is 12 → pdf = 6/12 = 0.5
    expected = ds.assign(pdf_normalized=xr.full_like(em, fill_value=0.5)).drop_vars(
        "em"
    )
    xr.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)


def test_single_emission_without_final():
    """
    emission variable without final
    """
    times = np.arange(2)
    latitudes = np.array([0, 1])
    longitudes = np.array([100])

    em = xr.DataArray(
        np.array([[[3], [1]], [[2], [2]]]),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        name="em",
    )
    initial = xr.DataArray(np.array([5]), dims=("dummy",), name="initial")

    ds = xr.Dataset({"em": em, "initial": initial})
    result = combine_emission_pdf(ds)

    # Normalized pdf calculation:
    # Time 0: em = [[3], [1]] → sum = 3+1 = 4 → pdf = [[3/4], [1/4]] = [[0.75], [0.25]]
    # Time 1: em = [[2], [2]] → sum = 2+2 = 4 → pdf = [[0.5], [0.5]]
    expected = ds.assign(
        pdf_normalized=em.copy(data=np.array([[[0.75], [0.25]], [[0.5], [0.5]]]))
    ).drop_vars("em")
    xr.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)


def test_normal_with_samples_split_along_cells():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, 10)).astype("float32")
    values[1, 2] = np.nan
    samples = xr.DataArray(values, dims=("time", "cells"), attrs={"units": "K"})

    actual = normal(
        samples.chunk({"time": 1, "cells": 4}), mean=0, std=0.5, dims=["cells"]
    )

    assert actual.name == "pdf"
    assert actual.dims == ("time", "cells")
    assert actual.attrs == {}
    assert actual.chunks == ((1, 1, 1), (4, 4, 2))
    computed = actual.compute()
    assert computed.dtype == np.float64
    np.testing.assert_array_equal(computed.values, scipy.stats.norm.pdf(values, 0, 0.5))


def test_normal_moves_dims_last():
    samples = xr.DataArray(np.zeros((2, 3, 4)), dims=("x", "time", "y"))
    actual = normal(samples, mean=0, std=1.0, dims=["y", "x"])
    assert actual.dims == ("time", "y", "x")


def test_normal_with_array_parameters():
    samples = xr.DataArray(np.zeros((2, 3)), dims=("time", "cells"))
    std = xr.DataArray([1.0, 2.0], dims="time")
    actual = normal(samples, mean=0, std=std, dims=["cells"])
    np.testing.assert_allclose(
        actual.values, scipy.stats.norm.pdf(0, 0, std.values)[:, None].repeat(3, 1)
    )
