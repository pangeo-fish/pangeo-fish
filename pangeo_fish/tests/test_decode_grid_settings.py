import healpix_geo.nested as hgn
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs  # noqa: F401
from tlz.functoolz import curry

from pangeo_fish.hmm import decode
from pangeo_fish.hmm.estimator import eager
from pangeo_fish.hmm.estimator.eager import EagerEstimator
from pangeo_fish.hmm.prediction import (
    FFTGaussian1DHealpix,
    Gaussian1DHealpix,
    LargeGaussianHealpixAnalyse,
    UpDownGaussian1DHealpix,
)

level = 6
sigma = 0.02


def create_emission(ellipsoid, as_attribute=False):
    """Three time steps on a small patch, with one certain cell per step."""
    centre = hgn.lonlat_to_healpix(np.array([-5.0]), np.array([48.5]), level)
    disk = np.asarray(hgn.kth_neighbourhood(centre, level, 3)).ravel()
    cells = np.unique(disk[disk >= 0]).astype(np.uint64)

    path = np.array([2, 3, 4])
    pdf = np.full((3, cells.size), 1e-6)
    pdf[np.arange(3), path] = 1
    initial = np.zeros(cells.size)
    initial[path[0]] = 1

    grid = {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"}
    if ellipsoid is not None and not as_attribute:
        grid["ellipsoid"] = ellipsoid

    emission = xr.Dataset(
        {
            "pdf": (("time", "cells"), pdf),
            "initial": ("cells", initial),
            "mask": ("cells", np.ones(cells.size, dtype=bool)),
            "predictor_index": ("time", np.zeros(3, dtype=np.int32)),
        },
        coords={
            "cell_ids": ("cells", cells),
            "time": pd.date_range("2025-01-01", periods=3),
        },
    ).dggs.decode(grid)
    if ellipsoid is not None and as_attribute:
        emission["cell_ids"].attrs["ellipsoid"] = ellipsoid

    return emission, cells[path]


@pytest.mark.parametrize(
    ["ellipsoid", "as_attribute", "expected"],
    (
        (None, False, "sphere"),
        ("sphere", False, "sphere"),
        ("WGS84", False, "WGS84"),
        ("WGS84", True, "WGS84"),
    ),
)
def test_viterbi_healpix_uses_the_ellipsoid_of_the_grid(
    ellipsoid, as_attribute, expected
):
    emission, cell_path = create_emission(ellipsoid, as_attribute)

    actual = decode.viterbi_healpix(emission, [sigma])

    np.testing.assert_array_equal(actual["cell_ids"], cell_path)
    lon, lat = hgn.healpix_to_lonlat(cell_path, level, ellipsoid=expected)
    np.testing.assert_allclose(actual["longitude"], lon)
    np.testing.assert_allclose(actual["latitude"], lat)


def test_viterbi_healpix_positions_differ_between_ellipsoids():
    # guards the test above: the two conventions are about 0.1 degree apart here
    _, cell_path = create_emission("WGS84")
    _, on_ellipsoid = hgn.healpix_to_lonlat(cell_path, level, ellipsoid="WGS84")
    _, on_sphere = hgn.healpix_to_lonlat(cell_path, level, ellipsoid="sphere")

    assert np.all(np.abs(on_ellipsoid - on_sphere) > 0.05)


def test_viterbi_healpix_explicit_ellipsoid_takes_precedence():
    emission, cell_path = create_emission("WGS84")

    actual = decode.viterbi_healpix(emission, [sigma], ellipsoid="sphere")

    _, lat = hgn.healpix_to_lonlat(cell_path, level, ellipsoid="sphere")
    np.testing.assert_allclose(actual["latitude"], lat)


@pytest.mark.parametrize(
    ["factory", "expected"],
    (
        (curry(FFTGaussian1DHealpix), {"truncate": 5.0}),
        (
            curry(FFTGaussian1DHealpix, truncate=3, ellipsoid="WGS84"),
            {"truncate": 3, "ellipsoid": "WGS84"},
        ),
        (curry(UpDownGaussian1DHealpix, ellipsoid="WGS84"), {"ellipsoid": "WGS84"}),
        (curry(Gaussian1DHealpix, truncate=4.0), {"truncate": 4.0}),
        (curry(LargeGaussianHealpixAnalyse), {}),
        (lambda sigma: None, {}),
    ),
)
def test_viterbi_healpix_settings(factory, expected):
    estimator = EagerEstimator(predictor_factory=factory, sigma=[sigma])

    assert estimator._viterbi_healpix_settings() == expected


def test_decode_passes_the_predictor_settings_to_viterbi_healpix(monkeypatch):
    emission, cell_path = create_emission("WGS84")
    received = {}

    def viterbi_healpix(emission, sigmas, **kwargs):
        received.update(kwargs)
        return decode.viterbi_healpix(emission, sigmas, **kwargs)

    monkeypatch.setattr(eager, "viterbi_healpix", viterbi_healpix)
    estimator = EagerEstimator(
        predictor_factory=curry(FFTGaussian1DHealpix, truncate=3, ellipsoid="WGS84"),
        sigma=[sigma],
    )

    trajectory = estimator.decode(emission, mode="viterbi_healpix")

    assert received == {"truncate": 3, "ellipsoid": "WGS84"}
    _, lat = hgn.healpix_to_lonlat(cell_path, level, ellipsoid="WGS84")
    np.testing.assert_allclose(trajectory.df.geometry.y, lat)


def test_build_transitions_on_an_ellipsoid_is_close_to_the_sphere():
    emission, _ = create_emission("WGS84")
    cells = emission["cell_ids"].values

    neighbours, weights = decode.build_transitions(
        cells, level, [sigma], truncate=4.0, ellipsoid="WGS84"
    )
    expected_neighbours, expected_weights = decode.build_transitions(
        cells, level, [sigma], truncate=4.0, ellipsoid="sphere"
    )

    # same disks; the distances differ by the flattening only (below 1 %)
    np.testing.assert_array_equal(neighbours[0], expected_neighbours[0])
    inside = np.isfinite(weights[0]) & np.isfinite(expected_weights[0])
    assert inside.any()
    actual, expected = weights[0][inside], expected_weights[0][inside]
    np.testing.assert_allclose(actual, expected, atol=0.1)
