import numpy as np
import pytest
import xarray as xr
from xdggs.healpix import HealpixInfo

from pangeo_fish.hmm.prediction import FFTGaussian1DHealpix

pytest.importorskip("healpix_analyse")


def _patch(level):
    """A compact patch of 1024 cells: the descendants of a 4x4 block of cells
    three levels coarser (one cell two levels up in the nested scheme)."""
    parents = 16 * 100 + np.arange(16)
    factor = 4**3
    ids = (parents[:, None] * factor + np.arange(factor)).ravel()
    return xr.DataArray(ids, dims="cells", attrs={"ellipsoid": "sphere"})


def _predictor(level, sigma, **kwargs):
    return FFTGaussian1DHealpix(
        cell_ids=_patch(level),
        grid_info=HealpixInfo(level=level, indexing_scheme="nested"),
        sigma=sigma,
        ellipsoid="sphere",
        **kwargs,
    )


@pytest.mark.parametrize("truncate", [3.0, 5.0])
def test_kernel_radius_follows_truncate(truncate):
    predictor = _predictor(8, 0.02, truncate=truncate)
    radius = int(np.ceil(truncate * predictor.sigma_px))
    assert predictor.kernel_size == 2 * radius + 1


def test_default_truncate_is_five_sigma():
    predictor = _predictor(8, 0.02)
    assert predictor.kernel_size == 2 * int(np.ceil(5 * predictor.sigma_px)) + 1


@pytest.mark.parametrize(
    "level, truncate, expected",
    [(10, 5.0, 97), (11, 5.0, 191), (12, 5.0, 379), (10, 3.0, 59)],
)
def test_kernel_cap_follows_max_sigma_and_level(level, truncate, expected):
    # max_sigma = 0.0094 rad: 60 km per day, the demo's maximum speed
    predictor = _predictor(level, 1e-3, truncate=truncate, max_sigma=0.0094)
    assert predictor.max_kernel_size == expected


def test_no_sigma_within_the_bounds_is_capped():
    predictor = _predictor(8, 0.05, max_sigma=0.05)
    assert predictor.kernel_size == 2 * int(np.ceil(5 * predictor.sigma_px)) + 1


def test_sigma_above_max_sigma_is_capped_with_a_warning():
    with pytest.warns(RuntimeWarning, match="kernel_size truncated"):
        predictor = _predictor(8, 0.05, max_sigma=0.01)
    assert predictor.kernel_size == predictor.max_kernel_size


def test_cap_without_max_sigma_keeps_the_former_level_10_cap():
    predictor = _predictor(10, 1e-3, truncate=3.0)
    assert predictor.max_kernel_size == 129


def test_predictor_factory_passes_max_sigma_and_truncate():
    import xdggs

    from pangeo_fish import helpers

    ids = _patch(8)
    ds = xr.Dataset(
        coords={
            "cell_ids": (
                "cells",
                ids.data,
                {
                    "grid_name": "healpix",
                    "level": 8,
                    "indexing_scheme": "nested",
                    "ellipsoid": "sphere",
                },
            )
        }
    ).pipe(xdggs.decode)
    factory = helpers._get_predictor_factory(
        ds, truncate=4.0, dims=["cells"], conv_method="FFTConv", max_sigma=0.02
    )
    assert factory.keywords["max_sigma"] == 0.02
    assert factory.keywords["truncate"] == 4.0


def test_explicit_kernel_cap_is_respected():
    with pytest.warns(RuntimeWarning, match="kernel_size truncated"):
        predictor = _predictor(8, 0.05, max_kernel_size=9)
    assert predictor.kernel_size == 9


def test_prediction_is_continuous_across_a_radius_switch():
    """Gaining one pixel of kernel radius must not change the prediction visibly."""
    level = 8
    pixel = np.sqrt(np.pi / 3) / 2**level
    switch = 6 * pixel / 5  # ceil(5 * sigma_px) goes from 6 to 7 here
    below = _predictor(level, switch * (1 - 1e-9))
    above = _predictor(level, switch * (1 + 1e-9))
    assert above.kernel_size == below.kernel_size + 2

    x = np.zeros(below.cell_ids.size)
    x[x.size // 2] = 1.0
    np.testing.assert_allclose(above.predict(x), below.predict(x), atol=1e-5)


def test_explicit_kernel_size_is_used():
    predictor = _predictor(8, 0.02, kernel_size=11)
    assert predictor.kernel_size == 11


def test_numpy_cell_ids_without_ellipsoid():
    predictor = FFTGaussian1DHealpix(
        cell_ids=np.asarray(_patch(8).data),
        grid_info=HealpixInfo(level=8, indexing_scheme="nested"),
        sigma=0.02,
    )
    assert predictor.ellipsoid == "sphere"
