import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs  # noqa: F401

pytest.importorskip("healpix_analyse")

from healpix_geo import nested  # noqa: E402

from pangeo_fish.helpers import (  # noqa: E402
    _build_estimator,
    _get_predictor_factory,
    _update_params_dict,
    to_healpix,
)
from pangeo_fish.hmm.prediction import (  # noqa: E402
    FFTGaussian1DHealpix,
    GaussianHealpixAnalyse,
    LargeGaussianHealpixAnalyse,
)

LEVEL = 8
R = 6371007.1809
CELL_RAD = np.sqrt(4 * np.pi / (12 * 4**LEVEL))


@pytest.fixture(scope="module")
def cells():
    centre = nested.lonlat_to_healpix(
        np.array([10.0]), np.array([45.0]), LEVEL, ellipsoid="WGS84"
    )
    ring = np.asarray(
        nested.kth_neighbourhood(np.asarray(centre, dtype=np.uint64), LEVEL, 20)
    )[0]
    return np.unique(ring[ring >= 0].astype(np.uint64))


def _grid():
    return SimpleNamespace(level=LEVEL)


def _dirac(cells, i=None):
    x = np.zeros(cells.size)
    x[cells.size // 2 if i is None else i] = 1.0
    return x


def _blocked_half(cells):
    """Ocean = cells west of the centre column, by index in the sorted ids."""
    lon, _ = nested.healpix_to_lonlat(cells, LEVEL, ellipsoid="WGS84")
    return lon < np.median(lon)


def test_default_options_are_unchanged(cells):
    default = GaussianHealpixAnalyse(cell_ids=cells, grid_info=_grid(), sigma=CELL_RAD)
    explicit = GaussianHealpixAnalyse(
        cell_ids=cells,
        grid_info=_grid(),
        sigma=CELL_RAD,
        normalization="full_disc",
        land_barrier=False,
    )
    x = np.random.default_rng(0).random(cells.size)
    np.testing.assert_array_equal(default.predict(x), explicit.predict(x))


def test_source_normalization_keeps_mass_at_the_edge(cells):
    kw = dict(cell_ids=cells, grid_info=_grid(), sigma=2 * CELL_RAD)
    edge = _dirac(cells, 0)
    assert GaussianHealpixAnalyse(**kw).predict(edge).sum() < 0.999
    source = GaussianHealpixAnalyse(**kw, normalization="source")
    assert source.predict(edge).sum() == pytest.approx(1.0, abs=1e-6)


def test_unknown_normalization_is_refused(cells):
    with pytest.raises(ValueError, match="normalization"):
        GaussianHealpixAnalyse(
            cell_ids=cells, grid_info=_grid(), sigma=CELL_RAD, normalization="x"
        )


def test_land_barrier_blocks_mass_and_honours_mask(cells):
    ocean = _blocked_half(cells)
    blocked = cells[~ocean]
    kw = dict(
        cell_ids=cells, grid_info=_grid(), sigma=2 * CELL_RAD, normalization="source"
    )
    barrier = GaussianHealpixAnalyse(**kw, land_barrier=True, blocked_cells=blocked)
    # a Dirac at the ocean cell closest to the coast
    idx = np.flatnonzero(ocean)
    src = idx[np.argmax(np.asarray(cells)[idx] * 0 + np.arange(idx.size))]
    out = barrier.predict(_dirac(cells, src), mask=ocean)
    assert out[~ocean].sum() == 0.0
    plain = GaussianHealpixAnalyse(**kw)
    assert plain.predict(_dirac(cells, src))[~ocean].sum() >= out[~ocean].sum()
    # mass stays in the ocean
    assert out.sum() == pytest.approx(1.0, abs=1e-6)


def test_land_barrier_needs_blocked_cells(cells):
    with pytest.raises(ValueError, match="blocked_cells"):
        GaussianHealpixAnalyse(
            cell_ids=cells, grid_info=_grid(), sigma=CELL_RAD, land_barrier=True
        )


def test_large_conv_mass_mask_and_minimum_sigma(cells):
    sigma = 6 * CELL_RAD
    predictor = LargeGaussianHealpixAnalyse(
        cell_ids=cells, grid_info=_grid(), sigma=sigma
    )
    out = predictor.predict(_dirac(cells))
    assert 0.9 < out.sum() < 1.05
    mask = _blocked_half(cells)
    np.testing.assert_array_equal(
        predictor.predict(_dirac(cells), mask=mask), np.where(mask, out, 0)
    )
    with pytest.raises(ValueError, match="HealpixAnalyseConv"):
        LargeGaussianHealpixAnalyse(
            cell_ids=cells, grid_info=_grid(), sigma=0.5 * CELL_RAD
        )


_FFT_MASS = {}


@pytest.mark.parametrize("projection", ["average", "conservative"])
def test_fft_projection(cells, projection):
    predictor = FFTGaussian1DHealpix(
        cell_ids=cells,
        grid_info=_grid(),
        sigma=3 * CELL_RAD,
        ellipsoid="WGS84",
        projection=projection,
    )
    out = predictor.predict(_dirac(cells))
    assert out.sum() == pytest.approx(1.0, abs=0.1)
    _FFT_MASS[projection] = out.sum()
    if len(_FFT_MASS) == 2:
        # the conservative pair keeps at least as much mass as the average one
        assert _FFT_MASS["conservative"] >= _FFT_MASS["average"] - 1e-9


def test_fft_unknown_projection(cells):
    with pytest.raises(ValueError, match="projection"):
        FFTGaussian1DHealpix(
            cell_ids=cells, grid_info=_grid(), sigma=3 * CELL_RAD, projection="x"
        )


@pytest.fixture
def emission(cells):
    n = cells.size
    ocean = _blocked_half(cells)
    ds = xr.Dataset(
        coords={
            "cell_ids": ("cells", cells.astype("int64")),
            "time": ("time", pd.date_range("2022-06-13", periods=3)),
        },
        data_vars={
            "pdf": (("time", "cells"), np.full((3, n), 1.0 / n)),
            "mask": ("cells", ocean),
            "predictor_index": ("time", np.zeros(3, dtype=np.int32)),
        },
    )
    ds["cell_ids"].attrs.update(
        {"level": LEVEL, "indexing_scheme": "nested", "ellipsoid": "WGS84"}
    )
    return ds


@pytest.mark.filterwarnings("ignore:A .predictor_index. entry:RuntimeWarning")
def test_helper_builds_options_and_json_round_trip(emission, tmp_path):
    factory = _get_predictor_factory(
        to_healpix(emission),
        truncate=4.0,
        dims=["cells"],
        conv_method="HealpixAnalyseConv",
        normalization="source",
        land_barrier=True,
    )
    predictor = factory(sigma=2 * CELL_RAD)
    assert predictor.normalization == "source"
    assert predictor.land_barrier
    np.testing.assert_array_equal(
        np.sort(predictor.blocked_cells),
        np.sort(emission["cell_ids"].values[~emission["mask"].values]),
    )

    params = _update_params_dict(factory, {"sigma": [2 * CELL_RAD]})
    params = json.loads(json.dumps(params))
    assert params["predictor_factory"]["kwargs"]["land_barrier"] == "True"
    assert "blocked_cells" not in params["predictor_factory"]["kwargs"]
    (tmp_path / "parameters.json").write_text(json.dumps(params))

    estimator, _, _ = _build_estimator(emission, str(tmp_path), None)
    restored = estimator.predictor_factory(sigma=2 * CELL_RAD)
    assert isinstance(restored, GaussianHealpixAnalyse)
    assert restored.normalization == "source"
    assert restored.land_barrier


@pytest.mark.filterwarnings("ignore:A .predictor_index. entry:RuntimeWarning")
def test_json_round_trip_fft_projection_and_large(emission, tmp_path):
    for method, cls_name, kw in [
        ("FFTConv", "FFTGaussian1DHealpix", {"projection": "conservative", "max_sigma": 0.1}),
        ("HealpixAnalyseLargeConv", "LargeGaussianHealpixAnalyse", {}),
    ]:
        factory = _get_predictor_factory(
            to_healpix(emission),
            truncate=5.0,
            dims=["cells"],
            conv_method=method,
            **kw,
        )
        params = json.loads(
            json.dumps(_update_params_dict(factory, {"sigma": [6 * CELL_RAD]}))
        )
        assert cls_name in params["predictor_factory"]["class"]
        (tmp_path / "parameters.json").write_text(json.dumps(params))
        estimator, _, _ = _build_estimator(emission, str(tmp_path), None)
        restored = estimator.predictor_factory(sigma=6 * CELL_RAD)
        assert type(restored).__name__ == cls_name
        if method == "FFTConv":
            assert restored.projection == "conservative"
