from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("healpix_analyse")

from healpix_geo import nested  # noqa: E402

from pangeo_fish.hmm.prediction import GaussianHealpixAnalyse  # noqa: E402

LEVEL = 8
R = 6371007.1809
CELL_M = np.sqrt(4 * np.pi * R**2 / (12 * 4**LEVEL))


def _cells(k: int = 12) -> np.ndarray:
    centre = nested.lonlat_to_healpix(
        np.array([10.0]), np.array([45.0]), LEVEL, ellipsoid="WGS84"
    )
    ring = np.asarray(
        nested.kth_neighbourhood(np.asarray(centre, dtype=np.uint64), LEVEL, k)
    )[0]
    return np.unique(ring[ring >= 0].astype(np.uint64))


@pytest.fixture(scope="module")
def cells():
    return _cells()


def _predictor(cells, sigma_cells, **kwargs):
    return GaussianHealpixAnalyse(
        cell_ids=cells,
        grid_info=SimpleNamespace(level=LEVEL),
        sigma=sigma_cells * CELL_M / R,
        **kwargs,
    )


@pytest.mark.parametrize("sigma_cells", [0.3, 1.0])
def test_full_disc_keeps_mass_inside_and_loses_it_at_the_edge(cells, sigma_cells):
    predictor = _predictor(cells, sigma_cells)
    centre = np.zeros(cells.size)
    centre[cells.size // 2] = 1.0
    assert predictor.predict(centre).sum() == pytest.approx(1.0, abs=1e-3)

    edge = np.zeros(cells.size)
    edge[0] = 1.0
    assert predictor.predict(edge).sum() < 0.999 or sigma_cells < 0.5


def test_predict_applies_the_mask(cells):
    predictor = _predictor(cells, 1.0)
    x = np.random.default_rng(0).random(cells.size)
    mask = np.arange(cells.size) % 2 == 0
    filtered = predictor.predict(x)
    np.testing.assert_array_equal(
        predictor.predict(x, mask=mask), np.where(mask, filtered, 0)
    )


def test_predict_accepts_dask_arrays(cells):
    da = pytest.importorskip("dask.array")
    predictor = _predictor(cells, 1.0)
    x = np.random.default_rng(1).random(cells.size)
    np.testing.assert_allclose(
        predictor.predict(da.from_array(x)), predictor.predict(x)
    )


def test_non_wgs84_ellipsoid_is_rejected(cells):
    with pytest.raises(NotImplementedError):
        GaussianHealpixAnalyse(
            cell_ids=cells,
            grid_info=SimpleNamespace(level=LEVEL, ellipsoid="sphere"),
            sigma=1e-4,
        )


def test_time_axis_is_kept(cells):
    predictor = _predictor(cells, 1.0)
    x = np.random.default_rng(2).random((3, cells.size))
    out = predictor.predict(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out[1], predictor.predict(x[1]))


def test_operator_is_built_once(cells):
    predictor = _predictor(cells, 1.0)
    first = predictor.operator
    predictor.predict(np.ones(cells.size))
    predictor.predict(np.ones(cells.size))
    assert predictor.operator is first
    assert predictor.operator.n_columns == cells.size
