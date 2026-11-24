import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs  # noqa: F401
from healpix_geo import nested
from pint_xarray import unit_registry as ureg

from pangeo_fish.helpers import optimize_pdf, stamp_parameter_indices_from_mask

# the optimiser may try sigmas smaller than a cell near the lower bound
pytestmark = pytest.mark.filterwarnings("ignore:sigma .* is below half a pixel")

LEVEL = 8
N_STEPS = 4


@pytest.fixture(scope="module")
def cells():
    centre = nested.lonlat_to_healpix(
        np.array([-5.0]), np.array([48.0]), LEVEL, ellipsoid="WGS84"
    )
    ring = np.asarray(
        nested.kth_neighbourhood(np.asarray(centre, dtype=np.uint64), LEVEL, 12)
    )[0]
    return np.unique(ring[ring >= 0].astype(np.uint64))


@pytest.fixture
def emission(cells):
    """A small emission dataset whose likelihood peak drifts east."""
    lon, lat = nested.healpix_to_lonlat(cells, LEVEL, ellipsoid="WGS84")
    lon0, lat0 = np.mean(lon), np.mean(lat)
    pdf = []
    for step in range(N_STEPS):
        d2 = (lon - (lon0 - 0.45 + 0.3 * step)) ** 2 + (lat - lat0) ** 2
        p = np.exp(-d2 / (2 * 0.5**2))
        pdf.append(p / p.sum())
    pdf = np.asarray(pdf)

    ds = xr.Dataset(
        coords={
            "cell_ids": ("cells", cells.astype("int64")),
            "time": ("time", pd.date_range("2022-06-13", periods=N_STEPS)),
        },
        data_vars={
            "pdf": (("time", "cells"), pdf),
            "initial": ("cells", pdf[0]),
            "mask": ("cells", np.ones(cells.size, dtype=bool)),
        },
    )
    ds["cell_ids"].attrs.update(
        {
            "grid_name": "healpix",
            "level": LEVEL,
            "indexing_scheme": "nested",
            "ellipsoid": "WGS84",
        }
    )
    return ds.assign_coords(
        latitude=("cells", lat, {"standard_name": "latitude"}),
        longitude=("cells", lon, {"standard_name": "longitude"}),
    )


def _optimize(ds, **kwargs):
    return optimize_pdf(
        ds=ds,
        adjustment_factor=5,
        truncate=4.0,
        maximum_speed=ureg.Quantity(60, "km / day"),
        tolerance=1e-3,
        conv_method="FFTConv",
        **kwargs,
    )


MASK = np.array([0, 0, 1, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"single_behav": True},
        {"single_behav": False},
        {"multivariate": True},
    ],
)
def test_multi_sigma_without_sigma_bounds(emission, kwargs):
    """Without ``sigma_bounds`` every behaviour uses the ``maximum_speed`` bound."""
    ds = emission.assign(predictor_index=("time", MASK.astype(np.int32)))
    assert "sigma_bounds" not in ds.attrs

    params, _ = _optimize(ds, **kwargs)

    sigmas = np.atleast_1d(params["sigma"])
    assert sigmas.size >= 2
    assert np.all(np.isfinite(sigmas))
    # 60 km/day over 1 day, times the adjustment factor 5, on the WGS84 semi-major axis
    upper = 5 * 60.0 / 6378.137
    assert np.all((sigmas >= 1e-4 * (1 - 1e-6)) & (sigmas <= upper * (1 + 1e-6)))


@pytest.mark.parametrize("as_strings", [False, True])
def test_sigma_bounds_keys_after_json_round_trip(emission, monkeypatch, as_strings):
    """``sigma_bounds`` keys read back from JSON (strings) still give the
    per-behaviour bounds to the optimiser, keyed by the integer predictor value."""
    from pangeo_fish import helpers

    captured = {}

    class Stop(Exception):
        pass

    def fake_search(estimator, param_bounds, optimizer_kwargs=None, sigma_bounds=None):
        captured["sigma_bounds"] = sigma_bounds
        raise Stop

    monkeypatch.setattr(helpers, "EagerBoundsSearch", fake_search)

    bounds = {
        0: (0, ureg.Quantity(60, "km / day")),
        1: (0, ureg.Quantity(2, "km / day")),
    }
    if as_strings:
        bounds = {str(k): v for k, v in bounds.items()}
    ds = stamp_parameter_indices_from_mask(
        pdf=emission, mask=MASK, sigma_bounds_dict=bounds
    )

    with pytest.raises(Stop):
        _optimize(ds, single_behav=True)

    sigma_bounds = captured["sigma_bounds"]
    assert set(sigma_bounds) == {0, 1}
    # speed over 1 day, times the adjustment factor 5, on the WGS84 semi-major axis
    for key, speed in [(0, 60.0), (1, 2.0)]:
        lower, upper = sigma_bounds[key]
        assert lower == 1e-4
        np.testing.assert_allclose(upper, 5 * speed / 6378.137, rtol=1e-6)


def test_stamp_parameter_indices_from_mask_does_not_modify_input(emission):
    before = dict(emission.attrs)
    out = stamp_parameter_indices_from_mask(
        pdf=emission, mask=MASK, sigma_bounds_dict={0: (0, 1), 1: (0, 2)}
    )
    assert emission.attrs == before
    assert out.attrs["sigma_bounds"] == {0: (0, 1), 1: (0, 2)}
    np.testing.assert_array_equal(out["predictor_index"].values, MASK)
