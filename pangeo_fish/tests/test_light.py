"""Unit tests for pangeo_fish.light."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.light.quality import dynamic_threshold
from pangeo_fish.light.solar import compute_solar_likelihood

# ---------------------------------------------------------------------------
# quality.dynamic_threshold
# ---------------------------------------------------------------------------


def test_dynamic_threshold_new_moon():
    assert dynamic_threshold(0) == pytest.approx(60.0)


def test_dynamic_threshold_full_moon():
    assert dynamic_threshold(100) == pytest.approx(80.0)


def test_dynamic_threshold_half_moon():
    assert dynamic_threshold(50) == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# solar.compute_solar_likelihood — output dimensions
# ---------------------------------------------------------------------------


def _make_minimal_pairs_and_qdf(n_nights):
    """Synthetic pairs and quality DataFrame for testing."""
    t0 = pd.Timestamp("2020-06-01 04:00:00", tz="UTC")
    pairs = []
    for i in range(n_nights):
        t_rise = t0 + pd.Timedelta(days=i)
        t_set = t_rise + pd.Timedelta(hours=15)
        pairs.append((t_rise, t_set))

    qdf = pd.DataFrame(
        {
            "night": list(range(n_nights)),
            "date": [p[0].date() for p in pairs],
            "flag": ["bad"] * n_nights,  # uniform maps → no ephem calls
            "valid_rise": [False] * n_nights,
            "valid_set": [False] * n_nights,
            "dl_rise": [0.0] * n_nights,
            "dl_set": [0.0] * n_nights,
            "dp_rise": [9999.0] * n_nights,
            "dp_set": [9999.0] * n_nights,
        }
    )
    return pairs, qdf


def test_solar_likelihood_dims():
    """compute_solar_likelihood returns DataArray with correct dims and shape."""
    n_nights = 5
    lons = np.arange(-10, 11, 5.0)  # 5 values
    lats = np.arange(30, 51, 5.0)  # 5 values

    pairs, qdf = _make_minimal_pairs_and_qdf(n_nights)
    da = compute_solar_likelihood(pairs, qdf, lons=lons, lats=lats, thresh_deg=-11.0)

    assert isinstance(da, xr.DataArray)
    assert set(da.dims) == {"time", "latitude", "longitude"}
    assert da.sizes["time"] == n_nights
    assert da.sizes["latitude"] == len(lats)
    assert da.sizes["longitude"] == len(lons)
    assert da.attrs["method"] == "solar_threshold"
