"""Unit tests for pangeo_fish.light."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.light.lunar import correlation_to_likelihood
from pangeo_fish.light.physics import moon_ground_illuminance
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
# physics.moon_ground_illuminance
# ---------------------------------------------------------------------------


def test_moon_illuminance_full_moon_zenith():
    """Full moon at zenith at mean distance should be ~0.25–0.3 lux."""
    lux = moon_ground_illuminance(100, 90, 384400)
    assert 0.2 <= lux <= 0.4, f"Expected ~0.25–0.3 lux, got {lux:.4f}"


def test_moon_illuminance_below_horizon():
    """Moon below the horizon → 0.0 lux."""
    assert moon_ground_illuminance(100, -5, 384400) == 0.0
    assert moon_ground_illuminance(100, 0, 384400) == 0.0


# ---------------------------------------------------------------------------
# lunar.correlation_to_likelihood
# ---------------------------------------------------------------------------


def test_correlation_to_likelihood_normalization():
    """After transformation: max == 1, all values > 0 where corr_map is finite."""
    rng = np.random.default_rng(42)
    # Synthetic correlation map with r in [-0.5, 0.8]
    corr_map = rng.uniform(-0.5, 0.8, size=(10, 10))

    moon_nights = [{"corr_map": corr_map}]
    result = correlation_to_likelihood(moon_nights, sigma_r=0.25)

    lh = result[0]["lh_map"]
    assert lh is not None
    assert np.nanmax(lh) == pytest.approx(1.0, abs=1e-9)
    assert np.all(lh > 0), "All likelihood values should be strictly positive"


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
