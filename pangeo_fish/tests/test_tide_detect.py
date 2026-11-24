"""Tests of the tide detector (``pangeo_fish.tide.detect_tide``) on synthetic pressure.

The previous detector (10 h window, smoothed residuals over raw variance, no trend
removal, flag at the window start) flagged a linear ascent 100 % of the time and a
3 m diel vertical migration most of the time.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish import tide

DT_MIN = 10
HOURS = np.arange(0, 8 * 24 * 60, DT_MIN) / 60.0
SEEDS = range(4)


def make_df(y):
    time = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(HOURS.size) * DT_MIN, "min")
    return pd.DataFrame({"time": time, "pressure": y})


def run(y, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return tide.tide_behav_extr(make_df(y), "synthetic", dt=DT_MIN, **kwargs)


def valid_fraction(ds, var="tide_found"):
    """Fraction of flagged windows among windows that have a statistic."""
    valid = np.isfinite(ds.rsq_tide.values)
    return float(ds[var].values[valid].mean())


def tidal(rng, amp=1.0):
    return (
        20
        + amp * np.sin(2 * np.pi * HOURS / 12.42 + rng.uniform(0, 6))
        + 0.3 * amp * np.sin(2 * np.pi * HOURS / 12.0 + rng.uniform(0, 6))
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_still_fish_with_tide_is_flagged(seed):
    rng = np.random.default_rng(seed)
    ds = run(tidal(rng) + rng.normal(0, 0.15, HOURS.size))
    assert valid_fraction(ds) > 0.95


def test_still_fish_with_tide_and_drift_is_flagged():
    rng = np.random.default_rng(0)
    y = tidal(rng) + 0.02 * (HOURS - 96) + rng.normal(0, 0.15, HOURS.size)
    assert valid_fraction(run(y)) > 0.95


@pytest.mark.parametrize("seed", SEEDS)
def test_linear_ascent_is_not_tide(seed):
    rng = np.random.default_rng(seed)
    y = 80 - 0.3 * HOURS + rng.normal(0, 0.15, HOURS.size)
    assert valid_fraction(run(y)) == 0


@pytest.mark.parametrize("shape", ["sine", "square"])
@pytest.mark.parametrize("seed", SEEDS)
def test_diel_migration_is_not_tide(seed, shape):
    rng = np.random.default_rng(seed)
    wave = np.sin(2 * np.pi * HOURS / 24 + rng.uniform(0, 6))
    if shape == "square":
        wave = np.tanh(3 * wave)
    y = 20 + 3 * wave + rng.normal(0, 0.15, HOURS.size)
    assert valid_fraction(run(y)) < 0.02


@pytest.mark.parametrize("seed", SEEDS)
def test_pure_noise_is_not_tide(seed):
    rng = np.random.default_rng(seed)
    ds = run(20 + rng.normal(0, 0.15, HOURS.size))
    assert valid_fraction(ds) == 0


def test_implausible_amplitude_is_not_tide():
    y = 20 + 8 * np.sin(2 * np.pi * HOURS / 12.42)
    assert valid_fraction(run(y)) == 0


def test_flag_timing_is_centred():
    # tide only between hours 72 and 120, noisy steady depth elsewhere
    rng = np.random.default_rng(1)
    y = 20 + rng.normal(0, 0.15, HOURS.size)
    on = (HOURS >= 72) & (HOURS < 120)
    y[on] += np.sin(2 * np.pi * HOURS[on] / 12.42)
    ds = run(y)
    flagged = HOURS[ds.tide_found.values > 0]
    assert flagged.size > 0
    # windows are centred, so the flagged span sits inside the tidal episode
    # (up to half a window of partial overlap) and is not shifted early
    assert 72 <= flagged.min() <= 72 + 13
    assert 120 - 13 <= flagged.max() <= 120 + 1
    assert abs(flagged.mean() - 96) < 3
    # no value in the first and last half windows
    half_h = 25 / 2
    assert ds.tide_found.values[HOURS < half_h - 0.2].sum() == 0
    assert np.isnan(ds.rsq_tide.values[HOURS < half_h - 0.2]).all()
    assert np.isnan(ds.rsq_tide.values[HOURS > HOURS[-1] - half_h + 0.2]).all()
    assert np.isfinite(ds.rsq_tide.values[HOURS == 96]).all()


def test_window_shorter_than_m2_is_rejected():
    y = 20 + np.zeros(HOURS.size)
    with pytest.raises(ValueError, match="M2"):
        run(y, tideFL=10)


def test_detect_tide_defaults_and_return_type():
    rng = np.random.default_rng(0)
    time = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(HOURS.size) * DT_MIN, "min")
    tag_log = xr.Dataset({"pressure": ("time", tidal(rng) + rng.normal(0, 0.15, time.size))},
                         coords={"time": time})
    with contextlib.redirect_stdout(io.StringIO()):
        ds = tide.detect_tide(tag_log, "tag")
    assert isinstance(ds, xr.Dataset)
    assert {"tide_found", "behav_found", "rsq_tide"} <= set(ds.data_vars)
    assert ds.attrs["tideFL"] == tide.DEFAULT_TIDE_WINDOW_H >= 12.42
    assert valid_fraction(ds) > 0.95
