"""Tests of the Pedersen et al. (2008) tide likelihood on HEALPix cells.

A synthetic tide (known constants, f = 1, V + u = omega t) replaces pyfes, so the
tests run without it. The checks follow the structure of the MATLAB
``datalikelihood.m``: Gaussian likelihood of a window around ``D + eta`` with a
cell-independent covariance, and a bathymetry-only likelihood on days without tide.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

from pangeo_fish import tide_healpix as th

healpix_geo = pytest.importorskip("healpix_geo")

LEVEL = 10
CONSTITUENTS = ("M2", "S2", "K1", "O1")
OMEGA_DEG_PER_HOUR = {"M2": 28.9841042, "S2": 30.0, "K1": 15.0410686, "O1": 13.9430356}
START = pd.Timestamp("2023-08-01")


def make_cells(n):
    lons = np.linspace(2.5, 4.0, n)
    lats = np.linspace(51.2, 52.2, n)
    ids = healpix_geo.nested.lonlat_to_healpix(lons, lats, LEVEL, ellipsoid="WGS84")
    ids = np.unique(ids)
    return th.make_cell_ids(ids, LEVEL, "WGS84")


@pytest.fixture
def synthetic_astronomy(monkeypatch):
    def fake(times, constituents):
        t = pd.DatetimeIndex(np.atleast_1d(times))
        hours = th._seconds_since(t, pd.Timestamp("2000-01-01")) / 3600.0
        omega = np.radians([OMEGA_DEG_PER_HOUR[c] for c in constituents])
        vu = hours[:, None] * omega[None, :]
        dims = ("time", th.CONSTITUENT_DIM)
        return xr.Dataset(
            {"f": (dims, np.ones_like(vu)), "vu": (dims, vu)},
            coords={"time": t.values, th.CONSTITUENT_DIM: list(constituents)},
        )

    monkeypatch.setattr(th, "nodal_modulations", fake)


@pytest.fixture
def setup(synthetic_astronomy):
    """Cells: 0 = truth, 1 = same amplitudes with shifted phases, others random."""
    cells = make_cells(6)
    n = cells.size
    assert n >= 4
    rng = np.random.default_rng(3)
    amp = np.tile(np.array([1.4, 0.5, 0.2, 0.15])[:, None], (1, n))
    amp = amp * rng.uniform(0.6, 1.4, amp.shape)
    pha = rng.uniform(0, 2 * np.pi, amp.shape)
    amp[:, 1], pha[:, 1] = amp[:, 0], pha[:, 0] + np.radians(40.0)
    const = th.harmonic_constants_dataset(
        amp * np.exp(1j * pha), list(CONSTITUENTS), cells
    )
    return cells, const


def record(const, cell=0, days=4, depth_m=20.0, noise=0.05, freq="10min", seed=0):
    times = pd.date_range(START, START + pd.Timedelta(days=days), freq=freq)
    eta = th.predict_tide(const, times.values).values[:, cell]
    rng = np.random.default_rng(seed)
    return pd.Series(depth_m + eta + noise * rng.standard_normal(len(times)), times)


def flags(series, window_hours=25.0, found=True):
    half = pd.Timedelta(hours=window_hours) / 2
    inside = (series.index >= series.index[0] + half) & (
        series.index <= series.index[-1] - half
    )
    return pd.Series(inside & found, index=series.index)


def run(const, cells, series, tide_found, wc=20.0, **kw):
    water = xr.DataArray(
        np.full(cells.size, wc) if np.isscalar(wc) else wc,
        dims="cells",
        coords={"cell_ids": cells},
    )
    return th.pedersen_tide_likelihood(
        series, const, water, tide_found=tide_found, **kw
    )


def test_true_cell_maximises_likelihood(setup):
    cells, const = setup
    series = record(const)
    out = run(const, cells, series, flags(series))
    found = out["tide_found"].values.astype(bool)
    assert found.sum() >= 3
    best = out["pdf"].values[found].argmax(axis=1)
    assert (best == 0).all()
    np.testing.assert_allclose(out["pdf"].sum("cells"), 1.0)
    assert out["pdf"].sel(time=out.time[found][0]).values[0] > 0.9


def test_shifted_phase_scores_lower(setup):
    cells, const = setup
    series = record(const)
    out = run(const, cells, series, flags(series))
    found = out["tide_found"].values.astype(bool)
    ll = out["log_likelihood"].values[found]
    # same amplitudes, phases shifted by 40 degrees: much less likely
    assert (ll[:, 0] - ll[:, 1] > 10).all()


def test_covariance_symmetric_positive_definite():
    rng = np.random.default_rng(1)
    hours = np.sort(rng.uniform(0, 25, 150))  # irregular sampling
    for kw in ({}, {"tidal_roughness_sd": 0.3}, {"epsilon": 0.0}):
        cov = th.pedersen_covariance(hours, **kw)
        np.testing.assert_allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > 0
        np.linalg.cholesky(cov)
    # MATLAB defaults: E = 0.2, lambda = 0.05^(1/40) per 10 min, s_eta_tid = 10^2
    cov = th.pedersen_covariance(np.array([0.0, 10 / 60.0]), epsilon=0.4)
    lam = 0.05 ** (1 / 40)
    assert cov[0, 0] == pytest.approx(0.2**2 + 0.4**2 + 100.0)
    assert cov[0, 1] == pytest.approx(0.4**2 * lam + 100.0)


def test_non_tidal_day_uses_bathymetry_only(setup):
    cells, const = setup
    n = cells.size
    series = record(const, depth_m=15.0, noise=0.0, cell=0)
    wc = np.linspace(10.0, 60.0, n)
    out = run(
        const,
        cells,
        series,
        pd.Series(False, series.index),
        wc=wc,
        bathy_variance=1.0,
    )
    assert (out["tide_found"] == 0).all()
    pdf = out["pdf"].values[1]
    depth = wc
    assert pdf[depth < 12].max() < 1e-6 * pdf.max()
    assert pdf.argmax() >= np.flatnonzero(depth > 25)[0]
    np.testing.assert_allclose(pdf.sum(), 1.0)
    # formula: normcdf(d; mu, s) / (eps + normcdf(0; mu, s)) in depth-negative-up form
    label = out.time.values[1]
    lo = pd.Timestamp(label) - pd.Timedelta("12h")
    sel = series[(series.index >= lo) & (series.index < lo + pd.Timedelta("1D"))]
    t = sel.idxmax()
    eta = th.predict_tide(const, np.array([t.to_datetime64()])).values[0]
    mu = wc + eta
    ref = norm.sf(sel.max(), mu, 1.0) / (np.finfo(float).eps + norm.sf(0, mu, 1.0))
    np.testing.assert_allclose(
        np.exp(out["log_likelihood"].values[1]), ref, rtol=1e-8, atol=1e-12
    )


def test_uniform_option_for_non_tidal_days(setup):
    cells, const = setup
    series = record(const)
    out = run(const, cells, series, pd.Series(False, series.index), non_tidal="uniform")
    np.testing.assert_allclose(out["pdf"].values, 1.0 / cells.size)


def test_no_underflow_for_long_windows(setup):
    cells, const = setup
    series = record(const, days=8, noise=0.02)
    # 96 h window: 576 samples, log-likelihoods of wrong cells hundreds below
    out = run(const, cells, series, flags(series, 96.0), window_hours=96.0, wc=20.0)
    assert out["tide_found"].sum() >= 1
    found = out["tide_found"].values.astype(bool)
    ll = out["log_likelihood"].values[found]
    assert np.isfinite(ll).all()
    assert (ll.max(axis=1) - ll.min(axis=1) > 100).all()
    pdf = out["pdf"].values[found]
    assert np.isfinite(pdf).all()
    np.testing.assert_allclose(pdf.sum(axis=1), 1.0)
    assert (pdf.argmax(axis=1) == 0).all()


def test_window_label_alignment():
    idx = pd.date_range("2023-08-01", "2023-08-06", freq="10min")
    found = pd.Series(False, idx)
    # centre at T + 11 h belongs to label T, centre at T + 13 h to label T + 1 day
    found[pd.Timestamp("2023-08-02 11:00")] = True
    found[pd.Timestamp("2023-08-03 13:00")] = True
    win = th.tide_windows(idx, found, window_hours=25.0)
    assert list(win["label"]) == [
        pd.Timestamp("2023-08-02"),
        pd.Timestamp("2023-08-04"),
    ]
    # [T - 12 h, T + 12 h): 11:59:59 -> T, 12:00 -> T + 1 day
    found[:] = False
    found[pd.Timestamp("2023-08-02 11:50")] = True
    found[pd.Timestamp("2023-08-04 12:00")] = True
    win = th.tide_windows(idx, found, window_hours=25.0)
    assert list(win["label"]) == [
        pd.Timestamp("2023-08-02"),
        pd.Timestamp("2023-08-05"),
    ]


def test_overlapping_windows_keep_best_fit():
    idx = pd.date_range("2023-08-01", "2023-08-06", freq="10min")
    found = pd.Series(False, idx)
    rmse = pd.Series(0.4, idx)
    for t, r in (("2023-08-02 00:00", 0.3), ("2023-08-02 03:00", 0.1)):
        found[pd.Timestamp(t)] = True
        rmse[pd.Timestamp(t)] = r
    win = th.tide_windows(idx, found, rmse=rmse, window_hours=25.0)
    assert len(win) == 1
    assert win["centre"][0] == pd.Timestamp("2023-08-02 03:00")
    # short windows do not overlap: both are kept
    win = th.tide_windows(idx, found, rmse=rmse, window_hours=2.0)
    assert len(win) == 2


def test_tidal_labels_follow_flags(setup):
    cells, const = setup
    series = record(const, days=5)
    tide_found = pd.Series(False, series.index)
    centre = pd.Timestamp("2023-08-03 00:00")  # label 2023-08-03
    tide_found[centre] = True
    out = run(const, cells, series, tide_found)
    times = pd.DatetimeIndex(out.time.values)
    assert out["tide_found"].values[times.get_loc(centre)] == 1
    assert out["tide_found"].sum() == 1
    assert out["pdf"].values[times.get_loc(centre)].argmax() == 0


def test_output_structure(setup):
    cells, const = setup
    series = record(const)
    out = run(const, cells, series, flags(series))
    ref = {"pdf", "log_likelihood", "tide_found", "ocean_mask"}
    assert ref <= set(out.data_vars)
    assert out["pdf"].dims == ("time", "cells")
    assert out["cell_ids"].attrs["level"] == LEVEL
    assert out["cell_ids"].attrs["ellipsoid"] == "WGS84"


def test_mean_depth_offset_is_not_decisive(setup):
    """A 3 m bias of the water column leaves the tide shape decisive (sd 10 m)."""
    cells, const = setup
    series = record(const)
    out = run(const, cells, series, flags(series), wc=23.0)
    found = out["tide_found"].values.astype(bool)
    assert (out["pdf"].values[found].argmax(axis=1) == 0).all()


def test_cell_bathymetry_mean_and_variance():
    cells = make_cells(6)
    info = th._grid_info(cells)
    lat = np.linspace(51.2, 52.2, 120)
    lon = np.linspace(2.5, 4.0, 150)
    elev = xr.DataArray(
        -(10.0 + 30.0 * (lon[None, :] - 2.5) + 0 * lat[:, None]),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    elev[:5] = 20.0  # land: not part of the water column
    out = th.cell_bathymetry(elev, cells)
    assert out["water_column"].dims == ("cells",)
    assert np.isfinite(out["water_column"]).any()
    assert (out["bathy_variance"].dropna("cells") >= 1.0).all()
    assert info["level"] == LEVEL
    ok = out["water_column"].dropna("cells")
    assert ((ok > 10) & (ok < 60)).all()
