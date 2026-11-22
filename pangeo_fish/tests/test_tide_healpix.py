"""Tests of pangeo_fish.tide_healpix.

Tests marked "pyfes" need pyfes (independent reference for the astronomy); the
others replace the astronomy by a synthetic one and run without it.

Old ``tide.py`` logic (main 2cefe3c) that these tests would reject:
- ``test_predict_matches_pyfes``, ``test_nodal_factors_match_pugh``: the old
  likelihood sums the equilibrium amplitude with the 1992 astronomical argument,
  ``omega`` in rad/s times hours since the record start, no nodal corrections and
  no local constants (``hRe``, ``hIm`` are not read), so its tide is the same at
  every cell and not the tide of any cell.
- ``test_prediction_time_reference``, ``test_likelihood_time_reference``: the old
  time axis is hours since the first hour of the record, so dropping or adding
  data before a day changes the tide predicted for that day.
- ``test_amphidrome_complex_resampling``: the old pipeline never resamples
  constants; it regrids the hourly pdf. Interpolating amplitude and phase (the
  natural alternative) fails this test, see ``test_amplitude_phase_interpolation_fails``.
- ``test_sphere_vs_wgs84_cells``: the old regrid used ``ellipsoid="sphere"`` cells
  with geodetic source latitudes (0.125 degrees, ~14 km, offset at 51 N).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish import tide_healpix as th

healpix_geo = pytest.importorskip("healpix_geo")

LEVEL = 10


def region_cells(level=LEVEL, ellipsoid="WGS84", lon=(2.5, 4.0), lat=(51.2, 52.2)):
    lons, lats = np.meshgrid(np.linspace(*lon, 60), np.linspace(*lat, 60))
    ids = healpix_geo.nested.lonlat_to_healpix(
        lons.ravel(), lats.ravel(), level, ellipsoid=ellipsoid
    )
    return th.make_cell_ids(np.unique(ids), level, ellipsoid)


OMEGA_DEG_PER_HOUR = {"M2": 28.9841042, "S2": 30.0, "K1": 15.0410686, "O1": 13.9430356}


@pytest.fixture
def synthetic_astronomy(monkeypatch):
    """f = 1 and vu = omega (t - 2000-01-01): no pyfes needed."""

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
    return fake


def random_constants(cell_ids, constituents=("M2", "S2", "K1", "O1"), seed=0):
    rng = np.random.default_rng(seed)
    n = cell_ids.size
    amp = rng.uniform(0.05, 2.0, (len(constituents), n))
    pha = rng.uniform(0, 2 * np.pi, (len(constituents), n))
    return th.harmonic_constants_dataset(
        amp * np.exp(1j * pha), list(constituents), cell_ids
    )


# --------------------------------------------------------------------------
# prediction vs independent references (pyfes)
# --------------------------------------------------------------------------


def test_predict_matches_pyfes():
    pyfes = pytest.importorskip("pyfes")
    names = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "2N2", "M4", "MS4", "Mf"]
    cells = region_cells()[:5]
    const = random_constants(cells, names, seed=1)
    times = pd.date_range("2023-08-01", "2023-09-01", freq="37min").values
    eta = th.predict_tide(const, times).values

    table = pyfes.wave_table_factory(pyfes.EngineType.DARWIN, names)
    order = list(table.constituents)
    c = th.as_complex(const).values
    for i in range(cells.size):
        wave = np.array([c[names.index(n), i] for n in order])[:, None]
        ref = table.tide_from_tide_series(times.astype("datetime64[us]"), wave)
        np.testing.assert_allclose(eta[:, i], np.ravel(ref), atol=1e-10)


# Astronomical arguments at 1992-01-01 00:00 (radians) of the OTPS/TMD tables, as
# stored in the variable ``phase`` of TPXO9-atlas v5 (TMD3 netCDF,
# Subset_NS_EC_forTMD30_21_2_2025.nc). Independent implementation of V0.
OTPS_V0_1992 = {
    "2N2": 4.08669963, "K1": 0.17300367, "K2": 3.4876, "M2": 1.73155755,
    "M4": 3.46311509, "Mf": 1.75604246, "Mm": 1.96402161, "MN4": 1.49909348,
    "MS4": 1.73155755, "N2": 6.05072124, "O1": 1.55855387, "P1": 6.11018163,
    "Q1": 5.87771757, "S2": 0.0,
}  # fmt: skip


def test_astronomical_arguments_match_otps():
    pyfes = pytest.importorskip("pyfes")
    names = list(OTPS_V0_1992)
    table = pyfes.wave_table_factory(pyfes.EngineType.DARWIN, names)
    angles = pyfes.AstronomicAngle(pyfes.Formulae.SCHUREMAN_ORDER_1)
    angles.update(pd.Timestamp("1992-01-01").to_pydatetime())
    table.compute_nodal_corrections(angles)
    for name, v0 in OTPS_V0_1992.items():
        diff = np.degrees(np.angle(np.exp(1j * (table[name].v - v0))))
        assert abs(diff) < 0.05, (name, diff)


def _pugh_nodal(name, date):
    """f and u (degrees) from the approximations of Pugh (1987), Table 4.3."""
    jd = pd.Timestamp(date).to_julian_date()
    t = (jd - 2451545.0) / 36525.0
    n = np.radians(125.04452 - 1934.136261 * t)  # Moon's ascending node (Meeus)
    if name == "M2":
        return 1.0004 - 0.0373 * np.cos(n) + 0.0002 * np.cos(2 * n), -2.14 * np.sin(n)
    if name == "K1":
        f = (
            1.0060
            + 0.1150 * np.cos(n)
            - 0.0088 * np.cos(2 * n)
            + 0.0006 * np.cos(3 * n)
        )
        u = -8.86 * np.sin(n) + 0.68 * np.sin(2 * n) - 0.07 * np.sin(3 * n)
        return f, u
    if name == "O1":
        f = (
            1.0089
            + 0.1871 * np.cos(n)
            - 0.0147 * np.cos(2 * n)
            + 0.0014 * np.cos(3 * n)
        )
        u = 10.80 * np.sin(n) - 1.34 * np.sin(2 * n) + 0.19 * np.sin(3 * n)
        return f, u
    raise KeyError(name)


@pytest.mark.parametrize(
    "date", ["1992-01-01", "2006-06-01", "2015-03-15", "2023-08-15"]
)
def test_nodal_factors_match_pugh(date):
    pyfes = pytest.importorskip("pyfes")
    names = ["M2", "K1", "O1"]
    table = pyfes.wave_table_factory(pyfes.EngineType.DARWIN, names)
    angles = pyfes.AstronomicAngle(pyfes.Formulae.SCHUREMAN_ORDER_1)
    angles.update(pd.Timestamp(date).to_pydatetime())
    table.compute_nodal_corrections(angles)
    mod = th.nodal_modulations(np.array([date], dtype="datetime64[ns]"), names)
    for name in names:
        f_ref, u_ref = _pugh_nodal(name, date)
        assert abs(table[name].f - f_ref) < 0.005, name
        assert abs(np.degrees(np.angle(np.exp(1j * table[name].u))) - u_ref) < 0.3
        # nodal_modulations returns f and V + u of the same table
        assert np.isclose(mod["f"].sel(constituent=name).item(), table[name].f)


# --------------------------------------------------------------------------
# prediction logic (no pyfes)
# --------------------------------------------------------------------------


def test_predict_with_modulations_is_harmonic_sum(synthetic_astronomy):
    cells = region_cells()[:7]
    const = random_constants(cells)
    times = pd.date_range("2023-01-01", periods=50, freq="1h")
    mod = synthetic_astronomy(times, ["M2", "S2", "K1", "O1"])
    eta = th.predict_tide(const, modulations=mod).values
    c = th.as_complex(const).values
    expected = (
        np.abs(c)[None] * np.cos(mod["vu"].values[:, :, None] - np.angle(c)[None])
    ).sum(1)
    np.testing.assert_allclose(eta, expected, atol=1e-12)
    # constituents order of the modulations does not matter
    eta2 = th.predict_tide(const, modulations=mod.isel(constituent=[3, 1, 0, 2])).values
    np.testing.assert_allclose(eta2, eta, atol=1e-12)


def test_prediction_time_reference(synthetic_astronomy):
    const = random_constants(region_cells()[:4])
    times = pd.date_range("2023-08-02", periods=72, freq="1h")
    full = th.predict_tide(const, times.values).values
    late = th.predict_tide(const, times[30:].values).values
    np.testing.assert_allclose(late, full[30:], atol=1e-12)


def test_timezone_aware_times(synthetic_astronomy):
    const = random_constants(region_cells()[:3])
    naive = pd.date_range("2023-08-02", periods=5, freq="1h")
    aware = naive.tz_localize("UTC").tz_convert("Europe/Paris")
    np.testing.assert_allclose(
        th.predict_tide(const, th._to_utc_datetime64(aware)).values,
        th.predict_tide(const, naive.values).values,
    )


# --------------------------------------------------------------------------
# likelihood
# --------------------------------------------------------------------------


def _synthetic_record(const, cell, start="2023-08-01", days=4, depth=19.0, noise=0.05):
    times = pd.date_range(start, periods=days * 24 * 6, freq="10min")
    eta = th.predict_tide(const.isel(cells=[cell]), times.values).values[:, 0]
    rng = np.random.default_rng(3)
    return pd.Series(depth + eta + rng.normal(0, noise, times.size), index=times)


def test_likelihood_peaks_at_true_cell(synthetic_astronomy):
    cells = region_cells()
    const = random_constants(cells, seed=5)
    true = 123
    record = _synthetic_record(const, true)
    lik = th.still_tag_tide_likelihood(record, const, sigma=0.3)
    assert lik.sizes["time"] == 4
    np.testing.assert_allclose(lik["pdf"].sum("cells").values, 1.0)
    assert (lik["pdf"].argmax("cells").values == true).all()
    assert (lik["tide_found"].values == 1).all()

    wc = np.full(cells.size, 30.0)
    wc[true] = 19.0
    lik_wc = th.still_tag_tide_likelihood(record, const, sigma=0.3, water_column=wc)
    assert (lik_wc["pdf"].argmax("cells").values == true).all()
    assert lik_wc.attrs["depth_reference"] == "water_column"


def test_likelihood_uniform_without_evidence(synthetic_astronomy):
    cells = region_cells()
    const = random_constants(cells)
    real = const["tide_real"].values.copy()
    real[:, :10] = np.nan  # land cells
    const["tide_real"] = (const["tide_real"].dims, real)
    record = _synthetic_record(const, 50, days=2)
    valid = pd.Series(record.index < pd.Timestamp("2023-08-02"), index=record.index)
    lik = th.still_tag_tide_likelihood(record, const, sigma=0.3, valid=valid)
    assert lik["tide_found"].values.tolist() == [1, 0]
    day2 = lik["pdf"].isel(time=1).values
    assert np.allclose(day2[10:], 1.0 / (cells.size - 10)) and (day2[:10] == 0).all()
    assert (lik["pdf"].isel(time=0).values[:10] == 0).all()


def test_likelihood_time_reference(synthetic_astronomy):
    const = random_constants(region_cells(), seed=7)
    record = _synthetic_record(const, 77, days=5)
    full = th.still_tag_tide_likelihood(record, const, sigma=0.3)
    late = th.still_tag_tide_likelihood(record["2023-08-03":], const, sigma=0.3)
    xr.testing.assert_allclose(
        full["pdf"].sel(time=late.time), late["pdf"], rtol=1e-10, atol=1e-300
    )


def test_segment_window_and_n_eff(synthetic_astronomy):
    const = random_constants(region_cells(), seed=2)
    record = _synthetic_record(const, 10, days=3, noise=0.2)
    seg = th.still_tag_tide_likelihood(
        record, const, sigma=0.4, time_step=None, n_eff=8
    )
    assert seg.sizes["time"] == 1 and seg.attrs["n_eff"] == 8
    rms = seg["rms"].isel(time=0).values
    np.testing.assert_allclose(
        seg["log_likelihood"].isel(time=0).values, -0.5 * 8 * (rms / 0.4) ** 2
    )


# --------------------------------------------------------------------------
# containers and grid handling
# --------------------------------------------------------------------------


def test_open_roundtrip(tmp_path):
    cells = region_cells()
    const = random_constants(cells)
    const.to_zarr(tmp_path / "c.zarr", zarr_format=2, consolidated=True)
    wanted = th.make_cell_ids(
        np.r_[cells.values[[5, 2]], np.uint64(12)], LEVEL, "WGS84"
    )
    out = th.open_harmonic_constants(tmp_path / "c.zarr", wanted, ["k1", "m2"]).load()
    assert out["constituent"].values.tolist() == ["K1", "M2"]
    np.testing.assert_array_equal(out["cell_ids"].values, wanted.values)
    np.testing.assert_allclose(
        out["tide_real"].values[:, :2],
        const["tide_real"].sel(constituent=["K1", "M2"]).values[:, [5, 2]],
    )
    assert np.isnan(out["tide_real"].values[:, 2]).all()
    assert out["cell_ids"].attrs["ellipsoid"] == "WGS84"
    with pytest.raises(ValueError, match="mismatch"):
        th.open_harmonic_constants(tmp_path / "c.zarr", ellipsoid="sphere")


def test_ellipsoid_required():
    ids = xr.DataArray(np.array([1, 2], "uint64"), dims="cells", attrs={"level": 10})
    with pytest.raises(ValueError, match="ellipsoid"):
        th.cell_centers(ids)


def _linear_latlon(field, lat, lon, constituents=("M2",)):
    """Lat/lon constants dataset from a complex field (lat, lon)."""
    values = np.broadcast_to(field, (len(constituents),) + field.shape)
    return th._latlon_constants(
        values, list(constituents), lat, lon, np.ones(field.shape, bool), {}
    )


def test_sphere_vs_wgs84_cells():
    lat = np.arange(50.0, 53.5, 1 / 30)
    lon = np.arange(1.5, 5.0, 1 / 30)
    field = lat[:, None] + 0j * lon[None, :]  # real part = latitude
    src = _linear_latlon(field, lat, lon)
    for ellipsoid in ("WGS84", "sphere"):
        cells = region_cells(ellipsoid=ellipsoid)
        out = th.constants_to_healpix(src, cells)
        _, lat_c = th.cell_centers(cells)
        np.testing.assert_allclose(out["tide_real"].values[0], lat_c, atol=1e-9)
        assert out["cell_ids"].attrs["ellipsoid"] == ellipsoid
    # same ids, different ellipsoid: the centres differ by ~0.125 degrees at 51 N
    ids = region_cells(ellipsoid="WGS84")
    sphere = th.make_cell_ids(ids.values, LEVEL, "sphere")
    dlat = th.cell_centers(ids)[1] - th.cell_centers(sphere)[1]
    assert 0.10 < np.abs(dlat).mean() < 0.14
    const = th.constants_to_healpix(src, ids)
    with pytest.raises(ValueError, match="mismatch"):
        th.select_constants(const, cell_ids=sphere)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def _amphidrome_source(lon0=3.2, lat0=51.7):
    lat = np.arange(51.0, 52.5, 1 / 30)
    lon = np.arange(2.0, 4.6, 1 / 30)
    x = (lon[None, :] - lon0) * np.cos(np.radians(lat0))
    y = lat[:, None] - lat0
    return lat, lon, 2.0 * (x + 1j * y)  # amplitude 0 at the amphidrome


def test_amphidrome_complex_resampling():
    lat, lon, field = _amphidrome_source()
    src = _linear_latlon(field, lat, lon)
    cells = region_cells(level=12, lon=(3.0, 3.4), lat=(51.55, 51.85))
    out = th.constants_to_healpix(src, cells)
    lon_c, lat_c = th.cell_centers(cells)
    expected = 2.0 * ((lon_c - 3.2) * np.cos(np.radians(51.7)) + 1j * (lat_c - 51.7))
    np.testing.assert_allclose(th.as_complex(out).values[0], expected, atol=1e-9)


def test_amplitude_phase_interpolation_fails():
    """Why the resampling uses complex values: amplitude/phase breaks at the wrap."""
    from scipy.interpolate import RegularGridInterpolator

    lat, lon, field = _amphidrome_source()
    cells = region_cells(level=12, lon=(3.0, 3.4), lat=(51.55, 51.85))
    lon_c, lat_c = th.cell_centers(cells)
    pts = np.c_[lat_c, lon_c]
    amp = RegularGridInterpolator((lat, lon), np.abs(field))(pts)
    pha = RegularGridInterpolator((lat, lon), np.angle(field))(pts)
    expected = 2.0 * ((lon_c - 3.2) * np.cos(np.radians(51.7)) + 1j * (lat_c - 51.7))
    assert np.abs(amp * np.exp(1j * pha) - expected).max() > 0.05


def test_land_renormalisation_and_fill():
    lat = np.arange(51.0, 52.5, 1 / 30)
    lon = np.arange(2.0, 4.6, 1 / 30)
    field = np.full((lat.size, lon.size), 0.5 + 0.5j)
    src = _linear_latlon(field, lat, lon)
    src["mask"] = (("lat", "lon"), lon[None, :] < 3.0 + 0 * lat[:, None])
    cells = region_cells()
    lon_c, _ = th.cell_centers(cells)
    out = th.constants_to_healpix(src, cells)
    c = th.as_complex(out).values[0]
    ok = np.isfinite(c)
    np.testing.assert_allclose(c[ok], 0.5 + 0.5j)  # no dilution by land zeros
    assert ok[lon_c < 2.95].all() and not ok[lon_c > 3.1].any()
    filled = th.as_complex(
        th.constants_to_healpix(src, cells, fill_distance_km=8)
    ).values[0]
    newly = np.isfinite(filled) & ~ok
    assert newly.any() and np.allclose(filled[newly], 0.5 + 0.5j)
    assert not np.isfinite(filled[lon_c > 3.3]).any()


def test_fes_style_amplitude_phase_input():
    lat = np.arange(51.0, 52.0, 1 / 16)
    lon = np.arange(-2.0, 2.0, 1 / 16) % 360  # FES files use 0..360
    amp_cm = np.full((2, lat.size, lon.size), 150.0)
    phase = np.empty_like(amp_cm)
    phase[0] = 359.0
    phase[1] = 1.0
    phase[:, :, ::2] = np.array([1.0, 359.0])[:, None, None]  # wrap between neighbours
    amp_cm[:, 0, 0] = np.nan  # land
    src = th.latlon_constants_from_amplitude_phase(
        amp_cm, phase, ["M2", "S2"], lat, lon, amplitude_scale=0.01
    )
    assert src["lon"].values.min() < 0 and not src["mask"].values.all()
    cells = region_cells(lon=(-1.5, 1.5), lat=(51.2, 51.8))
    out = th.constants_to_healpix(src, cells)
    c = th.as_complex(out).values
    np.testing.assert_allclose(np.abs(c), 1.5 * np.cos(np.radians(1.0)), rtol=1e-3)
    np.testing.assert_allclose(np.angle(c), 0.0, atol=np.radians(1.01))


def test_tmd_reader_rebases_s1(tmp_path):
    pytest.importorskip("pyfes")
    lat = np.array([51.0, 51.5])
    lon = np.array([2.0, 2.5])
    names = ["m2", "s1"]
    v0 = [OTPS_V0_1992["M2"], 0.0]  # OTPS uses V0(S1) = 0, pyfes 180 degrees
    shape = (2, lat.size, lon.size)
    ds = xr.Dataset(
        {
            "hRe": (("constituents", "lat", "lon"), np.ones(shape)),
            "hIm": (("constituents", "lat", "lon"), np.zeros(shape)),
            "mask": (("lat", "lon"), np.ones(shape[1:])),
            "phase": (("constituents", "lon"), np.repeat(np.array(v0)[:, None], 2, 1)),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "constituents": (
                "constituents",
                [1, 2],
                {"constituent_order": " ".join(names)},
            ),
        },
    )
    out = th.latlon_constants_from_tmd(ds)
    c = (out["tide_real"] + 1j * out["tide_imag"]).values
    assert np.allclose(np.degrees(np.angle(c[0])), 0.0, atol=0.05)
    assert np.allclose(np.abs(np.degrees(np.angle(c[1]))), 180.0, atol=0.05)


# --------------------------------------------------------------------------
# writer compatible with merge_tide_with_total
# --------------------------------------------------------------------------


def test_writer_roundtrip_with_merge(tmp_path, synthetic_astronomy):
    tide = pytest.importorskip("pangeo_fish.tide")
    cells = region_cells()
    const = random_constants(cells)
    record = _synthetic_record(const, 40, days=3)
    lik = th.still_tag_tide_likelihood(record, const, sigma=0.3)
    with pytest.raises(ValueError, match="behav_found"):
        th.write_tide_pdf_healpix(lik, tmp_path, behav_found=[0, 1])
    th.write_tide_pdf_healpix(lik, tmp_path, behav_found=[0, 1, 0])

    total = xr.Dataset(
        {"pdf": (("time", "cells"), np.full((3, cells.size), 1.0 / cells.size))},
        coords={
            "time": lik.time.values,
            "cell_ids": ("cells", cells.values, cells.attrs),
        },
        attrs={"sources": ["temperature"]},
    )
    total.to_zarr(tmp_path / "total_pdf.zarr", zarr_format=2, consolidated=True)
    merged, tide_mask, behav_mask = tide.merge_tide_with_total(tmp_path)
    np.testing.assert_allclose(merged["pdf_tide"].values, lik["pdf"].values)
    assert (tide_mask.values == 1).all()  # merge broadcasts it over cells
    assert (behav_mask.isel(cells=0).values == [0, 1, 0]).all()


@pytest.mark.parametrize(
    ("level", "ellipsoid", "mismatch"),
    [(2, "WGS84", None), (3, "WGS84", "level"), (2, "sphere", "ellipsoid")],
)
def test_merge_tide_matches_cell_ids_and_grid(tmp_path, level, ellipsoid, mismatch):
    from pangeo_fish.tide import merge_tide_with_total

    times = pd.date_range("2024-01-01", periods=2)
    likelihood = xr.Dataset(
        {
            "pdf": (("time", "cells"), [[0.1, 0.9], [0.8, 0.2]]),
            "ocean_mask": ("cells", [True, True]),
            "tide_found": ("time", [1, 1]),
        },
        coords={"time": times, "cell_ids": th.make_cell_ids([1, 2], 2, "WGS84")},
    )
    th.write_tide_pdf_healpix(likelihood, tmp_path, behav_found=[0, 1])
    total = xr.Dataset(
        {"pdf": (("time", "cells"), [[0.3, 0.7], [0.4, 0.6]])},
        coords={"time": times, "cell_ids": th.make_cell_ids([2, 1], level, ellipsoid)},
    )
    total.to_zarr(tmp_path / "total_pdf.zarr", zarr_format=2, consolidated=True)

    if mismatch:
        with pytest.raises(ValueError, match=f"grid mismatch for {mismatch}"):
            merge_tide_with_total(tmp_path)
        return

    merged, _, _ = merge_tide_with_total(tmp_path)
    np.testing.assert_array_equal(merged.cell_ids, total.cell_ids)
    np.testing.assert_array_equal(merged.pdf, total.pdf)
    np.testing.assert_array_equal(merged.pdf_tide, likelihood.pdf[:, ::-1])


@pytest.mark.parametrize("use_xarray", [False, True])
def test_writer_aligns_behaviour_dates(tmp_path, use_xarray):
    times = pd.date_range("2024-01-01", periods=3)
    likelihood = xr.Dataset(
        {
            "pdf": (("time", "cells"), [[1.0], [1.0], [1.0]]),
            "ocean_mask": ("cells", [True]),
            "tide_found": ("time", [1, 0, 1]),
        },
        coords={"time": times, "cell_ids": th.make_cell_ids([1], 2, "WGS84")},
    )
    flags = pd.Series([0, 0, 1], index=times[::-1])
    wrong_dates = pd.Series(flags.values, index=flags.index + pd.Timedelta(days=1))
    if use_xarray:
        flags = xr.DataArray(flags, dims="time", coords={"time": flags.index})
        wrong_dates = xr.DataArray(
            wrong_dates, dims="time", coords={"time": wrong_dates.index}
        )
    with pytest.raises(ValueError, match="behav_found.*each likelihood time"):
        th.write_tide_pdf_healpix(likelihood, tmp_path, behav_found=wrong_dates)
    store = th.write_tide_pdf_healpix(likelihood, tmp_path, behav_found=flags)
    with xr.open_zarr(store) as written:
        np.testing.assert_array_equal(written.behav_found.values, [1, 0, 0])


# --------------------------------------------------------------------------
# model-based detection, harmonic analysis, misfit, storage
# --------------------------------------------------------------------------


def test_model_based_detection(synthetic_astronomy):
    cells = region_cells()
    const = random_constants(cells, seed=11)
    record = _synthetic_record(const, 60, days=4, noise=0.03)
    # day 2: flat record (no tide), day 3: noise only (no tide of any cell)
    rng = np.random.default_rng(0)
    day2 = (record.index >= "2023-08-02") & (record.index < "2023-08-03")
    day3 = (record.index >= "2023-08-03") & (record.index < "2023-08-04")
    record[day2] = 19.0
    record[day3] = 19.0 + rng.normal(0, 0.5, day3.sum())
    lik = th.still_tag_tide_likelihood(record, const, sigma=0.3, detection_ratio=0.5)
    assert lik["tide_found"].values.tolist() == [1, 0, 0, 1]
    allowed = lik["ocean_mask"].values
    for k in (1, 2):
        np.testing.assert_allclose(
            lik["pdf"].isel(time=k).values[allowed], 1 / allowed.sum()
        )
    assert lik["best_rms"].isel(time=0) < 0.1 < lik["no_tide_rms"].isel(time=0)
    # without detection every window with data is used
    plain = th.still_tag_tide_likelihood(record, const, sigma=0.3)
    assert plain["tide_found"].values.tolist() == [1, 1, 1, 1]


def test_harmonic_analysis_recovers_constants():
    pytest.importorskip("pyfes")
    cells = region_cells()[:1]
    names = ["M2", "S2", "K1", "O1"]
    const = random_constants(cells, names, seed=4)
    times = pd.date_range("2023-08-01", periods=30 * 24 * 6, freq="10min")
    eta = th.predict_tide(const, times.values).values[:, 0]
    rng = np.random.default_rng(1)
    record = pd.Series(18.0 + eta + rng.normal(0, 0.02, times.size), index=times)
    ha = th.harmonic_analysis(record, names, freq=None)
    c_true = th.as_complex(const).values[:, 0]
    c_fit = (ha["tide_real"] + 1j * ha["tide_imag"]).values
    np.testing.assert_allclose(c_fit, c_true, atol=0.01)
    # hourly means (default): no phase shift, only the ~1 % box-average attenuation
    hourly = th.harmonic_analysis(record, names)
    c_hour = (hourly["tide_real"] + 1j * hourly["tide_imag"]).values
    np.testing.assert_allclose(np.angle(c_hour / c_true), 0, atol=np.radians(0.5))
    np.testing.assert_allclose(np.abs(c_hour / c_true), 1, atol=0.02)
    assert (
        abs(ha.attrs["mean_depth"] - 18.0) < 0.01 and ha.attrs["residual_rms"] < 0.025
    )
    # the misfit is ~0 at the true cell and larger elsewhere
    other = random_constants(region_cells()[:6], names, seed=9)
    for name in ("tide_real", "tide_imag"):
        other[name][:, 0] = const[name][:, 0]
    mis = th.constants_misfit(ha, other)
    assert mis.dims == ("cells",) and mis.values[0] < 0.01 < mis.values[1:].min()
    with pytest.raises(ValueError, match="Rayleigh"):
        th.harmonic_analysis(record[:"2023-08-10"], ["M2", "N2"])


def test_constants_misfit_latlon():
    lat = np.arange(51.0, 52.0, 0.1)
    lon = np.arange(2.0, 3.0, 0.1)
    field = (lon[None, :] - 2.5) + 1j * (lat[:, None] - 51.5)
    src = _linear_latlon(field, lat, lon)
    obs = xr.Dataset(
        {"tide_real": ("constituent", [0.0]), "tide_imag": ("constituent", [0.0])},
        coords={"constituent": ["m2"]},
    )
    mis = th.constants_misfit(obs, src)
    assert mis.dims == ("lat", "lon")
    i, j = np.unravel_index(int(np.argmin(mis.values)), mis.shape)
    assert abs(lat[i] - 51.5) < 0.06 and abs(lon[j] - 2.5) < 0.06


def test_write_harmonic_constants(tmp_path):
    cells = region_cells()
    const = random_constants(cells).isel(cells=slice(None, None, -1))  # unsorted
    th.write_harmonic_constants(const, tmp_path / "c.zarr", chunk_cells=100)
    out = th.open_harmonic_constants(tmp_path / "c.zarr")
    assert out["tide_real"].dtype == np.float32
    assert (np.diff(out["cell_ids"].values.astype(np.int64)) > 0).all()
    assert out["tide_real"].chunks == (
        (4,),
        tuple([100] * (cells.size // 100))
        + ((cells.size % 100,) if cells.size % 100 else ()),
    )
    back = th.select_constants(out, cell_ids=const["cell_ids"]).load()
    np.testing.assert_allclose(
        back["tide_real"].values, const["tide_real"].values, rtol=1e-6
    )


def test_hourly_means_at_mean_sample_time():
    times = pd.date_range("2023-08-01", periods=24 * 6, freq="10min")
    series = pd.Series(np.arange(times.size, dtype=float), index=times)
    out = th._bin_average(series, "1h")
    assert out.index[0] == pd.Timestamp("2023-08-01 00:25")  # samples :00..:50
    assert out.iloc[0] == 2.5
    tz = series.copy()
    tz.index = tz.index.tz_localize("UTC")
    assert th._bin_average(th._as_series(tz), "1h").index[0] == pd.Timestamp(
        "2023-08-01 00:25"
    )


# --------------------------------------------------------------------------
# review fixes: detection with a bathymetry bias, empty windows, alignment,
# partial windows, constituent names, bbox
# --------------------------------------------------------------------------


def test_detection_not_hidden_by_bathymetry_bias(synthetic_astronomy):
    cells = region_cells()
    const = random_constants(cells, seed=5)
    record = _synthetic_record(const, 123, days=3, noise=0.01)
    wc = np.full(cells.size, 20.0)  # 1 m deeper than the record mean everywhere
    lik = th.still_tag_tide_likelihood(
        record, const, sigma=0.3, water_column=wc, detection_ratio=0.5
    )
    assert (lik["tide_found"].values == 1).all()
    assert (lik["best_rms"] < 0.05).all()
    assert (lik["rms"].isel(cells=123) > 0.9).all()  # rms keeps the offset


def test_no_valid_sample_gives_uniform(synthetic_astronomy):
    cells = region_cells()
    const = random_constants(cells)
    record = _synthetic_record(const, 5, days=2)
    valid = pd.Series(False, index=record.index)
    lik = th.still_tag_tide_likelihood(record, const, sigma=0.3, valid=valid)
    assert (lik["tide_found"].values == 0).all()
    np.testing.assert_allclose(lik["pdf"].values, 1.0 / cells.size)
    with pytest.raises(ValueError, match="no samples"):
        th.still_tag_tide_likelihood(record.iloc[:0], const, sigma=0.3)


@pytest.mark.parametrize("missing", ["real", "imag", "ocean_mask", "water_column"])
def test_likelihood_rejects_no_eligible_cells(missing):
    cells = th.make_cell_ids([1, 2], 2, "WGS84")
    constants = th.harmonic_constants_dataset([[1j, 2j]], ["M2"], cells)
    depth = pd.Series([10.0, 10.0], index=pd.date_range("2024-01-01", periods=2))
    kwargs = {}
    if missing in ("real", "imag"):
        constants[f"tide_{missing}"][:] = np.nan
    elif missing == "ocean_mask":
        kwargs["ocean_mask"] = [False, False]
    else:
        kwargs["water_column"] = [np.nan, np.nan]
    with pytest.raises(ValueError, match="no eligible cells"):
        th.still_tag_tide_likelihood(depth, constants, sigma=0.3, **kwargs)


def test_likelihood_matches_averaged_prediction(synthetic_astronomy):
    cell_ids = th.make_cell_ids([1, 2], 2, "WGS84")
    times = pd.date_range("2024-01-01", periods=3 * 24 * 6, freq="10min")
    omega = np.radians(OMEGA_DEG_PER_HOUR["M2"])
    offsets = (np.arange(24) - 11.5) / 6
    attenuation = np.cos(omega * offsets).mean()
    constants = th.harmonic_constants_dataset(
        [[1 + 0j, attenuation + 0j]], ["M2"], cell_ids
    )
    tide = th.predict_tide(constants, times).isel(cells=0).to_series()
    valid = pd.Series(True, index=times)
    valid.iloc[:48] = False  # empty bins
    valid.iloc[67::29] = False  # irregular sampling within bins
    depth = 10.0 + tide
    depth.iloc[55::51] = np.nan

    lik = th.still_tag_tide_likelihood(
        depth, constants, sigma=0.1, valid=valid, freq="4h", time_step=None
    )
    assert lik.pdf.argmax("cells").item() == 0
    np.testing.assert_allclose(lik.rms.isel(cells=0), 0, atol=1e-12)


def test_harmonic_analysis_matches_averaged_basis(synthetic_astronomy, monkeypatch):
    constituents = ["M2"]
    monkeypatch.setattr(
        th,
        "_frequencies_rad_per_hour",
        lambda _: [np.radians(OMEGA_DEG_PER_HOUR["M2"])],
    )
    constants = th.harmonic_constants_dataset(
        [[0.8 + 0.6j]], constituents, th.make_cell_ids([1], 2, "WGS84")
    )
    times = pd.date_range("2024-01-01", periods=3 * 24 * 6, freq="10min")
    depth = 10 + th.predict_tide(constants, times).isel(cells=0).to_series()
    depth.iloc[:48] = np.nan
    depth.iloc[55::29] = np.nan

    fit = th.harmonic_analysis(depth, constituents, freq="4h")
    np.testing.assert_allclose(fit.tide_real, [0.8], atol=1e-12)
    np.testing.assert_allclose(fit.tide_imag, [0.6], atol=1e-12)
    assert fit.attrs["residual_rms"] < 1e-12


def test_per_cell_inputs_aligned_on_cell_ids(synthetic_astronomy):
    cells = region_cells()
    const = random_constants(cells, seed=5)
    record = _synthetic_record(const, 123, days=2)
    wc = np.full(cells.size, 30.0)
    wc[123] = 19.0
    mask = np.ones(cells.size, dtype=bool)
    mask[:20] = False
    ref = th.still_tag_tide_likelihood(
        record, const, sigma=0.3, water_column=wc, ocean_mask=mask
    )
    order = np.random.default_rng(0).permutation(cells.size)
    coords = {"cell_ids": ("cells", cells.values[order], cells.attrs)}
    shuffled = th.still_tag_tide_likelihood(
        record,
        const,
        sigma=0.3,
        water_column=xr.DataArray(wc[order], dims="cells", coords=coords),
        ocean_mask=xr.DataArray(mask[order], dims="cells", coords=coords),
    )
    np.testing.assert_allclose(shuffled["pdf"].values, ref["pdf"].values)
    with pytest.raises(ValueError, match="water_column"):
        th.still_tag_tide_likelihood(record, const, sigma=0.3, water_column=wc[:-1])
    missing = xr.DataArray(
        wc[:-1], dims="cells", coords={"cell_ids": ("cells", cells.values[:-1])}
    )
    with pytest.raises(ValueError, match="lacks 1 cells"):
        th.still_tag_tide_likelihood(record, const, sigma=0.3, water_column=missing)


def test_n_eff_scaled_by_window_coverage(synthetic_astronomy):
    cells = region_cells()
    const = random_constants(cells, seed=2)
    record = _synthetic_record(const, 10, days=2, noise=0.2)
    record = record[record.index < pd.Timestamp("2023-08-02T12:00")]  # 1.5 days
    lik = th.still_tag_tide_likelihood(record, const, sigma=0.4, n_eff=8)
    n = -2 * 0.4**2 * lik["log_likelihood"] / lik["rms"] ** 2
    np.testing.assert_allclose(n.isel(cells=0).values, [8.0, 4.0])


def test_predict_tide_modulation_names_any_case(synthetic_astronomy):
    const = random_constants(region_cells()[:5])
    times = pd.date_range("2023-08-01", periods=4, freq="1h")
    names = [str(c) for c in const[th.CONSTITUENT_DIM].values]
    mod = th.nodal_modulations(times, names[::-1])
    mod = mod.assign_coords({th.CONSTITUENT_DIM: [n.lower() for n in names[::-1]]})
    np.testing.assert_allclose(
        th.predict_tide(const, modulations=mod).values,
        th.predict_tide(const, times.values).values,
    )


def test_bbox_across_antimeridian_raises():
    bbox = {"latitude": [-10, 10], "longitude": [172, -172]}
    with pytest.raises(ValueError, match="antimeridian"):
        th._bbox_slices(np.arange(-20, 21.0), np.arange(-180, 180.0), bbox, 0.1)
