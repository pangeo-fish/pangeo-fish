"""Unit tests for pangeo_fish.light."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.light import (
    align_light_to_time,
    compute_quality_flags,
    compute_solar_likelihood,
    detect_twilight_events,
    self_calibrate_solar_threshold,
    solar_likelihood_cells,
    sun,
    twilight_quality,
)
from pangeo_fish.light.quality import dynamic_threshold

BISCAY = (-3.0, 47.1)
MAIZURU = (135.38, 35.46)
TRUE_ELEVATION = -6.0


def synthetic_record(lon, lat, start, days, seed=0):
    """One-minute light record of a tag at a fixed position.

    The light follows a logistic function of the solar elevation centred on
    ``TRUE_ELEVATION``, plus noise; the tag stays at 5 m.
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 1440, freq="1min")
    elevation = sun.solar_elevation(t.values, lon, lat)
    light = 20 + 120 / (1 + np.exp(-(elevation - TRUE_ELEVATION) / 0.7))
    light = light + rng.normal(0, 2, t.size)
    pressure = np.abs(rng.normal(5, 3, t.size))
    df = pd.DataFrame(
        {"time": t.tz_localize("UTC"), "light": light, "pressure": pressure}
    )
    df["dyn_thresh"] = 70.0
    return df


def healpix_cells(lon_range, lat_range, level, step=0.1):
    from healpix_geo.nested import lonlat_to_healpix

    from pangeo_fish.tide_healpix import make_cell_ids

    lon, lat = np.meshgrid(np.arange(*lon_range, step), np.arange(*lat_range, step))
    ids = np.unique(
        np.asarray(
            lonlat_to_healpix(lon.ravel(), lat.ravel(), level, ellipsoid="WGS84")
        )
    )
    return make_cell_ids(ids, level, "WGS84")


def distance_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon1) - lon2)
    cos = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return 6371.0 * np.arccos(np.clip(cos, -1, 1))


# ---------------------------------------------------------------------------
# quality.dynamic_threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase, expected", [(0, 60.0), (100, 80.0), (50, 70.0)])
def test_dynamic_threshold(phase, expected):
    assert dynamic_threshold(phase) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# sun
# ---------------------------------------------------------------------------


def test_sun_matches_ephem():
    ephem = pytest.importorskip("ephem")
    rng = np.random.default_rng(1)
    for _ in range(40):
        lon, lat = rng.uniform(-180, 180), rng.uniform(-60, 60)
        day = pd.Timestamp("2000-01-01") + pd.Timedelta(
            days=float(rng.uniform(0, 365 * 30))
        )
        obs = ephem.Observer()
        obs.lon, obs.lat, obs.pressure = str(lon), str(lat), 0
        obs.date = day.strftime("%Y/%m/%d %H:%M:%S")
        body = ephem.Sun()
        body.compute(obs)
        assert sun.solar_elevation(day.to_datetime64(), lon, lat) == pytest.approx(
            np.degrees(float(body.alt)), abs=0.05
        )

        obs.horizon = "-6"
        expected = pd.Timestamp(
            ephem.Date(obs.next_rising(ephem.Sun(), use_center=True)).datetime()
        )
        found = sun.crossing_time(
            (expected + pd.Timedelta(hours=3)).to_datetime64(), lon, lat, -6.0, "rise"
        )
        assert abs(pd.Timestamp(found.item()) - expected) < pd.Timedelta(minutes=1)


def test_crossing_time_polar_day_is_nat():
    # 80 N at the June solstice: the Sun never goes below -6 degrees
    t = np.datetime64("2022-06-21T00:00")
    assert np.isnat(sun.crossing_time(t, 0.0, 80.0, -6.0, "set"))


def test_crossing_time_window():
    t = np.datetime64("2022-06-21T12:00")
    # the setting at 47 N, 3 W is in the evening, more than 2 h away
    assert np.isnat(sun.crossing_time(t, -3.0, 47.0, -6.0, "set", window_h=2))
    assert not np.isnat(sun.crossing_time(t, -3.0, 47.0, -6.0, "set", window_h=12))


def test_daily_crossings_follow_local_day_in_japan():
    rise, sett = sun.daily_crossings(
        np.array(["2022-01-10"], dtype="datetime64[D]"), *MAIZURU, -6.0
    )
    # morning of 10 January in Japan is the evening of 9 January in UTC
    assert rise[0].astype("datetime64[D]") == np.datetime64("2022-01-09")
    assert sett[0].astype("datetime64[D]") == np.datetime64("2022-01-10")
    assert np.timedelta64(8, "h") < sett[0] - rise[0] < np.timedelta64(13, "h")


# ---------------------------------------------------------------------------
# detection, quality, calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [BISCAY, MAIZURU], ids=["biscay", "maizuru"])
def test_detection_quality_calibration(position):
    lon, lat = position
    df = synthetic_record(lon, lat, "2022-03-01", 15)

    pairs = detect_twilight_events(df, lon, lat)
    assert 13 <= len(pairs) <= 15
    for t_rise, t_set in pairs:
        assert t_rise < t_set < t_rise + pd.Timedelta(hours=24)

    qdf = compute_quality_flags(df, pairs)
    # the direction filter knows which event is a rising: also right in Japan
    assert (qdf["flag"] == "good").all()

    calib = self_calibrate_solar_threshold(
        pairs, n_nights=10, release_lon=lon, release_lat=lat
    )
    assert calib.thresh_deg == pytest.approx(TRUE_ELEVATION, abs=0.3)


def test_quality_direction_filter():
    lon, lat = BISCAY
    df = synthetic_record(lon, lat, "2022-03-01", 3)
    (t_rise, t_set), *_ = detect_twilight_events(df, lon, lat)

    assert twilight_quality(df, t_rise, direction="rise")[0]
    # a rising read as a setting fails the direction filter ...
    assert not twilight_quality(df, t_rise, direction="set")[0]
    # ... unless the filter is off
    assert twilight_quality(df, t_rise, direction="set", slope_threshold=None)[0]

    deep = df.assign(pressure=200.0)
    assert not twilight_quality(deep, t_set, direction="set")[0]


def test_quality_flags_empty():
    df = synthetic_record(*BISCAY, "2022-03-01", 1)
    qdf = compute_quality_flags(df, [])
    assert len(qdf) == 0
    assert "flag" in qdf.columns


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------


def _flags(n, flag="good", valid_rise=True, valid_set=True):
    return pd.DataFrame(
        {
            "flag": [flag] * n,
            "valid_rise": [valid_rise] * n,
            "valid_set": [valid_set] * n,
        }
    )


@pytest.fixture(scope="module")
def biscay_nights():
    lon, lat = BISCAY
    df = synthetic_record(lon, lat, "2022-05-01", 8)
    pairs = detect_twilight_events(df, lon, lat)
    qdf = compute_quality_flags(df, pairs)
    cell_ids = healpix_cells((-13, 7), (37, 57), level=6)
    return pairs, qdf, cell_ids


def test_solar_likelihood_cells_peaks_at_the_tag(biscay_nights):
    from pangeo_fish.tide_healpix import cell_centers

    pairs, qdf, cell_ids = biscay_nights
    nightly = solar_likelihood_cells(pairs, qdf, cell_ids, TRUE_ELEVATION, by="night")

    assert nightly.dims == ("time", "cells")
    assert nightly.sizes["time"] == len(pairs)
    assert nightly.name == "pdf_light"
    assert nightly["cell_ids"].attrs["level"] == 6
    np.testing.assert_allclose(nightly.max("cells"), 1.0)

    lon, lat = cell_centers(cell_ids)
    peak = nightly.values.argmax(axis=1)
    # level 6 cells are about 100 km wide
    assert (distance_km(lon[peak], lat[peak], *BISCAY) < 150).all()

    # the time is local noon, between the two crossings
    assert (nightly["rise_time"] < nightly["time"]).all()
    assert (nightly["time"] < nightly["set_time"]).all()


def test_solar_likelihood_cells_flags(biscay_nights):
    pairs, _, cell_ids = biscay_nights
    pairs = pairs[:2]

    bad = solar_likelihood_cells(
        pairs, _flags(2, "bad", False, False), cell_ids, -6.0, by="night"
    )
    np.testing.assert_array_equal(bad.values, 1.0)
    assert (
        solar_likelihood_cells(
            pairs, _flags(2, "bad", False, False), cell_ids, -6.0
        ).sizes["time"]
        == 0
    )

    good = solar_likelihood_cells(pairs, _flags(2), cell_ids, -6.0, by="night")
    rise_only = solar_likelihood_cells(
        pairs, _flags(2, "partial", True, False), cell_ids, -6.0, by="night"
    )
    # one event constrains less than two: more cells keep a high value
    assert (rise_only.values > 0.5).sum() > (good.values > 0.5).sum()

    with pytest.raises(ValueError, match="same length"):
        solar_likelihood_cells(pairs, _flags(3), cell_ids, -6.0)


def test_solar_likelihood_cells_good_night_needs_both_predictions():
    # polar summer: north of about 66 N the Sun never sets below -6 degrees,
    # so a "good" night gets the floor there instead of a one-sided value.
    pairs = [
        (
            pd.Timestamp("2022-06-21 03:30", tz="UTC"),
            pd.Timestamp("2022-06-21 20:30", tz="UTC"),
        )
    ]
    cell_ids = healpix_cells((-10, 10), (40, 85), level=4, step=0.5)
    nightly = solar_likelihood_cells(
        pairs, _flags(1), cell_ids, -6.0, lh_floor=1e-30, by="night"
    )

    from pangeo_fish.tide_healpix import cell_centers

    _, lat = cell_centers(cell_ids)
    polar = nightly.values[0, lat > 68]
    assert polar.size > 0
    # every polar cell has the floor (the night is divided by its maximum)
    np.testing.assert_allclose(polar, polar[0])
    assert polar[0] < 1e-20


def test_align_light_to_time():
    cell_ids = xr.DataArray(np.arange(3, dtype="uint64"), dims="cells")
    noon = pd.to_datetime(["2022-01-01 12:30", "2022-01-02 11:50", "2022-01-03 12:10"])
    nightly = xr.DataArray(
        np.array([[1.0, 0.5, 0.1], [0.2, 1.0, 0.5], [0.5, 0.5, 1.0]]),
        dims=("time", "cells"),
        coords={
            "time": noon.values,
            # the first rising is on the previous UTC date, as in Japan
            "rise_time": ("time", (noon - pd.Timedelta(hours=14)).values),
            "cell_ids": cell_ids,
        },
        name="pdf_light",
    )
    days = pd.date_range("2021-12-31", periods=5, freq="D").values

    by_date = align_light_to_time(nightly, days, how="date")
    assert by_date.dims == ("time", "cells")
    np.testing.assert_array_equal(by_date["n_light"], [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(by_date.values[0], 1.0)  # no night: neutral
    np.testing.assert_array_equal(by_date.values[1], nightly.values[0])

    by_rise = align_light_to_time(nightly, days, how="rise_date")
    np.testing.assert_array_equal(by_rise["n_light"], [1, 1, 1, 0, 0])

    # with bins centred on 00:00, a noon after 12:00 goes to the next day:
    # nights 1 and 2 share 2 January, multiplied and divided by the maximum
    by_bins = align_light_to_time(nightly, days, how="bins")
    np.testing.assert_array_equal(by_bins["n_light"], [0, 0, 2, 0, 1])
    product = nightly.values[0] * nightly.values[1]
    np.testing.assert_allclose(by_bins.values[2], product / product.max())

    # step i covers [day i, day i + 1): each noon stays on its own UTC date
    by_interval = align_light_to_time(nightly, days)
    np.testing.assert_array_equal(by_interval["n_light"], [0, 1, 1, 1, 0])

    with pytest.raises(ValueError, match="how must be"):
        align_light_to_time(nightly, days, how="nearest")
    with pytest.raises(ValueError, match="needs the nightly maps"):
        align_light_to_time(nightly.drop_vars("rise_time"), days, how="rise_date")


def test_solar_likelihood_cells_by_event(biscay_nights):
    from pangeo_fish.tide_healpix import cell_centers

    pairs, qdf, cell_ids = biscay_nights
    events = solar_likelihood_cells(pairs, qdf, cell_ids, TRUE_ELEVATION)
    nightly = solar_likelihood_cells(pairs, qdf, cell_ids, TRUE_ELEVATION, by="night")

    valid = qdf[qdf.flag != "bad"]
    assert events.sizes["time"] == valid.valid_rise.sum() + valid.valid_set.sum()
    assert events.attrs["by"] == "event"
    assert set(events["direction"].values) == {"rise", "set"}
    # stamped with the observed crossings, sorted
    observed = sorted(pd.Timestamp(t).tz_convert(None) for pair in pairs for t in pair)
    assert set(pd.DatetimeIndex(events["time"].values)) <= set(observed)
    assert (np.diff(events["time"].values) > np.timedelta64(0)).all()
    np.testing.assert_allclose(events.max("cells"), 1.0)

    # rise and set of one night multiplied again give the nightly map
    for night in range(len(pairs)):
        rows = events.where(events.night == night, drop=True)
        if rows.sizes["time"] != 2:
            continue
        product = rows.values.prod(axis=0)
        np.testing.assert_allclose(product / product.max(), nightly.values[night])

    # aligned by observation time on UTC days: every step peaks near the tag
    days = pd.date_range("2022-04-30", "2022-05-10", freq="D").values
    aligned = align_light_to_time(events, days)
    lon, lat = cell_centers(cell_ids)
    used = aligned.n_light.values > 0
    peak = aligned.values[used].argmax(axis=1)
    assert used.sum() >= len(pairs)
    # a single crossing is a curve, so only steps with both crossings locate
    both = aligned.n_light.values[used] == 2
    assert (distance_km(lon[peak[both]], lat[peak[both]], *BISCAY) < 150).all()


def test_solar_likelihood_cells_smoothing(biscay_nights):
    pairs, qdf, cell_ids = biscay_nights
    sharp = solar_likelihood_cells(pairs, qdf, cell_ids, TRUE_ELEVATION, by="night")
    smooth = solar_likelihood_cells(
        pairs, qdf, cell_ids, TRUE_ELEVATION, by="night", smooth_sigma_km=300
    )
    assert smooth.attrs["smooth_sigma_km"] == 300
    np.testing.assert_allclose(smooth.max("cells"), 1.0)
    # smoothing spreads the weight: more cells above one half
    assert (smooth.values > 0.5).sum() > (sharp.values > 0.5).sum()
    # and keeps the peak where it was (level 6 cells are about 100 km wide)
    from pangeo_fish.tide_healpix import cell_centers

    lon, lat = cell_centers(cell_ids)
    peak = smooth.values.argmax(axis=1)
    assert (distance_km(lon[peak], lat[peak], *BISCAY) < 250).all()


def test_light_settings():
    from pangeo_fish.light import LIGHT_SETTINGS, light_settings

    wc = light_settings({"tag_type": "wc_archival", "pit_tag_id": "x"})
    assert (
        wc is not LIGHT_SETTINGS["wc_archival"] or wc == LIGHT_SETTINGS["wc_archival"]
    )
    assert wc.quality == dict(
        window_h=1.0,
        slope_threshold=25.0,
        marge_threshold=8.0,
        min_amplitude=40.0,
        min_samples=10,
    )
    assert wc.threshold == {"base": 60.0, "amp": 20.0}
    np.testing.assert_array_equal(wc.transform_light(np.array([0.0, 9.0])), [0, 9])

    c6 = light_settings("LoggLaw_C6")
    assert c6.slope_threshold is None
    np.testing.assert_allclose(
        c6.transform_light(np.array([0.0, 9.0])), np.log1p([0, 9])
    )

    custom = light_settings(
        {"tag_type": "wc_archival", "light": {"min_amplitude": 30}}, window_h=2.0
    )
    assert custom.min_amplitude == 30
    assert custom.window_h == 2.0

    with pytest.raises(KeyError, match="no 'tag_type'"):
        light_settings({"pit_tag_id": "x"})
    with pytest.raises(KeyError, match="no light settings"):
        light_settings("lotek")
    # an unlisted type works when every value is given
    full = {
        k: v
        for k, v in vars(LIGHT_SETTINGS["wc_archival"]).items()
        if k not in ("tag_type", "description")
    }
    lotek = light_settings({"tag_type": "lotek", "light": full})
    assert lotek.tag_type == "lotek"
    assert lotek.quality == wc.quality
    with pytest.raises(TypeError, match="unknown light settings"):
        light_settings("wc_archival", min_amp=3)


def test_compute_solar_likelihood_regular_grid():
    lons = np.arange(-10, 11, 5.0)
    lats = np.arange(30, 51, 5.0)
    t0 = pd.Timestamp("2020-06-01 04:00:00", tz="UTC")
    pairs = [
        (t0 + pd.Timedelta(days=i), t0 + pd.Timedelta(days=i, hours=15))
        for i in range(5)
    ]
    da = compute_solar_likelihood(
        pairs, _flags(5, "bad", False, False), lons=lons, lats=lats, thresh_deg=-11.0
    )

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "latitude", "longitude")
    assert da.sizes["time"] == 5
    assert da.attrs["method"] == "solar_threshold"
    np.testing.assert_array_equal(da.values, 1.0)
