import warnings

import healpix_geo
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish import helpers

LEVEL = 8
PARENT_LEVEL = 3


def _grid(parent=300):
    """HEALPix cells (level 8) under one level-3 cell, ocean in the west half."""
    factor = 4 ** (LEVEL - PARENT_LEVEL)
    cell_ids = np.arange(parent * factor, (parent + 1) * factor, dtype="int64")
    lon, lat = healpix_geo.nested.healpix_to_lonlat(cell_ids, LEVEL, ellipsoid="sphere")
    ocean = lon < np.median(lon)
    ds = xr.Dataset(
        {"ocean_mask": ("cells", ocean.astype(float))},
        coords={
            "cell_ids": ("cells", cell_ids),
            "latitude": ("cells", lat),
            "longitude": ("cells", lon),
        },
    )
    ds["cell_ids"].attrs.update(
        grid_name="healpix", level=LEVEL, indexing_scheme="nested", ellipsoid="sphere"
    )
    return helpers.to_healpix(ds)


def _position(lat, lon):
    return xr.Dataset({"latitude": lat, "longitude": lon})


def _land_point(ds):
    """Easternmost cell, as far as possible from the ocean."""
    i = int(np.argmax(ds["longitude"].data))
    return _position(float(ds["latitude"][i]), float(ds["longitude"][i]))


def _ocean_point(ds):
    i = int(np.argmin(ds["longitude"].data))
    return _position(float(ds["latitude"][i]), float(ds["longitude"][i]))


def test_check_event_position_on_land_warns():
    ds = _grid()
    pos = _land_point(ds)

    with pytest.warns(UserWarning, match="release probability"):
        report = helpers.check_event_position(
            grid=ds, position=pos, sigma=1e-5, mask=ds["ocean_mask"]
        )

    assert not report["ok"]
    assert report["ocean_mass"] < 1e-6
    assert report["distance_to_ocean"] > 0
    assert report["distance_to_ocean_km"] == pytest.approx(
        report["distance_to_ocean"] * 6370.997
    )
    # the nearest ocean cell is an ocean cell of the grid
    nearest = (ds["latitude"] == report["nearest_ocean_latitude"]) & (
        ds["longitude"] == report["nearest_ocean_longitude"]
    )
    assert bool(ds["ocean_mask"].where(nearest, drop=True).item())

    # the suggested sigma reaches the target ocean share
    assert report["suggested_sigma"] > report["distance_to_ocean"] / 10
    check = helpers.check_event_position(
        grid=ds,
        position=pos,
        sigma=report["suggested_sigma"],
        mask=ds["ocean_mask"],
        warn=False,
    )
    assert check["ocean_mass"] >= 0.1


def test_check_event_position_matches_normal_at():
    ds = _grid()
    pos = _land_point(ds)
    sigma = 2e-3
    mask = ds["ocean_mask"] > 0

    report = helpers.check_event_position(
        grid=ds, position=pos, sigma=sigma, mask=mask, warn=False
    )
    pdf = helpers.distrib.healpix.normal_at(ds, pos=pos, sigma=sigma)

    expected = float(pdf.where(mask.data).sum() / pdf.sum())
    assert report["ocean_mass"] == pytest.approx(expected, rel=1e-6)


def test_check_event_position_in_ocean_is_silent():
    ds = _grid()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = helpers.check_event_position(
            grid=ds, position=_ocean_point(ds), sigma=1e-5, mask=ds["ocean_mask"]
        )

    assert report["ok"]
    assert report["ocean_mass"] == pytest.approx(1.0)
    assert report["distance_to_ocean"] == 0
    assert report["suggested_sigma"] == 1e-5


def test_check_event_position_without_ocean():
    ds = _grid()

    with pytest.warns(UserWarning, match="no ocean cell"):
        report = helpers.check_event_position(
            grid=ds,
            position=_land_point(ds),
            sigma=1e-3,
            mask=xr.zeros_like(ds["ocean_mask"]),
            event_name="recapture",
        )

    assert report["ocean_mass"] == 0
    assert report["distance_to_ocean"] is None
    assert report["suggested_sigma"] is None


def _diff_ds(ds):
    time = pd.date_range("2024-01-01", periods=3, freq="1D")
    ocean = ds["ocean_mask"].data > 0
    h0 = np.where(ocean, 20.0, np.nan)
    diff = np.where(ocean, 0.1, np.nan) * np.ones((time.size, 1))
    return ds.assign(
        diff=(("time", "cells"), diff),
        H0=("cells", h0),
    ).assign_coords(time=time)


def _events(release, recapture):
    return xr.Dataset(
        {
            "latitude": ("event_name", [release[0], recapture[0]]),
            "longitude": ("event_name", [release[1], recapture[1]]),
        },
        coords={"event_name": ["release", "fish_death"]},
    )


def _latlon(pos):
    return float(pos["latitude"]), float(pos["longitude"])


def test_compute_emission_pdf_warns_for_release_and_recapture():
    ds = _grid()
    land = _latlon(_land_point(ds))
    events = _events(land, land)
    kwargs = dict(differences_std=1.0, initial_std=1e-5, recapture_std=1e-5)

    with pytest.warns(UserWarning) as record:
        helpers.compute_emission_pdf(diff_ds=_diff_ds(ds), events_ds=events, **kwargs)
    messages = [str(w.message) for w in record]
    assert any("release probability" in m for m in messages)
    assert any("recapture probability" in m for m in messages)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        helpers.compute_emission_pdf(
            diff_ds=_diff_ds(ds), events_ds=events, check_positions=False, **kwargs
        )


def test_compute_emission_pdf_uses_h0_as_valid_cells():
    # an ocean mask regridded with a permissive threshold can be 1 everywhere;
    # cells without bathymetry are still not usable
    ds = _grid()
    diff_ds = _diff_ds(ds).assign(ocean_mask=lambda d: xr.ones_like(d["ocean_mask"]))
    land = _latlon(_land_point(ds))
    events = _events(land, (np.nan, np.nan))

    with pytest.warns(UserWarning, match="release probability"):
        emission, _ = helpers.compute_emission_pdf(
            diff_ds=diff_ds,
            events_ds=events,
            differences_std=1.0,
            initial_std=1e-5,
            recapture_std=1e-5,
        )
    assert "final" not in emission


def test_compute_emission_pdf_release_in_ocean_is_silent():
    ds = _grid()
    ocean = _latlon(_ocean_point(ds))
    events = _events(ocean, (np.nan, np.nan))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        helpers.compute_emission_pdf(
            diff_ds=_diff_ds(ds),
            events_ds=events,
            differences_std=1.0,
            initial_std=1e-5,
            recapture_std=1e-5,
        )
    assert not [w for w in record if "probability" in str(w.message)]
