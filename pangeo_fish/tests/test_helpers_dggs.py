import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr
import zarr
from healpix_geo.ellipsoid import resolve
from pint_xarray import unit_registry as ureg

from pangeo_fish import helpers


def _assert_wgs84(ellipsoid):
    # xdggs 0.6 can retain a name; 0.7 resolves it to numeric parameters.
    if isinstance(ellipsoid, str):
        ellipsoid = resolve(ellipsoid)
    elif hasattr(ellipsoid, "to_dict"):
        ellipsoid = ellipsoid.to_dict()
    assert ellipsoid["semimajor_axis"] == pytest.approx(6378137.0)
    assert ellipsoid["inverse_flattening"] == pytest.approx(298.257223563)


def _healpix_dataset(level=2, ellipsoid="sphere"):
    nside = 2**level
    cell_ids = np.arange(12 * nside**2, dtype="uint64")
    ds = xr.Dataset(
        {"pdf": (("time", "cells"), np.ones((3, cell_ids.size)))},
        coords={
            "cell_ids": ("cells", cell_ids),
            "time": pd.date_range("2024-01-01", periods=3, freq="1h"),
        },
    )
    ds["cell_ids"].attrs.update(
        grid_name="healpix", level=level, indexing_scheme="nested", ellipsoid=ellipsoid
    )
    return ds


@pytest.mark.parametrize(
    ["ellipsoid", "expected_km"],
    (
        ("sphere", 6370.997),
        ("WGS84", 6378.137),
        (None, 6370.997),
        ({"radius": 6371000.0}, 6371.0),
        ({"semimajor_axis": 6378137.0, "inverse_flattening": 298.257223563}, 6378.137),
    ),
)
def test_earth_radius_from_ellipsoid(ellipsoid, expected_km):
    radius = helpers.earth_radius_from_ellipsoid(ellipsoid)

    assert radius.units == pint.Unit("km")
    assert radius.magnitude == pytest.approx(expected_km)


def test_earth_radius_from_ellipsoid_unknown():
    with pytest.raises(ValueError):
        helpers.earth_radius_from_ellipsoid({"name": "no-size"})


def test_to_healpix_keeps_ellipsoid():
    decoded = helpers.to_healpix(_healpix_dataset(ellipsoid="WGS84"))

    assert decoded.dggs.grid_info.level == 2
    _assert_wgs84(decoded.dggs.grid_info.ellipsoid)
    _assert_wgs84(decoded["cell_ids"].attrs["ellipsoid"])


def test_save_zarr_writes_dggs_attrs(tmp_path):
    ds = helpers.to_healpix(_healpix_dataset(ellipsoid="WGS84"))
    path = str(tmp_path / "pdf")

    helpers._save_zarr(ds, path)

    group = zarr.open_group(path + ".zarr", mode="r")
    attrs = dict(group["cell_ids"].attrs)
    assert group.attrs["dggs"]["coordinate"] == "cell_ids"
    assert group.attrs["dggs"]["refinement_level"] == 2
    assert group["pdf"].chunks == (3, 192)
    assert attrs["grid_name"] == "healpix"
    assert attrs["level"] == 2
    assert attrs["indexing_scheme"] == "nested"
    _assert_wgs84(attrs["ellipsoid"])

    reopened = xr.open_zarr(path + ".zarr")
    _assert_wgs84(helpers.to_healpix(reopened).dggs.grid_info.ellipsoid)


def test_save_zarr_without_dggs_index(tmp_path):
    ds = _healpix_dataset(ellipsoid="sphere")
    del ds["cell_ids"].attrs["grid_name"]
    path = str(tmp_path / "pdf")

    helpers._save_zarr(ds, path)

    group = zarr.open_group(path + ".zarr", mode="r")
    attrs = dict(group["cell_ids"].attrs)
    assert group.attrs["dggs"]["coordinate"] == "cell_ids"
    assert group.attrs["dggs"]["refinement_level"] == 2
    assert group["pdf"].chunks == (3, 192)
    assert attrs["grid_name"] == "healpix"
    assert attrs["ellipsoid"] == "sphere"


def test_get_max_sigma_derives_earth_radius():
    ds = helpers.to_healpix(_healpix_dataset(ellipsoid="sphere"))
    speed = ureg.Quantity(60, "km / h")

    derived = helpers._get_max_sigma(ds, None, 1.0, speed, as_radians=True)
    explicit = helpers._get_max_sigma(
        ds, ureg.Quantity(6371, "km"), 1.0, speed, as_radians=True
    )

    assert derived == pytest.approx(60 / 6370.997)
    assert derived == pytest.approx(explicit, rel=1e-4)


def test_get_max_sigma_without_resolution_coordinate():
    ds = helpers.to_healpix(_healpix_dataset(ellipsoid="sphere"))
    speed = ureg.Quantity(60, "km / h")

    in_pixels = helpers._get_max_sigma(ds, None, 1.0, speed, as_radians=False)

    cell_size = 6370.997 * np.sqrt(4 * np.pi / (12 * 4**2))
    assert in_pixels == pytest.approx(60 / cell_size)


def test_with_dggs_attrs_from_index_only():
    ds = helpers.to_healpix(_healpix_dataset(ellipsoid="WGS84"))
    ds["cell_ids"].attrs.clear()

    attrs = helpers._with_dggs_attrs(ds)["cell_ids"].attrs

    assert attrs["level"] == 2
    _assert_wgs84(attrs["ellipsoid"])


def test_compute_acoustic_pdf_writes_dggs_attrs():
    from pangeo_fish.tests.test_acoustic import dummy_grid, dummy_tag

    grid = dummy_grid.__wrapped__().drop_indexes("cell_ids")
    grid["cell_ids"].attrs.update(
        grid_name="healpix", level=1, indexing_scheme="nested", ellipsoid="WGS84"
    )

    acoustic_pdf, _ = helpers.compute_acoustic_pdf(
        emission_ds=grid,
        tag=dummy_tag.__wrapped__(),
        receiver_buffer=ureg.Quantity(1000, "m"),
        chunk_time=1,
    )

    attrs = acoustic_pdf["cell_ids"].attrs
    assert attrs["grid_name"] == "healpix"
    assert attrs["level"] == 1
    assert attrs["indexing_scheme"] == "nested"
    _assert_wgs84(attrs["ellipsoid"])


def test_compute_emission_pdf_from_store_split_along_cells(tmp_path):
    """A store whose cells span several chunks must not break the emission pdf."""
    from pangeo_fish.storage import prepare_dggs_zarr

    ds = _healpix_dataset(level=2, ellipsoid="sphere")
    rng = np.random.default_rng(0)
    diff = rng.normal(size=ds["pdf"].shape)
    diff[0, :5] = np.nan
    ds = ds.drop_vars("pdf").assign(
        diff=(("time", "cells"), diff),
        ocean_mask=("cells", np.ones(ds.sizes["cells"], dtype=bool)),
        latitude=("cells", np.zeros(ds.sizes["cells"])),
        longitude=("cells", np.zeros(ds.sizes["cells"])),
    )
    path = str(tmp_path / "diff-regridded.zarr")
    prepare_dggs_zarr(ds, cell_chunk_size=50).to_zarr(path, zarr_format=2)
    reopened = xr.open_dataset(path, engine="zarr", chunks={})
    assert len(reopened["diff"].chunks[1]) > 1

    events = xr.Dataset(
        {
            "longitude": ("event_name", [10.0, np.nan]),
            "latitude": ("event_name", [10.0, np.nan]),
        },
        coords={"event_name": ["release", "fish_death"]},
    )
    emission, _ = helpers.compute_emission_pdf(
        diff_ds=reopened,
        events_ds=events,
        differences_std=0.5,
        initial_std=1e-3,
        recapture_std=1e-3,
        chunk_time=1,
    )

    expected = np.exp(-0.5 * (diff / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    assert emission["pdf"].dims == ("time", "cells")
    np.testing.assert_allclose(emission["pdf"].values, expected)
    assert emission["pdf"].chunks[1] == (ds.sizes["cells"],)
