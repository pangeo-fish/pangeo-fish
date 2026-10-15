import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr
import zarr
from pint_xarray import unit_registry as ureg

from pangeo_fish import helpers


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
    assert decoded.dggs.grid_info.ellipsoid == "WGS84"
    assert decoded["cell_ids"].attrs["ellipsoid"] == "WGS84"


def test_save_zarr_writes_dggs_attrs(tmp_path):
    ds = helpers.to_healpix(_healpix_dataset(ellipsoid="WGS84"))
    path = str(tmp_path / "pdf")

    helpers._save_zarr(ds, path)

    attrs = dict(zarr.open_group(path + ".zarr", mode="r")["cell_ids"].attrs)
    assert attrs["grid_name"] == "healpix"
    assert attrs["level"] == 2
    assert attrs["indexing_scheme"] == "nested"
    assert attrs["ellipsoid"] == "WGS84"

    reopened = xr.open_zarr(path + ".zarr")
    assert helpers.to_healpix(reopened).dggs.grid_info.ellipsoid == "WGS84"


def test_save_zarr_without_dggs_index(tmp_path):
    ds = _healpix_dataset(ellipsoid="sphere")
    del ds["cell_ids"].attrs["grid_name"]
    path = str(tmp_path / "pdf")

    helpers._save_zarr(ds, path)

    attrs = dict(zarr.open_group(path + ".zarr", mode="r")["cell_ids"].attrs)
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
