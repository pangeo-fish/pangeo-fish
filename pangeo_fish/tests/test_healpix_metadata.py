"""The HEALPix grid and CF metadata are described once and survive every step."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from pangeo_fish import helpers
from pangeo_fish.dggs import (
    CF_CONVENTIONS,
    canonical_ellipsoid,
    grid_attrs,
    grid_ellipsoid,
    with_cf_attrs,
    with_grid_attrs,
)
from pangeo_fish.storage import open_healpix_zarr, write_healpix_zarr

WGS84 = {"semimajor_axis": 6378137.0, "inverse_flattening": 298.257223563}


class _EllipsoidObject:
    """Stands for xdggs 0.7's ``Ellipsoid``: only ``to_dict`` is used."""

    def __init__(self, params):
        self._params = params

    def to_dict(self):
        return dict(self._params)


@pytest.mark.parametrize(
    "ellipsoid",
    [
        "WGS84",
        {"name": "WGS84"} | WGS84,
        {
            "name": "WGS84",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
        _EllipsoidObject({"name": "WGS84"} | WGS84),
    ],
)
def test_named_ellipsoid_forms_give_the_name(ellipsoid):
    assert canonical_ellipsoid(ellipsoid) == "WGS84"


def test_name_case_and_nameless_parameters():
    assert canonical_ellipsoid("wgs84") == "WGS84"
    assert canonical_ellipsoid({"name": "wgs84"}) == "WGS84"
    assert canonical_ellipsoid(dict(WGS84)) == "WGS84"
    assert (
        canonical_ellipsoid(_EllipsoidObject(dict(WGS84) | {"name": None})) == "WGS84"
    )


def test_sphere_forms():
    assert canonical_ellipsoid(None) == "sphere"
    assert canonical_ellipsoid("sphere") == "sphere"
    assert canonical_ellipsoid({"name": "sphere", "radius": 6370997.0}) == "sphere"


def test_custom_ellipsoid_stays_numeric():
    assert canonical_ellipsoid({"radius": 6371000.0}) == {"radius": 6371000.0}
    # A known name with other parameters is not that ellipsoid.
    custom = {"name": "WGS84", "semimajor_axis": 6378000.0, "inverse_flattening": 300.0}
    assert canonical_ellipsoid(custom) == {
        "semimajor_axis": 6378000.0,
        "inverse_flattening": 300.0,
    }


@pytest.mark.parametrize("bad", ["not-an-ellipsoid", {"name": "unknown"}, 42])
def test_invalid_ellipsoid(bad):
    with pytest.raises((ValueError, TypeError)):
        canonical_ellipsoid(bad)


def _maps(level=2, ellipsoid=None, n_time=3):
    cell_ids = np.arange(12 * 4**level, dtype="int64")
    ds = xr.Dataset(
        {
            "pdf": (
                ("time", "cells"),
                np.full((n_time, cell_ids.size), 1 / cell_ids.size),
            ),
            "mask": ("cells", np.ones(cell_ids.size, dtype=bool)),
            "latitude": ("cells", np.zeros(cell_ids.size)),
            "longitude": ("cells", np.zeros(cell_ids.size)),
        },
        coords={
            "cell_ids": ("cells", cell_ids),
            "time": pd.date_range("2024-01-01", periods=n_time, freq="1D"),
        },
    )
    ds["cell_ids"].attrs.update(
        grid_name="healpix", level=level, indexing_scheme="nested"
    )
    if ellipsoid is not None:
        ds["cell_ids"].attrs["ellipsoid"] = ellipsoid
    return ds


def test_decoded_dataset_keeps_canonical_attrs():
    decoded = helpers.to_healpix(_maps(ellipsoid={"name": "WGS84"} | WGS84))
    assert decoded["cell_ids"].attrs["ellipsoid"] == "WGS84"
    assert grid_ellipsoid(decoded) == "WGS84"
    assert grid_attrs(decoded) == {
        "grid_name": "healpix",
        "level": 2,
        "indexing_scheme": "nested",
        "ellipsoid": "WGS84",
    }


def test_grid_ellipsoid_defaults_to_sphere():
    assert grid_ellipsoid(_maps()) == "sphere"
    assert grid_ellipsoid(np.arange(3)) == "sphere"


def test_cf_attrs_do_not_override():
    ds = _maps()
    ds["pdf"].attrs["long_name"] = "my emission"
    out = with_cf_attrs(ds)
    assert out["pdf"].attrs == {"long_name": "my emission", "units": "1"}
    assert out["latitude"].attrs["units"] == "degrees_north"
    assert out.attrs["Conventions"] == CF_CONVENTIONS
    assert "long_name" not in ds["latitude"].attrs  # input unchanged


def test_write_and_open_round_trip(tmp_path):
    path = str(tmp_path / "combined.zarr")
    source = helpers.to_healpix(_maps(ellipsoid="WGS84"))
    write_healpix_zarr(source, path)

    group = zarr.open_group(path, mode="r")
    assert group.metadata.zarr_format == 2
    assert group.attrs["Conventions"] == CF_CONVENTIONS
    assert group.attrs["dggs"]["refinement_level"] == 2
    assert group.attrs["dggs"]["ellipsoid"]["name"] == "WGS84"
    assert dict(group["cell_ids"].attrs)["ellipsoid"] == "WGS84"
    assert group["pdf"].attrs["units"] == "1"
    assert group["latitude"].attrs["standard_name"] == "latitude"

    # xdggs reads the grid from the attributes of ``cell_ids`` alone
    assert xr.open_zarr(path).dggs.decode().dggs.grid_info.level == 2
    assert set(group["cell_ids"].attrs) <= {
        "grid_name",
        "level",
        "indexing_scheme",
        "ellipsoid",
        "_ARRAY_DIMENSIONS",
    }

    reopened = open_healpix_zarr(path, chunks={"time": 1, "cells": -1, "x": 3})
    assert reopened["pdf"].chunks == ((1, 1, 1), (192,))
    assert reopened["cell_ids"].attrs["ellipsoid"] == "WGS84"
    np.testing.assert_array_equal(reopened["pdf"].values, source["pdf"].values)
    # the reopened store decodes and is written again unchanged
    write_healpix_zarr(helpers.to_healpix(reopened), str(tmp_path / "again.zarr"))


def test_open_normalizes_stores_with_numeric_ellipsoid(tmp_path):
    """Stores written before the canonical form carry the ellipsoid as a dict."""
    path = str(tmp_path / "old.zarr")
    _maps(ellipsoid={"name": "WGS84"} | WGS84).to_zarr(path, zarr_format=2)
    reopened = open_healpix_zarr(path)
    assert reopened["cell_ids"].attrs["ellipsoid"] == "WGS84"


def test_write_refuses_existing_store_with_mode_w_minus(tmp_path):
    path = str(tmp_path / "s.zarr")
    write_healpix_zarr(_maps(), path)
    with pytest.raises(Exception):
        write_healpix_zarr(_maps(), path, mode="w-")


def test_save_zarr_raises_instead_of_warning(tmp_path):
    ds = _maps()
    ds["cell_ids"].attrs["level"] = 1  # cell ids beyond level 1: invalid grid
    with pytest.raises(ValueError):
        helpers._save_zarr(ds, str(tmp_path / "bad"))


def test_with_grid_attrs_without_grid_is_a_no_op():
    ds = xr.Dataset({"a": ("x", [1, 2])})
    assert with_grid_attrs(ds) is ds


def test_write_keeps_only_grid_attrs_on_cell_ids(tmp_path):
    """Extra keys (e.g. the acoustic ``lat``/``lon``) would break ``xdggs`` decode."""
    path = str(tmp_path / "acoustic.zarr")
    source = helpers.to_healpix(_maps(ellipsoid="WGS84"))
    source["cell_ids"].attrs.update(lat=0, lon=0)
    write_healpix_zarr(source, path)
    assert set(zarr.open_group(path, mode="r")["cell_ids"].attrs) <= {
        "grid_name",
        "level",
        "indexing_scheme",
        "ellipsoid",
        "_ARRAY_DIMENSIONS",
    }
    assert xr.open_zarr(path).dggs.decode().dggs.grid_info.level == 2
    assert source["cell_ids"].attrs["lat"] == 0  # input unchanged
