"""The bathymetry pdf is written as a regular HEALPix store."""

import healpix_geo.nested as hgn
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs  # noqa: F401

from pangeo_fish.bathy import (
    batch_compute_pdf_bathy,
    compute_healpix_histogram_region_bin_size,
)
from pangeo_fish.dggs import grid_attrs
from pangeo_fish.storage import write_healpix_zarr

LEVEL = 6


def _bathymetry():
    lat = np.linspace(47.0, 49.0, 40)
    lon = np.linspace(-6.0, -3.0, 50)
    elevation = -50.0 - 10.0 * np.add.outer(lat - 47.0, lon + 6.0)
    return xr.Dataset(
        {"elevation": (("latitude", "longitude"), elevation)},
        coords={"latitude": lat, "longitude": lon},
    )


def _reference(cell_ids, ellipsoid, n_time=3):
    return xr.Dataset(
        {"XE": (("time", "cells"), np.zeros((n_time, cell_ids.size)))},
        coords={
            "cell_ids": (
                "cells",
                cell_ids,
                {
                    "grid_name": "healpix",
                    "level": LEVEL,
                    "indexing_scheme": "nested",
                    "ellipsoid": ellipsoid,
                },
            ),
            "time": pd.date_range("2024-01-01", periods=n_time),
        },
    )


@pytest.mark.parametrize("ellipsoid", ["sphere", "WGS84"])
def test_histogram_grid_and_cell_centres(ellipsoid):
    hist = compute_healpix_histogram_region_bin_size(
        _bathymetry(),
        nside=2**LEVEL,
        max_depth_m=200,
        depth_bin_size=16,
        ellipsoid=ellipsoid,
    )
    assert grid_attrs(hist) == {
        "grid_name": "healpix",
        "level": LEVEL,
        "indexing_scheme": "nested",
        "ellipsoid": ellipsoid,
    }
    lon, lat = hgn.healpix_to_lonlat(
        hist.cell_ids.values.astype("uint64"), LEVEL, ellipsoid=ellipsoid
    )
    np.testing.assert_allclose(hist.latitude, lat)
    np.testing.assert_allclose(hist.longitude, lon)


def test_bathy_pdf_store_decodes(tmp_path):
    hist = compute_healpix_histogram_region_bin_size(
        _bathymetry(),
        nside=2**LEVEL,
        max_depth_m=200,
        depth_bin_size=16,
        ellipsoid="WGS84",
    )
    reference = _reference(hist.cell_ids.values, "WGS84")
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    tag = xr.Dataset(
        {"pressure": (("time", "obs"), np.full((3, 4), 60.0))},
        coords={"time": reference.time},
    )
    pdf = batch_compute_pdf_bathy(hist, tag, str(tmp_path), batch_size=7)
    assert grid_attrs(pdf) == grid_attrs(reference)

    path = str(tmp_path / "bathy_pdf.zarr")
    write_healpix_zarr((pdf / pdf.sum("cells")).compute(), path)
    decoded = xr.open_zarr(path).dggs.decode()
    assert decoded.dggs.grid_info.level == LEVEL


def test_histogram_on_another_grid_is_refused(tmp_path):
    hist = compute_healpix_histogram_region_bin_size(
        _bathymetry(),
        nside=2**LEVEL,
        max_depth_m=200,
        depth_bin_size=16,
        ellipsoid="sphere",
    )
    write_healpix_zarr(
        _reference(hist.cell_ids.values, "WGS84"),
        str(tmp_path / "diff-regridded.zarr"),
    )
    tag = xr.Dataset({"pressure": (("time", "obs"), np.full((3, 4), 60.0))})
    with pytest.raises(ValueError, match="different grids"):
        batch_compute_pdf_bathy(hist, tag, str(tmp_path))
