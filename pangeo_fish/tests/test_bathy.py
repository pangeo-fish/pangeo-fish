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


# --- the likelihood itself -------------------------------------------------


def _flat_seabed(depth_m, stdev=None):
    lat = np.linspace(47.0, 47.4, 24)
    lon = np.linspace(-5.0, -4.6, 24)
    data = {"elevation": (("latitude", "longitude"), np.full((24, 24), -depth_m))}
    if stdev is not None:
        data["stdev"] = (("latitude", "longitude"), np.full((24, 24), stdev))
    return xr.Dataset(data, coords={"latitude": lat, "longitude": lon})


def _likelihood(hist, fish_depth, xe=0.0):
    """Likelihood of the first cell for one time step with one observation."""
    from pangeo_fish.bathy import compute_pdf_bathy_numba_like_numpy

    values = np.ascontiguousarray(hist.bathy_pixel_hist.values[:1])
    out = compute_pdf_bathy_numba_like_numpy(
        values,
        np.array([[[float(fish_depth)]]]),
        np.array([[float(xe)]]),
        hist.depth_bins.values.astype("float64"),
    )
    return float(out[0, 0])


@pytest.mark.parametrize("bin_size", [1, 4, 16])
@pytest.mark.parametrize("seabed", [20.0, 80.0, 150.0])
def test_fish_shallower_than_the_seabed_only(bin_size, seabed):
    """~1 well above the seabed, ~0 well below it, whatever the bin width."""
    hist = compute_healpix_histogram_region_bin_size(
        _flat_seabed(seabed), nside=2**8, max_depth_m=400, depth_bin_size=bin_size
    )
    margin = 2 * bin_size + 3
    assert _likelihood(hist, seabed - margin) > 0.95
    assert _likelihood(hist, seabed + margin) < 0.05
    # deeper is never more likely
    depths = np.arange(0.0, 300.0, 5.0)
    values = [_likelihood(hist, d) for d in depths]
    assert np.all(np.diff(values) <= 1e-12)


def test_depth_bins_are_the_depths_of_the_exceedance():
    hist = compute_healpix_histogram_region_bin_size(
        _flat_seabed(100.0), nside=2**8, max_depth_m=400, depth_bin_size=16
    )
    edges = hist.depth_bins.values
    assert edges[0] == -160 and edges[-1] == 400
    assert hist.depth_bins.attrs["units"] == "m"
    row = hist.bathy_pixel_hist.values[0]
    assert row[0] == pytest.approx(1.0)
    assert np.all(row[edges <= 80] > 0.99) and np.all(row[edges >= 128] < 0.01)


def test_sea_surface_height_moves_the_fish_relative_to_the_seabed():
    hist = compute_healpix_histogram_region_bin_size(
        _flat_seabed(50.0), nside=2**8, max_depth_m=200, depth_bin_size=1
    )
    # 54 dbar below a surface raised by 6 m is 48 m below the mean sea level
    assert _likelihood(hist, 54.0, xe=6.0) == pytest.approx(_likelihood(hist, 48.0))
    assert _likelihood(hist, 54.0, xe=6.0) > 0.9 > _likelihood(hist, 54.0)


def test_depth_offset_is_applied_to_both_sides():
    seabed = _flat_seabed(60.0)
    plain = compute_healpix_histogram_region_bin_size(
        seabed, nside=2**8, max_depth_m=200, depth_bin_size=4
    )
    shifted = compute_healpix_histogram_region_bin_size(
        seabed, nside=2**8, max_depth_m=200, depth_bin_size=4, depth_offset=2
    )
    # only the position of the bin edges changes
    for depth in [30.0, 45.0, 75.0, 90.0]:
        assert _likelihood(shifted, depth) == pytest.approx(
            _likelihood(plain, depth), abs=0.01
        )
    for hist in (plain, shifted):
        assert _likelihood(hist, 50.0) > 0.9 > 0.1 > _likelihood(hist, 70.0)


def test_stdev_is_in_metres():
    narrow = compute_healpix_histogram_region_bin_size(
        _flat_seabed(100.0, stdev=1.0), nside=2**8, max_depth_m=300, depth_bin_size=1
    )
    wide = compute_healpix_histogram_region_bin_size(
        _flat_seabed(100.0, stdev=10.0), nside=2**8, max_depth_m=300, depth_bin_size=1
    )
    assert _likelihood(narrow, 90.0) > 0.99
    assert 0.6 < _likelihood(wide, 95.0) < 0.8  # P(N(100, 10) >= 95) ~ 0.69


def test_beyond_the_last_depth():
    shallow = compute_healpix_histogram_region_bin_size(
        _flat_seabed(100.0), nside=2**8, max_depth_m=200, depth_bin_size=10
    )
    deep = compute_healpix_histogram_region_bin_size(
        _flat_seabed(500.0), nside=2**8, max_depth_m=200, depth_bin_size=10
    )
    assert _likelihood(shallow, 350.0) < 0.01
    assert _likelihood(deep, 350.0) > 0.99


def test_histogram_is_not_modified():
    from pangeo_fish.bathy import compute_pdf_bathy_numba_like_numpy

    hist = compute_healpix_histogram_region_bin_size(
        _flat_seabed(50.0), nside=2**8, max_depth_m=100, depth_bin_size=4
    )
    values = np.ascontiguousarray(hist.bathy_pixel_hist.values[:2])
    before = values.copy()
    compute_pdf_bathy_numba_like_numpy(
        values,
        np.full((1, 2, 1), 30.0),
        np.array([[np.nan, 0.0]]),
        hist.depth_bins.values.astype("float64"),
    )
    np.testing.assert_array_equal(values, before)


def _pdf_inputs(tmp_path, n_time=3, depth=60.0):
    hist = compute_healpix_histogram_region_bin_size(
        _bathymetry(),
        nside=2**LEVEL,
        max_depth_m=200,
        depth_bin_size=16,
        ellipsoid="WGS84",
    )
    reference = _reference(hist.cell_ids.values, "WGS84", n_time=n_time)
    tag = xr.Dataset(
        {"pressure": (("time", "obs"), np.full((n_time, 4), depth))},
        coords={"time": reference.time},
    )
    return hist, reference, tag


def test_pressure_in_pa_is_refused(tmp_path):
    hist, reference, tag = _pdf_inputs(tmp_path, depth=60.0e4)
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    with pytest.raises(ValueError, match="in Pa"):
        batch_compute_pdf_bathy(hist, tag, str(tmp_path))


def test_tag_time_steps_must_be_in_the_model(tmp_path):
    hist, reference, tag = _pdf_inputs(tmp_path)
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    tag = tag.assign_coords(time=tag.time + pd.Timedelta("365D"))
    with pytest.raises(ValueError, match="not in diff-regridded"):
        batch_compute_pdf_bathy(hist, tag, str(tmp_path))


def test_model_cells_without_gebco_carry_no_information(tmp_path):
    hist, reference, tag = _pdf_inputs(tmp_path)
    reference["XE"][:, 0] = np.nan  # land
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    missing = hist.isel(cells=slice(1, 3)).cell_ids.values
    partial = hist.isel(cells=slice(3, None)).assign_attrs(hist.attrs)
    partial = xr.concat([hist.isel(cells=[0]), partial], dim="cells")
    with pytest.warns(UserWarning, match="2 cell"):
        pdf = batch_compute_pdf_bathy(partial, tag, str(tmp_path))
    np.testing.assert_array_equal(pdf.cell_ids, reference.cell_ids)
    on_missing = pdf.pdf_bathy.where(pdf.cell_ids.isin(missing), drop=True)
    np.testing.assert_array_equal(on_missing, 1.0)
    assert pdf.pdf_bathy.isel(cells=0).isnull().all()
