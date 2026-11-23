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


def write_healpix_zarr(ds, path):
    ds.to_zarr(path, mode="w", zarr_format=2, consolidated=True)


def grid_attrs(ds):
    attrs = ds["cell_ids"].attrs
    return {
        k: attrs.get(k) for k in ("grid_name", "level", "indexing_scheme", "ellipsoid")
    }


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
    # the complete demo merges it into total_pdf after dropping this index
    assert "cell_ids" in pdf.indexes
    pdf.drop_indexes(["cell_ids"])

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


def test_depth_offset_is_a_clearance_above_the_seabed():
    seabed = _flat_seabed(40.0)
    kwargs = dict(nside=2**8, max_depth_m=200, depth_bin_size=1)
    plain = compute_healpix_histogram_region_bin_size(seabed, **kwargs)
    above = compute_healpix_histogram_region_bin_size(seabed, depth_offset=10, **kwargs)
    below = compute_healpix_histogram_region_bin_size(
        seabed, depth_offset=-10, **kwargs
    )
    assert above.attrs["depth_offset"] == 10
    # 35 m is possible on a 40 m seabed, not with 10 m clearance
    assert _likelihood(plain, 35.0) > 0.99 and _likelihood(above, 35.0) < 0.01
    assert _likelihood(above, 25.0) > 0.99
    # 45 m is below a 40 m seabed, possible if 10 m below it are allowed
    assert _likelihood(plain, 45.0) < 0.01 and _likelihood(below, 45.0) > 0.99
    assert _likelihood(below, 55.0) < 0.01


@pytest.mark.parametrize("bin_size", [1, 16])
def test_stdev_is_in_metres_and_centred_on_the_depth(bin_size):
    from scipy.stats import norm

    hist = compute_healpix_histogram_region_bin_size(
        _flat_seabed(100.0, stdev=10.0),
        nside=2**8,
        max_depth_m=300,
        depth_bin_size=bin_size,
    )
    for depth in [85.0, 95.0, 100.0, 105.0, 115.0]:
        expected = norm.sf(depth, loc=100.0, scale=10.0)
        tolerance = 0.01 if bin_size == 1 else 0.06
        assert _likelihood(hist, depth) == pytest.approx(expected, abs=tolerance)


def test_linear_between_bin_edges():
    hist = compute_healpix_histogram_region_bin_size(
        _flat_seabed(100.0, stdev=20.0), nside=2**8, max_depth_m=300, depth_bin_size=16
    )
    edges = hist.depth_bins.values
    k = int(np.searchsorted(edges, 96.0))
    low, high = edges[k - 1], edges[k]
    middle = (low + high) / 2
    assert _likelihood(hist, middle) == pytest.approx(
        (_likelihood(hist, low) + _likelihood(hist, high)) / 2, rel=1e-9
    )


def test_land_points_are_not_seabed():
    lat = np.linspace(47.0, 47.4, 24)
    lon = np.linspace(-5.0, -4.6, 24)
    elevation = np.full((24, 24), -80.0)
    elevation[:, np.arange(24) % 5 < 2] = 25.0  # 40 % land in every cell
    ds = xr.Dataset(
        {"elevation": (("latitude", "longitude"), elevation)},
        coords={"latitude": lat, "longitude": lon},
    )
    hist = compute_healpix_histogram_region_bin_size(
        ds, nside=2**8, max_depth_m=200, depth_bin_size=4
    )
    for cell in range(hist.sizes["cells"]):
        assert _likelihood(hist.isel(cells=[cell]), 50.0) > 0.99


def test_deepest_record_of_the_time_step_counts(tmp_path):
    hist, reference, tag = _pdf_inputs(tmp_path)
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    shallow = tag.assign(pressure=tag.pressure * 0 + 5.0)
    one_deep = shallow.copy(deep=True)
    one_deep["pressure"][:, 0] = 190.0  # first record deep, the others shallow
    pdf_shallow = batch_compute_pdf_bathy(hist, shallow, str(tmp_path))
    pdf_deep = batch_compute_pdf_bathy(hist, one_deep, str(tmp_path))
    assert float(pdf_deep.pdf_bathy.max()) < 0.01 < float(pdf_shallow.pdf_bathy.min())


def test_time_step_without_record_carries_no_information(tmp_path):
    hist, reference, tag = _pdf_inputs(tmp_path)
    reference["XE"][:, 0] = np.nan  # land
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    tag["pressure"][1, :] = np.nan
    pdf = batch_compute_pdf_bathy(hist, tag, str(tmp_path))
    on_sea = pdf.pdf_bathy.isel(time=1, cells=slice(1, None))
    np.testing.assert_array_equal(on_sea, 1.0)
    assert pdf.pdf_bathy.isel(cells=0).isnull().all()


def test_old_histograms_are_refused(tmp_path):
    hist, reference, tag = _pdf_inputs(tmp_path)
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    old = hist.assign_attrs(bathy_histogram_version=None)
    with pytest.raises(ValueError, match="older pangeo-fish"):
        batch_compute_pdf_bathy(old, tag, str(tmp_path))


def test_model_cells_without_gebco_use_the_nearest_cell(tmp_path):
    hist, reference, tag = _pdf_inputs(tmp_path)
    reference["XE"][:, 0] = np.nan  # land
    write_healpix_zarr(reference, str(tmp_path / "diff-regridded.zarr"))
    full = batch_compute_pdf_bathy(hist, tag, str(tmp_path))
    partial = hist.drop_isel(cells=[1, 2])
    with pytest.warns(UserWarning, match="2 cell.*nearest cell"):
        pdf = batch_compute_pdf_bathy(partial, tag, str(tmp_path))
    np.testing.assert_array_equal(pdf.cell_ids, reference.cell_ids)
    assert pdf.pdf_bathy.isel(cells=[1, 2]).notnull().all()
    assert (pdf.pdf_bathy.isel(cells=[1, 2]) < 1).all()
    xr.testing.assert_equal(
        pdf.pdf_bathy.isel(cells=slice(3, None)),
        full.pdf_bathy.isel(cells=slice(3, None)),
    )
    assert pdf.pdf_bathy.isel(cells=0).isnull().all()
