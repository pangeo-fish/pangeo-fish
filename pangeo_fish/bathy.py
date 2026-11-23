"""Bathymetry likelihood: the fish cannot be deeper than the seabed.

For every HEALPix cell, :py:func:`compute_healpix_histogram_region_bin_size`
summarises the GEBCO depths that fall in the cell as the probability that the
seabed is at least ``d`` metres deep, for depths ``d`` on a regular grid
(``depth_bins``). For every time step, :py:func:`batch_compute_pdf_bathy` takes
the deepest depth of the fish below the model sea surface, ``max(P) - XE``, and
reads that probability at this depth (linear between grid depths).
"""

import warnings

import healpix_geo.nested as hpg_n
import numpy as np
import xarray as xr
from numba import njit, prange
from tqdm import tqdm

from pangeo_fish.dggs import canonical_ellipsoid, grid_attrs
from pangeo_fish.storage import open_healpix_zarr

#: Number of depth bins kept above ``depth_offset`` (land and intertidal points).
_ABOVE_BINS = 10
#: Default vertical uncertainty of the GEBCO depths (m) when there is no ``stdev``.
_DEFAULT_STDEV_M = 1.0
#: A pressure above this value (dbar) is deeper than any ocean: likely in Pa.
_MAX_PRESSURE_DBAR = 12000.0


def compute_healpix_histogram_region_bin_size(
    ds,
    nside,
    nb_depth_bins=None,
    max_depth_m=None,
    chunk_size=500,
    depth_offset=0,
    depth_bin_size=1,
    ellipsoid="sphere",
):
    """Probability that the seabed of each HEALPix cell is at least ``d`` deep.

    Parameters
    ----------
    ds : xarray.Dataset
        ``elevation`` on (latitude, longitude), in metres, negative below sea
        level (GEBCO), and optionally ``stdev`` (m), its vertical uncertainty.
    nside : int
        HEALPix resolution (power of 2).
    nb_depth_bins : int or None
        Number of depth bins of ``depth_bin_size`` metres below ``depth_offset``.
    max_depth_m : float or None
        Used when ``nb_depth_bins`` is None: ``ceil(max_depth_m / depth_bin_size)``
        bins. Seabeds deeper than this are counted as "deeper than max_depth_m".
    chunk_size : int
        Number of latitude rows read at once.
    depth_offset : float
        Depth (m) of the first bin edge below the surface; ten more bins are kept
        above it.
    depth_bin_size : float
        Bin width in metres.
    ellipsoid : str or dict
        Ellipsoid of the HEALPix grid (the one of ``diff-regridded.zarr``).

    Returns
    -------
    xarray.Dataset
        ``bathy_pixel_hist`` on (cells, depth_bins): the probability that the
        seabed is at least ``depth_bins`` metres deep (1 at the shallowest depth,
        decreasing with depth). The last depth is ``depth_offset + nb_depth_bins
        * depth_bin_size``.
    """
    if nb_depth_bins is None:
        if max_depth_m is None:
            raise ValueError("Either nb_depth_bins or max_depth_m must be provided.")
        nb_depth_bins = int(np.ceil(float(max_depth_m) / float(depth_bin_size)))
    else:
        nb_depth_bins = int(nb_depth_bins)
    if nb_depth_bins <= 0:
        raise ValueError("nb_depth_bins must be > 0")
    bin_size = float(depth_bin_size)
    if not bin_size > 0:
        raise ValueError("depth_bin_size must be > 0")

    # Bin j holds depths in [edge_j, edge_j+1); the last bin also holds every
    # deeper seabed. edge_j = depth_offset + (j - _ABOVE_BINS) * bin_size.
    n_edges = nb_depth_bins + _ABOVE_BINS + 1
    edges = depth_offset + (np.arange(n_edges) - _ABOVE_BINS) * bin_size
    n_bins = n_edges  # the last bin is the overflow bin

    lats = ds.latitude.values
    lons = ds.longitude.values
    nlat, nlon = len(lats), len(lons)
    level = int(np.log2(nside))

    # --- PASS 1: HEALPix cells that contain GEBCO points ---
    used_cells = set()
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 1 – collect cells"):
        j = min(i + chunk_size, nlat)
        elev = ds.elevation[i:j, :].load().values.flatten()
        valid = ~np.isnan(elev)
        if not valid.any():
            continue
        lat_blk = np.repeat(lats[i:j], nlon)[valid]
        lon_blk = np.tile(lons, j - i)[valid]
        hidx = hpg_n.lonlat_to_healpix(
            lon_blk, lat_blk, depth=level, ellipsoid=ellipsoid
        )
        used_cells.update(np.unique(hidx))

    used_cells = np.array(sorted(used_cells), dtype=np.int64)
    hist = np.zeros((used_cells.size, n_bins), dtype=np.float64)

    # --- PASS 2: histogram of depths, spread by the GEBCO uncertainty ---
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 2 – compute histogram"):
        j = min(i + chunk_size, nlat)
        elev = ds.elevation[i:j, :].load().values.flatten()
        valid = ~np.isnan(elev)
        if not valid.any():
            continue
        if "stdev" in ds:
            stdev = ds.stdev[i:j, :].load().values.flatten()[valid]
            stdev = np.nan_to_num(stdev, nan=_DEFAULT_STDEV_M)
        else:
            stdev = np.full(int(valid.sum()), _DEFAULT_STDEV_M)
        # uncertainty in bins; at least half a bin
        sigma = np.maximum(stdev / bin_size, 0.5)

        lat_v = np.repeat(lats[i:j], nlon)[valid]
        lon_v = np.tile(lons, j - i)[valid]
        hidx = hpg_n.lonlat_to_healpix(lon_v, lat_v, depth=level, ellipsoid=ellipsoid)

        depth_m = -elev[valid]  # positive below sea level
        position = (depth_m - edges[0]) / bin_size
        # spread before clipping, so that a seabed deeper than the last depth
        # stays entirely in the overflow bin
        bin_idx = np.floor(np.clip(position, -1e6, 1e6)).astype(np.int64)

        row = np.searchsorted(used_cells, hidx)
        radius = int(min(np.ceil(3 * sigma.max()), 50))
        offsets = np.arange(-radius, radius + 1)
        weights = np.exp(-(offsets[None, :] ** 2) / (2 * sigma[:, None] ** 2))
        weights /= weights.sum(axis=1, keepdims=True)
        for k, offset in enumerate(offsets):
            np.add.at(
                hist,
                (row, np.clip(bin_idx + offset, 0, n_bins - 1)),
                weights[:, k],
            )

    total = hist.sum(axis=1, keepdims=True)
    hist = np.divide(hist, total, out=np.full_like(hist, np.nan), where=total > 0)
    # P(seabed >= edge_k) = 1 - P(seabed in a bin below edge_k)
    exceedance = 1.0 - np.concatenate(
        [np.zeros((hist.shape[0], 1)), np.cumsum(hist, axis=1)[:, :-1]], axis=1
    )
    exceedance = np.clip(exceedance, 0.0, 1.0)

    var_cell_ids = xr.DataArray(
        used_cells,
        dims="cells",
        name="cell_ids",
        attrs={
            "grid_name": "healpix",
            "level": level,
            "indexing_scheme": "nested",
            "ellipsoid": canonical_ellipsoid(ellipsoid),
        },
    )
    lon, lat = hpg_n.healpix_to_lonlat(
        used_cells.astype("uint64"), level, ellipsoid=ellipsoid
    )
    return xr.Dataset(
        {
            "bathy_pixel_hist": (
                ("cells", "depth_bins"),
                exceedance,
                {
                    "long_name": "probability that the seabed is at least depth_bins deep",
                    "units": "1",
                },
            )
        },
        coords={
            "cell_ids": var_cell_ids,
            "depth_bins": (
                "depth_bins",
                edges,
                {"long_name": "depth below the mean sea level", "units": "m"},
            ),
            "latitude": ("cells", lat),
            "longitude": ("cells", lon),
        },
    )


@njit(parallel=True)
def compute_pdf_bathy_numba_like_numpy(hist, pressure, XE, depth_bins):
    """Probability that the seabed is deeper than the deepest fish depth.

    ``hist`` (C, K) is ``bathy_pixel_hist``, ``depth_bins`` (K) its depths,
    ``pressure`` (T, C, O) the fish depths (dbar ~ m) and ``XE`` (T, C) the model
    sea surface height (m). The fish depth below the mean sea level is
    ``max(pressure) - XE``. Returns (T, C), NaN on land, for time steps without
    observations and for cells without histogram. ``hist`` is not modified.
    """
    T, C, n_obs = pressure.shape
    K = hist.shape[1]
    first = depth_bins[0]
    step = depth_bins[1] - depth_bins[0]
    out = np.empty((T, C), dtype=np.float64)

    for t in prange(T):
        for c in range(C):
            if np.isnan(XE[t, c]) or np.isnan(hist[c, 0]):
                out[t, c] = np.nan
                continue

            deepest = -np.inf
            for o in range(n_obs):
                value = pressure[t, c, o]
                if not np.isnan(value) and value > deepest:
                    deepest = value
            if deepest == -np.inf:
                out[t, c] = np.nan
                continue

            position = (deepest - XE[t, c] - first) / step
            if position <= 0.0:
                value = hist[c, 0]
            elif position >= K - 1:
                # deeper than the last depth: the seabed share beyond it
                value = hist[c, K - 1]
            else:
                k = int(np.floor(position))
                frac = position - k
                value = hist[c, k] * (1.0 - frac) + hist[c, k + 1] * frac
            # never exactly impossible (keeps one bad depth from emptying a map)
            out[t, c] = max(value, 1e-14)

    return out


def _check_depth_bins(depth_bins):
    depth_bins = np.asarray(depth_bins, dtype=np.float64)
    if depth_bins.size < 2 or not np.allclose(
        np.diff(depth_bins), depth_bins[1] - depth_bins[0]
    ):
        raise ValueError("depth_bins must be regularly spaced")
    return depth_bins


def _check_pressure(pressure):
    deepest = np.nanmax(pressure) if np.isfinite(pressure).any() else np.nan
    if deepest > _MAX_PRESSURE_DBAR:
        raise ValueError(
            f"the deepest pressure is {deepest:.0f}, deeper than any ocean in dbar:"
            " is the pressure in Pa?"
        )


def compute_pdf_bathy_batch_numba(ds_chunk, reshaped_tag, copernicus_chunk):
    """Bathymetry likelihood of one batch of cells (see the module docstring)."""
    xe = copernicus_chunk["XE"]
    if "depth" in xe.dims:
        xe = xe.isel(depth=0)
    tag_time = reshaped_tag["pressure"]["time"].values
    if "time" in xe.coords and not np.array_equal(xe["time"].values, tag_time):
        raise ValueError("the tag and the model XE have different time steps")

    hist = np.asarray(ds_chunk["bathy_pixel_hist"].values, dtype=np.float64)
    pressure = np.asarray(reshaped_tag["pressure"].values, dtype=np.float64)
    XE = np.asarray(xe.transpose("time", ...).values, dtype=np.float64)
    depth_bins = _check_depth_bins(ds_chunk["depth_bins"].values)
    _check_pressure(pressure)

    if pressure.ndim == 2:
        pressure = np.broadcast_to(
            pressure[:, None, :], (XE.shape[0], XE.shape[1], pressure.shape[1])
        )
    result = compute_pdf_bathy_numba_like_numpy(
        hist, np.ascontiguousarray(pressure), XE, depth_bins
    )
    return xr.DataArray(
        result,
        dims=["time", "cells"],
        coords={
            "time": reshaped_tag["pressure"].coords["time"],
            "cells": ds_chunk["cells"],
        },
    )


def batch_compute_pdf_bathy(
    ds_lr, reshaped_tag, target_root: str, batch_size=50000, storage_options=None
):
    """Bathymetry likelihood on the cells of ``{target_root}/diff-regridded.zarr``.

    Parameters
    ----------
    ds_lr : xarray.Dataset
        Output of :py:func:`compute_healpix_histogram_region_bin_size`, on the grid
        of ``diff-regridded.zarr``.
    reshaped_tag : xarray.Dataset
        ``pressure`` (dbar) on (time, obs), on the time steps of the model.
    target_root : str
        Folder that holds ``diff-regridded.zarr`` (with ``XE``).
    batch_size : int
        Number of cells computed at once.
    storage_options : dict, optional
        Passed to :py:func:`pangeo_fish.storage.open_healpix_zarr`.

    Returns
    -------
    xarray.Dataset
        ``pdf_bathy`` on (time, cells) for every cell of ``diff-regridded.zarr``:
        NaN on land, 1 (no information) on sea cells without GEBCO points.
    """
    reference = open_healpix_zarr(
        f"{target_root}/diff-regridded.zarr", storage_options=storage_options
    )
    reference_grid = grid_attrs(reference)
    histogram_grid = grid_attrs(ds_lr)
    if reference_grid is None:
        raise ValueError("diff-regridded.zarr has no HEALPix grid (cell_ids level)")
    if histogram_grid is None:
        raise ValueError(
            "the bathymetry histogram has no HEALPix grid attributes; recompute it"
            " with compute_healpix_histogram_region_bin_size"
        )
    if histogram_grid != reference_grid:
        raise ValueError(
            "the bathymetry histogram and diff-regridded.zarr use different grids: "
            f"{histogram_grid} != {reference_grid}"
        )

    tag_time = reshaped_tag["pressure"]["time"].values
    missing_time = np.setdiff1d(tag_time, reference["time"].values)
    if missing_time.size:
        raise ValueError(
            f"{missing_time.size} time step(s) of the tag are not in"
            f" diff-regridded.zarr, e.g. {missing_time[0]}"
        )
    reference = reference[["XE"]].sel(time=tag_time)

    reference_ids = reference["cell_ids"].values
    common_ids = np.intersect1d(reference_ids, ds_lr.cell_ids.values)

    histogram = ds_lr.assign_coords(cells=("cells", ds_lr.cell_ids.values))
    histogram = histogram.drop_vars("cell_ids").sel(cells=common_ids)
    model = reference.assign_coords(cells=("cells", reference_ids))
    model = model.drop_vars("cell_ids").sel(cells=common_ids)

    pdf_chunks = []
    n_cells = common_ids.size
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        print(f"Batch cells {start}–{end}")
        pdf_chunks.append(
            compute_pdf_bathy_batch_numba(
                histogram.isel(cells=slice(start, end)),
                reshaped_tag,
                model.isel(cells=slice(start, end)),
            )
        )
    pdf = xr.concat(pdf_chunks, dim="cells")

    # Every cell of the model grid: sea cells without GEBCO points carry no
    # bathymetry information (1), land stays NaN.
    xe = reference["XE"]
    if "depth" in xe.dims:
        xe = xe.isel(depth=0)
    xe = xe.transpose("time", ...).assign_coords(cells=("cells", reference_ids))
    pdf = pdf.reindex(cells=reference_ids, fill_value=1.0).where(xe.notnull())
    n_missing = int(np.isin(reference_ids, common_ids, invert=True).sum())
    if n_missing:
        warnings.warn(
            f"{n_missing} cell(s) of the model grid have no GEBCO point: their"
            " bathymetry likelihood is 1 (no information)",
            UserWarning,
            stacklevel=2,
        )

    result = pdf.drop_vars("cells").to_dataset(name="pdf_bathy")
    result = result.assign_coords(
        cell_ids=("cells", reference_ids, dict(reference_grid))
    )
    result["pdf_bathy"].attrs = {
        "long_name": "probability that the seabed is deeper than the fish",
        "units": "1",
    }
    return result
