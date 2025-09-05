import warnings

import dask.array as da
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xdggs
from dask import delayed
from distributed import LocalCluster
from tqdm import tqdm

from pangeo_fish.cf import bounds_to_bins
from pangeo_fish.diff import diff_z
from pangeo_fish.helpers import compute_diff, load_model
from pangeo_fish.io import open_tag
from pangeo_fish.tags import adapt_model_time, reshape_by_bins, to_time_slice

warnings.filterwarnings("ignore", category=RuntimeWarning)

import healpy as hp
import numpy as np
import xarray as xr
from tqdm import tqdm


def compute_healpix_histogram_region_bin_size(
    ds,
    nside,
    nb_depth_bins=None,
    max_depth_m=None,
    chunk_size=500,
    depth_offset=0,
    depth_bin_size=1,
):
    """
    Compute HEALPix histogram over the geographic region covered by `ds`.

    Parameters
    ----------
    ds : xarray.Dataset
        Contains `elevation` (2D lat×lon) and optionally `stdev`.
    nside : int
        HEALPix resolution (power of 2).
    nb_depth_bins : int or None
        Number of depth bins (in units of `depth_bin_size`). If provided, used directly.
    max_depth_m : float or None
        If provided and nb_depth_bins is None, nb_depth_bins = ceil(max_depth_m / depth_bin_size).
    chunk_size : int
        Number of latitude rows per iteration.
    depth_offset : float
        Offset applied to bathymetry before binning.
    depth_bin_size : float
        Bin width in meters (default=1). Example: 100 → bins de 100 m.
    """

    # --- determine nb_depth_bins from max_depth_m if needed ---
    if nb_depth_bins is None:
        if max_depth_m is None:
            raise ValueError("Either nb_depth_bins or max_depth_m must be provided.")
        nb_depth_bins = int(np.ceil(float(max_depth_m) / float(depth_bin_size)))
    else:
        nb_depth_bins = int(nb_depth_bins)

    if nb_depth_bins <= 0:
        raise ValueError("nb_depth_bins must be > 0")

    nest = True
    # historic behaviour: we keep +10 buffer bins as in original code
    bins = nb_depth_bins + 10

    lats = ds.latitude.values
    lons = ds.longitude.values
    nlat, nlon = len(lats), len(lons)

    # --- PASS 1: Collect unique used HEALPix cells ---
    used_cells = set()
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 1 – collect cells"):
        j = min(i + chunk_size, nlat)

        elev = ds.elevation[i:j, :].load().values.flatten()
        valid = ~np.isnan(elev)
        if not valid.any():
            continue

        lat_blk = np.repeat(lats[i:j], nlon)[valid]
        lon_blk = np.tile(lons, j - i)[valid]

        hidx = hp.ang2pix(nside, lon_blk, lat_blk, lonlat=True, nest=nest)
        used_cells.update(np.unique(hidx))

    used_cells = np.array(sorted(used_cells), dtype=np.int64)
    n_cells = used_cells.size
    cell_to_idx = {cell: idx for idx, cell in enumerate(used_cells)}

    hist = np.zeros((n_cells, bins), dtype=np.float64)

    # --- PASS 2: Accumulate histogram ---
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 2 – compute histogram"):
        j = min(i + chunk_size, nlat)

        elev = ds.elevation[i:j, :].load().values.flatten()
        st = (
            ds.stdev[i:j, :].load().values.flatten()
            if "stdev" in ds
            else np.full_like(elev, 1.0)
        )

        valid = ~np.isnan(elev)
        if not valid.any():
            continue

        elev_v = elev[valid]
        st_v = np.maximum(np.nan_to_num(st[valid], nan=1.0), 0.5)
        lat_v = np.repeat(lats[i:j], nlon)[valid]
        lon_v = np.tile(lons, j - i)[valid]

        hidx = hp.ang2pix(nside, lon_v, lat_v, lonlat=True, nest=nest)

        # --- binning basé sur depth_bin_size ---
        # profondeur positive = -elev_v (elev is negative for bathy)
        depth_val = (-elev_v - depth_offset) / float(depth_bin_size)
        # clip entre -10 (buffer) et nb_depth_bins - tiny
        depth_idx = np.clip(depth_val, -10, nb_depth_bins - 0.01).astype(np.int64) + 10

        mask_zone = np.isin(hidx, used_cells)
        hidx = hidx[mask_zone]
        depth_idx = depth_idx[mask_zone]
        st_v = st_v[mask_zone]

        sum_w = np.zeros_like(st_v)
        for dj in range(-2, 3):
            sum_w += np.exp(-(dj**2) / (2 * st_v**2))
        inv_sum = 1.0 / sum_w

        uniq, inv = np.unique(hidx, return_inverse=True)
        hist_local = np.zeros((uniq.size, bins), dtype=np.float64)

        for dj in range(-2, 3):
            w = np.exp(-(dj**2) / (2 * st_v**2)) * inv_sum
            idx_shifted = np.clip(depth_idx + dj, 0, bins - 1)
            np.add.at(hist_local, (inv, idx_shifted), w)

        for u_idx, cell in enumerate(uniq):
            hist[cell_to_idx[cell], :] += hist_local[u_idx, :]

    with np.errstate(invalid="ignore"):
        hist /= hist.sum(axis=1, keepdims=True)
    h_im = 1 - np.cumsum(hist[:, :nb_depth_bins], axis=1)

    var_cell_ids = xr.DataArray(
        used_cells,
        dims="cells",
        name="cell_ids",
        attrs={"grid_name": "healpix", "nside": nside, "nest": nest},
    )

    ds_out = xr.Dataset(
        {"bathy_pixel_hist": (("cells", "depth_bins"), h_im)},
        coords={"cell_ids": var_cell_ids},
    )
    # depth_bins: valeur du début du bin en mètres (0, depth_bin_size, 2*depth_bin_size, ...)
    ds_out["depth_bins"] = np.arange(nb_depth_bins) * depth_bin_size

    cell_ids = ds_out.cell_ids.values
    lon, lat = hp.pix2ang(nside, cell_ids, nest=True, lonlat=True)
    ds_out = ds_out.assign_coords(
        {"latitude": ("cells", lat), "longitude": ("cells", lon)}
    )
    return ds_out


def compute_healpix_histogram_region(
    ds, nside, nb_depth_bins, chunk_size=500, depth_offset=0
):
    """
    Compute HEALPix histogram only over the geographic region covered by `ds`,
    using a two-pass approach:
      1) Identify the unique cells actually touched
      2) Accumulate histogram on this subset

    Parameters
    ----------
    ds : xarray.Dataset
        Contains `elevation` (2D lat×lon) and optionally `stdev`.
    nside : int
        HEALPix resolution (power of 2).
    nb_depth_bins : int
        Number of bins (excluding the +10 offset).
    chunk_size : int
        Number of latitude rows per iteration.
    depth_offset : float
        Offset applied to bathymetry before binning.
    """
    nest = True
    bins = nb_depth_bins + 10

    lats = ds.latitude.values
    lons = ds.longitude.values
    nlat, nlon = len(lats), len(lons)
    print(lons.min())
    print(lons.max())
    # --- PASS 1: Collect unique used HEALPix cells ---
    used_cells = set()
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 1 – collect cells"):
        j = min(i + chunk_size, nlat)

        elev = ds.elevation[i:j, :].load().values.flatten()
        valid = ~np.isnan(elev)
        if not valid.any():
            continue

        lat_blk = np.repeat(lats[i:j], nlon)[valid]
        lon_blk = np.tile(lons, j - i)[valid]

        hidx = hp.ang2pix(nside, lon_blk, lat_blk, lonlat=True, nest=nest)
        used_cells.update(np.unique(hidx))

    used_cells = np.array(sorted(used_cells), dtype=np.int64)
    n_cells = used_cells.size
    cell_to_idx = {cell: idx for idx, cell in enumerate(used_cells)}

    hist = np.zeros((n_cells, bins), dtype=np.float64)

    # --- PASS 2: Accumulate histogram ---
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 2 – compute histogram"):
        j = min(i + chunk_size, nlat)

        elev = ds.elevation[i:j, :].load().values.flatten()
        st = (
            ds.stdev[i:j, :].load().values.flatten()
            if "stdev" in ds
            else np.full_like(elev, 1.0)
        )

        valid = ~np.isnan(elev)
        if not valid.any():
            continue

        elev_v = elev[valid]
        st_v = np.maximum(np.nan_to_num(st[valid], nan=1.0), 0.5)
        lat_v = np.repeat(lats[i:j], nlon)[valid]
        lon_v = np.tile(lons, j - i)[valid]

        hidx = hp.ang2pix(nside, lon_v, lat_v, lonlat=True, nest=nest)
        depth_idx = (
            np.clip(-elev_v - depth_offset, -10, nb_depth_bins - 0.01).astype(np.int64)
            + 10
        )

        mask_zone = np.isin(hidx, used_cells)
        hidx = hidx[mask_zone]
        depth_idx = depth_idx[mask_zone]
        st_v = st_v[mask_zone]

        sum_w = np.zeros_like(st_v)
        for dj in range(-2, 3):
            sum_w += np.exp(-(dj**2) / (2 * st_v**2))
        inv_sum = 1.0 / sum_w

        uniq, inv = np.unique(hidx, return_inverse=True)
        hist_local = np.zeros((uniq.size, bins), dtype=np.float64)

        for dj in range(-2, 3):
            w = np.exp(-(dj**2) / (2 * st_v**2)) * inv_sum
            idx_shifted = np.clip(depth_idx + dj, 0, bins - 1)
            np.add.at(hist_local, (inv, idx_shifted), w)

        for u_idx, cell in enumerate(uniq):
            hist[cell_to_idx[cell], :] += hist_local[u_idx, :]

    with np.errstate(invalid="ignore"):
        hist /= hist.sum(axis=1, keepdims=True)
    h_im = 1 - np.cumsum(hist[:, :nb_depth_bins], axis=1)

    var_cell_ids = xr.DataArray(
        used_cells,
        dims="cells",
        name="cell_ids",
        attrs={"grid_name": "healpix", "nside": nside, "nest": nest},
    )

    ds_out = xr.Dataset(
        {"bathy_pixel_hist": (("cells", "depth_bins"), h_im)},
        coords={"cell_ids": var_cell_ids},
    )
    ds_out["depth_bins"] = np.arange(nb_depth_bins)
    cell_ids = ds_out.cell_ids.values
    lon, lat = hp.pix2ang(nside, cell_ids, nest=True, lonlat=True)

    ds_out = ds_out.assign_coords(
        {"latitude": ("cells", lat), "longitude": ("cells", lon)}
    )
    return ds_out


def compute_fish_histogram(reshaped_tag, depth_min=0, depth_max=200, bins=210):
    """
    Compute vertical histogram from fish pressure profiles.

    Parameters
    ----------
    reshaped_tag : xarray.Dataset
        Dataset containing "pressure" variable.
    depth_min : float
        Minimum depth for histogram.
    depth_max : float
        Maximum depth for histogram.
    bins : int
        Number of histogram bins.
    """
    bin_edges = np.linspace(depth_min, depth_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pressure = reshaped_tag.pressure.values  # (time, depth)

    hist2d = np.apply_along_axis(
        lambda x: np.histogram(x, bins=bin_edges)[0], axis=1, arr=pressure
    )
    hist2d = hist2d / hist2d.sum(axis=1, keepdims=True)
    fish_pdf = 1 - np.cumsum(hist2d, axis=1)

    return xr.DataArray(
        fish_pdf,
        dims=["time", "depth_bins"],
        coords={"time": reshaped_tag.time.values, "depth_bins": bin_centers},
        name="fish_hist",
    )


def compute_fish_histogram_bin_size(
    reshaped_tag, depth_min=0, depth_max=None, nb_depth_bins=None, depth_bin_size=1
):
    """
    Compute vertical histogram from fish pressure profiles.
    Parameters
    ----------
    reshaped_tag : xarray.Dataset
        Dataset containing "pressure" variable.
    depth_min : float
        Minimum depth for histogram (default=0).
    depth_max : float or None
        Maximum depth in meters. If nb_depth_bins is None, used to compute nb_depth_bins.
    nb_depth_bins : int or None
        Number of bins (in units of depth_bin_size). If None, inferred from depth_max.
    depth_bin_size : float
        Bin width in meters. Default=1 → 1 bin par mètre (ancien comportement).
        Exemple: depth_bin_size=100, depth_max=800 → 8 bins de 100 m.
    """

    # --- déterminer nb_depth_bins ---
    if nb_depth_bins is None:
        if depth_max is None:
            raise ValueError("Either nb_depth_bins or depth_max must be provided.")
        nb_depth_bins = int(np.ceil((depth_max - depth_min) / float(depth_bin_size)))
    else:
        nb_depth_bins = int(nb_depth_bins)

    # bornes des bins
    bin_edges = np.arange(
        depth_min, depth_min + (nb_depth_bins + 1) * depth_bin_size, depth_bin_size
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # récupérer la pression
    pressure = reshaped_tag.pressure.values  # (time, depth)

    # histogramme pour chaque profil
    hist2d = np.apply_along_axis(
        lambda x: np.histogram(x, bins=bin_edges)[0], axis=1, arr=pressure
    )

    # normalisation
    hist2d = hist2d / hist2d.sum(axis=1, keepdims=True)

    # PDF cumulée inversée
    fish_pdf = 1 - np.cumsum(hist2d, axis=1)

    return xr.DataArray(
        fish_pdf,
        dims=["time", "depth_bins"],
        coords={"time": reshaped_tag.time.values, "depth_bins": bin_centers},
        name="fish_hist",
    )


def compute_pdf_bathy_batch_numpy(ds_chunk, reshaped_tag, copernicus_chunk):
    """
    Generate PDF taking max depth of fish per time range in order to exclude non-possible
    pixel (bathymetry limitation).
    Parameters
    ----------
    ds_chunk : xarray.Dataset
        Histogram dataset (subset of full cells).
    reshaped_tag : xarray.Dataset
        Pressure profiles.
    copernicus_chunk : xarray.Dataset
        Modeled XE data for the same cells.
    """
    hist = ds_chunk["bathy_pixel_hist"].data
    pressure = reshaped_tag["pressure"].data
    XE = copernicus_chunk["XE"].data

    nan_mask = np.isnan(XE[0, :])
    hist[nan_mask, :] = np.nan
    hist = np.where(hist <= 0, 1e-14, hist)

    prof_corr = pressure[:, None, :] - XE[:, :, None] + 10  # (T, C, O)
    valid = ~np.isnan(prof_corr)

    bin_index = np.floor(prof_corr).astype("int32")
    bin_index[~valid] = 0
    max_bin = hist.shape[1] - 1
    bin_index = np.clip(bin_index, 0, max_bin)

    T, C, O = prof_corr.shape
    h = np.broadcast_to(hist[None, :, :], (T, C, hist.shape[1]))
    selected = np.take_along_axis(h, bin_index, axis=-1)

    safe_prof = np.where(np.isnan(prof_corr), 1e-14, prof_corr)
    idx_max = np.nanargmax(safe_prof, axis=-1)

    t_idx = np.arange(T)[:, None]
    c_idx = np.arange(C)[None, :]
    final_pdf = selected[t_idx, c_idx, idx_max]

    all_nan = np.all(np.isnan(prof_corr), axis=-1)
    final_pdf[all_nan] = np.nan

    return xr.DataArray(
        final_pdf,
        dims=["time", "cells"],
        coords={
            "time": reshaped_tag["pressure"].coords["time"],
            "cells": ds_chunk["cells"],
        },
    )


def batch_compute_pdf_bathy(
    ds_lr,
    reshaped_tag,
    target_root: str,
    batch_size=50000,
):
    """
    Dividing calculation into batch.
    Parameters
    ----------
    ds_lr : xarray.Dataset
        Full EMODnet histogram dataset.
    reshaped_tag : xarray.Dataset
        Fish reshaped pressure tag.
    batch_size : int
        Number of cells processed per batch.
    """
    n_cells = ds_lr.sizes["cells"]
    pdf_chunks = []

    # Here we make sure we are working on the same grids
    reference = xr.open_dataset(
        f"{target_root}/diff-regridded.zarr",
        engine="zarr",
        chunks={},
        inline_array=True,
        storage_options=None,
    )

    common_ids = np.intersect1d(reference.cell_ids.values, ds_lr.cell_ids.values)

    ds_histo_coords = ds_lr.assign_coords(cell_ids=("cells", ds_lr.cell_ids.values))
    ds_histo_coords = ds_histo_coords.set_index(cells="cell_ids")

    histogram_ds_subset_model = ds_histo_coords.sel(cells=common_ids)

    reference = reference.assign_coords(cells=("cell_ids", reference["cell_ids"].data))

    reference = reference.swap_dims({"cell_ids": "cells"})

    # then we loop on smaller dataset in order to avoid RAM increasing to much
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        print(f"Batch cells {start}–{end}")

        ds_chunk = histogram_ds_subset_model.isel(cells=slice(start, end))
        copernicus_chunk = reference.isel(cells=slice(start, end))

        pdf_chunk = compute_pdf_bathy_batch_numpy(
            ds_chunk, reshaped_tag, copernicus_chunk
        )
        pdf_chunks.append(pdf_chunk)

    concat_pdf = xr.concat(pdf_chunks, dim="cells")
    pdf_da_func = concat_pdf.to_dataset(name="pdf_bathy")
    pdf_da_func = pdf_da_func.rename_vars({"cells": "cell_ids"})

    return pdf_da_func
