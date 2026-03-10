import warnings

import healpy as hp
import numpy as np
import xarray as xr
from numba import njit, prange
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


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


@njit(parallel=True)
def compute_pdf_bathy_numba_like_numpy(hist, pressure, XE, depth_bins):

    T, C, n_obs = pressure.shape
    B = hist.shape[1]

    depth_bin_size = depth_bins[1] - depth_bins[0]
    out = np.empty((T, C), dtype=np.float64)

    # ---- land : hist[c,:] = NaN ----
    for c in range(C):
        if np.isnan(XE[0, c]):
            for b in range(B):
                hist[c, b] = np.nan
        else:
            # ---- RÈGLE 1 : hist <= 0 → 1e-14 ----
            for b in range(B):
                if hist[c, b] <= 0 or np.isnan(hist[c, b]):
                    hist[c, b] = 1e-14

    for t in prange(T):
        for c in range(C):

            if np.isnan(XE[t, c]):
                out[t, c] = np.nan
                continue

            max_val = -1e32

            all_nan = True

            for o in range(n_obs):
                val = pressure[t, c, o]

                if np.isnan(val):
                    val = 1e-14
                else:
                    all_nan = False

                pc = val - XE[t, c] + 10

                if pc > max_val:
                    max_val = pc

            if all_nan:
                out[t, c] = np.nan
                continue

            # Compute bin
            bin_idx = int(max_val / depth_bin_size)
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= B:
                bin_idx = B - 1

            out[t, c] = hist[c, bin_idx]

    return out


def compute_pdf_bathy_batch_numba(ds_chunk, reshaped_tag, copernicus_chunk):

    hist = np.asarray(ds_chunk["bathy_pixel_hist"].values)  # (C,B)
    pressure = np.asarray(reshaped_tag["pressure"].values)  # (T,n_obs) or (T,C,n_obs)
    XE = np.asarray(copernicus_chunk["XE"].values)  # (T,C)
    depth_bins = np.asarray(ds_chunk.depth_bins.values)

    if pressure.ndim == 2:
        pressure = pressure[:, None, :]
        pressure = np.broadcast_to(
            pressure, (XE.shape[0], XE.shape[1], pressure.shape[2])
        )

    result = compute_pdf_bathy_numba_like_numpy(hist, pressure, XE, depth_bins)

    return xr.DataArray(
        result,
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
    Dividing calculation into batches using the Numba-accelerated per-batch function.
    """

    pdf_chunks = []

    # Load reference grid
    reference = xr.open_dataset(
        f"{target_root}/diff-regridded.zarr",
        engine="zarr",
        chunks={},
        inline_array=True,
        storage_options=None,
    )

    # Get common cell IDs
    common_ids = np.intersect1d(reference.cell_ids.values, ds_lr.cell_ids.values)

    # Align histogram dataset
    ds_histo_coords = ds_lr.assign_coords(cell_ids=("cells", ds_lr.cell_ids.values))
    ds_histo_coords = ds_histo_coords.set_index(cells="cell_ids")
    histogram_ds_subset_model = ds_histo_coords.sel(cells=common_ids)

    # Align reference dataset
    reference = reference.assign_coords(cells=("cell_ids", reference["cell_ids"].data))
    reference = reference.swap_dims({"cell_ids": "cells"})

    # Loop over batches
    n_cells = histogram_ds_subset_model.sizes["cells"]
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        print(f"Batch cells {start}–{end}")

        ds_chunk = histogram_ds_subset_model.isel(cells=slice(start, end))
        copernicus_chunk = reference.isel(cells=slice(start, end))

        pdf_chunk = compute_pdf_bathy_batch_numba(
            ds_chunk, reshaped_tag, copernicus_chunk
        )

        pdf_chunks.append(pdf_chunk)

    # Concatenate results
    concat_pdf = xr.concat(pdf_chunks, dim="cells")
    pdf_da_func = concat_pdf.to_dataset(name="pdf_bathy")
    pdf_da_func = pdf_da_func.rename_vars({"cells": "cell_ids"})

    return pdf_da_func
