"""Bathymetry likelihood: the fish cannot be deeper than the seabed.

For every HEALPix cell, :py:func:`compute_healpix_histogram_region_bin_size`
summarises the GEBCO water depths that fall in the cell (land points excluded,
each depth spread by its uncertainty) as the probability that the seabed is at
least ``d`` metres deep, for depths ``d`` on a regular grid (``depth_bins``).
For every time step, :py:func:`batch_compute_pdf_bathy` takes the deepest depth
of the fish below the model sea surface, ``max(P) - XE``, and reads that
probability at this depth (linear between grid depths).
"""

import warnings

import healpix_geo.nested as hpg_n
import numpy as np
import xarray as xr
from numba import njit, prange
from scipy.special import ndtr
from tqdm import tqdm

#: Number of depth bins kept above the sea surface.
_ABOVE_BINS = 10
#: Format of the histogram; older histograms (shifted bins) are refused.
_HISTOGRAM_VERSION = 2
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
        Number of depth bins of ``depth_bin_size`` metres below the surface.
    max_depth_m : float or None
        Used when ``nb_depth_bins`` is None: ``ceil(max_depth_m / depth_bin_size)``
        bins. Seabeds deeper than this are counted as "deeper than max_depth_m".
    chunk_size : int
        Number of latitude rows read at once.
    depth_offset : float
        Clearance (m) between the fish and the seabed: the fish is taken to stay
        at least ``depth_offset`` above the GEBCO seabed. A negative value lets
        it go that much below (e.g. for a biased bathymetry).
    depth_bin_size : float
        Bin width in metres.
    ellipsoid : str or dict
        Ellipsoid of the HEALPix grid (the one of ``diff-regridded.zarr``).

    Returns
    -------
    xarray.Dataset
        ``bathy_pixel_hist`` on (cells, depth_bins): the probability that the
        seabed, minus ``depth_offset``, is at least ``depth_bins`` metres deep (1 at
        the shallowest depth, decreasing with depth). ``depth_bins`` go from ten
        bins above the surface to ``nb_depth_bins * depth_bin_size``. Cells
        whose GEBCO points are all land are not included.
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
    # deeper seabed. edge_j = (j - _ABOVE_BINS) * bin_size.
    n_edges = nb_depth_bins + _ABOVE_BINS + 1
    edges = (np.arange(n_edges) - _ABOVE_BINS) * bin_size
    n_bins = n_edges  # the last bin is the overflow bin

    lats = ds.latitude.values
    lons = ds.longitude.values
    nlat, nlon = len(lats), len(lons)
    level = int(np.log2(nside))

    # --- PASS 1: HEALPix cells that contain GEBCO water points ---
    # The fish is in the water: land points (elevation >= 0) are not seabed.
    used_cells = set()
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 1 – collect cells"):
        j = min(i + chunk_size, nlat)
        elev = ds.elevation[i:j, :].load().values.flatten()
        valid = elev < 0  # False for NaN
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

    # --- PASS 2: histogram of the water depths, each spread as N(depth, stdev) ---
    for i in tqdm(range(0, nlat, chunk_size), desc="Pass 2 – compute histogram"):
        j = min(i + chunk_size, nlat)
        elev = ds.elevation[i:j, :].load().values.flatten()
        valid = elev < 0
        if not valid.any():
            continue
        if "stdev" in ds:
            stdev = ds.stdev[i:j, :].load().values.flatten()[valid]
            stdev = np.nan_to_num(stdev, nan=_DEFAULT_STDEV_M)
        else:
            stdev = np.full(int(valid.sum()), _DEFAULT_STDEV_M)
        stdev = np.maximum(stdev, 1e-3)

        lat_v = np.repeat(lats[i:j], nlon)[valid]
        lon_v = np.tile(lons, j - i)[valid]
        hidx = hpg_n.lonlat_to_healpix(lon_v, lat_v, depth=level, ellipsoid=ellipsoid)
        row = np.searchsorted(used_cells, hidx)

        # depth the fish can reach: the seabed minus the clearance depth_offset
        reachable = -elev[valid] - depth_offset
        center = np.floor((reachable - edges[0]) / bin_size).astype(np.int64)
        # mass of N(reachable, stdev) in each bin, from the normal CDF at the bin
        # edges; the tails beyond +-radius go to the outermost bins of the window
        radius = int(min(np.ceil(4 * stdev.max() / bin_size), 50))
        for offset in range(-radius, radius + 1):
            low = edges[0] + (center + offset) * bin_size
            cdf_low = ndtr((low - reachable) / stdev) if offset > -radius else 0.0
            cdf_high = (
                ndtr((low + bin_size - reachable) / stdev) if offset < radius else 1.0
            )
            np.add.at(
                hist,
                (row, np.clip(center + offset, 0, n_bins - 1)),
                cdf_high - cdf_low,
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
            "ellipsoid": ellipsoid,
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
        attrs={
            "bathy_histogram_version": _HISTOGRAM_VERSION,
            "depth_offset": float(depth_offset),
        },
    )


@njit(parallel=True)
def compute_pdf_bathy_numba_like_numpy(hist, pressure, XE, depth_bins):
    """Probability that the seabed is deeper than the deepest fish depth.

    ``hist`` (C, K) is ``bathy_pixel_hist``, ``depth_bins`` (K) its depths,
    ``pressure`` (T, C, O) the fish depths (dbar ~ m) and ``XE`` (T, C) the model
    sea surface height (m). The fish depth below the mean sea level is
    ``max(pressure) - XE``. Returns (T, C): NaN on land and for cells without
    histogram, 1 (no information) for time steps without observation. ``hist``
    is not modified.
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
                out[t, c] = 1.0
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

    # only the deepest record of each time step matters: reduce before
    # broadcasting to the cells (a year of 90 s records would not fit otherwise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN time steps
        deepest = np.nanmax(pressure, axis=-1)
    if deepest.ndim == 1:
        deepest = np.broadcast_to(deepest[:, None], XE.shape)
    result = compute_pdf_bathy_numba_like_numpy(
        hist, np.ascontiguousarray(deepest[..., None]), XE, depth_bins
    )
    return xr.DataArray(
        result,
        dims=["time", "cells"],
        coords={
            "time": reshaped_tag["pressure"].coords["time"],
            "cells": ds_chunk["cells"],
        },
    )


def _grid(ds):
    """``{level, indexing_scheme, ellipsoid}`` of the ``cell_ids`` of ``ds``."""
    attrs = ds["cell_ids"].attrs
    if attrs.get("level") is None:
        return None
    return {
        "level": int(attrs["level"]),
        "indexing_scheme": attrs.get("indexing_scheme", "nested"),
        "ellipsoid": attrs.get("ellipsoid", "sphere"),
    }


def _same_grid(a, b):
    def key(grid):
        ellipsoid = grid["ellipsoid"]
        name = ellipsoid.lower() if isinstance(ellipsoid, str) else ellipsoid
        return grid["level"], grid["indexing_scheme"], name

    return key(a) == key(b)


def _nearest_histogram_cells(cell_ids, histogram_ids, grid, sea):
    """Index in ``histogram_ids`` of each cell, or of its nearest histogram cell.

    Only the sea cells (``sea``) without histogram are filled; land cells keep
    an arbitrary index, their likelihood is NaN anyway.
    """
    from scipy.spatial import cKDTree

    order = np.argsort(histogram_ids)
    sorted_ids = histogram_ids[order]
    position = np.clip(np.searchsorted(sorted_ids, cell_ids), 0, sorted_ids.size - 1)
    found = sorted_ids[position] == cell_ids
    index = order[position]
    fill = ~found & sea
    result = {"n_filled": int(fill.sum()), "max_distance_km": 0.0}
    if not fill.any():
        result["index"] = index
        return result

    def unit_vectors(ids):
        lon, lat = hpg_n.healpix_to_lonlat(
            np.asarray(ids, dtype="uint64"), grid["level"], ellipsoid=grid["ellipsoid"]
        )
        lon, lat = np.radians(lon), np.radians(lat)
        return np.column_stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )

    distance, nearest = cKDTree(unit_vectors(histogram_ids)).query(
        unit_vectors(cell_ids[fill])
    )
    index[fill] = nearest
    result["index"] = index
    result["max_distance_km"] = float(2 * np.arcsin(np.max(distance) / 2) * 6371.0)
    return result


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
        Passed to :py:func:`xarray.open_dataset` to read ``diff-regridded.zarr``.

    Returns
    -------
    xarray.Dataset
        ``pdf_bathy`` on (time, cells) for every cell of ``diff-regridded.zarr``,
        NaN on land. A sea cell without GEBCO water point uses the histogram of
        the nearest cell; a time step without record gets 1 (no information).
    """
    reference = xr.open_dataset(
        f"{target_root}/diff-regridded.zarr",
        engine="zarr",
        chunks={},
        inline_array=True,
        storage_options=storage_options,
    )
    cell_dim = reference["cell_ids"].dims[0]
    if cell_dim != "cells":
        reference = reference.drop_indexes("cell_ids", errors="ignore").rename_dims(
            {cell_dim: "cells"}
        )
    reference_grid = _grid(reference)
    histogram_grid = _grid(ds_lr)
    if reference_grid is None:
        raise ValueError("diff-regridded.zarr has no HEALPix grid (cell_ids level)")
    if histogram_grid is None:
        raise ValueError(
            "the bathymetry histogram has no HEALPix grid attributes; recompute it"
            " with compute_healpix_histogram_region_bin_size"
        )
    if not _same_grid(histogram_grid, reference_grid):
        raise ValueError(
            "the bathymetry histogram and diff-regridded.zarr use different grids: "
            f"{histogram_grid} != {reference_grid}"
        )

    if ds_lr.attrs.get("bathy_histogram_version") != _HISTOGRAM_VERSION:
        raise ValueError(
            "this bathymetry histogram was made by an older pangeo-fish, whose depth"
            " bins were shifted; recompute it with"
            " compute_healpix_histogram_region_bin_size"
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
    histogram_ids = ds_lr["cell_ids"].values
    # Each model cell uses its own histogram, or the one of the nearest cell
    # when it holds no GEBCO water point (fine levels, cells mostly on land).
    xe = reference["XE"]
    if "depth" in xe.dims:
        xe = xe.isel(depth=0)
    sea = np.asarray(xe.notnull().any("time").values)
    source = _nearest_histogram_cells(reference_ids, histogram_ids, reference_grid, sea)
    n_filled, max_km = source["n_filled"], source["max_distance_km"]
    if n_filled:
        warnings.warn(
            f"{n_filled} sea cell(s) of the model grid have no GEBCO water point: they"
            f" use the histogram of the nearest cell (up to {max_km:.1f} km away)",
            UserWarning,
            stacklevel=2,
        )

    histogram = ds_lr.drop_vars(
        ["cell_ids", "latitude", "longitude", "cells"], errors="ignore"
    )
    histogram = histogram.isel(cells=source["index"])
    histogram = histogram.assign_coords(cells=("cells", reference_ids))
    model = reference.drop_vars(["cell_ids", "cells"], errors="ignore")
    model = model.assign_coords(cells=("cells", reference_ids))

    pdf_chunks = []
    n_cells = reference_ids.size
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

    # as before: ``cell_ids`` is the indexed coordinate of the ``cells`` dimension
    # (the complete demo drops that index before merging into total_pdf)
    result = pdf.to_dataset(name="pdf_bathy").rename_vars({"cells": "cell_ids"})
    result["cell_ids"].attrs = dict(reference["cell_ids"].attrs)
    result["pdf_bathy"].attrs = {
        "long_name": "probability that the seabed is deeper than the fish",
        "units": "1",
    }
    return result


def sample_max_depth(elevation, latitude, longitude, *, window=None):
    """Sample a bathymetry (e.g. GEBCO) at given positions, as positive depth.

    Parameters
    ----------
    elevation : xarray.DataArray
        Elevation on a regular ``lat`` / ``lon`` grid, in meters, negative
        below sea level (GEBCO convention). The coordinates can be named
        ``lat``/``lon`` or ``latitude``/``longitude``.
    latitude, longitude : array-like or xarray.DataArray
        Positions where the depth is needed, e.g. the cell centers of the model
        grid (``diff_ds["latitude"]``, ``diff_ds["longitude"]``).
    window : float, optional
        Width (in degrees) of the box centred on each position. The deepest
        bathymetry value within the box is returned, which is the relevant
        value to decide whether a fish can reach a given depth somewhere in a
        model cell. Use the model resolution (e.g. ``1 / 36`` for a 1/36°
        grid). If ``None``, the nearest bathymetry value is returned.

    Returns
    -------
    depth : xarray.DataArray
        Depth in meters (positive downwards; land is negative), with the shape
        (and, for :py:class:`xarray.DataArray` positions, the dimensions and
        coordinates) of ``latitude``. ``NaN`` where the position is ``NaN``.

    Raises
    ------
    ValueError
        If a position lies outside the extent of the bathymetry grid.

    Notes
    -----
    The requested longitudes are wrapped to the convention of the grid
    (``0..360`` or ``-180..180``), so cell longitudes in either convention work.

    :py:func:`compute_healpix_histogram_region_bin_size` also describes the
    depths inside each cell, as the probability that the seabed is at least a
    given depth. It is smoothed by the GEBCO uncertainty, leaves out land
    points and only works on HEALPix cells, so its deepest depth is not this
    maximum.
    """
    names = {"latitude": "lat", "longitude": "lon"}
    elevation = elevation.rename(
        {k: v for k, v in names.items() if k in elevation.dims}
    )
    # ascending coordinates, so that the area of interest can be sliced
    for dim in ("lat", "lon"):
        if elevation[dim].values[0] > elevation[dim].values[-1]:
            elevation = elevation.isel({dim: slice(None, None, -1)})

    lat = xr.DataArray(latitude).astype("float64")
    lon = xr.DataArray(longitude).astype("float64")
    finite = lat.notnull() & lon.notnull()
    if not finite.any():
        return xr.full_like(lat, np.nan).rename("depth").assign_attrs(units="m")

    half = 0.0 if window is None else float(window) / 2
    dlat = float(np.abs(elevation["lat"].diff("lat")[0]))
    dlon = float(np.abs(elevation["lon"].diff("lon")[0]))

    # longitudes in the convention of the grid (0..360 or -180..180), and
    # positions outside the extent of the grid are an error
    lon_min = float(elevation["lon"].min()) - dlon / 2
    lon_max = float(elevation["lon"].max()) + dlon / 2
    lat_min = float(elevation["lat"].min()) - dlat / 2
    lat_max = float(elevation["lat"].max()) + dlat / 2
    lat_b, lon_b = xr.broadcast(lat.where(finite), lon.where(finite))
    lon_values = lon_b.values
    lat_values = lat_b.values
    finite_values = finite.broadcast_like(lon_b).transpose(*lon_b.dims).values
    with np.errstate(invalid="ignore"):
        candidates = [lon_values, lon_values % 360, (lon_values + 180) % 360 - 180]
        wrapped = np.full(lon_values.shape, np.nan)
        for candidate in reversed(candidates):
            inside = (candidate >= lon_min) & (candidate <= lon_max)
            wrapped = np.where(inside, candidate, wrapped)
        outside = finite_values & (
            np.isnan(wrapped) | (lat_values < lat_min) | (lat_values > lat_max)
        )
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} requested position(s) lie outside the extent"
            f" of the bathymetry grid (latitude {lat_min:.4g} to {lat_max:.4g},"
            f" longitude {lon_min:.4g} to {lon_max:.4g}), e.g. latitude"
            f" {float(lat_values[outside][0]):.4g}, longitude"
            f" {float(lon_values[outside][0]):.4g}"
        )
    lon = lon_b.copy(data=wrapped)

    margin_lat = half + 2 * dlat
    margin_lon = half + 2 * dlon
    depth = -elevation.sel(
        lat=slice(float(lat.min()) - margin_lat, float(lat.max()) + margin_lat),
        lon=slice(float(lon.min()) - margin_lon, float(lon.max()) + margin_lon),
    )
    if window is not None:
        size = {
            "lat": 2 * int(np.ceil(half / dlat)) + 1,
            "lon": 2 * int(np.ceil(half / dlon)) + 1,
        }
        depth = depth.rolling(size, center=True, min_periods=1).max()

    # NaN positions cannot be looked up: use any finite position, then mask
    sampled = depth.sel(
        lat=lat.where(finite, float(lat.where(finite).max())),
        lon=lon.where(finite, float(lon.where(finite).max())),
        method="nearest",
    )
    sampled = sampled.drop_vars(["lat", "lon"]).where(finite)
    return sampled.rename("depth").assign_attrs(units="m")
