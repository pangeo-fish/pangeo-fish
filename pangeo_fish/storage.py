"""Zarr-Python 3 APIs, with Zarr format 2 on disk for existing readers."""

from math import gcd
from numbers import Integral

import numpy as np

from pangeo_fish.dggs import canonical_ellipsoid, with_cf_attrs, with_grid_attrs


def write_time_blocks(ds, store, *, step):
    """Write bounded, chunk-aligned time regions and time-independent variables.

    Keep ``safe_chunks`` enabled: even a user-supplied step smaller than ten must
    end on storage chunk boundaries. The final, partial chunk is allowed.
    """
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    ds = ds.chunk({"time": gcd(10, step)})
    for var in ds.variables:
        ds[var].encoding.pop("chunks", None)
    ds.to_zarr(store, mode="w", compute=False, zarr_format=2, consolidated=False)
    static = [name for name in ds.variables if "time" not in ds[name].dims]
    temporal = ds.drop_vars(static)
    for start in range(0, ds.sizes["time"], step):
        end = min(start + step, ds.sizes["time"])
        temporal.isel(time=slice(start, end)).to_zarr(
            store,
            mode="r+",
            region={"time": slice(start, end)},
            zarr_format=2,
            consolidated=False,
        )
    if static:
        ds[static].to_zarr(store, mode="a", zarr_format=2, consolidated=False)


def _with_dggs_attrs(ds):
    """Return ``ds`` with the canonical HEALPix grid attributes on ``cell_ids``.

    Kept for compatibility; see :py:func:`pangeo_fish.dggs.with_grid_attrs`.
    """
    return with_grid_attrs(ds)


_DGGS_CONVENTION = {
    "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38",
    "name": "dggs",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "description": "Discrete Global Grid Systems convention for zarr",
}


DEFAULT_TARGET_CHUNK_BYTES = 64 * 2**20
"""Default upper bound (64 MiB) for one storage chunk of a HEALPix store."""


def _storage_chunks(ds, dim, target_chunk_bytes):
    """Choose storage chunks of at most ``target_chunk_bytes`` (see below).

    One time step of the largest variable holding ``dim`` (one "map") sets the
    scale. When a map fits in the target, keep all cells in one chunk and
    group as many time steps as fit. Otherwise use one time step per chunk
    and split the cells into nearly equal chunks.
    """
    map_bytes = max(
        var.dtype.itemsize
        * int(np.prod([size for d, size in var.sizes.items() if d != "time"]))
        for var in ds.variables.values()
        if dim in var.dims
    )
    n_cells = ds.sizes[dim]
    if map_bytes <= target_chunk_bytes:
        chunks = {dim: n_cells}
        if "time" in ds.dims:
            n_steps = target_chunk_bytes // map_bytes
            chunks["time"] = max(1, min(ds.sizes["time"], n_steps))
    else:
        bytes_per_cell = map_bytes / n_cells
        max_cells = max(1, int(target_chunk_bytes // bytes_per_cell))
        n_chunks = -(-n_cells // max_cells)
        chunks = {dim: -(-n_cells // n_chunks)}
        if "time" in ds.dims:
            chunks["time"] = 1
    return chunks


def with_dggs_metadata(ds):
    """Return ``ds`` with the full HEALPix metadata, ready to be written to Zarr.

    Sets the canonical grid attributes on ``cell_ids``, the CF attributes of the
    known variables and ``Conventions``, and the Zarr DGGS v1 group attributes
    (``dggs`` and ``zarr_conventions``). Checks the ``cell_ids``, clears inherited
    encodings and puts the cells last. Chunks are not changed. The input dataset
    and its encodings are not modified.
    """
    import healpix_geo.ellipsoid
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds = with_cf_attrs(with_grid_attrs(ds)).copy(deep=False)
    if "cell_ids" not in ds.coords or ds.cell_ids.ndim != 1:
        raise ValueError("a one-dimensional cell_ids coordinate is required")
    attrs = ds.cell_ids.attrs
    level = attrs.get("level")
    scheme = attrs.get("indexing_scheme")
    if attrs.get("grid_name") != "healpix" or scheme not in {"nested", "ring"}:
        raise ValueError("a fixed-level HEALPix nested or ring grid is required")
    if not isinstance(level, Integral) or not 0 <= level <= 29:
        raise ValueError("HEALPix level must be an integer between 0 and 29")
    ids = ds.cell_ids.values
    if ids.dtype.kind not in "iu" or ids.size == 0:
        raise ValueError("cell_ids must contain nonempty integer HEALPix indices")
    if np.any(ids < 0) or np.any(ids >= 12 * 4 ** int(level)):
        raise ValueError("cell_ids are outside the HEALPix grid")
    if np.unique(ids).size != ids.size:
        raise ValueError("cell_ids must be unique")
    dim = ds.cell_ids.dims[0]
    ellipsoid = canonical_ellipsoid(attrs.get("ellipsoid"))
    if isinstance(ellipsoid, str):
        ellipsoid = healpix_geo.ellipsoid.resolve(ellipsoid)
    ellipsoid = dict(ellipsoid)
    if "semimajor_axis" in ellipsoid:
        ellipsoid["semi_major_axis"] = ellipsoid.pop("semimajor_axis")
    if "semiminor_axis" in ellipsoid:
        ellipsoid["semi_minor_axis"] = ellipsoid.pop("semiminor_axis")
    if "name" not in ellipsoid:
        ellipsoid["name"] = "custom"
    ds.attrs["dggs"] = {
        "name": "healpix",
        "refinement_level": int(level),
        "indexing_scheme": scheme,
        "ellipsoid": ellipsoid,
        "spatial_dimension": dim,
        "coordinate": "cell_ids",
        "compression": "none",
    }
    conventions = [
        dict(item)
        for item in ds.attrs.get("zarr_conventions", [])
        if item.get("name") != "dggs" and item.get("uuid") != _DGGS_CONVENTION["uuid"]
    ]
    ds.attrs["zarr_conventions"] = [*conventions, dict(_DGGS_CONVENTION)]
    # Source stores can carry incompatible chunks, packing and fill values.
    # Let xarray encode the decoded values afresh, preserving real zeros and IDs.
    for name in ds.variables:
        ds[name].encoding = {}
    return ds.transpose(..., dim)


def prepare_dggs_zarr(
    ds, *, cell_chunk_size=None, target_chunk_bytes=DEFAULT_TARGET_CHUNK_BYTES
):
    """Prepare a HEALPix dataset for a new Zarr store without adding cells.

    Add the metadata of :py:func:`with_dggs_metadata` (grid, CF and DGGS v1),
    with cells last, and choose the storage chunks. Sparse cells retain their
    order.

    By default the chunks are sized from the data, up to ``target_chunk_bytes``
    per chunk. One time step of the largest variable with cells (one map) sets
    the scale:

    * when a map fits in the target, all the cells stay in one chunk and as
      many time steps as fit are grouped. This is the common case: one map of
      a coastal box at level 12 is about 1 MB, of the Mediterranean at
      level 12 about 20 MB;
    * when a single map is larger than the target (e.g. a basin-scale domain
      at level 12), each chunk holds one time step and the cells are split
      into the fewest chunks of (nearly) equal size that fit; only the last
      chunk can be slightly smaller. Sparse chunks need not coincide with
      HEALPix parent cells.

    Pass ``cell_chunk_size`` to force one time step per chunk and at most
    ``cell_chunk_size`` cells per chunk instead.
    The input dataset and its encodings are not modified.
    """
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    if cell_chunk_size is not None and (
        not isinstance(cell_chunk_size, Integral) or cell_chunk_size <= 0
    ):
        raise ValueError("cell_chunk_size must be None or a positive integer")
    if not isinstance(target_chunk_bytes, Integral) or target_chunk_bytes <= 0:
        raise ValueError("target_chunk_bytes must be a positive integer")
    ds = with_dggs_metadata(ds)
    dim = ds.cell_ids.dims[0]
    if cell_chunk_size is None:
        chunks = _storage_chunks(ds, dim, int(target_chunk_bytes))
    else:
        chunks = {dim: int(cell_chunk_size)}
        if "time" in ds.dims:
            chunks["time"] = 1
    return ds.chunk(chunks)


def write_dggs_zarr(ds, store, *, chunk_level=None, storage_options=None):
    """Export a map dataset with GRID4EARTH-style parent-cell spatial chunks.

    Each occupied nested HEALPix parent becomes one dense chunk containing
    ``4**(level - chunk_level)`` cells. Missing children are padded with NaN
    (False for boolean masks); no probabilities are interpolated or normalized.
    Unoccupied parents are omitted. This export is intended for viewing, not
    as a replacement for the sparse inference input.

    By default use at most 4096 cells per chunk (six levels below the data).
    Write a new Zarr v2 store with Blosc/LZ4 and consolidated metadata. Existing
    stores are refused. Supports local paths, Zarr stores and S3 URLs with
    ``storage_options``. Returns the prepared lazy dataset.
    """
    from numcodecs import Blosc

    ds = prepare_dggs_zarr(ds)
    grid = ds.attrs["dggs"]
    level = grid["refinement_level"]
    if grid["indexing_scheme"] != "nested":
        raise ValueError("parent-cell chunks require nested HEALPix indexing")
    if chunk_level is None:
        chunk_level = max(0, level - 6)
    if not isinstance(chunk_level, Integral) or not 0 <= chunk_level <= level:
        raise ValueError("chunk_level must be an integer between 0 and the data level")
    size = 4 ** (level - int(chunk_level))
    dim = grid["spatial_dimension"]
    ids = ds.cell_ids.values.astype("int64")
    parents = np.unique(ids // size)
    target_ids = (parents[:, None] * size + np.arange(size)).ravel()
    # Drop custom xdggs indexes before using ordinary xarray label reindexing.
    if "cell_ids" in ds.xindexes:
        ds = ds.drop_indexes("cell_ids")
    ds = ds.swap_dims({dim: "cell_ids"})
    fill_values = {
        name: False if var.dtype.kind == "b" else np.nan
        for name, var in ds.variables.items()
        if name != "cell_ids"
    }
    ds = ds.reindex(cell_ids=target_ids, fill_value=fill_values)
    if dim != "cell_ids":
        ds = ds.swap_dims({"cell_ids": dim})
    ds = prepare_dggs_zarr(ds, cell_chunk_size=size)
    encoding = {
        name: {"compressors": Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)}
        for name in ds.variables
    }
    ds.to_zarr(
        store,
        mode="w-",
        storage_options=storage_options,
        zarr_format=2,
        consolidated=True,
        encoding=encoding,
    )
    return ds


def write_healpix_zarr(
    ds,
    store,
    *,
    storage_options=None,
    mode="w",
    target_chunk_bytes=DEFAULT_TARGET_CHUNK_BYTES,
    consolidated=True,
):
    """Write a HEALPix dataset of the workflow to a new Zarr store.

    This is the single entry point for the intermediate and output stores
    (``diff-regridded``, ``emission``, ``combined``, ``states``, ...). It:

    - sets the canonical grid attributes on ``cell_ids``
      (:py:func:`pangeo_fish.dggs.with_grid_attrs`);
    - adds the CF attributes of the known variables and ``Conventions``
      (:py:func:`pangeo_fish.dggs.with_cf_attrs`);
    - adds the Zarr DGGS v1 group metadata and chooses the chunks
      (:py:func:`prepare_dggs_zarr`);
    - writes Zarr format 2 with consolidated metadata.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        A dataset with a one-dimensional ``cell_ids`` coordinate.
    store : str or zarr store
        Where to write. A path or URL; ``.zarr`` is not appended.
    storage_options : dict, optional
        Passed to :py:meth:`xarray.Dataset.to_zarr`.
    mode : {"w", "w-"}, default: "w"
        ``"w"`` replaces an existing store, ``"w-"`` refuses to.
    target_chunk_bytes : int, default: 64 MiB
        Upper bound of one storage chunk, see :py:func:`prepare_dggs_zarr`.
    consolidated : bool, default: True
        Write consolidated metadata.

    Returns
    -------
    xarray.Dataset
        The prepared dataset that was written (lazy).
    """
    prepared = prepare_dggs_zarr(ds, target_chunk_bytes=target_chunk_bytes)
    prepared.to_zarr(
        store,
        mode=mode,
        storage_options=storage_options,
        zarr_format=2,
        consolidated=consolidated,
    )
    return prepared


def open_healpix_zarr(store, *, storage_options=None, chunks=None, **kwargs):
    """Open a HEALPix Zarr store written by the workflow.

    The store is opened with its storage chunks (each read once) and then
    rechunked to ``chunks`` along the dimensions it has. The grid attributes of
    ``cell_ids`` are put in canonical form, so stores written by earlier versions
    (e.g. with the ellipsoid as a dict of parameters) give the same dataset.

    Parameters
    ----------
    store : str or zarr store
    storage_options : dict, optional
    chunks : dict, optional
        Dask chunks to use after opening, e.g. ``{"time": 1, "cells": -1}``.
    **kwargs
        Passed to :py:func:`xarray.open_dataset`.
    """
    import xarray as xr

    ds = xr.open_dataset(
        store,
        engine="zarr",
        chunks={},
        storage_options=storage_options,
        **({"inline_array": True} | kwargs),
    )
    if chunks:
        ds = ds.chunk({dim: size for dim, size in chunks.items() if dim in ds.dims})
    return with_grid_attrs(ds)
