"""Zarr-Python 3 APIs, with Zarr format 2 on disk for existing readers."""

from math import gcd
from numbers import Integral

import numpy as np


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
    """Return ``ds`` with the DGGS grid attributes written on ``cell_ids``.

    The attributes follow the Zarr DGGS convention used by xdggs (``grid_name``,
    ``level``, ``indexing_scheme`` and ``ellipsoid``). They are taken from the
    xdggs index when the dataset has one, else from the existing ``cell_ids``
    attributes, so that the grid definition survives a round trip to Zarr.
    """
    if "cell_ids" not in ds.variables:
        return ds

    try:
        grid_attrs = ds.dggs.grid_info.to_dict()
    except Exception:
        attrs = ds["cell_ids"].attrs
        if "level" not in attrs:
            return ds
        grid_attrs = {
            "grid_name": attrs.get("grid_name", "healpix"),
            "level": int(attrs["level"]),
            "indexing_scheme": attrs.get("indexing_scheme", "nested"),
        }
        if attrs.get("ellipsoid") is not None:
            grid_attrs["ellipsoid"] = attrs["ellipsoid"]

    ds = ds.copy()
    ds["cell_ids"].attrs.update(grid_attrs)
    return ds


_DGGS_CONVENTION = {
    "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38",
    "name": "dggs",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "description": "Discrete Global Grid Systems convention for zarr",
}


def prepare_dggs_zarr(ds, *, cell_chunk_size=4096):
    """Prepare a HEALPix dataset for a new Zarr store without adding cells.

    Write the DGGS v1 group metadata alongside the xdggs coordinate
    attributes. Use one time step per chunk and bounded spatial chunks, with
    cells last, so HTTP readers can fetch a single map. Sparse cells retain
    their order; these chunks need not coincide with HEALPix parent cells.
    The input dataset and its encodings are not modified.
    """
    import healpix_geo.ellipsoid
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    if not isinstance(cell_chunk_size, Integral) or cell_chunk_size <= 0:
        raise ValueError("cell_chunk_size must be a positive integer")
    ds = _with_dggs_attrs(ds).copy(deep=False)
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
    ellipsoid = attrs.get("ellipsoid")
    if ellipsoid is None:
        ellipsoid = "sphere"
    if isinstance(ellipsoid, str):
        ellipsoid = healpix_geo.ellipsoid.resolve(ellipsoid)
    elif hasattr(ellipsoid, "to_dict"):
        ellipsoid = ellipsoid.to_dict()
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
    ds = ds.transpose(..., dim)
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
