"""Zarr-Python 3 APIs, with Zarr format 2 on disk for existing readers."""

from math import gcd


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
