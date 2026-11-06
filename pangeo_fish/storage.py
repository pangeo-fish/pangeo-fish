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
            store, mode="r+", region={"time": slice(start, end)},
            zarr_format=2, consolidated=False,
        )
    if static:
        ds[static].to_zarr(store, mode="a", zarr_format=2, consolidated=False)

