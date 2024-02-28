import numpy as np
import xarray as xr


def to_time_slice(times):
    subset = times.where(times.notnull(), drop=True)

    min_ = subset.isel(event_name=0)
    max_ = subset.isel(event_name=1)

    return slice(min_.data, max_.data)


def adapt_model_time(slice_):
    model_start = slice_.start - np.timedelta64(30, "m")
    model_stop = slice_.stop + np.timedelta64(30, "m")

    return slice(model_start, model_stop)


def assign_group_labels(ds, *, dim, index, bin_dim, other_dim):
    value = ds[dim].isel({dim: 0}).to_pandas()
    indexer = index.get_loc(value)

    return ds.assign_coords(
        {
            bin_dim: xr.full_like(ds[dim], fill_value=indexer, dtype=int),
            other_dim: (dim, np.arange(ds.sizes[dim])),
        }
    )


def reshape_by_bins(ds, *, dim, bins, bin_dim="bincount", other_dim="obs"):
    index = bins.to_index().astype("interval")
    vertices = np.concatenate([index.left, index.right[-1:]])

    grouped = ds.groupby_bins(dim, bins=vertices)
    processed = grouped.map(
        assign_group_labels, dim=dim, bin_dim=bin_dim, other_dim=other_dim, index=index
    )

    return (
        processed.swap_dims({dim: other_dim})
        .set_index({"stacked": [bin_dim, other_dim]})
        .unstack("stacked")
        .assign_coords({dim: lambda ds: bins[dim].isel({dim: ds[bin_dim]})})
        .swap_dims({bin_dim: dim})
        .drop_vars([bin_dim, bins.name], errors="ignore")
    )
