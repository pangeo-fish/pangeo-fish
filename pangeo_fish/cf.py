import pandas as pd
import xarray as xr


def bounds_to_bins_(bounds, bounds_dim="bounds"):
    dims = [dim for dim in bounds.dims if dim != bounds_dim]
    bins = pd.IntervalIndex.from_arrays(
        bounds.isel({bounds_dim: 0}), bounds.isel({bounds_dim: 1})
    )

    return xr.Variable(dims, bins)


def bounds_to_bins(ds, bounds_dim="bounds"):
    variables = {
        name: var for name, var in ds.variables.items() if bounds_dim in var.dims
    }
    bins = {
        f"{name.removesuffix('_bounds')}_bins": bounds_to_bins_(
            var, bounds_dim=bounds_dim
        )
        for name, var in variables.items()
    }

    return ds.assign_coords(bins)
