import more_itertools
import scipy.ndimage as filters
import xarray as xr


def blur_edges(arr, *, dims, method, **filter_kwargs):
    dims = list(more_itertools.always_iterable(dims, base_type=(str, bytes, tuple)))

    available_filters = [name for name in dir(filters) if name.endswith("_filter")]
    if method not in available_filters:
        raise ValueError(
            f"unknown method {method}. Choose one of [{', '.join(available_filters)}]"
        )

    return xr.apply_ufunc(
        getattr(filters, method),
        arr,
        kwargs=filter_kwargs,
        input_core_dims=[dims],
        output_core_dims=[dims],
        vectorize=True,
        dask="parallelized",
    )
