"""Implements operations for merging probability distributions."""

import cf_xarray  # noqa: F401
import more_itertools
import numpy as np
import scipy.stats
import xarray as xr
from more_itertools import first

from pangeo_fish.utils import _detect_spatial_dims, normalize


def normal(samples, mean, std, *, dims):
    """
    Compute the combined pdf of independent layers.
    """

    def _pdf(samples, mean, std):
        return scipy.stats.norm.pdf(samples, mean, std)

    if not hasattr(mean, "dims"):
        mean = xr.zeros_like(samples) + mean

    # --- Choix des core-dims en fonction de std ---
    is_scalar_std = (
        np.isscalar(std)
        or (hasattr(std, "ndim") and std.ndim == 0)
        or (hasattr(std, "size") and np.size(std) == 1)
    )

    if is_scalar_std:
        # std scalar
        input_core_dims = [["cells"], ["cells"], []]
    elif hasattr(std, "dims") and ("time" in std.dims):
        # std(time)
        input_core_dims = [dims, dims, []]

    result = xr.apply_ufunc(
        _pdf,
        samples,
        mean,
        std,
        dask="parallelized",
        input_core_dims=input_core_dims,
        output_core_dims=[dims],
        output_dtypes=[samples.dtype],
        vectorize=True,
    )

    return result.rename("pdf").drop_attrs(deep=False)


# def normal(samples, mean, std, *, dims):
#     """
#     Compute the combined pdf of independent layers.
#     Works with:
#       - scalar std (σ constant)
#       - std(time) (σ dépend du temps)
#       - std(cells,time) (σ dépend des deux)
#     """

#     def _pdf(samples, mean, std):
#         return scipy.stats.norm.pdf(samples, mean, std)

#     # mean scalaire -> DataArray aligné à samples
#     if not hasattr(mean, "dims"):
#         mean = xr.zeros_like(samples) + mean

#     # Déterminer les dimensions de std
#     is_scalar_std = np.isscalar(std) or (
#         hasattr(std, "size") and np.size(std) == 1
#     )

#     if is_scalar_std:
#         param_dims = []
#     elif hasattr(std, "dims"):
#         param_dims = list(std.dims)
#     else:
#         # Cas liste -> on suppose variance déjà transformée en std(time)
#         param_dims = ["time"]
#         std = xr.DataArray(std, dims=param_dims)

#     # Pas de sqrt ici — std est déjà un écart-type
#     result = xr.apply_ufunc(
#         _pdf,
#         samples,
#         mean,
#         std,
#         dask="parallelized",
#         input_core_dims=[dims, dims, param_dims],
#         output_core_dims=[[]],
#         vectorize=True,
#         output_dtypes=[samples.dtype],
#         dask_gufunc_kwargs={"allow_rechunk": True},
#     )

#     return result.rename("pdf").drop_attrs(deep=False)


def combine_emission_pdf(raw, exclude=("initial", "final", "mask")):
    exclude = [n for n in more_itertools.always_iterable(exclude) if n in raw.variables]

    to_combine = [name for name in raw.data_vars if name not in exclude]
    if len(to_combine) == 1:
        pdf = raw[first(to_combine)].rename("pdf")
    else:
        pdf = (
            raw[to_combine]
            .to_array(dim="pdf")
            .prod(dim="pdf", skipna=False)
            .rename("pdf")
        )

    if "final" in raw:
        pdf[{"time": -1}] = pdf[{"time": -1}] * raw["final"]

    spatial_dims = _detect_spatial_dims(raw)
    return xr.merge([raw[exclude], pdf.pipe(normalize, spatial_dims)])
