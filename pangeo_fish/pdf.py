"""Implements operations for merging probability distributions."""

import warnings

import cf_xarray  # noqa: F401
import more_itertools
import numpy as np
import scipy.stats
import xarray as xr
from more_itertools import first

from pangeo_fish.utils import _detect_spatial_dims, normalize


# also try: multivariate_normal, gaussian_kde
# TODO: use numba to vectorize (using `guvectorize`?)
def normal(samples, mean, std, *, dims):
    """Compute the combined pdf of independent layers

    Parameters
    ----------
    samples : xarray.DataArray, Variable, or array-like
        The samples to compute the pdf from
    mean : float
        The mean of the distribution
    std : float
        The standard deviation of the distribution
    dims : list of hashable
        The dimension to compute the pdf along

    Returns
    -------
    pdf : xarray.DataArray
        The computed pdf
    """

    def _pdf(samples, mean, std):
        return scipy.stats.norm.pdf(samples, mean, std)

    if isinstance(std, int | float) or std.size == 1:
        param_dims = []
    else:
        param_dims = mean.dims

    result = xr.apply_ufunc(
        _pdf,
        samples,
        mean,
        std**2,
        dask="parallelized",
        input_core_dims=[dims, param_dims, param_dims],
        output_core_dims=[dims],
        exclude_dims=set(param_dims),
        vectorize=True,
    )
    return result.rename("pdf").drop_attrs(deep=False)


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


def normalize_pdf_by_mask(ds, mask_var="mask", pdf_var="pdf", tol=1e-12):
    """Replace zero-mass timesteps with a uniform distribution over ocean cells.

    When a PDF variable sums to zero or NaN over the ocean mask for a given
    timestep (e.g. because the fish was too deep, or all pixels had zero
    likelihood), this function replaces that timestep with a flat prior
    ``1 / n_ocean`` over all ocean cells.  Timesteps with valid mass are
    left unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``pdf_var`` (shape ``(time, cells)``) and
        ``mask_var`` (shape ``(cells,)``, boolean ocean mask).
    mask_var : str, default ``"mask"``
        Name of the ocean-mask variable (True = ocean cell).
    pdf_var : str, default ``"pdf"``
        Name of the PDF variable to fix.
    tol : float, default 1e-12
        Threshold below which a per-timestep sum is considered zero.

    Returns
    -------
    xr.Dataset
        Copy of ``ds`` with ``pdf_var`` corrected.
    """
    mask_cells = ds[mask_var].astype(bool)
    mask_time = mask_cells.expand_dims(time=ds["time"]).transpose("time", "cells")
    n_valid = int(mask_cells.sum().compute().item())
    if n_valid == 0:
        raise ValueError("Mask has 0 valid (ocean) cells — cannot normalise.")
    sums = ds[pdf_var].where(mask_time).fillna(0).sum(dim="cells").compute()
    to_fix = (sums <= tol) | np.isnan(sums)
    if not to_fix.any():
        return ds
    n_bad = int(to_fix.sum().item())
    warnings.warn(
        f"normalize_pdf_by_mask: {n_bad} timestep(s) had zero/NaN mass "
        f"in '{pdf_var}' — replaced with uniform prior.",
        UserWarning,
    )
    fill = xr.where(to_fix, 1.0 / float(n_valid), np.nan)
    fill = xr.DataArray(fill, coords={"time": ds["time"]}, dims=["time"])
    replacement = xr.where(mask_time, fill, np.nan)
    ds_fixed = ds.copy()
    ds_fixed[pdf_var] = xr.where(to_fix, replacement, ds[pdf_var])
    return ds_fixed
