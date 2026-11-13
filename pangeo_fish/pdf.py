"""Implements operations for merging probability distributions."""

import cf_xarray  # noqa: F401
import more_itertools
import scipy.stats
import xarray as xr
from more_itertools import first

from pangeo_fish.utils import _detect_spatial_dims, normalize


# also try: multivariate_normal, gaussian_kde
def normal(samples, mean, std, *, dims):
    """Evaluate the normal pdf element by element.

    The density of each sample only depends on that sample, so the computation
    works with any dask chunking: a store reopened with its on-disk chunks (for
    example several chunks along ``cells``) does not need rechunking first.

    Parameters
    ----------
    samples : xarray.DataArray
        The samples to compute the pdf from
    mean : float or xarray.DataArray
        The mean of the distribution. Arrays are broadcast by dimension name.
    std : float or xarray.DataArray
        The standard deviation of the distribution. Arrays are broadcast by
        dimension name.
    dims : list of hashable
        The spatial dimensions of ``samples``. They are moved to the end of the
        result, as before.

    Returns
    -------
    pdf : xarray.DataArray
        The computed pdf
    """
    result = xr.apply_ufunc(
        scipy.stats.norm.pdf,
        samples,
        mean,
        std,
        dask="parallelized",
        output_dtypes=[float],
    )
    result = result.transpose(..., *dims)
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
    normalized = pdf.pipe(normalize, spatial_dims).rename("pdf_normalized")
    return xr.merge([raw[exclude], normalized])
