import cf_xarray  # noqa: F401
import more_itertools
import scipy.stats
import xarray as xr

from .utils import _detect_spatial_dims, clear_attrs, normalize


# also try: multivariate_normal, gaussian_kde
# TODO: use numba to vectorize (using `guvectorize`?)
def normal(samples, mean, std, *, dim):
    """compute the combined pdf of independent layers

    Parameters
    ----------
    samples : DataArray, Variable, or array-like
        The samples to compute the pdf from
    mean : float
        The mean of the distribution
    std : float
        The standard deviation of the distribution
    dim : hashable or list of hashable
        The dimension to compute the pdf along

    Returns
    -------
    pdf : DataArray
        The computed pdf
    """

    def _pdf(samples, mean, std):
        return scipy.stats.norm.pdf(samples, mean, std)

    result = xr.apply_ufunc(
        _pdf,
        samples,
        mean,
        std**2,
        dask="parallelized",
        input_core_dims=[[dim], [dim], [dim]],
        output_core_dims=[[dim]],
    )
    return result.rename("pdf").pipe(clear_attrs)


def combine_emission_pdf(raw, exclude=("initial", "final", "mask")):
    exclude = list(more_itertools.always_iterable(exclude))

    pdfs = raw[[name for name in raw.data_vars if name not in exclude]]
    pdf = pdfs.to_array(dim="pdf").prod(dim="pdf", skipna=False).rename("pdf")

    spatial_dims = _detect_spatial_dims(raw)

    return xr.merge([raw[exclude], pdf.pipe(normalize, spatial_dims)])
