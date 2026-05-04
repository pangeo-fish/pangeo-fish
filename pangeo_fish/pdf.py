"""Implements operations for merging probability distributions."""

import cf_xarray  # noqa: F401
import more_itertools
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


def combine_emission_pdf_with_light(
    pdf_diff,
    pdf_bathy,
    pdf_light=None,
    pdf_moon=None,
):
    """Combine emission PDFs including optional light and moon components.

    Wraps :func:`combine_emission_pdf` to support up to four independent
    emission components.  All inputs **must already be on the same HEALPix
    grid** — regridding should be performed in the notebook before calling
    this function.

    When both ``pdf_light`` and ``pdf_moon`` are ``None``, the result is
    identical to calling :func:`combine_emission_pdf` directly on
    ``pdf_diff × pdf_bathy``, ensuring backwards compatibility.

    Days where ``pdf_light`` or ``pdf_moon`` is ``NaN`` are filled with
    1.0 (the multiplicative identity) so that missing light data never
    zeros out a timestep.

    Parameters
    ----------
    pdf_diff : xr.Dataset
        Temperature-difference emission PDF.  Must contain a ``'pdf'``
        data variable and optionally ``'initial'``, ``'final'``,
        ``'mask'``.
    pdf_bathy : xr.Dataset
        Bathymetry emission PDF.  Must contain a ``'pdf_bathy'`` data
        variable.
    pdf_light : xr.DataArray or None, default None
        Solar threshold likelihood (HEALPix, same grid as ``pdf_diff``).
        Named ``'pdf_light'``.  Missing timesteps should be ``NaN``.
    pdf_moon : xr.DataArray or None, default None
        Lunar template-fit likelihood (HEALPix, same grid as
        ``pdf_diff``).  Named ``'pdf_moon'``.  Missing nights should be
        ``NaN``.

    Returns
    -------
    xr.Dataset
        Merged and normalised emission PDF with a single ``'pdf'``
        variable, compatible with the downstream HMM workflow.
    """
    # Start from the two core components
    merged = pdf_diff.merge(pdf_bathy, compat="override")

    # Add light component if provided
    if pdf_light is not None:
        # NaN → 1.0 (neutral for multiplication)
        light_filled = pdf_light.fillna(1.0)
        if isinstance(light_filled, xr.DataArray):
            light_filled = light_filled.rename("pdf_light").to_dataset()
        merged = merged.merge(light_filled, compat="no_conflicts")

    # Add moon component if provided
    if pdf_moon is not None:
        moon_filled = pdf_moon.fillna(1.0)
        if isinstance(moon_filled, xr.DataArray):
            moon_filled = moon_filled.rename("pdf_moon").to_dataset()
        merged = merged.merge(moon_filled, compat="no_conflicts")

    return combine_emission_pdf(merged)
