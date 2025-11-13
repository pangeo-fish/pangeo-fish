"""Implements diff operations between tags and reference data."""

import numba
import numpy as np
import xarray as xr

_diff_z_signatures = [
    "void(float32[:], float32[:], float32, float32[:], float32[:], float32, float32[:])",
    "void(float64[:], float64[:], float64, float64[:], float64[:], float32, float64[:])",
    "void(float32[:], float32[:], float32, float64[:], float64[:], float32, float64[:])",
    "void(float64[:], float64[:], float64, float32[:], float32[:], float32, float64[:])",
]


@numba.guvectorize(_diff_z_signatures, "(z),(z),(),(o),(o),()->()", nopython=True)
def _diff_z(model_temp, model_depth, bottom, tag_temp, tag_depth, depth_thresh, result):
    if depth_thresh != 0 and bottom < np.max(tag_depth) * depth_thresh:
        result[0] = np.nan
        return

    diff_temp = np.full_like(tag_depth, fill_value=np.nan)
    mask = ~np.isnan(model_depth) & ~np.isnan(model_temp)
    model_depth_ = np.absolute(model_depth[mask])
    if model_depth_.size == 0:
        result[0] = np.nan
        return

    model_temp_ = model_temp[mask]

    for index in range(tag_depth.shape[0]):
        if not np.isnan(tag_depth[index]):
            diff_depth = np.absolute(model_depth_ - tag_depth[index])

            idx = np.argmin(diff_depth)

            diff_temp[index] = tag_temp[index] - np.absolute(model_temp_[idx])

    result[0] = np.mean(diff_temp[~np.isnan(diff_temp)])


def diff_z_numba(model_temp, model_depth, bottom, tag_temp, tag_depth, depth_thresh):
    with np.errstate(all="ignore"):
        # TODO: figure out why the "invalid value encountered" warning is raised
        return _diff_z(
            model_temp, model_depth, bottom, tag_temp, tag_depth, depth_thresh
        )


def diff_z(model, tag, depth_threshold=0.8):
    diff = xr.apply_ufunc(
        diff_z_numba,
        model["TEMP"],
        model["dynamic_depth"],
        model["dynamic_bathymetry"],
        tag["temperature"],
        tag["pressure"],
        kwargs={"depth_thresh": depth_threshold},
        input_core_dims=[["depth"], ["depth"], [], ["obs"], ["obs"]],
        output_core_dims=[[]],
        exclude_dims={},
        vectorize=False,
        dask="parallelized",
        output_dtypes=[model.dtypes["TEMP"]],
    )
    original_units = model["TEMP"].attrs["units"]

    return diff.assign_attrs({"units": original_units}).to_dataset(name="diff")


############### Second Function with sigma

import numba
import numpy as np

# Signatures: float32 and float64
_diff_z_var_signatures = [
    "void(float32[:], float32[:], float32, float32[:], float32[:], float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:], float64[:])",
]


@numba.njit
def interp1d_numba(x, y, xi):
    """
    Manual 1D linear interpolation (replacement for SciPy's interp1d).
    x, y: 1D vectors (monotonic)
    xi: vector of points where to interpolate
    """
    yi = np.empty_like(xi)
    for i in range(xi.shape[0]):
        v = xi[i]
        if np.isnan(v):
            yi[i] = np.nan
            continue
        if v <= x[0]:
            yi[i] = y[0]
        elif v >= x[-1]:
            yi[i] = y[-1]
        else:
            j = np.searchsorted(x, v) - 1
            x0, x1 = x[j], x[j + 1]
            y0, y1 = y[j], y[j + 1]
            yi[i] = y0 + (y1 - y0) * (v - x0) / (x1 - x0)
    return yi


@numba.njit
def interp_var_numba(var_depth, var_values, tag_depth):
    """Manual linear interpolation of variance values at the tag depths."""
    return interp1d_numba(var_depth, var_values, tag_depth)


@numba.guvectorize(
    _diff_z_var_signatures,
    "(z),(z),(),(o),(o),(p),(p)->()",
    nopython=True,
)
def _diff_z_var(
    model_temp, model_depth, bottom, tag_temp, tag_depth, var_depth, var_values, result
):
    """
    Computes the mean of ((tag_temp - model_temp_interp)**2) / var_at_depth,
    using manual linear interpolation.
    """
    model_temp_at_depth = interp1d_numba(model_depth, model_temp, tag_depth)
    var_at_the_depth = interp_var_numba(var_depth, var_values, tag_depth)

    diff_temp = (tag_temp - model_temp_at_depth) ** 2 / var_at_the_depth
    result[0] = np.nanmean(diff_temp)


def diff_z_var(model, tag, var_depth, var_values):
    """
    Compute the variance-weighted squared difference between model and tag temperatures.
    """
    diff = xr.apply_ufunc(
        _diff_z_var,
        model["TEMP"],  # Model temperature
        model["dynamic_depth"],  # Model depth
        model["dynamic_bathymetry"],  # Bottom depth
        tag["temperature"],  # Tag-measured temperature
        tag["pressure"],  # Tag-measured depth (pressure)
        var_depth,  # Depth bins of the variance
        var_values,  # Variance values
        input_core_dims=[
            ["depth"],
            ["depth"],
            [],
            ["obs"],
            ["obs"],
            ["depth_var"],
            ["depth_var"],
        ],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=False,  # Vectorization is handled by guvectorize
        output_dtypes=[model["TEMP"].dtype],
    )
    diff.attrs["units"] = model["TEMP"].attrs.get("units", "")
    return diff.to_dataset(name="diff")
