"""Implements diff operations between tags and reference data."""

import numba
import numpy as np
import xarray as xr
from numba import njit
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Numba guvectorize version of likelihood_fast — no Python loop, no scipy
# ---------------------------------------------------------------------------


@njit(inline="always")
def _interp1d_sorted(x_query, x_arr, y_arr):
    """Linear interpolation at x_query from sorted x_arr/y_arr, edge extrapolation."""
    n = len(x_arr)
    if x_query <= x_arr[0]:
        return y_arr[0]
    if x_query >= x_arr[n - 1]:
        return y_arr[n - 1]
    for i in range(n - 1):
        if x_arr[i] <= x_query <= x_arr[i + 1]:
            t = (x_query - x_arr[i]) / (x_arr[i + 1] - x_arr[i])
            return y_arr[i] + t * (y_arr[i + 1] - y_arr[i])
    return y_arr[n - 1]


_likelihood_gu_signatures = [
    "void(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
]


@numba.guvectorize(
    _likelihood_gu_signatures,
    "(z),(z),(o),(o),(o)->()",
    nopython=True,
)
def likelihood_fast_gu(
    model_temp, model_depth, tag_temp, tag_depth, var_at_depth, result
):
    """Guvectorized temperature likelihood — replaces likelihood_fast + vectorize=True.

    Same computation as likelihood_fast but runs entirely in compiled numba code:
    no Python loop, no scipy object creation per cell.

    Layout: model arrays have (z) depth levels, tag arrays have (o) observations.
    """
    # Check that model profile has at least one finite value
    has_model = False
    for i in range(len(model_temp)):
        if np.isfinite(model_temp[i]) and np.isfinite(model_depth[i]):
            has_model = True
            break
    if not has_model:
        result[0] = np.nan
        return

    log_sum = 0.0
    count = 0
    two_pi_log = 1.8378770664093453  # log(2*pi)

    for o in range(len(tag_depth)):
        if np.isnan(tag_depth[o]) or np.isnan(tag_temp[o]):
            continue
        mt = _interp1d_sorted(tag_depth[o], model_depth, model_temp)
        var = var_at_depth[o] + 0.01
        diff2 = (tag_temp[o] - mt) ** 2 / var
        log_sum += -0.5 * (two_pi_log + np.log(var) + diff2)
        count += 1

    if count == 0:
        result[0] = np.nan
    else:
        result[0] = np.exp(log_sum / count)


_diff_z_signatures = [
    "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    "void(float32[:], float32[:], float64[:], float64[:], float64[:])",
    "void(float64[:], float64[:], float32[:], float32[:], float64[:])",
]


@numba.guvectorize(_diff_z_signatures, "(z),(z),(o),(o)->()", nopython=True)
def _diff_z(model_temp, model_depth, tag_temp, tag_depth, result):
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


def diff_z_numba(model_temp, model_depth, tag_temp, tag_depth):
    with np.errstate(all="ignore"):
        # TODO: figure out why the "invalid value encountered" warning is raised
        return _diff_z(model_temp, model_depth, tag_temp, tag_depth)


def diff_z(model, tag, depth_threshold=0.8):
    diff = xr.apply_ufunc(
        diff_z_numba,
        model["TEMP"],
        model["dynamic_depth"],
        tag["temperature"],
        tag["pressure"],
        input_core_dims=[["depth"], ["depth"], ["obs"], ["obs"]],
        output_core_dims=[[]],
        exclude_dims={},
        vectorize=False,
        dask="parallelized",
        output_dtypes=[model.dtypes["TEMP"]],
    )
    original_units = model["TEMP"].attrs["units"]

    return diff.assign_attrs({"units": original_units}).to_dataset(name="diff")


def interp_var_at_depth(var_depth, var_values, tag_depth):
    """Interpolate variance values to target depths.

    Parameters
    ----------
    var_depth : array-like
        Depth bin centres at which *var_values* are defined.
    var_values : array-like
        Variance at each depth bin.
    tag_depth : array-like
        Target depths at which variance is needed.

    Returns
    -------
    np.ndarray
        Variance interpolated to *tag_depth*, with edge extrapolation.
    """
    f = interp1d(
        var_depth,
        var_values,
        bounds_error=False,
        fill_value=(var_values[0], var_values[-1]),
        axis=-1,
    )
    return f(tag_depth)


@njit
def fast_likelihood(tag_temp, model_temp, var_at_depth):
    """Compute Gaussian log-likelihood (numba-jitted inner loop).

    Parameters
    ----------
    tag_temp : 1-D float array
        Observed temperatures at each depth.
    model_temp : 1-D float array
        Model temperatures interpolated to the same depths.
    var_at_depth : 1-D float array
        Temperature variance at each depth.

    Returns
    -------
    float
        ``exp(mean(log p))`` over all depth levels.
    """
    diff_temp = (tag_temp - model_temp) ** 2 / (var_at_depth + 0.01)
    log_prob = -0.5 * (np.log(2 * np.pi) + np.log(var_at_depth + 0.01) + diff_temp)
    return np.exp(np.nanmean(log_prob))


def likelihood_fast(model_temp, model_depth, tag_temp, tag_depth, var_at_depth):
    """Compute temperature-profile likelihood for one grid cell / time step.

    Interpolates the model profile to the tag observation depths, then
    calls :func:`fast_likelihood`.

    Parameters
    ----------
    model_temp, model_depth : 1-D array
        Model temperature profile and corresponding depths.
    tag_temp, tag_depth : 1-D array
        Tag temperature observations and depths (NaN entries are dropped).
    var_at_depth : 1-D array
        Variance at each tag observation depth.

    Returns
    -------
    float
        Likelihood value in [0, 1], or NaN when the model profile is empty.
    """
    mask = ~np.isnan(tag_depth)
    tag_depth_ = tag_depth[mask]
    tag_temp_ = tag_temp[mask]
    var_at_depth_ = var_at_depth[mask]

    mask_model = np.isfinite(model_depth) & np.isfinite(model_temp)
    model_depth_ = model_depth[mask_model]
    model_temp_ = model_temp[mask_model]

    if model_depth_.size == 0:
        return np.nan

    model_temp_at_depth = interp1d(
        model_depth_,
        model_temp_,
        bounds_error=False,
        fill_value=(model_temp_[0], model_temp_[-1]),
    )(tag_depth_)

    return fast_likelihood(tag_temp_, model_temp_at_depth, var_at_depth_)
