import itertools

import dask
import dask.array as da
import numba
import numpy as np
import xarray as xr

from ..distributions import gaussian_kernel
from ..pdf import combine_emission_pdf


def mean_track(X, coords=["latitude", "longitude"]):
    probabilities = X["states"]
    dims = list(
        set(itertools.chain.from_iterable(probabilities[name].dims for name in coords))
    )
    grid = probabilities.reset_coords(coords).get(coords)
    return grid.weighted(probabilities).mean(dim=dims)


def modal_track(X, coords=["latitude", "longitude"]):
    probabilities = X["states"]
    dims = list(
        set(itertools.chain.from_iterable(probabilities[name].dims for name in coords))
    )
    (indices,) = dask.compute(probabilities.argmax(dim=dims))

    return xr.Dataset(
        {name: probabilities[name][indices] for name in coords}
    ).reset_coords(coords)


@numba.njit
def kernel_overlap(pos, shape, kernel_shape):
    px, py = pos
    xmax, ymax = shape
    kx, ky = kernel_shape

    a_xmin = max(px - kx // 2, 0)
    a_xmax = min(px + kx // 2 + 1, xmax)
    a_ymin = max(py - ky // 2, 0)
    a_ymax = min(py + ky // 2 + 1, ymax)

    k_xmin = max(0 - px + kx // 2, 0)
    k_xmax = min(xmax - px + kx // 2, kx)
    k_ymin = max(0 - py + ky // 2, 0)
    k_ymax = min(ymax - py + ky // 2, kx)

    return ((a_xmin, a_xmax), (a_ymin, a_ymax)), ((k_xmin, k_xmax), (k_ymin, k_ymax))


def kernel_state_metric(previous_state_metric, previous_positions, pdf, kernel):
    possible_positions = previous_state_metric != -np.inf

    # output arrays
    state_metric = np.full_like(previous_state_metric, fill_value=-np.inf)
    positions = np.full_like(previous_positions, fill_value=-1)

    # actual processing
    nx, ny = pdf.shape
    for x in range(nx):
        for y in range(ny):
            # skip if unreachable
            if not possible_positions[x, y]:
                continue

            # compute overlap
            (ax, ay), (kx, ky) = kernel_overlap((x, y), pdf.shape, kernel.shape)

            # current branch metric
            branch_metric = (
                pdf[ax[0] : ax[1], ay[0] : ay[1]] + kernel[kx[0] : kx[1], ky[0] : ky[1]]
            )

            # current state metric
            new_state_metric = branch_metric + previous_state_metric[x, y]

            # write the result
            region = state_metric[ax[0] : ax[1], ay[0] : ay[1]]

            mask = new_state_metric > region

            state_metric[ax[0] : ax[1], ay[0] : ay[1]] = np.where(
                mask, new_state_metric, region
            )
            positions[ax[0] : ax[1], ay[0] : ay[1]] = np.where(
                mask, x * ny + y, positions[ax[0] : ax[1], ay[0] : ay[1]]
            )

    return state_metric, positions


def viterbi(emission, sigma):
    """
    Parameters
    ----------
    emission : Dataset
        The emission probability dataset containing land mask, initial
        and final probabilities and the different components of the
        pdf.
    sigma : float
        The coefficient of diffusion in pixel

    Returns
    -------
    state_metrics : DataArray
        The evolution of state metrics over time
    positions : DataArray
        The integer positions of the tracks
        TODO: translate this to label space without computing
    """

    def decode_most_probable_track(pdf, sigma, land_mask):
        kernel = gaussian_kernel(np.full(shape=(2,), fill_value=sigma))

        pos0 = np.argmax(pdf[0, ...]).compute()

        # all in log space
        state_metrics = np.empty_like(pdf)
        state_metrics[0, ...] = pdf[0, ...]

        branches = np.full_like(pdf, fill_value=-1)
        branches[0, ...] = np.where(pdf[0, ...] != -np.inf, pos0, -1)

        for index in range(1, pdf.shape[0]):
            state_metric, positions = da.apply_gufunc(
                kernel_state_metric,
                "(x,y),(x,y),(x,y),(n,m)->(x,y),(x,y)",
                state_metrics[index - 1, ...],
                branches[index - 1, ...],
                pdf[index, ...],
                np.log(kernel),
                output_dtypes=[float, int],
            )

            state_metrics[index, ...] = np.where(land_mask, -np.inf, state_metric)
            branches[index, ...] = np.where(land_mask, -1, positions)

        reshaped_branches = branches.reshape(branches.shape[0], -1)

        most_likely_position = np.argmax(state_metrics[-1, ...])

        positions = np.empty_like(state_metrics[:, 0, 0], dtype=int)
        positions[0] = most_likely_position
        for index in range(1, positions.shape[0]):
            branch_index = reshaped_branches.shape[0] - index
            positions[index] = reshaped_branches[branch_index, positions[index - 1]]

        return state_metrics, positions[::-1]

    emission_ = combine_emission_pdf(emission)
    pdf = emission_.pdf
    pdf[{"time": 0}] = emission_.initial
    pdf[{"time": -1}] = emission_.final

    state_metrics, positions = xr.apply_ufunc(
        decode_most_probable_track,
        np.log(pdf),
        sigma,
        emission.mask,
        input_core_dims=[("x", "y"), (), ("x", "y")],
        output_core_dims=[("x", "y"), ()],
        dask="allowed",
    )

    return xr.Dataset({"state_metrics": state_metrics, "track": positions})
