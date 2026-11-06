import itertools

import dask
import dask.array as da
import healpix_geo.nested as hgn
import healpix_geo.nested as hgn
import numba
import numpy as np
import xarray as xr
from scipy.special import logsumexp

from pangeo_fish.distributions import gaussian_kernel


def mean_track(X, coords=["latitude", "longitude"]):
    probabilities = X["states"]
    dims = list(
        set(itertools.chain.from_iterable(probabilities[name].dims for name in coords))
    )
    grid = probabilities.reset_coords(coords).get(coords)
    return grid.weighted(probabilities).mean(dim=dims)


def modal_track(X, coords=["latitude", "longitude"]):
    probabilities = X["states"].drop_vars("cell_ids")
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


@numba.njit
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
    emission : xarray.Dataset
        The emission probability dataset containing land mask and initial
        probabilities and the pdf.
    sigma : float
        The coefficient of diffusion in pixel

    Returns
    -------
    state_metrics : xarray.DataArray
        The evolution of state metrics over time
    positions : xarray.DataArray
        The integer positions of the tracks
        TODO: translate this to label space without computing
    """

    def apply_kernel_state_metric(
        pdf, previous_state_metric, previous_branches, kernel
    ):
        if isinstance(pdf, da.Array):
            return da.apply_gufunc(
                kernel_state_metric,
                "(x,y),(x,y),(x,y),(n,m)->(x,y),(x,y)",
                previous_state_metric,
                previous_branches,
                pdf,
                kernel,
                output_dtypes=[float, int],
            )
        else:
            return kernel_state_metric(
                previous_state_metric, previous_branches, pdf, kernel
            )

    def decode_most_probable_track(pdf, sigma, ocean_mask):
        kernel = gaussian_kernel(np.full(shape=(2,), fill_value=sigma))

        [pos0] = dask.compute(np.argmax(pdf[0, ...]))

        # all in log space
        state_metrics_ = [pdf[0, ...]]

        initial_branches = np.where(pdf[0, ...] != -np.inf, pos0, -1)
        branches_ = [initial_branches]

        for index in range(1, pdf.shape[0]):
            state_metric, positions = apply_kernel_state_metric(
                pdf[index, ...],
                state_metrics_[index - 1],
                branches_[index - 1],
                np.log(kernel),
            )

            state_metrics_.append(np.where(ocean_mask, state_metric, -np.inf))
            branches_.append(np.where(ocean_mask, positions, -1))

        state_metrics = np.stack(state_metrics_, axis=0)
        branches = np.stack(branches_, axis=0)

        reshaped_branches = branches.reshape(branches.shape[0], -1)

        most_likely_position = np.argmax(state_metrics[-1, ...])

        positions = [most_likely_position]
        for index in range(1, state_metrics.shape[0]):
            branch_index = reshaped_branches.shape[0] - index
            positions.append(reshaped_branches[branch_index, positions[index - 1]])

        return state_metrics, np.stack(positions[::-1])

    pdf = emission["pdf"].copy()
    pdf[{"time": 0}] = emission.initial

    state_metrics, positions = xr.apply_ufunc(
        decode_most_probable_track,
        np.log(pdf.fillna(0)),
        sigma,
        emission.mask,
        input_core_dims=[("x", "y"), (), ("x", "y")],
        output_core_dims=[("x", "y"), ()],
        dask="allowed",
    )

    coords = emission[["longitude", "latitude"]].stack(z=["x", "y"])
    track = coords.isel(z=positions, drop=True)

    return track.drop_vars(["cell_ids", "z", "x", "y"])


@numba.njit
def _propagate_timestep(M, kernel, emission, Tprevx, Tprevy):
    row, col = M.shape
    ks = kernel.shape[0]

    Mtemp = np.full((row, col), fill_value=-np.inf)
    Ttempx = np.full((row, col), fill_value=-1, dtype="int16")
    Ttempy = np.full((row, col), fill_value=-1, dtype="int16")

    for x in range(col):
        for y in range(row):
            if M[y, x] == -np.inf:
                continue

            kminlat = max(ks // 2 - y, 0)
            kmaxlat = min((ks - 1) - (y + ks // 2 - (row - 1)), ks - 1)
            kminlong = max(ks // 2 - x, 0)
            kmaxlong = min((ks - 1) - (x + ks // 2 - (col - 1)), ks - 1)

            mminlat = max(y - ks // 2, 0)
            mmaxlat = min(y + ks // 2, row - 1)
            mminlong = max(x - ks // 2, 0)
            mmaxlong = min(x + ks // 2, col - 1)

            B = (
                emission[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]
                + kernel[kminlat : kmaxlat + 1, kminlong : kmaxlong + 1]
            )

            Msub = B + M[y, x]

            Mupdate = Mtemp[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]
            Txupdate = Ttempx[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]
            Tyupdate = Ttempy[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]

            update = Msub > Mupdate

            Mtemp[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1] = np.where(
                update, Msub, Mupdate
            )
            Ttempx[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1] = np.where(
                update, x, Txupdate
            )
            Ttempy[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1] = np.where(
                update, y, Tyupdate
            )

    return Mtemp, Ttempx, Ttempy


@numba.njit
def _reorder_track(Tprevx, Tprevy, Ttempx, Ttempy, index, M):
    row, col = M.shape

    Tx = np.full_like(Tprevx, fill_value=-1)
    Ty = np.full_like(Tprevy, fill_value=-1)

    for x in range(col):
        for y in range(row):
            if M[y, x] == -np.inf:
                continue

            Tx[y, x, :index] = Tprevx[Ttempy[y, x], Ttempx[y, x], :index]
            Ty[y, x, :index] = Tprevy[Ttempy[y, x], Ttempx[y, x], :index]
            Tx[y, x, index] = x
            Ty[y, x, index] = y

    return Tx, Ty


def _viterbi(emission, land_mask, sigma):
    kernel = np.log(gaussian_kernel(np.array([sigma, sigma]), type="discrete"))
    emission_ = np.log(emission)

    M = dask.compute(emission_[0, ...])[0]

    pos0 = np.argmax(M)
    y0, x0 = np.unravel_index(pos0, M.shape)

    Tprevx = np.full(M.shape + emission_.shape[:1], fill_value=-1, dtype="int16")
    Tprevx[y0, x0, 0] = x0
    Tprevy = np.full(M.shape + emission_.shape[:1], fill_value=-1, dtype="int16")
    Tprevy[y0, x0, 0] = y0

    for index in range(1, emission_.shape[0]):
        pdf = dask.compute(emission_[index, ...])[0]

        Mtemp, Ttempx, Ttempy = _propagate_timestep(M, kernel, pdf, Tprevx, Tprevy)
        Mtemp[land_mask] = -np.inf

        Tx, Ty = _reorder_track(Tprevx, Tprevy, Ttempx, Ttempy, index, Mtemp)

        M = Mtemp
        Tprevx = Tx
        Tprevy = Ty

    reshaped_M = np.reshape(M, -1)
    reshaped_x = np.reshape(Tprevx, (-1, emission.shape[0]))
    reshaped_y = np.reshape(Tprevy, (-1, emission.shape[0]))

    pos = np.argmax(reshaped_M)
    y, x = reshaped_y[pos, :], reshaped_x[pos, :]

    return y, x


def viterbi2(emission, sigma):
    pdf = emission["pdf"].copy()
    pdf[{"time": 0}] = emission.initial

    # different order for x and y, so we need to swap it
    x, y = xr.apply_ufunc(
        _viterbi,
        pdf.fillna(0),
        np.logical_not(emission.mask),  # ocean mask → land mask
        input_core_dims=[("x", "y"), ("x", "y")],
        output_core_dims=[(), ()],
        kwargs={"sigma": sigma},
        dask="allowed",
    )

    return (
        emission[["time", "latitude", "longitude"]].isel(x=x, y=y).drop_vars("cell_ids")
    )


### Healpix viterbi try ###





def _angular_distances(cell_ids, disk, level, ellipsoid="sphere"):
    """
    Angular distances (radians) between each cell centre and its neighbourhood.

    Parameters
    ----------
    cell_ids : numpy.ndarray of uint64, shape (n,)
        Source cell indexes at depth `level`.
    disk : numpy.ndarray of int64, shape (n, M)
        Neighbourhood of each source cell, as returned by `kth_neighbourhood`.
        Entries equal to -1 are padding.
    level : int
        HEALPix depth of the cells.
    ellipsoid : str, default "sphere"
        Reference ellipsoid.

    Returns
    -------
    numpy.ndarray of float, shape (n, M)
        Distances in radians, NaN where `disk` is padding.
    """
    valid = disk >= 0

    if ellipsoid == "sphere":
        return np.asarray(hgn.angular_distances(cell_ids, disk, level))

    # Ellipsoid case: geocentric angle between the centre vectors.
    # This is not the geodesic distance on the ellipsoid; the difference is
    # small but grows at high latitude.
    ids = np.where(valid, disk, 0).astype(np.uint64)

    depth_c = np.full(cell_ids.size, level, dtype=np.uint8)
    depth_d = np.full(ids.size, level, dtype=np.uint8)

    # centre vectors of the sources, shape (n, 3)
    pc = np.stack(
        hgn.healpix_to_cartesian(cell_ids, depth_c, ellipsoid=ellipsoid), axis=-1
    )
    # centre vectors of the neighbours, shape (n, M, 3)
    pd = np.stack(
        hgn.healpix_to_cartesian(ids.ravel(), depth_d, ellipsoid=ellipsoid), axis=-1
    ).reshape(ids.shape + (3,))

    nc = np.linalg.norm(pc, axis=-1)  # (n,)
    nd = np.linalg.norm(pd, axis=-1)  # (n, M)
    dot = np.einsum("nmk,nk->nm", pd, pc)

    cosang = dot / (nd * nc[:, None])
    d = np.arccos(np.clip(cosang, -1.0, 1.0))

    return np.where(valid, d, np.nan)


def build_transitions(cell_ids, level, sigmas, truncate=4.0, ellipsoid="sphere"):
    """
    Neighbours and log-transition weights for each sigma (radians).

    Returns
    -------
    neighbors : numpy.ndarray of int, shape (n_sigma, n, K)
        Local indices (0..n-1) of the predecessors j of each cell i.
        Padding entries point to i itself.
    logT : numpy.ndarray of float, shape (n_sigma, n, K)
        log P(i | j) for j = neighbors[s, i, k].
        Padding entries are -inf.

    Notes
    -----
    Pull convention: for each destination i, we list its predecessors j.
    P(i | j) = exp(w(i, j)) / Z_j, where Z_j sums over the whole disk of j,
    including cells outside the domain. This matches the normalisation used
    by Gaussian1DHealpix, where mass leaving the domain is not renormalised.
    """
    if ellipsoid is None:
        ellipsoid = "sphere"

    cell_ids = np.asarray(cell_ids, dtype=np.uint64)
    n = cell_ids.size

    # sorted lookup table, to map a HEALPix index to a local index
    order = np.argsort(cell_ids)
    sorted_ids = cell_ids[order]

    # approximate cell size (radians), used to choose the number of rings
    pix_size = np.sqrt(np.pi / 3) / (2**level)

    all_nbr, all_logT = [], []

    for sigma in np.atleast_1d(np.asarray(sigmas, dtype=float)):
        radius = truncate * sigma
        # kth_neighbourhood(ring=k) returns the full disk of radius k cells
        rings = int(np.ceil(radius / pix_size)) + 1

        # full disk around each cell, shape (n, M), -1 is padding
        disk = np.asarray(
            hgn.kth_neighbourhood(cell_ids, level, rings, num_threads=0),
            dtype=np.int64,
        )
        valid_hp = disk >= 0

        # angular distances, NaN on padding
        d = _angular_distances(cell_ids, disk, level, ellipsoid=ellipsoid)

        # Gaussian log-weights, -inf outside the radius and on padding
        inside = valid_hp & np.isfinite(d) & (d <= radius)
        w = np.where(inside, -(np.nan_to_num(d) ** 2) / (2.0 * sigma**2), -np.inf)

        # normalisation over the whole disk of each source (outside cells included)
        logZ = logsumexp(w, axis=1)  # (n,)

        # padding replaced by 0 so that searchsorted never sees -1
        ids = np.where(valid_hp, disk, 0).astype(np.uint64)

        # keep only neighbours that belong to the domain, and map them to local indices
        pos = np.searchsorted(sorted_ids, ids.ravel()).reshape(ids.shape)
        pos = np.clip(pos, 0, n - 1)
        in_dom = (sorted_ids[pos] == ids) & np.isfinite(w)
        loc = order[pos]

        # compact each row so the valid entries come first
        perm = np.argsort(~in_dom, axis=1, kind="stable")
        in_dom_s = np.take_along_axis(in_dom, perm, axis=1)
        w_s = np.take_along_axis(w, perm, axis=1)
        loc_s = np.take_along_axis(loc, perm, axis=1)

        # trim to the maximum number of valid neighbours
        K = max(int(in_dom.sum(axis=1).max()), 1)
        print(
            f"sigma={sigma:.4g} rad | rings={rings} | "
            f"M(candidats)={disk.shape[1]} | K(valides)={K} | "
            f"taille table ~{n * K * (4 + 4) / 1e6:.0f} Mo"
        )
        in_dom_s, w_s, loc_s = in_dom_s[:, :K], w_s[:, :K], loc_s[:, :K]

        nbr = np.where(in_dom_s, loc_s, np.arange(n)[:, None])
        logT = np.where(in_dom_s, w_s - logZ[nbr], -np.inf)

        all_nbr.append(nbr)
        all_logT.append(logT)

    return np.stack(all_nbr), np.stack(all_logT)


def viterbi_healpix(emission, sigmas, truncate=4.0, ellipsoid="sphere", level=None):
    """
    Viterbi decoding on a HEALPix grid, with a sigma that may change per time step.

    Parameters
    ----------
    emission : xarray.Dataset
        Dimensions (time, cells) for 'pdf'; 'initial' on cells;
        'mask' on cells (True = ocean); 'predictor_index' on time
        (index into `sigmas` used for the transition into each step);
        and 'cell_ids'.
    sigmas : sequence of float
        Standard deviations in radians.
    truncate : float, default 4.0
        Truncation radius in units of sigma.
    ellipsoid : str, default "sphere"
        Reference ellipsoid.

    Returns
    -------
    xarray.Dataset
        Most probable track, with 'cell_ids', 'longitude' and 'latitude' on time.
    """
    if ellipsoid is None:
        ellipsoid = "sphere"
    if level is None:
        level = emission.dggs.grid_info.level

    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=float))
    cell_ids = np.asarray(emission["cell_ids"].values, dtype=np.uint64)

    pdf = emission["pdf"].transpose("time", "cells").values
    initial = emission["initial"].transpose("cells").values
    ocean = np.asarray(emission["mask"].transpose("cells").values, dtype=bool)
    sig_idx = np.asarray(emission["predictor_index"].values, dtype=int)

    T, n = pdf.shape
    print(f"T={T},n={n}")
    if sig_idx.min() < 0 or sig_idx.max() >= len(sigmas):
        raise ValueError("predictor_index is out of range of the sigma list")

    # emissions in log space; the first step uses the initial distribution
    with np.errstate(divide="ignore"):
        logE = np.log(np.nan_to_num(pdf, nan=0.0))
        logE[0] = np.log(np.nan_to_num(initial, nan=0.0))
    print("début neighbors")
    # sigmas arrondis pour éviter les doublons dus au flottant
    sigmas_r = np.array([float(f"{s:.10g}") for s in sigmas])

    # sigmas réellement utilisés par predictor_index, sans doublons
    used = np.unique(sig_idx)
    unique_vals, inverse = np.unique(sigmas_r[used], return_inverse=True)

    # nouvel index de chaque pas, dans la liste réduite
    remap = np.full(len(sigmas), -1, dtype=int)
    remap[used] = inverse
    sig_idx_u = remap[sig_idx]
    print(f"nombre de sigmas demandés : {len(sigmas)}")
    print(f"sigmas uniques utilisés   : {len(unique_vals)}")
    print(f"valeurs                   : {unique_vals}")
    print(
        f"pas par sigma            : {np.bincount(sig_idx_u, minlength=len(unique_vals))}"
    )
    print(f"n={n}, T={T}, level={level}")
    neighbors, logT = build_transitions(
        cell_ids, level, unique_vals, truncate=truncate, ellipsoid=ellipsoid
    )
    print("fin neighbors")
    rows = np.arange(n)
    M = np.where(ocean, logE[0], -np.inf)
    back = np.zeros((T, n), dtype=np.int32)

    # forward pass
    for t in range(1, T):
        s = sig_idx_u[t]
        nbr = neighbors[s]
        cand = M[nbr] + logT[s]  # (n, K): score of each predecessor
        k = np.argmax(cand, axis=1)
        best = cand[rows, k]
        back[t] = nbr[rows, k]
        M = np.where(ocean, best + logE[t], -np.inf)

    if not np.isfinite(M).any():
        raise ValueError("no feasible path: all final scores are zero")

    # backtracking
    path = np.empty(T, dtype=np.int64)
    path[-1] = np.argmax(M)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    cell_path = cell_ids[path]
    lon, lat = hgn.healpix_to_lonlat(cell_path, level, ellipsoid=ellipsoid)

    return xr.Dataset(
        {
            "cell_ids": ("time", cell_path),
            "longitude": ("time", np.asarray(lon)),
            "latitude": ("time", np.asarray(lat)),
        },
        coords={"time": emission["time"].values},
    )
