"""Diagnostics that check the inputs of the pangeo-fish pipeline.

Each diagnostic returns an :py:class:`xarray.Dataset` with its findings and,
with ``warn=True`` (the default), emits a :py:class:`UserWarning` when it finds
a problem.
"""

import warnings

import numpy as np
import xarray as xr

from pangeo_fish.cf import bounds_to_bins
from pangeo_fish.tags import reshape_by_bins

__all__ = [
    "diagnose_depth_vs_bathymetry",
]


def _tag_max_depth(tag_log: xr.Dataset, time: xr.DataArray) -> xr.DataArray:
    """Maximum tag depth within each model time step (same bins as compute_diff)."""
    if time.size < 2:
        raise ValueError("at least two model time steps are needed")
    bins = (
        xr.Dataset(coords={"time": time})
        .cf.add_bounds(["time"], output_dim="bounds")
        .pipe(bounds_to_bins, bounds_dim="bounds")
        .get("time_bins")
    )
    reshaped = reshape_by_bins(
        tag_log[["pressure"]], dim="time", bins=bins, other_dim="obs"
    )
    return reshaped["pressure"].max("obs")


def diagnose_depth_vs_bathymetry(
    *,
    tag_log: xr.Dataset,
    bathymetry: xr.DataArray,
    time=None,
    mask: xr.DataArray = None,
    reference_bathymetry: xr.DataArray = None,
    relative_depth_threshold: float = 1.0,
    warn: bool = True,
) -> xr.Dataset:
    """Compare the depths reached by the tag with the model bathymetry.

    For each model time step, the maximum tag depth is compared with the
    bathymetry of every valid cell. A cell is "too shallow" when its
    bathymetry is lower than ``relative_depth_threshold`` times the maximum tag
    depth, which is the rule used by :py:func:`pangeo_fish.helpers.compute_diff`
    to drop cells when ``relative_depth_threshold != 0``. If the model
    bathymetry is too shallow almost everywhere while an independent
    bathymetry (e.g. GEBCO, sampled with
    :py:func:`pangeo_fish.bathy.sample_max_depth`) is deep enough, the model
    grid is too coarse for the area: do not use ``relative_depth_threshold``
    and use the GEBCO bathymetry evidence (:py:mod:`pangeo_fish.bathy`) instead.

    Parameters
    ----------
    tag_log : xarray.Dataset
        The DST data, with a ``pressure`` variable (depth in m) along ``time``.
        Hint: ``tag["dst"].ds``.
    bathymetry : xarray.DataArray
        Model depth (m, positive downwards), e.g. ``diff["H0"]`` from
        :py:func:`pangeo_fish.helpers.compute_diff` (2D grid or HEALPix
        ``cells``). It may also depend on ``time``: ``diff["H0"] + diff["XE"]``
        is exactly the bathymetry used by ``compute_diff``. The model time
        steps are then taken from its ``time`` coordinate.
    time : array-like, optional
        Model time steps. Required when ``bathymetry`` has no ``time``
        dimension, and not allowed when it has one.
    mask : xarray.DataArray, optional
        Area to consider (``True`` on cells to use), e.g. the ocean mask or an
        area of interest. The valid cells of each bathymetry are the cells of
        ``mask`` where that bathymetry is defined. Defaults to all cells.
    reference_bathymetry : xarray.DataArray, optional
        Independent bathymetry (m, positive downwards) on the same cells as
        ``bathymetry``, e.g. GEBCO from
        :py:func:`pangeo_fish.bathy.sample_max_depth`.
    relative_depth_threshold : float, default: 1.0
        A cell is too shallow when ``bathymetry < relative_depth_threshold *
        tag_max_depth``. Use ``1`` for the physical comparison, or the value
        passed to ``compute_diff`` to see which cells it removes.
    warn : bool, default: True
        Whether to emit a :py:class:`UserWarning` when, at some time steps, no
        valid cell of the model is deep enough.

    Returns
    -------
    diagnostic : xarray.Dataset
        Along ``time``: ``tag_max_depth``, ``n_valid_cells``,
        ``model_max_depth`` (deepest valid cell), ``model_too_shallow`` (share
        of valid cells that are too shallow, 0 to 1) and, with
        ``reference_bathymetry``, ``reference_n_valid_cells``,
        ``reference_max_depth``, ``reference_too_shallow`` and
        ``model_only_too_shallow`` (share of the cells valid in both
        bathymetries that are too shallow in the model but not in the
        reference: the cells a threshold would remove wrongly).
    """
    if "time" in bathymetry.dims:
        if time is not None:
            raise ValueError(
                "`bathymetry` has a time dimension: do not pass `time`,"
                " the time steps are taken from `bathymetry`"
            )
        time = bathymetry["time"]
    elif time is None:
        raise ValueError("`time` is required when `bathymetry` has no time dimension")
    else:
        time = xr.DataArray(np.asarray(time), dims="time", name="time")
        time = time.assign_coords(time=time)
    max_depth = _tag_max_depth(tag_log, time.reset_coords(drop=True)).compute()
    threshold = relative_depth_threshold * max_depth

    if mask is None:
        mask = True
    else:
        mask = mask.fillna(0).astype(bool)

    def compare(depth, prefix):
        valid = depth.notnull() & mask
        spatial_dims = [d for d in valid.dims if d != "time"]
        too_shallow = (depth < threshold) & valid
        n_valid = valid.sum(spatial_dims)
        variables = {
            f"{prefix}_max_depth": depth.where(valid).max(spatial_dims),
            f"{prefix}_too_shallow": (too_shallow.sum(spatial_dims) / n_valid).where(
                max_depth.notnull()
            ),
        }
        return variables, valid, too_shallow, n_valid, spatial_dims

    variables, model_valid, model_shallow, n_valid, spatial_dims = compare(
        bathymetry, "model"
    )
    variables["n_valid_cells"] = n_valid
    if reference_bathymetry is not None:
        reference_vars, reference_valid, reference_shallow, reference_n, _ = compare(
            reference_bathymetry, "reference"
        )
        variables |= reference_vars
        variables["reference_n_valid_cells"] = reference_n
        # cells removed by the threshold although the reference is deep enough
        both = model_valid & reference_valid
        contradicted = model_shallow & ~reference_shallow & both
        variables["model_only_too_shallow"] = (
            contradicted.sum(spatial_dims) / both.sum(spatial_dims)
        ).where(max_depth.notnull())

    diagnostic = (
        xr.Dataset(
            {"tag_max_depth": max_depth.assign_attrs(units="m")}
            | {name: var.broadcast_like(max_depth) for name, var in variables.items()}
        )
        .reset_coords(drop=True)
        .compute()
        .assign_attrs(relative_depth_threshold=relative_depth_threshold)
    )

    observed = diagnostic["tag_max_depth"].notnull()
    blocked = (diagnostic["model_too_shallow"] == 1) & observed
    n_blocked = int(blocked.sum())
    if warn and n_blocked > 0:
        message = (
            f"At {n_blocked} of {int(observed.sum())} time steps, the tag goes "
            f"deeper than {relative_depth_threshold:g} x the model bathymetry in "
            "every valid cell (deepest tag depth "
            f"{float(diagnostic['tag_max_depth'].where(blocked).max()):.1f} m, "
            "deepest model cell "
            f"{float(diagnostic['model_max_depth'].where(blocked).min()):.1f} m)."
        )
        if "reference_too_shallow" in diagnostic:
            ok = int(((diagnostic["reference_too_shallow"] < 1) & blocked).sum())
            message += (
                f" The reference bathymetry has deep enough cells at {ok} of"
                " these time steps."
            )
        message += (
            " compute_diff with relative_depth_threshold="
            f"{relative_depth_threshold:g} would remove all these cells at these"
            " time steps. If the tag depths are right, the model bathymetry is"
            " too coarse here: consider relative_depth_threshold=0 and the GEBCO"
            " bathymetry evidence of pangeo_fish.bathy."
        )
        warnings.warn(message, UserWarning, stacklevel=2)

    return diagnostic
