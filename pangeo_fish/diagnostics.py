"""Diagnostics that check the inputs of the pangeo-fish pipeline.

Each diagnostic returns an :py:class:`xarray.Dataset` with its findings and,
with ``warn=True`` (the default), emits a :py:class:`UserWarning` when it finds
a problem.
"""

import warnings

import numpy as np
import xarray as xr

from pangeo_fish.cf import bounds_to_bins
from pangeo_fish.distributions.healpix import _angular_distances
from pangeo_fish.helpers import _earth_radius_from_grid
from pangeo_fish.tags import reshape_by_bins

__all__ = [
    "check_event_position",
    "diagnose_depth_vs_bathymetry",
]


def _ocean_share(log_weights: np.ndarray, is_ocean: np.ndarray) -> float:
    """Share of ``exp(log_weights)`` located on ``is_ocean`` cells (underflow-safe)."""
    if not is_ocean.any():
        return 0.0
    total = np.logaddexp.reduce(log_weights)
    ocean = np.logaddexp.reduce(log_weights[is_ocean])
    return float(np.exp(ocean - total))


def check_event_position(
    *,
    grid: xr.Dataset,
    position,
    sigma: float,
    mask: xr.DataArray,
    event_name: str = "release",
    min_ocean_mass: float = 0.01,
    target_ocean_mass: float = 0.1,
    warn: bool = True,
) -> xr.Dataset:
    """Check that a tagging event leaves probability mass on the ocean cells.

    The initial (release) and final (recapture) distributions are Gaussians
    centred on the event position (see
    :py:func:`pangeo_fish.distributions.healpix.normal_at`). If the position is
    on land or outside the model domain and ``sigma`` is small, the whole
    distribution falls on cells that the ocean mask removes, and the filter
    starts (or ends) from an empty or degenerate distribution without any error.

    Parameters
    ----------
    grid : xarray.Dataset
        HEALPix grid (decoded with ``xdggs``), e.g. the ``latitude`` and
        ``longitude`` of the regridded diff dataset.
    position : mapping
        Event position with ``latitude`` and ``longitude`` (degrees), e.g.
        ``tag["tagging_events"].ds.sel(event_name="release")``.
    sigma : float
        Standard deviation of the Gaussian, in radians (``initial_std`` or
        ``recapture_std``).
    mask : xarray.DataArray
        Ocean mask along ``cells`` (``True`` / ``1`` on ocean cells), e.g. the
        ``ocean_mask`` of the diff dataset.
    event_name : str, default: "release"
        Name of the event, used in the warning only.
    min_ocean_mass : float, default: 0.01
        The check fails when the share of the Gaussian mass on ocean cells is
        below this value.
    target_ocean_mass : float, default: 0.1
        Ocean share used to compute ``suggested_sigma``.
    warn : bool, default: True
        Whether to emit a :py:class:`UserWarning` when the check fails. The
        warning also gives the distances in km.

    Returns
    -------
    report : xarray.Dataset
        Scalar variables, angles in radians: ``ocean_mass`` (share of the
        Gaussian on ocean cells, 0 to 1), ``distance_to_ocean`` (from the cell
        containing the event to the nearest ocean cell),
        ``nearest_ocean_latitude`` / ``nearest_ocean_longitude`` (a possible
        pseudo-event position, in degrees), ``suggested_sigma`` (smallest
        sigma whose ocean share reaches ``target_ocean_mass``) and ``ok``.
        Values that cannot be computed (no ocean cell, no sigma reaching the
        target) are ``NaN``.
    """
    distances = _angular_distances(grid, position).data
    is_ocean = np.asarray(mask.data, dtype=float)
    is_ocean = np.where(np.isnan(is_ocean), 0.0, is_ocean) > 0
    if is_ocean.shape != distances.shape:
        raise ValueError(
            f"mask has shape {is_ocean.shape}, expected {distances.shape} (one value per cell)"
        )

    def ocean_share(s):
        return _ocean_share(-0.5 * (distances / s) ** 2, is_ocean)

    ocean_mass = ocean_share(sigma)
    distance_to_ocean = np.nan
    nearest_lat = np.nan
    nearest_lon = np.nan
    suggested_sigma = np.nan

    if is_ocean.any():
        nearest = int(np.flatnonzero(is_ocean)[np.argmin(distances[is_ocean])])
        distance_to_ocean = float(distances[nearest])
        if "latitude" in grid.variables and "longitude" in grid.variables:
            nearest_lat = float(np.asarray(grid["latitude"].data).reshape(-1)[nearest])
            nearest_lon = float(np.asarray(grid["longitude"].data).reshape(-1)[nearest])

        if ocean_mass >= target_ocean_mass:
            suggested_sigma = float(sigma)
        else:
            positive = distances[distances > 0]
            lower = float(positive.min()) / 4 if positive.size else float(sigma)
            upper = max(float(distances.max()), lower)
            candidates = np.geomspace(lower, upper, 256)
            shares = np.array([ocean_share(s) for s in candidates])
            reached = np.flatnonzero(shares >= target_ocean_mass)
            if reached.size > 0:
                suggested_sigma = float(candidates[reached[0]])

    ok = ocean_mass >= min_ocean_mass
    report = xr.Dataset(
        {
            "ocean_mass": ((), ocean_mass),
            "distance_to_ocean": ((), distance_to_ocean, {"units": "rad"}),
            "nearest_ocean_latitude": ((), nearest_lat, {"units": "degrees_north"}),
            "nearest_ocean_longitude": ((), nearest_lon, {"units": "degrees_east"}),
            "suggested_sigma": ((), suggested_sigma, {"units": "rad"}),
            "ok": ((), ok),
        }
    )

    if warn and not ok:
        # km only in the message; the radius comes from the grid's ellipsoid
        radius_km = float(_earth_radius_from_grid(grid).magnitude)
        latitude = float(np.asarray(position["latitude"]))
        longitude = float(np.asarray(position["longitude"]))
        position_text = (
            f"({latitude:.4f} N, {longitude:.4f} E)"
            if np.isfinite(latitude)
            else "(unknown position)"
        )
        message = (
            f"Only {ocean_mass:.3g} of the {event_name} probability "
            f"(sigma = {sigma:.3g} rad ≈ {sigma * radius_km:.3g} km) lies on ocean "
            f"cells: the {event_name} position {position_text} is on land or "
            "outside the ocean mask of the model."
        )
        if np.isnan(distance_to_ocean):
            message += " The ocean mask has no ocean cell."
        else:
            nearest_text = (
                f" ({nearest_lat:.4f} N, {nearest_lon:.4f} E)"
                if np.isfinite(nearest_lat)
                else ""
            )
            message += (
                f" The nearest ocean cell{nearest_text} is "
                f"{distance_to_ocean * radius_km:.3g} km away."
                " Consider using that cell as a pseudo-event position"
            )
            if np.isfinite(suggested_sigma):
                message += (
                    f", or a sigma of at least {suggested_sigma:.3g} rad "
                    f"(≈ {suggested_sigma * radius_km:.3g} km), which puts "
                    f"{target_ocean_mass:.0%} of the mass on ocean cells"
                )
            message += "."
        warnings.warn(message, UserWarning, stacklevel=2)

    return report


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
    depth. This is a report only: :py:func:`pangeo_fish.helpers.compute_diff`
    does not remove cells because of the tag depth. Where the tag is deeper
    than the model bathymetry, it compares the tag temperature with the deepest
    model level, and the contradiction is not used. If the model bathymetry is
    too shallow almost everywhere while an independent bathymetry (e.g. GEBCO,
    sampled with :py:func:`pangeo_fish.bathy.sample_max_depth`) is deep enough,
    the model grid is too coarse for the area, and the GEBCO bathymetry
    evidence (:py:mod:`pangeo_fish.bathy`) is the way to use the tag depth.

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
        tag_max_depth``. Use ``1`` for the physical comparison, or a smaller
        value to count only the cells that are clearly too shallow. It only
        sets what this diagnostic reports: it does not change
        ``compute_diff``.
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
        reference: the cells wrongly counted as too shallow).
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
        # cells too shallow in the model although the reference is deep enough
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
            " compute_diff does not remove these cells: it compares the tag"
            " with the deepest model level there. If the tag depths are right,"
            " the model bathymetry is too coarse here: consider the GEBCO"
            " bathymetry evidence of pangeo_fish.bathy."
        )
        warnings.warn(message, UserWarning, stacklevel=2)

    return diagnostic
