"""Diagnostics that check the inputs of the pangeo-fish pipeline.

Each diagnostic returns an :py:class:`xarray.Dataset` with its findings and,
with ``warn=True`` (the default), emits a :py:class:`UserWarning` when it finds
a problem.
"""

import warnings

import numpy as np
import xarray as xr

from pangeo_fish.distributions.healpix import _angular_distances
from pangeo_fish.helpers import _earth_radius_from_grid

__all__ = [
    "check_event_position",
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
