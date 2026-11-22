"""Detect and mask dry cells (``H0 + XE <= min_water_depth``) in model data.

Shallow cells of tidal shelf models such as the Copernicus Marine NWSHELF hourly
products can have a negative water column at low tide: the distributed static
depth (``H0``) is the real bathymetry, while the model itself runs with a minimum
depth, so ``H0 + XE`` (``XE`` being the sea surface height) can drop below zero.
The temperature of those cells stays finite, so the dry cells have to be found
from the water column, not from the temperature.

The checks are meant for the native model grid (before regridding).
"""

import math
import warnings

import numpy as np
import xarray as xr

__all__ = [
    "DryCellWarning",
    "dry_mask",
    "count_dry_cells",
    "mask_dry_cells",
]


class DryCellWarning(UserWarning):
    """Warning about dry cells (``H0 + XE <= min_water_depth``) in the model.

    This is a :class:`UserWarning` (not a :class:`RuntimeWarning`) so that it is
    not hidden by the ``RuntimeWarning`` filter set in :mod:`pangeo_fish.bathy`.
    """


def _validate(min_water_depth):
    if isinstance(min_water_depth, (bool, np.bool_, str)):
        raise ValueError(
            f"min_water_depth must be a finite number >= 0, got {min_water_depth!r}"
        )
    try:
        value = float(min_water_depth)
    except (TypeError, ValueError):
        raise ValueError(
            f"min_water_depth must be a finite number >= 0, got {min_water_depth!r}"
        ) from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"min_water_depth must be a finite number >= 0, got {min_water_depth!r}"
        )
    return value


def _sea(ds):
    return ds["H0"].notnull()


def _spatial_dims(ds):
    return [dim for dim in ds["H0"].dims if dim != "time"]


def dry_mask(ds, min_water_depth=0.0):
    """Boolean mask of the dry (cell, time) pairs.

    A sea cell is dry at time ``t`` if ``XE(t) + H0 <= min_water_depth``.
    Land cells and cells with a missing ``XE`` are never dry (``NaN <= x`` is False).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the static depth ``H0`` and the sea surface height ``XE``.
    min_water_depth : float, default: 0.0
        Water column (m) at or below which a cell is considered dry.

    Returns
    -------
    mask : xarray.DataArray
        True where the cell is dry.
    """
    threshold = _validate(min_water_depth)
    return (ds["XE"] + ds["H0"]) <= threshold


def count_dry_cells(ds, min_water_depth=0.0):
    """Count the dry cells per time step.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``H0`` and ``XE`` (and optionally a ``depth`` coordinate).
    min_water_depth : float, default: 0.0
        Water column (m) at or below which a cell is considered dry.

    Returns
    -------
    counts : xarray.Dataset
        ``n_sea`` (number of sea cells), ``n_dry`` (dry cells per time step) and
        ``min_water_column`` (smallest ``H0 + XE`` over the sea cells per time
        step). If ``ds`` has a ``depth`` coordinate, ``n_top_level_above_surface``
        counts the wet sea cells whose shallowest level is above the
        instantaneous surface (``depth.min() + XE < 0``).
    """
    spatial = _spatial_dims(ds)
    sea = _sea(ds)
    dry = dry_mask(ds, min_water_depth) & sea
    column = (ds["XE"] + ds["H0"]).where(sea)
    counts = xr.Dataset(
        {
            "n_sea": sea.sum(spatial),
            "n_dry": dry.sum(spatial),
            "min_water_column": column.min(spatial),
        }
    )
    if "depth" in ds.coords:
        top = ds["depth"].min() + ds["XE"]
        counts["n_top_level_above_surface"] = (sea & ~dry & (top < 0)).sum(spatial)
    return counts


def mask_dry_cells(ds, min_water_depth=0.0, *, variable="diff"):
    """Set ``variable`` to NaN on the dry (cell, time) pairs.

    Time steps without any wet cell with a valid ``variable`` are left
    unchanged, so the masking never empties a time step. ``H0``, ``XE`` and the other variables are kept
    as they are. The operation is lazy for dask-backed data.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``H0``, ``XE`` and ``variable``.
    min_water_depth : float, default: 0.0
        Water column (m) at or below which a cell is considered dry.
    variable : str, default: "diff"
        Name of the variable to mask.

    Returns
    -------
    masked : xarray.Dataset
    """
    spatial = _spatial_dims(ds)
    dry = dry_mask(ds, min_water_depth)
    has_wet = (_sea(ds) & ds[variable].notnull() & ~dry).any(spatial)
    masked = ds[variable].where(~(dry & has_wet))
    return ds.assign({variable: masked.assign_attrs(ds[variable].attrs)})


def _format_times(times, limit=5):
    values = [np.datetime_as_string(t, unit="m") for t in np.asarray(times)[:limit]]
    suffix = ", ..." if len(times) > limit else ""
    return ", ".join(values) + suffix


def _apply_min_water_depth(diff, min_water_depth="auto"):
    """Mask (``"auto"`` or a number) or report (``None``) the dry cells of a diff.

    ``"auto"`` masks the cells with ``H0 + XE <= 0`` when there are any (in
    practice only the NWSHELF hourly product) and leaves the dataset untouched
    otherwise.
    """
    auto = isinstance(min_water_depth, str) and min_water_depth == "auto"
    if auto:
        min_water_depth = 0.0
    elif min_water_depth is not None:
        min_water_depth = _validate(min_water_depth)
    if "XE" not in diff or "H0" not in diff:
        return diff

    threshold = 0.0 if min_water_depth is None else min_water_depth
    spatial = _spatial_dims(diff)
    sea = _sea(diff)
    # only H0 and XE are read here: the diff itself stays lazy
    stats = xr.Dataset(
        {
            "n_sea": sea.sum(spatial),
            "n_dry": (dry_mask(diff, threshold) & sea).sum(spatial),
        }
    ).compute()
    n_dry_all = stats["n_dry"]
    total = int(n_dry_all.sum())
    if total == 0:
        if min_water_depth is None or auto:
            return diff
        return diff.assign_attrs(min_water_depth=min_water_depth)

    other = [dim for dim in n_dry_all.dims if dim != "time"]
    n_dry = n_dry_all.max(other) if other else n_dry_all
    has_time = "time" in n_dry.dims
    worst = int(n_dry.max())
    n_sea = int(stats["n_sea"].max())

    if min_water_depth is None:
        where = ""
        if has_time:
            where = f" (e.g. {_format_times(n_dry.time[n_dry.argmax('time')].values[None])})"
        warnings.warn(
            f"{total} sea (cell, time) pairs are dry (H0 + XE <= 0): at most {worst}"
            f" of {n_sea} sea cells per time step{where}. Their diff is kept."
            " Pass min_water_depth=0.0 to compute_diff to exclude them.",
            DryCellWarning,
            stacklevel=3,
        )
        return diff

    every_sea_cell_dry = (n_dry_all >= stats["n_sea"]) & (stats["n_sea"] > 0)
    masked = int(n_dry_all.where(~every_sea_cell_dry, 0).sum())
    message = (
        (
            "Dry cells found (automatic masking, pass min_water_depth=None to keep them). "
            if auto
            else ""
        )
        + f"Masking up to {masked} dry sea (cell, time) pairs"
        f" (H0 + XE <= {min_water_depth} m), at most {worst} of {n_sea} sea cells"
        " per time step. Time steps without any wet sea cell with a valid diff"
        " are left unmasked."
    )
    if has_time:
        all_dry = every_sea_cell_dry.any(other) if other else every_sea_cell_dry
        if bool(all_dry.any()):
            times = n_dry.time.values[all_dry.values]
            message += (
                f" Every sea cell is dry at {len(times)} time steps"
                f" ({_format_times(times)})."
            )
    warnings.warn(message, DryCellWarning, stacklevel=3)
    return mask_dry_cells(diff, min_water_depth).assign_attrs(
        min_water_depth=min_water_depth
    )
