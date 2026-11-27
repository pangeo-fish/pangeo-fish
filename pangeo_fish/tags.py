"""Miscellaneous operations around biologging data."""

import numpy as np
import pandas as pd
import xarray as xr


def to_time_slice(times):
    subset = times.where(times.notnull(), drop=True)

    min_ = subset.isel(event_name=0)
    max_ = subset.isel(event_name=1)

    return slice(min_.data, max_.data)


#: Names of the events that can end a track, by decreasing priority.
END_EVENT_NAMES = ("fish_death", "recapture")


def select_end_event(events):
    """Select the event that ends the track.

    The end event is the ``fish_death`` row. Without a ``fish_death`` row,
    the ``recapture`` row is used. When both are present, ``fish_death`` is
    used and ``recapture`` is ignored, whatever their positions are.

    Parameters
    ----------
    events : xarray.Dataset or xarray.DataArray
        The tagging events, indexed by ``event_name``.

    Returns
    -------
    event : xarray.Dataset or xarray.DataArray
        The selected event, without the ``event_name`` dimension.

    Raises
    ------
    ValueError
        If the events contain neither ``fish_death`` nor ``recapture``.
    """
    names = [str(name) for name in np.atleast_1d(events["event_name"].to_numpy())]
    for name in END_EVENT_NAMES:
        if name in names:
            return events.sel(event_name=name)

    expected = " or ".join(repr(name) for name in END_EVENT_NAMES)
    raise ValueError(
        f"The tagging events have no end event: expected {expected} in"
        f" `event_name`, got {names}. If the end position is unknown, add a"
        " 'fish_death' row with latitude and longitude set to NA."
    )


def adapt_model_time(slice_):
    start = np.datetime64(slice_.start)
    stop = np.datetime64(slice_.stop)

    # only if [minute, sec] part of `slice_.start` < 30:00
    if pd.Timestamp(start).minute < 30:
        model_start = start - np.timedelta64(30, "m")
    else:
        model_start = start

    # only if [minute, sec] part of `slice_.stop` > 30:00
    if pd.Timestamp(stop).minute > 30:
        model_stop = stop + np.timedelta64(30, "m")
    else:
        model_stop = stop

    return slice(model_start, model_stop)


def reshape_by_bins(ds, *, dim, bins, other_dim="obs"):
    def expand_group(group, *, dim, other_dim):
        return (
            group.rename_dims({dim: other_dim})
            .drop_indexes(dim)
            .assign_coords({other_dim: lambda ds: ds[other_dim]})
        )

    index = bins.to_index()
    grouper = xr.groupers.BinGrouper(index, include_lowest=True)

    if np.issubdtype(ds[dim].dtype, np.datetime64) and np.issubdtype(
        index.dtype.subtype, np.datetime64
    ):
        if ds[dim].dtype != index.dtype.subtype:
            ds = ds.assign_coords({dim: ds[dim].astype(index.dtype.subtype)})

    processed = ds.groupby({dim: grouper}).map(
        expand_group, dim=dim, other_dim=other_dim
    )

    return (
        processed.drop_vars("time")
        .rename_dims({"time_bins": "time"})
        .rename_vars({"time_bins": "time"})
        .assign_coords({dim: bins[dim]})
        .drop_vars("time_bins", errors="ignore")
    )
