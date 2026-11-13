"""Estimate and correct the offset of the pressure sensor.

In the air, a pressure sensor should read a depth of 0 m. Many archival tags read
a few tens of centimetres to a few metres instead (e.g. 1.8 m before the release),
and this offset matters for coastal fish and for depth-dependent comparisons
(bathymetry, tide, :py:func:`pangeo_fish.helpers.compute_diff`).

:py:func:`estimate_pressure_offset` estimates the offset from periods where the tag
is known to be in the air (before the release, after the recovery), and
:py:func:`correct_pressure_offset` subtracts it, recording the correction in the
attributes. The in-air periods can be found with
``pangeo_fish.track_bounds.detect_start_of_track`` (``pre_release_period``, pass
it as ``slice(*pre_release_period)``), from the tagging events, or by looking at the
records.
"""

import numpy as np
import pandas as pd
import xarray as xr

COMBINE_METHODS = ("median", "first", "last", "linear")


def _as_dataset(tag_log):
    if isinstance(tag_log, xr.DataTree):
        return tag_log["dst"].to_dataset()
    return tag_log


def _segments(periods, time, *, name):
    """Normalize ``periods`` to a list of (first, last) timestamps sorted by start.

    ``periods`` is a slice, a list of slices or a boolean DataArray along ``time``.
    """
    if isinstance(periods, xr.DataArray):
        if periods.dtype != bool or periods.dims != ("time",):
            raise TypeError(
                f"{name} as a DataArray must be a boolean mask along 'time', "
                f"got dtype {periods.dtype} and dims {periods.dims}"
            )
        mask = periods.reindex(time=time, fill_value=False).to_series().astype(bool)
        values = mask.to_numpy().astype(int)
        edges = np.diff(np.concatenate([[0], values, [0]]))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1) - 1
        return [(mask.index[a], mask.index[b]) for a, b in zip(starts, stops)]

    if isinstance(periods, slice):
        periods = [periods]
    if not isinstance(periods, (list, tuple)) or not all(
        isinstance(period, slice) for period in periods
    ):
        raise TypeError(
            f"{name} must be a slice, a list of slices or a boolean DataArray along "
            f"'time', got {periods!r} (write slice(start, stop) for one period)"
        )

    segments = []
    for period in periods:
        if period.step is not None:
            raise ValueError(f"{name}: slices with a step are not supported")
        first = pd.Timestamp(period.start if period.start is not None else time[0])
        last = pd.Timestamp(period.stop if period.stop is not None else time[-1])
        segments.append((first, last))
    return sorted(segments)


def estimate_pressure_offset(
    tag_log,
    in_air,
    *,
    variable="pressure",
    combine="median",
    max_std=0.25,
    min_records=10,
    plausible_range=(-10.0, 10.0),
    negative_tolerance=0.5,
    in_water=None,
):
    """Estimate the pressure offset from in-air periods.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The tag records (the *full* records, including the in-air periods). A tag
        DataTree uses its ``dst`` node.
    in_air : slice, list of slice or xarray.DataArray
        Periods where the tag is in the air: a slice ``slice(start, stop)`` (bounds
        included, ``None`` for the first / last record), a list of slices, or a
        boolean DataArray along ``time``. The periods are sorted by start time.
    variable : str, default: "pressure"
        The pressure variable (m).
    combine : {"median", "first", "last", "linear"}, default: "median"
        How to combine the offsets of several periods: median of the period
        offsets, offset of the first or of the last used period in time, or a
        linear interpolation in time between the periods (sensor drift; constant
        before the first and after the last period).
    max_std : float, default: 0.25
        Periods whose pressure standard deviation (m) is larger are not used (the
        tag was probably handled, heated by the sun or in water).
    min_records : int, default: 10
        Periods with fewer records are not used.
    plausible_range : (float, float), default: (-10, 10)
        Periods whose median pressure (m) is outside this range are not used
        (e.g. a sensor failure reading a constant 245 m).
    negative_tolerance : float, default: 0.5
        Records outside the in-air periods shallower than ``-negative_tolerance``
        after the correction are counted in ``fraction_negative_after_correction``.
    in_water : slice or xarray.DataArray, optional
        A single period (a slice, a one-element list of slices or a boolean mask
        with one contiguous period): only check these records for negative depths
        (e.g. the track, from the release to the end of the track). Defaults to all
        the records outside the in-air periods.

    Returns
    -------
    estimate : dict
        ``offset``: the offset (m) to subtract (``float``, or a DataArray along
        ``time`` for ``combine="linear"``, ``None`` if no valid period);
        ``segments``: one dict per period (``start``, ``stop``, ``n_records``,
        ``median``, ``std``, ``used``); ``spread``: range (m) of the offsets of the
        used periods; ``fraction_negative_after_correction``: fraction of the
        in-water records (see ``in_water``) shallower than ``-negative_tolerance``
        after the correction; ``warnings``: list of str; ``parameters``.

    Raises
    ------
    TypeError
        If ``in_air`` or ``in_water`` is not one of the accepted forms.
    ValueError
        If ``combine`` is unknown or ``in_water`` is not a single period.

    Examples
    --------
    >>> estimate = estimate_pressure_offset(
    ...     tag, in_air=slice("2023-05-03T08:00", "2023-05-03T11:06")
    ... )
    >>> estimate["offset"]
    """
    if combine not in COMBINE_METHODS:
        raise ValueError(f"combine must be one of {COMBINE_METHODS}, got {combine!r}")

    ds = _as_dataset(tag_log)
    pressure = ds[variable].to_series().dropna().sort_index()
    time = ds["time"].to_index()
    segments = _segments(in_air, time, name="in_air")
    in_water_segments = None
    if in_water is not None:
        in_water_segments = _segments(in_water, time, name="in_water")
        if len(in_water_segments) != 1:
            raise ValueError(
                "in_water must be a single period, got "
                f"{len(in_water_segments)} periods"
            )

    result = {
        "offset": None,
        "segments": [],
        "spread": None,
        "fraction_negative_after_correction": None,
        "warnings": [],
        "parameters": {
            "variable": variable,
            "combine": combine,
            "max_std": max_std,
            "min_records": min_records,
            "plausible_range": list(plausible_range),
            "negative_tolerance": negative_tolerance,
        },
    }

    in_air_mask = pd.Series(False, index=pressure.index)
    for first, last in segments:
        values = pressure[first:last]
        in_air_mask[first:last] = True
        std = float(values.std()) if len(values) > 1 else float("nan")
        median = float(values.median()) if len(values) else float("nan")
        used = (
            len(values) >= min_records
            and std <= max_std
            and plausible_range[0] <= median <= plausible_range[1]
        )
        result["segments"].append(
            {
                "start": first,
                "stop": last,
                "n_records": int(len(values)),
                "median": median,
                "std": std,
                "used": bool(used),
            }
        )
        if not used:
            result["warnings"].append(
                f"period {first} – {last} not used: {len(values)} records, "
                f"median {median:.2f} m, std {std:.2f} m (max_std={max_std}, "
                f"min_records={min_records}, plausible_range={plausible_range})"
            )

    used = [segment for segment in result["segments"] if segment["used"]]
    if not used:
        result["warnings"].append("no valid in-air period: offset not estimated")
        return result

    medians = np.array([segment["median"] for segment in used])
    result["spread"] = float(medians.max() - medians.min())
    if combine == "median":
        offset = float(np.median(medians))
    elif combine == "first":
        offset = float(medians[0])
    elif combine == "last":
        offset = float(medians[-1])
    else:
        centers = pd.DatetimeIndex(
            [
                segment["start"] + (segment["stop"] - segment["start"]) / 2
                for segment in used
            ]
        )
        order = np.argsort(centers)
        values = np.interp(
            time.as_unit("ns").asi8,
            centers.as_unit("ns").asi8[order],
            medians[order],
        )
        offset = xr.DataArray(
            values,
            dims="time",
            coords={"time": ds["time"]},
            name="pressure_offset",
            attrs={"units": "m"},
        )

    if len(used) > 1 and result["spread"] > max_std and combine != "linear":
        result["warnings"].append(
            f"the in-air periods disagree by {result['spread']:.2f} m (sensor drift?): "
            "consider combine='linear'"
        )

    in_water_records = pressure[~in_air_mask]
    if in_water_segments is not None:
        [(first, last)] = in_water_segments
        in_water_records = in_water_records[first:last]
    if len(in_water_records):
        if isinstance(offset, xr.DataArray):
            corrected = in_water_records - offset.to_series().reindex(
                in_water_records.index
            )
        else:
            corrected = in_water_records - offset
        negative = float((corrected < -negative_tolerance).mean())
        result["fraction_negative_after_correction"] = negative
        if negative > 0.01:
            result["warnings"].append(
                f"{negative:.1%} of the in-water records would be shallower than "
                f"{-negative_tolerance} m after the correction"
            )

    result["offset"] = offset
    return result


def correct_pressure_offset(tag_log, offset, *, variable="pressure"):
    """Subtract a pressure offset and record it in the attributes.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The tag records. For a tag DataTree, the pressure of its ``dst`` node is
        corrected.
    offset : float, xarray.DataArray or dict
        The offset (m) to subtract, or the dict returned by
        :py:func:`estimate_pressure_offset`. A DataArray is interpolated on the
        times of ``tag_log``.
    variable : str, default: "pressure"
        The pressure variable.

    Returns
    -------
    corrected : xarray.Dataset or xarray.DataTree
        A copy of ``tag_log`` (same type) with the corrected pressure. The
        ``pressure_offset`` attribute of the variable holds the subtracted offset
        (mean offset for a time-varying one) and ``pressure_offset_method``
        describes it.

    Raises
    ------
    ValueError
        If the pressure was already corrected or if no offset is given.
    """
    if isinstance(offset, dict):
        offset = offset.get("offset")
    if offset is None:
        raise ValueError("no offset to correct (the estimation failed?)")

    ds = _as_dataset(tag_log)
    attrs = dict(ds[variable].attrs)
    if "pressure_offset" in attrs:
        raise ValueError(
            f"{variable!r} was already corrected by {attrs['pressure_offset']} m"
        )

    if isinstance(offset, xr.DataArray):
        offset_ = offset.interp(time=ds["time"]).drop_vars(
            [name for name in offset.coords if name != "time"]
        )
        attrs["pressure_offset"] = float(offset_.mean())
        attrs["pressure_offset_method"] = (
            "time-varying offset interpolated between in-air periods"
        )
    else:
        offset_ = float(offset)
        attrs["pressure_offset"] = offset_
        attrs["pressure_offset_method"] = "constant offset from in-air periods"

    corrected = ds.assign({variable: (ds[variable] - offset_).assign_attrs(attrs)})
    if isinstance(tag_log, xr.DataTree):
        tree = tag_log.copy()
        tree["dst"] = corrected
        return tree
    return corrected
