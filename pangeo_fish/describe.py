"""Describe a tag before running the geolocation.

:py:func:`describe_tag` collects what can be learnt from the tag records
themselves: record period and sampling, depth and temperature ranges, values
outside the sensor range, duplicate and missing records, suggested start and end
of the track (:py:mod:`pangeo_fish.track_bounds`), pressure offset
(:py:mod:`pangeo_fish.pressure_offset`) and tidal signal
(:py:func:`pangeo_fish.tide.detect_tide`), and compares them with the tagging
events.

It only *describes* the tag: nothing is modified, and it returns facts only.
Deciding what to ask the data owner from these facts is left to the workflow
(notebooks, agent templates).
"""

import contextlib
import io

import numpy as np
import pandas as pd
import xarray as xr

from pangeo_fish.pressure_offset import estimate_pressure_offset
from pangeo_fish.track_bounds import (
    detect_end_of_track,
    detect_start_of_track,
    flag_records,
)

VARIABLES = ("pressure", "temperature")


def _split_tag(tag):
    """Return (records, tagging events, attributes) from a tag DataTree or Dataset."""
    if isinstance(tag, xr.DataTree):
        records = tag["dst"].to_dataset()
        events = (
            tag["tagging_events"].to_dataset()
            if "tagging_events" in tag.children
            else None
        )
        attrs = {**dict(tag.attrs), **records.attrs}
        return records, events, attrs
    return tag, None, dict(tag.attrs)


def _event_times(events):
    if events is None or "time" not in events:
        return {}
    times = events["time"].to_series()
    return {
        str(name): pd.Timestamp(time) if pd.notna(time) else None
        for name, time in times.items()
    }


def _event_positions(events):
    if events is None:
        return {}
    positions = {}
    for name in events["event_name"].to_numpy():
        event = events.sel(event_name=name)
        values = [
            float(event[coord]) if coord in event else float("nan")
            for coord in ("latitude", "longitude")
        ]
        positions[str(name)] = None if np.isnan(values).any() else tuple(values)
    return positions


def _records_summary(df, gap_factor):
    times = df.index
    steps = pd.Series(times[1:] - times[:-1])
    interval = steps.median() if len(steps) else None
    gaps = steps[steps > gap_factor * interval] if interval is not None else steps[:0]
    return {
        "first_record": times[0] if len(times) else None,
        "last_record": times[-1] if len(times) else None,
        "n_records": int(len(df)),
        "duration_days": (
            float((times[-1] - times[0]) / pd.Timedelta("1D")) if len(times) else 0.0
        ),
        "sampling_interval": interval,
        "regular_sampling": bool(len(steps) == 0 or (steps == interval).all()),
        "n_gaps": int(len(gaps)),
        "longest_gap": gaps.max() if len(gaps) else None,
        "n_duplicate_times": int(times.duplicated().sum()),
        "n_duplicate_records": int(df.reset_index().duplicated().sum()),
        "n_missing": {name: int(df[name].isna().sum()) for name in df.columns},
    }


def _ranges(df):
    summary = {}
    for name in df.columns:
        values = df[name].dropna()
        if values.empty:
            summary[name] = None
            continue
        summary[name] = {
            "min": float(values.min()),
            "median": float(values.median()),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }
    return summary


def _out_of_range(records, sensor_ranges, tolerance, after):
    """Summary of the records flagged ``out_of_sensor_range`` by ``flag_records``."""
    if not sensor_ranges:
        return None
    bad = (
        flag_records(
            records, sensor_ranges=sensor_ranges, sensor_range_tolerance=tolerance
        )["out_of_sensor_range"]
        .to_series()
        .astype(bool)
    )
    later = bad[bad.index > after] if after is not None else bad
    return {
        "sensor_ranges": {name: list(bounds) for name, bounds in sensor_ranges.items()},
        "n_records": int(bad.sum()),
        "fraction": float(bad.mean()) if len(bad) else 0.0,
        "first_after_start": (
            later.index[later.to_numpy()].min() if later.any() else None
        ),
    }


def _tide_summary(records, tag_name, start, end, after_end):
    from pangeo_fish.tide import detect_tide

    stop = end + pd.Timedelta(after_end) if end is not None else None
    subset = records[["pressure"]].sel(time=slice(start, stop))
    # detect_tide prints progress (print and tqdm in tide.tide_behav_extr): silence
    # it here until tide.py gets a verbose / progress option (separate follow-up
    # PR), then call detect_tide directly.
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = detect_tide(subset, tag_name)
    except (NotImplementedError, ValueError) as error:
        return {"available": False, "reason": str(error)}

    found = result["tide_found"].to_series().astype(bool)
    track = found[found.index <= end] if end is not None else found
    summary = {
        "available": True,
        "fraction_track": float(track.mean()) if len(track) else None,
        "first_found": track.index[track.to_numpy()].min() if track.any() else None,
    }
    if len(track):
        last_days = track[track.index > track.index[-1] - pd.Timedelta("3D")]
        summary["fraction_last_3_days"] = float(last_days.mean())
    if end is not None:
        after = found[found.index > end]
        summary["fraction_after_end"] = float(after.mean()) if len(after) else None
    return summary


def _days(delta):
    return float(delta / pd.Timedelta("1D"))


def describe_tag(
    tag,
    *,
    tag_name=None,
    sensor_ranges=None,
    sensor_range_tolerance=0.01,
    gap_factor=2,
    tide=True,
    tide_after_end="3D",
    start_kwargs=None,
    end_kwargs=None,
    offset_kwargs=None,
):
    """Summarize what the tag records tell before running the geolocation.

    Parameters
    ----------
    tag : xarray.DataTree or xarray.Dataset
        The tag, as returned by :py:func:`pangeo_fish.io.open_tag` (records in
        ``dst``, ``tagging_events``, metadata in the attributes), or the records
        only (no comparison with the tagging events).
    tag_name : str, optional
        Name of the tag. Defaults to the ``tag_name`` / ``pit_tag_id`` attribute.
    sensor_ranges : mapping of str to (float, float), optional
        Valid range of each sensor. Defaults to a ``sensor_ranges`` attribute of
        the tag, if present (no check otherwise).
    sensor_range_tolerance : float, default: 0.01
        Tolerance as a fraction of the sensor range.
    gap_factor : float, default: 2
        Steps longer than ``gap_factor`` times the sampling interval are gaps.
    tide : bool, default: True
        Run :py:func:`pangeo_fish.tide.detect_tide` on the suggested track (and
        ``tide_after_end`` after its end). It takes a few seconds per month of
        records.
    tide_after_end : str, default: "3D"
        Also look for the tide this long after the suggested end (a dead fish or
        a lost tag on the seabed records the tide).
    start_kwargs, end_kwargs : dict, optional
        Parameters of :py:func:`pangeo_fish.track_bounds.detect_start_of_track` and
        :py:func:`pangeo_fish.track_bounds.detect_end_of_track`.
    offset_kwargs : dict, optional
        Parameters of :py:func:`pangeo_fish.pressure_offset.estimate_pressure_offset`.

    Returns
    -------
    description : dict
        ``tag_name``; ``records`` (period, sampling interval, gaps, duplicates,
        missing values); ``ranges`` (min / median / p99 / max of each variable on
        the full record and on the suggested track); ``out_of_sensor_range``
        (records flagged by :py:func:`pangeo_fish.track_bounds.flag_records`:
        sensor ranges used, number, fraction and first one after the suggested
        start, or ``None`` without sensor ranges);
        ``tagging_events`` (times and positions); ``suggested_start`` and
        ``suggested_end`` (outputs of :py:mod:`pangeo_fish.track_bounds`);
        ``comparison`` (differences in days between the tagging events and the
        suggestions); ``pressure_offset`` (estimated from the pre-release period,
        checked on the suggested track); ``tide``.

    Examples
    --------
    >>> tag = pangeo_fish.io.open_tag(tag_root, "A19845")
    >>> description = describe_tag(tag)
    >>> description["suggested_end"]["likely_cause"]
    """
    records, events, attrs = _split_tag(tag)
    if tag_name is None:
        tag_name = attrs.get("tag_name") or attrs.get("pit_tag_id") or "tag"
    if sensor_ranges is None:
        sensor_ranges = attrs.get("sensor_ranges", {}) or {}

    variables = [name for name in VARIABLES if name in records]
    df = records[variables].to_dataframe().sort_index()
    records = records.sortby("time")
    records = records.isel(time=~records.indexes["time"].duplicated())

    description = {"tag_name": str(tag_name)}
    description["records"] = _records_summary(df, gap_factor)

    flag_kwargs = {
        "sensor_ranges": sensor_ranges,
        "sensor_range_tolerance": sensor_range_tolerance,
    }
    start = detect_start_of_track(records, **flag_kwargs, **(start_kwargs or {}))
    start_time = start["start"]
    end = detect_end_of_track(
        records, start=start_time, **flag_kwargs, **(end_kwargs or {})
    )
    end_time = end["end"]
    description["suggested_start"] = start
    description["suggested_end"] = end

    track = df.loc[start_time:end_time] if start_time is not None else df
    description["ranges"] = {"record": _ranges(df), "track": _ranges(track)}
    description["out_of_sensor_range"] = _out_of_range(
        records, sensor_ranges, sensor_range_tolerance, start_time
    )

    event_times = _event_times(events)
    description["tagging_events"] = {
        "times": event_times,
        "positions": _event_positions(events),
    }
    comparison = {}
    release = event_times.get("release")
    if release is not None and start_time is not None:
        comparison["release_minus_suggested_start_days"] = _days(release - start_time)
    for name in ("fish_death", "recapture"):
        time = event_times.get(name)
        if time is not None and end_time is not None:
            comparison[f"{name}_minus_suggested_end_days"] = _days(time - end_time)
    description["comparison"] = comparison

    pre_release = start.get("pre_release_period")
    if pre_release is not None:
        offset = estimate_pressure_offset(
            records,
            in_air=slice(*pre_release),
            **{"in_water": slice(start_time, end_time), **(offset_kwargs or {})},
        )
    else:
        offset = {
            "offset": None,
            "warnings": ["no pre-release period in the records"],
        }
    description["pressure_offset"] = offset

    if tide and start_time is not None:
        description["tide"] = _tide_summary(
            records, tag_name, start_time, end_time, tide_after_end
        )
    else:
        description["tide"] = {"available": False, "reason": "not computed"}

    return description
