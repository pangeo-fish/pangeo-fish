"""Describe a tag before running the geolocation.

:py:func:`describe_tag` collects what can be learnt from the tag records
themselves: record period and sampling, depth and temperature ranges, values
outside the sensor range, duplicate and missing records, suggested start and end
of the track (:py:mod:`pangeo_fish.track_bounds`), pressure offset
(:py:mod:`pangeo_fish.pressure_offset`) and tidal signal
(:py:func:`pangeo_fish.tide.detect_tide`). It compares them with the tagging
events and lists the questions that the data cannot answer.

It only *describes* the tag: nothing is modified.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import xarray as xr

from pangeo_fish.pressure_offset import estimate_pressure_offset
from pangeo_fish.track_bounds import detect_end_of_track, detect_start_of_track

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


def _out_of_range(df, sensor_ranges, tolerance, after):
    summary = {}
    for name, (low, high) in sensor_ranges.items():
        if name not in df:
            continue
        margin = tolerance * (high - low)
        bad = (df[name] < low - margin) | (df[name] > high + margin)
        later = bad[bad.index > after] if after is not None else bad
        summary[name] = {
            "range": [low, high],
            "n_records": int(bad.sum()),
            "fraction": float(bad.mean()) if len(bad) else 0.0,
            "first_after_start": (
                later.index[later.to_numpy()].min() if later.any() else None
            ),
        }
    return summary


def _tide_summary(records, tag_name, start, end, after_end):
    from pangeo_fish.tide import detect_tide

    stop = end + pd.Timedelta(after_end) if end is not None else None
    subset = records[["pressure"]].sel(time=slice(start, stop))
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
        Valid range of each sensor. Defaults to the ``sensor_ranges`` attribute
        (tag metadata).
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
        the full record and on the suggested track); ``out_of_sensor_range``;
        ``tagging_events`` (times and positions); ``suggested_start`` and
        ``suggested_end`` (outputs of :py:mod:`pangeo_fish.track_bounds`);
        ``comparison`` (differences in days between the tagging events and the
        suggestions); ``pressure_offset`` (estimated from the pre-release period);
        ``tide``; ``questions`` (what the data cannot tell, to ask the user).

    Examples
    --------
    >>> tag = pangeo_fish.io.open_tag(tag_root, "A19845")
    >>> description = describe_tag(tag)
    >>> print(format_description(description))
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
        df, sensor_ranges, sensor_range_tolerance, start_time
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
            in_air=pre_release,
            **{"in_water": (start_time, end_time), **(offset_kwargs or {})},
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

    description["questions"] = _questions(description)
    return description


def _questions(description):
    """What the records cannot tell: questions for the data owner."""
    questions = []
    events = description["tagging_events"]
    times, positions = events["times"], events["positions"]
    start = description["suggested_start"]
    end = description["suggested_end"]

    if "release" not in times or times.get("release") is None:
        questions.append("When was the fish released?")
    if positions.get("release") is None:
        questions.append("Where was the fish released (latitude, longitude)?")

    release_diff = description["comparison"].get("release_minus_suggested_start_days")
    if release_diff is not None and abs(release_diff) > 1 / 24:
        questions.append(
            f"The records suggest the tag entered the water at {start['start']}, "
            f"{abs(release_diff) * 24:.1f} h {'before' if release_diff > 0 else 'after'} "
            "the release time: which one is right (recovery tank, time zone)?"
        )

    end_events = [name for name in ("fish_death", "recapture") if times.get(name)]
    if end["end"] is None:
        questions.append(
            "No end of track found in the records: until when was the fish alive "
            "and carrying the tag?"
        )
    elif not end_events:
        questions.append(
            f"The records suggest the track ends at {end['end']} "
            f"({end['likely_cause']}): is this the end of the track?"
        )
    for name in end_events:
        diff = description["comparison"].get(f"{name}_minus_suggested_end_days")
        if diff is not None and abs(diff) > 1 / 24:
            questions.append(
                f"The records suggest the track ends at {end['end']} "
                f"({end['likely_cause']}), {abs(diff):.1f} days "
                f"{'before' if diff > 0 else 'after'} '{name}' ({times[name]}): "
                "which end should be used? Is the recapture position known?"
            )
    if end.get("warning"):
        questions.append(f"End of track: {end['warning']}.")

    offset = description["pressure_offset"]
    if offset.get("offset") is not None and abs(offset["offset"]) > 0.5:
        questions.append(
            f"The pressure reads {offset['offset']:.2f} m in the air before the "
            "release: correct the pressure by this offset?"
        )

    tide = description["tide"]
    if tide.get("available") and (tide.get("fraction_after_end") or 0) > 0.2:
        questions.append(
            "A tidal signal is recorded after the suggested end: the tag may be "
            "lying on the seabed (fish dead or tag lost). Was it found there?"
        )
    return questions


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, (tuple, list)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def format_description(description):
    """Format the output of :py:func:`describe_tag` as Markdown text.

    Parameters
    ----------
    description : dict
        The output of :py:func:`describe_tag`.

    Returns
    -------
    text : str
    """
    records = description["records"]
    start = description["suggested_start"]
    end = description["suggested_end"]
    lines = [f"## Tag {description['tag_name']}", ""]
    lines.append(
        f"- records: {records['first_record']} – {records['last_record']} "
        f"({records['n_records']} records, {records['duration_days']:.1f} days, "
        f"every {records['sampling_interval']}, {records['n_gaps']} gaps, "
        f"{records['n_duplicate_times']} duplicate times)"
    )
    for name, values in description["ranges"]["track"].items():
        if values:
            lines.append(
                f"- {name} on the suggested track: "
                + ", ".join(f"{key} {_format_value(v)}" for key, v in values.items())
            )
    for name, values in description["out_of_sensor_range"].items():
        if values["n_records"]:
            lines.append(
                f"- {name} outside the sensor range {values['range']}: "
                f"{values['n_records']} records, first after the start: "
                f"{values['first_after_start']}"
            )
    for name, time in description["tagging_events"]["times"].items():
        lines.append(f"- tagging event '{name}': {time}")
    lines.append(f"- suggested start: {start['start']} ({start['reason']})")
    lines.append(
        f"- suggested end: {end['end']} ({end['likely_cause'] or end['reason']})"
    )
    for key, value in description["comparison"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value:.2f}")
    offset = description["pressure_offset"]
    if offset.get("offset") is not None:
        lines.append(f"- pressure offset (in the air): {offset['offset']:.2f} m")
    for warning in offset.get("warnings", []):
        lines.append(f"- pressure offset: {warning}")
    tide = description["tide"]
    if tide.get("available"):
        lines.append(
            "- tidal signal: "
            + ", ".join(
                f"{key.replace('_', ' ')} {_format_value(value)}"
                for key, value in tide.items()
                if key != "available"
            )
        )
    if description["questions"]:
        lines += ["", "### Questions", ""]
        lines += [
            f"{i}. {question}" for i, question in enumerate(description["questions"], 1)
        ]
    return "\n".join(lines)
