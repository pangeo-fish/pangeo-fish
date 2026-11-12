"""Suggest the start and the end of a tag track from the records themselves.

Archival tags often record before the release (tag in the air, on the boat) and
after the end of the track (tag stranded on a beach, fish dead on the seabed, tag
picked up and stored, sensor failure). The functions of this module look at the
pressure and temperature records only and *suggest* the first and the last record
of the track. They never modify the tagging events: compare the suggestion with
the tagging events and decide.

The main criterion is the vertical activity of the tag: a swimming fish changes
depth all the time, while a tag in the air, stranded or lying on the seabed only
shows the sensor noise (and the slow tide for a tag on the seabed). The activity
is the mean absolute deviation of the pressure from its running median (see
:py:func:`pressure_activity`). Values outside the sensor range and temperatures
outside the plausible water range are also used.
"""

import numpy as np
import pandas as pd
import xarray as xr

DEFAULTS = {
    "window": "1h",
    "activity_threshold": 0.1,
    "max_gap": "30min",
    "sensor_range_tolerance": 0.01,
    "temperature_range": (-2.0, 30.0),
}


def _as_dataset(tag_log):
    """Return the records as a Dataset (accepts a tag DataTree or a Dataset)."""
    if isinstance(tag_log, xr.DataTree):
        attrs = dict(tag_log.attrs)
        ds = tag_log["dst"].to_dataset()
        return ds.assign_attrs({**attrs, **ds.attrs})
    return tag_log


def _to_frame(tag_log):
    ds = _as_dataset(tag_log)
    if "pressure" not in ds:
        raise ValueError("the tag records need a 'pressure' variable")
    variables = [name for name in ("pressure", "temperature") if name in ds]
    df = ds[variables].to_dataframe()
    df = df[df.index.notna()].sort_index()
    return df, ds.attrs


def _pressure_activity(pressure, window):
    median = pressure.rolling(window, center=True, min_periods=1).median()
    deviation = (pressure - median).abs()
    return deviation.rolling(window, center=True, min_periods=1).mean()


def pressure_activity(tag_log, window="1h"):
    """Vertical activity of the tag.

    The activity is the running mean (over ``window``) of the absolute deviation of
    the pressure from its running median (over ``window``). It is close to the
    sensor noise (a few cm) for a tag in the air, stranded or lying on the seabed
    (the tide is removed by the running median), and of the order of 1 m for a
    swimming fish.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The tag records, with a ``time`` dimension and a ``pressure`` variable (m).
        A tag DataTree (as returned by :py:func:`pangeo_fish.io.open_tag`) uses its
        ``dst`` node.
    window : str, default: "1h"
        Width of the running windows (pandas offset alias).

    Returns
    -------
    activity : xarray.DataArray
        The activity (m) along ``time``.
    """
    df, _ = _to_frame(tag_log)
    activity = _pressure_activity(df["pressure"], window)
    return xr.DataArray(
        activity.to_numpy(),
        dims="time",
        coords={"time": df.index.to_numpy()},
        name="activity",
        attrs={"units": "m", "long_name": "vertical activity", "window": window},
    )


def _flags(
    df,
    attrs,
    *,
    window,
    activity_threshold,
    sensor_ranges,
    sensor_range_tolerance,
    temperature_range,
):
    flags = pd.DataFrame(index=df.index)
    flags["activity"] = _pressure_activity(df["pressure"], window)
    flags["inactive"] = flags["activity"] < activity_threshold

    if sensor_ranges is None:
        sensor_ranges = attrs.get("sensor_ranges", {}) or {}
    out_of_range = pd.Series(False, index=df.index)
    for name, (low, high) in sensor_ranges.items():
        if name not in df:
            continue
        tolerance = sensor_range_tolerance * (high - low)
        out_of_range |= (df[name] < low - tolerance) | (df[name] > high + tolerance)
    flags["out_of_sensor_range"] = out_of_range

    if temperature_range is not None and "temperature" in df:
        low, high = temperature_range
        flags["temperature_outside_water_range"] = (df["temperature"] < low) | (
            df["temperature"] > high
        )
    else:
        flags["temperature_outside_water_range"] = False

    flags["suspect"] = (
        flags["inactive"]
        | flags["out_of_sensor_range"]
        | flags["temperature_outside_water_range"]
    )
    return flags


def flag_records(
    tag_log,
    *,
    window=DEFAULTS["window"],
    activity_threshold=DEFAULTS["activity_threshold"],
    sensor_ranges=None,
    sensor_range_tolerance=DEFAULTS["sensor_range_tolerance"],
    temperature_range=DEFAULTS["temperature_range"],
):
    """Flag the records that do not look like a swimming fish.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The tag records (``pressure`` in m, optionally ``temperature`` in °C).
    window : str, default: "1h"
        Window of :py:func:`pressure_activity`.
    activity_threshold : float, default: 0.1
        Records with an activity below this value (m) are flagged ``inactive``.
        The default suits 1-min records with a pressure resolution of a few cm.
    sensor_ranges : mapping of str to (float, float), optional
        Valid range of each sensor, e.g. ``{"pressure": (0, 300)}``. Defaults to the
        ``sensor_ranges`` attribute of the tag (tag metadata), if any.
    sensor_range_tolerance : float, default: 0.01
        Tolerance, as a fraction of the sensor range, before flagging a record as
        ``out_of_sensor_range``.
    temperature_range : (float, float) or None, default: (-2, 30)
        Plausible water temperature (°C). Records outside are flagged
        ``temperature_outside_water_range``. ``None`` disables this check.

    Returns
    -------
    flags : xarray.Dataset
        ``activity`` and the boolean flags ``inactive``, ``out_of_sensor_range``,
        ``temperature_outside_water_range`` and ``suspect`` (any of them).
    """
    df, attrs = _to_frame(tag_log)
    flags = _flags(
        df,
        attrs,
        window=window,
        activity_threshold=activity_threshold,
        sensor_ranges=sensor_ranges,
        sensor_range_tolerance=sensor_range_tolerance,
        temperature_range=temperature_range,
    )
    flags.index.name = "time"
    return flags.to_xarray()


def _runs(mask, max_gap):
    """Contiguous runs of True values as (first, last) timestamps.

    Runs separated by less than ``max_gap`` are merged.
    """
    values = mask.to_numpy().astype(int)
    index = mask.index
    edges = np.diff(np.concatenate([[0], values, [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1

    max_gap = pd.Timedelta(max_gap)
    runs = []
    for start, stop in zip(index[starts], index[stops]):
        if runs and start - runs[-1][1] <= max_gap:
            runs[-1] = (runs[-1][0], stop)
        else:
            runs.append((start, stop))
    return runs


def _timestamp(value):
    if value is None:
        return None
    return pd.Timestamp(np.asarray(value).ravel()[0])


def _daily_range(series):
    if series is None or series.empty:
        return None
    daily = series.resample("1D").agg(["min", "max", "count"])
    daily = daily[daily["count"] > 0]
    if daily.empty:
        return None
    return float((daily["max"] - daily["min"]).median())


def _parameters(**kwargs):
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in kwargs.items()
    }


def detect_start_of_track(
    tag_log,
    *,
    window=DEFAULTS["window"],
    activity_threshold=DEFAULTS["activity_threshold"],
    min_active_duration="2h",
    max_gap=DEFAULTS["max_gap"],
    level_change_threshold=0.5,
    sensor_ranges=None,
    sensor_range_tolerance=DEFAULTS["sensor_range_tolerance"],
    temperature_range=DEFAULTS["temperature_range"],
):
    """Suggest the first in-water record (end of the pre-release period).

    Tags are often switched on hours before the release and record in the air.
    The suggested start is the beginning of the first period of sustained vertical
    activity, refined to the first record (from one ``window`` before that period)
    deeper than the pre-release level by more than ``level_change_threshold`` and
    with a plausible water temperature (in the sun, the pressure reading of a tag in
    the air can drift by about 1 m).

    The suggestion is the first record *in water*: if the fish was kept in a
    recovery tank before the release, it can be earlier than the release.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The *full* tag records (not sliced to the tagging events).
    window, activity_threshold, sensor_ranges, sensor_range_tolerance, temperature_range
        See :py:func:`flag_records`.
    min_active_duration : str, default: "2h"
        Minimum duration of the first active period (short handling of the tag
        before the release is ignored).
    max_gap : str, default: "30min"
        Active periods separated by less than this are merged.
    level_change_threshold : float, default: 0.5
        Pressure increase (m) from the pre-release level marking the entry in the
        water.

    Returns
    -------
    suggestion : dict
        ``start``: suggested first in-water record (or the first record if the tag
        is active from the beginning, or ``None`` if never active);
        ``pre_release_period``: (first, last) records before ``start`` or ``None``;
        ``pre_release_level``: median pressure (m) of that period (an estimate of
        the in-air reading, see the pressure offset); ``reason`` and
        ``parameters``.
    """
    df, attrs = _to_frame(tag_log)
    parameters = _parameters(
        window=window,
        activity_threshold=activity_threshold,
        min_active_duration=min_active_duration,
        max_gap=max_gap,
        level_change_threshold=level_change_threshold,
        temperature_range=temperature_range,
    )
    result = {
        "start": None,
        "pre_release_period": None,
        "pre_release_level": None,
        "reason": "",
        "parameters": parameters,
    }
    if df.empty:
        result["reason"] = "no records"
        return result

    flags = _flags(
        df,
        attrs,
        window=window,
        activity_threshold=activity_threshold,
        sensor_ranges=sensor_ranges,
        sensor_range_tolerance=sensor_range_tolerance,
        temperature_range=temperature_range,
    )
    active_runs = [
        (first, last)
        for first, last in _runs(~flags["suspect"], max_gap)
        if last - first >= pd.Timedelta(min_active_duration)
    ]
    if not active_runs:
        result["reason"] = (
            f"no active period of at least {min_active_duration} in the record"
        )
        return result

    run_start = active_runs[0][0]
    first_record = df.index[0]
    window_ = pd.Timedelta(window)
    if run_start - first_record <= window_:
        result["start"] = first_record
        result["reason"] = "the tag is active from the first record (already in water?)"
        return result

    pressure = df["pressure"]
    leading = pressure[: run_start - window_]
    level = float(leading.median())
    search = pressure[run_start - window_ : active_runs[0][1]]
    plausible = ~(
        flags["out_of_sensor_range"] | flags["temperature_outside_water_range"]
    ).reindex(search.index)
    departed = search[(search - level > level_change_threshold) & plausible]
    start = departed.index[0] if not departed.empty else run_start

    before = pressure[pressure.index < start]
    result.update(
        {
            "start": start,
            "pre_release_period": (before.index[0], before.index[-1]),
            "pre_release_level": level,
            "reason": (
                f"no vertical activity before {start} (pressure {level:.2f} m), "
                f"then sustained activity"
            ),
        }
    )
    return result


def detect_end_of_track(
    tag_log,
    *,
    start=None,
    window=DEFAULTS["window"],
    activity_threshold=DEFAULTS["activity_threshold"],
    min_duration="12h",
    max_gap=DEFAULTS["max_gap"],
    level_change_threshold=0.5,
    sensor_ranges=None,
    sensor_range_tolerance=DEFAULTS["sensor_range_tolerance"],
    temperature_range=DEFAULTS["temperature_range"],
    air_temperature_daily_range=5.0,
    resumed_activity_warning=0.1,
):
    """Suggest the last in-water / active record of the track.

    The end of the track is the beginning of the first period, after ``start``,
    lasting at least ``min_duration`` in which the records do not look like a
    swimming fish (no vertical activity, values outside the sensor range or
    temperature outside the plausible water range, see :py:func:`flag_records`).
    A period reaching the last record is accepted even if shorter.

    The suggestion is refined to the last record whose pressure differs from the
    level at the beginning of that period by more than ``level_change_threshold``.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The *full* tag records (not sliced to the tagging events).
    start : datetime-like, optional
        Only look after this time (e.g. the release). Defaults to the start
        suggested by :py:func:`detect_start_of_track`.
    window, activity_threshold, sensor_ranges, sensor_range_tolerance, temperature_range
        See :py:func:`flag_records`.
    min_duration : str, default: "12h"
        Minimum duration of the inactive period. Shorter rests are ignored.
    max_gap : str, default: "30min"
        Inactive periods separated by less than this (e.g. a short handling of the
        tag) are merged.
    level_change_threshold : float, default: 0.5
        Pressure change (m) used to refine the end.
    air_temperature_daily_range : float, default: 5.0
        A median daily temperature range (°C) above this in the inactive period
        suggests the tag is in the air (diurnal cycle).
    resumed_activity_warning : float, default: 0.1
        Warn if more than this fraction of the records after the inactive period
        are active again (or if an active period after it lasts at least
        ``min_duration``).

    Returns
    -------
    suggestion : dict
        ``end``: suggested last record of the track (``None`` if no end found);
        ``inactive_period``: (first, last) record of the inactive period;
        ``likely_cause``: a short text; ``evidence``: the numbers behind it
        (fraction of each flag in the inactive period, median daily temperature
        range before and during it, fraction of active records after it and
        longest active period after it);
        ``warning`` (or ``None``); ``reason`` and ``parameters``.

    Examples
    --------
    >>> tag = pangeo_fish.io.open_tag(tag_root, "A19845")
    >>> start = detect_start_of_track(tag)
    >>> end = detect_end_of_track(tag, start=start["start"])
    >>> end["end"], end["likely_cause"]
    """
    df, attrs = _to_frame(tag_log)
    parameters = _parameters(
        window=window,
        activity_threshold=activity_threshold,
        min_duration=min_duration,
        max_gap=max_gap,
        level_change_threshold=level_change_threshold,
        temperature_range=temperature_range,
        air_temperature_daily_range=air_temperature_daily_range,
    )
    result = {
        "end": None,
        "inactive_period": None,
        "likely_cause": None,
        "evidence": {},
        "warning": None,
        "reason": "",
        "parameters": parameters,
    }
    if df.empty:
        result["reason"] = "no records"
        return result

    flag_kwargs = dict(
        window=window,
        activity_threshold=activity_threshold,
        sensor_ranges=sensor_ranges,
        sensor_range_tolerance=sensor_range_tolerance,
        temperature_range=temperature_range,
    )
    if start is None:
        start = detect_start_of_track(
            df.to_xarray().assign_attrs(attrs), **flag_kwargs
        )["start"]
        if start is None:
            start = df.index[0]
    start = _timestamp(start)
    parameters["start"] = start

    flags = _flags(df, attrs, **flag_kwargs)
    last_record = df.index[-1]
    candidates = [
        (first, last)
        for first, last in _runs(flags["suspect"], max_gap)
        if first > start
        and (last - first >= pd.Timedelta(min_duration) or last == last_record)
    ]
    if not candidates:
        result["reason"] = (
            f"no inactive period of at least {min_duration} after {start}: "
            "the tag looks active until the last record"
        )
        return result

    first, last = candidates[0]
    window_ = pd.Timedelta(window)
    pressure = df["pressure"]
    level = float(pressure[first : first + window_].median())
    search = pressure[
        (pressure.index >= first - window_) & (pressure.index < first + window_)
    ]
    moved = search[(search - level).abs() > level_change_threshold]
    end = moved.index[-1] if not moved.empty else first - pd.Timedelta(1, "ns")
    end = pressure.index[pressure.index <= end][-1]

    inactive = flags.loc[first : first + pd.Timedelta(min_duration)]
    temperature = df["temperature"] if "temperature" in df else None
    evidence = {
        "fraction_inactive": float(inactive["inactive"].mean()),
        "fraction_out_of_sensor_range": float(inactive["out_of_sensor_range"].mean()),
        "fraction_temperature_outside_water_range": float(
            inactive["temperature_outside_water_range"].mean()
        ),
        "temperature_daily_range_before": (
            _daily_range(temperature[first - pd.Timedelta("3D") : first])
            if temperature is not None
            else None
        ),
        "temperature_daily_range_during": (
            _daily_range(temperature[first : min(last, first + pd.Timedelta("3D"))])
            if temperature is not None
            else None
        ),
        "fraction_active_after": (
            float((~flags.loc[last:, "suspect"]).iloc[1:].mean())
            if last < last_record
            else 0.0
        ),
        "longest_active_period_after_hours": max(
            [
                (stop - begin) / pd.Timedelta("1h")
                for begin, stop in _runs(~flags.loc[last:, "suspect"], max_gap)
            ],
            default=0.0,
        ),
    }

    daily_range = evidence["temperature_daily_range_during"]
    if evidence["fraction_out_of_sensor_range"] > 0.5:
        cause = (
            "values outside the sensor range (sensor failure or tag out of the water)"
        )
    elif evidence["fraction_temperature_outside_water_range"] > 0.1 or (
        daily_range is not None and daily_range >= air_temperature_daily_range
    ):
        cause = "tag out of the water (air-like temperature or diurnal cycle)"
    else:
        cause = (
            "no vertical movement (fish dead or resting on the bottom, tag lost or "
            "stranded in water)"
        )

    warning = None
    longest = evidence["longest_active_period_after_hours"]
    if evidence[
        "fraction_active_after"
    ] > resumed_activity_warning or longest >= pd.Timedelta(
        min_duration
    ) / pd.Timedelta(
        "1h"
    ):
        warning = (
            f"activity resumes after the inactive period "
            f"({evidence['fraction_active_after']:.0%} of the later records are "
            f"active, longest active period {longest:.1f} h): check the end manually"
        )

    result.update(
        {
            "end": end,
            "inactive_period": (first, last),
            "likely_cause": cause,
            "evidence": evidence,
            "warning": warning,
            "reason": (
                f"inactive from {first} to {last} "
                f"({(last - first) / pd.Timedelta('1h'):.1f} h)"
            ),
        }
    )
    return result


def plot_track_bounds(
    tag_log, start=None, end=None, *, events=None, window="1h", ax=None
):
    """Plot the records with the suggested start and end of the track.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The tag records. For a tag DataTree, the tagging events are also drawn.
    start, end : datetime-like or dict, optional
        Suggested start / end, or the dicts returned by
        :py:func:`detect_start_of_track` / :py:func:`detect_end_of_track`.
    events : mapping of str to datetime-like, optional
        Other times to draw (e.g. ``{"release": ..., "fish_death": ...}``).
    window : str, default: "1h"
        Window of :py:func:`pressure_activity` (inactive records are shaded).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if events is None:
        events = {}
        if isinstance(tag_log, xr.DataTree) and "tagging_events" in tag_log:
            times = tag_log["tagging_events"].to_dataset()["time"].to_series()
            events = {name: time for name, time in times.items() if pd.notna(time)}

    if isinstance(start, dict):
        start = start.get("start")
    if isinstance(end, dict):
        end = end.get("end")

    df, _ = _to_frame(tag_log)
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.figure

    ax.plot(df.index, df["pressure"], color="tab:blue", lw=0.6, label="pressure")
    ax.invert_yaxis()
    ax.set_ylabel("pressure (m)")
    activity = _pressure_activity(df["pressure"], window)
    ax.fill_between(
        df.index,
        0,
        1,
        where=(activity < DEFAULTS["activity_threshold"]).to_numpy(),
        transform=ax.get_xaxis_transform(),
        color="0.85",
        label="no vertical activity",
    )
    if "temperature" in df:
        ax_t = ax.twinx()
        ax_t.plot(df.index, df["temperature"], color="tab:red", lw=0.6, alpha=0.7)
        ax_t.set_ylabel("temperature (°C)", color="tab:red")

    for name, time in events.items():
        ax.axvline(pd.Timestamp(time), color="k", ls=":", lw=1)
        ax.annotate(
            name,
            (pd.Timestamp(time), 1.0),
            xycoords=("data", "axes fraction"),
            rotation=90,
            va="top",
            fontsize=8,
        )
    if start is not None:
        ax.axvline(
            _timestamp(start), color="tab:green", ls="--", label="suggested start"
        )
    if end is not None:
        ax.axvline(_timestamp(end), color="tab:purple", ls="--", label="suggested end")
    ax.legend(loc="lower left", fontsize=8)
    return fig
