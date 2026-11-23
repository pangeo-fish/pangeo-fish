"""Suggest the start and the end of a tag track from the records themselves.

Archival tags often record before the release (tag in the air, on the boat) and
after the tag leaves the water (stranded on a beach, picked up and stored, sensor
failure). A fish that dies or sheds the tag leaves the tag in the water: that is
not the end of the track, the tag still moves. The functions of this module look at the
pressure and temperature records only and *suggest* the first and the last record
of the track. They never modify the tagging events: compare the suggestion with
the tagging events and decide.

The main criterion is the vertical activity of the tag: a swimming fish changes
depth all the time, while a tag in the air, stranded or lying on the seabed only
shows the sensor noise (and the slow tide for a tag on the seabed). The start is
the first sustained activity. The end is the last record before the tag leaves the
water, found among the periods without activity. The activity
is the mean absolute deviation of the pressure from its running median (the
``activity`` variable of :py:func:`flag_records`). Values outside the sensor range
and temperatures outside the plausible water range are also used. A pressure back
at the in-air reading of the tag is only reported: it does not end the track.

:py:func:`pangeo_fish.visualization.plot_track_bounds` shows the records with the
suggestions.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

DEFAULTS = {
    "window": "1h",
    "activity_threshold": 0.1,
    "max_gap": "30min",
    "sensor_range_tolerance": 0.01,
    "temperature_range": (-2.0, 30.0),
    "min_active_duration": "2h",
}


def _timedelta(value):
    """``pd.Timedelta(value)``, parsing strings like ``"30min"`` with ``pd.to_timedelta``.

    With pandas < 3 and numpy >= 2.5, ``pd.Timedelta("30min")`` creates a
    timedelta64 with the deprecated generic unit (DeprecationWarning).
    """
    if isinstance(value, str):
        return pd.to_timedelta([value])[0]
    return pd.Timedelta(value)


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
    # duplicated times: keep the first record
    df = df[~df.index.duplicated()]
    return df, ds.attrs


def _pressure_activity(pressure, window):
    """Vertical activity of the tag (m).

    The running mean (over ``window``) of the absolute deviation of the pressure
    from its running median (over ``window``). It is close to the sensor noise (a
    few cm) for a tag in the air, stranded or lying on the seabed (the tide is
    removed by the running median), and of the order of 1 m for a swimming fish.
    """
    median = pressure.rolling(window, center=True, min_periods=1).median()
    deviation = (pressure - median).abs()
    return deviation.rolling(window, center=True, min_periods=1).mean()


def _warn_if_undetectable(df, activity, *, window, activity_threshold, stacklevel=3):
    """Warn when the sampling or the noise of the record defeats the activity criterion.

    Detection itself is not changed.
    """
    interval = df.index.to_series().diff().median()
    if pd.notna(interval) and interval > _timedelta(window) / 3:
        warnings.warn(
            f"track_bounds: the median sampling interval ({interval}) is longer"
            f" than a third of the activity window ({window}): the activity is"
            " computed from too few records per window and the start and the end"
            " of the track cannot be detected. Use a longer `window`.",
            UserWarning,
            stacklevel=stacklevel,
        )

    activity = activity.dropna()
    if len(activity):
        # The quietest tenth of the record shows the sensor noise when the tag
        # has quiet periods (in the air, stranded, on the seabed). A fish that
        # is active throughout has no quiet level: its quietest tenth is not far
        # below its usual activity, so only a quiet level well below the active
        # one (a quarter of its 90th percentile) is taken as noise.
        n_quiet = max(1, len(activity) // 10)
        noise = float(activity.nsmallest(n_quiet).median())
        active = float(activity.quantile(0.9))
        if activity_threshold < noise < 0.25 * active:
            warnings.warn(
                f"track_bounds: the vertical activity of the quietest tenth of the"
                f" record is {noise:.2g} m (median), above activity_threshold ="
                f" {activity_threshold:g} m, while the usual activity is about"
                f" {active:.2g} m: periods without movement are not flagged"
                " inactive, so the start and the end of the track may be missed."
                " Raise `activity_threshold` above the noise of the record.",
                UserWarning,
                stacklevel=stacklevel,
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
    # no pressure for a whole window (NaN activity) is not activity either
    flags["inactive"] = ~(flags["activity"] >= activity_threshold)

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
        The tag records, with a ``time`` dimension, ``pressure`` (m) and optionally
        ``temperature`` (°C). A tag DataTree (as returned by
        :py:func:`pangeo_fish.io.open_tag`) uses its ``dst`` node.
    window : str, default: "1h"
        Width of the running windows used for the vertical activity (pandas
        offset alias).
    activity_threshold : float, default: 0.1
        Records with an activity below this value (m) are flagged ``inactive``.
        The default suits 1-min records with a pressure resolution of a few cm.
    sensor_ranges : mapping of str to (float, float), optional
        Valid range of each sensor, e.g. ``{"pressure": (0, 300)}``. Defaults to a
        ``sensor_ranges`` attribute of the records, if present (no check
        otherwise).
    sensor_range_tolerance : float, default: 0.01
        Tolerance, as a fraction of the sensor range, before flagging a record as
        ``out_of_sensor_range``.
    temperature_range : (float, float) or None, default: (-2, 30)
        Plausible water temperature (°C). Records outside are flagged
        ``temperature_outside_water_range``. ``None`` disables this check.

    Returns
    -------
    flags : xarray.Dataset
        ``activity`` (m): the running mean (over ``window``) of the absolute
        deviation of the pressure from its running median. It is close to the
        sensor noise (a few cm) for a tag in the air, stranded or lying on the
        seabed (the tide is removed by the running median), and of the order of
        1 m for a swimming fish. The boolean flags ``inactive`` (activity below
        ``activity_threshold``, or no pressure for a whole ``window``),
        ``out_of_sensor_range``, ``temperature_outside_water_range`` and
        ``suspect`` (any of them). Records without a time are dropped, and of
        records with the same time only the first is kept.
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

    max_gap = _timedelta(max_gap)
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
    min_active_duration=DEFAULTS["min_active_duration"],
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

    Warns
    -----
    UserWarning
        If the median sampling interval is longer than a third of ``window``
        (the activity is computed from too few records), or if the quietest
        tenth of the record is noisier than ``activity_threshold`` while the
        usual activity is much higher (periods without movement are then not
        flagged inactive: raise ``activity_threshold``). The detection itself is
        not changed.
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
    _warn_if_undetectable(
        df, flags["activity"], window=window, activity_threshold=activity_threshold
    )
    active_runs = [
        (first, last)
        for first, last in _runs(~flags["suspect"], max_gap)
        if last - first >= _timedelta(min_active_duration)
    ]
    if not active_runs:
        result["reason"] = (
            f"no active period of at least {min_active_duration} in the record"
        )
        return result

    run_start = active_runs[0][0]
    first_record = df.index[0]
    window_ = _timedelta(window)
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


def _long_runs(mask, max_gap, min_duration, last_record):
    """Runs of ``mask`` lasting at least ``min_duration`` (or reaching the last record)."""
    return [
        (first, last)
        for first, last in _runs(mask, max_gap)
        if last - first >= _timedelta(min_duration) or last == last_record
    ]


def _air_cycle_onset(temperature, window, threshold, margin=0.5):
    """First record of a diurnal temperature cycle (tag in the air), or None.

    The day bins start at the first record. A bin shows the cycle if its range
    and the median range of that bin and the next two are at least ``threshold``
    (°C). The onset is the first record, up to the end of that bin, whose
    temperature leaves by more than ``margin`` (°C) the range of the day before
    it (or of the records since the first one), not counting the last
    ``window``, and whose next day has a range of at least ``threshold``: a slow
    change of the water temperature (tide, trend) is not an onset. Without such
    a record, the onset is the beginning of the bin.
    """
    temperature = temperature.dropna()
    if temperature.empty:
        return None
    first = temperature.index[0]
    day = pd.Timedelta(1, "D")
    daily = temperature.resample("1D", origin=first).agg(["min", "max", "count"])
    daily = daily[daily["count"] > 0]
    ranges = daily["max"] - daily["min"]
    cycle = next(
        (
            begin
            for i, begin in enumerate(ranges.index)
            if ranges.iloc[i] >= threshold
            and ranges.iloc[i : i + 3].median() >= threshold
        ),
        None,
    )
    if cycle is None:
        return None

    # range of the records from one day to one window before each record
    window = _timedelta(window)
    previous = temperature.rolling(day - window, closed="both")
    lagged = temperature.index - window
    high = previous.max().reindex(lagged, method="ffill").to_numpy()
    low = previous.min().reindex(lagged, method="ffill").to_numpy()
    values = temperature.to_numpy()
    with np.errstate(invalid="ignore"):
        departed = (values > high + margin) | (values < low - margin)
    departed &= temperature.index < cycle + day
    for time in temperature.index[departed]:
        following = temperature[time : time + day]
        if following.max() - following.min() >= threshold:
            return time
    return cycle


def _water_exit(flags, temperature, first, last, *, last_record, parameters):
    """``(onset, cause)`` of the exit from the water inside one suspect run, or None.

    The earliest of: records outside the sensor range or with a temperature
    outside the water range (in a run lasting at least ``min_duration`` or
    reaching the last record), or the diurnal temperature cycle; a tie goes to
    the sensor range. A diurnal cycle found within three ``window`` of the
    beginning of the run starts with the run: the temperature of a tag in the
    air can take a few hours to change.
    """
    max_gap = parameters["max_gap"]
    min_duration = parameters["min_duration"]
    window = _timedelta(parameters["window"])
    inside = flags.loc[first:last]
    onsets = {}
    for cause, name in [
        ("out_of_sensor_range", "out_of_sensor_range"),
        ("out_of_water", "temperature_outside_water_range"),
    ]:
        runs = _long_runs(inside[name], max_gap, min_duration, last_record)
        if runs:
            onsets[cause] = runs[0][0]
    if temperature is not None:
        air = _air_cycle_onset(
            temperature[first:last],
            parameters["window"],
            parameters["air_temperature_daily_range"],
        )
        if air is not None and air - first <= 3 * window:
            air = first
        if air is not None and (
            "out_of_water" not in onsets or air < onsets["out_of_water"]
        ):
            onsets["out_of_water"] = air
    if not onsets:
        return None
    cause = min(onsets, key=lambda name: (onsets[name], name != "out_of_sensor_range"))
    return onsets[cause], cause


def _air_level_from(
    flags, pressure, *, after, end, last_record, min_active_duration, parameters
):
    """First record of the in-air pressure reading to report, or None.

    The runs without vertical movement at the in-air pressure reading, from
    ``after`` and starting before ``end``, lasting at least ``min_duration`` (or
    reaching the last record). A run is not reported if the tag is in the water
    later, up to ``end`` (or the last record): a recovered tag does not go back
    to the water. In the water means an active period of at least
    ``min_active_duration``, or a pressure deeper than the in-air reading for at
    least ``min_duration``.
    """
    max_gap = parameters["max_gap"]
    min_duration = _timedelta(parameters["min_duration"])
    tolerance = parameters["air_level_tolerance"]
    limit = end if end is not None else last_record
    depth = pressure.loc[after:] - parameters["air_level"]
    records = flags.loc[after:]
    at_air_level = records["inactive"] & (depth.abs() <= tolerance)
    runs = [
        begin
        for begin, _ in _long_runs(at_air_level, max_gap, min_duration, last_record)
        if begin < limit
    ]
    if not runs:
        return None
    active = [
        begin
        for begin, stop in _runs(~records.loc[:limit, "suspect"], max_gap)
        if stop - begin >= _timedelta(min_active_duration)
    ]
    deep = [
        begin
        for begin, stop in _runs(depth.loc[:limit] > tolerance, max_gap)
        if stop - begin >= min_duration
    ]
    in_water = max(active + deep, default=None)
    return next(
        (begin for begin in runs if in_water is None or begin >= in_water), None
    )


def _sustained_activity_start(flags, start, max_gap, duration):
    """First record from ``start`` of an active period lasting ``duration``, or None."""
    for first, last in _runs(~flags["suspect"], max_gap):
        first = max(first, start)
        if last - first >= _timedelta(duration):
            return first
    return None


def detect_end_of_track(
    tag_log,
    *,
    start=None,
    window=DEFAULTS["window"],
    activity_threshold=DEFAULTS["activity_threshold"],
    min_duration="12h",
    min_active_duration=DEFAULTS["min_active_duration"],
    max_gap=DEFAULTS["max_gap"],
    level_change_threshold=0.5,
    air_level=None,
    air_level_tolerance=0.5,
    sensor_ranges=None,
    sensor_range_tolerance=DEFAULTS["sensor_range_tolerance"],
    temperature_range=DEFAULTS["temperature_range"],
    air_temperature_daily_range=5.0,
    resumed_activity_warning=0.1,
):
    """Suggest the last record before the tag leaves the water.

    The track ends when the *tag leaves the water*, not when the fish stops
    moving: after the fish dies (or the tag is shed) the tag is still in the
    water, and the positions estimated for that time still show where the tag
    went. A period without vertical movement in the water (fish dead or resting,
    tag lost) does *not* end the track: it is only reported in
    ``inactive_period``.

    The periods that do not look like a swimming fish (no vertical activity,
    values outside the sensor range or temperature outside the plausible water
    range, see :py:func:`flag_records`), lasting at least ``min_duration`` (or
    reaching the last record), are examined from the first sustained activity in
    the water after ``start`` (at least ``min_active_duration``): the periods
    before it (tag in the air or on the deck before the release) are not. The
    temperature and the sensor range are used only inside these periods, so an
    active fish crossing a thermocline is not flagged. Inside a period, the tag
    leaves the water at the first of:

    - records outside the sensor range (for at least ``min_duration``);
    - temperatures outside the plausible water range (for at least
      ``min_duration``);
    - a diurnal temperature cycle (a daily range of at least
      ``air_temperature_daily_range``, kept for the next days).

    A period can hold both the death of the fish and the stranding of the tag
    (fish dead on the seabed for days, then tag on a beach): the end is the
    stranding. The suggestion is the last record before the onset; when the onset
    is at the beginning of the period it is refined to the last record whose
    pressure differs from the level at the onset by more than
    ``level_change_threshold``.

    A pressure back at the in-air reading ``air_level`` (within
    ``air_level_tolerance``) without vertical movement, for at least
    ``min_duration``, does *not* end the track. A tag kept indoors or in a fridge
    after its recovery records it (with a temperature that looks like water), but
    a tag floating at the surface or lying in very shallow water looks the same.
    It is only reported (``evidence["pressure_at_air_level_from"]`` and
    ``warning``), and only if the tag is not in the water again later (an active
    period of at least ``min_active_duration``, or a pressure deeper than the
    in-air reading for at least ``min_duration``): a recovered tag does not go
    back to the water. Check it against the recovery of the tag.

    Parameters
    ----------
    tag_log : xarray.Dataset or xarray.DataTree
        The *full* tag records (not sliced to the tagging events).
    start : datetime-like, optional
        Only look after this time (e.g. the release). Defaults to the start
        suggested by :py:func:`detect_start_of_track`, called with the same
        ``window``, ``activity_threshold``, ``min_active_duration``, ``max_gap``,
        ``level_change_threshold``, ``sensor_ranges``, ``sensor_range_tolerance``
        and ``temperature_range``.
    window, activity_threshold, sensor_ranges, sensor_range_tolerance, temperature_range
        See :py:func:`flag_records`.
    min_duration : str, default: "12h"
        Minimum duration of a period without vertical movement, and of a period
        outside the sensor or water temperature range. Shorter ones are ignored.
    min_active_duration : str, default: "2h"
        Minimum duration of the first active period after ``start``. The periods
        before it are not examined; if the tag is never active that long, all
        the periods after ``start`` are, but none is reported as
        ``inactive_period`` and the in-air pressure reading is not reported.
    max_gap : str, default: "30min"
        Periods separated by less than this (e.g. a short handling of the tag)
        are merged.
    level_change_threshold : float, default: 0.5
        Pressure change (m) used to refine the end (and the start, if ``start`` is
        not given).
    air_level : float, optional
        Pressure (m) read by the tag in the air. Defaults to the pre-release
        level found by :py:func:`detect_start_of_track` (called with the same
        parameters as above), or to 0 if the tag records no pre-release period.
    air_level_tolerance : float, default: 0.5
        Maximum difference (m) from ``air_level`` of a pressure reading in the
        air.
    air_temperature_daily_range : float, default: 5.0
        A daily temperature range (°C) of at least this, in a period without
        vertical movement, suggests the tag is in the air (diurnal cycle).
    resumed_activity_warning : float, default: 0.1
        Warn if more than this fraction of the records after the period that
        holds the end are active again (or if an active period after it lasts at
        least ``min_duration``).

    Returns
    -------
    suggestion : dict
        ``end``: suggested last record before the tag leaves the water (never
        before ``start``), or ``None`` if nothing shows that the tag leaves the
        water (the whole record after ``start`` can be used, but read the
        ``warning``);
        ``inactive_period``: (first, last) record of the first period without
        vertical movement *in the water*, or ``None``. It does **not** end the
        track: the fish is probably dead or resting, or the tag was shed. If the
        tag then leaves the water, ``last`` is the suggested ``end``;
        ``likely_cause``: ``None`` if no end, else:

        - ``"out_of_sensor_range"``: the records are outside the sensor range
          (sensor failure, or tag out of the water);
        - ``"out_of_water"``: temperatures outside the plausible water range, or
          a diurnal temperature cycle (tag in the air, e.g. stranded on a beach);

        ``evidence``: ``pressure_at_air_level_from`` (always present): first
        record of a period without vertical movement at the in-air pressure
        reading, before the ``end`` if there is one, else ``None``; and, if a
        period without vertical movement was examined, the numbers behind the
        suggestion (fraction of each flag, median daily temperature range before
        and after the onset of that period, fraction of active records after it
        and longest active period after it);
        ``warning`` (or ``None``): activity resumes after the suggested ``end``,
        or the pressure is back at the in-air reading (the tag may have left the
        water then);
        ``reason`` and ``parameters``.

    Warns
    -----
    UserWarning
        If the median sampling interval is longer than a third of ``window``
        (the activity is computed from too few records), or if the quietest
        tenth of the record is noisier than ``activity_threshold`` while the
        usual activity is much higher (periods without movement are then not
        flagged inactive: raise ``activity_threshold``). The detection itself is
        not changed.

    Notes
    -----
    Known limits: the onset of a diurnal cycle is the first record that leaves
    the temperature range of the previous day by 0.5 °C, so it is found to
    about an hour or a few hours (later at night, when the temperature in the
    air changes slowly). One found within three ``window`` of the beginning of
    the period is moved to the beginning: a tag floating less than about three
    hours before it strands ends when it starts floating. In water whose
    temperature changes by more than 0.5 °C within hours (tide, fronts), the
    onset can be found early when the tag strands less than a day after the
    beginning of the period. A tag that leaves the water without a
    temperature signal (a pressure-only tag, or a tag kept indoors) gets no
    ``end``: only a pressure back at ``air_level`` is reported, and not even
    that if the pressure reading drifts in the sun or the in-air reading changed
    since the release (pass ``air_level`` if the in-air reading is known).

    Examples
    --------
    >>> tag = pangeo_fish.io.open_tag(tag_root, "A19845")
    >>> start = detect_start_of_track(tag)
    >>> end = detect_end_of_track(tag, start=start["start"])
    >>> end["end"], end["likely_cause"], end["warning"]
    """
    df, attrs = _to_frame(tag_log)
    parameters = _parameters(
        window=window,
        activity_threshold=activity_threshold,
        min_duration=min_duration,
        min_active_duration=min_active_duration,
        max_gap=max_gap,
        level_change_threshold=level_change_threshold,
        air_level_tolerance=air_level_tolerance,
        temperature_range=temperature_range,
        air_temperature_daily_range=air_temperature_daily_range,
    )
    result = {
        "end": None,
        "inactive_period": None,
        "likely_cause": None,
        "evidence": {"pressure_at_air_level_from": None},
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
    if start is None or air_level is None:
        with warnings.catch_warnings():
            # the warnings of this call are emitted once, below
            warnings.filterwarnings("ignore", message="track_bounds:")
            suggestion = detect_start_of_track(
                df.to_xarray().assign_attrs(attrs),
                min_active_duration=min_active_duration,
                max_gap=max_gap,
                level_change_threshold=level_change_threshold,
                **flag_kwargs,
            )
        if start is None:
            start = suggestion["start"]
            if start is None:
                start = df.index[0]
        if air_level is None:
            air_level = suggestion["pre_release_level"]
            if air_level is None:
                air_level = 0.0
    start = _timestamp(start)
    parameters["start"] = start
    parameters["air_level"] = float(air_level)

    flags = _flags(df, attrs, **flag_kwargs)
    _warn_if_undetectable(
        df, flags["activity"], window=window, activity_threshold=activity_threshold
    )
    last_record = df.index[-1]
    window_ = _timedelta(window)
    pressure = df["pressure"]
    temperature = df["temperature"] if "temperature" in df else None
    # the periods before the fish is in the water (tag in the air or on the deck
    # before the release) are not examined
    in_water = _sustained_activity_start(flags, start, max_gap, min_active_duration)
    candidates = [
        (first, last)
        for first, last in _long_runs(
            flags["suspect"], max_gap, min_duration, last_record
        )
        if first > start and (in_water is None or first >= in_water)
    ]

    # the first period without movement in the water, and the exit from the water
    inactive_period = None
    exit_ = None
    for first, last in candidates:
        hard = _water_exit(
            flags,
            temperature,
            first,
            last,
            last_record=last_record,
            parameters=parameters,
        )
        if hard is None:
            if inactive_period is None and in_water is not None:
                inactive_period = (first, last)
            continue
        onset, cause = hard
        end = onset - pd.Timedelta(1, "ns")
        if onset - first <= window_:
            level = float(pressure[onset : onset + window_].median())
            search = pressure[
                (pressure.index >= onset - window_) & (pressure.index < onset)
            ]
            if np.isnan(level):
                # no pressure at the onset: the last record with a pressure
                moved = search.dropna()
            else:
                moved = search[(search - level).abs() > level_change_threshold]
            if not moved.empty:
                end = moved.index[-1]
        # the last record before the onset, but never before ``start``
        after_start = pressure.index[pressure.index >= start]
        before = after_start[after_start <= end]
        end = before[-1] if len(before) else after_start[0]
        if (
            inactive_period is None
            and in_water is not None
            and end - first >= _timedelta(min_duration)
        ):
            inactive_period = (first, end)
        exit_ = (first, last, onset, cause, end)
        break

    air_from = None
    if in_water is not None:
        air_from = _air_level_from(
            flags,
            pressure,
            after=in_water,
            end=exit_[4] if exit_ is not None else None,
            last_record=last_record,
            min_active_duration=min_active_duration,
            parameters=parameters,
        )

    air_reading = f"{parameters['air_level']:.2f} ± {air_level_tolerance:g} m"
    if exit_ is None:
        result["inactive_period"] = inactive_period
        result["reason"] = (
            f"the tag stays in the water until the last record: no period out of"
            f" the water after {start}"
        )
        if in_water is None:
            result["reason"] = (
                f"no sustained vertical activity (at least {min_active_duration})"
                f" after {start}: the tag may not have been in the water, or"
                " `start` is too late"
            )
        if inactive_period is not None:
            first, last = inactive_period
            result["reason"] += (
                f"; no vertical movement from {first} to {last}"
                f" ({(last - first) / pd.Timedelta(1, 'h'):.1f} h): fish probably"
                " dead or resting, or tag shed"
            )
            result["evidence"] = _evidence(
                flags, temperature, first, last, last_record, min_duration, max_gap
            )
        result["evidence"]["pressure_at_air_level_from"] = air_from
        if air_from is not None:
            result[
                "reason"
            ] += f"; the pressure is at the in-air reading from {air_from}"
            result["warning"] = (
                f"the pressure is back at the in-air reading ({air_reading}) without"
                f" vertical movement from {air_from}: the tag may have left the water"
                " then (kept indoors or in a fridge after its recovery), or it floats"
                " at the surface or lies in very shallow water. The end is not set"
                " from the pressure alone: check it against the recovery of the tag"
            )
        return result

    first, last, onset, cause, end = exit_
    evidence = _evidence(
        flags, temperature, onset, last, last_record, min_duration, max_gap
    )
    evidence["pressure_at_air_level_from"] = air_from
    warnings_ = []
    longest = evidence["longest_active_period_after_hours"]
    min_duration_hours = _timedelta(min_duration) / pd.Timedelta(1, "h")
    if (
        evidence["fraction_active_after"] > resumed_activity_warning
        or longest >= min_duration_hours
    ):
        warnings_.append(
            f"activity resumes after the suggested end "
            f"({evidence['fraction_active_after']:.0%} of the later records are "
            f"active, longest active period {longest:.1f} h): check the end manually"
        )
    if air_from is not None:
        warnings_.append(
            f"the pressure is back at the in-air reading ({air_reading}) without"
            f" vertical movement from {air_from}, before the {cause} evidence: the"
            " tag may have left the water then (or it floated at the surface or lay"
            " in very shallow water), check the end against the recovery of the tag"
        )
    warning = "; ".join(warnings_) or None

    reason = (
        f"the tag leaves the water at {onset} ({cause}); the period without"
        f" vertical movement or with unusable records runs from {first} to {last}"
        f" ({(last - first) / pd.Timedelta(1, 'h'):.1f} h)"
    )
    if inactive_period is not None:
        reason += (
            f"; the tag was in the water without vertical movement from"
            f" {inactive_period[0]} (fish probably dead or resting, or tag shed)"
        )
    result.update(
        {
            "end": end,
            "inactive_period": inactive_period,
            "likely_cause": cause,
            "evidence": evidence,
            "warning": warning,
            "reason": reason,
        }
    )
    return result


def _evidence(flags, temperature, begin, last, last_record, min_duration, max_gap):
    """Numbers behind the suggestion, for the period from ``begin`` to ``last``."""
    inactive = flags.loc[begin : begin + _timedelta(min_duration)]
    return {
        "fraction_inactive": float(inactive["inactive"].mean()),
        "fraction_out_of_sensor_range": float(inactive["out_of_sensor_range"].mean()),
        "fraction_temperature_outside_water_range": float(
            inactive["temperature_outside_water_range"].mean()
        ),
        "temperature_daily_range_before": (
            _daily_range(temperature[begin - pd.Timedelta(3, "D") : begin])
            if temperature is not None
            else None
        ),
        "temperature_daily_range_during": (
            _daily_range(temperature[begin : min(last, begin + pd.Timedelta(3, "D"))])
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
                (stop - begin_) / pd.Timedelta(1, "h")
                for begin_, stop in _runs(~flags.loc[last:, "suspect"], max_gap)
            ],
            default=0.0,
        ),
    }
