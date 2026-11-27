"""Twilight quality flagging shared between the solar and lunar pipelines."""

import numpy as np
import pandas as pd


def _tz(t):
    """Ensure a timestamp is timezone-aware (UTC).

    Parameters
    ----------
    t : datetime-like
        Any timestamp accepted by ``pd.Timestamp``.

    Returns
    -------
    pd.Timestamp
        UTC-aware timestamp.
    """
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def dynamic_threshold(phase, base=60.0, amp=20.0):
    """Phase-dependent light threshold (counts).

    The threshold is raised on bright-moon nights to avoid the moonlight
    being misclassified as 'day'. Linear in moon phase fraction.

    Parameters
    ----------
    phase : float or array-like
        Moon illuminated fraction in percent (0–100).
    base : float, default 60.0
        Baseline threshold (counts) on a new-moon night.
    amp : float, default 20.0
        Amplitude of the moon-phase correction (counts).

    Returns
    -------
    float or array-like
        Dynamic threshold = ``base + amp * (phase / 100)``.
        Same shape as ``phase``.

    Examples
    --------
    >>> dynamic_threshold(0)
    60.0
    >>> dynamic_threshold(100)
    80.0
    >>> dynamic_threshold(50)
    70.0
    """
    return base + amp * (phase / 100.0)


def _event_window(df, t_event, window_h):
    """Rows of ``df`` within ``window_h`` hours of ``t_event`` (both ends included)."""
    t = _tz(t_event)
    times = df["time"]
    # Match the time zone of df["time"]: keep UTC if aware, drop it if naive
    t_cmp = t if getattr(times.dtype, "tz", None) is not None else t.tz_convert(None)
    # an explicit unit: keyword Timedelta construction warns with numpy >= 2.5
    half = np.timedelta64(int(round(window_h * 3_600_000_000_000)), "ns")
    if times.is_monotonic_increasing:
        start = times.searchsorted(t_cmp - half, side="left")
        stop = times.searchsorted(t_cmp + half, side="right")
        return df.iloc[start:stop]
    return df[(times >= t_cmp - half) & (times <= t_cmp + half)]


def twilight_quality(
    df,
    t_event,
    window_h=1.0,
    max_depth=70,
    slope_threshold=25,
    marge_threshold=8.0,
    min_amplitude=40,
    min_samples=10,
    direction=None,
):
    """Evaluate the quality of a single twilight event (rising or setting).

    Four filters are applied:

    1. **Direction**: the light must go up through a rising and down through
       a setting: the mean of the last 10 records of the window minus the
       mean of the first 10 must be above ``slope_threshold`` for a rising
       and below ``-slope_threshold`` for a setting. ``slope_threshold=None``
       turns this filter off.
    2. **Lunar margin**: ``light.max() - dyn_thresh.mean() > marge_threshold``.
    3. **Depth**: ``pressure.max() < max_depth``.
    4. **Amplitude**: ``light.max() - light.min() > min_amplitude``.

    Parameters
    ----------
    df : pandas.DataFrame
        Tag data with columns ``time`` (UTC), ``light``, ``pressure`` (dbar)
        and ``dyn_thresh`` (light threshold per row).
    t_event : datetime-like
        UTC time of the event.
    window_h : float, default 1.0
        Half-width of the window around ``t_event``, hours.
    max_depth : float, default 70
        Largest allowed pressure in the window, dbar.
    slope_threshold : float or None, default 25
        Smallest light change across the window for the direction filter.
        ``None`` turns the filter off.
    marge_threshold : float, default 8.0
        Smallest margin of the peak light above the threshold.
    min_amplitude : float, default 40
        Smallest light range (max - min) in the window.
    min_samples : int, default 10
        Smallest number of records in the window.
    direction : {"rise", "set"}, optional
        Kind of event. :func:`compute_quality_flags` always gives it. Without
        it the event is taken as a rising when its UTC hour is before 12,
        which is only right near the Greenwich meridian.

    Returns
    -------
    valid : bool
        True if the event passes all filters.
    dl : float
        Light range in the window.
    p_max : float
        Largest pressure in the window, dbar.
    """
    t = _tz(t_event)
    sub = _event_window(df, t, window_h)

    if len(sub) < min_samples:
        return False, 0.0, 9999.0

    # 1. Direction filter (slope)
    if slope_threshold is None:
        direction_ok = True
    else:
        if direction is None:
            direction = "rise" if t.hour < 12 else "set"
        if direction not in ("rise", "set"):
            raise ValueError(f"direction must be 'rise' or 'set', got {direction!r}")
        v_start = sub["light"].head(10).mean()
        v_end = sub["light"].tail(10).mean()
        slope = v_end - v_start
        if direction == "rise":
            direction_ok = slope > slope_threshold
        else:
            direction_ok = slope < -slope_threshold

    # 2. Lunar margin filter
    current_thresh = sub["dyn_thresh"].mean()
    marge = sub["light"].max() - current_thresh
    marge_ok = marge > marge_threshold

    # 3. Depth filter
    p_max = sub["pressure"].max()
    depth_ok = p_max < max_depth

    # 4. Amplitude filter
    dl = sub["light"].max() - sub["light"].min()
    amplitude_ok = dl > min_amplitude

    valid = bool(direction_ok and marge_ok and depth_ok and amplitude_ok)
    return valid, float(dl), float(p_max)


def compute_quality_flags(
    df,
    pairs,
    max_depth=80,
    *,
    window_h=1.0,
    slope_threshold=25,
    marge_threshold=8.0,
    min_amplitude=40,
    min_samples=10,
):
    """Compute quality flags for each twilight pair.

    For each ``(t_rise, t_set)`` pair, calls :func:`twilight_quality` on
    both events (with ``direction="rise"`` and ``direction="set"``) and
    assigns:

    * ``'good'`` -- both events valid
    * ``'partial'`` -- exactly one event valid
    * ``'bad'`` -- neither event valid

    Parameters
    ----------
    df : pandas.DataFrame
        Tag data with columns ``time``, ``light``, ``pressure``,
        ``dyn_thresh``.
    pairs : list of (pandas.Timestamp, pandas.Timestamp)
        ``(t_rise, t_set)`` pairs from
        :func:`~pangeo_fish.light.solar.detect_twilight_events`.
    max_depth : float, default 80
        Largest allowed pressure in the window, dbar.
    window_h, slope_threshold, marge_threshold, min_amplitude, min_samples
        Passed to :func:`twilight_quality`. ``slope_threshold=None`` turns the
        direction filter off.

    Returns
    -------
    pandas.DataFrame
        Columns: ``night``, ``date``, ``flag``, ``valid_rise``,
        ``valid_set``, ``dl_rise``, ``dl_set``, ``dp_rise``, ``dp_set``.
        ``date`` is the UTC date of the rising.
    """
    if not df["time"].is_monotonic_increasing:
        df = df.sort_values("time")
    options = dict(
        window_h=window_h,
        max_depth=max_depth,
        slope_threshold=slope_threshold,
        marge_threshold=marge_threshold,
        min_amplitude=min_amplitude,
        min_samples=min_samples,
    )
    records = []
    for i, (t_rise, t_set) in enumerate(pairs):
        v_r, dl_r, dp_r = twilight_quality(df, t_rise, direction="rise", **options)
        v_s, dl_s, dp_s = twilight_quality(df, t_set, direction="set", **options)

        if v_r and v_s:
            flag = "good"
        elif v_r or v_s:
            flag = "partial"
        else:
            flag = "bad"

        records.append(
            {
                "night": i,
                "date": _tz(t_rise).date(),
                "flag": flag,
                "valid_rise": v_r,
                "valid_set": v_s,
                "dl_rise": dl_r,
                "dl_set": dl_s,
                "dp_rise": dp_r,
                "dp_set": dp_s,
            }
        )

    columns = ["night", "date", "flag", "valid_rise", "valid_set"]
    columns += ["dl_rise", "dl_set", "dp_rise", "dp_set"]
    qdf = pd.DataFrame(records, columns=columns)
    qdf["date"] = pd.to_datetime(qdf["date"])
    return qdf
