"""Twilight quality flagging shared between the solar and lunar pipelines."""

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


def twilight_quality(
    df,
    t_event,
    window_h=1.0,
    max_depth=70,
    slope_threshold=25,
    marge_threshold=8.0,
    min_amplitude=40,
    min_samples=10,
):
    """Evaluate the quality of a single twilight event (RISE or SET).

    This is the calibrated version, tuned on ``nights_profile.csv``.
    Four filters are applied in sequence:

    1. **Slope / direction**: morning events must have positive slope
       > ``slope_threshold``; evening events must have negative slope
       < ``-slope_threshold``.
    2. **Lunar margin**: ``light.max() - dyn_thresh.mean() > marge_threshold``.
    3. **Depth**: ``pressure.max() < max_depth``.
    4. **Amplitude**: ``light.max() - light.min() > min_amplitude``.

    Parameters
    ----------
    df : pd.DataFrame
        Tag data with columns ``time`` (UTC Timestamps), ``light`` (counts),
        ``pressure`` (dbar), and ``dyn_thresh`` (dynamic light threshold).
    t_event : datetime-like
        UTC timestamp of the twilight event (RISE or SET).
    window_h : float, default 1.0
        Half-width of the inspection window around ``t_event`` (hours).
    max_depth : float, default 70
        Maximum allowed pressure (dbar) during the event window.
    slope_threshold : float, default 25
        Minimum absolute slope (counts per window) for direction filter.
    marge_threshold : float, default 8.0
        Minimum margin above the dynamic threshold.
    min_amplitude : float, default 40
        Minimum light range (max − min, counts) inside the window.
    min_samples : int, default 10
        Minimum number of records required inside the window.

    Returns
    -------
    valid : bool
        True if the event passes all four filters.
    dl : float
        Light amplitude inside the window (counts).
    p_max : float
        Maximum pressure inside the window (dbar).
    """
    t = _tz(t_event)
    # Match timezone of df["time"]: keep UTC if tz-aware, strip if tz-naive
    df_tz = getattr(df["time"].dtype, "tz", None)
    t_cmp = t if df_tz is not None else t.tz_convert(None)
    mask = (df["time"] >= t_cmp - pd.Timedelta(hours=window_h)) & (
        df["time"] <= t_cmp + pd.Timedelta(hours=window_h)
    )
    sub = df[mask]

    if len(sub) < min_samples:
        return False, 0.0, 9999.0

    # 1. Direction filter (slope)
    v_start = sub["light"].head(10).mean()
    v_end = sub["light"].tail(10).mean()
    slope = v_end - v_start
    is_morning = t.hour < 12
    if is_morning:
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

    valid = direction_ok and marge_ok and depth_ok and amplitude_ok
    return valid, float(dl), float(p_max)


def compute_quality_flags(df, pairs, max_depth=80):
    """Compute quality flags for each twilight pair.

    For each ``(t_rise, t_set)`` pair, calls :func:`twilight_quality` on
    both events and assigns:

    * ``'good'`` — both events valid
    * ``'partial'`` — exactly one event valid
    * ``'bad'`` — neither event valid

    Parameters
    ----------
    df : pd.DataFrame
        Tag data with columns ``time``, ``light``, ``pressure``,
        ``dyn_thresh``.
    pairs : list of (pd.Timestamp, pd.Timestamp)
        Paired RISE / SET events, as returned by
        :func:`~pangeo_fish.light.solar.detect_twilight_events`.
    max_depth : float, default 80
        Maximum depth (dbar) passed to :func:`twilight_quality`.

    Returns
    -------
    pd.DataFrame
        Columns: ``night``, ``date``, ``flag``, ``valid_rise``,
        ``valid_set``, ``dl_rise``, ``dl_set``, ``dp_rise``, ``dp_set``.
    """
    records = []
    for i, (t_rise, t_set) in enumerate(pairs):
        v_r, dl_r, dp_r = twilight_quality(df, t_rise, max_depth=max_depth)
        v_s, dl_s, dp_s = twilight_quality(df, t_set, max_depth=max_depth)

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

    qdf = pd.DataFrame(records)
    qdf["date"] = pd.to_datetime(qdf["date"])
    return qdf
