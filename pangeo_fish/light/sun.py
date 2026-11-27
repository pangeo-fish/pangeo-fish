"""Vectorised solar geometry for light-based geolocation.

The functions here work on NumPy arrays of times and positions at once, so
that a likelihood can be evaluated on every cell of a HEALPix grid without a
Python loop over cells.

The formulas are the low-precision solar position of the NOAA Solar
Calculator (Meeus, *Astronomical Algorithms*, chapter 25): declination and
equation of time from the apparent solar longitude. Over 1950--2050 they agree
with :mod:`ephem` (geometric centre, no refraction) to well under a minute of
crossing time, which is small compared with the timing uncertainty of a light
tag (``sigma_timing`` of several minutes).

Conventions
-----------
- Times are UTC, as ``numpy.datetime64`` (any unit) or anything
  :func:`pandas.to_datetime` accepts. Time-zone-aware inputs are converted to
  UTC first.
- Longitudes are degrees East, latitudes degrees North.
- Elevations are the geometric elevation of the centre of the Sun, in degrees,
  without atmospheric refraction (the same as ``ephem`` with ``pressure = 0``
  and ``use_center=True``).
"""

import numpy as np
import pandas as pd

_MINUTES_PER_DAY = 1440.0
_NS_PER_MINUTE = 60_000_000_000


def _as_utc_ns(times):
    """Return ``times`` as ``datetime64[ns]`` in UTC (time-zone naive)."""
    if isinstance(times, pd.Timestamp | pd.DatetimeIndex | pd.Series):
        values = pd.to_datetime(times)
    else:
        arr = np.asarray(times)
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]")
        values = pd.to_datetime(arr.ravel()).values.reshape(arr.shape)
        return np.asarray(values, dtype="datetime64[ns]")
    if isinstance(values, pd.Timestamp):
        if values.tzinfo is not None:
            values = values.tz_convert("UTC").tz_localize(None)
        return np.datetime64(values.to_datetime64(), "ns")
    if isinstance(values, pd.Series):
        values = pd.DatetimeIndex(values)
    if values.tz is not None:
        values = values.tz_convert("UTC").tz_localize(None)
    return np.asarray(values.values, dtype="datetime64[ns]")


def _minutes_since_epoch(times_ns):
    """Minutes since 1970-01-01 as float64."""
    return times_ns.astype("int64") / _NS_PER_MINUTE


def _from_minutes_since_epoch(minutes):
    """Inverse of :func:`_minutes_since_epoch`; NaN gives ``NaT``."""
    minutes = np.asarray(minutes, dtype="float64")
    out = np.full(minutes.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = np.isfinite(minutes)
    out[ok] = (
        np.round(minutes[ok] * _NS_PER_MINUTE).astype("int64").view("datetime64[ns]")
    )
    return out


def declination_and_equation_of_time(times):
    """Solar declination (degrees) and equation of time (minutes).

    Parameters
    ----------
    times : array-like of datetime
        UTC times.

    Returns
    -------
    declination : numpy.ndarray
        Declination of the Sun, degrees.
    equation_of_time : numpy.ndarray
        Apparent minus mean solar time, minutes.
    """
    minutes = _minutes_since_epoch(_as_utc_ns(times))
    return _declination_eot_from_minutes(minutes)


def _declination_eot_from_minutes(minutes):
    jd = minutes / _MINUTES_PER_DAY + 2440587.5
    t = (jd - 2451545.0) / 36525.0

    l0 = np.mod(280.46646 + t * (36000.76983 + t * 0.0003032), 360.0)
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m_rad = np.radians(m)
    c = (
        np.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + np.sin(3 * m_rad) * 0.000289
    )
    true_long = l0 + c
    omega = np.radians(125.04 - 1934.136 * t)
    apparent_long = np.radians(true_long - 0.00569 - 0.00478 * np.sin(omega))

    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    mean_obliquity = 23.0 + (26.0 + seconds / 60.0) / 60.0
    obliquity = np.radians(mean_obliquity + 0.00256 * np.cos(omega))

    declination = np.degrees(np.arcsin(np.sin(obliquity) * np.sin(apparent_long)))

    y = np.tan(obliquity / 2.0) ** 2
    l0_rad = np.radians(l0)
    eot = 4.0 * np.degrees(
        y * np.sin(2 * l0_rad)
        - 2 * e * np.sin(m_rad)
        + 4 * e * y * np.sin(m_rad) * np.cos(2 * l0_rad)
        - 0.5 * y * y * np.sin(4 * l0_rad)
        - 1.25 * e * e * np.sin(2 * m_rad)
    )
    return declination, eot


def _hour_angle_deg(minutes, lon, eot):
    """Hour angle of the Sun (degrees, increasing 15 per hour)."""
    minutes_of_day = np.mod(minutes, _MINUTES_PER_DAY)
    true_solar_time = minutes_of_day + eot + 4.0 * lon
    return true_solar_time / 4.0 - 180.0


def solar_elevation(times, lon, lat):
    """Geometric elevation of the centre of the Sun, in degrees.

    ``times``, ``lon`` and ``lat`` are broadcast against each other.

    Parameters
    ----------
    times : array-like of datetime
        UTC times.
    lon, lat : float or array-like
        Observer position, degrees East and North.

    Returns
    -------
    numpy.ndarray
        Elevation in degrees (no refraction).
    """
    minutes = _minutes_since_epoch(_as_utc_ns(times))
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    decl, eot = _declination_eot_from_minutes(minutes)
    hour_angle = np.radians(_hour_angle_deg(minutes, lon, eot))
    lat_r = np.radians(lat)
    decl_r = np.radians(decl)
    sin_el = np.sin(lat_r) * np.sin(decl_r) + np.cos(lat_r) * np.cos(decl_r) * np.cos(
        hour_angle
    )
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def _wrap180(deg):
    return np.mod(deg + 180.0, 360.0) - 180.0


def _crossing_minutes(minutes_near, lon, lat, elevation_deg, sign, iterations=3):
    """Crossing nearest to ``minutes_near`` (minutes since epoch, float).

    ``sign`` is -1 for a rising (morning) and +1 for a setting (evening).
    Returns NaN where the Sun never reaches ``elevation_deg`` that day.
    """
    minutes = np.asarray(minutes_near, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    lat_r = np.radians(np.asarray(lat, dtype="float64"))
    sin_h = np.sin(np.radians(elevation_deg))

    estimate = minutes
    for _ in range(iterations):
        decl, eot = _declination_eot_from_minutes(estimate)
        decl_r = np.radians(decl)
        cos_h0 = (sin_h - np.sin(lat_r) * np.sin(decl_r)) / (
            np.cos(lat_r) * np.cos(decl_r)
        )
        with np.errstate(invalid="ignore"):
            h0 = np.degrees(np.arccos(cos_h0))
        target = sign * h0
        current = _hour_angle_deg(estimate, lon, eot)
        # Wrapping to [-180, 180) degrees of hour angle picks the solution
        # nearest to the current estimate: within 12 h on the first step, and
        # a small correction (same day) on the next ones.
        estimate = estimate + _wrap180(target - current) / 15.0 * 60.0
    return np.where(np.abs(cos_h0) <= 1.0, estimate, np.nan)


def crossing_time(t_near, lon, lat, elevation_deg, direction, window_h=None):
    """Time at which the Sun crosses ``elevation_deg``, nearest to ``t_near``.

    All array arguments are broadcast against each other, so one observed
    crossing time can be evaluated at every cell of a grid at once.

    Parameters
    ----------
    t_near : array-like of datetime
        UTC times near which the crossing is searched (for instance the
        observed crossing times).
    lon, lat : float or array-like
        Observer position, degrees East and North.
    elevation_deg : float
        Solar elevation defining the crossing, degrees.
    direction : {"rise", "set"}
        ``"rise"`` for the morning crossing (Sun going up), ``"set"`` for the
        evening crossing.
    window_h : float, optional
        If given, crossings more than ``window_h`` hours away from ``t_near``
        are discarded (``NaT``).

    Returns
    -------
    numpy.ndarray of datetime64[ns]
        Crossing times, ``NaT`` where the Sun does not reach ``elevation_deg``
        (polar day or night) or the crossing is outside ``window_h``.
    """
    sign = {"rise": -1.0, "set": 1.0}.get(direction)
    if sign is None:
        raise ValueError(f"direction must be 'rise' or 'set', got {direction!r}")
    minutes_near = _minutes_since_epoch(_as_utc_ns(t_near))
    minutes_near, lon, lat = np.broadcast_arrays(
        minutes_near, np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    )
    result = _crossing_minutes(minutes_near, lon, lat, elevation_deg, sign)
    if window_h is not None:
        result = np.where(
            np.abs(result - minutes_near) < window_h * 60.0, result, np.nan
        )
    return _from_minutes_since_epoch(result)


def solar_noon(dates, lon):
    """UTC time of the solar noon of each local solar day.

    Parameters
    ----------
    dates : array-like of datetime
        Calendar dates of the local solar day (the time of day is ignored).
    lon : float or array-like
        Longitude, degrees East.

    Returns
    -------
    numpy.ndarray of datetime64[ns]
    """
    days = _as_utc_ns(dates).astype("datetime64[D]").astype("datetime64[ns]")
    lon = np.asarray(lon, dtype="float64")
    minutes = _minutes_since_epoch(days) + 720.0 - 4.0 * lon
    for _ in range(2):
        _, eot = _declination_eot_from_minutes(minutes)
        minutes = _minutes_since_epoch(days) + 720.0 - 4.0 * lon - eot
    return _from_minutes_since_epoch(minutes)


def daily_crossings(dates, lon, lat, elevation_deg):
    """Predicted rising and setting of each local solar day.

    The day of a crossing is the local solar day, centred on the solar noon at
    ``lon``. East of about 90 degrees East the morning crossing therefore
    falls on the previous UTC date (for example 21:00 UTC for a 06:00 sunrise
    in Japan).

    Parameters
    ----------
    dates : array-like of datetime
        Calendar dates (local solar days).
    lon, lat : float
        Position, degrees East and North.
    elevation_deg : float
        Solar elevation defining the crossings, degrees.

    Returns
    -------
    rise, set : numpy.ndarray of datetime64[ns]
        ``NaT`` where the Sun does not reach ``elevation_deg`` that day.
    """
    noon = solar_noon(dates, lon)
    rise = crossing_time(noon - np.timedelta64(6, "h"), lon, lat, elevation_deg, "rise")
    sett = crossing_time(noon + np.timedelta64(6, "h"), lon, lat, elevation_deg, "set")
    # a crossing found more than 12 h from noon belongs to another day
    half_day = np.timedelta64(12, "h")
    rise = np.where(np.abs(rise - noon) < half_day, rise, np.datetime64("NaT", "ns"))
    sett = np.where(np.abs(sett - noon) < half_day, sett, np.datetime64("NaT", "ns"))
    return rise, sett
