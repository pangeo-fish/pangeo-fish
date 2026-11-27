"""Solar threshold geolocation: twilight detection, self-calibration, likelihood.

The pipeline is:

1. :func:`detect_twilight_events` finds the times at which the light record
   crosses a light threshold, keeps the crossing nearest to the predicted
   rising and setting at the release position, and pairs them by local solar
   day.
2. :func:`~pangeo_fish.light.quality.compute_quality_flags` flags each event.
3. :func:`self_calibrate_solar_threshold` finds the solar elevation at which
   the tag sees the threshold, from the first nights at the release position.
4. :func:`solar_likelihood_cells` evaluates, on every HEALPix cell, how well
   the predicted crossing times at that cell match the observed ones.
   :func:`align_light_to_time` puts the result on the time steps of the
   emission, ready to be multiplied with the other terms.

All solar geometry comes from :mod:`pangeo_fish.light.sun` (vectorised, no
refraction, centre of the Sun). :func:`compute_solar_likelihood` keeps the
older regular latitude/longitude grid output.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from pangeo_fish.light import sun
from pangeo_fish.light.quality import _tz

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _threshold_per_row(df, thresh_light, use_fixed_threshold):
    if "dyn_thresh" in df.columns:
        thresh = df["dyn_thresh"].to_numpy(dtype="float64")
    elif "moon_phase" in df.columns:
        from pangeo_fish.light.quality import dynamic_threshold

        thresh = np.asarray(
            dynamic_threshold(df["moon_phase"].to_numpy(dtype="float64")),
            dtype="float64",
        )
    else:
        thresh = np.full(len(df), float(thresh_light))
    if use_fixed_threshold:
        thresh = np.full(len(thresh), float(np.nanmax(thresh)))
    return thresh


def _nearest_within(candidates, targets, window):
    """For each target, the nearest candidate within ``window`` (else NaT).

    ``candidates`` must be sorted.
    """
    out = np.full(targets.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    if candidates.size == 0:
        return out
    valid = ~np.isnat(targets)
    pos = np.searchsorted(candidates, targets[valid])
    left = candidates[np.clip(pos - 1, 0, candidates.size - 1)]
    right = candidates[np.clip(pos, 0, candidates.size - 1)]
    d_left = np.abs(targets[valid] - left)
    d_right = np.abs(targets[valid] - right)
    best = np.where(d_left <= d_right, left, right)
    best_d = np.minimum(d_left, d_right)
    out[valid] = np.where(best_d < window, best, np.datetime64("NaT"))
    return out


def detect_twilight_events(
    df,
    release_lon,
    release_lat,
    thresh_light=60,
    window_h=2.5,
    use_fixed_threshold=False,
    detection_elevation=-10.0,
):
    """Detect and pair the rising and setting crossings of the light record.

    Parameters
    ----------
    df : pandas.DataFrame
        Light record with columns:

        * ``time`` -- UTC timestamps (naive or time-zone aware), sorted
        * ``light`` -- light values
        * ``dyn_thresh`` (optional) -- light threshold per row, from
          :func:`~pangeo_fish.light.quality.dynamic_threshold`. If absent it
          is computed from ``moon_phase`` when that column exists, otherwise
          ``thresh_light`` is used.
        * ``moon_phase`` (optional) -- illuminated fraction of the Moon, %.
    release_lon, release_lat : float
        Release position, degrees East and North. The predicted crossings at
        this position select which threshold crossings are kept.
    thresh_light : float, default 60
        Constant light threshold, used only without ``dyn_thresh`` and
        ``moon_phase``.
    window_h : float, default 2.5
        Largest distance, in hours, between a threshold crossing and the
        predicted crossing at the release position.
    use_fixed_threshold : bool, default False
        If True, use one threshold for all rows, the largest value of the
        per-row threshold (its full-moon value).
    detection_elevation : float, default -10.0
        Solar elevation, degrees, of the predicted crossings used to select the
        threshold crossings. It only needs to be close to the true one:
        ``window_h`` absorbs the difference.

    Returns
    -------
    list of (pandas.Timestamp, pandas.Timestamp)
        One ``(t_rise, t_set)`` pair (UTC) per local solar day on which both
        crossings were found. The rising is the morning crossing of the same
        local day as the setting, so east of about 90 degrees East it can fall
        on the previous UTC date.
    """
    times = sun._as_utc_ns(df["time"])
    light = df["light"].to_numpy(dtype="float64")
    thresh = _threshold_per_row(df, thresh_light, use_fixed_threshold)

    prev, cur, th = light[:-1], light[1:], thresh[1:]
    with np.errstate(invalid="ignore"):
        going_up = (prev <= th) & (th < cur)
        going_down = (prev > th) & (th >= cur)
    rise_raw = np.sort(times[1:][going_up])
    set_raw = np.sort(times[1:][going_down])

    if times.size == 0:
        return []
    first = times.min().astype("datetime64[D]") - np.timedelta64(1, "D")
    last = times.max().astype("datetime64[D]") + np.timedelta64(1, "D")
    days = np.arange(first, last + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    rise_pred, set_pred = sun.daily_crossings(
        days, release_lon, release_lat, detection_elevation
    )

    window = np.timedelta64(int(round(window_h * 3600e9)), "ns")
    rises = _nearest_within(rise_raw, rise_pred, window)
    sets = _nearest_within(set_raw, set_pred, window)

    keep = ~np.isnat(rises) & ~np.isnat(sets) & (sets > rises)
    keep &= (sets - rises) < np.timedelta64(24, "h")
    return [
        (pd.Timestamp(r).tz_localize("UTC"), pd.Timestamp(s).tz_localize("UTC"))
        for r, s in zip(rises[keep], sets[keep])
    ]


def pairs_from_lightloc(lightloc_df, max_depth_dbar=80):
    """Build twilight pairs and quality flags from a standardised LightLoc DataFrame.

    Used when only the crossing times transmitted by a Wildlife Computers tag
    are available (tags that were not recovered). Replaces
    :func:`detect_twilight_events` and
    :func:`~pangeo_fish.light.quality.compute_quality_flags`.

    Parameters
    ----------
    lightloc_df : pandas.DataFrame
        Table written by :func:`~pangeo_fish.light.ingest.ingest_lightloc`, with
        columns ``time`` (UTC), ``type`` (``"rise"`` or ``"set"``) and
        ``max_depth``.
    max_depth_dbar : float, default 80
        Events with ``max_depth > max_depth_dbar`` are flagged invalid.

    Returns
    -------
    pairs : list of (pandas.Timestamp, pandas.Timestamp)
        ``(t_rise, t_set)`` pairs, one per UTC date with both events.
    qdf : pandas.DataFrame
        Same columns as
        :func:`~pangeo_fish.light.quality.compute_quality_flags`.
    """
    df = lightloc_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date

    rises = df[df["type"] == "rise"].set_index("date")
    sets = df[df["type"] == "set"].set_index("date")

    pairs = []
    records = []
    night_idx = 0

    all_dates = sorted(set(rises.index) | set(sets.index))
    for date in all_dates:
        has_rise = date in rises.index
        has_set = date in sets.index
        if not (has_rise and has_set):
            continue

        t_rise = _tz(rises.loc[date, "time"])
        t_set = _tz(sets.loc[date, "time"])
        dp_rise = float(rises.loc[date, "max_depth"])
        dp_set = float(sets.loc[date, "max_depth"])

        v_r = dp_rise <= max_depth_dbar
        v_s = dp_set <= max_depth_dbar
        flag = "good" if (v_r and v_s) else ("partial" if (v_r or v_s) else "bad")

        pairs.append((t_rise, t_set))
        records.append(
            {
                "night": night_idx,
                "date": pd.Timestamp(date),
                "flag": flag,
                "valid_rise": v_r,
                "valid_set": v_s,
                "dl_rise": 0.0,
                "dl_set": 0.0,
                "dp_rise": dp_rise,
                "dp_set": dp_set,
            }
        )
        night_idx += 1

    return pairs, pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Self-calibration
# ---------------------------------------------------------------------------


class CalibResult(NamedTuple):
    """Return type of :func:`self_calibrate_solar_threshold`.

    Attributes
    ----------
    thresh_deg : float
        Median solar elevation, degrees, at the observed crossings of the
        calibration nights: the elevation to pass to the likelihood.
    lon_self : float
        Longitude, degrees East, estimated from the timing of noon.
    elevations : numpy.ndarray
        Elevation at each rising, then each setting, of the calibration
        nights.
    thresh_deg_end : float or None
        Same median over the last nights, at the recapture position.
        ``None`` without a recapture position.
    elevations_end : numpy.ndarray or None
        Elevations of the end nights.
    delta_deg : float or None
        ``thresh_deg_end - thresh_deg``.
    """

    thresh_deg: float
    lon_self: float
    elevations: np.ndarray
    thresh_deg_end: float | None = None
    elevations_end: np.ndarray | None = None
    delta_deg: float | None = None


def _pair_times(pairs):
    rises = sun._as_utc_ns(pd.DatetimeIndex([_tz(r) for r, _ in pairs]))
    sets = sun._as_utc_ns(pd.DatetimeIndex([_tz(s) for _, s in pairs]))
    return rises, sets


def _elevations_at(pairs, lon, lat):
    if len(pairs) == 0:
        return np.array([], dtype="float64")
    rises, sets = _pair_times(pairs)
    return np.concatenate(
        [sun.solar_elevation(rises, lon, lat), sun.solar_elevation(sets, lon, lat)]
    )


def self_calibrate_solar_threshold(
    pairs,
    df=None,
    n_nights=20,
    release_lon=None,
    release_lat=None,
    recapture_lon=None,
    recapture_lat=None,
    n_nights_end=None,
):
    """Solar elevation at which the tag sees its light threshold.

    The solar elevation is computed at each crossing of the first
    ``n_nights`` pairs, at the release position; the median is the
    calibrated elevation. With a recapture position, the same is done on the
    last ``n_nights_end`` pairs as a check; a small ``delta_deg`` (below
    1--2 degrees) means the calibration holds over the deployment.

    Parameters
    ----------
    pairs : list of (pandas.Timestamp, pandas.Timestamp)
        ``(t_rise, t_set)`` pairs from :func:`detect_twilight_events`.
    df : pandas.DataFrame, optional
        Not used. Kept so that older calls still work.
    n_nights : int, default 20
        Number of pairs, from the start, used for the calibration.
    release_lon : float, optional
        Release longitude, degrees East. Without it the noon-timing longitude
        ``lon_self`` is used.
    release_lat : float
        Release latitude, degrees North.
    recapture_lon, recapture_lat : float, optional
        Recapture position, for the end-of-deployment check.
    n_nights_end : int, optional
        Number of pairs, from the end, for the check. Defaults to
        ``n_nights``.

    Returns
    -------
    CalibResult

    Notes
    -----
    ``lon_self = (12 - noon_utc) * 15`` where ``noon_utc`` is the UTC hour of
    the midpoint between rising and setting, averaged as a median over the
    calibration nights.
    """
    if release_lat is None:
        raise ValueError("release_lat is required")
    calib_pairs = list(pairs[:n_nights])
    if not calib_pairs:
        raise ValueError("no twilight pair to calibrate on")

    rises, sets = _pair_times(calib_pairs)
    midpoints = rises + (sets - rises) // 2
    noon_utc_hours = (midpoints - midpoints.astype("datetime64[D]")) / np.timedelta64(
        1, "h"
    )
    lon_self = float(np.median((12.0 - noon_utc_hours) * 15.0))

    # The release longitude is more reliable than lon_self for the first
    # nights, when the fish is still near the release site: lon_self comes
    # from noisy noon timing and would make the threshold depend on itself.
    lon_eval = release_lon if release_lon is not None else lon_self
    elevations = _elevations_at(calib_pairs, lon_eval, release_lat)
    thresh_deg = float(np.median(elevations))

    thresh_deg_end = elevations_end = delta_deg = None
    if recapture_lon is not None and recapture_lat is not None:
        n_end = n_nights_end if n_nights_end is not None else n_nights
        if n_end and len(pairs) > n_end:
            elevations_end = _elevations_at(
                list(pairs[-n_end:]), recapture_lon, recapture_lat
            )
            thresh_deg_end = float(np.median(elevations_end))
            delta_deg = thresh_deg_end - thresh_deg

    return CalibResult(
        thresh_deg=thresh_deg,
        lon_self=lon_self,
        elevations=elevations,
        thresh_deg_end=thresh_deg_end,
        elevations_end=elevations_end,
        delta_deg=delta_deg,
    )


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def _event_log_likelihood(t_obs, direction, lon, lat, thresh_deg, sigma, window_h):
    """Log-likelihood of one observed crossing at positions ``lon``, ``lat``.

    Returns ``(log_lh, ok)``: ``-dt**2 / (2 sigma**2)`` with ``dt`` in minutes
    between the observed crossing and the one predicted at each position, and
    ``ok`` where a prediction exists within ``window_h`` hours.
    """
    t_obs = sun._as_utc_ns(_tz(t_obs))
    predicted = sun.crossing_time(
        t_obs, lon, lat, thresh_deg, direction, window_h=window_h
    )
    ok = ~np.isnat(predicted)
    dt = (t_obs - predicted).astype("timedelta64[ns]").astype("int64") / 60e9
    return np.where(ok, -(dt**2) / (2.0 * sigma**2), 0.0), ok


def _finish(lh, lh_floor):
    lh = np.where(np.isfinite(lh), lh, lh_floor)
    mx = lh.max() if lh.size else 0.0
    if mx > 0:
        lh = lh / mx
    return lh


def _night_likelihood(
    t_rise,
    t_set,
    valid_rise,
    valid_set,
    flag,
    lon,
    lat,
    thresh_deg,
    sigma,
    window_h,
    lh_floor,
):
    """Timing likelihood of one night at positions ``lon``, ``lat`` (1-D arrays).

    - ``flag == "bad"``: 1 everywhere (no information).
    - Each valid event contributes ``exp(-dt**2 / (2 sigma**2))``, ``dt`` in
      minutes between the observed and the predicted crossing.
    - A position where no valid event has a predicted crossing within
      ``window_h`` gets ``lh_floor``. On a ``"good"`` night both predictions
      are required: a position with only one of them gets ``lh_floor`` (a
      single event would otherwise give ghost maxima near the poles).
    - Non-finite values become ``lh_floor``. The result is divided by its
      maximum.
    """
    shape = np.shape(lon)
    if flag == "bad":
        return np.ones(shape)

    log_lh = np.zeros(shape)
    found = {}
    for valid, t_obs, direction in (
        (valid_rise, t_rise, "rise"),
        (valid_set, t_set, "set"),
    ):
        if not valid:
            continue
        event_log_lh, ok = _event_log_likelihood(
            t_obs, direction, lon, lat, thresh_deg, sigma, window_h
        )
        log_lh = log_lh + event_log_lh
        found[direction] = ok

    if not found:
        lh = np.full(shape, lh_floor)
    else:
        lh = np.exp(log_lh)
        any_found = np.logical_or.reduce(list(found.values()))
        lh = np.where(any_found, lh, lh_floor)
        if flag == "good":
            both = found.get("rise", False) & found.get("set", False)
            lh = np.where(both, lh, lh_floor)
    return _finish(lh, lh_floor)


def _event_likelihood(
    t_obs, direction, lon, lat, thresh_deg, sigma, window_h, lh_floor
):
    """Timing likelihood of one crossing; ``lh_floor`` where it is not predicted."""
    log_lh, ok = _event_log_likelihood(
        t_obs, direction, lon, lat, thresh_deg, sigma, window_h
    )
    return _finish(np.where(ok, np.exp(log_lh), lh_floor), lh_floor)


def _night_rows(pairs, qdf):
    if len(pairs) != len(qdf):
        raise ValueError(
            f"pairs ({len(pairs)}) and qdf ({len(qdf)}) must have the same length"
        )
    flags = qdf["flag"].to_numpy()
    valid_rise = qdf["valid_rise"].to_numpy(dtype=bool)
    valid_set = qdf["valid_set"].to_numpy(dtype=bool)
    return zip(pairs, flags, valid_rise, valid_set)


def _as_ns(times):
    if not times:
        return np.array([], "datetime64[ns]")
    return sun._as_utc_ns(pd.DatetimeIndex([_tz(t) for t in times]))


def light_smoothing_operator(cell_ids, sigma_km, truncate=4.0):
    """Gaussian smoothing operator on the cells of a HEALPix grid.

    Built with :func:`healpix_analyse.gaussian_filter_operator`
    (``normalization="present"``, ``width_rule="matched"``): a constant
    field stays constant, also at the edge of the domain, and the width of
    the response is ``sigma_km`` even below one cell. Distances are measured
    on WGS84, whatever the ellipsoid of the grid; at the scale of the
    smoothing (tens of kilometres) the difference is negligible.

    Parameters
    ----------
    cell_ids : xarray.DataArray
        Nested HEALPix cell ids with the attribute ``level``.
    sigma_km : float
        Standard deviation of the Gaussian, kilometres.
    truncate : float, default 4.0
        Support radius in units of ``sigma_km``.

    Returns
    -------
    tuple
        The operator, for :func:`healpix_analyse.apply_filter_operator`.
    """
    from healpix_analyse import gaussian_filter_operator

    attrs = getattr(cell_ids, "attrs", {})
    if attrs.get("indexing_scheme", "nested") != "nested":
        raise ValueError("light smoothing needs nested HEALPix cell ids")
    if "level" not in attrs:
        raise ValueError("cell_ids must carry the HEALPix level in its attributes")
    return gaussian_filter_operator(
        np.asarray(cell_ids),
        int(attrs["level"]),
        sigma_m=float(sigma_km) * 1000.0,
        truncate=float(truncate),
        normalization="present",
        width_rule="matched",
        ellipsoid="WGS84",
    )


def solar_likelihood_cells(
    pairs,
    qdf,
    cell_ids,
    thresh_deg,
    sigma_timing=10.0,
    window_h=6.0,
    lh_floor=1e-30,
    by="event",
    smooth_sigma_km=None,
    smooth_truncate=4.0,
):
    """Light likelihood on the cells of a HEALPix grid, per crossing or per night.

    For an observed crossing ``e`` (rising or setting) and a cell, the
    likelihood is

    .. math::

        L_e = \\exp\\left(-\\frac{\\Delta t_e^2}{2\\sigma^2}\\right)

    where :math:`\\Delta t_e` is the difference, in minutes, between the
    observed crossing and the crossing of ``thresh_deg`` predicted at the
    cell centre. A cell where no crossing is predicted within ``window_h``
    hours (polar day or night, far away) gets ``lh_floor``.

    With ``by="event"`` (default) every valid crossing is a map of its own,
    stamped with the observed time of the crossing. No position is needed to
    decide to which time step a map belongs: :func:`align_light_to_time` puts
    it in the step that contains the observation, as for the temperature and
    pressure records. A rising and a setting that fall in the same step are
    multiplied there, which gives the nightly likelihood.

    With ``by="night"`` the valid crossings of a night are multiplied into one
    map stamped with the midpoint between rising and setting; see
    :func:`_night_likelihood` for the special cases.

    Nights flagged ``"bad"`` give no map with ``by="event"`` and a map of ones
    with ``by="night"``. Every map is divided by its maximum, so it is a
    relative weight over the cells.

    Parameters
    ----------
    pairs : list of (pandas.Timestamp, pandas.Timestamp)
        ``(t_rise, t_set)`` pairs, in the order of the rows of ``qdf``.
    qdf : pandas.DataFrame
        Quality flags from
        :func:`~pangeo_fish.light.quality.compute_quality_flags`, with
        columns ``flag``, ``valid_rise`` and ``valid_set``.
    cell_ids : xarray.DataArray
        Cell ids on the dimension ``cells``, with the attributes
        ``grid_name``, ``level``, ``indexing_scheme`` and ``ellipsoid``, as
        in the emission dataset (``emission["cell_ids"]``). The cell centres
        are computed on that ellipsoid.
    thresh_deg : float
        Solar elevation of the crossings, degrees, from
        :func:`self_calibrate_solar_threshold`.
    sigma_timing : float, default 10.0
        Timing uncertainty, minutes.
    window_h : float, default 6.0
        A predicted crossing more than ``window_h`` hours from the observed
        one is treated as missing.
    lh_floor : float, default 1e-30
        Smallest value; it also replaces missing predictions.
    by : {"event", "night"}, default "event"
        One map per valid crossing, or one map per night.
    smooth_sigma_km : float, optional
        If given, every map is smoothed with a Gaussian of this standard
        deviation (kilometres) on the HEALPix grid
        (:func:`light_smoothing_operator`), then divided by its maximum
        again.
    smooth_truncate : float, default 4.0
        Support radius of the smoothing, in units of ``smooth_sigma_km``.

    Returns
    -------
    xarray.DataArray
        ``pdf_light`` with dimensions ``("time", "cells")``.

        - ``by="event"``: ``time`` is the observed crossing, with the
          coordinates ``direction`` (``"rise"`` or ``"set"``), ``night``
          (row of ``qdf``) and ``flag`` (flag of the night).
        - ``by="night"``: ``time`` is the midpoint between rising and setting
          (local noon, UTC), with the coordinates ``rise_time``, ``set_time``
          and ``flag``.

        ``cell_ids`` is copied from the input.
    """
    from pangeo_fish.tide_healpix import cell_centers

    if by not in ("event", "night"):
        raise ValueError(f"by must be 'event' or 'night', got {by!r}")

    lon, lat = cell_centers(cell_ids)
    sigma = float(sigma_timing)

    rows = []
    coords = {}
    if by == "night":
        rise_times, set_times, flags = [], [], []
        for (t_rise, t_set), flag, v_r, v_s in _night_rows(pairs, qdf):
            rows.append(
                _night_likelihood(
                    t_rise,
                    t_set,
                    v_r,
                    v_s,
                    flag,
                    lon,
                    lat,
                    thresh_deg,
                    sigma,
                    window_h,
                    lh_floor,
                )
            )
            rise_times.append(t_rise)
            set_times.append(t_set)
            flags.append(flag)
        rise_ns, set_ns = _as_ns(rise_times), _as_ns(set_times)
        time = rise_ns + (set_ns - rise_ns) // 2
        coords = {
            "rise_time": ("time", rise_ns),
            "set_time": ("time", set_ns),
            "flag": ("time", np.asarray(flags, dtype=object).astype(str)),
        }
    else:
        times, directions, nights, flags = [], [], [], []
        for night, ((t_rise, t_set), flag, v_r, v_s) in enumerate(
            _night_rows(pairs, qdf)
        ):
            if flag == "bad":
                continue
            for valid, t_obs, direction in ((v_r, t_rise, "rise"), (v_s, t_set, "set")):
                if not valid:
                    continue
                rows.append(
                    _event_likelihood(
                        t_obs,
                        direction,
                        lon,
                        lat,
                        thresh_deg,
                        sigma,
                        window_h,
                        lh_floor,
                    )
                )
                times.append(t_obs)
                directions.append(direction)
                nights.append(night)
                flags.append(flag)
        time = _as_ns(times)
        order = np.argsort(time, kind="stable")
        time = time[order]
        rows = [rows[i] for i in order]
        coords = {
            "direction": (
                "time",
                np.asarray(directions, dtype=object)[order].astype(str),
            ),
            "night": ("time", np.asarray(nights, dtype="int64")[order]),
            "flag": ("time", np.asarray(flags, dtype=object)[order].astype(str)),
        }

    data = np.stack(rows) if rows else np.empty((0, lon.size))
    if smooth_sigma_km is not None and data.shape[0] > 0:
        from healpix_analyse import apply_filter_operator

        operator = light_smoothing_operator(cell_ids, smooth_sigma_km, smooth_truncate)
        data = np.stack(
            [_finish(row, lh_floor) for row in apply_filter_operator(data, *operator)]
        )
        data = np.maximum(data, lh_floor)

    return xr.DataArray(
        data,
        dims=("time", "cells"),
        coords={
            "time": time,
            **coords,
            "cell_ids": (
                cell_ids.variable
                if hasattr(cell_ids, "variable")
                else ("cells", np.asarray(cell_ids))
            ),
        },
        name="pdf_light",
        attrs={
            "method": "solar_threshold",
            "by": by,
            "sigma_timing_min": sigma,
            "thresh_deg": float(thresh_deg),
            "window_h": float(window_h),
            "smooth_sigma_km": (
                float(smooth_sigma_km) if smooth_sigma_km is not None else "none"
            ),
        },
    )


def _time_bins(time):
    """Bins of the time coordinate, built as in :func:`pangeo_fish.helpers.compute_diff`."""
    import cf_xarray  # noqa: F401

    from pangeo_fish.cf import bounds_to_bins

    ds = xr.Dataset(coords={"time": ("time", np.asarray(time))})
    ds["time"].attrs["standard_name"] = "time"
    return (
        ds.cf.add_bounds(["time"], output_dim="bounds")
        .pipe(bounds_to_bins, bounds_dim="bounds")
        .get("time_bins")
    )


def _interval_index(time, values):
    """Step ``i`` covers ``[time[i], time[i+1])``; the last one has the length of the one before."""
    if time.size == 0:
        return np.full(values.shape, -1)
    if time.size > 1:
        end = time[-1] + (time[-1] - time[-2])
    else:
        end = time[-1] + np.timedelta64(1, "D")
    index = np.searchsorted(time, values, side="right") - 1
    outside = (values < time[0]) | (values >= end) | np.isnat(values)
    return np.where(outside, -1, index)


def align_light_to_time(pdf_light, time, how="interval", fill_value=1.0):
    """Put the light likelihood on the time steps of the emission.

    Parameters
    ----------
    pdf_light : xarray.DataArray
        Output of :func:`solar_likelihood_cells` (dimensions ``time``,
        ``cells``).
    time : array-like or xarray.DataArray
        Time coordinate of the emission (sorted).
    how : {"interval", "bins", "date", "rise_date"}, default "interval"
        How a map finds its time step, from its ``time`` (the observed
        crossing with ``by="event"``).

        - ``"interval"``: step ``i`` covers ``[time[i], time[i+1])``. This is
          the step that holds the records of the same period when the tag log
          is floored to the step before it is binned, as in the daily
          notebooks (``tag_log["time"].dt.floor("D")``).
        - ``"bins"``: the bins built from the bounds of ``time`` as in
          :func:`pangeo_fish.helpers.compute_diff`, centred on the labels.
          Use it when the tag log is binned without flooring.
        - ``"date"`` and ``"rise_date"`` (for ``by="night"`` maps only): the
          step labelled 00:00 of the UTC date of the midpoint, or of the
          rising. Both depend on where the night was observed (the local
          noon), unlike the two options above.
    fill_value : float, default 1.0
        Value of the time steps without any map. Any constant is neutral:
        the product of the terms is normalised over the cells.

    Returns
    -------
    xarray.DataArray
        ``pdf_light`` on ``("time", "cells")`` with the given time coordinate.
        Maps that fall in the same time step are multiplied, then divided by
        the maximum. Maps outside every time step are dropped. ``n_light``
        counts the maps of each time step.
    """
    time = np.asarray(getattr(time, "values", time), dtype="datetime64[ns]")
    stamps = pdf_light["time"].values.astype("datetime64[ns]")
    if how == "interval":
        index = _interval_index(time, stamps)
    elif how == "bins":
        bins = pd.IntervalIndex(_time_bins(time).values)
        index = bins.get_indexer(stamps)
    elif how in ("date", "rise_date"):
        if "rise_time" not in pdf_light.coords:
            raise ValueError(f"how={how!r} needs the nightly maps of by='night'")
        source = "time" if how == "date" else "rise_time"
        labels = (
            pdf_light[source].values.astype("datetime64[D]").astype("datetime64[ns]")
        )
        index = pd.Index(time).get_indexer(labels)
    else:
        raise ValueError(
            f"how must be 'interval', 'bins', 'date' or 'rise_date', got {how!r}"
        )

    n_cells = pdf_light.sizes["cells"]
    product = np.ones((time.size, n_cells))
    counts = np.zeros(time.size, dtype="int64")
    values = pdf_light.values
    for row, target in enumerate(index):
        if target < 0:
            continue
        product[target] *= values[row]
        counts[target] += 1

    out = np.where(counts[:, None] > 0, product, fill_value)
    # several maps in one step: divide by the maximum again
    for step in np.flatnonzero(counts > 1):
        mx = out[step].max()
        if mx > 0:
            out[step] /= mx

    return xr.DataArray(
        out,
        dims=("time", "cells"),
        coords={
            "time": time,
            "cell_ids": pdf_light["cell_ids"].variable,
            "n_light": ("time", counts),
        },
        name="pdf_light",
        attrs=dict(pdf_light.attrs, alignment=how),
    )


def compute_solar_likelihood(
    pairs,
    qdf,
    lons,
    lats,
    thresh_deg,
    sigma_timing=18,
    smooth_sigma=1.0,
    lh_floor=1e-30,
    window_h=6.0,
):
    """Light likelihood of each night on a regular latitude/longitude grid.

    Same likelihood as :func:`solar_likelihood_cells`, evaluated at the nodes
    of a regular grid, then smoothed with a Gaussian filter of
    ``smooth_sigma`` grid steps and divided by its maximum. This is the output
    used by the notebooks of pull request 267, which then regrid it to
    HEALPix; new code should use :func:`solar_likelihood_cells`, which needs
    neither the intermediate grid nor the regridding.

    Parameters
    ----------
    pairs : list of (pandas.Timestamp, pandas.Timestamp)
        ``(t_rise, t_set)`` pairs, in the order of the rows of ``qdf``.
    qdf : pandas.DataFrame
        Quality flags with columns ``flag``, ``valid_rise``, ``valid_set``.
    lons, lats : array-like
        1-D grid axes, degrees East and North.
    thresh_deg : float
        Solar elevation of the crossings, degrees.
    sigma_timing : float, default 18
        Timing uncertainty, minutes.
    smooth_sigma : float, default 1.0
        Standard deviation, in grid steps, of the spatial Gaussian filter.
    lh_floor : float, default 1e-30
        Smallest value.
    window_h : float, default 6.0
        A predicted crossing more than ``window_h`` hours from the observed
        one is treated as missing.

    Returns
    -------
    xarray.DataArray
        ``pdf_light`` with dimensions ``("time", "latitude", "longitude")``.
        ``time`` is the UTC date of the rising.
    """
    lons = np.asarray(lons, dtype="float64")
    lats = np.asarray(lats, dtype="float64")
    lon2d, lat2d = np.meshgrid(lons, lats)
    shape = lon2d.shape

    maps = []
    night_times = []
    for (t_rise, t_set), flag, v_r, v_s in _night_rows(pairs, qdf):
        night_times.append(pd.Timestamp(_tz(t_rise).date()))
        lh = _night_likelihood(
            t_rise,
            t_set,
            v_r,
            v_s,
            flag,
            lon2d.ravel(),
            lat2d.ravel(),
            thresh_deg,
            float(sigma_timing),
            window_h,
            lh_floor,
        ).reshape(shape)
        if flag != "bad":
            lh = gaussian_filter(lh, sigma=smooth_sigma)
            mx = lh.max()
            if mx > 0:
                lh = lh / mx
        maps.append(lh)

    data = np.stack(maps) if maps else np.empty((0, *shape))
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(night_times, dtype="datetime64[ns]"),
            "latitude": lats,
            "longitude": lons,
        },
        name="pdf_light",
        attrs={"method": "solar_threshold", "sigma_timing_min": sigma_timing},
    )
