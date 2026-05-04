"""Solar threshold geolocation method (Part A of the light pipeline).

Detects twilight events (RISE / SET) by threshold-crossing, self-calibrates
the solar elevation threshold from the first nights of deployment, evaluates
event quality, and computes per-night solar likelihood maps on a regular
lat/lon grid.
"""

from typing import NamedTuple

import ephem
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from pangeo_fish.light.quality import _tz

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_astro_prediction(date, lon, lat, horizon_deg=-10.0):
    """Return predicted astronomical RISE and SET for *date* at *(lon, lat)*.

    Parameters
    ----------
    date : pd.Timestamp or datetime.date
        Calendar date.
    lon : float
        Observer longitude (degrees East).
    lat : float
        Observer latitude (degrees North).
    horizon_deg : float, default -10.0
        Solar elevation defining "rise" and "set" (degrees).

    Returns
    -------
    rise : pd.Timestamp or None
    sett : pd.Timestamp or None
        UTC timestamps of the predicted rise and set.  Both are ``None``
        if :mod:`ephem` cannot find a crossing (e.g., polar day/night).
    """
    obs = ephem.Observer()
    obs.lon = str(lon)
    obs.lat = str(lat)
    obs.pressure = 0
    obs.horizon = str(horizon_deg)
    sun = ephem.Sun()
    try:
        obs.date = f"{date.year}/{date.month:02d}/{date.day:02d} 00:00:00"
        rise = pd.Timestamp(
            ephem.Date(obs.next_rising(sun, use_center=True)).datetime()
        ).tz_localize("UTC")
        obs.date = f"{date.year}/{date.month:02d}/{date.day:02d} 10:00:00"
        sett = pd.Timestamp(
            ephem.Date(obs.next_setting(sun, use_center=True)).datetime()
        ).tz_localize("UTC")
        return rise, sett
    except Exception:
        return None, None


def _get_crossing_fast(t_approx, lon, lat, horizon_deg, direction="up", window_h=6.0):
    """Return the predicted solar crossing near *t_approx* for a grid pixel.

    Parameters
    ----------
    t_approx : pd.Timestamp
        Approximate observed crossing time (UTC).
    lon : float
        Observer longitude (degrees East).
    lat : float
        Observer latitude (degrees North).
    horizon_deg : float
        Solar elevation defining the crossing (degrees).
    direction : {'up', 'down'}, default 'up'
        ``'up'`` for RISE, ``'down'`` for SET.
    window_h : float, default 6.0
        Search half-window (hours).

    Returns
    -------
    pd.Timestamp or None
        Predicted crossing time (UTC), or ``None`` if not found within
        ``window_h`` of ``t_approx``.

    Notes
    -----
    **Why ``window_h=6.0`` (was 3.0):** at high-latitude grid pixels in
    summer, the predicted crossing can shift by several hours relative to
    the observed time at mid-latitudes.  The old 3-hour window caused
    silent ``None`` returns → single-event fallback → ghost likelihood
    peaks.  6 hours is wide enough to capture any real crossing while
    remaining tighter than a full 12-hour day/night cycle.
    """
    obs = ephem.Observer()
    obs.lon = str(lon)
    obs.lat = str(lat)
    obs.pressure = 0
    obs.horizon = str(horizon_deg)
    sun = ephem.Sun()
    t_start = _tz(t_approx) - pd.Timedelta(hours=window_h)
    obs.date = t_start.strftime("%Y/%m/%d %H:%M:%S")
    try:
        if direction == "up":
            t_cross = obs.next_rising(sun, use_center=True)
        else:
            t_cross = obs.next_setting(sun, use_center=True)
        t_result = pd.Timestamp(ephem.Date(t_cross).datetime()).tz_localize("UTC")
        if abs((t_result - _tz(t_approx)).total_seconds()) < window_h * 3600:
            return t_result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_twilight_events(
    df,
    release_lon,
    release_lat,
    thresh_light=60,
    window_h=2.5,
    use_fixed_threshold=False,
):
    """Detect and pair RISE / SET twilight events in the tag light record.

    Parameters
    ----------
    df : pd.DataFrame
        Tag data resampled to 1-minute resolution with columns:

        * ``time`` — UTC timestamps
        * ``light`` — light counts
        * ``dyn_thresh`` (optional) — pre-computed dynamic threshold from
          :func:`~pangeo_fish.light.quality.dynamic_threshold`; if absent,
          computed on-the-fly from ``moon_phase`` if that column exists,
          otherwise falls back to the static ``thresh_light``.
        * ``moon_phase`` (optional) — moon illuminated fraction (%).
    release_lon : float
        Release-site longitude (degrees East), used to compute
        astronomical predictions.
    release_lat : float
        Release-site latitude (degrees North).
    thresh_light : float, default 60
        Static light threshold (counts), used only if ``df`` has neither
        ``dyn_thresh`` nor ``moon_phase`` columns.
    window_h : float, default 2.5
        Maximum allowed offset (hours) between a raw crossing and the
        astronomical prediction at the release position.
    use_fixed_threshold : bool, default False
        If ``True``, replace the per-row dynamic threshold with a single
        constant equal to its maximum observed value (i.e. the threshold
        value at full moon).  This is more conservative — it avoids
        misclassifying moonlit nights as daytime — at the cost of slightly
        fewer detections on faint-moon nights.

    Returns
    -------
    list of (pd.Timestamp, pd.Timestamp)
        Paired (t_rise, t_set) tuples in UTC for each detected
        astronomical night.
    """
    light_arr = df["light"].values
    times_arr = df["time"].values

    # Resolve per-row threshold
    if "dyn_thresh" in df.columns:
        thresh_arr = df["dyn_thresh"].values
    elif "moon_phase" in df.columns:
        from pangeo_fish.light.quality import dynamic_threshold

        thresh_arr = dynamic_threshold(df["moon_phase"].values)
    else:
        thresh_arr = np.full(len(df), thresh_light, dtype=float)

    # Option: collapse to the maximum value (full-moon level) for all rows.
    # More conservative — prevents misclassifying moonlit nights as day —
    # at the cost of slightly fewer detections on new-moon nights.
    if use_fixed_threshold:
        thresh_arr = np.full(len(thresh_arr), float(np.max(thresh_arr)))

    # Raw threshold crossings
    rise_raw, set_raw = [], []
    for i in range(1, len(light_arr)):
        if light_arr[i - 1] <= thresh_arr[i] < light_arr[i]:
            rise_raw.append(pd.Timestamp(times_arr[i]))
        elif light_arr[i - 1] > thresh_arr[i] >= light_arr[i]:
            set_raw.append(pd.Timestamp(times_arr[i]))

    # Keep only the crossing closest to the astronomical prediction
    window = pd.Timedelta(hours=window_h)
    all_dates = pd.date_range(
        df["time"].min().date(), df["time"].max().date(), freq="D"
    )
    rise_events, set_events = [], []

    for date in all_dates:
        rise_pred, set_pred = _get_astro_prediction(date, release_lon, release_lat)
        if rise_pred is None:
            continue
        cands_r = [
            t
            for t in rise_raw
            if abs((_tz(t) - rise_pred).total_seconds()) < window.total_seconds()
        ]
        cands_s = [
            t
            for t in set_raw
            if abs((_tz(t) - set_pred).total_seconds()) < window.total_seconds()
        ]
        if cands_r:
            rise_events.append(
                min(cands_r, key=lambda t: abs((_tz(t) - rise_pred).total_seconds()))
            )
        if cands_s:
            set_events.append(
                min(cands_s, key=lambda t: abs((_tz(t) - set_pred).total_seconds()))
            )

    # Pair each RISE with the nearest next SET (same astronomical night, < 24 h)
    pairs = []
    for t_rise in rise_events:
        t_rise = _tz(t_rise)
        next_sets = [
            _tz(t)
            for t in set_events
            if _tz(t) > t_rise and (_tz(t) - t_rise).total_seconds() < 86400
        ]
        if next_sets:
            # Prefer ~55 000 s ≈ 15 h gap (typical day length at mid-latitudes)
            t_set = min(
                next_sets,
                key=lambda t: abs((t - t_rise).total_seconds() - 55000),
            )
            pairs.append((t_rise, t_set))

    return pairs


class CalibResult(NamedTuple):
    """Return type of :func:`self_calibrate_solar_threshold`.

    Attributes
    ----------
    thresh_deg : float
        Median solar elevation (degrees) at observed crossings — the
        calibrated threshold to pass to :func:`compute_solar_likelihood`.
    lon_self : float
        Self-calibrated longitude (degrees East) derived from noon UTC.
    elevations : np.ndarray
        Per-event solar elevation at each RISE and SET used for
        calibration (start period).
    thresh_deg_end : float or None
        Median solar elevation at the recapture location over the last
        ``n_nights_end`` pairs.  ``None`` if no recapture position was
        provided.
    elevations_end : np.ndarray or None
        Per-event elevations at recapture location (end period).
    delta_deg : float or None
        ``thresh_deg_end − thresh_deg``.  Small |delta| (< 1–2°)
        validates the calibration; large delta may indicate tag drift
        or strong spatial variability.
    """

    thresh_deg: float
    lon_self: float
    elevations: np.ndarray
    thresh_deg_end: float | None = None
    elevations_end: np.ndarray | None = None
    delta_deg: float | None = None


def self_calibrate_solar_threshold(
    pairs,
    df,
    n_nights=20,
    release_lon=None,
    release_lat=None,
    recapture_lon=None,
    recapture_lat=None,
    n_nights_end=None,
):
    """Self-calibrate the solar elevation threshold from the first nights.

    Uses the first ``n_nights`` paired events to derive a longitude
    estimate from local noon timing, then computes the actual solar
    elevation at each RISE and SET event at that longitude.  The median
    elevation is the calibrated threshold.

    Optionally verifies the calibration against the last ``n_nights_end``
    events evaluated at the known recapture position.  A small
    ``|delta_deg|`` (< 1–2°) confirms that the calibration is consistent
    across the deployment.

    Parameters
    ----------
    pairs : list of (pd.Timestamp, pd.Timestamp)
        Paired (t_rise, t_set) events, as returned by
        :func:`detect_twilight_events`.
    df : pd.DataFrame
        Tag data (not used directly; kept for API symmetry).
    n_nights : int, default 20
        Number of nights to use for calibration (from the start).
    release_lon : float or None
        Release-site longitude (degrees East).  If ``None``, the
        self-calibrated longitude is used for computing elevations.
    release_lat : float
        Release-site latitude (degrees North).  Required.
    recapture_lon : float or None, default None
        Recapture longitude (degrees East).  If provided together with
        ``recapture_lat``, enables end-period verification.
    recapture_lat : float or None, default None
        Recapture latitude (degrees North).
    n_nights_end : int or None, default None
        Number of nights before recapture to use for verification.
        Defaults to ``n_nights`` if not provided.

    Returns
    -------
    CalibResult
        Named tuple with ``thresh_deg``, ``lon_self``, ``elevations``,
        and (if recapture provided) ``thresh_deg_end``,
        ``elevations_end``, ``delta_deg``.

    Notes
    -----
    Self-calibrated longitude: ``(12 - noon_utc) * 15`` where
    ``noon_utc`` is the UTC hour of the midpoint between t_rise and
    t_set.
    """
    calib_pairs = pairs[:n_nights]

    # Step 1: estimate longitude from noon UTC
    noon_lons = []
    for t_rise, t_set in calib_pairs:
        day_h = (_tz(t_set) - _tz(t_rise)).total_seconds() / 3600
        noon_utc = _tz(t_rise).hour + _tz(t_rise).minute / 60 + day_h / 2
        if noon_utc >= 24:
            noon_utc -= 24
        noon_lons.append((12 - noon_utc) * 15)

    lon_self = float(np.median(noon_lons))
    lat_eval = release_lat

    # Step 2: solar elevation at each crossing (start period)
    # Use release_lon if provided — it is more reliable than lon_self for
    # the first N nights when the fish is still near the release site.
    # lon_self is derived from noon-UTC timing which is noisy; using it to
    # compute elevations creates a circular dependency that can bias
    # thresh_deg by several degrees relative to the true sensor threshold.
    lon_eval = release_lon if release_lon is not None else lon_self

    def _elevations_at(event_pairs, lon, lat):
        elevs_r, elevs_s = [], []
        for t_rise, t_set in event_pairs:
            obs = ephem.Observer()
            obs.lon = str(lon)
            obs.lat = str(lat)
            obs.pressure = 0
            sun = ephem.Sun()
            obs.date = _tz(t_rise).strftime("%Y/%m/%d %H:%M:%S")
            sun.compute(obs)
            elevs_r.append(np.degrees(float(sun.alt)))
            obs.date = _tz(t_set).strftime("%Y/%m/%d %H:%M:%S")
            sun.compute(obs)
            elevs_s.append(np.degrees(float(sun.alt)))
        return np.array(elevs_r + elevs_s)

    elevations = _elevations_at(calib_pairs, lon_eval, lat_eval)
    thresh_deg = float(np.median(elevations))

    # Step 3 (optional): end-period verification at recapture position
    thresh_deg_end = None
    elevations_end = None
    delta_deg = None

    if recapture_lon is not None and recapture_lat is not None:
        n_end = n_nights_end if n_nights_end is not None else n_nights
        if len(pairs) > n_end:
            end_pairs = pairs[-n_end:]
            elevations_end = _elevations_at(end_pairs, recapture_lon, recapture_lat)
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


def compute_solar_likelihood(
    pairs,
    qdf,
    lons,
    lats,
    thresh_deg,
    sigma_timing=18,
    smooth_sigma=1.0,
    lh_floor=1e-30,
):
    """Compute per-night solar likelihood maps on a regular lat/lon grid.

    For each night and each grid pixel, the likelihood is the product of
    Gaussian terms in timing residual:

        L_k = ∏_{e in valid events}  exp(−Δt_e² / (2 σ_timing²))

    where Δt_e is the difference (minutes) between the observed crossing
    time and the astronomically predicted crossing at pixel k.

    Parameters
    ----------
    pairs : list of (pd.Timestamp, pd.Timestamp)
        Paired (t_rise, t_set) events, aligned with rows of ``qdf``.
    qdf : pd.DataFrame
        Quality flags as returned by
        :func:`~pangeo_fish.light.quality.compute_quality_flags`.
        Must have columns ``flag``, ``valid_rise``, ``valid_set``.
    lons : array-like
        1-D longitude axis of the output grid (degrees East).
    lats : array-like
        1-D latitude axis of the output grid (degrees North).
    thresh_deg : float
        Solar elevation threshold (degrees) defining RISE and SET.
        Typically the output of :func:`self_calibrate_solar_threshold`.
    sigma_timing : float, default 18
        Timing uncertainty (minutes).
    smooth_sigma : float, default 1.0
        Standard deviation (pixels) of the Gaussian spatial smoother
        applied to each night's raw likelihood map.
    lh_floor : float, default 1e-30
        Minimum likelihood value; replaces non-finite values and guards
        against log(0).

    Returns
    -------
    xr.DataArray
        Dims ``('time', 'latitude', 'longitude')``.  ``time`` is the
        date of each night (``t_rise.date()``) as UTC ``datetime64``.
        Attributes: ``method='solar_threshold'``,
        ``sigma_timing_min=sigma_timing``.

    Notes
    -----
    **Bug fix — ``window_h=6.0``:** :func:`_get_crossing_fast` is called
    with ``window_h=6.0`` (was 3.0).  See that function's docstring for
    the rationale.

    **Bug fix — ``need_both`` guard:** for nights flagged ``'good'``, a
    pixel where either crossing prediction is unavailable (e.g., polar
    regions where :mod:`ephem` cannot find a rising) is forced to
    ``lh_floor`` instead of falling back to single-event mode.  Without
    this guard, "good" nights produce ghost likelihood peaks at high
    latitudes.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    grid_shape = (len(lats), len(lons))
    lon_flat, lat_flat = np.meshgrid(lons, lats)
    lon_flat = lon_flat.ravel()
    lat_flat = lat_flat.ravel()
    n_pix = len(lon_flat)

    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm

    n_nights = len(pairs)
    lh_smooth = np.zeros((n_nights, len(lats), len(lons)))
    night_times = []

    for i, (t_rise, t_set) in enumerate(
        tqdm(pairs, desc="Solar likelihood", unit="night")
    ):
        night_times.append(pd.Timestamp(t_rise.date()))
        flag = qdf.iloc[i]["flag"]
        v_r = qdf.iloc[i]["valid_rise"]
        v_s = qdf.iloc[i]["valid_set"]

        # Bad night → uniform likelihood (no information)
        if flag == "bad":
            lh_smooth[i] = np.ones(grid_shape)
            continue

        # Good or partial night → compute from valid events
        # KEY FIX: for 'good' nights, BOTH predictions must exist.
        # If one is missing (ephem can't find it at a polar pixel),
        # we refuse single-event fallback to avoid ghost peaks.
        need_both = flag == "good"

        lh = np.full(n_pix, lh_floor)
        for k in range(n_pix):
            parts = []
            got_rise = False
            got_set = False

            if v_s:
                tp_s = _get_crossing_fast(
                    t_set, lon_flat[k], lat_flat[k], thresh_deg, "down"
                )
                if tp_s is not None:
                    dt = (_tz(t_set) - tp_s).total_seconds() / 60.0
                    parts.append(np.exp(-(dt**2) / (2.0 * sigma_timing**2)))
                    got_set = True

            if v_r:
                tp_r = _get_crossing_fast(
                    t_rise, lon_flat[k], lat_flat[k], thresh_deg, "up"
                )
                if tp_r is not None:
                    dt = (_tz(t_rise) - tp_r).total_seconds() / 60.0
                    parts.append(np.exp(-(dt**2) / (2.0 * sigma_timing**2)))
                    got_rise = True

            # need_both guard: refuse single-event fallback for 'good' nights
            if need_both and not (got_rise and got_set):
                lh[k] = lh_floor
            elif parts:
                lh[k] = np.prod(parts)

        # Replace non-finite values before smoothing
        lh[~np.isfinite(lh)] = lh_floor

        lh_map = gaussian_filter(lh.reshape(grid_shape), sigma=smooth_sigma)
        mx = lh_map.max()
        if mx > 0:
            lh_map /= mx  # normalize so max = 1 (NOT sum = 1)
        lh_smooth[i] = lh_map

    da = xr.DataArray(
        lh_smooth,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(night_times, dtype="datetime64[ns]"),
            "latitude": lats,
            "longitude": lons,
        },
        name="pdf_light",
        attrs={
            "method": "solar_threshold",
            "sigma_timing_min": sigma_timing,
        },
    )
    return da
