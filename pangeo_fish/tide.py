import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sparse
import xarray as xr
import zarr
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
from tqdm import tqdm

from pangeo_fish.helpers import _with_dggs_attrs, regrid_dataset_hpresample
from pangeo_fish.storage import write_time_blocks

#####################################################################
#  First part : extraction of a tidal signal
#####################################################################


M2_PERIOD_H = 12.42  # h, principal lunar semi-diurnal tide
DEFAULT_TIDE_WINDOW_H = 25.0  # h, about two M2 cycles
MAX_TIDE_AMPLITUDE_M = 5.0  # m, largest plausible M2 amplitude in pressure


def lssinfit(time, depth, periods_h=(M2_PERIOD_H,), trend_order=1):
    """Fit a tidal harmonic model on a window of depth measurements.

    A polynomial trend of order ``trend_order`` (in time) is fitted together with a
    sine and a cosine at each period of ``periods_h``. The trend takes care of the
    low-frequency part of the record (ascents, descents, slow drifts), so that only
    what the polynomial cannot explain is attributed to the tide.

    Parameters
    ----------
    time : array-like
        Time in days.
    depth : array-like
        Depth (m) measured at ``time``. Non-finite values are ignored.
    periods_h : sequence of float, optional
        Tidal periods (h). Default: M2 (12.42 h) only. M2 and S2 cannot be separated
        in a window shorter than about 15 days, so S2 is left to the residuals.
    trend_order : int, optional
        Order of the polynomial trend removed before assessing the fit. Default: 1 (linear; a quadratic trend starts to absorb diel migration).

    Returns
    -------
    rmse : float
        Root mean square of the residuals of the full fit (m).
    rsq : float
        R² of the harmonic part, computed on the detrended series: share of the
        variance left by the trend that the harmonics explain.
    amplitude : float
        Amplitude (m) of the first period.
        All values are NaN if the window has too few valid samples.
    """
    time = np.asarray(time, dtype=float)
    depth = np.asarray(depth, dtype=float)
    ok = np.isfinite(time) & np.isfinite(depth)
    n_par = trend_order + 1 + 2 * len(periods_h)
    if ok.sum() <= n_par:
        return np.nan, np.nan, np.nan
    t = time[ok]
    y = depth[ok]
    # centred and scaled time (-1..1) keeps the polynomial well conditioned
    span = np.ptp(t)
    x = (t - t.mean()) / (span / 2) if span > 0 else np.zeros_like(t)
    poly = np.column_stack([x**k for k in range(trend_order + 1)])
    cols = []
    for period in periods_h:
        w = 2 * np.pi / (period / 24)  # rad/day
        cols += [np.sin(w * t), np.cos(w * t)]
    X = np.column_stack([poly, *cols])

    coef, _, _, _ = lstsq(X, y, rcond=None)
    residuals = y - X @ coef
    coef_trend, _, _, _ = lstsq(poly, y, rcond=None)
    detrended = y - poly @ coef_trend

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(detrended**2)
    rsq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))
    amplitude = np.hypot(coef[trend_order + 1], coef[trend_order + 2])
    return rmse, rsq, amplitude


def _sliding_fit(time, depth, window, **fit_kwargs):
    """Fit ``lssinfit`` on every window of ``window`` samples.

    Results are stored at the sample at the CENTRE of the window (``window // 2``
    samples after its start); the first and last half windows are NaN.
    """
    n = len(depth)
    out = np.full((3, n), np.nan)
    half = window // 2
    for i in tqdm(range(0, n - window + 1)):
        idx = slice(i, i + window)
        out[:, i + half] = lssinfit(time[idx], depth[idx], **fit_kwargs)
    return out


def tide_behav_extr(
    df,
    tagno,
    tideFL=DEFAULT_TIDE_WINDOW_H,
    tideLV=[0.5, 0.7, 0.5],
    behavFL=DEFAULT_TIDE_WINDOW_H,
    behavLV=[0.5, 0.7, 0.5],
    dt=10,
    ampli_max=MAX_TIDE_AMPLITUDE_M,
    trend_order=1,
    periods_h=(M2_PERIOD_H,),
):
    """Sliding-window detection of the tide in a pressure record.

    Python translation of the MATLAB function tidebehavextr.m, corrected so that a
    flag means "the record looks like a fish lying on the bottom under the M2 tide".

    For every window of ``tideFL`` hours, a polynomial trend of order ``trend_order``
    and a harmonic at each of ``periods_h`` are fitted to the raw pressure (see
    :func:`lssinfit`). The window is flagged when

    - the residual RMS is below ``tideLV[0]`` (m),
    - the R² on the detrended series is above ``tideLV[1]``,
    - the amplitude is between ``tideLV[2]`` and ``ampli_max`` (m).

    Each flag and each statistic is stored at the CENTRE of its window. The first and
    last half windows have no value (NaN statistics, flag 0).

    ``behav_found`` is the same test with the window ``behavFL`` and the thresholds
    ``behavLV``. It is a relaxed tide fit (the demo uses it as a switch between
    "still" and "moving" days), not an independent behaviour classifier.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``time`` and ``pressure`` (m), regularly sampled every ``dt`` minutes.
    tagno : str
        Tag identifier.
    tideFL, behavFL : float, optional
        Windows (h), at least one M2 period (12.42 h). Default: 25 h, two M2 cycles,
        so that the trend and the harmonic are well separated. Windows are made of
        ``round(60 / dt * FL)`` samples.
    tideLV, behavLV : list of float, optional
        ``[max residual RMS (m), min R², min amplitude (m)]``.
    dt : float, optional
        Time step (min).
    ampli_max : float, optional
        Largest plausible tidal amplitude (m). Default: 5.
    trend_order : int, optional
        Order of the trend removed in each window. Default: 1 (linear; a quadratic trend starts to absorb diel migration).
    periods_h : sequence of float, optional
        Tidal periods fitted (h). Default: M2.

    Returns
    -------
    xarray.Dataset
        Statistics (``rmse_*``, ``rsq_*``, ``ampli_*``) and flags (``tide_found``,
        ``behav_found``, integer 0/1) along ``time``.
    """
    # --- Reading of the brute datas ---

    if "time" not in df.columns or "pressure" not in df.columns:
        raise ValueError("'time' and 'pressure' shoud be part of the CSV")
    for name, fl in (("tideFL", tideFL), ("behavFL", behavFL)):
        if fl < M2_PERIOD_H:
            raise ValueError(
                f"{name}={fl} h is shorter than one M2 period ({M2_PERIOD_H} h)"
            )
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    time = (
        (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 3600.0 / 24.0
    )  # in days
    time = time.to_numpy()
    depth = df["pressure"].to_numpy()
    n = len(depth)
    tide_window = int(round(60 / dt * tideFL))
    behav_window = int(round(60 / dt * behavFL))
    fit_kwargs = dict(periods_h=tuple(periods_h), trend_order=trend_order)

    def found(stats, levels):
        rmse, rsq, ampli = stats
        with np.errstate(invalid="ignore"):
            return (
                (rmse < levels[0])
                & (rsq > levels[1])
                & (ampli > levels[2])
                & (ampli < ampli_max)
            )

    # --- Sliding loops (results at the window centres) ---
    print(f"Tide signal extraction ({tideFL} h)...")
    stats_tide = _sliding_fit(time, depth, tide_window, **fit_kwargs)
    tide_found = found(stats_tide, tideLV)
    print(f"Relaxed behaviour fit ({behavFL} h)...")
    stats_behav = _sliding_fit(time, depth, behav_window, **fit_kwargs)
    behav_found = found(stats_behav, behavLV)
    rmse_tide, rsq_tide, ampli_tide = stats_tide
    rmse_behav, rsq_behav, ampli_behav = stats_behav

    # --- Output dataset ---
    ds = xr.Dataset(
        data_vars=dict(
            depth=("time", depth),
            rmse_tide=("time", rmse_tide),
            rsq_tide=("time", rsq_tide),
            ampli_tide=("time", ampli_tide),
            tide_found=("time", tide_found.astype(int)),
            rmse_behav=("time", rmse_behav),
            rsq_behav=("time", rsq_behav),
            ampli_behav=("time", ampli_behav),
            behav_found=("time", behav_found.astype(int)),
        ),
        coords=dict(
            time=("time", df["time"]),
        ),
        attrs=dict(
            tagno=str(tagno),
            tideFL=tideFL,
            tideLV=tideLV,
            behavFL=behavFL,
            behavLV=behavLV,
            dt=dt,
            ampli_max=ampli_max,
            trend_order=trend_order,
            periods_h=list(periods_h),
        ),
    )

    print("\nExtraction done")
    return ds


def convert_lon_360_to_180(ds, bbox=None):
    """
    Convert longitudes from 0-360 to -180-180,
    apply a bbox, and use nearest if the exact born does not match.

    Parameters
    ----------
    ds : xarray.Dataset

    bbox : dict or None
        Exemple :
        bbox = {"latitude": [lat_min, lat_max], "longitude": [lon_min, lon_max]}
    """

    ds = ds.copy()

    # --- 1. 0–360 → –180–180 ---
    lon_new = ((ds.lon + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon_new)
    ds = ds.sortby("lon")

    # --- 2. bbox ---
    if bbox is not None:
        # -------- LATITUDE --------
        if "latitude" in bbox:
            lat_min, lat_max = bbox["latitude"]
            print(lat_min, lat_max)
            # Vérifie si les valeurs existent
            lat_vals = ds.lat.values
            # min
            if lat_min in lat_vals:
                print("here")
                lat_min_sel = lat_min
            else:
                lat_min_sel = ds.lat.sel(lat=lat_min, method="nearest").item()
            # max
            if lat_max in lat_vals:
                print("HERE")
                lat_max_sel = lat_max
            else:
                lat_max_sel = ds.lat.sel(lat=lat_max, method="nearest").item()
                print(lat_max_sel)
            print(slice(lat_min_sel, lat_max_sel))
            ds = ds.sortby("lat")  # latitudes croissantes
            ds = ds.sel(lat=slice(lat_min_sel, lat_max_sel))
        # -------- LONGITUDE --------
        if "longitude" in bbox:
            lon_min, lon_max = bbox["longitude"]
            lon_vals = ds.lon.values
            # min
            if lon_min in lon_vals:
                lon_min_sel = lon_min
            else:
                lon_min_sel = ds.lon.sel(lon=lon_min, method="nearest").item()
            # max
            if lon_max in lon_vals:
                lon_max_sel = lon_max
            else:
                lon_max_sel = ds.lon.sel(lon=lon_max, method="nearest").item()
            ds = ds.sel(lon=slice(lon_min_sel, lon_max_sel))

    return ds


###_______________________________________________________________________________________________________###
### METHOD TIDE FULL NORMALIZED ###
###_______________________________________________________________________________________________________###


def datalikelihood_tide_only_full_normalized(
    td_depth, td_time, tidal_ds, tide_found, sigma_tid=2.0
):
    """
    Compute a normalized PDF for each steptime.
    - If tide detected : PDF = likelihood  (gaussian)
    - Else :  uniform PDF on the ocean (sum == 1)
    Output: Dataset with noramlized PDF for each (t,lat,lon).
    """
    eps = 1e-200
    lat = tidal_ds.lat.values
    lon = tidal_ds.lon.values
    mask = tidal_ds.mask.values.astype(bool)  # True = océan
    n_times, n_meas = td_depth.shape
    n_lat, n_lon = len(lat), len(lon)
    # Tide parameters from the database
    amp = tidal_ds.amplitude.values
    phase = tidal_ds.phase.values
    omega = tidal_ds.omega.values
    depth_mean = tidal_ds.wct.values

    # PDF finale
    PDF = np.full((n_times, n_lat, n_lon), np.nan)
    tide_idx = np.where(tide_found == 1)[0]
    notide_idx = np.where(tide_found == 0)[0]

    # --- Case 1 : Tide found → normalized gaussian PDF ---
    for t in tqdm(tide_idx, desc="PDF with Tide found"):
        x = td_depth[t, :]  # shape (n_meas,)
        # Empirical covariance or fallback if n_meas=1
        if n_meas == 1:
            cov = np.array([[sigma_tid**2]])
        else:
            cov = np.cov(x, rowvar=False)
            if cov.shape == ():  # particular case
                cov = np.array([[cov]])
            else:
                cov = np.atleast_2d(cov)
        sigma_vec = np.sqrt(np.diag(cov)).mean()
        if sigma_vec < 1e-6:  # seuil arbitraire
            sigma_vec = sigma_tid

        # --- mu computed only for this steptime t ---
        tidal_pred_t = amp * np.cos(omega * td_time[t] + phase)  # shape (dim0, n_lon)
        tidal_sum_t = tidal_pred_t.sum(axis=0)  # shape (n_lon,)
        mu_ij = -tidal_sum_t[None, :] + depth_mean  # shape (n_lat, n_lon)
        ###

        x_mean = np.nanmean(td_depth[t, :])
        L = np.exp(-0.5 * ((x_mean - mu_ij) / sigma_vec) ** 2) / (
            sigma_vec * np.sqrt(2 * np.pi)
        )
        L = L * mask
        s = np.nansum(L)  # normalization
        if s == 0:
            PDF[t, :, :] = mask / np.sum(mask)  # fallback uniforme
        else:
            PDF[t, :, :] = L / s
        PDF[t, :, :] = np.maximum(PDF[t, :, :], eps)
        PDF[t, :, :] /= np.nansum(PDF[t, :, :])

    # --- Cas 2 : No Tide found → uniform PDF  ---
    ocean_pixels = np.sum(mask)
    uniform_pdf = mask / ocean_pixels
    uniform_pdf = np.maximum(uniform_pdf, eps)
    uniform_pdf /= np.nansum(uniform_pdf)
    for t in notide_idx:
        PDF[t, :, :] = uniform_pdf

    # Outpud dataset
    ds_pdf = xr.Dataset(
        {
            "pdf": (("time", "latitude", "longitude"), PDF),
            "ocean_mask": (("latitude", "longitude"), mask),
            "tide_found": (("time"), tide_found),
        },
        coords={"time": td_time, "latitude": lat, "longitude": lon},
    )
    return ds_pdf


def tide_pdf(tide_behav, data):
    """Compute the tide-based likelihood map of a tag from its depth time series.

    The function resamples the depth record into hourly groups, builds a matrix of
    measurements (one row per hour, padded with NaN) and a binary hourly indicator of tide
    detection. These are compared with the tidal model ``data`` through
    ``datalikelihood_tide_only_full_normalized`` (with ``sigma_tid=2.0``) to obtain, for each
    hour and each grid cell, the likelihood of the observations. The output is re-indexed on
    the hourly datetimes of the tag record.

    Parameters
    ----------
    tide_behav : xarray.Dataset
        Output of the tide/behavior detection (see ``detect_tide``), with a ``time``
        dimension and the variables ``depth`` (m) and ``tide_found`` (bool, True when a tide
        is detected at that instant).
    data : xarray.Dataset
        Tidal model dataset on a longitude/latitude grid

    Returns
    -------
    ds_pdf : xarray.Dataset
        Likelihood dataset indexed by an hourly ``time`` coordinate (datetimes of the tag
        record)

    Notes
    -----
    - The hourly indicator ``tide_found`` is True if a tide is detected at least once
      during the hour.
    - Hours are counted from the first hour of the record (``td_time`` in hours); gaps in the
      record are therefore kept in the time axis of the likelihood.
    - ``tide_behav`` is copied and not modified.

    Examples
    --------
    >>> tide_behav = detect_tide(tag_log, "tag_01")
    >>> ds_pdf = tide_pdf(tide_behav, data)
    """
    ds = tide_behav.copy()

    time = pd.to_datetime(ds["time"].values)  # shaping

    # Temporar Dataset to ease the group gestion
    tmp_df = pd.DataFrame(
        {
            "depth": ds["depth"].values,
            "tide_found": ds["tide_found"].values,
            "behav_found": ds["behav_found"].values,
        },
        index=time,
    )
    # Mean column for an hour
    tmp_df["hour"] = tmp_df.index.floor("h")

    # If Tide is found at least once in the column the all the values gain tide
    tmp_df["hour_include"] = tmp_df["hour"]
    for h, group in tmp_df.groupby("hour"):
        if group["tide_found"].any():
            tmp_df.loc[group.index, "hour_include"] = h

    # Group by "hour_include"
    grouped_obs = {
        hour: group[["depth"]].values for hour, group in tmp_df.groupby("hour_include")
    }
    hours_sorted = sorted(grouped_obs.keys())

    # td_time : number of hour since the begining
    td_time = np.array(
        [(h - hours_sorted[0]).total_seconds() / 3600 for h in hours_sorted]
    )

    # td_depth : matrice (n_times, n_meas_max)
    n_times = len(hours_sorted)
    n_meas_max = max(len(grouped_obs[h]) for h in hours_sorted)
    td_depth = np.full((n_times, n_meas_max), np.nan)
    for i, h in enumerate(hours_sorted):
        vals = grouped_obs[h].flatten()
        td_depth[i, : len(vals)] = vals

    tide_found_hourly = np.array(
        [tmp_df[tmp_df["hour_include"] == h]["tide_found"].any() for h in hours_sorted],
        dtype=int,
    )
    behav_found_hourly = np.array(
        [
            tmp_df[tmp_df["hour_include"] == h]["behav_found"].any()
            for h in hours_sorted
        ],
        dtype=int,
    )
    Likelihood = datalikelihood_tide_only_full_normalized(
        td_depth, td_time, data, tide_found_hourly, sigma_tid=2.0
    )
    Likelihood = Likelihood.assign(behav_found=("time", behav_found_hourly))
    ds_pdf = Likelihood.assign_coords(datetime=("time", hours_sorted))
    ds_pdf = ds_pdf.swap_dims({"time": "datetime"})
    ds_pdf = ds_pdf.rename({"time": "instant"})
    ds_pdf = ds_pdf.rename({"datetime": "time"}).set_index(time="time")
    ds_pdf = ds_pdf.rename({"ocean_mask": "mask"})
    return ds_pdf


def plot_tide(tag_test):
    # function that shows where tide have been detected should be implemented here
    # Dataset : ds
    time = tag_test["time"].values
    depth = tag_test["depth"].values
    tide_found = tag_test["tide_found"].values  # 0 or 1
    behav_found = tag_test["behav_found"].values
    fig, ax = plt.subplots(figsize=(15, 4))

    ax.fill_between(
        time,
        depth.min(),
        depth.max(),
        where=(tide_found == 1),
        color="green",
        alpha=0.3,
        label="Marée détectée",
    )
    ax.fill_between(
        time,
        depth.min(),
        depth.max(),
        where=(behav_found == 1),
        color="red",
        alpha=0.3,
        label="Comportement détecté",
    )

    ax.plot(time, depth, color="blue", label="Profondeur")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Profondeur (m)")
    ax.set_title(f'Temporal Depth & Tidal Detection - Tag {tag_test.attrs["tagno"]}')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


###_______________________________________________________________________________________________________###
### FACTORIZATION ###
###_______________________________________________________________________________________________________###

from scipy.ndimage import median_filter

DT_TARGET = 10  # min, time step expected by tide_behav_extr

TIDE_PARAMS = dict(
    tideFL=DEFAULT_TIDE_WINDOW_H,  # h, window used for tide detection (>= one M2 period)
    tideLV=[0.42, 0.85, 0.6],  # max RMSE (m), min R², min amplitude (m) (cf. Pedersen)
    behavFL=DEFAULT_TIDE_WINDOW_H,  # h, window of the relaxed fit (behav_found)
    behavLV=[0.42, 0.70, 0.2],
)


def time_step_min(da, dim="time"):
    """Time step (min) of a regular time series."""
    return float(np.median(np.diff(da[dim].values)) / np.timedelta64(1, "m"))


def prepare_pressure(
    tag_log, var="pressure", dt_target=DT_TARGET, smooth=False, alpha=5, threshold=1.0
):
    """Resample to dt_target if needed, optionally smooth, return a DataFrame."""
    da = tag_log[var]
    dt = time_step_min(da)

    if dt < dt_target:
        da = da.resample(time=f"{dt_target}min").first()
    elif dt > dt_target:
        raise NotImplementedError(f"Time step {dt} min > {dt_target} min not supported")

    if smooth:  # remove spikes shorter than `alpha` time steps
        med = median_filter(da.values, size=2 * alpha + 1, mode="nearest")
        da = da.where(np.abs(da - med) <= threshold, med)

    return da.to_dataframe().reset_index().dropna(subset=[var])


def detect_tide(tag_log, tag_name, smooth=False, **kwargs):
    """Detect tidal and behavioral patterns from a tag pressure time series.

    A flag (``tide_found``) is set at the CENTRE of every window long enough to hold the
    M2 tide whose detrended pressure is explained by an M2 harmonic (see
    :func:`tide_behav_extr`: polynomial trend removed, R² on the detrended series,
    residual RMS and amplitude range checked).

    The function extracts the ``pressure`` variable from the tag dataset, resamples it to a
    10 min time step if the data are recorded at a finer resolution, optionally removes short
    spikes with a median filter, and converts the result to a DataFrame. The detection itself
    is performed by ``tide_behav_extr`` using the default parameters stored in ``TIDE_PARAMS``,
    which can be overridden through ``kwargs``.

    Parameters
    ----------
    tag_log : xarray.Dataset
        A dataset containing the tag measurements, with a regular `time` dimension and a
        `pressure` variable (in meters). Time steps larger than 10 min are not supported.
    tag_name : str
        Identifier of the tag
    smooth : bool, optional
        If True, apply a median filter to replace spikes (differences to the median larger than
        1 m, lasting less than 5 time steps) before detection. Default is False.
    **kwargs
        Overrides of the default parameters passed to ``tide_behav_extr``
        (see ``TIDE_PARAMS``).

    Other Parameters
    ----------------
    tideFL : float, optional
        Window (h) used for tide detection, at least 12.42 h (one M2 period).
        Default is 25 (about two cycles).
    ampli_max : float, optional
        Largest plausible tidal amplitude (m). Default is 5.
    trend_order : int, optional
        Order of the polynomial trend removed in each window. Default is 1 (linear).
    tideLV : list of float, optional
        Tide thresholds ``[max RMSE (m), min R², min amplitude (m)]``.
        Default is ``[0.42, 0.85, 0.6]``. A given list replaces the whole default.
    behavFL : float, optional
        Window (h) of the relaxed fit behind ``behav_found``. Default is 25.
    behavLV : list of float, optional
        Thresholds of the relaxed fit, same structure as ``tideLV``.
        Default is ``[0.42, 0.70, 0.2]``.

        Any other keyword accepted by ``tide_behav_extr`` can also be passed, except ``df`` and
        ``tagno`` (set by the function). ``dt`` is fixed to ``DT_TARGET`` (10 min) and should
        not be overridden since the data are resampled accordingly.

    Returns
    -------
    tag_test : pandas.DataFrame
        The output of ``tide_behav_extr``: the detection results for the tag (same type as
        returned by this function).

    Raises
    ------
    NotImplementedError
        If the time step of the data is larger than 10 min.

    Examples
    --------
    >>> detect_tide(tag_log, "tag_01")
    >>> detect_tide(tag_log, "tag_01", smooth=True, tideFL=30, behavLV=[0.5, 0.7, 0.3])
    """
    df = prepare_pressure(tag_log, smooth=smooth)
    return tide_behav_extr(
        df=df, tagno=tag_name, dt=DT_TARGET, **{**TIDE_PARAMS, **kwargs}
    )


VARS_NO_TIME = ["cell_ids", "latitude", "longitude", "resolution", "ocean_mask"]


def _centered_daily_mean(obj):
    """Daily mean over [T - 12 h, T + 12 h), labelled T (midnight).

    This is the window of the temperature emission (``cf_xarray`` time bounds of a
    daily axis), unlike ``resample(time="1D")`` which covers [T, T + 24 h).
    """
    half = pd.Timedelta("12h")
    out = obj.resample(time="1D", offset=half).mean(skipna=True)
    return out.assign_coords(time=out.time + half)


def tide_to_healpix(
    target_root,
    refinement_level,
    ellipsoid,
    chunk_size=2000,
    step=200,
    dataset=None,
):
    """Regrid the tide likelihood to HEALPix, resample daily and save as zarr.

    Parameters
    ----------
    target_root : str or Path
        Folder containing ``tide_pdf.zarr`` and receiving ``tide_pdf_healpix.zarr``.
    refinement_level : int
        HEALPix refinement level.
    ellipsoid : object
        Ellipsoid passed to ``regrid_dataset_hpresample``.
    chunk_size : int, optional
        Time steps regridded at once (limits resampler memory). Default is 2000.
    step : int, optional
        Daily time steps written at once to zarr. Default is 200.
    dataset : xarray dataset, optional
        avoid to use open_dataset by passing directly in argument the dataset
    Returns
    -------
    tide_found_resampled : xarray.DataArray
        Daily mean of ``tide_found``, also written to the zarr store.
    behav_found_resampled : xarray.DataArray
        Daily mean of ``behav_found``, also written to the zarr store.
    ds : xarray.dataset dataset with the pdf from tide and healpix coordinate
    """
    if dataset is None:
        lik_full = xr.open_dataset(
            f"{target_root}/tide_pdf.zarr", engine="zarr", chunks={}
        )
    else:
        lik_full = dataset.copy()
    lik_full = lik_full.rename({"mask": "ocean_mask"})
    lik_full["ocean_mask"] = lik_full["ocean_mask"].compute()
    tide_found = lik_full["tide_found"]
    behav_found = lik_full["behav_found"]

    lon2d, lat2d = xr.broadcast(lik_full.longitude, lik_full.latitude)
    lon2d = lon2d.transpose("latitude", "longitude")
    lat2d = lat2d.transpose("latitude", "longitude")

    # Regrid by temporal chunks to limit memory
    times = lik_full.time.to_index()
    n_time = len(times)
    # chunks are cut at 12:00 so that no centred daily window (label +- 12 h) is split
    days = (times - pd.Timedelta("12h")).normalize()

    day_starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])

    boundaries = [0]
    for s in day_starts:
        if s - boundaries[-1] >= chunk_size:
            boundaries.append(int(s))
    boundaries.append(n_time)

    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        print(f"Processing time chunk [{start}:{end}] / {n_time}")
        prepped = (
            lik_full.isel(time=slice(start, end))
            .rename({"latitude": "yi", "longitude": "xi"})
            .assign_coords(
                longitude=(("yi", "xi"), lon2d.values),
                latitude=(("yi", "xi"), lat2d.values),
            )
        )
        reshaped, _ = regrid_dataset_hpresample(
            ds=prepped, refinement_level=refinement_level, ellipsoid=ellipsoid
        )
        mask = reshaped["ocean_mask"].compute()
        if isinstance(mask.data, sparse.SparseArray):
            reshaped["ocean_mask"] = mask.copy(data=mask.data.todense())
        chunks.append(_centered_daily_mean(reshaped))

    print("Concatenating chunks...")
    ds = xr.concat(chunks, dim="time")
    ds = _with_dggs_attrs(ds)
    # Write format 2, with region boundaries aligned to the storage chunks.
    store_path = f"{target_root}/tide_pdf_healpix.zarr"
    write_time_blocks(ds.chunk({"cells": -1}), store_path, step=step)

    tide_found_resampled = _centered_daily_mean(tide_found)
    tide_found_resampled.sel(time=ds.time).to_zarr(
        store_path, mode="a", zarr_format=2, consolidated=False
    )
    behav_found_resampled = _centered_daily_mean(behav_found)
    behav_found_resampled.sel(time=ds.time).to_zarr(
        store_path, mode="a", zarr_format=2, consolidated=False
    )
    zarr.consolidate_metadata(store_path)
    print("DONE")
    return ds, tide_found_resampled, behav_found_resampled


def merge_tide_with_total(target_root, storage_options=None):
    """Merge the daily tide likelihood with the other likelihoods on their common cells and days.

    Parameters
    ----------
    target_root : str or Path
        Folder containing ``total_pdf.zarr`` and ``tide_pdf_healpix.zarr``.
    storage_options : dict, optional
        Options passed to ``xarray.open_dataset`` for remote storage.

    Returns
    -------
    emission_with_bathy_tide : xarray.Dataset
        Merged dataset restricted to the common ``cell_ids`` and ``time``, with the tide
        likelihood stored as ``pdf_tide`` (``tide_found`` and ``ocean_mask`` removed).
    tide_mask : xarray.DataArray
        Daily ``tide_found`` on the common time steps.
    """
    open_kw = dict(engine="zarr", chunks={}, storage_options=storage_options)
    total = xr.open_dataset(f"{target_root}/total_pdf.zarr", **open_kw).load()
    daily = xr.open_dataset(f"{target_root}/tide_pdf_healpix.zarr", **open_kw)
    for name in ("grid_name", "level", "indexing_scheme", "ellipsoid"):
        tide_value = daily.cell_ids.attrs.get(name)
        total_value = total.cell_ids.attrs.get(name)
        if (
            tide_value is not None
            and total_value is not None
            and str(tide_value).casefold() != str(total_value).casefold()
        ):
            raise ValueError(
                f"HEALPix grid mismatch for {name}: {tide_value} vs {total_value}"
            )

    # Keep the HMM cell order and match the tide values by ID.
    daily = daily.groupby("time").mean()
    total_ids = total.cell_ids.to_index()
    common_pix = total_ids.intersection(daily.cell_ids.to_index())
    common_time = daily.time.to_index().intersection(total.time.to_index())

    def restrict(ds):
        selected = ds.isel(cells=ds.cell_ids.to_index().get_indexer(common_pix))
        return selected.sel(time=common_time)

    merged = (
        restrict(daily)
        .rename({"pdf": "pdf_tide"})
        .merge(restrict(total), compat="override", join="exact")
    )
    merged.attrs = total.attrs.copy()

    tide_mask = merged["tide_found"].broadcast_like(merged["pdf_tide"])
    behav_mask = merged["behav_found"].broadcast_like(merged["pdf_tide"])
    merged = merged.drop_vars(["tide_found", "ocean_mask", "behav_found"])

    # Rechunk for Zarr writes and discard the source stores' chunk encodings.
    merged = merged.chunk({"time": 3, "cells": -1})
    for var in merged.variables.values():
        var.encoding.pop("chunks", None)
    merged.attrs["sources"] = list(set(total.attrs.get("sources", [])) | {"pdf_tide"})
    return merged, tide_mask, behav_mask
