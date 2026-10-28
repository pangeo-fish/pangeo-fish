import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
from tqdm import tqdm
import sparse
from pangeo_fish.helpers import regrid_dataset_hpresample

#####################################################################
#  First part : extraction of a tidal signal
#####################################################################


def lssinfit(time, depth):
    """
    Fit a simple sinusoidal model on depth measure.
    Return rmse, rsquare, amplitude.
    """

    depth_smoothed = savgol_filter(depth, window_length=15, polyorder=2, mode="mirror")

    w = 2 * np.pi / (12.42 / 24)  # tide frequency (rad/day)
    X = np.column_stack([np.ones_like(time), np.sin(w * time), np.cos(w * time)])
    coef, _, _, _ = lstsq(X, depth_smoothed, rcond=None)
    fitted = X @ coef
    residuals = depth_smoothed - fitted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((depth - np.mean(depth)) ** 2)
    rsq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))
    amplitude = np.sqrt(coef[1] ** 2 + coef[2] ** 2)
    return rmse, rsq, amplitude  # , depth_smoothed


def tide_behav_extr(
    df,
    tagno,
    tideFL=10,
    tideLV=[0.5, 0.7, 0.5],
    behavFL=16,
    behavLV=[0.5, 0.7, 0.5],
    dt=10,
):
    """
    Traduction Python of MATLAB function tidebehavextr.m
    Parameter : CSV with column 'time', 'depth' (and enventually 'temp')
    Output : xarray.Dataset with result of the classification tide / behavior
    """
    # --- Reading of the brute datas ---

    if "time" not in df.columns or "pressure" not in df.columns:
        raise ValueError("'time' and 'pressure' shoud be part of the CSV")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    time = (
        (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 3600.0 / 24.0
    )  # in days
    depth = df["pressure"].to_numpy()
    print(df["time"] - df["time"].iloc[0])
    n = len(depth)
    tide_window = int(round(60 / dt * tideFL))
    print(tide_window)
    behav_window = int(round(60 / dt * behavFL))

    # --- Initialisation ---
    rmse_tide = np.full(n, np.nan)
    rsq_tide = np.full(n, np.nan)
    ampli_tide = np.full(n, np.nan)

    rmse_behav = np.full(n, np.nan)
    rsq_behav = np.full(n, np.nan)
    ampli_behav = np.full(n, np.nan)

    # --- Sliding loop for Tide ---
    print(f"Extraction du signal de marée ({tideFL} h)...")
    for i in tqdm(range(0, n - tide_window)):
        idx = slice(i, i + tide_window)
        rmse, rsq, ampli = lssinfit(time[idx], depth[idx])
        rmse_tide[i] = rmse
        rsq_tide[i] = rsq
        ampli_tide[i] = ampli
        # depth_smoothed_list[i] = depth_lissage

    # --- Tide intervals detections ---
    tide_found = (
        (rmse_tide < tideLV[0]) & (rsq_tide > tideLV[1]) & (ampli_tide > tideLV[2])
    )

    # --- Sliding loop for behavior ---
    print(f"Classification comportementale ({behavFL} h)...")
    for i in tqdm(range(0, n - behav_window)):
        idx = slice(i, i + behav_window)
        rmse, rsq, ampli = lssinfit(time[idx], depth[idx])
        rmse_behav[i] = rmse
        rsq_behav[i] = rsq
        ampli_behav[i] = ampli

    behav_found = (
        (rmse_behav < behavLV[0])
        & (rsq_behav > behavLV[1])
        & (ampli_behav > behavLV[2])
    )

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
        ),
    )

    print("\nExtraction terminée ✅")
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
        [tmp_df[tmp_df["hour_include"] == h]["behav_found"].any() for h in hours_sorted],
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

import numpy as np
from scipy.ndimage import median_filter

DT_TARGET = 10  # min, time step expected by tide_behav_extr

TIDE_PARAMS = dict(
    tideFL=10,                 # h, window used for tide detection
    tideLV=[0.42, 0.85, 0.6],  # max RMSE (m), min R², min amplitude (m) (cf. Pedersen)
    behavFL=16,                # h, window used for behavior detection
    behavLV=[0.42, 0.70, 0.2],
)

def time_step_min(da, dim="time"):
    """Time step (min) of a regular time series."""
    return float(np.median(np.diff(da[dim].values)) / np.timedelta64(1, "m"))

def prepare_pressure(tag_log, var="pressure", dt_target=DT_TARGET,
                     smooth=False, alpha=5, threshold=1.0):
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
        Window (h) used for tide detection. Default is 10.
    tideLV : list of float, optional
        Tide thresholds ``[max RMSE (m), min R², min amplitude (m)]``.
        Default is ``[0.42, 0.85, 0.6]``. A given list replaces the whole default.
    behavFL : float, optional
        Window (h) used for behavior detection. Default is 16.
    behavLV : list of float, optional
        Behavior thresholds, same structure as ``tideLV``.
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
    >>> detect_tide(tag_log, "tag_01", smooth=True, tideFL=12, behavLV=[0.5, 0.7, 0.3])
    """
    df = prepare_pressure(tag_log, smooth=smooth)
    return tide_behav_extr(df=df, tagno=tag_name, dt=DT_TARGET,
                           **{**TIDE_PARAMS, **kwargs})


VARS_NO_TIME = ["cell_ids", "latitude", "longitude", "resolution", "ocean_mask"]


def tide_to_healpix(target_root, refinement_level, ellipsoid,
                    chunk_size=2000, step=200):
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

    Returns
    -------
    tide_found_resampled : xarray.DataArray
        Daily mean of ``tide_found``, also written to the zarr store.
    behav_found_resampled : xarray.DataArray
        Daily mean of ``behav_found``, also written to the zarr store.
    ds : xarray.dataset dataset with the pdf from tide and healpix coordinate
    """
    lik_full = xr.open_dataset(f"{target_root}/tide_pdf.zarr", engine="zarr", chunks={})
    lik_full = lik_full.rename({"mask": "ocean_mask"})
    lik_full["ocean_mask"] = lik_full["ocean_mask"].compute()
    tide_found = lik_full["tide_found"]
    behav_found = lik_full["behav_found"]

    lon2d, lat2d = xr.broadcast(lik_full.longitude, lik_full.latitude)
    lon2d = lon2d.transpose("latitude", "longitude")
    lat2d = lat2d.transpose("latitude", "longitude")

    # Regrid by temporal chunks to limit memory
    n_time = lik_full.sizes["time"]
    chunks = []
    for start in range(0, n_time, chunk_size):
        end = min(start + chunk_size, n_time)
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
        if isinstance(mask.data, sparse.SparseArray):  # densify if sparse
            reshaped["ocean_mask"] = mask.copy(data=mask.data.todense())

        chunks.append(reshaped.resample(time="1D").mean(skipna=True))

    print("Concatenating chunks...")
    ds = xr.concat(chunks, dim="time")
    ds = ds.drop_vars("pdf").assign(pdf=ds.pdf.chunk({"time": 10, "cells": -1}))
    ds = ds.chunk({"time": 10})
    for var in ds.variables:
        ds[var].encoding.pop("chunks", None)

    # Write: structure first, then time blocks, then time-independent variables
    store_path = f"{target_root}/tide_pdf_healpix.zarr"
    ds.to_zarr(store_path, mode="w", compute=False)

    n_time = ds.sizes["time"]
    for start in range(0, n_time, step):
        end = min(start + step, n_time)
        print(f"Writing time[{start}:{end}] / {n_time}")
        ds.isel(time=slice(start, end)).drop_vars(VARS_NO_TIME).to_zarr(
            store_path, mode="r+", region={"time": slice(start, end)}
        )
    ds[VARS_NO_TIME].to_zarr(store_path, mode="a")

    tide_found_resampled = tide_found.resample(time="1D").mean(skipna=True)
    tide_found_resampled.sel(time=ds.time).to_zarr(store_path, mode="a")
    behav_found_resampled = behav_found.resample(time="1D").mean(skipna=True)
    behav_found_resampled.sel(time=ds.time).to_zarr(store_path, mode="a")
    print("DONE")
    return ds, tide_found_resampled, behav_found_resampled

import xarray as xr


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
    daily = daily.groupby("time").mean()

    # Common cells and days
    common_pix = daily.cell_ids.to_index().intersection(total.cell_ids.to_index())
    common_time = daily.time.to_index().intersection(total.time.to_index())

    def restrict(ds):
        return ds.where(ds.cell_ids.isin(common_pix).compute(), drop=True).sel(
            time=common_time
        )

    merged = restrict(daily).rename({"pdf": "pdf_tide"}).merge(
        restrict(total), compat="override", join="exact"
    )
    merged.attrs = total.attrs.copy()

    tide_mask = merged["tide_found"]
    behav_mask = merged["behav_found"]
    merged = merged.drop_vars(["tide_found", "ocean_mask","behav_found"])

    # Chunking for zarr and source tracking
    merged = merged.chunk({"time": 3, "cells": -1})
    merged.attrs["sources"] = list(set(total.attrs.get("sources", [])) | {"pdf_tide"})
    return merged, tide_mask, behav_mask