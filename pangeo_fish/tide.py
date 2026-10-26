import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
from tqdm import tqdm

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

    ds = tide_behav.copy()

    time = pd.to_datetime(ds["time"].values)  # shaping

    # Temporar Dataset to ease the group gestion
    tmp_df = pd.DataFrame(
        {
            "depth": ds["depth"].values,
            "tide_found": ds["tide_found"].values,
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
    Likelihood = datalikelihood_tide_only_full_normalized(
        td_depth, td_time, data, tide_found_hourly, sigma_tid=2.0
    )

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
